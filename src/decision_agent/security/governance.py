"""Request-bound provider egress governance for the formal runtime only."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Lock
from time import monotonic
from typing import Any, Protocol, TypeVar
from uuid import uuid4

from decision_agent.security.audit import AuditChainError, AuditEvent, new_audit_event
from decision_agent.security.models import SecurityContext
from decision_agent.security.provider_policy import (
    DataClassification,
    ProviderPolicy,
    ProviderPolicyError,
    ProviderStage,
    require_provider_egress,
)
from decision_agent.security.redaction import ProviderRedactor

T = TypeVar("T")
ProviderPayloadProjector = Callable[
    [tuple[Any, ...], dict[str, Any]], tuple[dict[str, Any], dict[str, Any], int]
]


class AuditSink(Protocol):
    """The narrow synchronous durability boundary required by request execution."""

    def append(self, event: AuditEvent) -> AuditEvent: ...

    def close(self) -> None: ...


class InMemoryAuditSink:
    """Explicitly non-durable sink reserved for isolated offline unit composition."""

    def __init__(self) -> None:
        self.events: list[AuditEvent] = []
        self.closed = False

    def append(self, event: AuditEvent) -> AuditEvent:
        if self.closed:
            raise AuditChainError("audit_sink_closed")
        self.events.append(event)
        return event

    def close(self) -> None:
        self.closed = True


@dataclass(frozen=True, slots=True)
class ProviderRequestContext:
    request_id: str
    trace_id: str
    principal_type: str
    tenant_hash: str
    policy_id: str
    policy_version: str


_REQUEST_CONTEXT: ContextVar[ProviderRequestContext | None] = ContextVar(
    "provider_governance_request_context", default=None
)


@dataclass(slots=True)
class ProviderRequestState:
    """One task-shared request budget; child asyncio contexts retain the same object."""

    call_count: int = 0
    lock: Lock = field(default_factory=Lock, repr=False)


_REQUEST_STATE: ContextVar[ProviderRequestState | None] = ContextVar(
    "provider_governance_request_state", default=None
)


class ProviderGovernance:
    """Policy, deterministic redaction, durable audit, transport, and output gate."""

    def __init__(
        self,
        *,
        policy: ProviderPolicy | None,
        audit_sink: AuditSink | None,
        redactor: ProviderRedactor | None,
    ) -> None:
        self._policy = policy
        self._audit_sink = audit_sink
        self._redactor = redactor

    @property
    def policy(self) -> ProviderPolicy | None:
        return self._policy

    @contextmanager
    def bind_request(
        self,
        *,
        request_id: str,
        trace_id: str,
        security_context: SecurityContext,
    ) -> Iterator[None]:
        if self._policy is None:
            raise ProviderPolicyError("provider_policy_missing")
        if self._audit_sink is None:
            raise ProviderPolicyError("audit_sink_missing")
        if self._redactor is None:
            raise ProviderPolicyError("provider_redaction_failed")
        context = ProviderRequestContext(
            request_id=request_id,
            trace_id=trace_id,
            principal_type=security_context.principal.principal_type.value,
            tenant_hash=_hash_identifier(security_context.principal.tenant_id),
            policy_id=self._policy.policy_id,
            policy_version=self._policy.version,
        )
        context_token = _REQUEST_CONTEXT.set(context)
        state_token = _REQUEST_STATE.set(ProviderRequestState())
        try:
            yield
        finally:
            _REQUEST_STATE.reset(state_token)
            _REQUEST_CONTEXT.reset(context_token)

    async def call(
        self,
        *,
        stage: ProviderStage,
        payload: Any,
        classification: DataClassification,
        evidence_count: int,
        transport: Callable[[Any], Awaitable[T]],
    ) -> T:
        """Run one provider transport only after its non-payload governance facts persist."""
        context = self._require_context()
        state = self._require_state()
        try:
            if self._redactor is None:
                raise ProviderPolicyError("provider_redaction_failed")
            sanitized, redaction_count = self._redactor.sanitize_payload(
                payload, classification=classification
            )
            payload_size = len(_canonical_size(sanitized))
            with state.lock:
                entry = require_provider_egress(
                    policy=self._policy,
                    stage=stage,
                    classification=classification,
                    payload_size=payload_size,
                    call_count=state.call_count,
                )
                if evidence_count > entry.max_evidence_items:
                    raise ProviderPolicyError("provider_payload_too_large")
                self._append(
                    context=context,
                    event_type="provider_call_allowed",
                    stage=stage,
                    outcome="allowed",
                    classification=classification,
                    redaction_count=redaction_count,
                    provider_call_count=state.call_count,
                )
                state.call_count += 1
        except ProviderPolicyError as exc:
            self._append_blocked(
                context=context,
                stage=stage,
                code=exc.code,
                classification=classification,
            )
            raise
        except AuditChainError as exc:
            raise ProviderPolicyError(_audit_code(exc)) from None

        started = monotonic()
        try:
            result = await transport(sanitized)
            if self._redactor is None:
                raise ProviderPolicyError("provider_redaction_failed")
            self._redactor.ensure_safe_output(
                _safe_output_text(result),
                allow_sql=stage is ProviderStage.DATA_PLANNING,
            )
        except ProviderPolicyError as exc:
            self._append_blocked(
                context=context,
                stage=stage,
                code=exc.code,
                classification=classification,
            )
            raise
        except Exception:
            self._append(
                context=context,
                event_type="provider_call_failed",
                stage=stage,
                outcome="failed",
                classification=classification,
                latency_ms=_elapsed_ms(started),
                error_code="provider_transport_failed",
            )
            raise
        self._append(
            context=context,
            event_type="provider_call_completed",
            stage=stage,
            outcome="completed",
            classification=classification,
            latency_ms=_elapsed_ms(started),
        )
        return result

    def audit(self, *, event_type: str, outcome: str, error_code: str | None = None) -> None:
        context = self._require_context()
        self._append(
            context=context,
            event_type=event_type,
            stage=None,
            outcome=outcome,
            classification=None,
            error_code=error_code,
        )

    def audit_for_request(
        self,
        *,
        request_id: str,
        trace_id: str,
        security_context: SecurityContext,
        event_type: str,
        outcome: str,
        error_code: str | None = None,
    ) -> None:
        """Emit a durable boundary event outside a provider-call scope."""
        with self.bind_request(
            request_id=request_id, trace_id=trace_id, security_context=security_context
        ):
            self.audit(event_type=event_type, outcome=outcome, error_code=error_code)

    def _require_context(self) -> ProviderRequestContext:
        context = _REQUEST_CONTEXT.get()
        if context is None:
            raise ProviderPolicyError("provider_policy_missing")
        if self._audit_sink is None:
            raise ProviderPolicyError("audit_sink_missing")
        return context

    @staticmethod
    def _require_state() -> ProviderRequestState:
        state = _REQUEST_STATE.get()
        if state is None:
            raise ProviderPolicyError("provider_policy_missing")
        return state

    def _append_blocked(
        self,
        *,
        context: ProviderRequestContext,
        stage: ProviderStage,
        code: str,
        classification: DataClassification,
    ) -> None:
        try:
            self._append(
                context=context,
                event_type="provider_call_blocked",
                stage=stage,
                outcome="blocked",
                classification=classification,
                error_code=code,
            )
        except AuditChainError as exc:
            raise ProviderPolicyError(_audit_code(exc)) from None

    def _append(
        self,
        *,
        context: ProviderRequestContext,
        event_type: str,
        outcome: str,
        stage: ProviderStage | None,
        classification: DataClassification | None,
        redaction_count: int | None = None,
        latency_ms: int | None = None,
        error_code: str | None = None,
        provider_call_count: int | None = None,
    ) -> None:
        if self._audit_sink is None:
            raise AuditChainError("audit_sink_missing")
        event = new_audit_event(
            event_id=str(uuid4()),
            request_id=context.request_id,
            trace_id=context.trace_id,
            principal_type=context.principal_type,
            tenant_hash=context.tenant_hash,
            event_type=event_type,
            component="provider",
            action="egress",
            policy_id=context.policy_id,
            policy_version=context.policy_version,
            outcome=outcome,
            error_code=error_code,
            provider_call_count=(
                self._require_state().call_count
                if provider_call_count is None
                else provider_call_count
            ),
            provider_stage=stage,
            data_classification=classification,
            redaction_status=None if redaction_count is None else f"redacted_{redaction_count}",
        )
        self._audit_sink.append(event)


def _canonical_size(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _safe_output_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _hash_identifier(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _elapsed_ms(started: float) -> int:
    return max(0, int((monotonic() - started) * 1_000))


def _audit_code(error: AuditChainError) -> str:
    return (
        "audit_chain_invalid"
        if "chain" in str(error) or "integrity" in str(error)
        else "audit_write_failed"
    )


class GovernedChatCompletionClient:
    """Role-fixed adapter that prevents a shared transport from choosing its own stage."""

    def __init__(
        self,
        *,
        client: Any,
        governance: ProviderGovernance,
        stage: ProviderStage,
        classification: DataClassification = DataClassification.CONFIDENTIAL,
    ) -> None:
        self._client = client
        self._governance = governance
        self._stage = stage
        self._classification = classification

    async def complete_chat(
        self, *, messages: list[dict[str, object]], response_format: dict[str, str]
    ) -> dict[str, Any]:
        payload = {"messages": messages, "response_format": response_format}

        async def transport(sanitized: Any) -> dict[str, Any]:
            assert isinstance(sanitized, dict)
            return await self._client.complete_chat(
                messages=sanitized["messages"], response_format=sanitized["response_format"]
            )

        return await self._governance.call(
            stage=self._stage,
            payload=payload,
            classification=self._classification,
            evidence_count=_evidence_count(messages),
            transport=transport,
        )

    def provider_trace_metadata(self) -> Any:
        return self._client.provider_trace_metadata()


class GovernedNativeToolCallingModel(GovernedChatCompletionClient):
    """Formal native-tool boundary with the fixed knowledge-answer egress stage."""

    async def complete(
        self,
        *,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
        tool_choice: str,
        response_format: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "response_format": response_format,
        }

        async def transport(sanitized: Any) -> dict[str, Any]:
            assert isinstance(sanitized, dict)
            return await self._client.complete(
                messages=sanitized["messages"],
                tools=sanitized["tools"],
                tool_choice=sanitized["tool_choice"],
                response_format=sanitized["response_format"],
            )

        return await self._governance.call(
            stage=self._stage,
            payload=payload,
            classification=self._classification,
            evidence_count=_evidence_count(messages),
            transport=transport,
        )


def _evidence_count(messages: list[dict[str, object]]) -> int:
    """Bound only structured evidence arrays; message text is governed by character budget."""
    return sum(
        len(message.get("evidence", ()))
        for message in messages
        if isinstance(message.get("evidence"), (list, tuple))
    )


class GovernedProviderRole:
    """Transparent role adapter for typed formal providers with a fixed stage."""

    _BYPASS_METHODS = frozenset({"provider_trace_metadata"})

    def __init__(
        self,
        *,
        provider: Any,
        governance: ProviderGovernance,
        stage: ProviderStage,
        classification: DataClassification,
        payload_projector: ProviderPayloadProjector | None = None,
    ) -> None:
        self._provider = provider
        self._governance = governance
        self._stage = stage
        self._classification = classification
        self._payload_projector = payload_projector

    def __getattr__(self, name: str) -> Any:
        target = getattr(self._provider, name)
        if name in self._BYPASS_METHODS or not callable(target):
            return target

        async def governed(*args: Any, **kwargs: Any) -> Any:
            if self._payload_projector is None:
                payload = {"args": list(args), "kwargs": kwargs}
                local_kwargs: dict[str, Any] = {}
                evidence_count = _evidence_count_from_payload(payload)
            else:
                payload, local_kwargs, evidence_count = self._payload_projector(args, kwargs)

            async def transport(sanitized: Any) -> Any:
                assert isinstance(sanitized, dict)
                provider_kwargs = dict(sanitized["kwargs"])
                if provider_kwargs.keys() & local_kwargs.keys():
                    raise ProviderPolicyError("provider_projection_failed")
                return await target(
                    *sanitized["args"],
                    **provider_kwargs,
                    **local_kwargs,
                )

            return await self._governance.call(
                stage=self._stage,
                payload=payload,
                classification=self._classification,
                evidence_count=evidence_count,
                transport=transport,
            )

        return governed


def _evidence_count_from_payload(value: Any) -> int:
    if isinstance(value, dict):
        return sum(_evidence_count_from_payload(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_evidence_count_from_payload(item) for item in value)
    return 1 if value.__class__.__name__.endswith("Evidence") else 0
