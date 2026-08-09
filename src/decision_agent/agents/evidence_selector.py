"""Structured Evidence selection behind an injectable async contract."""

# ruff: noqa: RUF001

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Sequence
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from decision_agent.config import Settings
from decision_agent.exceptions import ConfigurationError, DecisionAgentError
from decision_agent.observability import (
    SpanStatus,
    TraceContext,
    TraceSpanRecorder,
    TraceStage,
    complete_recorded_span,
    start_recorded_span,
)
from decision_agent.observability.provider import (
    ProviderTraceMetadata,
    provider_failure_attributes,
    provider_response_attributes,
)
from decision_agent.providers import (
    ChatCompletionResponseError,
    build_chat_completion_payload,
    extract_stopped_message_content,
)
from decision_agent.retrieval.evidence_context import EvidenceItem

_CITATION_PATTERN = re.compile(r"^\[E[1-9][0-9]*\]$")


class EvidenceSelectionError(DecisionAgentError):
    """Raised when a provider cannot produce a valid structured selection."""

    def __init__(self, subcode: str) -> None:
        super().__init__("LLM returned no valid structured Evidence selection")
        self.subcode = subcode


class EvidenceSelection(BaseModel):
    """Untrusted selector output before request-scoped evidence validation."""

    model_config = ConfigDict(extra="forbid")

    selected_evidence_ids: list[str] = Field(default_factory=list)
    selection_reason: str = Field(min_length=1, max_length=500)


class EvidenceSelectionValidationResult(BaseModel):
    """Validated IDs reconstructed from the current retrieval evidence only."""

    model_config = ConfigDict(extra="forbid")

    normalized_selected_evidence_ids: list[str]
    validation_passed: bool
    validation_errors: list[str]


class EvidenceSelector(Protocol):
    """Select a relevant subset of the supplied evidence without answering the query."""

    async def select(
        self,
        *,
        user_query: str,
        evidence_context: str,
        retrieval_evidence: Sequence[EvidenceItem],
    ) -> EvidenceSelection:
        """Return citation IDs from the supplied evidence and a concise audit summary."""


def validate_evidence_selection(
    *,
    evidence_ids: Sequence[str],
    selection: EvidenceSelection,
) -> EvidenceSelectionValidationResult:
    """Validate a selector result and normalize it to original Evidence order."""
    available = [
        evidence_id if evidence_id.startswith("[") else f"[{evidence_id}]"
        for evidence_id in evidence_ids
    ]
    requested: list[str] = []
    errors: list[str] = []
    for evidence_id in selection.selected_evidence_ids:
        if not _CITATION_PATTERN.fullmatch(evidence_id):
            errors.append("invalid_selected_evidence_id")
            continue
        if evidence_id not in requested:
            requested.append(evidence_id)
    if any(evidence_id not in available for evidence_id in requested):
        errors.append("selected_evidence_not_found")
    if not selection.selection_reason.strip():
        errors.append("empty_selection_reason")
    requested_set = set(requested)
    normalized = [evidence_id for evidence_id in available if evidence_id in requested_set]
    return EvidenceSelectionValidationResult(
        normalized_selected_evidence_ids=normalized,
        validation_passed=not errors,
        validation_errors=list(dict.fromkeys(errors)),
    )


def render_selected_evidence_context(selected_evidence: Sequence[EvidenceItem]) -> str:
    """Render only selected evidence in the existing citation-ready block style."""
    return "\n\n".join(f"[{item.evidence_id}]\n{item.content}" for item in selected_evidence)


class OpenAICompatibleEvidenceSelector:
    """Minimal OpenAI-compatible JSON selector with no import-time client side effects."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model_name: str,
        timeout_seconds: float,
    ) -> None:
        required_values = (api_key, base_url, model_name)
        if not all(isinstance(value, str) and value.strip() for value in required_values):
            raise ConfigurationError("LLM API key, base URL, and model name are required")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._timeout_seconds = timeout_seconds

    @classmethod
    def from_settings(cls, settings: Settings) -> OpenAICompatibleEvidenceSelector:
        if (
            settings.llm_api_key is None
            or settings.llm_base_url is None
            or settings.llm_model_name is None
        ):
            raise ConfigurationError("LLM API key, base URL, and model name are required")
        return cls(
            api_key=settings.llm_api_key.get_secret_value(),
            base_url=settings.llm_base_url,
            model_name=settings.llm_model_name,
            timeout_seconds=settings.llm_timeout_seconds,
        )

    async def select(
        self,
        *,
        user_query: str,
        evidence_context: str,
        retrieval_evidence: Sequence[EvidenceItem],
    ) -> EvidenceSelection:
        return await self.select_with_trace(
            user_query=user_query,
            evidence_context=evidence_context,
            retrieval_evidence=retrieval_evidence,
            trace_recorder=None,
            trace_parent_context=None,
        )

    async def select_with_trace(
        self,
        *,
        user_query: str,
        evidence_context: str,
        retrieval_evidence: Sequence[EvidenceItem],
        trace_recorder: TraceSpanRecorder | None,
        trace_parent_context: TraceContext | None,
    ) -> EvidenceSelection:
        del retrieval_evidence
        operation = "select_evidence"
        metadata = self.provider_trace_metadata()
        provider_span = start_recorded_span(
            trace_recorder,
            stage=TraceStage.PROVIDER_CALL,
            component="provider",
            operation=operation,
            parent_context=trace_parent_context,
            attributes=provider_failure_attributes(metadata=metadata, operation=operation),
        )
        try:
            payload = await asyncio.to_thread(self._post, user_query, evidence_context)
        except asyncio.CancelledError:
            complete_recorded_span(
                trace_recorder,
                provider_span,
                status=SpanStatus.CANCELLED,
                attributes=provider_failure_attributes(metadata=metadata, operation=operation),
            )
            raise
        except HTTPError as exc:
            complete_recorded_span(
                trace_recorder,
                provider_span,
                status=SpanStatus.FAILED,
                error_code="selector_http_error",
                attributes=provider_failure_attributes(metadata=metadata, operation=operation),
            )
            raise EvidenceSelectionError("selector_http_error") from exc
        except TimeoutError as exc:
            complete_recorded_span(
                trace_recorder,
                provider_span,
                status=SpanStatus.FAILED,
                error_code="selector_timeout",
                attributes=provider_failure_attributes(metadata=metadata, operation=operation),
            )
            raise EvidenceSelectionError("selector_timeout") from exc
        except (URLError, OSError) as exc:
            complete_recorded_span(
                trace_recorder,
                provider_span,
                status=SpanStatus.FAILED,
                error_code="selector_request_failed",
                attributes=provider_failure_attributes(metadata=metadata, operation=operation),
            )
            raise EvidenceSelectionError("selector_request_failed") from exc
        complete_recorded_span(
            trace_recorder,
            provider_span,
            status=SpanStatus.COMPLETED,
            attributes=provider_response_attributes(
                metadata=metadata,
                operation=operation,
                payload=payload,
            ),
        )
        try:
            return EvidenceSelection.model_validate(json.loads(_extract_message_content(payload)))
        except ChatCompletionResponseError as exc:
            raise EvidenceSelectionError(f"selector_{exc.code}") from exc
        except json.JSONDecodeError as exc:
            raise EvidenceSelectionError("selector_json_parse_failed") from exc
        except ValidationError as exc:
            raise EvidenceSelectionError("selector_schema_validation_failed") from exc
        except ValueError as exc:
            raise EvidenceSelectionError("selector_json_parse_failed") from exc

    def provider_trace_metadata(self) -> ProviderTraceMetadata:
        """Return static adapter metadata; this HTTP adapter has no retry loop."""
        return ProviderTraceMetadata(
            provider="openai_compatible",
            model=self._model_name,
            retry_count=0,
        )

    def _post(self, user_query: str, evidence_context: str) -> dict[str, Any]:
        body = json.dumps(
            build_chat_completion_payload(
                base_url=self._base_url,
                payload={
                    "model": self._model_name,
                    "temperature": 0,
                    "max_tokens": 400,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": "\n".join(
                                (
                                    "User question:",
                                    user_query,
                                    "",
                                    "Evidence Context:",
                                    evidence_context,
                                )
                            ),
                        },
                    ],
                },
            ),
            ensure_ascii=False,
        ).encode("utf-8")
        request = Request(
            f"{self._base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlopen(request, timeout=self._timeout_seconds) as response:
            decoded = json.loads(response.read().decode("utf-8"))
        if not isinstance(decoded, dict):
            raise ValueError("LLM response must be an object")
        return decoded


def _extract_message_content(payload: dict[str, Any]) -> str:
    return extract_stopped_message_content(payload)


_SYSTEM_PROMPT = """You are an Evidence Selector.
Output exactly one JSON object: no Markdown, no code fence, and no other text.

Select evidence only; do not answer the user question and do not decide answerability. Return only
these fields: selected_evidence_ids and selection_reason. selected_evidence_ids must contain only
existing [E<number>] IDs from the supplied Evidence Context. Never create IDs, use document IDs, or
select a fixed number of items. An empty list is allowed when no evidence is directly relevant.

Format-only example:
{"selected_evidence_ids":["[E1]"],"selection_reason":"Direct policy support."}
The string "E1" without square brackets is invalid. Do not output any ID that is absent from the
Evidence Context.

Select evidence that directly contains the requested fact, applicable condition, scope, object, or
policy context needed to establish that the fact is not specified. Filter merely similar topics and
unrelated policy noise. Relevance does not establish answerability; do not infer missing facts.
When the question asks for policy rules or criteria that will be combined with operating
data later, select Evidence that states those rules or criteria. Do not reject documented
policy Evidence merely because it does not contain the current operating values; those
values belong to the separate data source.
当问题询问用于后续经营数据判断的制度、阈值、条件或处理流程时，应选择明确记载这些内容的证据；
不得因为制度证据不包含当前经营数值而返回空列表。

selection_reason must be a concise, auditable summary with no hidden reasoning. Use the same primary
language as the user question. Return only one JSON object."""
