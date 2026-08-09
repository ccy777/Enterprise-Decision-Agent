"""Strict, dependency-free contracts for inventory-risk synthesis."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from decision_agent.config import Settings
from decision_agent.exceptions import DecisionAgentError
from decision_agent.observability.execution import (
    TraceSpanRecorder,
    complete_recorded_span,
    start_recorded_span,
)
from decision_agent.observability.models import TraceContext
from decision_agent.observability.provider import (
    ProviderTraceMetadata,
    provider_failure_attributes,
    provider_response_attributes,
)
from decision_agent.observability.stages import SpanStatus, TraceStage
from decision_agent.tool_calling.runtime import (
    NativeToolCallingError,
    OpenAICompatibleNativeToolCallingModel,
)

_DATA_CITATION = re.compile(r"^\[D\d+\]$")
_KNOWLEDGE_CITATION = re.compile(r"^\[E\d+\]$")
_SAFE_FAILURE_STAGES = frozenset(
    {
        "response_choices",
        "response_finish",
        "response_content",
        "response_json",
        "response_schema",
        "response_citations",
    }
)
_SAFE_FINISH_REASONS = frozenset({"stop", "length", "tool_calls", "content_filter"})
_SYNTHESIS_FIELDS = frozenset({"risk_summary", "policy_basis", "recommended_actions", "citations"})
_SAFE_SCHEMA_ERROR_LABEL = re.compile(r"^[a-z_]+:[a-z_]+$")


class InventoryRiskSynthesisInput(BaseModel):
    """The sole public information boundary for a mixed inventory synthesis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    original_request: str = Field(min_length=1, max_length=4_000)
    data_subquery: str = Field(min_length=1, max_length=2_000)
    data_answer: str = Field(min_length=1, max_length=8_000)
    data_citations: tuple[str, ...] = Field(min_length=1, max_length=100)
    knowledge_subquery: str = Field(min_length=1, max_length=2_000)
    knowledge_answer: str = Field(min_length=1, max_length=8_000)
    knowledge_citations: tuple[str, ...] = Field(min_length=1, max_length=100)

    @field_validator(
        "original_request",
        "data_subquery",
        "data_answer",
        "knowledge_subquery",
        "knowledge_answer",
    )
    @classmethod
    def _nonblank_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("text cannot be blank")
        return normalized

    @field_validator("data_citations", "knowledge_citations")
    @classmethod
    def _unique_citations(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(set(value)) != len(value):
            raise ValueError("citations must be nonempty and unique")
        return value

    @model_validator(mode="after")
    def _citation_domains(self) -> InventoryRiskSynthesisInput:
        if not all(_DATA_CITATION.fullmatch(citation) for citation in self.data_citations):
            raise ValueError("data_citations must use [D#]")
        if not all(
            _KNOWLEDGE_CITATION.fullmatch(citation) for citation in self.knowledge_citations
        ):
            raise ValueError("knowledge_citations must use [E#]")
        return self


class InventoryRiskSynthesisResult(BaseModel):
    """Structured synthesis content before deterministic public rendering."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    risk_summary: str = Field(min_length=1, max_length=8_000)
    policy_basis: str = Field(min_length=1, max_length=8_000)
    recommended_actions: tuple[str, ...] = Field(min_length=1, max_length=20)
    citations: tuple[str, ...] = Field(min_length=1, max_length=200)

    @field_validator("risk_summary", "policy_basis")
    @classmethod
    def _nonblank_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("text cannot be blank")
        return normalized

    @field_validator("recommended_actions")
    @classmethod
    def _nonblank_actions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not action.strip() for action in value):
            raise ValueError("recommended_actions must be nonempty")
        return tuple(action.strip() for action in value)

    @field_validator("citations")
    @classmethod
    def _valid_citations(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(
            not (_DATA_CITATION.fullmatch(citation) or _KNOWLEDGE_CITATION.fullmatch(citation))
            for citation in value
        ):
            raise ValueError("citations must use [D#] or [E#]")
        return value


class InventoryRiskSynthesizer(Protocol):
    """Injectable one-shot synthesizer with no tool or transport access."""

    async def synthesize(
        self,
        input_data: InventoryRiskSynthesisInput,
        *,
        trace_recorder: TraceSpanRecorder | None = None,
        trace_parent_context: TraceContext | None = None,
    ) -> InventoryRiskSynthesisResult:
        """Return only the strict public synthesis result."""


class OpenAICompatibleChatCompletionClient(Protocol):
    """The existing safe request boundary needed by ordinary synthesis only."""

    async def complete_chat(
        self,
        *,
        messages: list[dict[str, object]],
        response_format: dict[str, str],
    ) -> dict[str, Any]:
        """Return one OpenAI-compatible Chat Completions response."""


class InventoryRiskSynthesizerError(DecisionAgentError):
    """A safe provider failure without response, URL, credential, or traceback details."""

    def __init__(
        self,
        code: str,
        *,
        http_status: int | None = None,
        failure_stage: str | None = None,
        finish_reason: str | None = None,
        schema_error_fields: tuple[str, ...] = (),
    ) -> None:
        super().__init__("Inventory-risk synthesis could not be completed")
        if http_status is not None and not 100 <= http_status <= 599:
            raise ValueError("http_status must be a valid HTTP status code")
        if failure_stage is not None and failure_stage not in _SAFE_FAILURE_STAGES:
            raise ValueError("failure_stage must be a safe synthesis stage")
        if finish_reason is not None and finish_reason not in _SAFE_FINISH_REASONS:
            raise ValueError("finish_reason must be a safe provider termination value")
        if schema_error_fields and code != "inventory_risk_synthesizer_schema_invalid":
            raise ValueError("schema_error_fields require a schema validation failure")
        if not all(_SAFE_SCHEMA_ERROR_LABEL.fullmatch(field) for field in schema_error_fields):
            raise ValueError("schema_error_fields must be safe field:type labels")
        self.code = code
        self.http_status = http_status
        self.failure_stage = failure_stage
        self.finish_reason = finish_reason
        self.schema_error_fields = schema_error_fields


class OpenAICompatibleInventoryRiskSynthesizer:
    """One ordinary structured Chat Completions call over trusted mixed-skill evidence."""

    def __init__(self, *, client: OpenAICompatibleChatCompletionClient) -> None:
        self._client = client

    @classmethod
    def from_settings(cls, settings: Settings) -> OpenAICompatibleInventoryRiskSynthesizer:
        """Build only the existing safe OpenAI-compatible request adapter."""
        return cls(client=OpenAICompatibleNativeToolCallingModel.from_settings(settings))

    async def synthesize(
        self,
        input_data: InventoryRiskSynthesisInput,
        *,
        trace_recorder: TraceSpanRecorder | None = None,
        trace_parent_context: TraceContext | None = None,
    ) -> InventoryRiskSynthesisResult:
        operation = "generate_inventory_synthesis"
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
            response = await self._client.complete_chat(
                messages=_messages(input_data), response_format={"type": "json_object"}
            )
        except asyncio.CancelledError:
            complete_recorded_span(
                trace_recorder,
                provider_span,
                status=SpanStatus.CANCELLED,
                attributes=provider_failure_attributes(metadata=metadata, operation=operation),
            )
            raise
        except NativeToolCallingError as exc:
            mapped = _map_provider_error(exc)
            complete_recorded_span(
                trace_recorder,
                provider_span,
                status=SpanStatus.FAILED,
                error_code=mapped.code,
                attributes=provider_failure_attributes(metadata=metadata, operation=operation),
            )
            raise mapped from exc
        except (OSError, TimeoutError) as exc:
            complete_recorded_span(
                trace_recorder,
                provider_span,
                status=SpanStatus.FAILED,
                error_code="inventory_risk_synthesizer_unavailable",
                attributes=provider_failure_attributes(metadata=metadata, operation=operation),
            )
            raise InventoryRiskSynthesizerError("inventory_risk_synthesizer_unavailable") from exc
        except Exception as exc:
            complete_recorded_span(
                trace_recorder,
                provider_span,
                status=SpanStatus.FAILED,
                error_code="inventory_risk_synthesizer_invalid_response",
                attributes=provider_failure_attributes(metadata=metadata, operation=operation),
            )
            raise InventoryRiskSynthesizerError(
                "inventory_risk_synthesizer_invalid_response"
            ) from exc
        complete_recorded_span(
            trace_recorder,
            provider_span,
            status=SpanStatus.COMPLETED,
            attributes=provider_response_attributes(
                metadata=metadata, operation=operation, payload=response
            ),
        )
        return _parse_response(response, input_data)

    def provider_trace_metadata(self) -> ProviderTraceMetadata:
        """Return configured client metadata when its actual adapter exposes it."""
        metadata = getattr(self._client, "provider_trace_metadata", None)
        if callable(metadata):
            candidate = metadata()
            if isinstance(candidate, ProviderTraceMetadata):
                return candidate
        return ProviderTraceMetadata(provider=None, model=None, retry_count=None)


def _messages(input_data: InventoryRiskSynthesisInput) -> list[dict[str, object]]:
    return [
        {
            "role": "system",
            "content": (
                "Return exactly one JSON object with risk_summary, policy_basis, "
                "recommended_actions, and citations. Output JSON only, without Markdown. "
                'Use this exact shape: {"risk_summary":"...","policy_basis":"...",'
                '"recommended_actions":["..."],"citations":["[D1]","[E1]"]}. '
                "Use only the supplied trusted answers and citations. Do not create facts or "
                "citations. citations must include at least one [D#] and one [E#], and each must "
                "be copied from the supplied evidence."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original request:\n{input_data.original_request}\n\n"
                f"Data subquery:\n{input_data.data_subquery}\n"
                f"Trusted data answer:\n{input_data.data_answer}\n"
                f"Data citations:\n{', '.join(input_data.data_citations)}\n\n"
                f"Knowledge subquery:\n{input_data.knowledge_subquery}\n"
                f"Trusted knowledge answer:\n{input_data.knowledge_answer}\n"
                f"Knowledge citations:\n{', '.join(input_data.knowledge_citations)}"
            ),
        },
    ]


def _parse_response(
    response: Mapping[str, Any], input_data: InventoryRiskSynthesisInput
) -> InventoryRiskSynthesisResult:
    choices = response.get("choices") if isinstance(response, Mapping) else None
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
        raise InventoryRiskSynthesizerError(
            "inventory_risk_synthesizer_missing_choice", failure_stage="response_choices"
        )

    choice = choices[0]
    finish_reason = _safe_finish_reason(choice.get("finish_reason"))
    if choice.get("finish_reason") == "length":
        raise InventoryRiskSynthesizerError(
            "inventory_risk_synthesizer_truncated",
            failure_stage="response_finish",
            finish_reason="length",
        )
    if choice.get("finish_reason") != "stop":
        raise InventoryRiskSynthesizerError(
            "inventory_risk_synthesizer_invalid_finish_reason",
            failure_stage="response_finish",
            finish_reason=finish_reason,
        )

    try:
        content = choice["message"]["content"]
    except (KeyError, TypeError) as exc:
        raise InventoryRiskSynthesizerError(
            "inventory_risk_synthesizer_empty_content",
            failure_stage="response_content",
            finish_reason=finish_reason,
        ) from exc
    if not isinstance(content, str) or not content.strip():
        raise InventoryRiskSynthesizerError(
            "inventory_risk_synthesizer_empty_content",
            failure_stage="response_content",
            finish_reason=finish_reason,
        )
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise InventoryRiskSynthesizerError(
            "inventory_risk_synthesizer_invalid_json",
            failure_stage="response_json",
            finish_reason=finish_reason,
        ) from exc
    if not isinstance(parsed, dict):
        raise InventoryRiskSynthesizerError(
            "inventory_risk_synthesizer_schema_invalid",
            failure_stage="response_schema",
            finish_reason=finish_reason,
            schema_error_fields=("root:model_type",),
        )
    try:
        result = InventoryRiskSynthesisResult.model_validate(parsed)
    except ValidationError as exc:
        raise InventoryRiskSynthesizerError(
            "inventory_risk_synthesizer_schema_invalid",
            failure_stage="response_schema",
            finish_reason=finish_reason,
            schema_error_fields=_schema_error_fields(exc),
        ) from exc
    except TypeError as exc:
        raise InventoryRiskSynthesizerError(
            "inventory_risk_synthesizer_schema_invalid",
            failure_stage="response_schema",
            finish_reason=finish_reason,
            schema_error_fields=("root:invalid",),
        ) from exc
    if not _citations_are_bounded(result, input_data):
        raise InventoryRiskSynthesizerError(
            "inventory_risk_synthesis_citations_invalid",
            failure_stage="response_citations",
            finish_reason=finish_reason,
        )
    return result


def _safe_finish_reason(value: object) -> str | None:
    return value if isinstance(value, str) and value in _SAFE_FINISH_REASONS else None


def _schema_error_fields(error: ValidationError) -> tuple[str, ...]:
    """Return deduplicated field:type labels without retaining invalid provider values."""
    fields: list[str] = []
    for detail in error.errors():
        error_type = detail.get("type")
        safe_type = (
            error_type if isinstance(error_type, str) and error_type.isidentifier() else "invalid"
        )
        location = detail.get("loc")
        top_level = location[0] if isinstance(location, tuple) and location else None
        if error_type == "extra_forbidden":
            field = "extra_field"
        elif isinstance(top_level, str) and top_level in _SYNTHESIS_FIELDS:
            field = top_level
        else:
            field = "root"
        fields.append(f"{field}:{safe_type}")
    return tuple(dict.fromkeys(fields)) or ("root:invalid",)


def _citations_are_bounded(
    result: InventoryRiskSynthesisResult, input_data: InventoryRiskSynthesisInput
) -> bool:
    citations = set(result.citations)
    allowed = set(input_data.data_citations) | set(input_data.knowledge_citations)
    return (
        bool(citations)
        and citations <= allowed
        and any(_DATA_CITATION.fullmatch(citation) for citation in citations)
        and any(_KNOWLEDGE_CITATION.fullmatch(citation) for citation in citations)
    )


def _map_provider_error(error: NativeToolCallingError) -> InventoryRiskSynthesizerError:
    if error.code == "tool_calling_provider_http_error":
        return InventoryRiskSynthesizerError(
            "inventory_risk_synthesizer_http_error", http_status=error.http_status
        )
    if error.code == "tool_calling_provider_unavailable":
        return InventoryRiskSynthesizerError("inventory_risk_synthesizer_unavailable")
    return InventoryRiskSynthesizerError("inventory_risk_synthesizer_invalid_response")
