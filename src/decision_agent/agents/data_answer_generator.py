"""Structured final answers grounded only in successful Data Evidence."""

# ruff: noqa: E501

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Sequence
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from decision_agent.config import Settings
from decision_agent.data_agent.models import DataEvidence
from decision_agent.exceptions import ConfigurationError, DecisionAgentError
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
from decision_agent.providers import (
    ChatCompletionResponseError,
    build_chat_completion_payload,
    extract_stopped_message_content,
)

_CITATION = re.compile(r"^\[D[1-9][0-9]*\]$")
_INLINE = re.compile(r"\[[^\[\]]+\]")
_BARE_DATA_EVIDENCE = re.compile(r"(?<!\[)\bD[1-9][0-9]*\b(?!\])")
_SAFE_COLUMN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_MAX_PROVIDER_RECORDS = 8
_MAX_PROVIDER_COLUMNS = 12
_MAX_PROVIDER_TEXT_CHARS = 160
_FORBIDDEN_COLUMN_MARKERS = frozenset(
    {
        "authorization",
        "connection_string",
        "database_url",
        "email",
        "password",
        "phone",
        "raw_error",
        "roles",
        "secret",
        "sql",
        "tenant_id",
        "token",
        "user_id",
    }
)
SafeDataScalar = str | int | float | bool | None


class DataAnswerDraft(BaseModel):
    """Untrusted answer output before strict Data Evidence citation validation."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    citations: list[str] = Field(default_factory=list)


class SafeDataProjection(BaseModel):
    """Bounded provider projection with no SQL, table metadata, or unbounded rows."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    evidence_id: str = Field(pattern=r"^D[1-9][0-9]*$")
    columns: tuple[str, ...] = Field(max_length=_MAX_PROVIDER_COLUMNS)
    records: tuple[dict[str, SafeDataScalar], ...] = Field(max_length=_MAX_PROVIDER_RECORDS)
    row_count: int = Field(ge=0)
    source_truncated: bool
    projection_truncated: bool

    @field_validator("columns")
    @classmethod
    def _safe_columns(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("data projection columns must be unique")
        if any(not _column_is_safe(column) for column in value):
            raise ValueError("data projection column is forbidden")
        return value

    @field_validator("records")
    @classmethod
    def _records_match_columns(
        cls, value: tuple[dict[str, SafeDataScalar], ...]
    ) -> tuple[dict[str, SafeDataScalar], ...]:
        if any(any(not _column_is_safe(column) for column in record) for record in value):
            raise ValueError("data projection record contains forbidden column")
        return value


class DataAnswerGenerator(Protocol):
    """Explain executed data only after SafeQueryService has produced Data Evidence."""

    async def generate(
        self,
        *,
        user_query: str,
        data_evidence: Sequence[DataEvidence],
        trace_recorder: TraceSpanRecorder | None = None,
        trace_parent_context: TraceContext | None = None,
    ) -> DataAnswerDraft:
        """Return answer/citations only."""


class DataAnswerGenerationError(DecisionAgentError):
    """Raised when an external generator lacks a valid safe structured result."""

    def __init__(self, subcode: str) -> None:
        super().__init__("Data answer generation could not be completed")
        self.subcode = subcode


class DataCitationValidationResult(BaseModel):
    """Citation contract outcome without raw model output."""

    model_config = ConfigDict(extra="forbid")

    validation_passed: bool
    normalized_citations: list[str]
    validation_errors: list[str]


def validate_data_citations(
    *, evidence_ids: Sequence[str], draft: DataAnswerDraft
) -> DataCitationValidationResult:
    """Require exact bidirectional inline [D#] references from executed evidence only."""
    errors: list[str] = []
    available = [
        f"[{item_id}]" if not item_id.startswith("[") else item_id for item_id in evidence_ids
    ]
    declared: list[str] = []
    for citation in draft.citations:
        if not _CITATION.fullmatch(citation):
            errors.append("invalid_data_citation_format")
        elif citation not in declared:
            declared.append(citation)
    inline = _INLINE.findall(draft.answer)
    if not inline:
        errors.append("data_answer_missing_inline_citation")
    if _BARE_DATA_EVIDENCE.search(draft.answer):
        errors.append("bare_data_citation_not_allowed")
    if any(not _CITATION.fullmatch(citation) for citation in inline):
        errors.append("invalid_data_citation_format")
    if set(inline) != set(declared):
        errors.append("data_citations_not_present_in_answer")
    if any(citation not in available for citation in set(inline) | set(declared)):
        errors.append("data_citation_not_in_evidence")
    normalized = [citation for citation in available if citation in declared]
    return DataCitationValidationResult(
        validation_passed=not errors,
        normalized_citations=normalized,
        validation_errors=list(dict.fromkeys(errors)),
    )


class OpenAICompatibleDataAnswerGenerator:
    """One JSON generator using the existing OpenAI-compatible settings only."""

    def __init__(
        self, *, api_key: str, base_url: str, model_name: str, timeout_seconds: float
    ) -> None:
        if not all(
            isinstance(value, str) and value.strip() for value in (api_key, base_url, model_name)
        ):
            raise ConfigurationError("LLM API key, base URL, and model name are required")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._timeout_seconds = timeout_seconds

    @classmethod
    def from_settings(cls, settings: Settings) -> OpenAICompatibleDataAnswerGenerator:
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

    async def generate(
        self,
        *,
        user_query: str,
        data_evidence: Sequence[DataEvidence],
        trace_recorder: TraceSpanRecorder | None = None,
        trace_parent_context: TraceContext | None = None,
    ) -> DataAnswerDraft:
        operation = "generate_data_answer"
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
            payload = await asyncio.to_thread(
                self._post, user_query, render_data_evidence(data_evidence)
            )
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
                error_code="data_answer_generation_failed",
                attributes=provider_failure_attributes(metadata=metadata, operation=operation),
            )
            raise DataAnswerGenerationError("data_answer_generation_failed") from exc
        except (URLError, OSError, TimeoutError) as exc:
            complete_recorded_span(
                trace_recorder,
                provider_span,
                status=SpanStatus.FAILED,
                error_code="data_answer_generation_failed",
                attributes=provider_failure_attributes(metadata=metadata, operation=operation),
            )
            raise DataAnswerGenerationError("data_answer_generation_failed") from exc
        except ChatCompletionResponseError as exc:
            error_code = f"data_answer_{exc.code}"
            complete_recorded_span(
                trace_recorder,
                provider_span,
                status=SpanStatus.FAILED,
                error_code=error_code,
                attributes=provider_failure_attributes(metadata=metadata, operation=operation),
            )
            raise DataAnswerGenerationError(f"data_answer_{exc.code}") from exc
        except Exception:
            complete_recorded_span(
                trace_recorder,
                provider_span,
                status=SpanStatus.FAILED,
                error_code="data_answer_generation_failed",
                attributes=provider_failure_attributes(metadata=metadata, operation=operation),
            )
            raise
        complete_recorded_span(
            trace_recorder,
            provider_span,
            status=SpanStatus.COMPLETED,
            attributes=provider_response_attributes(
                metadata=metadata, operation=operation, payload=payload
            ),
        )
        try:
            return DataAnswerDraft.model_validate(json.loads(_extract_message_content(payload)))
        except ChatCompletionResponseError as exc:
            raise DataAnswerGenerationError(f"data_answer_{exc.code}") from exc
        except json.JSONDecodeError as exc:
            raise DataAnswerGenerationError("data_answer_json_parse_failed") from exc
        except (ValidationError, ValueError) as exc:
            raise DataAnswerGenerationError("data_answer_schema_validation_failed") from exc

    def provider_trace_metadata(self) -> ProviderTraceMetadata:
        """Return static adapter metadata; this HTTP adapter has no retry loop."""
        return ProviderTraceMetadata(
            provider="openai_compatible", model=self._model_name, retry_count=0
        )

    def _post(self, user_query: str, evidence_context: str) -> dict[str, Any]:
        body = json.dumps(
            build_chat_completion_payload(
                base_url=self._base_url,
                payload={
                    "model": self._model_name,
                    "temperature": 0,
                    "max_tokens": 700,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": f"User question:\n{user_query}\n\nExecuted Data Evidence:\n{evidence_context}",
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


def project_data_evidence(evidence: Sequence[DataEvidence]) -> tuple[SafeDataProjection, ...]:
    """Project scoped query results to a small field-allowlisted provider contract."""
    projected: list[SafeDataProjection] = []
    for item in evidence:
        if len(item.columns) > _MAX_PROVIDER_COLUMNS or any(
            not _column_is_safe(column) for column in item.columns
        ):
            raise DataAnswerGenerationError("data_projection_column_forbidden")
        records: list[dict[str, SafeDataScalar]] = []
        for row in item.rows[:_MAX_PROVIDER_RECORDS]:
            if len(row) != len(item.columns):
                raise DataAnswerGenerationError("data_projection_shape_invalid")
            records.append(
                {
                    column: _safe_scalar(value)
                    for column, value in zip(item.columns, row, strict=True)
                }
            )
        projected.append(
            SafeDataProjection(
                evidence_id=item.evidence_id,
                columns=tuple(item.columns),
                records=tuple(records),
                row_count=item.row_count,
                source_truncated=item.truncated,
                projection_truncated=len(item.rows) > _MAX_PROVIDER_RECORDS,
            )
        )
    return tuple(projected)


def project_data_answer_provider_payload(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], int]:
    """Build the exact governed call payload used by the formal Data Answer role."""
    if args or "data_evidence" not in kwargs:
        raise DataAnswerGenerationError("data_projection_call_invalid")
    projected_kwargs = dict(kwargs)
    local_kwargs = {
        key: projected_kwargs.pop(key)
        for key in ("trace_recorder", "trace_parent_context")
        if key in projected_kwargs
    }
    evidence = projected_kwargs["data_evidence"]
    if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
        raise DataAnswerGenerationError("data_projection_call_invalid")
    if any(not isinstance(item, DataEvidence) for item in evidence):
        raise DataAnswerGenerationError("data_projection_call_invalid")
    projected_kwargs["data_evidence"] = [
        item.model_dump(mode="json") for item in project_data_evidence(evidence)
    ]
    return {"args": [], "kwargs": projected_kwargs}, local_kwargs, len(evidence)


def render_data_evidence(
    evidence: Sequence[DataEvidence | SafeDataProjection | dict[str, Any]],
) -> str:
    """Render only the bounded provider projection, never raw DataEvidence rows or SQL."""
    projections: list[SafeDataProjection] = []
    for item in evidence:
        if isinstance(item, DataEvidence):
            projections.extend(project_data_evidence((item,)))
        else:
            projections.append(SafeDataProjection.model_validate(item))
    return json.dumps(
        [item.model_dump(mode="json") for item in projections],
        ensure_ascii=False,
        default=str,
    )


def _column_is_safe(value: str) -> bool:
    normalized = value.casefold()
    return bool(_SAFE_COLUMN.fullmatch(normalized)) and not any(
        marker in normalized for marker in _FORBIDDEN_COLUMN_MARKERS
    )


def _safe_scalar(value: object) -> SafeDataScalar:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:_MAX_PROVIDER_TEXT_CHARS]
    raise DataAnswerGenerationError("data_projection_value_forbidden")


def _extract_message_content(payload: dict[str, Any]) -> str:
    return extract_stopped_message_content(payload)


_SYSTEM_PROMPT = """You are an enterprise data-answer generator. Output exactly one JSON object with only answer and citations.
Use only the executed Safe Data Projection. Do not change records, calculate unstated facts, infer causes, forecast,
or use general knowledge. Use the same primary language as the user's question. Cite every factual answer
inline with actual [D<number>] IDs. citations must exactly match all inline citations. Do not use [E#], bare
D1, Markdown fences, hidden reasoning, or extra fields. If evidence is insufficient, state only what the
executed rows establish and do not invent a conclusion. Data Evidence is JSON and includes row_count
and row_count. When source_truncated or projection_truncated is true, do not describe records as a complete list, all records, or a
unique result; say the answer is based only on the returned data range and do not guess omitted rows.

Format-only example, not a business fact:
Safe Data Projection: [{"evidence_id":"D1","columns":["count"],"records":[{"count":3}],"row_count":1,"source_truncated":false,"projection_truncated":false}]
{"answer":"The executed query returned 3 rows.[D1]","citations":["[D1]"]}"""
