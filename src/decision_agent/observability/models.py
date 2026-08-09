"""Immutable, dependency-free contracts for safe request tracing."""

from __future__ import annotations

import math
import re
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from decision_agent.observability.stages import TERMINAL_SPAN_STATUSES, SpanStatus, TraceStage

TraceAttributeValue: TypeAlias = str | int | float | bool | None
TraceIdFactory: TypeAlias = Callable[[], str]

_IDENTIFIER_MAX_LENGTH = 128
_SAFE_NAME_MAX_LENGTH = 64
_SAFE_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def new_trace_id() -> str:
    """Generate a server-owned opaque trace identifier."""
    return str(uuid.uuid4())


def _nonblank_identifier(value: str, field_name: str) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def _aware_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


class TraceContext(BaseModel):
    """Immutable trace identity explicitly passed through future request boundaries."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    trace_id: str = Field(min_length=1, max_length=_IDENTIFIER_MAX_LENGTH)
    request_id: str = Field(min_length=1, max_length=_IDENTIFIER_MAX_LENGTH)
    session_present: bool
    current_span_id: str | None = Field(default=None, max_length=_IDENTIFIER_MAX_LENGTH)
    parent_span_id: str | None = Field(default=None, max_length=_IDENTIFIER_MAX_LENGTH)

    @field_validator("trace_id", "request_id", "current_span_id", "parent_span_id")
    @classmethod
    def _valid_identifiers(cls, value: str | None, info: object) -> str | None:
        if value is None:
            return None
        field_name = getattr(info, "field_name", "identifier")
        return _nonblank_identifier(value, field_name)

    @classmethod
    def create(
        cls,
        *,
        request_id: str,
        session_present: bool = False,
        id_factory: TraceIdFactory = new_trace_id,
    ) -> TraceContext:
        """Create a server-owned trace ID without accepting external trace identity."""
        return cls(trace_id=id_factory(), request_id=request_id, session_present=session_present)

    def for_child_span(self, span_id: str) -> TraceContext:
        """Return a new child context; the original context is never mutated."""
        return self.model_copy(
            update={"current_span_id": span_id, "parent_span_id": self.current_span_id}
        )


class TraceAttribute(BaseModel):
    """One already allowlisted, scalar, JSON-safe trace attribute."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = Field(min_length=1, max_length=_SAFE_NAME_MAX_LENGTH)
    value: TraceAttributeValue

    @field_validator("key")
    @classmethod
    def _valid_key(cls, value: str) -> str:
        return _nonblank_identifier(value, "attribute key").lower()

    @field_validator("value")
    @classmethod
    def _scalar_finite_value(cls, value: TraceAttributeValue) -> TraceAttributeValue:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("attribute floats must be finite")
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise ValueError("attribute values must be scalar JSON-safe values")


class TraceSpan(BaseModel):
    """Immutable completed span; running state exists only inside TraceCollector."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    span_id: str = Field(min_length=1, max_length=_IDENTIFIER_MAX_LENGTH)
    trace_id: str = Field(min_length=1, max_length=_IDENTIFIER_MAX_LENGTH)
    parent_span_id: str | None = Field(default=None, max_length=_IDENTIFIER_MAX_LENGTH)
    stage: TraceStage
    component: str = Field(min_length=1, max_length=_SAFE_NAME_MAX_LENGTH)
    operation: str = Field(min_length=1, max_length=_SAFE_NAME_MAX_LENGTH)
    started_at: datetime
    completed_at: datetime
    duration_ms: float = Field(ge=0, allow_inf_nan=False)
    status: SpanStatus
    error_code: str | None = Field(default=None, max_length=_SAFE_NAME_MAX_LENGTH)
    attributes: tuple[TraceAttribute, ...] = ()
    dropped_attribute_count: int = Field(default=0, ge=0)

    @field_validator(
        "span_id", "trace_id", "parent_span_id", "component", "operation", "error_code"
    )
    @classmethod
    def _valid_text(cls, value: str | None, info: object) -> str | None:
        if value is None:
            return None
        field_name = getattr(info, "field_name", "trace field")
        return _nonblank_identifier(value, field_name)

    @field_validator("component", "operation")
    @classmethod
    def _stable_safe_names(cls, value: str) -> str:
        if not _SAFE_NAME.fullmatch(value):
            raise ValueError("component and operation must be stable lowercase identifiers")
        return value

    @field_validator("started_at", "completed_at")
    @classmethod
    def _valid_times(cls, value: datetime, info: object) -> datetime:
        return _aware_utc(value, getattr(info, "field_name", "timestamp"))

    @model_validator(mode="after")
    def _completed_contract(self) -> TraceSpan:
        if self.status not in TERMINAL_SPAN_STATUSES:
            raise ValueError("TraceSpan must be terminal")
        if self.status is SpanStatus.FAILED and self.error_code is None:
            raise ValueError("failed spans require a safe error_code")
        if self.status is not SpanStatus.FAILED and self.error_code is not None:
            raise ValueError("only failed spans may carry an error_code")
        keys = tuple(attribute.key for attribute in self.attributes)
        if len(set(keys)) != len(keys):
            raise ValueError("TraceSpan attributes must have unique keys")
        return self


class RequestTrace(BaseModel):
    """Immutable final request-level snapshot emitted only after collection ends."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    trace_id: str = Field(min_length=1, max_length=_IDENTIFIER_MAX_LENGTH)
    request_id: str = Field(min_length=1, max_length=_IDENTIFIER_MAX_LENGTH)
    session_present: bool
    started_at: datetime
    completed_at: datetime
    duration_ms: float = Field(ge=0, allow_inf_nan=False)
    final_status: SpanStatus
    route: str | None = Field(default=None, max_length=_SAFE_NAME_MAX_LENGTH)
    skill_name: str | None = Field(default=None, max_length=_SAFE_NAME_MAX_LENGTH)
    error_code: str | None = Field(default=None, max_length=_SAFE_NAME_MAX_LENGTH)
    spans: tuple[TraceSpan, ...] = ()
    span_count: int = Field(ge=0)
    dropped_span_count: int = Field(default=0, ge=0)
    dropped_attribute_count: int = Field(default=0, ge=0)
    sink_failure_count: int = Field(default=0, ge=0)

    @field_validator("trace_id", "request_id", "route", "skill_name", "error_code")
    @classmethod
    def _valid_trace_text(cls, value: str | None, info: object) -> str | None:
        if value is None:
            return None
        return _nonblank_identifier(value, getattr(info, "field_name", "trace field"))

    @field_validator("started_at", "completed_at")
    @classmethod
    def _valid_trace_times(cls, value: datetime, info: object) -> datetime:
        return _aware_utc(value, getattr(info, "field_name", "timestamp"))

    @model_validator(mode="after")
    def _final_contract(self) -> RequestTrace:
        if self.final_status not in TERMINAL_SPAN_STATUSES:
            raise ValueError("RequestTrace final_status must be terminal")
        if self.span_count != len(self.spans):
            raise ValueError("span_count must equal the number of spans")
        if any(span.trace_id != self.trace_id for span in self.spans):
            raise ValueError("all spans must belong to the request trace")
        span_ids = tuple(span.span_id for span in self.spans)
        if len(set(span_ids)) != len(span_ids):
            raise ValueError("span IDs must be unique")
        if self.final_status is SpanStatus.FAILED and self.error_code is None:
            raise ValueError("failed traces require a safe error_code")
        if self.final_status is not SpanStatus.FAILED and self.error_code is not None:
            raise ValueError("only failed traces may carry an error_code")
        return self


class TraceStageSummary(BaseModel):
    """Bounded public projection of one completed span for the current response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: TraceStage
    operation: str = Field(min_length=1, max_length=_SAFE_NAME_MAX_LENGTH)
    status: SpanStatus
    duration_ms: float = Field(ge=0, allow_inf_nan=False)
    error_code: str | None = Field(default=None, max_length=_SAFE_NAME_MAX_LENGTH)
    attributes: tuple[TraceAttribute, ...] = ()

    @field_validator("operation", "error_code")
    @classmethod
    def _safe_summary_text(cls, value: str | None, info: object) -> str | None:
        if value is None:
            return None
        field_name = getattr(info, "field_name", "summary field")
        return _nonblank_identifier(value, field_name)


class TraceSummary(BaseModel):
    """Read-only, bounded, payload-free response projection of one RequestTrace."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    trace_id: str = Field(min_length=1, max_length=_IDENTIFIER_MAX_LENGTH)
    request_id: str = Field(min_length=1, max_length=_IDENTIFIER_MAX_LENGTH)
    final_status: SpanStatus
    duration_ms: float = Field(ge=0, allow_inf_nan=False)
    span_count: int = Field(ge=0)
    dropped_span_count: int = Field(ge=0)
    dropped_attribute_count: int = Field(ge=0)
    sink_failure_count: int = Field(ge=0)
    truncated_stage_count: int = Field(ge=0)
    stages: tuple[TraceStageSummary, ...] = ()

    @field_validator("trace_id", "request_id")
    @classmethod
    def _summary_identifiers(cls, value: str, info: object) -> str:
        return _nonblank_identifier(value, getattr(info, "field_name", "summary identifier"))
