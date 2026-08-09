"""Request-local trace collection with explicit lifecycle contracts."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

from decision_agent.observability.attributes import AttributeLimits, sanitize_attributes
from decision_agent.observability.models import (
    RequestTrace,
    TraceContext,
    TraceIdFactory,
    TraceSpan,
)
from decision_agent.observability.stages import TERMINAL_SPAN_STATUSES, SpanStatus, TraceStage


class TraceLifecycleError(RuntimeError):
    """Programming-contract failure in explicit request-local trace collection."""


@dataclass(frozen=True, slots=True)
class TraceLimits:
    """Trace retention bounds that prevent one request from growing without limit."""

    max_spans_per_trace: int = 64
    max_attributes_per_span: int = 24
    max_attribute_value_length: int = 256
    max_serialized_trace_bytes: int = 65_536
    max_in_memory_traces: int = 256

    def __post_init__(self) -> None:
        if any(
            value <= 0
            for value in (
                self.max_spans_per_trace,
                self.max_attributes_per_span,
                self.max_attribute_value_length,
                self.max_serialized_trace_bytes,
                self.max_in_memory_traces,
            )
        ):
            raise ValueError("trace limits must be positive")

    @property
    def attribute_limits(self) -> AttributeLimits:
        return AttributeLimits(
            max_attributes_per_span=self.max_attributes_per_span,
            max_attribute_value_length=self.max_attribute_value_length,
        )


@dataclass(slots=True)
class _OpenSpan:
    context: TraceContext
    stage: TraceStage
    component: str
    operation: str
    started_at: datetime
    started_monotonic: float
    attributes: dict[str, object]
    dropped_attribute_count: int


class TraceCollector:
    """Collect one trace explicitly; instances are never shared across requests."""

    def __init__(
        self,
        *,
        context: TraceContext,
        limits: TraceLimits | None = None,
        utc_now: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
        id_factory: TraceIdFactory,
    ) -> None:
        self._context = context
        self._limits = limits or TraceLimits()
        self._utc_now = utc_now or (lambda: datetime.now(UTC))
        self._monotonic = monotonic or time.monotonic
        self._id_factory = id_factory
        self._started_at = self._require_utc(self._utc_now())
        self._started_monotonic = self._monotonic()
        self._open_spans: dict[str, _OpenSpan] = {}
        self._closed_spans: list[TraceSpan] = []
        self._span_order: dict[str, int] = {}
        self._next_span_order = 0
        self._dropped_span_ids: set[str] = set()
        self._dropped_span_count = 0
        self._dropped_attribute_count = 0
        self._finalized = False

    @property
    def context(self) -> TraceContext:
        return self._context

    def start_span(
        self,
        *,
        stage: TraceStage,
        component: str,
        operation: str,
        parent_context: TraceContext | None = None,
        attributes: Mapping[str, object] | None = None,
    ) -> TraceContext:
        """Start one bounded root/child span and return its immutable child context."""
        self._require_open()
        parent_span_id = self._parent_span_id(parent_context)
        span_id = self._id_factory()
        child_context = TraceContext(
            trace_id=self._context.trace_id,
            request_id=self._context.request_id,
            session_present=self._context.session_present,
            current_span_id=span_id,
            parent_span_id=parent_span_id,
        )
        if len(self._closed_spans) + len(self._open_spans) >= self._limits.max_spans_per_trace:
            self._dropped_span_ids.add(span_id)
            self._dropped_span_count += 1
            return child_context
        sanitized = sanitize_attributes(
            stage=stage, values=attributes, limits=self._limits.attribute_limits
        )
        self._dropped_attribute_count += sanitized.dropped_attribute_count
        self._open_spans[span_id] = _OpenSpan(
            context=child_context,
            stage=stage,
            component=component,
            operation=operation,
            started_at=self._require_utc(self._utc_now()),
            started_monotonic=self._monotonic(),
            attributes={attribute.key: attribute.value for attribute in sanitized.attributes},
            dropped_attribute_count=sanitized.dropped_attribute_count,
        )
        self._span_order[span_id] = self._next_span_order
        self._next_span_order += 1
        return child_context

    def complete_span(
        self,
        context: TraceContext,
        *,
        status: SpanStatus = SpanStatus.COMPLETED,
        error_code: str | None = None,
        attributes: Mapping[str, object] | None = None,
    ) -> TraceSpan | None:
        """Close a recorded span exactly once; dropped-capacity spans remain a no-op."""
        self._require_open()
        span_id = self._require_context(context)
        if span_id in self._dropped_span_ids:
            self._dropped_span_ids.remove(span_id)
            return None
        if status not in TERMINAL_SPAN_STATUSES:
            raise TraceLifecycleError("a span must close with a terminal status")
        open_span = self._open_spans.pop(span_id, None)
        if open_span is None:
            raise TraceLifecycleError("unknown or already completed span")
        merged_attributes = dict(open_span.attributes)
        if attributes:
            merged_attributes.update(attributes)
        sanitized = sanitize_attributes(
            stage=open_span.stage,
            values=merged_attributes,
            limits=self._limits.attribute_limits,
        )
        dropped = open_span.dropped_attribute_count + sanitized.dropped_attribute_count
        self._dropped_attribute_count += sanitized.dropped_attribute_count
        completed_at = self._require_utc(self._utc_now())
        duration_ms = max(0.0, (self._monotonic() - open_span.started_monotonic) * 1000)
        span = TraceSpan(
            span_id=span_id,
            trace_id=self._context.trace_id,
            parent_span_id=open_span.context.parent_span_id,
            stage=open_span.stage,
            component=open_span.component,
            operation=open_span.operation,
            started_at=open_span.started_at,
            completed_at=completed_at,
            duration_ms=duration_ms,
            status=status,
            error_code=error_code,
            attributes=sanitized.attributes,
            dropped_attribute_count=dropped,
        )
        self._closed_spans.append(span)
        return span

    def finalize(
        self,
        *,
        final_status: SpanStatus,
        route: str | None = None,
        skill_name: str | None = None,
        error_code: str | None = None,
    ) -> RequestTrace:
        """Close any leaked spans safely and return one immutable request snapshot."""
        self._require_open()
        if final_status not in TERMINAL_SPAN_STATUSES:
            raise TraceLifecycleError("a trace must finalize with a terminal status")
        for span_id in tuple(self._open_spans):
            open_span = self._open_spans.pop(span_id)
            completed_at = self._require_utc(self._utc_now())
            sanitized = sanitize_attributes(
                stage=open_span.stage,
                values=open_span.attributes,
                limits=self._limits.attribute_limits,
            )
            dropped = open_span.dropped_attribute_count + sanitized.dropped_attribute_count
            self._dropped_attribute_count += sanitized.dropped_attribute_count
            span = TraceSpan(
                span_id=span_id,
                trace_id=self._context.trace_id,
                parent_span_id=open_span.context.parent_span_id,
                stage=open_span.stage,
                component=open_span.component,
                operation=open_span.operation,
                started_at=open_span.started_at,
                completed_at=completed_at,
                duration_ms=max(0.0, (self._monotonic() - open_span.started_monotonic) * 1000),
                status=SpanStatus.FAILED,
                error_code="observability_span_unclosed",
                attributes=sanitized.attributes,
                dropped_attribute_count=dropped,
            )
            self._closed_spans.append(span)
        completed_at = self._require_utc(self._utc_now())
        trace = RequestTrace(
            trace_id=self._context.trace_id,
            request_id=self._context.request_id,
            session_present=self._context.session_present,
            started_at=self._started_at,
            completed_at=completed_at,
            duration_ms=max(0.0, (self._monotonic() - self._started_monotonic) * 1000),
            final_status=final_status,
            route=route,
            skill_name=skill_name,
            error_code=error_code,
            spans=self._ordered_spans(),
            span_count=len(self._closed_spans),
            dropped_span_count=self._dropped_span_count,
            dropped_attribute_count=self._dropped_attribute_count,
        )
        self._finalized = True
        return trace

    def _parent_span_id(self, parent_context: TraceContext | None) -> str | None:
        if parent_context is None:
            return self._context.current_span_id
        if parent_context.trace_id != self._context.trace_id:
            raise TraceLifecycleError("parent context belongs to another trace")
        if (
            parent_context.current_span_id is not None
            and parent_context.current_span_id not in self._span_order
            and parent_context.current_span_id not in self._dropped_span_ids
        ):
            raise TraceLifecycleError("parent context does not identify a collector span")
        return parent_context.current_span_id

    def _ordered_spans(self) -> tuple[TraceSpan, ...]:
        return tuple(sorted(self._closed_spans, key=lambda span: self._span_order[span.span_id]))

    def _require_context(self, context: TraceContext) -> str:
        if context.trace_id != self._context.trace_id:
            raise TraceLifecycleError("span context belongs to another trace")
        if context.current_span_id is None:
            raise TraceLifecycleError("span context does not identify a current span")
        return context.current_span_id

    def _require_open(self) -> None:
        if self._finalized:
            raise TraceLifecycleError("trace collector is finalized")

    @staticmethod
    def _require_utc(value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise TraceLifecycleError("collector clock must return timezone-aware time")
        return value.astimezone(UTC)
