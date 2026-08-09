"""Best-effort, request-local span recording for production execution seams."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Protocol

from decision_agent.observability.collector import TraceCollector
from decision_agent.observability.models import RequestTrace, TraceContext
from decision_agent.observability.sinks import BestEffortTraceDispatcher
from decision_agent.observability.stages import SpanStatus, TraceStage

_OBSERVABILITY_FAILURE_CODE = "observability_instrumentation_failed"
_LOGGER = logging.getLogger("decision_agent.observability")


class TraceSpanRecorder(Protocol):
    """Minimal observability-only capability passed between execution layers."""

    def start_span(
        self,
        *,
        stage: TraceStage,
        component: str,
        operation: str,
        parent_context: TraceContext | None = None,
        attributes: Mapping[str, object] | None = None,
    ) -> TraceContext | None:
        """Start one best-effort span without affecting business execution."""

    def complete_span(
        self,
        context: TraceContext | None,
        *,
        status: SpanStatus,
        error_code: str | None = None,
        attributes: Mapping[str, object] | None = None,
    ) -> None:
        """Complete one best-effort span without affecting business execution."""


class TraceExecution:
    """Fault-isolated collector owner for one request; never shared globally."""

    def __init__(
        self,
        *,
        collector: TraceCollector | None,
        dispatcher: BestEffortTraceDispatcher | None,
    ) -> None:
        self._collector = collector
        self._dispatcher = dispatcher
        self._active_span: TraceContext | None = None

    @classmethod
    def start(
        cls,
        *,
        collector_factory: Callable[[], TraceCollector],
        dispatcher: BestEffortTraceDispatcher | None,
    ) -> TraceExecution:
        try:
            collector = collector_factory()
            if not isinstance(collector, TraceCollector):
                raise TypeError("trace collector factory returned an invalid collector")
        except Exception:
            _safe_observability_warning()
            collector = None
        return cls(collector=collector, dispatcher=dispatcher)

    def start_span(
        self,
        *,
        stage: TraceStage,
        component: str,
        operation: str,
        parent_context: TraceContext | None = None,
        attributes: Mapping[str, object] | None = None,
    ) -> TraceContext | None:
        if self._collector is None:
            return None
        try:
            context = self._collector.start_span(
                stage=stage,
                component=component,
                operation=operation,
                parent_context=parent_context,
                attributes=attributes,
            )
        except Exception:
            _safe_observability_warning()
            return None
        self._active_span = context
        return context

    def complete_span(
        self,
        context: TraceContext | None,
        *,
        status: SpanStatus,
        error_code: str | None = None,
        attributes: Mapping[str, object] | None = None,
    ) -> None:
        if self._collector is None or context is None:
            return
        try:
            self._collector.complete_span(
                context,
                status=status,
                error_code=error_code,
                attributes=attributes,
            )
        except Exception:
            _safe_observability_warning()
            return
        if self._active_span == context:
            self._active_span = None

    def memory_not_requested(self, *, parent_context: TraceContext | None) -> None:
        span = self.start_span(
            stage=TraceStage.MEMORY_READ,
            component="application",
            operation="load_conversation_memory",
            parent_context=parent_context,
            attributes={"memory_requested": False},
        )
        self.complete_span(span, status=SpanStatus.NOT_REQUESTED)

    def cancel_active_span(self) -> None:
        self.complete_span(self._active_span, status=SpanStatus.CANCELLED)

    def finish(
        self,
        *,
        root_span: TraceContext | None,
        final_status: SpanStatus,
        error_code: str | None,
        route: str | None = None,
        skill_name: str | None = None,
    ) -> RequestTrace | None:
        self.complete_span(
            root_span,
            status=final_status,
            error_code=error_code,
            attributes={"final_status": final_status.value},
        )
        if self._collector is None:
            return None
        try:
            trace = self._collector.finalize(
                final_status=final_status,
                route=route,
                skill_name=skill_name,
                error_code=error_code,
            )
        except Exception:
            _safe_observability_warning()
            return None
        if self._dispatcher is None:
            return trace
        try:
            self._dispatcher.emit(trace)
        except Exception:
            _safe_observability_warning()
        return trace


def start_recorded_span(
    recorder: TraceSpanRecorder | None,
    *,
    stage: TraceStage,
    component: str,
    operation: str,
    parent_context: TraceContext | None = None,
    attributes: Mapping[str, object] | None = None,
) -> TraceContext | None:
    """Call an optional recorder without letting instrumentation alter execution."""
    if recorder is None:
        return None
    try:
        return recorder.start_span(
            stage=stage,
            component=component,
            operation=operation,
            parent_context=parent_context,
            attributes=attributes,
        )
    except Exception:
        _safe_observability_warning()
        return None


def complete_recorded_span(
    recorder: TraceSpanRecorder | None,
    context: TraceContext | None,
    *,
    status: SpanStatus,
    error_code: str | None = None,
    attributes: Mapping[str, object] | None = None,
) -> None:
    """Complete an optional recorder span without letting instrumentation alter execution."""
    if recorder is None or context is None:
        return
    try:
        recorder.complete_span(
            context,
            status=status,
            error_code=error_code,
            attributes=attributes,
        )
    except Exception:
        _safe_observability_warning()


def _safe_observability_warning() -> None:
    try:
        _LOGGER.warning(_OBSERVABILITY_FAILURE_CODE)
    except (OSError, TypeError, ValueError):
        return
