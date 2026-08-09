"""Dependency-free, request-local observability domain contracts and sinks."""

from decision_agent.observability.attributes import (
    AttributeLimits,
    AttributeSanitization,
    sanitize_attributes,
)
from decision_agent.observability.collector import TraceCollector, TraceLifecycleError, TraceLimits
from decision_agent.observability.execution import (
    TraceExecution,
    TraceSpanRecorder,
    complete_recorded_span,
    start_recorded_span,
)
from decision_agent.observability.models import (
    RequestTrace,
    TraceAttribute,
    TraceContext,
    TraceSpan,
    TraceStageSummary,
    TraceSummary,
    new_trace_id,
)
from decision_agent.observability.sinks import (
    BestEffortTraceDispatcher,
    InMemoryTraceSink,
    StructuredLoggingTraceSink,
    TraceEmissionResult,
    TraceSink,
    TraceSinkError,
)
from decision_agent.observability.stages import SpanStatus, TraceStage
from decision_agent.observability.summary import (
    DEFAULT_TRACE_SUMMARY_MAX_STAGES,
    build_trace_summary,
)

__all__ = [
    "DEFAULT_TRACE_SUMMARY_MAX_STAGES",
    "AttributeLimits",
    "AttributeSanitization",
    "BestEffortTraceDispatcher",
    "InMemoryTraceSink",
    "RequestTrace",
    "SpanStatus",
    "StructuredLoggingTraceSink",
    "TraceAttribute",
    "TraceCollector",
    "TraceContext",
    "TraceEmissionResult",
    "TraceExecution",
    "TraceLifecycleError",
    "TraceLimits",
    "TraceSink",
    "TraceSinkError",
    "TraceSpan",
    "TraceSpanRecorder",
    "TraceStage",
    "TraceStageSummary",
    "TraceSummary",
    "build_trace_summary",
    "complete_recorded_span",
    "new_trace_id",
    "sanitize_attributes",
    "start_recorded_span",
]
