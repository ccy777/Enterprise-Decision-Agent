"""Bounded public Trace projections with an explicit allowlist."""

from __future__ import annotations

from decision_agent.observability.models import (
    RequestTrace,
    TraceAttribute,
    TraceStageSummary,
    TraceSummary,
)

DEFAULT_TRACE_SUMMARY_MAX_STAGES = 24
_PUBLIC_ATTRIBUTE_KEYS = frozenset(
    {
        "answerable",
        "denied",
        "input_tokens",
        "model",
        "output_tokens",
        "provider",
        "reranked_count",
        "retrieved_count",
        "retry_count",
        "review_outcome",
        "route",
        "row_count",
        "selected_evidence_count",
        "skill_name",
        "timeout",
        "tool_name",
    }
)


def build_trace_summary(
    trace: RequestTrace,
    *,
    max_stages: int = DEFAULT_TRACE_SUMMARY_MAX_STAGES,
) -> TraceSummary:
    """Project one final trace without copying request, memory, or business payloads."""
    if max_stages <= 0:
        raise ValueError("max_stages must be positive")
    included_spans = trace.spans[:max_stages]
    return TraceSummary(
        trace_id=trace.trace_id,
        request_id=trace.request_id,
        final_status=trace.final_status,
        duration_ms=trace.duration_ms,
        span_count=trace.span_count,
        dropped_span_count=trace.dropped_span_count,
        dropped_attribute_count=trace.dropped_attribute_count,
        sink_failure_count=trace.sink_failure_count,
        truncated_stage_count=len(trace.spans) - len(included_spans),
        stages=tuple(
            TraceStageSummary(
                stage=span.stage,
                operation=span.operation,
                status=span.status,
                duration_ms=span.duration_ms,
                error_code=span.error_code,
                attributes=_safe_attributes(span.attributes),
            )
            for span in included_spans
        ),
    )


def _safe_attributes(attributes: tuple[TraceAttribute, ...]) -> tuple[TraceAttribute, ...]:
    return tuple(attribute for attribute in attributes if attribute.key in _PUBLIC_ATTRIBUTE_KEYS)
