"""Public TraceSummary projections stay bounded, ordered, and payload-free."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from decision_agent.observability import (
    SpanStatus,
    TraceCollector,
    TraceContext,
    TraceStage,
    build_trace_summary,
)


class _Ids:
    def __init__(self) -> None:
        self._value = 0

    def __call__(self) -> str:
        self._value += 1
        return f"span_{self._value}"


def _collector() -> TraceCollector:
    return TraceCollector(
        context=TraceContext.create(request_id="request_1", id_factory=lambda: "trace_1"),
        utc_now=lambda: datetime(2026, 7, 28, tzinfo=UTC),
        monotonic=lambda: 10.0,
        id_factory=_Ids(),
    )


def test_trace_summary_preserves_span_order_and_exposes_only_allowlisted_attributes() -> None:
    collector = _collector()
    root = collector.start_span(stage=TraceStage.REQUEST, component="app", operation="execute")
    routing = collector.start_span(
        stage=TraceStage.ROUTING,
        component="routing",
        operation="route_request",
        parent_context=root,
        attributes={"route": "knowledge", "query": "PRIVATE_QUERY"},
    )
    collector.complete_span(routing, status=SpanStatus.COMPLETED, attributes={"success": True})
    collector.complete_span(root, status=SpanStatus.COMPLETED)
    summary = build_trace_summary(collector.finalize(final_status=SpanStatus.COMPLETED))

    assert [stage.operation for stage in summary.stages] == ["execute", "route_request"]
    assert [(item.key, item.value) for item in summary.stages[1].attributes] == [
        ("route", "knowledge")
    ]
    assert summary.truncated_stage_count == 0
    assert "PRIVATE_QUERY" not in summary.model_dump_json()


def test_trace_summary_keeps_unknown_distinct_from_zero_and_reports_truncation() -> None:
    collector = _collector()
    root = collector.start_span(stage=TraceStage.REQUEST, component="app", operation="execute")
    data = collector.start_span(
        stage=TraceStage.DATA_ACCESS,
        component="mcp",
        operation="execute_safe_query",
        parent_context=root,
        attributes={"row_count": 0, "query": "PRIVATE_SQL"},
    )
    collector.complete_span(data, status=SpanStatus.COMPLETED)
    collector.complete_span(root, status=SpanStatus.UNSUPPORTED)
    trace = collector.finalize(final_status=SpanStatus.UNSUPPORTED)

    summary = build_trace_summary(trace, max_stages=1)

    assert summary.final_status is SpanStatus.UNSUPPORTED
    assert summary.truncated_stage_count == 1
    assert summary.stages[0].attributes == ()
    full_summary = build_trace_summary(trace)
    assert [(item.key, item.value) for item in full_summary.stages[1].attributes] == [
        ("row_count", 0)
    ]
    with pytest.raises(ValueError, match="max_stages"):
        build_trace_summary(trace, max_stages=0)
