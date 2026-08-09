"""Behavioral contracts for request-local trace collection."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from decision_agent.observability import (
    SpanStatus,
    TraceCollector,
    TraceContext,
    TraceLifecycleError,
    TraceLimits,
    TraceStage,
)


class FakeClock:
    def __init__(self) -> None:
        self.wall = datetime(2026, 7, 27, 12, 0, tzinfo=UTC)
        self.monotonic = 10.0

    def utc_now(self) -> datetime:
        return self.wall

    def tick(self, *, seconds: float, wall_seconds: float | None = None) -> None:
        self.monotonic += seconds
        self.wall += timedelta(seconds=seconds if wall_seconds is None else wall_seconds)


class FakeIds:
    def __init__(self, *values: str) -> None:
        self._values = iter(values)

    def __call__(self) -> str:
        return next(self._values)


def _collector(*, limits: TraceLimits | None = None) -> tuple[TraceCollector, FakeClock]:
    clock = FakeClock()
    context = TraceContext.create(request_id="request_1", id_factory=lambda: "trace_1")
    return (
        TraceCollector(
            context=context,
            limits=limits,
            utc_now=clock.utc_now,
            monotonic=lambda: clock.monotonic,
            id_factory=FakeIds("span_root", "span_child", "span_extra"),
        ),
        clock,
    )


def test_collector_creates_parented_spans_and_uses_monotonic_duration() -> None:
    collector, clock = _collector()
    root = collector.start_span(stage=TraceStage.REQUEST, component="executor", operation="execute")
    clock.tick(seconds=0.25, wall_seconds=-20)
    child = collector.start_span(
        stage=TraceStage.ROUTING,
        component="router",
        operation="route",
        parent_context=root,
    )
    clock.tick(seconds=0.5)
    child_span = collector.complete_span(child)
    root_span = collector.complete_span(root)
    trace = collector.finalize(final_status=SpanStatus.COMPLETED)

    assert child_span is not None
    assert root_span is not None
    assert child_span.parent_span_id == "span_root"
    assert child_span.duration_ms == 500.0
    assert root_span.duration_ms == 750.0
    assert [span.span_id for span in trace.spans] == ["span_root", "span_child"]
    assert trace.duration_ms == 750.0


def test_collector_rejects_unknown_duplicate_cross_trace_and_post_finalize_operations() -> None:
    collector, _ = _collector()
    root = collector.start_span(stage=TraceStage.REQUEST, component="executor", operation="execute")
    collector.complete_span(root)
    with pytest.raises(TraceLifecycleError):
        collector.complete_span(root)
    other_context = TraceContext.create(request_id="request_2", id_factory=lambda: "trace_2")
    with pytest.raises(TraceLifecycleError):
        collector.start_span(
            stage=TraceStage.ROUTING,
            component="router",
            operation="route",
            parent_context=other_context,
        )
    forged_context = other_context.model_copy(
        update={"trace_id": "trace_1", "current_span_id": "not_collector_owned"}
    )
    with pytest.raises(TraceLifecycleError):
        collector.start_span(
            stage=TraceStage.ROUTING,
            component="router",
            operation="route",
            parent_context=forged_context,
        )
    collector.finalize(final_status=SpanStatus.UNSUPPORTED)
    with pytest.raises(TraceLifecycleError):
        collector.start_span(stage=TraceStage.ROUTING, component="router", operation="route")


def test_finalize_converts_unclosed_span_to_safe_failed_result() -> None:
    collector, clock = _collector()
    collector.start_span(
        stage=TraceStage.TOOL_EXECUTION,
        component="tool_runtime",
        operation="execute",
        attributes={"tool_name": "run_data_agent"},
    )
    clock.tick(seconds=0.1)
    trace = collector.finalize(final_status=SpanStatus.COMPLETED)

    assert trace.spans[0].status is SpanStatus.FAILED
    assert trace.spans[0].error_code == "observability_span_unclosed"
    assert trace.spans[0].attributes[0].value == "run_data_agent"


def test_capacity_drops_new_spans_without_changing_saved_span_history() -> None:
    collector, _ = _collector(
        limits=TraceLimits(
            max_spans_per_trace=1,
            max_attributes_per_span=24,
            max_attribute_value_length=256,
            max_serialized_trace_bytes=65536,
            max_in_memory_traces=256,
        )
    )
    root = collector.start_span(stage=TraceStage.REQUEST, component="executor", operation="execute")
    dropped = collector.start_span(stage=TraceStage.ROUTING, component="router", operation="route")
    assert collector.complete_span(dropped) is None
    collector.complete_span(root)
    trace = collector.finalize(final_status=SpanStatus.COMPLETED)

    assert [span.span_id for span in trace.spans] == ["span_root"]
    assert trace.dropped_span_count == 1


def test_independent_collectors_do_not_share_trace_or_span_state() -> None:
    first, _ = _collector()
    second_clock = FakeClock()
    second = TraceCollector(
        context=TraceContext.create(request_id="request_2", id_factory=lambda: "trace_2"),
        utc_now=second_clock.utc_now,
        monotonic=lambda: second_clock.monotonic,
        id_factory=FakeIds("span_second"),
    )

    first_span = first.start_span(
        stage=TraceStage.REQUEST, component="executor", operation="execute"
    )
    second_span = second.start_span(
        stage=TraceStage.REQUEST, component="executor", operation="execute"
    )
    first.complete_span(first_span)
    second.complete_span(second_span)

    assert first.finalize(final_status=SpanStatus.COMPLETED).trace_id == "trace_1"
    assert second.finalize(final_status=SpanStatus.COMPLETED).trace_id == "trace_2"


@pytest.mark.parametrize("status", [SpanStatus.UNSUPPORTED, SpanStatus.NOT_REQUESTED])
def test_non_failure_terminal_statuses_remain_distinct(status: SpanStatus) -> None:
    collector, _ = _collector()
    span_context = collector.start_span(
        stage=TraceStage.MEMORY_READ, component="memory", operation="read"
    )
    span = collector.complete_span(span_context, status=status)
    trace = collector.finalize(final_status=status)

    assert span is not None
    assert span.status is status
    assert trace.final_status is status
