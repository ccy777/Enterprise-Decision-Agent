"""Contracts for immutable trace domain models."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from decision_agent.observability import (
    RequestTrace,
    SpanStatus,
    TraceAttribute,
    TraceContext,
    TraceSpan,
    TraceStage,
)


def _timestamp() -> datetime:
    return datetime(2026, 7, 27, 12, 0, tzinfo=UTC)


def _span(*, value: object = 0) -> TraceSpan:
    return TraceSpan(
        span_id="span_1",
        trace_id="trace_1",
        parent_span_id=None,
        stage=TraceStage.REQUEST,
        component="executor",
        operation="execute",
        started_at=_timestamp(),
        completed_at=_timestamp(),
        duration_ms=0.0,
        status=SpanStatus.COMPLETED,
        attributes=(TraceAttribute(key="retry_count", value=value),),
    )


def test_context_is_immutable_and_child_context_preserves_trace_relationship() -> None:
    context = TraceContext.create(
        request_id="request_1", session_present=True, id_factory=lambda: "trace_server_1"
    )
    child = context.for_child_span("span_root")

    assert context.trace_id == "trace_server_1"
    assert context.request_id == "request_1"
    assert context.current_span_id is None
    assert child.current_span_id == "span_root"
    assert child.parent_span_id is None
    with pytest.raises(ValidationError):
        context.trace_id = "replacement"  # type: ignore[misc]


def test_request_trace_and_span_are_immutable_and_keep_none_distinct_from_zero() -> None:
    none_span = _span(value=None)
    zero_span = _span(value=0).model_copy(update={"span_id": "span_2"})
    trace = RequestTrace(
        trace_id="trace_1",
        request_id="request_1",
        session_present=False,
        started_at=_timestamp(),
        completed_at=_timestamp(),
        duration_ms=0.0,
        final_status=SpanStatus.COMPLETED,
        spans=(none_span, zero_span),
        span_count=2,
    )

    assert none_span.attributes[0].value is None
    assert zero_span.attributes[0].value == 0
    assert trace.model_dump(mode="json")["spans"][0]["attributes"][0]["value"] is None
    assert trace.model_dump(mode="json")["spans"][1]["attributes"][0]["value"] == 0
    with pytest.raises(ValidationError):
        trace.span_count = 3  # type: ignore[misc]


@pytest.mark.parametrize("invalid", ["Executor", "execute-request", "", "with space"])
def test_span_rejects_unstable_component_or_operation_names(invalid: str) -> None:
    with pytest.raises(ValidationError):
        TraceSpan(
            span_id="span_1",
            trace_id="trace_1",
            stage=TraceStage.REQUEST,
            component=invalid,
            operation="execute",
            started_at=_timestamp(),
            completed_at=_timestamp(),
            duration_ms=0.0,
            status=SpanStatus.COMPLETED,
        )


def test_trace_rejects_naive_time_and_running_terminal_state() -> None:
    with pytest.raises(ValidationError):
        RequestTrace(
            trace_id="trace_1",
            request_id="request_1",
            session_present=False,
            started_at=datetime(2026, 7, 27, 12, 0),
            completed_at=_timestamp(),
            duration_ms=0.0,
            final_status=SpanStatus.COMPLETED,
            span_count=0,
        )
    with pytest.raises(ValidationError):
        RequestTrace(
            trace_id="trace_1",
            request_id="request_1",
            session_present=False,
            started_at=_timestamp(),
            completed_at=_timestamp(),
            duration_ms=0.0,
            final_status=SpanStatus.RUNNING,
            span_count=0,
        )
