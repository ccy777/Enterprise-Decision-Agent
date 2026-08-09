"""Behavioral tests for bounded trace sinks and best-effort fan-out."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from decision_agent.observability import (
    BestEffortTraceDispatcher,
    InMemoryTraceSink,
    RequestTrace,
    SpanStatus,
    StructuredLoggingTraceSink,
    TraceAttribute,
    TraceLimits,
    TraceSpan,
    TraceStage,
)


class ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


class FailingSink:
    def emit(self, trace: RequestTrace) -> None:
        raise OSError("private path and exception text must not be logged")


def _trace(*, attribute_value: str = "safe") -> RequestTrace:
    timestamp = datetime(2026, 7, 27, 12, 0, tzinfo=UTC)
    span = TraceSpan(
        span_id="span_1",
        trace_id="trace_1",
        stage=TraceStage.RETRIEVAL,
        component="retrieval",
        operation="retrieve",
        started_at=timestamp,
        completed_at=timestamp,
        duration_ms=0.0,
        status=SpanStatus.COMPLETED,
        attributes=(TraceAttribute(key="provider", value=attribute_value),),
    )
    return RequestTrace(
        trace_id="trace_1",
        request_id="request_1",
        session_present=False,
        started_at=timestamp,
        completed_at=timestamp,
        duration_ms=0.0,
        final_status=SpanStatus.COMPLETED,
        spans=(span,),
        span_count=1,
    )


def test_in_memory_sink_is_bounded_isolated_and_returns_immutable_snapshots() -> None:
    first = InMemoryTraceSink(max_traces=1)
    second = InMemoryTraceSink(max_traces=2)
    trace_one = _trace()
    trace_two = trace_one.model_copy(
        update={
            "trace_id": "trace_2",
            "request_id": "request_2",
            "spans": (trace_one.spans[0].model_copy(update={"trace_id": "trace_2"}),),
        }
    )

    first.emit(trace_one)
    first.emit(trace_two)
    second.emit(trace_one)

    assert first.snapshot() == (trace_two,)
    assert second.snapshot() == (trace_one,)
    assert first.find("trace_2") == trace_two


def test_structured_sink_outputs_one_safe_parseable_json_line_without_global_configuration() -> (
    None
):
    logger = logging.getLogger("test.observability.json")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = ListHandler()
    logger.addHandler(handler)
    StructuredLoggingTraceSink(logger=logger).emit(_trace())

    payload = json.loads(handler.messages[0])
    assert payload["event"] == "decision_agent.request_trace"
    assert payload["trace_id"] == "trace_1"
    assert payload["spans"][0]["attributes"] == {"provider": "safe"}
    serialized = handler.messages[0].lower()
    assert all(value not in serialized for value in ("query", "prompt", "sql", "session_id"))


def test_structured_sink_emits_safe_summary_when_serialized_trace_is_too_large() -> None:
    logger = logging.getLogger("test.observability.truncated")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = ListHandler()
    logger.addHandler(handler)
    limits = TraceLimits(max_serialized_trace_bytes=120)
    StructuredLoggingTraceSink(logger=logger, limits=limits).emit(_trace(attribute_value="x" * 256))

    payload = json.loads(handler.messages[0])
    assert payload["trace_truncated"] is True
    assert "spans" not in payload
    assert "x" * 256 not in handler.messages[0]


def test_best_effort_dispatcher_continues_after_failure_with_safe_fallback() -> None:
    logger = logging.getLogger("test.observability.fallback")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.WARNING)
    handler = ListHandler()
    logger.addHandler(handler)
    successful = InMemoryTraceSink()

    result = BestEffortTraceDispatcher([FailingSink(), successful], fallback_logger=logger).emit(
        _trace()
    )

    assert result.attempted_sink_count == 2
    assert result.succeeded_sink_count == 1
    assert result.failed_sink_count == 1
    assert successful.snapshot()[0].trace_id == "trace_1"
    fallback = json.loads(handler.messages[0])
    assert fallback["error_code"] == "observability_sink_failed"
    assert "private path" not in handler.messages[0]


def test_best_effort_dispatcher_does_not_raise_when_every_sink_fails() -> None:
    result = BestEffortTraceDispatcher([FailingSink(), FailingSink()]).emit(_trace())

    assert result.attempted_sink_count == 2
    assert result.succeeded_sink_count == 0
    assert result.failed_sink_count == 2
