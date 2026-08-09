"""Request-level trace integration tests without Router or Coordinator instrumentation."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime

import pytest

import decision_agent.application.executor as executor_module
from decision_agent.application import FormalRequest, FormalRequestExecutor, SessionMemoryReadError
from decision_agent.context import ConversationMemoryProjector
from decision_agent.coordination.models import CoordinatorResult, CoordinatorStatus
from decision_agent.memory import SessionMemorySnapshot, SessionTurn
from decision_agent.observability import (
    BestEffortTraceDispatcher,
    InMemoryTraceSink,
    SpanStatus,
    StructuredLoggingTraceSink,
    TraceCollector,
    TraceContext,
)
from decision_agent.routing.models import RequestRoute


class _Ids:
    def __init__(self, *values: str) -> None:
        self._values = iter(values)

    def __call__(self) -> str:
        return next(self._values)


class _Coordinator:
    def __init__(self, result: CoordinatorResult | BaseException) -> None:
        self._result = result
        self.calls: list[dict[str, object]] = []

    async def execute(self, **kwargs: object) -> CoordinatorResult:
        self.calls.append(kwargs)
        if isinstance(self._result, BaseException):
            raise self._result
        return self._result


class _Store:
    def __init__(self, snapshot: SessionMemorySnapshot | BaseException) -> None:
        self._snapshot = snapshot
        self.calls: list[str] = []

    def read(self, session_id: str) -> SessionMemorySnapshot:
        self.calls.append(session_id)
        if isinstance(self._snapshot, BaseException):
            raise self._snapshot
        return self._snapshot

    def append_turn(self, *_: object, **__: object) -> SessionMemorySnapshot:
        assert isinstance(self._snapshot, SessionMemorySnapshot)
        return self._snapshot


class _FailingStartCollector(TraceCollector):
    def start_span(self, **_: object) -> TraceContext:
        raise RuntimeError("OBSERVABILITY_PRIVATE_PATH")


class _FailingCompleteCollector(TraceCollector):
    def complete_span(self, *_: object, **__: object) -> None:
        raise RuntimeError("OBSERVABILITY_PRIVATE_PATH")


class _FailingFinalizeCollector(TraceCollector):
    def finalize(self, **_: object):  # type: ignore[no-untyped-def]
        raise RuntimeError("OBSERVABILITY_PRIVATE_PATH")


class _FailingDispatcher:
    def emit(self, _: object) -> None:
        raise RuntimeError("OBSERVABILITY_PRIVATE_PATH")


class _FailingStructuredLogger:
    def info(self, _: str) -> None:
        raise OSError("OBSERVABILITY_PRIVATE_PATH")


def _completed_result() -> CoordinatorResult:
    return CoordinatorResult(
        status=CoordinatorStatus.COMPLETED,
        route=RequestRoute.KNOWLEDGE,
        skill_name="knowledge",
        answer="ANSWER_SECRET_DO_NOT_LEAK",
        citations=["[E1]"],
        coordinator_steps=("route_request", "execute_skill"),
        tool_steps=("run_knowledge_agent",),
    )


def _failed_result() -> CoordinatorResult:
    return CoordinatorResult(
        status=CoordinatorStatus.FAILED,
        route=RequestRoute.DATA,
        error_code="safe_data_failure",
        coordinator_steps=("route_request",),
    )


def _unsupported_result() -> CoordinatorResult:
    return CoordinatorResult(
        status=CoordinatorStatus.UNSUPPORTED,
        route=RequestRoute.UNSUPPORTED,
        coordinator_steps=("route_request", "short_circuit_unsupported"),
    )


def _snapshot() -> SessionMemorySnapshot:
    return SessionMemorySnapshot(
        session_id="SESSION_SECRET_DO_NOT_LEAK",
        version=1,
        turns=(
            SessionTurn(
                session_id="SESSION_SECRET_DO_NOT_LEAK",
                turn_id="turn_1",
                request_id="old_request",
                user_text="MEMORY_SECRET_DO_NOT_LEAK",
                assistant_text="OLD_ANSWER_SECRET_DO_NOT_LEAK",
                created_at=datetime(2026, 7, 27, tzinfo=UTC),
            ),
        ),
    )


def _factory(
    *, collector_type: type[TraceCollector] = TraceCollector
) -> Callable[[FormalRequest], TraceCollector]:
    counter = 0

    def create(request: FormalRequest) -> TraceCollector:
        nonlocal counter
        counter += 1
        return collector_type(
            context=TraceContext.create(
                request_id=request.request_id,
                session_present=request.session_id is not None,
                id_factory=lambda: f"trace_{counter}",
            ),
            utc_now=lambda: datetime(2026, 7, 27, tzinfo=UTC),
            monotonic=lambda: 10.0,
            id_factory=_Ids(
                "request_span",
                "memory_read_span",
                "memory_write_span",
                "memory_summary_span",
                "response_span",
            ),
        )

    return create


def _executor(
    *,
    coordinator: _Coordinator,
    store: _Store | None = None,
    sink: InMemoryTraceSink | None = None,
    factory: Callable[[FormalRequest], TraceCollector] | None = None,
    dispatcher: object | None = None,
) -> FormalRequestExecutor:
    resolved_dispatcher = (
        dispatcher
        if dispatcher is not None
        else (None if sink is None else BestEffortTraceDispatcher([sink]))
    )
    return FormalRequestExecutor(
        coordinator=coordinator,  # type: ignore[arg-type]
        memory_store=store,  # type: ignore[arg-type]
        memory_projector=ConversationMemoryProjector(),
        trace_collector_factory=factory or _factory(),
        trace_dispatcher=resolved_dispatcher,  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_completed_request_emits_one_safe_request_memory_and_response_trace() -> None:
    sink = InMemoryTraceSink()
    response = await _executor(coordinator=_Coordinator(_completed_result()), sink=sink).execute(
        FormalRequest(request_id="request_1", user_query="QUERY_SECRET_DO_NOT_LEAK")
    )
    trace = sink.snapshot()[0]

    assert response.result.status is CoordinatorStatus.COMPLETED
    assert len(sink.snapshot()) == 1
    assert trace.trace_id != trace.request_id
    assert trace.request_id == "request_1"
    assert trace.final_status is SpanStatus.COMPLETED
    assert trace.route == "knowledge"
    assert trace.skill_name == "knowledge"
    assert [(span.stage, span.status) for span in trace.spans] == [
        ("request", SpanStatus.COMPLETED),
        ("memory_read", SpanStatus.NOT_REQUESTED),
        ("memory_write", SpanStatus.NOT_REQUESTED),
        ("memory_summary", SpanStatus.NOT_REQUESTED),
        ("response_mapping", SpanStatus.COMPLETED),
    ]
    serialized = str(trace.model_dump(mode="json"))
    assert all(
        marker not in serialized
        for marker in ("QUERY_SECRET", "SESSION_SECRET", "ANSWER_SECRET", "[E1]")
    )


@pytest.mark.asyncio
async def test_disabled_memory_with_session_is_not_requested_without_store_access() -> None:
    sink = InMemoryTraceSink()
    response = await _executor(coordinator=_Coordinator(_completed_result()), sink=sink).execute(
        FormalRequest(
            request_id="request_1",
            session_id="SESSION_SECRET_DO_NOT_LEAK",
            user_query="query",
        )
    )
    memory_span = sink.snapshot()[0].spans[1]

    assert response.memory_context_status.value == "not_requested"
    assert memory_span.status is SpanStatus.NOT_REQUESTED
    assert dict((item.key, item.value) for item in memory_span.attributes) == {
        "memory_requested": False
    }


@pytest.mark.asyncio
async def test_memory_read_success_records_only_turn_count_without_memory_content() -> None:
    sink = InMemoryTraceSink()
    store = _Store(_snapshot())
    await _executor(coordinator=_Coordinator(_completed_result()), store=store, sink=sink).execute(
        FormalRequest(
            request_id="request_1", session_id="SESSION_SECRET_DO_NOT_LEAK", user_query="q"
        )
    )
    trace = sink.snapshot()[0]
    memory_span = trace.spans[1]

    assert store.calls == ["SESSION_SECRET_DO_NOT_LEAK"]
    assert memory_span.status is SpanStatus.COMPLETED
    assert dict((item.key, item.value) for item in memory_span.attributes) == {
        "memory_requested": True,
        "result_count": 1,
        "success": True,
    }
    assert "MEMORY_SECRET" not in str(trace.model_dump(mode="json"))


@pytest.mark.asyncio
async def test_memory_read_failure_preserves_fail_closed_exception_and_emits_failed_trace() -> None:
    sink = InMemoryTraceSink()
    executor = _executor(
        coordinator=_Coordinator(_completed_result()),
        store=_Store(RuntimeError("MEMORY_PRIVATE_PATH")),
        sink=sink,
    )

    with pytest.raises(SessionMemoryReadError):
        await executor.execute(
            FormalRequest(request_id="request_1", session_id="session_1", user_query="q")
        )

    trace = sink.snapshot()[0]
    assert trace.final_status is SpanStatus.FAILED
    assert trace.error_code == "session_memory_read_failed"
    assert trace.spans[1].status is SpanStatus.FAILED
    assert "MEMORY_PRIVATE_PATH" not in str(trace.model_dump(mode="json"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result", "status", "error_code"),
    [
        (_completed_result(), SpanStatus.COMPLETED, None),
        (_failed_result(), SpanStatus.FAILED, "safe_data_failure"),
        (_unsupported_result(), SpanStatus.UNSUPPORTED, None),
    ],
)
async def test_formal_result_status_maps_to_root_span(
    result: CoordinatorResult,
    status: SpanStatus,
    error_code: str | None,
) -> None:
    sink = InMemoryTraceSink()
    await _executor(coordinator=_Coordinator(result), sink=sink).execute(
        FormalRequest(request_id="request_1", user_query="q")
    )
    trace = sink.snapshot()[0]

    assert trace.final_status is status
    assert trace.error_code == error_code
    assert trace.spans[0].status is status


@pytest.mark.asyncio
async def test_cancellation_emits_cancelled_trace_and_propagates() -> None:
    sink = InMemoryTraceSink()
    executor = _executor(coordinator=_Coordinator(asyncio.CancelledError()), sink=sink)

    with pytest.raises(asyncio.CancelledError):
        await executor.execute(FormalRequest(request_id="request_1", user_query="q"))

    trace = sink.snapshot()[0]
    assert trace.final_status is SpanStatus.CANCELLED
    assert trace.spans[0].status is SpanStatus.CANCELLED


@pytest.mark.asyncio
async def test_response_mapping_failure_emits_failed_mapping_span_and_reraises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = InMemoryTraceSink()

    def fail_response_mapping(**_: object) -> None:
        raise ValueError("RESPONSE_MAPPING_PRIVATE_PATH")

    monkeypatch.setattr(executor_module, "FormalResponse", fail_response_mapping)
    with pytest.raises(ValueError):
        await _executor(coordinator=_Coordinator(_completed_result()), sink=sink).execute(
            FormalRequest(request_id="request_1", user_query="q")
        )

    trace = sink.snapshot()[0]
    mapping_span = trace.spans[4]
    assert trace.final_status is SpanStatus.FAILED
    assert mapping_span.status is SpanStatus.FAILED
    assert mapping_span.error_code == "response_mapping_failed"
    assert "RESPONSE_MAPPING_PRIVATE_PATH" not in str(trace.model_dump(mode="json"))


@pytest.mark.asyncio
async def test_unexpected_coordinator_failure_reraises_and_emits_safe_failed_trace() -> None:
    sink = InMemoryTraceSink()
    executor = _executor(
        coordinator=_Coordinator(RuntimeError("COORDINATOR_PRIVATE_PATH")),
        sink=sink,
    )

    with pytest.raises(RuntimeError):
        await executor.execute(FormalRequest(request_id="request_1", user_query="q"))

    trace = sink.snapshot()[0]
    assert trace.final_status is SpanStatus.FAILED
    assert trace.error_code == "formal_request_execution_failed"
    assert "COORDINATOR_PRIVATE_PATH" not in str(trace.model_dump(mode="json"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "factory",
    [
        lambda _: (_ for _ in ()).throw(RuntimeError("OBSERVABILITY_PRIVATE_PATH")),
        _factory(collector_type=_FailingStartCollector),
        _factory(collector_type=_FailingCompleteCollector),
        _factory(collector_type=_FailingFinalizeCollector),
    ],
)
async def test_collector_failures_do_not_change_completed_business_response(
    factory: Callable[[FormalRequest], TraceCollector],
) -> None:
    response = await _executor(
        coordinator=_Coordinator(_completed_result()),
        factory=factory,
        dispatcher=_FailingDispatcher(),
    ).execute(FormalRequest(request_id="request_1", user_query="q"))

    assert response.result.status is CoordinatorStatus.COMPLETED
    assert response.result.error_code is None


@pytest.mark.asyncio
async def test_structured_logging_sink_failure_does_not_change_business_response(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="decision_agent.observability")
    dispatcher = BestEffortTraceDispatcher(
        [StructuredLoggingTraceSink(logger=_FailingStructuredLogger())]  # type: ignore[arg-type]
    )

    response = await _executor(
        coordinator=_Coordinator(_completed_result()),
        dispatcher=dispatcher,
    ).execute(FormalRequest(request_id="request_1", user_query="q"))

    assert response.result.status is CoordinatorStatus.COMPLETED
    assert response.result.error_code is None
    assert "OBSERVABILITY_PRIVATE_PATH" not in caplog.text


@pytest.mark.asyncio
async def test_two_concurrent_requests_get_isolated_trace_ids_and_request_ids() -> None:
    sink = InMemoryTraceSink()
    executor = _executor(coordinator=_Coordinator(_completed_result()), sink=sink)
    await asyncio.gather(
        executor.execute(FormalRequest(request_id="request_1", user_query="q1")),
        executor.execute(FormalRequest(request_id="request_2", user_query="q2")),
    )
    traces = sink.snapshot()

    assert [trace.request_id for trace in traces] == ["request_1", "request_2"]
    assert {trace.trace_id for trace in traces} == {"trace_1", "trace_2"}
    assert all(all(span.trace_id == trace.trace_id for span in trace.spans) for trace in traces)
