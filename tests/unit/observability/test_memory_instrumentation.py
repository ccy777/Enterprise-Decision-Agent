"""Trace contracts for the existing executor-owned Memory boundaries."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

import decision_agent.application.executor as executor_module
from decision_agent.application import FormalRequest, FormalRequestExecutor
from decision_agent.context import ConversationMemoryProjector
from decision_agent.coordination.models import CoordinatorResult, CoordinatorStatus
from decision_agent.memory import (
    RollingSummaryGenerationError,
    RollingSummaryStatus,
    SessionMemorySnapshot,
    SessionTurn,
)
from decision_agent.observability import (
    BestEffortTraceDispatcher,
    InMemoryTraceSink,
    SpanStatus,
    TraceCollector,
    TraceContext,
)
from decision_agent.routing.models import RequestRoute

NOW = datetime(2026, 7, 28, tzinfo=UTC)


class _Ids:
    def __init__(self) -> None:
        self._value = 0

    def __call__(self) -> str:
        self._value += 1
        return f"span_{self._value}"


class _Coordinator:
    async def execute(self, **_: object) -> CoordinatorResult:
        return CoordinatorResult(
            status=CoordinatorStatus.COMPLETED,
            route=RequestRoute.KNOWLEDGE,
            skill_name="knowledge",
            answer="PRIVATE_ANSWER[E1]",
            citations=["[E1]"],
            coordinator_steps=("route",),
            tool_steps=("tool",),
            memory_context_selected=False,
        )


class _Store:
    def __init__(self, *, append_error: BaseException | None = None) -> None:
        self.append_error = append_error
        self.read_calls = 0
        self.append_calls = 0
        self.snapshot = SessionMemorySnapshot(session_id="PRIVATE_SESSION", version=0)

    def read(self, session_id: str) -> SessionMemorySnapshot:
        assert session_id == "PRIVATE_SESSION"
        self.read_calls += 1
        return self.snapshot

    def append_turn(self, turn: SessionTurn, *, expected_version: int) -> SessionMemorySnapshot:
        assert expected_version == self.snapshot.version
        self.append_calls += 1
        if self.append_error is not None:
            raise self.append_error
        self.snapshot = SessionMemorySnapshot(
            session_id=turn.session_id,
            version=expected_version + 1,
            turns=(turn,),
        )
        return self.snapshot


class _SummaryService:
    def __init__(self, outcome: object | BaseException) -> None:
        self.outcome = outcome
        self.calls = 0

    def compact_snapshot_if_needed(
        self, session_id: str, snapshot: SessionMemorySnapshot
    ) -> object:
        assert session_id == "PRIVATE_SESSION"
        self.calls += 1
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def _executor(
    *,
    store: _Store | None,
    summary_service: object | None = None,
) -> tuple[FormalRequestExecutor, InMemoryTraceSink]:
    sink = InMemoryTraceSink()
    ids = _Ids()

    def collector_factory(_: FormalRequest) -> TraceCollector:
        return TraceCollector(
            context=TraceContext.create(request_id="request_1", id_factory=lambda: "trace_1"),
            utc_now=lambda: NOW,
            monotonic=lambda: 10.0,
            id_factory=ids,
        )

    return (
        FormalRequestExecutor(
            coordinator=_Coordinator(),  # type: ignore[arg-type]
            memory_store=store,  # type: ignore[arg-type]
            memory_projector=ConversationMemoryProjector(),
            rolling_summary_service=summary_service,  # type: ignore[arg-type]
            trace_collector_factory=collector_factory,
            trace_dispatcher=BestEffortTraceDispatcher([sink]),
        ),
        sink,
    )


def _request(*, session_id: str | None = "PRIVATE_SESSION") -> FormalRequest:
    return FormalRequest(request_id="request_1", session_id=session_id, user_query="PRIVATE_QUERY")


def _span(trace: object, operation: str):
    return next(item for item in trace.spans if item.operation == operation)  # type: ignore[attr-defined]


@pytest.mark.asyncio
@pytest.mark.parametrize("store,session_id", [(None, "PRIVATE_SESSION"), (_Store(), None)])
async def test_memory_write_and_summary_are_not_requested_without_an_active_memory_path(
    store: _Store | None, session_id: str | None
) -> None:
    executor, sink = _executor(store=store)

    response = await executor.execute(_request(session_id=session_id))
    trace = sink.snapshot()[0]

    assert response.memory_persistence_status.value == "not_requested"
    assert response.memory_summarization_status.value == "not_requested"
    assert _span(trace, "persist_conversation_memory").status is SpanStatus.NOT_REQUESTED
    assert _span(trace, "summarize_conversation_memory").status is SpanStatus.NOT_REQUESTED
    assert "PRIVATE_SESSION" not in response.trace.model_dump_json()  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_memory_write_success_is_traced_once_without_turn_content() -> None:
    store = _Store()
    executor, sink = _executor(store=store)

    response = await executor.execute(_request())
    trace = sink.snapshot()[0]
    write = _span(trace, "persist_conversation_memory")

    assert store.append_calls == 1 and response.memory_persistence_status.value == "persisted"
    assert write.status is SpanStatus.COMPLETED
    assert {item.key: item.value for item in write.attributes} == {
        "written_item_count": 1,
        "success": True,
    }
    assert all(
        marker not in response.trace.model_dump_json()  # type: ignore[union-attr]
        for marker in ("PRIVATE_QUERY", "PRIVATE_ANSWER")
    )


@pytest.mark.asyncio
async def test_memory_write_failure_preserves_existing_business_status() -> None:
    store = _Store(append_error=OSError("PRIVATE_STORE_FAILURE"))
    executor, sink = _executor(store=store)

    response = await executor.execute(_request())
    write = _span(sink.snapshot()[0], "persist_conversation_memory")

    assert store.append_calls == 1 and response.memory_persistence_status.value == "store_failure"
    assert write.status is SpanStatus.FAILED and write.error_code == "memory_write_failed"


@pytest.mark.asyncio
async def test_memory_write_cancellation_is_rethrown_once() -> None:
    store = _Store(append_error=asyncio.CancelledError())
    executor, sink = _executor(store=store)

    with pytest.raises(asyncio.CancelledError):
        await executor.execute(_request())

    write = _span(sink.snapshot()[0], "persist_conversation_memory")
    assert store.append_calls == 1 and write.status is SpanStatus.CANCELLED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "expected_status", "expected_span_status"),
    [
        (
            SimpleNamespace(status=RollingSummaryStatus.NOT_REQUIRED, compacted_turn_count=0),
            "not_needed",
            SpanStatus.NOT_REQUESTED,
        ),
        (
            SimpleNamespace(status=RollingSummaryStatus.COMPACTED, compacted_turn_count=1),
            "compacted",
            SpanStatus.COMPLETED,
        ),
        (RollingSummaryGenerationError(stage="provider"), "provider_failure", SpanStatus.FAILED),
    ],
)
async def test_summary_trace_reflects_the_existing_summary_outcome(
    outcome: object | BaseException,
    expected_status: str,
    expected_span_status: SpanStatus,
) -> None:
    store = _Store()
    summary_service = _SummaryService(outcome)
    executor, sink = _executor(store=store, summary_service=summary_service)

    response = await executor.execute(_request())
    summary = _span(sink.snapshot()[0], "summarize_conversation_memory")

    assert summary_service.calls == 1
    assert response.memory_summarization_status.value == expected_status
    assert summary.status is expected_span_status


@pytest.mark.asyncio
async def test_summary_cancellation_is_rethrown_once() -> None:
    store = _Store()
    summary_service = _SummaryService(asyncio.CancelledError())
    executor, sink = _executor(store=store, summary_service=summary_service)

    with pytest.raises(asyncio.CancelledError):
        await executor.execute(_request())

    summary = _span(sink.snapshot()[0], "summarize_conversation_memory")
    assert summary_service.calls == 1 and summary.status is SpanStatus.CANCELLED


@pytest.mark.asyncio
async def test_memory_instrumentation_failure_does_not_repeat_the_write() -> None:
    class _BrokenCollector(TraceCollector):
        def start_span(self, **_: object) -> TraceContext:
            raise RuntimeError("PRIVATE_TRACE_FAILURE")

    store = _Store()
    executor = FormalRequestExecutor(
        coordinator=_Coordinator(),  # type: ignore[arg-type]
        memory_store=store,  # type: ignore[arg-type]
        memory_projector=ConversationMemoryProjector(),
        trace_collector_factory=lambda _: _BrokenCollector(
            context=TraceContext.create(request_id="request_1", id_factory=lambda: "trace_1"),
            id_factory=lambda: "span_1",
        ),
    )

    response = await executor.execute(_request())

    assert response.memory_persistence_status.value == "persisted"
    assert store.append_calls == 1


@pytest.mark.asyncio
async def test_trace_summary_failure_does_not_change_the_formal_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _Store()
    executor, _ = _executor(store=store)
    monkeypatch.setattr(
        executor_module,
        "build_trace_summary",
        lambda _: (_ for _ in ()).throw(RuntimeError("PRIVATE_SUMMARY_FAILURE")),
    )

    response = await executor.execute(_request())

    assert response.memory_persistence_status.value == "persisted"
    assert response.result.answer == "PRIVATE_ANSWER[E1]"
    assert response.trace is None and store.append_calls == 1


@pytest.mark.asyncio
async def test_concurrent_memory_trace_summaries_do_not_cross_request_identity() -> None:
    sink = InMemoryTraceSink()
    ids = _Ids()
    trace_count = 0

    def collector_factory(request: FormalRequest) -> TraceCollector:
        nonlocal trace_count
        trace_count += 1
        return TraceCollector(
            context=TraceContext.create(
                request_id=request.request_id,
                id_factory=lambda: f"trace_{trace_count}",
            ),
            utc_now=lambda: NOW,
            monotonic=lambda: 10.0,
            id_factory=ids,
        )

    executor = FormalRequestExecutor(
        coordinator=_Coordinator(),  # type: ignore[arg-type]
        memory_store=None,
        memory_projector=ConversationMemoryProjector(),
        trace_collector_factory=collector_factory,
        trace_dispatcher=BestEffortTraceDispatcher([sink]),
    )
    first, second = await asyncio.gather(
        executor.execute(FormalRequest(request_id="request_a", user_query="PRIVATE_QUERY_A")),
        executor.execute(FormalRequest(request_id="request_b", user_query="PRIVATE_QUERY_B")),
    )

    assert first.trace is not None and second.trace is not None
    assert first.trace.request_id == "request_a" and second.trace.request_id == "request_b"
    assert first.trace.trace_id != second.trace.trace_id
    assert all(
        marker not in first.trace.model_dump_json() and marker not in second.trace.model_dump_json()
        for marker in ("PRIVATE_QUERY_A", "PRIVATE_QUERY_B")
    )
