from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta, timezone
from threading import Barrier, Lock

import pytest

from decision_agent.application import (
    FormalRequest,
    FormalRequestExecutor,
    MemoryContextStatus,
    MemoryPersistenceStatus,
    MemorySummarizationStatus,
    SessionMemoryReadError,
)
from decision_agent.context import ConversationMemoryProjector
from decision_agent.coordination import Coordinator
from decision_agent.coordination.models import CoordinatorResult, CoordinatorStatus
from decision_agent.memory import (
    InMemorySessionMemoryStore,
    RollingSummaryDraft,
    RollingSummaryGenerationError,
    RollingSummaryPolicy,
    RollingSummaryService,
    SessionMemoryContentionError,
    SessionMemoryCorruptionError,
    SessionMemorySnapshot,
    SessionMemoryUnavailableError,
    SessionTurn,
    SessionTurnConflictError,
    SessionVersionConflictError,
)
from decision_agent.routing.models import RequestRoute, RouterDecision
from decision_agent.skills.registry import SkillRegistry

pytestmark = pytest.mark.offline_integration

NOW = datetime(2026, 7, 24, tzinfo=UTC)


def _result(*, selected: bool = False) -> CoordinatorResult:
    return CoordinatorResult(
        status=CoordinatorStatus.COMPLETED,
        route=RequestRoute.KNOWLEDGE,
        skill_name="knowledge",
        answer="ANSWER_SECRET_DO_NOT_LEAK",
        citations=["[E1]"],
        coordinator_steps=("route_request", "select_skill", "execute_skill"),
        tool_steps=("run",),
        memory_context_selected=selected,
    )


class _Coordinator:
    def __init__(self, *, selected: bool = False) -> None:
        self.selected = selected
        self.calls: list[dict[str, object]] = []

    async def execute(self, **kwargs: object) -> CoordinatorResult:
        self.calls.append(kwargs)
        return _result(selected=self.selected)


class _ForbiddenProjector:
    def __init__(self) -> None:
        self.calls = 0

    def project(self, _: SessionMemorySnapshot) -> None:
        self.calls += 1
        raise AssertionError("Memory projector must not run when Memory is disabled")


class _ForbiddenSummaryService:
    def __init__(self) -> None:
        self.calls = 0

    def compact_snapshot_if_needed(self, *_: object) -> None:
        self.calls += 1
        raise AssertionError("Rolling summary must not run when Memory is disabled")


class _Store:
    def __init__(self, snapshot: SessionMemorySnapshot | Exception) -> None:
        self.snapshot = snapshot
        self.read_calls: list[str] = []
        self.mutations = 0
        self.append_calls: list[tuple[SessionTurn, int]] = []

    def read(self, session_id: str) -> SessionMemorySnapshot:
        self.read_calls.append(session_id)
        if isinstance(self.snapshot, Exception):
            raise self.snapshot
        return self.snapshot

    def append_turn(self, turn: SessionTurn, *, expected_version: int) -> SessionMemorySnapshot:
        self.mutations += 1
        self.append_calls.append((turn, expected_version))
        assert isinstance(self.snapshot, SessionMemorySnapshot)
        return self.snapshot

    def clear(self, *_: object, **__: object) -> None:
        self.mutations += 1

    def compact(self, *_: object, **__: object) -> None:
        self.mutations += 1


class _LegacyRouter:
    async def route(self, *, user_query: str) -> RouterDecision:
        return _unsupported_decision()


class _ContextRouter(_LegacyRouter):
    def __init__(self) -> None:
        self.selected_items: tuple[object, ...] = ()

    async def route_with_context(
        self, *, user_query: str, selected_items: tuple[object, ...]
    ) -> RouterDecision:
        self.selected_items = selected_items
        return _unsupported_decision()


def _snapshot() -> SessionMemorySnapshot:
    return SessionMemorySnapshot(
        session_id="session-1",
        version=3,
        turns=(
            SessionTurn(
                session_id="session-1",
                turn_id="turn-1",
                request_id="request-old",
                user_text="HISTORY_USER_SECRET [D12]",
                assistant_text="HISTORY_ASSISTANT_SECRET [E2]",
                created_at=NOW,
            ),
        ),
    )


def _unsupported_decision() -> RouterDecision:
    return RouterDecision(
        route=RequestRoute.UNSUPPORTED,
        normalized_query="query",
        decision_reason="unsupported",
        knowledge_subquery=None,
        data_subquery=None,
        missing_information=None,
        confidence=1,
    )


@pytest.mark.asyncio
async def test_no_session_keeps_stateless_execution_and_zero_store_io() -> None:
    coordinator = _Coordinator()
    store = _Store(_snapshot())
    response = await FormalRequestExecutor(
        coordinator=coordinator, memory_store=store, memory_projector=ConversationMemoryProjector()
    ).execute(FormalRequest(request_id="request-1", user_query="query"))
    assert response.memory_context_status is MemoryContextStatus.NOT_REQUESTED
    assert response.memory_persistence_status is MemoryPersistenceStatus.NOT_REQUESTED
    assert response.memory_summarization_status is MemorySummarizationStatus.NOT_REQUESTED
    assert store.read_calls == [] and store.mutations == 0
    assert coordinator.calls[0]["user_query"] == "query"
    assert coordinator.calls[0]["request_id"] == "request-1"


@pytest.mark.asyncio
async def test_no_session_without_configured_store_keeps_stateless_execution() -> None:
    coordinator = _Coordinator()
    response = await FormalRequestExecutor(
        coordinator=coordinator,
        memory_store=None,
        memory_projector=ConversationMemoryProjector(),
    ).execute(FormalRequest(request_id="request-1", user_query="query"))

    assert response.memory_context_status is MemoryContextStatus.NOT_REQUESTED
    assert response.memory_persistence_status is MemoryPersistenceStatus.NOT_REQUESTED
    assert response.memory_summarization_status is MemorySummarizationStatus.NOT_REQUESTED
    assert coordinator.calls[0]["user_query"] == "query"
    assert coordinator.calls[0]["request_id"] == "request-1"


@pytest.mark.asyncio
async def test_session_reads_once_and_passes_only_typed_projection() -> None:
    coordinator = _Coordinator(selected=True)
    store = _Store(_snapshot())
    response = await FormalRequestExecutor(
        coordinator=coordinator, memory_store=store, memory_projector=ConversationMemoryProjector()
    ).execute(FormalRequest(request_id="request-1", user_query="current", session_id="session-1"))
    projection = coordinator.calls[0]["conversation_memory"]
    assert response.memory_context_status is MemoryContextStatus.PROJECTED
    assert response.memory_persistence_status is MemoryPersistenceStatus.PERSISTED
    assert store.read_calls == ["session-1"] and store.mutations == 1
    turn, expected_version = store.append_calls[0]
    assert expected_version == 3
    assert turn.session_id == "session-1"
    assert turn.request_id == "request-1"
    assert turn.user_text == "current"
    assert turn.assistant_text == "ANSWER_SECRET_DO_NOT_LEAK"
    assert turn.created_at.tzinfo is UTC
    assert projection is not None
    assert "[historical-D12]" in projection.content and "[historical-E2]" in projection.content
    assert "[D12]" not in projection.content and "[E2]" not in projection.content


@pytest.mark.asyncio
async def test_context_aware_router_consumption_maps_to_projected() -> None:
    router = _ContextRouter()
    response = await FormalRequestExecutor(
        coordinator=Coordinator(router=router, registry=SkillRegistry()),
        memory_store=_Store(_snapshot()),
        memory_projector=ConversationMemoryProjector(),
    ).execute(FormalRequest(request_id="request-1", user_query="current", session_id="session-1"))
    assert any(
        getattr(item, "kind", None).value == "conversation_memory" for item in router.selected_items
    )
    assert response.result.memory_context_selected is True
    assert response.memory_context_status is MemoryContextStatus.PROJECTED
    assert response.memory_persistence_status is MemoryPersistenceStatus.SKIPPED


@pytest.mark.asyncio
async def test_legacy_router_without_skill_consumption_maps_to_omitted() -> None:
    response = await FormalRequestExecutor(
        coordinator=Coordinator(router=_LegacyRouter(), registry=SkillRegistry()),
        memory_store=_Store(_snapshot()),
        memory_projector=ConversationMemoryProjector(),
    ).execute(FormalRequest(request_id="request-1", user_query="current", session_id="session-1"))
    assert response.result.memory_context_selected is False
    assert response.memory_context_status is MemoryContextStatus.OMITTED_BY_BUDGET
    assert response.memory_persistence_status is MemoryPersistenceStatus.SKIPPED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        SessionMemoryUnavailableError(operation="read"),
        SessionMemoryCorruptionError(session_id="SESSION_SECRET_DO_NOT_LEAK"),
        RuntimeError(
            "RAW_STORE_SECRET_DO_NOT_LEAK "
            "redis://user:password@example.invalid/15 "
            "SESSION_SECRET_DO_NOT_LEAK"
        ),
    ],
)
async def test_read_failure_is_safe_and_blocks_coordinator(failure: Exception) -> None:
    coordinator = _Coordinator()
    store = _Store(failure)
    executor = FormalRequestExecutor(
        coordinator=coordinator, memory_store=store, memory_projector=ConversationMemoryProjector()
    )
    with pytest.raises(SessionMemoryReadError) as raised:
        await executor.execute(
            FormalRequest(request_id="request-1", user_query="query", session_id="x")
        )
    assert raised.value.code == "session_memory_read_failed"
    assert raised.value.__cause__ is None and raised.value.__context__ is None
    for rendered in (str(raised.value), repr(raised.value)):
        assert "RAW_STORE_SECRET_DO_NOT_LEAK" not in rendered
        assert "SESSION_SECRET_DO_NOT_LEAK" not in rendered
        assert "redis://" not in rendered
    assert coordinator.calls == [] and store.read_calls == ["x"]


@pytest.mark.asyncio
async def test_disabled_memory_with_session_bypasses_memory_services() -> None:
    coordinator = _Coordinator()
    projector = _ForbiddenProjector()
    summary_service = _ForbiddenSummaryService()
    executor = FormalRequestExecutor(
        coordinator=coordinator,
        memory_store=None,
        memory_projector=projector,  # type: ignore[arg-type]
        rolling_summary_service=summary_service,  # type: ignore[arg-type]
    )
    response = await executor.execute(
        FormalRequest(request_id="request-1", user_query="query", session_id="x")
    )

    assert response.result.status is CoordinatorStatus.COMPLETED
    assert response.memory_context_status is MemoryContextStatus.NOT_REQUESTED
    assert response.memory_persistence_status is MemoryPersistenceStatus.NOT_REQUESTED
    assert response.memory_summarization_status is MemorySummarizationStatus.NOT_REQUESTED
    assert coordinator.calls[0]["user_query"] == "query"
    assert coordinator.calls[0]["request_id"] == "request-1"
    assert projector.calls == summary_service.calls == 0


class _ResultCoordinator(_Coordinator):
    def __init__(self, result: CoordinatorResult) -> None:
        super().__init__()
        self.result = result

    async def execute(self, **kwargs: object) -> CoordinatorResult:
        self.calls.append(kwargs)
        return self.result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        CoordinatorResult(status=CoordinatorStatus.UNSUPPORTED, route=RequestRoute.UNSUPPORTED),
        CoordinatorResult(
            status=CoordinatorStatus.COMPLETED,
            route=RequestRoute.KNOWLEDGE,
            skill_name="knowledge",
            answer="   ",
            tool_steps=("run",),
        ),
    ],
)
async def test_unsuccessful_or_blank_results_skip_persistence(result: CoordinatorResult) -> None:
    store = _Store(_snapshot())
    response = await FormalRequestExecutor(
        coordinator=_ResultCoordinator(result),
        memory_store=store,
        memory_projector=ConversationMemoryProjector(),
    ).execute(FormalRequest(request_id="request-1", user_query="query", session_id="session-1"))
    assert response.memory_persistence_status is MemoryPersistenceStatus.SKIPPED
    assert response.memory_summarization_status is MemorySummarizationStatus.SKIPPED
    assert store.read_calls == ["session-1"] and store.append_calls == []


@pytest.mark.asyncio
async def test_new_turn_uses_injected_utc_clock() -> None:
    store = _Store(_snapshot())
    local_time = datetime(2026, 7, 24, 16, 30, tzinfo=timezone(timedelta(hours=8)))
    response = await FormalRequestExecutor(
        coordinator=_Coordinator(),
        memory_store=store,
        memory_projector=ConversationMemoryProjector(),
        clock=lambda: local_time,
    ).execute(FormalRequest(request_id="request-1", user_query="query", session_id="session-1"))
    assert response.memory_persistence_status is MemoryPersistenceStatus.PERSISTED
    assert store.append_calls[0][0].created_at == local_time.astimezone(UTC)


class _FailingCoordinator:
    async def execute(self, **_: object) -> CoordinatorResult:
        raise RuntimeError("coordinator failure")


@pytest.mark.asyncio
async def test_coordinator_exception_does_not_append() -> None:
    store = _Store(_snapshot())
    executor = FormalRequestExecutor(
        coordinator=_FailingCoordinator(),  # type: ignore[arg-type]
        memory_store=store,
        memory_projector=ConversationMemoryProjector(),
    )
    with pytest.raises(RuntimeError, match="coordinator failure"):
        await executor.execute(
            FormalRequest(request_id="request-1", user_query="query", session_id="session-1")
        )
    assert store.read_calls == ["session-1"] and store.append_calls == []


class _AppendFailureStore(_Store):
    def __init__(self, failure: BaseException) -> None:
        super().__init__(_snapshot())
        self.failure = failure

    def append_turn(self, turn: SessionTurn, *, expected_version: int) -> SessionMemorySnapshot:
        self.mutations += 1
        self.append_calls.append((turn, expected_version))
        raise self.failure


class _RecordingInMemoryStore(InMemorySessionMemoryStore):
    def __init__(self) -> None:
        super().__init__()
        self.read_calls: list[str] = []
        self.append_calls: list[tuple[SessionTurn, int]] = []

    def read(self, session_id: str) -> SessionMemorySnapshot:
        self.read_calls.append(session_id)
        return super().read(session_id)

    def append_turn(self, turn: SessionTurn, *, expected_version: int) -> SessionMemorySnapshot:
        self.append_calls.append((turn, expected_version))
        return super().append_turn(turn, expected_version=expected_version)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected_status"),
    [
        (
            SessionVersionConflictError(
                session_id="SESSION_SECRET", expected_version=3, actual_version=4
            ),
            MemoryPersistenceStatus.VERSION_CONFLICT,
        ),
        (
            SessionTurnConflictError(
                session_id="SESSION_SECRET", turn_id="turn", request_id="request"
            ),
            MemoryPersistenceStatus.IDEMPOTENCY_CONFLICT,
        ),
        (
            SessionMemoryUnavailableError(operation="append_turn"),
            MemoryPersistenceStatus.STORE_FAILURE,
        ),
        (
            SessionMemoryCorruptionError(session_id="SESSION_SECRET"),
            MemoryPersistenceStatus.STORE_FAILURE,
        ),
        (
            SessionMemoryContentionError(session_id="SESSION_SECRET", retries=2),
            MemoryPersistenceStatus.STORE_FAILURE,
        ),
        (
            RuntimeError("redis://user:password@example.invalid/15 SESSION_SECRET"),
            MemoryPersistenceStatus.STORE_FAILURE,
        ),
    ],
)
async def test_append_failure_maps_safely_without_losing_business_result(
    failure: BaseException, expected_status: MemoryPersistenceStatus
) -> None:
    store = _AppendFailureStore(failure)
    response = await FormalRequestExecutor(
        coordinator=_Coordinator(),
        memory_store=store,
        memory_projector=ConversationMemoryProjector(),
    ).execute(FormalRequest(request_id="request-1", user_query="query", session_id="session-1"))
    assert response.result.answer == "ANSWER_SECRET_DO_NOT_LEAK"
    assert response.memory_persistence_status is expected_status
    rendered = repr(response) + str(response)
    assert "SESSION_SECRET" not in rendered and "redis://" not in rendered
    assert len(store.append_calls) == 1


@pytest.mark.asyncio
async def test_append_cancellation_propagates() -> None:
    executor = FormalRequestExecutor(
        coordinator=_Coordinator(),
        memory_store=_AppendFailureStore(asyncio.CancelledError()),
        memory_projector=ConversationMemoryProjector(),
    )
    with pytest.raises(asyncio.CancelledError):
        await executor.execute(
            FormalRequest(request_id="request-1", user_query="query", session_id="session-1")
        )


@pytest.mark.asyncio
async def test_real_store_accepts_idempotent_retry_without_adding_a_turn() -> None:
    store = _RecordingInMemoryStore()
    created_at = datetime(2026, 7, 24, 10, 0, tzinfo=UTC)
    retry_clock = datetime(2026, 7, 24, 11, 0, tzinfo=UTC)
    executor = FormalRequestExecutor(
        coordinator=_Coordinator(),
        memory_store=store,
        memory_projector=ConversationMemoryProjector(),
        clock=lambda: created_at,
    )
    request = FormalRequest(request_id="request-1", user_query="query", session_id="session-1")
    first = await executor.execute(request)
    second = await FormalRequestExecutor(
        coordinator=_Coordinator(),
        memory_store=store,
        memory_projector=ConversationMemoryProjector(),
        clock=lambda: retry_clock,
    ).execute(request)
    snapshot = store.read("session-1")
    assert first.memory_persistence_status is MemoryPersistenceStatus.PERSISTED
    assert second.memory_persistence_status is MemoryPersistenceStatus.PERSISTED
    assert snapshot.version == 1 and len(snapshot.turns) == 1
    assert store.append_calls[-1][0].created_at == created_at
    assert store.read_calls[:2] == ["session-1", "session-1"]


@pytest.mark.asyncio
async def test_same_request_id_with_different_turn_id_uses_new_clock_before_conflict() -> None:
    store = _RecordingInMemoryStore()
    request = FormalRequest(request_id="request-1", user_query="query", session_id="session-1")
    stale_time = datetime(2026, 7, 24, 9, 0, tzinfo=UTC)
    recent_time = datetime(2026, 7, 24, 9, 30, tzinfo=UTC)
    new_time = datetime(2026, 7, 24, 10, 0, tzinfo=UTC)
    store.append_turn(
        SessionTurn(
            session_id="session-1",
            turn_id="legacy-turn",
            request_id=request.request_id,
            user_text="legacy query",
            assistant_text="legacy answer",
            created_at=stale_time,
        ),
        expected_version=0,
    )
    store.append_turn(
        SessionTurn(
            session_id="session-1",
            turn_id="other-turn",
            request_id="request-other",
            user_text="other query",
            assistant_text="other answer",
            created_at=recent_time,
        ),
        expected_version=1,
    )

    response = await FormalRequestExecutor(
        coordinator=_Coordinator(),
        memory_store=store,
        memory_projector=ConversationMemoryProjector(),
        clock=lambda: new_time,
    ).execute(request)

    candidate, expected_version = store.append_calls[-1]
    assert response.memory_persistence_status is MemoryPersistenceStatus.IDEMPOTENCY_CONFLICT
    assert candidate.turn_id != "legacy-turn"
    assert candidate.request_id == request.request_id
    assert candidate.created_at == new_time
    assert candidate.created_at not in {stale_time, recent_time}
    assert expected_version == 2
    assert store.read_calls == ["session-1"]


@pytest.mark.asyncio
async def test_reused_request_id_with_changed_content_fails_closed() -> None:
    store = InMemorySessionMemoryStore()
    executor = FormalRequestExecutor(
        coordinator=_Coordinator(),
        memory_store=store,
        memory_projector=ConversationMemoryProjector(),
    )
    first = await executor.execute(
        FormalRequest(request_id="request-1", user_query="original", session_id="session-1")
    )
    changed = await executor.execute(
        FormalRequest(request_id="request-1", user_query="changed", session_id="session-1")
    )
    snapshot = store.read("session-1")
    assert first.memory_persistence_status is MemoryPersistenceStatus.PERSISTED
    assert changed.memory_persistence_status is MemoryPersistenceStatus.IDEMPOTENCY_CONFLICT
    assert snapshot.version == 1 and snapshot.turns[0].user_text == "original"


class _BarrierStore(InMemorySessionMemoryStore):
    def __init__(self) -> None:
        super().__init__()
        self._barrier = Barrier(2)
        self._read_lock = Lock()
        self._read_count = 0

    def read(self, session_id: str) -> SessionMemorySnapshot:
        snapshot = super().read(session_id)
        with self._read_lock:
            self._read_count += 1
            should_wait = self._read_count <= 2
        if should_wait:
            self._barrier.wait()
        return snapshot


@pytest.mark.asyncio
async def test_concurrent_stale_snapshots_allow_one_persisted_turn() -> None:
    store = _BarrierStore()
    executor = FormalRequestExecutor(
        coordinator=_Coordinator(),
        memory_store=store,
        memory_projector=ConversationMemoryProjector(),
    )
    first, second = await asyncio.gather(
        executor.execute(
            FormalRequest(request_id="request-1", user_query="first", session_id="session-1")
        ),
        executor.execute(
            FormalRequest(request_id="request-2", user_query="second", session_id="session-1")
        ),
    )
    assert {first.memory_persistence_status, second.memory_persistence_status} == {
        MemoryPersistenceStatus.PERSISTED,
        MemoryPersistenceStatus.VERSION_CONFLICT,
    }
    snapshot = store.read("session-1")
    assert snapshot.version == 1 and len(snapshot.turns) == 1


class _CapturingSummarizer:
    def __init__(self, result: object = RollingSummaryDraft(summary_text="safe summary")) -> None:
        self.result = result
        self.requests: list[object] = []

    def summarize(self, request: object) -> object:
        self.requests.append(request)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class _SummaryRecordingStore(_RecordingInMemoryStore):
    def __init__(self) -> None:
        super().__init__()
        self.compact_calls: list[int] = []

    def compact(self, summary, compacted_turn_ids, *, expected_version):  # type: ignore[no-untyped-def]
        self.compact_calls.append(expected_version)
        return super().compact(summary, compacted_turn_ids, expected_version=expected_version)


def _summary_service(
    store: InMemorySessionMemoryStore, summarizer: _CapturingSummarizer
) -> RollingSummaryService:
    return RollingSummaryService(
        store=store,
        summarizer=summarizer,
        policy=RollingSummaryPolicy(
            trigger_turns=3, retain_recent_turns=1, max_source_chars=1_000, max_summary_chars=100
        ),
        clock=lambda: NOW,
    )


@pytest.mark.asyncio
async def test_persisted_turn_compacts_from_post_append_snapshot_without_second_read() -> None:
    store = _SummaryRecordingStore()
    for index in range(1, 3):
        store.append_turn(
            SessionTurn(
                session_id="session-1",
                turn_id=f"old-turn-{index}",
                request_id=f"old-request-{index}",
                user_text=f"old-user-{index}",
                assistant_text=f"old-assistant-{index}",
                created_at=NOW,
            ),
            expected_version=index - 1,
        )
    store.read_calls.clear()
    store.append_calls.clear()
    summarizer = _CapturingSummarizer()
    response = await FormalRequestExecutor(
        coordinator=_Coordinator(),
        memory_store=store,
        memory_projector=ConversationMemoryProjector(),
        rolling_summary_service=_summary_service(store, summarizer),
        clock=lambda: NOW,
    ).execute(FormalRequest(request_id="request-1", user_query="query", session_id="session-1"))

    assert response.result.answer == "ANSWER_SECRET_DO_NOT_LEAK"
    assert response.memory_persistence_status is MemoryPersistenceStatus.PERSISTED
    assert response.memory_summarization_status is MemorySummarizationStatus.COMPACTED
    assert store.read_calls == ["session-1"]
    assert len(store.append_calls) == 1
    assert len(summarizer.requests) == 1
    assert store.compact_calls == [3]
    final_snapshot = store.read("session-1")
    assert final_snapshot.version == 4 and final_snapshot.summary is not None


@pytest.mark.asyncio
async def test_persisted_turn_below_policy_threshold_is_not_needed() -> None:
    store = _SummaryRecordingStore()
    summarizer = _CapturingSummarizer()
    response = await FormalRequestExecutor(
        coordinator=_Coordinator(),
        memory_store=store,
        memory_projector=ConversationMemoryProjector(),
        rolling_summary_service=_summary_service(store, summarizer),
        clock=lambda: NOW,
    ).execute(FormalRequest(request_id="request-1", user_query="query", session_id="session-1"))
    assert response.memory_persistence_status is MemoryPersistenceStatus.PERSISTED
    assert response.memory_summarization_status is MemorySummarizationStatus.NOT_NEEDED
    assert store.read_calls == ["session-1"]
    assert summarizer.requests == [] and store.compact_calls == []


@pytest.mark.asyncio
async def test_provider_failure_preserves_persisted_turn_and_business_result() -> None:
    store = _SummaryRecordingStore()
    for index in range(1, 3):
        store.append_turn(
            SessionTurn(
                session_id="session-1",
                turn_id=f"old-turn-{index}",
                request_id=f"old-request-{index}",
                user_text=f"old-user-{index}",
                assistant_text=f"old-assistant-{index}",
                created_at=NOW,
            ),
            expected_version=index - 1,
        )
    store.read_calls.clear()
    store.append_calls.clear()
    summarizer = _CapturingSummarizer(RollingSummaryGenerationError(stage="provider"))
    response = await FormalRequestExecutor(
        coordinator=_Coordinator(),
        memory_store=store,
        memory_projector=ConversationMemoryProjector(),
        rolling_summary_service=_summary_service(store, summarizer),
        clock=lambda: NOW,
    ).execute(FormalRequest(request_id="request-1", user_query="query", session_id="session-1"))
    assert response.result.answer == "ANSWER_SECRET_DO_NOT_LEAK"
    assert response.memory_persistence_status is MemoryPersistenceStatus.PERSISTED
    assert response.memory_summarization_status is MemorySummarizationStatus.PROVIDER_FAILURE
    assert store.read_calls == ["session-1"]
    assert len(store.append_calls) == 1
    assert len(summarizer.requests) == 1 and store.compact_calls == []


class _CompactConflictStore(_SummaryRecordingStore):
    def compact(self, summary, compacted_turn_ids, *, expected_version):  # type: ignore[no-untyped-def]
        self.append_turn(
            SessionTurn(
                session_id=summary.session_id,
                turn_id="concurrent-turn",
                request_id="concurrent-request",
                user_text="concurrent-user",
                assistant_text="concurrent-assistant",
                created_at=NOW,
            ),
            expected_version=expected_version,
        )
        return super().compact(summary, compacted_turn_ids, expected_version=expected_version)


@pytest.mark.asyncio
async def test_compact_conflict_uses_post_append_version_without_retry() -> None:
    store = _CompactConflictStore()
    for index in range(1, 3):
        store.append_turn(
            SessionTurn(
                session_id="session-1",
                turn_id=f"old-turn-{index}",
                request_id=f"old-request-{index}",
                user_text=f"old-user-{index}",
                assistant_text=f"old-assistant-{index}",
                created_at=NOW,
            ),
            expected_version=index - 1,
        )
    store.read_calls.clear()
    store.append_calls.clear()
    summarizer = _CapturingSummarizer()
    response = await FormalRequestExecutor(
        coordinator=_Coordinator(),
        memory_store=store,
        memory_projector=ConversationMemoryProjector(),
        rolling_summary_service=_summary_service(store, summarizer),
        clock=lambda: NOW,
    ).execute(FormalRequest(request_id="request-1", user_query="query", session_id="session-1"))
    assert response.result.answer == "ANSWER_SECRET_DO_NOT_LEAK"
    assert response.memory_persistence_status is MemoryPersistenceStatus.PERSISTED
    assert response.memory_summarization_status is MemorySummarizationStatus.VERSION_CONFLICT
    assert store.read_calls == ["session-1"]
    assert len(summarizer.requests) == 1 and store.compact_calls == [3]
    assert store.read("session-1").summary is None


class _CompactFailureStore(_SummaryRecordingStore):
    def compact(self, summary, compacted_turn_ids, *, expected_version):  # type: ignore[no-untyped-def]
        self.compact_calls.append(expected_version)
        raise SessionMemoryUnavailableError(operation="compact")


@pytest.mark.asyncio
async def test_compact_store_failure_preserves_persisted_turn_and_hides_details() -> None:
    store = _CompactFailureStore()
    for index in range(1, 3):
        store.append_turn(
            SessionTurn(
                session_id="session-1",
                turn_id=f"old-turn-{index}",
                request_id=f"old-request-{index}",
                user_text=f"old-user-{index}",
                assistant_text=f"old-assistant-{index}",
                created_at=NOW,
            ),
            expected_version=index - 1,
        )
    store.read_calls.clear()
    store.append_calls.clear()
    summarizer = _CapturingSummarizer()
    response = await FormalRequestExecutor(
        coordinator=_Coordinator(),
        memory_store=store,
        memory_projector=ConversationMemoryProjector(),
        rolling_summary_service=_summary_service(store, summarizer),
        clock=lambda: NOW,
    ).execute(FormalRequest(request_id="request-1", user_query="query", session_id="session-1"))
    assert response.result.answer == "ANSWER_SECRET_DO_NOT_LEAK"
    assert response.memory_persistence_status is MemoryPersistenceStatus.PERSISTED
    assert response.memory_summarization_status is MemorySummarizationStatus.STORE_FAILURE
    assert store.read_calls == ["session-1"]
    assert len(summarizer.requests) == 1 and store.compact_calls == [3]
    assert "session-1" not in repr(response)


class _CancelledSummaryService:
    def compact_snapshot_if_needed(self, *_: object) -> object:
        raise asyncio.CancelledError()


@pytest.mark.asyncio
async def test_summary_cancellation_propagates_after_append() -> None:
    store = _RecordingInMemoryStore()
    with pytest.raises(asyncio.CancelledError):
        await FormalRequestExecutor(
            coordinator=_Coordinator(),
            memory_store=store,
            memory_projector=ConversationMemoryProjector(),
            rolling_summary_service=_CancelledSummaryService(),  # type: ignore[arg-type]
        ).execute(FormalRequest(request_id="request-1", user_query="query", session_id="session-1"))
    assert store.read_calls == ["session-1"] and len(store.append_calls) == 1


class _UnexpectedSummaryFailureService:
    def compact_snapshot_if_needed(self, *_: object) -> object:
        raise RuntimeError("redis://user:password@example.invalid/15 SUMMARY_SECRET_DO_NOT_LEAK")


@pytest.mark.asyncio
async def test_unexpected_summary_failure_is_safe_and_preserves_append() -> None:
    store = _RecordingInMemoryStore()
    response = await FormalRequestExecutor(
        coordinator=_Coordinator(),
        memory_store=store,
        memory_projector=ConversationMemoryProjector(),
        rolling_summary_service=_UnexpectedSummaryFailureService(),  # type: ignore[arg-type]
    ).execute(FormalRequest(request_id="request-1", user_query="query", session_id="session-1"))
    assert response.result.answer == "ANSWER_SECRET_DO_NOT_LEAK"
    assert response.memory_persistence_status is MemoryPersistenceStatus.PERSISTED
    assert response.memory_summarization_status is MemorySummarizationStatus.STORE_FAILURE
    assert "SUMMARY_SECRET_DO_NOT_LEAK" not in repr(response)
    assert "redis://" not in repr(response)
