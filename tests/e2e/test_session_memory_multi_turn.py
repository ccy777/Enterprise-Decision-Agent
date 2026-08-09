"""Offline end-to-end acceptance for the formal multi-turn session-memory path."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest

from decision_agent.application import (
    FormalRequest,
    FormalRequestExecutor,
    MemoryContextStatus,
    MemoryPersistenceStatus,
    MemorySummarizationStatus,
)
from decision_agent.context import ConversationMemoryProjector
from decision_agent.context.models import ContextItem, ContextKind
from decision_agent.coordination import Coordinator
from decision_agent.coordination.models import CoordinatorStatus
from decision_agent.memory import (
    InMemorySessionMemoryStore,
    RollingSummaryDraft,
    RollingSummaryPolicy,
    RollingSummaryRequest,
    RollingSummaryService,
    SessionMemorySnapshot,
    SessionSummary,
    SessionTurn,
)
from decision_agent.routing.models import RequestRoute, RouterDecision
from decision_agent.skills.enterprise_knowledge_qa import EnterpriseKnowledgeQASkill
from decision_agent.skills.native_runtime import NativeToolCallingSkillExecutor
from decision_agent.skills.registry import SkillRegistry
from decision_agent.tool_calling.models import AgentToolResult

pytestmark = [pytest.mark.e2e, pytest.mark.offline_integration]

NOW = datetime(2026, 7, 25, tzinfo=UTC)


@dataclass(frozen=True)
class _RouterCall:
    user_query: str
    selected_kinds: tuple[ContextKind, ...]
    conversation_memory: str | None


class _RecordingRouter:
    """Deterministic Router replacement that records only formal selected Context."""

    def __init__(self) -> None:
        self.calls: list[_RouterCall] = []

    async def route(self, *, user_query: str) -> RouterDecision:
        return self._decision(user_query)

    async def route_with_context(
        self, *, user_query: str, selected_items: tuple[ContextItem, ...]
    ) -> RouterDecision:
        memory_items = [
            item for item in selected_items if item.kind is ContextKind.CONVERSATION_MEMORY
        ]
        assert len(memory_items) <= 1
        self.calls.append(
            _RouterCall(
                user_query=user_query,
                selected_kinds=tuple(item.kind for item in selected_items),
                conversation_memory=None if not memory_items else memory_items[0].content,
            )
        )
        return self._decision(user_query)

    @staticmethod
    def _decision(user_query: str) -> RouterDecision:
        return RouterDecision(
            route=RequestRoute.KNOWLEDGE,
            normalized_query=user_query,
            decision_reason="deterministic_e2e_knowledge_route",
            knowledge_subquery=user_query,
            data_subquery=None,
            missing_information=None,
            confidence=1,
        )


class _RecordingNativeToolModel:
    """Script the formal native-tool protocol while retaining initial provider messages."""

    def __init__(self) -> None:
        self.initial_messages: list[tuple[dict[str, object], ...]] = []
        self.calls = 0

    async def complete(
        self,
        *,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
        tool_choice: str,
        response_format: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        del tools, response_format
        self.calls += 1
        if tool_choice == "required":
            self.initial_messages.append(tuple(messages))
            user_content = messages[1]["content"]
            assert isinstance(user_content, str)
            query = user_content.split("Required subquery: ", maxsplit=1)[1].split(
                "\n\n", maxsplit=1
            )[0]
            return {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"call-{self.calls}",
                                    "type": "function",
                                    "function": {
                                        "name": "run_knowledge_agent",
                                        "arguments": json.dumps({"query": query}),
                                    },
                                }
                            ],
                        },
                    }
                ]
            }

        tool_message = next(message for message in messages if message["role"] == "tool")
        tool_payload = json.loads(str(tool_message["content"]))
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            {
                                "answer": tool_payload["answer"],
                                "citations": tool_payload["citations"],
                            }
                        )
                    },
                }
            ]
        }


class _RecordingKnowledgeTool:
    """Deterministic high-level tool whose answer proves the current request boundary."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    async def run(self, *, query: str) -> AgentToolResult:
        self.queries.append(query)
        return AgentToolResult(
            status="succeeded",
            answer=f"CURRENT_ANSWER::{query} [E1]",
            citations=["[E1]"],
        )


class _RecordingStore:
    """Count formal Store calls while delegating all semantics to the real InMemory Store."""

    def __init__(self) -> None:
        self._store = InMemorySessionMemoryStore(clock=lambda: NOW)
        self.read_calls: dict[str, int] = defaultdict(int)
        self.append_calls: dict[str, int] = defaultdict(int)
        self.compact_calls: dict[str, int] = defaultdict(int)
        self.clear_calls: dict[str, int] = defaultdict(int)

    def read(self, session_id: str) -> SessionMemorySnapshot:
        self.read_calls[session_id] += 1
        return self._store.read(session_id)

    def append_turn(self, turn: SessionTurn, *, expected_version: int) -> SessionMemorySnapshot:
        self.append_calls[turn.session_id] += 1
        return self._store.append_turn(turn, expected_version=expected_version)

    def compact(
        self,
        summary: SessionSummary,
        compacted_turn_ids: tuple[str, ...],
        *,
        expected_version: int,
    ) -> SessionMemorySnapshot:
        self.compact_calls[summary.session_id] += 1
        return self._store.compact(summary, compacted_turn_ids, expected_version=expected_version)

    def clear(self, session_id: str, *, expected_version: int) -> SessionMemorySnapshot:
        self.clear_calls[session_id] += 1
        return self._store.clear(session_id, expected_version=expected_version)

    def snapshot_for_assertion(self, session_id: str) -> SessionMemorySnapshot:
        """Read the real Store outside the formal request path for final-state assertions."""
        return self._store.read(session_id)


@dataclass(frozen=True)
class _SummaryCall:
    session_id: str
    source_version: int
    turn_ids: tuple[str, ...]


class _RecordingSummarizer:
    """Deterministic no-network RollingSummarizer that records no historical bodies."""

    def __init__(self) -> None:
        self.calls: list[_SummaryCall] = []

    def summarize(self, request: RollingSummaryRequest) -> RollingSummaryDraft:
        self.calls.append(
            _SummaryCall(
                session_id=request.session_id,
                source_version=request.source_version,
                turn_ids=tuple(turn.turn_id for turn in request.turns),
            )
        )
        return RollingSummaryDraft(summary_text="SUMMARY_MARKER")


@dataclass(frozen=True)
class _Harness:
    executor: FormalRequestExecutor
    store: _RecordingStore
    router: _RecordingRouter
    native_model: _RecordingNativeToolModel
    knowledge_tool: _RecordingKnowledgeTool
    summarizer: _RecordingSummarizer


def _harness() -> _Harness:
    store = _RecordingStore()
    router = _RecordingRouter()
    native_model = _RecordingNativeToolModel()
    knowledge_tool = _RecordingKnowledgeTool()
    runtime = NativeToolCallingSkillExecutor(
        model=native_model,
        knowledge_tool=knowledge_tool,
        data_tool=knowledge_tool,
    )
    registry = SkillRegistry()
    registry.register(EnterpriseKnowledgeQASkill(runtime=runtime))
    summarizer = _RecordingSummarizer()
    executor = FormalRequestExecutor(
        coordinator=Coordinator(router=router, registry=registry),
        memory_store=store,
        memory_projector=ConversationMemoryProjector(),
        rolling_summary_service=RollingSummaryService(
            store=store,
            summarizer=summarizer,
            policy=RollingSummaryPolicy(
                trigger_turns=3,
                retain_recent_turns=1,
                max_source_chars=4_000,
                max_summary_chars=200,
            ),
            clock=lambda: NOW,
        ),
        clock=lambda: NOW,
    )
    return _Harness(executor, store, router, native_model, knowledge_tool, summarizer)


def _initial_system_message(model: _RecordingNativeToolModel, index: int) -> str:
    message = model.initial_messages[index][0]
    assert message["role"] == "system" and isinstance(message["content"], str)
    return message["content"]


def _initial_user_message(model: _RecordingNativeToolModel, index: int) -> str:
    message = model.initial_messages[index][1]
    assert message["role"] == "user" and isinstance(message["content"], str)
    return message["content"]


@pytest.mark.asyncio
async def test_requests_without_session_remain_stateless_end_to_end() -> None:
    harness = _harness()
    first_marker = "FIRST_TURN_USER_MARKER"
    second_marker = "SECOND_TURN_USER_MARKER"

    first = await harness.executor.execute(
        FormalRequest(request_id="stateless-1", user_query=first_marker)
    )
    second = await harness.executor.execute(
        FormalRequest(request_id="stateless-2", user_query=second_marker)
    )

    for response, marker in ((first, first_marker), (second, second_marker)):
        assert response.result.status is CoordinatorStatus.COMPLETED
        assert response.result.answer == f"CURRENT_ANSWER::{marker} [E1]"
        assert response.memory_context_status is MemoryContextStatus.NOT_REQUESTED
        assert response.memory_persistence_status is MemoryPersistenceStatus.NOT_REQUESTED
        assert response.memory_summarization_status is MemorySummarizationStatus.NOT_REQUESTED
    assert all(call.conversation_memory is None for call in harness.router.calls)
    assert all(
        not kinds.count(ContextKind.CONVERSATION_MEMORY)
        for kinds in (call.selected_kinds for call in harness.router.calls)
    )
    assert all(
        second_marker not in message
        for message in (_initial_user_message(harness.native_model, 0),)
    )
    assert first_marker not in _initial_user_message(harness.native_model, 1)
    assert not harness.store.read_calls
    assert not harness.store.append_calls
    assert not harness.store.compact_calls
    assert not harness.summarizer.calls


@pytest.mark.asyncio
async def test_second_request_consumes_first_persisted_turn_in_same_session() -> None:
    harness = _harness()
    session_id = "same-session"
    first_marker = "FIRST_TURN_USER_MARKER [D1]"
    second_marker = "SECOND_TURN_USER_MARKER"

    first = await harness.executor.execute(
        FormalRequest(request_id="req-1", user_query=first_marker, session_id=session_id)
    )
    second = await harness.executor.execute(
        FormalRequest(request_id="req-2", user_query=second_marker, session_id=session_id)
    )

    assert first.memory_context_status is MemoryContextStatus.EMPTY
    assert first.memory_persistence_status is MemoryPersistenceStatus.PERSISTED
    assert first.memory_summarization_status is MemorySummarizationStatus.NOT_NEEDED
    assert second.result.status is CoordinatorStatus.COMPLETED
    assert second.memory_context_status is MemoryContextStatus.PROJECTED
    assert second.memory_persistence_status is MemoryPersistenceStatus.PERSISTED
    assert second.memory_summarization_status is MemorySummarizationStatus.NOT_NEEDED
    assert harness.router.calls[0].conversation_memory is None
    memory = harness.router.calls[1].conversation_memory
    assert memory is not None
    assert first_marker.split(" ")[0] in memory
    assert "[historical-D1]" in memory and "[historical-E1]" in memory
    user_message = _initial_user_message(harness.native_model, 1)
    system_message = _initial_system_message(harness.native_model, 1)
    assert first_marker.split(" ")[0] in user_message
    assert first_marker.split(" ")[0] not in system_message
    assert "[historical-D1]" in user_message and "[historical-E1]" in user_message
    assert second.result.citations == ["[E1]"]
    snapshot = harness.store.snapshot_for_assertion(session_id)
    assert snapshot.version == 2 and snapshot.summary is None
    assert [turn.request_id for turn in snapshot.turns] == ["req-1", "req-2"]
    assert snapshot.turns[0].turn_id != snapshot.turns[1].turn_id
    assert harness.store.read_calls[session_id] == 2
    assert harness.store.append_calls[session_id] == 2
    assert harness.store.compact_calls[session_id] == 0
    assert not harness.summarizer.calls
    assert first_marker not in repr(second) and session_id not in repr(second)


@pytest.mark.asyncio
async def test_request_after_compaction_consumes_summary_and_retained_turns() -> None:
    harness = _harness()
    session_id = "summary-session"

    await harness.executor.execute(
        FormalRequest(
            request_id="summary-1", user_query="COMPACTED_TURN_1 [D1]", session_id=session_id
        )
    )
    await harness.executor.execute(
        FormalRequest(
            request_id="summary-2", user_query="COMPACTED_TURN_2 [E1]", session_id=session_id
        )
    )
    compacted = await harness.executor.execute(
        FormalRequest(
            request_id="summary-3",
            user_query="RETAINED_TURN_MARKER [D1]",
            session_id=session_id,
        )
    )
    after_compact = harness.store.snapshot_for_assertion(session_id)
    follow_up = await harness.executor.execute(
        FormalRequest(
            request_id="summary-4",
            user_query="POST_COMPACTION_CURRENT_MARKER",
            session_id=session_id,
        )
    )

    assert compacted.memory_persistence_status is MemoryPersistenceStatus.PERSISTED
    assert compacted.memory_summarization_status is MemorySummarizationStatus.COMPACTED
    assert after_compact.version == 4 and after_compact.summary is not None
    assert after_compact.summary.summary_text == "SUMMARY_MARKER"
    assert [turn.request_id for turn in after_compact.turns] == ["summary-3"]
    assert len(harness.summarizer.calls) == 1
    assert harness.summarizer.calls[0].source_version == 3
    assert len(harness.summarizer.calls[0].turn_ids) == 2
    assert follow_up.memory_context_status is MemoryContextStatus.PROJECTED
    assert follow_up.memory_persistence_status is MemoryPersistenceStatus.PERSISTED
    assert follow_up.memory_summarization_status is MemorySummarizationStatus.NOT_NEEDED
    memory = harness.router.calls[3].conversation_memory
    assert memory is not None
    assert memory.index("SUMMARY_MARKER") < memory.index("RETAINED_TURN_MARKER")
    assert "[historical-D1]" in memory and "[historical-E1]" in memory
    user_message = _initial_user_message(harness.native_model, 3)
    system_message = _initial_system_message(harness.native_model, 3)
    assert "SUMMARY_MARKER" in user_message and "RETAINED_TURN_MARKER" in user_message
    assert "SUMMARY_MARKER" not in system_message and "RETAINED_TURN_MARKER" not in system_message
    assert follow_up.result.answer == "CURRENT_ANSWER::POST_COMPACTION_CURRENT_MARKER [E1]"
    assert follow_up.result.citations == ["[E1]"]
    final_snapshot = harness.store.snapshot_for_assertion(session_id)
    assert final_snapshot.version == 5 and final_snapshot.summary is not None
    assert [turn.request_id for turn in final_snapshot.turns] == ["summary-3", "summary-4"]
    assert harness.store.read_calls[session_id] == 4
    assert harness.store.append_calls[session_id] == 4
    assert harness.store.compact_calls[session_id] == 1


@pytest.mark.asyncio
async def test_interleaved_sessions_remain_fully_isolated() -> None:
    harness = _harness()
    session_a = "session-a"
    session_b = "session-b"
    marker_a = "SESSION_A_PRIVATE_MARKER"
    marker_b = "SESSION_B_PRIVATE_MARKER"

    await harness.executor.execute(
        FormalRequest(request_id="a-1", user_query=marker_a, session_id=session_a)
    )
    await harness.executor.execute(
        FormalRequest(request_id="b-1", user_query=marker_b, session_id=session_b)
    )
    second_a = await harness.executor.execute(
        FormalRequest(request_id="a-2", user_query="A_FOLLOW_UP", session_id=session_a)
    )
    second_b = await harness.executor.execute(
        FormalRequest(request_id="b-2", user_query="B_FOLLOW_UP", session_id=session_b)
    )

    for response in (second_a, second_b):
        assert response.memory_context_status is MemoryContextStatus.PROJECTED
        assert response.memory_persistence_status is MemoryPersistenceStatus.PERSISTED
        assert response.memory_summarization_status is MemorySummarizationStatus.NOT_NEEDED
        assert response.result.citations == ["[E1]"]
    memory_a = harness.router.calls[2].conversation_memory
    memory_b = harness.router.calls[3].conversation_memory
    assert memory_a is not None and marker_a in memory_a and marker_b not in memory_a
    assert memory_b is not None and marker_b in memory_b and marker_a not in memory_b
    for index, own_marker, other_marker in ((2, marker_a, marker_b), (3, marker_b, marker_a)):
        assert own_marker in _initial_user_message(harness.native_model, index)
        assert other_marker not in _initial_user_message(harness.native_model, index)
        assert own_marker not in _initial_system_message(harness.native_model, index)
    snapshot_a = harness.store.snapshot_for_assertion(session_a)
    snapshot_b = harness.store.snapshot_for_assertion(session_b)
    assert snapshot_a.version == snapshot_b.version == 2
    assert all(marker_b not in turn.user_text for turn in snapshot_a.turns)
    assert all(marker_a not in turn.user_text for turn in snapshot_b.turns)
    assert {turn.turn_id for turn in snapshot_a.turns}.isdisjoint(
        turn.turn_id for turn in snapshot_b.turns
    )
    assert harness.store.read_calls == {session_a: 2, session_b: 2}
    assert harness.store.append_calls == {session_a: 2, session_b: 2}
    assert not harness.store.compact_calls and not harness.summarizer.calls
