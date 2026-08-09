"""Minimal offline acceptance for formal runtime composition."""

from __future__ import annotations

import json
from typing import Any

import pytest

from decision_agent.application import (
    FormalMemoryConfiguration,
    FormalRequest,
    FormalRequestExecutor,
    MemoryContextStatus,
    MemoryPersistenceStatus,
    MemorySummarizationStatus,
    build_formal_request_executor,
)
from decision_agent.context.models import ContextItem, ContextKind
from decision_agent.coordination import build_default_coordinator
from decision_agent.coordination.models import CoordinatorStatus
from decision_agent.routing.models import RequestRoute, RouterDecision
from decision_agent.skills.native_runtime import NativeToolCallingSkillExecutor
from decision_agent.tool_calling.models import AgentToolResult

pytestmark = [pytest.mark.e2e, pytest.mark.offline_integration]


class _RecordingRouter:
    def __init__(self) -> None:
        self.memory_inputs: list[str | None] = []

    async def route_with_context(
        self, *, user_query: str, selected_items: tuple[ContextItem, ...]
    ) -> RouterDecision:
        memory = next(
            (
                item.content
                for item in selected_items
                if item.kind is ContextKind.CONVERSATION_MEMORY
            ),
            None,
        )
        self.memory_inputs.append(memory)
        return RouterDecision(
            route=RequestRoute.KNOWLEDGE,
            normalized_query=user_query,
            decision_reason="deterministic_runtime_composition_e2e",
            knowledge_subquery=user_query,
            data_subquery=None,
            missing_information=None,
            confidence=1,
        )


class _RecordingNativeModel:
    def __init__(self) -> None:
        self.initial_messages: list[tuple[dict[str, object], ...]] = []

    async def complete(
        self,
        *,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
        tool_choice: str,
        response_format: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        del tools, response_format
        if tool_choice == "required":
            self.initial_messages.append(tuple(messages))
            user_content = str(messages[1]["content"])
            query = user_content.split("Required subquery: ", 1)[1].split("\n\n", 1)[0]
            return {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "knowledge-call",
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
        payload = json.loads(str(tool_message["content"]))
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            {"answer": payload["answer"], "citations": payload["citations"]}
                        )
                    },
                }
            ]
        }


class _KnowledgeTool:
    async def run(self, *, query: str) -> AgentToolResult:
        return AgentToolResult(
            status="succeeded", answer=f"CURRENT::{query} [E1]", citations=["[E1]"]
        )


class _UnusedSynthesizer:
    async def synthesize(self, _: object) -> object:
        raise AssertionError("knowledge E2E must not invoke the mixed-route synthesizer")


def _coordinator(router: _RecordingRouter, model: _RecordingNativeModel):
    tool = _KnowledgeTool()
    return build_default_coordinator(
        router=router,  # type: ignore[arg-type]
        tool_calling_executor=NativeToolCallingSkillExecutor(
            model=model, knowledge_tool=tool, data_tool=tool
        ),
        inventory_risk_synthesizer=_UnusedSynthesizer(),  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_factory_runtime_executes_request_with_memory_disabled() -> None:
    router = _RecordingRouter()
    model = _RecordingNativeModel()
    executor = build_formal_request_executor(coordinator=_coordinator(router, model))

    response = await executor.execute(
        FormalRequest(request_id="runtime-disabled-1", user_query="CURRENT")
    )

    assert isinstance(executor, FormalRequestExecutor)
    assert response.result.status is CoordinatorStatus.COMPLETED
    assert response.result.answer == "CURRENT::CURRENT [E1]" and response.result.citations == [
        "[E1]"
    ]
    assert response.memory_context_status is MemoryContextStatus.NOT_REQUESTED
    assert response.memory_persistence_status is MemoryPersistenceStatus.NOT_REQUESTED
    assert response.memory_summarization_status is MemorySummarizationStatus.NOT_REQUESTED
    assert executor._memory_store is None and executor._rolling_summary_service is None  # type: ignore[attr-defined]
    assert router.memory_inputs == [None]


@pytest.mark.asyncio
async def test_factory_runtime_persists_and_projects_same_session_memory() -> None:
    router = _RecordingRouter()
    model = _RecordingNativeModel()
    executor = build_formal_request_executor(
        coordinator=_coordinator(router, model), memory=FormalMemoryConfiguration.in_memory()
    )
    first_marker = "FIRST_RUNTIME_TURN_MARKER [D1]"

    first = await executor.execute(
        FormalRequest(
            request_id="runtime-memory-1", user_query=first_marker, session_id="runtime-session"
        )
    )
    second = await executor.execute(
        FormalRequest(
            request_id="runtime-memory-2",
            user_query="SECOND_RUNTIME_TURN_MARKER",
            session_id="runtime-session",
        )
    )

    assert first.memory_persistence_status is MemoryPersistenceStatus.PERSISTED
    assert second.memory_context_status is MemoryContextStatus.PROJECTED
    assert second.memory_persistence_status is MemoryPersistenceStatus.PERSISTED
    assert second.memory_summarization_status is MemorySummarizationStatus.NOT_REQUESTED
    assert router.memory_inputs[0] is None
    assert (
        router.memory_inputs[1] is not None
        and "FIRST_RUNTIME_TURN_MARKER" in router.memory_inputs[1]
    )
    second_user = str(model.initial_messages[1][1]["content"])
    second_system = str(model.initial_messages[1][0]["content"])
    assert "FIRST_RUNTIME_TURN_MARKER" in second_user
    assert "FIRST_RUNTIME_TURN_MARKER" not in second_system
