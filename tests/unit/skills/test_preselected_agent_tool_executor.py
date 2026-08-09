"""Deterministic coverage for controlled-Mixed preselected child Tools."""

from __future__ import annotations

import pytest

from decision_agent.routing.models import RequestRoute, RouterDecision
from decision_agent.skills.native_runtime import PreselectedAgentToolSkillExecutor
from decision_agent.tool_calling.models import AgentToolResult, NativeToolCallingStatus


class RecordingTool:
    def __init__(self, *, citation: str) -> None:
        self._citation = citation
        self.queries: list[str] = []

    async def run(self, *, query: str) -> AgentToolResult:
        self.queries.append(query)
        return AgentToolResult(
            status="succeeded",
            answer="bounded child answer",
            citations=[self._citation],
        )


def _decision(route: RequestRoute) -> RouterDecision:
    return RouterDecision(
        route=route,
        normalized_query="original",
        decision_reason="bounded_test",
        knowledge_subquery="approved knowledge query"
        if route in {RequestRoute.KNOWLEDGE, RequestRoute.MIXED}
        else None,
        data_subquery="approved data query"
        if route in {RequestRoute.DATA, RequestRoute.MIXED}
        else None,
        confidence=1,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route", "expected_tool", "expected_query", "expected_citation"),
    [
        (RequestRoute.DATA, "run_data_agent", "approved data query", "[D1]"),
        (
            RequestRoute.KNOWLEDGE,
            "run_knowledge_agent",
            "approved knowledge query",
            "[E1]",
        ),
    ],
)
async def test_preselected_executor_uses_only_router_owned_child_query(
    route: RequestRoute,
    expected_tool: str,
    expected_query: str,
    expected_citation: str,
) -> None:
    knowledge = RecordingTool(citation="[E1]")
    data = RecordingTool(citation="[D1]")
    executor = PreselectedAgentToolSkillExecutor(
        knowledge_tool=knowledge,
        data_tool=data,
    )

    result = await executor.execute(
        user_query="untrusted model query must not be used",
        decision=_decision(route),
    )

    assert result.status is NativeToolCallingStatus.COMPLETED
    assert result.selected_tool == expected_tool
    assert result.citations == [expected_citation]
    assert knowledge.queries + data.queries == [expected_query]


@pytest.mark.asyncio
async def test_preselected_executor_rejects_mixed_route_without_executing_a_tool() -> None:
    knowledge = RecordingTool(citation="[E1]")
    data = RecordingTool(citation="[D1]")
    executor = PreselectedAgentToolSkillExecutor(
        knowledge_tool=knowledge,
        data_tool=data,
    )

    result = await executor.execute(
        user_query="mixed request",
        decision=_decision(RequestRoute.MIXED),
    )

    assert result.status is NativeToolCallingStatus.FAILED
    assert result.error_code == "tool_calling_route_invalid"
    assert knowledge.queries == data.queries == []


@pytest.mark.asyncio
async def test_preselected_executor_accepts_only_empty_memory_projection() -> None:
    knowledge = RecordingTool(citation="[E1]")
    data = RecordingTool(citation="[D1]")
    executor = PreselectedAgentToolSkillExecutor(
        knowledge_tool=knowledge,
        data_tool=data,
    )
    decision = _decision(RequestRoute.DATA)

    result = await executor.execute_with_memory(
        user_query="untrusted model query",
        decision=decision,
        conversation_memory="",
    )
    rejected = await executor.execute_with_memory(
        user_query="untrusted model query",
        decision=decision,
        conversation_memory="untrusted prior turn",
    )

    assert result.status is NativeToolCallingStatus.COMPLETED
    assert data.queries == ["approved data query"]
    assert rejected.status is NativeToolCallingStatus.FAILED
    assert rejected.error_code == "tool_calling_memory_unsupported"
    assert data.queries == ["approved data query"]
