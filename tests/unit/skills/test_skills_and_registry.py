"""Offline contracts for executable single-domain Skills."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from decision_agent.coordination.models import SkillResult, SkillStatus
from decision_agent.routing.models import RequestRoute, RouterDecision
from decision_agent.skills.contracts import SkillDefinition
from decision_agent.skills.enterprise_data_analysis import EnterpriseDataAnalysisSkill
from decision_agent.skills.enterprise_knowledge_qa import EnterpriseKnowledgeQASkill
from decision_agent.skills.registry import SkillRegistry, SkillRegistryError
from decision_agent.tool_calling.models import NativeToolCallingStatus, ToolCallingResult


def decision(route: RequestRoute) -> RouterDecision:
    return RouterDecision(
        route=route,
        normalized_query="query",
        decision_reason="reason",
        knowledge_subquery="knowledge" if route is RequestRoute.KNOWLEDGE else None,
        data_subquery="data" if route is RequestRoute.DATA else None,
        missing_information=None,
        confidence=0.9,
    )


class FakeRuntime:
    def __init__(self, result: ToolCallingResult) -> None:
        self.result = result
        self.calls: list[RouterDecision] = []

    async def execute(self, *, user_query: str, decision: RouterDecision) -> ToolCallingResult:
        self.calls.append(decision)
        return self.result


def completed(route: RequestRoute, tool: str, citation: str) -> ToolCallingResult:
    return ToolCallingResult(
        status=NativeToolCallingStatus.COMPLETED,
        route=route,
        selected_tool=tool,
        tool_call_id="call_1",
        answer="answer" + citation,
        citations=[citation],
        steps=2,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route", "skill_type", "tool", "citation"),
    [
        (RequestRoute.KNOWLEDGE, EnterpriseKnowledgeQASkill, "run_knowledge_agent", "[E1]"),
        (RequestRoute.DATA, EnterpriseDataAnalysisSkill, "run_data_agent", "[D1]"),
    ],
)
async def test_skill_delegates_to_runtime_and_preserves_route_evidence(
    route, skill_type, tool, citation
) -> None:
    runtime = FakeRuntime(completed(route, tool, citation))
    skill = skill_type(runtime=runtime)
    result = await skill.execute(user_query="query", decision=decision(route))
    assert result.status is SkillStatus.COMPLETED
    assert result.selected_tool == tool and result.citations == [citation]
    assert runtime.calls == [decision(route)] and skill.definition.allowed_tools == (tool,)


@pytest.mark.asyncio
async def test_skill_rejects_wrong_route_or_selected_tool_without_exposing_answer() -> None:
    knowledge = EnterpriseKnowledgeQASkill(
        runtime=FakeRuntime(completed(RequestRoute.KNOWLEDGE, "run_data_agent", "[E1]"))
    )
    wrong_tool = await knowledge.execute(
        user_query="query", decision=decision(RequestRoute.KNOWLEDGE)
    )
    wrong_route = await knowledge.execute(user_query="query", decision=decision(RequestRoute.DATA))
    assert wrong_tool.error_code == "skill_tool_not_allowed" and wrong_tool.answer is None
    assert wrong_route.error_code == "skill_route_invalid"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("skill_type", "decision_route", "runtime_route", "tool", "citation"),
    [
        (
            EnterpriseKnowledgeQASkill,
            RequestRoute.KNOWLEDGE,
            RequestRoute.DATA,
            "run_knowledge_agent",
            "[E1]",
        ),
        (
            EnterpriseDataAnalysisSkill,
            RequestRoute.DATA,
            RequestRoute.KNOWLEDGE,
            "run_data_agent",
            "[D1]",
        ),
    ],
)
async def test_skill_rejects_runtime_route_mismatch(
    skill_type, decision_route, runtime_route, tool, citation
) -> None:
    runtime = FakeRuntime(completed(runtime_route, tool, citation))
    result = await skill_type(runtime=runtime).execute(
        user_query="query", decision=decision(decision_route)
    )
    assert result.error_code == "skill_runtime_route_mismatch"
    assert result.answer is None and result.citations == [] and result.selected_tool is None
    assert len(runtime.calls) == 1


def test_registry_rejects_duplicate_name_and_route_and_returns_snapshots() -> None:
    runtime = FakeRuntime(completed(RequestRoute.KNOWLEDGE, "run_knowledge_agent", "[E1]"))
    registry = SkillRegistry()
    first = EnterpriseKnowledgeQASkill(runtime=runtime)
    registry.register(first)
    assert registry.get(first.definition.name) is first and registry.names == (
        first.definition.name,
    )
    with pytest.raises(SkillRegistryError, match="skill_name_already_registered"):
        registry.register(first)

    class SameRouteDifferentName:
        definition = SkillDefinition(
            name="different-knowledge-skill",
            version="1",
            description="test",
            supported_route=RequestRoute.KNOWLEDGE,
            input_contract=("query",),
            allowed_tools=("run_knowledge_agent",),
            steps=("run",),
            output_contract=("answer",),
            failure_codes=("failed",),
        )

        async def execute(self, *, user_query: str, decision: RouterDecision) -> SkillResult:
            raise AssertionError("registry test never executes this Skill")

    with pytest.raises(SkillRegistryError, match="skill_route_already_registered"):
        registry.register(SameRouteDifferentName())
    with pytest.raises(SkillRegistryError, match="skill_not_registered"):
        registry.get("untrusted")


def test_skill_result_rejects_cross_domain_or_failed_output() -> None:
    with pytest.raises(ValidationError):
        SkillResult(
            status=SkillStatus.COMPLETED,
            skill_name="x",
            skill_version="1",
            route=RequestRoute.KNOWLEDGE,
            answer="answer",
            citations=["[D1]"],
            executed_steps=("x",),
            selected_tool="run_knowledge_agent",
        )
    with pytest.raises(ValidationError):
        SkillResult(
            status=SkillStatus.FAILED,
            skill_name="x",
            skill_version="1",
            route=RequestRoute.DATA,
            answer="leak",
            executed_steps=("x",),
            error_code="failed",
        )


@pytest.mark.asyncio
async def test_cancelled_error_is_not_swallowed() -> None:
    class CancelledRuntime:
        async def execute(self, *, user_query: str, decision: RouterDecision) -> ToolCallingResult:
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await EnterpriseKnowledgeQASkill(runtime=CancelledRuntime()).execute(
            user_query="query", decision=decision(RequestRoute.KNOWLEDGE)
        )
