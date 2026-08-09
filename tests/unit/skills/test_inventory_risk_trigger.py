"""Offline contracts for the bounded inventory-risk mixed Skill."""

from __future__ import annotations

import asyncio

import pytest

from decision_agent.coordination.models import SkillResult, SkillStatus
from decision_agent.routing.models import RequestRoute, RouterDecision
from decision_agent.skills.contracts import SkillDefinition
from decision_agent.skills.inventory_risk_diagnosis import InventoryRiskDiagnosisSkill
from decision_agent.skills.inventory_risk_synthesizer import (
    InventoryRiskSynthesisInput,
    InventoryRiskSynthesisResult,
)


def mixed(
    data: str | None = "库存低于安全库存", knowledge: str | None = "补货制度"
) -> RouterDecision:
    return RouterDecision(
        route=RequestRoute.MIXED,
        normalized_query="库存风险诊断",
        decision_reason="需要数据和制度证据",
        data_subquery=data,
        knowledge_subquery=knowledge,
        missing_information=None,
        confidence=0.9,
    )


class FakeSubSkill:
    def __init__(self, *, route: RequestRoute, result: object, calls: list[str]) -> None:
        self.definition = SkillDefinition(
            name=(
                "enterprise-data-analysis"
                if route is RequestRoute.DATA
                else "enterprise-knowledge-qa"
            ),
            version="1.0.0",
            description="offline fake",
            supported_route=route,
            input_contract=("query",),
            allowed_tools=(
                "run_data_agent" if route is RequestRoute.DATA else "run_knowledge_agent",
            ),
            steps=("fake",),
            output_contract=("answer",),
            failure_codes=("fake_failed",),
        )
        self._result = result
        self.calls: list[tuple[str, RouterDecision]] = []
        self._calls = calls

    async def execute(self, *, user_query: str, decision: RouterDecision) -> SkillResult:
        self._calls.append(self.definition.name)
        self.calls.append((user_query, decision))
        if isinstance(self._result, BaseException):
            raise self._result
        return self._result  # type: ignore[return-value]


class FakeSynthesizer:
    async def synthesize(
        self, input_data: InventoryRiskSynthesisInput
    ) -> InventoryRiskSynthesisResult:
        return InventoryRiskSynthesisResult(
            risk_summary="库存存在风险。[D1]",
            policy_basis="制度要求及时补货。[E1]",
            recommended_actions=("安排补货。",),
            citations=("[D1]", "[E1]"),
        )


def completed(route: RequestRoute) -> SkillResult:
    tool = "run_data_agent" if route is RequestRoute.DATA else "run_knowledge_agent"
    citation = "[D1]" if route is RequestRoute.DATA else "[E1]"
    name = "enterprise-data-analysis" if route is RequestRoute.DATA else "enterprise-knowledge-qa"
    return SkillResult(
        status=SkillStatus.COMPLETED,
        skill_name=name,
        skill_version="1.0.0",
        route=route,
        answer="可信答案" + citation,
        citations=[citation],
        executed_steps=("fake",),
        selected_tool=tool,
    )


def failed(route: RequestRoute) -> SkillResult:
    return SkillResult(
        status=SkillStatus.FAILED,
        skill_name="enterprise-data-analysis"
        if route is RequestRoute.DATA
        else "enterprise-knowledge-qa",
        skill_version="1.0.0",
        route=route,
        executed_steps=("fake",),
        error_code="fake_failed",
    )


def unchecked_completed(base_route: RequestRoute, **updates: object) -> SkillResult:
    result = completed(base_route).model_dump()
    result.update(updates)
    return SkillResult.model_construct(**result)


def make_skill(
    *, data_result: object | None = None, knowledge_result: object | None = None
) -> tuple[InventoryRiskDiagnosisSkill, FakeSubSkill, FakeSubSkill, list[str]]:
    calls: list[str] = []
    data_skill = FakeSubSkill(
        route=RequestRoute.DATA,
        result=completed(RequestRoute.DATA) if data_result is None else data_result,
        calls=calls,
    )
    knowledge_skill = FakeSubSkill(
        route=RequestRoute.KNOWLEDGE,
        result=completed(RequestRoute.KNOWLEDGE) if knowledge_result is None else knowledge_result,
        calls=calls,
    )
    return (
        InventoryRiskDiagnosisSkill(
            data_skill=data_skill,
            knowledge_skill=knowledge_skill,
            synthesizer=FakeSynthesizer(),
        ),
        data_skill,
        knowledge_skill,
        calls,
    )


@pytest.mark.parametrize("query", ["库存不足如何补货", "inventory stockout replenishment"])
def test_inventory_topics_apply(query: str) -> None:
    skill, _, _, _ = make_skill()
    assert skill.is_applicable(query, mixed()) is True


@pytest.mark.parametrize("query", ["供应商绩效和采购制度", "hello"])
def test_irrelevant_mixed_does_not_apply(query: str) -> None:
    skill, _, _, _ = make_skill()
    assert skill.is_applicable(query, mixed("销售额", "采购制度")) is False


@pytest.mark.parametrize(
    ("data", "knowledge"),
    [(None, "补货"), ("库存", None), ("   ", "补货"), ("库存", "   ")],
)
def test_malformed_mixed_subqueries_do_not_apply(data: str | None, knowledge: str | None) -> None:
    malformed = RouterDecision.model_construct(
        route=RequestRoute.MIXED,
        normalized_query="q",
        decision_reason="r",
        data_subquery=data,
        knowledge_subquery=knowledge,
        missing_information=None,
        confidence=0.9,
    )
    skill, _, _, _ = make_skill()
    assert skill.is_applicable("库存", malformed) is False


@pytest.mark.parametrize(
    "route",
    [RequestRoute.KNOWLEDGE, RequestRoute.DATA, RequestRoute.UNSUPPORTED],
)
def test_non_mixed_decision_does_not_apply(route: RequestRoute) -> None:
    decision = RouterDecision(
        route=route,
        normalized_query="q",
        decision_reason="r",
        knowledge_subquery="库存制度" if route is RequestRoute.KNOWLEDGE else None,
        data_subquery="库存数据" if route is RequestRoute.DATA else None,
        missing_information="不支持" if route is RequestRoute.UNSUPPORTED else None,
        confidence=0.9,
    )
    skill, _, _, _ = make_skill()
    assert skill.is_applicable("库存风险", decision) is False


def test_dependencies_match_immutable_definition() -> None:
    skill, _, _, _ = make_skill()
    assert skill.definition.required_skills == (
        "enterprise-data-analysis",
        "enterprise-knowledge-qa",
    )
    assert skill.definition.allowed_tools == ("run_data_agent", "run_knowledge_agent")

    calls: list[str] = []
    invalid = FakeSubSkill(
        route=RequestRoute.DATA, result=completed(RequestRoute.DATA), calls=calls
    )
    invalid.definition = invalid.definition.model_copy(update={"name": "wrong"})
    with pytest.raises(ValueError, match="inventory_risk_dependency_invalid"):
        InventoryRiskDiagnosisSkill(
            data_skill=invalid,
            knowledge_skill=make_skill()[2],
            synthesizer=FakeSynthesizer(),
        )


@pytest.mark.asyncio
async def test_data_then_knowledge_subdecisions_are_isolated_and_ordered() -> None:
    skill, data_skill, knowledge_skill, calls = make_skill()
    original = mixed("当前库存风险", "补货制度要求")

    result = await skill.execute(user_query="库存风险与补货制度", decision=original)

    assert result.status is SkillStatus.COMPLETED
    assert result.error_code is None and result.selected_tool is None
    assert result.citations == ["[D1]", "[E1]"]
    assert result.executed_steps == (
        "enterprise-data-analysis",
        "run_data_agent",
        "enterprise-knowledge-qa",
        "run_knowledge_agent",
        "synthesize_inventory_risk",
    )
    assert "风险概览:" in result.answer
    assert calls == ["enterprise-data-analysis", "enterprise-knowledge-qa"]
    data_query, data_decision = data_skill.calls[0]
    knowledge_query, knowledge_decision = knowledge_skill.calls[0]
    assert data_query == original.data_subquery
    assert data_decision.route is RequestRoute.DATA
    assert data_decision.normalized_query == original.data_subquery
    assert data_decision.data_subquery == original.data_subquery
    assert data_decision.knowledge_subquery is None
    assert data_decision.missing_information is None
    assert knowledge_query == original.knowledge_subquery
    assert knowledge_decision.route is RequestRoute.KNOWLEDGE
    assert knowledge_decision.normalized_query == original.knowledge_subquery
    assert knowledge_decision.knowledge_subquery == original.knowledge_subquery
    assert knowledge_decision.data_subquery is None
    assert knowledge_decision.missing_information is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data_result",
    [
        failed(RequestRoute.DATA),
        unchecked_completed(RequestRoute.DATA, skill_name="wrong"),
        unchecked_completed(RequestRoute.DATA, route=RequestRoute.KNOWLEDGE),
        unchecked_completed(RequestRoute.DATA, selected_tool="run_knowledge_agent"),
        unchecked_completed(RequestRoute.DATA, answer="   "),
        unchecked_completed(RequestRoute.DATA, citations=[]),
        unchecked_completed(RequestRoute.DATA, citations=["[E1]"]),
        unchecked_completed(RequestRoute.DATA, citations=["D1"]),
    ],
)
async def test_invalid_data_result_short_circuits_knowledge(data_result: SkillResult) -> None:
    skill, data_skill, knowledge_skill, _ = make_skill(data_result=data_result)

    result = await skill.execute(user_query="库存风险", decision=mixed())

    expected = (
        "inventory_risk_data_skill_failed"
        if data_result.status is SkillStatus.FAILED
        else "inventory_risk_data_skill_invalid"
    )
    assert result.error_code == expected
    assert len(data_skill.calls) == 1
    assert knowledge_skill.calls == []
    assert result.answer is None and result.citations == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "knowledge_result",
    [
        failed(RequestRoute.KNOWLEDGE),
        unchecked_completed(RequestRoute.KNOWLEDGE, skill_name="wrong"),
        unchecked_completed(RequestRoute.KNOWLEDGE, route=RequestRoute.DATA),
        unchecked_completed(RequestRoute.KNOWLEDGE, selected_tool="run_data_agent"),
        unchecked_completed(RequestRoute.KNOWLEDGE, answer="   "),
        unchecked_completed(RequestRoute.KNOWLEDGE, citations=[]),
        unchecked_completed(RequestRoute.KNOWLEDGE, citations=["[D1]"]),
        unchecked_completed(RequestRoute.KNOWLEDGE, citations=["E1"]),
    ],
)
async def test_invalid_knowledge_result_fails_after_data(knowledge_result: SkillResult) -> None:
    skill, data_skill, knowledge_skill, _ = make_skill(knowledge_result=knowledge_result)

    result = await skill.execute(user_query="库存风险", decision=mixed())

    expected = (
        "inventory_risk_knowledge_skill_failed"
        if knowledge_result.status is SkillStatus.FAILED
        else "inventory_risk_knowledge_skill_invalid"
    )
    assert result.error_code == expected
    assert len(data_skill.calls) == 1 and len(knowledge_skill.calls) == 1
    assert result.answer is None and result.citations == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("data_result", "knowledge_result", "error_code"),
    [
        (RuntimeError("internal"), None, "inventory_risk_data_skill_failed"),
        (None, RuntimeError("internal"), "inventory_risk_knowledge_skill_failed"),
    ],
)
async def test_ordinary_subskill_exceptions_are_desensitized(
    data_result: object | None, knowledge_result: object | None, error_code: str
) -> None:
    skill, _, knowledge_skill, _ = make_skill(
        data_result=data_result, knowledge_result=knowledge_result
    )

    result = await skill.execute(user_query="库存风险", decision=mixed())

    assert result.error_code == error_code
    assert "internal" not in result.model_dump_json()
    if data_result is not None:
        assert knowledge_skill.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [asyncio.CancelledError(), KeyboardInterrupt(), SystemExit()])
async def test_control_flow_exceptions_are_not_swallowed(error: BaseException) -> None:
    skill, _, _, _ = make_skill(data_result=error)
    with pytest.raises(type(error)):
        await skill.execute(user_query="库存风险", decision=mixed())
