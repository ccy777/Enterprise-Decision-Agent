"""Offline synthesis contracts for inventory-risk diagnosis."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from decision_agent.coordination.models import SkillResult, SkillStatus
from decision_agent.routing.models import RequestRoute, RouterDecision
from decision_agent.skills.contracts import SkillDefinition
from decision_agent.skills.inventory_risk_diagnosis import InventoryRiskDiagnosisSkill
from decision_agent.skills.inventory_risk_synthesizer import (
    InventoryRiskSynthesisInput,
    InventoryRiskSynthesisResult,
)


def mixed() -> RouterDecision:
    return RouterDecision(
        route=RequestRoute.MIXED,
        normalized_query="库存风险与补货制度",
        decision_reason="需要数据和制度证据",
        data_subquery="哪些产品库存不足",
        knowledge_subquery="公司补货制度是什么",
        missing_information=None,
        confidence=0.9,
    )


def completed(route: RequestRoute, citations: list[str]) -> SkillResult:
    return SkillResult(
        status=SkillStatus.COMPLETED,
        skill_name=(
            "enterprise-data-analysis" if route is RequestRoute.DATA else "enterprise-knowledge-qa"
        ),
        skill_version="1.0.0",
        route=route,
        answer="可信子结论",
        citations=citations,
        executed_steps=("fake",),
        selected_tool=("run_data_agent" if route is RequestRoute.DATA else "run_knowledge_agent"),
    )


def failed(route: RequestRoute) -> SkillResult:
    return SkillResult(
        status=SkillStatus.FAILED,
        skill_name=(
            "enterprise-data-analysis" if route is RequestRoute.DATA else "enterprise-knowledge-qa"
        ),
        skill_version="1.0.0",
        route=route,
        executed_steps=("fake",),
        error_code="fake_failed",
    )


class FakeSubSkill:
    def __init__(self, route: RequestRoute, result: SkillResult) -> None:
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
        self.result = result

    async def execute(self, *, user_query: str, decision: RouterDecision) -> SkillResult:
        return self.result


class FakeSynthesizer:
    def __init__(self, result: object) -> None:
        self.result = result
        self.inputs: list[InventoryRiskSynthesisInput] = []

    async def synthesize(
        self, input_data: InventoryRiskSynthesisInput
    ) -> InventoryRiskSynthesisResult:
        self.inputs.append(input_data)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result  # type: ignore[return-value]


def synthesis(citations: tuple[str, ...] = ("[D1]", "[E1]")) -> InventoryRiskSynthesisResult:
    return InventoryRiskSynthesisResult(
        risk_summary="库存低于安全库存。",
        policy_basis="制度要求及时补货。",
        recommended_actions=("安排补货。", "跟踪库存。"),
        citations=citations,
    )


def make_skill(
    synthesis_result: object,
    *,
    data_citations: list[str] | None = None,
    knowledge_citations: list[str] | None = None,
) -> tuple[InventoryRiskDiagnosisSkill, FakeSynthesizer]:
    data = FakeSubSkill(RequestRoute.DATA, completed(RequestRoute.DATA, data_citations or ["[D1]"]))
    knowledge = FakeSubSkill(
        RequestRoute.KNOWLEDGE,
        completed(RequestRoute.KNOWLEDGE, knowledge_citations or ["[E1]"]),
    )
    synthesizer = FakeSynthesizer(synthesis_result)
    return (
        InventoryRiskDiagnosisSkill(
            data_skill=data,
            knowledge_skill=knowledge,
            synthesizer=synthesizer,
        ),
        synthesizer,
    )


def test_synthesis_models_are_strict_immutable_and_evidence_bounded() -> None:
    input_data = InventoryRiskSynthesisInput(
        original_request="库存风险",
        data_subquery="库存查询",
        data_answer="数据结论",
        data_citations=("[D1]",),
        knowledge_subquery="补货制度",
        knowledge_answer="制度结论",
        knowledge_citations=("[E1]",),
    )
    assert input_data.data_citations == ("[D1]",)
    with pytest.raises(ValidationError):
        InventoryRiskSynthesisInput.model_validate({**input_data.model_dump(), "sql": "SELECT 1"})
    with pytest.raises(ValidationError):
        InventoryRiskSynthesisInput.model_validate(
            {**input_data.model_dump(), "data_citations": ("[E1]",)}
        )
    with pytest.raises(ValidationError):
        input_data.original_request = "changed"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        input_data.data_citations = ("[D2]",)  # type: ignore[misc]
    output_data = synthesis()
    with pytest.raises(ValidationError):
        output_data.risk_summary = "changed"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        InventoryRiskSynthesisResult.model_validate(
            {**output_data.model_dump(), "traceback": "not allowed"}
        )


@pytest.mark.asyncio
async def test_synthesizer_receives_only_trusted_public_input_once() -> None:
    skill, synthesizer = make_skill(synthesis())

    result = await skill.execute(user_query="库存风险与补货制度", decision=mixed())

    assert result.status is SkillStatus.COMPLETED
    assert len(synthesizer.inputs) == 1
    input_data = synthesizer.inputs[0]
    assert input_data.original_request == "库存风险与补货制度"
    assert input_data.data_subquery == mixed().data_subquery
    assert input_data.knowledge_subquery == mixed().knowledge_subquery
    assert set(input_data.model_dump()) == {
        "original_request",
        "data_subquery",
        "data_answer",
        "data_citations",
        "knowledge_subquery",
        "knowledge_answer",
        "knowledge_citations",
    }


@pytest.mark.asyncio
async def test_valid_synthesis_is_rendered_with_stable_deduplicated_citations() -> None:
    skill, _ = make_skill(
        synthesis(("[E1]", "[D1]", "[D1]")),
        data_citations=["[D2]", "[D1]"],
        knowledge_citations=["[E2]", "[E1]"],
    )

    result = await skill.execute(user_query="库存风险与补货制度", decision=mixed())

    assert result.status is SkillStatus.COMPLETED
    assert result.selected_tool is None
    assert result.citations == ["[D1]", "[E1]"]
    assert result.answer == (
        "风险概览:\n库存低于安全库存。\n\n"
        "制度依据:\n制度要求及时补货。\n\n"
        "建议措施:\n1. 安排补货。\n2. 跟踪库存。\n\n"
        "引用:\n[D1] [E1]"
    )
    assert result.executed_steps == (
        "enterprise-data-analysis",
        "run_data_agent",
        "enterprise-knowledge-qa",
        "run_knowledge_agent",
        "synthesize_inventory_risk",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_result",
    [
        object(),
        InventoryRiskSynthesisResult.model_construct(
            risk_summary=" ",
            policy_basis="政策",
            recommended_actions=("行动",),
            citations=("[D1]", "[E1]"),
        ),
        InventoryRiskSynthesisResult.model_construct(
            risk_summary="风险",
            policy_basis=" ",
            recommended_actions=("行动",),
            citations=("[D1]", "[E1]"),
        ),
        InventoryRiskSynthesisResult.model_construct(
            risk_summary="风险",
            policy_basis="政策",
            recommended_actions=(" ",),
            citations=("[D1]", "[E1]"),
        ),
    ],
)
async def test_invalid_synthesis_content_fails_closed(invalid_result: object) -> None:
    skill, synthesizer = make_skill(invalid_result)

    result = await skill.execute(user_query="库存风险", decision=mixed())

    assert result.status is SkillStatus.FAILED
    assert result.error_code == "inventory_risk_synthesis_invalid"
    assert result.answer is None and result.citations == []
    assert len(synthesizer.inputs) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "citations",
    [
        ("[D1]",),
        ("[E1]",),
        ("[D99]", "[E1]"),
        ("[D1]", "[E99]"),
        ("D1", "[E1]"),
    ],
)
async def test_invalid_synthesis_citations_fail_closed(citations: tuple[str, ...]) -> None:
    result_data = InventoryRiskSynthesisResult.model_construct(
        risk_summary="风险",
        policy_basis="政策",
        recommended_actions=("行动",),
        citations=citations,
    )
    skill, _ = make_skill(result_data)

    result = await skill.execute(user_query="库存风险", decision=mixed())

    assert result.status is SkillStatus.FAILED
    assert result.error_code == "inventory_risk_synthesis_citations_invalid"
    assert result.answer is None and result.citations == []


@pytest.mark.asyncio
async def test_synthesizer_exception_is_desensitized() -> None:
    skill, _ = make_skill(RuntimeError("internal failure"))

    result = await skill.execute(user_query="库存风险", decision=mixed())

    assert result.error_code == "inventory_risk_synthesizer_failed"
    assert "internal failure" not in result.model_dump_json()


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_route", [RequestRoute.DATA, RequestRoute.KNOWLEDGE])
async def test_subskill_failure_does_not_call_synthesizer(failed_route: RequestRoute) -> None:
    data = FakeSubSkill(
        RequestRoute.DATA,
        failed(RequestRoute.DATA)
        if failed_route is RequestRoute.DATA
        else completed(RequestRoute.DATA, ["[D1]"]),
    )
    knowledge = FakeSubSkill(
        RequestRoute.KNOWLEDGE,
        failed(RequestRoute.KNOWLEDGE)
        if failed_route is RequestRoute.KNOWLEDGE
        else completed(RequestRoute.KNOWLEDGE, ["[E1]"]),
    )
    synthesizer = FakeSynthesizer(synthesis())
    skill = InventoryRiskDiagnosisSkill(
        data_skill=data,
        knowledge_skill=knowledge,
        synthesizer=synthesizer,
    )

    result = await skill.execute(user_query="库存风险", decision=mixed())

    assert result.error_code == (
        "inventory_risk_data_skill_failed"
        if failed_route is RequestRoute.DATA
        else "inventory_risk_knowledge_skill_failed"
    )
    assert synthesizer.inputs == []


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [asyncio.CancelledError(), KeyboardInterrupt(), SystemExit()])
async def test_synthesizer_control_flow_exceptions_are_not_swallowed(error: BaseException) -> None:
    skill, _ = make_skill(error)
    with pytest.raises(type(error)):
        await skill.execute(user_query="库存风险", decision=mixed())
