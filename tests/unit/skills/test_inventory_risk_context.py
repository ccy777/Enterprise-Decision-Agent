from __future__ import annotations

from datetime import UTC, datetime

import pytest

from decision_agent.context import ContextProjectionError, RequestContextRuntime
from decision_agent.context.runtime import MixedSynthesisContext
from decision_agent.coordination.models import SkillResult, SkillStatus
from decision_agent.routing.models import RequestRoute, RouterDecision
from decision_agent.skills.contracts import SkillDefinition
from decision_agent.skills.inventory_risk_diagnosis import InventoryRiskDiagnosisSkill
from decision_agent.skills.inventory_risk_synthesizer import (
    InventoryRiskSynthesisInput,
    InventoryRiskSynthesisResult,
)


def _decision() -> RouterDecision:
    return RouterDecision(
        route=RequestRoute.MIXED,
        normalized_query="inventory",
        decision_reason="test",
        knowledge_subquery="knowledge",
        data_subquery="data",
        missing_information=None,
        confidence=1,
    )


class _Child:
    def __init__(
        self, route: RequestRoute, answer: str, citation: str, *, failed: bool = False
    ) -> None:
        self.definition = SkillDefinition(
            name="enterprise-data-analysis"
            if route is RequestRoute.DATA
            else "enterprise-knowledge-qa",
            version="1",
            description="fake",
            supported_route=route,
            input_contract=("q",),
            allowed_tools=(
                "run_data_agent" if route is RequestRoute.DATA else "run_knowledge_agent",
            ),
            steps=("run",),
            output_contract=("answer",),
            failure_codes=("failed",),
        )
        self.answer = answer
        self.citation = citation
        self.failed = failed
        self.calls = 0

    async def execute(self, *, user_query: str, decision: RouterDecision) -> SkillResult:
        self.calls += 1
        if self.failed:
            return SkillResult(
                status=SkillStatus.FAILED,
                skill_name=self.definition.name,
                skill_version="1",
                route=decision.route,
                executed_steps=("run",),
                error_code="failed",
            )
        return SkillResult(
            status=SkillStatus.COMPLETED,
            skill_name=self.definition.name,
            skill_version="1",
            route=decision.route,
            answer=self.answer,
            citations=[self.citation],
            executed_steps=("run",),
            selected_tool=self.definition.allowed_tools[0],
        )


class _Synth:
    def __init__(self) -> None:
        self.inputs: list[InventoryRiskSynthesisInput] = []

    async def synthesize(
        self, input_data: InventoryRiskSynthesisInput
    ) -> InventoryRiskSynthesisResult:
        self.inputs.append(input_data)
        return InventoryRiskSynthesisResult(
            risk_summary="risk",
            policy_basis="policy",
            recommended_actions=("act",),
            citations=("[D1]", "[E1]"),
        )


def _skill(
    data: _Child | None = None, knowledge: _Child | None = None, synth: _Synth | None = None
) -> tuple[InventoryRiskDiagnosisSkill, _Synth]:
    synth = synth or _Synth()
    return InventoryRiskDiagnosisSkill(
        data_skill=data or _Child(RequestRoute.DATA, "ORIGINAL_DATA_ANSWER", "[D1]"),
        knowledge_skill=knowledge
        or _Child(RequestRoute.KNOWLEDGE, "ORIGINAL_KNOWLEDGE_ANSWER", "[E1]"),
        synthesizer=synth,
    ), synth


@pytest.mark.asyncio
async def test_mixed_normal_input_preserves_dual_domain_citations() -> None:
    skill, synth = _skill()
    result = await skill.execute(user_query="inventory ORIGINAL_MIXED_QUERY", decision=_decision())
    assert (
        result.citations == ["[D1]", "[E1]"]
        and synth.inputs[0].data_answer == "ORIGINAL_DATA_ANSWER"
    )


@pytest.mark.asyncio
async def test_mixed_data_failure_blocks_synthesizer() -> None:
    data = _Child(RequestRoute.DATA, "ignored", "[D1]", failed=True)
    skill, synth = _skill(data=data)
    result = await skill.execute(user_query="inventory", decision=_decision())
    assert result.status is SkillStatus.FAILED and synth.inputs == []


@pytest.mark.asyncio
async def test_mixed_knowledge_failure_blocks_synthesizer() -> None:
    knowledge = _Child(RequestRoute.KNOWLEDGE, "ignored", "[E1]", failed=True)
    skill, synth = _skill(knowledge=knowledge)
    result = await skill.execute(user_query="inventory", decision=_decision())
    assert result.status is SkillStatus.FAILED and synth.inputs == []


def test_synthesis_input_has_no_sql_schema_or_provider_fields() -> None:
    assert {"sql", "schema", "rows", "mcp_log", "provider_response"}.isdisjoint(
        InventoryRiskSynthesisInput.model_fields
    )


def test_synthesis_input_requires_separate_citation_domains() -> None:
    with pytest.raises(ValueError):
        InventoryRiskSynthesisInput(
            original_request="q",
            data_subquery="d",
            data_answer="a",
            data_citations=("[E1]",),
            knowledge_subquery="k",
            knowledge_answer="a",
            knowledge_citations=("[D1]",),
        )


@pytest.mark.asyncio
async def test_mixed_synthesis_uses_selected_projection_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill, synth = _skill()
    runtime = RequestContextRuntime(
        request_id="mixed", created_at=datetime(2026, 7, 24, tzinfo=UTC)
    )
    user_id = runtime.user_request("inventory ORIGINAL_MIXED_QUERY").item_id

    class Projection:
        def mixed(self, **_: object) -> MixedSynthesisContext:
            return MixedSynthesisContext(
                original_request="SELECTED_MIXED_QUERY",
                data_subquery="SELECTED_DATA_ANSWER",
                data_answer="SELECTED_DATA_ANSWER",
                data_citations=("[D1]",),
                knowledge_subquery="SELECTED_KNOWLEDGE_ANSWER",
                knowledge_answer="SELECTED_KNOWLEDGE_ANSWER",
                knowledge_citations=("[E1]",),
            )

    monkeypatch.setattr(RequestContextRuntime, "project", staticmethod(lambda _: Projection()))
    await skill.execute_with_context(
        user_query="inventory ORIGINAL_MIXED_QUERY",
        decision=_decision(),
        context_runtime=runtime,
        user_item_id=user_id,
    )

    captured = synth.inputs[0]
    assert captured.model_dump() == {
        "original_request": "SELECTED_MIXED_QUERY",
        "data_subquery": "SELECTED_DATA_ANSWER",
        "data_answer": "SELECTED_DATA_ANSWER",
        "data_citations": ("[D1]",),
        "knowledge_subquery": "SELECTED_KNOWLEDGE_ANSWER",
        "knowledge_answer": "SELECTED_KNOWLEDGE_ANSWER",
        "knowledge_citations": ("[E1]",),
    }


@pytest.mark.asyncio
async def test_mixed_projection_failure_blocks_synthesizer(monkeypatch: pytest.MonkeyPatch) -> None:
    skill, synth = _skill()
    runtime = RequestContextRuntime(
        request_id="mixed", created_at=datetime(2026, 7, 24, tzinfo=UTC)
    )
    user_id = runtime.user_request("inventory").item_id
    monkeypatch.setattr(
        RequestContextRuntime,
        "project",
        staticmethod(lambda _: (_ for _ in ()).throw(ContextProjectionError("safe"))),
    )

    with pytest.raises(ContextProjectionError):
        await skill.execute_with_context(
            user_query="inventory",
            decision=_decision(),
            context_runtime=runtime,
            user_item_id=user_id,
        )
    assert synth.inputs == []
