"""One opt-in real-provider smoke for constrained inventory-risk synthesis."""

from __future__ import annotations

import os

import pytest

from decision_agent.config import Settings
from decision_agent.coordination.models import SkillResult, SkillStatus
from decision_agent.routing.models import RequestRoute, RouterDecision
from decision_agent.skills.contracts import SkillDefinition
from decision_agent.skills.inventory_risk_diagnosis import InventoryRiskDiagnosisSkill
from decision_agent.skills.inventory_risk_synthesizer import (
    InventoryRiskSynthesisInput,
    InventoryRiskSynthesisResult,
    InventoryRiskSynthesizerError,
    OpenAICompatibleInventoryRiskSynthesizer,
)

pytestmark = pytest.mark.integration


class DeterministicSubSkill:
    def __init__(self, route: RequestRoute) -> None:
        self.definition = SkillDefinition(
            name=(
                "enterprise-data-analysis"
                if route is RequestRoute.DATA
                else "enterprise-knowledge-qa"
            ),
            version="1.0.0",
            description="real-smoke deterministic fake",
            supported_route=route,
            input_contract=("query",),
            allowed_tools=(
                "run_data_agent" if route is RequestRoute.DATA else "run_knowledge_agent",
            ),
            steps=("fake",),
            output_contract=("answer",),
            failure_codes=("fake_failed",),
        )
        self.calls = 0

    async def execute(self, *, user_query: str, decision: RouterDecision) -> SkillResult:
        self.calls += 1
        citation = "[D1]" if decision.route is RequestRoute.DATA else "[E1]"
        tool = "run_data_agent" if decision.route is RequestRoute.DATA else "run_knowledge_agent"
        answer = (
            "产品 A 当前库存低于安全库存,需要关注补货风险。[D1]"
            if decision.route is RequestRoute.DATA
            else "公司补货制度要求库存低于安全库存时及时安排补货。[E1]"
        )
        return SkillResult(
            status=SkillStatus.COMPLETED,
            skill_name=self.definition.name,
            skill_version=self.definition.version,
            route=decision.route,
            answer=answer,
            citations=[citation],
            executed_steps=("fake",),
            selected_tool=tool,
        )


class RecordingSynthesizer:
    def __init__(self, delegate: OpenAICompatibleInventoryRiskSynthesizer) -> None:
        self._delegate = delegate
        self.calls = 0
        self.http_status: int | None = None
        self.failure_stage: str | None = None
        self.finish_reason: str | None = None
        self.schema_error_fields: tuple[str, ...] = ()

    async def synthesize(
        self, input_data: InventoryRiskSynthesisInput
    ) -> InventoryRiskSynthesisResult:
        self.calls += 1
        try:
            return await self._delegate.synthesize(input_data)
        except InventoryRiskSynthesizerError as exc:
            self.http_status = exc.http_status
            self.failure_stage = exc.failure_stage
            self.finish_reason = exc.finish_reason
            self.schema_error_fields = exc.schema_error_fields
            raise


@pytest.mark.asyncio
async def test_real_provider_synthesizes_fake_inventory_evidence_once() -> None:
    if os.getenv("RUN_INVENTORY_RISK_SYNTHESIZER_REAL_LLM_SMOKE") != "1":
        pytest.skip("set RUN_INVENTORY_RISK_SYNTHESIZER_REAL_LLM_SMOKE=1 to run this smoke")
    settings = Settings()
    if (
        settings.llm_api_key is None
        or settings.llm_base_url is None
        or settings.llm_model_name is None
    ):
        pytest.skip("LLM settings are not fully configured")

    data_skill = DeterministicSubSkill(RequestRoute.DATA)
    knowledge_skill = DeterministicSubSkill(RequestRoute.KNOWLEDGE)
    synthesizer = RecordingSynthesizer(
        OpenAICompatibleInventoryRiskSynthesizer.from_settings(settings)
    )
    skill = InventoryRiskDiagnosisSkill(
        data_skill=data_skill,
        knowledge_skill=knowledge_skill,
        synthesizer=synthesizer,
    )
    decision = RouterDecision(
        route=RequestRoute.MIXED,
        normalized_query="产品 A 库存不足,并结合补货制度给出建议",
        decision_reason="需要库存数据和补货制度证据",
        data_subquery="产品 A 当前库存是否低于安全库存?",
        knowledge_subquery="库存低于安全库存时公司的补货制度要求是什么?",
        missing_information=None,
        confidence=0.9,
    )

    result = await skill.execute(user_query=decision.normalized_query, decision=decision)

    if result.status is not SkillStatus.COMPLETED:
        pytest.fail(
            "real synthesis smoke failed safely: "
            f"error_code={result.error_code}; http_status={synthesizer.http_status}; "
            f"failure_stage={synthesizer.failure_stage}; "
            f"finish_reason={synthesizer.finish_reason}; "
            f"schema_error_fields={synthesizer.schema_error_fields}"
        )
    assert data_skill.calls == 1 and knowledge_skill.calls == 1 and synthesizer.calls == 1
    assert result.selected_tool is None
    assert result.citations == ["[D1]", "[E1]"]
    assert result.answer is not None and "风险概览:" in result.answer
