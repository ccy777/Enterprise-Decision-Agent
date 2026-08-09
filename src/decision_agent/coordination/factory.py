"""Explicit composition root for the two default single-domain Skills."""

from __future__ import annotations

from collections.abc import Callable

from decision_agent.agent_workflow.workflow import ControlledAgentWorkflow
from decision_agent.coordination.coordinator import Coordinator
from decision_agent.routing.request_router import RequestRouter
from decision_agent.skills.contracts import NativeToolCallingExecutor
from decision_agent.skills.enterprise_data_analysis import EnterpriseDataAnalysisSkill
from decision_agent.skills.enterprise_knowledge_qa import EnterpriseKnowledgeQASkill
from decision_agent.skills.inventory_risk_diagnosis import InventoryRiskDiagnosisSkill
from decision_agent.skills.inventory_risk_synthesizer import InventoryRiskSynthesizer
from decision_agent.skills.registry import SkillRegistry


def build_default_coordinator(
    *,
    router: RequestRouter,
    tool_calling_executor: NativeToolCallingExecutor,
    inventory_risk_synthesizer: InventoryRiskSynthesizer,
    controlled_workflow_builder: Callable[[SkillRegistry], ControlledAgentWorkflow] | None = None,
    inventory_child_executor: NativeToolCallingExecutor | None = None,
) -> Coordinator:
    """Assemble trusted defaults from supplied dependencies only."""
    registry = SkillRegistry()
    knowledge_skill = EnterpriseKnowledgeQASkill(runtime=tool_calling_executor)
    data_skill = EnterpriseDataAnalysisSkill(runtime=tool_calling_executor)
    registry.register(knowledge_skill)
    registry.register(data_skill)
    mixed_child_runtime = inventory_child_executor or tool_calling_executor
    registry.register(
        InventoryRiskDiagnosisSkill(
            data_skill=EnterpriseDataAnalysisSkill(runtime=mixed_child_runtime),
            knowledge_skill=EnterpriseKnowledgeQASkill(runtime=mixed_child_runtime),
            synthesizer=inventory_risk_synthesizer,
        )
    )
    controlled_workflow = (
        None if controlled_workflow_builder is None else controlled_workflow_builder(registry)
    )
    return Coordinator(router=router, registry=registry, controlled_workflow=controlled_workflow)
