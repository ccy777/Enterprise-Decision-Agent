"""Bounded executor for already validated M8B-1 plans."""

from __future__ import annotations

import asyncio
from typing import Any

from decision_agent.agent_workflow.models import (
    INVENTORY_RISK_SKILL,
    ExecutionBudget,
    ExecutionPlan,
    StepExecutionResult,
    StepExecutionStatus,
)
from decision_agent.coordination.models import SkillResult, SkillStatus
from decision_agent.coordination.skill_execution import execute_registered_skill
from decision_agent.observability import TraceContext, TraceSpanRecorder
from decision_agent.routing.models import RouterDecision
from decision_agent.security import (
    AuthorizationPolicy,
    SecurityAuthorizationError,
    SecurityContext,
    SecurityErrorCode,
)
from decision_agent.skills.registry import SkillRegistry, SkillRegistryError


class BoundedExecutor:
    """Execute exactly one validated allowlisted Skill call per execution round."""

    def __init__(self, *, registry: SkillRegistry) -> None:
        self._registry = registry

    async def execute(
        self,
        *,
        plan: ExecutionPlan,
        user_query: str,
        decision: RouterDecision,
        context_runtime: Any,
        user_item_id: str,
        memory_item_id: str | None,
        budget: ExecutionBudget,
        trace_recorder: TraceSpanRecorder | None,
        trace_parent_context: TraceContext | None,
        security_context: SecurityContext | None = None,
        authorization_policy: AuthorizationPolicy | None = None,
    ) -> StepExecutionResult:
        """Reserve capacity, then invoke the single allowlisted Skill once."""
        step = plan.steps[0]
        if not budget.reserve_execution_round():
            return _technical_failure(
                step.step_id, step.skill_name, budget, "workflow_round_budget_exhausted"
            )
        if not budget.reserve_skill_call():
            return _technical_failure(
                step.step_id, step.skill_name, budget, "workflow_skill_budget_exhausted"
            )
        if step.skill_name != INVENTORY_RISK_SKILL:
            return _technical_failure(
                step.step_id, step.skill_name, budget, "workflow_skill_not_allowed"
            )
        if authorization_policy is not None:
            try:
                if security_context is None:
                    raise SecurityAuthorizationError(SecurityErrorCode.UNAUTHENTICATED)
                authorization_policy.require_skill(security_context, step.skill_name)
            except SecurityAuthorizationError as exc:
                return _technical_failure(step.step_id, step.skill_name, budget, exc.code)
        try:
            skill = self._registry.get(step.skill_name)
        except SkillRegistryError:
            return _technical_failure(
                step.step_id, step.skill_name, budget, "workflow_skill_not_allowed"
            )
        if skill.definition.name != INVENTORY_RISK_SKILL or not skill.is_applicable(
            user_query, decision
        ):
            return _technical_failure(
                step.step_id, step.skill_name, budget, "workflow_skill_not_allowed"
            )
        try:
            result, memory_context_selected = await execute_registered_skill(
                skill=skill,
                user_query=user_query,
                decision=decision,
                context_runtime=context_runtime,
                user_item_id=user_item_id,
                memory_item_id=memory_item_id,
                trace_recorder=trace_recorder,
                trace_parent_context=trace_parent_context,
                execution_index=budget.skill_calls - 1,
                security_context=security_context,
                authorization_policy=authorization_policy,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            return _technical_failure(
                step.step_id, step.skill_name, budget, "workflow_skill_execution_failed"
            )
        if not isinstance(result, SkillResult):
            return _technical_failure(
                step.step_id, step.skill_name, budget, "workflow_skill_result_invalid"
            )
        if result.skill_name != skill.definition.name or result.route is not decision.route:
            return _technical_failure(
                step.step_id, step.skill_name, budget, "workflow_skill_result_invalid"
            )
        if result.status is SkillStatus.FAILED:
            return StepExecutionResult(
                step_id=step.step_id,
                skill_name=step.skill_name,
                status=StepExecutionStatus.BUSINESS_FAILED,
                skill_result=result,
                error_code=result.error_code,
                execution_round=budget.execution_rounds,
                memory_context_selected=memory_context_selected,
            )
        return StepExecutionResult(
            step_id=step.step_id,
            skill_name=step.skill_name,
            status=StepExecutionStatus.COMPLETED,
            skill_result=result,
            execution_round=budget.execution_rounds,
            memory_context_selected=memory_context_selected,
        )


def _technical_failure(
    step_id: str, skill_name: str, budget: ExecutionBudget, error_code: str
) -> StepExecutionResult:
    return StepExecutionResult(
        step_id=step_id,
        skill_name=skill_name,
        status=StepExecutionStatus.TECHNICAL_FAILED,
        error_code=error_code,
        execution_round=max(1, budget.execution_rounds),
    )
