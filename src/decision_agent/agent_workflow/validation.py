"""Deterministic validation for untrusted structured execution plans."""

from __future__ import annotations

from dataclasses import dataclass

from decision_agent.agent_workflow.models import (
    INVENTORY_RISK_SKILL,
    PLAN_VERSION,
    ExecutionBudget,
    ExecutionPlan,
    PlanObjectiveType,
    RequiredOutputType,
)
from decision_agent.routing.models import RequestRoute
from decision_agent.skills.registry import SkillRegistry, SkillRegistryError


@dataclass(frozen=True, slots=True)
class PlanValidationResult:
    """Safe validation outcome that never carries provider output or exception text."""

    is_valid: bool
    error_code: str | None = None


class PlanValidator:
    """Validate only the M8B-1 single-step Mixed inventory-diagnosis plan."""

    def __init__(self, *, registry: SkillRegistry, budget: ExecutionBudget) -> None:
        self._registry = registry
        self._budget = budget

    def validate(self, plan: object) -> PlanValidationResult:
        if not isinstance(plan, ExecutionPlan):
            return _invalid("workflow_plan_invalid")
        if plan.plan_version != PLAN_VERSION:
            return _invalid("workflow_plan_version_invalid")
        if plan.objective_type is not PlanObjectiveType.MIXED_INVENTORY_DIAGNOSIS:
            return _invalid("workflow_plan_objective_invalid")
        if not plan.steps:
            return _invalid("workflow_plan_empty")
        step_ids = tuple(step.step_id for step in plan.steps)
        if len(set(step_ids)) != len(step_ids):
            return _invalid("workflow_plan_step_id_duplicate")
        if tuple(step.sequence for step in plan.steps) != tuple(range(1, len(plan.steps) + 1)):
            return _invalid("workflow_plan_sequence_invalid")
        known_steps = set(step_ids)
        for step in plan.steps:
            if step.objective_type is not plan.objective_type:
                return _invalid("workflow_plan_objective_invalid")
            if step.required_output_type is not RequiredOutputType.MIXED_DIAGNOSIS:
                return _invalid("workflow_plan_output_invalid")
            if any(dependency not in known_steps for dependency in step.depends_on):
                return _invalid("workflow_plan_dependency_invalid")
            if any(dependency == step.step_id for dependency in step.depends_on):
                return _invalid("workflow_plan_dependency_cycle")
            if any(
                _sequence_for_step(plan, dependency) >= step.sequence
                for dependency in step.depends_on
            ):
                return _invalid("workflow_plan_dependency_invalid")
        if len(plan.steps) > self._budget.max_plan_steps:
            return _invalid("workflow_plan_step_count_invalid")
        if len(plan.steps) != 1:
            return _invalid("workflow_plan_step_count_invalid")
        if (
            plan.max_execution_rounds > self._budget.max_execution_rounds
            or plan.max_skill_calls > self._budget.max_skill_calls
        ):
            return _invalid("workflow_plan_budget_invalid")
        step = plan.steps[0]
        if step.step_id != step.step_id.strip():
            return _invalid("workflow_plan_step_id_invalid")
        if step.skill_name != INVENTORY_RISK_SKILL:
            return _invalid("workflow_plan_skill_not_allowed")
        if step.optional or step.depends_on:
            return _invalid("workflow_plan_dependency_invalid")
        try:
            skill = self._registry.get(step.skill_name)
        except SkillRegistryError:
            return _invalid("workflow_plan_skill_not_allowed")
        if skill.definition.supported_route is not RequestRoute.MIXED:
            return _invalid("workflow_plan_skill_not_allowed")
        return PlanValidationResult(is_valid=True)


def _invalid(error_code: str) -> PlanValidationResult:
    return PlanValidationResult(is_valid=False, error_code=error_code)


def _sequence_for_step(plan: ExecutionPlan, step_id: str) -> int:
    return next(step.sequence for step in plan.steps if step.step_id == step_id)
