"""Deterministic Planner--Executor--Reviewer state machine for M8B-1."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from decision_agent.agent_workflow.contracts import Planner, WorkflowReviewer
from decision_agent.agent_workflow.executor import BoundedExecutor
from decision_agent.agent_workflow.models import (
    INVENTORY_RISK_SKILL,
    ControlledWorkflowResult,
    ExecutionBudget,
    PlanningRequest,
    PlanObjectiveType,
    ReviewerDecision,
    ReviewerFinalStatus,
    ReviewerOutcome,
    ReviewStepSummary,
    StepExecutionResult,
    StepExecutionStatus,
    WorkflowReviewRequest,
    WorkflowStatus,
)
from decision_agent.agent_workflow.providers import WorkflowProviderError
from decision_agent.agent_workflow.validation import PlanValidator
from decision_agent.observability import (
    SpanStatus,
    TraceContext,
    TraceSpanRecorder,
    TraceStage,
    complete_recorded_span,
    start_recorded_span,
)
from decision_agent.routing.models import RequestRoute, RouterDecision
from decision_agent.security import AuthorizationPolicy, SecurityContext
from decision_agent.skills.contracts import ExecutableSkill
from decision_agent.skills.registry import SkillRegistry


@dataclass(frozen=True, slots=True)
class ControlledWorkflowPolicy:
    """Explicit internal switch; the production default remains disabled."""

    enabled: bool = False
    repairable_skill_names: tuple[str, ...] = ()


class ControlledAgentWorkflow:
    """Run a single approved Mixed Skill through a bounded deterministic loop."""

    def __init__(
        self,
        *,
        planner: Planner,
        reviewer: WorkflowReviewer,
        registry: SkillRegistry,
        policy: ControlledWorkflowPolicy | None = None,
    ) -> None:
        self._planner = planner
        self._reviewer = reviewer
        self._registry = registry
        self._policy = ControlledWorkflowPolicy() if policy is None else policy
        self._executor = BoundedExecutor(registry=registry)

    def is_enabled_for(self, *, decision: RouterDecision, skill: ExecutableSkill) -> bool:
        """Allow only the explicitly configured Mixed inventory-diagnosis path."""
        return (
            self._policy.enabled
            and decision.route is RequestRoute.MIXED
            and skill.definition.name == INVENTORY_RISK_SKILL
            and skill.definition.supported_route is RequestRoute.MIXED
        )

    async def execute(
        self,
        *,
        user_query: str,
        decision: RouterDecision,
        context_runtime: Any,
        user_item_id: str,
        memory_item_id: str | None,
        trace_recorder: TraceSpanRecorder | None,
        trace_parent_context: TraceContext | None,
        security_context: SecurityContext | None = None,
        authorization_policy: AuthorizationPolicy | None = None,
    ) -> ControlledWorkflowResult:
        """Own the optional M8B workflow span without changing business call counts."""
        span = start_recorded_span(
            trace_recorder,
            stage=TraceStage.AGENT_WORKFLOW,
            component="agent_workflow",
            operation="execute_controlled_workflow",
            parent_context=trace_parent_context,
        )
        try:
            result = await self._execute_core(
                user_query=user_query,
                decision=decision,
                context_runtime=context_runtime,
                user_item_id=user_item_id,
                memory_item_id=memory_item_id,
                trace_recorder=trace_recorder,
                trace_parent_context=span,
                security_context=security_context,
                authorization_policy=authorization_policy,
            )
        except asyncio.CancelledError:
            complete_recorded_span(trace_recorder, span, status=SpanStatus.CANCELLED)
            raise
        complete_recorded_span(
            trace_recorder,
            span,
            status=SpanStatus.COMPLETED
            if result.status is not WorkflowStatus.FAILED
            else SpanStatus.FAILED,
            error_code=result.error_code if result.status is WorkflowStatus.FAILED else None,
            attributes={
                "plan_version": "m8b-v1",
                "plan_step_count": len(result.steps),
                "repair_allowed": False,
                "success": result.status is WorkflowStatus.ACCEPTED,
                "result_status": result.status.value,
            },
        )
        return result

    async def _execute_core(
        self,
        *,
        user_query: str,
        decision: RouterDecision,
        context_runtime: Any,
        user_item_id: str,
        memory_item_id: str | None,
        trace_recorder: TraceSpanRecorder | None,
        trace_parent_context: TraceContext | None,
        security_context: SecurityContext | None,
        authorization_policy: AuthorizationPolicy | None,
    ) -> ControlledWorkflowResult:
        """Run Planner -> validate -> execute -> review with at most one repair."""
        budget = ExecutionBudget()
        planning_span = start_recorded_span(
            trace_recorder,
            stage=TraceStage.PLANNING,
            component="agent_workflow",
            operation="validate_workflow_plan",
            parent_context=trace_parent_context,
        )
        try:
            plan = await _plan_with_trace(
                self._planner,
                request=PlanningRequest(
                    route=decision.route,
                    user_request=user_query,
                    objective_type=PlanObjectiveType.MIXED_INVENTORY_DIAGNOSIS,
                    allowed_skill_names=(INVENTORY_RISK_SKILL,),
                ),
                trace_recorder=trace_recorder,
                trace_parent_context=planning_span,
            )
        except asyncio.CancelledError:
            complete_recorded_span(trace_recorder, planning_span, status=SpanStatus.CANCELLED)
            raise
        except WorkflowProviderError as exc:
            complete_recorded_span(
                trace_recorder,
                planning_span,
                status=SpanStatus.FAILED,
                error_code=exc.code,
            )
            return _failed(exc.code)
        except Exception:
            complete_recorded_span(
                trace_recorder,
                planning_span,
                status=SpanStatus.FAILED,
                error_code="workflow_planner_failed",
            )
            return _failed("workflow_planner_failed")
        validation = PlanValidator(registry=self._registry, budget=budget).validate(plan)
        if not validation.is_valid:
            complete_recorded_span(
                trace_recorder,
                planning_span,
                status=SpanStatus.FAILED,
                error_code=validation.error_code,
            )
            return _failed(validation.error_code or "workflow_plan_invalid")
        complete_recorded_span(
            trace_recorder,
            planning_span,
            status=SpanStatus.COMPLETED,
            attributes={
                "plan_version": plan.plan_version,
                "plan_step_count": len(plan.steps),
                "success": True,
            },
        )

        first = await self._execute_step(
            plan=plan,
            user_query=user_query,
            decision=decision,
            context_runtime=context_runtime,
            user_item_id=user_item_id,
            memory_item_id=memory_item_id,
            budget=budget,
            trace_recorder=trace_recorder,
            trace_parent_context=trace_parent_context,
            security_context=security_context,
            authorization_policy=authorization_policy,
        )
        if first.status is StepExecutionStatus.TECHNICAL_FAILED:
            return _failed(first.error_code or "workflow_skill_execution_failed", (first,))
        review = await self._review(
            plan_id=plan.plan_id,
            user_request=user_query,
            objective_type=plan.objective_type,
            steps=(first,),
            budget=budget,
            trace_recorder=trace_recorder,
            trace_parent_context=trace_parent_context,
        )
        if isinstance(review, ControlledWorkflowResult):
            return _with_memory(review, (first,))
        accepted = _accept_result(review, (first,))
        if accepted is not None:
            return _accepted(accepted, (first,))
        if _is_unanswerable(review):
            return _unanswerable((first,))
        if review.outcome is ReviewerOutcome.FAIL_CLOSED:
            return _failed("workflow_reviewer_fail_closed", (first,))
        if review.outcome is not ReviewerOutcome.REPAIR:
            return _failed("workflow_reviewer_invalid", (first,))
        if not self._repair_is_allowed(review, first, budget):
            return _failed("workflow_repair_not_permitted", (first,))
        if not budget.reserve_repair_attempt():
            return _failed("workflow_repair_budget_exhausted", (first,))

        repaired = await self._execute_step(
            plan=plan,
            user_query=user_query,
            decision=decision,
            context_runtime=context_runtime,
            user_item_id=user_item_id,
            memory_item_id=memory_item_id,
            budget=budget,
            trace_recorder=trace_recorder,
            trace_parent_context=trace_parent_context,
        )
        all_steps = (first, repaired)
        if repaired.status is StepExecutionStatus.TECHNICAL_FAILED:
            return _failed(repaired.error_code or "workflow_skill_execution_failed", all_steps)
        second_review = await self._review(
            plan_id=plan.plan_id,
            user_request=user_query,
            objective_type=plan.objective_type,
            steps=all_steps,
            budget=budget,
            trace_recorder=trace_recorder,
            trace_parent_context=trace_parent_context,
        )
        if isinstance(second_review, ControlledWorkflowResult):
            return _with_memory(second_review, all_steps)
        accepted = _accept_result(second_review, all_steps)
        if accepted is not None:
            return _accepted(accepted, all_steps)
        if _is_unanswerable(second_review):
            return _unanswerable(all_steps)
        if second_review.outcome is ReviewerOutcome.REPAIR:
            return _failed("workflow_repair_not_permitted", all_steps)
        if second_review.outcome is ReviewerOutcome.FAIL_CLOSED:
            return _failed("workflow_reviewer_fail_closed", all_steps)
        return _failed("workflow_reviewer_invalid", all_steps)

    async def _review(
        self,
        *,
        plan_id: str,
        user_request: str,
        objective_type: PlanObjectiveType,
        steps: tuple[StepExecutionResult, ...],
        budget: ExecutionBudget,
        trace_recorder: TraceSpanRecorder | None,
        trace_parent_context: TraceContext | None,
    ) -> ReviewerDecision | ControlledWorkflowResult:
        if not budget.reserve_reviewer_call():
            return _failed("workflow_reviewer_budget_exhausted", steps)
        span = start_recorded_span(
            trace_recorder,
            stage=TraceStage.WORKFLOW_REVIEW,
            component="agent_workflow",
            operation="review_workflow_result",
            parent_context=trace_parent_context,
        )
        request = WorkflowReviewRequest(
            plan_id=plan_id,
            user_request=user_request,
            objective_type=objective_type,
            steps=tuple(_summary(step) for step in steps),
            remaining_skill_calls=budget.remaining_skill_calls,
            remaining_repair_attempts=budget.remaining_repair_attempts,
        )
        try:
            decision = await _review_with_trace(
                self._reviewer,
                request=request,
                trace_recorder=trace_recorder,
                trace_parent_context=span,
            )
        except asyncio.CancelledError:
            complete_recorded_span(trace_recorder, span, status=SpanStatus.CANCELLED)
            raise
        except WorkflowProviderError as exc:
            complete_recorded_span(
                trace_recorder,
                span,
                status=SpanStatus.FAILED,
                error_code=exc.code,
            )
            return _failed(exc.code, steps)
        except Exception:
            complete_recorded_span(
                trace_recorder,
                span,
                status=SpanStatus.FAILED,
                error_code="workflow_reviewer_failed",
            )
            return _failed("workflow_reviewer_failed", steps)
        if not isinstance(decision, ReviewerDecision):
            complete_recorded_span(
                trace_recorder,
                span,
                status=SpanStatus.FAILED,
                error_code="workflow_reviewer_invalid",
            )
            return _failed("workflow_reviewer_invalid", steps)
        complete_recorded_span(
            trace_recorder,
            span,
            status=SpanStatus.COMPLETED,
            attributes={
                "execution_round": steps[-1].execution_round,
                "reviewer_outcome": decision.outcome.value,
                "reason_code": decision.reason_code,
                "repair_attempts": budget.repair_attempts,
                "reviewer_calls_used": budget.reviewer_calls,
                "budget_remaining": budget.remaining_skill_calls,
                "success": True,
                "result_status": decision.final_status.value,
            },
        )
        return decision

    async def _execute_step(self, **kwargs: Any) -> StepExecutionResult:
        """Add one request-local plan-step span around the existing single Skill call."""
        budget = kwargs["budget"]
        span = start_recorded_span(
            kwargs["trace_recorder"],
            stage=TraceStage.PLAN_STEP_EXECUTION,
            component="agent_workflow",
            operation="execute_plan_step",
            parent_context=kwargs["trace_parent_context"],
        )
        try:
            execution_kwargs = {**kwargs, "trace_parent_context": span}
            result = await self._executor.execute(**execution_kwargs)
        except asyncio.CancelledError:
            complete_recorded_span(kwargs["trace_recorder"], span, status=SpanStatus.CANCELLED)
            raise
        complete_recorded_span(
            kwargs["trace_recorder"],
            span,
            status=SpanStatus.COMPLETED
            if result.status is StepExecutionStatus.COMPLETED
            else SpanStatus.FAILED,
            error_code=result.error_code
            if result.status is not StepExecutionStatus.COMPLETED
            else None,
            attributes={
                "execution_round": result.execution_round,
                "skill_name": result.skill_name,
                "skill_calls_used": budget.skill_calls,
                "success": result.status is StepExecutionStatus.COMPLETED,
                "result_status": result.status.value,
            },
        )
        return result

    def _repair_is_allowed(
        self,
        review: ReviewerDecision,
        initial: StepExecutionResult,
        budget: ExecutionBudget,
    ) -> bool:
        return (
            review.final_status is ReviewerFinalStatus.REPAIR
            and review.accepted_step_id is None
            and review.repair_target == initial.step_id
            and initial.skill_name in self._policy.repairable_skill_names
            and budget.repair_attempts < budget.max_repair_attempts
            and budget.execution_rounds < budget.max_execution_rounds
            and budget.skill_calls < budget.max_skill_calls
            and budget.reviewer_calls < budget.max_reviewer_calls
        )


def _summary(step: StepExecutionResult) -> ReviewStepSummary:
    return ReviewStepSummary(
        step_id=step.step_id,
        skill_name=step.skill_name,
        status=step.status,
        error_code=step.error_code,
        citation_count=0 if step.skill_result is None else len(step.skill_result.citations),
        answer_available=step.skill_result is not None and step.skill_result.answer is not None,
        execution_round=step.execution_round,
    )


async def _plan_with_trace(planner: Any, **kwargs: Any) -> Any:
    method = getattr(planner, "plan_with_trace", None)
    if callable(method):
        return await method(**kwargs)
    return await planner.plan(request=kwargs["request"])


async def _review_with_trace(reviewer: Any, **kwargs: Any) -> Any:
    method = getattr(reviewer, "review_with_trace", None)
    if callable(method):
        return await method(**kwargs)
    return await reviewer.review(request=kwargs["request"])


def _accept_result(
    review: ReviewerDecision,
    steps: tuple[StepExecutionResult, ...],
):
    if (
        review.outcome is not ReviewerOutcome.ACCEPT
        or review.final_status is not ReviewerFinalStatus.ACCEPTED
        or review.accepted_step_id is None
        or review.repair_target is not None
    ):
        return None
    for step in reversed(steps):
        if (
            step.step_id == review.accepted_step_id
            and step.status is StepExecutionStatus.COMPLETED
            and step.skill_result is not None
        ):
            return step.skill_result
    return None


def _is_unanswerable(review: ReviewerDecision) -> bool:
    return (
        review.outcome is ReviewerOutcome.UNANSWERABLE
        and review.final_status is ReviewerFinalStatus.UNANSWERABLE
        and review.accepted_step_id is None
        and review.repair_target is None
    )


def _accepted(result: Any, steps: tuple[StepExecutionResult, ...]) -> ControlledWorkflowResult:
    return ControlledWorkflowResult(
        status=WorkflowStatus.ACCEPTED,
        accepted_result=result,
        steps=steps,
        memory_context_selected=any(step.memory_context_selected for step in steps),
    )


def _unanswerable(steps: tuple[StepExecutionResult, ...]) -> ControlledWorkflowResult:
    return ControlledWorkflowResult(
        status=WorkflowStatus.UNANSWERABLE,
        error_code="workflow_unanswerable",
        steps=steps,
        memory_context_selected=any(step.memory_context_selected for step in steps),
    )


def _failed(
    error_code: str, steps: tuple[StepExecutionResult, ...] = ()
) -> ControlledWorkflowResult:
    return ControlledWorkflowResult(
        status=WorkflowStatus.FAILED,
        error_code=error_code,
        steps=steps,
        memory_context_selected=any(step.memory_context_selected for step in steps),
    )


def _with_memory(
    result: ControlledWorkflowResult, steps: tuple[StepExecutionResult, ...]
) -> ControlledWorkflowResult:
    return result.model_copy(
        update={
            "steps": steps,
            "memory_context_selected": any(step.memory_context_selected for step in steps),
        }
    )
