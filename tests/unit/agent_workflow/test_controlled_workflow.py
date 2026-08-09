"""Deterministic M8B-1 controlled-workflow tests with no external dependencies."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from decision_agent.agent_workflow import (
    INVENTORY_RISK_SKILL,
    PLAN_VERSION,
    ControlledAgentWorkflow,
    ControlledWorkflowPolicy,
    ExecutionBudget,
    ExecutionPlan,
    PlanObjectiveType,
    PlanStep,
    RequiredOutputType,
    ReviewerDecision,
    ReviewerFinalStatus,
    ReviewerOutcome,
    WorkflowStatus,
)
from decision_agent.agent_workflow.executor import BoundedExecutor
from decision_agent.agent_workflow.validation import PlanValidator
from decision_agent.context import RequestContextRuntime
from decision_agent.coordination.coordinator import Coordinator
from decision_agent.coordination.models import CoordinatorStatus, SkillResult, SkillStatus
from decision_agent.routing.models import RequestRoute, RouterDecision
from decision_agent.skills.contracts import SkillDefinition
from decision_agent.skills.registry import SkillRegistry

pytestmark = pytest.mark.offline_integration


def mixed_decision() -> RouterDecision:
    return RouterDecision(
        route=RequestRoute.MIXED,
        normalized_query="router-owned-mixed-request",
        decision_reason="mixed",
        knowledge_subquery="router-owned-knowledge-subquery",
        data_subquery="router-owned-data-subquery",
        missing_information=None,
        confidence=0.9,
    )


def single_step(**updates: object) -> PlanStep:
    values: dict[str, object] = {
        "step_id": "inventory_step",
        "sequence": 1,
        "skill_name": INVENTORY_RISK_SKILL,
        "objective_type": PlanObjectiveType.MIXED_INVENTORY_DIAGNOSIS,
        "depends_on": (),
        "required_output_type": RequiredOutputType.MIXED_DIAGNOSIS,
        "optional": False,
    }
    values.update(updates)
    return PlanStep(**values)


def plan(**updates: object) -> ExecutionPlan:
    values: dict[str, object] = {
        "plan_id": "plan_1",
        "plan_version": PLAN_VERSION,
        "objective_type": PlanObjectiveType.MIXED_INVENTORY_DIAGNOSIS,
        "steps": (single_step(),),
        "max_execution_rounds": 2,
        "max_skill_calls": 2,
    }
    values.update(updates)
    return ExecutionPlan(**values)


def completed_result() -> SkillResult:
    return SkillResult(
        status=SkillStatus.COMPLETED,
        skill_name=INVENTORY_RISK_SKILL,
        skill_version="1",
        route=RequestRoute.MIXED,
        answer="trusted mixed answer",
        citations=["[D1]", "[E1]"],
        executed_steps=("data", "knowledge", "synthesis"),
        selected_tool=None,
    )


def failed_result() -> SkillResult:
    return SkillResult(
        status=SkillStatus.FAILED,
        skill_name=INVENTORY_RISK_SKILL,
        skill_version="1",
        route=RequestRoute.MIXED,
        error_code="inventory_risk_synthesizer_failed",
        executed_steps=("data",),
    )


class FakeSkill:
    def __init__(self, result: SkillResult | BaseException | None = None) -> None:
        self.definition = SkillDefinition(
            name=INVENTORY_RISK_SKILL,
            version="1",
            description="fake inventory diagnosis",
            supported_route=RequestRoute.MIXED,
            input_contract=("request", "decision"),
            allowed_tools=("run_data_agent", "run_knowledge_agent"),
            steps=("diagnose",),
            output_contract=("mixed diagnosis",),
            failure_codes=("inventory_risk_synthesizer_failed",),
        )
        self._result = completed_result() if result is None else result
        self.calls = 0
        self.queries: list[str] = []
        self.trace_calls = 0
        self.legacy_calls = 0

    def is_applicable(self, request: str, decision: RouterDecision) -> bool:
        return decision.route is RequestRoute.MIXED and bool(request)

    async def execute(self, *, user_query: str, decision: RouterDecision) -> SkillResult:
        del decision
        self.calls += 1
        self.legacy_calls += 1
        self.queries.append(user_query)
        if isinstance(self._result, BaseException):
            raise self._result
        return self._result


class TraceAwareFakeSkill(FakeSkill):
    async def execute_with_trace(self, **kwargs: object) -> SkillResult:
        self.trace_calls += 1
        return await self.execute(
            user_query=kwargs["user_query"],  # type: ignore[arg-type]
            decision=kwargs["decision"],  # type: ignore[arg-type]
        )


class FailingTraceFakeSkill(FakeSkill):
    async def execute_with_trace(self, **_: object) -> SkillResult:
        self.trace_calls += 1
        raise RuntimeError("instrumentation entrypoint failure")


class FakePlanner:
    def __init__(self, result: object = None) -> None:
        self._result = plan() if result is None else result
        self.calls = 0
        self.requests = []

    async def plan(self, *, request: object) -> ExecutionPlan:
        self.calls += 1
        self.requests.append(request)
        if isinstance(self._result, BaseException):
            raise self._result
        return self._result  # type: ignore[return-value]


class FakeReviewer:
    def __init__(self, *decisions: object) -> None:
        self._decisions = list(decisions)
        self.calls = 0
        self.requests = []

    async def review(self, *, request: object) -> ReviewerDecision:
        self.calls += 1
        self.requests.append(request)
        decision = self._decisions.pop(0)
        if isinstance(decision, BaseException):
            raise decision
        return decision  # type: ignore[return-value]


class FakeRouter:
    def __init__(self, result: RouterDecision) -> None:
        self._result = result
        self.calls = 0

    async def route(self, *, user_query: str) -> RouterDecision:
        del user_query
        self.calls += 1
        return self._result


def accept(step_id: str = "inventory_step") -> ReviewerDecision:
    return ReviewerDecision(
        outcome=ReviewerOutcome.ACCEPT,
        accepted_step_id=step_id,
        repair_target=None,
        reason_code="accepted",
        final_status=ReviewerFinalStatus.ACCEPTED,
    )


def repair(step_id: str = "inventory_step") -> ReviewerDecision:
    return ReviewerDecision(
        outcome=ReviewerOutcome.REPAIR,
        accepted_step_id=None,
        repair_target=step_id,
        reason_code="retry_once",
        final_status=ReviewerFinalStatus.REPAIR,
    )


def unanswerable() -> ReviewerDecision:
    return ReviewerDecision(
        outcome=ReviewerOutcome.UNANSWERABLE,
        accepted_step_id=None,
        repair_target=None,
        reason_code="insufficient_result",
        final_status=ReviewerFinalStatus.UNANSWERABLE,
    )


def registry_with(skill: FakeSkill | None = None) -> SkillRegistry:
    registry = SkillRegistry()
    registry.register(FakeSkill() if skill is None else skill)
    return registry


async def run_workflow(
    workflow: ControlledAgentWorkflow,
    *,
    query: str = "router-owned-request",
) -> object:
    runtime = RequestContextRuntime(request_id="request_1", created_at=datetime.now(UTC))
    user_item = runtime.user_request(query)
    return await workflow.execute(
        user_query=query,
        decision=mixed_decision(),
        context_runtime=runtime,
        user_item_id=user_item.item_id,
        memory_item_id=None,
        trace_recorder=None,
        trace_parent_context=None,
    )


def workflow(
    *,
    planner: FakePlanner | None = None,
    reviewer: FakeReviewer | None = None,
    skill: FakeSkill | None = None,
    repairable: bool = False,
) -> tuple[ControlledAgentWorkflow, FakePlanner, FakeReviewer, FakeSkill]:
    actual_skill = FakeSkill() if skill is None else skill
    actual_planner = FakePlanner() if planner is None else planner
    actual_reviewer = FakeReviewer(accept()) if reviewer is None else reviewer
    controlled = ControlledAgentWorkflow(
        planner=actual_planner,
        reviewer=actual_reviewer,
        registry=registry_with(actual_skill),
        policy=ControlledWorkflowPolicy(
            enabled=True,
            repairable_skill_names=(INVENTORY_RISK_SKILL,) if repairable else (),
        ),
    )
    return controlled, actual_planner, actual_reviewer, actual_skill


def test_valid_single_step_inventory_plan_passes() -> None:
    result = PlanValidator(registry=registry_with(), budget=ExecutionBudget()).validate(plan())
    assert result.is_valid


@pytest.mark.parametrize(
    ("candidate", "code"),
    [
        (plan(steps=()), "workflow_plan_empty"),
        (plan(steps=(single_step(), single_step())), "workflow_plan_step_id_duplicate"),
        (
            plan(
                steps=(
                    single_step(),
                    single_step(step_id="second_step", sequence=2),
                )
            ),
            "workflow_plan_step_count_invalid",
        ),
        (plan(steps=(single_step(sequence=2),)), "workflow_plan_sequence_invalid"),
        (plan(steps=(single_step(depends_on=("missing",)),)), "workflow_plan_dependency_invalid"),
        (
            plan(
                steps=(
                    single_step(depends_on=("second_step",)),
                    single_step(step_id="second_step", sequence=2, depends_on=("inventory_step",)),
                )
            ),
            "workflow_plan_dependency_invalid",
        ),
        (
            plan(steps=(single_step(skill_name="unknown-skill"),)),
            "workflow_plan_skill_not_allowed",
        ),
        (
            plan(steps=(single_step(skill_name="enterprise-data-analysis"),)),
            "workflow_plan_skill_not_allowed",
        ),
        (plan(max_execution_rounds=3), "workflow_plan_budget_invalid"),
    ],
)
def test_plan_validator_rejects_invalid_controlled_plans(
    candidate: ExecutionPlan, code: str
) -> None:
    result = PlanValidator(registry=registry_with(), budget=ExecutionBudget()).validate(candidate)
    assert result.is_valid is False and result.error_code == code


def test_plan_contract_forbids_query_tool_and_sql_fields() -> None:
    payload = plan().model_dump()
    for forbidden in ("query", "tool_arguments", "sql"):
        with pytest.raises(ValidationError):
            ExecutionPlan.model_validate({**payload, forbidden: "untrusted"})


@pytest.mark.asyncio
async def test_accept_executes_one_skill_once_and_preserves_skill_result() -> None:
    controlled, planner, reviewer, skill = workflow()
    result = await run_workflow(controlled)

    assert result.status is WorkflowStatus.ACCEPTED  # type: ignore[attr-defined]
    assert result.accepted_result is skill._result  # type: ignore[attr-defined]
    assert planner.calls == reviewer.calls == skill.calls == 1
    assert skill.queries == ["router-owned-request"]
    assert not hasattr(planner.requests[0], "query")


@pytest.mark.asyncio
async def test_repair_reuses_the_same_skill_and_router_owned_query_once() -> None:
    controlled, planner, reviewer, skill = workflow(
        reviewer=FakeReviewer(repair(), accept()), repairable=True
    )
    result = await run_workflow(controlled)

    assert result.status is WorkflowStatus.ACCEPTED  # type: ignore[attr-defined]
    assert planner.calls == 1 and reviewer.calls == 2 and skill.calls == 2
    assert skill.queries == ["router-owned-request", "router-owned-request"]


@pytest.mark.asyncio
async def test_second_repair_fails_closed_without_a_third_skill_call() -> None:
    controlled, planner, reviewer, skill = workflow(
        reviewer=FakeReviewer(repair(), repair()), repairable=True
    )
    result = await run_workflow(controlled)

    assert result.status is WorkflowStatus.FAILED  # type: ignore[attr-defined]
    assert result.error_code == "workflow_repair_not_permitted"  # type: ignore[attr-defined]
    assert planner.calls == 1 and reviewer.calls == 2 and skill.calls == 2


@pytest.mark.asyncio
async def test_unpermitted_repair_does_not_consume_a_second_skill_call() -> None:
    controlled, planner, reviewer, skill = workflow(reviewer=FakeReviewer(repair()))
    result = await run_workflow(controlled)

    assert result.error_code == "workflow_repair_not_permitted"  # type: ignore[attr-defined]
    assert planner.calls == reviewer.calls == skill.calls == 1


@pytest.mark.asyncio
async def test_repair_cannot_target_another_step_or_change_the_plan() -> None:
    controlled, planner, reviewer, skill = workflow(
        reviewer=FakeReviewer(repair("other_step")), repairable=True
    )
    result = await run_workflow(controlled)

    assert result.error_code == "workflow_repair_not_permitted"  # type: ignore[attr-defined]
    assert planner.calls == reviewer.calls == skill.calls == 1


@pytest.mark.asyncio
async def test_unanswerable_stops_without_repair() -> None:
    controlled, planner, reviewer, skill = workflow(reviewer=FakeReviewer(unanswerable()))
    result = await run_workflow(controlled)

    assert result.status is WorkflowStatus.UNANSWERABLE  # type: ignore[attr-defined]
    assert result.error_code == "workflow_unanswerable"  # type: ignore[attr-defined]
    assert planner.calls == reviewer.calls == skill.calls == 1


@pytest.mark.asyncio
async def test_planner_reviewer_and_technical_skill_failures_fail_closed() -> None:
    for planner, reviewer, skill, code in (
        (
            FakePlanner(RuntimeError("planner")),
            FakeReviewer(accept()),
            FakeSkill(),
            "workflow_planner_failed",
        ),
        (
            FakePlanner(),
            FakeReviewer(RuntimeError("reviewer")),
            FakeSkill(),
            "workflow_reviewer_failed",
        ),
        (
            FakePlanner(),
            FakeReviewer(accept()),
            FakeSkill(RuntimeError("skill")),
            "workflow_skill_execution_failed",
        ),
    ):
        controlled, _, _, _ = workflow(planner=planner, reviewer=reviewer, skill=skill)
        result = await run_workflow(controlled)
        assert result.status is WorkflowStatus.FAILED and result.error_code == code  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_business_failed_skill_is_reviewable_but_cannot_be_accepted() -> None:
    controlled, _, reviewer, skill = workflow(
        reviewer=FakeReviewer(accept()), skill=FakeSkill(failed_result())
    )
    result = await run_workflow(controlled)

    assert result.error_code == "workflow_reviewer_invalid"  # type: ignore[attr-defined]
    assert skill.calls == reviewer.calls == 1


@pytest.mark.asyncio
async def test_invalid_reviewer_output_and_missing_or_failed_accept_fail_closed() -> None:
    cases = (
        (FakeReviewer("not-a-decision"), FakeSkill(), "workflow_reviewer_invalid"),
        (FakeReviewer(accept("missing")), FakeSkill(), "workflow_reviewer_invalid"),
        (FakeReviewer(accept()), FakeSkill(failed_result()), "workflow_reviewer_invalid"),
    )
    for reviewer, skill, code in cases:
        controlled, _, _, _ = workflow(reviewer=reviewer, skill=skill)
        result = await run_workflow(controlled)
        assert result.status is WorkflowStatus.FAILED and result.error_code == code  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_cancellation_is_reraised_from_planner_skill_and_reviewer() -> None:
    cases = (
        workflow(planner=FakePlanner(asyncio.CancelledError())),
        workflow(skill=FakeSkill(asyncio.CancelledError())),
        workflow(reviewer=FakeReviewer(asyncio.CancelledError())),
    )
    for controlled, _, _, _ in cases:
        with pytest.raises(asyncio.CancelledError):
            await run_workflow(controlled)


@pytest.mark.asyncio
async def test_trace_compatibility_failure_does_not_fallback_to_legacy_or_repeat_skill() -> None:
    skill = FailingTraceFakeSkill()
    controlled, _, reviewer, _ = workflow(skill=skill)
    result = await run_workflow(controlled)

    assert result.error_code == "workflow_skill_execution_failed"  # type: ignore[attr-defined]
    assert skill.trace_calls == 1 and skill.legacy_calls == skill.calls == 0
    assert reviewer.calls == 0


@pytest.mark.asyncio
async def test_trace_aware_skill_uses_one_internal_trace_entrypoint() -> None:
    skill = TraceAwareFakeSkill()
    controlled, _, _, _ = workflow(skill=skill)
    result = await run_workflow(controlled)

    assert result.status is WorkflowStatus.ACCEPTED  # type: ignore[attr-defined]
    assert skill.trace_calls == skill.legacy_calls == skill.calls == 1


@pytest.mark.asyncio
async def test_budget_blocks_skill_and_reviewer_calls_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import decision_agent.agent_workflow.workflow as workflow_module

    controlled, planner, reviewer, skill = workflow()
    monkeypatch.setattr(
        workflow_module, "ExecutionBudget", lambda: ExecutionBudget(max_skill_calls=0)
    )
    skill_result = await run_workflow(controlled)
    assert skill_result.error_code == "workflow_plan_budget_invalid"  # type: ignore[attr-defined]
    assert planner.calls == 1 and skill.calls == reviewer.calls == 0

    controlled, planner, reviewer, skill = workflow()
    monkeypatch.setattr(
        workflow_module, "ExecutionBudget", lambda: ExecutionBudget(max_reviewer_calls=0)
    )
    reviewer_result = await run_workflow(controlled)
    assert reviewer_result.error_code == "workflow_reviewer_budget_exhausted"  # type: ignore[attr-defined]
    assert planner.calls == skill.calls == 1 and reviewer.calls == 0


@pytest.mark.asyncio
async def test_bounded_executor_blocks_an_exhausted_skill_budget_before_call() -> None:
    skill = FakeSkill()
    runtime = RequestContextRuntime(request_id="request_1", created_at=datetime.now(UTC))
    user_item = runtime.user_request("router-owned-request")
    result = await BoundedExecutor(registry=registry_with(skill)).execute(
        plan=plan(),
        user_query="router-owned-request",
        decision=mixed_decision(),
        context_runtime=runtime,
        user_item_id=user_item.item_id,
        memory_item_id=None,
        budget=ExecutionBudget(max_skill_calls=0),
        trace_recorder=None,
        trace_parent_context=None,
    )

    assert result.error_code == "workflow_skill_budget_exhausted"
    assert skill.calls == 0


@pytest.mark.asyncio
async def test_concurrent_workflows_keep_budgets_results_and_queries_isolated() -> None:
    first, _, _, first_skill = workflow()
    second, _, _, second_skill = workflow()
    first_result, second_result = await asyncio.gather(
        run_workflow(first, query="first-router-query"),
        run_workflow(second, query="second-router-query"),
    )

    assert first_result.status is second_result.status is WorkflowStatus.ACCEPTED  # type: ignore[attr-defined]
    assert first_skill.queries == ["first-router-query"]
    assert second_skill.queries == ["second-router-query"]


@pytest.mark.asyncio
async def test_planner_failure_is_isolated_and_budgets_are_not_module_global() -> None:
    failed, _, _, failed_skill = workflow(planner=FakePlanner(RuntimeError("planner")))
    accepted, _, _, accepted_skill = workflow()
    failed_result, accepted_result = await asyncio.gather(
        run_workflow(failed), run_workflow(accepted)
    )

    first_budget = ExecutionBudget()
    second_budget = ExecutionBudget()
    assert failed_result.error_code == "workflow_planner_failed"  # type: ignore[attr-defined]
    assert accepted_result.status is WorkflowStatus.ACCEPTED  # type: ignore[attr-defined]
    assert failed_skill.calls == 0 and accepted_skill.calls == 1
    assert first_budget.reserve_skill_call() and second_budget.skill_calls == 0


@pytest.mark.asyncio
async def test_coordinator_defaults_to_existing_mixed_path_and_injected_workflow_is_opt_in() -> (
    None
):
    router = FakeRouter(mixed_decision())
    direct_skill = FakeSkill()
    direct_registry = registry_with(direct_skill)
    direct_result = await Coordinator(router=router, registry=direct_registry).execute(
        user_query="router-owned-request"
    )

    controlled, planner, reviewer, controlled_skill = workflow()
    controlled_router = FakeRouter(mixed_decision())
    controlled_result = await Coordinator(
        router=controlled_router,
        registry=controlled._registry,  # type: ignore[attr-defined]
        controlled_workflow=controlled,
    ).execute(user_query="router-owned-request")

    assert direct_result.status is CoordinatorStatus.COMPLETED and direct_skill.calls == 1
    assert router.calls == 1
    assert controlled_result.status is CoordinatorStatus.COMPLETED
    assert controlled_router.calls == planner.calls == reviewer.calls == controlled_skill.calls == 1
    assert controlled_result.answer == controlled_skill._result.answer


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route", [RequestRoute.KNOWLEDGE, RequestRoute.DATA, RequestRoute.UNSUPPORTED]
)
async def test_non_mixed_routes_do_not_enter_controlled_workflow(route: RequestRoute) -> None:
    class RouteSkill(FakeSkill):
        def __init__(self) -> None:
            super().__init__()
            self.definition = self.definition.model_copy(
                update={"name": f"{route}-skill", "supported_route": route}
            )

        def is_applicable(self, request: str, decision: RouterDecision) -> bool:
            return True

        async def execute(self, *, user_query: str, decision: RouterDecision) -> SkillResult:
            self.calls += 1
            citation = "[E1]" if route is RequestRoute.KNOWLEDGE else "[D1]"
            return SkillResult(
                status=SkillStatus.COMPLETED,
                skill_name=self.definition.name,
                skill_version="1",
                route=route,
                answer="answer",
                citations=[] if route is RequestRoute.UNSUPPORTED else [citation],
                executed_steps=("step",),
                selected_tool=None if route is RequestRoute.UNSUPPORTED else "tool",
            )

    controlled, planner, reviewer, _ = workflow()
    registry = SkillRegistry()
    if route is not RequestRoute.UNSUPPORTED:
        skill = RouteSkill()
        registry.register(skill)
    result = await Coordinator(
        router=FakeRouter(
            RouterDecision(
                route=route,
                normalized_query="query",
                decision_reason="reason",
                knowledge_subquery="knowledge" if route is RequestRoute.KNOWLEDGE else None,
                data_subquery="data" if route is RequestRoute.DATA else None,
                missing_information=None,
                confidence=0.9,
            )
        ),
        registry=registry,
        controlled_workflow=controlled,
    ).execute(user_query="query")

    assert planner.calls == reviewer.calls == 0
    assert result.status is (
        CoordinatorStatus.UNSUPPORTED
        if route is RequestRoute.UNSUPPORTED
        else CoordinatorStatus.COMPLETED
    )
