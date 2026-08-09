"""Fake-transport tests for M8B structured provider adapters."""

from __future__ import annotations

import asyncio
import json

import pytest

from decision_agent.agent_workflow.models import (
    INVENTORY_RISK_SKILL,
    PLAN_VERSION,
    PlanningRequest,
    PlanObjectiveType,
    ReviewerFinalStatus,
    ReviewerOutcome,
    WorkflowReviewRequest,
)
from decision_agent.agent_workflow.providers import (
    OpenAICompatibleWorkflowPlanner,
    OpenAICompatibleWorkflowReviewer,
    WorkflowProviderError,
)
from decision_agent.observability.provider import ProviderTraceMetadata
from decision_agent.routing.models import RequestRoute


class FakeProvider:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.calls = 0

    def provider_trace_metadata(self) -> ProviderTraceMetadata:
        return ProviderTraceMetadata(provider="fake", model="fake-model", retry_count=0)

    async def complete_chat(self, **_: object) -> dict[str, object]:
        self.calls += 1
        if isinstance(self.payload, BaseException):
            raise self.payload
        return self.payload  # type: ignore[return-value]


def payload(value: object) -> dict[str, object]:
    return {
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
    }


def planning_request() -> PlanningRequest:
    return PlanningRequest(
        route=RequestRoute.MIXED,
        user_request="planner may read this request",
        objective_type=PlanObjectiveType.MIXED_INVENTORY_DIAGNOSIS,
        allowed_skill_names=(INVENTORY_RISK_SKILL,),
    )


@pytest.mark.asyncio
async def test_real_planner_adapter_parses_one_safe_plan_once() -> None:
    provider = FakeProvider(
        payload(
            {
                "plan_id": "plan_1",
                "plan_version": PLAN_VERSION,
                "objective_type": "mixed_inventory_diagnosis",
                "steps": [
                    {
                        "step_id": "inventory_step",
                        "sequence": 1,
                        "skill_name": INVENTORY_RISK_SKILL,
                        "objective_type": "mixed_inventory_diagnosis",
                        "depends_on": [],
                        "required_output_type": "mixed_diagnosis",
                        "optional": False,
                    }
                ],
                "max_execution_rounds": 2,
                "max_skill_calls": 2,
            }
        )
    )
    result = await OpenAICompatibleWorkflowPlanner(provider=provider).plan(
        request=planning_request()
    )

    assert result.steps[0].skill_name == INVENTORY_RISK_SKILL and provider.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value",
    [
        {"plan_id": "missing-fields"},
        {"plan_id": "x", "query": "forbidden"},
    ],
)
async def test_planner_schema_failures_are_safe_and_single_call(value: object) -> None:
    provider = FakeProvider(payload(value))
    with pytest.raises(WorkflowProviderError) as error:
        await OpenAICompatibleWorkflowPlanner(provider=provider).plan(request=planning_request())
    assert error.value.code == "workflow_planner_schema_invalid" and provider.calls == 1


@pytest.mark.asyncio
async def test_reviewer_parses_safe_unanswerable_and_reraises_cancellation() -> None:
    provider = FakeProvider(
        payload(
            {
                "outcome": "unanswerable",
                "accepted_step_id": None,
                "repair_target": None,
                "reason_code": "insufficient_result",
                "final_status": "unanswerable",
            }
        )
    )
    request = WorkflowReviewRequest(
        plan_id="plan_1",
        user_request="review target",
        objective_type=PlanObjectiveType.MIXED_INVENTORY_DIAGNOSIS,
        steps=(),
        remaining_skill_calls=1,
        remaining_repair_attempts=0,
    )
    result = await OpenAICompatibleWorkflowReviewer(provider=provider).review(request=request)
    assert result.outcome is ReviewerOutcome.UNANSWERABLE
    assert result.final_status is ReviewerFinalStatus.UNANSWERABLE and provider.calls == 1

    with pytest.raises(asyncio.CancelledError):
        await OpenAICompatibleWorkflowReviewer(
            provider=FakeProvider(asyncio.CancelledError())
        ).review(request=request)
