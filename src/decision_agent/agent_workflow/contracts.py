"""Internal protocols for deterministic M8B-1 planning and workflow review."""

from __future__ import annotations

from typing import Protocol

from decision_agent.agent_workflow.models import (
    ExecutionPlan,
    PlanningRequest,
    ReviewerDecision,
    WorkflowReviewRequest,
)


class Planner(Protocol):
    """Return a structured plan from safe route/capability metadata only."""

    async def plan(self, *, request: PlanningRequest) -> ExecutionPlan:
        """Produce one non-executable plan without query, Tool, or SQL fields."""


class WorkflowReviewer(Protocol):
    """Review safe execution summaries without invoking any business capability."""

    async def review(self, *, request: WorkflowReviewRequest) -> ReviewerDecision:
        """Return one bounded transition decision."""
