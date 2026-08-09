"""Immutable contracts for the deterministic controlled-agent workflow."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from decision_agent.coordination.models import SkillResult
from decision_agent.routing.models import RequestRoute

_SAFE_CODE = re.compile(r"^[a-z][a-z0-9_]{0,119}$")

PLAN_VERSION = "m8b-v1"
INVENTORY_RISK_SKILL = "inventory-risk-diagnosis"


class PlanObjectiveType(StrEnum):
    """Objectives explicitly permitted in the first controlled-workflow pilot."""

    MIXED_INVENTORY_DIAGNOSIS = "mixed_inventory_diagnosis"


class RequiredOutputType(StrEnum):
    """Typed output expectations; never provider-defined free text."""

    MIXED_DIAGNOSIS = "mixed_diagnosis"


class ReviewerOutcome(StrEnum):
    """The only workflow-reviewer transitions."""

    ACCEPT = "accept"
    REPAIR = "repair"
    UNANSWERABLE = "unanswerable"
    FAIL_CLOSED = "fail_closed"


class ReviewerFinalStatus(StrEnum):
    """Stable terminal or transition status declared by a Reviewer."""

    ACCEPTED = "accepted"
    REPAIR = "repair"
    UNANSWERABLE = "unanswerable"
    FAILED = "failed"


class WorkflowStatus(StrEnum):
    """Safe terminal state returned to Coordinator."""

    ACCEPTED = "accepted"
    UNANSWERABLE = "unanswerable"
    FAILED = "failed"


class StepExecutionStatus(StrEnum):
    """Safe per-step execution status without duplicating business output."""

    COMPLETED = "completed"
    BUSINESS_FAILED = "business_failed"
    TECHNICAL_FAILED = "technical_failed"


class PlanStep(BaseModel):
    """One declarative, non-executable plan step."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step_id: str = Field(min_length=1, max_length=80)
    sequence: int = Field(ge=1, le=8)
    skill_name: str = Field(min_length=1, max_length=120)
    objective_type: PlanObjectiveType
    depends_on: tuple[str, ...] = ()
    required_output_type: RequiredOutputType
    optional: bool = False

    @field_validator("step_id", "skill_name")
    @classmethod
    def _nonblank_identifier(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("identifier cannot be blank")
        return normalized


class ExecutionPlan(BaseModel):
    """Structured planner output, intentionally without query, Tool, or SQL fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: str = Field(min_length=1, max_length=80)
    plan_version: str = Field(min_length=1, max_length=40)
    objective_type: PlanObjectiveType
    steps: tuple[PlanStep, ...] = Field(max_length=8)
    max_execution_rounds: int = Field(ge=1, le=8)
    max_skill_calls: int = Field(ge=1, le=8)

    @field_validator("plan_id", "plan_version")
    @classmethod
    def _nonblank_plan_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("plan text cannot be blank")
        return normalized


class ReviewerDecision(BaseModel):
    """Structured Reviewer output with stable reason codes only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: ReviewerOutcome
    accepted_step_id: str | None = Field(default=None, max_length=80)
    repair_target: str | None = Field(default=None, max_length=80)
    reason_code: str = Field(min_length=1, max_length=120)
    final_status: ReviewerFinalStatus

    @field_validator("accepted_step_id", "repair_target")
    @classmethod
    def _nonblank_optional_identifier(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("identifier cannot be blank")
        return normalized

    @field_validator("reason_code")
    @classmethod
    def _safe_reason_code(cls, value: str) -> str:
        if not _SAFE_CODE.fullmatch(value):
            raise ValueError("reason_code must be a stable safe code")
        return value


class PlanningRequest(BaseModel):
    """Safe Planner input: route/capability metadata only, never Router-owned query text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    route: RequestRoute
    user_request: str = Field(min_length=1, max_length=4_000)
    objective_type: PlanObjectiveType
    allowed_skill_names: tuple[str, ...]


class ReviewStepSummary(BaseModel):
    """Reviewer-visible step status without Skill answer, citations, or business payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step_id: str = Field(min_length=1, max_length=80)
    skill_name: str = Field(min_length=1, max_length=120)
    status: StepExecutionStatus
    error_code: str | None = Field(default=None, max_length=120)
    citation_count: int = Field(ge=0, le=32)
    answer_available: bool
    execution_round: int = Field(ge=1, le=8)


class WorkflowReviewRequest(BaseModel):
    """Safe workflow-review input with counters, not full plan/result payloads."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: str = Field(min_length=1, max_length=80)
    user_request: str = Field(min_length=1, max_length=4_000)
    objective_type: PlanObjectiveType
    steps: tuple[ReviewStepSummary, ...]
    remaining_skill_calls: int = Field(ge=0, le=8)
    remaining_repair_attempts: int = Field(ge=0, le=8)


class StepExecutionResult(BaseModel):
    """Request-local execution control record referencing the original SkillResult only."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    step_id: str = Field(min_length=1, max_length=80)
    skill_name: str = Field(min_length=1, max_length=120)
    status: StepExecutionStatus
    skill_result: SkillResult | None = None
    error_code: str | None = Field(default=None, max_length=120)
    execution_round: int = Field(ge=1, le=8)
    memory_context_selected: bool = False


class ControlledWorkflowResult(BaseModel):
    """Internal terminal result; Coordinator alone maps it into its public contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: WorkflowStatus
    accepted_result: SkillResult | None = None
    error_code: str | None = Field(default=None, max_length=120)
    steps: tuple[StepExecutionResult, ...] = ()
    memory_context_selected: bool = False


@dataclass(slots=True)
class ExecutionBudget:
    """Per-request hard counters that reserve capacity before each external call."""

    max_plan_steps: int = 1
    max_execution_rounds: int = 2
    max_skill_calls: int = 2
    max_repair_attempts: int = 1
    max_reviewer_calls: int = 2
    execution_rounds: int = 0
    skill_calls: int = 0
    repair_attempts: int = 0
    reviewer_calls: int = 0

    def reserve_execution_round(self) -> bool:
        if self.execution_rounds >= self.max_execution_rounds:
            return False
        self.execution_rounds += 1
        return True

    def reserve_skill_call(self) -> bool:
        if self.skill_calls >= self.max_skill_calls:
            return False
        self.skill_calls += 1
        return True

    def reserve_repair_attempt(self) -> bool:
        if self.repair_attempts >= self.max_repair_attempts:
            return False
        self.repair_attempts += 1
        return True

    def reserve_reviewer_call(self) -> bool:
        if self.reviewer_calls >= self.max_reviewer_calls:
            return False
        self.reviewer_calls += 1
        return True

    @property
    def remaining_skill_calls(self) -> int:
        return self.max_skill_calls - self.skill_calls

    @property
    def remaining_repair_attempts(self) -> int:
        return self.max_repair_attempts - self.repair_attempts
