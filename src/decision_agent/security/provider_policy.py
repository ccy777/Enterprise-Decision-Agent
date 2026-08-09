"""Default-deny provider egress contracts used by configured runtimes."""

from __future__ import annotations

from enum import IntEnum, StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from decision_agent.exceptions import DecisionAgentError


class DataClassification(IntEnum):
    """Ordered, server-owned classifications; callers cannot lower a value."""

    PUBLIC = 0
    INTERNAL = 1
    CONFIDENTIAL = 2
    RESTRICTED = 3
    SECRET = 4


class ProviderStage(StrEnum):
    ROUTING = "routing"
    PLANNING = "planning"
    DATA_PLANNING = "data_planning"
    DATA_ANSWER = "data_answer"
    EVIDENCE_SELECTION = "evidence_selection"
    ANSWERABILITY_REVIEW = "answerability_review"
    KNOWLEDGE_ANSWER = "knowledge_answer"
    INVENTORY_SYNTHESIS = "inventory_synthesis"
    WORKFLOW_REVIEW = "workflow_review"


class ProviderPolicyError(DecisionAgentError):
    """Stable egress denial that never retains a payload."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class ProviderStagePolicy(BaseModel):
    """One immutable stage entry in the server-owned egress matrix."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: ProviderStage
    allowed_classifications: frozenset[DataClassification]
    allow_query: bool = False
    allow_knowledge_summary: bool = False
    allow_data_aggregate: bool = False
    allow_identity: bool = False
    allow_raw_document: bool = False
    allow_raw_rows: bool = False
    max_context_chars: int = Field(gt=0, le=16_000)
    max_evidence_items: int = Field(ge=0, le=32)

    @model_validator(mode="after")
    def _forbid_unsafe_permanent_classes(self) -> ProviderStagePolicy:
        if DataClassification.SECRET in self.allowed_classifications:
            raise ValueError("secret data can never be provider eligible")
        if self.allow_raw_rows or self.allow_raw_document or self.allow_identity:
            raise ValueError("raw rows, documents, and identity never leave the boundary")
        return self


class ProviderPolicy(BaseModel):
    """Versioned, closed-set policy with no implicit allow-all fallback."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_id: str = Field(min_length=1, max_length=80)
    version: str = Field(min_length=1, max_length=40)
    enabled: bool = True
    max_provider_calls: int = Field(gt=0, le=32)
    require_input_redaction: bool = True
    require_output_check: bool = True
    stages: tuple[ProviderStagePolicy, ...] = Field(min_length=1, max_length=9)

    @model_validator(mode="after")
    def _verify_closed_stage_matrix(self) -> ProviderPolicy:
        seen = {entry.stage for entry in self.stages}
        if len(seen) != len(self.stages):
            raise ValueError("provider stages must be unique")
        return self

    def stage_policy(self, stage: ProviderStage) -> ProviderStagePolicy:
        if not self.enabled:
            raise ProviderPolicyError("provider_stage_forbidden")
        for entry in self.stages:
            if entry.stage is stage:
                return entry
        raise ProviderPolicyError("provider_stage_forbidden")

    @classmethod
    def controlled_mixed(cls) -> ProviderPolicy:
        """Return the least-privilege matrix for the existing nine-call path."""
        query_only = frozenset({DataClassification.PUBLIC, DataClassification.INTERNAL})
        summary = frozenset(
            {
                DataClassification.PUBLIC,
                DataClassification.INTERNAL,
                DataClassification.CONFIDENTIAL,
            }
        )
        return cls(
            policy_id="m8c-provider-egress",
            version="1",
            max_provider_calls=9,
            stages=(
                ProviderStagePolicy(
                    stage=ProviderStage.ROUTING,
                    allowed_classifications=query_only,
                    allow_query=True,
                    max_context_chars=2_000,
                    max_evidence_items=0,
                ),
                ProviderStagePolicy(
                    stage=ProviderStage.PLANNING,
                    allowed_classifications=query_only,
                    allow_query=True,
                    max_context_chars=2_000,
                    max_evidence_items=0,
                ),
                ProviderStagePolicy(
                    stage=ProviderStage.DATA_PLANNING,
                    allowed_classifications=query_only,
                    allow_query=True,
                    max_context_chars=4_000,
                    max_evidence_items=0,
                ),
                ProviderStagePolicy(
                    stage=ProviderStage.DATA_ANSWER,
                    allowed_classifications=summary,
                    allow_query=True,
                    allow_data_aggregate=True,
                    max_context_chars=8_000,
                    max_evidence_items=8,
                ),
                ProviderStagePolicy(
                    stage=ProviderStage.EVIDENCE_SELECTION,
                    allowed_classifications=summary,
                    allow_knowledge_summary=True,
                    max_context_chars=8_000,
                    max_evidence_items=16,
                ),
                ProviderStagePolicy(
                    stage=ProviderStage.ANSWERABILITY_REVIEW,
                    allowed_classifications=summary,
                    allow_knowledge_summary=True,
                    max_context_chars=8_000,
                    max_evidence_items=16,
                ),
                ProviderStagePolicy(
                    stage=ProviderStage.KNOWLEDGE_ANSWER,
                    allowed_classifications=summary,
                    allow_query=True,
                    allow_knowledge_summary=True,
                    max_context_chars=8_000,
                    max_evidence_items=16,
                ),
                ProviderStagePolicy(
                    stage=ProviderStage.INVENTORY_SYNTHESIS,
                    allowed_classifications=summary,
                    allow_query=True,
                    allow_knowledge_summary=True,
                    allow_data_aggregate=True,
                    max_context_chars=12_000,
                    max_evidence_items=16,
                ),
                ProviderStagePolicy(
                    stage=ProviderStage.WORKFLOW_REVIEW,
                    allowed_classifications=summary,
                    max_context_chars=4_000,
                    max_evidence_items=8,
                ),
            ),
        )


def require_provider_egress(
    *,
    policy: ProviderPolicy | None,
    stage: ProviderStage,
    classification: DataClassification,
    payload_size: int,
    call_count: int,
) -> ProviderStagePolicy:
    """Validate all fixed inputs before a transport is allowed to run."""
    if policy is None:
        raise ProviderPolicyError("provider_policy_missing")
    entry = policy.stage_policy(stage)
    if classification not in entry.allowed_classifications:
        raise ProviderPolicyError("provider_data_classification_forbidden")
    if payload_size > entry.max_context_chars:
        raise ProviderPolicyError("provider_payload_too_large")
    if call_count >= policy.max_provider_calls:
        raise ProviderPolicyError("provider_budget_exceeded")
    return entry
