"""Pure Pydantic contracts used across the application."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

MetadataScalar = str | int | float | bool | None
# Most provenance stays scalar, but deterministic chunking also needs to preserve
# ordered structural identifiers such as the Clause IDs grouped in a parent.
MetadataValue = MetadataScalar | tuple[str, ...]
Metadata = dict[str, MetadataValue]
NonEmptyString = Annotated[str, Field(min_length=1)]


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(UTC)


class ContractModel(BaseModel):
    """Base contract with strict input handling and JSON-safe serialization."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class EvidenceType(StrEnum):
    """Supported evidence origins for V1."""

    DOCUMENT = "document"
    GRAPH = "graph"
    SQL = "sql"
    WEB = "web"


class TaskType(StrEnum):
    """Bounded task categories independent of future orchestration code."""

    KNOWLEDGE = "knowledge"
    DATA = "data"
    HYBRID = "hybrid"
    CLARIFICATION = "clarification"


class TaskStatus(StrEnum):
    """Lifecycle values shared by tasks and task results."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class ReviewDecision(StrEnum):
    """Bounded outcomes produced by a future reviewer."""

    ACCEPT = "accept"
    RETRY = "retry"
    REPLAN = "replan"
    REJECT = "reject"


class AgentStateStatus(StrEnum):
    """Overall workflow state without implementing a workflow engine."""

    INITIAL = "initial"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentBlock(ContractModel):
    """Normalized source block retaining document provenance."""

    block_id: NonEmptyString
    document_id: NonEmptyString
    document_version: NonEmptyString
    content: str
    block_index: int = Field(default=0, ge=0)
    page_number: int | None = Field(default=None, ge=1)
    source: str | None = None
    file_name: str | None = None
    file_suffix: str | None = None
    mime_type: str | None = None
    file_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    block_content_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    parser_name: str | None = None
    section: str | None = None
    metadata: Metadata = Field(default_factory=dict)


class ParentChunk(ContractModel):
    """Large context chunk from which searchable child chunks are derived."""

    chunk_id: NonEmptyString
    document_id: NonEmptyString
    document_version: NonEmptyString
    content: NonEmptyString
    block_ids: list[NonEmptyString] = Field(default_factory=list)
    page_number: int | None = Field(default=None, ge=1)
    source: str | None = None
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)
    metadata: Metadata = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_offsets(self) -> ParentChunk:
        """Require a non-empty source span."""
        if self.end_offset <= self.start_offset:
            raise ValueError("end_offset must be greater than start_offset")
        return self


class ChildChunk(ContractModel):
    """Searchable child chunk linked to a stable parent identifier."""

    chunk_id: NonEmptyString
    parent_id: NonEmptyString
    document_id: NonEmptyString
    document_version: NonEmptyString
    content: NonEmptyString
    page_number: int | None = Field(default=None, ge=1)
    source: str | None = None
    start_offset: int | None = Field(default=None, ge=0)
    end_offset: int | None = Field(default=None, ge=0)
    metadata: Metadata = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_offsets(self) -> ChildChunk:
        """Require offsets to be paired and ordered when present."""
        if (self.start_offset is None) != (self.end_offset is None):
            raise ValueError("start_offset and end_offset must be provided together")
        if self.start_offset is not None and self.end_offset <= self.start_offset:
            raise ValueError("end_offset must be greater than start_offset")
        return self


class Evidence(ContractModel):
    """Unified, citation-ready evidence returned by every future data source."""

    evidence_id: UUID = Field(default_factory=uuid4)
    evidence_type: EvidenceType
    content: NonEmptyString
    source_name: NonEmptyString
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    document_id: str | None = None
    chunk_id: str | None = None
    document_version: str | None = None
    extraction_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: Metadata = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_graph_provenance(self) -> Evidence:
        """Ensure graph facts can always be traced to their source chunk."""
        if self.evidence_type is EvidenceType.GRAPH:
            required = (self.document_id, self.chunk_id, self.document_version)
            if not all(required) or self.extraction_confidence is None:
                raise ValueError(
                    "graph evidence requires document_id, chunk_id, document_version, "
                    "and extraction_confidence"
                )
        return self


class Task(ContractModel):
    """Stable unit of planned work with explicit dependencies."""

    task_id: UUID = Field(default_factory=uuid4)
    task_type: TaskType
    description: NonEmptyString
    status: TaskStatus = TaskStatus.PENDING
    dependency_ids: list[UUID] = Field(default_factory=list)
    metadata: Metadata = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_dependencies(self) -> Task:
        """Reject self-references and duplicate task dependencies."""
        if self.task_id in self.dependency_ids:
            raise ValueError("task cannot depend on itself")
        if len(self.dependency_ids) != len(set(self.dependency_ids)):
            raise ValueError("task dependencies must be unique")
        return self


class ErrorRecord(ContractModel):
    """Serializable error information safe to pass between layers."""

    code: NonEmptyString
    message: NonEmptyString
    retryable: bool = False
    task_id: UUID | None = None
    details: Metadata = Field(default_factory=dict)
    occurred_at: datetime = Field(default_factory=utc_now)


class TaskResult(ContractModel):
    """Typed output of a task, including evidence and structured errors."""

    task_id: UUID
    status: TaskStatus
    summary: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    errors: list[ErrorRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_terminal_result(self) -> TaskResult:
        """Keep successful and failed results internally consistent."""
        if self.status is TaskStatus.FAILED and not self.errors:
            raise ValueError("failed task result requires at least one error")
        if self.status is TaskStatus.SUCCEEDED and self.errors:
            raise ValueError("successful task result cannot contain errors")
        return self


class ReviewResult(ContractModel):
    """Structured evidence review decision."""

    decision: ReviewDecision
    rationale: NonEmptyString
    evidence_ids: list[UUID] = Field(default_factory=list)
    missing_requirements: list[NonEmptyString] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class AgentState(ContractModel):
    """JSON-safe workflow state that supports an empty initial request."""

    request_id: UUID = Field(default_factory=uuid4)
    status: AgentStateStatus = AgentStateStatus.INITIAL
    query: str = ""
    tasks: list[Task] = Field(default_factory=list)
    task_results: list[TaskResult] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    errors: list[ErrorRecord] = Field(default_factory=list)
    review: ReviewResult | None = None
    final_report: str | None = None
    metadata: Metadata = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
