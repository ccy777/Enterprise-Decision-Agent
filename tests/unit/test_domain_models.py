"""Offline tests for canonical domain invariants and serialization."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from decision_agent.domain import (
    AgentState,
    ChildChunk,
    ErrorRecord,
    Evidence,
    EvidenceType,
    Task,
    TaskResult,
    TaskStatus,
    TaskType,
)


def test_agent_state_supports_empty_json_safe_initial_state() -> None:
    state = AgentState()

    payload = state.model_dump(mode="json")

    assert payload["status"] == "initial"
    assert payload["tasks"] == []
    assert isinstance(payload["request_id"], str)


def test_child_chunk_requires_parent_and_valid_offset_pair() -> None:
    with pytest.raises(ValidationError, match="parent_id"):
        ChildChunk(
            chunk_id="child-1",
            parent_id="",
            document_id="doc-1",
            document_version="v1",
            content="text",
        )

    with pytest.raises(ValidationError, match="provided together"):
        ChildChunk(
            chunk_id="child-1",
            parent_id="parent-1",
            document_id="doc-1",
            document_version="v1",
            content="text",
            start_offset=0,
        )


def test_evidence_type_and_scores_are_bounded() -> None:
    with pytest.raises(ValidationError):
        Evidence(evidence_type="email", content="x", source_name="source")

    with pytest.raises(ValidationError, match="less than or equal to 1"):
        Evidence(
            evidence_type=EvidenceType.DOCUMENT,
            content="x",
            source_name="source",
            score=1.1,
        )


def test_graph_evidence_requires_complete_provenance() -> None:
    with pytest.raises(ValidationError, match="graph evidence requires"):
        Evidence(
            evidence_type=EvidenceType.GRAPH,
            content="Product A is owned by Department X",
            source_name="knowledge-graph",
        )


def test_task_rejects_self_and_duplicate_dependencies() -> None:
    task_id = uuid4()
    with pytest.raises(ValidationError, match="depend on itself"):
        Task(
            task_id=task_id,
            task_type=TaskType.KNOWLEDGE,
            description="Find evidence",
            dependency_ids=[task_id],
        )

    dependency_id = uuid4()
    with pytest.raises(ValidationError, match="must be unique"):
        Task(
            task_type=TaskType.DATA,
            description="Analyze data",
            dependency_ids=[dependency_id, dependency_id],
        )


def test_failed_result_requires_structured_error() -> None:
    with pytest.raises(ValidationError, match="requires at least one error"):
        TaskResult(task_id=uuid4(), status=TaskStatus.FAILED)

    result = TaskResult(
        task_id=uuid4(),
        status=TaskStatus.FAILED,
        errors=[ErrorRecord(code="worker_failed", message="Worker failed")],
    )
    assert result.model_dump(mode="json")["errors"][0]["retryable"] is False
