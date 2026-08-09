"""Versioned, deterministic enterprise-operations question-set loading and verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from decision_agent.data.models import SafeQueryRequest
from decision_agent.data.safe_query_service import SafeQueryService


class OperationsQuestion(BaseModel):
    """One business question with database-recomputable expected output."""

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(pattern=r"^ops-v1-q[0-9]{2}$")
    question: str = Field(min_length=1)
    business_domain: str = Field(min_length=1)
    expected_result: dict[str, Any]
    verification_sql: str = Field(min_length=1)
    important_filters: list[str] = Field(min_length=1)
    notes: str = Field(min_length=1)


def default_question_set_path() -> Path:
    """Return the versioned business question-set path without reading it at import time."""
    return Path("datasets/enterprise_operations/v1/questions.json")


def load_operations_questions(path: Path | None = None) -> list[OperationsQuestion]:
    """Load strict JSON data and reject duplicate question identifiers."""
    source = path or default_question_set_path()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("operations question set must be a JSON array")
    questions = [OperationsQuestion.model_validate(item) for item in payload]
    if len(questions) < 8:
        raise ValueError("operations question set must contain at least eight questions")
    ids = [question.question_id for question in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("operations question IDs must be unique")
    return questions


async def verify_operations_ground_truth(
    *,
    service: SafeQueryService,
    questions: list[OperationsQuestion],
) -> list[tuple[str, bool, dict[str, Any]]]:
    """Run each versioned SQL rule and compare its public columns/rows exactly."""
    outcomes: list[tuple[str, bool, dict[str, Any]]] = []
    for question in questions:
        result = await service.execute(SafeQueryRequest(sql=question.verification_sql))
        actual = {"columns": result.columns, "rows": result.rows}
        outcomes.append(
            (
                question.question_id,
                result.error_code is None and actual == question.expected_result,
                actual,
            )
        )
    return outcomes
