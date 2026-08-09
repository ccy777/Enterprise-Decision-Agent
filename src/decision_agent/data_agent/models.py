"""Stable public evidence contracts for the single-query Data Agent."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from decision_agent.data.models import SafeQueryResult


class DataEvidence(BaseModel):
    """Traceable public result derived only from a successful guarded query."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(pattern=r"^D[1-9][0-9]*$")
    normalized_sql: str = Field(min_length=1)
    columns: list[str]
    rows: list[list[Any]]
    row_count: int = Field(ge=0)
    truncated: bool
    accessed_tables: list[str]
    elapsed_ms: float = Field(ge=0)

    @classmethod
    def from_safe_query_result(cls, *, evidence_id: str, result: SafeQueryResult) -> DataEvidence:
        """Reject unsuccessful results instead of allowing unverified data evidence."""
        if (
            result.error_code is not None
            or not result.audit.allowed
            or not result.audit.normalized_sql
            or not result.accessed_tables
        ):
            raise ValueError("DataEvidence requires a successful guarded query")
        return cls(
            evidence_id=evidence_id,
            normalized_sql=result.audit.normalized_sql,
            columns=result.columns,
            rows=result.rows,
            row_count=result.row_count,
            truncated=result.truncated,
            accessed_tables=result.accessed_tables,
            elapsed_ms=result.elapsed_ms,
        )
