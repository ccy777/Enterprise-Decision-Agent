"""Typed contracts for safe, auditable read-only business queries."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

SAFE_QUERY_SQL_MIN_LENGTH = 1
SAFE_QUERY_SQL_MAX_LENGTH = 20_000


class _DataContract(BaseModel):
    """Strict, serializable data-layer contract."""

    model_config = ConfigDict(extra="forbid")


class SafeQueryRequest(_DataContract):
    """One untrusted SQL query submitted to the read-only service."""

    sql: str = Field(min_length=SAFE_QUERY_SQL_MIN_LENGTH, max_length=SAFE_QUERY_SQL_MAX_LENGTH)
    request_id: UUID = Field(default_factory=uuid4)


class QueryAudit(_DataContract):
    """Non-sensitive record of a safe-query decision and execution outcome."""

    request_id: UUID
    normalized_sql: str | None = None
    allowed: bool
    rejection_code: str | None = None
    accessed_tables: list[str] = Field(default_factory=list)
    elapsed_ms: float = Field(ge=0)
    row_count: int = Field(ge=0)
    truncated: bool = False


class SafeQueryResult(_DataContract):
    """Public result that never includes a connection URL or credentials."""

    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)
    row_count: int = Field(default=0, ge=0)
    truncated: bool = False
    elapsed_ms: float = Field(ge=0)
    accessed_tables: list[str] = Field(default_factory=list)
    audit: QueryAudit
    error_code: str | None = None


@dataclass(frozen=True)
class QueryExecution:
    """Internal raw executor response before public-value normalization."""

    columns: list[str]
    rows: list[tuple[Any, ...]]
    elapsed_started_at: float = 0.0

    @classmethod
    def started(cls, *, columns: list[str], rows: list[tuple[Any, ...]]) -> QueryExecution:
        """Create an execution response with a monotonic timing marker for test doubles."""
        return cls(columns=columns, rows=rows, elapsed_started_at=monotonic())
