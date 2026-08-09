"""Strict public MCP tool contracts with no database connection details."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from decision_agent.data.models import SAFE_QUERY_SQL_MAX_LENGTH, SAFE_QUERY_SQL_MIN_LENGTH


class _MCPContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EnterpriseSchemaResponse(_MCPContract):
    tables: dict[str, list[str]]


class BusinessDefinitionsResponse(_MCPContract):
    definitions: dict[str, str]


class ExecuteSafeQueryInput(_MCPContract):
    sql: str = Field(min_length=SAFE_QUERY_SQL_MIN_LENGTH, max_length=SAFE_QUERY_SQL_MAX_LENGTH)


class ExecuteSafeQueryResponse(_MCPContract):
    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)
    row_count: int = Field(ge=0)
    truncated: bool
    normalized_sql: str | None = None
    accessed_tables: list[str] = Field(default_factory=list)
    elapsed_ms: float = Field(ge=0)
    error_code: str | None = None
