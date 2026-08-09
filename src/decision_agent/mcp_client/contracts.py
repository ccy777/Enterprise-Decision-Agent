"""Strict client-side contracts for public Enterprise Data MCP tool responses."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _MCPClientContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EnterpriseSchema(_MCPClientContract):
    tables: dict[str, list[str]] = Field(min_length=1)


class BusinessDefinitions(_MCPClientContract):
    definitions: dict[str, str] = Field(min_length=1)


class MCPQueryResult(_MCPClientContract):
    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)
    row_count: int = Field(ge=0)
    truncated: bool
    normalized_sql: str | None = None
    accessed_tables: list[str] = Field(default_factory=list)
    elapsed_ms: float = Field(ge=0)
    error_code: str | None = None
