"""Independent stdio MCP server exposing the existing guarded enterprise data boundary."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from decision_agent.config import Settings
from decision_agent.data.business_definitions import BUSINESS_DEFINITIONS
from decision_agent.data.models import (
    SAFE_QUERY_SQL_MAX_LENGTH,
    SAFE_QUERY_SQL_MIN_LENGTH,
    SafeQueryRequest,
)
from decision_agent.data.safe_query_service import SafeQueryService
from decision_agent.data.sql_guard import BUSINESS_TABLE_COLUMNS
from decision_agent.exceptions import ConfigurationError
from decision_agent.mcp_server.contracts import (
    BusinessDefinitionsResponse,
    EnterpriseSchemaResponse,
    ExecuteSafeQueryInput,
    ExecuteSafeQueryResponse,
)

SafeQueryServiceFactory = Callable[[], SafeQueryService]
_DATABASE_PREFLIGHT_ENVIRONMENT_KEY = "DECISION_AGENT_MCP_DATABASE_PREFLIGHT"
_DATABASE_PREFLIGHT_RESULT_FILE_KEY = "DECISION_AGENT_MCP_DATABASE_PREFLIGHT_RESULT_FILE"


class EnterpriseDataToolService:
    """MCP tool adapter that delegates all SQL execution to ``SafeQueryService``."""

    def __init__(self, service_factory: SafeQueryServiceFactory) -> None:
        self._service_factory = service_factory
        self._service: SafeQueryService | None = None

    async def get_enterprise_schema(self) -> EnterpriseSchemaResponse:
        """Return the SQLGuard allowlist without maintaining a second schema definition."""
        return EnterpriseSchemaResponse(
            tables={table: sorted(columns) for table, columns in BUSINESS_TABLE_COLUMNS.items()}
        )

    async def get_business_definitions(self) -> BusinessDefinitionsResponse:
        """Return the canonical business definitions, never benchmark answers or SQL."""
        return BusinessDefinitionsResponse(definitions=BUSINESS_DEFINITIONS)

    async def execute_safe_query(self, request: ExecuteSafeQueryInput) -> ExecuteSafeQueryResponse:
        """Execute one untrusted statement through the existing SafeQueryService only."""
        try:
            result = await self._get_service().execute(SafeQueryRequest(sql=request.sql))
        except ConfigurationError:
            return ExecuteSafeQueryResponse(
                row_count=0,
                truncated=False,
                elapsed_ms=0,
                error_code="database_unavailable",
            )
        return ExecuteSafeQueryResponse(
            columns=result.columns,
            rows=result.rows,
            row_count=result.row_count,
            truncated=result.truncated,
            normalized_sql=result.audit.normalized_sql,
            accessed_tables=result.accessed_tables,
            elapsed_ms=result.elapsed_ms,
            error_code=result.error_code,
        )

    def _get_service(self) -> SafeQueryService:
        if self._service is None:
            self._service = self._service_factory()
        return self._service

    async def preflight_database(self) -> dict[str, object]:
        """Run the opt-in transport probe inside the owning MCP Server process."""
        try:
            result = await self._get_service().preflight_database()
        except ConfigurationError:
            return _database_preflight_payload(
                error_code="mysql_configuration_missing",
                connection_established=False,
                readonly_ready=False,
                sql_guard_ready=False,
                shutdown="not_started",
                exception_type="ConfigurationError",
            )
        return _database_preflight_payload(
            error_code=result.error_code,
            connection_established=result.connection_established,
            readonly_ready=result.readonly_ready,
            sql_guard_ready=result.sql_guard_ready,
            shutdown=result.shutdown,
            exception_type=result.exception_type,
        )

    async def aclose(self) -> None:
        """Close a lazily constructed service when the MCP process stops."""
        if self._service is not None:
            await self._service.aclose()


def create_enterprise_data_server(
    *, service_factory: SafeQueryServiceFactory | None = None
) -> FastMCP:
    """Create a stdio-only MCP application without opening a database connection at import time."""
    tools = EnterpriseDataToolService(
        service_factory or (lambda: SafeQueryService.from_settings(Settings()))
    )

    @asynccontextmanager
    async def lifespan(_: FastMCP):
        try:
            if os.environ.get(_DATABASE_PREFLIGHT_ENVIRONMENT_KEY) == "true":
                _write_database_preflight_stage("protocol_path_received")
                _write_database_preflight_stage("preflight_started")
                result = await tools.preflight_database()
                _write_database_preflight_result(result)
                _write_database_preflight_stage("preflight_result_written")
                if result["status"] != "passed":
                    raise RuntimeError(f"mcp_database_preflight_{result['error_code']}")
            yield
        finally:
            await tools.aclose()

    server = FastMCP(
        "enterprise-data-mcp",
        instructions="Read-only enterprise operations schema and SafeQueryService access.",
        lifespan=lifespan,
    )

    @server.tool()
    async def get_enterprise_schema() -> dict[str, list[str]]:
        """Get the SafeQueryService-authorized enterprise tables and fields."""
        return (await tools.get_enterprise_schema()).tables

    @server.tool()
    async def get_business_definitions() -> dict[str, str]:
        """Get canonical sales, purchase, inventory, delivery, and natural-month definitions."""
        return (await tools.get_business_definitions()).definitions

    @server.tool()
    async def execute_safe_query(
        sql: Annotated[
            str,
            Field(
                min_length=SAFE_QUERY_SQL_MIN_LENGTH,
                max_length=SAFE_QUERY_SQL_MAX_LENGTH,
                description=(
                    "One untrusted SQL statement. "
                    "Only SafeQueryService-approved SELECT queries run."
                ),
            ),
        ],
    ) -> ExecuteSafeQueryResponse:
        """Execute one read-only SQL statement through SQLGuard and SafeQueryService."""
        request = ExecuteSafeQueryInput(sql=sql)
        return await tools.execute_safe_query(request)

    return server


def _database_preflight_payload(
    *,
    error_code: str,
    connection_established: bool,
    readonly_ready: bool,
    sql_guard_ready: bool,
    shutdown: str,
    exception_type: str | None,
) -> dict[str, object]:
    return {
        "component": "mysql",
        "phase": "mcp_server_preflight",
        "status": "passed" if error_code == "none" else "failed",
        "error_code": error_code,
        "connection_established": connection_established,
        "readonly_ready": readonly_ready,
        "sql_guard_ready": sql_guard_ready,
        "shutdown": shutdown,
        "exception_type": exception_type,
    }


def _write_database_preflight_result(result: dict[str, object]) -> None:
    """Publish only fixed lifecycle facts for the parent MCP client to consume."""
    result_file = os.environ.get(_DATABASE_PREFLIGHT_RESULT_FILE_KEY)
    if not result_file:
        return
    payload = {
        key: result[key]
        for key in (
            "status",
            "error_code",
            "connection_established",
            "readonly_ready",
            "sql_guard_ready",
            "shutdown",
        )
    }
    target = Path(result_file)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, separators=(",", ":"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)


def _write_database_preflight_stage(stage: str) -> None:
    """Publish a fixed lifecycle stage without child configuration or payloads."""
    result_file = os.environ.get(_DATABASE_PREFLIGHT_RESULT_FILE_KEY)
    if not result_file:
        return
    target = Path(result_file).with_suffix(".stage.json")
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump({"stage": stage}, stream, separators=(",", ":"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)


def run_stdio_server() -> None:
    """Run the MCP protocol over stdio; SDK owns stdout framing."""
    _write_database_preflight_stage("server_process_entered")
    create_enterprise_data_server().run(transport="stdio")
