"""Unit coverage for the public Enterprise Data MCP tool adapters."""

from __future__ import annotations

import pytest

from decision_agent.data.models import QueryAudit, SafeQueryResult
from decision_agent.data.safe_query_service import DatabasePreflightResult
from decision_agent.data.sql_guard import BUSINESS_TABLE_COLUMNS
from decision_agent.exceptions import ConfigurationError
from decision_agent.mcp_server.contracts import ExecuteSafeQueryInput
from decision_agent.mcp_server.enterprise_data_server import EnterpriseDataToolService


class RecordingSafeQueryService:
    def __init__(self, result: SafeQueryResult) -> None:
        self.result = result
        self.sql: list[str] = []

    async def execute(self, request: object) -> SafeQueryResult:
        self.sql.append(request.sql)  # type: ignore[attr-defined]
        return self.result

    async def aclose(self) -> None:
        return None


def _result(*, error_code: str | None = None) -> SafeQueryResult:
    return SafeQueryResult(
        columns=["product_id"] if error_code is None else [],
        rows=[["P100"]] if error_code is None else [],
        row_count=1 if error_code is None else 0,
        truncated=False,
        elapsed_ms=1.5,
        accessed_tables=["products"] if error_code is None else [],
        audit=QueryAudit(
            request_id="00000000-0000-0000-0000-000000000001",
            normalized_sql="SELECT product_id FROM products LIMIT 200"
            if error_code is None
            else None,
            allowed=error_code is None,
            rejection_code=error_code,
            accessed_tables=["products"] if error_code is None else [],
            elapsed_ms=1.5,
            row_count=1 if error_code is None else 0,
        ),
        error_code=error_code,
    )


@pytest.mark.asyncio
async def test_schema_tool_uses_the_existing_sql_guard_allowlist() -> None:
    adapter = EnterpriseDataToolService(lambda: RecordingSafeQueryService(_result()))  # type: ignore[arg-type]
    response = await adapter.get_enterprise_schema()
    assert response.tables == {
        table: sorted(columns) for table, columns in BUSINESS_TABLE_COLUMNS.items()
    }


@pytest.mark.asyncio
async def test_definitions_tool_returns_shared_business_rules_without_benchmark_answers() -> None:
    adapter = EnterpriseDataToolService(lambda: RecordingSafeQueryService(_result()))  # type: ignore[arg-type]
    response = await adapter.get_business_definitions()
    rendered = " ".join(response.definitions.values())
    assert {
        "effective_sales",
        "current_inventory",
        "on_time_delivery_rate",
        "natural_month",
    } <= set(response.definitions)
    assert "P100" not in rendered
    assert "SELECT" not in rendered


@pytest.mark.asyncio
async def test_execute_tool_delegates_to_safe_query_service_and_projects_safe_fields() -> None:
    service = RecordingSafeQueryService(_result())
    adapter = EnterpriseDataToolService(lambda: service)  # type: ignore[arg-type]
    response = await adapter.execute_safe_query(
        ExecuteSafeQueryInput(sql="SELECT product_id FROM products")
    )
    assert service.sql == ["SELECT product_id FROM products"]
    assert response.model_dump() == {
        "columns": ["product_id"],
        "rows": [["P100"]],
        "row_count": 1,
        "truncated": False,
        "normalized_sql": "SELECT product_id FROM products LIMIT 200",
        "accessed_tables": ["products"],
        "elapsed_ms": 1.5,
        "error_code": None,
    }


@pytest.mark.asyncio
async def test_execute_tool_preserves_controlled_guard_error_without_sensitive_audit_fields() -> (
    None
):
    service = RecordingSafeQueryService(_result(error_code="write_statement_not_allowed"))
    adapter = EnterpriseDataToolService(lambda: service)  # type: ignore[arg-type]
    response = await adapter.execute_safe_query(ExecuteSafeQueryInput(sql="DELETE FROM products"))
    serialized = response.model_dump_json().lower()
    assert service.sql == ["DELETE FROM products"]
    assert response.error_code == "write_statement_not_allowed"
    assert "password" not in serialized
    assert "mysql+pymysql" not in serialized
    assert "traceback" not in serialized


@pytest.mark.asyncio
async def test_execute_tool_maps_startup_configuration_failure_to_a_safe_error() -> None:
    def unavailable_service() -> RecordingSafeQueryService:
        raise ConfigurationError("database readonly password is required")

    adapter = EnterpriseDataToolService(unavailable_service)  # type: ignore[arg-type]
    response = await adapter.execute_safe_query(
        ExecuteSafeQueryInput(sql="SELECT product_id FROM products")
    )
    assert response.error_code == "database_unavailable"
    assert "password" not in response.model_dump_json().lower()


@pytest.mark.parametrize("sql", ["", "x" * 20_001])
def test_execute_input_rejects_invalid_sql_parameters(sql: str) -> None:
    with pytest.raises(ValueError):
        ExecuteSafeQueryInput(sql=sql)


class PreflightSafeQueryService(RecordingSafeQueryService):
    def __init__(self, result: DatabasePreflightResult) -> None:
        self.preflight_result = result
        self.closed = False

    async def preflight_database(self) -> DatabasePreflightResult:
        return self.preflight_result

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_internal_database_preflight_returns_only_content_safe_fields() -> None:
    service = PreflightSafeQueryService(
        DatabasePreflightResult(True, True, True, "passed", "none", None)
    )
    adapter = EnterpriseDataToolService(lambda: service)  # type: ignore[arg-type]

    result = await adapter.preflight_database()
    await adapter.aclose()

    assert result == {
        "component": "mysql",
        "phase": "mcp_server_preflight",
        "status": "passed",
        "error_code": "none",
        "connection_established": True,
        "readonly_ready": True,
        "sql_guard_ready": True,
        "shutdown": "passed",
        "exception_type": None,
    }
    assert service.closed is True
    assert "select" not in repr(result).lower()
    assert "password" not in repr(result).lower()
