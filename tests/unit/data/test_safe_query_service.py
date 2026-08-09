from __future__ import annotations

from decimal import Decimal
from time import sleep

import pytest

from decision_agent.data.executor import (
    DatabaseAuthenticationFailed,
    DatabaseHealthCheckFailed,
    DatabaseUnavailable,
    QueryExecutionFailed,
    QueryTimeout,
)
from decision_agent.data.models import QueryExecution, SafeQueryRequest
from decision_agent.data.safe_query_service import SafeQueryService
from decision_agent.data.sql_guard import SQLGuard

pytestmark = pytest.mark.offline_integration


class RecordingExecutor:
    def __init__(self, result: QueryExecution | Exception) -> None:
        self.result = result
        self.calls: list[tuple[str, int]] = []

    def execute(self, *, normalized_sql: str, timeout_ms: int) -> QueryExecution:
        self.calls.append((normalized_sql, timeout_ms))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def make_service(
    executor: RecordingExecutor,
    *,
    max_rows: int = 5,
    max_cells: int = 20,
    query_timeout_seconds: float = 1.5,
    application_timeout_seconds: float | None = None,
) -> SafeQueryService:
    return SafeQueryService(
        guard=SQLGuard(max_rows=max_rows),
        executor=executor,
        query_timeout_seconds=query_timeout_seconds,
        max_rows=max_rows,
        max_result_cells=max_cells,
        application_timeout_seconds=application_timeout_seconds,
    )


@pytest.mark.asyncio
async def test_service_executes_only_guarded_sql_and_preserves_exact_decimal() -> None:
    executor = RecordingExecutor(QueryExecution(columns=["amount"], rows=[(Decimal("12.50"),)]))
    result = await make_service(executor).execute(
        SafeQueryRequest(sql="SELECT safety_stock FROM products")
    )
    assert result.error_code is None
    assert result.rows == [["12.50"]]
    assert result.audit.allowed is True
    assert result.audit.normalized_sql == "SELECT safety_stock FROM products LIMIT 5"
    assert executor.calls == [("SELECT safety_stock FROM products LIMIT 5", 1500)]


@pytest.mark.asyncio
async def test_guard_rejection_does_not_access_database() -> None:
    executor = RecordingExecutor(QueryExecution(columns=[], rows=[]))
    result = await make_service(executor).execute(SafeQueryRequest(sql="DELETE FROM products"))
    assert result.error_code == "write_statement_not_allowed"
    assert result.audit.allowed is False
    assert executor.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "code"),
    [
        (QueryTimeout(), "query_timeout"),
        (DatabaseUnavailable(), "database_unavailable"),
        (QueryExecutionFailed(), "safe_query_execution_failed"),
    ],
)
async def test_service_maps_executor_failures_without_driver_details(
    failure: Exception, code: str
) -> None:
    result = await make_service(RecordingExecutor(failure)).execute(
        SafeQueryRequest(sql="SELECT product_name FROM products")
    )
    assert result.error_code == code
    serialized = result.model_dump_json()
    assert "password" not in serialized.lower()
    assert "mysql+pymysql" not in serialized.lower()


@pytest.mark.asyncio
async def test_service_rejects_too_many_rows_from_an_untrusted_executor() -> None:
    executor = RecordingExecutor(QueryExecution(columns=["product_id"], rows=[("P1",)] * 6))
    result = await make_service(executor).execute(
        SafeQueryRequest(sql="SELECT product_id FROM products")
    )
    assert result.error_code == "result_too_large"
    assert result.truncated is True
    assert result.audit.row_count == 6


@pytest.mark.asyncio
async def test_exactly_maximum_rows_are_not_claimed_as_truncated() -> None:
    executor = RecordingExecutor(QueryExecution(columns=["product_id"], rows=[("P1",)] * 5))
    result = await make_service(executor).execute(
        SafeQueryRequest(sql="SELECT product_id FROM products")
    )
    assert result.error_code is None
    assert result.truncated is False
    assert result.audit.truncated is False


@pytest.mark.asyncio
async def test_service_rejects_oversized_cell_result() -> None:
    executor = RecordingExecutor(
        QueryExecution(columns=["a", "b", "c"], rows=[("x", "y", "z")] * 2)
    )
    result = await make_service(executor, max_cells=5).execute(
        SafeQueryRequest(sql="SELECT product_id FROM products")
    )
    assert result.error_code == "result_too_large"
    assert result.audit.truncated is False


class SlowExecutor(RecordingExecutor):
    def __init__(self) -> None:
        super().__init__(QueryExecution(columns=[], rows=[]))
        self.discarded = False

    def execute(self, *, normalized_sql: str, timeout_ms: int) -> QueryExecution:
        self.calls.append((normalized_sql, timeout_ms))
        sleep(0.05)
        return QueryExecution(columns=[], rows=[])

    def discard_after_timeout(self) -> None:
        self.discarded = True


@pytest.mark.asyncio
async def test_application_timeout_maps_to_safe_error_and_discards_connections() -> None:
    executor = SlowExecutor()
    result = await make_service(
        executor,
        query_timeout_seconds=0.01,
        application_timeout_seconds=0.02,
    ).execute(SafeQueryRequest(sql="SELECT product_name FROM products"))
    assert result.error_code == "query_timeout"
    assert result.audit.rejection_code == "query_timeout"
    assert executor.calls == [("SELECT product_name FROM products LIMIT 5", 10)]
    assert executor.discarded is True


class PreflightExecutor:
    def __init__(
        self, outcome: Exception | None = None, close_error: Exception | None = None
    ) -> None:
        self.outcome = outcome
        self.close_error = close_error
        self.probe_calls = 0
        self.close_calls = 0

    def execute(self, *, normalized_sql: str, timeout_ms: int) -> QueryExecution:
        raise AssertionError("preflight must not execute application SQL")

    def preflight_connection(self) -> None:
        self.probe_calls += 1
        if self.outcome is not None:
            raise self.outcome

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "code"),
    [
        (None, "none"),
        (DatabaseAuthenticationFailed(), "mysql_authentication_failed"),
        (DatabaseUnavailable(), "mysql_connection_failed"),
        (QueryTimeout(), "mysql_timeout"),
        (DatabaseHealthCheckFailed(), "mysql_health_query_failed"),
    ],
)
async def test_database_preflight_maps_safe_results_and_always_closes(
    outcome: Exception | None, code: str
) -> None:
    executor = PreflightExecutor(outcome)
    result = await make_service(executor).preflight_database()  # type: ignore[arg-type]

    assert result.error_code == code
    assert executor.probe_calls == executor.close_calls == 1
    assert "SELECT" not in repr(result)
    assert "password" not in repr(result).lower()


@pytest.mark.asyncio
async def test_database_preflight_reports_shutdown_failure_without_driver_details() -> None:
    executor = PreflightExecutor(close_error=RuntimeError("database-password-marker"))
    result = await make_service(executor).preflight_database()  # type: ignore[arg-type]

    assert result.error_code == "mysql_shutdown_failed"
    assert result.shutdown == "failed"
    assert "database-password-marker" not in repr(result)
