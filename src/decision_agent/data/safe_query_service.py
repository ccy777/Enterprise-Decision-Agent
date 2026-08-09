"""Service boundary combining AST authorization, resource limits, and safe auditing."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from time import monotonic
from typing import Any

from decision_agent.config import Settings
from decision_agent.data.executor import (
    DatabaseAuthenticationFailed,
    DatabaseHealthCheckFailed,
    DatabaseUnavailable,
    QueryExecutionFailed,
    QueryExecutor,
    QueryTimeout,
    SQLAlchemyQueryExecutor,
)
from decision_agent.data.models import QueryAudit, SafeQueryRequest, SafeQueryResult
from decision_agent.data.sql_guard import SQLGuard


@dataclass(frozen=True, slots=True)
class DatabasePreflightResult:
    """Content-safe result for an opt-in database transport health probe."""

    connection_established: bool
    readonly_ready: bool
    sql_guard_ready: bool
    shutdown: str
    error_code: str
    exception_type: str | None


class SafeQueryService:
    """Execute only AST-approved SELECT queries through an injected read-only executor."""

    def __init__(
        self,
        *,
        guard: SQLGuard,
        executor: QueryExecutor,
        query_timeout_seconds: float,
        max_rows: int,
        max_result_cells: int,
        application_timeout_seconds: float | None = None,
    ) -> None:
        if query_timeout_seconds <= 0 or max_rows <= 0 or max_result_cells <= 0:
            raise ValueError("SafeQueryService resource limits must be positive")
        self._guard = guard
        self._executor = executor
        self._query_timeout_seconds = query_timeout_seconds
        self._application_timeout_seconds = application_timeout_seconds or (
            query_timeout_seconds + 1.0
        )
        if self._application_timeout_seconds <= query_timeout_seconds:
            raise ValueError("application timeout must exceed database query timeout")
        self._max_rows = max_rows
        self._max_result_cells = max_result_cells

    @classmethod
    def from_settings(cls, settings: Settings) -> SafeQueryService:
        """Create one runtime service with the application read-only account."""
        return cls(
            guard=SQLGuard(max_rows=settings.db_max_rows),
            executor=SQLAlchemyQueryExecutor.from_settings(settings),
            query_timeout_seconds=settings.db_query_timeout_seconds,
            max_rows=settings.db_max_rows,
            max_result_cells=settings.db_max_result_cells,
        )

    async def execute(self, request: SafeQueryRequest) -> SafeQueryResult:
        """Validate first, then execute exactly one bounded query without leaking driver details."""
        started = monotonic()
        decision = self._guard.validate(request.sql)
        if not decision.allowed:
            return self._failure(
                request=request,
                started=started,
                code=decision.rejection_code or "safe_query_execution_failed",
            )
        try:
            execution = await asyncio.wait_for(
                asyncio.to_thread(
                    self._executor.execute,
                    normalized_sql=decision.normalized_sql or "",
                    timeout_ms=round(self._query_timeout_seconds * 1000),
                ),
                timeout=self._application_timeout_seconds,
            )
        except TimeoutError:
            self._discard_connections_after_timeout()
            return self._failure(
                request=request, started=started, code="query_timeout", decision=decision
            )
        except QueryTimeout:
            return self._failure(
                request=request, started=started, code="query_timeout", decision=decision
            )
        except DatabaseUnavailable:
            return self._failure(
                request=request, started=started, code="database_unavailable", decision=decision
            )
        except QueryExecutionFailed:
            return self._failure(
                request=request,
                started=started,
                code="safe_query_execution_failed",
                decision=decision,
            )

        row_count = len(execution.rows)
        elapsed_ms = _elapsed_ms(started)
        if row_count > self._max_rows:
            return self._failure(
                request=request,
                started=started,
                code="result_too_large",
                decision=decision,
                row_count=row_count,
                truncated=True,
            )
        if row_count * len(execution.columns) > self._max_result_cells:
            return self._failure(
                request=request,
                started=started,
                code="result_too_large",
                decision=decision,
                row_count=row_count,
            )
        rows = [[_public_value(value) for value in row] for row in execution.rows]
        audit = QueryAudit(
            request_id=request.request_id,
            normalized_sql=decision.normalized_sql,
            allowed=True,
            accessed_tables=decision.accessed_tables,
            elapsed_ms=elapsed_ms,
            row_count=row_count,
            truncated=False,
        )
        return SafeQueryResult(
            columns=execution.columns,
            rows=rows,
            row_count=row_count,
            truncated=False,
            elapsed_ms=elapsed_ms,
            accessed_tables=decision.accessed_tables,
            audit=audit,
        )

    def _failure(
        self,
        *,
        request: SafeQueryRequest,
        started: float,
        code: str,
        decision: Any | None = None,
        row_count: int = 0,
        truncated: bool = False,
    ) -> SafeQueryResult:
        accessed_tables = decision.accessed_tables if decision is not None else []
        normalized_sql = decision.normalized_sql if decision is not None else None
        elapsed_ms = _elapsed_ms(started)
        audit = QueryAudit(
            request_id=request.request_id,
            normalized_sql=normalized_sql,
            allowed=False,
            rejection_code=code,
            accessed_tables=accessed_tables,
            elapsed_ms=elapsed_ms,
            row_count=row_count,
            truncated=truncated,
        )
        return SafeQueryResult(
            row_count=0,
            truncated=truncated,
            elapsed_ms=elapsed_ms,
            accessed_tables=accessed_tables,
            audit=audit,
            error_code=code,
        )

    def _discard_connections_after_timeout(self) -> None:
        """Ask capable executors to evict pooled connections after a wait timeout."""
        discard = getattr(self._executor, "discard_after_timeout", None)
        if callable(discard):
            discard()

    async def preflight_database(self) -> DatabasePreflightResult:
        """Exercise the existing read-only executor transport once without business SQL."""
        probe = getattr(self._executor, "preflight_connection", None)
        close = getattr(self._executor, "close", None)
        if not callable(probe):
            return DatabasePreflightResult(
                False, False, True, "not_required", "mysql_health_query_failed", None
            )

        error_code = "none"
        exception_type: str | None = None
        connected = False
        try:
            await asyncio.to_thread(probe)
            connected = True
        except DatabaseAuthenticationFailed as exc:
            error_code = "mysql_authentication_failed"
            exception_type = type(exc).__name__
        except DatabaseUnavailable as exc:
            error_code = "mysql_connection_failed"
            exception_type = type(exc).__name__
        except QueryTimeout as exc:
            error_code = "mysql_timeout"
            exception_type = type(exc).__name__
        except DatabaseHealthCheckFailed as exc:
            error_code = "mysql_health_query_failed"
            exception_type = type(exc).__name__
        finally:
            if callable(close):
                try:
                    await asyncio.to_thread(close)
                except (OSError, RuntimeError) as exc:
                    error_code = "mysql_shutdown_failed"
                    exception_type = type(exc).__name__

        return DatabasePreflightResult(
            connection_established=connected,
            readonly_ready=connected and error_code == "none",
            sql_guard_ready=True,
            shutdown="passed" if error_code != "mysql_shutdown_failed" else "failed",
            error_code=error_code,
            exception_type=exception_type,
        )

    async def aclose(self) -> None:
        """Release the optional executor resource owned by this service."""
        close = getattr(self._executor, "close", None)
        if callable(close):
            await asyncio.to_thread(close)


def _elapsed_ms(started: float) -> float:
    return round((monotonic() - started) * 1000, 3)


def _public_value(value: Any) -> Any:
    """Make values JSON-safe while keeping monetary values exact rather than float-converted."""
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value
