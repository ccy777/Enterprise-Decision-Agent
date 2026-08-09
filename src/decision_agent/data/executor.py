"""Lazy SQLAlchemy implementation of the injectable read-only execution boundary."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Protocol, runtime_checkable

from sqlalchemy import URL, Engine, create_engine, text
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from decision_agent.config import Settings
from decision_agent.data.models import QueryExecution
from decision_agent.exceptions import ConfigurationError


class QueryExecutorError(Exception):
    """Base execution failure that intentionally has no connection details."""


class QueryTimeout(QueryExecutorError):
    """Raised when MySQL interrupts a query because of the configured timeout."""


class DatabaseUnavailable(QueryExecutorError):
    """Raised when a connection cannot be created or maintained."""


class QueryExecutionFailed(QueryExecutorError):
    """Raised for non-sensitive database execution failures."""


class DatabaseAuthenticationFailed(DatabaseUnavailable):
    """Raised when the configured read-only account cannot authenticate."""


class DatabaseHealthCheckFailed(QueryExecutorError):
    """Raised when the driver cannot complete its non-business health probe."""


@runtime_checkable
class QueryExecutor(Protocol):
    """Synchronous external I/O boundary used through ``asyncio.to_thread``."""

    def execute(self, *, normalized_sql: str, timeout_ms: int) -> QueryExecution:
        """Run a prevalidated SELECT query with a MySQL execution timeout."""


@dataclass
class SQLAlchemyQueryExecutor:
    """Run validated queries with a runtime read-only MySQL account only."""

    engine: Engine

    @classmethod
    def from_settings(cls, settings: Settings) -> SQLAlchemyQueryExecutor:
        """Construct a lazy Engine from non-empty read-only connection settings."""
        if settings.db_readonly_password is None:
            raise ConfigurationError("database readonly password is required")
        url = URL.create(
            "mysql+pymysql",
            username=settings.db_readonly_username,
            password=settings.db_readonly_password.get_secret_value(),
            host=settings.db_host,
            port=settings.db_port,
            database=settings.db_database,
            query={"charset": "utf8mb4"},
        )
        return cls(
            engine=create_engine(
                url,
                pool_pre_ping=True,
                connect_args={"connect_timeout": settings.db_connect_timeout_seconds},
            )
        )

    def execute(self, *, normalized_sql: str, timeout_ms: int) -> QueryExecution:
        """Execute only Guard-normalized SQL and map driver details to safe exceptions."""
        started = monotonic()
        try:
            with self.engine.connect() as connection:
                connection.execute(
                    text("SET SESSION MAX_EXECUTION_TIME = :timeout_ms"), {"timeout_ms": timeout_ms}
                )
                result = connection.execute(text(normalized_sql))
                columns = list(result.keys())
                rows = [tuple(row) for row in result.fetchall()]
        except OperationalError as exc:
            message = str(exc).lower()
            if "max_execution_time" in message or "query execution was interrupted" in message:
                raise QueryTimeout() from exc
            if "connect" in message or "connection" in message:
                raise DatabaseUnavailable() from exc
            raise QueryExecutionFailed() from exc
        except SQLAlchemyError as exc:
            raise QueryExecutionFailed() from exc
        return QueryExecution(columns=columns, rows=rows, elapsed_started_at=started)

    def discard_after_timeout(self) -> None:
        """Discard pooled connections after an application-level timeout fallback."""
        self.engine.dispose()

    def preflight_connection(self) -> None:
        """Verify the configured read-only transport without issuing application SQL."""
        try:
            with self.engine.connect() as connection:
                driver_connection = connection.connection.driver_connection
                ping = getattr(driver_connection, "ping", None)
                if not callable(ping):
                    raise DatabaseHealthCheckFailed()
                ping(reconnect=False)
        except DatabaseHealthCheckFailed:
            raise
        except TimeoutError as exc:
            raise QueryTimeout() from exc
        except OperationalError as exc:
            if _is_authentication_error(exc):
                raise DatabaseAuthenticationFailed() from exc
            raise DatabaseUnavailable() from exc
        except SQLAlchemyError as exc:
            raise DatabaseHealthCheckFailed() from exc

    def close(self) -> None:
        """Release this executor's engine and all pooled driver connections."""
        self.engine.dispose()


def _is_authentication_error(error: OperationalError) -> bool:
    """Classify the standard MySQL authentication code without retaining driver text."""
    original = getattr(error, "orig", None)
    arguments = getattr(original, "args", ())
    return bool(arguments and arguments[0] == 1045)
