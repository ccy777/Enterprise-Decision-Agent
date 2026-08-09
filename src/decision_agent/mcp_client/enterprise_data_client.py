"""Official stdio MCP client for the independent Enterprise Data server."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import threading
from collections.abc import AsyncIterator, Mapping
from contextlib import AbstractAsyncContextManager, suppress
from enum import Enum
from pathlib import Path
from typing import Any

from anyio import BrokenResourceError, ClosedResourceError, EndOfStream
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import get_default_environment, stdio_client
from mcp.shared.exceptions import McpError
from pydantic import SecretStr, ValidationError

from decision_agent.config import Settings
from decision_agent.mcp_client.contracts import (
    BusinessDefinitions,
    EnterpriseSchema,
    MCPQueryResult,
)
from decision_agent.mcp_client.errors import EnterpriseDataMCPError

_REQUIRED_TOOLS = frozenset(
    {"get_enterprise_schema", "get_business_definitions", "execute_safe_query"}
)
_TRANSPORT_ERRORS = (
    McpError,
    OSError,
    TimeoutError,
    BrokenResourceError,
    ClosedResourceError,
    EndOfStream,
)
_CLOSE_ERRORS = (*_TRANSPORT_ERRORS, RuntimeError)
_MCP_CHILD_SETTINGS_FIELDS = (
    "app_name",
    "environment",
    "db_host",
    "db_port",
    "db_database",
    "db_readonly_username",
    "db_readonly_password",
    "db_connect_timeout_seconds",
    "db_query_timeout_seconds",
    "db_max_rows",
    "db_max_result_cells",
)
_DATABASE_PREFLIGHT_ENVIRONMENT_KEY = "DECISION_AGENT_MCP_DATABASE_PREFLIGHT"
_DATABASE_PREFLIGHT_RESULT_FILE_KEY = "DECISION_AGENT_MCP_DATABASE_PREFLIGHT_RESULT_FILE"
_CHILD_STDERR_CAPTURE_LIMIT = 8_192
_PREFLIGHT_RESULT_FIELDS = frozenset(
    {
        "status",
        "error_code",
        "connection_established",
        "readonly_ready",
        "sql_guard_ready",
        "shutdown",
    }
)


class _ChildStderrCapture:
    """Bounded in-memory stderr summary that continuously drains the child pipe."""

    def __init__(self, *, max_bytes: int = _CHILD_STDERR_CAPTURE_LIMIT) -> None:
        self._max_bytes = max_bytes
        self._read_handle, write_handle = os.pipe()
        self._write_stream = os.fdopen(write_handle, "wb", buffering=0)
        self._captured = bytearray()
        self._bytes = 0
        self._truncated = False
        self._reader = threading.Thread(target=self._drain, daemon=True)
        self._reader.start()

    @property
    def errlog(self) -> Any:
        """Return the subprocess-compatible stderr destination."""
        return self._write_stream

    def close(self) -> dict[str, object]:
        """Close both pipe ends and expose only fixed-field safety metadata."""
        if not self._write_stream.closed:
            self._write_stream.close()
        self._reader.join(timeout=1.0)
        text = self._captured.decode("utf-8", errors="replace").lower()
        return {
            "child_stderr_captured": True,
            "child_stderr_bytes": self._bytes,
            "child_stderr_truncated": self._truncated,
            "child_warning_detected": "warning" in text,
            "child_error_class": _child_error_class(text),
            "sensitive_content_detected": _sensitive_content_detected(text),
        }

    def _drain(self) -> None:
        with os.fdopen(self._read_handle, "rb", buffering=0) as stream:
            while chunk := stream.read(4_096):
                self._bytes += len(chunk)
                remaining = self._max_bytes - len(self._captured)
                if remaining > 0:
                    self._captured.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    self._truncated = True


def _child_error_class(content: str) -> str | None:
    for marker, code in (
        ("mcp_database_preflight_mysql_authentication_failed", "mysql_authentication_failed"),
        ("mcp_database_preflight_mysql_connection_failed", "mysql_connection_failed"),
        ("mcp_database_preflight_mysql_timeout", "mysql_timeout"),
        ("mcp_database_preflight_mysql_health_query_failed", "mysql_health_query_failed"),
        ("mcp_database_preflight_mysql_shutdown_failed", "mysql_shutdown_failed"),
        ("mcp_database_preflight_failed", "mysql_preflight_failed"),
    ):
        if marker in content:
            return code
    if "traceback" in content or "error" in content:
        return "child_reported_error"
    return None


def _sensitive_content_detected(content: str) -> bool:
    return any(marker in content for marker in ("password", "api_key", "authorization"))


class EnterpriseDataMCPClient:
    """One initialized stdio session with cached discovery, schema, and business definitions."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 10.0,
        server_environment: Mapping[str, str] | None = None,
        database_preflight: bool = False,
        database_preflight_result_file: Path | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._timeout_seconds = timeout_seconds
        self._server_environment = None if server_environment is None else dict(server_environment)
        self._database_preflight = database_preflight
        self._database_preflight_result_file = database_preflight_result_file
        self._stdio_context: AbstractAsyncContextManager[tuple[AsyncIterator[Any], Any]] | None = (
            None
        )
        self._session_context: AbstractAsyncContextManager[ClientSession] | None = None
        self._session: ClientSession | None = None
        self._schema: EnterpriseSchema | None = None
        self._definitions: BusinessDefinitions | None = None
        self._active = False
        self._database_preflight_passed = False
        self._database_preflight_error_code = "mysql_preflight_failed"
        self._database_preflight_status = "not_started"
        self._stderr_summary: dict[str, object] | None = None
        self._stderr_capture: _ChildStderrCapture | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> EnterpriseDataMCPClient:
        """Capture the MCP child configuration without starting external I/O."""
        return cls(
            timeout_seconds=settings.mcp_timeout_seconds,
            server_environment=_capture_mcp_child_environment(settings),
        )

    @classmethod
    def for_database_preflight(cls, settings: Settings) -> EnterpriseDataMCPClient:
        """Create one opt-in client whose child validates its own database transport."""
        environment = _capture_mcp_child_environment(settings)
        environment[_DATABASE_PREFLIGHT_ENVIRONMENT_KEY] = "true"
        descriptor, result_name = tempfile.mkstemp(
            prefix="decision-agent-mcp-preflight-", suffix=".json"
        )
        os.close(descriptor)
        result_file = Path(result_name)
        # The Server publishes with os.replace().  Leaving mkstemp's empty
        # destination in place makes that replacement unreliable on Windows.
        result_file.unlink()
        environment[_DATABASE_PREFLIGHT_RESULT_FILE_KEY] = str(result_file)
        return cls(
            timeout_seconds=settings.mcp_timeout_seconds,
            server_environment=environment,
            database_preflight=True,
            database_preflight_result_file=result_file,
        )

    @property
    def database_preflight_passed(self) -> bool:
        """Expose only whether the requested Server-side probe completed."""
        return self._database_preflight_passed

    @property
    def database_preflight_error_code(self) -> str:
        """Return a fixed preflight category without exposing child stderr."""
        return self._database_preflight_error_code

    @property
    def database_preflight_status(self) -> str:
        """Expose only the fixed MCP/MySQL preflight state."""
        return self._database_preflight_status

    @property
    def child_stderr_summary(self) -> dict[str, object]:
        """Expose bounded metadata without returning any child stderr content."""
        return dict(self._stderr_summary or _empty_stderr_summary())

    async def __aenter__(self) -> EnterpriseDataMCPClient:
        if self._active:
            raise EnterpriseDataMCPError("mcp_client_already_active")
        self._active = True
        self._stderr_capture = _ChildStderrCapture()
        try:
            parameters = StdioServerParameters(
                command=sys.executable,
                args=["-m", "decision_agent.mcp_server"],
                env=(None if self._server_environment is None else dict(self._server_environment)),
                cwd=Path(__file__).resolve().parents[3],
            )
            self._stdio_context = stdio_client(parameters, errlog=self._stderr_capture.errlog)
            reader, writer = await asyncio.wait_for(
                self._stdio_context.__aenter__(), timeout=self._timeout_seconds
            )
        except _TRANSPORT_ERRORS:
            await self._abort_enter()
            raise EnterpriseDataMCPError(
                await self._enter_error_code("mcp_server_unavailable")
            ) from None
        except BaseException:
            await self._abort_enter()
            raise
        try:
            self._session_context = ClientSession(reader, writer)
            self._session = await asyncio.wait_for(
                self._session_context.__aenter__(), timeout=self._timeout_seconds
            )
            await asyncio.wait_for(self._session.initialize(), timeout=self._timeout_seconds)
            tools = await asyncio.wait_for(
                self._session.list_tools(), timeout=self._timeout_seconds
            )
        except _TRANSPORT_ERRORS:
            await self._abort_enter()
            raise EnterpriseDataMCPError(
                await self._enter_error_code("mcp_initialization_failed")
            ) from None
        except BaseException:
            await self._abort_enter()
            raise
        try:
            available_tools = {tool.name for tool in tools.tools}
        except (AttributeError, TypeError):
            await self._abort_enter()
            raise EnterpriseDataMCPError("mcp_initialization_failed") from None
        if not available_tools >= _REQUIRED_TOOLS:
            await self._abort_enter()
            raise EnterpriseDataMCPError("mcp_required_tool_missing")
        self._database_preflight_passed = self._database_preflight
        self._database_preflight_error_code = "none"
        self._database_preflight_status = "mysql_preflight_passed"
        return self

    async def _enter_error_code(self, fallback: str) -> str:
        if not self._database_preflight:
            return fallback
        await self._read_database_preflight_result()
        error_class = self.child_stderr_summary["child_error_class"]
        if isinstance(error_class, str) and error_class.startswith("mysql_"):
            self._database_preflight_error_code = error_class
        return self._database_preflight_error_code

    async def _read_database_preflight_result(self) -> None:
        result_file = self._database_preflight_result_file
        if result_file is None:
            return
        payload: object | None = None
        for _ in range(20):
            try:
                if result_file.stat().st_size:
                    payload = json.loads(result_file.read_text(encoding="utf-8"))
                    break
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                pass
            await asyncio.sleep(0.05)
        self._remove_database_preflight_result()
        if payload is None:
            self._database_preflight_status = "mysql_preflight_protocol_failed"
            return
        if not isinstance(payload, dict) or set(payload) != _PREFLIGHT_RESULT_FIELDS:
            self._database_preflight_status = "mysql_preflight_protocol_failed"
            return
        error_code = payload.get("error_code")
        if not isinstance(error_code, str):
            self._database_preflight_status = "mysql_preflight_protocol_failed"
            return
        self._database_preflight_status = error_code
        self._database_preflight_error_code = error_code

    def _remove_database_preflight_result(self) -> None:
        if self._database_preflight_result_file is not None:
            result_file = self._database_preflight_result_file
            for path in (result_file, result_file.with_suffix(".stage.json")):
                with suppress(OSError):
                    path.unlink(missing_ok=True)

    async def _abort_enter(self) -> None:
        try:
            await self._close_contexts()
        finally:
            self._active = False

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        try:
            await self._close_contexts()
        finally:
            self._active = False

    async def get_enterprise_schema(self) -> EnterpriseSchema:
        if self._schema is None:
            try:
                self._schema = EnterpriseSchema(
                    tables=await self._call_payload("get_enterprise_schema", {})
                )
            except ValidationError:
                raise EnterpriseDataMCPError("mcp_response_invalid") from None
        return self._schema

    async def get_business_definitions(self) -> BusinessDefinitions:
        if self._definitions is None:
            try:
                self._definitions = BusinessDefinitions(
                    definitions=await self._call_payload("get_business_definitions", {})
                )
            except ValidationError:
                raise EnterpriseDataMCPError("mcp_response_invalid") from None
        return self._definitions

    async def execute_safe_query(self, sql: str) -> MCPQueryResult:
        try:
            return MCPQueryResult.model_validate(
                await self._call_payload("execute_safe_query", {"sql": sql})
            )
        except ValidationError:
            raise EnterpriseDataMCPError("mcp_response_invalid") from None

    async def _call_payload(self, tool_name: str, arguments: dict[str, object]) -> dict[str, Any]:
        if self._session is None:
            raise EnterpriseDataMCPError("mcp_initialization_failed")
        try:
            result = await asyncio.wait_for(
                self._session.call_tool(tool_name, arguments), timeout=self._timeout_seconds
            )
        except TimeoutError:
            raise EnterpriseDataMCPError("mcp_tool_timeout") from None
        except _TRANSPORT_ERRORS:
            raise EnterpriseDataMCPError("mcp_tool_call_failed") from None
        if result.isError:
            raise EnterpriseDataMCPError("mcp_tool_call_failed")
        try:
            structured = result.structuredContent
            if isinstance(structured, dict):
                return structured
            if len(result.content) != 1:
                raise ValueError("tool response content must contain one item")
            payload = json.loads(result.content[0].text)
            if not isinstance(payload, dict):
                raise ValueError("tool response must be an object")
            return payload
        except (AttributeError, IndexError, TypeError, ValueError, json.JSONDecodeError):
            raise EnterpriseDataMCPError("mcp_response_invalid") from None

    async def _close_contexts(self) -> None:
        session_context, stdio_context = self._session_context, self._stdio_context
        self._session = None
        self._session_context = None
        self._stdio_context = None
        self._schema = None
        self._definitions = None
        if session_context is not None:
            await self._close_context(session_context)
        if stdio_context is not None:
            await self._close_context(stdio_context)
        if self._stderr_capture is not None:
            self._stderr_summary = self._stderr_capture.close()
            self._stderr_capture = None
        if self._database_preflight_passed:
            self._remove_database_preflight_result()

    async def _close_context(self, context: AbstractAsyncContextManager[Any]) -> None:
        try:
            # MCP/AnyIO context managers bind their cancel scopes to the task that entered them.
            # Running __aexit__ inside asyncio.wait_for would create another task on Windows.
            async with asyncio.timeout(self._timeout_seconds):
                await context.__aexit__(None, None, None)
        except _CLOSE_ERRORS:
            return


def _capture_mcp_child_environment(settings: Settings) -> dict[str, str]:
    """Overlay a fixed Settings whitelist on the MCP SDK's safe child environment."""
    environment = dict(get_default_environment())
    source_root = Path(__file__).resolve().parents[2]
    inherited_pythonpath = os.environ.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(source_root), inherited_pythonpath) if value
    )
    for field_name in _MCP_CHILD_SETTINGS_FIELDS:
        environment[f"DECISION_AGENT_{field_name.upper()}"] = _serialize_setting_value(
            getattr(settings, field_name)
        )
    return environment


def _serialize_setting_value(value: object) -> str:
    """Serialize one whitelisted Settings value for the stdio subprocess boundary."""
    if value is None:
        return ""
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        return str(value)
    raise TypeError("unsupported MCP child setting type")


def _empty_stderr_summary() -> dict[str, object]:
    return {
        "child_stderr_captured": False,
        "child_stderr_bytes": 0,
        "child_stderr_truncated": False,
        "child_warning_detected": False,
        "child_error_class": None,
        "sensitive_content_detected": False,
    }
