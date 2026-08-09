"""Deterministic lifecycle and safety tests for the official Enterprise Data MCP client."""

from __future__ import annotations

import asyncio
import hmac
import sys
import traceback
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
from anyio import EndOfStream

from decision_agent.config import Environment, Settings
from decision_agent.mcp_client.enterprise_data_client import (
    EnterpriseDataMCPClient,
    _child_error_class,
    _ChildStderrCapture,
)
from decision_agent.mcp_client.errors import EnterpriseDataMCPError

pytestmark = pytest.mark.offline_integration


@pytest.fixture(autouse=True)
def isolate_decision_agent_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore every Settings-owned environment value after each test."""
    for field_name in Settings.model_fields:
        monkeypatch.delenv(f"DECISION_AGENT_{field_name.upper()}", raising=False)


@dataclass
class FakeTool:
    name: str


class FakeSession:
    def __init__(
        self, *, tools: set[str] | None = None, payloads: dict[str, object] | None = None
    ) -> None:
        self.tools = tools or {
            "get_enterprise_schema",
            "get_business_definitions",
            "execute_safe_query",
        }
        self.payloads = payloads or {
            "get_enterprise_schema": {"products": ["product_id"]},
            "get_business_definitions": {"natural_month": "half-open"},
            "execute_safe_query": {
                "columns": ["product_id"],
                "rows": [["P100"]],
                "row_count": 1,
                "truncated": False,
                "normalized_sql": "SELECT product_id FROM products LIMIT 200",
                "accessed_tables": ["products"],
                "elapsed_ms": 1.0,
                "error_code": None,
            },
        }
        self.initialize_calls = 0
        self.list_calls = 0
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def initialize(self) -> None:
        self.initialize_calls += 1

    async def list_tools(self) -> SimpleNamespace:
        self.list_calls += 1
        return SimpleNamespace(tools=[FakeTool(name) for name in self.tools])

    async def call_tool(self, name: str, arguments: dict[str, object]) -> SimpleNamespace:
        self.calls.append((name, arguments))
        payload = self.payloads[name]
        if isinstance(payload, Exception):
            raise payload
        return SimpleNamespace(isError=False, structuredContent=payload, content=[])


class FakeContext:
    def __init__(self, value: object | Exception) -> None:
        self.value = value
        self.closed = False
        self.enter_calls = 0

    async def __aenter__(self):  # type: ignore[no-untyped-def]
        self.enter_calls += 1
        if isinstance(self.value, Exception):
            raise self.value
        return self.value

    async def __aexit__(self, exc_type, exc, traceback):  # type: ignore[no-untyped-def]
        self.closed = True


def install_fake_transport(
    monkeypatch: pytest.MonkeyPatch,
    session: FakeSession | None = None,
    *,
    startup_error: Exception | None = None,
    captured_parameters: list[Any] | None = None,
) -> tuple[FakeSession, FakeContext, FakeContext | None]:
    fake_session = session or FakeSession()
    stdio_context = FakeContext(startup_error or (object(), object()))
    session_context: FakeContext | None = None

    def fake_stdio_client(parameters, errlog=None):  # type: ignore[no-untyped-def]
        if captured_parameters is not None:
            captured_parameters.append(parameters)
        return stdio_context

    def fake_client_session(reader, writer):  # type: ignore[no-untyped-def]
        nonlocal session_context
        session_context = FakeContext(fake_session)
        return session_context

    monkeypatch.setattr(
        "decision_agent.mcp_client.enterprise_data_client.stdio_client", fake_stdio_client
    )
    monkeypatch.setattr(
        "decision_agent.mcp_client.enterprise_data_client.ClientSession", fake_client_session
    )
    return fake_session, stdio_context, session_context


def make_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "app_name": "Parent Agent",
        "environment": Environment.PRODUCTION,
        "mcp_timeout_seconds": 7.25,
        "db_host": "db.internal",
        "db_port": 4406,
        "db_database": "operations_prod",
        "db_readonly_username": "readonly_service",
        "db_readonly_password": "database-password-marker",
        "db_connect_timeout_seconds": 9,
        "db_query_timeout_seconds": 4.5,
        "db_max_rows": 321,
        "db_max_result_cells": 4321,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def child_settings_from_environment(
    monkeypatch: pytest.MonkeyPatch, environment: dict[str, str]
) -> Settings:
    for field_name in Settings.model_fields:
        monkeypatch.delenv(f"DECISION_AGENT_{field_name.upper()}", raising=False)
    for name, value in environment.items():
        if name.startswith("DECISION_AGENT_"):
            monkeypatch.setenv(name, value)
    return Settings(_env_file=None)


def mcp_environment(parameters: Any) -> dict[str, str]:
    assert isinstance(parameters.env, dict)
    return {
        name: value for name, value in parameters.env.items() if name.startswith("DECISION_AGENT_")
    }


def test_database_preflight_reserves_an_absent_atomic_publish_target() -> None:
    client = EnterpriseDataMCPClient.for_database_preflight(make_settings())
    result_file = client._database_preflight_result_file

    assert result_file is not None
    assert result_file.parent.is_dir()
    assert not result_file.exists()


def assert_mcp_settings_equal(parent: Settings, child: Settings) -> None:
    assert child.app_name == parent.app_name
    assert child.environment is parent.environment
    assert child.db_host == parent.db_host
    assert child.db_port == parent.db_port
    assert child.db_database == parent.db_database
    assert child.db_readonly_username == parent.db_readonly_username
    assert child.db_readonly_password == parent.db_readonly_password
    assert child.db_connect_timeout_seconds == parent.db_connect_timeout_seconds
    assert child.db_query_timeout_seconds == parent.db_query_timeout_seconds
    assert child.db_max_rows == parent.db_max_rows
    assert child.db_max_result_cells == parent.db_max_result_cells


def test_from_settings_captures_without_starting_stdio_and_maps_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_stdio_client(parameters):  # type: ignore[no-untyped-def]
        raise AssertionError("from_settings must not start stdio")

    monkeypatch.setattr(
        "decision_agent.mcp_client.enterprise_data_client.stdio_client",
        fail_stdio_client,
    )
    client = EnterpriseDataMCPClient.from_settings(make_settings())

    assert client._timeout_seconds == 7.25


@pytest.mark.asyncio
async def test_from_settings_passes_fixed_command_and_complete_child_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DECISION_AGENT_DB_HOST", "ambient-host")
    monkeypatch.setenv("DECISION_AGENT_DB_PORT", "9999")
    monkeypatch.setenv("DECISION_AGENT_LLM_API_KEY", "ambient-llm-secret")
    parent = make_settings(
        llm_api_key="unrelated-llm-secret",
        llm_base_url="https://llm.example.invalid",
        llm_model_name="unrelated-model",
        milvus_token="unrelated-milvus-secret",
        memory_mode="redis",
        memory_redis_url="redis://cache.example.invalid:6379/0",
    )
    monkeypatch.setattr(
        "decision_agent.mcp_client.enterprise_data_client.get_default_environment",
        lambda: {"PATH": "safe-path", "SYSTEMROOT": "safe-system-root"},
    )
    captured: list[Any] = []
    install_fake_transport(monkeypatch, captured_parameters=captured)

    async with EnterpriseDataMCPClient.from_settings(parent):
        pass

    assert len(captured) == 1
    parameters = captured[0]
    assert parameters.command == sys.executable
    assert parameters.args == ["-m", "decision_agent.mcp_server"]
    assert parameters.env["PATH"] == "safe-path"
    assert parameters.env["SYSTEMROOT"] == "safe-system-root"
    environment = mcp_environment(parameters)
    assert set(environment) == {
        "DECISION_AGENT_APP_NAME",
        "DECISION_AGENT_ENVIRONMENT",
        "DECISION_AGENT_DB_HOST",
        "DECISION_AGENT_DB_PORT",
        "DECISION_AGENT_DB_DATABASE",
        "DECISION_AGENT_DB_READONLY_USERNAME",
        "DECISION_AGENT_DB_READONLY_PASSWORD",
        "DECISION_AGENT_DB_CONNECT_TIMEOUT_SECONDS",
        "DECISION_AGENT_DB_QUERY_TIMEOUT_SECONDS",
        "DECISION_AGENT_DB_MAX_ROWS",
        "DECISION_AGENT_DB_MAX_RESULT_CELLS",
    }
    assert environment["DECISION_AGENT_APP_NAME"] == "Parent Agent"
    assert environment["DECISION_AGENT_ENVIRONMENT"] == "production"
    assert environment["DECISION_AGENT_DB_HOST"] == "db.internal"
    assert environment["DECISION_AGENT_DB_PORT"] == "4406"
    assert environment["DECISION_AGENT_DB_DATABASE"] == "operations_prod"
    assert environment["DECISION_AGENT_DB_READONLY_USERNAME"] == "readonly_service"
    assert hmac.compare_digest(
        environment["DECISION_AGENT_DB_READONLY_PASSWORD"],
        "database-password-marker",
    )
    assert environment["DECISION_AGENT_DB_CONNECT_TIMEOUT_SECONDS"] == "9"
    assert environment["DECISION_AGENT_DB_QUERY_TIMEOUT_SECONDS"] == "4.5"
    assert environment["DECISION_AGENT_DB_MAX_ROWS"] == "321"
    assert environment["DECISION_AGENT_DB_MAX_RESULT_CELLS"] == "4321"
    assert "DECISION_AGENT_LLM_API_KEY" not in environment
    assert "DECISION_AGENT_MILVUS_TOKEN" not in environment
    assert "DECISION_AGENT_MEMORY_REDIS_URL" not in environment

    child = child_settings_from_environment(monkeypatch, environment)
    assert_mcp_settings_equal(parent, child)


@pytest.mark.asyncio
async def test_database_preflight_client_keeps_tool_schema_and_marks_only_child_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[Any] = []
    install_fake_transport(monkeypatch, captured_parameters=captured)

    async with EnterpriseDataMCPClient.for_database_preflight(make_settings()) as client:
        assert client.database_preflight_passed is True

    environment = mcp_environment(captured[0])
    assert environment["DECISION_AGENT_MCP_DATABASE_PREFLIGHT"] == "true"
    assert set(environment) - {
        "DECISION_AGENT_MCP_DATABASE_PREFLIGHT",
        "DECISION_AGENT_MCP_DATABASE_PREFLIGHT_RESULT_FILE",
    } == {
        "DECISION_AGENT_APP_NAME",
        "DECISION_AGENT_ENVIRONMENT",
        "DECISION_AGENT_DB_HOST",
        "DECISION_AGENT_DB_PORT",
        "DECISION_AGENT_DB_DATABASE",
        "DECISION_AGENT_DB_READONLY_USERNAME",
        "DECISION_AGENT_DB_READONLY_PASSWORD",
        "DECISION_AGENT_DB_CONNECT_TIMEOUT_SECONDS",
        "DECISION_AGENT_DB_QUERY_TIMEOUT_SECONDS",
        "DECISION_AGENT_DB_MAX_ROWS",
        "DECISION_AGENT_DB_MAX_RESULT_CELLS",
    }


def test_child_stderr_capture_bounds_retention_without_exposing_raw_content() -> None:
    capture = _ChildStderrCapture(max_bytes=8)
    capture.errlog.write(b"warning password-marker more-output")
    summary = capture.close()

    assert summary["child_stderr_captured"] is True
    assert summary["child_stderr_bytes"] > 8
    assert summary["child_stderr_truncated"] is True
    assert summary["child_warning_detected"] is True
    assert summary["sensitive_content_detected"] is False
    assert "password-marker" not in repr(summary)


def test_child_stderr_capture_detects_sensitive_markers_without_returning_them() -> None:
    capture = _ChildStderrCapture(max_bytes=128)
    capture.errlog.write(b"authorization password-marker")
    summary = capture.close()

    assert summary["sensitive_content_detected"] is True
    assert "password-marker" not in repr(summary)


@pytest.mark.parametrize(
    ("marker", "expected"),
    [
        ("mcp_database_preflight_mysql_authentication_failed", "mysql_authentication_failed"),
        ("mcp_database_preflight_mysql_connection_failed", "mysql_connection_failed"),
        ("mcp_database_preflight_mysql_timeout", "mysql_timeout"),
        ("mcp_database_preflight_mysql_health_query_failed", "mysql_health_query_failed"),
        ("mcp_database_preflight_mysql_shutdown_failed", "mysql_shutdown_failed"),
    ],
)
def test_preflight_child_errors_map_to_fixed_codes(marker: str, expected: str) -> None:
    assert _child_error_class(marker) == expected


@pytest.mark.asyncio
async def test_from_settings_serializes_none_to_block_other_configuration_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = make_settings(db_readonly_password=None)
    monkeypatch.setenv("DECISION_AGENT_DB_READONLY_PASSWORD", "ambient-password")
    monkeypatch.setattr(
        "decision_agent.mcp_client.enterprise_data_client.get_default_environment",
        lambda: {"PATH": "safe-path"},
    )
    captured: list[Any] = []
    install_fake_transport(monkeypatch, captured_parameters=captured)

    async with EnterpriseDataMCPClient.from_settings(parent):
        pass

    environment = mcp_environment(captured[0])
    assert environment["DECISION_AGENT_DB_READONLY_PASSWORD"] == ""
    child = child_settings_from_environment(monkeypatch, environment)
    assert child.db_readonly_password is None
    assert_mcp_settings_equal(parent, child)


@pytest.mark.asyncio
async def test_from_settings_snapshot_is_immune_to_later_host_environment_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = make_settings()
    safe_environment = {"PATH": "captured-path"}
    monkeypatch.setattr(
        "decision_agent.mcp_client.enterprise_data_client.get_default_environment",
        lambda: safe_environment,
    )
    client = EnterpriseDataMCPClient.from_settings(parent)
    safe_environment["PATH"] = "mutated-path"
    monkeypatch.setenv("DECISION_AGENT_DB_HOST", "mutated-host")
    captured: list[Any] = []
    install_fake_transport(monkeypatch, captured_parameters=captured)

    async with client:
        pass

    assert captured[0].env["PATH"] == "captured-path"
    assert captured[0].env["DECISION_AGENT_DB_HOST"] == "db.internal"


@pytest.mark.asyncio
async def test_direct_environment_mapping_is_defensively_copied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = {"PATH": "captured-path", "CUSTOM_VALUE": "captured-value"}
    client = EnterpriseDataMCPClient(server_environment=environment)
    environment["PATH"] = "mutated-path"
    environment["CUSTOM_VALUE"] = "mutated-value"
    captured: list[Any] = []
    install_fake_transport(monkeypatch, captured_parameters=captured)

    async with client:
        pass

    assert captured[0].env == {
        "PATH": "captured-path",
        "CUSTOM_VALUE": "captured-value",
    }


@pytest.mark.asyncio
async def test_clients_from_different_settings_keep_isolated_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "decision_agent.mcp_client.enterprise_data_client.get_default_environment",
        lambda: {"PATH": "safe-path"},
    )
    first = EnterpriseDataMCPClient.from_settings(
        make_settings(db_host="first.internal", db_readonly_password="first-password")
    )
    second = EnterpriseDataMCPClient.from_settings(
        make_settings(db_host="second.internal", db_readonly_password="second-password")
    )
    captured: list[Any] = []
    install_fake_transport(monkeypatch, captured_parameters=captured)

    async with first:
        pass
    async with second:
        pass

    assert captured[0].env["DECISION_AGENT_DB_HOST"] == "first.internal"
    assert captured[1].env["DECISION_AGENT_DB_HOST"] == "second.internal"
    assert hmac.compare_digest(
        captured[0].env["DECISION_AGENT_DB_READONLY_PASSWORD"],
        "first-password",
    )
    assert hmac.compare_digest(
        captured[1].env["DECISION_AGENT_DB_READONLY_PASSWORD"],
        "second-password",
    )


def test_client_repr_and_stable_error_do_not_expose_database_password() -> None:
    marker = "database-password-marker"
    client = EnterpriseDataMCPClient.from_settings(make_settings(db_readonly_password=marker))
    error = EnterpriseDataMCPError("mcp_server_unavailable")

    assert marker not in repr(client)
    assert marker not in str(error)
    assert marker not in repr(error)
    assert marker not in "".join(traceback.format_exception(error))


@pytest.mark.asyncio
async def test_client_preserves_cancelled_error_during_stdio_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CancelledContext:
        async def __aenter__(self) -> None:
            raise asyncio.CancelledError

        async def __aexit__(self, exc_type, exc, traceback) -> None:  # type: ignore[no-untyped-def]
            return None

    monkeypatch.setattr(
        "decision_agent.mcp_client.enterprise_data_client.stdio_client",
        lambda parameters, errlog=None: CancelledContext(),
    )

    with pytest.raises(asyncio.CancelledError):
        async with EnterpriseDataMCPClient.from_settings(make_settings()):
            pass


@pytest.mark.asyncio
async def test_client_initializes_once_and_caches_schema_and_definitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, stdio_context, _ = install_fake_transport(monkeypatch)
    async with EnterpriseDataMCPClient() as client:
        assert (await client.get_enterprise_schema()).tables == {"products": ["product_id"]}
        assert (await client.get_enterprise_schema()).tables == {"products": ["product_id"]}
        assert (await client.get_business_definitions()).definitions == {
            "natural_month": "half-open"
        }
        await client.get_business_definitions()
        result = await client.execute_safe_query("SELECT product_id FROM products")
    assert result.rows == [["P100"]]
    assert session.initialize_calls == session.list_calls == 1
    assert [name for name, _ in session.calls] == [
        "get_enterprise_schema",
        "get_business_definitions",
        "execute_safe_query",
    ]
    assert stdio_context.closed is True


@pytest.mark.asyncio
async def test_client_rejects_missing_required_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_transport(monkeypatch, FakeSession(tools={"get_enterprise_schema"}))
    with pytest.raises(EnterpriseDataMCPError, match="MCP request") as raised:
        async with EnterpriseDataMCPClient():
            pass
    assert raised.value.code == "mcp_required_tool_missing"


@pytest.mark.asyncio
async def test_client_releases_active_state_after_malformed_tools_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, stdio_context, _ = install_fake_transport(monkeypatch)
    original_list_tools = session.list_tools

    async def malformed_list_tools() -> object:
        return object()

    session.list_tools = malformed_list_tools  # type: ignore[method-assign]
    client = EnterpriseDataMCPClient()
    with pytest.raises(EnterpriseDataMCPError) as raised:
        async with client:
            pass
    assert raised.value.code == "mcp_initialization_failed"
    assert stdio_context.closed is True

    session.list_tools = original_list_tools  # type: ignore[method-assign]
    async with client:
        assert (await client.get_enterprise_schema()).tables == {"products": ["product_id"]}


@pytest.mark.asyncio
async def test_client_rejects_concurrent_enter_and_allows_sequential_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, stdio_context, _ = install_fake_transport(monkeypatch)
    client = EnterpriseDataMCPClient()
    async with client:
        with pytest.raises(EnterpriseDataMCPError) as raised:
            async with client:
                pass
        assert raised.value.code == "mcp_client_already_active"
        assert stdio_context.enter_calls == 1
        assert (await client.get_enterprise_schema()).tables == {"products": ["product_id"]}
    async with client:
        assert (await client.get_enterprise_schema()).tables == {"products": ["product_id"]}
    assert stdio_context.enter_calls == 2
    assert session.initialize_calls == 2


@pytest.mark.asyncio
async def test_client_maps_startup_failure_without_transport_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = "database-password-marker"
    install_fake_transport(monkeypatch, startup_error=OSError(marker))
    with pytest.raises(EnterpriseDataMCPError) as raised:
        async with EnterpriseDataMCPClient():
            pass
    assert raised.value.code == "mcp_server_unavailable"
    assert marker not in str(raised.value)
    assert marker not in repr(raised.value)
    assert marker not in "".join(traceback.format_exception(raised.value))


@pytest.mark.asyncio
async def test_client_maps_invalid_tool_payload_without_leaking_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = FakeSession(payloads={"get_enterprise_schema": {"bad": "payload"}})
    install_fake_transport(monkeypatch, session)
    async with EnterpriseDataMCPClient() as client:
        with pytest.raises(EnterpriseDataMCPError) as raised:
            await client.get_enterprise_schema()
    assert raised.value.code == "mcp_response_invalid"


@pytest.mark.asyncio
async def test_client_maps_tool_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    class SlowSession(FakeSession):
        async def call_tool(self, name: str, arguments: dict[str, object]) -> SimpleNamespace:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    install_fake_transport(monkeypatch, SlowSession())
    async with EnterpriseDataMCPClient(timeout_seconds=0.01) as client:
        with pytest.raises(EnterpriseDataMCPError) as raised:
            await client.get_enterprise_schema()
    assert raised.value.code == "mcp_tool_timeout"


@pytest.mark.asyncio
async def test_client_maps_server_exit_during_tool_call_without_transport_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = FakeSession(payloads={"get_enterprise_schema": EndOfStream()})
    install_fake_transport(monkeypatch, session)
    async with EnterpriseDataMCPClient() as client:
        with pytest.raises(EnterpriseDataMCPError) as raised:
            await client.get_enterprise_schema()
    assert raised.value.code == "mcp_tool_call_failed"
    assert "traceback" not in str(raised.value).lower()


@pytest.mark.asyncio
async def test_client_bounds_stalled_transport_close_in_its_existing_timeout() -> None:
    class StalledContext:
        async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
            await asyncio.Event().wait()

    client = EnterpriseDataMCPClient(timeout_seconds=0.01)

    await client._close_context(StalledContext())  # type: ignore[arg-type]
