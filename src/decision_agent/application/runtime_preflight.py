"""Opt-in, payload-free lifecycle preflights for configured external dependencies."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from decision_agent.application.bootstrap import (
    BootstrappedRuntime,
    RuntimeBootstrapError,
    RuntimeBuilder,
    build_bootstrapped_runtime,
)
from decision_agent.application.configured_runtime import (
    ConfiguredRuntimeDependencies,
    _default_configured_runtime_dependencies,
    create_configured_runtime_builder,
)
from decision_agent.config import Settings
from decision_agent.mcp_client.enterprise_data_client import EnterpriseDataMCPClient
from decision_agent.mcp_client.errors import EnterpriseDataMCPError


@dataclass(frozen=True, slots=True)
class RuntimePreflightResult:
    """Content-safe projection of one opt-in Runtime bootstrap and close cycle."""

    component: str
    phase: str
    status: str
    error_code: str
    bootstrap_stage: str
    failure_stage: str
    cleanup_stage: str
    mysql_preflight: str
    runtime_close: str
    child_stderr_captured: bool
    child_stderr_truncated: bool
    child_warning_detected: bool
    child_error_class: str | None
    sensitive_content_detected: bool
    exception_type: str | None = None

    def safe_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


_RESULT_FIELDS = frozenset(RuntimePreflightResult.__dataclass_fields__)
_BOOTSTRAP_STAGES = frozenset(
    {
        "settings_loading",
        "runtime_builder",
        "provider_adapters",
        "retrieval_runtime",
        "retrieval_initialize",
        "mcp_database_preflight",
        "retrieval_close",
        "runtime_close",
    }
)


@dataclass(slots=True)
class _PreflightLifecycle:
    current_stage: str = "runtime_builder"
    failure_stage: str = "none"
    cleanup_stage: str = "not_started"

    def enter(self, stage: str) -> None:
        self.current_stage = stage

    def fail(self) -> None:
        if self.failure_stage == "none":
            self.failure_stage = self.current_stage


class _DatabasePreflightClientFactory:
    """Retain only safe lifecycle metadata from the Runtime-owned MCP handshake."""

    def __init__(self, lifecycle: _PreflightLifecycle) -> None:
        self._lifecycle = lifecycle
        self.client: EnterpriseDataMCPClient | None = None

    def __call__(self, settings: Settings) -> EnterpriseDataMCPClient:
        self._lifecycle.enter("mcp_database_preflight")
        self.client = EnterpriseDataMCPClient.for_database_preflight(settings)
        return self.client


class _PreflightRetrievalRuntime:
    """Proxy the formal retrieval owner while retaining one fixed lifecycle stage."""

    def __init__(self, runtime: object, lifecycle: _PreflightLifecycle) -> None:
        self._runtime = runtime
        self._lifecycle = lifecycle

    @property
    def pipeline(self) -> object:
        return self._runtime.pipeline  # type: ignore[attr-defined]

    async def initialize(self) -> None:
        self._lifecycle.enter("retrieval_initialize")
        await self._runtime.initialize()  # type: ignore[attr-defined]

    async def aclose(self) -> None:
        self._lifecycle.cleanup_stage = "retrieval_close"
        await self._runtime.aclose()  # type: ignore[attr-defined]


async def preflight_mcp() -> dict[str, object]:
    """Verify the normal formal MCP handshake and static schema discovery once."""
    try:
        async with EnterpriseDataMCPClient.from_settings(Settings()) as client:
            schema = await client.get_enterprise_schema()
            definitions = await client.get_business_definitions()
        return {
            "component": "mcp",
            "status": "passed",
            "handshake": "passed",
            "schema_valid": bool(schema.tables and definitions.definitions),
            "shutdown": "passed",
            "error_code": "none",
        }
    except EnterpriseDataMCPError as exc:
        return {
            "component": "mcp",
            "status": "failed",
            "handshake": "failed",
            "schema_valid": False,
            "shutdown": "unknown",
            "error_code": exc.code,
        }


async def preflight_configured_runtime(
    settings: Settings,
    *,
    dependencies: ConfiguredRuntimeDependencies | None = None,
    bootstrap: Callable[
        [RuntimeBuilder], Awaitable[BootstrappedRuntime]
    ] = build_bootstrapped_runtime,
) -> RuntimePreflightResult:
    """Build and close the formal Runtime once with an MCP-owned MySQL transport probe."""
    lifecycle = _PreflightLifecycle()
    client_factory = _DatabasePreflightClientFactory(lifecycle)
    resolved = dependencies or _default_configured_runtime_dependencies()

    def provider_factory(current_settings: Settings) -> object:
        lifecycle.enter("provider_adapters")
        return resolved.provider_adapters_factory(current_settings)

    def knowledge_factory(current_settings: Settings) -> _PreflightRetrievalRuntime:
        lifecycle.enter("retrieval_runtime")
        return _PreflightRetrievalRuntime(
            resolved.knowledge_runtime_factory(current_settings), lifecycle
        )

    preflight_dependencies = replace(
        resolved,
        provider_adapters_factory=provider_factory,  # type: ignore[arg-type]
        knowledge_runtime_factory=knowledge_factory,  # type: ignore[arg-type]
        enterprise_data_client_factory=client_factory,
    )
    runtime: BootstrappedRuntime | None = None
    error_code = "none"
    runtime_close = "not_started"
    try:
        builder = create_configured_runtime_builder(settings, preflight_dependencies)
        runtime = await bootstrap(builder)
    except RuntimeBootstrapError as exc:
        lifecycle.fail()
        error_code = _preflight_error_code(exc.code, client_factory.client, lifecycle.failure_stage)
    finally:
        if runtime is not None:
            try:
                lifecycle.enter("runtime_close")
                await runtime.aclose()
                runtime_close = "passed"
            except RuntimeBootstrapError as exc:
                lifecycle.fail()
                error_code = exc.code
                runtime_close = "failed"

    client = client_factory.client
    stderr = client.child_stderr_summary if client is not None else _empty_stderr_summary()
    mysql_preflight = "not_reached"
    if client is not None:
        mysql_preflight = "passed" if client.database_preflight_passed else "failed"
    if error_code == "none" and mysql_preflight != "passed":
        error_code = "mysql_preflight_failed"
    return RuntimePreflightResult(
        component="runtime",
        phase="bootstrap_close",
        status="passed" if error_code == "none" else "failed",
        error_code=error_code,
        bootstrap_stage=lifecycle.current_stage,
        failure_stage=lifecycle.failure_stage,
        cleanup_stage=lifecycle.cleanup_stage,
        mysql_preflight=mysql_preflight,
        runtime_close=runtime_close,
        child_stderr_captured=bool(stderr["child_stderr_captured"]),
        child_stderr_truncated=bool(stderr["child_stderr_truncated"]),
        child_warning_detected=bool(stderr["child_warning_detected"]),
        child_error_class=stderr["child_error_class"]
        if isinstance(stderr["child_error_class"], str)
        else None,
        sensitive_content_detected=bool(stderr["sensitive_content_detected"]),
        exception_type=None,
    )


def _preflight_error_code(
    bootstrap_error_code: str,
    client: EnterpriseDataMCPClient | None,
    stage: str,
) -> str:
    """Refine only the opt-in MCP probe failure without changing Runtime semantics."""
    if client is not None and not client.database_preflight_passed:
        return client.database_preflight_error_code
    if bootstrap_error_code == "bootstrap_runtime_unavailable":
        return f"{stage}_failed"
    return bootstrap_error_code


def write_result_file(result_file: Path, result: RuntimePreflightResult) -> None:
    """Atomically publish one UTF-8 safety result after the child lifecycle ends."""
    temporary = result_file.with_suffix(result_file.suffix + ".tmp")
    payload = result.safe_json()
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, result_file)


def write_stage_file(stage_file: Path, stage: str) -> None:
    """Atomically retain one fixed, payload-free child lifecycle stage."""
    temporary = stage_file.with_suffix(stage_file.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps({"stage": stage}, separators=(",", ":")))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, stage_file)


def validate_result_file(
    result_file: Path, exit_code: int
) -> tuple[RuntimePreflightResult | None, str]:
    """Validate a child result before its parent is allowed to remove it."""
    if not result_file.is_file():
        return None, "runtime_preflight_result_missing"
    try:
        payload: Any = json.loads(result_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, "runtime_preflight_result_invalid"
    if not isinstance(payload, dict) or set(payload) != _RESULT_FIELDS:
        return None, "runtime_preflight_result_incomplete"
    if any(isinstance(value, str) and _unsafe_value(value) for value in payload.values()):
        return None, "runtime_preflight_result_unsafe"
    try:
        result = RuntimePreflightResult(**payload)
    except TypeError:
        return None, "runtime_preflight_result_invalid"
    if result.bootstrap_stage not in _BOOTSTRAP_STAGES:
        return None, "runtime_preflight_result_invalid"
    if (exit_code == 0) != (result.status == "passed"):
        return None, "runtime_preflight_result_inconsistent"
    return result, "none"


def _unsafe_value(value: str) -> bool:
    normalized = value.lower()
    return any(marker in normalized for marker in ("password", "api_key", "mysql://", "select "))


def _empty_stderr_summary() -> dict[str, object]:
    return {
        "child_stderr_captured": False,
        "child_stderr_truncated": False,
        "child_warning_detected": False,
        "child_error_class": None,
        "sensitive_content_detected": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--result-file", type=Path)
    parser.add_argument("--stage-file", type=Path)
    arguments = parser.parse_args(argv)
    result_file = arguments.result_file
    if arguments.stage_file is not None:
        write_stage_file(arguments.stage_file, "process_entered")
    try:
        if arguments.stage_file is not None:
            write_stage_file(arguments.stage_file, "settings_loading")
        result = asyncio.run(preflight_configured_runtime(Settings()))
    except Exception as exc:
        result = RuntimePreflightResult(
            component="runtime",
            phase="bootstrap_close",
            status="failed",
            error_code="runtime_preflight_unavailable",
            bootstrap_stage="settings_loading",
            failure_stage="settings_loading",
            cleanup_stage="not_started",
            mysql_preflight="not_reached",
            runtime_close="not_reached",
            child_stderr_captured=False,
            child_stderr_truncated=False,
            child_warning_detected=False,
            child_error_class=None,
            sensitive_content_detected=False,
            exception_type=type(exc).__name__,
        )
    if result_file is not None:
        if arguments.stage_file is not None:
            write_stage_file(arguments.stage_file, "child_result_writing")
        write_result_file(result_file, result)
        if arguments.stage_file is not None:
            write_stage_file(arguments.stage_file, "child_result_written")
    else:
        sys.stdout.write(result.safe_json() + "\n")
    return 0 if result.status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
