"""Isolation coverage for the opt-in configured Runtime preflight."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from decision_agent.application.runtime_preflight import (
    RuntimePreflightResult,
    _preflight_error_code,
    main,
    preflight_configured_runtime,
    validate_result_file,
    write_result_file,
)
from decision_agent.config import Settings


class _Client:
    def __init__(self) -> None:
        self.database_preflight_passed = True
        self.child_stderr_summary = {
            "child_stderr_captured": True,
            "child_stderr_truncated": False,
            "child_warning_detected": False,
            "child_error_class": None,
            "sensitive_content_detected": False,
        }


class _Runtime:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class _FailedPreflightClient:
    database_preflight_passed = False
    database_preflight_error_code = "mysql_connection_failed"


def test_preflight_failure_categories_do_not_mislabel_unreached_mysql() -> None:
    assert (
        _preflight_error_code("bootstrap_runtime_unavailable", None, "retrieval_initialize")
        == "retrieval_initialize_failed"
    )
    assert (
        _preflight_error_code(
            "bootstrap_runtime_unavailable", _FailedPreflightClient(), "mcp_database_preflight"
        )
        == "mysql_connection_failed"
    )


@pytest.mark.asyncio
async def test_configured_runtime_preflight_uses_only_the_opt_in_mcp_factory_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _Client()
    runtime = _Runtime()
    builder_calls = 0

    def fake_preflight_client(_: Settings) -> _Client:
        return client

    def fake_builder(settings: Settings, dependencies: object) -> object:
        nonlocal builder_calls
        builder_calls += 1
        dependencies.enterprise_data_client_factory(settings)  # type: ignore[attr-defined]
        return object()

    async def fake_bootstrap(_: object) -> _Runtime:
        return runtime

    monkeypatch.setattr(
        "decision_agent.application.runtime_preflight.EnterpriseDataMCPClient.for_database_preflight",
        fake_preflight_client,
    )
    monkeypatch.setattr(
        "decision_agent.application.runtime_preflight.create_configured_runtime_builder",
        fake_builder,
    )

    result = await preflight_configured_runtime(
        Settings(_env_file=None),
        bootstrap=fake_bootstrap,  # type: ignore[arg-type]
    )

    assert result.status == "passed"
    assert result.error_code == "none"
    assert runtime.closed is True
    assert builder_calls == 1


def _passed_result() -> RuntimePreflightResult:
    return RuntimePreflightResult(
        component="runtime",
        phase="bootstrap_close",
        status="passed",
        error_code="none",
        bootstrap_stage="runtime_close",
        failure_stage="none",
        cleanup_stage="retrieval_close",
        mysql_preflight="passed",
        runtime_close="passed",
        child_stderr_captured=True,
        child_stderr_truncated=False,
        child_warning_detected=False,
        child_error_class=None,
        sensitive_content_detected=False,
    )


def test_result_file_is_atomic_json_and_remains_available_for_parent_validation(
    tmp_path: Path,
) -> None:
    result_file = tmp_path / "result.json"
    write_result_file(result_file, _passed_result())

    result, code = validate_result_file(result_file, 0)

    assert code == "none"
    assert result == _passed_result()
    assert result_file.is_file()
    assert not list(tmp_path.glob("*.tmp"))


def test_child_main_returns_immediately_after_writing_a_legal_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def passed_preflight(_: Settings) -> RuntimePreflightResult:
        return _passed_result()

    monkeypatch.setattr(
        "decision_agent.application.runtime_preflight.preflight_configured_runtime",
        passed_preflight,
    )
    result_file = tmp_path / "result.json"
    stage_file = tmp_path / "stage.json"

    assert main(["--result-file", str(result_file), "--stage-file", str(stage_file)]) == 0
    assert validate_result_file(result_file, 0)[1] == "none"
    assert json.loads(stage_file.read_text(encoding="utf-8"))["stage"] == "child_result_written"


@pytest.mark.parametrize(
    ("payload", "exit_code", "expected"),
    [
        (None, 1, "runtime_preflight_result_missing"),
        ("not-json", 1, "runtime_preflight_result_invalid"),
        (json.dumps({"status": "failed"}), 1, "runtime_preflight_result_incomplete"),
        (json.dumps(_passed_result().safe_json()), 1, "runtime_preflight_result_incomplete"),
        (None, 1, "runtime_preflight_result_missing"),
    ],
)
def test_result_file_rejects_invalid_or_incomplete_payloads(
    tmp_path: Path, payload: str | None, exit_code: int, expected: str
) -> None:
    result_file = tmp_path / "result.json"
    if payload is not None:
        result_file.write_text(payload, encoding="utf-8")

    _, code = validate_result_file(result_file, exit_code)

    assert code == expected


def test_result_file_rejects_exit_status_contradiction_and_unsafe_content(tmp_path: Path) -> None:
    result_file = tmp_path / "result.json"
    write_result_file(result_file, _passed_result())
    assert validate_result_file(result_file, 1)[1] == "runtime_preflight_result_inconsistent"

    payload = json.loads(result_file.read_text(encoding="utf-8"))
    payload["exception_type"] = "password-marker"
    result_file.write_text(json.dumps(payload), encoding="utf-8")
    assert validate_result_file(result_file, 1)[1] == "runtime_preflight_result_unsafe"

    payload["exception_type"] = None
    payload["bootstrap_stage"] = "untrusted-stage"
    result_file.write_text(json.dumps(payload), encoding="utf-8")
    assert validate_result_file(result_file, 0)[1] == "runtime_preflight_result_invalid"
