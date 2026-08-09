"""Offline evidence-retention coverage for the Runtime preflight supervisor."""

from __future__ import annotations

from pathlib import Path

import pytest

from decision_agent.application.runtime_preflight import RuntimePreflightResult, write_result_file
from decision_agent.application.runtime_preflight_supervisor import _verify_child
from decision_agent.application.runtime_preflight_verify import verify


def _result(status: str = "passed") -> RuntimePreflightResult:
    return RuntimePreflightResult(
        "runtime",
        "bootstrap_close",
        status,
        "none" if status == "passed" else "runtime_bootstrap_failed",
        "runtime_close",
        "none" if status == "passed" else "mcp_database_preflight",
        "retrieval_close",
        "passed",
        "passed",
        True,
        False,
        False,
        None,
        False,
    )


def test_supervisor_verifies_child_result_before_external_verifier_reads_envelope(
    tmp_path: Path,
) -> None:
    result_file = tmp_path / "child-result.json"
    stderr_file = tmp_path / "child-stderr.bin"
    write_result_file(result_file, _result())
    stderr_file.write_bytes(b"third-party warning")

    envelope = _verify_child(result_file, tmp_path / "child-stage.json", stderr_file, 0, False)
    (tmp_path / "verified-envelope.json").write_text(envelope.safe_json(), encoding="utf-8")
    verified, code = verify(tmp_path)

    assert envelope.evidence_status == "verified"
    assert envelope.child_warning_detected is True
    assert verified == envelope
    assert code == 0
    assert result_file.is_file()


def test_supervisor_retains_precise_evidence_failure_for_missing_or_inconsistent_child_result(
    tmp_path: Path,
) -> None:
    stderr_file = tmp_path / "child-stderr.bin"
    stderr_file.write_bytes(b"password-marker" + b"x" * 9_000)

    envelope = _verify_child(
        tmp_path / "missing.json", tmp_path / "child-stage.json", stderr_file, 1, False
    )

    assert envelope.evidence_status == "verified"
    assert envelope.runtime_error_code == "runtime_child_exited_without_result"
    assert envelope.child_stderr_truncated is True
    assert envelope.sensitive_content_detected is True


def test_supervisor_verifies_a_failed_child_result_and_verifier_returns_ten(tmp_path: Path) -> None:
    result_file = tmp_path / "child-result.json"
    stderr_file = tmp_path / "child-stderr.bin"
    write_result_file(result_file, _result("failed"))
    stderr_file.write_bytes(b"")

    envelope = _verify_child(result_file, tmp_path / "child-stage.json", stderr_file, 1, False)
    (tmp_path / "verified-envelope.json").write_text(envelope.safe_json(), encoding="utf-8")
    _, verifier_code = verify(tmp_path)

    assert envelope.evidence_status == "verified"
    assert envelope.runtime_status == "failed"
    assert verifier_code == 10


def test_supervisor_marks_timeout_without_deleting_evidence(tmp_path: Path) -> None:
    stderr_file = tmp_path / "child-stderr.bin"
    stderr_file.write_bytes(b"")

    envelope = _verify_child(
        tmp_path / "child-result.json", tmp_path / "child-stage.json", stderr_file, 124, True
    )

    assert envelope.evidence_error_code == "evidence_child_timeout"
    assert envelope.child_timed_out is True
    assert stderr_file.exists()


@pytest.mark.parametrize("status", ["passed", "failed"])
def test_supervisor_keeps_legal_result_as_verified_runtime_failure_on_timeout(
    tmp_path: Path, status: str
) -> None:
    result_file = tmp_path / "child-result.json"
    stderr_file = tmp_path / "child-stderr.bin"
    write_result_file(result_file, _result(status))
    stderr_file.write_bytes(b"")

    envelope = _verify_child(result_file, tmp_path / "child-stage.json", stderr_file, -9, True)
    (tmp_path / "verified-envelope.json").write_text(envelope.safe_json(), encoding="utf-8")
    _, verifier_code = verify(tmp_path)

    assert envelope.evidence_status == "verified"
    assert envelope.runtime_status == "failed"
    assert envelope.runtime_error_code == "runtime_child_shutdown_timeout"
    assert envelope.child_result_valid is True
    assert envelope.result_exit_consistent is True
    assert verifier_code == 10
