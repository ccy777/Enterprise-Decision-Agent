"""Independent evidence-owning supervisor for one Runtime preflight child process."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from decision_agent.application.runtime_preflight import validate_result_file

_STDERR_LIMIT = 8_192


@dataclass(frozen=True, slots=True)
class VerifiedEnvelope:
    component: str
    phase: str
    evidence_status: str
    evidence_error_code: str
    runtime_status: str
    runtime_error_code: str
    child_process_started: bool
    child_exit_code: int | None
    child_timed_out: bool
    child_result_present: bool
    child_result_valid: bool
    result_exit_consistent: bool
    mysql_preflight: str
    runtime_close: str
    child_stderr_captured: bool
    child_stderr_bytes: int
    child_stderr_truncated: bool
    child_warning_detected: bool
    child_error_class: str | None
    child_error_signature_matched: bool
    sensitive_content_detected: bool
    exception_type: str | None
    last_supervisor_stage: str
    last_child_stage: str | None
    child_stage_present: bool
    child_stage_valid: bool

    def safe_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


def supervise(evidence_dir: Path, *, timeout_seconds: float = 120.0) -> VerifiedEnvelope:
    """Run one child, verify its atomic result, and retain evidence for an external reader."""
    evidence_dir.mkdir(parents=True, exist_ok=True)
    result_file = evidence_dir / "child-result.json"
    stage_file = evidence_dir / "child-stage.json"
    stdout_file = evidence_dir / "child-stdout.bin"
    stderr_file = evidence_dir / "child-stderr.bin"
    process: subprocess.Popen[bytes] | None = None
    timed_out = False
    exit_code: int | None = None
    try:
        with stdout_file.open("wb") as stdout, stderr_file.open("wb") as stderr:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "decision_agent.application.runtime_preflight",
                    "--result-file",
                    str(result_file),
                    "--stage-file",
                    str(stage_file),
                ],
                stdout=stdout,
                stderr=stderr,
            )
            try:
                exit_code = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                process.kill()
                exit_code = process.wait()
    except OSError:
        envelope = _failure_envelope(
            "evidence_write_failed", process is not None, exit_code, timed_out
        )
    else:
        envelope = _verify_child(result_file, stage_file, stderr_file, exit_code, timed_out)
    _atomic_write(evidence_dir / "verified-envelope.json", envelope.safe_json())
    return envelope


def _verify_child(
    result_file: Path,
    stage_file: Path,
    stderr_file: Path,
    exit_code: int | None,
    timed_out: bool,
) -> VerifiedEnvelope:
    summary = _stderr_summary(stderr_file)
    stage, stage_present, stage_valid = _read_stage(stage_file)
    if timed_out:
        result, code = _validate_result_without_observed_exit(result_file)
        if result is not None:
            return VerifiedEnvelope(
                "runtime_preflight_supervisor",
                "collect_and_verify",
                "verified",
                "none",
                "failed",
                "runtime_child_shutdown_timeout",
                True,
                exit_code,
                True,
                True,
                True,
                True,
                result.mysql_preflight,
                "failed",
                True,
                int(summary["bytes"]),
                bool(summary["truncated"]),
                bool(summary["warning"]),
                summary["error_class"] if isinstance(summary["error_class"], str) else None,
                bool(summary["matched"]),
                bool(summary["sensitive"]),
                result.exception_type,
                "envelope_writing",
                stage,
                stage_present,
                stage_valid,
            )
        return _failure_envelope(
            "evidence_child_timeout",
            True,
            exit_code,
            True,
            summary,
            stage,
            stage_present,
            stage_valid,
        )
    result, code = validate_result_file(result_file, exit_code if exit_code is not None else 1)
    if result is None:
        if code == "runtime_preflight_result_missing" and exit_code is not None:
            return _runtime_failure_without_result(
                exit_code, summary, stage, stage_present, stage_valid
            )
        return _failure_envelope(
            code.replace("runtime_preflight_", "evidence_"), True, exit_code, False, summary
        )
    return VerifiedEnvelope(
        component="runtime_preflight_supervisor",
        phase="collect_and_verify",
        evidence_status="verified",
        evidence_error_code="none",
        runtime_status=result.status,
        runtime_error_code=result.error_code,
        child_process_started=True,
        child_exit_code=exit_code,
        child_timed_out=False,
        child_result_present=True,
        child_result_valid=True,
        result_exit_consistent=True,
        mysql_preflight=result.mysql_preflight,
        runtime_close=result.runtime_close,
        child_stderr_captured=True,
        child_stderr_bytes=summary["bytes"],
        child_stderr_truncated=summary["truncated"],
        child_warning_detected=summary["warning"],
        child_error_class=summary["error_class"],
        child_error_signature_matched=summary["matched"],
        sensitive_content_detected=summary["sensitive"],
        exception_type=result.exception_type,
        last_supervisor_stage="envelope_writing",
        last_child_stage=stage,
        child_stage_present=stage_present,
        child_stage_valid=stage_valid,
    )


def _validate_result_without_observed_exit(
    result_file: Path,
) -> tuple[object | None, str]:
    """Validate a written result against its declared child exit contract.

    A forced shutdown changes the observed exit code. It must not invalidate an
    atomically published result that was legal before the child stalled.
    """
    try:
        status = json.loads(result_file.read_text(encoding="utf-8")).get("status")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, AttributeError):
        return validate_result_file(result_file, 1)
    expected_exit = 0 if status == "passed" else 1
    return validate_result_file(result_file, expected_exit)


def _failure_envelope(
    code: str,
    started: bool,
    exit_code: int | None,
    timed_out: bool,
    summary: dict[str, int | bool | str | None] | None = None,
    stage: str | None = None,
    stage_present: bool = False,
    stage_valid: bool = False,
) -> VerifiedEnvelope:
    values = summary or {
        "bytes": 0,
        "truncated": False,
        "warning": False,
        "sensitive": False,
        "error_class": None,
        "matched": False,
    }
    return VerifiedEnvelope(
        "runtime_preflight_supervisor",
        "collect_and_verify",
        "failed",
        code,
        "unknown",
        "runtime_bootstrap_failed",
        started,
        exit_code,
        timed_out,
        False,
        False,
        False,
        "not_reached",
        "not_reached",
        True,
        int(values["bytes"]),
        bool(values["truncated"]),
        bool(values["warning"]),
        values["error_class"] if isinstance(values["error_class"], str) else None,
        bool(values["matched"]),
        bool(values["sensitive"]),
        None,
        "envelope_writing",
        stage,
        stage_present,
        stage_valid,
    )


def _runtime_failure_without_result(
    exit_code: int,
    summary: dict[str, int | bool | str | None],
    stage: str | None,
    stage_present: bool,
    stage_valid: bool,
) -> VerifiedEnvelope:
    return VerifiedEnvelope(
        "runtime_preflight_supervisor",
        "collect_and_verify",
        "verified",
        "none",
        "failed",
        "runtime_child_exited_without_result",
        True,
        exit_code,
        False,
        False,
        False,
        True,
        "not_reached",
        "not_reached",
        True,
        int(summary["bytes"]),
        bool(summary["truncated"]),
        bool(summary["warning"]),
        summary["error_class"]
        if isinstance(summary["error_class"], str)
        else "unknown_child_error",
        bool(summary["matched"]),
        bool(summary["sensitive"]),
        None,
        "envelope_writing",
        stage,
        stage_present,
        stage_valid,
    )


def _stderr_summary(stderr_file: Path) -> dict[str, int | bool | str | None]:
    content = stderr_file.read_bytes()
    retained = content[:_STDERR_LIMIT].lower()
    signatures = (
        (b"modulenotfounderror", "module_not_found"),
        (b"importerror", "import_error"),
        (b"syntaxerror", "syntax_error"),
        (b"permissionerror", "permission_denied"),
        (b"filenotfounderror", "file_not_found"),
        (b"timeout", "timeout_error"),
        (b"brokenpipe", "broken_pipe"),
    )
    matched = next((name for marker, name in signatures if marker in retained), None)
    return {
        "bytes": len(content),
        "truncated": len(content) > _STDERR_LIMIT,
        "warning": b"warning" in retained,
        "error_class": matched
        or ("unhandled_python_exception" if b"traceback" in retained else "unknown_child_error"),
        "matched": matched is not None,
        "sensitive": any(
            marker in retained for marker in (b"password", b"api_key", b"authorization")
        ),
    }


def _read_stage(stage_file: Path) -> tuple[str | None, bool, bool]:
    if not stage_file.is_file():
        return None, False, False
    try:
        value = json.loads(stage_file.read_text(encoding="utf-8")).get("stage")
    except (OSError, json.JSONDecodeError, AttributeError):
        return None, True, False
    return (
        value,
        True,
        value
        in {"process_entered", "settings_loading", "child_result_writing", "child_result_written"},
    )


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    arguments = parser.parse_args(argv)
    try:
        supervise(arguments.evidence_dir, timeout_seconds=arguments.timeout_seconds)
    except OSError:
        return 22
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
