"""Persist pytest evidence outside the repository without relying on terminal output."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
import xml.etree.ElementTree as element_tree
from pathlib import Path


def _write(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, separators=(",", ":"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _stage(run: Path, name: str, args: list[str]) -> dict[str, object]:
    stdout, stderr, junit = (
        run / f"{name}-stdout.log",
        run / f"{name}-stderr.log",
        run / f"{name}-junit.xml",
    )
    started = time.monotonic()
    with stdout.open("wb") as output, stderr.open("wb") as errors:
        process = subprocess.Popen(
            [sys.executable, "-m", "pytest", *args, "-q", f"--junitxml={junit}"],
            stdout=output,
            stderr=errors,
            cwd=Path.cwd(),
        )
        timed_out = False
        try:
            code = process.wait(timeout=300)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            code = process.wait()
    tests = failures = errors_count = skipped = 0
    valid = False
    if junit.is_file():
        try:
            root = element_tree.parse(junit).getroot()
            suites = list(root.iter("testsuite"))
            tests, failures, errors_count, skipped = (
                sum(int(suite.attrib.get(key, 0)) for suite in suites)
                for key in ("tests", "failures", "errors", "skipped")
            )
            valid = True
        except (element_tree.ParseError, ValueError):
            pass
    result = {
        "component": "pytest_stage",
        "stage": name,
        "state": "completed" if code == 0 else "failed",
        "pytest_started": True,
        "pytest_exit_code": code,
        "timed_out": timed_out,
        "junit_present": junit.is_file(),
        "junit_valid": valid,
        "tests": tests,
        "failures": failures,
        "errors": errors_count,
        "skipped": skipped,
        "duration_seconds": round(time.monotonic() - started, 3),
        "stdout_bytes": stdout.stat().st_size,
        "stderr_bytes": stderr.stat().st_size,
        "orphan_process": False,
        "error_code": "none" if code == 0 else "pytest_failed",
    }
    _write(run / f"{name}-status.json", result)
    return result


def main() -> int:
    root = Path(tempfile.gettempdir()) / "enterprise-decision-agent-m8d-b1-r11"
    root.mkdir(exist_ok=True)
    run = root / uuid.uuid4().hex
    run.mkdir()
    _write(
        root / "latest-run.json",
        {
            "run_id": run.name,
            "state": "running",
            "active_stage": "targeted",
            "run_started": time.time(),
            "run_completed": None,
        },
    )
    targeted = _stage(
        run,
        "targeted",
        [
            "tests/unit/application/test_runtime_preflight.py",
            "tests/unit/application/test_runtime_preflight_supervisor.py",
            "tests/unit/application/test_provider_preflight.py",
            "tests/unit/mcp_client",
            "tests/unit/mcp_server",
            "tests/unit/data",
            "tests/unit/retrieval",
        ],
    )
    unit = (
        _stage(run, "unit", ["tests/unit"])
        if targeted["state"] == "completed"
        else {"state": "not_run"}
    )
    overall = {
        "state": "completed",
        "targeted_state": targeted["state"],
        "unit_state": unit["state"],
        "all_required_tests_passed": targeted["state"] == unit["state"] == "completed",
        "static_checks_state": "not_run",
        "started_at": 0,
        "completed_at": time.time(),
    }
    _write(run / "overall-status.json", overall)
    _write(
        root / "latest-run.json",
        {
            "run_id": run.name,
            "state": "completed",
            "active_stage": "none",
            "run_started": 0,
            "run_completed": time.time(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
