from __future__ import annotations

from pathlib import Path


def test_ci_is_offline_minimum_permission_and_has_stable_required_jobs() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "pull_request_target" not in workflow
    assert "continue-on-error" not in workflow
    assert "contents: read" in workflow
    assert "pull-requests: read" in workflow
    assert "concurrency:" in workflow
    assert "cancel-in-progress: true" in workflow
    assert "tests/unit" in workflow
    assert "docker compose config --quiet --no-interpolate" in workflow
    assert "gitleaks/gitleaks-action@v2" in workflow
    assert "pip-audit==2.9.0" in workflow
    assert "python -m pip_audit --strict" in workflow
    assert workflow.count("python -m pip install -r requirements.lock") >= 4
    assert "python scripts/validate_dependency_lock.py" in workflow
    assert (
        "python -m pytest -q -m offline_integration --strict-markers --import-mode=importlib"
        in workflow
    )
    for job_name in (
        "quality:",
        "unit:",
        "security-evaluation:",
        "secret-scan:",
        "dependency-scan:",
        "offline-integration:",
    ):
        assert job_name in workflow


def test_dependabot_covers_python_and_github_actions_without_automerge() -> None:
    dependabot = Path(".github/dependabot.yml").read_text(encoding="utf-8")

    assert "package-ecosystem: pip" in dependabot
    assert "package-ecosystem: github-actions" in dependabot
    assert "interval: weekly" in dependabot
    assert "automerge" not in dependabot.lower()
