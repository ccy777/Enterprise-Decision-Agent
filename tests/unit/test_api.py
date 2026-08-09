"""Offline tests for process liveness and honest readiness reporting."""

from fastapi.testclient import TestClient

from decision_agent.api import create_app
from decision_agent.config import Environment, Settings


def make_settings(required_dependencies: list[str] | None = None) -> Settings:
    return Settings(
        app_name="Test Agent",
        environment=Environment.TEST,
        required_dependencies=required_dependencies or [],
        _env_file=None,
    )


def test_health_is_live_without_external_dependencies() -> None:
    with TestClient(create_app(make_settings())) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_is_ready_when_no_dependencies_are_required() -> None:
    with TestClient(create_app(make_settings())) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "dependencies": {}}


def test_ready_reports_missing_check_as_unavailable() -> None:
    app = create_app(make_settings(["vector-store"]))

    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "dependencies": {"vector-store": False},
    }


def test_ready_reports_false_check_as_unavailable() -> None:
    app = create_app(make_settings(["vector-store"]), {"vector-store": lambda: False})

    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "dependencies": {"vector-store": False},
    }


def test_ready_reports_check_exception_as_unavailable() -> None:
    def fail() -> bool:
        raise RuntimeError("probe failed")

    app = create_app(make_settings(["vector-store"]), {"vector-store": fail})

    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "dependencies": {"vector-store": False},
    }


def test_ready_evaluates_injected_check_lazily() -> None:
    calls = 0

    def available() -> bool:
        nonlocal calls
        calls += 1
        return True

    app = create_app(make_settings(["metadata-store"]), {"metadata-store": available})
    assert calls == 0

    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    assert calls == 1
