"""Unit coverage for the package-local Module 7 demonstration UI."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from decision_agent import web as demo_web
from decision_agent.api import create_app
from decision_agent.config import Environment, Settings


@pytest.fixture(autouse=True)
def _isolate_settings_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for field_name in Settings.model_fields:
        monkeypatch.delenv(f"DECISION_AGENT_{field_name.upper()}", raising=False)


def _settings() -> Settings:
    return Settings(
        app_name="Module 7 Demo UI Test",
        environment=Environment.TEST,
        required_dependencies=[],
        _env_file=None,
    )


def _client() -> TestClient:
    return TestClient(create_app(_settings()))


def test_root_and_package_local_assets_are_served() -> None:
    with _client() as client:
        root = client.get("/")
        styles = client.get("/assets/styles.css")
        script = client.get("/assets/app.js")

    assert root.status_code == styles.status_code == script.status_code == 200
    assert root.headers["content-type"].startswith("text/html")
    assert styles.headers["content-type"].startswith("text/css")
    assert "javascript" in script.headers["content-type"]
    expected_styles = (Path(demo_web.__file__).parent / "styles.css").read_text(encoding="utf-8")
    assert styles.text
    assert styles.text.replace("\r\n", "\n") == expected_styles
    assert "<title>企业决策 AI Agent</title>" in root.text
    assert '<link rel="stylesheet" href="/assets/styles.css" />' in root.text
    assert "--brand:" in styles.text
    for selector in (
        ".app-header",
        ".header-inner",
        ".page-shell",
        ".workspace-card",
        ".result-card",
        "@media (max-width: 860px)",
    ):
        assert selector in styles.text
    assert '"use strict"' in script.text


def test_app_construction_does_not_read_static_files(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_static_read(path: Path, *_: object, **__: object) -> str:
        if "decision_agent" in path.parts and "web" in path.parts:
            raise AssertionError("static file read during app construction")
        return ""

    monkeypatch.setattr(Path, "read_text", fail_static_read)
    monkeypatch.setattr(Path, "read_bytes", fail_static_read)
    monkeypatch.setattr(Path, "open", fail_static_read)

    app = create_app(_settings())

    assert app.title == "Module 7 Demo UI Test"


def test_html_has_accessible_form_status_and_result_regions() -> None:
    with _client() as client:
        html = client.get("/").text

    assert 'lang="zh-CN"' in html
    assert 'for="query"' in html
    assert 'id="query-form"' in html
    assert 'aria-live="polite"' in html
    assert 'aria-label="服务状态"' in html
    assert 'class="page-shell"' in html
    assert 'class="status-panel"' in html
    assert 'id="answer"' in html
    assert 'id="citations"' in html
    assert 'id="metadata"' in html
    assert 'id="cancel-request"' in html
    assert 'id="trace-card"' in html
    assert 'id="trace-stages"' in html
    assert "自动判断 Knowledge、Data 或 Mixed 路由" in html


def test_script_uses_only_the_formal_api_contract_for_requests() -> None:
    with _client() as client:
        script = client.get("/assets/app.js").text

    assert 'const HEALTH_ENDPOINT = "/health"' in script
    assert 'const READY_ENDPOINT = "/ready"' in script
    assert 'const EXECUTE_ENDPOINT = "/api/v1/agent/execute"' in script
    assert "const payload = { request_id: requestId, session_id: sessionId, query };" in script
    payload_start = script.index("const payload =")
    payload_end = script.index(";", payload_start)
    assert "route" not in script[payload_start:payload_end]
    assert "JSON.stringify(payload)" in script


def test_session_request_identity_and_controls_are_browser_local() -> None:
    with _client() as client:
        script = client.get("/assets/app.js").text

    assert "localStorage.getItem(SESSION_STORAGE_KEY)" in script
    assert "localStorage.setItem(SESSION_STORAGE_KEY, nextSession)" in script
    assert "crypto.randomUUID" in script
    assert "crypto.getRandomValues" in script
    assert 'randomIdentifier("request")' in script
    assert 'randomIdentifier("session")' in script
    assert "activeController !== null" in script
    assert "createAndStoreSession()" in script
    assert "session_id=" not in script


def test_status_polling_visibility_and_abort_are_bounded() -> None:
    with _client() as client:
        script = client.get("/assets/app.js").text

    assert "const STATUS_POLL_MS = 12000" in script
    assert "visibilitychange" in script
    assert "document.hidden" in script
    assert "stopStatusPolling()" in script
    assert "new AbortController()" in script
    assert "activeController.abort()" in script
    assert 'error.name === "AbortError"' in script


def test_dynamic_rendering_uses_text_only_dom_apis() -> None:
    with _client() as client:
        script = client.get("/assets/app.js").text

    forbidden = (
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "document.write",
        "eval(",
        "new Function",
    )
    assert all(token not in script for token in forbidden)
    assert "textContent" in script
    assert "document.createElement" in script
    assert "replaceChildren()" in script


def test_static_assets_have_no_external_or_sensitive_configuration() -> None:
    with _client() as client:
        combined = "\n".join(
            (
                client.get("/").text,
                client.get("/assets/styles.css").text,
                client.get("/assets/app.js").text,
            )
        ).lower()

    forbidden = (
        "sk-",
        "api_key",
        "password",
        "bearer ",
        "private key",
        "http://",
        "https://",
        "c:\\users\\",
        "e:\\ai-agent-study\\",
    )
    assert all(token not in combined for token in forbidden)
    assert "cdn" not in combined
    assert "analytics" not in combined


def test_runtime_unavailable_keeps_ui_and_health_available() -> None:
    with _client() as client:
        root = client.get("/")
        health = client.get("/health")
        response = client.post(
            "/api/v1/agent/execute",
            json={
                "request_id": "demo-runtime-unavailable",
                "session_id": "demo-session",
                "query": "请回答一个企业问题。",
            },
        )

    assert root.status_code == 200
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert response.status_code == 503
    assert response.json() == {
        "code": "runtime_unavailable",
        "message": "The Agent runtime is unavailable.",
    }


def test_invalid_input_is_rejected_without_breaking_the_ui() -> None:
    with _client() as client:
        response = client.post(
            "/api/v1/agent/execute",
            json={
                "request_id": "demo-invalid",
                "query": "合法问题",
                "route": "knowledge",
            },
        )
        root = client.get("/")

    assert response.status_code == 422
    assert root.status_code == 200


def test_script_has_stable_public_error_messages_and_complete_response_fields() -> None:
    with _client() as client:
        script = client.get("/assets/app.js").text

    for token in (
        "statusCode === 422",
        "statusCode === 503",
        "statusCode >= 500",
        "data.answer",
        "data.citations",
        "data.route",
        "data.skill",
        "data.error_code",
        "data.memory_context_status",
        "data.memory_persistence_status",
        "data.memory_summarization_status",
        "renderTrace(data.trace)",
    ):
        assert token in script


def test_script_distinguishes_unsupported_from_execution_failure() -> None:
    with _client() as client:
        script = client.get("/assets/app.js").text

    assert 'if (status === "unsupported")' in script
    assert 'label: "暂不支持"' in script
    assert 'message: "当前 Agent 暂不支持此类请求。"' in script
    assert 'presentation.tone === "failed"' in script
    assert 'data.status === "unsupported"' in script


def test_trace_renderer_uses_a_safe_allowlist_and_status_semantics() -> None:
    with _client() as client:
        script = client.get("/assets/app.js").text

    assert "const TRACE_ATTRIBUTE_LABELS = Object.freeze" in script
    assert "const TRACE_STAGE_ALLOWLIST = new Set" in script
    assert "Object.hasOwn(TRACE_ATTRIBUTE_LABELS, attribute.key)" in script
    assert "TRACE_STAGE_ALLOWLIST.has(stage.stage)" in script
    assert "detail.textContent = renderTraceAttributes(stage.attributes);" in script
    for forbidden_attribute in ("input_tokens", "output_tokens", "provider", "model"):
        assert forbidden_attribute not in script
    for status in ("completed", "failed", "unsupported", "cancelled", "not_requested"):
        assert f"{status}: {{ label:" in script
    assert "innerHTML" not in script
