from __future__ import annotations

from starlette.requests import Request

from decision_agent.demo import DemoCase
from decision_agent.demo.web import LocalDemoSecurityContextResolver
from decision_agent.security import SecurityAuthorizationError, SecurityErrorCode


def _request(client_host: str | None) -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/agent/execute",
        "headers": [],
        "client": (client_host, 50000) if client_host is not None else None,
        "server": ("127.0.0.1", 8000),
        "scheme": "http",
        "query_string": b"",
    }
    return Request(scope)


def test_local_demo_resolver_accepts_only_loopback_and_builds_server_owned_context() -> None:
    context = LocalDemoSecurityContextResolver(DemoCase.MIXED).resolve(
        request=_request("127.0.0.1"),
        request_id="browser-request",
    )

    assert context.request_id == "browser-request"
    assert context.principal.subject_id == "local-demo-principal"
    assert context.allowed_scenarios == frozenset({"mixed"})
    assert context.allowed_workflows == frozenset({"controlled_mixed"})
    assert context.allowed_skills == frozenset({"inventory-risk-diagnosis"})
    assert context.allowed_tools == frozenset({"run_data_agent", "run_knowledge_agent"})
    assert context.data_scope is not None
    assert context.data_scope.allowed_query_capabilities == frozenset({"read"})
    assert context.data_scope.allowed_resources == frozenset(
        {"products", "inventory_snapshots", "purchase_orders", "suppliers"}
    )
    assert context.knowledge_scope is not None
    assert context.knowledge_scope.allowed_document_ids == frozenset({"DOC-INV-001"})
    assert context.session_scope is not None
    assert context.session_scope.tenant_id == context.principal.tenant_id
    assert context.session_scope.subject_id == context.principal.subject_id


def test_local_demo_resolver_rejects_non_loopback_before_runtime() -> None:
    resolver = LocalDemoSecurityContextResolver(DemoCase.MIXED)

    try:
        resolver.resolve(request=_request("192.0.2.10"), request_id="remote-request")
    except SecurityAuthorizationError as exc:
        assert exc.code == SecurityErrorCode.UNAUTHENTICATED.value
    else:  # pragma: no cover - explicit fail-closed assertion
        raise AssertionError("non-loopback request was accepted")
