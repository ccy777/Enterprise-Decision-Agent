from __future__ import annotations

from pathlib import Path

import pytest

from decision_agent.application.bootstrap import RuntimeBootstrapError
from decision_agent.config import Environment, Settings
from decision_agent.demo import (
    DemoCase,
    build_demo_request,
    build_demo_security_context,
    prepare_demo_settings,
    run_demo,
)
from decision_agent.security import PrincipalType


def test_demo_principal_and_scopes_are_fixed_and_least_privilege() -> None:
    request = build_demo_request(
        DemoCase.MIXED,
        request_id="local-demo-request",
        trace_id="local-demo-trace",
    )
    context = request.security_context

    assert context is not None
    assert context.principal.principal_type is PrincipalType.SYSTEM
    assert context.principal.roles == frozenset({"local_demo_reader"})
    assert "admin" not in context.principal.roles
    assert context.allowed_scenarios == frozenset({"mixed"})
    assert context.allowed_workflows == frozenset({"controlled_mixed"})
    assert context.allowed_skills == frozenset({"inventory-risk-diagnosis"})
    assert context.allowed_tools == frozenset({"run_data_agent", "run_knowledge_agent"})
    assert context.data_scope is not None
    assert context.data_scope.allowed_query_capabilities == frozenset({"read"})
    assert context.knowledge_scope is not None
    assert context.knowledge_scope.allowed_document_ids == frozenset({"DOC-INV-001"})
    assert request.session_id is None


def test_combined_demo_scope_is_the_union_of_only_frozen_cases() -> None:
    context = build_demo_security_context(
        tuple(DemoCase),
        request_id="local-web-request",
        trace_id="local-web-trace",
    )

    assert context.allowed_scenarios == frozenset({"knowledge", "data", "mixed"})
    assert context.allowed_workflows == frozenset({"direct", "controlled_mixed"})
    assert context.allowed_skills == frozenset(
        {
            "enterprise-knowledge-qa",
            "enterprise-data-analysis",
            "inventory-risk-diagnosis",
        }
    )
    assert context.allowed_tools == frozenset({"run_knowledge_agent", "run_data_agent"})
    assert context.data_scope is not None
    assert context.data_scope.allowed_query_capabilities == frozenset({"read"})
    assert context.data_scope.allowed_resources == frozenset(
        {"products", "inventory_snapshots", "purchase_orders", "suppliers"}
    )
    assert context.knowledge_scope is not None
    assert context.knowledge_scope.allowed_document_ids == frozenset(
        {"DOC-ORG-001", "DOC-AGENT-001", "DOC-INV-001"}
    )
    assert context.session_scope is None


def test_demo_security_context_requires_at_least_one_case() -> None:
    with pytest.raises(ValueError, match="at least one demo case"):
        build_demo_security_context((), request_id="request", trace_id="trace")


@pytest.mark.parametrize("case", list(DemoCase))
def test_demo_accepts_only_closed_cases_without_identity_or_scope_inputs(case: DemoCase) -> None:
    request = build_demo_request(case, request_id="request-fixed", trace_id="trace-fixed")

    assert request.user_query
    assert request.security_context is not None
    assert request.security_context.request_id == request.request_id


def test_demo_audit_path_is_forced_outside_repository(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    external = tmp_path / "external-audit"
    repository.mkdir()
    settings = Settings(environment=Environment.TEST, _env_file=None)

    prepared = prepare_demo_settings(
        settings,
        repository_root=repository,
        audit_root=external,
    )

    assert prepared.audit_log_path is not None
    assert prepared.audit_log_path.parent == external.resolve()
    assert repository.resolve() not in prepared.audit_log_path.resolve().parents
    assert prepared.controlled_workflow_enabled is True


def test_demo_rejects_repository_local_audit_directory(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    settings = Settings(environment=Environment.TEST, _env_file=None)

    with pytest.raises(ValueError, match="outside"):
        prepare_demo_settings(
            settings,
            repository_root=repository,
            audit_root=repository / "audit",
        )


@pytest.mark.asyncio
async def test_demo_missing_formal_configuration_fails_closed_before_external_io(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    external = tmp_path / "audit"
    repository.mkdir()
    settings = prepare_demo_settings(
        Settings(environment=Environment.DEVELOPMENT, _env_file=None),
        repository_root=repository,
        audit_root=external,
    )

    with pytest.raises(RuntimeBootstrapError) as raised:
        await run_demo(DemoCase.KNOWLEDGE, settings=settings)

    assert raised.value.code == "bootstrap_configuration_invalid"
