"""Offline M8C-A coverage for explicit identity and default-deny authorization."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from decision_agent.api import create_app
from decision_agent.application import (
    FormalRequest,
    FormalRequestExecutor,
    MemoryContextStatus,
)
from decision_agent.config import Environment, Settings
from decision_agent.context.conversation_memory import ConversationMemoryProjector
from decision_agent.coordination import Coordinator
from decision_agent.coordination.models import SkillResult, SkillStatus
from decision_agent.routing.models import RequestRoute, RouterDecision
from decision_agent.security import (
    AuthenticationMethod,
    DataScope,
    DefaultDenyAuthorizationPolicy,
    KnowledgeScope,
    PrincipalType,
    RequestPrincipal,
    SecurityAuthorizationError,
    SecurityContext,
    SecurityErrorCode,
    SessionScope,
    build_security_context,
    make_system_principal,
    make_test_principal,
)
from decision_agent.skills.contracts import SkillDefinition
from decision_agent.skills.registry import SkillRegistry

pytestmark = pytest.mark.offline_integration


def _context(
    *,
    scenarios: frozenset[str] = frozenset({"knowledge"}),
    workflows: frozenset[str] = frozenset({"direct"}),
    skills: frozenset[str] = frozenset({"enterprise-knowledge-qa"}),
    tools: frozenset[str] = frozenset({"run_knowledge_agent"}),
) -> SecurityContext:
    return build_security_context(
        principal=make_test_principal(
            subject_id="test-subject",
            tenant_id="tenant-a",
            roles=frozenset({"knowledge_reader"}),
        ),
        request_id="request-1",
        trace_id="trace-1",
        allowed_scenarios=scenarios,
        allowed_workflows=workflows,
        allowed_skills=skills,
        allowed_tools=tools,
        data_scope=DataScope(
            tenant_id="tenant-a",
            allowed_domains=frozenset({"enterprise_operations"}),
            allowed_resources=frozenset({"products"}),
            allowed_query_capabilities=frozenset({"read"}),
        ),
        knowledge_scope=KnowledgeScope(
            tenant_id="tenant-a",
            allowed_namespaces=frozenset({"enterprise_kb"}),
            allowed_document_ids=frozenset({"DOC-TEST-001"}),
        ),
        session_scope=SessionScope(tenant_id="tenant-a", subject_id="test-subject"),
    )


def _decision(route: RequestRoute = RequestRoute.KNOWLEDGE) -> RouterDecision:
    return RouterDecision(
        route=route,
        normalized_query="inventory policy",
        decision_reason="test route",
        knowledge_subquery="inventory policy" if route is RequestRoute.KNOWLEDGE else None,
        data_subquery="inventory level" if route is RequestRoute.DATA else None,
        missing_information=None,
        confidence=1.0,
    )


@dataclass
class _Router:
    decision: RouterDecision
    calls: int = 0

    async def route(self, user_query: str) -> RouterDecision:
        del user_query
        self.calls += 1
        return self.decision


class _Skill:
    _definition = SkillDefinition(
        name="enterprise-knowledge-qa",
        version="1.0.0",
        description="test-only trusted skill",
        supported_route=RequestRoute.KNOWLEDGE,
        input_contract=("user_query",),
        allowed_tools=("run_knowledge_agent",),
        steps=("execute",),
        output_contract=("answer",),
        failure_codes=("skill_failed",),
    )

    def __init__(self) -> None:
        self.calls = 0

    @property
    def definition(self) -> SkillDefinition:
        return self._definition

    def is_applicable(self, request: str, decision: RouterDecision) -> bool:
        return bool(request) and decision.route is RequestRoute.KNOWLEDGE

    async def execute(self, *, user_query: str, decision: RouterDecision) -> SkillResult:
        del user_query
        self.calls += 1
        return SkillResult(
            status=SkillStatus.COMPLETED,
            skill_name=self.definition.name,
            skill_version=self.definition.version,
            route=decision.route,
            answer="safe answer",
            citations=["[E1]"],
            executed_steps=("execute",),
            selected_tool="run_knowledge_agent",
        )


def _coordinator(router: _Router, skill: _Skill) -> Coordinator:
    registry = SkillRegistry()
    registry.register(skill)
    return Coordinator(router=router, registry=registry)


@pytest.mark.asyncio
async def test_missing_context_blocks_before_router_or_skill() -> None:
    router, skill = _Router(_decision()), _Skill()
    executor = FormalRequestExecutor(
        coordinator=_coordinator(router, skill),
        memory_store=None,
        memory_projector=ConversationMemoryProjector(),
        authorization_policy=DefaultDenyAuthorizationPolicy(),
    )

    response = await executor.execute(FormalRequest(request_id="request-1", user_query="query"))

    assert response.result.error_code == SecurityErrorCode.UNAUTHENTICATED.value
    assert response.memory_context_status is MemoryContextStatus.NOT_REQUESTED
    assert router.calls == skill.calls == 0


@pytest.mark.asyncio
async def test_missing_context_blocks_before_session_memory_read() -> None:
    class MemoryStore:
        read_calls = 0

        def read(self, session_id: str) -> object:
            del session_id
            self.read_calls += 1
            raise AssertionError("authorization must run before memory access")

    router, skill = _Router(_decision()), _Skill()
    memory = MemoryStore()
    executor = FormalRequestExecutor(
        coordinator=_coordinator(router, skill),
        memory_store=memory,  # type: ignore[arg-type]
        memory_projector=ConversationMemoryProjector(),
        authorization_policy=DefaultDenyAuthorizationPolicy(),
    )

    response = await executor.execute(
        FormalRequest(request_id="request-1", session_id="session-1", user_query="query")
    )

    assert response.result.error_code == SecurityErrorCode.UNAUTHENTICATED.value
    assert memory.read_calls == router.calls == skill.calls == 0


@pytest.mark.asyncio
async def test_router_output_cannot_expand_scenario_authority() -> None:
    router, skill = _Router(_decision(RequestRoute.DATA)), _Skill()

    result = await _coordinator(router, skill).execute(
        user_query="query",
        request_id="request-1",
        security_context=_context(),
        authorization_policy=DefaultDenyAuthorizationPolicy(),
    )

    assert result.error_code == SecurityErrorCode.SCENARIO_FORBIDDEN.value
    assert router.calls == 1
    assert skill.calls == 0


@pytest.mark.asyncio
async def test_skill_and_tool_denials_prevent_skill_execution() -> None:
    router, skill = _Router(_decision()), _Skill()
    coordinator = _coordinator(router, skill)

    skill_denied = await coordinator.execute(
        user_query="query",
        request_id="request-1",
        security_context=_context(skills=frozenset({"other-skill"})),
        authorization_policy=DefaultDenyAuthorizationPolicy(),
    )
    tool_denied = await coordinator.execute(
        user_query="query",
        request_id="request-1",
        security_context=_context(tools=frozenset({"other-tool"})),
        authorization_policy=DefaultDenyAuthorizationPolicy(),
    )

    assert skill_denied.error_code == SecurityErrorCode.SKILL_FORBIDDEN.value
    assert tool_denied.error_code == SecurityErrorCode.TOOL_FORBIDDEN.value
    assert skill.calls == 0


@pytest.mark.asyncio
async def test_route_scope_denials_prevent_skill_execution() -> None:
    knowledge_router, knowledge_skill = _Router(_decision()), _Skill()
    knowledge_result = await _coordinator(knowledge_router, knowledge_skill).execute(
        user_query="query",
        request_id="request-1",
        security_context=_context().model_copy(update={"knowledge_scope": None}),
        authorization_policy=DefaultDenyAuthorizationPolicy(),
    )
    data_router, data_skill = _Router(_decision(RequestRoute.DATA)), _Skill()
    data_result = await _coordinator(data_router, data_skill).execute(
        user_query="query",
        request_id="request-1",
        security_context=_context(
            scenarios=frozenset({"data"}),
        ).model_copy(update={"data_scope": None}),
        authorization_policy=DefaultDenyAuthorizationPolicy(),
    )

    assert knowledge_result.error_code == SecurityErrorCode.KNOWLEDGE_SCOPE_MISSING.value
    assert data_result.error_code == SecurityErrorCode.DATA_SCOPE_MISSING.value
    assert knowledge_skill.calls == data_skill.calls == 0


@pytest.mark.asyncio
async def test_workflow_denial_prevents_skill_and_authorized_direct_path_is_unchanged() -> None:
    router, skill = _Router(_decision()), _Skill()
    coordinator = _coordinator(router, skill)

    denied = await coordinator.execute(
        user_query="query",
        request_id="request-1",
        security_context=_context(workflows=frozenset({"controlled_mixed"})),
        authorization_policy=DefaultDenyAuthorizationPolicy(),
    )
    allowed = await coordinator.execute(
        user_query="query",
        request_id="request-1",
        security_context=_context(),
        authorization_policy=DefaultDenyAuthorizationPolicy(),
    )

    assert denied.error_code == SecurityErrorCode.WORKFLOW_FORBIDDEN.value
    assert allowed.status.value == "completed"
    assert skill.calls == 1


@pytest.mark.asyncio
async def test_explicit_test_and_system_principals_have_only_explicit_grants() -> None:
    test_context = _context()
    system_context = build_security_context(
        principal=make_system_principal(
            subject_id="bootstrap",
            tenant_id="tenant-a",
            roles=frozenset({"runtime_operator"}),
        ),
        request_id="request-2",
        trace_id="trace-2",
        allowed_scenarios=frozenset({"knowledge"}),
        allowed_workflows=frozenset({"direct"}),
        allowed_skills=frozenset({"enterprise-knowledge-qa"}),
        allowed_tools=frozenset({"run_knowledge_agent"}),
    )
    policy = DefaultDenyAuthorizationPolicy()

    assert policy.require_tool(test_context, "run_knowledge_agent").decision == "allowed"
    assert policy.require_skill(system_context, "enterprise-knowledge-qa").decision == "allowed"
    with pytest.raises(SecurityAuthorizationError, match=SecurityErrorCode.TOOL_FORBIDDEN.value):
        policy.require_tool(system_context, "run_data_agent")


def test_privileged_principal_types_require_their_explicit_factories() -> None:
    with pytest.raises(ValueError, match="system principals require"):
        RequestPrincipal(
            principal_type=PrincipalType.SYSTEM,
            subject_id="bootstrap",
            tenant_id="tenant-a",
            roles=frozenset({"runtime_operator"}),
            authentication_method=AuthenticationMethod.INTERNAL_SYSTEM,
        )
    with pytest.raises(ValueError, match="test principals require"):
        RequestPrincipal(
            principal_type=PrincipalType.TEST,
            subject_id="test-subject",
            tenant_id="tenant-a",
            roles=frozenset({"knowledge_reader"}),
            authentication_method=AuthenticationMethod.TEST_FIXTURE,
        )


def test_security_event_is_payload_free_and_tenant_is_hashed() -> None:
    event = DefaultDenyAuthorizationPolicy().require_skill(_context(), "enterprise-knowledge-qa")
    serialized = event.model_dump_json()

    assert "tenant-a" not in serialized
    assert "query" not in serialized
    assert set(event.model_dump()) == {
        "request_id",
        "trace_id",
        "principal_type",
        "tenant_id_digest",
        "action",
        "resource_type",
        "decision",
        "policy_id",
        "policy_version",
        "scope_version",
        "error_code",
    }


def test_scope_policy_rejects_tenant_and_session_binding_mismatches() -> None:
    policy = DefaultDenyAuthorizationPolicy()
    context = _context().model_copy(
        update={
            "data_scope": DataScope(
                tenant_id="tenant-b",
                allowed_domains=frozenset({"enterprise_operations"}),
                allowed_resources=frozenset({"products"}),
                allowed_query_capabilities=frozenset({"read"}),
            ),
            "session_scope": SessionScope(tenant_id="tenant-a", subject_id="other-subject"),
        }
    )

    with pytest.raises(
        SecurityAuthorizationError, match=SecurityErrorCode.TENANT_SCOPE_MISMATCH.value
    ):
        policy.require_data_scope(context, "enterprise_operations")
    with pytest.raises(
        SecurityAuthorizationError, match=SecurityErrorCode.SESSION_SCOPE_VIOLATION.value
    ):
        policy.require_session_scope(context)


def test_scoped_session_keys_are_opaque_and_principal_bound() -> None:
    first = _context()
    second = build_security_context(
        principal=make_test_principal(
            subject_id="other-subject",
            tenant_id="tenant-a",
            roles=frozenset({"knowledge_reader"}),
        ),
        request_id="request-2",
        trace_id="trace-2",
        allowed_scenarios=frozenset({"knowledge"}),
        allowed_workflows=frozenset({"direct"}),
        allowed_skills=frozenset({"enterprise-knowledge-qa"}),
        allowed_tools=frozenset({"run_knowledge_agent"}),
        session_scope=SessionScope(tenant_id="tenant-a", subject_id="other-subject"),
    )

    first_key = first.scoped_session_key("shared-label")
    second_key = second.scoped_session_key("shared-label")

    assert first_key != second_key
    assert len(first_key) == len(second_key) == 64
    assert "shared-label" not in first_key + second_key


class _SecurityRequiredExecutor:
    requires_security_context = True

    async def execute(self, request: FormalRequest) -> object:  # pragma: no cover - must not run
        raise AssertionError(request)


def test_api_default_identity_adapter_fails_closed_without_calling_executor() -> None:
    app = create_app(
        Settings(environment=Environment.TEST, required_dependencies=[], _env_file=None),
        formal_request_executor=_SecurityRequiredExecutor(),  # type: ignore[arg-type]
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/agent/execute",
            json={"request_id": "request-1", "query": "query"},
        )

    assert response.status_code == 401
    assert response.json() == {
        "code": SecurityErrorCode.UNAUTHENTICATED.value,
        "message": "The Agent request is not authorized.",
    }


@pytest.mark.parametrize(
    "identity_field",
    [
        "roles",
        "tenant_id",
        "allowed_tools",
        "principal",
        "data_scope",
        "knowledge_scope",
        "session_scope",
    ],
)
def test_api_rejects_self_asserted_identity_fields_before_resolver(
    identity_field: str,
) -> None:
    app = create_app(
        Settings(environment=Environment.TEST, required_dependencies=[], _env_file=None),
        formal_request_executor=_SecurityRequiredExecutor(),  # type: ignore[arg-type]
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/agent/execute",
            json={
                "request_id": "request-1",
                "query": "query",
                identity_field: "untrusted-input",
            },
        )

    assert response.status_code == 422
