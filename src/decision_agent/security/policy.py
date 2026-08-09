"""Central default-deny authorization policy with safe decision events."""

from __future__ import annotations

from typing import Protocol

from decision_agent.security.models import (
    SecurityAuthorizationError,
    SecurityContext,
    SecurityErrorCode,
    SecurityEvent,
)


class AuthorizationPolicy(Protocol):
    """Authorize one already-known capability without reading model output."""

    def require_scenario(self, context: SecurityContext, scenario: str) -> SecurityEvent:
        """Require access to one Router-selected scenario."""

    def require_workflow(self, context: SecurityContext, workflow: str) -> SecurityEvent:
        """Require access to one Coordinator-selected workflow."""

    def require_skill(self, context: SecurityContext, skill: str) -> SecurityEvent:
        """Require access to one registered Skill."""

    def require_tool(self, context: SecurityContext, tool: str) -> SecurityEvent:
        """Require access to one declared high-level Tool."""

    def require_data_scope(self, context: SecurityContext, domain: str) -> SecurityEvent:
        """Require a tenant-bound read grant before Data execution begins."""

    def require_knowledge_scope(self, context: SecurityContext) -> SecurityEvent:
        """Require a tenant-bound document scope before Retrieval begins."""

    def require_session_scope(self, context: SecurityContext) -> SecurityEvent:
        """Require a tenant/principal session binding before Store access."""


class DefaultDenyAuthorizationPolicy:
    """Check immutable grants only; absent grants are never inferred or expanded."""

    def require_scenario(self, context: SecurityContext, scenario: str) -> SecurityEvent:
        return self._require(
            context=context,
            action="enter_scenario",
            resource_type="scenario",
            resource=scenario,
            grants=context.allowed_scenarios,
            denied_code=SecurityErrorCode.SCENARIO_FORBIDDEN,
        )

    def require_workflow(self, context: SecurityContext, workflow: str) -> SecurityEvent:
        return self._require(
            context=context,
            action="enter_workflow",
            resource_type="workflow",
            resource=workflow,
            grants=context.allowed_workflows,
            denied_code=SecurityErrorCode.WORKFLOW_FORBIDDEN,
        )

    def require_skill(self, context: SecurityContext, skill: str) -> SecurityEvent:
        return self._require(
            context=context,
            action="execute_skill",
            resource_type="skill",
            resource=skill,
            grants=context.allowed_skills,
            denied_code=SecurityErrorCode.SKILL_FORBIDDEN,
        )

    def require_tool(self, context: SecurityContext, tool: str) -> SecurityEvent:
        return self._require(
            context=context,
            action="execute_tool",
            resource_type="tool",
            resource=tool,
            grants=context.allowed_tools,
            denied_code=SecurityErrorCode.TOOL_FORBIDDEN,
        )

    def require_data_scope(self, context: SecurityContext, domain: str) -> SecurityEvent:
        self._require_context_binding(context)
        scope = context.data_scope
        if scope is None:
            raise SecurityAuthorizationError(SecurityErrorCode.DATA_SCOPE_MISSING)
        if scope.tenant_id != context.principal.tenant_id:
            raise SecurityAuthorizationError(SecurityErrorCode.TENANT_SCOPE_MISMATCH)
        return self._scope_event(
            context=context,
            action="access_data_scope",
            resource_type="data_scope",
            allowed=scope.permits(domain=domain) and bool(scope.allowed_resources),
            denied_code=SecurityErrorCode.DATA_SCOPE_VIOLATION,
            scope_version=scope.scope_version,
        )

    def require_knowledge_scope(self, context: SecurityContext) -> SecurityEvent:
        self._require_context_binding(context)
        scope = context.knowledge_scope
        if scope is None:
            raise SecurityAuthorizationError(SecurityErrorCode.KNOWLEDGE_SCOPE_MISSING)
        if scope.tenant_id != context.principal.tenant_id:
            raise SecurityAuthorizationError(SecurityErrorCode.TENANT_SCOPE_MISMATCH)
        return self._scope_event(
            context=context,
            action="access_knowledge_scope",
            resource_type="knowledge_scope",
            allowed=bool(scope.allowed_document_ids),
            denied_code=SecurityErrorCode.KNOWLEDGE_SCOPE_VIOLATION,
            scope_version=scope.scope_version,
        )

    def require_session_scope(self, context: SecurityContext) -> SecurityEvent:
        self._require_context_binding(context)
        scope = context.session_scope
        if scope is None:
            raise SecurityAuthorizationError(SecurityErrorCode.SESSION_SCOPE_VIOLATION)
        if (
            scope.tenant_id != context.principal.tenant_id
            or scope.subject_id != context.principal.subject_id
        ):
            raise SecurityAuthorizationError(SecurityErrorCode.SESSION_SCOPE_VIOLATION)
        return self._scope_event(
            context=context,
            action="access_session_scope",
            resource_type="session_scope",
            allowed=True,
            denied_code=SecurityErrorCode.SESSION_SCOPE_VIOLATION,
            scope_version=scope.scope_version,
        )

    def _require(
        self,
        *,
        context: SecurityContext,
        action: str,
        resource_type: str,
        resource: str,
        grants: frozenset[str],
        denied_code: SecurityErrorCode,
    ) -> SecurityEvent:
        if not isinstance(context, SecurityContext):
            raise SecurityAuthorizationError(SecurityErrorCode.SECURITY_CONTEXT_INVALID)
        allowed = resource in grants
        event = SecurityEvent(
            request_id=context.request_id,
            trace_id=context.trace_id,
            principal_type=context.principal.principal_type,
            tenant_id_digest=context.tenant_id_digest,
            action=action,
            resource_type=resource_type,
            decision="allowed" if allowed else "denied",
            policy_id=context.policy_id,
            policy_version=context.policy_version,
            error_code=None if allowed else denied_code,
        )
        if not allowed:
            raise SecurityAuthorizationError(denied_code)
        return event

    @staticmethod
    def _require_context_binding(context: SecurityContext) -> None:
        if not isinstance(context, SecurityContext):
            raise SecurityAuthorizationError(SecurityErrorCode.SECURITY_CONTEXT_INVALID)

    def _scope_event(
        self,
        *,
        context: SecurityContext,
        action: str,
        resource_type: str,
        allowed: bool,
        denied_code: SecurityErrorCode,
        scope_version: str,
    ) -> SecurityEvent:
        event = SecurityEvent(
            request_id=context.request_id,
            trace_id=context.trace_id,
            principal_type=context.principal.principal_type,
            tenant_id_digest=context.tenant_id_digest,
            action=action,
            resource_type=resource_type,
            decision="allowed" if allowed else "denied",
            policy_id=context.policy_id,
            policy_version=context.policy_version,
            scope_version=scope_version,
            error_code=None if allowed else denied_code,
        )
        if not allowed:
            raise SecurityAuthorizationError(denied_code)
        return event
