"""Immutable, payload-safe security contracts for formal request execution."""

from __future__ import annotations

from enum import StrEnum
from hashlib import sha256

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationInfo,
    field_validator,
    model_validator,
)

from decision_agent.exceptions import DecisionAgentError

_IDENTIFIER_MAX_LENGTH = 128
_NAME_MAX_LENGTH = 120
_POLICY_ID = "m8c-default-deny"
_POLICY_VERSION = "1"
_SYSTEM_FACTORY_MARKER = object()
_TEST_FACTORY_MARKER = object()


class PrincipalType(StrEnum):
    """Trusted origins accepted by the M8C-A security boundary."""

    HUMAN = "human"
    SYSTEM = "system"
    TEST = "test"


class AuthenticationMethod(StrEnum):
    """Authentication provenance; no method itself grants a capability."""

    UPSTREAM_ASSERTED = "upstream_asserted"
    INTERNAL_SYSTEM = "internal_system"
    TEST_FIXTURE = "test_fixture"


class SecurityErrorCode(StrEnum):
    """Stable, content-safe authorization outcomes."""

    UNAUTHENTICATED = "unauthenticated"
    SECURITY_CONTEXT_INVALID = "security_context_invalid"
    TENANT_CONTEXT_MISSING = "tenant_context_missing"
    SCENARIO_FORBIDDEN = "scenario_forbidden"
    WORKFLOW_FORBIDDEN = "workflow_forbidden"
    SKILL_FORBIDDEN = "skill_forbidden"
    TOOL_FORBIDDEN = "tool_forbidden"
    DATA_SCOPE_MISSING = "data_scope_missing"
    DATA_SCOPE_INVALID = "data_scope_invalid"
    DATA_SCOPE_VIOLATION = "data_scope_violation"
    KNOWLEDGE_SCOPE_MISSING = "knowledge_scope_missing"
    KNOWLEDGE_SCOPE_INVALID = "knowledge_scope_invalid"
    KNOWLEDGE_SCOPE_VIOLATION = "knowledge_scope_violation"
    TENANT_SCOPE_MISMATCH = "tenant_scope_mismatch"
    SESSION_SCOPE_VIOLATION = "session_scope_violation"
    EVIDENCE_SCOPE_VIOLATION = "evidence_scope_violation"
    CITATION_SCOPE_VIOLATION = "citation_scope_violation"


class RequestPrincipal(BaseModel):
    """Verified identity data, never projected to prompt, trace, or API response."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    principal_type: PrincipalType
    subject_id: str = Field(min_length=1, max_length=_IDENTIFIER_MAX_LENGTH, repr=False)
    tenant_id: str = Field(min_length=1, max_length=_IDENTIFIER_MAX_LENGTH, repr=False)
    roles: frozenset[str] = Field(min_length=1, max_length=16, repr=False)
    authentication_method: AuthenticationMethod
    _factory_marker: object | None = PrivateAttr(default=None)

    @field_validator("subject_id", "tenant_id")
    @classmethod
    def _validate_identifier(cls, value: str) -> str:
        return _validate_text(value, "identity identifier")

    @field_validator("roles")
    @classmethod
    def _validate_roles(cls, value: frozenset[str]) -> frozenset[str]:
        if not value:
            raise ValueError("roles must not be empty")
        return frozenset(_validate_text(role, "role") for role in value)

    @model_validator(mode="after")
    def _validate_provenance(self, info: ValidationInfo) -> RequestPrincipal:
        """Keep privileged principal types behind their explicit construction paths."""
        expected_authentication = {
            PrincipalType.HUMAN: AuthenticationMethod.UPSTREAM_ASSERTED,
            PrincipalType.SYSTEM: AuthenticationMethod.INTERNAL_SYSTEM,
            PrincipalType.TEST: AuthenticationMethod.TEST_FIXTURE,
        }[self.principal_type]
        if self.authentication_method is not expected_authentication:
            raise ValueError("principal type and authentication method are incompatible")
        marker = self._factory_marker
        if info.context is not None:
            marker = info.context.get("principal_factory_marker", marker)
        if self.principal_type is PrincipalType.SYSTEM and marker is not _SYSTEM_FACTORY_MARKER:
            raise ValueError("system principals require the explicit system factory")
        if self.principal_type is PrincipalType.TEST and marker is not _TEST_FACTORY_MARKER:
            raise ValueError("test principals require the explicit test factory")
        return self


class DataScope(BaseModel):
    """Immutable, server-owned grant for read-only enterprise-data access."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    tenant_id: str = Field(min_length=1, max_length=_IDENTIFIER_MAX_LENGTH, repr=False)
    allowed_domains: frozenset[str] = Field(default_factory=frozenset, max_length=8, repr=False)
    allowed_resources: frozenset[str] = Field(default_factory=frozenset, max_length=32, repr=False)
    allowed_query_capabilities: frozenset[str] = Field(
        default_factory=frozenset, max_length=4, repr=False
    )
    scope_version: str = Field(default="1", min_length=1, max_length=40)

    @field_validator("tenant_id", "scope_version")
    @classmethod
    def _validate_text_field(cls, value: str) -> str:
        return _validate_text(value, "data scope field")

    @field_validator("allowed_domains", "allowed_resources", "allowed_query_capabilities")
    @classmethod
    def _validate_items(cls, value: frozenset[str]) -> frozenset[str]:
        return frozenset(_validate_text(item, "data scope grant") for item in value)

    def permits(self, *, domain: str, resource: str | None = None) -> bool:
        """Return whether this explicit scope grants one fixed data access."""
        return (
            domain in self.allowed_domains
            and "read" in self.allowed_query_capabilities
            and (resource is None or resource in self.allowed_resources)
        )


class KnowledgeScope(BaseModel):
    """Immutable, server-owned grant for document retrieval and citations."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    tenant_id: str = Field(min_length=1, max_length=_IDENTIFIER_MAX_LENGTH, repr=False)
    allowed_namespaces: frozenset[str] = Field(default_factory=frozenset, max_length=16, repr=False)
    allowed_document_ids: frozenset[str] = Field(
        default_factory=frozenset, max_length=128, repr=False
    )
    scope_version: str = Field(default="1", min_length=1, max_length=40)

    @field_validator("tenant_id", "scope_version")
    @classmethod
    def _validate_text_field(cls, value: str) -> str:
        return _validate_text(value, "knowledge scope field")

    @field_validator("allowed_namespaces", "allowed_document_ids")
    @classmethod
    def _validate_items(cls, value: frozenset[str]) -> frozenset[str]:
        return frozenset(_validate_text(item, "knowledge scope grant") for item in value)

    def permits_document(self, document_id: str) -> bool:
        """Return whether a retrieval result may enter Evidence or citations."""
        return document_id in self.allowed_document_ids


class SessionScope(BaseModel):
    """Bind a caller-provided session label to exactly one tenant and principal."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    tenant_id: str = Field(min_length=1, max_length=_IDENTIFIER_MAX_LENGTH, repr=False)
    subject_id: str = Field(min_length=1, max_length=_IDENTIFIER_MAX_LENGTH, repr=False)
    scope_version: str = Field(default="1", min_length=1, max_length=40)

    @field_validator("tenant_id", "subject_id", "scope_version")
    @classmethod
    def _validate_text_field(cls, value: str) -> str:
        return _validate_text(value, "session scope field")


class SecurityContext(BaseModel):
    """Immutable server-owned authorization input propagated through one request."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    principal: RequestPrincipal = Field(repr=False)
    request_id: str = Field(min_length=1, max_length=_IDENTIFIER_MAX_LENGTH)
    trace_id: str = Field(min_length=1, max_length=_IDENTIFIER_MAX_LENGTH)
    allowed_scenarios: frozenset[str] = Field(min_length=1, max_length=8, repr=False)
    allowed_workflows: frozenset[str] = Field(min_length=1, max_length=8, repr=False)
    allowed_skills: frozenset[str] = Field(min_length=1, max_length=16, repr=False)
    allowed_tools: frozenset[str] = Field(min_length=1, max_length=16, repr=False)
    data_scope: DataScope | None = Field(default=None, repr=False)
    knowledge_scope: KnowledgeScope | None = Field(default=None, repr=False)
    session_scope: SessionScope | None = Field(default=None, repr=False)
    policy_id: str = Field(default=_POLICY_ID, min_length=1, max_length=_NAME_MAX_LENGTH)
    policy_version: str = Field(default=_POLICY_VERSION, min_length=1, max_length=40)

    @field_validator(
        "request_id",
        "trace_id",
        "policy_id",
        "policy_version",
    )
    @classmethod
    def _validate_text_field(cls, value: str) -> str:
        return _validate_text(value, "security context field")

    @field_validator(
        "allowed_scenarios",
        "allowed_workflows",
        "allowed_skills",
        "allowed_tools",
    )
    @classmethod
    def _validate_grants(cls, value: frozenset[str]) -> frozenset[str]:
        if not value:
            raise ValueError("authorization grants must not be empty")
        return frozenset(_validate_text(item, "authorization grant") for item in value)

    @property
    def tenant_id_digest(self) -> str:
        """Return a non-reversible tenant identifier suitable for a future audit event."""
        return _digest(self.principal.tenant_id)

    def scoped_session_key(self, session_id: str) -> str:
        """Derive an opaque Store key that cannot collide across tenant/principal pairs."""
        if self.session_scope is None:
            raise SecurityAuthorizationError(SecurityErrorCode.SESSION_SCOPE_VIOLATION)
        return _digest(
            "\x1f".join(
                (
                    self.session_scope.tenant_id,
                    self.session_scope.subject_id,
                    _validate_text(session_id, "session identifier"),
                )
            )
        )


class SecurityEvent(BaseModel):
    """Safe authorization decision record for M8C-C audit persistence."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    request_id: str = Field(min_length=1, max_length=_IDENTIFIER_MAX_LENGTH)
    trace_id: str = Field(min_length=1, max_length=_IDENTIFIER_MAX_LENGTH)
    principal_type: PrincipalType
    tenant_id_digest: str = Field(min_length=16, max_length=64)
    action: str = Field(min_length=1, max_length=_NAME_MAX_LENGTH)
    resource_type: str = Field(min_length=1, max_length=40)
    decision: str = Field(pattern="^(allowed|denied)$")
    policy_id: str = Field(min_length=1, max_length=_NAME_MAX_LENGTH)
    policy_version: str = Field(min_length=1, max_length=40)
    scope_version: str | None = Field(default=None, min_length=1, max_length=40)
    error_code: SecurityErrorCode | None = None


class SecurityAuthorizationError(DecisionAgentError):
    """Stable denial that never carries policy internals or caller content."""

    def __init__(self, code: SecurityErrorCode) -> None:
        self.code = code.value
        super().__init__(self.code)


def make_system_principal(
    *,
    subject_id: str,
    tenant_id: str,
    roles: frozenset[str],
) -> RequestPrincipal:
    """Create an explicit narrow system principal; callers still supply grants."""
    principal = RequestPrincipal.model_validate(
        {
            "principal_type": PrincipalType.SYSTEM,
            "subject_id": subject_id,
            "tenant_id": tenant_id,
            "roles": roles,
            "authentication_method": AuthenticationMethod.INTERNAL_SYSTEM,
        },
        context={"principal_factory_marker": _SYSTEM_FACTORY_MARKER},
    )
    principal._factory_marker = _SYSTEM_FACTORY_MARKER
    return principal


def make_test_principal(
    *,
    subject_id: str,
    tenant_id: str,
    roles: frozenset[str],
) -> RequestPrincipal:
    """Create an explicit test principal without granting any implicit capability."""
    principal = RequestPrincipal.model_validate(
        {
            "principal_type": PrincipalType.TEST,
            "subject_id": subject_id,
            "tenant_id": tenant_id,
            "roles": roles,
            "authentication_method": AuthenticationMethod.TEST_FIXTURE,
        },
        context={"principal_factory_marker": _TEST_FACTORY_MARKER},
    )
    principal._factory_marker = _TEST_FACTORY_MARKER
    return principal


def build_security_context(
    *,
    principal: RequestPrincipal,
    request_id: str,
    trace_id: str,
    allowed_scenarios: frozenset[str],
    allowed_workflows: frozenset[str],
    allowed_skills: frozenset[str],
    allowed_tools: frozenset[str],
    data_scope: DataScope | None = None,
    knowledge_scope: KnowledgeScope | None = None,
    session_scope: SessionScope | None = None,
) -> SecurityContext:
    """Build an explicit context for trusted API, CLI, system, or test entrypoints."""
    return SecurityContext(
        principal=principal,
        request_id=request_id,
        trace_id=trace_id,
        allowed_scenarios=allowed_scenarios,
        allowed_workflows=allowed_workflows,
        allowed_skills=allowed_skills,
        allowed_tools=allowed_tools,
        data_scope=data_scope,
        knowledge_scope=knowledge_scope,
        session_scope=session_scope,
    )


def _validate_text(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty")
    normalized = value.strip()
    if any(ord(character) < 32 or ord(character) == 127 for character in normalized):
        raise ValueError(f"{label} must not contain control characters")
    return normalized


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()
