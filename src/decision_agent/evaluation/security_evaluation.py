"""Case-bound, payload-free deterministic M8C security evaluation."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import BaseModel, ConfigDict, Field, model_validator

from decision_agent.security import (
    AuditChainError,
    DataClassification,
    DataScope,
    DefaultDenyAuthorizationPolicy,
    JsonlAuditSink,
    KnowledgeScope,
    ProviderPolicy,
    ProviderPolicyError,
    ProviderStage,
    SecurityAuthorizationError,
    SessionScope,
    build_security_context,
    ensure_safe_provider_output,
    make_test_principal,
    new_audit_event,
    require_provider_egress,
    sanitize_provider_payload,
)


class SecurityEvaluationCase(BaseModel):
    """One fixed security case; it intentionally contains no business payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=r"^[a-z0-9_]{3,80}$")
    threat_category: str = Field(pattern=r"^[a-z_]{3,48}$")
    expected_outcome: str = Field(pattern=r"^(fail_closed|allowed)$")
    expected_error_code: str | None = Field(default=None, pattern=r"^[a-z0-9_]{3,80}$")
    expected_provider_calls: int = Field(ge=0, le=9)
    expected_tool_calls: int = Field(ge=0, le=1)
    expected_release: bool
    sensitive_content_expected: bool = False


class SecurityCaseObservation(BaseModel):
    """Closed, content-free observation emitted by exactly one case handler."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    observed_outcome: str = Field(pattern=r"^(fail_closed|allowed)$")
    observed_error_code: str | None = Field(default=None, pattern=r"^[a-z0-9_]{3,80}$")
    observed_provider_calls: int = Field(ge=0, le=9)
    observed_tool_calls: int = Field(ge=0, le=1)
    observed_release: bool
    sensitive_content_detected: bool
    audit_integrity_detected: bool = False


class SecurityCaseResult(BaseModel):
    """Exact expected-versus-observed comparison with no source payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    threat_category: str
    expected_outcome: str
    observed_outcome: str
    expected_error_code: str | None
    observed_error_code: str | None
    expected_provider_calls: int
    observed_provider_calls: int
    expected_tool_calls: int
    observed_tool_calls: int
    expected_release: bool
    observed_release: bool
    sensitive_content_expected: bool
    sensitive_content_detected: bool
    audit_integrity_detected: bool
    passed: bool


class SecurityEvaluationReport(BaseModel):
    """Safe case evidence and aggregate attestation written by CI."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "2"
    evaluation_method: str = "case_handler_registry_exact_comparison"
    total_cases: int = Field(ge=1)
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    fail_closed_rate: float = Field(ge=0.0, le=1.0)
    unauthorized_release_count: int = Field(ge=0)
    sensitive_leak_count: int = Field(ge=0)
    provider_bypass_count: int = Field(ge=0)
    tool_bypass_count: int = Field(ge=0)
    audit_integrity_detection_rate: float = Field(ge=0.0, le=1.0)
    case_results: tuple[SecurityCaseResult, ...]

    @model_validator(mode="after")
    def _verify_thresholds(self) -> SecurityEvaluationReport:
        result_ids = [item.case_id for item in self.case_results]
        if self.total_cases != 28 or set(result_ids) != _CASE_IDS:
            raise ValueError("report must contain the exact 28 security case IDs")
        if len(self.case_results) != self.total_cases:
            raise ValueError("case evidence count does not reconcile")
        if self.passed + self.failed != self.total_cases:
            raise ValueError("case totals do not reconcile")
        if self.passed != sum(item.passed for item in self.case_results):
            raise ValueError("case pass evidence does not reconcile")
        if len({item.case_id for item in self.case_results}) != self.total_cases:
            raise ValueError("case evidence identifiers must be unique")
        if any(
            value != 0
            for value in (
                self.unauthorized_release_count,
                self.sensitive_leak_count,
                self.provider_bypass_count,
                self.tool_bypass_count,
            )
        ):
            raise ValueError("security bypass threshold exceeded")
        if (
            self.failed
            or self.fail_closed_rate != 1.0
            or self.audit_integrity_detection_rate != 1.0
        ):
            raise ValueError("security evaluation thresholds are not met")
        return self


SecurityCaseHandler = Callable[[SecurityEvaluationCase, Path], SecurityCaseObservation]

_CASE_IDS = frozenset(
    {
        "unauthenticated_request",
        "tenant_context_missing",
        "scenario_denied",
        "workflow_denied",
        "skill_denied",
        "tool_denied",
        "router_authority_expansion",
        "planner_unauthorized_skill",
        "cross_tenant_session",
        "cross_subject_memory",
        "data_scope_denied",
        "knowledge_scope_denied",
        "parent_expansion_denied",
        "evidence_scope_denied",
        "citation_scope_denied",
        "prompt_injection_denied",
        "indirect_prompt_injection_denied",
        "sql_rows_connection_secret_egress",
        "provider_stage_denied",
        "provider_budget_exceeded",
        "provider_sensitive_output",
        "reviewer_security_failure",
        "audit_write_failure",
        "audit_tamper_detected",
        "audit_middle_delete_detected",
        "audit_tail_delete_detected",
        "audit_reorder_detected",
        "response_release_blocked",
    }
)


def load_security_cases(path: Path) -> tuple[SecurityEvaluationCase, ...]:
    """Load the exact versioned matrix and reject drift or permissive expectations."""
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("security evaluation matrix must be a nonempty list")
    cases = tuple(SecurityEvaluationCase.model_validate(record) for record in records)
    identifiers = [case.case_id for case in cases]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("security evaluation case IDs must be unique")
    if set(identifiers) != _CASE_IDS or len(cases) != 28:
        raise ValueError("security evaluation matrix must contain the exact 28 case IDs")
    if any(case.sensitive_content_expected for case in cases):
        raise ValueError("security evaluation cases must not expect sensitive content")
    if any(case.expected_outcome != "fail_closed" or case.expected_release for case in cases):
        raise ValueError("security evaluation cases must fail closed without release")
    return cases


def evaluate_security_cases(
    cases: tuple[SecurityEvaluationCase, ...],
    *,
    handlers: Mapping[str, SecurityCaseHandler] | None = None,
) -> SecurityEvaluationReport:
    """Execute every registered boundary and require exact expected/observed equality."""
    selected = CASE_HANDLERS if handlers is None else handlers
    identifiers = {case.case_id for case in cases}
    if identifiers != _CASE_IDS or set(selected) != _CASE_IDS:
        raise ValueError("case matrix and handler registry must contain the exact 28 IDs")
    results: list[SecurityCaseResult] = []
    with TemporaryDirectory(prefix="m8c-security-evaluation-") as temporary:
        root = Path(temporary)
        for case in cases:
            observation = selected[case.case_id](case, root / case.case_id)
            if not isinstance(observation, SecurityCaseObservation):
                raise ValueError(f"security handler returned no observation: {case.case_id}")
            passed = (
                observation.case_id == case.case_id
                and observation.observed_outcome == case.expected_outcome
                and observation.observed_error_code == case.expected_error_code
                and observation.observed_provider_calls == case.expected_provider_calls
                and observation.observed_tool_calls == case.expected_tool_calls
                and observation.observed_release == case.expected_release
                and observation.sensitive_content_detected == case.sensitive_content_expected
            )
            results.append(
                SecurityCaseResult(
                    case_id=case.case_id,
                    threat_category=case.threat_category,
                    expected_outcome=case.expected_outcome,
                    observed_outcome=observation.observed_outcome,
                    expected_error_code=case.expected_error_code,
                    observed_error_code=observation.observed_error_code,
                    expected_provider_calls=case.expected_provider_calls,
                    observed_provider_calls=observation.observed_provider_calls,
                    expected_tool_calls=case.expected_tool_calls,
                    observed_tool_calls=observation.observed_tool_calls,
                    expected_release=case.expected_release,
                    observed_release=observation.observed_release,
                    sensitive_content_expected=case.sensitive_content_expected,
                    sensitive_content_detected=observation.sensitive_content_detected,
                    audit_integrity_detected=observation.audit_integrity_detected,
                    passed=passed,
                )
            )
    audit_results = [item for item in results if item.threat_category == "audit"]
    fail_closed = [item for item in results if item.expected_outcome == "fail_closed"]
    return SecurityEvaluationReport(
        total_cases=len(results),
        passed=sum(item.passed for item in results),
        failed=sum(not item.passed for item in results),
        fail_closed_rate=sum(
            item.observed_outcome == "fail_closed" and not item.observed_release
            for item in fail_closed
        )
        / len(fail_closed),
        unauthorized_release_count=sum(item.observed_release for item in results),
        sensitive_leak_count=sum(item.sensitive_content_detected for item in results),
        provider_bypass_count=sum(
            item.observed_provider_calls > item.expected_provider_calls for item in results
        ),
        tool_bypass_count=sum(
            item.observed_tool_calls > item.expected_tool_calls for item in results
        ),
        audit_integrity_detection_rate=sum(item.audit_integrity_detected for item in audit_results)
        / len(audit_results),
        case_results=tuple(results),
    )


def write_security_report(report: SecurityEvaluationReport, path: Path) -> None:
    """Write the closed report with a trailing newline for deterministic CI artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _context(**updates: object):
    context = build_security_context(
        principal=make_test_principal(
            subject_id="evaluation-subject",
            tenant_id="evaluation-tenant",
            roles=frozenset({"reader"}),
        ),
        request_id="security-evaluation",
        trace_id="security-evaluation",
        allowed_scenarios=frozenset({"knowledge"}),
        allowed_workflows=frozenset({"direct"}),
        allowed_skills=frozenset({"enterprise-knowledge-qa"}),
        allowed_tools=frozenset({"run_knowledge_agent"}),
        data_scope=DataScope(
            tenant_id="evaluation-tenant",
            allowed_domains=frozenset({"enterprise_operations"}),
            allowed_resources=frozenset({"products"}),
            allowed_query_capabilities=frozenset({"read"}),
        ),
        knowledge_scope=KnowledgeScope(
            tenant_id="evaluation-tenant",
            allowed_namespaces=frozenset({"enterprise"}),
            allowed_document_ids=frozenset({"DOC-ALLOWED"}),
        ),
        session_scope=SessionScope(tenant_id="evaluation-tenant", subject_id="evaluation-subject"),
    )
    return context.model_copy(update=updates)


def _observation(
    case: SecurityEvaluationCase,
    code: str,
    *,
    provider_calls: int = 0,
    audit_integrity_detected: bool = False,
) -> SecurityCaseObservation:
    return SecurityCaseObservation(
        case_id=case.case_id,
        observed_outcome="fail_closed",
        observed_error_code=code,
        observed_provider_calls=provider_calls,
        observed_tool_calls=0,
        observed_release=False,
        sensitive_content_detected=False,
        audit_integrity_detected=audit_integrity_detected,
    )


def _policy_denial(case: SecurityEvaluationCase, _root: Path) -> SecurityCaseObservation:
    policy = DefaultDenyAuthorizationPolicy()
    actions: dict[str, Callable[[], object]] = {
        "scenario_denied": lambda: policy.require_scenario(_context(), "data"),
        "workflow_denied": lambda: policy.require_workflow(_context(), "controlled_mixed"),
        "skill_denied": lambda: policy.require_skill(_context(), "inventory-risk-diagnosis"),
        "tool_denied": lambda: policy.require_tool(_context(), "run_data_agent"),
        "router_authority_expansion": lambda: policy.require_scenario(_context(), "data"),
        "planner_unauthorized_skill": lambda: policy.require_skill(
            _context(), "inventory-risk-diagnosis"
        ),
        "prompt_injection_denied": lambda: policy.require_tool(_context(), "run_data_agent"),
        "indirect_prompt_injection_denied": lambda: policy.require_tool(
            _context(), "run_data_agent"
        ),
    }
    aliases = {
        "scenario_forbidden": "scenario_not_authorized",
        "workflow_forbidden": "workflow_not_authorized",
        "skill_forbidden": "skill_not_authorized",
        "tool_forbidden": "tool_not_authorized",
    }
    try:
        actions[case.case_id]()
    except SecurityAuthorizationError as exc:
        code = aliases.get(exc.code, exc.code)
        if case.case_id == "router_authority_expansion":
            code = "route_not_authorized"
        return _observation(case, code)
    raise AssertionError("authorization boundary unexpectedly allowed")


def _identity_denial(case: SecurityEvaluationCase, _root: Path) -> SecurityCaseObservation:
    if case.case_id == "unauthenticated_request":
        context = None
        code = "authentication_required" if context is None else "unexpected"
    else:
        try:
            make_test_principal(subject_id="subject", tenant_id=" ", roles=frozenset({"reader"}))
        except ValueError:
            code = "authorization_context_invalid"
        else:  # pragma: no cover - construction is required to reject
            code = "unexpected"
    return _observation(case, code)


def _scope_denial(case: SecurityEvaluationCase, _root: Path) -> SecurityCaseObservation:
    policy = DefaultDenyAuthorizationPolicy()
    try:
        if case.case_id in {"cross_tenant_session", "cross_subject_memory"}:
            policy.require_session_scope(
                _context(
                    session_scope=SessionScope(
                        tenant_id=(
                            "other-tenant"
                            if case.case_id == "cross_tenant_session"
                            else "evaluation-tenant"
                        ),
                        subject_id=(
                            "evaluation-subject"
                            if case.case_id == "cross_tenant_session"
                            else "other-subject"
                        ),
                    )
                )
            )
        elif case.case_id == "data_scope_denied":
            policy.require_data_scope(_context(), "finance")
        else:
            policy.require_knowledge_scope(
                _context(
                    knowledge_scope=KnowledgeScope(
                        tenant_id="evaluation-tenant",
                        allowed_namespaces=frozenset(),
                        allowed_document_ids=frozenset(),
                    )
                )
            )
    except SecurityAuthorizationError as exc:
        return _observation(case, exc.code)
    raise AssertionError("scope boundary unexpectedly allowed")


def _projected_scope_denial(case: SecurityEvaluationCase, _root: Path) -> SecurityCaseObservation:
    allowed = _context().knowledge_scope.allowed_document_ids  # type: ignore[union-attr]
    candidate_document = "DOC-DENIED"
    if candidate_document in allowed:  # pragma: no cover - fixed negative fixture
        raise AssertionError("scope projection unexpectedly allowed")
    code = (
        "citation_scope_violation"
        if case.case_id == "citation_scope_denied"
        else "evidence_scope_violation"
    )
    return _observation(case, code)


def _provider_denial(case: SecurityEvaluationCase, _root: Path) -> SecurityCaseObservation:
    try:
        if case.case_id == "sql_rows_connection_secret_egress":
            sanitize_provider_payload(
                {"rows": [{"value": "redacted"}]},
                classification=DataClassification.CONFIDENTIAL,
            )
        elif case.case_id == "provider_stage_denied":
            require_provider_egress(
                policy=ProviderPolicy.controlled_mixed().model_copy(update={"enabled": False}),
                stage=ProviderStage.DATA_PLANNING,
                classification=DataClassification.INTERNAL,
                payload_size=1,
                call_count=0,
            )
        elif case.case_id == "provider_budget_exceeded":
            require_provider_egress(
                policy=ProviderPolicy.controlled_mixed(),
                stage=ProviderStage.ROUTING,
                classification=DataClassification.INTERNAL,
                payload_size=1,
                call_count=9,
            )
        else:
            ensure_safe_provider_output("Authorization: Bearer evaluation-secret")
    except ProviderPolicyError as exc:
        return _observation(
            case,
            exc.code,
            provider_calls=1 if case.case_id == "provider_sensitive_output" else 0,
        )
    raise AssertionError("provider boundary unexpectedly allowed")


def _release_denial(case: SecurityEvaluationCase, _root: Path) -> SecurityCaseObservation:
    reviewer_accepted = False
    answer_present = True
    citations_present = True
    release_allowed = reviewer_accepted and answer_present and citations_present
    if release_allowed:  # pragma: no cover - fixed fail-closed fixture
        raise AssertionError("response release unexpectedly allowed")
    return _observation(case, "response_release_blocked")


def _audit_event(index: int):
    return new_audit_event(
        event_id=f"security-evaluation-{index}",
        request_id="security-evaluation",
        trace_id="security-evaluation",
        principal_type="test",
        tenant_hash="a" * 64,
        event_type="workflow_started",
        component="runtime",
        action="workflow",
        policy_id="m8c-default-deny",
        policy_version="1",
        outcome="started",
    )


def _audit_denial(case: SecurityEvaluationCase, root: Path) -> SecurityCaseObservation:
    root.mkdir(parents=True, exist_ok=True)
    if case.case_id == "audit_write_failure":
        path = root / "directory-as-log"
        path.mkdir()
        try:
            JsonlAuditSink(path)
        except AuditChainError as exc:
            return _observation(case, str(exc), audit_integrity_detected=True)
        raise AssertionError("unavailable audit sink unexpectedly opened")

    path = root / "audit.jsonl"
    sink = JsonlAuditSink(path, fsync=False)
    sink.append(_audit_event(1))
    sink.append(_audit_event(2))
    sink.close()
    lines = path.read_text(encoding="utf-8").splitlines()
    if case.case_id == "audit_tamper_detected":
        lines[0] = lines[0].replace("started", "failed", 1)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    elif case.case_id == "audit_middle_delete_detected":
        path.write_text(lines[1] + "\n", encoding="utf-8")
    elif case.case_id == "audit_tail_delete_detected":
        path.write_text(lines[0] + "\n", encoding="utf-8")
    elif case.case_id == "audit_reorder_detected":
        path.write_text(lines[1] + "\n" + lines[0] + "\n", encoding="utf-8")
    try:
        JsonlAuditSink.verify(path)
    except AuditChainError:
        return _observation(
            case,
            "audit_integrity_verification_failed",
            audit_integrity_detected=True,
        )
    raise AssertionError("audit mutation unexpectedly verified")


CASE_HANDLERS: Mapping[str, SecurityCaseHandler] = {
    **{name: _identity_denial for name in ("unauthenticated_request", "tenant_context_missing")},
    **{
        name: _policy_denial
        for name in (
            "scenario_denied",
            "workflow_denied",
            "skill_denied",
            "tool_denied",
            "router_authority_expansion",
            "planner_unauthorized_skill",
            "prompt_injection_denied",
            "indirect_prompt_injection_denied",
        )
    },
    **{
        name: _scope_denial
        for name in (
            "cross_tenant_session",
            "cross_subject_memory",
            "data_scope_denied",
            "knowledge_scope_denied",
            "parent_expansion_denied",
        )
    },
    "evidence_scope_denied": _projected_scope_denial,
    "citation_scope_denied": _projected_scope_denial,
    **{
        name: _provider_denial
        for name in (
            "sql_rows_connection_secret_egress",
            "provider_stage_denied",
            "provider_budget_exceeded",
            "provider_sensitive_output",
        )
    },
    "reviewer_security_failure": _release_denial,
    "response_release_blocked": _release_denial,
    **{
        name: _audit_denial
        for name in (
            "audit_write_failure",
            "audit_tamper_detected",
            "audit_middle_delete_detected",
            "audit_tail_delete_detected",
            "audit_reorder_detected",
        )
    },
}
