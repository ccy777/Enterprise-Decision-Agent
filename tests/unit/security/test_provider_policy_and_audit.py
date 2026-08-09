from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from decision_agent.security import (
    AuditChainError,
    DataClassification,
    DeterministicProviderRedactor,
    InMemoryAuditSink,
    JsonlAuditSink,
    ProviderGovernance,
    ProviderPolicy,
    ProviderPolicyError,
    ProviderStage,
    build_security_context,
    ensure_safe_provider_output,
    make_test_principal,
    new_audit_event,
    require_provider_egress,
    sanitize_provider_payload,
)

pytestmark = pytest.mark.offline_integration


def _context():
    return build_security_context(
        principal=make_test_principal(
            subject_id="test-subject", tenant_id="tenant-a", roles=frozenset({"decision_analyst"})
        ),
        request_id="request-1",
        trace_id="trace-1",
        allowed_scenarios=frozenset({"knowledge"}),
        allowed_workflows=frozenset({"direct"}),
        allowed_skills=frozenset({"enterprise-knowledge-qa"}),
        allowed_tools=frozenset({"run_knowledge_agent"}),
    )


def _event(index: int = 1):
    return new_audit_event(
        event_id=f"event-{index}",
        request_id="request-1",
        trace_id="trace-1",
        principal_type="test",
        tenant_hash="a" * 64,
        event_type="provider_call_allowed",
        component="provider",
        action="complete",
        policy_id="policy",
        policy_version="1",
        outcome="allowed",
    )


def test_policy_is_default_deny_for_missing_stage_and_forbidden_classification() -> None:
    policy = ProviderPolicy.controlled_mixed()

    with pytest.raises(ProviderPolicyError, match="provider_data_classification_forbidden"):
        require_provider_egress(
            policy=policy,
            stage=ProviderStage.ROUTING,
            classification=DataClassification.CONFIDENTIAL,
            payload_size=10,
            call_count=0,
        )
    with pytest.raises(ProviderPolicyError, match="provider_policy_missing"):
        require_provider_egress(
            policy=None,
            stage=ProviderStage.ROUTING,
            classification=DataClassification.PUBLIC,
            payload_size=10,
            call_count=0,
        )


def test_policy_blocks_size_and_budget_before_transport() -> None:
    policy = ProviderPolicy.controlled_mixed()
    with pytest.raises(ProviderPolicyError, match="provider_payload_too_large"):
        require_provider_egress(
            policy=policy,
            stage=ProviderStage.ROUTING,
            classification=DataClassification.PUBLIC,
            payload_size=2_001,
            call_count=0,
        )
    with pytest.raises(ProviderPolicyError, match="provider_budget_exceeded"):
        require_provider_egress(
            policy=policy,
            stage=ProviderStage.ROUTING,
            classification=DataClassification.PUBLIC,
            payload_size=1,
            call_count=9,
        )


def test_redactor_denies_restricted_and_structurally_masks_identity() -> None:
    sanitized, count = sanitize_provider_payload(
        {"tenant_id": "tenant-a", "nested": {"user_id": "user-a", "text": "normal"}},
        classification=DataClassification.INTERNAL,
    )
    assert sanitized == {
        "tenant_id": "[REDACTED]",
        "nested": {"user_id": "[REDACTED]", "text": "normal"},
    }
    assert count == 2
    with pytest.raises(ProviderPolicyError, match="provider_data_classification_forbidden"):
        sanitize_provider_payload("row", classification=DataClassification.RESTRICTED)
    with pytest.raises(ProviderPolicyError, match="provider_sensitive_input_detected"):
        sanitize_provider_payload(
            {"sql": "SELECT name FROM customers"}, classification=DataClassification.INTERNAL
        )
    with pytest.raises(ProviderPolicyError, match="provider_sensitive_input_detected"):
        sanitize_provider_payload(
            {"rows": [{"name": "customer"}]}, classification=DataClassification.INTERNAL
        )
    assert (
        sanitize_provider_payload(
            "Please select a regional owner for the meeting.",
            classification=DataClassification.INTERNAL,
        )[0]
        == "Please select a regional owner for the meeting."
    )
    with pytest.raises(ProviderPolicyError, match="provider_sensitive_output_detected"):
        ensure_safe_provider_output("Authorization: Bearer token")
    ensure_safe_provider_output("SELECT value FROM safe_table", allow_sql=True)
    with pytest.raises(ProviderPolicyError, match="provider_sensitive_output_detected"):
        ensure_safe_provider_output("SELECT value FROM safe_table")


def test_audit_chain_detects_mutation_deletion_and_reordering(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    sink = JsonlAuditSink(path)
    sink.append(_event(1))
    sink.append(_event(2))
    sink.close()
    assert JsonlAuditSink.verify(path)

    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text(
        lines[0].replace("allowed", "denied", 1) + "\n" + lines[1] + "\n", encoding="utf-8"
    )
    with pytest.raises(AuditChainError, match="audit_integrity_verification_failed"):
        JsonlAuditSink.verify(path)

    path.write_text(lines[0] + "\n", encoding="utf-8")
    with pytest.raises(AuditChainError, match="audit_chain_invalid"):
        JsonlAuditSink.verify(path)
    path.write_text(lines[1] + "\n" + lines[0] + "\n", encoding="utf-8")
    with pytest.raises(AuditChainError, match="audit_integrity_verification_failed"):
        JsonlAuditSink.verify(path)


def test_concurrent_audit_append_keeps_a_valid_chain(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    sink = JsonlAuditSink(path)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda index: sink.append(_event(index)), range(16)))
    sink.close()
    assert JsonlAuditSink.verify(path)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 16


def test_audit_chain_rejects_missing_anchor_and_partial_final_write(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    sink = JsonlAuditSink(path)
    sink.append(_event())
    sink.close()
    path.with_suffix(".jsonl.anchor").unlink()
    with pytest.raises(AuditChainError, match="audit_chain_invalid"):
        JsonlAuditSink.verify(path)

    path.write_text('{"event_id":"partial"', encoding="utf-8")
    with pytest.raises(AuditChainError, match="audit_integrity_verification_failed"):
        JsonlAuditSink.verify(path)


@pytest.mark.asyncio
async def test_governance_fails_closed_before_transport_when_dependencies_are_invalid() -> None:
    calls = 0

    async def transport(_payload: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"choices": []}

    governance = ProviderGovernance(
        policy=ProviderPolicy.controlled_mixed(),
        audit_sink=None,
        redactor=DeterministicProviderRedactor(),
    )
    with (
        pytest.raises(ProviderPolicyError, match="audit_sink_missing"),
        governance.bind_request(
            request_id="request-1", trace_id="trace-1", security_context=_context()
        ),
    ):
        await governance.call(
            stage=ProviderStage.ROUTING,
            payload={"query": "normal request"},
            classification=DataClassification.INTERNAL,
            evidence_count=0,
            transport=transport,
        )
    assert calls == 0

    missing_redactor = ProviderGovernance(
        policy=ProviderPolicy.controlled_mixed(), audit_sink=InMemoryAuditSink(), redactor=None
    )
    with (
        pytest.raises(ProviderPolicyError, match="provider_redaction_failed"),
        missing_redactor.bind_request(
            request_id="request-1", trace_id="trace-1", security_context=_context()
        ),
    ):
        pass


@pytest.mark.asyncio
async def test_governance_audits_allow_block_and_sensitive_output_without_payloads() -> None:
    sink = InMemoryAuditSink()
    governance = ProviderGovernance(
        policy=ProviderPolicy.controlled_mixed(),
        audit_sink=sink,
        redactor=DeterministicProviderRedactor(),
    )
    calls = 0

    async def safe_transport(_payload: object) -> dict[str, str]:
        nonlocal calls
        calls += 1
        return {"content": "safe response"}

    async def secret_transport(_payload: object) -> dict[str, str]:
        nonlocal calls
        calls += 1
        return {"content": "Authorization: Bearer secret-value"}

    with governance.bind_request(
        request_id="request-1", trace_id="trace-1", security_context=_context()
    ):
        assert await governance.call(
            stage=ProviderStage.ROUTING,
            payload={"query": "normal request"},
            classification=DataClassification.INTERNAL,
            evidence_count=0,
            transport=safe_transport,
        ) == {"content": "safe response"}
        with pytest.raises(ProviderPolicyError, match="provider_sensitive_input_detected"):
            await governance.call(
                stage=ProviderStage.DATA_ANSWER,
                payload={"sql": "SELECT revenue FROM sales"},
                classification=DataClassification.CONFIDENTIAL,
                evidence_count=0,
                transport=safe_transport,
            )
        with pytest.raises(ProviderPolicyError, match="provider_sensitive_output_detected"):
            await governance.call(
                stage=ProviderStage.KNOWLEDGE_ANSWER,
                payload={"summary": "safe"},
                classification=DataClassification.CONFIDENTIAL,
                evidence_count=0,
                transport=secret_transport,
            )
    assert calls == 2
    assert [event.event_type.value for event in sink.events] == [
        "provider_call_allowed",
        "provider_call_completed",
        "provider_call_blocked",
        "provider_call_allowed",
        "provider_call_blocked",
    ]
    assert all("normal request" not in event.model_dump_json() for event in sink.events)


@pytest.mark.asyncio
async def test_provider_budget_is_shared_across_child_asyncio_contexts() -> None:
    sink = InMemoryAuditSink()
    governance = ProviderGovernance(
        policy=ProviderPolicy.controlled_mixed(),
        audit_sink=sink,
        redactor=DeterministicProviderRedactor(),
    )
    transport_calls = 0

    async def transport(_payload: object) -> dict[str, str]:
        nonlocal transport_calls
        transport_calls += 1
        await asyncio.sleep(0)
        return {"content": "safe"}

    async def invoke(index: int) -> object:
        return await governance.call(
            stage=ProviderStage.ROUTING,
            payload={"query": f"normal request {index}"},
            classification=DataClassification.INTERNAL,
            evidence_count=0,
            transport=transport,
        )

    with governance.bind_request(
        request_id="request-1", trace_id="trace-1", security_context=_context()
    ):
        results = await asyncio.gather(
            *(asyncio.create_task(invoke(index)) for index in range(10)),
            return_exceptions=True,
        )

    failures = [result for result in results if isinstance(result, ProviderPolicyError)]
    assert len(failures) == 1
    assert failures[0].code == "provider_budget_exceeded"
    assert transport_calls == 9
