from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from decision_agent.context import (
    ContextDropReason,
    ContextItem,
    ContextKind,
    ContextManager,
    ContextPolicy,
    ContextProvenance,
    ContextSelectionResult,
    ContextSource,
    ContextTokenBudgetExceededError,
    DroppedContextItem,
    EvidenceDomain,
    RequiredContextItemMissingError,
    RequiredContextItemRejectedError,
    TokenBudget,
    TrustLevel,
)

NOW = datetime(2026, 7, 18, tzinfo=UTC)


def make_item(
    item_id: str,
    *,
    kind: ContextKind = ContextKind.USER_REQUEST,
    trust: TrustLevel = TrustLevel.UNTRUSTED_USER,
    tokens: int = 1,
    domain: EvidenceDomain | None = None,
    expired: bool = False,
) -> ContextItem:
    source = ContextSource.USER if kind is ContextKind.USER_REQUEST else ContextSource.AGENT
    provenance_ids = () if kind is ContextKind.USER_REQUEST else ("origin",)
    citations = ()
    if kind is ContextKind.KNOWLEDGE_EVIDENCE:
        domain, citations = EvidenceDomain.KNOWLEDGE, ("[E7]",)
    if kind is ContextKind.DATA_EVIDENCE:
        domain, citations = EvidenceDomain.DATA, ("[D8]",)
    return ContextItem(
        item_id=item_id,
        kind=kind,
        content=item_id,
        source=source,
        trust_level=trust,
        provenance=ContextProvenance(
            producer="test", request_id="r", generated_at=NOW, source_item_ids=provenance_ids
        ),
        created_at=NOW + timedelta(seconds=len(item_id)),
        expires_at=NOW + timedelta(seconds=len(item_id)) if expired else None,
        estimated_tokens=tokens,
        evidence_domain=domain,
        citation_ids=citations,
    )


def policy(
    *,
    required: tuple[str, ...] = (),
    budget: int = 10,
    domains: frozenset[EvidenceDomain] | None = None,
) -> ContextPolicy:
    return ContextPolicy(
        node_name="test-node",
        allowed_kinds=frozenset(ContextKind),
        allowed_trust_levels=frozenset(TrustLevel),
        allowed_evidence_domains=frozenset(EvidenceDomain) if domains is None else domains,
        token_budget=TokenBudget(max_tokens=budget),
        required_item_ids=required,
    )


def test_required_items_are_first_stable_and_fail_closed_when_missing() -> None:
    manager = ContextManager()
    manager.add(make_item("second"))
    manager.add(make_item("first"))

    result = manager.select(policy(required=("first", "second")), at=NOW)
    assert result.selected_item_ids[:2] == ("first", "second")
    with pytest.raises(RequiredContextItemMissingError):
        manager.select(policy(required=("absent",)), at=NOW)


def test_ordinary_candidates_are_filtered_and_ordered_deterministically() -> None:
    manager = ContextManager()
    manager.add(make_item("expired", expired=True))
    manager.add(
        make_item(
            "knowledge", kind=ContextKind.KNOWLEDGE_EVIDENCE, trust=TrustLevel.VERIFIED_INTERNAL
        )
    )
    manager.add(
        make_item("data", kind=ContextKind.DATA_EVIDENCE, trust=TrustLevel.TRUSTED_TOOL_RESULT)
    )
    manager.add(make_item("too-large", tokens=3))
    manager.add(make_item("fits", tokens=1))

    result = manager.select(
        policy(budget=2, domains=frozenset({EvidenceDomain.KNOWLEDGE})),
        at=NOW + timedelta(days=1),
    )
    assert result.selected_item_ids == ("knowledge", "fits")
    assert {(drop.item_id, drop.reason) for drop in result.dropped_items} == {
        ("expired", ContextDropReason.EXPIRED),
        ("data", ContextDropReason.EVIDENCE_DOMAIN_NOT_ALLOWED),
        ("too-large", ContextDropReason.TOKEN_BUDGET_EXCEEDED),
    }


def test_knowledge_data_and_mixed_policies_preserve_original_citations() -> None:
    manager = ContextManager()
    manager.add(make_item("knowledge", kind=ContextKind.KNOWLEDGE_EVIDENCE))
    manager.add(make_item("data", kind=ContextKind.DATA_EVIDENCE))

    knowledge = manager.select(policy(domains=frozenset({EvidenceDomain.KNOWLEDGE})), at=NOW)
    data = manager.select(policy(domains=frozenset({EvidenceDomain.DATA})), at=NOW)
    mixed = manager.select(policy(), at=NOW)

    assert knowledge.selected_items[0].citation_ids == ("[E7]",)
    assert data.selected_items[0].citation_ids == ("[D8]",)
    assert {item.citation_ids for item in mixed.selected_items} == {("[E7]",), ("[D8]",)}


def test_manager_mapping_is_read_only_and_naive_selection_time_is_rejected() -> None:
    manager = ContextManager()
    manager.add(make_item("one"))
    with pytest.raises(TypeError):
        manager.items["two"] = make_item("two")  # type: ignore[index]
    with pytest.raises(ValueError):
        manager.select(policy(), at=datetime(2026, 1, 1))


@pytest.mark.parametrize(
    ("required", "configured_policy", "error_type", "reason"),
    (
        (None, policy(required=("missing",)), RequiredContextItemMissingError, None),
        (
            make_item("expired", expired=True),
            policy(required=("expired",)),
            RequiredContextItemRejectedError,
            "expired",
        ),
        (
            make_item("kind", kind=ContextKind.STRUCTURED_SUMMARY),
            ContextPolicy(
                node_name="test-node",
                allowed_kinds=frozenset({ContextKind.USER_REQUEST}),
                allowed_trust_levels=frozenset(TrustLevel),
                allowed_evidence_domains=frozenset(EvidenceDomain),
                token_budget=TokenBudget(max_tokens=10),
                required_item_ids=("kind",),
            ),
            RequiredContextItemRejectedError,
            "kind_not_allowed",
        ),
        (
            make_item("trust"),
            ContextPolicy(
                node_name="test-node",
                allowed_kinds=frozenset(ContextKind),
                allowed_trust_levels=frozenset({TrustLevel.VERIFIED_INTERNAL}),
                allowed_evidence_domains=frozenset(EvidenceDomain),
                token_budget=TokenBudget(max_tokens=10),
                required_item_ids=("trust",),
            ),
            RequiredContextItemRejectedError,
            "trust_not_allowed",
        ),
        (
            make_item("domain", kind=ContextKind.KNOWLEDGE_EVIDENCE),
            policy(required=("domain",), domains=frozenset({EvidenceDomain.DATA})),
            RequiredContextItemRejectedError,
            "evidence_domain_not_allowed",
        ),
        (
            make_item("large", tokens=3),
            policy(required=("large",), budget=2),
            ContextTokenBudgetExceededError,
            "token_budget_exceeded",
        ),
    ),
)
def test_required_item_rejections_fail_closed_with_safe_errors(
    required: ContextItem | None,
    configured_policy: ContextPolicy,
    error_type: type[Exception],
    reason: str | None,
) -> None:
    manager = ContextManager()
    if required is not None:
        manager.add(required.model_copy(update={"content": "sensitive required content"}))
    manager.add(make_item("ordinary"))

    with pytest.raises(error_type) as raised:
        manager.select(configured_policy, at=NOW + timedelta(days=1))
    message = str(raised.value)
    assert "test-node" in message
    if reason is not None:
        assert reason in message
    assert "ordinary" not in message
    assert "sensitive required content" not in message


def test_required_total_token_overflow_fails_closed_before_ordinary_selection() -> None:
    manager = ContextManager()
    manager.add(make_item("one", tokens=2))
    manager.add(make_item("two", tokens=2))
    manager.add(make_item("ordinary"))

    with pytest.raises(ContextTokenBudgetExceededError, match="item=two"):
        manager.select(policy(required=("one", "two"), budget=3), at=NOW)


def test_trust_and_same_trust_sorting_are_stable_and_repeatable() -> None:
    manager = ContextManager()
    system = ContextItem(
        item_id="system",
        kind=ContextKind.SYSTEM_INSTRUCTION,
        content="system",
        source=ContextSource.SYSTEM,
        trust_level=TrustLevel.TRUSTED_SYSTEM,
        provenance=ContextProvenance(producer="test", request_id="r", generated_at=NOW),
        created_at=NOW,
        estimated_tokens=1,
    )
    verified = make_item(
        "verified", kind=ContextKind.STRUCTURED_SUMMARY, trust=TrustLevel.VERIFIED_INTERNAL
    )
    tool = make_item("tool", kind=ContextKind.TOOL_RESULT, trust=TrustLevel.TRUSTED_TOOL_RESULT)
    user_a = make_item("a", tokens=1)
    user_z = make_item("z", tokens=1)
    external = ContextItem(
        item_id="external",
        kind=ContextKind.STRUCTURED_SUMMARY,
        content="external",
        source=ContextSource.EXTERNAL,
        trust_level=TrustLevel.UNTRUSTED_EXTERNAL,
        provenance=ContextProvenance(
            producer="test", request_id="r", generated_at=NOW, source_item_ids=("p",)
        ),
        created_at=NOW,
        estimated_tokens=1,
    )
    for item in (external, user_z, tool, verified, user_a, system):
        manager.add(item)

    result = manager.select(policy(), at=NOW)
    repeated = manager.select(policy(), at=NOW)
    assert result.selected_item_ids == ("system", "verified", "tool", "a", "z", "external")
    assert repeated == result


def test_selection_result_rejects_all_inconsistent_external_constructions() -> None:
    item = make_item("item")
    dropped = DroppedContextItem(item_id="dropped", reason=ContextDropReason.EXPIRED)
    payload = {
        "selected_items": (item,),
        "selected_item_ids": ("item",),
        "dropped_items": (dropped,),
        "total_estimated_tokens": 1,
        "available_tokens": 1,
    }
    invalid_payloads = (
        {**payload, "selected_item_ids": ("wrong",)},
        {
            **payload,
            "selected_items": (item, item),
            "selected_item_ids": ("item", "item"),
            "total_estimated_tokens": 2,
            "available_tokens": 2,
        },
        {**payload, "dropped_items": (dropped, dropped)},
        {
            **payload,
            "dropped_items": (
                DroppedContextItem(item_id="item", reason=ContextDropReason.EXPIRED),
            ),
        },
        {**payload, "total_estimated_tokens": 0},
        {**payload, "total_estimated_tokens": 2, "available_tokens": 1},
    )
    for invalid_payload in invalid_payloads:
        with pytest.raises(ValueError):
            ContextSelectionResult(**invalid_payload)
