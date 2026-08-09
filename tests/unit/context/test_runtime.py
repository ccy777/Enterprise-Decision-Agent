from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

from decision_agent.context import (
    ContextBudgetConfig,
    EvidenceDomain,
    RequestContextRuntime,
    TokenBudget,
)
from decision_agent.routing.models import RequestRoute, RouterDecision

NOW = datetime(2026, 7, 18, tzinfo=UTC)


def test_budget_config_is_immutable_and_keeps_node_budgets_centralized() -> None:
    config = ContextBudgetConfig(knowledge=TokenBudget(max_tokens=9, reserved_tokens=8))
    assert config.knowledge.available_tokens == 1
    with pytest.raises(FrozenInstanceError):
        config.knowledge = TokenBudget(max_tokens=10)  # type: ignore[misc]


def _runtime() -> tuple[RequestContextRuntime, object]:
    runtime = RequestContextRuntime(request_id="request-1", created_at=NOW)
    return runtime, runtime.user_request("Analyze inventory risk")


def test_router_and_coordinator_do_not_select_tool_or_evidence_items() -> None:
    runtime, user = _runtime()
    instruction = runtime.system_instruction("router-instruction", "Route only.")
    data_summary = runtime.verified_summary(
        "data-summary", "answer [D1]", source_item_ids=(user.item_id,)
    )
    data_evidence = runtime.evidence(
        "data-evidence",
        "answer [D1]",
        domain=EvidenceDomain.DATA,
        citation_ids=("[D1]",),
        source_item_ids=(data_summary.item_id,),
    )
    selection = runtime.select_for_router(user_item=user, instruction_item=instruction, at=NOW)
    assert {user.item_id, instruction.item_id} <= set(selection.selected_item_ids)
    assert data_evidence.item_id not in selection.selected_item_ids

    decision = RouterDecision(
        route=RequestRoute.DATA,
        normalized_query="inventory",
        decision_reason="data",
        knowledge_subquery=None,
        data_subquery="inventory",
        missing_information=None,
        confidence=0.9,
    )
    decision_item = runtime.router_decision(decision, source_item_id=user.item_id)
    coordinator = runtime.select_for_coordinator(
        user_item=user, decision_item=decision_item, at=NOW
    )
    assert {user.item_id, decision_item.item_id} <= set(coordinator.selected_item_ids)
    assert data_evidence.item_id not in coordinator.selected_item_ids


def test_knowledge_data_and_mixed_runtime_policies_keep_evidence_domains_isolated() -> None:
    runtime, user = _runtime()
    knowledge_instruction = runtime.skill_instruction(
        "knowledge-instruction", "knowledge", source_item_id=user.item_id
    )
    data_instruction = runtime.skill_instruction(
        "data-instruction", "data", source_item_id=user.item_id
    )
    knowledge_summary = runtime.verified_summary(
        "knowledge-summary", "answer [E1]", source_item_ids=(user.item_id,)
    )
    data_summary = runtime.verified_summary(
        "data-summary", "answer [D1]", source_item_ids=(user.item_id,)
    )
    knowledge = runtime.evidence(
        "knowledge-evidence",
        "answer [E1]",
        domain=EvidenceDomain.KNOWLEDGE,
        citation_ids=("[E1]",),
        source_item_ids=(knowledge_summary.item_id,),
    )
    data = runtime.evidence(
        "data-evidence",
        "answer [D1]",
        domain=EvidenceDomain.DATA,
        citation_ids=("[D1]",),
        source_item_ids=(data_summary.item_id,),
    )

    knowledge_selection = runtime.select_for_knowledge(
        user_item=user, instruction_item=knowledge_instruction, at=NOW
    )
    data_selection = runtime.select_for_data(
        user_item=user, instruction_item=data_instruction, at=NOW
    )
    mixed_selection = runtime.select_for_mixed_synthesis(
        user_item=user, data_summary=data_summary, knowledge_summary=knowledge_summary, at=NOW
    )

    assert knowledge.item_id in knowledge_selection.selected_item_ids
    assert data.item_id not in knowledge_selection.selected_item_ids
    assert data.item_id in data_selection.selected_item_ids
    assert knowledge.item_id not in data_selection.selected_item_ids
    assert {knowledge.item_id, data.item_id} <= set(mixed_selection.selected_item_ids)
    assert all("answer" not in str(diagnostic) for diagnostic in runtime.diagnostics)
