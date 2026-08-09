"""Fixed, per-node context policies for the request-scoped runtime."""

from __future__ import annotations

from dataclasses import dataclass, field

from decision_agent.context.models import ContextKind, ContextPolicy, EvidenceDomain, TrustLevel
from decision_agent.context.token_budget import TokenBudget


@dataclass(frozen=True)
class ContextBudgetConfig:
    """Immutable per-node budgets; the default remains the production policy."""

    router: TokenBudget = field(
        default_factory=lambda: TokenBudget(max_tokens=2_400, reserved_tokens=600)
    )
    coordinator: TokenBudget = field(
        default_factory=lambda: TokenBudget(max_tokens=2_400, reserved_tokens=400)
    )
    knowledge: TokenBudget = field(
        default_factory=lambda: TokenBudget(max_tokens=4_800, reserved_tokens=1_200)
    )
    data: TokenBudget = field(
        default_factory=lambda: TokenBudget(max_tokens=4_800, reserved_tokens=1_200)
    )
    mixed_synthesis: TokenBudget = field(
        default_factory=lambda: TokenBudget(max_tokens=6_400, reserved_tokens=1_600)
    )


DEFAULT_CONTEXT_BUDGET_CONFIG = ContextBudgetConfig()


def _policy(
    node_name: str,
    *,
    kinds: frozenset[ContextKind],
    trusts: frozenset[TrustLevel],
    domains: frozenset[EvidenceDomain],
    required: tuple[str, ...],
    budget_config: ContextBudgetConfig,
) -> ContextPolicy:
    return ContextPolicy(
        node_name=node_name,
        allowed_kinds=kinds,
        allowed_trust_levels=trusts,
        allowed_evidence_domains=domains,
        token_budget=getattr(budget_config, node_name),
        required_item_ids=required,
    )


def router_policy(
    *,
    user_item_id: str,
    instruction_item_id: str,
    budget_config: ContextBudgetConfig = DEFAULT_CONTEXT_BUDGET_CONFIG,
) -> ContextPolicy:
    return _policy(
        "router",
        kinds=frozenset(
            {
                ContextKind.SYSTEM_INSTRUCTION,
                ContextKind.USER_REQUEST,
                ContextKind.STRUCTURED_SUMMARY,
                ContextKind.CONVERSATION_MEMORY,
            }
        ),
        trusts=frozenset(
            {
                TrustLevel.TRUSTED_SYSTEM,
                TrustLevel.VERIFIED_INTERNAL,
                TrustLevel.UNTRUSTED_USER,
                TrustLevel.UNTRUSTED_EXTERNAL,
            }
        ),
        domains=frozenset(),
        required=(user_item_id, instruction_item_id),
        budget_config=budget_config,
    )


def coordinator_policy(
    *,
    user_item_id: str,
    decision_item_id: str,
    budget_config: ContextBudgetConfig = DEFAULT_CONTEXT_BUDGET_CONFIG,
) -> ContextPolicy:
    return _policy(
        "coordinator",
        kinds=frozenset(
            {
                ContextKind.SYSTEM_INSTRUCTION,
                ContextKind.USER_REQUEST,
                ContextKind.ROUTER_DECISION,
                ContextKind.STRUCTURED_SUMMARY,
            }
        ),
        trusts=frozenset(
            {TrustLevel.TRUSTED_SYSTEM, TrustLevel.VERIFIED_INTERNAL, TrustLevel.UNTRUSTED_USER}
        ),
        domains=frozenset(),
        required=(user_item_id, decision_item_id),
        budget_config=budget_config,
    )


def knowledge_policy(
    *,
    user_item_id: str,
    instruction_item_id: str,
    budget_config: ContextBudgetConfig = DEFAULT_CONTEXT_BUDGET_CONFIG,
) -> ContextPolicy:
    return _policy(
        "knowledge",
        kinds=frozenset(
            {
                ContextKind.SYSTEM_INSTRUCTION,
                ContextKind.USER_REQUEST,
                ContextKind.SKILL_INSTRUCTION,
                ContextKind.KNOWLEDGE_EVIDENCE,
                ContextKind.STRUCTURED_SUMMARY,
                ContextKind.CONVERSATION_MEMORY,
            }
        ),
        trusts=frozenset(
            {
                TrustLevel.TRUSTED_SYSTEM,
                TrustLevel.VERIFIED_INTERNAL,
                TrustLevel.UNTRUSTED_USER,
                TrustLevel.UNTRUSTED_EXTERNAL,
            }
        ),
        domains=frozenset({EvidenceDomain.KNOWLEDGE}),
        required=(user_item_id, instruction_item_id),
        budget_config=budget_config,
    )


def data_policy(
    *,
    user_item_id: str,
    instruction_item_id: str,
    budget_config: ContextBudgetConfig = DEFAULT_CONTEXT_BUDGET_CONFIG,
) -> ContextPolicy:
    return _policy(
        "data",
        kinds=frozenset(
            {
                ContextKind.SYSTEM_INSTRUCTION,
                ContextKind.USER_REQUEST,
                ContextKind.SKILL_INSTRUCTION,
                ContextKind.TOOL_RESULT,
                ContextKind.DATA_EVIDENCE,
                ContextKind.STRUCTURED_SUMMARY,
                ContextKind.CONVERSATION_MEMORY,
            }
        ),
        trusts=frozenset(
            {
                TrustLevel.TRUSTED_SYSTEM,
                TrustLevel.VERIFIED_INTERNAL,
                TrustLevel.TRUSTED_TOOL_RESULT,
                TrustLevel.UNTRUSTED_USER,
                TrustLevel.UNTRUSTED_EXTERNAL,
            }
        ),
        domains=frozenset({EvidenceDomain.DATA}),
        required=(user_item_id, instruction_item_id),
        budget_config=budget_config,
    )


def mixed_synthesis_policy(
    *,
    user_item_id: str,
    data_summary_item_id: str,
    knowledge_summary_item_id: str,
    budget_config: ContextBudgetConfig = DEFAULT_CONTEXT_BUDGET_CONFIG,
) -> ContextPolicy:
    return _policy(
        "mixed_synthesis",
        kinds=frozenset(
            {
                ContextKind.SYSTEM_INSTRUCTION,
                ContextKind.USER_REQUEST,
                ContextKind.STRUCTURED_SUMMARY,
                ContextKind.KNOWLEDGE_EVIDENCE,
                ContextKind.DATA_EVIDENCE,
            }
        ),
        trusts=frozenset(
            {TrustLevel.TRUSTED_SYSTEM, TrustLevel.VERIFIED_INTERNAL, TrustLevel.UNTRUSTED_USER}
        ),
        domains=frozenset({EvidenceDomain.KNOWLEDGE, EvidenceDomain.DATA}),
        required=(user_item_id, data_summary_item_id, knowledge_summary_item_id),
        budget_config=budget_config,
    )
