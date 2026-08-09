"""Deterministic in-memory selection of immutable context items."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from types import MappingProxyType

from decision_agent.context.exceptions import (
    ContextTokenBudgetExceededError,
    DuplicateContextItemError,
    RequiredContextItemMissingError,
    RequiredContextItemRejectedError,
)
from decision_agent.context.models import (
    ContextDropReason,
    ContextItem,
    ContextKind,
    ContextPolicy,
    ContextSelectionResult,
    DroppedContextItem,
    TrustLevel,
)

_TRUST_PRIORITY = {
    TrustLevel.TRUSTED_SYSTEM: 0,
    TrustLevel.VERIFIED_INTERNAL: 1,
    TrustLevel.TRUSTED_TOOL_RESULT: 2,
    TrustLevel.UNTRUSTED_USER: 3,
    TrustLevel.UNTRUSTED_EXTERNAL: 4,
}
_EVIDENCE_KINDS = frozenset({ContextKind.KNOWLEDGE_EVIDENCE, ContextKind.DATA_EVIDENCE})


class ContextManager:
    """Store immutable items and select a policy-safe deterministic subset."""

    def __init__(self) -> None:
        self._items: dict[str, ContextItem] = {}

    @property
    def items(self) -> Mapping[str, ContextItem]:
        """Expose a read-only view rather than the mutable backing dictionary."""
        return MappingProxyType(self._items.copy())

    def add(self, item: ContextItem) -> None:
        """Add one pre-validated item, refusing duplicate identifiers."""
        if item.item_id in self._items:
            raise DuplicateContextItemError(item.item_id)
        self._items[item.item_id] = item

    def get(self, item_id: str) -> ContextItem | None:
        """Return an immutable item or ``None`` without exposing mutable storage."""
        return self._items.get(item_id)

    @staticmethod
    def _rejection_reason(
        item: ContextItem, policy: ContextPolicy, at: datetime
    ) -> ContextDropReason | None:
        if item.is_expired(at):
            return ContextDropReason.EXPIRED
        if item.kind not in policy.allowed_kinds:
            return ContextDropReason.KIND_NOT_ALLOWED
        if item.trust_level not in policy.allowed_trust_levels:
            return ContextDropReason.TRUST_NOT_ALLOWED
        if (
            item.kind in _EVIDENCE_KINDS
            and item.evidence_domain not in policy.allowed_evidence_domains
        ):
            return ContextDropReason.EVIDENCE_DOMAIN_NOT_ALLOWED
        return None

    def select(self, policy: ContextPolicy, *, at: datetime) -> ContextSelectionResult:
        """Select required items first, then eligible candidates in stable priority order."""
        if policy is None:
            raise ValueError("policy must not be None")
        if at.tzinfo is None or at.utcoffset() is None:
            raise ValueError("at must be timezone-aware")

        selected: list[ContextItem] = []
        current_tokens = 0
        required_ids = set(policy.required_item_ids)
        for item_id in policy.required_item_ids:
            item = self._items.get(item_id)
            if item is None:
                raise RequiredContextItemMissingError(policy.node_name, item_id)
            reason = self._rejection_reason(item, policy, at)
            if reason is not None:
                raise RequiredContextItemRejectedError(policy.node_name, item_id, reason.value)
            if not policy.token_budget.can_fit(current_tokens, item.estimated_tokens):
                raise ContextTokenBudgetExceededError(policy.node_name, item_id)
            selected.append(item)
            current_tokens += item.estimated_tokens

        candidates = sorted(
            (item for item_id, item in self._items.items() if item_id not in required_ids),
            key=lambda item: (_TRUST_PRIORITY[item.trust_level], item.created_at, item.item_id),
        )
        dropped: list[DroppedContextItem] = []
        for item in candidates:
            reason = self._rejection_reason(item, policy, at)
            if reason is not None:
                dropped.append(DroppedContextItem(item_id=item.item_id, reason=reason))
            elif policy.token_budget.can_fit(current_tokens, item.estimated_tokens):
                selected.append(item)
                current_tokens += item.estimated_tokens
            else:
                dropped.append(
                    DroppedContextItem(
                        item_id=item.item_id,
                        reason=ContextDropReason.TOKEN_BUDGET_EXCEEDED,
                    )
                )

        return ContextSelectionResult(
            selected_items=tuple(selected),
            selected_item_ids=tuple(item.item_id for item in selected),
            dropped_items=tuple(dropped),
            total_estimated_tokens=current_tokens,
            available_tokens=policy.token_budget.available_tokens,
        )
