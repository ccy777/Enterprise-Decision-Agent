"""Stable, content-safe errors raised by the context foundation."""

from __future__ import annotations


class ContextManagerError(Exception):
    """Base error for deterministic context selection failures."""


class DuplicateContextItemError(ContextManagerError):
    """Raised when an item identifier is already present."""

    def __init__(self, item_id: str) -> None:
        super().__init__(f"duplicate context item: {item_id}")


class RequiredContextItemMissingError(ContextManagerError):
    """Raised when a policy-required item cannot be found."""

    def __init__(self, node_name: str, item_id: str) -> None:
        super().__init__(f"required context item missing: node={node_name}, item={item_id}")


class RequiredContextItemRejectedError(ContextManagerError):
    """Raised when a policy-required item is not eligible for selection."""

    def __init__(self, node_name: str, item_id: str, reason: str) -> None:
        super().__init__(
            f"required context item rejected: node={node_name}, item={item_id}, reason={reason}"
        )


class ContextTokenBudgetExceededError(ContextManagerError):
    """Raised when a required item cannot fit the configured token budget."""

    def __init__(self, node_name: str, item_id: str, reason: str = "token_budget_exceeded") -> None:
        super().__init__(
            f"context token budget exceeded: node={node_name}, item={item_id}, reason={reason}"
        )
