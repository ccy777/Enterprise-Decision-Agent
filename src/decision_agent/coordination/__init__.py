"""Coordinator contracts and routing-to-Skill execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

from decision_agent.coordination.models import (
    CoordinatorResult,
    CoordinatorStatus,
    SkillResult,
    SkillStatus,
)

if TYPE_CHECKING:
    from decision_agent.coordination.coordinator import Coordinator


def build_default_coordinator(*args: object, **kwargs: object) -> Coordinator:
    """Lazily import the composition root to keep shared contracts importable."""
    from decision_agent.coordination.factory import build_default_coordinator as _build

    return _build(*args, **kwargs)  # type: ignore[arg-type]


def __getattr__(name: str) -> object:
    """Defer Coordinator import so shared result contracts remain cycle-free."""
    if name == "Coordinator":
        from decision_agent.coordination.coordinator import Coordinator

        return Coordinator
    raise AttributeError(name)


__all__ = [
    "Coordinator",
    "CoordinatorResult",
    "CoordinatorStatus",
    "SkillResult",
    "SkillStatus",
    "build_default_coordinator",
]
