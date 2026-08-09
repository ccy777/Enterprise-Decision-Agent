"""A small, immutable-from-callers registry of executable Skills."""

from __future__ import annotations

from decision_agent.routing.models import RequestRoute
from decision_agent.skills.contracts import ExecutableSkill


class SkillRegistryError(ValueError):
    """Raised for invalid or missing registry bindings."""


class SkillRegistry:
    """Register each trusted Skill once and bind at most one Skill per route."""

    def __init__(self) -> None:
        self._by_name: dict[str, ExecutableSkill] = {}
        self._by_route: dict[RequestRoute, ExecutableSkill] = {}

    def register(self, skill: ExecutableSkill) -> None:
        definition = skill.definition
        if definition.name in self._by_name:
            raise SkillRegistryError("skill_name_already_registered")
        if definition.supported_route in self._by_route:
            raise SkillRegistryError("skill_route_already_registered")
        self._by_name[definition.name] = skill
        self._by_route[definition.supported_route] = skill

    def get(self, name: str) -> ExecutableSkill:
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise SkillRegistryError("skill_not_registered") from exc

    def for_route(self, route: RequestRoute) -> ExecutableSkill:
        try:
            return self._by_route[route]
        except KeyError as exc:
            raise SkillRegistryError("skill_route_not_registered") from exc

    @property
    def names(self) -> tuple[str, ...]:
        """Expose a snapshot rather than a mutable registry mapping."""
        return tuple(self._by_name)
