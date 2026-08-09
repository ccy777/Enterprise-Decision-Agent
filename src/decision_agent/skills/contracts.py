"""Small immutable contracts for executable Skills."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from decision_agent.coordination.models import SkillResult
from decision_agent.routing.models import RequestRoute, RouterDecision
from decision_agent.tool_calling.models import ToolCallingResult


class SkillDefinition(BaseModel):
    """Metadata that documents a Skill's bounded executable contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=120)
    version: str = Field(min_length=1, max_length=40)
    description: str = Field(min_length=1, max_length=600)
    supported_route: RequestRoute
    input_contract: tuple[str, ...]
    allowed_tools: tuple[str, ...]
    steps: tuple[str, ...]
    output_contract: tuple[str, ...]
    failure_codes: tuple[str, ...]
    required_skills: tuple[str, ...] = ()


class NativeToolCallingExecutor(Protocol):
    """The only execution dependency available to single-domain Skills."""

    async def execute(self, *, user_query: str, decision: RouterDecision) -> ToolCallingResult:
        """Invoke the existing bounded native-tool-calling runtime."""


class ExecutableSkill(Protocol):
    """A registered business capability; never constructed from user input."""

    @property
    def definition(self) -> SkillDefinition:
        """Expose immutable capability metadata."""

    async def execute(self, *, user_query: str, decision: RouterDecision) -> SkillResult:
        """Run the Skill only for its declared route."""

    def is_applicable(self, request: str, decision: RouterDecision) -> bool:
        """Return whether this registered Skill applies to the bounded request."""
