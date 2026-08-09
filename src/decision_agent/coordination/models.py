"""Safe public results shared by the Coordinator and executable Skills."""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from decision_agent.routing.models import RequestRoute

_KNOWLEDGE_CITATION = re.compile(r"^\[E\d+\]$")
_DATA_CITATION = re.compile(r"^\[D\d+\]$")


class SkillStatus(StrEnum):
    """Terminal result of one executable business Skill."""

    COMPLETED = "completed"
    FAILED = "failed"


class CoordinatorStatus(StrEnum):
    """Terminal result of the bounded Coordinator."""

    COMPLETED = "completed"
    UNSUPPORTED = "unsupported"
    REQUIRES_MIXED_SKILL = "requires_mixed_skill"
    FAILED = "failed"


class SkillResult(BaseModel):
    """Validated result produced by a registered single-domain Skill."""

    model_config = ConfigDict(extra="forbid")

    status: SkillStatus
    skill_name: str = Field(min_length=1, max_length=120)
    skill_version: str = Field(min_length=1, max_length=40)
    route: RequestRoute
    answer: str | None = Field(default=None, max_length=8_000)
    citations: list[str] = Field(default_factory=list)
    executed_steps: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    selected_tool: str | None = Field(default=None, max_length=120)
    error_code: str | None = Field(default=None, max_length=120)

    @field_validator("answer", "error_code")
    @classmethod
    def _nonblank_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("text cannot be blank")
        return normalized

    @field_validator("executed_steps")
    @classmethod
    def _valid_steps(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not step.strip() for step in value):
            raise ValueError("executed_steps must be nonempty named steps")
        return value

    @model_validator(mode="after")
    def _terminal_contract(self) -> SkillResult:
        if self.status is SkillStatus.COMPLETED:
            if self.answer is None or self.error_code is not None:
                raise ValueError("completed SkillResult requires answer and no error")
            if self.route is RequestRoute.KNOWLEDGE:
                citations_valid = self.selected_tool is not None and all(
                    _KNOWLEDGE_CITATION.fullmatch(citation) for citation in self.citations
                )
            elif self.route is RequestRoute.DATA:
                citations_valid = self.selected_tool is not None and all(
                    _DATA_CITATION.fullmatch(citation) for citation in self.citations
                )
            elif self.route is RequestRoute.MIXED:
                citations_valid = self.selected_tool is None and all(
                    _KNOWLEDGE_CITATION.fullmatch(citation) or _DATA_CITATION.fullmatch(citation)
                    for citation in self.citations
                )
            else:
                citations_valid = False
            if not citations_valid:
                raise ValueError("completed SkillResult has invalid route citations")
        elif (
            self.answer is not None
            or self.citations
            or self.selected_tool is not None
            or self.error_code is None
        ):
            raise ValueError("failed SkillResult cannot expose tool output and requires error_code")
        return self


class CoordinatorResult(BaseModel):
    """Public Coordinator output without provider, MCP, or database internals."""

    model_config = ConfigDict(extra="forbid")

    status: CoordinatorStatus
    route: RequestRoute | None = None
    skill_name: str | None = Field(default=None, max_length=120)
    answer: str | None = Field(default=None, max_length=8_000)
    citations: list[str] = Field(default_factory=list)
    coordinator_steps: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    tool_steps: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    error_code: str | None = Field(default=None, max_length=120)
    memory_context_selected: bool = False

    @model_validator(mode="after")
    def _terminal_contract(self) -> CoordinatorResult:
        if self.status is CoordinatorStatus.COMPLETED:
            if (
                self.route not in {RequestRoute.KNOWLEDGE, RequestRoute.DATA, RequestRoute.MIXED}
                or self.skill_name is None
            ):
                raise ValueError("completed CoordinatorResult requires a selected Skill")
            if self.answer is None or self.error_code is not None or not self.tool_steps:
                raise ValueError("completed CoordinatorResult requires trusted Skill output")
        elif self.status is CoordinatorStatus.UNSUPPORTED:
            if self.route is not RequestRoute.UNSUPPORTED or self.skill_name is not None:
                raise ValueError("unsupported result cannot select a Skill")
        elif self.status is CoordinatorStatus.REQUIRES_MIXED_SKILL:
            if self.route is not RequestRoute.MIXED or self.skill_name is not None:
                raise ValueError("mixed result cannot select a single-domain Skill")
            if self.error_code != "requires_mixed_skill":
                raise ValueError("mixed result requires stable error_code")
        elif (
            self.answer is not None
            or self.citations
            or self.skill_name is not None
            or self.error_code is None
        ):
            raise ValueError(
                "failed CoordinatorResult cannot expose output and requires error_code"
            )
        return self
