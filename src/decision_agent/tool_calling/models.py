"""Strict contracts for a bounded native function-calling turn."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from decision_agent.routing.models import RequestRoute


class NativeToolCallingStatus(StrEnum):
    """Terminal outcomes for one controlled native-tool-calling turn."""

    COMPLETED = "completed"
    UNSUPPORTED = "unsupported"
    REQUIRES_COORDINATOR = "requires_coordinator"
    FAILED = "failed"


class AgentToolResult(BaseModel):
    """Safe, high-level Agent output sent back through a tool message only."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(pattern=r"^(succeeded|failed)$")
    answer: str | None = Field(default=None, max_length=8_000)
    citations: list[str] = Field(default_factory=list)
    error_code: str | None = Field(default=None, max_length=120)

    @field_validator("answer", "error_code")
    @classmethod
    def reject_blank_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("text cannot be blank")
        return normalized

    @model_validator(mode="after")
    def validate_terminal_contract(self) -> AgentToolResult:
        if self.status == "succeeded":
            if self.answer is None or self.error_code is not None:
                raise ValueError("succeeded tool result requires answer and no error_code")
        elif self.answer is not None or self.citations or self.error_code is None:
            raise ValueError("failed tool result requires error_code only")
        return self


class FinalAnswerDraft(BaseModel):
    """Untrusted final model output, constrained to exact tool-result preservation."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1, max_length=8_000)
    citations: list[str] = Field(default_factory=list)

    @field_validator("answer")
    @classmethod
    def reject_blank_answer(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("answer cannot be blank")
        return normalized


class ToolCallingResult(BaseModel):
    """Public terminal result with no raw provider, MCP, or Agent internals."""

    model_config = ConfigDict(extra="forbid")

    status: NativeToolCallingStatus
    route: RequestRoute
    selected_tool: str | None = None
    tool_call_id: str | None = None
    answer: str | None = Field(default=None, max_length=8_000)
    citations: list[str] = Field(default_factory=list)
    steps: int = Field(ge=0, le=2)
    error_code: str | None = Field(default=None, max_length=120)
    http_status: int | None = Field(default=None, ge=100, le=599)

    @model_validator(mode="after")
    def validate_terminal_contract(self) -> ToolCallingResult:
        if self.status is NativeToolCallingStatus.COMPLETED:
            if (
                self.selected_tool is None
                or self.tool_call_id is None
                or self.answer is None
                or self.error_code is not None
                or self.steps != 2
            ):
                raise ValueError("completed result requires one tool and final answer")
        elif self.status is NativeToolCallingStatus.UNSUPPORTED:
            if self.selected_tool is not None or self.tool_call_id is not None or self.steps != 0:
                raise ValueError("unsupported result cannot call a tool")
        elif self.status is NativeToolCallingStatus.REQUIRES_COORDINATOR:
            if (
                self.selected_tool is not None
                or self.tool_call_id is not None
                or self.answer is not None
                or self.citations
                or self.error_code != "requires_coordinator"
                or self.steps != 0
            ):
                raise ValueError("mixed route must await a coordinator")
        elif self.answer is not None or self.citations or self.error_code is None:
            raise ValueError("failed result cannot expose unvalidated answer output")
        if self.http_status is not None and self.status is not NativeToolCallingStatus.FAILED:
            raise ValueError("http_status is valid only for a failed result")
        return self
