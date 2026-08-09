"""Stable HTTP contracts for the formal Agent execution boundary."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from decision_agent.application import (
    FormalRequest,
    FormalResponse,
    MemoryContextStatus,
    MemoryPersistenceStatus,
    MemorySummarizationStatus,
)
from decision_agent.coordination.models import CoordinatorStatus
from decision_agent.observability import TraceSummary
from decision_agent.routing.models import RequestRoute
from decision_agent.security import SecurityContext

_REQUEST_ID_MAX_LENGTH = 128
_SESSION_ID_MAX_LENGTH = 128
_QUERY_MAX_LENGTH = 8_000


class AgentExecutionRequest(BaseModel):
    """Public request fields mapped explicitly to one internal FormalRequest."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    request_id: str = Field(min_length=1, max_length=_REQUEST_ID_MAX_LENGTH)
    session_id: str | None = Field(default=None, max_length=_SESSION_ID_MAX_LENGTH)
    query: str = Field(min_length=1, max_length=_QUERY_MAX_LENGTH, repr=False)

    @field_validator("request_id", "query")
    @classmethod
    def _required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("request field must be non-empty")
        return value

    @field_validator("session_id")
    @classmethod
    def _optional_session_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("session_id must be non-empty when supplied")
        if any(character.isspace() and character not in {" ", "\t"} for character in normalized):
            raise ValueError("session_id must not contain control characters")
        if any(ord(character) < 32 or ord(character) == 127 for character in normalized):
            raise ValueError("session_id must not contain control characters")
        return normalized

    def to_formal_request(
        self,
        *,
        security_context: SecurityContext | None = None,
    ) -> FormalRequest:
        """Map only the reviewed public fields to the internal application contract."""
        return FormalRequest(
            request_id=self.request_id,
            session_id=self.session_id,
            user_query=self.query,
            security_context=security_context,
        )


class AgentExecutionResponse(BaseModel):
    """Reviewed public projection of FormalResponse and CoordinatorResult."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str
    status: CoordinatorStatus
    route: RequestRoute | None
    skill: str | None
    answer: str | None
    citations: list[str]
    error_code: str | None
    memory_context_status: MemoryContextStatus
    memory_persistence_status: MemoryPersistenceStatus
    memory_summarization_status: MemorySummarizationStatus
    trace: TraceSummary | None = Field(default=None, exclude_if=lambda value: value is None)

    @classmethod
    def from_formal_response(cls, response: FormalResponse) -> AgentExecutionResponse:
        """Copy only the stable fields approved for the HTTP contract."""
        result = response.result
        return cls(
            request_id=response.request_id,
            status=result.status,
            route=result.route,
            skill=result.skill_name,
            answer=result.answer,
            citations=list(result.citations),
            error_code=result.error_code,
            memory_context_status=response.memory_context_status,
            memory_persistence_status=response.memory_persistence_status,
            memory_summarization_status=response.memory_summarization_status,
            trace=_safe_trace_projection(response.trace),
        )


def _safe_trace_projection(value: object) -> TraceSummary | None:
    """Drop an invalid optional Trace projection without changing the business response."""
    if value is None:
        return None
    try:
        return TraceSummary.model_validate(value)
    except (TypeError, ValidationError, ValueError):
        return None


class ApiErrorResponse(BaseModel):
    """Content-safe transport failure without internal exception details."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    message: str
