"""Content-safe internal envelope models for the formal request path."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from decision_agent.coordination.models import CoordinatorResult
from decision_agent.observability import TraceSummary
from decision_agent.security import SecurityContext

_SESSION_ID_MAX_LENGTH = 128


class MemoryContextStatus(StrEnum):
    """Read-projection outcome only; this does not describe persistence."""

    NOT_REQUESTED = "not_requested"
    EMPTY = "empty"
    PROJECTED = "projected"
    OMITTED_BY_BUDGET = "omitted_by_budget"


class MemoryPersistenceStatus(StrEnum):
    """Content-safe outcome of the optional successful-turn write."""

    NOT_REQUESTED = "not_requested"
    SKIPPED = "skipped"
    PERSISTED = "persisted"
    VERSION_CONFLICT = "version_conflict"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"
    STORE_FAILURE = "store_failure"


class MemorySummarizationStatus(StrEnum):
    """Content-safe outcome of post-append rolling summarization."""

    NOT_REQUESTED = "not_requested"
    SKIPPED = "skipped"
    NOT_NEEDED = "not_needed"
    COMPACTED = "compacted"
    VERSION_CONFLICT = "version_conflict"
    PROVIDER_FAILURE = "provider_failure"
    STORE_FAILURE = "store_failure"


class FormalRequest(BaseModel):
    """Validated internal request with distinct request and session identities."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    request_id: str = Field(min_length=1)
    user_query: str = Field(min_length=1, repr=False)
    session_id: str | None = Field(default=None, repr=False)
    security_context: SecurityContext | None = Field(default=None, repr=False)

    @field_validator("request_id", "user_query")
    @classmethod
    def _required_text(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
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
        if len(normalized) > _SESSION_ID_MAX_LENGTH:
            raise ValueError("session_id exceeds maximum length")
        if any(character.isspace() and character not in {" ", "\t"} for character in normalized):
            raise ValueError("session_id must not contain control characters")
        if any(ord(character) < 32 or ord(character) == 127 for character in normalized):
            raise ValueError("session_id must not contain control characters")
        return normalized


class FormalResponse(BaseModel):
    """Content-safe internal response envelope for the formal memory boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    request_id: str = Field(min_length=1)
    result: CoordinatorResult = Field(repr=False)
    memory_context_status: MemoryContextStatus
    memory_persistence_status: MemoryPersistenceStatus = MemoryPersistenceStatus.NOT_REQUESTED
    memory_summarization_status: MemorySummarizationStatus = MemorySummarizationStatus.NOT_REQUESTED
    trace: TraceSummary | None = Field(default=None, repr=False)
