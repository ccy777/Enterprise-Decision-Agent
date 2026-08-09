"""Immutable, content-safe contracts for short-lived session memory."""

from __future__ import annotations

import re
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_EVIDENCE_MARKER_PATTERN = re.compile(r"\[(?:D|E)\d+\]")


def _nonblank(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty")
    return value


def _normalize_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


class SessionTurn(BaseModel):
    """One completed user-assistant turn; text is intentionally absent from ``repr``."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    session_id: str
    turn_id: str
    request_id: str
    user_text: str = Field(repr=False)
    assistant_text: str = Field(repr=False)
    created_at: datetime

    @field_validator("session_id", "turn_id", "request_id", "user_text", "assistant_text")
    @classmethod
    def _nonblank_fields(cls, value: str, info) -> str:  # type: ignore[no-untyped-def]
        return _nonblank(value, info.field_name)

    @field_validator("created_at")
    @classmethod
    def _utc_created_at(cls, value: datetime) -> datetime:
        return _normalize_utc(value, "created_at")


class SessionSummary(BaseModel):
    """Immutable rolling summary state; its body is never shown by default."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    session_id: str
    summary_id: str
    previous_summary_id: str | None = None
    source_version: int = Field(ge=0)
    covered_turn_count: int = Field(gt=0)
    covered_through_turn_id: str
    summary_text: str = Field(repr=False)
    created_at: datetime

    @field_validator("session_id", "summary_id", "covered_through_turn_id", "summary_text")
    @classmethod
    def _nonblank_fields(cls, value: str, info) -> str:  # type: ignore[no-untyped-def]
        return _nonblank(value, info.field_name)

    @field_validator("previous_summary_id")
    @classmethod
    def _nonblank_previous_summary_id(cls, value: str | None) -> str | None:
        return None if value is None else _nonblank(value, "previous_summary_id")

    @field_validator("summary_text")
    @classmethod
    def _reject_evidence_markers(cls, value: str) -> str:
        if _EVIDENCE_MARKER_PATTERN.search(value):
            raise ValueError("summary_text must not contain evidence markers")
        return value

    @field_validator("created_at")
    @classmethod
    def _utc_created_at(cls, value: datetime) -> datetime:
        return _normalize_utc(value, "created_at")

    @model_validator(mode="after")
    def _different_previous_summary_id(self) -> SessionSummary:
        if self.previous_summary_id == self.summary_id:
            raise ValueError("previous_summary_id must differ from summary_id")
        return self


class SessionMemorySnapshot(BaseModel):
    """An immutable session view that cannot mutate a store."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    session_id: str
    version: int = Field(ge=0)
    turns: tuple[SessionTurn, ...] = Field(default=(), repr=False)
    summary: SessionSummary | None = Field(default=None, repr=False)
    expires_at: datetime | None = None

    @field_validator("session_id")
    @classmethod
    def _nonblank_session_id(cls, value: str) -> str:
        return _nonblank(value, "session_id")

    @field_validator("expires_at")
    @classmethod
    def _utc_expiry(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _normalize_utc(value, "expires_at")

    @model_validator(mode="after")
    def _summary_matches_session(self) -> SessionMemorySnapshot:
        if self.summary is not None and self.summary.session_id != self.session_id:
            raise ValueError("summary session_id must match snapshot session_id")
        return self


class SessionMemoryPolicy(BaseModel):
    """Fixed retention policy, independent of settings or environment variables."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    ttl_seconds: int = Field(default=1_800, gt=0)
    max_turns: int = Field(default=20, gt=0)


DEFAULT_SESSION_MEMORY_POLICY = SessionMemoryPolicy()
