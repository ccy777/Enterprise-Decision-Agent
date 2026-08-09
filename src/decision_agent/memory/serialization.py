"""Strict versioned JSON serialization for Redis session-memory values."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from decision_agent.memory.models import SessionMemorySnapshot, SessionSummary, SessionTurn
from decision_agent.memory.store import SessionMemoryCorruptionError

_SCHEMA_VERSION = 2
_V1_STATE_FIELDS = frozenset({"schema_version", "session_id", "version", "turns"})
_V2_STATE_FIELDS = frozenset({"schema_version", "session_id", "version", "summary", "turns"})
_TURN_FIELDS = frozenset(
    {"session_id", "turn_id", "request_id", "user_text", "assistant_text", "created_at"}
)
_SUMMARY_FIELDS = frozenset(
    {
        "session_id",
        "summary_id",
        "previous_summary_id",
        "source_version",
        "covered_turn_count",
        "covered_through_turn_id",
        "summary_text",
        "created_at",
    }
)


def serialize_session_snapshot(snapshot: SessionMemorySnapshot) -> str:
    """Encode a complete bounded snapshot deterministically without Python object metadata."""
    value = {
        "schema_version": _SCHEMA_VERSION,
        "session_id": snapshot.session_id,
        "version": snapshot.version,
        "summary": None if snapshot.summary is None else _serialize_summary(snapshot.summary),
        "turns": [
            {
                "session_id": turn.session_id,
                "turn_id": turn.turn_id,
                "request_id": turn.request_id,
                "user_text": turn.user_text,
                "assistant_text": turn.assistant_text,
                "created_at": turn.created_at.isoformat(),
            }
            for turn in snapshot.turns
        ],
    }
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def deserialize_session_snapshot(
    payload: bytes | str, *, expected_session_id: str, expires_at: datetime
) -> SessionMemorySnapshot:
    """Decode and fully validate one stored snapshot without exposing bad payload content."""
    try:
        text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        if not isinstance(text, str):
            raise TypeError("payload must be text")
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("invalid state fields")
        schema_version = value.get("schema_version")
        if schema_version == 1:
            if frozenset(value) != _V1_STATE_FIELDS:
                raise ValueError("invalid v1 state fields")
            summary = None
        elif schema_version == _SCHEMA_VERSION:
            if frozenset(value) != _V2_STATE_FIELDS:
                raise ValueError("invalid v2 state fields")
            summary = _deserialize_summary(value.get("summary"), expected_session_id)
        else:
            raise ValueError("unsupported schema")
        if value.get("session_id") != expected_session_id:
            raise ValueError("session mismatch")
        version = value.get("version")
        turns_value = value.get("turns")
        if not isinstance(version, int) or isinstance(version, bool) or version < 0:
            raise ValueError("invalid version")
        if not isinstance(turns_value, list):
            raise ValueError("invalid turns")
        turns = tuple(_deserialize_turn(item) for item in turns_value)
        if any(turn.session_id != expected_session_id for turn in turns):
            raise ValueError("turn session mismatch")
        return SessionMemorySnapshot(
            session_id=expected_session_id,
            version=version,
            turns=turns,
            summary=summary,
            expires_at=expires_at,
        )
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        raise SessionMemoryCorruptionError(session_id=expected_session_id) from None


def _deserialize_turn(value: Any) -> SessionTurn:
    if not isinstance(value, dict) or frozenset(value) != _TURN_FIELDS:
        raise ValueError("invalid turn fields")
    created_at = value.get("created_at")
    if not isinstance(created_at, str):
        raise ValueError("invalid timestamp")
    return SessionTurn(
        session_id=value.get("session_id"),
        turn_id=value.get("turn_id"),
        request_id=value.get("request_id"),
        user_text=value.get("user_text"),
        assistant_text=value.get("assistant_text"),
        created_at=datetime.fromisoformat(created_at),
    )


def _serialize_summary(summary: SessionSummary) -> dict[str, object]:
    return {
        "session_id": summary.session_id,
        "summary_id": summary.summary_id,
        "previous_summary_id": summary.previous_summary_id,
        "source_version": summary.source_version,
        "covered_turn_count": summary.covered_turn_count,
        "covered_through_turn_id": summary.covered_through_turn_id,
        "summary_text": summary.summary_text,
        "created_at": summary.created_at.isoformat(),
    }


def _deserialize_summary(value: Any, expected_session_id: str) -> SessionSummary | None:
    if value is None:
        return None
    if not isinstance(value, dict) or frozenset(value) != _SUMMARY_FIELDS:
        raise ValueError("invalid summary fields")
    if value.get("session_id") != expected_session_id:
        raise ValueError("summary session mismatch")
    created_at = value.get("created_at")
    if not isinstance(created_at, str):
        raise ValueError("invalid summary timestamp")
    return SessionSummary(
        session_id=value.get("session_id"),
        summary_id=value.get("summary_id"),
        previous_summary_id=value.get("previous_summary_id"),
        source_version=value.get("source_version"),
        covered_turn_count=value.get("covered_turn_count"),
        covered_through_turn_id=value.get("covered_through_turn_id"),
        summary_text=value.get("summary_text"),
        created_at=datetime.fromisoformat(created_at),
    )
