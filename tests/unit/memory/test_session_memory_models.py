from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from decision_agent.memory import SessionMemoryPolicy, SessionMemorySnapshot, SessionTurn

NOW = datetime(2026, 7, 24, 8, 30, tzinfo=UTC)


def turn(**overrides: object) -> SessionTurn:
    values: dict[str, object] = {
        "session_id": "session-1",
        "turn_id": "turn-1",
        "request_id": "request-1",
        "user_text": "USER_SECRET_BODY_DO_NOT_LEAK",
        "assistant_text": "ASSISTANT_SECRET_BODY_DO_NOT_LEAK",
        "created_at": NOW,
    }
    values.update(overrides)
    return SessionTurn(**values)


def test_turn_is_immutable_utc_normalized_and_content_safe_in_repr() -> None:
    value = turn(created_at=NOW.astimezone(timezone(timedelta(hours=8))))
    assert value.created_at == NOW
    assert "USER_SECRET_BODY_DO_NOT_LEAK" not in repr(value)
    assert "ASSISTANT_SECRET_BODY_DO_NOT_LEAK" not in repr(value)
    assert "USER_SECRET_BODY_DO_NOT_LEAK" not in str(value)
    assert "ASSISTANT_SECRET_BODY_DO_NOT_LEAK" not in str(value)
    with pytest.raises(ValidationError):
        value.user_text = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "field", ["session_id", "turn_id", "request_id", "user_text", "assistant_text"]
)
def test_turn_rejects_blank_required_fields(field: str) -> None:
    with pytest.raises(ValidationError):
        turn(**{field: "  "})


def test_turn_rejects_naive_timestamp_without_echoing_text() -> None:
    with pytest.raises(ValidationError) as raised:
        turn(created_at=datetime(2026, 7, 24, 8, 30))
    assert "USER_SECRET_BODY_DO_NOT_LEAK" not in str(raised.value)
    assert "ASSISTANT_SECRET_BODY_DO_NOT_LEAK" not in str(raised.value)


@pytest.mark.parametrize("values", [{"ttl_seconds": 0}, {"max_turns": 0}, {"ttl_seconds": -1}])
def test_policy_rejects_non_positive_values(values: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        SessionMemoryPolicy(**values)


def test_snapshot_is_frozen_and_hides_turn_text() -> None:
    snapshot = SessionMemorySnapshot(session_id="session-1", version=1, turns=(turn(),))
    assert isinstance(snapshot.turns, tuple)
    assert "USER_SECRET_BODY_DO_NOT_LEAK" not in repr(snapshot)
    assert "ASSISTANT_SECRET_BODY_DO_NOT_LEAK" not in repr(snapshot)
    assert "USER_SECRET_BODY_DO_NOT_LEAK" not in str(snapshot)
    assert "ASSISTANT_SECRET_BODY_DO_NOT_LEAK" not in str(snapshot)
    with pytest.raises(ValidationError):
        snapshot.version = 2  # type: ignore[misc]
