"""Contracts for immutable, non-evidence rolling summary state."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from decision_agent.memory import SessionMemorySnapshot, SessionSummary, SessionTurn

NOW = datetime(2026, 7, 24, 9, 30, tzinfo=UTC)
SUMMARY_SECRET = "SUMMARY_SECRET_BODY_DO_NOT_LEAK"
USER_SECRET = "USER_SECRET_BODY_DO_NOT_LEAK"
ASSISTANT_SECRET = "ASSISTANT_SECRET_BODY_DO_NOT_LEAK"


def summary(**overrides: object) -> SessionSummary:
    values: dict[str, object] = {
        "session_id": "session-1",
        "summary_id": "summary-1",
        "previous_summary_id": None,
        "source_version": 2,
        "covered_turn_count": 1,
        "covered_through_turn_id": "turn-1",
        "summary_text": SUMMARY_SECRET,
        "created_at": NOW,
    }
    values.update(overrides)
    return SessionSummary(**values)


def turn() -> SessionTurn:
    return SessionTurn(
        session_id="session-1",
        turn_id="turn-1",
        request_id="request-1",
        user_text=USER_SECRET,
        assistant_text=ASSISTANT_SECRET,
        created_at=NOW,
    )


def test_summary_is_frozen_utc_normalized_and_hides_body() -> None:
    value = summary(created_at=NOW.astimezone(timezone(timedelta(hours=8))))
    assert value.created_at == NOW
    assert SUMMARY_SECRET not in repr(value)
    assert SUMMARY_SECRET not in str(value)
    with pytest.raises(ValidationError):
        value.summary_text = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("session_id", " "),
        ("summary_id", " "),
        ("covered_through_turn_id", " "),
        ("summary_text", " "),
        ("previous_summary_id", " "),
        ("source_version", -1),
        ("covered_turn_count", 0),
    ],
)
def test_summary_rejects_invalid_required_values(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        summary(**{field: value})


def test_summary_rejects_naive_time_and_self_predecessor() -> None:
    with pytest.raises(ValidationError):
        summary(created_at=datetime(2026, 7, 24, 9, 30))
    with pytest.raises(ValidationError):
        summary(previous_summary_id="summary-1")


@pytest.mark.parametrize("marker", ["[D1]", "[D12]", "[E1]", "[E12]"])
def test_summary_rejects_evidence_markers(marker: str) -> None:
    with pytest.raises(ValidationError) as raised:
        summary(summary_text=f"summary body {marker}")
    assert "summary body" not in str(raised.value)


def test_summary_allows_ordinary_bracketed_text() -> None:
    assert (
        summary(summary_text="[draft] item (not evidence)").summary_text
        == "[draft] item (not evidence)"
    )


def test_snapshot_rejects_summary_for_another_session_and_hides_all_bodies() -> None:
    with pytest.raises(ValidationError):
        SessionMemorySnapshot(session_id="session-2", version=1, summary=summary())
    snapshot = SessionMemorySnapshot(
        session_id="session-1", version=1, turns=(turn(),), summary=summary()
    )
    rendered = f"{snapshot!r} {snapshot}"
    assert SUMMARY_SECRET not in rendered
    assert USER_SECRET not in rendered
    assert ASSISTANT_SECRET not in rendered


def test_snapshot_defaults_to_no_summary() -> None:
    assert SessionMemorySnapshot(session_id="session-1", version=0).summary is None
