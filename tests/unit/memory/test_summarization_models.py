"""Offline contracts for rolling-summary service models."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from decision_agent.memory import (
    RollingSummaryDraft,
    RollingSummaryOutcome,
    RollingSummaryPolicy,
    RollingSummaryRequest,
    RollingSummaryStatus,
    SessionMemorySnapshot,
    SessionSummary,
    SessionTurn,
)

NOW = datetime(2026, 7, 24, tzinfo=UTC)
USER_SECRET = "USER_SECRET_BODY_DO_NOT_LEAK"
ASSISTANT_SECRET = "ASSISTANT_SECRET_BODY_DO_NOT_LEAK"
SUMMARY_SECRET = "SUMMARY_SECRET_BODY_DO_NOT_LEAK"


def turn(index: int = 1, *, session_id: str = "session-1") -> SessionTurn:
    return SessionTurn(
        session_id=session_id,
        turn_id=f"turn-{index}",
        request_id=f"request-{index}",
        user_text=USER_SECRET,
        assistant_text=ASSISTANT_SECRET,
        created_at=NOW,
    )


def request(**overrides: object) -> RollingSummaryRequest:
    values: dict[str, object] = {
        "session_id": "session-1",
        "source_version": 3,
        "previous_summary_text": SUMMARY_SECRET,
        "turns": (turn(),),
        "target_summary_id": "rs1_safe",
        "max_summary_chars": 200,
    }
    values.update(overrides)
    return RollingSummaryRequest(**values)


@pytest.mark.parametrize(
    "overrides",
    [
        {"trigger_turns": 1},
        {"retain_recent_turns": 0},
        {"trigger_turns": 3, "retain_recent_turns": 3},
        {"max_source_chars": 0},
        {"max_summary_chars": 0},
    ],
)
def test_policy_rejects_invalid_limits(overrides: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        RollingSummaryPolicy(**overrides)


def test_policy_is_frozen_and_forbids_extra_fields() -> None:
    policy = RollingSummaryPolicy()
    with pytest.raises(ValidationError):
        policy.trigger_turns = 7  # type: ignore[misc]
    with pytest.raises(ValidationError):
        RollingSummaryPolicy(unexpected=True)  # type: ignore[call-arg]


def test_request_is_frozen_requires_nonempty_same_session_turns_and_hides_bodies() -> None:
    value = request()
    rendered = f"{value!r} {value}"
    assert all(secret not in rendered for secret in (USER_SECRET, ASSISTANT_SECRET, SUMMARY_SECRET))
    with pytest.raises(ValidationError):
        value.source_version = 4  # type: ignore[misc]
    with pytest.raises(ValidationError):
        request(turns=())
    with pytest.raises(ValidationError):
        request(turns=(turn(session_id="other-session"),))


@pytest.mark.parametrize(
    "summary_text", ["", "  ", "ordinary [D1] marker", "ordinary [E12] marker"]
)
def test_draft_rejects_blank_or_evidence_marked_text(summary_text: str) -> None:
    with pytest.raises(ValidationError) as raised:
        RollingSummaryDraft(summary_text=summary_text)
    if summary_text.strip():
        assert summary_text not in str(raised.value)


def test_draft_is_frozen_hides_text_and_forbids_extra_fields() -> None:
    draft = RollingSummaryDraft(summary_text=SUMMARY_SECRET)
    assert SUMMARY_SECRET not in repr(draft)
    assert SUMMARY_SECRET not in str(draft)
    with pytest.raises(ValidationError):
        draft.summary_text = "changed"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        RollingSummaryDraft(summary_text="safe", extra=True)  # type: ignore[call-arg]


def test_outcome_enforces_status_metadata_without_snapshot_body_leakage() -> None:
    snapshot = SessionMemorySnapshot(session_id="session-1", version=0)
    no_op = RollingSummaryOutcome(
        status=RollingSummaryStatus.NOT_REQUIRED, snapshot=snapshot, compacted_turn_count=0
    )
    assert no_op.summary_id is None
    assert USER_SECRET not in repr(no_op)
    with pytest.raises(ValidationError):
        RollingSummaryOutcome(
            status=RollingSummaryStatus.NOT_REQUIRED,
            snapshot=snapshot,
            compacted_turn_count=1,
        )


def test_compacted_outcome_requires_matching_authoritative_summary() -> None:
    summary = SessionSummary(
        session_id="session-1",
        summary_id="rs1_safe",
        source_version=2,
        covered_turn_count=1,
        covered_through_turn_id="turn-1",
        summary_text=SUMMARY_SECRET,
        created_at=NOW,
    )
    snapshot = SessionMemorySnapshot(
        session_id="session-1", version=3, turns=(turn(2),), summary=summary
    )
    outcome = RollingSummaryOutcome(
        status=RollingSummaryStatus.COMPACTED,
        snapshot=snapshot,
        compacted_turn_count=1,
        summary_id="rs1_safe",
    )
    assert outcome.snapshot == snapshot
    assert SUMMARY_SECRET not in repr(outcome)
    with pytest.raises(ValidationError):
        RollingSummaryOutcome(
            status=RollingSummaryStatus.COMPACTED,
            snapshot=snapshot,
            compacted_turn_count=1,
            summary_id="rs1_other",
        )
