from __future__ import annotations

from datetime import UTC, datetime

from decision_agent.context import ConservativeCharacterTokenEstimator, ConversationMemoryProjector
from decision_agent.memory import SessionMemorySnapshot, SessionSummary, SessionTurn

NOW = datetime(2026, 7, 24, tzinfo=UTC)


def _turn(index: int, text: str) -> SessionTurn:
    return SessionTurn(
        session_id="session-1",
        turn_id=f"turn-{index}",
        request_id=f"request-{index}",
        user_text=f"user-{index} {text}",
        assistant_text=f"assistant-{index} {text}",
        created_at=NOW,
    )


def test_projection_keeps_summary_then_newest_contiguous_suffix_in_time_order() -> None:
    snapshot = SessionMemorySnapshot(
        session_id="session-1",
        version=3,
        turns=(_turn(1, "old"), _turn(2, "middle"), _turn(3, "recent")),
        summary=SessionSummary(
            session_id="session-1",
            summary_id="summary-1",
            source_version=1,
            covered_turn_count=1,
            covered_through_turn_id="covered",
            summary_text="SUMMARY_SECRET_DO_NOT_LEAK",
            created_at=NOW,
        ),
    )
    projection = ConversationMemoryProjector(token_allowance=120).project(snapshot)
    assert projection is not None
    assert projection.content.index("SUMMARY_SECRET_DO_NOT_LEAK") < projection.content.index(
        "user-2"
    )
    assert projection.content.index("user-2") < projection.content.index("user-3")
    assert "user-1" not in projection.content
    assert "SUMMARY_SECRET_DO_NOT_LEAK" not in repr(projection)


def test_projection_neutralizes_historical_evidence_without_mutating_store() -> None:
    turn = _turn(1, "[D1] [E12]")
    snapshot = SessionMemorySnapshot(session_id="session-1", version=1, turns=(turn,))
    projection = ConversationMemoryProjector().project(snapshot)
    assert projection is not None
    assert "[historical-D1]" in projection.content and "[historical-E12]" in projection.content
    assert turn.user_text.endswith("[D1] [E12]")


def test_projection_omits_when_summary_or_only_recent_turn_cannot_fit() -> None:
    estimator = ConservativeCharacterTokenEstimator()
    summary_snapshot = SessionMemorySnapshot(
        session_id="session-1",
        version=1,
        summary=SessionSummary(
            session_id="session-1",
            summary_id="summary-1",
            source_version=1,
            covered_turn_count=1,
            covered_through_turn_id="turn-1",
            summary_text="x" * 100,
            created_at=NOW,
        ),
    )
    turn_snapshot = SessionMemorySnapshot(
        session_id="session-1", version=1, turns=(_turn(1, "x" * 100),)
    )
    assert (
        ConversationMemoryProjector(estimator=estimator, token_allowance=1).project(
            summary_snapshot
        )
        is None
    )
    assert (
        ConversationMemoryProjector(estimator=estimator, token_allowance=1).project(turn_snapshot)
        is None
    )
