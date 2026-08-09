"""Atomic rolling-summary compaction tests for the local session-memory store."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from threading import Barrier

import pytest

from decision_agent.memory import (
    InMemorySessionMemoryStore,
    SessionCompactionPrefixError,
    SessionMemoryPolicy,
    SessionSummary,
    SessionSummaryIdentityConflictError,
    SessionSummaryLineageConflictError,
    SessionTurn,
    SessionVersionConflictError,
)


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 7, 24, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: int) -> None:
        self.now += timedelta(seconds=seconds)


def turn(index: int, *, session_id: str = "session-1") -> SessionTurn:
    return SessionTurn(
        session_id=session_id,
        turn_id=f"turn-{index}",
        request_id=f"request-{index}",
        user_text="USER_SECRET_BODY_DO_NOT_LEAK",
        assistant_text="ASSISTANT_SECRET_BODY_DO_NOT_LEAK",
        created_at=datetime(2026, 7, 24, tzinfo=UTC),
    )


def summary(
    *,
    summary_id: str = "summary-1",
    previous_summary_id: str | None = None,
    source_version: int = 3,
    covered_turn_count: int = 2,
    covered_through_turn_id: str = "turn-2",
    session_id: str = "session-1",
    summary_text: str = "SUMMARY_SECRET_BODY_DO_NOT_LEAK",
) -> SessionSummary:
    return SessionSummary(
        session_id=session_id,
        summary_id=summary_id,
        previous_summary_id=previous_summary_id,
        source_version=source_version,
        covered_turn_count=covered_turn_count,
        covered_through_turn_id=covered_through_turn_id,
        summary_text=summary_text,
        created_at=datetime(2026, 7, 24, tzinfo=UTC),
    )


def store(clock: Clock | None = None) -> InMemorySessionMemoryStore:
    return InMemorySessionMemoryStore(
        policy=SessionMemoryPolicy(ttl_seconds=10, max_turns=5), clock=clock
    )


def seeded(memory: InMemorySessionMemoryStore, count: int = 3) -> None:
    for index in range(1, count + 1):
        memory.append_turn(turn(index), expected_version=index - 1)


def test_first_compact_preserves_summary_and_newest_turns() -> None:
    memory = store()
    seeded(memory)
    result = memory.compact(summary(), ("turn-1", "turn-2"), expected_version=3)
    assert result.version == 4
    assert result.summary == summary()
    assert [item.turn_id for item in result.turns] == ["turn-3"]


def test_compact_refreshes_ttl_but_idempotent_retry_does_not() -> None:
    clock = Clock()
    memory = store(clock)
    seeded(memory)
    result = memory.compact(summary(), ("turn-1", "turn-2"), expected_version=3)
    assert result.expires_at == clock.now + timedelta(seconds=10)
    clock.advance(5)
    retry = memory.compact(summary(), ("turn-1", "turn-2"), expected_version=3)
    assert retry == result
    assert retry.expires_at == result.expires_at


def test_second_rolling_compact_requires_lineage_and_accumulates_coverage() -> None:
    memory = store()
    seeded(memory)
    first = memory.compact(summary(), ("turn-1", "turn-2"), expected_version=3)
    memory.append_turn(turn(4), expected_version=first.version)
    second = summary(
        summary_id="summary-2",
        previous_summary_id="summary-1",
        source_version=5,
        covered_turn_count=3,
        covered_through_turn_id="turn-3",
    )
    result = memory.compact(second, ("turn-3",), expected_version=5)
    assert result.version == 6
    assert result.summary == second
    assert [item.turn_id for item in result.turns] == ["turn-4"]


@pytest.mark.parametrize(
    "compacted_ids",
    [(), ("turn-1", "turn-1"), ("turn-2",), ("turn-2", "turn-1"), ("turn-1", "turn-2", "turn-3")],
)
def test_compact_rejects_non_strict_oldest_prefix(compacted_ids: tuple[str, ...]) -> None:
    memory = store()
    seeded(memory)
    candidate = summary(
        covered_turn_count=max(1, len(compacted_ids)),
        covered_through_turn_id="turn-1" if not compacted_ids else compacted_ids[-1],
    )
    with pytest.raises(SessionCompactionPrefixError):
        memory.compact(candidate, compacted_ids, expected_version=3)
    assert memory.read("session-1").version == 3


def test_compact_rejects_stale_version_before_mutating() -> None:
    memory = store()
    seeded(memory)
    with pytest.raises(SessionVersionConflictError):
        memory.compact(summary(), ("turn-1", "turn-2"), expected_version=2)
    assert memory.read("session-1").summary is None


@pytest.mark.parametrize(
    "candidate",
    [
        summary(source_version=2),
        summary(covered_turn_count=1),
        summary(covered_through_turn_id="turn-1"),
        summary(previous_summary_id="unexpected"),
    ],
)
def test_first_compact_rejects_summary_lineage_mismatch(candidate: SessionSummary) -> None:
    memory = store()
    seeded(memory)
    with pytest.raises(SessionSummaryLineageConflictError):
        memory.compact(candidate, ("turn-1", "turn-2"), expected_version=3)


def test_same_summary_id_with_changed_body_fails_closed_without_leaking_body() -> None:
    memory = store()
    seeded(memory)
    memory.compact(summary(), ("turn-1", "turn-2"), expected_version=3)
    changed = summary(summary_text="SUMMARY_CHANGED_BODY_DO_NOT_LEAK")
    with pytest.raises(SessionSummaryIdentityConflictError) as raised:
        memory.compact(changed, ("turn-3",), expected_version=4)
    assert "SUMMARY_CHANGED_BODY_DO_NOT_LEAK" not in str(raised.value)
    assert memory.read("session-1").version == 4


def test_append_preserves_summary_and_clear_removes_it() -> None:
    memory = store()
    seeded(memory)
    compacted = memory.compact(summary(), ("turn-1", "turn-2"), expected_version=3)
    appended = memory.append_turn(turn(4), expected_version=compacted.version)
    assert appended.summary == summary()
    cleared = memory.clear("session-1", expected_version=5)
    assert cleared.version == 0 and cleared.summary is None and cleared.turns == ()


def test_expired_session_compact_fails_closed() -> None:
    clock = Clock()
    memory = store(clock)
    seeded(memory)
    clock.advance(10)
    with pytest.raises(SessionCompactionPrefixError):
        memory.compact(
            summary(source_version=0, covered_turn_count=1, covered_through_turn_id="turn-1"),
            ("turn-1",),
            expected_version=0,
        )
    assert memory.read("session-1").version == 0


def test_concurrent_append_and_compact_allow_only_one_success() -> None:
    memory = store()
    seeded(memory, count=2)
    barrier = Barrier(2)
    compact_candidate = summary(
        source_version=2, covered_turn_count=1, covered_through_turn_id="turn-1"
    )

    def append() -> object:
        barrier.wait()
        try:
            return memory.append_turn(turn(3), expected_version=2)
        except SessionVersionConflictError as error:
            return error

    def compact() -> object:
        barrier.wait()
        try:
            return memory.compact(compact_candidate, ("turn-1",), expected_version=2)
        except SessionVersionConflictError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        append_future = executor.submit(append)
        compact_future = executor.submit(compact)
        results = [append_future.result(), compact_future.result()]
    assert sum(not isinstance(result, SessionVersionConflictError) for result in results) == 1
    assert sum(isinstance(result, SessionVersionConflictError) for result in results) == 1
