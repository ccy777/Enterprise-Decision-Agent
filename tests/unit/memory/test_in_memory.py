from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from threading import Barrier

import pytest

from decision_agent.memory import (
    InMemorySessionMemoryStore,
    SessionMemoryPolicy,
    SessionTurn,
    SessionTurnConflictError,
    SessionVersionConflictError,
)


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 7, 24, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: int) -> None:
        self.now += timedelta(seconds=seconds)


def make_turn(
    *,
    session_id: str = "session-1",
    turn_id: str = "turn-1",
    request_id: str = "request-1",
    user_text: str = "USER_SECRET_BODY_DO_NOT_LEAK",
    assistant_text: str = "ASSISTANT_SECRET_BODY_DO_NOT_LEAK",
) -> SessionTurn:
    return SessionTurn(
        session_id=session_id,
        turn_id=turn_id,
        request_id=request_id,
        user_text=user_text,
        assistant_text=assistant_text,
        created_at=datetime(2026, 7, 24, tzinfo=UTC),
    )


def store(*, clock: Clock | None = None, max_turns: int = 3) -> InMemorySessionMemoryStore:
    return InMemorySessionMemoryStore(
        policy=SessionMemoryPolicy(ttl_seconds=10, max_turns=max_turns), clock=clock
    )


def test_missing_session_and_clear_are_deterministically_empty() -> None:
    memory = store()
    assert memory.read("session-1").model_dump() == {
        "session_id": "session-1",
        "version": 0,
        "turns": (),
        "summary": None,
        "expires_at": None,
    }
    assert memory.clear("session-1", expected_version=0).version == 0


def test_append_preserves_order_and_snapshot_cannot_change_store() -> None:
    memory = store()
    first = memory.append_turn(make_turn(), expected_version=0)
    second = memory.append_turn(
        make_turn(turn_id="turn-2", request_id="request-2"), expected_version=1
    )
    assert first.version == 1
    assert [turn.turn_id for turn in second.turns] == ["turn-1", "turn-2"]
    assert memory.read("session-1") == second


def test_sessions_and_store_instances_are_isolated() -> None:
    first = store()
    second = store()
    first.append_turn(make_turn(), expected_version=0)
    first.append_turn(make_turn(session_id="session-2"), expected_version=0)
    assert second.read("session-1").version == 0
    assert first.read("session-2").turns[0].session_id == "session-2"


def test_stale_version_and_clear_do_not_change_existing_session() -> None:
    memory = store()
    memory.append_turn(make_turn(), expected_version=0)
    with pytest.raises(SessionVersionConflictError) as append_error:
        memory.append_turn(make_turn(turn_id="turn-2", request_id="request-2"), expected_version=0)
    with pytest.raises(SessionVersionConflictError):
        memory.clear("session-1", expected_version=0)
    assert memory.read("session-1").version == 1
    assert "USER_SECRET_BODY_DO_NOT_LEAK" not in str(append_error.value)
    assert "ASSISTANT_SECRET_BODY_DO_NOT_LEAK" not in str(append_error.value)


def test_clear_version_match_removes_only_one_session() -> None:
    memory = store()
    memory.append_turn(make_turn(), expected_version=0)
    memory.append_turn(make_turn(session_id="session-2"), expected_version=0)
    assert memory.clear("session-1", expected_version=1).version == 0
    assert memory.read("session-2").version == 1


@pytest.mark.parametrize("identity", ["turn_id", "request_id"])
def test_idempotent_retry_precedes_stale_version(identity: str) -> None:
    memory = store()
    original = make_turn()
    memory.append_turn(original, expected_version=0)
    retry = original.model_copy()
    if identity == "turn_id":
        retry = original.model_copy(update={"request_id": original.request_id})
    result = memory.append_turn(retry, expected_version=0)
    assert result.version == 1 and len(result.turns) == 1


def test_idempotent_retry_does_not_refresh_ttl() -> None:
    clock = Clock()
    memory = store(clock=clock)
    original = make_turn()
    first = memory.append_turn(original, expected_version=0)
    clock.advance(5)
    retry = memory.append_turn(original, expected_version=0)
    assert retry.version == 1
    assert retry.expires_at == first.expires_at


@pytest.mark.parametrize("identity", ["turn_id", "request_id"])
def test_reused_identity_with_different_content_fails_closed(identity: str) -> None:
    memory = store()
    memory.append_turn(make_turn(), expected_version=0)
    changed = (
        make_turn(request_id="request-2", user_text="SECRET_CHANGED")
        if identity == "turn_id"
        else make_turn(turn_id="turn-2", user_text="SECRET_CHANGED")
    )
    with pytest.raises(SessionTurnConflictError) as raised:
        memory.append_turn(changed, expected_version=1)
    assert memory.read("session-1").version == 1
    assert "SECRET_CHANGED" not in str(raised.value)
    assert "USER_SECRET_BODY_DO_NOT_LEAK" not in str(raised.value)
    assert "ASSISTANT_SECRET_BODY_DO_NOT_LEAK" not in str(raised.value)


def test_cross_identifier_conflict_fails_closed_without_mutating_session() -> None:
    memory = store()
    memory.append_turn(make_turn(turn_id="t1", request_id="r1"), expected_version=0)
    memory.append_turn(make_turn(turn_id="t2", request_id="r2"), expected_version=1)
    with pytest.raises(SessionTurnConflictError) as raised:
        memory.append_turn(make_turn(turn_id="t1", request_id="r2"), expected_version=2)
    snapshot = memory.read("session-1")
    assert snapshot.version == 2
    assert [(turn.turn_id, turn.request_id) for turn in snapshot.turns] == [
        ("t1", "r1"),
        ("t2", "r2"),
    ]
    assert "USER_SECRET_BODY_DO_NOT_LEAK" not in str(raised.value)


def test_different_sessions_may_reuse_turn_and_request_identifiers() -> None:
    memory = store()
    memory.append_turn(make_turn(), expected_version=0)
    second = memory.append_turn(make_turn(session_id="session-2"), expected_version=0)
    assert second.version == 1
    assert (
        memory.read("session-1").turns[0].request_id == memory.read("session-2").turns[0].request_id
    )


def test_ttl_read_does_not_refresh_but_append_does() -> None:
    clock = Clock()
    memory = store(clock=clock)
    first = memory.append_turn(make_turn(), expected_version=0)
    assert first.expires_at == clock.now + timedelta(seconds=10)
    clock.advance(5)
    assert memory.read("session-1").expires_at == first.expires_at
    refreshed = memory.append_turn(
        make_turn(turn_id="turn-2", request_id="request-2"), expected_version=1
    )
    assert refreshed.expires_at == clock.now + timedelta(seconds=10)


def test_expiry_resets_version_and_lazily_removes_old_turns() -> None:
    clock = Clock()
    memory = store(clock=clock)
    memory.append_turn(make_turn(), expected_version=0)
    clock.advance(10)
    assert memory.read("session-1").version == 0
    replacement = memory.append_turn(
        make_turn(turn_id="turn-2", request_id="request-2"), expected_version=0
    )
    assert replacement.version == 1 and [turn.turn_id for turn in replacement.turns] == ["turn-2"]


def test_expiry_removes_old_identifier_from_idempotency_checks() -> None:
    clock = Clock()
    memory = store(clock=clock)
    memory.append_turn(make_turn(), expected_version=0)
    clock.advance(10)
    replacement = memory.append_turn(
        make_turn(user_text="replacement", assistant_text="replacement"), expected_version=0
    )
    assert replacement.version == 1
    assert replacement.turns[0].user_text == "replacement"


def test_ttl_clock_error_does_not_leak_stored_turn_text() -> None:
    clock = Clock()
    memory = store(clock=clock)
    memory.append_turn(make_turn(), expected_version=0)
    clock.now = datetime(2026, 7, 24)
    with pytest.raises(ValueError) as raised:
        memory.read("session-1")
    assert "USER_SECRET_BODY_DO_NOT_LEAK" not in str(raised.value)
    assert "ASSISTANT_SECRET_BODY_DO_NOT_LEAK" not in str(raised.value)


def test_retention_drops_oldest_turn_and_version_advances_once_per_append() -> None:
    memory = store(max_turns=2)
    for index in range(1, 4):
        snapshot = memory.append_turn(
            make_turn(turn_id=f"turn-{index}", request_id=f"request-{index}"),
            expected_version=index - 1,
        )
        assert snapshot.version == index
    assert [turn.turn_id for turn in snapshot.turns] == ["turn-2", "turn-3"]


def test_retention_with_one_turn_keeps_only_the_latest_turn() -> None:
    memory = store(max_turns=1)
    memory.append_turn(make_turn(), expected_version=0)
    snapshot = memory.append_turn(
        make_turn(turn_id="turn-2", request_id="request-2"), expected_version=1
    )
    assert snapshot.version == 2
    assert [turn.turn_id for turn in snapshot.turns] == ["turn-2"]


def test_concurrent_stale_appends_allow_exactly_one_write_without_leaking_text() -> None:
    memory = store()
    barrier = Barrier(2)

    def append(turn: SessionTurn) -> object:
        barrier.wait()
        try:
            return memory.append_turn(turn, expected_version=0)
        except SessionVersionConflictError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                append,
                (
                    make_turn(turn_id="turn-a", request_id="request-a"),
                    make_turn(turn_id="turn-b", request_id="request-b"),
                ),
            )
        )
    snapshots = [
        result for result in results if not isinstance(result, SessionVersionConflictError)
    ]
    conflicts = [result for result in results if isinstance(result, SessionVersionConflictError)]
    assert len(snapshots) == 1 and len(conflicts) == 1
    assert memory.read("session-1").version == 1
    assert len(memory.read("session-1").turns) == 1
    assert "USER_SECRET_BODY_DO_NOT_LEAK" not in str(conflicts[0])
    assert "ASSISTANT_SECRET_BODY_DO_NOT_LEAK" not in str(conflicts[0])
