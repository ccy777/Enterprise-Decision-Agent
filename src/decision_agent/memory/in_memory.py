"""Thread-safe, process-local implementation for tests and local development only."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from threading import RLock

from decision_agent.memory.models import (
    DEFAULT_SESSION_MEMORY_POLICY,
    SessionMemoryPolicy,
    SessionMemorySnapshot,
    SessionSummary,
    SessionTurn,
)
from decision_agent.memory.store import (
    SessionTurnConflictError,
    SessionVersionConflictError,
    validate_compaction,
)


@dataclass(frozen=True)
class _StoredSession:
    version: int
    turns: tuple[SessionTurn, ...]
    summary: SessionSummary | None
    expires_at: datetime


class InMemorySessionMemoryStore:
    """Per-instance locked memory store; it has no persistence or cross-process sharing."""

    def __init__(
        self,
        *,
        policy: SessionMemoryPolicy = DEFAULT_SESSION_MEMORY_POLICY,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._policy = policy
        self._clock = clock or (lambda: datetime.now(UTC))
        self._lock = RLock()
        self._sessions: dict[str, _StoredSession] = {}

    def read(self, session_id: str) -> SessionMemorySnapshot:
        """Read a live session without refreshing TTL."""
        _validate_session_id(session_id)
        with self._lock:
            stored = self._live_session(session_id)
            return _snapshot(session_id, stored)

    def append_turn(self, turn: SessionTurn, *, expected_version: int) -> SessionMemorySnapshot:
        """Append atomically, retaining only the newest configured number of turns."""
        _validate_expected_version(expected_version)
        with self._lock:
            stored = self._live_session(turn.session_id)
            if stored is not None:
                duplicate = _matching_turn(stored.turns, turn)
                if duplicate is not None:
                    if duplicate == turn:
                        return _snapshot(turn.session_id, stored)
                    raise SessionTurnConflictError(
                        session_id=turn.session_id, turn_id=turn.turn_id, request_id=turn.request_id
                    )
            actual_version = 0 if stored is None else stored.version
            if expected_version != actual_version:
                raise SessionVersionConflictError(
                    session_id=turn.session_id,
                    expected_version=expected_version,
                    actual_version=actual_version,
                )
            turns = () if stored is None else stored.turns
            retained_turns = (*turns, turn)[-self._policy.max_turns :]
            updated = _StoredSession(
                version=actual_version + 1,
                turns=retained_turns,
                summary=None if stored is None else stored.summary,
                expires_at=self._now() + timedelta(seconds=self._policy.ttl_seconds),
            )
            self._sessions[turn.session_id] = updated
            return _snapshot(turn.session_id, updated)

    def clear(self, session_id: str, *, expected_version: int) -> SessionMemorySnapshot:
        """Clear one session atomically; an absent session has deterministic version zero."""
        _validate_session_id(session_id)
        _validate_expected_version(expected_version)
        with self._lock:
            stored = self._live_session(session_id)
            actual_version = 0 if stored is None else stored.version
            if expected_version != actual_version:
                raise SessionVersionConflictError(
                    session_id=session_id,
                    expected_version=expected_version,
                    actual_version=actual_version,
                )
            self._sessions.pop(session_id, None)
            return _snapshot(session_id, None)

    def compact(
        self,
        summary: SessionSummary,
        compacted_turn_ids: tuple[str, ...],
        *,
        expected_version: int,
    ) -> SessionMemorySnapshot:
        """Atomically replace an oldest turn prefix with validated summary state."""
        _validate_expected_version(expected_version)
        with self._lock:
            stored = self._live_session(summary.session_id)
            current = _snapshot(summary.session_id, stored)
            if validate_compaction(
                current, summary, compacted_turn_ids, expected_version=expected_version
            ):
                return current
            updated = _StoredSession(
                version=current.version + 1,
                turns=current.turns[len(compacted_turn_ids) :],
                summary=summary,
                expires_at=self._now() + timedelta(seconds=self._policy.ttl_seconds),
            )
            self._sessions[summary.session_id] = updated
            return _snapshot(summary.session_id, updated)

    def _live_session(self, session_id: str) -> _StoredSession | None:
        stored = self._sessions.get(session_id)
        if stored is not None and self._now() >= stored.expires_at:
            del self._sessions[session_id]
            return None
        return stored

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return value.astimezone(UTC)


def _matching_turn(turns: tuple[SessionTurn, ...], candidate: SessionTurn) -> SessionTurn | None:
    for turn in turns:
        if turn.turn_id == candidate.turn_id or turn.request_id == candidate.request_id:
            return turn
    return None


def _snapshot(session_id: str, stored: _StoredSession | None) -> SessionMemorySnapshot:
    if stored is None:
        return SessionMemorySnapshot(session_id=session_id, version=0)
    return SessionMemorySnapshot(
        session_id=session_id,
        version=stored.version,
        turns=stored.turns,
        summary=stored.summary,
        expires_at=stored.expires_at,
    )


def _validate_session_id(session_id: str) -> None:
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be non-empty")


def _validate_expected_version(expected_version: int) -> None:
    if (
        not isinstance(expected_version, int)
        or isinstance(expected_version, bool)
        or expected_version < 0
    ):
        raise ValueError("expected_version must be a non-negative integer")
