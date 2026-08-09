"""Synchronous Redis-backed implementation of the session-memory protocol."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any

from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisError, WatchError
from redis.exceptions import TimeoutError as RedisTimeoutError

from decision_agent.memory.models import (
    DEFAULT_SESSION_MEMORY_POLICY,
    SessionMemoryPolicy,
    SessionMemorySnapshot,
    SessionSummary,
    SessionTurn,
)
from decision_agent.memory.serialization import (
    deserialize_session_snapshot,
    serialize_session_snapshot,
)
from decision_agent.memory.store import (
    SessionMemoryContentionError,
    SessionMemoryCorruptionError,
    SessionMemoryUnavailableError,
    SessionTurnConflictError,
    SessionVersionConflictError,
    validate_compaction,
)


class RedisSessionMemoryStore:
    """One-key WATCH/MULTI/EXEC store with no fallback to process-local memory."""

    def __init__(
        self,
        *,
        client: Any,
        policy: SessionMemoryPolicy = DEFAULT_SESSION_MEMORY_POLICY,
        key_prefix: str = "decision-agent:session-memory:v1",
        max_transaction_retries: int = 8,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if client is None:
            raise ValueError("client must not be None")
        if not isinstance(key_prefix, str) or not key_prefix.strip():
            raise ValueError("key_prefix must be non-empty")
        if (
            not isinstance(max_transaction_retries, int)
            or isinstance(max_transaction_retries, bool)
            or max_transaction_retries <= 0
        ):
            raise ValueError("max_transaction_retries must be a positive integer")
        self._client = client
        self._policy = policy
        self._key_prefix = key_prefix
        self._max_transaction_retries = max_transaction_retries
        self._clock = clock or (lambda: datetime.now(UTC))

    def read(self, session_id: str) -> SessionMemorySnapshot:
        """Read one key without extending its Redis TTL."""
        _validate_session_id(session_id)
        key = self._key(session_id)
        try:
            payload = self._client.get(key)
            if payload is None:
                return _empty_snapshot(session_id)
            ttl = self._client.ttl(key)
            expires_at = self._expires_at(session_id, ttl)
            return deserialize_session_snapshot(
                payload, expected_session_id=session_id, expires_at=expires_at
            )
        except (SessionMemoryCorruptionError, SessionMemoryUnavailableError):
            raise
        except (RedisConnectionError, RedisTimeoutError, RedisError, OSError, TimeoutError):
            raise SessionMemoryUnavailableError(operation="read") from None

    def append_turn(self, turn: SessionTurn, *, expected_version: int) -> SessionMemorySnapshot:
        """Atomically append one new turn or safely return an idempotent retry."""
        _validate_expected_version(expected_version)
        key = self._key(turn.session_id)
        for _ in range(self._max_transaction_retries):
            try:
                with self._client.pipeline() as pipe:
                    pipe.watch(key)
                    snapshot = self._watched_snapshot(pipe, key, turn.session_id)
                    duplicate = _matching_turn(snapshot.turns, turn)
                    if duplicate is not None:
                        if duplicate == turn:
                            return snapshot
                        raise SessionTurnConflictError(
                            session_id=turn.session_id,
                            turn_id=turn.turn_id,
                            request_id=turn.request_id,
                        )
                    if expected_version != snapshot.version:
                        raise SessionVersionConflictError(
                            session_id=turn.session_id,
                            expected_version=expected_version,
                            actual_version=snapshot.version,
                        )
                    updated = SessionMemorySnapshot(
                        session_id=turn.session_id,
                        version=snapshot.version + 1,
                        turns=(*snapshot.turns, turn)[-self._policy.max_turns :],
                        summary=snapshot.summary,
                        expires_at=self._now() + timedelta(seconds=self._policy.ttl_seconds),
                    )
                    pipe.multi()
                    pipe.set(key, serialize_session_snapshot(updated), ex=self._policy.ttl_seconds)
                    pipe.execute()
                    return updated
            except WatchError:
                continue
            except (
                SessionMemoryCorruptionError,
                SessionMemoryUnavailableError,
                SessionTurnConflictError,
                SessionVersionConflictError,
            ):
                raise
            except (
                RedisConnectionError,
                RedisTimeoutError,
                RedisError,
                OSError,
                TimeoutError,
            ):
                raise SessionMemoryUnavailableError(operation="append_turn") from None
        raise SessionMemoryContentionError(
            session_id=turn.session_id, retries=self._max_transaction_retries
        )

    def clear(self, session_id: str, *, expected_version: int) -> SessionMemorySnapshot:
        """Atomically delete one session key when its version matches."""
        _validate_session_id(session_id)
        _validate_expected_version(expected_version)
        key = self._key(session_id)
        for _ in range(self._max_transaction_retries):
            try:
                with self._client.pipeline() as pipe:
                    pipe.watch(key)
                    snapshot = self._watched_snapshot(pipe, key, session_id)
                    if expected_version != snapshot.version:
                        raise SessionVersionConflictError(
                            session_id=session_id,
                            expected_version=expected_version,
                            actual_version=snapshot.version,
                        )
                    if snapshot.version == 0:
                        return snapshot
                    pipe.multi()
                    pipe.delete(key)
                    pipe.execute()
                    return _empty_snapshot(session_id)
            except WatchError:
                continue
            except (
                SessionMemoryCorruptionError,
                SessionMemoryUnavailableError,
                SessionVersionConflictError,
            ):
                raise
            except (
                RedisConnectionError,
                RedisTimeoutError,
                RedisError,
                OSError,
                TimeoutError,
            ):
                raise SessionMemoryUnavailableError(operation="clear") from None
        raise SessionMemoryContentionError(
            session_id=session_id, retries=self._max_transaction_retries
        )

    def compact(
        self,
        summary: SessionSummary,
        compacted_turn_ids: tuple[str, ...],
        *,
        expected_version: int,
    ) -> SessionMemorySnapshot:
        """Atomically save a summary while removing a strict oldest turn prefix."""
        _validate_expected_version(expected_version)
        key = self._key(summary.session_id)
        for _ in range(self._max_transaction_retries):
            try:
                with self._client.pipeline() as pipe:
                    pipe.watch(key)
                    snapshot = self._watched_snapshot(pipe, key, summary.session_id)
                    if validate_compaction(
                        snapshot,
                        summary,
                        compacted_turn_ids,
                        expected_version=expected_version,
                    ):
                        return snapshot
                    updated = SessionMemorySnapshot(
                        session_id=summary.session_id,
                        version=snapshot.version + 1,
                        turns=snapshot.turns[len(compacted_turn_ids) :],
                        summary=summary,
                        expires_at=self._now() + timedelta(seconds=self._policy.ttl_seconds),
                    )
                    pipe.multi()
                    pipe.set(key, serialize_session_snapshot(updated), ex=self._policy.ttl_seconds)
                    pipe.execute()
                    return updated
            except WatchError:
                continue
            except (
                SessionMemoryCorruptionError,
                SessionMemoryUnavailableError,
                SessionTurnConflictError,
                SessionVersionConflictError,
            ):
                raise
            except (RedisConnectionError, RedisTimeoutError, RedisError, OSError, TimeoutError):
                raise SessionMemoryUnavailableError(operation="compact") from None
        raise SessionMemoryContentionError(
            session_id=summary.session_id, retries=self._max_transaction_retries
        )

    def _watched_snapshot(self, pipe: Any, key: str, session_id: str) -> SessionMemorySnapshot:
        payload = pipe.get(key)
        if payload is None:
            return _empty_snapshot(session_id)
        return deserialize_session_snapshot(
            payload,
            expected_session_id=session_id,
            expires_at=self._expires_at(session_id, pipe.ttl(key)),
        )

    def _key(self, session_id: str) -> str:
        _validate_session_id(session_id)
        digest = sha256(session_id.encode("utf-8")).hexdigest()
        return f"{self._key_prefix}:{digest}"

    def _expires_at(self, session_id: str, ttl: Any) -> datetime:
        if not isinstance(ttl, int) or isinstance(ttl, bool) or ttl <= 0:
            raise SessionMemoryCorruptionError(session_id=session_id)
        return self._now() + timedelta(seconds=ttl)

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return value.astimezone(UTC)


def _empty_snapshot(session_id: str) -> SessionMemorySnapshot:
    return SessionMemorySnapshot(session_id=session_id, version=0)


def _matching_turn(turns: tuple[SessionTurn, ...], candidate: SessionTurn) -> SessionTurn | None:
    for turn in turns:
        if turn.turn_id == candidate.turn_id or turn.request_id == candidate.request_id:
            return turn
    return None


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
