"""Offline contract tests for the injected-client Redis session-memory store."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import WatchError

from decision_agent.memory import (
    RedisSessionMemoryStore,
    SessionMemoryContentionError,
    SessionMemoryCorruptionError,
    SessionMemoryPolicy,
    SessionMemoryUnavailableError,
    SessionTurn,
    SessionTurnConflictError,
    SessionVersionConflictError,
)


class FakeRedisClient:
    """Tiny fake with immediate WATCH reads and queued transaction writes."""

    def __init__(self) -> None:
        self.values: dict[str, tuple[bytes | str, int]] = {}
        self.set_calls = 0
        self.delete_calls = 0
        self.watch_failures = 0
        self.raise_on_get: BaseException | None = None

    def get(self, key: str) -> bytes | str | None:
        if self.raise_on_get is not None:
            raise self.raise_on_get
        item = self.values.get(key)
        return None if item is None else item[0]

    def ttl(self, key: str) -> int:
        item = self.values.get(key)
        return -2 if item is None else item[1]

    def pipeline(self) -> FakePipeline:
        return FakePipeline(self)


class FakePipeline:
    def __init__(self, client: FakeRedisClient) -> None:
        self.client = client
        self.commands: list[tuple[object, ...]] = []

    def __enter__(self) -> FakePipeline:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def watch(self, key: str) -> None:
        return None

    def get(self, key: str) -> bytes | str | None:
        return self.client.get(key)

    def ttl(self, key: str) -> int:
        return self.client.ttl(key)

    def multi(self) -> None:
        return None

    def set(self, key: str, value: str, *, ex: int) -> None:
        self.commands.append(("set", key, value, ex))

    def delete(self, key: str) -> None:
        self.commands.append(("delete", key))

    def execute(self) -> list[object]:
        if self.client.watch_failures:
            self.client.watch_failures -= 1
            raise WatchError("changed")
        for command in self.commands:
            if command[0] == "set":
                _, key, value, ttl = command
                self.client.values[str(key)] = (str(value).encode("utf-8"), int(ttl))
                self.client.set_calls += 1
            else:
                _, key = command
                self.client.values.pop(str(key), None)
                self.client.delete_calls += 1
        return [True] * len(self.commands)


@pytest.fixture
def client() -> FakeRedisClient:
    return FakeRedisClient()


@pytest.fixture
def store(client: FakeRedisClient) -> RedisSessionMemoryStore:
    return RedisSessionMemoryStore(
        client=client,
        policy=SessionMemoryPolicy(ttl_seconds=90, max_turns=2),
        clock=lambda: datetime(2026, 7, 24, tzinfo=UTC),
    )


def _turn(
    *,
    session_id: str = "session-A",
    turn_id: str = "turn-1",
    request_id: str = "request-1",
    user_text: str = "question",
    assistant_text: str = "answer",
) -> SessionTurn:
    return SessionTurn(
        session_id=session_id,
        turn_id=turn_id,
        request_id=request_id,
        user_text=user_text,
        assistant_text=assistant_text,
        created_at=datetime(2026, 7, 24, tzinfo=UTC),
    )


def test_read_missing_returns_empty_without_writing(
    store: RedisSessionMemoryStore, client: FakeRedisClient
) -> None:
    assert store.read("session-A").model_dump() == {
        "session_id": "session-A",
        "version": 0,
        "turns": (),
        "summary": None,
        "expires_at": None,
    }
    assert client.set_calls == client.delete_calls == 0


def test_key_is_versioned_stable_and_hides_raw_session_id(store: RedisSessionMemoryStore) -> None:
    first = store._key("session-A")
    assert first == store._key("session-A")
    assert first != store._key("session-B")
    assert first.startswith("decision-agent:session-memory:v1:")
    assert "session-A" not in first


def test_append_round_trips_version_turn_and_ttl(
    store: RedisSessionMemoryStore, client: FakeRedisClient
) -> None:
    written = store.append_turn(_turn(), expected_version=0)
    reread = store.read("session-A")
    assert written.version == reread.version == 1
    assert reread.turns == (_turn(),)
    assert reread.expires_at == datetime(2026, 7, 24, 0, 1, 30, tzinfo=UTC)
    assert client.values[store._key("session-A")][1] == 90


def test_identical_retry_is_noop_even_with_stale_expected_version(
    store: RedisSessionMemoryStore, client: FakeRedisClient
) -> None:
    stored = store.append_turn(_turn(), expected_version=0)
    retry = store.append_turn(_turn(), expected_version=0)
    assert retry == stored
    assert client.set_calls == 1
    assert client.values[store._key("session-A")][1] == 90


@pytest.mark.parametrize("field", ["turn_id", "request_id"])
def test_reused_id_with_different_content_conflicts_before_version_check(
    store: RedisSessionMemoryStore, field: str
) -> None:
    store.append_turn(_turn(), expected_version=0)
    values = {"turn_id": "turn-2", "request_id": "request-2", "assistant_text": "changed"}
    values[field] = "turn-1" if field == "turn_id" else "request-1"
    with pytest.raises(SessionTurnConflictError):
        store.append_turn(_turn(**values), expected_version=0)


def test_new_turn_requires_matching_version(store: RedisSessionMemoryStore) -> None:
    store.append_turn(_turn(), expected_version=0)
    with pytest.raises(SessionVersionConflictError):
        store.append_turn(_turn(turn_id="turn-2", request_id="request-2"), expected_version=0)


def test_retention_keeps_only_latest_policy_max_turns(store: RedisSessionMemoryStore) -> None:
    store.append_turn(_turn(), expected_version=0)
    store.append_turn(_turn(turn_id="turn-2", request_id="request-2"), expected_version=1)
    result = store.append_turn(_turn(turn_id="turn-3", request_id="request-3"), expected_version=2)
    assert result.version == 3
    assert [turn.turn_id for turn in result.turns] == ["turn-2", "turn-3"]


def test_clear_deletes_key_and_version_returns_to_empty(
    store: RedisSessionMemoryStore, client: FakeRedisClient
) -> None:
    store.append_turn(_turn(), expected_version=0)
    assert store.clear("session-A", expected_version=1).version == 0
    assert client.delete_calls == 1
    assert store.read("session-A").turns == ()


def test_clear_empty_is_noop_and_wrong_version_conflicts(
    store: RedisSessionMemoryStore, client: FakeRedisClient
) -> None:
    assert store.clear("session-A", expected_version=0).version == 0
    assert client.delete_calls == 0
    with pytest.raises(SessionVersionConflictError):
        store.clear("session-A", expected_version=1)


def test_watch_conflict_retries_then_succeeds(
    store: RedisSessionMemoryStore, client: FakeRedisClient
) -> None:
    client.watch_failures = 1
    assert store.append_turn(_turn(), expected_version=0).version == 1
    assert client.set_calls == 1


def test_watch_conflict_has_bounded_safe_failure(client: FakeRedisClient) -> None:
    client.watch_failures = 2
    store = RedisSessionMemoryStore(client=client, max_transaction_retries=2)
    with pytest.raises(SessionMemoryContentionError) as exc_info:
        store.append_turn(_turn(), expected_version=0)
    assert "session-A" in str(exc_info.value)


@pytest.mark.parametrize(
    ("payload", "session_id"),
    [
        (b"not json", "session-A"),
        (b'{"schema_version":2}', "session-A"),
        (b'{"schema_version":1,"session_id":"other","version":0,"turns":[]}', "session-A"),
        (b'{"schema_version":1,"session_id":"session-A","version":false,"turns":[]}', "session-A"),
        (b'{"schema_version":1,"session_id":"session-A","version":0,"turns":[]}', "session-A"),
    ],
)
def test_invalid_payload_or_ttl_maps_to_safe_corruption(
    store: RedisSessionMemoryStore, client: FakeRedisClient, payload: bytes, session_id: str
) -> None:
    client.values[store._key(session_id)] = (payload, 0 if payload.endswith(b"[]}") else 90)
    with pytest.raises(SessionMemoryCorruptionError) as exc_info:
        store.read(session_id)
    assert "not json" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


def test_unavailable_backend_maps_to_safe_error(
    store: RedisSessionMemoryStore, client: FakeRedisClient
) -> None:
    client.raise_on_get = RedisConnectionError("redis://secret:password@host")
    with pytest.raises(SessionMemoryUnavailableError) as exc_info:
        store.read("session-A")
    assert "secret" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


def test_constructor_rejects_missing_client_and_invalid_retries(client: FakeRedisClient) -> None:
    with pytest.raises(ValueError):
        RedisSessionMemoryStore(client=None)
    with pytest.raises(ValueError):
        RedisSessionMemoryStore(client=client, max_transaction_retries=0)
