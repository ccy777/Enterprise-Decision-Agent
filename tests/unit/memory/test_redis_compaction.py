"""Offline Redis v2 serialization and atomic compact tests using a fake client."""

from __future__ import annotations

import json
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
    SessionSummary,
    SessionTurn,
    SessionVersionConflictError,
)


class FakeRedisClient:
    def __init__(self) -> None:
        self.values: dict[str, tuple[bytes, int]] = {}
        self.set_calls = 0
        self.delete_calls = 0
        self.watch_failures = 0
        self.raise_on_get: BaseException | None = None

    def get(self, key: str) -> bytes | None:
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

    def get(self, key: str) -> bytes | None:
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
                self.client.values[str(key)] = (str(value).encode(), int(ttl))
                self.client.set_calls += 1
            else:
                _, key = command
                self.client.values.pop(str(key), None)
                self.client.delete_calls += 1
        return [True] * len(self.commands)


def turn(index: int) -> SessionTurn:
    return SessionTurn(
        session_id="session-1",
        turn_id=f"turn-{index}",
        request_id=f"request-{index}",
        user_text="USER_SECRET_BODY_DO_NOT_LEAK",
        assistant_text="ASSISTANT_SECRET_BODY_DO_NOT_LEAK",
        created_at=datetime(2026, 7, 24, tzinfo=UTC),
    )


def summary(**overrides: object) -> SessionSummary:
    values: dict[str, object] = {
        "session_id": "session-1",
        "summary_id": "summary-1",
        "previous_summary_id": None,
        "source_version": 3,
        "covered_turn_count": 2,
        "covered_through_turn_id": "turn-2",
        "summary_text": "SUMMARY_SECRET_BODY_DO_NOT_LEAK",
        "created_at": datetime(2026, 7, 24, tzinfo=UTC),
    }
    values.update(overrides)
    return SessionSummary(**values)


@pytest.fixture
def client() -> FakeRedisClient:
    return FakeRedisClient()


@pytest.fixture
def store(client: FakeRedisClient) -> RedisSessionMemoryStore:
    return RedisSessionMemoryStore(
        client=client,
        policy=SessionMemoryPolicy(ttl_seconds=90, max_turns=5),
        clock=lambda: datetime(2026, 7, 24, tzinfo=UTC),
    )


def seeded(store: RedisSessionMemoryStore) -> None:
    for index in range(1, 4):
        store.append_turn(turn(index), expected_version=index - 1)


def test_schema_v2_round_trip_with_and_without_summary(
    store: RedisSessionMemoryStore, client: FakeRedisClient
) -> None:
    store.append_turn(turn(1), expected_version=0)
    key = store._key("session-1")
    assert json.loads(client.values[key][0])["schema_version"] == 2
    assert store.read("session-1").summary is None
    store.append_turn(turn(2), expected_version=1)
    store.append_turn(turn(3), expected_version=2)
    compacted = store.compact(summary(), ("turn-1", "turn-2"), expected_version=3)
    assert compacted.summary == summary()
    assert [item.turn_id for item in compacted.turns] == ["turn-3"]


def test_schema_v1_reads_without_writing_and_next_append_upgrades_to_v2(
    store: RedisSessionMemoryStore, client: FakeRedisClient
) -> None:
    key = store._key("session-1")
    client.values[key] = (
        json.dumps(
            {
                "schema_version": 1,
                "session_id": "session-1",
                "version": 1,
                "turns": [
                    {
                        "session_id": "session-1",
                        "turn_id": "turn-1",
                        "request_id": "request-1",
                        "user_text": "historical [D1] display text",
                        "assistant_text": "historical text",
                        "created_at": "2026-07-24T00:00:00+00:00",
                    }
                ],
            }
        ).encode(),
        90,
    )
    assert store.read("session-1").summary is None
    assert client.set_calls == 0
    store.append_turn(turn(2), expected_version=1)
    assert json.loads(client.values[key][0])["schema_version"] == 2


def test_compact_idempotent_is_read_only_and_does_not_refresh_ttl(
    store: RedisSessionMemoryStore, client: FakeRedisClient
) -> None:
    seeded(store)
    first = store.compact(summary(), ("turn-1", "turn-2"), expected_version=3)
    key = store._key("session-1")
    writes = client.set_calls
    ttl = client.values[key][1]
    retry = store.compact(summary(), ("turn-1", "turn-2"), expected_version=3)
    assert retry == first
    assert client.set_calls == writes
    assert client.values[key][1] == ttl


def test_compact_watch_retry_and_bounded_contention(
    store: RedisSessionMemoryStore, client: FakeRedisClient
) -> None:
    seeded(store)
    client.watch_failures = 1
    assert store.compact(summary(), ("turn-1", "turn-2"), expected_version=3).version == 4
    store.append_turn(turn(4), expected_version=4)
    exhausted = RedisSessionMemoryStore(client=client, max_transaction_retries=1)
    client.watch_failures = 1
    with pytest.raises(SessionMemoryContentionError):
        exhausted.compact(
            summary(
                summary_id="summary-2",
                previous_summary_id="summary-1",
                source_version=5,
                covered_turn_count=3,
                covered_through_turn_id="turn-3",
            ),
            ("turn-3",),
            expected_version=5,
        )


def test_compact_stale_or_unavailable_fails_closed_without_payload_leak(
    store: RedisSessionMemoryStore, client: FakeRedisClient
) -> None:
    seeded(store)
    with pytest.raises(SessionVersionConflictError):
        store.compact(summary(), ("turn-1", "turn-2"), expected_version=2)
    client.raise_on_get = RedisConnectionError("redis://secret:password@host")
    with pytest.raises(SessionMemoryUnavailableError) as raised:
        store.compact(summary(), ("turn-1", "turn-2"), expected_version=3)
    assert "secret" not in str(raised.value)
    assert raised.value.__cause__ is None


def test_append_preserves_summary_and_clear_removes_summary(store: RedisSessionMemoryStore) -> None:
    seeded(store)
    compacted = store.compact(summary(), ("turn-1", "turn-2"), expected_version=3)
    appended = store.append_turn(turn(4), expected_version=compacted.version)
    assert appended.summary == summary()
    cleared = store.clear("session-1", expected_version=5)
    assert cleared.version == 0 and cleared.summary is None


@pytest.mark.parametrize(
    "payload",
    [
        {
            "schema_version": 99,
            "session_id": "session-1",
            "version": 0,
            "summary": None,
            "turns": [],
        },
        {"schema_version": 2, "session_id": "session-1", "version": 0, "summary": {}, "turns": []},
        {
            "schema_version": 2,
            "session_id": "session-1",
            "version": 0,
            "summary": {
                "session_id": "other-session",
                "summary_id": "summary-1",
                "previous_summary_id": None,
                "source_version": 0,
                "covered_turn_count": 1,
                "covered_through_turn_id": "turn-1",
                "summary_text": "SUMMARY_SECRET_BODY_DO_NOT_LEAK",
                "created_at": "2026-07-24T00:00:00+00:00",
            },
            "turns": [],
        },
    ],
)
def test_invalid_v2_summary_or_schema_is_safe_corruption(
    store: RedisSessionMemoryStore, client: FakeRedisClient, payload: dict[str, object]
) -> None:
    client.values[store._key("session-1")] = (json.dumps(payload).encode(), 90)
    with pytest.raises(SessionMemoryCorruptionError) as raised:
        store.read("session-1")
    assert "SUMMARY_SECRET_BODY_DO_NOT_LEAK" not in str(raised.value)
    assert raised.value.__cause__ is None
