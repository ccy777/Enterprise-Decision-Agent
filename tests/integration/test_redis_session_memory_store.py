"""Optional Redis contract test; it never discovers a default Redis URL."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from threading import Barrier
from uuid import uuid4

import pytest
from redis import Redis

from decision_agent.memory import (
    RedisSessionMemoryStore,
    SessionMemoryPolicy,
    SessionSummary,
    SessionTurn,
    SessionVersionConflictError,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def redis_url() -> str:
    url = os.environ.get("DECISION_AGENT_TEST_REDIS_URL")
    if not url:
        pytest.skip("DECISION_AGENT_TEST_REDIS_URL is not configured")
    return url


def _turn(session_id: str, turn_id: str) -> SessionTurn:
    return SessionTurn(
        session_id=session_id,
        turn_id=turn_id,
        request_id=f"request-{turn_id}",
        user_text="integration question",
        assistant_text="integration answer",
        created_at=datetime.now(UTC),
    )


def _summary(
    session_id: str,
    *,
    summary_id: str,
    previous_summary_id: str | None,
    source_version: int,
    covered_turn_count: int,
    covered_through_turn_id: str,
) -> SessionSummary:
    return SessionSummary(
        session_id=session_id,
        summary_id=summary_id,
        previous_summary_id=previous_summary_id,
        source_version=source_version,
        covered_turn_count=covered_turn_count,
        covered_through_turn_id=covered_through_turn_id,
        summary_text="integration rolling summary",
        created_at=datetime.now(UTC),
    )


def test_redis_store_contract(redis_url: str) -> None:
    """Exercise v1 compatibility and v2 atomic compaction in an isolated namespace."""
    client = Redis.from_url(redis_url, decode_responses=False)
    prefix = f"decision-agent:test-session-memory:v1:{uuid4().hex}"
    policy = SessionMemoryPolicy(ttl_seconds=30, max_turns=5)
    store = RedisSessionMemoryStore(client=client, key_prefix=prefix, policy=policy)
    session_id = f"session-{uuid4().hex}"
    v1_session_id = f"session-v1-{uuid4().hex}"
    key = store._key(session_id)
    v1_key = store._key(v1_session_id)
    try:
        first = _turn(session_id, "1")
        assert store.read(session_id).version == 0
        assert store.append_turn(first, expected_version=0).version == 1
        assert store.append_turn(first, expected_version=0).version == 1
        assert json.loads(client.get(key))["schema_version"] == 2
        assert store.append_turn(_turn(session_id, "2"), expected_version=1).version == 2
        assert store.append_turn(_turn(session_id, "3"), expected_version=2).version == 3
        first_summary = _summary(
            session_id,
            summary_id="summary-1",
            previous_summary_id=None,
            source_version=3,
            covered_turn_count=2,
            covered_through_turn_id="2",
        )
        assert store.compact(first_summary, ("1", "2"), expected_version=3).version == 4
        assert client.ttl(key) > 0
        assert store.append_turn(_turn(session_id, "4"), expected_version=4).version == 5
        second_summary = _summary(
            session_id,
            summary_id="summary-2",
            previous_summary_id="summary-1",
            source_version=5,
            covered_turn_count=3,
            covered_through_turn_id="3",
        )
        assert store.compact(second_summary, ("3",), expected_version=5).version == 6
        ttl = client.ttl(key)
        assert store.compact(second_summary, ("3",), expected_version=5).version == 6
        assert client.ttl(key) == ttl
        with pytest.raises(SessionVersionConflictError):
            store.append_turn(_turn(session_id, "5"), expected_version=5)
        assert store.clear(session_id, expected_version=6).model_dump() == {
            "session_id": session_id,
            "version": 0,
            "turns": (),
            "summary": None,
            "expires_at": None,
        }

        client.set(
            v1_key,
            json.dumps(
                {
                    "schema_version": 1,
                    "session_id": v1_session_id,
                    "version": 1,
                    "turns": [
                        {
                            "session_id": v1_session_id,
                            "turn_id": "v1-turn",
                            "request_id": "v1-request",
                            "user_text": "historical display text",
                            "assistant_text": "historical display text",
                            "created_at": datetime.now(UTC).isoformat(),
                        }
                    ],
                }
            ),
            ex=30,
        )
        assert store.read(v1_session_id).summary is None
        assert store.append_turn(_turn(v1_session_id, "v2"), expected_version=1).version == 2
        assert json.loads(client.get(v1_key))["schema_version"] == 2
    finally:
        client.delete(key, v1_key)
        client.close()


def test_redis_append_and_compact_race(redis_url: str) -> None:
    """Two independently injected stores must not both commit version-two writes."""
    first_client = Redis.from_url(redis_url, decode_responses=False)
    second_client = Redis.from_url(redis_url, decode_responses=False)
    prefix = f"decision-agent:test-session-memory:v1:{uuid4().hex}"
    policy = SessionMemoryPolicy(ttl_seconds=30, max_turns=5)
    first = RedisSessionMemoryStore(client=first_client, key_prefix=prefix, policy=policy)
    second = RedisSessionMemoryStore(client=second_client, key_prefix=prefix, policy=policy)
    session_id = f"session-{uuid4().hex}"
    key = first._key(session_id)
    barrier = Barrier(2)
    candidate = _summary(
        session_id,
        summary_id="summary-race",
        previous_summary_id=None,
        source_version=2,
        covered_turn_count=1,
        covered_through_turn_id="1",
    )
    try:
        first.append_turn(_turn(session_id, "1"), expected_version=0)
        first.append_turn(_turn(session_id, "2"), expected_version=1)

        def append() -> object:
            barrier.wait()
            try:
                return first.append_turn(_turn(session_id, "3"), expected_version=2)
            except SessionVersionConflictError as error:
                return error

        def compact() -> object:
            barrier.wait()
            try:
                return second.compact(candidate, ("1",), expected_version=2)
            except SessionVersionConflictError as error:
                return error

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = [executor.submit(append), executor.submit(compact)]
            resolved = [result.result() for result in results]
        assert sum(not isinstance(item, SessionVersionConflictError) for item in resolved) == 1
        assert sum(isinstance(item, SessionVersionConflictError) for item in resolved) == 1
    finally:
        first_client.delete(key)
        first_client.close()
        second_client.close()
