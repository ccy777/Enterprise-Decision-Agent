"""Offline service tests for deterministic rolling-summary compaction."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from decision_agent.memory import (
    InMemorySessionMemoryStore,
    RollingSummaryDraft,
    RollingSummaryGenerationError,
    RollingSummaryInputTooLarge,
    RollingSummaryOutputInvalid,
    RollingSummaryPolicy,
    RollingSummaryService,
    RollingSummaryStatus,
    SessionMemoryPolicy,
    SessionTurn,
    SessionVersionConflictError,
)

NOW = datetime(2026, 7, 24, 10, 0, tzinfo=UTC)


def turn(index: int, *, user: str | None = None, assistant: str | None = None) -> SessionTurn:
    return SessionTurn(
        session_id="session-1",
        turn_id=f"turn-{index}",
        request_id=f"request-{index}",
        user_text=user or f"user-{index}",
        assistant_text=assistant or f"assistant-{index}",
        created_at=NOW,
    )


def memory_with_turns(count: int) -> InMemorySessionMemoryStore:
    memory = InMemorySessionMemoryStore(policy=SessionMemoryPolicy(max_turns=20))
    for index in range(1, count + 1):
        memory.append_turn(turn(index), expected_version=index - 1)
    return memory


class CapturingSummarizer:
    def __init__(self, result: object = RollingSummaryDraft(summary_text="safe summary")) -> None:
        self.result = result
        self.requests: list[object] = []

    def summarize(self, request: object) -> object:
        self.requests.append(request)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def service(
    memory: InMemorySessionMemoryStore,
    summarizer: CapturingSummarizer,
    *,
    policy: RollingSummaryPolicy | None = None,
) -> RollingSummaryService:
    return RollingSummaryService(
        store=memory,
        summarizer=summarizer,
        policy=policy
        or RollingSummaryPolicy(
            trigger_turns=3, retain_recent_turns=1, max_source_chars=1_000, max_summary_chars=100
        ),
        clock=lambda: NOW,
    )


def test_empty_and_below_trigger_are_noops_without_summarizer_or_compact() -> None:
    empty = InMemorySessionMemoryStore()
    summarizer = CapturingSummarizer()
    empty_result = service(empty, summarizer).compact_if_needed("session-1")
    assert empty_result.status is RollingSummaryStatus.NOT_REQUIRED
    assert empty_result.snapshot.version == 0

    below = memory_with_turns(2)
    below_result = service(below, summarizer).compact_if_needed("session-1")
    assert below_result.status is RollingSummaryStatus.NOT_REQUIRED
    assert below_result.snapshot.version == 2
    assert summarizer.requests == []


def test_snapshot_entry_uses_supplied_snapshot_without_a_second_read() -> None:
    memory = memory_with_turns(3)
    summarizer = CapturingSummarizer()
    snapshot = memory.read("session-1")
    result = service(memory, summarizer).compact_snapshot_if_needed("session-1", snapshot)
    assert result.status is RollingSummaryStatus.COMPACTED
    assert len(summarizer.requests) == 1
    assert result.snapshot.version == snapshot.version + 1


def test_snapshot_entry_rejects_a_different_session_without_store_access() -> None:
    memory = memory_with_turns(3)
    snapshot = memory.read("session-1")
    with pytest.raises(ValueError, match="snapshot session_id must match session_id"):
        service(memory, CapturingSummarizer()).compact_snapshot_if_needed("other-session", snapshot)


def test_exact_trigger_compacts_oldest_prefix_and_retains_recent_turn() -> None:
    memory = memory_with_turns(3)
    summarizer = CapturingSummarizer()
    result = service(memory, summarizer).compact_if_needed("session-1")
    request = summarizer.requests[0]
    assert result.status is RollingSummaryStatus.COMPACTED
    assert result.compacted_turn_count == 2
    assert [turn.turn_id for turn in request.turns] == ["turn-1", "turn-2"]  # type: ignore[attr-defined]
    assert [item.turn_id for item in result.snapshot.turns] == ["turn-3"]
    assert result.snapshot.summary is not None
    assert result.snapshot.summary.source_version == 3
    assert result.snapshot.summary.covered_turn_count == 2
    assert result.snapshot.summary.covered_through_turn_id == "turn-2"
    assert result.snapshot.summary.previous_summary_id is None
    assert result.snapshot.version == 4


def test_longest_budgeted_oldest_prefix_never_skips_middle_turns() -> None:
    memory = InMemorySessionMemoryStore(policy=SessionMemoryPolicy(max_turns=20))
    for index, body in enumerate(("a", "b" * 200, "c", "d"), start=1):
        memory.append_turn(turn(index, user=body, assistant=body), expected_version=index - 1)
    summarizer = CapturingSummarizer()
    result = service(
        memory,
        summarizer,
        policy=RollingSummaryPolicy(
            trigger_turns=4, retain_recent_turns=1, max_source_chars=400, max_summary_chars=100
        ),
    ).compact_if_needed("session-1")
    request = summarizer.requests[0]
    assert [item.turn_id for item in request.turns] == ["turn-1"]  # type: ignore[attr-defined]
    assert result.compacted_turn_count == 1
    assert [item.turn_id for item in result.snapshot.turns] == ["turn-2", "turn-3", "turn-4"]


def test_previous_summary_is_in_next_request_and_counts_against_budget() -> None:
    memory = memory_with_turns(3)
    first_summarizer = CapturingSummarizer(RollingSummaryDraft(summary_text="old summary"))
    first = service(memory, first_summarizer).compact_if_needed("session-1")
    memory.append_turn(turn(4), expected_version=first.snapshot.version)
    memory.append_turn(turn(5), expected_version=first.snapshot.version + 1)
    second_summarizer = CapturingSummarizer(RollingSummaryDraft(summary_text="new summary"))
    second = service(memory, second_summarizer).compact_if_needed("session-1")
    request = second_summarizer.requests[0]
    assert request.previous_summary_text == "old summary"  # type: ignore[attr-defined]
    assert second.snapshot.summary is not None
    assert second.snapshot.summary.previous_summary_id == first.summary_id
    assert second.snapshot.summary.covered_turn_count == 4
    assert [item.turn_id for item in second.snapshot.turns] == ["turn-5"]


def test_first_candidate_over_budget_fails_closed_without_provider_or_store_mutation() -> None:
    memory = memory_with_turns(3)
    summarizer = CapturingSummarizer()
    original = memory.read("session-1")
    with pytest.raises(RollingSummaryInputTooLarge) as raised:
        service(
            memory,
            summarizer,
            policy=RollingSummaryPolicy(
                trigger_turns=3, retain_recent_turns=1, max_source_chars=1, max_summary_chars=100
            ),
        ).compact_if_needed("session-1")
    assert "user-1" not in str(raised.value)
    assert summarizer.requests == []
    assert memory.read("session-1") == original


@pytest.mark.parametrize(
    "draft",
    [
        {"summary_text": ""},
        {"summary_text": "summary [D1]"},
        {"summary_text": "x" * 101},
        {"summary_text": "safe", "unexpected": True},
    ],
)
def test_invalid_output_never_compacts_and_never_leaks_body(draft: object) -> None:
    memory = memory_with_turns(3)
    summarizer = CapturingSummarizer(draft)
    original = memory.read("session-1")
    with pytest.raises(RollingSummaryOutputInvalid) as raised:
        service(memory, summarizer).compact_if_needed("session-1")
    assert "summary [D1]" not in str(raised.value)
    assert memory.read("session-1") == original
    assert len(summarizer.requests) == 1


def test_summarizer_failure_does_not_compact_or_retry() -> None:
    memory = memory_with_turns(3)
    summarizer = CapturingSummarizer(RuntimeError("SECRET_HISTORY_BODY"))
    original = memory.read("session-1")
    with pytest.raises(RollingSummaryGenerationError) as raised:
        service(memory, summarizer).compact_if_needed("session-1")
    assert "SECRET_HISTORY_BODY" not in str(raised.value)
    assert len(summarizer.requests) == 1
    assert memory.read("session-1") == original


class ConflictStore:
    def __init__(self, delegate: InMemorySessionMemoryStore) -> None:
        self.delegate = delegate
        self.compact_calls = 0

    def read(self, session_id: str):  # type: ignore[no-untyped-def]
        return self.delegate.read(session_id)

    def compact(self, summary, compacted_turn_ids, *, expected_version):  # type: ignore[no-untyped-def]
        self.compact_calls += 1
        raise SessionVersionConflictError(
            session_id=summary.session_id,
            expected_version=expected_version,
            actual_version=expected_version + 1,
        )


def test_compact_conflict_is_propagated_without_second_summarizer_call() -> None:
    delegate = memory_with_turns(3)
    store = ConflictStore(delegate)
    summarizer = CapturingSummarizer()
    summary_service = RollingSummaryService(
        store=store,
        summarizer=summarizer,
        policy=RollingSummaryPolicy(
            trigger_turns=3, retain_recent_turns=1, max_source_chars=1_000, max_summary_chars=100
        ),
        clock=lambda: NOW,
    )
    with pytest.raises(SessionVersionConflictError):
        summary_service.compact_if_needed("session-1")
    assert store.compact_calls == 1
    assert len(summarizer.requests) == 1
    assert delegate.read("session-1").version == 3


def test_summary_id_is_deterministic_and_changes_with_version_or_prefix() -> None:
    first_memory = memory_with_turns(3)
    first = service(first_memory, CapturingSummarizer()).compact_if_needed("session-1")
    second_memory = memory_with_turns(3)
    second = service(second_memory, CapturingSummarizer()).compact_if_needed("session-1")
    assert first.summary_id == second.summary_id
    assert first.summary_id is not None
    assert "session-1" not in first.summary_id and "user-1" not in first.summary_id

    versioned = memory_with_turns(3)
    versioned.append_turn(turn(4), expected_version=3)
    changed = service(versioned, CapturingSummarizer()).compact_if_needed("session-1")
    assert changed.summary_id != first.summary_id
