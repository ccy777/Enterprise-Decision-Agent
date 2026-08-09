"""Opt-in single-call smoke test for rolling-summary generation."""

from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest

from decision_agent.memory import (
    InMemorySessionMemoryStore,
    ProviderRollingSummarizer,
    RollingSummaryPolicy,
    RollingSummaryService,
    RollingSummaryStatus,
    SessionMemoryPolicy,
    SessionTurn,
)

pytestmark = pytest.mark.integration


def test_real_provider_rolling_summary_is_explicit_opt_in() -> None:
    if os.getenv("DECISION_AGENT_RUN_REAL_SUMMARY_TEST") != "1":
        pytest.skip("set DECISION_AGENT_RUN_REAL_SUMMARY_TEST=1 to run the real summary smoke")

    # Imports and configuration construction remain after the opt-in guard so the default skip path
    # creates no provider and performs no network activity.
    from decision_agent.config import Settings
    from decision_agent.tool_calling.runtime import OpenAICompatibleNativeToolCallingModel

    now = datetime(2026, 7, 24, tzinfo=UTC)
    store = InMemorySessionMemoryStore(policy=SessionMemoryPolicy(max_turns=10))
    for index, user_text, assistant_text in (
        (1, "Decision: keep the weekly review.", "Acknowledged."),
        (2, "Constraint: no production writes.", "I will keep all queries read-only."),
        (3, "Preference: concise status updates.", "I will use concise updates."),
    ):
        store.append_turn(
            SessionTurn(
                session_id="real-summary-smoke",
                turn_id=f"turn-{index}",
                request_id=f"request-{index}",
                user_text=user_text,
                assistant_text=assistant_text,
                created_at=now,
            ),
            expected_version=index - 1,
        )
    service = RollingSummaryService(
        store=store,
        summarizer=ProviderRollingSummarizer(
            provider=OpenAICompatibleNativeToolCallingModel.from_settings(Settings())
        ),
        policy=RollingSummaryPolicy(
            trigger_turns=3, retain_recent_turns=1, max_source_chars=2_000, max_summary_chars=300
        ),
        clock=lambda: now,
    )

    outcome = service.compact_if_needed("real-summary-smoke")

    assert outcome.status is RollingSummaryStatus.COMPACTED
    assert outcome.snapshot.version == 4
    assert outcome.snapshot.summary is not None
    assert outcome.snapshot.summary.summary_text.strip()
    assert "[D" not in outcome.snapshot.summary.summary_text
    assert "[E" not in outcome.snapshot.summary.summary_text
    assert outcome.snapshot.summary.covered_turn_count == 2
    assert [turn.turn_id for turn in outcome.snapshot.turns] == ["turn-3"]
