"""Tests for deny-by-default scalar trace attributes."""

from __future__ import annotations

import pytest

from decision_agent.observability import AttributeLimits, TraceStage, sanitize_attributes


def test_attribute_allowlist_preserves_none_and_zero_without_recording_forbidden_values() -> None:
    result = sanitize_attributes(
        stage=TraceStage.ROUTING,
        values={
            "route": "knowledge",
            "retry_count": 0,
            "input_tokens": None,
            "QUERY": "do not retain",
            "Api_Key": "do not retain",
        },
        limits=AttributeLimits(),
    )

    assert [(item.key, item.value) for item in result.attributes] == [
        ("route", "knowledge"),
        ("retry_count", 0),
        ("input_tokens", None),
    ]
    assert result.dropped_attribute_count == 2
    assert "do not retain" not in str(result.attributes)


def test_attribute_policy_rejects_complex_and_unknown_values_and_truncates_safe_strings() -> None:
    result = sanitize_attributes(
        stage=TraceStage.TOOL_EXECUTION,
        values={
            "tool_name": "x" * 6,
            "rows": [["secret"]],
            "unknown": "not allowed",
            "denied": True,
            "timeout": float("inf"),
        },
        limits=AttributeLimits(max_attributes_per_span=2, max_attribute_value_length=4),
    )

    assert [(item.key, item.value) for item in result.attributes] == [
        ("tool_name", "xxxx"),
        ("denied", True),
    ]
    assert result.dropped_attribute_count == 4


def test_attribute_count_limit_is_stable_and_does_not_admit_later_values() -> None:
    result = sanitize_attributes(
        stage=TraceStage.RETRIEVAL,
        values={"requested_top_k": 10, "retrieved_count": 0, "parent_count": 1},
        limits=AttributeLimits(max_attributes_per_span=2, max_attribute_value_length=16),
    )

    assert [item.key for item in result.attributes] == ["requested_top_k", "retrieved_count"]
    assert result.dropped_attribute_count == 1


def test_planning_attributes_keep_only_bounded_plan_metadata() -> None:
    result = sanitize_attributes(
        stage=TraceStage.PLANNING,
        values={
            "plan_version": "m8b-v1",
            "plan_step_count": 1,
            "success": True,
            "plan_json": "must not appear",
        },
        limits=AttributeLimits(),
    )

    assert [(item.key, item.value) for item in result.attributes] == [
        ("plan_version", "m8b-v1"),
        ("plan_step_count", 1),
        ("success", True),
    ]
    assert result.dropped_attribute_count == 1


@pytest.mark.parametrize("key", ["sql", "SYSTEM_PROMPT", "session_id", "connection_string"])
def test_forbidden_keys_are_rejected_case_insensitively(key: str) -> None:
    result = sanitize_attributes(
        stage=TraceStage.TOOL_EXECUTION,
        values={key: "must not appear"},
        limits=AttributeLimits(),
    )

    assert result.attributes == ()
    assert result.dropped_attribute_count == 1
