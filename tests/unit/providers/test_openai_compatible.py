"""Offline contracts for neutral OpenAI-compatible HTTP payload handling."""

from __future__ import annotations

import pytest

from decision_agent.providers import (
    ChatCompletionResponseError,
    build_chat_completion_payload,
    extract_stopped_message_content,
)


@pytest.mark.parametrize(
    ("base_url", "expect_thinking"),
    [
        ("https://api.deepseek.com/v1", True),
        ("https://example.test/v1", False),
    ],
)
def test_direct_http_payload_only_adds_deepseek_thinking(
    base_url: str, *, expect_thinking: bool
) -> None:
    payload = build_chat_completion_payload(
        base_url=base_url,
        payload={
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "extra_body": {"thinking": {"type": "enabled"}},
            "reasoning_content": "must-not-be-sent",
        },
    )

    assert payload["model"] == "test-model"
    assert payload["messages"] == [{"role": "user", "content": "test"}]
    assert "extra_body" not in payload
    assert "reasoning_content" not in payload
    if expect_thinking:
        assert payload["thinking"] == {"type": "disabled"}
    else:
        assert "thinking" not in payload


def _response(*, content: object = "{}", finish_reason: object = "stop") -> dict[str, object]:
    return {"choices": [{"finish_reason": finish_reason, "message": {"content": content}}]}


@pytest.mark.parametrize(
    ("payload", "expected_code"),
    [
        ({}, "missing_choice"),
        ({"choices": []}, "missing_choice"),
        (_response(finish_reason="length"), "truncated"),
        (_response(finish_reason="tool_calls"), "invalid_finish_reason"),
        (_response(finish_reason=None), "invalid_finish_reason"),
        (_response(content=None), "empty_content"),
        (_response(content=""), "empty_content"),
        (_response(content="   "), "empty_content"),
    ],
)
def test_structured_completion_rejects_nonterminal_or_empty_responses(
    payload: dict[str, object], expected_code: str
) -> None:
    with pytest.raises(ChatCompletionResponseError) as raised:
        extract_stopped_message_content(payload)

    assert raised.value.code == expected_code
    assert "content" not in str(raised.value).lower()


def test_structured_completion_returns_only_normal_stop_content() -> None:
    assert (
        extract_stopped_message_content(_response(content='{"answer":"safe"}'))
        == '{"answer":"safe"}'
    )
