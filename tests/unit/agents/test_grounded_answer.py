"""Unit tests for the OpenAI-compatible structured-answer boundary."""

from __future__ import annotations

import json

import pytest

from decision_agent.agents.grounded_answer import (
    _SYSTEM_PROMPT,
    AnswerGenerationError,
    OpenAICompatibleAnswerGenerator,
)


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_prompt_includes_answer_only_json_example_and_inline_citation_rules() -> None:
    assert "Correct JSON:" in _SYSTEM_PROMPT
    assert "exactly these fields: answer, citations" in _SYSTEM_PROMPT
    assert "Do not make or alter an" in _SYSTEM_PROMPT
    assert "answer must contain every citation inline" in _SYSTEM_PROMPT
    assert "[E<number>]" in _SYSTEM_PROMPT
    assert "no Markdown, no code fence" in _SYSTEM_PROMPT


def test_prompt_requires_the_user_query_primary_language() -> None:
    assert "same primary language as the user question" in _SYSTEM_PROMPT
    assert "Chinese question must receive\nChinese output" in _SYSTEM_PROMPT


@pytest.mark.parametrize(
    ("base_url", "expect_thinking"),
    [("https://api.deepseek.com/v1", True), ("https://example.invalid/v1", False)],
)
def test_request_uses_the_configured_provider_payload_contract(
    monkeypatch: pytest.MonkeyPatch, base_url: str, *, expect_thinking: bool
) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request: object, *, timeout: float) -> _Response:
        captured["body"] = json.loads(request.data.decode("utf-8"))  # type: ignore[attr-defined]
        captured["timeout"] = timeout
        return _Response(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer": "Supported fact.[E1]",
                                    "citations": ["[E1]"],
                                }
                            )
                        },
                    }
                ]
            }
        )

    monkeypatch.setattr("decision_agent.agents.grounded_answer.urlopen", fake_urlopen)
    generator = OpenAICompatibleAnswerGenerator(
        api_key="test-key",
        base_url=base_url,
        model_name="test-model",
        timeout_seconds=12,
    )

    payload = generator._post(
        "question", "[E1] evidence", "answerable", None, "Evidence supports the answer."
    )

    body = captured["body"]
    assert isinstance(body, dict)
    assert body["response_format"] == {"type": "json_object"}
    assert "extra_body" not in body and "reasoning_content" not in body
    if expect_thinking:
        assert body["thinking"] == {"type": "disabled"}
    else:
        assert "thinking" not in body
    assert body["max_tokens"] == 600
    assert payload["choices"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider_payload",
    [
        {},
        {"choices": []},
        {"choices": [{"finish_reason": "length", "message": {"content": "sensitive"}}]},
        {"choices": [{"finish_reason": "tool_calls", "message": {"content": "sensitive"}}]},
        {"choices": [{"finish_reason": None, "message": {"content": "sensitive"}}]},
        {"choices": [{"finish_reason": "stop", "message": {"content": None}}]},
        {"choices": [{"finish_reason": "stop", "message": {"content": ""}}]},
        {"choices": [{"finish_reason": "stop", "message": {"content": "   "}}]},
    ],
)
async def test_generator_fails_closed_before_parsing_invalid_structured_completion(
    monkeypatch: pytest.MonkeyPatch, provider_payload: dict[str, object]
) -> None:
    generator = OpenAICompatibleAnswerGenerator(
        api_key="test-key", base_url="https://example.invalid", model_name="test", timeout_seconds=1
    )
    monkeypatch.setattr(generator, "_post", lambda *_: provider_payload)

    with pytest.raises(AnswerGenerationError) as raised:
        await generator.generate(
            user_query="question",
            selected_evidence_context="[E1] evidence",
            selected_evidence=(),
            answerability="answerable",
            missing_information=None,
            decision_reason="supported",
        )

    assert "sensitive" not in str(raised.value)
