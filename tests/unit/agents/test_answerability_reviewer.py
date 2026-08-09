"""Offline contracts for the A2 answerability reviewer."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from decision_agent.agents.answerability_reviewer import (
    _SYSTEM_PROMPT,
    AnswerabilityDecision,
    AnswerabilityReviewError,
    OpenAICompatibleAnswerabilityReviewer,
    _build_user_prompt,
    validate_answerability_decision,
)


def _answerable() -> AnswerabilityDecision:
    return AnswerabilityDecision(
        answerability="answerable",
        missing_information=None,
        decision_reason="选中的证据直接规定了所需条件。",
    )


def _unanswerable() -> AnswerabilityDecision:
    return AnswerabilityDecision(
        answerability="unanswerable",
        missing_information="维修完成后的新增免费保修期限",
        decision_reason="选中的证据没有规定维修完成后的新增免费保修期限。",
    )


def test_reviewer_schema_accepts_valid_answerable_and_unanswerable_decisions() -> None:
    assert _answerable().missing_information is None
    assert _unanswerable().missing_information == "维修完成后的新增免费保修期限"


@pytest.mark.parametrize(
    "payload",
    [
        {
            "answerability": "answerable",
            "missing_information": "missing",
            "decision_reason": "理由",
        },
        {"answerability": "unanswerable", "decision_reason": "理由"},
        {"answerability": "answerable", "missing_information": None, "decision_reason": "   "},
        {"answerability": "failed", "missing_information": None, "decision_reason": "理由"},
        {
            "answerability": "answerable",
            "missing_information": None,
            "decision_reason": "理由",
            "answer": "not allowed",
        },
    ],
)
def test_reviewer_schema_rejects_contract_and_extra_field_violations(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        AnswerabilityDecision.model_validate(payload)


def test_reviewer_schema_has_no_hidden_reasoning_field() -> None:
    assert "hidden_reasoning" not in AnswerabilityDecision.model_fields


def test_chinese_query_rejects_english_reason_and_missing_information() -> None:
    english_reason = AnswerabilityDecision(
        answerability="answerable",
        missing_information=None,
        decision_reason="Evidence is sufficient.",
    )
    english_missing = AnswerabilityDecision(
        answerability="unanswerable",
        missing_information="the requested warranty period",
        decision_reason="证据没有规定所需期限。",
    )

    assert validate_answerability_decision(
        user_query="产品 A 保修多久?", decision=english_reason
    ).validation_errors == ["reviewer_language_mismatch"]
    assert validate_answerability_decision(
        user_query="产品 A 保修多久?", decision=english_missing
    ).validation_errors == ["reviewer_language_mismatch"]
    assert validate_answerability_decision(
        user_query="产品 A 保修多久?", decision=_unanswerable()
    ).validation_passed


def test_reviewer_prompt_declares_scope_and_output_constraints() -> None:
    assert "Selected Evidence" in _SYSTEM_PROMPT
    assert "Do not generate a final answer" in _SYSTEM_PROMPT
    assert "no Markdown" in _SYSTEM_PROMPT
    assert "same primary language" in _SYSTEM_PROMPT
    assert "operating data in a later stage" in _SYSTEM_PROMPT
    assert "不得因为制度证据不包含当前经营数值" in _SYSTEM_PROMPT


def test_chinese_query_prompt_repeats_mandatory_simplified_chinese_constraints() -> None:
    prompt = _build_user_prompt(
        user_query="请说明当前政策是否足够。",
        selected_evidence_context="[E1]\n示例证据",
    )

    assert "Mandatory Chinese output requirement" in prompt
    assert "Keep JSON field names in English" in prompt
    assert "Simplified Chinese" in prompt
    assert '"answerability"' in _SYSTEM_PROMPT
    assert "选中的证据明确规定了所需条件。" in _SYSTEM_PROMPT
    assert "所请求的生效日期" in _SYSTEM_PROMPT


def test_english_query_prompt_does_not_force_chinese_output() -> None:
    prompt = _build_user_prompt(
        user_query="Is the selected policy sufficient?",
        selected_evidence_context="[E1]\nExample Evidence",
    )

    assert "Mandatory Chinese output requirement" not in prompt


@pytest.mark.parametrize(
    ("base_url", "expect_thinking"),
    [("https://api.deepseek.com/v1", True), ("https://llm.example.invalid/v1", False)],
)
def test_reviewer_request_uses_the_configured_provider_payload_contract(
    monkeypatch: pytest.MonkeyPatch, base_url: str, *, expect_thinking: bool
) -> None:
    reviewer = OpenAICompatibleAnswerabilityReviewer(
        api_key="test-key",
        base_url=base_url,
        model_name="test-model",
        timeout_seconds=30,
    )
    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": json.dumps(
                                    {
                                        "answerability": "answerable",
                                        "missing_information": None,
                                        "decision_reason": "Selected Evidence is sufficient.",
                                    }
                                )
                            },
                        }
                    ]
                }
            ).encode("utf-8")

    def fake_urlopen(request: object, *, timeout: float) -> FakeResponse:
        captured["body"] = json.loads(request.data.decode("utf-8"))  # type: ignore[attr-defined]
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("decision_agent.agents.answerability_reviewer.urlopen", fake_urlopen)

    response = reviewer._post("English question?", "[E1]\nEvidence")

    assert response["choices"]
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["response_format"] == {"type": "json_object"}
    assert "extra_body" not in body and "reasoning_content" not in body
    if expect_thinking:
        assert body["thinking"] == {"type": "disabled"}
    else:
        assert "thinking" not in body
    assert body["temperature"] == 0


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
async def test_reviewer_fails_closed_before_parsing_invalid_structured_completion(
    monkeypatch: pytest.MonkeyPatch, provider_payload: dict[str, object]
) -> None:
    reviewer = OpenAICompatibleAnswerabilityReviewer(
        api_key="test-key", base_url="https://example.invalid", model_name="test", timeout_seconds=1
    )
    monkeypatch.setattr(reviewer, "_post", lambda *_: provider_payload)

    with pytest.raises(AnswerabilityReviewError) as raised:
        await reviewer.review(
            user_query="question", selected_evidence_context="[E1] evidence", selected_evidence=()
        )

    assert "sensitive" not in str(raised.value)
