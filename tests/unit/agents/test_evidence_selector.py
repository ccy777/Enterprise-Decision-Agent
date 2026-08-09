"""Offline contracts for the A2 Evidence Selector."""

from __future__ import annotations

import json
from urllib.error import HTTPError

import pytest
from pydantic import ValidationError

from decision_agent.agents.evidence_selector import (
    _SYSTEM_PROMPT,
    EvidenceSelection,
    EvidenceSelectionError,
    OpenAICompatibleEvidenceSelector,
    validate_evidence_selection,
)


def _completion(content: object, *, finish_reason: object = "stop") -> dict[str, object]:
    return {"choices": [{"finish_reason": finish_reason, "message": {"content": content}}]}


def test_valid_and_empty_selection_results_are_supported() -> None:
    selected = EvidenceSelection(
        selected_evidence_ids=["[E3]", "[E1]", "[E1]"],
        selection_reason="Direct policy evidence is relevant.",
    )
    empty = EvidenceSelection(
        selected_evidence_ids=[], selection_reason="No direct evidence exists."
    )

    result = validate_evidence_selection(evidence_ids=["E1", "E2", "E3"], selection=selected)

    assert result.validation_passed
    assert result.normalized_selected_evidence_ids == ["[E1]", "[E3]"]
    assert (
        validate_evidence_selection(
            evidence_ids=["E1"], selection=empty
        ).normalized_selected_evidence_ids
        == []
    )


@pytest.mark.parametrize(
    ("selected_ids", "expected_error"),
    [
        (["E1"], "invalid_selected_evidence_id"),
        (["[E9]"], "selected_evidence_not_found"),
        (["DOC-CS-001"], "invalid_selected_evidence_id"),
    ],
)
def test_invalid_or_unknown_selection_ids_are_rejected(
    selected_ids: list[str], expected_error: str
) -> None:
    selection = EvidenceSelection(
        selected_evidence_ids=selected_ids,
        selection_reason="Selection audit summary.",
    )

    result = validate_evidence_selection(evidence_ids=["E1"], selection=selection)

    assert result.validation_passed is False
    assert expected_error in result.validation_errors


def test_empty_selection_reason_is_rejected() -> None:
    with pytest.raises(ValidationError, match="selection_reason"):
        EvidenceSelection(selected_evidence_ids=[], selection_reason="")


def test_evidence_selector_schema_has_no_hidden_reasoning_field() -> None:
    assert "hidden_reasoning" not in EvidenceSelection.model_fields


def test_selector_prompt_includes_exact_bracketed_id_example() -> None:
    assert '"selected_evidence_ids":["[E1]"]' in _SYSTEM_PROMPT
    assert 'The string "E1" without square brackets is invalid.' in _SYSTEM_PROMPT
    assert "policy rules or criteria" in _SYSTEM_PROMPT
    assert "separate data source" in _SYSTEM_PROMPT
    assert "不得因为制度证据不包含当前经营数值" in _SYSTEM_PROMPT


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_result", "expected_subcode"),
    [
        (HTTPError("https://llm.example.invalid", 500, "error", {}, None), "selector_http_error"),
        (_completion("not-json"), "selector_json_parse_failed"),
        (
            _completion('{"selected_evidence_ids": []}'),
            "selector_schema_validation_failed",
        ),
    ],
)
async def test_selector_provider_failures_have_safe_distinct_subcodes(
    monkeypatch: pytest.MonkeyPatch,
    provider_result: object,
    expected_subcode: str,
) -> None:
    selector = OpenAICompatibleEvidenceSelector(
        api_key="test-key",
        base_url="https://llm.example.invalid/v1",
        model_name="test-model",
        timeout_seconds=30,
    )

    def fake_post(*_: object) -> object:
        if isinstance(provider_result, Exception):
            raise provider_result
        return provider_result

    monkeypatch.setattr(selector, "_post", fake_post)

    with pytest.raises(EvidenceSelectionError) as exc_info:
        await selector.select(
            user_query="test", evidence_context="[E1]\nEvidence", retrieval_evidence=[]
        )

    assert exc_info.value.subcode == expected_subcode
    assert "test-key" not in str(exc_info.value)


@pytest.mark.parametrize(
    ("base_url", "expect_thinking"),
    [("https://api.deepseek.com/v1", True), ("https://example.invalid/v1", False)],
)
def test_selector_payload_uses_the_configured_provider_contract(
    monkeypatch: pytest.MonkeyPatch, base_url: str, *, expect_thinking: bool
) -> None:
    captured: dict[str, object] = {}

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                _completion('{"selected_evidence_ids":[],"selection_reason":"None."}')
            ).encode()

    def fake_urlopen(request: object, *, timeout: float) -> Response:
        captured["body"] = json.loads(request.data.decode("utf-8"))  # type: ignore[attr-defined]
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("decision_agent.agents.evidence_selector.urlopen", fake_urlopen)
    selector = OpenAICompatibleEvidenceSelector(
        api_key="test-key", base_url=base_url, model_name="test", timeout_seconds=1
    )
    selector._post("question", "[E1] evidence")

    body = captured["body"]
    assert isinstance(body, dict)
    assert body["response_format"] == {"type": "json_object"}
    assert "extra_body" not in body and "reasoning_content" not in body
    if expect_thinking:
        assert body["thinking"] == {"type": "disabled"}
    else:
        assert "thinking" not in body


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_payload", "expected_subcode"),
    [
        ({}, "selector_missing_choice"),
        ({"choices": []}, "selector_missing_choice"),
        (_completion("sensitive", finish_reason="length"), "selector_truncated"),
        (_completion("sensitive", finish_reason="tool_calls"), "selector_invalid_finish_reason"),
        (_completion("sensitive", finish_reason=None), "selector_invalid_finish_reason"),
        (_completion(None), "selector_empty_content"),
        (_completion(""), "selector_empty_content"),
        (_completion("   "), "selector_empty_content"),
    ],
)
async def test_selector_fails_closed_before_json_parsing(
    monkeypatch: pytest.MonkeyPatch,
    provider_payload: dict[str, object],
    expected_subcode: str,
) -> None:
    selector = OpenAICompatibleEvidenceSelector(
        api_key="test-key", base_url="https://example.invalid", model_name="test", timeout_seconds=1
    )
    monkeypatch.setattr(selector, "_post", lambda *_: provider_payload)

    with pytest.raises(EvidenceSelectionError) as raised:
        await selector.select(
            user_query="question", evidence_context="[E1] evidence", retrieval_evidence=()
        )

    assert raised.value.subcode == expected_subcode
    assert "sensitive" not in str(raised.value)
