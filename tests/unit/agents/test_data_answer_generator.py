# ruff: noqa: RUF001

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from decision_agent.agents.data_answer_generator import (
    _SYSTEM_PROMPT,
    DataAnswerDraft,
    DataAnswerGenerationError,
    OpenAICompatibleDataAnswerGenerator,
    project_data_answer_provider_payload,
    project_data_evidence,
    render_data_evidence,
    validate_data_citations,
)
from decision_agent.data_agent.models import DataEvidence


def test_data_answer_requires_exact_inline_d_citation() -> None:
    draft = DataAnswerDraft(answer="销售额为 12000 元。[D1]", citations=["[D1]"])
    result = validate_data_citations(evidence_ids=["D1"], draft=draft)
    assert result.validation_passed is True
    assert result.normalized_citations == ["[D1]"]


@pytest.mark.parametrize(
    "draft",
    [
        DataAnswerDraft(answer="销售额为 12000 元。", citations=["[D1]"]),
        DataAnswerDraft(answer="销售额为 12000 元。[E1]", citations=["[E1]"]),
        DataAnswerDraft(answer="销售额为 12000 元。[D1]", citations=["D1"]),
        DataAnswerDraft(answer="销售额为 12000 元。[D2]", citations=["[D2]"]),
        DataAnswerDraft(answer="根据 D1，销售额为 12000 元。[D1]", citations=["[D1]"]),
    ],
)
def test_data_answer_rejects_invalid_or_unavailable_citations(draft: DataAnswerDraft) -> None:
    assert validate_data_citations(evidence_ids=["D1"], draft=draft).validation_passed is False


def test_data_answer_schema_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        DataAnswerDraft(answer="x", citations=[], reasoning="hidden")


def test_generator_prompt_requires_only_executed_data_and_inline_d_citations() -> None:
    assert "Use only the executed Safe Data Projection" in _SYSTEM_PROMPT
    assert "inline with actual [D<number>] IDs" in _SYSTEM_PROMPT
    assert "[E#]" in _SYSTEM_PROMPT
    assert "When source_truncated or projection_truncated is true" in _SYSTEM_PROMPT


@pytest.mark.parametrize("truncated", [False, True])
def test_rendered_data_evidence_is_bounded_and_omits_sql_and_raw_rows(truncated: bool) -> None:
    rendered = json.loads(
        render_data_evidence(
            (
                DataEvidence(
                    evidence_id="D1",
                    normalized_sql="SELECT product_id FROM products LIMIT 1",
                    columns=["product_id"],
                    rows=[["P100"]],
                    row_count=1,
                    truncated=truncated,
                    accessed_tables=["products"],
                    elapsed_ms=1.0,
                ),
            )
        )
    )
    assert rendered == [
        {
            "evidence_id": "D1",
            "columns": ["product_id"],
            "records": [{"product_id": "P100"}],
            "row_count": 1,
            "source_truncated": truncated,
            "projection_truncated": False,
        }
    ]


def test_projection_limits_records_and_rejects_sensitive_columns_and_values() -> None:
    evidence = DataEvidence(
        evidence_id="D1",
        normalized_sql="SELECT product_id FROM products LIMIT 20",
        columns=["product_id"],
        rows=[[f"P{index:03d}"] for index in range(12)],
        row_count=12,
        truncated=False,
        accessed_tables=["products"],
        elapsed_ms=1.0,
    )
    projected = project_data_evidence((evidence,))[0]
    assert len(projected.records) == 8
    assert projected.projection_truncated is True
    assert "SELECT" not in projected.model_dump_json()
    assert "accessed_tables" not in projected.model_dump_json()

    with pytest.raises(DataAnswerGenerationError) as forbidden_column:
        project_data_evidence((evidence.model_copy(update={"columns": ["tenant_id"]}),))
    assert forbidden_column.value.subcode == "data_projection_column_forbidden"
    with pytest.raises(DataAnswerGenerationError) as forbidden_value:
        project_data_evidence((evidence.model_copy(update={"rows": [[{"nested": "value"}]]}),))
    assert forbidden_value.value.subcode == "data_projection_value_forbidden"


def test_governance_payload_projection_contains_no_data_evidence_internals() -> None:
    evidence = DataEvidence(
        evidence_id="D1",
        normalized_sql="SELECT effective_purchase_amount FROM totals LIMIT 1",
        columns=["effective_purchase_amount"],
        rows=[["19840.00"]],
        row_count=1,
        truncated=False,
        accessed_tables=["purchase_orders"],
        elapsed_ms=1.0,
    )
    payload = project_data_answer_provider_payload(
        (), {"user_query": "question", "data_evidence": (evidence,)}
    )
    serialized = json.dumps(payload, ensure_ascii=False, default=str).casefold()
    assert "19840.00" in serialized
    assert all(
        forbidden not in serialized
        for forbidden in ("select ", "normalized_sql", "accessed_tables", '"rows"')
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("base_url", "expect_thinking"),
    [("https://api.deepseek.com/v1", True), ("https://example.test/v1", False)],
)
async def test_generator_request_uses_the_configured_provider_payload_contract(
    monkeypatch: pytest.MonkeyPatch, base_url: str, *, expect_thinking: bool
) -> None:
    captured: dict[str, object] = {}

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            content = json.dumps({"answer": "x[D1]", "citations": ["[D1]"]})
            return json.dumps(
                {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}
            ).encode()

    def fake_urlopen(request, timeout: float):  # type: ignore[no-untyped-def]
        captured["body"] = request.data
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("decision_agent.agents.data_answer_generator.urlopen", fake_urlopen)
    generator = OpenAICompatibleDataAnswerGenerator(
        api_key="test-key", base_url=base_url, model_name="test", timeout_seconds=9
    )
    draft = await generator.generate(
        user_query="question",
        data_evidence=(
            DataEvidence(
                evidence_id="D1",
                normalized_sql="SELECT 1",
                columns=["value"],
                rows=[[1]],
                row_count=1,
                truncated=False,
                accessed_tables=[],
                elapsed_ms=1.0,
            ),
        ),
    )
    assert draft.citations == ["[D1]"]
    body = json.loads(captured["body"])
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
        ({}, "data_answer_missing_choice"),
        ({"choices": []}, "data_answer_missing_choice"),
        (
            {"choices": [{"finish_reason": "length", "message": {"content": "sensitive"}}]},
            "data_answer_truncated",
        ),
        (
            {"choices": [{"finish_reason": "tool_calls", "message": {"content": "sensitive"}}]},
            "data_answer_invalid_finish_reason",
        ),
        (
            {"choices": [{"finish_reason": None, "message": {"content": "sensitive"}}]},
            "data_answer_invalid_finish_reason",
        ),
        (
            {"choices": [{"finish_reason": "stop", "message": {"content": None}}]},
            "data_answer_empty_content",
        ),
        (
            {"choices": [{"finish_reason": "stop", "message": {"content": ""}}]},
            "data_answer_empty_content",
        ),
        (
            {"choices": [{"finish_reason": "stop", "message": {"content": "   "}}]},
            "data_answer_empty_content",
        ),
    ],
)
async def test_data_generator_fails_closed_before_json_parsing(
    monkeypatch: pytest.MonkeyPatch,
    provider_payload: dict[str, object],
    expected_subcode: str,
) -> None:
    generator = OpenAICompatibleDataAnswerGenerator(
        api_key="test-key", base_url="https://example.invalid", model_name="test", timeout_seconds=1
    )
    monkeypatch.setattr(generator, "_post", lambda *_: provider_payload)

    with pytest.raises(DataAnswerGenerationError) as raised:
        await generator.generate(user_query="question", data_evidence=())

    assert raised.value.subcode == expected_subcode
    assert "sensitive" not in str(raised.value)
