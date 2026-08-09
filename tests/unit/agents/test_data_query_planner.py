# ruff: noqa: RUF001

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from decision_agent.agents.data_query_planner import (
    _SYSTEM_PROMPT,
    DataPlanStatus,
    DataQueryPlan,
    DataQueryPlanningError,
    OpenAICompatibleDataQueryPlanner,
    _build_system_prompt,
    validate_data_query_plan,
)


def test_ready_plan_requires_sql_and_rejects_extra_answer_fields() -> None:
    plan = DataQueryPlan(
        status=DataPlanStatus.READY,
        intent="查询产品销售额",
        sql="SELECT product_name FROM products",
        decision_reason="已具备查询所需条件。",
    )
    assert plan.missing_information is None
    with pytest.raises(ValidationError):
        DataQueryPlan(status="ready", intent="x", sql=None, decision_reason="x")
    with pytest.raises(ValidationError):
        DataQueryPlan(status="ready", intent="x", sql="SELECT 1", decision_reason="x", answer="x")


@pytest.mark.parametrize("status", [DataPlanStatus.NEEDS_CLARIFICATION, DataPlanStatus.UNSUPPORTED])
def test_non_ready_plan_cannot_carry_sql(status: DataPlanStatus) -> None:
    with pytest.raises(ValidationError):
        DataQueryPlan(status=status, intent="澄清", sql="SELECT 1", decision_reason="需要更多条件")


def test_chinese_plan_requires_chinese_natural_language_fields() -> None:
    plan = DataQueryPlan(
        status="needs_clarification",
        intent="clarify period",
        decision_reason="period is missing",
        missing_information="统计期间",
    )
    assert validate_data_query_plan(
        user_query="销售额最高的产品是什么？", plan=plan
    ).validation_errors == ["data_planner_language_mismatch"]


def test_planner_prompt_exposes_only_schema_and_business_rules() -> None:
    assert "Only create one SELECT" in _SYSTEM_PROMPT
    assert "current database snapshot across all authorized rows" in _SYSTEM_PROMPT
    assert "inventory-risk screening" in _SYSTEM_PROMPT
    assert "query the authorized raw fields" in _SYSTEM_PROMPT
    assert "current_inventory definition resolves the period" in _SYSTEM_PROMPT
    assert "status 必须为 ready" in _SYSTEM_PROMPT
    prompt = _build_system_prompt(
        {"products": ["product_id"]}, {"current_inventory": "per-product latest snapshot"}
    )
    assert '"products"' in prompt
    assert "per-product latest snapshot" in prompt
    assert "Ground Truth" not in _SYSTEM_PROMPT
    assert "expected_result" not in _SYSTEM_PROMPT


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("base_url", "expect_thinking"),
    [("https://api.deepseek.com/v1", True), ("https://example.test/v1", False)],
)
async def test_planner_request_uses_the_configured_provider_payload_contract(
    monkeypatch: pytest.MonkeyPatch, base_url: str, *, expect_thinking: bool
) -> None:
    captured: dict[str, object] = {}

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            content = json.dumps(
                {
                    "status": "unsupported",
                    "intent": "not supported",
                    "sql": None,
                    "decision_reason": "not available",
                    "missing_information": None,
                }
            )
            return json.dumps(
                {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}
            ).encode()

    def fake_urlopen(request, timeout: float):  # type: ignore[no-untyped-def]
        captured["body"] = request.data
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("decision_agent.agents.data_query_planner.urlopen", fake_urlopen)
    planner = OpenAICompatibleDataQueryPlanner(
        api_key="test-key", base_url=base_url, model_name="test", timeout_seconds=9
    )
    plan = await planner.plan(
        user_query="question",
        enterprise_schema={"products": ["product_id"]},
        business_definitions={"current_inventory": "per-product latest snapshot"},
    )
    assert plan.status is DataPlanStatus.UNSUPPORTED
    body = json.loads(captured["body"])
    assert body["response_format"] == {"type": "json_object"}
    assert body["max_tokens"] == 1500
    assert "extra_body" not in body and "reasoning_content" not in body
    if expect_thinking:
        assert body["thinking"] == {"type": "disabled"}
    else:
        assert "thinking" not in body


def _payload(*, content: object, finish_reason: object = "stop") -> dict[str, object]:
    return {
        "choices": [{"finish_reason": finish_reason, "message": {"content": content}}],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "expected_code", "expected_details"),
    [
        (
            _payload(content=""),
            "data_plan_response_empty",
            {"finish_reason": "stop", "response_empty": True, "token_limit_reached": False},
        ),
        (
            _payload(content="{"),
            "data_plan_json_parse_failed",
            {"finish_reason": "stop", "response_empty": False, "token_limit_reached": False},
        ),
        (
            _payload(content="{", finish_reason="length"),
            "data_plan_output_truncated",
            {"finish_reason": "length", "response_empty": False, "token_limit_reached": True},
        ),
        (
            _payload(content="{", finish_reason="provider-specific-detail"),
            "data_query_provider_invalid_response",
            {"finish_reason": "unknown", "response_empty": False, "token_limit_reached": False},
        ),
        (
            _payload(content="sensitive", finish_reason="tool_calls"),
            "data_query_provider_invalid_response",
            {"finish_reason": "tool_calls", "response_empty": False, "token_limit_reached": False},
        ),
        (
            {"choices": [{"finish_reason": "length"}]},
            "data_plan_output_truncated",
            {"finish_reason": "length", "response_empty": False, "token_limit_reached": True},
        ),
        (
            {"choices": [{"finish_reason": "length", "message": {}}]},
            "data_plan_output_truncated",
            {"finish_reason": "length", "response_empty": False, "token_limit_reached": True},
        ),
        (
            _payload(content="sensitive", finish_reason=None),
            "data_query_provider_invalid_response",
            {"finish_reason": None, "response_empty": False, "token_limit_reached": False},
        ),
        (
            _payload(content=None),
            "data_plan_response_empty",
            {"finish_reason": "stop", "response_empty": True, "token_limit_reached": False},
        ),
        (
            _payload(content="   "),
            "data_plan_response_empty",
            {"finish_reason": "stop", "response_empty": True, "token_limit_reached": False},
        ),
        (
            _payload(
                content=json.dumps(
                    {
                        "status": "unsupported",
                        "intent": "unsupported",
                        "sql": None,
                        "decision_reason": "unsupported",
                        "missing_information": None,
                        "answer": "forbidden",
                    }
                )
            ),
            "data_plan_schema_validation_failed",
            {"finish_reason": "stop", "response_empty": False, "token_limit_reached": False},
        ),
    ],
)
async def test_planner_classifies_safe_provider_and_output_failures(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object],
    expected_code: str,
    expected_details: dict[str, str | bool],
) -> None:
    planner = OpenAICompatibleDataQueryPlanner(
        api_key="test-key", base_url="https://example.test", model_name="test", timeout_seconds=9
    )
    monkeypatch.setattr(planner, "_post", lambda user_query, schema, definitions: payload)
    with pytest.raises(DataQueryPlanningError) as raised:
        await planner.plan(
            user_query="question",
            enterprise_schema={"products": ["product_id"]},
            business_definitions={"current_inventory": "per-product latest snapshot"},
        )
    assert raised.value.subcode == expected_code
    assert raised.value.details == expected_details
    assert "question" not in str(raised.value.details)


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [{}, {"choices": []}])
async def test_planner_rejects_missing_choices_without_parsing_content(
    monkeypatch: pytest.MonkeyPatch, payload: dict[str, object]
) -> None:
    planner = OpenAICompatibleDataQueryPlanner(
        api_key="test-key", base_url="https://example.test", model_name="test", timeout_seconds=9
    )
    monkeypatch.setattr(planner, "_post", lambda *_: payload)

    with pytest.raises(DataQueryPlanningError) as raised:
        await planner.plan(
            user_query="private question",
            enterprise_schema={"products": ["product_id"]},
            business_definitions={"current_inventory": "latest snapshot"},
        )

    assert raised.value.subcode == "data_query_provider_invalid_response"
    assert "private question" not in str(raised.value)
