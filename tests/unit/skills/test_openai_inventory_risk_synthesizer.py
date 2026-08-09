"""Offline provider-boundary tests for inventory-risk synthesis."""

from __future__ import annotations

import json

import pytest

from decision_agent.skills.inventory_risk_synthesizer import (
    InventoryRiskSynthesisInput,
    InventoryRiskSynthesizerError,
    OpenAICompatibleInventoryRiskSynthesizer,
)
from decision_agent.tool_calling.runtime import (
    NativeToolCallingError,
    OpenAICompatibleNativeToolCallingModel,
)


def input_data() -> InventoryRiskSynthesisInput:
    return InventoryRiskSynthesisInput(
        original_request="库存风险与补货制度",
        data_subquery="哪些产品库存不足",
        data_answer="产品 A 库存低于安全库存。[D1]",
        data_citations=("[D1]",),
        knowledge_subquery="公司的补货制度是什么",
        knowledge_answer="制度要求及时补货。[E1]",
        knowledge_citations=("[E1]",),
    )


def valid_content() -> str:
    return json.dumps(
        {
            "risk_summary": "产品 A 存在库存风险。",
            "policy_basis": "制度要求及时补货。",
            "recommended_actions": ["安排补货。"],
            "citations": ["[D1]", "[E1]"],
        },
        ensure_ascii=False,
    )


def provider_response(
    content: object,
    *,
    finish_reason: object = "stop",
) -> dict[str, object]:
    return {
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {"content": content},
            }
        ]
    }


class FakeChatClient:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    async def complete_chat(
        self, *, messages: list[dict[str, object]], response_format: dict[str, str]
    ) -> dict[str, object]:
        self.calls.append({"messages": messages, "response_format": response_format})
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response  # type: ignore[return-value]


@pytest.mark.asyncio
async def test_synthesizer_uses_one_plain_json_request_with_minimal_prompt() -> None:
    captured: list[dict[str, object]] = []

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(provider_response(valid_content())).encode("utf-8")

    def fake_urlopen(request, timeout: float):  # type: ignore[no-untyped-def]
        captured.append(json.loads(request.data.decode("utf-8")))
        return Response()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("decision_agent.tool_calling.runtime.urlopen", fake_urlopen)
    try:
        synthesizer = OpenAICompatibleInventoryRiskSynthesizer(
            client=OpenAICompatibleNativeToolCallingModel(
                api_key="test-key",
                base_url="https://api.deepseek.com",
                model_name="test-model",
                timeout_seconds=5,
            )
        )
        result = await synthesizer.synthesize(input_data())
    finally:
        monkeypatch.undo()

    assert result.citations == ("[D1]", "[E1]")
    assert len(captured) == 1
    payload = captured[0]
    assert payload["model"] == "test-model"
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["response_format"] == {"type": "json_object"}
    assert "tools" not in payload and "tool_choice" not in payload
    assert "extra_body" not in payload and "reasoning_content" not in payload
    prompt = "\n".join(message["content"] for message in payload["messages"])
    assert "Output JSON only, without Markdown." in prompt
    assert '"recommended_actions":["..."]' in prompt
    assert '"citations":["[D1]","[E1]"]' in prompt
    assert "产品 A 库存低于安全库存。[D1]" in prompt
    assert "制度要求及时补货。[E1]" in prompt
    assert all(forbidden not in prompt.lower() for forbidden in ("sql", "schema", "mcp", "http"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "error_code", "failure_stage", "finish_reason"),
    [
        ({}, "inventory_risk_synthesizer_missing_choice", "response_choices", None),
        ({"choices": []}, "inventory_risk_synthesizer_missing_choice", "response_choices", None),
        (
            {"choices": [provider_response(valid_content())["choices"][0]] * 2},
            "inventory_risk_synthesizer_missing_choice",
            "response_choices",
            None,
        ),
        (
            provider_response(valid_content(), finish_reason="length"),
            "inventory_risk_synthesizer_truncated",
            "response_finish",
            "length",
        ),
        (
            provider_response(valid_content(), finish_reason="tool_calls"),
            "inventory_risk_synthesizer_invalid_finish_reason",
            "response_finish",
            "tool_calls",
        ),
        (
            provider_response(None),
            "inventory_risk_synthesizer_empty_content",
            "response_content",
            None,
        ),
        (
            provider_response(""),
            "inventory_risk_synthesizer_empty_content",
            "response_content",
            None,
        ),
        (
            provider_response("   "),
            "inventory_risk_synthesizer_empty_content",
            "response_content",
            None,
        ),
        (
            provider_response("not-json"),
            "inventory_risk_synthesizer_invalid_json",
            "response_json",
            None,
        ),
        (
            provider_response("[]"),
            "inventory_risk_synthesizer_schema_invalid",
            "response_schema",
            None,
        ),
        (
            provider_response(
                json.dumps(
                    {
                        "policy_basis": "y",
                        "recommended_actions": ["z"],
                        "citations": ["[D1]", "[E1]"],
                    }
                )
            ),
            "inventory_risk_synthesizer_schema_invalid",
            "response_schema",
            None,
        ),
        (
            provider_response(
                json.dumps(
                    {
                        "risk_summary": "x",
                        "policy_basis": "y",
                        "recommended_actions": ["z"],
                        "citations": ["[D1]", "[E1]"],
                        "unexpected": True,
                    }
                )
            ),
            "inventory_risk_synthesizer_schema_invalid",
            "response_schema",
            None,
        ),
        (
            provider_response(
                json.dumps(
                    {
                        "risk_summary": "x",
                        "policy_basis": "y",
                        "recommended_actions": "z",
                        "citations": ["[D1]", "[E1]"],
                    }
                )
            ),
            "inventory_risk_synthesizer_schema_invalid",
            "response_schema",
            None,
        ),
        (
            provider_response(
                json.dumps(
                    {
                        "risk_summary": "x",
                        "policy_basis": "y",
                        "recommended_actions": ["z"],
                        "citations": ["[D1]"],
                    }
                )
            ),
            "inventory_risk_synthesis_citations_invalid",
            "response_citations",
            None,
        ),
        (
            provider_response(
                json.dumps(
                    {
                        "risk_summary": "x",
                        "policy_basis": "y",
                        "recommended_actions": ["z"],
                        "citations": ["[E1]"],
                    }
                )
            ),
            "inventory_risk_synthesis_citations_invalid",
            "response_citations",
            None,
        ),
        (
            provider_response(
                json.dumps(
                    {
                        "risk_summary": "x",
                        "policy_basis": "y",
                        "recommended_actions": ["z"],
                        "citations": ["[D99]", "[E1]"],
                    }
                )
            ),
            "inventory_risk_synthesis_citations_invalid",
            "response_citations",
            None,
        ),
    ],
)
async def test_provider_response_failures_are_classified_safely(
    response: dict[str, object],
    error_code: str,
    failure_stage: str,
    finish_reason: str | None,
) -> None:
    client = FakeChatClient(response)
    synthesizer = OpenAICompatibleInventoryRiskSynthesizer(client=client)

    with pytest.raises(InventoryRiskSynthesizerError, match="could not be completed") as raised:
        await synthesizer.synthesize(input_data())

    assert raised.value.code == error_code
    assert raised.value.failure_stage == failure_stage
    assert raised.value.finish_reason == (
        "stop" if finish_reason is None and failure_stage != "response_choices" else finish_reason
    )
    assert "not-json" not in str(raised.value)
    assert "[D99]" not in str(raised.value)
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_standard_json_arrays_and_different_field_order_are_accepted() -> None:
    content = json.dumps(
        {
            "citations": ["[D1]", "[E1]"],
            "recommended_actions": ["采取补货行动"],
            "policy_basis": "制度要求及时补货。",
            "risk_summary": "库存存在风险。",
        },
        ensure_ascii=False,
    )
    client = FakeChatClient(provider_response(content))

    result = await OpenAICompatibleInventoryRiskSynthesizer(client=client).synthesize(input_data())

    assert result.recommended_actions == ("采取补货行动",)
    assert result.citations == ("[D1]", "[E1]")
    assert len(client.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "expected_fields"),
    [
        (
            {
                "risk_summary": "x",
                "policy_basis": "y",
                "recommended_actions": "sensitive-provider-value",
                "citations": ["[D1]", "[E1]"],
            },
            ("recommended_actions:tuple_type",),
        ),
        (
            {
                "risk_summary": "x",
                "policy_basis": "y",
                "recommended_actions": [{"hidden": "sensitive-provider-value"}],
                "citations": ["[D1]", "[E1]"],
            },
            ("recommended_actions:string_type", "recommended_actions:too_short"),
        ),
        (
            {
                "risk_summary": "x",
                "policy_basis": "y",
                "recommended_actions": ["z"],
                "citations": "[D1]",
            },
            ("citations:tuple_type",),
        ),
        (
            {
                "policy_basis": "y",
                "recommended_actions": ["z"],
                "citations": ["[D1]", "[E1]"],
            },
            ("risk_summary:missing",),
        ),
        (
            {
                "risk_summary": "x",
                "policy_basis": "y",
                "recommended_actions": ["z"],
                "citations": ["[D1]", "[E1]"],
                "untrusted_provider_field": "sensitive-provider-value",
            },
            ("extra_field:extra_forbidden",),
        ),
        (
            {
                "risk_summary": {"hidden": "sensitive-provider-value"},
                "policy_basis": "y",
                "recommended_actions": ["z"],
                "citations": ["[D1]", "[E1]"],
            },
            ("risk_summary:string_type",),
        ),
        (
            {
                "risk_summary": "x",
                "policy_basis": None,
                "recommended_actions": ["z"],
                "citations": ["[D1]", "[E1]"],
            },
            ("policy_basis:string_type",),
        ),
    ],
)
async def test_schema_diagnostics_retain_only_safe_field_type_labels(
    payload: dict[str, object], expected_fields: tuple[str, ...]
) -> None:
    client = FakeChatClient(provider_response(json.dumps(payload)))

    with pytest.raises(InventoryRiskSynthesizerError) as raised:
        await OpenAICompatibleInventoryRiskSynthesizer(client=client).synthesize(input_data())

    assert raised.value.code == "inventory_risk_synthesizer_schema_invalid"
    assert raised.value.failure_stage == "response_schema"
    assert raised.value.finish_reason == "stop"
    assert raised.value.schema_error_fields == expected_fields
    assert "sensitive-provider-value" not in str(raised.value)
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_valid_response_is_parsed_once_with_bounded_dual_citations() -> None:
    client = FakeChatClient(provider_response(valid_content()))

    result = await OpenAICompatibleInventoryRiskSynthesizer(client=client).synthesize(input_data())

    assert result.citations == ("[D1]", "[E1]")
    assert len(client.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 429, 500])
async def test_http_status_is_safe_without_provider_details(status: int) -> None:
    client = FakeChatClient(
        NativeToolCallingError("tool_calling_provider_http_error", http_status=status)
    )
    synthesizer = OpenAICompatibleInventoryRiskSynthesizer(client=client)

    with pytest.raises(InventoryRiskSynthesizerError) as raised:
        await synthesizer.synthesize(input_data())

    assert raised.value.code == "inventory_risk_synthesizer_http_error"
    assert raised.value.http_status == status
    assert "http" not in str(raised.value).lower()
    assert len(client.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        NativeToolCallingError("tool_calling_provider_unavailable"),
        OSError("https://secret.example.test"),
        TimeoutError(),
    ],
)
async def test_network_and_timeout_errors_are_safe(failure: BaseException) -> None:
    client = FakeChatClient(failure)
    synthesizer = OpenAICompatibleInventoryRiskSynthesizer(client=client)

    with pytest.raises(InventoryRiskSynthesizerError) as raised:
        await synthesizer.synthesize(input_data())

    assert raised.value.code == "inventory_risk_synthesizer_unavailable"
    assert raised.value.http_status is None
    assert "secret" not in str(raised.value)
    assert len(client.calls) == 1
