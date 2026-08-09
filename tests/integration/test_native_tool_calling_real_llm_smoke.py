"""Opt-in real native-tool-calling smoke with deterministic safe high-level tool results."""

# ruff: noqa: RUF001

from __future__ import annotations

import os

import pytest

from decision_agent.config import Settings
from decision_agent.routing.models import RouterDecision
from decision_agent.tool_calling.models import AgentToolResult, NativeToolCallingStatus
from decision_agent.tool_calling.runtime import (
    OpenAICompatibleNativeToolCallingModel,
    run_native_tool_calling,
)

pytestmark = pytest.mark.integration


class DeterministicTool:
    def __init__(self, result: AgentToolResult) -> None:
        self.result = result
        self.calls: list[str] = []

    async def run(self, *, query: str) -> AgentToolResult:
        self.calls.append(query)
        return self.result


def _decision(route: str) -> RouterDecision:
    return RouterDecision(
        route=route,
        normalized_query="企业问题",
        decision_reason="已完成路由。",
        knowledge_subquery="产品 A 的售后保修政策是什么？" if route == "knowledge" else None,
        data_subquery="五月销售额最高的产品是什么？" if route == "data" else None,
        missing_information=None,
        confidence=0.9,
    )


@pytest.mark.asyncio
async def test_real_provider_uses_native_tool_calls_for_knowledge_and_data() -> None:
    """Use real tools/tool_calls only; fake tools avoid retrieval, MCP, and database execution."""
    if os.getenv("RUN_NATIVE_TOOL_CALLING_REAL_LLM_SMOKE") != "1":
        pytest.skip(
            "set RUN_NATIVE_TOOL_CALLING_REAL_LLM_SMOKE=1 to run the real native-tools smoke"
        )
    settings = Settings()
    if (
        settings.llm_api_key is None
        or settings.llm_base_url is None
        or settings.llm_model_name is None
    ):
        pytest.skip("LLM settings are not fully configured")
    model = OpenAICompatibleNativeToolCallingModel.from_settings(settings)
    knowledge = DeterministicTool(
        AgentToolResult(
            status="succeeded",
            answer="产品 A 的原装电池保修期为 12 个月。[E1]",
            citations=["[E1]"],
        )
    )
    data = DeterministicTool(
        AgentToolResult(
            status="succeeded",
            answer="五月销售额最高的产品是 Aster 工业泵。[D1]",
            citations=["[D1]"],
        )
    )

    observed: list[dict[str, object]] = []
    for query, route in (
        ("产品 A 的售后保修政策是什么？", "knowledge"),
        ("五月销售额最高的产品是什么？", "data"),
    ):
        result = await run_native_tool_calling(
            user_query=query,
            decision=_decision(route),
            model=model,
            knowledge_tool=knowledge,
            data_tool=data,
        )
        assert result.status is NativeToolCallingStatus.COMPLETED
        assert result.tool_call_id is not None
        assert result.selected_tool == f"run_{route}_agent"
        observed.append(
            {
                "route": route,
                "selected_tool": result.selected_tool,
                "native_tool_call_id_present": True,
            }
        )
    assert knowledge.calls == ["产品 A 的售后保修政策是什么？"]
    assert data.calls == ["五月销售额最高的产品是什么？"]
    print({"native_tool_calling_smoke": observed})
