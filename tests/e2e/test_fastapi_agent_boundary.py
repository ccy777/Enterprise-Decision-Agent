"""HTTP application acceptance for the factory-built formal Knowledge runtime."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from decision_agent.api import create_app
from decision_agent.application import (
    FormalMemoryConfiguration,
    build_formal_request_executor,
)
from decision_agent.config import Environment, Settings
from decision_agent.context.models import ContextItem, ContextKind
from decision_agent.coordination import build_default_coordinator
from decision_agent.routing.models import RequestRoute, RouterDecision
from decision_agent.skills.inventory_risk_synthesizer import (
    InventoryRiskSynthesisInput,
    InventoryRiskSynthesisResult,
)
from decision_agent.skills.native_runtime import NativeToolCallingSkillExecutor
from decision_agent.tool_calling.models import AgentToolResult

pytestmark = [pytest.mark.e2e, pytest.mark.offline_integration]


class _DeterministicRouter:
    def __init__(self, query: str) -> None:
        self._query = query
        self.calls: list[tuple[str, tuple[ContextItem, ...]]] = []

    async def route_with_context(
        self,
        *,
        user_query: str,
        selected_items: tuple[ContextItem, ...],
    ) -> RouterDecision:
        self.calls.append((user_query, selected_items))
        return RouterDecision(
            route=RequestRoute.KNOWLEDGE,
            normalized_query=self._query,
            decision_reason="deterministic_module_6c1_http_acceptance",
            knowledge_subquery=self._query,
            confidence=1,
        )


class _DeterministicNativeModel:
    def __init__(self, query: str) -> None:
        self._query = query
        self.calls: list[dict[str, object]] = []

    async def complete(
        self,
        *,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
        tool_choice: str,
        response_format: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "messages": tuple(messages),
                "tools": tuple(tools),
                "tool_choice": tool_choice,
                "response_format": response_format,
            }
        )
        if tool_choice == "required":
            return {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "module-6c1-knowledge-call",
                                    "type": "function",
                                    "function": {
                                        "name": "run_knowledge_agent",
                                        "arguments": json.dumps({"query": self._query}),
                                    },
                                }
                            ],
                        },
                    }
                ]
            }

        tool_message = next(message for message in reversed(messages) if message["role"] == "tool")
        payload = json.loads(str(tool_message["content"]))
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            {
                                "answer": payload["answer"],
                                "citations": payload["citations"],
                            }
                        )
                    },
                }
            ]
        }


class _RecordingKnowledgeTool:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def run(self, *, query: str) -> AgentToolResult:
        self.queries.append(query)
        return AgentToolResult(
            status="succeeded",
            answer="FASTAPI_FORMAL_RUNTIME_MARKER [E1]",
            citations=["[E1]"],
        )


class _UnusedDataTool:
    async def run(self, *, query: str) -> AgentToolResult:
        raise AssertionError(f"Knowledge HTTP path must not call the Data Tool: {query}")


class _UnusedSynthesizer:
    async def synthesize(
        self,
        input_data: InventoryRiskSynthesisInput,
    ) -> InventoryRiskSynthesisResult:
        raise AssertionError(
            f"Knowledge HTTP path must not synthesize Mixed output: {input_data!r}"
        )


def test_fastapi_agent_endpoint_executes_formal_runtime() -> None:
    query = "说明库存补货制度"
    router = _DeterministicRouter(query)
    native_model = _DeterministicNativeModel(query)
    knowledge_tool = _RecordingKnowledgeTool()
    coordinator = build_default_coordinator(
        router=router,  # type: ignore[arg-type]
        tool_calling_executor=NativeToolCallingSkillExecutor(
            model=native_model,  # type: ignore[arg-type]
            knowledge_tool=knowledge_tool,
            data_tool=_UnusedDataTool(),
        ),
        inventory_risk_synthesizer=_UnusedSynthesizer(),
    )
    executor = build_formal_request_executor(
        coordinator=coordinator,
        memory=FormalMemoryConfiguration.disabled(),
    )
    settings = Settings(
        app_name="Module 6C-1 HTTP Acceptance",
        environment=Environment.TEST,
        required_dependencies=[],
        _env_file=None,
    )
    app = create_app(settings, formal_request_executor=executor)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/agent/execute",
            json={"request_id": "m6c1-http-knowledge", "query": query},
        )

    body = response.json()
    trace = body.pop("trace")
    assert response.status_code == 200
    assert body == {
        "request_id": "m6c1-http-knowledge",
        "status": "completed",
        "route": "knowledge",
        "skill": "enterprise-knowledge-qa",
        "answer": "FASTAPI_FORMAL_RUNTIME_MARKER [E1]",
        "citations": ["[E1]"],
        "error_code": None,
        "memory_context_status": "not_requested",
        "memory_persistence_status": "not_requested",
        "memory_summarization_status": "not_requested",
    }
    assert trace["final_status"] == "completed"
    assert trace["request_id"] == "m6c1-http-knowledge"
    assert "session_id" not in body
    assert len(router.calls) == 1
    assert router.calls[0][0] == query
    assert all(item.kind is not ContextKind.CONVERSATION_MEMORY for item in router.calls[0][1])
    assert knowledge_tool.queries == [query]
    assert len(native_model.calls) == 2
    public_body = response.text.lower()
    assert not any(
        forbidden in public_body
        for forbidden in ("system prompt", "traceback", "secret", "mysql://", "https://")
    )
