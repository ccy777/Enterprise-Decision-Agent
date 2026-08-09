"""Offline HTTP acceptance for app-local runtime bootstrap and lifespan ownership."""

from __future__ import annotations

import json
from contextlib import AsyncExitStack
from typing import Any

import pytest
from fastapi.testclient import TestClient

from decision_agent.api.runtime import create_bootstrapped_app
from decision_agent.application import (
    FormalMemoryConfiguration,
    FormalRequestExecutor,
    build_formal_request_executor,
)
from decision_agent.config import Environment, Settings
from decision_agent.context.models import ContextItem
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
        self.query = query
        self.calls = 0

    async def route_with_context(
        self,
        *,
        user_query: str,
        selected_items: tuple[ContextItem, ...],
    ) -> RouterDecision:
        del user_query, selected_items
        self.calls += 1
        return RouterDecision(
            route=RequestRoute.KNOWLEDGE,
            normalized_query=self.query,
            decision_reason="deterministic_module_6c2a_bootstrap_acceptance",
            knowledge_subquery=self.query,
            confidence=1,
        )


class _DeterministicNativeModel:
    def __init__(self, query: str) -> None:
        self.query = query
        self.calls = 0

    async def complete(
        self,
        *,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
        tool_choice: str,
        response_format: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        del tools, response_format
        self.calls += 1
        if tool_choice == "required":
            return {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "module-6c2a-knowledge-call",
                                    "type": "function",
                                    "function": {
                                        "name": "run_knowledge_agent",
                                        "arguments": json.dumps({"query": self.query}),
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


class _KnowledgeTool:
    def __init__(self, answer_marker: str) -> None:
        self.answer_marker = answer_marker
        self.queries: list[str] = []

    async def run(self, *, query: str) -> AgentToolResult:
        self.queries.append(query)
        return AgentToolResult(
            status="succeeded",
            answer=f"{self.answer_marker} [E1]",
            citations=["[E1]"],
        )


class _UnusedDataTool:
    async def run(self, *, query: str) -> AgentToolResult:
        raise AssertionError(f"Knowledge bootstrap E2E must not call Data: {query}")


class _UnusedSynthesizer:
    async def synthesize(
        self,
        input_data: InventoryRiskSynthesisInput,
    ) -> InventoryRiskSynthesisResult:
        raise AssertionError(f"Knowledge bootstrap E2E must not synthesize: {input_data!r}")


class _RecordingResource:
    def __init__(self, marker: str) -> None:
        self.marker = marker
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1


class _FormalRuntimeBuilder:
    def __init__(self, *, query: str, answer_marker: str) -> None:
        self.calls = 0
        self.router = _DeterministicRouter(query)
        self.model = _DeterministicNativeModel(query)
        self.knowledge_tool = _KnowledgeTool(answer_marker)
        self.resource = _RecordingResource(answer_marker)

    async def __call__(self, stack: AsyncExitStack) -> FormalRequestExecutor:
        self.calls += 1
        stack.push_async_callback(self.resource.aclose)
        coordinator = build_default_coordinator(
            router=self.router,  # type: ignore[arg-type]
            tool_calling_executor=NativeToolCallingSkillExecutor(
                model=self.model,
                knowledge_tool=self.knowledge_tool,
                data_tool=_UnusedDataTool(),
            ),
            inventory_risk_synthesizer=_UnusedSynthesizer(),
        )
        return build_formal_request_executor(
            coordinator=coordinator,
            memory=FormalMemoryConfiguration.disabled(),
        )


class _FailingBuilder:
    def __init__(self) -> None:
        self.calls = 0
        self.resource = _RecordingResource("failed-startup-resource")

    async def __call__(self, stack: AsyncExitStack) -> FormalRequestExecutor:
        self.calls += 1
        stack.push_async_callback(self.resource.aclose)
        raise RuntimeError("SECRET_BOOTSTRAP_MARKER https://private.example token=should-not-leak")


def _settings(name: str, *, required_dependencies: list[str] | None = None) -> Settings:
    return Settings(
        app_name=name,
        environment=Environment.TEST,
        required_dependencies=required_dependencies or [],
        _env_file=None,
    )


def _request(request_id: str, query: str) -> dict[str, str]:
    return {"request_id": request_id, "query": query}


def test_bootstrapped_fastapi_app_publishes_formal_runtime_during_lifespan() -> None:
    query = "说明库存补货制度"
    builder = _FormalRuntimeBuilder(
        query=query,
        answer_marker="BOOTSTRAPPED_FORMAL_RUNTIME_MARKER",
    )
    app = create_bootstrapped_app(
        _settings("6C-2A Success", required_dependencies=["metadata-store"]),
        builder,
        {"metadata-store": lambda: True},
    )

    assert builder.calls == 0
    with TestClient(app) as client:
        health = client.get("/health")
        ready = client.get("/ready")
        response = client.post(
            "/api/v1/agent/execute",
            json=_request("m6c2a-success-1", query),
        )

        assert health.status_code == 200
        assert ready.status_code == 200
        assert ready.json() == {
            "status": "ready",
            "dependencies": {"metadata-store": True, "agent_runtime": True},
        }
        assert response.status_code == 200
        body = response.json()
        trace = body.pop("trace")
        assert body == {
            "request_id": "m6c2a-success-1",
            "status": "completed",
            "route": "knowledge",
            "skill": "enterprise-knowledge-qa",
            "answer": "BOOTSTRAPPED_FORMAL_RUNTIME_MARKER [E1]",
            "citations": ["[E1]"],
            "error_code": None,
            "memory_context_status": "not_requested",
            "memory_persistence_status": "not_requested",
            "memory_summarization_status": "not_requested",
        }
        assert trace["final_status"] == "completed"
        assert trace["request_id"] == "m6c2a-success-1"
        assert builder.resource.close_calls == 0

    assert builder.calls == 1
    assert builder.router.calls == 1
    assert builder.model.calls == 2
    assert builder.knowledge_tool.queries == [query]
    assert builder.resource.close_calls == 1


def test_bootstrapped_fastapi_app_stays_unready_after_startup_failure() -> None:
    builder = _FailingBuilder()
    app = create_bootstrapped_app(_settings("6C-2A Failed"), builder)

    with TestClient(app) as client:
        health = client.get("/health")
        ready = client.get("/ready")
        response = client.post(
            "/api/v1/agent/execute",
            json=_request("m6c2a-failed-1", "合法问题"),
        )

    assert health.status_code == 200
    assert ready.status_code == 503
    assert ready.json() == {
        "status": "not_ready",
        "dependencies": {"agent_runtime": False},
    }
    assert response.status_code == 503
    assert response.json() == {
        "code": "runtime_unavailable",
        "message": "The Agent runtime is unavailable.",
    }
    public_text = f"{health.text} {ready.text} {response.text}".lower()
    assert all(
        marker not in public_text
        for marker in ("secret_bootstrap_marker", "private.example", "token")
    )
    assert builder.calls == 1
    assert builder.resource.close_calls == 1


def test_bootstrapped_fastapi_apps_isolate_runtime_lifecycle() -> None:
    first_query, second_query = "第一个应用问题", "第二个应用问题"
    first = _FormalRuntimeBuilder(query=first_query, answer_marker="FIRST_APP_RUNTIME")
    second = _FormalRuntimeBuilder(query=second_query, answer_marker="SECOND_APP_RUNTIME")
    first_app = create_bootstrapped_app(_settings("6C-2A First"), first)
    second_app = create_bootstrapped_app(_settings("6C-2A Second"), second)

    with TestClient(second_app) as second_client:
        with TestClient(first_app) as first_client:
            first_response = first_client.post(
                "/api/v1/agent/execute",
                json=_request("m6c2a-first", first_query),
            )
            second_response = second_client.post(
                "/api/v1/agent/execute",
                json=_request("m6c2a-second", second_query),
            )

            assert first_client.get("/ready").status_code == 200
            assert second_client.get("/ready").status_code == 200
            assert first_response.json()["answer"] == "FIRST_APP_RUNTIME [E1]"
            assert second_response.json()["answer"] == "SECOND_APP_RUNTIME [E1]"
            assert (first.resource.close_calls, second.resource.close_calls) == (0, 0)

        assert first.resource.close_calls == 1
        assert second.resource.close_calls == 0
        still_ready = second_client.get("/ready")
        still_usable = second_client.post(
            "/api/v1/agent/execute",
            json=_request("m6c2a-second-after-first-close", second_query),
        )
        assert still_ready.status_code == 200
        assert still_usable.json()["answer"] == "SECOND_APP_RUNTIME [E1]"

    assert first.calls == 1 and second.calls == 1
    assert first.resource.close_calls == 1
    assert second.resource.close_calls == 1
