"""Offline HTTP acceptance for the production-configured runtime builder."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from decision_agent.agents.answerability_reviewer import AnswerabilityDecision
from decision_agent.agents.evidence_selector import EvidenceSelection
from decision_agent.agents.grounded_answer import AnswerDraft
from decision_agent.api.runtime import create_bootstrapped_app
from decision_agent.application.configured_runtime import (
    ConfiguredProviderAdapters,
    ConfiguredRuntimeDependencies,
    create_configured_runtime_builder,
)
from decision_agent.config import Environment, Settings
from decision_agent.context.models import ContextItem
from decision_agent.retrieval.evidence_context import (
    EvidenceContext,
    EvidenceItem,
    EvidenceReference,
)
from decision_agent.retrieval.parent_expansion import MatchedChild
from decision_agent.routing.models import RequestRoute, RouterDecision

pytestmark = pytest.mark.e2e

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _isolate_settings_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for field_name in Settings.model_fields:
        monkeypatch.delenv(f"DECISION_AGENT_{field_name.upper()}", raising=False)


def _settings() -> Settings:
    return Settings(
        app_name="Configured Runtime E2E",
        environment=Environment.TEST,
        llm_api_key="e2e-provider-secret",
        llm_base_url="https://provider.invalid/v1",
        llm_model_name="e2e-model",
        knowledge_dataset_root=ROOT / "datasets/enterprise_kb/m2c1",
        db_readonly_password="e2e-db-secret",
        memory_mode="disabled",
        _env_file=None,
    )


def _evidence_context() -> EvidenceContext:
    content = "Product A battery warranty is twelve months."
    child = MatchedChild(
        child_id="CHILD-1",
        parent_id="PARENT-1",
        document_id="DOC-1",
        content=content,
        upstream_rank=1,
    )
    item = EvidenceItem(
        evidence_id="E1",
        final_rank=1,
        parent_id="PARENT-1",
        document_id="DOC-1",
        content=content,
        original_content_length=len(content),
        included_content_length=len(content),
        truncated=False,
        matched_child_count=1,
        best_child_rank=1,
        matched_children=(child,),
    )
    reference = EvidenceReference(
        evidence_id="E1",
        parent_id="PARENT-1",
        document_id="DOC-1",
        source="policies/warranty.md",
        start_offset=0,
        end_offset=len(content),
    )
    return EvidenceContext(
        rendered_context=f"[E1]\n{content}",
        evidence_items=(item,),
        references=(reference,),
        included_evidence_count=1,
        omitted_evidence_count=0,
        total_original_chars=len(content),
        total_included_chars=len(content),
        truncated=False,
    )


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
            decision_reason="deterministic_configured_runtime_e2e",
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
                                    "id": "configured-knowledge-call",
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

    async def complete_chat(self, **_: object) -> dict[str, Any]:
        raise AssertionError("Knowledge E2E must not execute Mixed or Summary")


class _KnowledgePipeline:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def retrieve(self, query: str) -> object:
        self.queries.append(query)
        return SimpleNamespace(evidence_context=_evidence_context())


class _Selector:
    def __init__(self) -> None:
        self.calls = 0

    async def select(self, **_: object) -> EvidenceSelection:
        self.calls += 1
        return EvidenceSelection(
            selected_evidence_ids=["[E1]"],
            selection_reason="The evidence directly supports the answer.",
        )


class _Reviewer:
    def __init__(self) -> None:
        self.calls = 0

    async def review(self, **_: object) -> AnswerabilityDecision:
        self.calls += 1
        return AnswerabilityDecision(
            answerability="answerable",
            missing_information=None,
            decision_reason="选中的证据明确支持答案。",
        )


class _KnowledgeGenerator:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, **_: object) -> AnswerDraft:
        self.calls += 1
        return AnswerDraft(
            answer="Product A battery warranty is twelve months. [E1]",
            citations=["[E1]"],
        )


class _UnusedDataProvider:
    async def plan(self, **_: object) -> object:
        raise AssertionError("Knowledge E2E must not plan Data")

    async def generate(self, **_: object) -> object:
        raise AssertionError("Knowledge E2E must not generate Data")


class _KnowledgeRuntime:
    def __init__(self, *, fail_initialize: bool = False) -> None:
        self.pipeline = _KnowledgePipeline()
        self.fail_initialize = fail_initialize
        self.initialize_calls = 0
        self.close_calls = 0

    async def initialize(self) -> None:
        self.initialize_calls += 1
        if self.fail_initialize:
            raise RuntimeError(
                "PRIVATE_KNOWLEDGE_STARTUP_MARKER https://private.invalid PRIVATE_DATASET_MARKER"
            )

    async def aclose(self) -> None:
        self.close_calls += 1


class _MCPClient:
    def __init__(self) -> None:
        self.enter_calls = 0
        self.exit_calls = 0

    async def __aenter__(self) -> _MCPClient:
        self.enter_calls += 1
        return self

    async def __aexit__(self, *_: object) -> None:
        self.exit_calls += 1

    async def get_enterprise_schema(self) -> object:
        raise AssertionError("Knowledge E2E must not call Data")

    async def get_business_definitions(self) -> object:
        raise AssertionError("Knowledge E2E must not call Data")

    async def execute_safe_query(self, _: str) -> object:
        raise AssertionError("Knowledge E2E must not execute SQL")


class _ConfiguredBoundaries:
    def __init__(self, query: str, *, fail_initialize: bool = False) -> None:
        self.router = _DeterministicRouter(query)
        self.native = _DeterministicNativeModel(query)
        self.selector = _Selector()
        self.reviewer = _Reviewer()
        self.generator = _KnowledgeGenerator()
        self.data = _UnusedDataProvider()
        self.knowledge = _KnowledgeRuntime(fail_initialize=fail_initialize)
        self.mcp_clients: list[_MCPClient] = []
        self.provider_calls = 0
        self.knowledge_calls = 0

    def providers(self, _: Settings) -> ConfiguredProviderAdapters:
        self.provider_calls += 1
        return ConfiguredProviderAdapters(
            router=self.router,  # type: ignore[arg-type]
            native_model=self.native,  # type: ignore[arg-type]
            evidence_selector=self.selector,
            answerability_reviewer=self.reviewer,
            answer_generator=self.generator,
            data_query_planner=self.data,  # type: ignore[arg-type]
            data_answer_generator=self.data,  # type: ignore[arg-type]
        )

    def knowledge_runtime(self, _: Settings) -> _KnowledgeRuntime:
        self.knowledge_calls += 1
        return self.knowledge

    def mcp(self, _: Settings) -> _MCPClient:
        client = _MCPClient()
        self.mcp_clients.append(client)
        return client

    def redis(self, _: str, __: float) -> object:
        raise AssertionError("Disabled-memory E2E must not create Redis")

    def dependencies(self) -> ConfiguredRuntimeDependencies:
        return ConfiguredRuntimeDependencies(
            provider_adapters_factory=self.providers,
            knowledge_runtime_factory=self.knowledge_runtime,  # type: ignore[arg-type]
            enterprise_data_client_factory=self.mcp,
            redis_client_factory=self.redis,  # type: ignore[arg-type]
        )


def test_configured_fastapi_success_runs_formal_knowledge_chain_and_shutdown() -> None:
    query = "What is the Product A battery warranty?"
    boundaries = _ConfiguredBoundaries(query)
    settings = _settings()
    builder = create_configured_runtime_builder(settings, boundaries.dependencies())
    app = create_bootstrapped_app(settings, builder)

    assert boundaries.provider_calls == boundaries.knowledge_calls == 0
    with TestClient(app) as client:
        health = client.get("/health")
        ready = client.get("/ready")
        response = client.post(
            "/api/v1/agent/execute",
            json={"request_id": "configured-e2e-success", "query": query},
        )

        assert health.status_code == 200
        assert ready.status_code == 200
        assert ready.json() == {
            "status": "ready",
            "dependencies": {"agent_runtime": True},
        }
        assert response.status_code == 200
        body = response.json()
        trace = body.pop("trace")
        assert body == {
            "request_id": "configured-e2e-success",
            "status": "completed",
            "route": "knowledge",
            "skill": "enterprise-knowledge-qa",
            "answer": "Product A battery warranty is twelve months. [E1]",
            "citations": ["[E1]"],
            "error_code": None,
            "memory_context_status": "not_requested",
            "memory_persistence_status": "not_requested",
            "memory_summarization_status": "not_requested",
        }
        assert trace["final_status"] == "completed"
        assert trace["request_id"] == "configured-e2e-success"
        assert boundaries.knowledge.close_calls == 0

    assert boundaries.provider_calls == boundaries.knowledge_calls == 1
    assert boundaries.knowledge.initialize_calls == boundaries.knowledge.close_calls == 1
    assert len(boundaries.mcp_clients) == 1
    assert (
        boundaries.mcp_clients[0].enter_calls,
        boundaries.mcp_clients[0].exit_calls,
    ) == (1, 1)
    assert boundaries.router.calls == 1
    assert boundaries.native.calls == 2
    assert boundaries.knowledge.pipeline.queries == [query]
    assert boundaries.selector.calls == boundaries.reviewer.calls == boundaries.generator.calls == 1


def test_configured_fastapi_startup_failure_stays_fail_closed_and_rolls_back() -> None:
    query = "What is the Product A battery warranty?"
    boundaries = _ConfiguredBoundaries(query, fail_initialize=True)
    settings = _settings()
    app = create_bootstrapped_app(
        settings,
        create_configured_runtime_builder(settings, boundaries.dependencies()),
    )

    with TestClient(app) as client:
        health = client.get("/health")
        ready = client.get("/ready")
        response = client.post(
            "/api/v1/agent/execute",
            json={"request_id": "configured-e2e-failed", "query": query},
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
        for marker in (
            "private_knowledge_startup_marker",
            "private.invalid",
            "private_dataset_marker",
            "e2e-provider-secret",
            "e2e-db-secret",
        )
    )
    assert boundaries.knowledge.initialize_calls == boundaries.knowledge.close_calls == 1
    assert boundaries.mcp_clients == []
    assert boundaries.router.calls == boundaries.native.calls == 0
