"""Offline M8B closed-loop coverage over real production orchestration.

Only external provider, vector/reranker, database, and MCP-stdio transport ports are
deterministic here.  The test intentionally runs the production Router, Coordinator,
controlled workflow, Skills, Knowledge/Data graphs, retrieval pipeline, SafeQuery,
MCP client mapping, and request trace path.
"""

# ruff: noqa: RUF001

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from decision_agent.agent_workflow import ControlledAgentWorkflow, ControlledWorkflowPolicy
from decision_agent.agent_workflow.providers import (
    OpenAICompatibleWorkflowPlanner,
    OpenAICompatibleWorkflowReviewer,
)
from decision_agent.agents.answerability_reviewer import OpenAICompatibleAnswerabilityReviewer
from decision_agent.agents.data_answer_generator import OpenAICompatibleDataAnswerGenerator
from decision_agent.agents.data_query_planner import OpenAICompatibleDataQueryPlanner
from decision_agent.agents.evidence_selector import OpenAICompatibleEvidenceSelector
from decision_agent.agents.grounded_answer import OpenAICompatibleAnswerGenerator
from decision_agent.application.executor import FormalRequestExecutor
from decision_agent.application.models import FormalRequest
from decision_agent.context.conversation_memory import ConversationMemoryProjector
from decision_agent.coordination.factory import build_default_coordinator
from decision_agent.data.executor import QueryExecutor
from decision_agent.data.models import QueryExecution
from decision_agent.data.safe_query_service import SafeQueryService
from decision_agent.data.sql_guard import SQLGuard
from decision_agent.mcp_client.enterprise_data_client import EnterpriseDataMCPClient
from decision_agent.mcp_server.contracts import ExecuteSafeQueryInput
from decision_agent.mcp_server.enterprise_data_server import EnterpriseDataToolService
from decision_agent.memory import InMemorySessionMemoryStore
from decision_agent.observability import (
    BestEffortTraceDispatcher,
    InMemoryTraceSink,
    TraceStage,
)
from decision_agent.retrieval import (
    DeterministicHashEmbeddingProvider,
    EnterpriseRetrievalPipeline,
    InMemoryVectorStore,
    RetrievalPipelineConfig,
)
from decision_agent.retrieval.reranking import RerankCandidate, RerankedResult
from decision_agent.routing.request_router import OpenAICompatibleRequestRouter
from decision_agent.skills.inventory_risk_synthesizer import (
    OpenAICompatibleInventoryRiskSynthesizer,
)
from decision_agent.skills.native_runtime import NativeToolCallingSkillExecutor
from decision_agent.tool_calling.runtime import OpenAICompatibleNativeToolCallingModel
from decision_agent.tool_calling.tools import DataAgentTool, KnowledgeAgentTool
from decision_agent.workflows.knowledge_qa import build_knowledge_qa_graph

_USER_QUERY = "请结合库存数据和库存制度给出补货建议"
_DATA_QUERY = "查询当前库存与安全库存"
_KNOWLEDGE_QUERY = "查询库存预警与补货制度"


class CountingEmbedding:
    """The deterministic embedding is the substituted external model port only."""

    def __init__(self) -> None:
        self._inner = DeterministicHashEmbeddingProvider(dimension=16)
        self.document_calls = 0
        self.query_calls = 0

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    async def initialize(self) -> None:
        return None

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_calls += 1
        return await self._inner.embed_documents(texts)

    async def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return await self._inner.embed_query(text)


class CountingReranker:
    """A fixed cross-encoder endpoint response while production reranking stays in the pipeline."""

    def __init__(self) -> None:
        self.calls = 0

    async def initialize(self) -> None:
        return None

    async def rerank(
        self, query: str, candidates: Sequence[RerankCandidate], *, top_k: int | None = None
    ) -> list[RerankedResult]:
        del query
        self.calls += 1
        limit = len(candidates) if top_k is None else min(len(candidates), top_k)
        return [
            RerankedResult(
                final_rank=index,
                candidate_id=candidate.candidate_id,
                record_id=candidate.record_id,
                document_id=candidate.document_id,
                content=candidate.content,
                reranker_score=float(limit - index + 1),
                upstream_rank=candidate.upstream_rank,
                upstream_score=candidate.upstream_score,
                metadata=candidate.model_copy(deep=True).metadata,
                provenance=candidate.model_copy(deep=True).provenance,
            )
            for index, candidate in enumerate(candidates[:limit], start=1)
        ]


class CountingGuard(SQLGuard):
    """Instrumentation around the actual SQLGuard implementation."""

    def __init__(self) -> None:
        super().__init__(max_rows=20)
        self.calls = 0

    def validate(self, sql: str):  # type: ignore[no-untyped-def]
        self.calls += 1
        return super().validate(sql)


class CountingDatabase(QueryExecutor):
    """The sole database I/O replacement; it receives SQL already approved by SQLGuard."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def execute(self, *, normalized_sql: str, timeout_ms: int) -> QueryExecution:
        self.calls.append((normalized_sql, timeout_ms))
        return QueryExecution(
            columns=["product_id", "on_hand_quantity", "safety_stock"],
            rows=[("P-TEST-1", 8, 20)],
        )


class _AsyncContext:
    def __init__(self, value: object) -> None:
        self._value = value

    async def __aenter__(self) -> object:
        return self._value

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None


class OfflineMCPSession:
    """A stdio transport substitute which delegates to the real MCP tool adapter."""

    def __init__(self, tools: EnterpriseDataToolService) -> None:
        self._tools = tools
        self.transport_calls = 0

    async def initialize(self) -> None:
        return None

    async def list_tools(self) -> SimpleNamespace:
        return SimpleNamespace(
            tools=[
                SimpleNamespace(name="get_enterprise_schema"),
                SimpleNamespace(name="get_business_definitions"),
                SimpleNamespace(name="execute_safe_query"),
            ]
        )

    async def call_tool(self, name: str, arguments: dict[str, object]) -> SimpleNamespace:
        self.transport_calls += 1
        if name == "get_enterprise_schema":
            payload = (await self._tools.get_enterprise_schema()).tables
        elif name == "get_business_definitions":
            payload = (await self._tools.get_business_definitions()).definitions
        elif name == "execute_safe_query":
            payload = (
                await self._tools.execute_safe_query(
                    ExecuteSafeQueryInput(sql=str(arguments["sql"]))
                )
            ).model_dump()
        else:  # pragma: no cover - EnterpriseDataMCPClient has a fixed required-tool surface.
            raise AssertionError(f"unexpected MCP tool: {name}")
        return SimpleNamespace(isError=False, structuredContent=payload, content=[])


class ScriptedProviderTransport:
    """Deterministic HTTP/chat boundary used by the real production adapters."""

    def __init__(self, *, review_outcome: str) -> None:
        self.review_outcome = review_outcome
        self.router_calls = 0
        self.planner_calls = 0
        self.workflow_reviewer_calls = 0
        self.native_tool_calls = 0
        self.native_tool_queries: list[str] = []
        self.inventory_synthesis_calls = 0

    def router_post(self, user_query: str, messages: object = None) -> dict[str, Any]:
        del user_query, messages
        self.router_calls += 1
        return _completion(
            {
                "route": "mixed",
                "normalized_query": _USER_QUERY,
                "decision_reason": "需要同时核对库存数据和制度。",
                "knowledge_subquery": _KNOWLEDGE_QUERY,
                "data_subquery": _DATA_QUERY,
                "missing_information": None,
                "confidence": 0.95,
            }
        )

    def native_post(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None,
        tool_choice: str | None,
        response_format: dict[str, str] | None,
    ) -> dict[str, Any]:
        del tool_choice, response_format
        if tools:
            self.native_tool_calls += 1
            name = str(tools[0]["function"]["name"])
            query = _KNOWLEDGE_QUERY if name == "run_knowledge_agent" else _DATA_QUERY
            self.native_tool_queries.append(query)
            return {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"offline-{name}",
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": json.dumps({"query": query}),
                                    },
                                }
                            ],
                        },
                    }
                ]
            }
        tool_message = next(message for message in messages if message.get("role") == "tool")
        result = json.loads(str(tool_message["content"]))
        return _completion({"answer": result["answer"], "citations": result["citations"]})

    def chat_post(
        self,
        messages: list[dict[str, object]],
        tools: object,
        tool_choice: object,
        response_format: dict[str, str] | None,
    ) -> dict[str, Any]:
        del tools, tool_choice, response_format
        system = str(messages[0]["content"])
        if "execution-plan schema" in system:
            self.planner_calls += 1
            return _completion(_valid_plan())
        if "reviewer-decision schema" in system:
            self.workflow_reviewer_calls += 1
            return _completion(_workflow_review(self.review_outcome))
        if "risk_summary" in system:
            self.inventory_synthesis_calls += 1
            return _completion(
                {
                    "risk_summary": "当前库存低于安全库存。[D1]",
                    "policy_basis": "库存制度要求触发补货评估。[E1]",
                    "recommended_actions": ["执行补货评估。"],
                    "citations": ["[D1]", "[E1]"],
                }
            )
        raise AssertionError("unexpected production chat adapter request")


class OfflineRuntime:
    def __init__(
        self,
        *,
        executor: FormalRequestExecutor,
        transport: ScriptedProviderTransport,
        pipeline: EnterpriseRetrievalPipeline,
        embedding: CountingEmbedding,
        reranker: CountingReranker,
        guard: CountingGuard,
        database: CountingDatabase,
        mcp_session: OfflineMCPSession,
        sink: InMemoryTraceSink,
        memory_store: InMemorySessionMemoryStore,
        data_tool: DataAgentTool,
    ) -> None:
        self.executor = executor
        self.transport = transport
        self.pipeline = pipeline
        self.embedding = embedding
        self.reranker = reranker
        self.guard = guard
        self.database = database
        self.mcp_session = mcp_session
        self.sink = sink
        self.memory_store = memory_store
        self.data_tool = data_tool


async def build_offline_controlled_inventory_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    review_outcome: str = "accept",
    controlled_workflow_enabled: bool = True,
) -> OfflineRuntime:
    """Compose real production classes with deterministic lowest-level I/O ports."""
    dataset_root = _write_inventory_dataset(tmp_path)
    embedding = CountingEmbedding()
    reranker = CountingReranker()
    pipeline = EnterpriseRetrievalPipeline(
        dataset_root=dataset_root,
        embedding_provider=embedding,
        vector_store=InMemoryVectorStore(dimension=embedding.dimension),
        reranker=reranker,
        config=RetrievalPipelineConfig(
            dense_top_k=1,
            bm25_top_k=1,
            rrf_top_k=1,
            reranker_candidate_count=1,
            reranker_top_k=1,
            parent_top_k=1,
            evidence_max_count=1,
        ),
    )
    await pipeline.initialize()

    transport = ScriptedProviderTransport(review_outcome=review_outcome)
    native_model = OpenAICompatibleNativeToolCallingModel(
        api_key="offline-test-key",
        base_url="https://offline.invalid",
        model_name="offline-test-model",
        timeout_seconds=1,
    )
    native_model._post = transport.native_post  # type: ignore[method-assign]
    native_model.complete_chat = (  # type: ignore[method-assign]
        lambda *, messages, response_format: _complete_via_transport(
            transport, messages, response_format
        )
    )

    router = OpenAICompatibleRequestRouter(
        api_key="offline-test-key",
        base_url="https://offline.invalid",
        model_name="offline-test-model",
        timeout_seconds=1,
    )
    router._post = transport.router_post  # type: ignore[method-assign]

    selector = OpenAICompatibleEvidenceSelector(
        api_key="offline-test-key",
        base_url="https://offline.invalid",
        model_name="offline",
        timeout_seconds=1,
    )
    selector._post = lambda query, context: _completion(  # type: ignore[method-assign]
        {"selected_evidence_ids": ["[E1]"], "selection_reason": "库存制度直接相关。"}
    )
    knowledge_reviewer = OpenAICompatibleAnswerabilityReviewer(
        api_key="offline-test-key",
        base_url="https://offline.invalid",
        model_name="offline",
        timeout_seconds=1,
    )
    knowledge_reviewer._post = lambda query, context: _completion(  # type: ignore[method-assign]
        {
            "answerability": "answerable",
            "missing_information": None,
            "decision_reason": "选中证据直接说明库存制度。",
        }
    )
    knowledge_answer = OpenAICompatibleAnswerGenerator(
        api_key="offline-test-key",
        base_url="https://offline.invalid",
        model_name="offline",
        timeout_seconds=1,
    )
    knowledge_answer._post = lambda *args: _completion(  # type: ignore[method-assign]
        {"answer": "库存低于安全库存时需要补货评估。[E1]", "citations": ["[E1]"]}
    )
    knowledge_tool = KnowledgeAgentTool(
        graph=build_knowledge_qa_graph(
            retrieval_pipeline=pipeline,
            evidence_selector=selector,
            answerability_reviewer=knowledge_reviewer,
            answer_generator=knowledge_answer,
        )
    )

    guard = CountingGuard()
    database = CountingDatabase()
    service = SafeQueryService(
        guard=guard,
        executor=database,
        query_timeout_seconds=1,
        max_rows=20,
        max_result_cells=100,
    )
    mcp_session = OfflineMCPSession(EnterpriseDataToolService(lambda: service))
    monkeypatch.setattr(
        "decision_agent.mcp_client.enterprise_data_client.stdio_client",
        lambda parameters: _AsyncContext((object(), object())),
    )
    monkeypatch.setattr(
        "decision_agent.mcp_client.enterprise_data_client.ClientSession",
        lambda reader, writer: _AsyncContext(mcp_session),
    )
    data_planner = OpenAICompatibleDataQueryPlanner(
        api_key="offline-test-key",
        base_url="https://offline.invalid",
        model_name="offline",
        timeout_seconds=1,
    )
    data_planner._post = lambda query, schema, definitions: _completion(  # type: ignore[method-assign]
        {
            "status": "ready",
            "intent": "查询库存状态",
            "sql": "SELECT product_id, on_hand_quantity FROM inventory_snapshots",
            "decision_reason": "库存字段已授权。",
            "missing_information": None,
        }
    )
    data_answer = OpenAICompatibleDataAnswerGenerator(
        api_key="offline-test-key",
        base_url="https://offline.invalid",
        model_name="offline",
        timeout_seconds=1,
    )
    data_answer._post = lambda query, evidence: _completion(  # type: ignore[method-assign]
        {"answer": "当前库存为 8，安全库存为 20。[D1]", "citations": ["[D1]"]}
    )
    data_tool = DataAgentTool(
        planner=data_planner,
        enterprise_data_client_factory=lambda: EnterpriseDataMCPClient(timeout_seconds=1),
        answer_generator=data_answer,
    )
    native_executor = NativeToolCallingSkillExecutor(
        model=native_model, knowledge_tool=knowledge_tool, data_tool=data_tool
    )
    synthesizer = OpenAICompatibleInventoryRiskSynthesizer(client=native_model)

    def controlled_builder(registry):  # type: ignore[no-untyped-def]
        return ControlledAgentWorkflow(
            planner=OpenAICompatibleWorkflowPlanner(provider=native_model),
            reviewer=OpenAICompatibleWorkflowReviewer(provider=native_model),
            registry=registry,
            policy=ControlledWorkflowPolicy(enabled=controlled_workflow_enabled),
        )

    coordinator = build_default_coordinator(
        router=router,
        tool_calling_executor=native_executor,
        inventory_risk_synthesizer=synthesizer,
        controlled_workflow_builder=controlled_builder if controlled_workflow_enabled else None,
    )
    sink = InMemoryTraceSink()
    memory_store = InMemorySessionMemoryStore()
    return OfflineRuntime(
        executor=FormalRequestExecutor(
            coordinator=coordinator,
            memory_store=memory_store,
            memory_projector=ConversationMemoryProjector(),
            trace_dispatcher=BestEffortTraceDispatcher((sink,)),
        ),
        transport=transport,
        pipeline=pipeline,
        embedding=embedding,
        reranker=reranker,
        guard=guard,
        database=database,
        mcp_session=mcp_session,
        sink=sink,
        memory_store=memory_store,
        data_tool=data_tool,
    )


@pytest.mark.asyncio
async def test_accept_runs_real_offline_controlled_inventory_closed_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = await build_offline_controlled_inventory_runtime(tmp_path, monkeypatch)

    response = await runtime.executor.execute(
        FormalRequest(
            request_id="offline-accept", user_query=_USER_QUERY, session_id="session-accept"
        )
    )

    assert response.result.status.value == "completed", (
        response.result.error_code,
        [
            (span.stage.value, span.operation, span.error_code)
            for span in runtime.sink.snapshot()[0].spans
            if span.error_code is not None
        ],
    )
    assert response.result.route.value == "mixed"
    assert response.result.skill_name == "inventory-risk-diagnosis"
    assert (
        response.result.answer is not None
        and "[D1]" in response.result.answer
        and "[E1]" in response.result.answer
    )
    assert runtime.transport.router_calls == runtime.transport.planner_calls == 1
    assert (
        runtime.transport.workflow_reviewer_calls
        == runtime.transport.inventory_synthesis_calls
        == 1
    )
    assert runtime.transport.native_tool_calls == 2
    assert runtime.transport.native_tool_queries == [_DATA_QUERY, _KNOWLEDGE_QUERY]
    assert (
        runtime.embedding.document_calls
        == runtime.embedding.query_calls
        == runtime.reranker.calls
        == 1
    )
    assert runtime.guard.calls == len(runtime.database.calls) == 1
    assert runtime.mcp_session.transport_calls == 3
    assert runtime.database.calls[0][0].endswith("LIMIT 20")
    memory = runtime.memory_store.read("session-accept")
    assert len(memory.turns) == 1 and memory.turns[0].assistant_text == response.result.answer
    assert "plan-1" not in memory.turns[0].assistant_text

    trace = runtime.sink.snapshot()[0]
    _assert_trace_tree(trace)
    rendered_trace = json.dumps(trace.model_dump(mode="json"), ensure_ascii=False)
    for forbidden in (
        _USER_QUERY,
        _DATA_QUERY,
        _KNOWLEDGE_QUERY,
        "SELECT product_id",
        "P-TEST-1",
        "session-accept",
    ):
        assert forbidden not in rendered_trace
    assert response.trace is not None and response.trace.trace_id == trace.trace_id
    await runtime.pipeline.close()


@pytest.mark.asyncio
async def test_repair_is_fail_closed_without_a_second_skill_or_memory_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = await build_offline_controlled_inventory_runtime(
        tmp_path, monkeypatch, review_outcome="repair"
    )

    response = await runtime.executor.execute(
        FormalRequest(
            request_id="offline-repair", user_query=_USER_QUERY, session_id="session-repair"
        )
    )

    assert response.result.status.value == "failed"
    assert response.result.error_code == "workflow_repair_not_permitted"
    assert runtime.transport.planner_calls == runtime.transport.workflow_reviewer_calls == 1
    assert runtime.transport.native_tool_calls == 2
    assert runtime.transport.inventory_synthesis_calls == 1
    assert runtime.memory_store.read("session-repair").turns == ()
    trace = runtime.sink.snapshot()[0]
    assert sum(span.stage is TraceStage.PLAN_STEP_EXECUTION for span in trace.spans) == 1
    assert sum(span.stage is TraceStage.WORKFLOW_REVIEW for span in trace.spans) == 1
    await runtime.pipeline.close()


@pytest.mark.asyncio
async def test_disabled_controlled_workflow_keeps_existing_direct_mixed_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = await build_offline_controlled_inventory_runtime(
        tmp_path, monkeypatch, controlled_workflow_enabled=False
    )

    response = await runtime.executor.execute(
        FormalRequest(request_id="offline-disabled", user_query=_USER_QUERY)
    )

    assert response.result.status.value == "completed", (
        response.result.error_code,
        [
            (span.stage.value, span.operation, span.error_code)
            for span in runtime.sink.snapshot()[0].spans
            if span.error_code is not None
        ],
    )
    assert runtime.transport.router_calls == runtime.transport.inventory_synthesis_calls == 1
    assert runtime.transport.planner_calls == runtime.transport.workflow_reviewer_calls == 0
    assert runtime.transport.native_tool_calls == 2
    trace = runtime.sink.snapshot()[0]
    assert all(
        span.stage
        not in {TraceStage.AGENT_WORKFLOW, TraceStage.PLANNING, TraceStage.WORKFLOW_REVIEW}
        for span in trace.spans
    )
    assert response.trace is not None
    await runtime.pipeline.close()


async def _complete_via_transport(
    transport: ScriptedProviderTransport,
    messages: list[dict[str, object]],
    response_format: dict[str, str],
) -> dict[str, Any]:
    return transport.chat_post(messages, None, None, response_format)


def _completion(content: object) -> dict[str, Any]:
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": json.dumps(content, ensure_ascii=False)},
            }
        ]
    }


def _valid_plan() -> dict[str, object]:
    return {
        "plan_id": "plan-1",
        "plan_version": "m8b-v1",
        "objective_type": "mixed_inventory_diagnosis",
        "steps": [
            {
                "step_id": "step-1",
                "sequence": 1,
                "skill_name": "inventory-risk-diagnosis",
                "objective_type": "mixed_inventory_diagnosis",
                "depends_on": [],
                "required_output_type": "mixed_diagnosis",
                "optional": False,
            }
        ],
        "max_execution_rounds": 1,
        "max_skill_calls": 1,
    }


def _workflow_review(outcome: str) -> dict[str, object]:
    if outcome == "accept":
        return {
            "outcome": "accept",
            "accepted_step_id": "step-1",
            "repair_target": None,
            "reason_code": "evidence_complete",
            "final_status": "accepted",
        }
    return {
        "outcome": "repair",
        "accepted_step_id": None,
        "repair_target": "step-1",
        "reason_code": "repair_requested",
        "final_status": "repair",
    }


def _write_inventory_dataset(tmp_path: Path) -> Path:
    generated = tmp_path / "generated"
    generated.mkdir()
    content = "库存低于安全库存时需要补货评估，并依据库存制度安排补货。"
    common = {
        "schema_version": "1.0",
        "document_id": "DOC-TEST-INV",
        "document_version": "v1",
        "content": content,
        "source": "fixtures/inventory_policy.md",
        "start_offset": 0,
        "end_offset": len(content),
        "metadata": {"category": "inventory"},
        "provenance": {"parser_name": "offline_test"},
    }
    parent = {**common, "chunk_id": "parent-test-1", "block_ids": ["block-test-1"]}
    child = {**common, "chunk_id": "child-test-1", "parent_id": "parent-test-1"}
    (generated / "parent_chunks.jsonl").write_text(
        json.dumps(parent, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (generated / "child_chunks.jsonl").write_text(
        json.dumps(child, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return tmp_path


def _assert_trace_tree(trace) -> None:  # type: ignore[no-untyped-def]
    by_stage = {span.stage: span for span in trace.spans}
    required = {
        TraceStage.COORDINATION,
        TraceStage.AGENT_WORKFLOW,
        TraceStage.PLANNING,
        TraceStage.PLAN_STEP_EXECUTION,
        TraceStage.WORKFLOW_REVIEW,
        TraceStage.RETRIEVAL,
        TraceStage.RERANKING,
        TraceStage.EVIDENCE_SELECTION,
        TraceStage.DATA_ACCESS,
    }
    assert required <= set(by_stage)
    workflow = by_stage[TraceStage.AGENT_WORKFLOW]
    assert by_stage[TraceStage.PLANNING].parent_span_id == workflow.span_id
    assert by_stage[TraceStage.PLAN_STEP_EXECUTION].parent_span_id == workflow.span_id
    assert by_stage[TraceStage.WORKFLOW_REVIEW].parent_span_id == workflow.span_id
    assert all(span.trace_id == trace.trace_id for span in trace.spans)
    assert any(span.operation == "parent_expansion" for span in trace.spans)
    assert any(span.operation == "plan_workflow" for span in trace.spans)
    assert any(span.operation == "review_workflow" for span in trace.spans)
