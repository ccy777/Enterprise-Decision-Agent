"""Offline application acceptance for the unified formal runtime."""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

import pytest

from decision_agent.agents.data_answer_generator import DataAnswerDraft
from decision_agent.agents.data_query_planner import DataPlanStatus, DataQueryPlan
from decision_agent.application import (
    FormalMemoryConfiguration,
    FormalRequest,
    FormalResponse,
    MemoryContextStatus,
    MemoryPersistenceStatus,
    MemorySummarizationStatus,
    build_formal_request_executor,
)
from decision_agent.context.models import ContextItem, ContextKind
from decision_agent.coordination import build_default_coordinator
from decision_agent.coordination.models import CoordinatorStatus
from decision_agent.data.models import SafeQueryRequest
from decision_agent.data.safe_query_service import SafeQueryService
from decision_agent.data.sql_guard import SQLGuard
from decision_agent.mcp_client.contracts import (
    BusinessDefinitions,
    EnterpriseSchema,
    MCPQueryResult,
)
from decision_agent.routing.models import RequestRoute, RouterDecision
from decision_agent.skills.inventory_risk_synthesizer import (
    InventoryRiskSynthesisInput,
    InventoryRiskSynthesisResult,
)
from decision_agent.skills.native_runtime import NativeToolCallingSkillExecutor
from decision_agent.tool_calling.models import AgentToolResult
from decision_agent.tool_calling.tools import DataAgentTool

pytestmark = [pytest.mark.e2e, pytest.mark.offline_integration]

_SAFE_QUERY_REQUEST_ID = UUID("00000000-0000-0000-0000-00000000006b")


class _DeterministicRouter:
    def __init__(self, decision: RouterDecision) -> None:
        self._decision = decision
        self.calls: list[tuple[str, tuple[ContextItem, ...]]] = []

    async def route_with_context(
        self, *, user_query: str, selected_items: tuple[ContextItem, ...]
    ) -> RouterDecision:
        self.calls.append((user_query, selected_items))
        return self._decision


class _DeterministicNativeModel:
    def __init__(self, queries_by_tool: dict[str, str]) -> None:
        self._queries_by_tool = queries_by_tool
        self.calls: list[dict[str, object]] = []
        self.required_tools: list[str] = []

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
            function = tools[0]["function"]
            assert isinstance(function, dict)
            tool_name = function["name"]
            assert isinstance(tool_name, str)
            self.required_tools.append(tool_name)
            query = self._queries_by_tool[tool_name]
            return {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"{tool_name}-call-{len(self.required_tools)}",
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "arguments": json.dumps({"query": query}),
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


class _RecordingAgentTool:
    def __init__(self, result: AgentToolResult) -> None:
        self._result = result
        self.queries: list[str] = []

    async def run(self, *, query: str) -> AgentToolResult:
        self.queries.append(query)
        return self._result


class _DeterministicDataPlanner:
    def __init__(self, *, sql: str) -> None:
        self._sql = sql
        self.calls: list[tuple[str, dict[str, list[str]], dict[str, str]]] = []

    async def plan(
        self,
        *,
        user_query: str,
        enterprise_schema: dict[str, list[str]],
        business_definitions: dict[str, str],
    ) -> DataQueryPlan:
        self.calls.append((user_query, enterprise_schema, business_definitions))
        return DataQueryPlan(
            status=DataPlanStatus.READY,
            intent="查询库存风险",
            sql=self._sql,
            decision_reason="只读查询条件完整。",
        )


class _DeterministicDataAnswerGenerator:
    def __init__(self, answer: str) -> None:
        self._answer = answer
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    async def generate(
        self, *, user_query: str, data_evidence: tuple[object, ...]
    ) -> DataAnswerDraft:
        self.calls.append((user_query, data_evidence))
        return DataAnswerDraft(answer=self._answer, citations=["[D1]"])


class _DeterministicEnterpriseDataClient:
    def __init__(self, *, result: MCPQueryResult | None) -> None:
        self._result = result
        self.enter_calls = 0
        self.exit_calls = 0
        self.schema_calls = 0
        self.definition_calls = 0
        self.queries: list[str] = []

    async def __aenter__(self) -> _DeterministicEnterpriseDataClient:
        self.enter_calls += 1
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.exit_calls += 1

    async def get_enterprise_schema(self) -> EnterpriseSchema:
        self.schema_calls += 1
        return EnterpriseSchema(
            tables={
                "inventory_snapshots": [
                    "product_id",
                    "snapshot_date",
                    "on_hand_quantity",
                ]
            }
        )

    async def get_business_definitions(self) -> BusinessDefinitions:
        self.definition_calls += 1
        return BusinessDefinitions(definitions={"inventory_risk": "库存低于安全库存。"})

    async def execute_safe_query(self, sql: str) -> MCPQueryResult:
        self.queries.append(sql)
        assert self._result is not None
        return self._result


class _InProcessSafeQueryClient(_DeterministicEnterpriseDataClient):
    def __init__(self, *, service: SafeQueryService) -> None:
        super().__init__(result=None)
        self._service = service

    async def execute_safe_query(self, sql: str) -> MCPQueryResult:
        self.queries.append(sql)
        result = await self._service.execute(
            SafeQueryRequest(sql=sql, request_id=_SAFE_QUERY_REQUEST_ID)
        )
        return MCPQueryResult(
            columns=result.columns,
            rows=result.rows,
            row_count=result.row_count,
            truncated=result.truncated,
            normalized_sql=result.audit.normalized_sql,
            accessed_tables=result.accessed_tables,
            elapsed_ms=result.elapsed_ms,
            error_code=result.error_code,
        )


class _OneClientFactory:
    def __init__(self, client: object) -> None:
        self._client = client
        self.calls = 0

    def __call__(self) -> object:
        self.calls += 1
        return self._client


class _NeverExecuteDatabase:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, *, normalized_sql: str, timeout_ms: int) -> object:
        self.calls += 1
        raise AssertionError(
            f"SQLGuard must reject before database execution: {normalized_sql!r}, {timeout_ms}"
        )


class _RecordingSynthesizer:
    def __init__(self) -> None:
        self.inputs: list[InventoryRiskSynthesisInput] = []

    async def synthesize(
        self, input_data: InventoryRiskSynthesisInput
    ) -> InventoryRiskSynthesisResult:
        self.inputs.append(input_data)
        return InventoryRiskSynthesisResult(
            risk_summary=f"MIXED_APPLICATION_MARKER {input_data.data_answer}",
            policy_basis=input_data.knowledge_answer,
            recommended_actions=("按制度复核并补货。",),
            citations=("[D1]", "[E1]"),
        )


class _UnusedSynthesizer:
    async def synthesize(self, _: InventoryRiskSynthesisInput) -> InventoryRiskSynthesisResult:
        raise AssertionError("single-domain application path must not invoke mixed synthesis")


def _decision(
    route: RequestRoute,
    *,
    knowledge_subquery: str | None = None,
    data_subquery: str | None = None,
) -> RouterDecision:
    return RouterDecision(
        route=route,
        normalized_query=knowledge_subquery or data_subquery or "库存风险诊断",
        decision_reason="deterministic_module_6b_application_acceptance",
        knowledge_subquery=knowledge_subquery,
        data_subquery=data_subquery,
        confidence=1,
    )


def _data_tool(
    *,
    sql: str,
    client: object,
    answer: str,
) -> tuple[
    DataAgentTool,
    _DeterministicDataPlanner,
    _DeterministicDataAnswerGenerator,
    _OneClientFactory,
]:
    planner = _DeterministicDataPlanner(sql=sql)
    generator = _DeterministicDataAnswerGenerator(answer)
    factory = _OneClientFactory(client)
    return (
        DataAgentTool(
            planner=planner,
            enterprise_data_client_factory=factory,  # type: ignore[arg-type]
            answer_generator=generator,
        ),
        planner,
        generator,
        factory,
    )


def _executor(
    *,
    router: _DeterministicRouter,
    model: _DeterministicNativeModel,
    knowledge_tool: object,
    data_tool: object,
    synthesizer: object,
):
    coordinator = build_default_coordinator(
        router=router,  # type: ignore[arg-type]
        tool_calling_executor=NativeToolCallingSkillExecutor(
            model=model,  # type: ignore[arg-type]
            knowledge_tool=knowledge_tool,  # type: ignore[arg-type]
            data_tool=data_tool,  # type: ignore[arg-type]
        ),
        inventory_risk_synthesizer=synthesizer,  # type: ignore[arg-type]
    )
    return build_formal_request_executor(
        coordinator=coordinator,
        memory=FormalMemoryConfiguration.disabled(),
    )


def _assert_memory_not_requested(response: FormalResponse) -> None:
    assert response.memory_context_status is MemoryContextStatus.NOT_REQUESTED
    assert response.memory_persistence_status is MemoryPersistenceStatus.NOT_REQUESTED
    assert response.memory_summarization_status is MemorySummarizationStatus.NOT_REQUESTED


def _assert_router_used_without_memory(router: _DeterministicRouter) -> None:
    assert len(router.calls) == 1
    assert all(item.kind is not ContextKind.CONVERSATION_MEMORY for item in router.calls[0][1])


def _successful_data_result(sql: str) -> MCPQueryResult:
    return MCPQueryResult(
        columns=["product_id", "on_hand_quantity"],
        rows=[["PRODUCT-6B", 3]],
        row_count=1,
        truncated=False,
        normalized_sql=sql,
        accessed_tables=["inventory_snapshots"],
        elapsed_ms=1,
    )


@pytest.mark.asyncio
async def test_formal_runtime_executes_knowledge_application_path() -> None:
    query = "说明库存补货制度"
    router = _DeterministicRouter(_decision(RequestRoute.KNOWLEDGE, knowledge_subquery=query))
    model = _DeterministicNativeModel({"run_knowledge_agent": query})
    knowledge_tool = _RecordingAgentTool(
        AgentToolResult(
            status="succeeded",
            answer="KNOWLEDGE_APPLICATION_MARKER [E1]",
            citations=["[E1]"],
        )
    )
    data_tool = _RecordingAgentTool(
        AgentToolResult(status="succeeded", answer="unused [D1]", citations=["[D1]"])
    )
    executor = _executor(
        router=router,
        model=model,
        knowledge_tool=knowledge_tool,
        data_tool=data_tool,
        synthesizer=_UnusedSynthesizer(),
    )

    response = await executor.execute(FormalRequest(request_id="m6b-knowledge", user_query=query))

    assert isinstance(response, FormalResponse)
    assert response.result.status is CoordinatorStatus.COMPLETED
    assert response.result.route is RequestRoute.KNOWLEDGE
    assert response.result.skill_name == "enterprise-knowledge-qa"
    assert response.result.answer == "KNOWLEDGE_APPLICATION_MARKER [E1]"
    assert response.result.citations == ["[E1]"]
    assert knowledge_tool.queries == [query] and data_tool.queries == []
    assert model.required_tools == ["run_knowledge_agent"] and len(model.calls) == 2
    _assert_router_used_without_memory(router)
    _assert_memory_not_requested(response)


@pytest.mark.asyncio
async def test_formal_runtime_executes_data_application_path() -> None:
    query = "查询当前库存风险产品"
    sql = "SELECT product_id, on_hand_quantity FROM inventory_snapshots LIMIT 1"
    client = _DeterministicEnterpriseDataClient(result=_successful_data_result(sql))
    data_tool, planner, generator, factory = _data_tool(
        sql=sql,
        client=client,
        answer="DATA_APPLICATION_MARKER [D1]",
    )
    router = _DeterministicRouter(_decision(RequestRoute.DATA, data_subquery=query))
    model = _DeterministicNativeModel({"run_data_agent": query})
    knowledge_tool = _RecordingAgentTool(
        AgentToolResult(status="succeeded", answer="unused [E1]", citations=["[E1]"])
    )
    executor = _executor(
        router=router,
        model=model,
        knowledge_tool=knowledge_tool,
        data_tool=data_tool,
        synthesizer=_UnusedSynthesizer(),
    )

    response = await executor.execute(FormalRequest(request_id="m6b-data", user_query=query))

    assert isinstance(response, FormalResponse)
    assert response.result.status is CoordinatorStatus.COMPLETED
    assert response.result.route is RequestRoute.DATA
    assert response.result.skill_name == "enterprise-data-analysis"
    assert response.result.answer == "DATA_APPLICATION_MARKER [D1]"
    assert response.result.citations == ["[D1]"]
    assert len(planner.calls) == len(generator.calls) == factory.calls == 1
    assert client.queries == [sql]
    assert (
        client.enter_calls,
        client.exit_calls,
        client.schema_calls,
        client.definition_calls,
    ) == (1, 1, 1, 1)
    assert knowledge_tool.queries == []
    assert model.required_tools == ["run_data_agent"] and len(model.calls) == 2
    _assert_router_used_without_memory(router)
    _assert_memory_not_requested(response)


@pytest.mark.asyncio
async def test_formal_runtime_executes_mixed_application_path() -> None:
    data_query = "查询当前库存风险产品"
    knowledge_query = "说明库存补货制度"
    sql = "SELECT product_id, on_hand_quantity FROM inventory_snapshots LIMIT 1"
    client = _DeterministicEnterpriseDataClient(result=_successful_data_result(sql))
    data_tool, planner, generator, factory = _data_tool(
        sql=sql,
        client=client,
        answer="DATA_MIXED_MARKER [D1]",
    )
    knowledge_tool = _RecordingAgentTool(
        AgentToolResult(
            status="succeeded",
            answer="KNOWLEDGE_MIXED_MARKER [E1]",
            citations=["[E1]"],
        )
    )
    router = _DeterministicRouter(
        _decision(
            RequestRoute.MIXED,
            knowledge_subquery=knowledge_query,
            data_subquery=data_query,
        )
    )
    model = _DeterministicNativeModel(
        {
            "run_data_agent": data_query,
            "run_knowledge_agent": knowledge_query,
        }
    )
    synthesizer = _RecordingSynthesizer()
    executor = _executor(
        router=router,
        model=model,
        knowledge_tool=knowledge_tool,
        data_tool=data_tool,
        synthesizer=synthesizer,
    )

    response = await executor.execute(
        FormalRequest(request_id="m6b-mixed", user_query="诊断当前库存风险并给出补货建议")
    )

    assert isinstance(response, FormalResponse)
    assert response.result.status is CoordinatorStatus.COMPLETED
    assert response.result.route is RequestRoute.MIXED
    assert response.result.skill_name == "inventory-risk-diagnosis"
    assert response.result.answer is not None
    assert "MIXED_APPLICATION_MARKER" in response.result.answer
    assert "DATA_MIXED_MARKER" in response.result.answer
    assert "KNOWLEDGE_MIXED_MARKER" in response.result.answer
    assert response.result.citations == ["[D1]", "[E1]"]
    assert len(planner.calls) == len(generator.calls) == factory.calls == 1
    assert client.queries == [sql] and knowledge_tool.queries == [knowledge_query]
    assert model.required_tools == ["run_data_agent", "run_knowledge_agent"]
    assert len(model.calls) == 4
    assert len(synthesizer.inputs) == 1
    assert synthesizer.inputs[0].data_answer == "DATA_MIXED_MARKER [D1]"
    assert synthesizer.inputs[0].knowledge_answer == "KNOWLEDGE_MIXED_MARKER [E1]"
    system_messages = [
        str(message["content"])
        for call in model.calls
        for message in call["messages"]  # type: ignore[union-attr]
        if message["role"] == "system"
    ]
    assert all(
        marker not in content
        for content in system_messages
        for marker in (
            "DATA_MIXED_MARKER",
            "KNOWLEDGE_MIXED_MARKER",
            "MIXED_APPLICATION_MARKER",
        )
    )
    _assert_router_used_without_memory(router)
    _assert_memory_not_requested(response)


@pytest.mark.asyncio
async def test_formal_runtime_projects_safe_query_rejection() -> None:
    query = "验证危险库存写操作会被拒绝"
    dangerous_sql = "DELETE FROM products"
    database_executor = _NeverExecuteDatabase()
    safe_query_service = SafeQueryService(
        guard=SQLGuard(max_rows=5),
        executor=database_executor,  # type: ignore[arg-type]
        query_timeout_seconds=1,
        max_rows=5,
        max_result_cells=20,
    )
    client = _InProcessSafeQueryClient(service=safe_query_service)
    data_tool, planner, generator, factory = _data_tool(
        sql=dangerous_sql,
        client=client,
        answer="must not be generated [D1]",
    )
    router = _DeterministicRouter(_decision(RequestRoute.DATA, data_subquery=query))
    model = _DeterministicNativeModel({"run_data_agent": query})
    knowledge_tool = _RecordingAgentTool(
        AgentToolResult(status="succeeded", answer="unused [E1]", citations=["[E1]"])
    )
    executor = _executor(
        router=router,
        model=model,
        knowledge_tool=knowledge_tool,
        data_tool=data_tool,
        synthesizer=_UnusedSynthesizer(),
    )

    response = await executor.execute(
        FormalRequest(request_id="m6b-safe-query-rejection", user_query=query)
    )

    assert isinstance(response, FormalResponse)
    assert response.result.status is CoordinatorStatus.FAILED
    assert response.result.route is RequestRoute.DATA
    assert response.result.skill_name is None
    assert response.result.answer is None and response.result.citations == []
    assert response.result.error_code == "write_statement_not_allowed"
    assert len(planner.calls) == factory.calls == 1
    assert generator.calls == []
    assert client.queries == [dangerous_sql]
    assert (
        client.enter_calls,
        client.exit_calls,
        client.schema_calls,
        client.definition_calls,
    ) == (1, 1, 1, 1)
    assert database_executor.calls == 0
    assert model.required_tools == ["run_data_agent"] and len(model.calls) == 1
    assert knowledge_tool.queries == []
    public_response = response.model_dump_json().lower()
    assert dangerous_sql.lower() not in public_response
    assert not any(
        forbidden in public_response
        for forbidden in ("traceback", "secret", "http://", "https://", "mysql", "prompt")
    )
    _assert_router_used_without_memory(router)
    _assert_memory_not_requested(response)
