"""Provider-span contracts for Data answers and inventory-risk synthesis."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from decision_agent.agents.data_answer_generator import OpenAICompatibleDataAnswerGenerator
from decision_agent.agents.data_query_planner import DataQueryPlan
from decision_agent.coordination.models import SkillResult, SkillStatus
from decision_agent.mcp_client.contracts import MCPQueryResult
from decision_agent.observability import SpanStatus, TraceCollector, TraceContext, TraceStage
from decision_agent.routing.models import RequestRoute, RouterDecision
from decision_agent.skills.contracts import SkillDefinition
from decision_agent.skills.inventory_risk_diagnosis import InventoryRiskDiagnosisSkill
from decision_agent.skills.inventory_risk_synthesizer import (
    OpenAICompatibleInventoryRiskSynthesizer,
)
from decision_agent.tool_calling.runtime import run_native_tool_calling
from decision_agent.tool_calling.tools import DataAgentTool
from decision_agent.workflows.data_agent import DataAgentStatus, run_data_agent


class _Ids:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self) -> str:
        self.value += 1
        return f"span_{self.value}"


def _collector() -> tuple[TraceCollector, TraceContext, TraceContext]:
    collector = TraceCollector(
        context=TraceContext.create(request_id="request_1", id_factory=lambda: "trace_1"),
        utc_now=lambda: datetime(2026, 7, 28, tzinfo=UTC),
        monotonic=lambda: 10.0,
        id_factory=_Ids(),
    )
    root = collector.start_span(stage=TraceStage.REQUEST, component="executor", operation="execute")
    parent = collector.start_span(
        stage=TraceStage.TOOL_EXECUTION,
        component="tool_calling",
        operation="execute_authorized_tool",
        parent_context=root,
    )
    return collector, root, parent


def _finish(collector: TraceCollector, root: TraceContext, parent: TraceContext):
    collector.complete_span(parent, status=SpanStatus.COMPLETED)
    collector.complete_span(root, status=SpanStatus.COMPLETED)
    return collector.finalize(final_status=SpanStatus.COMPLETED)


class _Planner:
    async def plan(self, **_: object) -> DataQueryPlan:
        return DataQueryPlan(
            status="ready",
            intent="query",
            sql="SELECT product_id FROM products",
            decision_reason="ready",
        )


class _DataClient:
    async def __aenter__(self) -> _DataClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def get_enterprise_schema(self) -> object:
        return SimpleNamespace(tables={"products": ["product_id"]})

    async def get_business_definitions(self) -> object:
        return SimpleNamespace(definitions={})

    async def execute_safe_query(self, sql: str) -> MCPQueryResult:
        del sql
        return MCPQueryResult(
            columns=["product_id"],
            rows=[["P1"]],
            row_count=1,
            truncated=False,
            normalized_sql="SELECT product_id FROM products",
            accessed_tables=["products"],
            elapsed_ms=1.0,
        )


def _data_generator() -> OpenAICompatibleDataAnswerGenerator:
    return OpenAICompatibleDataAnswerGenerator(
        api_key="test-key",
        base_url="https://example.test",
        model_name="data-model",
        timeout_seconds=1,
    )


def _data_response(answer: str = "Product P1 returned.[D1]") -> dict[str, Any]:
    return {
        "usage": {"prompt_tokens": 0, "completion_tokens": 3},
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": json.dumps({"answer": answer, "citations": ["[D1]"]})},
            }
        ],
    }


@pytest.mark.asyncio
async def test_data_answer_provider_is_a_tool_execution_grandchild_and_keeps_sql_private(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = _data_generator()
    monkeypatch.setattr(generator, "_post", lambda *_: _data_response())
    collector, root, tool = _collector()
    result = await run_data_agent(
        query="PRIVATE_QUERY",
        planner=_Planner(),
        enterprise_data_client_factory=_DataClient,
        answer_generator=generator,
        trace_recorder=collector,
        trace_parent_context=tool,
    )
    trace = _finish(collector, root, tool)
    answer = next(span for span in trace.spans if span.stage is TraceStage.ANSWER_GENERATION)
    provider = next(span for span in trace.spans if span.stage is TraceStage.PROVIDER_CALL)
    attrs = {item.key: item.value for item in provider.attributes}

    assert result.status is DataAgentStatus.ANSWERABLE_FINAL
    assert (
        answer.parent_span_id == tool.current_span_id and provider.parent_span_id == answer.span_id
    )
    assert attrs["input_tokens"] == 0 and attrs["output_tokens"] == 3
    serialized = str(trace.model_dump(mode="json"))
    assert all(
        value not in serialized for value in ("PRIVATE_QUERY", "SELECT product_id", "Product P1")
    )


@pytest.mark.asyncio
async def test_data_provider_completion_followed_by_citation_failure_fails_only_answer_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = _data_generator()
    monkeypatch.setattr(generator, "_post", lambda *_: _data_response("Wrong.[D9]"))
    collector, root, tool = _collector()
    result = await run_data_agent(
        query="q",
        planner=_Planner(),
        enterprise_data_client_factory=_DataClient,
        answer_generator=generator,
        trace_recorder=collector,
        trace_parent_context=tool,
    )
    trace = _finish(collector, root, tool)
    answer = next(span for span in trace.spans if span.stage is TraceStage.ANSWER_GENERATION)
    provider = next(span for span in trace.spans if span.stage is TraceStage.PROVIDER_CALL)

    assert result.errors[0].code == "data_citations_not_present_in_answer"
    assert provider.status is SpanStatus.COMPLETED
    assert answer.status is SpanStatus.FAILED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected_status"),
    [(OSError(), SpanStatus.FAILED), (asyncio.CancelledError(), SpanStatus.CANCELLED)],
)
async def test_data_provider_failure_and_cancellation_preserve_one_call(
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
    expected_status: SpanStatus,
) -> None:
    generator = _data_generator()
    calls = 0

    def fail(*_: object) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise failure

    monkeypatch.setattr(generator, "_post", fail)
    collector, root, tool = _collector()
    if isinstance(failure, asyncio.CancelledError):
        with pytest.raises(asyncio.CancelledError):
            await run_data_agent(
                query="q",
                planner=_Planner(),
                enterprise_data_client_factory=_DataClient,
                answer_generator=generator,
                trace_recorder=collector,
                trace_parent_context=tool,
            )
    else:
        result = await run_data_agent(
            query="q",
            planner=_Planner(),
            enterprise_data_client_factory=_DataClient,
            answer_generator=generator,
            trace_recorder=collector,
            trace_parent_context=tool,
        )
        assert result.status is DataAgentStatus.FAILED
    trace = _finish(collector, root, tool)
    statuses = [
        span.status
        for span in trace.spans
        if span.stage in {TraceStage.ANSWER_GENERATION, TraceStage.PROVIDER_CALL}
    ]
    assert calls == 1 and statuses == [expected_status, expected_status]


@pytest.mark.asyncio
async def test_runtime_uses_data_trace_entry_once_with_router_owned_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = _data_generator()
    monkeypatch.setattr(generator, "_post", lambda *_: _data_response())

    class _Model:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, **_: object) -> dict[str, Any]:
            self.calls += 1
            if self.calls == 1:
                return {
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "tool_calls": [
                                    {
                                        "id": "call",
                                        "type": "function",
                                        "function": {
                                            "name": "run_data_agent",
                                            "arguments": json.dumps(
                                                {"query": "MODEL_QUERY_SECRET"}
                                            ),
                                        },
                                    }
                                ]
                            },
                        }
                    ]
                }
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {"answer": "Product P1 returned.[D1]", "citations": ["[D1]"]}
                            )
                        },
                    }
                ]
            }

    class _UnusedKnowledge:
        async def run(self, *, query: str) -> object:
            del query
            raise AssertionError("knowledge tool must not run")

    collector, root, skill = _collector()
    result = await run_native_tool_calling(
        user_query="PRIVATE_QUERY",
        decision=RouterDecision(
            route=RequestRoute.DATA,
            normalized_query="PRIVATE",
            decision_reason="PRIVATE",
            knowledge_subquery=None,
            data_subquery="ROUTER_DATA_QUERY",
            missing_information=None,
            confidence=0.9,
        ),
        model=_Model(),
        knowledge_tool=_UnusedKnowledge(),
        data_tool=DataAgentTool(
            planner=_Planner(),
            enterprise_data_client_factory=_DataClient,
            answer_generator=generator,
        ),
        trace_recorder=collector,
        trace_parent_context=skill,
    )
    trace = _finish(collector, root, skill)
    execution = [span for span in trace.spans if span.stage is TraceStage.TOOL_EXECUTION][-1]
    answer = next(
        span
        for span in trace.spans
        if span.stage is TraceStage.ANSWER_GENERATION and span.operation == "generate_data_answer"
    )

    assert result.status.value == "completed" and answer.parent_span_id == execution.span_id
    assert "MODEL_QUERY_SECRET" not in str(trace.model_dump(mode="json"))


class _ChildSkill:
    def __init__(self, route: RequestRoute, citation: str) -> None:
        self.definition = SkillDefinition(
            name="enterprise-data-analysis"
            if route is RequestRoute.DATA
            else "enterprise-knowledge-qa",
            version="1",
            description="fake",
            supported_route=route,
            input_contract=("q",),
            allowed_tools=("run_data_agent",)
            if route is RequestRoute.DATA
            else ("run_knowledge_agent",),
            steps=("fake",),
            output_contract=("answer",),
            failure_codes=("failed",),
        )
        self._route, self._citation = route, citation

    async def execute(self, *, user_query: str, decision: RouterDecision) -> SkillResult:
        del user_query, decision
        return SkillResult(
            status=SkillStatus.COMPLETED,
            skill_name=self.definition.name,
            skill_version="1",
            route=self._route,
            answer="PRIVATE_CHILD_ANSWER",
            citations=[self._citation],
            executed_steps=("fake",),
            selected_tool="run_data_agent"
            if self._route is RequestRoute.DATA
            else "run_knowledge_agent",
        )


class _ChatClient:
    def __init__(self, response: object) -> None:
        self.response, self.calls = response, 0

    async def complete_chat(self, **_: object) -> dict[str, Any]:
        self.calls += 1
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response  # type: ignore[return-value]


def _mixed_decision() -> RouterDecision:
    return RouterDecision(
        route=RequestRoute.MIXED,
        normalized_query="PRIVATE_QUERY",
        decision_reason="PRIVATE_REASON",
        data_subquery="inventory DATA_QUERY",
        knowledge_subquery="inventory KNOWLEDGE_QUERY",
        missing_information=None,
        confidence=0.9,
    )


def _synthesis_response() -> dict[str, Any]:
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(
                        {
                            "risk_summary": "PRIVATE_RISK",
                            "policy_basis": "PRIVATE_POLICY",
                            "recommended_actions": ["PRIVATE_ACTION"],
                            "citations": ["[D1]", "[E1]"],
                        }
                    )
                },
            }
        ]
    }


@pytest.mark.asyncio
async def test_inventory_synthesis_is_skill_child_with_one_provider_call() -> None:
    client = _ChatClient(_synthesis_response())
    skill = InventoryRiskDiagnosisSkill(
        data_skill=_ChildSkill(RequestRoute.DATA, "[D1]"),
        knowledge_skill=_ChildSkill(RequestRoute.KNOWLEDGE, "[E1]"),
        synthesizer=OpenAICompatibleInventoryRiskSynthesizer(client=client),
    )
    collector = TraceCollector(
        context=TraceContext.create(request_id="request_1", id_factory=lambda: "trace_1"),
        utc_now=lambda: datetime(2026, 7, 28, tzinfo=UTC),
        monotonic=lambda: 10.0,
        id_factory=_Ids(),
    )
    root = collector.start_span(stage=TraceStage.REQUEST, component="executor", operation="execute")
    parent = collector.start_span(
        stage=TraceStage.SKILL_EXECUTION,
        component="skill",
        operation="execute_skill",
        parent_context=root,
    )
    result = await skill.execute_with_trace(
        user_query="PRIVATE_QUERY",
        decision=_mixed_decision(),
        trace_recorder=collector,
        trace_parent_context=parent,
    )
    trace = _finish(collector, root, parent)
    answer = next(span for span in trace.spans if span.stage is TraceStage.ANSWER_GENERATION)
    provider = next(span for span in trace.spans if span.stage is TraceStage.PROVIDER_CALL)

    assert result.status is SkillStatus.COMPLETED and client.calls == 1
    assert (
        answer.parent_span_id == parent.current_span_id
        and provider.parent_span_id == answer.span_id
    )
    assert "PRIVATE_RISK" not in str(trace.model_dump(mode="json"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "provider_status", "code"),
    [
        ({"choices": []}, SpanStatus.COMPLETED, "inventory_risk_synthesizer_missing_choice"),
        (OSError(), SpanStatus.FAILED, "inventory_risk_synthesizer_unavailable"),
    ],
)
async def test_inventory_local_validation_and_provider_failure_are_distinct(
    response: object, provider_status: SpanStatus, code: str
) -> None:
    client = _ChatClient(response)
    skill = InventoryRiskDiagnosisSkill(
        data_skill=_ChildSkill(RequestRoute.DATA, "[D1]"),
        knowledge_skill=_ChildSkill(RequestRoute.KNOWLEDGE, "[E1]"),
        synthesizer=OpenAICompatibleInventoryRiskSynthesizer(client=client),
    )
    collector, root, parent = _collector()
    result = await skill.execute_with_trace(
        user_query="inventory q",
        decision=_mixed_decision(),
        trace_recorder=collector,
        trace_parent_context=parent,
    )
    trace = _finish(collector, root, parent)
    answer = next(span for span in trace.spans if span.stage is TraceStage.ANSWER_GENERATION)
    provider = next(span for span in trace.spans if span.stage is TraceStage.PROVIDER_CALL)

    assert result.error_code == code and client.calls == 1
    assert provider.status is provider_status and answer.status is SpanStatus.FAILED


@pytest.mark.asyncio
async def test_inventory_provider_cancellation_cancels_both_spans() -> None:
    client = _ChatClient(asyncio.CancelledError())
    skill = InventoryRiskDiagnosisSkill(
        data_skill=_ChildSkill(RequestRoute.DATA, "[D1]"),
        knowledge_skill=_ChildSkill(RequestRoute.KNOWLEDGE, "[E1]"),
        synthesizer=OpenAICompatibleInventoryRiskSynthesizer(client=client),
    )
    collector, root, parent = _collector()
    with pytest.raises(asyncio.CancelledError):
        await skill.execute_with_trace(
            user_query="q",
            decision=_mixed_decision(),
            trace_recorder=collector,
            trace_parent_context=parent,
        )
    trace = _finish(collector, root, parent)
    states = [
        span.status
        for span in trace.spans
        if span.stage in {TraceStage.ANSWER_GENERATION, TraceStage.PROVIDER_CALL}
    ]
    assert states == [SpanStatus.CANCELLED, SpanStatus.CANCELLED] and client.calls == 1
