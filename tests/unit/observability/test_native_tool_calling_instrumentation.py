"""Trace contracts at the real Skill-to-native-tool runtime boundary."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any

import pytest

from decision_agent.coordination import Coordinator
from decision_agent.coordination.models import CoordinatorStatus
from decision_agent.observability import SpanStatus, TraceCollector, TraceContext, TraceStage
from decision_agent.routing.models import RequestRoute, RouterDecision
from decision_agent.skills.enterprise_knowledge_qa import EnterpriseKnowledgeQASkill
from decision_agent.skills.native_runtime import NativeToolCallingSkillExecutor
from decision_agent.skills.registry import SkillRegistry
from decision_agent.tool_calling.models import AgentToolResult, NativeToolCallingStatus
from decision_agent.tool_calling.runtime import run_native_tool_calling


class _Ids:
    def __init__(self) -> None:
        self._value = 0

    def __call__(self) -> str:
        self._value += 1
        return f"span_{self._value}"


class _Model:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = responses
        self.calls = 0

    async def complete(self, **_: object) -> dict[str, Any]:
        self.calls += 1
        return self._responses.pop(0)


class _Tool:
    def __init__(self, result: AgentToolResult | BaseException) -> None:
        self._result = result
        self.queries: list[str] = []

    async def run(self, *, query: str) -> AgentToolResult:
        self.queries.append(query)
        if isinstance(self._result, BaseException):
            raise self._result
        return self._result


class _Router:
    async def route(self, *, user_query: str) -> RouterDecision:
        del user_query
        return _decision()


class _FailingRecorder:
    def __init__(self, *, fail_start: bool) -> None:
        self._fail_start = fail_start

    def start_span(self, **_: object) -> TraceContext:
        if self._fail_start:
            raise RuntimeError("OBSERVABILITY_PRIVATE_PATH")
        return TraceContext.create(request_id="request_1", id_factory=lambda: "trace_bad")

    def complete_span(self, *_: object, **__: object) -> None:
        raise RuntimeError("OBSERVABILITY_PRIVATE_PATH")


def _decision(route: RequestRoute = RequestRoute.KNOWLEDGE) -> RouterDecision:
    return RouterDecision(
        route=route,
        normalized_query="ROUTER_QUERY_SECRET",
        decision_reason="ROUTER_REASON_SECRET",
        knowledge_subquery="KNOWLEDGE_QUERY_SECRET" if route is RequestRoute.KNOWLEDGE else None,
        data_subquery="DATA_QUERY_SECRET" if route is RequestRoute.DATA else None,
        missing_information=None,
        confidence=0.9,
    )


def _tool_call(name: str, query: str = "MODEL_QUERY_SECRET") -> dict[str, Any]:
    return {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps({"query": query})},
                        }
                    ],
                },
            }
        ]
    }


def _final(answer: str, citations: list[str]) -> dict[str, Any]:
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": json.dumps({"answer": answer, "citations": citations})},
            }
        ]
    }


def _collector() -> tuple[TraceCollector, TraceContext, TraceContext]:
    collector, root = _root_collector()
    skill = collector.start_span(
        stage=TraceStage.SKILL_EXECUTION,
        component="skill",
        operation="execute_skill",
        parent_context=root,
    )
    return collector, root, skill


def _root_collector() -> tuple[TraceCollector, TraceContext]:
    collector = TraceCollector(
        context=TraceContext.create(request_id="request_1", id_factory=lambda: "trace_1"),
        utc_now=lambda: datetime(2026, 7, 27, tzinfo=UTC),
        monotonic=lambda: 10.0,
        id_factory=_Ids(),
    )
    root = collector.start_span(stage=TraceStage.REQUEST, component="executor", operation="execute")
    return collector, root


def _finish(collector: TraceCollector, root: TraceContext, skill: TraceContext):
    collector.complete_span(skill, status=SpanStatus.COMPLETED)
    collector.complete_span(root, status=SpanStatus.COMPLETED)
    return collector.finalize(final_status=SpanStatus.COMPLETED)


def _finish_root(collector: TraceCollector, root: TraceContext):
    collector.complete_span(root, status=SpanStatus.COMPLETED)
    return collector.finalize(final_status=SpanStatus.COMPLETED)


def _attributes(span: object) -> dict[str, object]:
    return {attribute.key: attribute.value for attribute in span.attributes}  # type: ignore[attr-defined]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route", "tool_name", "answer", "citation"),
    [
        (RequestRoute.KNOWLEDGE, "run_knowledge_agent", "answer [E1]", "[E1]"),
        (RequestRoute.DATA, "run_data_agent", "answer [D1]", "[D1]"),
    ],
)
async def test_native_tool_spans_are_skill_children_and_keep_router_query_private(
    route: RequestRoute, tool_name: str, answer: str, citation: str
) -> None:
    collector, root, skill = _collector()
    model = _Model([_tool_call(tool_name), _final(answer, [citation])])
    knowledge = _Tool(AgentToolResult(status="succeeded", answer=answer, citations=[citation]))
    data = _Tool(AgentToolResult(status="succeeded", answer=answer, citations=[citation]))

    result = await run_native_tool_calling(
        user_query="USER_QUERY_SECRET",
        decision=_decision(route),
        model=model,
        knowledge_tool=knowledge,
        data_tool=data,
        trace_recorder=collector,
        trace_parent_context=skill,
    )
    trace = _finish(collector, root, skill)
    selection, execution = [
        span
        for span in trace.spans
        if span.stage in {TraceStage.TOOL_SELECTION, TraceStage.TOOL_EXECUTION}
    ]

    assert result.status is NativeToolCallingStatus.COMPLETED
    assert selection.parent_span_id == execution.parent_span_id == skill.current_span_id
    assert selection.trace_id == execution.trace_id == trace.trace_id
    assert selection.status is execution.status is SpanStatus.COMPLETED
    assert _attributes(selection) == {
        "tool_name": tool_name,
        "authorized": True,
        "argument_validation": "passed",
        "tool_call_count": 1,
        "selection_index": 0,
        "query_source": "router_owned",
        "success": True,
    }
    assert _attributes(execution) == {
        "tool_name": tool_name,
        "authorized": True,
        "execution_index": 0,
        "success": True,
        "result_status": "succeeded",
    }
    assert knowledge.queries == (
        ["KNOWLEDGE_QUERY_SECRET"] if route is RequestRoute.KNOWLEDGE else []
    )
    assert data.queries == (["DATA_QUERY_SECRET"] if route is RequestRoute.DATA else [])
    serialized = str(trace.model_dump(mode="json"))
    assert all(
        secret not in serialized
        for secret in ("USER_QUERY_SECRET", "ROUTER_QUERY_SECRET", "MODEL_QUERY_SECRET")
    )
    assert all(span.stage.value != "provider" for span in trace.spans)


@pytest.mark.asyncio
async def test_real_knowledge_skill_propagates_its_explicit_skill_parent_to_native_runtime() -> (
    None
):
    collector, root, skill_span = _collector()
    model = _Model([_tool_call("run_knowledge_agent"), _final("answer [E1]", ["[E1]"])])
    tool = _Tool(AgentToolResult(status="succeeded", answer="answer [E1]", citations=["[E1]"]))
    skill = EnterpriseKnowledgeQASkill(
        runtime=NativeToolCallingSkillExecutor(model=model, knowledge_tool=tool, data_tool=tool)
    )

    result = await skill.execute_with_trace(
        user_query="USER_QUERY_SECRET",
        decision=_decision(),
        trace_recorder=collector,
        trace_parent_context=skill_span,
    )
    trace = _finish(collector, root, skill_span)

    assert result.status.value == "completed"
    selection, provider, execution, answer_generation, answer_provider = trace.spans[-5:]
    assert [selection.parent_span_id, execution.parent_span_id] == [
        skill_span.current_span_id,
        skill_span.current_span_id,
    ]
    assert provider.parent_span_id == selection.span_id
    assert answer_generation.parent_span_id == skill_span.current_span_id
    assert answer_provider.parent_span_id == answer_generation.span_id
    assert tool.queries == ["KNOWLEDGE_QUERY_SECRET"]


@pytest.mark.asyncio
async def test_coordinator_to_real_skill_to_native_tool_runtime_keeps_the_exact_span_tree() -> None:
    collector, root = _root_collector()
    model = _Model([_tool_call("run_knowledge_agent"), _final("answer [E1]", ["[E1]"])])
    tool = _Tool(AgentToolResult(status="succeeded", answer="answer [E1]", citations=["[E1]"]))
    skill = EnterpriseKnowledgeQASkill(
        runtime=NativeToolCallingSkillExecutor(model=model, knowledge_tool=tool, data_tool=tool)
    )
    registry = SkillRegistry()
    registry.register(skill)

    result = await Coordinator(router=_Router(), registry=registry).execute(
        user_query="USER_QUERY_SECRET",
        trace_recorder=collector,
        trace_parent_context=root,
    )
    trace = _finish_root(collector, root)
    spans = {span.stage: span for span in trace.spans}

    assert result.status is CoordinatorStatus.COMPLETED
    assert [span.stage for span in trace.spans] == [
        TraceStage.REQUEST,
        TraceStage.COORDINATION,
        TraceStage.ROUTING,
        TraceStage.SKILL_EXECUTION,
        TraceStage.TOOL_SELECTION,
        TraceStage.PROVIDER_CALL,
        TraceStage.TOOL_EXECUTION,
        TraceStage.ANSWER_GENERATION,
        TraceStage.PROVIDER_CALL,
    ]
    assert (
        spans[TraceStage.TOOL_SELECTION].parent_span_id
        == spans[TraceStage.TOOL_EXECUTION].parent_span_id
        == spans[TraceStage.SKILL_EXECUTION].span_id
    )
    providers = [span for span in trace.spans if span.stage is TraceStage.PROVIDER_CALL]
    assert providers[0].parent_span_id == spans[TraceStage.TOOL_SELECTION].span_id
    assert providers[1].parent_span_id == spans[TraceStage.ANSWER_GENERATION].span_id
    assert (
        spans[TraceStage.SKILL_EXECUTION].parent_span_id == spans[TraceStage.COORDINATION].span_id
    )
    assert spans[TraceStage.TOOL_SELECTION].trace_id == trace.trace_id
    assert tool.queries == ["KNOWLEDGE_QUERY_SECRET"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "code", "attributes"),
    [
        (_tool_call("unknown"), "native_tool_unknown", {"authorized": False}),
        (
            _tool_call("run_data_agent"),
            "native_tool_route_unauthorized",
            {"authorized": False, "denied": True},
        ),
        (
            _tool_call("run_knowledge_agent", query="x")
            | {"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": []}}]},
            "native_tool_call_count_invalid",
            {"tool_call_count": 0},
        ),
        (
            _tool_call("run_knowledge_agent", query="x")
            | {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "run_knowledge_agent", "arguments": "{}"},
                                }
                            ]
                        },
                    }
                ]
            },
            "native_tool_arguments_invalid",
            {"authorized": True, "argument_validation": "failed"},
        ),
    ],
)
async def test_rejected_tool_selection_has_no_execution_span(
    response: dict[str, Any], code: str, attributes: dict[str, object]
) -> None:
    collector, root, skill = _collector()
    tool = _Tool(AgentToolResult(status="succeeded", answer="answer [E1]", citations=["[E1]"]))
    result = await run_native_tool_calling(
        user_query="q",
        decision=_decision(),
        model=_Model([response]),
        knowledge_tool=tool,
        data_tool=tool,
        trace_recorder=collector,
        trace_parent_context=skill,
    )
    trace = _finish(collector, root, skill)
    selection = next(span for span in trace.spans if span.stage is TraceStage.TOOL_SELECTION)

    assert result.error_code == code
    assert selection.status is SpanStatus.FAILED and selection.error_code == code
    assert attributes.items() <= _attributes(selection).items()
    assert not [span for span in trace.spans if span.stage is TraceStage.TOOL_EXECUTION]
    assert tool.queries == []


@pytest.mark.asyncio
async def test_tool_failure_and_cancellation_are_distinct_from_selection() -> None:
    collector, root, skill = _collector()
    failed_tool = _Tool(AgentToolResult(status="failed", error_code="knowledge_agent_failed"))
    failed = await run_native_tool_calling(
        user_query="q",
        decision=_decision(),
        model=_Model([_tool_call("run_knowledge_agent")]),
        knowledge_tool=failed_tool,
        data_tool=failed_tool,
        trace_recorder=collector,
        trace_parent_context=skill,
    )
    trace = _finish(collector, root, skill)
    selection, execution = [
        span
        for span in trace.spans
        if span.stage in {TraceStage.TOOL_SELECTION, TraceStage.TOOL_EXECUTION}
    ]
    assert failed.error_code == "knowledge_agent_failed"
    assert selection.status is SpanStatus.COMPLETED
    assert (
        execution.status is SpanStatus.FAILED and execution.error_code == "knowledge_agent_failed"
    )

    collector, root, skill = _collector()
    cancelled_tool = _Tool(asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await run_native_tool_calling(
            user_query="q",
            decision=_decision(),
            model=_Model([_tool_call("run_knowledge_agent")]),
            knowledge_tool=cancelled_tool,
            data_tool=cancelled_tool,
            trace_recorder=collector,
            trace_parent_context=skill,
        )
    trace = _finish(collector, root, skill)
    assert (
        next(span for span in trace.spans if span.stage is TraceStage.TOOL_EXECUTION).status
        is SpanStatus.CANCELLED
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_start", [True, False])
async def test_native_tool_observability_failures_do_not_repeat_business_work(
    fail_start: bool, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("WARNING", logger="decision_agent.observability")
    model = _Model([_tool_call("run_knowledge_agent"), _final("answer [E1]", ["[E1]"])])
    tool = _Tool(AgentToolResult(status="succeeded", answer="answer [E1]", citations=["[E1]"]))

    result = await run_native_tool_calling(
        user_query="q",
        decision=_decision(),
        model=model,
        knowledge_tool=tool,
        data_tool=tool,
        trace_recorder=_FailingRecorder(fail_start=fail_start),
        trace_parent_context=None,
    )

    assert result.status is NativeToolCallingStatus.COMPLETED
    assert model.calls == 2 and tool.queries == ["KNOWLEDGE_QUERY_SECRET"]
    assert "OBSERVABILITY_PRIVATE_PATH" not in caplog.text
