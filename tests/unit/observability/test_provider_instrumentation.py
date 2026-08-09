"""Provider-span contracts at the Router and native Tool Selection boundaries."""

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
from decision_agent.routing.request_router import OpenAICompatibleRequestRouter
from decision_agent.skills.registry import SkillRegistry
from decision_agent.tool_calling.models import AgentToolResult, NativeToolCallingStatus
from decision_agent.tool_calling.runtime import (
    NativeToolCallingError,
    OpenAICompatibleNativeToolCallingModel,
    run_native_tool_calling,
)


class _Ids:
    def __init__(self) -> None:
        self._value = 0

    def __call__(self) -> str:
        self._value += 1
        return f"span_{self._value}"


class _Tool:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def run(self, *, query: str) -> AgentToolResult:
        self.queries.append(query)
        return AgentToolResult(status="succeeded", answer="TOOL_ANSWER_SECRET", citations=["[E1]"])


class _NativeModel(OpenAICompatibleNativeToolCallingModel):
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        super().__init__(
            api_key="test-key",
            base_url="https://example.test",
            model_name="test-model",
            timeout_seconds=1,
        )
        self._responses = responses
        self.calls = 0

    async def complete(self, **_: object) -> dict[str, Any]:
        self.calls += 1
        return self._responses.pop(0)


class _FailingNativeModel:
    async def complete(self, **_: object) -> dict[str, Any]:
        raise NativeToolCallingError("tool_calling_provider_unavailable")


class _CancelledNativeModel:
    async def complete(self, **_: object) -> dict[str, Any]:
        raise asyncio.CancelledError


class _FinalProviderFailureModel(_NativeModel):
    async def complete(self, **_: object) -> dict[str, Any]:
        self.calls += 1
        if self.calls == 1:
            return self._responses.pop(0)
        raise NativeToolCallingError("tool_calling_provider_unavailable")


class _FinalProviderCancellationModel(_NativeModel):
    async def complete(self, **_: object) -> dict[str, Any]:
        self.calls += 1
        if self.calls == 1:
            return self._responses.pop(0)
        raise asyncio.CancelledError


def _collector() -> tuple[TraceCollector, TraceContext]:
    collector = TraceCollector(
        context=TraceContext.create(request_id="request_1", id_factory=lambda: "trace_1"),
        utc_now=lambda: datetime(2026, 7, 27, tzinfo=UTC),
        monotonic=lambda: 10.0,
        id_factory=_Ids(),
    )
    root = collector.start_span(stage=TraceStage.REQUEST, component="executor", operation="execute")
    return collector, root


def _finalize(collector: TraceCollector, root: TraceContext, status: SpanStatus):
    error_code = "test_failed" if status is SpanStatus.FAILED else None
    collector.complete_span(root, status=status, error_code=error_code)
    return collector.finalize(final_status=status, error_code=error_code)


def _attributes(span: object) -> dict[str, object]:
    return {attribute.key: attribute.value for attribute in span.attributes}  # type: ignore[attr-defined]


def _router_payload(*, extra: bool = False, usage: object | None = None) -> dict[str, Any]:
    decision: dict[str, object] = {
        "route": "unsupported",
        "normalized_query": "SAFE_NORMALIZED_QUERY_SECRET",
        "decision_reason": "The request is outside supported enterprise capabilities.",
        "knowledge_subquery": None,
        "data_subquery": None,
        "missing_information": None,
        "confidence": 0.9,
    }
    if extra:
        decision["tool_arguments"] = "MODEL_TOOL_ARGUMENTS_SECRET"
    payload: dict[str, Any] = {
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(decision)}}]
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


def _native_call(
    *, usage: object | None = None, name: str = "run_knowledge_agent"
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps({"query": "MODEL_QUERY_SECRET"}),
                            },
                        }
                    ],
                },
            }
        ]
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


def _native_final(
    *, usage: object | None = None, answer: str = "TOOL_ANSWER_SECRET"
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": json.dumps({"answer": answer, "citations": ["[E1]"]})},
            }
        ]
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


def _decision() -> RouterDecision:
    return RouterDecision(
        route=RequestRoute.KNOWLEDGE,
        normalized_query="ROUTER_NORMALIZED_QUERY_SECRET",
        decision_reason="ROUTER_REASON_SECRET",
        knowledge_subquery="ROUTER_OWNED_QUERY_SECRET",
        data_subquery=None,
        missing_information=None,
        confidence=0.9,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("usage", "expected_input", "expected_output", "available"),
    [({"prompt_tokens": 0, "completion_tokens": 7}, 0, 7, True), (None, None, None, False)],
)
async def test_router_provider_span_is_nested_and_projects_only_real_safe_metadata(
    monkeypatch: pytest.MonkeyPatch,
    usage: object | None,
    expected_input: int | None,
    expected_output: int | None,
    available: bool,
) -> None:
    router = OpenAICompatibleRequestRouter(
        api_key="test-key",
        base_url="https://example.test",
        model_name="test-model",
        timeout_seconds=1,
    )
    calls = 0

    def post(*_: object) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return _router_payload(usage=usage)

    monkeypatch.setattr(router, "_post", post)
    collector, root = _collector()
    result = await Coordinator(router=router, registry=SkillRegistry()).execute(
        user_query="USER_QUERY_SECRET",
        trace_recorder=collector,
        trace_parent_context=root,
    )
    trace = _finalize(collector, root, SpanStatus.UNSUPPORTED)
    spans = {span.stage: span for span in trace.spans}
    provider = spans[TraceStage.PROVIDER_CALL]

    assert result.status is CoordinatorStatus.UNSUPPORTED and calls == 1
    assert provider.parent_span_id == spans[TraceStage.ROUTING].span_id
    assert provider.status is SpanStatus.COMPLETED
    assert _attributes(provider) == {
        "provider": "openai_compatible",
        "model": "test-model",
        "operation": "route_request",
        "usage_available": available,
        "input_tokens": expected_input,
        "output_tokens": expected_output,
        "retry_count": 0,
        "finish_reason": "stop",
        "success": True,
    }
    serialized = str(trace.model_dump(mode="json"))
    assert all(
        secret not in serialized for secret in ("USER_QUERY_SECRET", "SAFE_NORMALIZED_QUERY_SECRET")
    )


@pytest.mark.asyncio
async def test_router_provider_success_is_not_rewritten_when_local_decision_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = OpenAICompatibleRequestRouter(
        api_key="test-key",
        base_url="https://example.test",
        model_name="test-model",
        timeout_seconds=1,
    )
    monkeypatch.setattr(router, "_post", lambda *_: _router_payload(extra=True))
    collector, root = _collector()
    result = await Coordinator(router=router, registry=SkillRegistry()).execute(
        user_query="USER_QUERY_SECRET",
        trace_recorder=collector,
        trace_parent_context=root,
    )
    trace = _finalize(collector, root, SpanStatus.FAILED)
    spans = {span.stage: span for span in trace.spans}

    assert result.error_code == "coordinator_router_router_schema_validation_failed"
    assert spans[TraceStage.PROVIDER_CALL].status is SpanStatus.COMPLETED
    assert spans[TraceStage.ROUTING].status is SpanStatus.FAILED
    assert "MODEL_TOOL_ARGUMENTS_SECRET" not in str(trace.model_dump(mode="json"))


@pytest.mark.asyncio
async def test_router_provider_failure_and_cancellation_keep_safe_statuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = OpenAICompatibleRequestRouter(
        api_key="test-key",
        base_url="https://example.test",
        model_name="test-model",
        timeout_seconds=1,
    )
    monkeypatch.setattr(router, "_post", lambda *_: (_ for _ in ()).throw(OSError("PRIVATE_PATH")))
    collector, root = _collector()
    failed = await Coordinator(router=router, registry=SkillRegistry()).execute(
        user_query="q", trace_recorder=collector, trace_parent_context=root
    )
    trace = _finalize(collector, root, SpanStatus.FAILED)
    provider = next(span for span in trace.spans if span.stage is TraceStage.PROVIDER_CALL)
    assert failed.error_code == "coordinator_router_router_provider_unavailable"
    assert (
        provider.status is SpanStatus.FAILED
        and provider.error_code == "router_provider_unavailable"
    )
    assert "PRIVATE_PATH" not in str(trace.model_dump(mode="json"))

    async def cancelled_to_thread(*_: object, **__: object) -> object:
        raise asyncio.CancelledError

    monkeypatch.setattr(
        "decision_agent.routing.request_router.asyncio.to_thread", cancelled_to_thread
    )
    collector, root = _collector()
    routing = collector.start_span(
        stage=TraceStage.ROUTING,
        component="routing",
        operation="route_request",
        parent_context=root,
    )
    with pytest.raises(asyncio.CancelledError):
        await router.route_with_trace(
            user_query="q",
            selected_items=None,
            trace_recorder=collector,
            trace_parent_context=routing,
        )
    collector.complete_span(routing, status=SpanStatus.CANCELLED)
    trace = _finalize(collector, root, SpanStatus.CANCELLED)
    assert (
        next(span for span in trace.spans if span.stage is TraceStage.PROVIDER_CALL).status
        is SpanStatus.CANCELLED
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        ({"prompt_tokens": 3, "completion_tokens": 0}, (True, 3, 0)),
        ({"prompt_tokens": "bad", "completion_tokens": 2}, (False, None, None)),
    ],
)
async def test_tool_selection_provider_span_keeps_usage_and_query_boundaries(
    usage: object, expected: tuple[bool, int | None, int | None]
) -> None:
    collector, root = _collector()
    skill = collector.start_span(
        stage=TraceStage.SKILL_EXECUTION,
        component="skill",
        operation="execute_skill",
        parent_context=root,
    )
    model = _NativeModel([_native_call(usage=usage), _native_final(usage=usage)])
    tool = _Tool()
    result = await run_native_tool_calling(
        user_query="USER_QUERY_SECRET",
        decision=_decision(),
        model=model,
        knowledge_tool=tool,
        data_tool=tool,
        trace_recorder=collector,
        trace_parent_context=skill,
    )
    collector.complete_span(skill, status=SpanStatus.COMPLETED)
    trace = _finalize(collector, root, SpanStatus.COMPLETED)
    selection = next(span for span in trace.spans if span.stage is TraceStage.TOOL_SELECTION)
    generation = next(span for span in trace.spans if span.stage is TraceStage.ANSWER_GENERATION)
    providers = {
        _attributes(span)["operation"]: span
        for span in trace.spans
        if span.stage is TraceStage.PROVIDER_CALL
    }
    provider = providers["select_tool"]
    answer_provider = providers["generate_tool_answer"]
    attributes = _attributes(provider)
    answer_attributes = _attributes(answer_provider)

    assert result.status is NativeToolCallingStatus.COMPLETED and model.calls == 2
    assert provider.parent_span_id == selection.span_id
    assert provider.status is SpanStatus.COMPLETED
    assert attributes["provider"] == "openai_compatible" and attributes["model"] == "test-model"
    assert (
        attributes["usage_available"],
        attributes["input_tokens"],
        attributes["output_tokens"],
    ) == expected
    assert attributes["retry_count"] == 0 and attributes["finish_reason"] == "tool_calls"
    assert generation.parent_span_id == skill.current_span_id
    assert answer_provider.parent_span_id == generation.span_id
    assert generation.status is answer_provider.status is SpanStatus.COMPLETED
    assert answer_attributes["operation"] == "generate_tool_answer"
    assert (
        answer_attributes["usage_available"],
        answer_attributes["input_tokens"],
        answer_attributes["output_tokens"],
    ) == expected
    assert tool.queries == ["ROUTER_OWNED_QUERY_SECRET"]
    serialized = str(trace.model_dump(mode="json"))
    assert all(
        secret not in serialized
        for secret in (
            "USER_QUERY_SECRET",
            "MODEL_QUERY_SECRET",
            "ROUTER_OWNED_QUERY_SECRET",
            "TOOL_ANSWER_SECRET",
        )
    )
    assert set(providers) == {"select_tool", "generate_tool_answer"}


@pytest.mark.asyncio
async def test_tool_selection_provider_failure_is_failed_without_a_tool_execution_span() -> None:
    collector, root = _collector()
    skill = collector.start_span(
        stage=TraceStage.SKILL_EXECUTION,
        component="skill",
        operation="execute_skill",
        parent_context=root,
    )
    tool = _Tool()
    result = await run_native_tool_calling(
        user_query="q",
        decision=_decision(),
        model=_FailingNativeModel(),
        knowledge_tool=tool,
        data_tool=tool,
        trace_recorder=collector,
        trace_parent_context=skill,
    )
    collector.complete_span(skill, status=SpanStatus.FAILED, error_code="test_failed")
    trace = _finalize(collector, root, SpanStatus.FAILED)
    provider = next(span for span in trace.spans if span.stage is TraceStage.PROVIDER_CALL)

    assert result.error_code == "tool_calling_provider_unavailable"
    assert provider.status is SpanStatus.FAILED
    assert provider.error_code == "tool_calling_provider_unavailable"
    assert tool.queries == []
    assert not [span for span in trace.spans if span.stage is TraceStage.TOOL_EXECUTION]


@pytest.mark.asyncio
async def test_tool_selection_provider_success_stays_completed_when_local_tool_contract_fails() -> (
    None
):
    collector, root = _collector()
    skill = collector.start_span(
        stage=TraceStage.SKILL_EXECUTION,
        component="skill",
        operation="execute_skill",
        parent_context=root,
    )
    tool = _Tool()
    result = await run_native_tool_calling(
        user_query="q",
        decision=_decision(),
        model=_NativeModel([_native_call(name="unknown")]),
        knowledge_tool=tool,
        data_tool=tool,
        trace_recorder=collector,
        trace_parent_context=skill,
    )
    collector.complete_span(skill, status=SpanStatus.FAILED, error_code="native_tool_unknown")
    trace = _finalize(collector, root, SpanStatus.FAILED)
    provider = next(span for span in trace.spans if span.stage is TraceStage.PROVIDER_CALL)
    selection = next(span for span in trace.spans if span.stage is TraceStage.TOOL_SELECTION)

    assert result.error_code == "native_tool_unknown"
    assert provider.status is SpanStatus.COMPLETED
    assert selection.status is SpanStatus.FAILED
    assert tool.queries == []


@pytest.mark.asyncio
async def test_tool_selection_provider_cancellation_is_re_raised_and_traced() -> None:
    collector, root = _collector()
    skill = collector.start_span(
        stage=TraceStage.SKILL_EXECUTION,
        component="skill",
        operation="execute_skill",
        parent_context=root,
    )
    tool = _Tool()
    with pytest.raises(asyncio.CancelledError):
        await run_native_tool_calling(
            user_query="q",
            decision=_decision(),
            model=_CancelledNativeModel(),
            knowledge_tool=tool,
            data_tool=tool,
            trace_recorder=collector,
            trace_parent_context=skill,
        )
    collector.complete_span(skill, status=SpanStatus.CANCELLED)
    trace = _finalize(collector, root, SpanStatus.CANCELLED)

    provider = next(span for span in trace.spans if span.stage is TraceStage.PROVIDER_CALL)
    selection = next(span for span in trace.spans if span.stage is TraceStage.TOOL_SELECTION)
    assert provider.status is selection.status is SpanStatus.CANCELLED
    assert tool.queries == []


@pytest.mark.asyncio
async def test_final_answer_provider_stays_completed_when_local_mapping_fails() -> None:
    collector, root = _collector()
    skill = collector.start_span(
        stage=TraceStage.SKILL_EXECUTION,
        component="skill",
        operation="execute_skill",
        parent_context=root,
    )
    tool = _Tool()
    result = await run_native_tool_calling(
        user_query="q",
        decision=_decision(),
        model=_NativeModel([_native_call(), _native_final(answer="MODEL_REWRITE_SECRET")]),
        knowledge_tool=tool,
        data_tool=tool,
        trace_recorder=collector,
        trace_parent_context=skill,
    )
    collector.complete_span(
        skill,
        status=SpanStatus.FAILED,
        error_code="tool_calling_final_answer_mismatch",
    )
    trace = _finalize(collector, root, SpanStatus.FAILED)
    generation = next(span for span in trace.spans if span.stage is TraceStage.ANSWER_GENERATION)
    provider = next(
        span
        for span in trace.spans
        if span.stage is TraceStage.PROVIDER_CALL
        and _attributes(span)["operation"] == "generate_tool_answer"
    )

    assert result.error_code == "tool_calling_final_answer_mismatch"
    assert provider.status is SpanStatus.COMPLETED
    assert generation.status is SpanStatus.FAILED
    assert "MODEL_REWRITE_SECRET" not in str(trace.model_dump(mode="json"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model_type", "expected_status"),
    [
        (_FinalProviderFailureModel, SpanStatus.FAILED),
        (_FinalProviderCancellationModel, SpanStatus.CANCELLED),
    ],
)
async def test_final_answer_provider_failure_and_cancellation_preserve_semantics(
    model_type: type[_FinalProviderFailureModel] | type[_FinalProviderCancellationModel],
    expected_status: SpanStatus,
) -> None:
    collector, root = _collector()
    skill = collector.start_span(
        stage=TraceStage.SKILL_EXECUTION,
        component="skill",
        operation="execute_skill",
        parent_context=root,
    )
    model = model_type([_native_call()])
    tool = _Tool()
    if expected_status is SpanStatus.CANCELLED:
        with pytest.raises(asyncio.CancelledError):
            await run_native_tool_calling(
                user_query="q",
                decision=_decision(),
                model=model,
                knowledge_tool=tool,
                data_tool=tool,
                trace_recorder=collector,
                trace_parent_context=skill,
            )
        collector.complete_span(skill, status=SpanStatus.CANCELLED)
    else:
        result = await run_native_tool_calling(
            user_query="q",
            decision=_decision(),
            model=model,
            knowledge_tool=tool,
            data_tool=tool,
            trace_recorder=collector,
            trace_parent_context=skill,
        )
        assert result.error_code == "tool_calling_provider_unavailable"
        collector.complete_span(
            skill,
            status=SpanStatus.FAILED,
            error_code="tool_calling_provider_unavailable",
        )
    trace = _finalize(collector, root, expected_status)
    generation = next(span for span in trace.spans if span.stage is TraceStage.ANSWER_GENERATION)
    provider = next(
        span
        for span in trace.spans
        if span.stage is TraceStage.PROVIDER_CALL
        and _attributes(span)["operation"] == "generate_tool_answer"
    )

    assert model.calls == 2
    assert provider.status is generation.status is expected_status
