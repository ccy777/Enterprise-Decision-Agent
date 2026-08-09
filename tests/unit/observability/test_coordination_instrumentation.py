"""Precise Coordinator/Router trace tests with no external runtime dependencies."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime

import pytest

from decision_agent.application import FormalRequest, FormalRequestExecutor
from decision_agent.context import ConversationMemoryProjector
from decision_agent.coordination import Coordinator
from decision_agent.coordination.models import CoordinatorStatus, SkillResult, SkillStatus
from decision_agent.memory import SessionMemorySnapshot, SessionTurn
from decision_agent.observability import (
    BestEffortTraceDispatcher,
    InMemoryTraceSink,
    SpanStatus,
    TraceCollector,
    TraceContext,
)
from decision_agent.routing.models import RequestRoute, RouterDecision
from decision_agent.routing.request_router import RequestRoutingError
from decision_agent.skills.contracts import SkillDefinition
from decision_agent.skills.registry import SkillRegistry


class _Ids:
    def __init__(self) -> None:
        self._value = 0

    def __call__(self) -> str:
        self._value += 1
        return f"span_{self._value}"


class _Router:
    def __init__(
        self, result: RouterDecision | BaseException, *, with_context: bool = False
    ) -> None:
        self._result = result
        self._with_context = with_context
        self.route_calls = 0
        self.context_calls = 0
        self.context_items: tuple[object, ...] = ()

    def __getattribute__(self, name: str) -> object:
        if name == "route_with_context" and not object.__getattribute__(self, "_with_context"):
            raise AttributeError(name)
        return object.__getattribute__(self, name)

    async def route(self, *, user_query: str) -> RouterDecision:
        self.route_calls += 1
        return self._resolve()

    async def route_with_context(
        self, *, user_query: str, selected_items: tuple[object, ...]
    ) -> RouterDecision:
        if not self._with_context:
            raise AssertionError("route_with_context must not be called")
        self.context_calls += 1
        self.context_items = selected_items
        return self._resolve()

    def _resolve(self) -> RouterDecision:
        if isinstance(self._result, BaseException):
            raise self._result
        return self._result


class _Skill:
    def __init__(
        self, route: RequestRoute, result: SkillResult | BaseException | None = None
    ) -> None:
        self.definition = SkillDefinition(
            name=f"{route}-skill",
            version="1",
            description="fake",
            supported_route=route,
            input_contract=("query",),
            allowed_tools=("fake-tool",),
            steps=("fake-step",),
            output_contract=("answer",),
            failure_codes=("safe_skill_failure",),
        )
        self._result = result
        self.calls = 0

    def is_applicable(self, user_query: str, decision: RouterDecision) -> bool:
        del user_query, decision
        return True

    async def execute(self, *, user_query: str, decision: RouterDecision) -> SkillResult:
        del user_query
        self.calls += 1
        if isinstance(self._result, BaseException):
            raise self._result
        if self._result is not None:
            return self._result
        citations = (
            ["[D1]", "[E1]"]
            if decision.route is RequestRoute.MIXED
            else ["[D1]"]
            if decision.route is RequestRoute.DATA
            else ["[E1]"]
        )
        return SkillResult(
            status=SkillStatus.COMPLETED,
            skill_name=self.definition.name,
            skill_version="1",
            route=decision.route,
            answer="ANSWER_SECRET_DO_NOT_LEAK",
            citations=citations,
            executed_steps=("fake-step",),
            selected_tool=None if decision.route is RequestRoute.MIXED else "fake-tool",
        )


class _Store:
    def __init__(self) -> None:
        self.read_calls: list[str] = []

    def read(self, session_id: str) -> SessionMemorySnapshot:
        self.read_calls.append(session_id)
        return SessionMemorySnapshot(
            session_id=session_id,
            version=0,
            turns=(
                SessionTurn(
                    session_id=session_id,
                    turn_id="turn_1",
                    request_id="previous_request",
                    user_text="MEMORY_SECRET_DO_NOT_LEAK",
                    assistant_text="MEMORY_ANSWER_SECRET_DO_NOT_LEAK",
                    created_at=datetime(2026, 7, 27, tzinfo=UTC),
                ),
            ),
        )

    def append_turn(self, *_: object, **__: object) -> SessionMemorySnapshot:
        raise AssertionError("unsupported request must not persist a Memory turn")


def _decision(route: RequestRoute) -> RouterDecision:
    return RouterDecision(
        route=route,
        normalized_query="ROUTER_SUBQUERY_SECRET_DO_NOT_LEAK",
        decision_reason="ROUTER_REASON_SECRET_DO_NOT_LEAK",
        knowledge_subquery=(
            "KNOWLEDGE_SUBQUERY_SECRET_DO_NOT_LEAK"
            if route in {RequestRoute.KNOWLEDGE, RequestRoute.MIXED}
            else None
        ),
        data_subquery=(
            "DATA_SUBQUERY_SECRET_DO_NOT_LEAK"
            if route in {RequestRoute.DATA, RequestRoute.MIXED}
            else None
        ),
        missing_information=None,
        confidence=0.9,
    )


def _collector_factory(*, trace_prefix: str = "trace") -> Callable[[FormalRequest], TraceCollector]:
    trace_counter = 0

    def create(request: FormalRequest) -> TraceCollector:
        nonlocal trace_counter
        trace_counter += 1
        return TraceCollector(
            context=TraceContext.create(
                request_id=request.request_id,
                session_present=request.session_id is not None,
                id_factory=lambda: f"{trace_prefix}_{trace_counter}",
            ),
            utc_now=lambda: datetime(2026, 7, 27, tzinfo=UTC),
            monotonic=lambda: 10.0,
            id_factory=_Ids(),
        )

    return create


def _executor(
    *,
    router: _Router,
    skill: _Skill | None = None,
    store: _Store | None = None,
    sink: InMemoryTraceSink,
    trace_prefix: str = "trace",
) -> FormalRequestExecutor:
    registry = SkillRegistry()
    if skill is not None:
        registry.register(skill)  # type: ignore[arg-type]
    return FormalRequestExecutor(
        coordinator=Coordinator(router=router, registry=registry),
        memory_store=store,  # type: ignore[arg-type]
        memory_projector=ConversationMemoryProjector(),
        trace_collector_factory=_collector_factory(trace_prefix=trace_prefix),
        trace_dispatcher=BestEffortTraceDispatcher([sink]),
    )


def _spans(trace, *stages: str):  # type: ignore[no-untyped-def]
    return [span for span in trace.spans if span.stage.value in stages]


@pytest.mark.asyncio
@pytest.mark.parametrize("route", [RequestRoute.KNOWLEDGE, RequestRoute.DATA, RequestRoute.MIXED])
async def test_real_coordinator_records_nested_routing_and_coordination_spans(
    route: RequestRoute,
) -> None:
    sink = InMemoryTraceSink()
    router = _Router(_decision(route))
    skill = _Skill(route)
    response = await _executor(router=router, skill=skill, sink=sink).execute(
        FormalRequest(request_id="request_1", user_query="QUERY_SECRET_DO_NOT_LEAK")
    )
    trace = sink.snapshot()[0]
    (
        request,
        memory_read,
        memory_write,
        memory_summary,
        coordination,
        routing,
        skill_span,
        mapping,
    ) = trace.spans

    assert response.result.status is CoordinatorStatus.COMPLETED
    assert router.route_calls == 1 and router.context_calls == 0 and skill.calls == 1
    assert [span.stage.value for span in trace.spans] == [
        "request",
        "memory_read",
        "memory_write",
        "memory_summary",
        "coordination",
        "routing",
        "skill_execution",
        "response_mapping",
    ]
    assert (
        memory_read.parent_span_id
        == memory_write.parent_span_id
        == memory_summary.parent_span_id
        == coordination.parent_span_id
        == mapping.parent_span_id
        == request.span_id
    )
    assert routing.parent_span_id == coordination.span_id
    assert skill_span.parent_span_id == coordination.span_id
    assert coordination.trace_id == routing.trace_id == skill_span.trace_id == trace.trace_id
    assert coordination.status is SpanStatus.COMPLETED and routing.status is SpanStatus.COMPLETED
    assert skill_span.status is SpanStatus.COMPLETED
    coordination_attributes = {item.key: item.value for item in coordination.attributes}
    assert coordination_attributes["route"] == route.value
    assert coordination_attributes["selected_skill_count"] == 1
    assert coordination_attributes["success"] is True
    skill_attributes = {item.key: item.value for item in skill_span.attributes}
    assert skill_attributes["route"] == route.value
    assert skill_attributes["skill_name"] == skill.definition.name
    assert skill_attributes["execution_index"] == 0
    assert skill_attributes["selected_skill_count"] == 1
    assert skill_attributes["result_status"] == "completed"
    if route is RequestRoute.MIXED:
        assert "skill_name" not in coordination_attributes
    serialized = str(trace.model_dump(mode="json"))
    assert all(
        secret not in serialized
        for secret in (
            "QUERY_SECRET",
            "ROUTER_SUBQUERY_SECRET",
            "ROUTER_REASON_SECRET",
            "KNOWLEDGE_SUBQUERY_SECRET",
            "DATA_SUBQUERY_SECRET",
            "ANSWER_SECRET",
        )
    )


@pytest.mark.asyncio
async def test_route_with_context_is_nested_and_memory_content_is_not_traced() -> None:
    sink = InMemoryTraceSink()
    router = _Router(_decision(RequestRoute.UNSUPPORTED), with_context=True)
    store = _Store()
    response = await _executor(router=router, store=store, sink=sink).execute(
        FormalRequest(
            request_id="request_1",
            session_id="SESSION_SECRET_DO_NOT_LEAK",
            user_query="QUERY_SECRET_DO_NOT_LEAK",
        )
    )
    trace = sink.snapshot()[0]
    coordination, routing = _spans(trace, "coordination", "routing")

    assert response.result.status is CoordinatorStatus.UNSUPPORTED
    assert store.read_calls == ["SESSION_SECRET_DO_NOT_LEAK"]
    assert router.route_calls == 0 and router.context_calls == 1
    assert routing.parent_span_id == coordination.span_id
    assert routing.status is SpanStatus.COMPLETED
    assert coordination.status is SpanStatus.UNSUPPORTED
    assert {item.key: item.value for item in coordination.attributes}["selected_skill_count"] == 0
    assert _spans(trace, "skill_execution") == []
    serialized = str(trace.model_dump(mode="json"))
    assert "SESSION_SECRET" not in serialized
    assert "MEMORY_SECRET" not in serialized
    assert "MEMORY_ANSWER_SECRET" not in serialized


@pytest.mark.asyncio
async def test_disabled_memory_with_session_keeps_legacy_route_and_nested_trace_contract() -> None:
    sink = InMemoryTraceSink()
    router = _Router(_decision(RequestRoute.UNSUPPORTED))
    response = await _executor(router=router, sink=sink).execute(
        FormalRequest(
            request_id="request_1",
            session_id="SESSION_SECRET_DO_NOT_LEAK",
            user_query="q",
        )
    )
    trace = sink.snapshot()[0]
    memory, coordination, routing = _spans(trace, "memory_read", "coordination", "routing")

    assert response.result.status is CoordinatorStatus.UNSUPPORTED
    assert response.memory_context_status.value == "not_requested"
    assert router.route_calls == 1 and router.context_calls == 0
    assert memory.status is SpanStatus.NOT_REQUESTED
    assert routing.parent_span_id == coordination.span_id


@pytest.mark.asyncio
async def test_router_failure_is_distinct_from_skill_failure_and_preserves_safe_codes() -> None:
    sink = InMemoryTraceSink()
    router = _Router(RequestRoutingError("router_provider_unavailable"))
    response = await _executor(router=router, sink=sink).execute(
        FormalRequest(request_id="request_1", user_query="q")
    )
    trace = sink.snapshot()[0]
    coordination, routing = _spans(trace, "coordination", "routing")

    assert response.result.error_code == "coordinator_router_router_provider_unavailable"
    assert trace.final_status is SpanStatus.FAILED
    assert routing.status is coordination.status is SpanStatus.FAILED
    assert routing.error_code == coordination.error_code == response.result.error_code

    failed_skill = _Skill(
        RequestRoute.KNOWLEDGE,
        SkillResult(
            status=SkillStatus.FAILED,
            skill_name="knowledge-skill",
            skill_version="1",
            route=RequestRoute.KNOWLEDGE,
            error_code="safe_skill_failure",
            executed_steps=("fake-step",),
        ),
    )
    sink = InMemoryTraceSink()
    response = await _executor(
        router=_Router(_decision(RequestRoute.KNOWLEDGE)), skill=failed_skill, sink=sink
    ).execute(FormalRequest(request_id="request_2", user_query="q"))
    trace = sink.snapshot()[0]
    coordination, routing, skill_span = _spans(trace, "coordination", "routing", "skill_execution")

    assert response.result.error_code == "safe_skill_failure"
    assert routing.status is SpanStatus.COMPLETED
    assert skill_span.status is SpanStatus.FAILED
    assert skill_span.error_code == "safe_skill_failure"
    assert coordination.status is SpanStatus.FAILED
    assert coordination.error_code == "safe_skill_failure"


@pytest.mark.asyncio
async def test_skill_technical_failure_is_traced_without_exception_text() -> None:
    sink = InMemoryTraceSink()
    response = await _executor(
        router=_Router(_decision(RequestRoute.KNOWLEDGE)),
        skill=_Skill(RequestRoute.KNOWLEDGE, RuntimeError("SKILL_PRIVATE_PATH")),
        sink=sink,
    ).execute(FormalRequest(request_id="request_1", user_query="q"))
    trace = sink.snapshot()[0]
    coordination, skill_span = _spans(trace, "coordination", "skill_execution")

    assert response.result.error_code == "skill_execution_failed"
    assert trace.final_status is SpanStatus.FAILED
    assert skill_span.status is coordination.status is SpanStatus.FAILED
    assert skill_span.error_code == "skill_execution_failed"
    assert "SKILL_PRIVATE_PATH" not in str(trace.model_dump(mode="json"))


@pytest.mark.asyncio
@pytest.mark.parametrize("during", ["router", "skill"])
async def test_cancellation_marks_the_correct_nested_spans_and_reraises(during: str) -> None:
    sink = InMemoryTraceSink()
    if during == "router":
        executor = _executor(router=_Router(asyncio.CancelledError()), sink=sink)
    else:
        executor = _executor(
            router=_Router(_decision(RequestRoute.KNOWLEDGE)),
            skill=_Skill(RequestRoute.KNOWLEDGE, asyncio.CancelledError()),
            sink=sink,
        )

    with pytest.raises(asyncio.CancelledError):
        await executor.execute(FormalRequest(request_id="request_1", user_query="q"))

    trace = sink.snapshot()[0]
    coordination = _spans(trace, "coordination")[0]
    routing = _spans(trace, "routing")[0]
    assert trace.final_status is SpanStatus.CANCELLED
    assert coordination.status is SpanStatus.CANCELLED
    assert routing.status is (SpanStatus.CANCELLED if during == "router" else SpanStatus.COMPLETED)
    if during == "skill":
        assert _spans(trace, "skill_execution")[0].status is SpanStatus.CANCELLED


class _FailingRecorder:
    def __init__(self, *, fail_start: bool) -> None:
        self._fail_start = fail_start
        self.calls: list[str] = []

    def start_span(self, **kwargs: object) -> TraceContext:
        self.calls.append(str(kwargs["stage"]))
        if self._fail_start:
            raise RuntimeError("OBSERVABILITY_PRIVATE_PATH")
        return TraceContext(
            trace_id="trace_1",
            request_id="request_1",
            session_present=False,
            current_span_id=f"span_{len(self.calls)}",
        )

    def complete_span(self, *_: object, **__: object) -> None:
        raise RuntimeError("OBSERVABILITY_PRIVATE_PATH")


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_start", [True, False])
async def test_coordinator_observability_faults_do_not_skip_router_or_skill(
    fail_start: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("WARNING", logger="decision_agent.observability")
    router = _Router(_decision(RequestRoute.KNOWLEDGE))
    skill = _Skill(RequestRoute.KNOWLEDGE)
    registry = SkillRegistry()
    registry.register(skill)  # type: ignore[arg-type]

    result = await Coordinator(router=router, registry=registry).execute(
        user_query="q",
        trace_recorder=_FailingRecorder(fail_start=fail_start),
    )

    assert result.status is CoordinatorStatus.COMPLETED
    assert router.route_calls == 1 and skill.calls == 1
    assert result.error_code is None
    assert "OBSERVABILITY_PRIVATE_PATH" not in caplog.text


@pytest.mark.asyncio
async def test_concurrent_mixed_and_knowledge_skill_spans_remain_trace_local() -> None:
    sink = InMemoryTraceSink()
    knowledge = _executor(
        router=_Router(_decision(RequestRoute.KNOWLEDGE)),
        skill=_Skill(RequestRoute.KNOWLEDGE),
        sink=sink,
        trace_prefix="knowledge",
    )
    mixed = _executor(
        router=_Router(_decision(RequestRoute.MIXED)),
        skill=_Skill(RequestRoute.MIXED),
        sink=sink,
        trace_prefix="mixed",
    )

    responses = await asyncio.gather(
        knowledge.execute(FormalRequest(request_id="knowledge_request", user_query="q")),
        mixed.execute(FormalRequest(request_id="mixed_request", user_query="q")),
    )
    traces = sink.snapshot()

    assert all(response.result.status is CoordinatorStatus.COMPLETED for response in responses)
    assert {trace.request_id for trace in traces} == {"knowledge_request", "mixed_request"}
    assert len({trace.trace_id for trace in traces}) == 2
    assert all(
        len(_spans(trace, "skill_execution")) == 1
        and _spans(trace, "skill_execution")[0].trace_id == trace.trace_id
        for trace in traces
    )
