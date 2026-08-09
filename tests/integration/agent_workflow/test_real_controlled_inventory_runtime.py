"""Opt-in real M8D-B2 acceptance through the formal configured Runtime."""

# ruff: noqa: RUF001

from __future__ import annotations

import json
import os

import pytest

from decision_agent.application import FormalRequest
from decision_agent.application.bootstrap import build_bootstrapped_runtime
from decision_agent.application.configured_runtime import create_configured_runtime_builder
from decision_agent.config import Settings
from decision_agent.coordination.models import CoordinatorStatus
from decision_agent.observability import (
    BestEffortTraceDispatcher,
    InMemoryTraceSink,
    SpanStatus,
    TraceStage,
)
from decision_agent.routing.models import RequestRoute

_RUN_ENV = "RUN_M8D_B2_REAL_JOINT"
_USER_QUERY = (
    "请依据企业库存制度中的安全库存、补货触发和紧急处理规则，"
    "结合当前各产品库存数据，识别需要优先关注的库存风险，"
    "区分制度依据与数据依据并给出对应处理建议。"
)
_EXPECTED_PROVIDER_OPERATIONS = {
    "route_request",
    "plan_workflow",
    "plan_data_query",
    "generate_data_answer",
    "select_evidence",
    "review_answerability",
    "generate_grounded_answer",
    "generate_inventory_synthesis",
    "review_workflow",
}


def _attributes(span: object) -> dict[str, object]:
    return {attribute.key: attribute.value for attribute in span.attributes}  # type: ignore[attr-defined]


def _safe_failure_summary(error_code: str | None, spans: tuple[object, ...]) -> str:
    span_errors = [
        f"{getattr(span, 'operation', 'unknown')}={getattr(span, 'error_code', None)}"
        for span in spans
        if getattr(span, "error_code", None) is not None
    ]
    return "\n".join((f"result={error_code}", *span_errors))


@pytest.mark.skipif(
    os.environ.get(_RUN_ENV) != "1",
    reason="set RUN_M8D_B2_REAL_JOINT=1 for the real joint Runtime smoke",
)
@pytest.mark.asyncio
async def test_real_controlled_inventory_runtime_is_bounded_grounded_and_private() -> None:
    settings = Settings()
    assert settings.controlled_workflow_enabled is True
    assert settings.memory_mode == "disabled"
    assert settings.knowledge_dataset_root is not None

    runtime = await build_bootstrapped_runtime(create_configured_runtime_builder(settings))
    sink = InMemoryTraceSink(max_traces=1)
    executor = runtime.executor.with_trace_dispatcher(BestEffortTraceDispatcher([sink]))
    try:
        response = await executor.execute(
            FormalRequest(
                request_id="m8d-b2-real-mixed-1",
                user_query=_USER_QUERY,
            )
        )
    finally:
        await runtime.aclose()

    traces = sink.snapshot()
    assert len(traces) == 1
    trace = traces[0]
    assert runtime.closed is True
    assert response.result.status is CoordinatorStatus.COMPLETED, _safe_failure_summary(
        response.result.error_code,
        trace.spans,
    )
    assert response.result.route is RequestRoute.MIXED
    assert response.result.skill_name == "inventory-risk-diagnosis"
    assert response.result.answer is not None
    assert "风险" in response.result.answer
    assert any(term in response.result.answer for term in ("建议", "补货", "处理"))
    assert any(citation.startswith("[D") for citation in response.result.citations)
    assert any(citation.startswith("[E") for citation in response.result.citations)
    assert all(citation in response.result.answer for citation in response.result.citations)

    assert trace.final_status is SpanStatus.COMPLETED
    assert trace.route == "mixed"
    assert trace.skill_name == "inventory-risk-diagnosis"
    assert response.trace is not None and response.trace.trace_id == trace.trace_id

    provider_spans = [span for span in trace.spans if span.stage is TraceStage.PROVIDER_CALL]
    assert len(provider_spans) == 9
    assert {span.operation for span in provider_spans} == _EXPECTED_PROVIDER_OPERATIONS
    assert all(span.status is SpanStatus.COMPLETED for span in provider_spans)
    assert all(
        span.operation not in {"select_tool", "generate_tool_answer"} for span in provider_spans
    )

    assert sum(span.stage is TraceStage.PLAN_STEP_EXECUTION for span in trace.spans) == 1
    assert sum(span.stage is TraceStage.WORKFLOW_REVIEW for span in trace.spans) == 1
    assert sum(span.stage is TraceStage.DATA_ACCESS for span in trace.spans) == 1
    assert (
        sum(
            span.stage is TraceStage.TOOL_EXECUTION and span.operation == "execute_preselected_tool"
            for span in trace.spans
        )
        == 2
    )
    assert any(
        span.stage is TraceStage.RETRIEVAL and span.operation == "hybrid_retrieve"
        for span in trace.spans
    )
    assert any(span.operation == "parent_expansion" for span in trace.spans)
    assert any(span.stage is TraceStage.RERANKING for span in trace.spans)

    planning = next(span for span in trace.spans if span.stage is TraceStage.PLANNING)
    reviewer = next(span for span in trace.spans if span.stage is TraceStage.WORKFLOW_REVIEW)
    selection = next(span for span in trace.spans if span.stage is TraceStage.EVIDENCE_SELECTION)
    assert _attributes(planning)["plan_step_count"] == 1
    assert _attributes(reviewer)["reviewer_outcome"] == "accept"
    assert _attributes(reviewer)["reviewer_calls_used"] == 1
    assert _attributes(reviewer)["repair_attempts"] == 0
    assert _attributes(selection)["selected_evidence_count"] > 0
    assert (
        _attributes(selection)["selected_evidence_count"]
        <= _attributes(selection)["candidate_evidence_count"]
    )

    serialized_trace = json.dumps(trace.model_dump(mode="json"), ensure_ascii=False)
    forbidden = (
        _USER_QUERY,
        response.result.answer,
        "SELECT ",
        "mysql://",
        "authorization",
        "api_key",
        "password",
    )
    assert all(value.lower() not in serialized_trace.lower() for value in forbidden)
