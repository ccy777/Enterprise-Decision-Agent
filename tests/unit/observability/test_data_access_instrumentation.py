"""Trace contracts for the existing Data workflow MCP access boundary."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from decision_agent.agents.data_answer_generator import DataAnswerDraft
from decision_agent.agents.data_query_planner import DataQueryPlan
from decision_agent.mcp_client.contracts import (
    BusinessDefinitions,
    EnterpriseSchema,
    MCPQueryResult,
)
from decision_agent.observability import SpanStatus, TraceCollector, TraceContext, TraceStage
from decision_agent.workflows.data_agent import DataAgentStatus, run_data_agent


class _Ids:
    def __init__(self, prefix: str) -> None:
        self._prefix = prefix
        self._value = 0

    def __call__(self) -> str:
        self._value += 1
        return f"{self._prefix}_{self._value}"


class _Planner:
    async def plan(self, **_: object) -> DataQueryPlan:
        return DataQueryPlan(
            status="ready",
            intent="data query",
            sql="SELECT PRIVATE_COLUMN FROM products",
            decision_reason="PRIVATE_REASON",
        )


class _Client:
    def __init__(self, result: MCPQueryResult) -> None:
        self._result = result
        self.query_calls = 0

    async def __aenter__(self) -> _Client:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def get_enterprise_schema(self) -> EnterpriseSchema:
        return EnterpriseSchema(tables={"products": ["product_id"]})

    async def get_business_definitions(self) -> BusinessDefinitions:
        return BusinessDefinitions(definitions={"scope": "private"})

    async def execute_safe_query(self, sql: str) -> MCPQueryResult:
        assert sql == "SELECT PRIVATE_COLUMN FROM products"
        self.query_calls += 1
        return self._result


class _Generator:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, **_: object) -> DataAnswerDraft:
        self.calls += 1
        return DataAnswerDraft(answer="PRIVATE_ANSWER[D1]", citations=["[D1]"])


class _BrokenRecorder:
    def start_span(self, **_: object) -> TraceContext:
        raise RuntimeError("PRIVATE_RECORDER_FAILURE")

    def complete_span(self, *_: object, **__: object) -> None:
        raise RuntimeError("PRIVATE_RECORDER_FAILURE")


def _collector(request_id: str) -> tuple[TraceCollector, TraceContext, TraceContext]:
    collector = TraceCollector(
        context=TraceContext.create(
            request_id=request_id, id_factory=lambda: f"trace_{request_id}"
        ),
        utc_now=lambda: datetime(2026, 7, 28, tzinfo=UTC),
        monotonic=lambda: 10.0,
        id_factory=_Ids(request_id),
    )
    root = collector.start_span(stage=TraceStage.REQUEST, component="executor", operation="execute")
    tool = collector.start_span(
        stage=TraceStage.TOOL_EXECUTION,
        component="tool_calling",
        operation="execute_authorized_tool",
        parent_context=root,
    )
    return collector, root, tool


def _finish(collector: TraceCollector, root: TraceContext, tool: TraceContext):
    collector.complete_span(tool, status=SpanStatus.COMPLETED)
    collector.complete_span(root, status=SpanStatus.COMPLETED)
    return collector.finalize(final_status=SpanStatus.COMPLETED)


def _attributes(span: object) -> dict[str, object]:
    return {attribute.key: attribute.value for attribute in span.attributes}  # type: ignore[attr-defined]


def _success_result() -> MCPQueryResult:
    return MCPQueryResult(
        columns=["product_id"],
        rows=[["PRIVATE_ROW"]],
        row_count=1,
        truncated=False,
        normalized_sql="SELECT PRIVATE_COLUMN FROM products",
        accessed_tables=["products"],
        elapsed_ms=1.0,
    )


@pytest.mark.asyncio
async def test_data_access_is_one_tool_child_and_does_not_retain_business_payloads() -> None:
    client = _Client(_success_result())
    generator = _Generator()
    collector, root, tool = _collector("request_1")

    state = await run_data_agent(
        query="PRIVATE_DATA_QUERY",
        planner=_Planner(),
        enterprise_data_client_factory=lambda: client,
        answer_generator=generator,
        trace_recorder=collector,
        trace_parent_context=tool,
    )
    trace = _finish(collector, root, tool)
    access = next(span for span in trace.spans if span.stage is TraceStage.DATA_ACCESS)

    assert state.status is DataAgentStatus.ANSWERABLE_FINAL
    assert (client.query_calls, generator.calls) == (1, 1)
    assert access.parent_span_id == tool.current_span_id
    assert _attributes(access) == {
        "tool_name": "execute_safe_query",
        "authorized": True,
        "argument_validation": "passed",
        "row_count": 1,
        "result_truncated": False,
        "success": True,
        "result_status": "completed",
    }
    serialized = str(trace.model_dump(mode="json"))
    assert all(
        value not in serialized
        for value in ("PRIVATE_DATA_QUERY", "PRIVATE_COLUMN", "PRIVATE_ROW", "PRIVATE_ANSWER")
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_code",
    [
        "dangerous_function_not_allowed",
        "limit_exceeded",
        "locking_read_not_allowed",
        "multiple_statements_not_allowed",
        "sql_not_allowed",
        "sql_parse_failed",
        "system_schema_not_allowed",
        "unauthorized_column",
        "unauthorized_table",
        "wildcard_not_allowed",
        "write_statement_not_allowed",
    ],
)
async def test_sql_guard_rejection_is_denied_and_runs_the_mcp_call_once(
    error_code: str,
) -> None:
    client = _Client(
        MCPQueryResult(
            row_count=0,
            truncated=False,
            elapsed_ms=1.0,
            error_code=error_code,
        )
    )
    generator = _Generator()
    collector, root, tool = _collector("request_2")

    state = await run_data_agent(
        query="PRIVATE_DATA_QUERY",
        planner=_Planner(),
        enterprise_data_client_factory=lambda: client,
        answer_generator=generator,
        trace_recorder=collector,
        trace_parent_context=tool,
    )
    trace = _finish(collector, root, tool)
    access = next(span for span in trace.spans if span.stage is TraceStage.DATA_ACCESS)

    assert state.status is DataAgentStatus.FAILED
    assert state.errors[0].code == error_code
    assert (client.query_calls, generator.calls) == (1, 0)
    assert access.status is SpanStatus.FAILED
    assert _attributes(access)["denied"] is True
    assert _attributes(access)["authorized"] is False
    assert _attributes(access)["argument_validation"] == "failed"


@pytest.mark.asyncio
async def test_data_access_timeout_is_failed_but_not_an_authorization_denial() -> None:
    client = _Client(
        MCPQueryResult(row_count=0, truncated=False, elapsed_ms=1.0, error_code="query_timeout")
    )
    generator = _Generator()
    collector, root, tool = _collector("request_3")

    state = await run_data_agent(
        query="PRIVATE_DATA_QUERY",
        planner=_Planner(),
        enterprise_data_client_factory=lambda: client,
        answer_generator=generator,
        trace_recorder=collector,
        trace_parent_context=tool,
    )
    trace = _finish(collector, root, tool)
    access = next(span for span in trace.spans if span.stage is TraceStage.DATA_ACCESS)

    assert state.status is DataAgentStatus.FAILED
    assert (client.query_calls, generator.calls) == (1, 0)
    assert _attributes(access)["timeout"] is True
    assert _attributes(access)["denied"] is False


@pytest.mark.asyncio
async def test_data_access_recorder_failure_does_not_repeat_business_calls() -> None:
    client = _Client(_success_result())
    generator = _Generator()

    state = await run_data_agent(
        query="PRIVATE_DATA_QUERY",
        planner=_Planner(),
        enterprise_data_client_factory=lambda: client,
        answer_generator=generator,
        trace_recorder=_BrokenRecorder(),
    )

    assert state.status is DataAgentStatus.ANSWERABLE_FINAL
    assert (client.query_calls, generator.calls) == (1, 1)


@pytest.mark.asyncio
async def test_concurrent_data_traces_keep_contexts_and_private_values_isolated() -> None:
    async def run_one(request_id: str, query: str):
        client = _Client(_success_result())
        collector, root, tool = _collector(request_id)
        state = await run_data_agent(
            query=query,
            planner=_Planner(),
            enterprise_data_client_factory=lambda: client,
            answer_generator=_Generator(),
            trace_recorder=collector,
            trace_parent_context=tool,
        )
        return state, client, _finish(collector, root, tool), tool, query

    first, second = await asyncio.gather(
        run_one("request_a", "PRIVATE_QUERY_A"), run_one("request_b", "PRIVATE_QUERY_B")
    )

    for state, client, trace, tool, query in (first, second):
        access = next(span for span in trace.spans if span.stage is TraceStage.DATA_ACCESS)
        assert state.status is DataAgentStatus.ANSWERABLE_FINAL and client.query_calls == 1
        assert access.parent_span_id == tool.current_span_id
        assert query not in str(trace.model_dump(mode="json"))
    assert first[2].trace_id != second[2].trace_id
