"""Lifecycle, request isolation, and safety tests for the Data Agent workflow."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from itertools import count

import pytest
from pydantic import ValidationError

import decision_agent.workflows.data_agent as data_agent_workflow
from decision_agent.agents.data_answer_generator import DataAnswerDraft
from decision_agent.agents.data_query_planner import DataQueryPlan
from decision_agent.data.models import QueryAudit, SafeQueryResult
from decision_agent.domain import ErrorRecord
from decision_agent.mcp_client.contracts import (
    BusinessDefinitions,
    EnterpriseSchema,
    MCPQueryResult,
)
from decision_agent.mcp_client.errors import EnterpriseDataMCPError
from decision_agent.observability import TraceCollector, TraceContext, TraceExecution
from decision_agent.security import DataScope
from decision_agent.workflows.data_agent import DataAgentState, DataAgentStatus, run_data_agent


class StubPlanner:
    def __init__(self, plan: DataQueryPlan) -> None:
        self.result = plan
        self.calls = 0
        self.schemas: list[dict[str, list[str]]] = []
        self.definitions: list[dict[str, str]] = []

    async def plan(
        self,
        *,
        user_query: str,
        enterprise_schema: dict[str, list[str]],
        business_definitions: dict[str, str],
    ) -> DataQueryPlan:
        self.calls += 1
        self.schemas.append(enterprise_schema)
        self.definitions.append(business_definitions)
        return self.result


class StubMCPClient:
    def __init__(self, result: SafeQueryResult) -> None:
        self.result = result
        self.enter_calls = 0
        self.exit_calls = 0
        self.schema_calls = 0
        self.definition_calls = 0
        self.query_calls = 0

    async def __aenter__(self) -> StubMCPClient:
        self.enter_calls += 1
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.exit_calls += 1

    async def get_enterprise_schema(self) -> EnterpriseSchema:
        self.schema_calls += 1
        return EnterpriseSchema(tables={"products": ["product_id", "product_name"]})

    async def get_business_definitions(self) -> BusinessDefinitions:
        self.definition_calls += 1
        return BusinessDefinitions(definitions={"natural_month": "half-open month"})

    async def execute_safe_query(self, sql: str) -> MCPQueryResult:
        self.query_calls += 1
        return MCPQueryResult(
            columns=self.result.columns,
            rows=self.result.rows,
            row_count=self.result.row_count,
            truncated=self.result.truncated,
            normalized_sql=self.result.audit.normalized_sql,
            accessed_tables=self.result.accessed_tables,
            elapsed_ms=self.result.elapsed_ms,
            error_code=self.result.error_code,
        )


class CoordinatedMCPClient(StubMCPClient):
    def __init__(
        self,
        result: SafeQueryResult,
        *,
        both_entered: asyncio.Event,
        hold_query: asyncio.Event | None = None,
        on_enter: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(result)
        self._both_entered = both_entered
        self._hold_query = hold_query
        self._on_enter = on_enter

    async def __aenter__(self) -> CoordinatedMCPClient:
        await super().__aenter__()
        if self._on_enter is not None:
            self._on_enter()
        return self

    async def execute_safe_query(self, sql: str) -> MCPQueryResult:
        result = await super().execute_safe_query(sql)
        await self._both_entered.wait()
        if self._hold_query is not None:
            await self._hold_query.wait()
        return result


class StubGenerator:
    def __init__(self, draft: DataAnswerDraft) -> None:
        self.draft = draft
        self.calls = 0

    async def generate(
        self, *, user_query: str, data_evidence: tuple[object, ...]
    ) -> DataAnswerDraft:
        self.calls += 1
        return self.draft


class FailingMCPClient(StubMCPClient):
    async def get_enterprise_schema(self) -> EnterpriseSchema:
        raise EnterpriseDataMCPError("mcp_server_unavailable")


class MultiTableMCPClient(StubMCPClient):
    async def get_enterprise_schema(self) -> EnterpriseSchema:
        self.schema_calls += 1
        return EnterpriseSchema(
            tables={
                "products": ["product_id", "product_name"],
                "orders": ["order_id", "product_id"],
            }
        )


def successful_result(
    *, rows: list[list[object]], accessed_tables: list[str] | None = None
) -> SafeQueryResult:
    return SafeQueryResult(
        columns=["product_name"],
        rows=rows,
        row_count=len(rows),
        truncated=False,
        elapsed_ms=1.0,
        accessed_tables=accessed_tables if accessed_tables is not None else ["products"],
        audit=QueryAudit(
            request_id="00000000-0000-0000-0000-000000000001",
            normalized_sql="SELECT product_name FROM products LIMIT 1",
            allowed=True,
            elapsed_ms=1.0,
            row_count=len(rows),
        ),
    )


def failed_result(*, code: str) -> SafeQueryResult:
    return SafeQueryResult(
        elapsed_ms=1.0,
        audit=QueryAudit(
            request_id="00000000-0000-0000-0000-000000000001",
            allowed=False,
            rejection_code=code,
            elapsed_ms=1.0,
            row_count=0,
        ),
        error_code=code,
    )


async def _run(
    *,
    query: str,
    planner: StubPlanner,
    client_factory: object,
    generator: StubGenerator,
) -> DataAgentState:
    return await run_data_agent(
        query=query,
        planner=planner,
        enterprise_data_client_factory=client_factory,  # type: ignore[arg-type]
        answer_generator=generator,
    )


@pytest.mark.parametrize(
    ("status", "kwargs"),
    [
        (DataAgentStatus.RUNNING, {"answer": ""}),
        (DataAgentStatus.PLANNED_READY, {}),
        (DataAgentStatus.NEEDS_CLARIFICATION, {}),
        (DataAgentStatus.UNSUPPORTED, {}),
        (DataAgentStatus.QUERIED, {}),
        (DataAgentStatus.ANSWERABLE_FINAL, {}),
        (DataAgentStatus.EMPTY_RESULT_FINAL, {}),
        (DataAgentStatus.FAILED, {}),
    ],
)
def test_state_rejects_incomplete_or_untrusted_lifecycle_values(
    status: DataAgentStatus, kwargs: dict[str, object]
) -> None:
    with pytest.raises(ValidationError):
        DataAgentState(query="question", status=status, **kwargs)


def test_running_and_failed_state_contracts_are_explicit() -> None:
    assert DataAgentState(query="question").status is DataAgentStatus.RUNNING
    failed = DataAgentState(
        query="question",
        status=DataAgentStatus.FAILED,
        errors=[ErrorRecord(code="safe_query_execution_failed", message="safe")],
    )
    assert failed.answer is None
    assert failed.citations == []


@pytest.mark.asyncio
async def test_ready_query_uses_one_factory_client_and_builds_d1() -> None:
    planner = StubPlanner(
        DataQueryPlan(
            status="ready",
            intent="product query",
            sql="SELECT product_name FROM products",
            decision_reason="the required data is available",
        )
    )
    service = StubMCPClient(successful_result(rows=[["Aster"]]))
    generator = StubGenerator(
        DataAnswerDraft(answer="The product is Aster.[D1]", citations=["[D1]"])
    )
    state = await _run(
        query="What is the product?",
        planner=planner,
        client_factory=lambda: service,
        generator=generator,
    )
    assert state.status is DataAgentStatus.ANSWERABLE_FINAL
    assert state.data_evidence[0].evidence_id == "D1"
    assert state.citations == ["[D1]"]
    assert (service.enter_calls, service.exit_calls, service.query_calls) == (1, 1, 1)
    assert planner.schemas == [{"products": ["product_id", "product_name"]}]
    assert planner.definitions == [{"natural_month": "half-open month"}]


@pytest.mark.asyncio
async def test_legacy_generator_still_runs_once_when_request_has_trace() -> None:
    planner = StubPlanner(
        DataQueryPlan(
            status="ready",
            intent="product query",
            sql="SELECT product_name FROM products",
            decision_reason="the required data is available",
        )
    )
    service = StubMCPClient(successful_result(rows=[["Aster"]]))
    generator = StubGenerator(
        DataAnswerDraft(answer="The product is Aster.[D1]", citations=["[D1]"])
    )
    span_ids = count()
    trace = TraceExecution(
        collector=TraceCollector(
            context=TraceContext.create(request_id="legacy-generator"),
            id_factory=lambda: f"span-{next(span_ids)}",
        ),
        dispatcher=None,
    )

    state = await run_data_agent(
        query="What is the product?",
        planner=planner,
        enterprise_data_client_factory=lambda: service,
        answer_generator=generator,
        trace_recorder=trace,
    )

    assert state.status is DataAgentStatus.ANSWERABLE_FINAL
    assert generator.calls == service.query_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["needs_clarification", "unsupported"])
async def test_non_query_plans_short_circuit_sql_and_generator(status: str) -> None:
    plan = DataQueryPlan(
        status=status,
        intent="scope decision",
        decision_reason="information is missing or unsupported",
        missing_information="reporting period" if status == "needs_clarification" else None,
    )
    service = StubMCPClient(successful_result(rows=[]))
    generator = StubGenerator(DataAnswerDraft(answer="x[D1]", citations=["[D1]"]))
    state = await _run(
        query="question",
        planner=StubPlanner(plan),
        client_factory=lambda: service,
        generator=generator,
    )
    assert state.status in (DataAgentStatus.NEEDS_CLARIFICATION, DataAgentStatus.UNSUPPORTED)
    assert service.query_calls == generator.calls == 0


@pytest.mark.asyncio
async def test_planner_language_failure_does_not_access_database() -> None:
    planner = StubPlanner(
        DataQueryPlan(
            status="ready",
            intent="query",
            sql="SELECT product_name FROM products",
            decision_reason="query is defined",
        )
    )
    service = StubMCPClient(successful_result(rows=[]))
    generator = StubGenerator(DataAnswerDraft(answer="x[D1]", citations=["[D1]"]))
    state = await _run(
        query="\u4e2d\u6587\u95ee\u9898",
        planner=planner,
        client_factory=lambda: service,
        generator=generator,
    )
    assert state.status is DataAgentStatus.FAILED
    assert state.errors[0].code == "data_planner_language_mismatch"
    assert service.query_calls == generator.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("error_code", ["write_statement_not_allowed", "database_unavailable"])
async def test_safe_query_failure_skips_generator(error_code: str) -> None:
    plan = DataQueryPlan(
        status="ready",
        intent="query",
        sql="SELECT product_name FROM products",
        decision_reason="query is defined",
    )
    service = StubMCPClient(failed_result(code=error_code))
    generator = StubGenerator(DataAnswerDraft(answer="x[D1]", citations=["[D1]"]))
    state = await _run(
        query="question",
        planner=StubPlanner(plan),
        client_factory=lambda: service,
        generator=generator,
    )
    assert state.status is DataAgentStatus.FAILED
    assert state.errors[0].code == error_code
    assert service.query_calls == 1
    assert generator.calls == 0


@pytest.mark.asyncio
async def test_mcp_initialization_failure_skips_planner_query_and_generator() -> None:
    planner = StubPlanner(
        DataQueryPlan(
            status="ready",
            intent="query",
            sql="SELECT product_name FROM products",
            decision_reason="query is defined",
        )
    )
    service = FailingMCPClient(successful_result(rows=[]))
    generator = StubGenerator(DataAnswerDraft(answer="x[D1]", citations=["[D1]"]))
    state = await _run(
        query="question",
        planner=planner,
        client_factory=lambda: service,
        generator=generator,
    )
    assert state.status is DataAgentStatus.FAILED
    assert state.errors[0].code == "mcp_server_unavailable"
    assert planner.calls == service.query_calls == generator.calls == 0


@pytest.mark.asyncio
async def test_denied_data_scope_does_not_create_or_open_mcp_client() -> None:
    planner = StubPlanner(
        DataQueryPlan(
            status="ready",
            intent="query",
            sql="SELECT product_name FROM products",
            decision_reason="query is defined",
        )
    )
    generator = StubGenerator(DataAnswerDraft(answer="x[D1]", citations=["[D1]"]))
    factory_calls = 0

    def factory() -> StubMCPClient:
        nonlocal factory_calls
        factory_calls += 1
        return StubMCPClient(successful_result(rows=[["Aster"]]))

    state = await run_data_agent(
        query="question",
        planner=planner,
        enterprise_data_client_factory=factory,
        answer_generator=generator,
        data_scope=DataScope(
            tenant_id="tenant-a",
            allowed_domains=frozenset(),
            allowed_resources=frozenset(),
            allowed_query_capabilities=frozenset(),
        ),
    )

    assert state.status is DataAgentStatus.FAILED
    assert state.errors[0].code == "data_scope_violation"
    assert factory_calls == planner.calls == generator.calls == 0


@pytest.mark.asyncio
async def test_data_scope_filters_schema_and_blocks_unscoped_table_before_query() -> None:
    planner = StubPlanner(
        DataQueryPlan(
            status="ready",
            intent="query",
            sql="SELECT order_id FROM orders",
            decision_reason="query is defined",
        )
    )
    service = MultiTableMCPClient(successful_result(rows=[["Aster"]]))
    generator = StubGenerator(DataAnswerDraft(answer="x[D1]", citations=["[D1]"]))

    state = await run_data_agent(
        query="question",
        planner=planner,
        enterprise_data_client_factory=lambda: service,
        answer_generator=generator,
        data_scope=DataScope(
            tenant_id="tenant-a",
            allowed_domains=frozenset({"enterprise_operations"}),
            allowed_resources=frozenset({"products"}),
            allowed_query_capabilities=frozenset({"read"}),
        ),
    )

    assert state.status is DataAgentStatus.FAILED
    assert state.errors[0].code == "data_scope_violation"
    assert planner.schemas == [{"products": ["product_id", "product_name"]}]
    assert service.query_calls == generator.calls == 0


@pytest.mark.asyncio
async def test_result_without_business_table_fails_without_generator() -> None:
    plan = DataQueryPlan(
        status="ready",
        intent="query",
        sql="SELECT 1",
        decision_reason="query is defined",
    )
    service = StubMCPClient(successful_result(rows=[[1]], accessed_tables=[]))
    generator = StubGenerator(DataAnswerDraft(answer="x[D1]", citations=["[D1]"]))
    state = await _run(
        query="question",
        planner=StubPlanner(plan),
        client_factory=lambda: service,
        generator=generator,
    )
    assert state.status is DataAgentStatus.FAILED
    assert state.errors[0].code == "data_query_without_business_table"
    assert state.data_evidence == ()
    assert service.query_calls == 1
    assert generator.calls == 0


@pytest.mark.asyncio
async def test_invalid_generator_citation_fails_without_publishing_an_answer() -> None:
    plan = DataQueryPlan(
        status="ready",
        intent="product query",
        sql="SELECT product_name FROM products",
        decision_reason="the required data is available",
    )
    service = StubMCPClient(successful_result(rows=[["Aster"]]))
    state = await _run(
        query="What is the product?",
        planner=StubPlanner(plan),
        client_factory=lambda: service,
        generator=StubGenerator(
            DataAnswerDraft(answer="The product is Aster.[E1]", citations=["[E1]"])
        ),
    )
    assert state.status is DataAgentStatus.FAILED
    assert state.answer is None
    assert state.data_evidence[0].evidence_id == "D1"
    assert state.errors[0].code == "invalid_data_citation_format"


@pytest.mark.asyncio
async def test_empty_result_is_deterministic_and_skips_generator() -> None:
    plan = DataQueryPlan(
        status="ready",
        intent="product query",
        sql="SELECT product_name FROM products",
        decision_reason="the required data is available",
    )
    service = StubMCPClient(successful_result(rows=[]))
    generator = StubGenerator(DataAnswerDraft(answer="x[D1]", citations=["[D1]"]))
    state = await _run(
        query="What product is missing?",
        planner=StubPlanner(plan),
        client_factory=lambda: service,
        generator=generator,
    )
    assert state.status is DataAgentStatus.EMPTY_RESULT_FINAL
    assert state.citations == ["[D1]"]
    assert generator.calls == 0


@pytest.mark.asyncio
async def test_concurrent_requests_use_distinct_factory_clients_and_do_not_close_each_other() -> (
    None
):
    both_entered = asyncio.Event()
    release_second_query = asyncio.Event()
    entered_count = 0

    def note_entered() -> None:
        nonlocal entered_count
        entered_count += 1
        if entered_count == 2:
            both_entered.set()

    first = CoordinatedMCPClient(
        successful_result(rows=[["Aster"]]),
        both_entered=both_entered,
        on_enter=note_entered,
    )
    second = CoordinatedMCPClient(
        successful_result(rows=[["Boreal"]]),
        both_entered=both_entered,
        hold_query=release_second_query,
        on_enter=note_entered,
    )
    clients = [first, second]
    factory_calls = 0

    def factory() -> StubMCPClient:
        nonlocal factory_calls
        factory_calls += 1
        client = clients.pop(0)
        return client

    planner = StubPlanner(
        DataQueryPlan(
            status="ready",
            intent="query",
            sql="SELECT product_name FROM products",
            decision_reason="query is defined",
        )
    )
    generator = StubGenerator(DataAnswerDraft(answer="result[D1]", citations=["[D1]"]))
    first_request = asyncio.create_task(
        _run(query="one", planner=planner, client_factory=factory, generator=generator)
    )
    second_request = asyncio.create_task(
        _run(query="two", planner=planner, client_factory=factory, generator=generator)
    )
    await asyncio.wait_for(both_entered.wait(), timeout=1)
    await asyncio.wait_for(
        _wait_until(lambda: first.exit_calls == 1 and second.query_calls == 1), timeout=1
    )
    assert second.exit_calls == 0
    release_second_query.set()
    first_state, second_state = await asyncio.gather(first_request, second_request)

    assert factory_calls == 2
    assert first is not second
    assert (first.enter_calls, first.exit_calls, first.schema_calls, first.definition_calls) == (
        1,
        1,
        1,
        1,
    )
    assert (
        second.enter_calls,
        second.exit_calls,
        second.schema_calls,
        second.definition_calls,
    ) == (1, 1, 1, 1)
    assert first.query_calls == second.query_calls == 1
    assert first_state.status is second_state.status is DataAgentStatus.ANSWERABLE_FINAL


async def _wait_until(predicate) -> None:  # type: ignore[no-untyped-def]
    while not predicate():
        await asyncio.sleep(0)


def test_run_data_agent_accepts_factory_without_a_graph_or_client_argument() -> None:
    parameters = inspect.signature(run_data_agent).parameters
    assert "enterprise_data_client_factory" in parameters
    assert "graph" not in parameters
    assert "enterprise_data_client" not in parameters


def test_production_workflow_has_no_direct_safe_query_service_dependency() -> None:
    assert "SafeQueryService" not in data_agent_workflow.__dict__
