"""The bounded LangGraph loop for one enterprise data query and answer."""

from __future__ import annotations

import asyncio
import inspect
import re
from collections.abc import Callable
from enum import StrEnum
from typing import Protocol
from uuid import UUID, uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.errors import NodeCancelledError
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field, model_validator

from decision_agent.agents.data_answer_generator import (
    DataAnswerDraft,
    DataAnswerGenerationError,
    DataAnswerGenerator,
    validate_data_citations,
)
from decision_agent.agents.data_query_planner import (
    DataPlanStatus,
    DataQueryPlan,
    DataQueryPlanner,
    DataQueryPlanningError,
    validate_data_query_plan,
)
from decision_agent.data_agent.models import DataEvidence
from decision_agent.domain import ErrorRecord
from decision_agent.mcp_client.contracts import EnterpriseSchema, MCPQueryResult
from decision_agent.mcp_client.errors import EnterpriseDataMCPError
from decision_agent.observability.execution import (
    TraceSpanRecorder,
    complete_recorded_span,
    start_recorded_span,
)
from decision_agent.observability.models import TraceContext
from decision_agent.observability.stages import SpanStatus, TraceStage
from decision_agent.security import DataScope


class DataAgentStatus(StrEnum):
    """Explicit lifecycle values for the single-query data loop."""

    RUNNING = "running"
    PLANNED_READY = "planned-ready"
    NEEDS_CLARIFICATION = "needs-clarification"
    UNSUPPORTED = "unsupported"
    QUERIED = "queried"
    ANSWERABLE_FINAL = "answerable-final"
    EMPTY_RESULT_FINAL = "empty-result-final"
    FAILED = "failed"


class DataAgentState(BaseModel):
    """Serializable state; providers and database clients remain outside graph state."""

    model_config = ConfigDict(extra="forbid")

    request_id: UUID = Field(default_factory=uuid4)
    query: str = Field(min_length=1)
    status: DataAgentStatus = DataAgentStatus.RUNNING
    plan_status: DataPlanStatus | None = None
    intent: str | None = None
    planned_sql: str | None = None
    decision_reason: str | None = None
    missing_information: str | None = None
    data_evidence: tuple[DataEvidence, ...] = ()
    answer: str | None = None
    citations: list[str] = Field(default_factory=list)
    errors: list[ErrorRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_terminal_contract(self) -> DataAgentState:
        if self.status is DataAgentStatus.RUNNING:
            if (
                self.plan_status is not None
                or self.intent is not None
                or self.planned_sql is not None
                or self.decision_reason is not None
                or self.missing_information is not None
                or self.answer is not None
            ):
                raise ValueError("running state cannot contain plan or answer fields")
            if self.citations or self.errors or self.data_evidence:
                raise ValueError("running state cannot contain citations, errors, or evidence")
        if self.status is DataAgentStatus.PLANNED_READY and (
            self.plan_status is not DataPlanStatus.READY
            or not self.intent
            or not self.planned_sql
            or not self.decision_reason
            or self.missing_information is not None
            or self.answer is not None
            or self.citations
            or self.errors
            or self.data_evidence
        ):
            raise ValueError("planned-ready state requires only a ready plan")
        if self.status is DataAgentStatus.NEEDS_CLARIFICATION and (
            self.plan_status is not DataPlanStatus.NEEDS_CLARIFICATION
            or not self.intent
            or not self.decision_reason
            or not self.missing_information
            or not self.answer
            or self.planned_sql is not None
            or self.citations
            or self.errors
            or self.data_evidence
        ):
            raise ValueError("clarification state requires a deterministic answer without SQL")
        if self.status is DataAgentStatus.UNSUPPORTED and (
            self.plan_status is not DataPlanStatus.UNSUPPORTED
            or not self.intent
            or not self.decision_reason
            or self.missing_information is not None
            or not self.answer
            or self.planned_sql is not None
            or self.citations
            or self.errors
            or self.data_evidence
        ):
            raise ValueError("unsupported state requires a deterministic answer without SQL")
        if self.status is DataAgentStatus.QUERIED and (
            self.plan_status is not DataPlanStatus.READY
            or not self.intent
            or not self.planned_sql
            or not self.decision_reason
            or not self.data_evidence
            or self.answer is not None
            or self.citations
            or self.errors
        ):
            raise ValueError("queried state requires successful data evidence only")
        if self.status is DataAgentStatus.ANSWERABLE_FINAL and (
            self.plan_status is not DataPlanStatus.READY
            or not self.intent
            or not self.planned_sql
            or not self.decision_reason
            or not self.answer
            or not self.citations
            or not self.data_evidence
            or self.errors
        ):
            raise ValueError("answerable final state requires answer, citations, and evidence")
        if self.status is DataAgentStatus.EMPTY_RESULT_FINAL and (
            self.plan_status is not DataPlanStatus.READY
            or not self.intent
            or not self.planned_sql
            or not self.decision_reason
            or not self.answer
            or self.citations != ["[D1]"]
            or not self.data_evidence
            or self.errors
        ):
            raise ValueError("empty result state requires deterministic D1 answer")
        if self.status is DataAgentStatus.FAILED and (
            self.plan_status is not None
            or self.intent is not None
            or self.planned_sql is not None
            or self.decision_reason is not None
            or self.missing_information is not None
            or self.answer is not None
            or self.citations
            or not self.errors
        ):
            raise ValueError("failed state requires errors and no untrusted output")
        return self


class EnterpriseDataClient(Protocol):
    """Session-scoped MCP boundary used by the Data Agent workflow."""

    async def __aenter__(self) -> EnterpriseDataClient:
        """Open one initialized MCP session."""

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """Close the MCP session and child server process."""

    async def get_enterprise_schema(self):  # type: ignore[no-untyped-def]
        """Return the authorized schema from MCP."""

    async def get_business_definitions(self):  # type: ignore[no-untyped-def]
        """Return canonical business rules from MCP."""

    async def execute_safe_query(self, sql: str) -> MCPQueryResult:
        """Run one statement through MCP and the remote SafeQueryService."""


_TABLE_REFERENCE = re.compile(r"\b(?:from|join)\s+`?([a-zA-Z_][a-zA-Z0-9_]*)`?\b", re.IGNORECASE)


class ScopedEnterpriseDataClient:
    """Constrain the existing MCP client without changing its public Tool Schema."""

    def __init__(self, *, client: EnterpriseDataClient, scope: DataScope) -> None:
        self._client = client
        self._scope = scope
        self._schema_tables: frozenset[str] = frozenset()

    async def __aenter__(self) -> ScopedEnterpriseDataClient:
        await self._client.__aenter__()
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self._client.__aexit__(exc_type, exc, traceback)

    async def get_enterprise_schema(self) -> EnterpriseSchema:
        schema = await self._client.get_enterprise_schema()
        self._schema_tables = frozenset(table.casefold() for table in schema.tables)
        allowed_tables = {
            table: columns for table, columns in schema.tables.items() if self._permits_table(table)
        }
        if not allowed_tables:
            raise EnterpriseDataMCPError("data_scope_violation")
        return EnterpriseSchema(tables=allowed_tables)

    async def get_business_definitions(self):  # type: ignore[no-untyped-def]
        return await self._client.get_business_definitions()

    async def execute_safe_query(self, sql: str) -> MCPQueryResult:
        referenced = frozenset(
            match.group(1).casefold() for match in _TABLE_REFERENCE.finditer(sql)
        )
        if any(
            table in self._schema_tables and not self._permits_table(table) for table in referenced
        ):
            return MCPQueryResult(
                row_count=0,
                truncated=False,
                elapsed_ms=0.0,
                error_code="data_scope_violation",
            )
        result = await self._client.execute_safe_query(sql)
        if any(not self._permits_table(table) for table in result.accessed_tables):
            return MCPQueryResult(
                row_count=0,
                truncated=False,
                elapsed_ms=result.elapsed_ms,
                error_code="data_scope_violation",
            )
        return result

    def _permits_table(self, table: str) -> bool:
        return self._scope.permits(domain="enterprise_operations") and any(
            resource.casefold() == table.casefold() for resource in self._scope.allowed_resources
        )


_SQL_GUARD_DENIAL_CODES = frozenset(
    {
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
    }
)


def build_data_agent_graph(
    *,
    planner: DataQueryPlanner,
    enterprise_data_client: EnterpriseDataClient,
    answer_generator: DataAnswerGenerator,
):
    """Build START → plan → guarded query → answer with terminal conditional short-circuits."""

    async def plan_data_query(
        state: DataAgentState,
        config: RunnableConfig,
    ) -> dict[str, object]:
        try:
            schema = await enterprise_data_client.get_enterprise_schema()
            definitions = await enterprise_data_client.get_business_definitions()
            trace_recorder, trace_parent_context = _trace_from_config(config)
            plan_with_trace = getattr(planner, "plan_with_trace", None)
            if callable(plan_with_trace):
                planned = await plan_with_trace(
                    user_query=state.query,
                    enterprise_schema=schema.tables,
                    business_definitions=definitions.definitions,
                    trace_recorder=trace_recorder,
                    trace_parent_context=trace_parent_context,
                )
            else:
                planned = await planner.plan(
                    user_query=state.query,
                    enterprise_schema=schema.tables,
                    business_definitions=definitions.definitions,
                )
            plan = DataQueryPlan.model_validate(planned)
            validation = validate_data_query_plan(user_query=state.query, plan=plan)
            if not validation.validation_passed:
                return _failed(validation.validation_errors[0], "Data plan failed validation.")
        except EnterpriseDataMCPError as exc:
            return _failed(exc.code, "Enterprise Data MCP could not be completed.")
        except DataQueryPlanningError as exc:
            return _failed(
                exc.subcode,
                "Data planning could not be completed.",
                details=exc.details,
            )
        except Exception:
            return _failed("data_query_planning_failed", "Data planning could not be completed.")
        if plan.status is DataPlanStatus.NEEDS_CLARIFICATION:
            return {
                "status": DataAgentStatus.NEEDS_CLARIFICATION,
                "plan_status": plan.status,
                "intent": plan.intent,
                "decision_reason": plan.decision_reason,
                "missing_information": plan.missing_information,
                "answer": _clarification_answer(state.query, plan.missing_information or ""),
            }
        if plan.status is DataPlanStatus.UNSUPPORTED:
            return {
                "status": DataAgentStatus.UNSUPPORTED,
                "plan_status": plan.status,
                "intent": plan.intent,
                "decision_reason": plan.decision_reason,
                "answer": _unsupported_answer(state.query),
            }
        return {
            "status": DataAgentStatus.PLANNED_READY,
            "plan_status": plan.status,
            "intent": plan.intent,
            "planned_sql": plan.sql,
            "decision_reason": plan.decision_reason,
        }

    async def execute_safe_query(
        state: DataAgentState, config: RunnableConfig
    ) -> dict[str, object]:
        if state.status is not DataAgentStatus.PLANNED_READY or not state.planned_sql:
            return _failed("invalid_data_query_plan", "Data query execution requires a ready plan.")
        trace_recorder, trace_parent_context = _trace_from_config(config)
        access_span = start_recorded_span(
            trace_recorder,
            stage=TraceStage.DATA_ACCESS,
            component="mcp",
            operation="execute_safe_query",
            parent_context=trace_parent_context,
            attributes={"tool_name": "execute_safe_query"},
        )
        try:
            result = await enterprise_data_client.execute_safe_query(state.planned_sql)
        except asyncio.CancelledError:
            complete_recorded_span(trace_recorder, access_span, status=SpanStatus.CANCELLED)
            raise
        except EnterpriseDataMCPError as exc:
            complete_recorded_span(
                trace_recorder,
                access_span,
                status=SpanStatus.FAILED,
                error_code=exc.code,
                attributes={
                    "authorized": True,
                    "success": False,
                    "result_status": "failed",
                    "timeout": exc.code == "mcp_tool_timeout",
                },
            )
            return _failed(exc.code, "Enterprise Data MCP could not be completed.")
        except Exception:
            complete_recorded_span(
                trace_recorder,
                access_span,
                status=SpanStatus.FAILED,
                error_code="mcp_tool_call_failed",
                attributes={"authorized": True, "success": False, "result_status": "failed"},
            )
            return _failed("mcp_tool_call_failed", "Data query could not be completed.")
        if result.error_code is not None:
            denied = result.error_code in _SQL_GUARD_DENIAL_CODES
            complete_recorded_span(
                trace_recorder,
                access_span,
                status=SpanStatus.FAILED,
                error_code=result.error_code,
                attributes={
                    "authorized": not denied,
                    "denied": denied,
                    "argument_validation": "failed" if denied else "passed",
                    "timeout": result.error_code == "query_timeout",
                    "success": False,
                    "result_status": "failed",
                },
            )
            return _failed(result.error_code, "Data query could not be completed.")
        if not result.accessed_tables:
            complete_recorded_span(
                trace_recorder,
                access_span,
                status=SpanStatus.FAILED,
                error_code="data_query_without_business_table",
                attributes={"authorized": True, "success": False, "result_status": "failed"},
            )
            return _failed(
                "data_query_without_business_table",
                "Data query did not access an authorized business table.",
            )
        try:
            evidence = DataEvidence(
                evidence_id="D1",
                normalized_sql=result.normalized_sql or "",
                columns=result.columns,
                rows=result.rows,
                row_count=result.row_count,
                truncated=result.truncated,
                accessed_tables=result.accessed_tables,
                elapsed_ms=result.elapsed_ms,
            )
        except ValueError:
            complete_recorded_span(
                trace_recorder,
                access_span,
                status=SpanStatus.FAILED,
                error_code="safe_query_execution_failed",
                attributes={"authorized": True, "success": False, "result_status": "failed"},
            )
            return _failed("safe_query_execution_failed", "Data query could not be completed.")
        if evidence.row_count == 0:
            complete_recorded_span(
                trace_recorder,
                access_span,
                status=SpanStatus.COMPLETED,
                attributes={
                    "authorized": True,
                    "argument_validation": "passed",
                    "row_count": 0,
                    "result_truncated": evidence.truncated,
                    "success": True,
                    "result_status": "completed",
                },
            )
            return {
                "status": DataAgentStatus.EMPTY_RESULT_FINAL,
                "data_evidence": (evidence,),
                "answer": _empty_answer(state.query),
                "citations": ["[D1]"],
            }
        complete_recorded_span(
            trace_recorder,
            access_span,
            status=SpanStatus.COMPLETED,
            attributes={
                "authorized": True,
                "argument_validation": "passed",
                "row_count": evidence.row_count,
                "result_truncated": evidence.truncated,
                "success": True,
                "result_status": "completed",
            },
        )
        return {"status": DataAgentStatus.QUERIED, "data_evidence": (evidence,)}

    async def generate_data_answer(
        state: DataAgentState, config: RunnableConfig
    ) -> dict[str, object]:
        if state.status is not DataAgentStatus.QUERIED or not state.data_evidence:
            return _failed(
                "invalid_data_query_plan", "Data answer generation requires executed evidence."
            )
        trace_recorder, trace_parent_context = _trace_from_config(config)
        answer_span = start_recorded_span(
            trace_recorder,
            stage=TraceStage.ANSWER_GENERATION,
            component="answer_generation",
            operation="generate_data_answer",
            parent_context=trace_parent_context,
            attributes={"answer_type": "data"},
        )
        try:
            if trace_recorder is None or not _supports_trace_arguments(answer_generator.generate):
                raw_draft = await answer_generator.generate(
                    user_query=state.query, data_evidence=state.data_evidence
                )
            else:
                raw_draft = await answer_generator.generate(
                    user_query=state.query,
                    data_evidence=state.data_evidence,
                    trace_recorder=trace_recorder,
                    trace_parent_context=answer_span,
                )
            draft = DataAnswerDraft.model_validate(raw_draft)
            validation = validate_data_citations(
                evidence_ids=[item.evidence_id for item in state.data_evidence], draft=draft
            )
            if not validation.validation_passed:
                error_code = validation.validation_errors[0]
                complete_recorded_span(
                    trace_recorder,
                    answer_span,
                    status=SpanStatus.FAILED,
                    error_code=error_code,
                    attributes={"success": False, "result_status": "failed"},
                )
                return _failed(
                    error_code,
                    "Data answer failed citation validation.",
                    evidence=state.data_evidence,
                )
            complete_recorded_span(
                trace_recorder,
                answer_span,
                status=SpanStatus.COMPLETED,
                attributes={
                    "citation_count": len(validation.normalized_citations),
                    "success": True,
                    "result_status": "completed",
                },
            )
            return {
                "status": DataAgentStatus.ANSWERABLE_FINAL,
                "answer": draft.answer,
                "citations": validation.normalized_citations,
            }
        except asyncio.CancelledError:
            complete_recorded_span(trace_recorder, answer_span, status=SpanStatus.CANCELLED)
            raise
        except DataAnswerGenerationError as exc:
            complete_recorded_span(
                trace_recorder,
                answer_span,
                status=SpanStatus.FAILED,
                error_code=exc.subcode,
                attributes={"success": False, "result_status": "failed"},
            )
            return _failed(
                exc.subcode,
                "Data answer could not be completed.",
                evidence=state.data_evidence,
            )
        except Exception:
            complete_recorded_span(
                trace_recorder,
                answer_span,
                status=SpanStatus.FAILED,
                error_code="data_answer_generation_failed",
                attributes={"success": False, "result_status": "failed"},
            )
            return _failed(
                "data_answer_generation_failed",
                "Data answer could not be completed.",
                evidence=state.data_evidence,
            )

    def route_after_plan(state: DataAgentState) -> str:
        return "execute_safe_query" if state.status is DataAgentStatus.PLANNED_READY else END

    def route_after_query(state: DataAgentState) -> str:
        return "generate_data_answer" if state.status is DataAgentStatus.QUERIED else END

    builder = StateGraph(DataAgentState)
    builder.add_node("plan_data_query", plan_data_query)
    builder.add_node("execute_safe_query", execute_safe_query)
    builder.add_node("generate_data_answer", generate_data_answer)
    builder.add_edge(START, "plan_data_query")
    builder.add_conditional_edges("plan_data_query", route_after_plan)
    builder.add_conditional_edges("execute_safe_query", route_after_query)
    builder.add_edge("generate_data_answer", END)
    return builder.compile()


async def run_data_agent(
    *,
    query: str,
    planner: DataQueryPlanner,
    enterprise_data_client_factory: Callable[[], EnterpriseDataClient],
    answer_generator: DataAnswerGenerator,
    request_id: UUID | None = None,
    trace_recorder: TraceSpanRecorder | None = None,
    trace_parent_context: TraceContext | None = None,
    data_scope: DataScope | None = None,
) -> DataAgentState:
    """Create one request-scoped MCP client, graph, and serializable final state."""
    initial = DataAgentState(request_id=request_id or uuid4(), query=query)
    if data_scope is not None and not data_scope.permits(domain="enterprise_operations"):
        return DataAgentState(
            request_id=initial.request_id,
            query=query,
            status=DataAgentStatus.FAILED,
            errors=[
                ErrorRecord(
                    code="data_scope_violation",
                    message="Data access is outside the authorized scope.",
                )
            ],
        )
    enterprise_data_client: EnterpriseDataClient = enterprise_data_client_factory()
    if data_scope is not None:
        enterprise_data_client = ScopedEnterpriseDataClient(
            client=enterprise_data_client,
            scope=data_scope,
        )
    async with enterprise_data_client:
        graph = build_data_agent_graph(
            planner=planner,
            enterprise_data_client=enterprise_data_client,
            answer_generator=answer_generator,
        )
        try:
            if trace_recorder is None and trace_parent_context is None:
                result = await graph.ainvoke(initial.model_dump(mode="python"))  # type: ignore[attr-defined]
            else:
                result = await graph.ainvoke(  # type: ignore[attr-defined]
                    initial.model_dump(mode="python"),
                    config={
                        "configurable": {
                            "trace_recorder": trace_recorder,
                            "trace_parent_context": trace_parent_context,
                        }
                    },
                )
        except NodeCancelledError as error:
            if isinstance(error.__cause__, asyncio.CancelledError):
                raise asyncio.CancelledError from error
            raise
    return DataAgentState.model_validate(result)


def _trace_from_config(
    config: RunnableConfig,
) -> tuple[TraceSpanRecorder | None, TraceContext | None]:
    configurable = config.get("configurable")
    if not isinstance(configurable, dict):
        return None, None
    recorder = configurable.get("trace_recorder")
    parent_context = configurable.get("trace_parent_context")
    return (
        recorder if _is_trace_recorder(recorder) else None,
        parent_context if isinstance(parent_context, TraceContext) else None,
    )


def _is_trace_recorder(value: object) -> bool:
    return callable(getattr(value, "start_span", None)) and callable(
        getattr(value, "complete_span", None)
    )


def _supports_trace_arguments(method: object) -> bool:
    """Use optional trace kwargs only when an injected generator accepts them."""
    try:
        parameters = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):
        return False
    names = {parameter.name for parameter in parameters}
    return any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters) or {
        "trace_recorder",
        "trace_parent_context",
    }.issubset(names)


def _failed(
    code: str,
    message: str,
    *,
    evidence: tuple[DataEvidence, ...] = (),
    details: dict[str, str | bool | int | None] | None = None,
) -> dict[str, object]:
    return {
        "status": DataAgentStatus.FAILED,
        "plan_status": None,
        "intent": None,
        "planned_sql": None,
        "decision_reason": None,
        "missing_information": None,
        "data_evidence": evidence,
        "answer": None,
        "citations": [],
        "errors": [ErrorRecord(code=code, message=message, details=details or {})],
    }


def _is_chinese(query: str) -> bool:
    return any("\u4e00" <= character <= "\u9fff" for character in query)


def _clarification_answer(query: str, missing: str) -> str:
    return f"请补充{missing}。" if _is_chinese(query) else f"Please provide {missing}."


def _unsupported_answer(query: str) -> str:
    return (
        "当前经营数据库不包含回答该问题所需的数据。"
        if _is_chinese(query)
        else "The current operations database does not contain the required data."
    )


def _empty_answer(query: str) -> str:
    return (
        "未查询到符合当前条件的数据。[D1]"
        if _is_chinese(query)
        else "No data matched the current conditions.[D1]"
    )
