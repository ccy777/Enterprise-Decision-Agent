"""High-level tool adapters that reuse existing Agent entry points only."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from decision_agent.agents.data_answer_generator import DataAnswerGenerator
from decision_agent.agents.data_query_planner import DataQueryPlanner
from decision_agent.observability.execution import TraceSpanRecorder
from decision_agent.observability.models import TraceContext
from decision_agent.security import SecurityContext, SecurityErrorCode
from decision_agent.tool_calling.models import AgentToolResult
from decision_agent.workflows.data_agent import (
    DataAgentState,
    DataAgentStatus,
    EnterpriseDataClient,
    run_data_agent,
)
from decision_agent.workflows.knowledge_qa import Answerability, KnowledgeQAState, run_knowledge_qa


class HighLevelAgentTool(Protocol):
    """The only tool shape visible to the native tool-calling runtime."""

    async def run(self, *, query: str) -> AgentToolResult:
        """Run one existing high-level Agent and project its safe public result."""


class KnowledgeAgentTool:
    """Adapter over the existing compiled Knowledge QA graph; no retrieval is reimplemented."""

    def __init__(self, *, graph: object) -> None:
        self._graph = graph

    async def run(self, *, query: str) -> AgentToolResult:
        state = await run_knowledge_qa(self._graph, user_query=query)
        return _knowledge_result(state)

    async def run_with_trace(
        self,
        *,
        query: str,
        trace_recorder: TraceSpanRecorder | None = None,
        trace_parent_context: TraceContext | None = None,
        security_context: SecurityContext | None = None,
    ) -> AgentToolResult:
        """Run the reusable graph with optional request-local observability context."""
        if security_context is not None and security_context.knowledge_scope is None:
            return AgentToolResult(
                status="failed",
                error_code=SecurityErrorCode.KNOWLEDGE_SCOPE_MISSING.value,
            )
        state = await run_knowledge_qa(
            self._graph,
            user_query=query,
            trace_recorder=trace_recorder,
            trace_parent_context=trace_parent_context,
            knowledge_scope=None if security_context is None else security_context.knowledge_scope,
        )
        return _knowledge_result(state)

    async def run_with_scope(
        self, *, query: str, security_context: SecurityContext
    ) -> AgentToolResult:
        """Run only with a context propagated from the secured formal request path."""
        if security_context.knowledge_scope is None:
            return AgentToolResult(
                status="failed",
                error_code=SecurityErrorCode.KNOWLEDGE_SCOPE_MISSING.value,
            )
        state = await run_knowledge_qa(
            self._graph,
            user_query=query,
            knowledge_scope=security_context.knowledge_scope,
        )
        return _knowledge_result(state)


class DataAgentTool:
    """Adapter over the existing Data Agent, which keeps its MCP-only data path."""

    def __init__(
        self,
        *,
        planner: DataQueryPlanner,
        enterprise_data_client_factory: Callable[[], EnterpriseDataClient],
        answer_generator: DataAnswerGenerator,
    ) -> None:
        self._planner = planner
        self._enterprise_data_client_factory = enterprise_data_client_factory
        self._answer_generator = answer_generator

    async def run(self, *, query: str) -> AgentToolResult:
        state = await run_data_agent(
            query=query,
            planner=self._planner,
            enterprise_data_client_factory=self._enterprise_data_client_factory,
            answer_generator=self._answer_generator,
        )
        return _data_result(state)

    async def run_with_trace(
        self,
        *,
        query: str,
        trace_recorder: TraceSpanRecorder | None = None,
        trace_parent_context: TraceContext | None = None,
        security_context: SecurityContext | None = None,
    ) -> AgentToolResult:
        """Run the existing Data workflow with optional request-local trace context."""
        if security_context is not None and security_context.data_scope is None:
            return AgentToolResult(
                status="failed",
                error_code=SecurityErrorCode.DATA_SCOPE_MISSING.value,
            )
        state = await run_data_agent(
            query=query,
            planner=self._planner,
            enterprise_data_client_factory=self._enterprise_data_client_factory,
            answer_generator=self._answer_generator,
            trace_recorder=trace_recorder,
            trace_parent_context=trace_parent_context,
            data_scope=None if security_context is None else security_context.data_scope,
        )
        return _data_result(state)

    async def run_with_scope(
        self, *, query: str, security_context: SecurityContext
    ) -> AgentToolResult:
        """Fail before MCP construction unless the secured request grants Data access."""
        scope = security_context.data_scope
        if scope is None:
            error_code = SecurityErrorCode.DATA_SCOPE_MISSING.value
        elif not scope.permits(domain="enterprise_operations"):
            error_code = SecurityErrorCode.DATA_SCOPE_VIOLATION.value
        else:
            error_code = None
        if error_code is not None:
            return AgentToolResult(status="failed", error_code=error_code)
        state = await run_data_agent(
            query=query,
            planner=self._planner,
            enterprise_data_client_factory=self._enterprise_data_client_factory,
            answer_generator=self._answer_generator,
            data_scope=scope,
        )
        return _data_result(state)


def _knowledge_result(state: KnowledgeQAState) -> AgentToolResult:
    if state.answerability is Answerability.FAILED or state.answer is None:
        return AgentToolResult(
            status="failed",
            error_code=state.errors[0].code if state.errors else "knowledge_agent_failed",
        )
    return AgentToolResult(status="succeeded", answer=state.answer, citations=state.citations)


def _data_result(state: DataAgentState) -> AgentToolResult:
    if state.status is DataAgentStatus.FAILED or state.answer is None:
        return AgentToolResult(
            status="failed",
            error_code=state.errors[0].code if state.errors else "data_agent_failed",
        )
    return AgentToolResult(status="succeeded", answer=state.answer, citations=state.citations)
