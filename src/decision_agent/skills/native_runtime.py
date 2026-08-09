"""Production adapter that reuses the existing Native Tool Calling Runtime."""

from __future__ import annotations

import asyncio

from decision_agent.observability import (
    SpanStatus,
    TraceSpanRecorder,
    TraceStage,
    complete_recorded_span,
    start_recorded_span,
)
from decision_agent.observability.models import TraceContext
from decision_agent.routing.models import RequestRoute, RouterDecision
from decision_agent.security import SecurityContext
from decision_agent.tool_calling.models import (
    NativeToolCallingStatus,
    ToolCallingResult,
)
from decision_agent.tool_calling.runtime import NativeToolCallingModel, run_native_tool_calling
from decision_agent.tool_calling.tools import DataAgentTool, HighLevelAgentTool, KnowledgeAgentTool


class NativeToolCallingSkillExecutor:
    """Bind existing high-level tools once; Skills cannot access their internals."""

    def __init__(
        self,
        *,
        model: NativeToolCallingModel,
        knowledge_tool: HighLevelAgentTool,
        data_tool: HighLevelAgentTool,
    ) -> None:
        self._model = model
        self._knowledge_tool = knowledge_tool
        self._data_tool = data_tool

    async def execute(
        self,
        *,
        user_query: str,
        decision: RouterDecision,
        trace_recorder: TraceSpanRecorder | None = None,
        trace_parent_context: TraceContext | None = None,
        security_context: SecurityContext | None = None,
    ) -> ToolCallingResult:
        return await run_native_tool_calling(
            user_query=user_query,
            decision=decision,
            model=self._model,
            knowledge_tool=self._knowledge_tool,
            data_tool=self._data_tool,
            trace_recorder=trace_recorder,
            trace_parent_context=trace_parent_context,
            security_context=security_context,
        )

    async def execute_with_memory(
        self,
        *,
        user_query: str,
        decision: RouterDecision,
        conversation_memory: str,
        trace_recorder: TraceSpanRecorder | None = None,
        trace_parent_context: TraceContext | None = None,
        security_context: SecurityContext | None = None,
    ) -> ToolCallingResult:
        """Forward only selected untrusted Memory to the real provider boundary."""
        return await run_native_tool_calling(
            user_query=user_query,
            decision=decision,
            model=self._model,
            knowledge_tool=self._knowledge_tool,
            data_tool=self._data_tool,
            conversation_memory=conversation_memory,
            trace_recorder=trace_recorder,
            trace_parent_context=trace_parent_context,
            security_context=security_context,
        )


class PreselectedAgentToolSkillExecutor:
    """Execute the Router-approved child Tool inside the controlled Mixed Skill.

    The controlled workflow already allowlists one inventory-risk Skill, and the
    Router already owns both child queries. Repeating model-driven Tool selection
    and answer copying would add four unrelated Provider calls without adding an
    authorization decision.
    """

    def __init__(
        self,
        *,
        knowledge_tool: HighLevelAgentTool,
        data_tool: HighLevelAgentTool,
    ) -> None:
        self._knowledge_tool = knowledge_tool
        self._data_tool = data_tool

    async def execute(
        self,
        *,
        user_query: str,
        decision: RouterDecision,
        trace_recorder: TraceSpanRecorder | None = None,
        trace_parent_context: TraceContext | None = None,
        security_context: SecurityContext | None = None,
    ) -> ToolCallingResult:
        if not isinstance(user_query, str) or not user_query.strip():
            return _preselected_failure(decision.route, "tool_calling_query_invalid")
        selected = self._preselected_tool(decision)
        if selected is None:
            return _preselected_failure(decision.route, "tool_calling_route_invalid")
        selected_tool, query, tool = selected
        span = start_recorded_span(
            trace_recorder,
            stage=TraceStage.TOOL_EXECUTION,
            component="tool_calling",
            operation="execute_preselected_tool",
            parent_context=trace_parent_context,
            attributes={
                "tool_name": selected_tool,
                "authorized": True,
                "execution_index": 0,
                "query_source": "router_owned",
            },
        )
        try:
            run_with_trace = getattr(tool, "run_with_trace", None)
            if callable(run_with_trace):
                trace_kwargs: dict[str, object] = {
                    "query": query,
                    "trace_recorder": trace_recorder,
                    "trace_parent_context": span,
                }
                if security_context is not None and isinstance(
                    tool, (KnowledgeAgentTool, DataAgentTool)
                ):
                    trace_kwargs["security_context"] = security_context
                result = await run_with_trace(**trace_kwargs)
            else:
                result = await tool.run(query=query)
        except asyncio.CancelledError:
            complete_recorded_span(trace_recorder, span, status=SpanStatus.CANCELLED)
            raise
        except Exception:
            complete_recorded_span(
                trace_recorder,
                span,
                status=SpanStatus.FAILED,
                error_code="agent_tool_execution_failed",
                attributes={"success": False},
            )
            return _preselected_failure(
                decision.route,
                "agent_tool_execution_failed",
                selected_tool=selected_tool,
            )
        if result.status != "succeeded":
            error_code = result.error_code or "agent_tool_execution_failed"
            complete_recorded_span(
                trace_recorder,
                span,
                status=SpanStatus.FAILED,
                error_code=error_code,
                attributes={"success": False, "result_status": "failed"},
            )
            return _preselected_failure(
                decision.route,
                error_code,
                selected_tool=selected_tool,
            )
        complete_recorded_span(
            trace_recorder,
            span,
            status=SpanStatus.COMPLETED,
            attributes={"success": True, "result_status": "succeeded"},
        )
        return ToolCallingResult(
            status=NativeToolCallingStatus.COMPLETED,
            route=decision.route,
            selected_tool=selected_tool,
            tool_call_id="server_preselected_tool",
            answer=result.answer,
            citations=result.citations,
            steps=2,
        )

    async def execute_with_memory(
        self,
        *,
        user_query: str,
        decision: RouterDecision,
        conversation_memory: str,
        trace_recorder: TraceSpanRecorder | None = None,
        trace_parent_context: TraceContext | None = None,
        security_context: SecurityContext | None = None,
    ) -> ToolCallingResult:
        """Accept the disabled-memory projection without discarding real Memory."""
        if conversation_memory.strip():
            return _preselected_failure(
                decision.route,
                "tool_calling_memory_unsupported",
            )
        return await self.execute(
            user_query=user_query,
            decision=decision,
            trace_recorder=trace_recorder,
            trace_parent_context=trace_parent_context,
            security_context=security_context,
        )

    def _preselected_tool(
        self,
        decision: RouterDecision,
    ) -> tuple[str, str, HighLevelAgentTool] | None:
        if decision.route is RequestRoute.KNOWLEDGE and decision.knowledge_subquery is not None:
            return (
                "run_knowledge_agent",
                decision.knowledge_subquery,
                self._knowledge_tool,
            )
        if decision.route is RequestRoute.DATA and decision.data_subquery is not None:
            return ("run_data_agent", decision.data_subquery, self._data_tool)
        return None


def _preselected_failure(
    route: RequestRoute,
    error_code: str,
    *,
    selected_tool: str | None = None,
) -> ToolCallingResult:
    return ToolCallingResult(
        status=NativeToolCallingStatus.FAILED,
        route=route,
        selected_tool=selected_tool,
        tool_call_id="server_preselected_tool" if selected_tool is not None else None,
        steps=1 if selected_tool is not None else 0,
        error_code=error_code,
    )
