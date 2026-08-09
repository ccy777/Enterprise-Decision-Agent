"""Shared, request-local Skill invocation mechanics for Coordinator-owned paths."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

from decision_agent.coordination.models import SkillResult, SkillStatus
from decision_agent.observability import (
    SpanStatus,
    TraceContext,
    TraceSpanRecorder,
    TraceStage,
    complete_recorded_span,
    start_recorded_span,
)
from decision_agent.routing.models import RouterDecision
from decision_agent.security import (
    AuthorizationPolicy,
    SecurityAuthorizationError,
    SecurityContext,
    SecurityErrorCode,
)


async def execute_registered_skill(
    *,
    skill: Any,
    user_query: str,
    decision: RouterDecision,
    context_runtime: Any,
    user_item_id: str,
    memory_item_id: str | None,
    trace_recorder: TraceSpanRecorder | None,
    trace_parent_context: TraceContext | None,
    execution_index: int = 0,
    security_context: SecurityContext | None = None,
    authorization_policy: AuthorizationPolicy | None = None,
) -> tuple[object, bool]:
    """Invoke one selected Skill exactly once with its optional formal extensions."""
    execute_with_trace = getattr(skill, "execute_with_trace", None)
    execute_with_context = getattr(skill, "execute_with_context", None)
    diagnostics_start = len(context_runtime.diagnostics)
    skill_span = start_recorded_span(
        trace_recorder,
        stage=TraceStage.SKILL_EXECUTION,
        component="skill",
        operation="execute_skill",
        parent_context=trace_parent_context,
        attributes={
            "route": decision.route.value,
            "skill_name": skill.definition.name,
            "execution_index": execution_index,
            "selected_skill_count": 1,
        },
    )
    try:
        if authorization_policy is not None:
            if security_context is None:
                raise SecurityAuthorizationError(SecurityErrorCode.UNAUTHENTICATED)
            for tool_name in skill.definition.allowed_tools:
                authorization_policy.require_tool(security_context, tool_name)
        if callable(execute_with_trace):
            trace_kwargs: dict[str, object] = {
                "user_query": user_query,
                "decision": decision,
                "trace_recorder": trace_recorder,
                "trace_parent_context": skill_span,
                "context_runtime": context_runtime if callable(execute_with_context) else None,
                "user_item_id": user_item_id if callable(execute_with_context) else None,
            }
            if security_context is not None and _supports_keyword(
                execute_with_trace, "security_context"
            ):
                trace_kwargs["security_context"] = security_context
            result = await execute_with_trace(**trace_kwargs)
        elif callable(execute_with_context):
            result = await execute_with_context(
                user_query=user_query,
                decision=decision,
                context_runtime=context_runtime,
                user_item_id=user_item_id,
            )
        else:
            result = await skill.execute(user_query=user_query, decision=decision)
    except asyncio.CancelledError:
        complete_recorded_span(trace_recorder, skill_span, status=SpanStatus.CANCELLED)
        raise
    except Exception:
        complete_recorded_span(
            trace_recorder,
            skill_span,
            status=SpanStatus.FAILED,
            error_code="skill_execution_failed",
            attributes={"success": False},
        )
        raise
    if isinstance(result, SkillResult):
        status = (
            SpanStatus.COMPLETED if result.status is SkillStatus.COMPLETED else SpanStatus.FAILED
        )
        complete_recorded_span(
            trace_recorder,
            skill_span,
            status=status,
            error_code=result.error_code if status is SpanStatus.FAILED else None,
            attributes={
                "success": status is SpanStatus.COMPLETED,
                "result_status": result.status.value,
            },
        )
    else:
        complete_recorded_span(
            trace_recorder,
            skill_span,
            status=SpanStatus.FAILED,
            error_code="skill_result_contract_invalid",
            attributes={"success": False},
        )
    if callable(execute_with_context):
        return result, skill_memory_consumed(context_runtime, diagnostics_start, memory_item_id)
    return result, False


def skill_memory_consumed(runtime: Any, diagnostics_start: int, memory_item_id: str | None) -> bool:
    """Return request-scoped selection metadata without exposing Memory content."""
    if memory_item_id is None:
        return False
    return any(
        diagnostic.node_name in {"knowledge", "data"}
        and memory_item_id in diagnostic.selected_item_ids
        for diagnostic in runtime.diagnostics[diagnostics_start:]
    )


def _supports_keyword(method: object, keyword: str) -> bool:
    """Keep legacy isolated Skill fakes compatible with optional security propagation."""
    try:
        parameters = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):
        return False
    return keyword in {parameter.name for parameter in parameters} or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
    )
