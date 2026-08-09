"""Structured real-provider adapters for the controlled workflow."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any, Protocol

from pydantic import ValidationError

from decision_agent.agent_workflow.models import ExecutionPlan, ReviewerDecision
from decision_agent.exceptions import DecisionAgentError
from decision_agent.observability import (
    SpanStatus,
    TraceContext,
    TraceSpanRecorder,
    TraceStage,
    complete_recorded_span,
    start_recorded_span,
)
from decision_agent.observability.provider import (
    ProviderTraceMetadata,
    provider_failure_attributes,
    provider_response_attributes,
)
from decision_agent.providers import extract_stopped_message_content


class StructuredWorkflowProvider(Protocol):
    """Existing OpenAI-compatible non-tool completion surface."""

    async def complete_chat(
        self, *, messages: list[dict[str, object]], response_format: dict[str, str]
    ) -> dict[str, Any]: ...

    def provider_trace_metadata(self) -> ProviderTraceMetadata: ...


class WorkflowProviderError(DecisionAgentError):
    """Safe provider/structured-output error without retaining raw completion text."""

    def __init__(self, code: str) -> None:
        super().__init__("Controlled workflow provider result is unavailable")
        self.code = code


class OpenAICompatibleWorkflowPlanner:
    """Call the existing provider once and parse only the bounded ExecutionPlan schema."""

    def __init__(self, *, provider: StructuredWorkflowProvider) -> None:
        self._provider = provider

    async def plan_with_trace(self, *, request, trace_recorder, trace_parent_context):  # type: ignore[no-untyped-def]
        payload = await _complete(
            provider=self._provider,
            messages=[
                {"role": "system", "content": _PLANNER_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(request.model_dump(), ensure_ascii=False, sort_keys=True),
                },
            ],
            operation="plan_workflow",
            trace_recorder=trace_recorder,
            trace_parent_context=trace_parent_context,
        )
        return _parse(payload, ExecutionPlan, "workflow_planner_schema_invalid")

    async def plan(self, *, request):  # type: ignore[no-untyped-def]
        return await self.plan_with_trace(
            request=request, trace_recorder=None, trace_parent_context=None
        )


class OpenAICompatibleWorkflowReviewer:
    """Call the existing provider once and parse only the bounded ReviewerDecision schema."""

    def __init__(self, *, provider: StructuredWorkflowProvider) -> None:
        self._provider = provider

    async def review_with_trace(self, *, request, trace_recorder, trace_parent_context):  # type: ignore[no-untyped-def]
        payload = await _complete(
            provider=self._provider,
            messages=[
                {"role": "system", "content": _REVIEWER_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(request.model_dump(), ensure_ascii=False, sort_keys=True),
                },
            ],
            operation="review_workflow",
            trace_recorder=trace_recorder,
            trace_parent_context=trace_parent_context,
        )
        return _parse(payload, ReviewerDecision, "workflow_reviewer_schema_invalid")

    async def review(self, *, request):  # type: ignore[no-untyped-def]
        return await self.review_with_trace(
            request=request, trace_recorder=None, trace_parent_context=None
        )


async def _complete(
    *,
    provider: StructuredWorkflowProvider,
    messages: list[dict[str, object]],
    operation: str,
    trace_recorder: TraceSpanRecorder | None,
    trace_parent_context: TraceContext | None,
) -> dict[str, Any]:
    metadata = provider.provider_trace_metadata()
    span = start_recorded_span(
        trace_recorder,
        stage=TraceStage.PROVIDER_CALL,
        component="provider",
        operation=operation,
        parent_context=trace_parent_context,
        attributes=provider_failure_attributes(metadata=metadata, operation=operation),
    )
    try:
        payload = await provider.complete_chat(
            messages=messages, response_format={"type": "json_object"}
        )
    except asyncio.CancelledError:
        complete_recorded_span(
            trace_recorder,
            span,
            status=SpanStatus.CANCELLED,
            attributes=provider_failure_attributes(metadata=metadata, operation=operation),
        )
        raise
    except Exception:
        complete_recorded_span(
            trace_recorder,
            span,
            status=SpanStatus.FAILED,
            error_code="workflow_provider_failed",
            attributes=provider_failure_attributes(metadata=metadata, operation=operation),
        )
        raise WorkflowProviderError("workflow_provider_failed") from None
    complete_recorded_span(
        trace_recorder,
        span,
        status=SpanStatus.COMPLETED,
        attributes=provider_response_attributes(
            metadata=metadata, operation=operation, payload=payload
        ),
    )
    return payload


def _parse(payload: Mapping[str, Any], model, code: str):  # type: ignore[no-untyped-def]
    try:
        return model.model_validate(json.loads(extract_stopped_message_content(payload)))
    except (TypeError, ValueError, ValidationError, json.JSONDecodeError):
        raise WorkflowProviderError(code) from None


_PLANNER_PROMPT = """Return exactly this JSON structure and no other keys:
{"plan_id":"plan_1","plan_version":"m8b-v1",
"objective_type":"mixed_inventory_diagnosis","steps":[{"step_id":"inventory_step",
"sequence":1,"skill_name":"inventory-risk-diagnosis",
"objective_type":"mixed_inventory_diagnosis","depends_on":[],
"required_output_type":"mixed_diagnosis","optional":false}],
"max_execution_rounds":1,"max_skill_calls":1}
The supplied request is classification context only. Never output a query, subquery, parameters,
tool arguments, SQL, answer, rationale, prompt, markdown, or any extra field."""

_REVIEWER_PROMPT = """Return exactly one JSON object with only these keys:
{"outcome":"accept|repair|unanswerable|fail_closed",
"accepted_step_id":"step id or null","repair_target":"step id or null",
"reason_code":"lowercase_safe_code","final_status":"accepted|repair|unanswerable|failed"}
For one completed answer-bearing step with citations, use accept/accepted and its step id.
For insufficient evidence use unanswerable/unanswerable. For a technical or invalid result use
fail_closed/failed. Never create a plan, choose a Skill, alter a query, invoke a tool, include an
answer, reasoning, markdown, or any extra field."""
