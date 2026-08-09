"""Coordinator that routes only to registered single-domain executable Skills."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

from decision_agent.agent_workflow.models import WorkflowStatus
from decision_agent.agent_workflow.workflow import ControlledAgentWorkflow
from decision_agent.context import RequestContextRuntime
from decision_agent.context.conversation_memory import ConversationMemoryProjection
from decision_agent.coordination.models import (
    CoordinatorResult,
    CoordinatorStatus,
    SkillResult,
    SkillStatus,
)
from decision_agent.coordination.skill_execution import (
    execute_registered_skill,
    skill_memory_consumed,
)
from decision_agent.observability import (
    SpanStatus,
    TraceContext,
    TraceSpanRecorder,
    TraceStage,
    complete_recorded_span,
    start_recorded_span,
)
from decision_agent.routing.models import RequestRoute, RouterDecision
from decision_agent.routing.prompt import ROUTER_SYSTEM_PROMPT
from decision_agent.routing.request_router import RequestRouter, RequestRoutingError
from decision_agent.security import (
    AuthorizationPolicy,
    SecurityAuthorizationError,
    SecurityContext,
    SecurityErrorCode,
)
from decision_agent.skills.registry import SkillRegistry, SkillRegistryError


class Coordinator:
    """Compose Router, registry, and Skill without accessing Agent or MCP internals."""

    def __init__(
        self,
        *,
        router: RequestRouter,
        registry: SkillRegistry,
        controlled_workflow: ControlledAgentWorkflow | None = None,
    ) -> None:
        self._router = router
        self._registry = registry
        self._controlled_workflow = controlled_workflow

    async def execute(
        self,
        *,
        user_query: str,
        request_id: str | None = None,
        conversation_memory: ConversationMemoryProjection | None = None,
        trace_recorder: TraceSpanRecorder | None = None,
        trace_parent_context: TraceContext | None = None,
        security_context: SecurityContext | None = None,
        authorization_policy: AuthorizationPolicy | None = None,
    ) -> CoordinatorResult:
        """Coordinate one request with optional request-local observability only."""
        coordination_span = start_recorded_span(
            trace_recorder,
            stage=TraceStage.COORDINATION,
            component="coordination",
            operation="coordinate_request",
            parent_context=trace_parent_context,
        )
        try:
            result = await self._execute_business(
                user_query=user_query,
                request_id=request_id,
                conversation_memory=conversation_memory,
                trace_recorder=trace_recorder,
                coordination_span=coordination_span,
                security_context=security_context,
                authorization_policy=authorization_policy,
            )
        except asyncio.CancelledError:
            complete_recorded_span(
                trace_recorder,
                coordination_span,
                status=SpanStatus.CANCELLED,
            )
            raise
        except Exception:
            complete_recorded_span(
                trace_recorder,
                coordination_span,
                status=SpanStatus.FAILED,
                error_code="coordinator_execution_failed",
                attributes={"success": False},
            )
            raise
        complete_recorded_span(
            trace_recorder,
            coordination_span,
            status=_coordination_span_status(result),
            error_code=result.error_code if result.status is CoordinatorStatus.FAILED else None,
            attributes=_coordination_attributes(result),
        )
        return result

    async def _execute_business(
        self,
        *,
        user_query: str,
        request_id: str | None = None,
        conversation_memory: ConversationMemoryProjection | None = None,
        trace_recorder: TraceSpanRecorder | None,
        coordination_span: TraceContext | None,
        security_context: SecurityContext | None,
        authorization_policy: AuthorizationPolicy | None,
    ) -> CoordinatorResult:
        steps = ("route_request",)
        if authorization_policy is not None and security_context is None:
            return _failed(None, ("authorize_request",), SecurityErrorCode.UNAUTHENTICATED.value)
        resolved_request_id = str(uuid4()) if request_id is None else request_id
        runtime = RequestContextRuntime(
            request_id=resolved_request_id, created_at=datetime.now(UTC)
        )
        memory_item = None
        memory_context_selected = False
        router_memory_consumed = False
        try:
            user_item = runtime.user_request(user_query)
            if conversation_memory is not None:
                memory_item = runtime.add_conversation_memory(conversation_memory)
            router_instruction = runtime.system_instruction(
                "router-instruction", ROUTER_SYSTEM_PROMPT
            )
            router_selection = runtime.select_for_router(
                user_item=user_item, instruction_item=router_instruction, at=runtime.created_at
            )
            selected_router = RequestContextRuntime.project(router_selection).router(
                user_item_id=user_item.item_id
            )
            router_memory_selected = (
                memory_item is not None
                and memory_item.item_id in router_selection.selected_item_ids
            )
            route_with_context = getattr(self._router, "route_with_context", None)
            router_memory_consumed = (
                router_memory_selected if callable(route_with_context) else False
            )
            decision = await _route_with_trace(
                router=self._router,
                route_with_context=route_with_context,
                user_query=selected_router.user_request,
                selected_items=router_selection.selected_items,
                trace_recorder=trace_recorder,
                coordination_span=coordination_span,
            )
        except RequestRoutingError as exc:
            return _failed(
                None,
                steps,
                "coordinator_router_" + exc.subcode,
                memory_context_selected=router_memory_consumed,
            )
        except Exception:
            return _failed(
                None,
                steps,
                "coordinator_router_failed",
                memory_context_selected=router_memory_consumed,
            )
        memory_context_selected = router_memory_consumed
        if not isinstance(decision, RouterDecision):
            return _failed(
                None,
                steps,
                "invalid_router_decision",
                memory_context_selected=memory_context_selected,
            )
        try:
            decision_item = runtime.router_decision(decision, source_item_id=user_item.item_id)
            coordinator_selection = runtime.select_for_coordinator(
                user_item=user_item, decision_item=decision_item, at=runtime.created_at
            )
            selected = RequestContextRuntime.project(coordinator_selection).coordinator(
                user_item_id=user_item.item_id, decision_item_id=decision_item.item_id
            )
        except Exception:
            return _failed(
                decision.route,
                steps,
                "coordinator_context_failed",
                memory_context_selected=memory_context_selected,
            )
        decision = selected.decision
        selected_request = selected.user_request
        if decision.route is RequestRoute.UNSUPPORTED:
            return CoordinatorResult(
                status=CoordinatorStatus.UNSUPPORTED,
                route=decision.route,
                coordinator_steps=(*steps, "short_circuit_unsupported"),
                memory_context_selected=memory_context_selected,
            )
        if authorization_policy is not None:
            assert security_context is not None
            try:
                authorization_policy.require_scenario(security_context, decision.route.value)
                _require_route_scopes(
                    authorization_policy=authorization_policy,
                    security_context=security_context,
                    route=decision.route,
                )
            except SecurityAuthorizationError as exc:
                return _failed(
                    decision.route,
                    (*steps, "authorize_scenario", "authorize_scope"),
                    exc.code,
                    memory_context_selected=memory_context_selected,
                )
        if decision.route is RequestRoute.MIXED:
            try:
                skill = self._registry.for_route(decision.route)
            except SkillRegistryError as exc:
                return _failed(
                    decision.route,
                    (*steps, "select_mixed_skill"),
                    exc.args[0],
                    memory_context_selected=memory_context_selected,
                )
            if not skill.is_applicable(user_query, decision):
                return _failed(
                    decision.route,
                    (*steps, "select_mixed_skill"),
                    "no_matching_skill",
                    memory_context_selected=memory_context_selected,
                )
            skill_diagnostics_start = len(runtime.diagnostics)
            workflow_enabled = (
                self._controlled_workflow is not None
                and self._controlled_workflow.is_enabled_for(decision=decision, skill=skill)
            )
            try:
                if authorization_policy is not None:
                    assert security_context is not None
                    authorization_policy.require_workflow(
                        security_context,
                        "controlled_mixed" if workflow_enabled else "direct",
                    )
                    authorization_policy.require_skill(security_context, skill.definition.name)
                if workflow_enabled:
                    assert self._controlled_workflow is not None
                    workflow_result = await self._controlled_workflow.execute(
                        user_query=selected_request,
                        decision=decision,
                        context_runtime=runtime,
                        user_item_id=user_item.item_id,
                        memory_item_id=None if memory_item is None else memory_item.item_id,
                        trace_recorder=trace_recorder,
                        trace_parent_context=coordination_span,
                        security_context=security_context,
                        authorization_policy=authorization_policy,
                    )
                    memory_context_selected = (
                        memory_context_selected or workflow_result.memory_context_selected
                    )
                    completed_steps = (
                        *steps,
                        "select_mixed_skill",
                        "execute_skill",
                        "controlled_workflow",
                    )
                    if workflow_result.status is not WorkflowStatus.ACCEPTED:
                        return _failed(
                            decision.route,
                            completed_steps,
                            workflow_result.error_code or "workflow_execution_failed",
                            memory_context_selected=memory_context_selected,
                        )
                    result = workflow_result.accepted_result
                else:
                    result, skill_memory_selected = await execute_registered_skill(
                        skill=skill,
                        user_query=selected_request,
                        decision=decision,
                        context_runtime=runtime,
                        user_item_id=user_item.item_id,
                        memory_item_id=None if memory_item is None else memory_item.item_id,
                        trace_recorder=trace_recorder,
                        trace_parent_context=coordination_span,
                        security_context=security_context,
                        authorization_policy=authorization_policy,
                    )
                    memory_context_selected = memory_context_selected or skill_memory_selected
            except SecurityAuthorizationError as exc:
                return _failed(
                    decision.route,
                    (*steps, "select_mixed_skill", "authorize_skill"),
                    exc.code,
                    memory_context_selected=memory_context_selected,
                )
            except Exception:
                memory_context_selected = memory_context_selected or skill_memory_consumed(
                    runtime,
                    skill_diagnostics_start,
                    None if memory_item is None else memory_item.item_id,
                )
                return _failed(
                    decision.route,
                    (*steps, "select_mixed_skill", "execute_skill"),
                    "workflow_execution_failed" if workflow_enabled else "skill_execution_failed",
                    memory_context_selected=memory_context_selected,
                )
            if not workflow_enabled:
                completed_steps = (*steps, "select_mixed_skill", "execute_skill")
            if not isinstance(result, SkillResult):
                return _failed(
                    decision.route,
                    completed_steps,
                    "skill_result_contract_invalid",
                    memory_context_selected=memory_context_selected,
                )
            if result.route is not decision.route or result.skill_name != skill.definition.name:
                return _failed(
                    decision.route,
                    completed_steps,
                    "skill_result_contract_invalid",
                    memory_context_selected=memory_context_selected,
                )
            if result.status is SkillStatus.COMPLETED:
                return CoordinatorResult(
                    status=CoordinatorStatus.COMPLETED,
                    route=decision.route,
                    skill_name=result.skill_name,
                    answer=result.answer,
                    citations=result.citations,
                    coordinator_steps=completed_steps,
                    tool_steps=result.executed_steps,
                    memory_context_selected=memory_context_selected,
                )
            return _failed(
                decision.route,
                completed_steps,
                result.error_code or "skill_execution_failed",
                memory_context_selected=memory_context_selected,
            )
        try:
            skill = self._registry.for_route(decision.route)
        except SkillRegistryError as exc:
            return _failed(
                decision.route,
                (*steps, "select_skill"),
                exc.args[0],
                memory_context_selected=memory_context_selected,
            )
        if skill.definition.supported_route is not decision.route:
            return _failed(
                decision.route,
                (*steps, "select_skill"),
                "skill_route_mismatch",
                memory_context_selected=memory_context_selected,
            )
        skill_diagnostics_start = len(runtime.diagnostics)
        try:
            if authorization_policy is not None:
                assert security_context is not None
                authorization_policy.require_workflow(security_context, "direct")
                authorization_policy.require_skill(security_context, skill.definition.name)
            skill_result, skill_memory_selected = await execute_registered_skill(
                skill=skill,
                user_query=selected_request,
                decision=decision,
                context_runtime=runtime,
                user_item_id=user_item.item_id,
                memory_item_id=None if memory_item is None else memory_item.item_id,
                trace_recorder=trace_recorder,
                trace_parent_context=coordination_span,
                security_context=security_context,
                authorization_policy=authorization_policy,
            )
            memory_context_selected = memory_context_selected or skill_memory_selected
        except SecurityAuthorizationError as exc:
            return _failed(
                decision.route,
                (*steps, "select_skill", "authorize_skill"),
                exc.code,
                memory_context_selected=memory_context_selected,
            )
        except Exception:
            memory_context_selected = memory_context_selected or skill_memory_consumed(
                runtime,
                skill_diagnostics_start,
                None if memory_item is None else memory_item.item_id,
            )
            return _failed(
                decision.route,
                (*steps, "select_skill", "execute_skill"),
                "skill_execution_failed",
                memory_context_selected=memory_context_selected,
            )
        completed_steps = (*steps, "select_skill", "execute_skill")
        if (
            skill_result.route is not decision.route
            or skill_result.skill_name != skill.definition.name
        ):
            return _failed(
                decision.route,
                completed_steps,
                "skill_result_contract_invalid",
                memory_context_selected=memory_context_selected,
            )
        if skill_result.status is SkillStatus.FAILED:
            return _failed(
                decision.route,
                completed_steps,
                skill_result.error_code or "skill_execution_failed",
                memory_context_selected=memory_context_selected,
            )
        return CoordinatorResult(
            status=CoordinatorStatus.COMPLETED,
            route=decision.route,
            skill_name=skill_result.skill_name,
            answer=skill_result.answer,
            citations=skill_result.citations,
            coordinator_steps=completed_steps,
            tool_steps=skill_result.executed_steps,
            memory_context_selected=memory_context_selected,
        )


async def _route_with_trace(
    *,
    router: RequestRouter,
    route_with_context: object,
    user_query: str,
    selected_items: tuple[object, ...],
    trace_recorder: TraceSpanRecorder | None,
    coordination_span: TraceContext | None,
) -> object:
    """Measure only the real Router invocation and its strict output contract."""
    routing_span = start_recorded_span(
        trace_recorder,
        stage=TraceStage.ROUTING,
        component="routing",
        operation="route_request",
        parent_context=coordination_span,
    )
    try:
        route_with_trace = getattr(router, "route_with_trace", None)
        if callable(route_with_trace):
            decision = await route_with_trace(
                user_query=user_query,
                selected_items=selected_items if callable(route_with_context) else None,
                trace_recorder=trace_recorder,
                trace_parent_context=routing_span,
            )
        elif callable(route_with_context):
            decision = await route_with_context(
                user_query=user_query,
                selected_items=selected_items,
            )
        else:
            decision = await router.route(user_query=user_query)
    except asyncio.CancelledError:
        complete_recorded_span(
            trace_recorder,
            routing_span,
            status=SpanStatus.CANCELLED,
        )
        raise
    except RequestRoutingError as exc:
        complete_recorded_span(
            trace_recorder,
            routing_span,
            status=SpanStatus.FAILED,
            error_code="coordinator_router_" + exc.subcode,
        )
        raise
    except Exception:
        complete_recorded_span(
            trace_recorder,
            routing_span,
            status=SpanStatus.FAILED,
            error_code="coordinator_router_failed",
        )
        raise
    if not isinstance(decision, RouterDecision):
        complete_recorded_span(
            trace_recorder,
            routing_span,
            status=SpanStatus.FAILED,
            error_code="invalid_router_decision",
        )
        return decision
    complete_recorded_span(
        trace_recorder,
        routing_span,
        status=SpanStatus.COMPLETED,
        attributes={
            "route": decision.route.value,
            "has_knowledge_subquery": decision.knowledge_subquery is not None,
            "has_data_subquery": decision.data_subquery is not None,
            "success": True,
        },
    )
    return decision


def _coordination_span_status(result: CoordinatorResult) -> SpanStatus:
    if result.status is CoordinatorStatus.COMPLETED:
        return SpanStatus.COMPLETED
    if result.status is CoordinatorStatus.UNSUPPORTED:
        return SpanStatus.UNSUPPORTED
    return SpanStatus.FAILED


def _coordination_attributes(result: CoordinatorResult) -> dict[str, str | int | bool | None]:
    selected_skill_count = int("execute_skill" in result.coordinator_steps)
    attributes: dict[str, str | int | bool | None] = {
        "route": None if result.route is None else result.route.value,
        "selected_skill_count": selected_skill_count,
        "success": result.status is CoordinatorStatus.COMPLETED,
    }
    if (
        result.status is CoordinatorStatus.COMPLETED
        and result.route is not RequestRoute.MIXED
        and result.skill_name is not None
    ):
        attributes["skill_name"] = result.skill_name
    return attributes


def _require_route_scopes(
    *,
    authorization_policy: AuthorizationPolicy,
    security_context: SecurityContext,
    route: RequestRoute,
) -> None:
    """Reject domain access before Skill, Tool, Retriever, MCP, or data execution."""
    if route is RequestRoute.KNOWLEDGE:
        authorization_policy.require_knowledge_scope(security_context)
    elif route is RequestRoute.DATA:
        authorization_policy.require_data_scope(security_context, "enterprise_operations")
    elif route is RequestRoute.MIXED:
        authorization_policy.require_knowledge_scope(security_context)
        authorization_policy.require_data_scope(security_context, "enterprise_operations")


def _failed(
    route: RequestRoute | None,
    steps: tuple[str, ...],
    code: str,
    *,
    memory_context_selected: bool = False,
) -> CoordinatorResult:
    return CoordinatorResult(
        status=CoordinatorStatus.FAILED,
        route=route,
        coordinator_steps=steps,
        error_code=code,
        memory_context_selected=memory_context_selected,
    )
