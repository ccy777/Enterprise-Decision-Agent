"""Executable single-domain Skill for enterprise data analysis."""

from __future__ import annotations

from decision_agent.context import EvidenceDomain, RequestContextRuntime
from decision_agent.coordination.models import SkillResult, SkillStatus
from decision_agent.observability.execution import TraceSpanRecorder
from decision_agent.observability.models import TraceContext
from decision_agent.routing.models import RequestRoute, RouterDecision
from decision_agent.security import SecurityContext
from decision_agent.skills.contracts import NativeToolCallingExecutor, SkillDefinition
from decision_agent.tool_calling.models import NativeToolCallingStatus


class EnterpriseDataAnalysisSkill:
    """Delegate data requests to native tools; never access MCP or SQL directly."""

    _definition = SkillDefinition(
        name="enterprise-data-analysis",
        version="1.0.0",
        description="Answer operations-data questions through the existing cited Data Agent.",
        supported_route=RequestRoute.DATA,
        input_contract=("user_query", "RouterDecision(data)"),
        allowed_tools=("run_data_agent",),
        steps=("validate_route", "native_tool_calling", "validate_selected_tool", "map_result"),
        output_contract=("answer", "[D#] citations", "safe error_code"),
        failure_codes=(
            "skill_route_invalid",
            "skill_runtime_failed",
            "skill_selected_tool_invalid",
        ),
    )

    def __init__(self, *, runtime: NativeToolCallingExecutor) -> None:
        self._runtime = runtime

    @property
    def definition(self) -> SkillDefinition:
        return self._definition

    async def execute(self, *, user_query: str, decision: RouterDecision) -> SkillResult:
        return await self._execute(user_query=user_query, decision=decision)

    async def execute_with_trace(
        self,
        *,
        user_query: str,
        decision: RouterDecision,
        trace_recorder: TraceSpanRecorder | None = None,
        trace_parent_context: TraceContext | None = None,
        context_runtime: RequestContextRuntime | None = None,
        user_item_id: str | None = None,
        security_context: SecurityContext | None = None,
    ) -> SkillResult:
        """Internal observability extension; public business inputs remain unchanged."""
        if context_runtime is not None and user_item_id is not None:
            return await self.execute_with_context(
                user_query=user_query,
                decision=decision,
                context_runtime=context_runtime,
                user_item_id=user_item_id,
                trace_recorder=trace_recorder,
                trace_parent_context=trace_parent_context,
                security_context=security_context,
            )
        return await self._execute(
            user_query=user_query,
            decision=decision,
            trace_recorder=trace_recorder,
            trace_parent_context=trace_parent_context,
            security_context=security_context,
        )

    async def execute_with_context(
        self,
        *,
        user_query: str,
        decision: RouterDecision,
        context_runtime: RequestContextRuntime,
        user_item_id: str,
        trace_recorder: TraceSpanRecorder | None = None,
        trace_parent_context: TraceContext | None = None,
        security_context: SecurityContext | None = None,
    ) -> SkillResult:
        instruction = context_runtime.skill_instruction(
            "data-skill-instruction",
            "Execute only the approved read-only data question.",
            source_item_id=user_item_id,
        )
        user_item = context_runtime.get(user_item_id)
        if user_item is None:
            raise ValueError("context user item is missing")
        selection = context_runtime.select_for_data(
            user_item=user_item,
            instruction_item=instruction,
            at=context_runtime.created_at,
        )
        selected = RequestContextRuntime.project(selection).skill(
            user_item_id=user_item_id, instruction_item_id=instruction.item_id
        )
        result = await self._execute(
            user_query=selected.user_request,
            decision=decision,
            conversation_memory=selected.conversation_memory,
            trace_recorder=trace_recorder,
            trace_parent_context=trace_parent_context,
            security_context=security_context,
        )
        if (
            result.status is SkillStatus.COMPLETED
            and result.answer is not None
            and result.citations
        ):
            summary = context_runtime.verified_summary(
                "data-answer-summary", result.answer, source_item_ids=(instruction.item_id,)
            )
            context_runtime.evidence(
                "data-answer-evidence",
                result.answer,
                domain=EvidenceDomain.DATA,
                citation_ids=tuple(result.citations),
                source_item_ids=(summary.item_id,),
            )
        return result

    async def _execute(
        self,
        *,
        user_query: str,
        decision: RouterDecision,
        conversation_memory: str | None = None,
        trace_recorder: TraceSpanRecorder | None = None,
        trace_parent_context: TraceContext | None = None,
        security_context: SecurityContext | None = None,
    ) -> SkillResult:
        steps = ("validate_route",)
        if decision.route is not RequestRoute.DATA:
            return _failed(self.definition, decision.route, steps, "skill_route_invalid")
        scope_kwargs = {} if security_context is None else {"security_context": security_context}
        try:
            if conversation_memory is None:
                if trace_recorder is None and trace_parent_context is None:
                    runtime_result = await self._runtime.execute(
                        user_query=user_query,
                        decision=decision,
                        **scope_kwargs,
                    )
                else:
                    runtime_result = await self._runtime.execute(
                        user_query=user_query,
                        decision=decision,
                        trace_recorder=trace_recorder,
                        trace_parent_context=trace_parent_context,
                        **scope_kwargs,
                    )
            else:
                execute_with_memory = getattr(self._runtime, "execute_with_memory", None)
                if not callable(execute_with_memory):
                    raise RuntimeError("runtime memory projection is unsupported")
                if trace_recorder is None and trace_parent_context is None:
                    runtime_result = await execute_with_memory(
                        user_query=user_query,
                        decision=decision,
                        conversation_memory=conversation_memory,
                        **scope_kwargs,
                    )
                else:
                    runtime_result = await execute_with_memory(
                        user_query=user_query,
                        decision=decision,
                        conversation_memory=conversation_memory,
                        trace_recorder=trace_recorder,
                        trace_parent_context=trace_parent_context,
                        **scope_kwargs,
                    )
        except Exception:
            return _failed(
                self.definition,
                decision.route,
                (*steps, "native_tool_calling"),
                "skill_runtime_failed",
            )
        steps = (*steps, "native_tool_calling")
        if runtime_result.status is not NativeToolCallingStatus.COMPLETED:
            return _failed(
                self.definition,
                decision.route,
                steps,
                runtime_result.error_code or "skill_runtime_failed",
            )
        if (
            runtime_result.route is not decision.route
            or runtime_result.route is not self.definition.supported_route
        ):
            return _failed(self.definition, decision.route, steps, "skill_runtime_route_mismatch")
        if runtime_result.selected_tool not in self.definition.allowed_tools:
            return _failed(self.definition, decision.route, steps, "skill_tool_not_allowed")
        return SkillResult(
            status=SkillStatus.COMPLETED,
            skill_name=self.definition.name,
            skill_version=self.definition.version,
            route=decision.route,
            answer=runtime_result.answer,
            citations=runtime_result.citations,
            executed_steps=(*steps, "validate_selected_tool", "map_result"),
            selected_tool=runtime_result.selected_tool,
        )


def _failed(
    definition: SkillDefinition, route: RequestRoute, steps: tuple[str, ...], code: str
) -> SkillResult:
    return SkillResult(
        status=SkillStatus.FAILED,
        skill_name=definition.name,
        skill_version=definition.version,
        route=route,
        executed_steps=steps,
        error_code=code,
    )
