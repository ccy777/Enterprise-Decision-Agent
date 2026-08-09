"""Bounded mixed Skill that composes trusted inventory data and policy Skills."""

from __future__ import annotations

import asyncio
import inspect
import re

from decision_agent.context import EvidenceDomain, RequestContextRuntime
from decision_agent.coordination.models import SkillResult, SkillStatus
from decision_agent.observability.execution import (
    TraceSpanRecorder,
    complete_recorded_span,
    start_recorded_span,
)
from decision_agent.observability.models import TraceContext
from decision_agent.observability.stages import SpanStatus, TraceStage
from decision_agent.routing.models import RequestRoute, RouterDecision
from decision_agent.security import SecurityContext
from decision_agent.skills.contracts import ExecutableSkill, SkillDefinition
from decision_agent.skills.inventory_risk_synthesizer import (
    InventoryRiskSynthesisInput,
    InventoryRiskSynthesisResult,
    InventoryRiskSynthesizer,
    InventoryRiskSynthesizerError,
)

_DATA_CITATION = re.compile(r"^\[D\d+\]$")
_KNOWLEDGE_CITATION = re.compile(r"^\[E\d+\]$")
_DATA_SKILL_NAME = "enterprise-data-analysis"
_KNOWLEDGE_SKILL_NAME = "enterprise-knowledge-qa"
_TOPICS = (
    "库存",
    "缺货",
    "积压",
    "补货",
    "安全库存",
    "库存周转",
    "断供",
    "供应风险",
    "inventory",
    "stockout",
    "overstock",
    "replenishment",
    "safety stock",
    "inventory turnover",
    "supply risk",
)


class InventoryRiskDiagnosisSkill:
    """Run trusted data then knowledge Skills; synthesis remains deliberately absent."""

    _definition = SkillDefinition(
        name="inventory-risk-diagnosis",
        version="1.0.0",
        description="Diagnose inventory risk from data and policy evidence.",
        supported_route=RequestRoute.MIXED,
        input_contract=("request", "RouterDecision(mixed)"),
        allowed_tools=("run_data_agent", "run_knowledge_agent"),
        steps=(
            "validate_mixed_request",
            "execute_data_skill",
            "validate_data_result",
            "execute_knowledge_skill",
            "validate_knowledge_result",
        ),
        output_contract=("future mixed diagnosis",),
        failure_codes=(
            "inventory_risk_route_invalid",
            "inventory_risk_data_skill_failed",
            "inventory_risk_data_skill_invalid",
            "inventory_risk_knowledge_skill_failed",
            "inventory_risk_knowledge_skill_invalid",
            "inventory_risk_synthesizer_failed",
            "inventory_risk_synthesizer_http_error",
            "inventory_risk_synthesizer_unavailable",
            "inventory_risk_synthesizer_invalid_response",
            "inventory_risk_synthesizer_missing_choice",
            "inventory_risk_synthesizer_invalid_finish_reason",
            "inventory_risk_synthesizer_truncated",
            "inventory_risk_synthesizer_empty_content",
            "inventory_risk_synthesizer_invalid_json",
            "inventory_risk_synthesizer_schema_invalid",
            "inventory_risk_synthesis_invalid",
            "inventory_risk_synthesis_citations_invalid",
            "inventory_risk_synthesis_not_implemented",
        ),
        required_skills=(_DATA_SKILL_NAME, _KNOWLEDGE_SKILL_NAME),
    )

    def __init__(
        self,
        *,
        data_skill: ExecutableSkill,
        knowledge_skill: ExecutableSkill,
        synthesizer: InventoryRiskSynthesizer,
    ) -> None:
        _validate_dependency(data_skill, _DATA_SKILL_NAME, RequestRoute.DATA)
        _validate_dependency(knowledge_skill, _KNOWLEDGE_SKILL_NAME, RequestRoute.KNOWLEDGE)
        self._data_skill = data_skill
        self._knowledge_skill = knowledge_skill
        self._synthesizer = synthesizer

    @property
    def definition(self) -> SkillDefinition:
        return self._definition

    def is_applicable(self, request: str, decision: RouterDecision) -> bool:
        if (
            not isinstance(decision, RouterDecision)
            or decision.route is not RequestRoute.MIXED
            or not decision.data_subquery
            or not decision.knowledge_subquery
            or not decision.data_subquery.strip()
            or not decision.knowledge_subquery.strip()
        ):
            return False
        text = " ".join((request, decision.data_subquery, decision.knowledge_subquery)).lower()
        return any(topic in text for topic in _TOPICS)

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
        return await self._execute(
            user_query=user_query,
            decision=decision,
            context_runtime=context_runtime,
            user_item_id=user_item_id,
            trace_recorder=trace_recorder,
            trace_parent_context=trace_parent_context,
            security_context=security_context,
        )

    async def _execute(
        self,
        *,
        user_query: str,
        decision: RouterDecision,
        context_runtime: RequestContextRuntime | None = None,
        user_item_id: str | None = None,
        trace_recorder: TraceSpanRecorder | None = None,
        trace_parent_context: TraceContext | None = None,
        security_context: SecurityContext | None = None,
    ) -> SkillResult:
        steps = ("validate_mixed_request",)
        if not self.is_applicable(user_query, decision):
            return _failed(steps, "inventory_risk_route_invalid")

        data_decision = _subdecision(decision, RequestRoute.DATA, decision.data_subquery)
        try:
            data_result = await _execute_child(
                self._data_skill,
                user_query=decision.data_subquery,
                decision=data_decision,
                context_runtime=context_runtime,
                user_item_id=user_item_id,
                trace_recorder=trace_recorder,
                trace_parent_context=trace_parent_context,
                security_context=security_context,
            )
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            return _failed((*steps, "execute_data_skill"), "inventory_risk_data_skill_failed")
        steps = (*steps, "execute_data_skill")
        if not _is_valid_result(
            data_result,
            skill_name=_DATA_SKILL_NAME,
            route=RequestRoute.DATA,
            selected_tool="run_data_agent",
            citation_pattern=_DATA_CITATION,
        ):
            code = (
                "inventory_risk_data_skill_failed"
                if isinstance(data_result, SkillResult) and data_result.status is SkillStatus.FAILED
                else "inventory_risk_data_skill_invalid"
            )
            return _failed(steps, code)
        steps = (*steps, "validate_data_result")

        knowledge_decision = _subdecision(
            decision, RequestRoute.KNOWLEDGE, decision.knowledge_subquery
        )
        try:
            knowledge_result = await _execute_child(
                self._knowledge_skill,
                user_query=decision.knowledge_subquery,
                decision=knowledge_decision,
                context_runtime=context_runtime,
                user_item_id=user_item_id,
                trace_recorder=trace_recorder,
                trace_parent_context=trace_parent_context,
                security_context=security_context,
            )
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            return _failed(
                (*steps, "execute_knowledge_skill"), "inventory_risk_knowledge_skill_failed"
            )
        steps = (*steps, "execute_knowledge_skill")
        if not _is_valid_result(
            knowledge_result,
            skill_name=_KNOWLEDGE_SKILL_NAME,
            route=RequestRoute.KNOWLEDGE,
            selected_tool="run_knowledge_agent",
            citation_pattern=_KNOWLEDGE_CITATION,
        ):
            code = (
                "inventory_risk_knowledge_skill_failed"
                if isinstance(knowledge_result, SkillResult)
                and knowledge_result.status is SkillStatus.FAILED
                else "inventory_risk_knowledge_skill_invalid"
            )
            return _failed(steps, code)
        if context_runtime is not None and user_item_id is not None:
            data_summary = context_runtime.verified_answer_summary(
                "mixed-data-answer-summary",
                answer=data_result.answer,
                subquery=data_decision.normalized_query,
                citations=tuple(data_result.citations),
                source_item_ids=(user_item_id,),
            )
            knowledge_summary = context_runtime.verified_answer_summary(
                "mixed-knowledge-answer-summary",
                answer=knowledge_result.answer,
                subquery=knowledge_decision.normalized_query,
                citations=tuple(knowledge_result.citations),
                source_item_ids=(user_item_id,),
            )
            context_runtime.evidence(
                "mixed-data-evidence",
                data_result.answer,
                domain=EvidenceDomain.DATA,
                citation_ids=tuple(data_result.citations),
                source_item_ids=(data_summary.item_id,),
            )
            context_runtime.evidence(
                "mixed-knowledge-evidence",
                knowledge_result.answer,
                domain=EvidenceDomain.KNOWLEDGE,
                citation_ids=tuple(knowledge_result.citations),
                source_item_ids=(knowledge_summary.item_id,),
            )
            user_item = context_runtime.get(user_item_id)
            if user_item is None:
                return _failed(steps, "inventory_risk_context_failed")
            selection = context_runtime.select_for_mixed_synthesis(
                user_item=user_item,
                data_summary=data_summary,
                knowledge_summary=knowledge_summary,
                at=context_runtime.created_at,
            )
            selected = RequestContextRuntime.project(selection).mixed(
                user_item_id=user_item.item_id,
                data_summary_id=data_summary.item_id,
                knowledge_summary_id=knowledge_summary.item_id,
            )
            input_data = InventoryRiskSynthesisInput(
                original_request=selected.original_request,
                data_subquery=selected.data_subquery,
                data_answer=selected.data_answer,
                data_citations=selected.data_citations,
                knowledge_subquery=selected.knowledge_subquery,
                knowledge_answer=selected.knowledge_answer,
                knowledge_citations=selected.knowledge_citations,
            )
        else:
            input_data = InventoryRiskSynthesisInput(
                original_request=user_query,
                data_subquery=decision.data_subquery,
                data_answer=data_result.answer,
                data_citations=tuple(data_result.citations),
                knowledge_subquery=decision.knowledge_subquery,
                knowledge_answer=knowledge_result.answer,
                knowledge_citations=tuple(knowledge_result.citations),
            )
        synthesis_span = start_recorded_span(
            trace_recorder,
            stage=TraceStage.ANSWER_GENERATION,
            component="answer_generation",
            operation="generate_inventory_synthesis",
            parent_context=trace_parent_context,
            attributes={"answer_type": "inventory_synthesis"},
        )
        try:
            if trace_recorder is None or not _supports_trace_arguments(
                self._synthesizer.synthesize
            ):
                synthesis_result = await self._synthesizer.synthesize(input_data)
            else:
                synthesis_result = await self._synthesizer.synthesize(
                    input_data,
                    trace_recorder=trace_recorder,
                    trace_parent_context=synthesis_span,
                )
        except asyncio.CancelledError:
            complete_recorded_span(trace_recorder, synthesis_span, status=SpanStatus.CANCELLED)
            raise
        except (KeyboardInterrupt, SystemExit):
            raise
        except InventoryRiskSynthesizerError as exc:
            complete_recorded_span(
                trace_recorder,
                synthesis_span,
                status=SpanStatus.FAILED,
                error_code=exc.code,
                attributes={"success": False, "result_status": "failed"},
            )
            return _failed((*steps, "synthesize_inventory_risk"), exc.code)
        except Exception:
            complete_recorded_span(
                trace_recorder,
                synthesis_span,
                status=SpanStatus.FAILED,
                error_code="inventory_risk_synthesizer_failed",
                attributes={"success": False, "result_status": "failed"},
            )
            return _failed(
                (*steps, "synthesize_inventory_risk"),
                "inventory_risk_synthesizer_failed",
            )
        if not isinstance(synthesis_result, InventoryRiskSynthesisResult):
            complete_recorded_span(
                trace_recorder,
                synthesis_span,
                status=SpanStatus.FAILED,
                error_code="inventory_risk_synthesis_invalid",
                attributes={"success": False, "result_status": "failed"},
            )
            return _failed(
                (*steps, "synthesize_inventory_risk"), "inventory_risk_synthesis_invalid"
            )
        if not _is_valid_synthesis_content(synthesis_result):
            complete_recorded_span(
                trace_recorder,
                synthesis_span,
                status=SpanStatus.FAILED,
                error_code="inventory_risk_synthesis_invalid",
                attributes={"success": False, "result_status": "failed"},
            )
            return _failed(
                (*steps, "synthesize_inventory_risk"), "inventory_risk_synthesis_invalid"
            )
        citations = _ordered_mixed_citations(
            synthesis_result.citations,
            data_citations=data_result.citations,
            knowledge_citations=knowledge_result.citations,
        )
        if citations is None:
            complete_recorded_span(
                trace_recorder,
                synthesis_span,
                status=SpanStatus.FAILED,
                error_code="inventory_risk_synthesis_citations_invalid",
                attributes={"success": False, "result_status": "failed"},
            )
            return _failed(
                (*steps, "synthesize_inventory_risk"),
                "inventory_risk_synthesis_citations_invalid",
            )
        complete_recorded_span(
            trace_recorder,
            synthesis_span,
            status=SpanStatus.COMPLETED,
            attributes={
                "citation_count": len(citations),
                "source_count": 2,
                "success": True,
                "result_status": "completed",
            },
        )
        return SkillResult(
            status=SkillStatus.COMPLETED,
            skill_name=self.definition.name,
            skill_version=self.definition.version,
            route=RequestRoute.MIXED,
            answer=_render_answer(synthesis_result, citations=citations),
            citations=citations,
            executed_steps=(
                "enterprise-data-analysis",
                "run_data_agent",
                "enterprise-knowledge-qa",
                "run_knowledge_agent",
                "synthesize_inventory_risk",
            ),
            selected_tool=None,
        )


def _validate_dependency(skill: ExecutableSkill, name: str, route: RequestRoute) -> None:
    definition = skill.definition
    if definition.name != name or definition.supported_route is not route:
        raise ValueError("inventory_risk_dependency_invalid")


async def _execute_child(
    skill: ExecutableSkill,
    *,
    user_query: str,
    decision: RouterDecision,
    context_runtime: RequestContextRuntime | None,
    user_item_id: str | None,
    trace_recorder: TraceSpanRecorder | None,
    trace_parent_context: TraceContext | None,
    security_context: SecurityContext | None,
) -> SkillResult:
    execute_with_trace = getattr(skill, "execute_with_trace", None)
    execute_with_context = getattr(skill, "execute_with_context", None)
    if callable(execute_with_trace):
        kwargs: dict[str, object] = {
            "user_query": user_query,
            "decision": decision,
            "trace_recorder": trace_recorder,
            "trace_parent_context": trace_parent_context,
            "context_runtime": context_runtime,
            "user_item_id": user_item_id,
        }
        if security_context is not None and _supports_keyword(
            execute_with_trace, "security_context"
        ):
            kwargs["security_context"] = security_context
        return await execute_with_trace(**kwargs)
    if context_runtime is not None and user_item_id is not None and callable(execute_with_context):
        return await execute_with_context(
            user_query=user_query,
            decision=decision,
            context_runtime=context_runtime,
            user_item_id=user_item_id,
        )
    return await skill.execute(user_query=user_query, decision=decision)


def _subdecision(decision: RouterDecision, route: RequestRoute, subquery: str) -> RouterDecision:
    return RouterDecision(
        route=route,
        normalized_query=subquery,
        decision_reason=decision.decision_reason,
        knowledge_subquery=subquery if route is RequestRoute.KNOWLEDGE else None,
        data_subquery=subquery if route is RequestRoute.DATA else None,
        missing_information=None,
        confidence=decision.confidence,
    )


def _supports_trace_arguments(method: object) -> bool:
    """Use optional trace kwargs only when an injected synthesizer accepts them."""
    try:
        parameters = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):
        return False
    names = {parameter.name for parameter in parameters}
    return any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters) or {
        "trace_recorder",
        "trace_parent_context",
    }.issubset(names)


def _supports_keyword(method: object, keyword: str) -> bool:
    """Avoid widening legacy injected Skill test doubles with internal kwargs."""
    try:
        parameters = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD or parameter.name == keyword
        for parameter in parameters
    )


def _is_valid_result(
    result: object,
    *,
    skill_name: str,
    route: RequestRoute,
    selected_tool: str,
    citation_pattern: re.Pattern[str],
) -> bool:
    return (
        isinstance(result, SkillResult)
        and result.status is SkillStatus.COMPLETED
        and result.skill_name == skill_name
        and result.route is route
        and result.selected_tool == selected_tool
        and result.answer is not None
        and bool(result.answer.strip())
        and bool(result.citations)
        and all(citation_pattern.fullmatch(citation) for citation in result.citations)
    )


def _ordered_mixed_citations(
    citations: tuple[str, ...],
    *,
    data_citations: list[str],
    knowledge_citations: list[str],
) -> list[str] | None:
    if not isinstance(citations, tuple) or any(
        not isinstance(citation, str) for citation in citations
    ):
        return None
    allowed = set(data_citations) | set(knowledge_citations)
    selected = set(citations)
    if (
        not selected
        or not selected <= allowed
        or not any(_DATA_CITATION.fullmatch(citation) for citation in selected)
        or not any(_KNOWLEDGE_CITATION.fullmatch(citation) for citation in selected)
    ):
        return None
    return [
        citation for citation in (*data_citations, *knowledge_citations) if citation in selected
    ]


def _is_valid_synthesis_content(result: InventoryRiskSynthesisResult) -> bool:
    return (
        isinstance(result.risk_summary, str)
        and bool(result.risk_summary.strip())
        and isinstance(result.policy_basis, str)
        and bool(result.policy_basis.strip())
        and isinstance(result.recommended_actions, tuple)
        and bool(result.recommended_actions)
        and all(isinstance(action, str) and action.strip() for action in result.recommended_actions)
        and isinstance(result.citations, tuple)
    )


def _render_answer(
    result: InventoryRiskSynthesisResult,
    *,
    citations: list[str],
) -> str:
    actions = "\n".join(
        f"{index}. {action}" for index, action in enumerate(result.recommended_actions, start=1)
    )
    rendered_citations = " ".join(citations)
    return (
        f"风险概览:\n{result.risk_summary}\n\n"
        f"制度依据:\n{result.policy_basis}\n\n"
        f"建议措施:\n{actions}\n\n"
        f"引用:\n{rendered_citations}"
    )


def _failed(steps: tuple[str, ...], code: str) -> SkillResult:
    return SkillResult(
        status=SkillStatus.FAILED,
        skill_name=InventoryRiskDiagnosisSkill._definition.name,
        skill_version=InventoryRiskDiagnosisSkill._definition.version,
        route=RequestRoute.MIXED,
        executed_steps=steps,
        error_code=code,
    )
