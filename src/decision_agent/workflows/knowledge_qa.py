"""The fixed retrieval, selection, review, and answer-generation QA workflow."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from enum import StrEnum
from typing import Protocol
from uuid import UUID, uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.errors import NodeCancelledError
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from decision_agent.agents.answerability_reviewer import (
    AnswerabilityDecision,
    AnswerabilityReviewer,
    validate_answerability_decision,
)
from decision_agent.agents.evidence_selector import (
    EvidenceSelection,
    EvidenceSelectionError,
    EvidenceSelector,
    render_selected_evidence_context,
    validate_evidence_selection,
)
from decision_agent.agents.grounded_answer import AnswerDraft, AnswerGenerator
from decision_agent.domain import ErrorRecord
from decision_agent.observability.execution import (
    TraceSpanRecorder,
    complete_recorded_span,
    start_recorded_span,
)
from decision_agent.observability.models import TraceContext
from decision_agent.observability.stages import SpanStatus, TraceStage
from decision_agent.retrieval.evidence_context import EvidenceItem
from decision_agent.retrieval.pipeline import RetrievalPipelineResult
from decision_agent.security import KnowledgeScope

_CITATION_PATTERN = re.compile(r"^\[E[1-9][0-9]*\]$")
_INLINE_BRACKET_PATTERN = re.compile(r"\[[^\[\]]+\]")


class Answerability(StrEnum):
    """Terminal answerability outcomes for the minimal QA flow."""

    ANSWERABLE = "answerable"
    UNANSWERABLE = "unanswerable"
    FAILED = "failed"


class KnowledgeQAState(BaseModel):
    """Serializable state only; runtime services remain outside this contract."""

    model_config = ConfigDict(extra="forbid")

    request_id: UUID = Field(default_factory=uuid4)
    user_query: str = Field(min_length=1)
    retrieval_evidence: tuple[EvidenceItem, ...] = ()
    evidence_context: str = ""
    selected_evidence: tuple[EvidenceItem, ...] = ()
    selected_evidence_context: str = ""
    selection_reason: str | None = None
    answerability: Answerability | None = None
    answer: str | None = None
    citations: list[str] = Field(default_factory=list)
    missing_information: str | None = None
    decision_reason: str | None = None
    errors: list[ErrorRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_state(self) -> KnowledgeQAState:
        if self.answerability is None:
            if (
                self.answer is not None
                or self.citations != []
                or self.missing_information is not None
                or self.decision_reason is not None
                or self.errors != []
            ):
                raise ValueError("running state cannot retain terminal output")
        elif self.answerability is Answerability.FAILED:
            if (
                self.answer is not None
                or self.citations != []
                or self.missing_information is not None
                or self.decision_reason is not None
            ):
                raise ValueError("failed state cannot retain answer output")
            if not self.errors:
                raise ValueError("failed state requires an error")
        else:
            if self.answerability is Answerability.ANSWERABLE:
                if self.missing_information is not None:
                    raise ValueError("answerable state cannot retain missing information")
            elif (
                not isinstance(self.missing_information, str)
                or not self.missing_information.strip()
            ):
                raise ValueError("unanswerable state requires missing information")
            if not isinstance(self.decision_reason, str) or not self.decision_reason.strip():
                raise ValueError("reviewed state requires a decision reason")
            if self.errors:
                raise ValueError("reviewed state cannot retain errors")
            if self.answer is None:
                if self.citations != []:
                    raise ValueError("reviewed state cannot retain citations")
                return self
            if not self.answer.strip():
                raise ValueError("terminal state requires a nonempty answer")
            if self.answerability is Answerability.ANSWERABLE and not self.citations:
                raise ValueError("answerable state requires citations")
            if self.answerability is Answerability.UNANSWERABLE and self.citations:
                raise ValueError("unanswerable state cannot retain citations")
        return self


class CitationValidationResult(BaseModel):
    """Pure citation-validation result used before final state is emitted."""

    model_config = ConfigDict(extra="forbid")

    normalized_citations: list[str]
    validation_passed: bool
    validation_errors: list[str]


class RetrievalPipeline(Protocol):
    """The narrow existing-pipeline dependency used by retrieve_evidence."""

    async def retrieve(
        self,
        query: str,
        *,
        allowed_document_ids: frozenset[str] | None = None,
    ) -> RetrievalPipelineResult:
        """Return the already-ranked real retrieval result."""


def validate_citations(
    *,
    evidence_ids: Sequence[str],
    draft: AnswerDraft,
) -> CitationValidationResult:
    """Validate declared and inline citations against this request's evidence only."""
    available = {
        evidence_id if _CITATION_PATTERN.fullmatch(evidence_id) else f"[{evidence_id}]"
        for evidence_id in evidence_ids
    }
    normalized: list[str] = []
    errors: list[str] = []
    for citation in draft.citations:
        if not isinstance(citation, str) or not _CITATION_PATTERN.fullmatch(citation):
            errors.append("invalid_citation_format")
            continue
        if citation not in normalized:
            normalized.append(citation)
    if any(citation not in available for citation in normalized):
        errors.append("citation_not_in_evidence")

    inline = _INLINE_BRACKET_PATTERN.findall(draft.answer)
    if any(not _CITATION_PATTERN.fullmatch(citation) for citation in inline):
        errors.append("invalid_citation_format")
    inline_normalized = list(dict.fromkeys(inline))
    if set(normalized) - set(inline_normalized):
        errors.append("citations_not_present_in_answer")
    if set(inline_normalized) - set(normalized):
        errors.append("answer_contains_undeclared_citation")
    if not normalized:
        errors.append("answer_missing_inline_citation")
    return CitationValidationResult(
        normalized_citations=normalized,
        validation_passed=not errors,
        validation_errors=list(dict.fromkeys(errors)),
    )


def build_knowledge_qa_graph(
    *,
    retrieval_pipeline: RetrievalPipeline,
    evidence_selector: EvidenceSelector,
    answerability_reviewer: AnswerabilityReviewer,
    answer_generator: AnswerGenerator,
):
    """Compile the fixed, dependency-injected Evidence-selection QA workflow."""

    async def retrieve_evidence(
        state: KnowledgeQAState, config: RunnableConfig
    ) -> dict[str, object]:
        trace_recorder, trace_parent_context = _trace_from_config(config)
        knowledge_scope = _knowledge_scope_from_config(config)
        try:
            if knowledge_scope is not None:
                result = await retrieval_pipeline.retrieve(
                    state.user_query,
                    allowed_document_ids=knowledge_scope.allowed_document_ids,
                )
            elif trace_recorder is None or not hasattr(retrieval_pipeline, "_reranker"):
                result = await retrieval_pipeline.retrieve(state.user_query)
            else:
                result = await retrieval_pipeline.retrieve(
                    state.user_query,
                    trace_recorder=trace_recorder,
                    trace_parent_context=trace_parent_context,
                )
            evidence_items = result.evidence_context.evidence_items
            if knowledge_scope is not None and any(
                item.document_id not in knowledge_scope.allowed_document_ids
                for item in evidence_items
            ):
                return _failed_update(
                    "evidence_scope_violation", "Knowledge Evidence exceeded the authorized scope."
                )
            if knowledge_scope is not None and any(
                reference.document_id not in knowledge_scope.allowed_document_ids
                for reference in result.evidence_context.references
            ):
                return _failed_update(
                    "citation_scope_violation", "Knowledge citations exceeded the authorized scope."
                )
            return {
                "retrieval_evidence": evidence_items,
                "evidence_context": result.evidence_context.rendered_context,
            }
        except Exception:  # Boundary: external retrieval failures are converted to a safe state.
            return _failed_update("retrieval_failed", "Knowledge retrieval could not be completed.")

    async def select_evidence(state: KnowledgeQAState, config: RunnableConfig) -> dict[str, object]:
        if state.answerability is Answerability.FAILED or state.errors:
            return {}
        trace_recorder, trace_parent_context = _trace_from_config(config)
        selection_span = start_recorded_span(
            trace_recorder,
            stage=TraceStage.EVIDENCE_SELECTION,
            component="evidence",
            operation="select_evidence",
            parent_context=trace_parent_context,
            attributes={"candidate_evidence_count": len(state.retrieval_evidence)},
        )
        try:
            select_with_trace = getattr(evidence_selector, "select_with_trace", None)
            if callable(select_with_trace):
                raw_selection = await select_with_trace(
                    user_query=state.user_query,
                    evidence_context=state.evidence_context,
                    retrieval_evidence=state.retrieval_evidence,
                    trace_recorder=trace_recorder,
                    trace_parent_context=selection_span,
                )
            else:
                raw_selection = await evidence_selector.select(
                    user_query=state.user_query,
                    evidence_context=state.evidence_context,
                    retrieval_evidence=state.retrieval_evidence,
                )
            selection = EvidenceSelection.model_validate(raw_selection)
            validation = validate_evidence_selection(
                evidence_ids=[item.evidence_id for item in state.retrieval_evidence],
                selection=selection,
            )
            if not validation.validation_passed:
                complete_recorded_span(
                    trace_recorder,
                    selection_span,
                    status=SpanStatus.FAILED,
                    error_code=validation.validation_errors[0],
                    attributes={"success": False, "result_status": "failed"},
                )
                return _failed_update(
                    validation.validation_errors[0], "Evidence selection failed validation."
                )
            selected_ids = set(validation.normalized_selected_evidence_ids)
            selected_evidence = tuple(
                item for item in state.retrieval_evidence if f"[{item.evidence_id}]" in selected_ids
            )
            complete_recorded_span(
                trace_recorder,
                selection_span,
                status=SpanStatus.COMPLETED,
                attributes={
                    "selected_evidence_count": len(selected_evidence),
                    "answerable": bool(selected_evidence),
                    "empty_result": not selected_evidence,
                    "success": True,
                    "result_status": "completed",
                },
            )
            return {
                "selected_evidence": selected_evidence,
                "selected_evidence_context": render_selected_evidence_context(selected_evidence),
                "selection_reason": selection.selection_reason,
            }
        except asyncio.CancelledError:
            complete_recorded_span(trace_recorder, selection_span, status=SpanStatus.CANCELLED)
            raise
        except EvidenceSelectionError as exc:
            complete_recorded_span(
                trace_recorder,
                selection_span,
                status=SpanStatus.FAILED,
                error_code="evidence_selection_failed",
                attributes={"success": False, "result_status": "failed"},
            )
            return _failed_update(
                "evidence_selection_failed",
                "Evidence selection could not be completed.",
                subcode=exc.subcode,
            )
        except (ValidationError, ValueError, TypeError):
            complete_recorded_span(
                trace_recorder,
                selection_span,
                status=SpanStatus.FAILED,
                error_code="evidence_selection_failed",
                attributes={"success": False, "result_status": "failed"},
            )
            return _failed_update(
                "evidence_selection_failed",
                "Evidence selection could not be completed.",
                subcode="selector_response_invalid",
            )
        except Exception:
            complete_recorded_span(
                trace_recorder,
                selection_span,
                status=SpanStatus.FAILED,
                error_code="evidence_selection_failed",
                attributes={"success": False, "result_status": "failed"},
            )
            return _failed_update(
                "evidence_selection_failed",
                "Evidence selection could not be completed.",
                subcode="selector_unexpected_failure",
            )

    async def review_answerability(
        state: KnowledgeQAState, config: RunnableConfig
    ) -> dict[str, object]:
        if state.answerability is Answerability.FAILED or state.errors:
            return {}
        trace_recorder, trace_parent_context = _trace_from_config(config)
        reviewer_type = "rule" if not state.selected_evidence else "provider"
        review_span = start_recorded_span(
            trace_recorder,
            stage=TraceStage.REVIEW,
            component="review",
            operation="review_answerability",
            parent_context=trace_parent_context,
            attributes={"reviewer_type": reviewer_type},
        )
        if not state.selected_evidence:
            missing_information, decision_reason = _empty_evidence_decision(state.user_query)
            complete_recorded_span(
                trace_recorder,
                review_span,
                status=SpanStatus.COMPLETED,
                attributes={
                    "review_passed": False,
                    "review_outcome": "no_evidence",
                    "answerable": False,
                    "success": True,
                    "result_status": "completed",
                },
            )
            return {
                "answerability": Answerability.UNANSWERABLE,
                "missing_information": missing_information,
                "decision_reason": decision_reason,
            }
        try:
            raw_decision = await answerability_reviewer.review(
                user_query=state.user_query,
                selected_evidence_context=state.selected_evidence_context,
                selected_evidence=state.selected_evidence,
                trace_recorder=trace_recorder,
                trace_parent_context=review_span,
            )
            decision = AnswerabilityDecision.model_validate(raw_decision)
            decision_validation = validate_answerability_decision(
                user_query=state.user_query, decision=decision
            )
            if not decision_validation.validation_passed:
                error_code = decision_validation.validation_errors[0]
                complete_recorded_span(
                    trace_recorder,
                    review_span,
                    status=SpanStatus.FAILED,
                    error_code=error_code,
                    attributes={"success": False, "result_status": "failed"},
                )
                return _failed_update(
                    error_code,
                    "Answerability review failed language validation.",
                    clear_selected_evidence=False,
                )
            answerable = decision.answerability == Answerability.ANSWERABLE.value
            complete_recorded_span(
                trace_recorder,
                review_span,
                status=SpanStatus.COMPLETED,
                attributes={
                    "review_passed": answerable,
                    "review_outcome": decision.answerability,
                    "answerable": answerable,
                    "success": True,
                    "result_status": "completed",
                },
            )
            return {
                "answerability": Answerability(decision.answerability),
                "missing_information": decision.missing_information,
                "decision_reason": decision.decision_reason,
            }
        except asyncio.CancelledError:
            complete_recorded_span(trace_recorder, review_span, status=SpanStatus.CANCELLED)
            raise
        except (ValidationError, ValueError, TypeError):
            complete_recorded_span(
                trace_recorder,
                review_span,
                status=SpanStatus.FAILED,
                error_code="invalid_answerability_decision",
                attributes={"success": False, "result_status": "failed"},
            )
            return _failed_update(
                "invalid_answerability_decision",
                "Answerability review failed validation.",
                clear_selected_evidence=False,
            )
        except Exception:
            complete_recorded_span(
                trace_recorder,
                review_span,
                status=SpanStatus.FAILED,
                error_code="answerability_review_failed",
                attributes={"success": False, "result_status": "failed"},
            )
            return _failed_update(
                "answerability_review_failed",
                "Answerability review could not be completed.",
                clear_selected_evidence=False,
            )

    async def generate_answer(state: KnowledgeQAState, config: RunnableConfig) -> dict[str, object]:
        if state.answerability is Answerability.FAILED or state.errors:
            return {}
        if state.answerability is Answerability.UNANSWERABLE:
            return {
                "answer": _unanswerable_answer(state.user_query),
                "citations": [],
            }
        if state.answerability is not Answerability.ANSWERABLE:
            return _failed_update(
                "invalid_answerability_decision", "Answer generation requires a reviewed decision."
            )
        trace_recorder, trace_parent_context = _trace_from_config(config)
        answer_span = start_recorded_span(
            trace_recorder,
            stage=TraceStage.ANSWER_GENERATION,
            component="answer_generation",
            operation="generate_grounded_answer",
            parent_context=trace_parent_context,
            attributes={"answer_type": "grounded"},
        )
        try:
            raw_draft = await answer_generator.generate(
                user_query=state.user_query,
                selected_evidence_context=state.selected_evidence_context,
                selected_evidence=state.selected_evidence,
                answerability=state.answerability.value,
                missing_information=state.missing_information,
                decision_reason=state.decision_reason or "",
                trace_recorder=trace_recorder,
                trace_parent_context=answer_span,
            )
            draft = AnswerDraft.model_validate(raw_draft)
            validation = validate_citations(
                evidence_ids=[item.evidence_id for item in state.selected_evidence], draft=draft
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
                return _failed_update(
                    error_code,
                    "Generated answer failed validation.",
                    clear_selected_evidence=False,
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
                "answer": draft.answer,
                "citations": validation.normalized_citations,
            }
        except asyncio.CancelledError:
            complete_recorded_span(trace_recorder, answer_span, status=SpanStatus.CANCELLED)
            raise
        except (ValidationError, ValueError, TypeError):
            complete_recorded_span(
                trace_recorder,
                answer_span,
                status=SpanStatus.FAILED,
                error_code="answer_generation_failed",
                attributes={"success": False, "result_status": "failed"},
            )
            return _failed_update(
                "answer_generation_failed",
                "Grounded answer generation could not be completed.",
                clear_selected_evidence=False,
            )
        except Exception:
            complete_recorded_span(
                trace_recorder,
                answer_span,
                status=SpanStatus.FAILED,
                error_code="answer_generation_failed",
                attributes={"success": False, "result_status": "failed"},
            )
            return _failed_update(
                "answer_generation_failed",
                "Grounded answer generation could not be completed.",
                clear_selected_evidence=False,
            )

    builder = StateGraph(KnowledgeQAState)
    builder.add_node("retrieve_evidence", retrieve_evidence)
    builder.add_node("select_evidence", select_evidence)
    builder.add_node("review_answerability", review_answerability)
    builder.add_node("generate_answer", generate_answer)
    builder.add_edge(START, "retrieve_evidence")
    builder.add_edge("retrieve_evidence", "select_evidence")
    builder.add_edge("select_evidence", "review_answerability")
    builder.add_edge("review_answerability", "generate_answer")
    builder.add_edge("generate_answer", END)
    return builder.compile()


async def run_knowledge_qa(
    graph: object,
    *,
    user_query: str,
    request_id: UUID | None = None,
    trace_recorder: TraceSpanRecorder | None = None,
    trace_parent_context: TraceContext | None = None,
    knowledge_scope: KnowledgeScope | None = None,
) -> KnowledgeQAState:
    """Run a compiled graph and restore the checked serializable state contract."""
    initial = KnowledgeQAState(request_id=request_id or uuid4(), user_query=user_query)
    try:
        if trace_recorder is None and trace_parent_context is None and knowledge_scope is None:
            result = await graph.ainvoke(initial.model_dump(mode="python"))  # type: ignore[attr-defined]
        else:
            result = await graph.ainvoke(  # type: ignore[attr-defined]
                initial.model_dump(mode="python"),
                config={
                    "configurable": {
                        "trace_recorder": trace_recorder,
                        "trace_parent_context": trace_parent_context,
                        "knowledge_scope": knowledge_scope,
                    }
                },
            )
    except NodeCancelledError as error:
        if isinstance(error.__cause__, asyncio.CancelledError):
            raise asyncio.CancelledError from error
        raise
    return KnowledgeQAState.model_validate(result)


def _trace_from_config(
    config: RunnableConfig,
) -> tuple[TraceSpanRecorder | None, TraceContext | None]:
    """Read optional non-business runtime configuration without retaining it in graph state."""
    configurable = config.get("configurable")
    if not isinstance(configurable, dict):
        return None, None
    recorder = configurable.get("trace_recorder")
    parent_context = configurable.get("trace_parent_context")
    return (
        recorder if _is_trace_recorder(recorder) else None,
        parent_context if isinstance(parent_context, TraceContext) else None,
    )


def _knowledge_scope_from_config(config: RunnableConfig) -> KnowledgeScope | None:
    configurable = config.get("configurable")
    if not isinstance(configurable, dict):
        return None
    scope = configurable.get("knowledge_scope")
    return scope if isinstance(scope, KnowledgeScope) else None


def _is_trace_recorder(value: object) -> bool:
    return callable(getattr(value, "start_span", None)) and callable(
        getattr(value, "complete_span", None)
    )


def _failed_update(
    code: str,
    message: str,
    *,
    clear_selected_evidence: bool = True,
    subcode: str | None = None,
) -> dict[str, object]:
    update: dict[str, object] = {
        "answerability": Answerability.FAILED,
        "answer": None,
        "citations": [],
        "missing_information": None,
        "decision_reason": None,
        "errors": [
            ErrorRecord(
                code=code,
                message=message,
                details={"subcode": subcode} if subcode is not None else {},
            )
        ],
    }
    if clear_selected_evidence:
        update.update(
            {
                "selected_evidence": (),
                "selected_evidence_context": "",
                "selection_reason": None,
            }
        )
    return update


def _empty_evidence_decision(user_query: str) -> tuple[str, str]:
    if any("\u4e00" <= character <= "\u9fff" for character in user_query):
        return (
            "能够直接支持该问题答案的证据",
            "当前没有选中能够直接支持该问题的证据。",
        )
    return (
        "evidence that directly supports the requested information",
        "No selected Evidence directly supports the requested information.",
    )


def _unanswerable_answer(user_query: str) -> str:
    if any("\u4e00" <= character <= "\u9fff" for character in user_query):
        return "现有证据不足以确定该问题的答案。"
    return "The available Evidence is insufficient to determine the answer."
