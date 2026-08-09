"""Trace contracts for the existing Knowledge workflow's review and answer seams."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from decision_agent.agents.answerability_reviewer import OpenAICompatibleAnswerabilityReviewer
from decision_agent.agents.evidence_selector import EvidenceSelection
from decision_agent.agents.grounded_answer import OpenAICompatibleAnswerGenerator
from decision_agent.observability import SpanStatus, TraceCollector, TraceContext, TraceStage
from decision_agent.retrieval.evidence_context import (
    EvidenceContext,
    EvidenceItem,
    EvidenceReference,
)
from decision_agent.retrieval.parent_expansion import MatchedChild
from decision_agent.routing.models import RequestRoute, RouterDecision
from decision_agent.tool_calling.models import AgentToolResult, NativeToolCallingStatus
from decision_agent.tool_calling.runtime import run_native_tool_calling
from decision_agent.tool_calling.tools import KnowledgeAgentTool
from decision_agent.workflows.knowledge_qa import (
    Answerability,
    build_knowledge_qa_graph,
    run_knowledge_qa,
)


class _Ids:
    def __init__(self) -> None:
        self._value = 0

    def __call__(self) -> str:
        self._value += 1
        return f"span_{self._value}"


class _Pipeline:
    async def retrieve(self, query: str) -> object:
        del query
        return SimpleNamespace(evidence_context=_context())


class _Selector:
    def __init__(self, *, selected: bool = True) -> None:
        self._selected = selected
        self.calls = 0

    async def select(self, **_: object) -> EvidenceSelection:
        self.calls += 1
        return EvidenceSelection(
            selected_evidence_ids=["[E1]"] if self._selected else [],
            selection_reason="direct match" if self._selected else "no match",
        )


def _context() -> EvidenceContext:
    child = MatchedChild(
        child_id="child-1",
        parent_id="parent-1",
        document_id="doc-1",
        content="Evidence content must not enter the trace.",
        upstream_rank=1,
    )
    item = EvidenceItem(
        evidence_id="E1",
        final_rank=1,
        parent_id="parent-1",
        document_id="doc-1",
        content="Evidence content must not enter the trace.",
        original_content_length=41,
        included_content_length=41,
        truncated=False,
        matched_child_count=1,
        best_child_rank=1,
        matched_children=(child,),
    )
    return EvidenceContext(
        rendered_context="[E1] Evidence content must not enter the trace.",
        evidence_items=(item,),
        references=(
            EvidenceReference(
                evidence_id="E1",
                parent_id="parent-1",
                document_id="doc-1",
                source="private-source.md",
                start_offset=0,
                end_offset=41,
            ),
        ),
        included_evidence_count=1,
        omitted_evidence_count=0,
        total_original_chars=41,
        total_included_chars=41,
        truncated=False,
    )


def _collector() -> tuple[TraceCollector, TraceContext, TraceContext]:
    collector = TraceCollector(
        context=TraceContext.create(request_id="request_1", id_factory=lambda: "trace_1"),
        utc_now=lambda: datetime(2026, 7, 28, tzinfo=UTC),
        monotonic=lambda: 10.0,
        id_factory=_Ids(),
    )
    root = collector.start_span(stage=TraceStage.REQUEST, component="executor", operation="execute")
    tool = collector.start_span(
        stage=TraceStage.TOOL_EXECUTION,
        component="tool_calling",
        operation="execute_authorized_tool",
        parent_context=root,
    )
    return collector, root, tool


def _finish(collector: TraceCollector, root: TraceContext, tool: TraceContext):
    collector.complete_span(tool, status=SpanStatus.COMPLETED)
    collector.complete_span(root, status=SpanStatus.COMPLETED)
    return collector.finalize(final_status=SpanStatus.COMPLETED)


def _attributes(span: object) -> dict[str, object]:
    return {attribute.key: attribute.value for attribute in span.attributes}  # type: ignore[attr-defined]


def _reviewer() -> OpenAICompatibleAnswerabilityReviewer:
    return OpenAICompatibleAnswerabilityReviewer(
        api_key="test-key",
        base_url="https://example.test",
        model_name="knowledge-reviewer",
        timeout_seconds=1,
    )


def _generator() -> OpenAICompatibleAnswerGenerator:
    return OpenAICompatibleAnswerGenerator(
        api_key="test-key",
        base_url="https://example.test",
        model_name="knowledge-generator",
        timeout_seconds=1,
    )


def _graph(*, selector: _Selector, reviewer: object, generator: object):
    return build_knowledge_qa_graph(
        retrieval_pipeline=_Pipeline(),
        evidence_selector=selector,  # type: ignore[arg-type]
        answerability_reviewer=reviewer,  # type: ignore[arg-type]
        answer_generator=generator,  # type: ignore[arg-type]
    )


class _NativeModel:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = responses

    async def complete(self, **_: object) -> dict[str, Any]:
        return self._responses.pop(0)


class _UnusedDataTool:
    async def run(self, *, query: str) -> AgentToolResult:
        del query
        raise AssertionError("data tool must not run")


def _tool_call_response() -> dict[str, Any]:
    return {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "run_knowledge_agent",
                                "arguments": json.dumps({"query": "MODEL_QUERY_SECRET"}),
                            },
                        }
                    ]
                },
            }
        ]
    }


def _final_response() -> dict[str, Any]:
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps({"answer": "Supported fact.[E1]", "citations": ["[E1]"]})
                },
            }
        ]
    }


@pytest.mark.asyncio
async def test_answerable_knowledge_review_and_generation_are_tool_execution_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reviewer = _reviewer()
    generator = _generator()
    monkeypatch.setattr(
        reviewer,
        "_post",
        lambda *_: {
            "usage": {"prompt_tokens": 7, "completion_tokens": 0},
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            {
                                "answerability": "answerable",
                                "missing_information": None,
                                "decision_reason": "Selected Evidence is sufficient.",
                            }
                        )
                    },
                }
            ],
        },
    )
    monkeypatch.setattr(
        generator,
        "_post",
        lambda *_: {
            "usage": {"prompt_tokens": 0, "completion_tokens": 9},
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            {"answer": "Supported fact.[E1]", "citations": ["[E1]"]}
                        )
                    },
                }
            ],
        },
    )
    collector, root, tool = _collector()

    result = await run_knowledge_qa(
        _graph(selector=_Selector(), reviewer=reviewer, generator=generator),
        user_query="PRIVATE_QUERY",
        trace_recorder=collector,
        trace_parent_context=tool,
    )
    trace = _finish(collector, root, tool)
    review, review_provider, answer, answer_provider = trace.spans[-4:]

    assert result.answerability is Answerability.ANSWERABLE
    assert [span.operation for span in (review, review_provider, answer, answer_provider)] == [
        "review_answerability",
        "review_answerability",
        "generate_grounded_answer",
        "generate_grounded_answer",
    ]
    assert review.parent_span_id == answer.parent_span_id == tool.current_span_id
    assert review_provider.parent_span_id == review.span_id
    assert answer_provider.parent_span_id == answer.span_id
    assert _attributes(review) == {
        "reviewer_type": "provider",
        "review_passed": True,
        "review_outcome": "answerable",
        "answerable": True,
        "success": True,
        "result_status": "completed",
    }
    assert _attributes(answer) == {
        "answer_type": "grounded",
        "citation_count": 1,
        "success": True,
        "result_status": "completed",
    }
    assert _attributes(review_provider)["input_tokens"] == 7
    assert _attributes(review_provider)["output_tokens"] == 0
    assert _attributes(answer_provider)["input_tokens"] == 0
    assert _attributes(answer_provider)["output_tokens"] == 9
    serialized = str(trace.model_dump(mode="json"))
    assert all(
        secret not in serialized
        for secret in ("PRIVATE_QUERY", "Evidence content", "Supported fact")
    )


@pytest.mark.asyncio
async def test_native_runtime_passes_its_tool_execution_context_into_knowledge_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reviewer = _reviewer()
    generator = _generator()
    review_payload = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(
                        {
                            "answerability": "answerable",
                            "missing_information": None,
                            "decision_reason": "Selected Evidence is sufficient.",
                        }
                    )
                },
            }
        ]
    }
    answer_payload = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps({"answer": "Supported fact.[E1]", "citations": ["[E1]"]})
                },
            }
        ]
    }
    monkeypatch.setattr(reviewer, "_post", lambda *_: review_payload)
    monkeypatch.setattr(generator, "_post", lambda *_: answer_payload)
    collector = TraceCollector(
        context=TraceContext.create(request_id="request_1", id_factory=lambda: "trace_1"),
        utc_now=lambda: datetime(2026, 7, 28, tzinfo=UTC),
        monotonic=lambda: 10.0,
        id_factory=_Ids(),
    )
    root = collector.start_span(stage=TraceStage.REQUEST, component="executor", operation="execute")
    decision = RouterDecision(
        route=RequestRoute.KNOWLEDGE,
        normalized_query="PRIVATE_QUERY",
        decision_reason="PRIVATE_REASON",
        knowledge_subquery="KNOWLEDGE_QUERY_SECRET",
        data_subquery=None,
        missing_information=None,
        confidence=0.9,
    )

    result = await run_native_tool_calling(
        user_query="PRIVATE_QUERY",
        decision=decision,
        model=_NativeModel([_tool_call_response(), _final_response()]),
        knowledge_tool=KnowledgeAgentTool(
            graph=_graph(selector=_Selector(), reviewer=reviewer, generator=generator)
        ),
        data_tool=_UnusedDataTool(),
        trace_recorder=collector,
        trace_parent_context=root,
    )
    collector.complete_span(root, status=SpanStatus.COMPLETED)
    trace = collector.finalize(final_status=SpanStatus.COMPLETED)
    execution = next(span for span in trace.spans if span.stage is TraceStage.TOOL_EXECUTION)
    review = next(span for span in trace.spans if span.stage is TraceStage.REVIEW)
    grounded = next(
        span
        for span in trace.spans
        if span.stage is TraceStage.ANSWER_GENERATION
        and span.operation == "generate_grounded_answer"
    )

    assert result.status is NativeToolCallingStatus.COMPLETED
    assert review.parent_span_id == grounded.parent_span_id == execution.span_id
    assert sum(span.operation == "generate_tool_answer" for span in trace.spans) == 2


@pytest.mark.asyncio
async def test_no_evidence_uses_rule_review_without_provider_or_answer_generation() -> None:
    class _UnexpectedReviewer:
        async def review(self, **_: object) -> object:
            raise AssertionError("reviewer provider must not run")

    class _UnexpectedGenerator:
        async def generate(self, **_: object) -> object:
            raise AssertionError("generator provider must not run")

    collector, root, tool = _collector()
    result = await run_knowledge_qa(
        _graph(
            selector=_Selector(selected=False),
            reviewer=_UnexpectedReviewer(),
            generator=_UnexpectedGenerator(),
        ),
        user_query="PRIVATE_QUERY",
        trace_recorder=collector,
        trace_parent_context=tool,
    )
    trace = _finish(collector, root, tool)
    review = trace.spans[-1]

    assert result.answerability is Answerability.UNANSWERABLE
    assert review.stage is TraceStage.REVIEW and review.parent_span_id == tool.current_span_id
    assert _attributes(review) == {
        "reviewer_type": "rule",
        "review_passed": False,
        "review_outcome": "no_evidence",
        "answerable": False,
        "success": True,
        "result_status": "completed",
    }
    assert not [span for span in trace.spans if span.stage is TraceStage.PROVIDER_CALL]
    assert not [span for span in trace.spans if span.stage is TraceStage.ANSWER_GENERATION]


@pytest.mark.asyncio
@pytest.mark.parametrize("selected", [True, False])
async def test_evidence_selection_is_a_tool_child_with_safe_outcome_metadata(
    selected: bool,
) -> None:
    selector = _Selector(selected=selected)
    collector, root, tool = _collector()

    result = await run_knowledge_qa(
        _graph(
            selector=selector,
            reviewer=type("UnusedReviewer", (), {"review": None})(),
            generator=type("UnusedGenerator", (), {"generate": None})(),
        ),
        user_query="PRIVATE_QUERY",
        trace_recorder=collector,
        trace_parent_context=tool,
    )
    trace = _finish(collector, root, tool)
    selection = next(span for span in trace.spans if span.stage is TraceStage.EVIDENCE_SELECTION)

    assert selector.calls == 1
    assert selection.parent_span_id == tool.current_span_id
    assert _attributes(selection) == {
        "candidate_evidence_count": 1,
        "selected_evidence_count": int(selected),
        "answerable": selected,
        "empty_result": not selected,
        "success": True,
        "result_status": "completed",
    }
    assert result.answerability is (
        Answerability.UNANSWERABLE if not selected else Answerability.FAILED
    )
    serialized = str(trace.model_dump(mode="json"))
    assert all(value not in serialized for value in ("PRIVATE_QUERY", "Evidence content"))


@pytest.mark.asyncio
async def test_evidence_selection_recorder_failure_does_not_repeat_selection() -> None:
    class _BrokenRecorder:
        def start_span(self, **_: object) -> TraceContext:
            raise RuntimeError("PRIVATE_TRACE_FAILURE")

        def complete_span(self, *_: object, **__: object) -> None:
            raise RuntimeError("PRIVATE_TRACE_FAILURE")

    selector = _Selector(selected=False)
    result = await run_knowledge_qa(
        _graph(
            selector=selector,
            reviewer=type("UnusedReviewer", (), {"review": None})(),
            generator=type("UnusedGenerator", (), {"generate": None})(),
        ),
        user_query="PRIVATE_QUERY",
        trace_recorder=_BrokenRecorder(),
    )

    assert result.answerability is Answerability.UNANSWERABLE
    assert selector.calls == 1


@pytest.mark.asyncio
async def test_reviewer_rejection_is_completed_and_skips_grounded_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reviewer = _reviewer()
    generator = _generator()
    monkeypatch.setattr(
        reviewer,
        "_post",
        lambda *_: {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            {
                                "answerability": "unanswerable",
                                "missing_information": "the requested fact",
                                "decision_reason": (
                                    "The selected Evidence lacks the requested fact."
                                ),
                            }
                        )
                    },
                }
            ]
        },
    )
    monkeypatch.setattr(generator, "_post", lambda *_: pytest.fail("generator must not run"))
    collector, root, tool = _collector()

    result = await run_knowledge_qa(
        _graph(selector=_Selector(), reviewer=reviewer, generator=generator),
        user_query="PRIVATE_QUERY",
        trace_recorder=collector,
        trace_parent_context=tool,
    )
    trace = _finish(collector, root, tool)
    review = next(span for span in trace.spans if span.stage is TraceStage.REVIEW)

    assert result.answerability is Answerability.UNANSWERABLE
    assert review.status is SpanStatus.COMPLETED
    assert _attributes(review)["review_passed"] is False
    assert _attributes(review)["review_outcome"] == "unanswerable"
    assert not [span for span in trace.spans if span.stage is TraceStage.ANSWER_GENERATION]


@pytest.mark.asyncio
async def test_provider_success_with_local_grounding_failure_marks_only_answer_generation_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reviewer = _reviewer()
    generator = _generator()
    monkeypatch.setattr(
        reviewer,
        "_post",
        lambda *_: {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            {
                                "answerability": "answerable",
                                "missing_information": None,
                                "decision_reason": "Selected Evidence is sufficient.",
                            }
                        )
                    },
                }
            ]
        },
    )
    monkeypatch.setattr(
        generator,
        "_post",
        lambda *_: {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps({"answer": "Unsupported.[E9]", "citations": ["[E9]"]})
                    },
                }
            ]
        },
    )
    collector, root, tool = _collector()

    result = await run_knowledge_qa(
        _graph(selector=_Selector(), reviewer=reviewer, generator=generator),
        user_query="PRIVATE_QUERY",
        trace_recorder=collector,
        trace_parent_context=tool,
    )
    trace = _finish(collector, root, tool)
    answer = next(span for span in trace.spans if span.stage is TraceStage.ANSWER_GENERATION)
    provider = [span for span in trace.spans if span.stage is TraceStage.PROVIDER_CALL][-1]

    assert result.answerability is Answerability.FAILED
    assert result.errors[0].code == "citation_not_in_evidence"
    assert provider.status is SpanStatus.COMPLETED
    assert answer.status is SpanStatus.FAILED and answer.error_code == "citation_not_in_evidence"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failed_stage", "expected_code"),
    [
        ("review", "answerability_review_failed"),
        ("generation", "answer_generation_failed"),
    ],
)
async def test_provider_technical_failures_fail_the_enclosing_workflow_span(
    monkeypatch: pytest.MonkeyPatch, failed_stage: str, expected_code: str
) -> None:
    reviewer = _reviewer()
    generator = _generator()
    review_payload = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(
                        {
                            "answerability": "answerable",
                            "missing_information": None,
                            "decision_reason": "Selected Evidence is sufficient.",
                        }
                    )
                },
            }
        ]
    }
    answer_payload = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps({"answer": "Supported fact.[E1]", "citations": ["[E1]"]})
                },
            }
        ]
    }
    monkeypatch.setattr(
        reviewer,
        "_post",
        (lambda *_: (_ for _ in ()).throw(OSError()))
        if failed_stage == "review"
        else lambda *_: review_payload,
    )
    monkeypatch.setattr(
        generator,
        "_post",
        (lambda *_: (_ for _ in ()).throw(OSError()))
        if failed_stage == "generation"
        else lambda *_: answer_payload,
    )
    collector, root, tool = _collector()

    result = await run_knowledge_qa(
        _graph(selector=_Selector(), reviewer=reviewer, generator=generator),
        user_query="PRIVATE_QUERY",
        trace_recorder=collector,
        trace_parent_context=tool,
    )
    trace = _finish(collector, root, tool)
    failed = next(span for span in trace.spans if span.error_code == expected_code)
    provider = [span for span in trace.spans if span.stage is TraceStage.PROVIDER_CALL][-1]

    assert result.answerability is Answerability.FAILED
    assert result.errors[0].code == expected_code
    assert failed.status is provider.status is SpanStatus.FAILED


@pytest.mark.asyncio
async def test_cancellation_and_recorder_failures_do_not_repeat_reviewer_or_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reviewer = _reviewer()
    generator = _generator()
    calls = {"review": 0, "generate": 0}

    def cancel(*_: object) -> dict[str, Any]:
        calls["review"] += 1
        raise asyncio.CancelledError

    monkeypatch.setattr(reviewer, "_post", cancel)
    monkeypatch.setattr(generator, "_post", lambda *_: calls.__setitem__("generate", 1))
    collector, root, tool = _collector()

    with pytest.raises(asyncio.CancelledError):
        await run_knowledge_qa(
            _graph(selector=_Selector(), reviewer=reviewer, generator=generator),
            user_query="PRIVATE_QUERY",
            trace_recorder=collector,
            trace_parent_context=tool,
        )
    trace = _finish(collector, root, tool)
    review = next(span for span in trace.spans if span.stage is TraceStage.REVIEW)
    provider = next(span for span in trace.spans if span.stage is TraceStage.PROVIDER_CALL)

    assert calls == {"review": 1, "generate": 0}
    assert review.status is provider.status is SpanStatus.CANCELLED


@pytest.mark.asyncio
async def test_recorder_failures_do_not_change_or_repeat_knowledge_provider_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BrokenRecorder:
        def start_span(self, **_: object) -> TraceContext:
            raise RuntimeError("PRIVATE_TRACE_FAILURE")

        def complete_span(self, *_: object, **__: object) -> None:
            raise RuntimeError("PRIVATE_TRACE_FAILURE")

    reviewer = _reviewer()
    generator = _generator()
    calls = {"review": 0, "generate": 0}

    def review_post(*_: object) -> dict[str, Any]:
        calls["review"] += 1
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            {
                                "answerability": "answerable",
                                "missing_information": None,
                                "decision_reason": "Selected Evidence is sufficient.",
                            }
                        )
                    },
                }
            ]
        }

    def generate_post(*_: object) -> dict[str, Any]:
        calls["generate"] += 1
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            {"answer": "Supported fact.[E1]", "citations": ["[E1]"]}
                        )
                    },
                }
            ]
        }

    monkeypatch.setattr(reviewer, "_post", review_post)
    monkeypatch.setattr(generator, "_post", generate_post)

    result = await run_knowledge_qa(
        _graph(selector=_Selector(), reviewer=reviewer, generator=generator),
        user_query="PRIVATE_QUERY",
        trace_recorder=_BrokenRecorder(),
        trace_parent_context=None,
    )

    assert result.answerability is Answerability.ANSWERABLE
    assert calls == {"review": 1, "generate": 1}
