"""Offline behavior tests for the fixed evidence-grounded QA graph."""

# ruff: noqa: RUF001

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from decision_agent.agents.answerability_reviewer import AnswerabilityDecision
from decision_agent.agents.evidence_selector import EvidenceSelection, EvidenceSelectionError
from decision_agent.agents.grounded_answer import AnswerDraft
from decision_agent.domain import ErrorRecord
from decision_agent.retrieval.evidence_context import (
    EvidenceContext,
    EvidenceItem,
    EvidenceReference,
)
from decision_agent.retrieval.parent_expansion import MatchedChild
from decision_agent.security import KnowledgeScope
from decision_agent.workflows.knowledge_qa import (
    Answerability,
    KnowledgeQAState,
    build_knowledge_qa_graph,
    run_knowledge_qa,
    validate_citations,
)


def _evidence_context() -> EvidenceContext:
    child = MatchedChild(
        child_id="CHILD-1",
        parent_id="PARENT-1",
        document_id="DOC-1",
        content="产品 A 的原装电池保修期为 12 个月。",
        upstream_rank=1,
    )
    item = EvidenceItem(
        evidence_id="E1",
        final_rank=1,
        parent_id="PARENT-1",
        document_id="DOC-1",
        content="产品 A 的原装电池保修期为 12 个月。",
        original_content_length=21,
        included_content_length=21,
        truncated=False,
        matched_child_count=1,
        best_child_rank=1,
        matched_children=(child,),
    )
    reference = EvidenceReference(
        evidence_id="E1",
        parent_id="PARENT-1",
        document_id="DOC-1",
        source="policies/warranty.md",
        start_offset=0,
        end_offset=21,
    )
    return EvidenceContext(
        rendered_context="[E1]\n产品 A 的原装电池保修期为 12 个月。",
        evidence_items=(item,),
        references=(reference,),
        included_evidence_count=1,
        omitted_evidence_count=0,
        total_original_chars=21,
        total_included_chars=21,
        truncated=False,
    )


class FakePipeline:
    def __init__(self, context: EvidenceContext | None = None, *, fails: bool = False) -> None:
        self.context = context or _evidence_context()
        self.fails = fails
        self.queries: list[str] = []
        self.allowed_document_ids: list[frozenset[str] | None] = []

    async def retrieve(
        self,
        query: str,
        *,
        allowed_document_ids: frozenset[str] | None = None,
    ) -> object:
        self.queries.append(query)
        self.allowed_document_ids.append(allowed_document_ids)
        if self.fails:
            raise RuntimeError("private retrieval detail")
        return SimpleNamespace(evidence_context=self.context)


class RecordingSelector:
    def __init__(self, selection: object | None = None, *, fails: bool = False) -> None:
        self.selection = selection or EvidenceSelection(
            selected_evidence_ids=["[E1]"], selection_reason="Directly relevant evidence."
        )
        self.fails = fails
        self.calls = 0

    async def select(self, **kwargs: object) -> object:
        self.calls += 1
        if self.fails:
            raise RuntimeError("private selection detail")
        return self.selection


class RecordingReviewer:
    def __init__(self, decision: object | None = None, *, fails: bool = False) -> None:
        self.decision = decision or _answerable_decision()
        self.fails = fails
        self.calls = 0
        self.received_evidence_ids: list[tuple[str, ...]] = []
        self.received_contexts: list[str] = []

    async def review(self, **kwargs: object) -> object:
        self.calls += 1
        evidence = kwargs["selected_evidence"]
        context = kwargs["selected_evidence_context"]
        assert isinstance(evidence, tuple)
        assert isinstance(context, str)
        self.received_evidence_ids.append(tuple(item.evidence_id for item in evidence))
        self.received_contexts.append(context)
        if self.fails:
            raise RuntimeError("private reviewer detail")
        return self.decision


class RecordingGenerator:
    def __init__(self, draft: object | None = None, *, fails: bool = False) -> None:
        self.draft = draft or _answer_draft()
        self.fails = fails
        self.calls = 0
        self.received_evidence_ids: list[tuple[str, ...]] = []
        self.received_contexts: list[str] = []
        self.received_answerability: list[str] = []

    async def generate(self, **kwargs: object) -> object:
        self.calls += 1
        evidence = kwargs["selected_evidence"]
        context = kwargs["selected_evidence_context"]
        answerability = kwargs["answerability"]
        assert isinstance(evidence, tuple)
        assert isinstance(context, str)
        assert isinstance(answerability, str)
        self.received_evidence_ids.append(tuple(item.evidence_id for item in evidence))
        self.received_contexts.append(context)
        self.received_answerability.append(answerability)
        if self.fails:
            raise RuntimeError("private generator detail")
        return self.draft


def _answerable_decision() -> AnswerabilityDecision:
    return AnswerabilityDecision(
        answerability="answerable",
        missing_information=None,
        decision_reason="选中的证据明确规定了原装电池保修期。",
    )


def _unanswerable_decision() -> AnswerabilityDecision:
    return AnswerabilityDecision(
        answerability="unanswerable",
        missing_information="维修完成后的新增免费保修期限",
        decision_reason="选中的证据没有规定维修完成后的新增免费保修期限。",
    )


def _answer_draft() -> AnswerDraft:
    return AnswerDraft(answer="产品 A 的原装电池保修期为 12 个月。[E1]", citations=["[E1]"])


def _graph(
    *,
    pipeline: FakePipeline | None = None,
    selector: RecordingSelector | None = None,
    reviewer: RecordingReviewer | None = None,
    generator: RecordingGenerator | None = None,
):
    return build_knowledge_qa_graph(
        retrieval_pipeline=pipeline or FakePipeline(),
        evidence_selector=selector or RecordingSelector(),
        answerability_reviewer=reviewer or RecordingReviewer(),
        answer_generator=generator or RecordingGenerator(),
    )


@pytest.mark.parametrize(
    ("draft", "passed", "expected"),
    [
        (_answer_draft(), True, ["[E1]"]),
        (_answer_draft().model_copy(update={"citations": ["[E9]"]}), False, ["[E9]"]),
        (_answer_draft().model_copy(update={"citations": []}), False, []),
        (_answer_draft().model_copy(update={"answer": "没有内联引用。"}), False, ["[E1]"]),
        (
            _answer_draft().model_copy(update={"answer": "答案使用未声明引用。[E2]"}),
            False,
            ["[E1]"],
        ),
        (_answer_draft().model_copy(update={"citations": ["DOC-CS-001"]}), False, []),
        (_answer_draft().model_copy(update={"citations": ["[E1]", "[E1]"]}), True, ["[E1]"]),
    ],
)
def test_validate_citations_remains_strict(
    draft: AnswerDraft, passed: bool, expected: list[str]
) -> None:
    result = validate_citations(evidence_ids=["[E1]"], draft=draft)

    assert result.validation_passed is passed
    assert result.normalized_citations == expected


def test_answer_generator_schema_cannot_modify_reviewer_decision() -> None:
    with pytest.raises(ValidationError, match="answerability"):
        AnswerDraft.model_validate(
            {"answer": "Supported.[E1]", "citations": ["[E1]"], "answerability": "unanswerable"}
        )


@pytest.mark.parametrize(
    ("state_update", "error_match"),
    [
        ({"answer": "premature"}, "running state cannot retain terminal output"),
        (
            {
                "answerability": Answerability.ANSWERABLE,
                "citations": ["[E1]"],
                "decision_reason": "理由",
            },
            "reviewed state cannot retain citations",
        ),
        (
            {
                "answerability": Answerability.ANSWERABLE,
                "missing_information": "缺失",
                "decision_reason": "理由",
            },
            "answerable state cannot retain missing information",
        ),
        (
            {"answerability": Answerability.UNANSWERABLE, "decision_reason": "理由"},
            "unanswerable state requires missing information",
        ),
        (
            {
                "answerability": Answerability.UNANSWERABLE,
                "missing_information": "缺失",
                "decision_reason": "理由",
                "answer": "",
                "citations": [],
            },
            "terminal state requires a nonempty answer",
        ),
        (
            {
                "answerability": Answerability.FAILED,
                "answer": "",
                "errors": [ErrorRecord(code="safe", message="safe")],
            },
            "failed state cannot retain answer output",
        ),
    ],
)
def test_state_rejects_invalid_running_reviewed_and_failed_states(
    state_update: dict[str, object], error_match: str
) -> None:
    with pytest.raises(ValidationError, match=error_match):
        KnowledgeQAState(user_query="测试", **state_update)


def test_state_accepts_selected_reviewed_and_final_contracts() -> None:
    selected = KnowledgeQAState(
        user_query="测试", selected_evidence=_evidence_context().evidence_items
    )
    reviewed_answerable = KnowledgeQAState(
        user_query="测试", answerability=Answerability.ANSWERABLE, decision_reason="证据充分。"
    )
    reviewed_unanswerable = KnowledgeQAState(
        user_query="测试",
        answerability=Answerability.UNANSWERABLE,
        missing_information="缺失信息",
        decision_reason="证据不足。",
    )
    final_answerable = KnowledgeQAState(
        user_query="测试",
        answerability=Answerability.ANSWERABLE,
        answer="答案。[E1]",
        citations=["[E1]"],
        decision_reason="证据充分。",
    )
    final_unanswerable = reviewed_unanswerable.model_copy(update={"answer": "现有证据不足以确定。"})
    failed = KnowledgeQAState(
        user_query="测试",
        answerability=Answerability.FAILED,
        errors=[ErrorRecord(code="safe", message="safe")],
    )

    assert selected.answerability is None
    assert reviewed_answerable.answer is None and reviewed_answerable.citations == []
    assert reviewed_unanswerable.answer is None and reviewed_unanswerable.citations == []
    assert final_answerable.citations == ["[E1]"]
    assert final_unanswerable.citations == []
    assert failed.answer is None


def test_knowledge_qa_cli_parser_has_no_model_side_effects() -> None:
    from scripts.run_knowledge_qa_agent_demo import ROOT, build_parser

    args = build_parser().parse_args(["--query", "测试问题"])

    assert args.query == "测试问题"
    assert args.dataset_root == ROOT / "datasets/enterprise_kb/m2c1"


def test_cli_public_projection_excludes_retrieval_selection_and_review_internals() -> None:
    from scripts.run_knowledge_qa_agent_demo import project_public_result

    state = KnowledgeQAState(
        user_query="测试",
        retrieval_evidence=_evidence_context().evidence_items,
        evidence_context="private raw context",
        selected_evidence=_evidence_context().evidence_items,
        selected_evidence_context="private selected context",
        selection_reason="private selection reason",
        answerability=Answerability.ANSWERABLE,
        answer="支持的答案。[E1]",
        citations=["[E1]"],
        decision_reason="证据充分。",
    )

    assert project_public_result(state) == {
        "answerability": Answerability.ANSWERABLE,
        "answer": "支持的答案。[E1]",
        "citations": ["[E1]"],
        "missing_information": None,
        "decision_reason": "证据充分。",
    }


@pytest.mark.asyncio
async def test_graph_has_the_required_four_node_topology() -> None:
    graph = _graph()
    names = set(graph.get_graph().nodes)
    edges = {(edge.source, edge.target) for edge in graph.get_graph().edges}

    assert {
        "__start__",
        "retrieve_evidence",
        "select_evidence",
        "review_answerability",
        "generate_answer",
        "__end__",
    } <= names
    assert {
        ("__start__", "retrieve_evidence"),
        ("retrieve_evidence", "select_evidence"),
        ("select_evidence", "review_answerability"),
        ("review_answerability", "generate_answer"),
        ("generate_answer", "__end__"),
    } <= edges


@pytest.mark.asyncio
async def test_answerable_path_uses_only_selected_evidence_and_reviewer_decision() -> None:
    context = _evidence_context()
    e1 = context.evidence_items[0]
    e2 = e1.model_copy(update={"evidence_id": "E2", "content": "unselected inventory noise"})
    pipeline = FakePipeline(context.model_copy(update={"evidence_items": (e1, e2)}))
    reviewer = RecordingReviewer()
    generator = RecordingGenerator()

    result = await run_knowledge_qa(
        _graph(pipeline=pipeline, reviewer=reviewer, generator=generator),
        user_query="产品 A 的原装电池保修期多久？",
    )

    assert result.answerability is Answerability.ANSWERABLE
    assert result.answer == _answer_draft().answer
    assert result.citations == ["[E1]"]
    assert reviewer.received_evidence_ids == [("E1",)]
    assert "[E2]" not in reviewer.received_contexts[0]
    assert generator.received_evidence_ids == [("E1",)]
    assert generator.received_answerability == ["answerable"]


@pytest.mark.asyncio
async def test_knowledge_scope_reaches_retrieval_and_blocks_out_of_scope_evidence() -> None:
    pipeline = FakePipeline()
    reviewer = RecordingReviewer()
    generator = RecordingGenerator()

    result = await run_knowledge_qa(
        _graph(pipeline=pipeline, reviewer=reviewer, generator=generator),
        user_query="产品 A 的原装电池保修期多久？",
        knowledge_scope=KnowledgeScope(
            tenant_id="tenant-a",
            allowed_namespaces=frozenset({"enterprise_kb"}),
            allowed_document_ids=frozenset({"DOC-2"}),
        ),
    )

    assert result.answerability is Answerability.FAILED
    assert result.errors[0].code == "evidence_scope_violation"
    assert pipeline.allowed_document_ids == [frozenset({"DOC-2"})]
    assert reviewer.calls == generator.calls == 0


@pytest.mark.asyncio
async def test_empty_selection_is_deterministic_unanswerable_without_reviewer_or_generator() -> (
    None
):
    reviewer = RecordingReviewer()
    generator = RecordingGenerator()
    result = await run_knowledge_qa(
        _graph(
            selector=RecordingSelector(
                EvidenceSelection(selected_evidence_ids=[], selection_reason="No match.")
            ),
            reviewer=reviewer,
            generator=generator,
        ),
        user_query="维修完成后免费保修多久？",
    )

    assert result.answerability is Answerability.UNANSWERABLE
    assert result.answer is not None
    assert result.citations == []
    assert reviewer.calls == 0
    assert generator.calls == 0


@pytest.mark.asyncio
async def test_reviewer_unanswerable_skips_generator_and_does_not_expose_missing_number() -> None:
    reviewer = RecordingReviewer(_unanswerable_decision())
    generator = RecordingGenerator()
    result = await run_knowledge_qa(
        _graph(reviewer=reviewer, generator=generator), user_query="维修问题？"
    )

    assert result.answerability is Answerability.UNANSWERABLE
    assert result.citations == []
    assert result.missing_information == "维修完成后的新增免费保修期限"
    assert result.answer is not None and not any(character.isdigit() for character in result.answer)
    assert generator.calls == 0


@pytest.mark.asyncio
async def test_unanswerable_answer_does_not_echo_a_reviewer_missing_number() -> None:
    reviewer = RecordingReviewer(
        AnswerabilityDecision(
            answerability="unanswerable",
            missing_information="维修完成后的 30 天期限",
            decision_reason="选中的证据没有规定维修完成后的新增期限。",
        )
    )
    result = await run_knowledge_qa(
        _graph(reviewer=reviewer), user_query="维修完成后免费保修多久？"
    )

    assert result.answerability is Answerability.UNANSWERABLE
    assert result.answer is not None and not any(character.isdigit() for character in result.answer)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reviewer", "expected_code"),
    [
        (RecordingReviewer(fails=True), "answerability_review_failed"),
        (
            RecordingReviewer({"answerability": "answerable", "decision_reason": "English only."}),
            "reviewer_language_mismatch",
        ),
        (
            RecordingReviewer(
                {
                    "answerability": "unanswerable",
                    "missing_information": "missing",
                    "decision_reason": "中文说明",
                }
            ),
            "reviewer_language_mismatch",
        ),
        (
            RecordingReviewer({"answerability": "unanswerable", "decision_reason": "中文说明"}),
            "invalid_answerability_decision",
        ),
    ],
)
async def test_invalid_or_failed_reviewer_enters_safe_failed_state(
    reviewer: RecordingReviewer, expected_code: str
) -> None:
    generator = RecordingGenerator()
    result = await run_knowledge_qa(
        _graph(reviewer=reviewer, generator=generator), user_query="测试问题"
    )

    assert result.answerability is Answerability.FAILED
    assert result.answer is None and result.citations == []
    assert result.missing_information is None and result.decision_reason is None
    assert result.errors[0].code == expected_code
    assert generator.calls == 0


@pytest.mark.asyncio
async def test_reviewer_failure_retains_verified_selected_evidence_for_audit() -> None:
    result = await run_knowledge_qa(
        _graph(reviewer=RecordingReviewer(fails=True)), user_query="测试问题"
    )

    assert result.answerability is Answerability.FAILED
    assert tuple(item.evidence_id for item in result.selected_evidence) == ("E1",)
    assert result.selected_evidence_context.startswith("[E1]")
    assert result.errors[0].code == "answerability_review_failed"


@pytest.mark.asyncio
async def test_selector_provider_subcode_is_safe_and_clears_unverified_selection() -> None:
    class SelectorWithProviderFailure:
        async def select(self, **_: object) -> object:
            raise EvidenceSelectionError("selector_timeout")

    result = await run_knowledge_qa(
        _graph(selector=SelectorWithProviderFailure()), user_query="测试问题"
    )

    assert result.answerability is Answerability.FAILED
    assert result.errors[0].code == "evidence_selection_failed"
    assert result.errors[0].details == {"subcode": "selector_timeout"}
    assert result.selected_evidence == ()
    assert result.selected_evidence_context == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "generator",
    [
        RecordingGenerator(fails=True),
        RecordingGenerator("not a structured draft"),
        RecordingGenerator(AnswerDraft(answer="Uncited answer.", citations=["[E1]"])),
        RecordingGenerator(AnswerDraft(answer="Unsupported.[E2]", citations=["[E2]"])),
    ],
)
async def test_invalid_generator_output_enters_safe_failed_state(
    generator: RecordingGenerator,
) -> None:
    result = await run_knowledge_qa(_graph(generator=generator), user_query="测试问题")

    assert result.answerability is Answerability.FAILED
    assert result.answer is None and result.citations == []
    assert result.missing_information is None and result.decision_reason is None
    assert result.errors[0].code in {
        "answer_generation_failed",
        "citations_not_present_in_answer",
        "citation_not_in_evidence",
    }


@pytest.mark.asyncio
async def test_failed_upstream_short_circuits_reviewer_and_generator() -> None:
    reviewer = RecordingReviewer()
    generator = RecordingGenerator()
    result = await run_knowledge_qa(
        _graph(pipeline=FakePipeline(fails=True), reviewer=reviewer, generator=generator),
        user_query="测试问题",
    )

    assert result.answerability is Answerability.FAILED
    assert result.errors[0].code == "retrieval_failed"
    assert reviewer.calls == 0 and generator.calls == 0


@pytest.mark.asyncio
async def test_selector_failure_short_circuits_reviewer_and_generator() -> None:
    reviewer = RecordingReviewer()
    generator = RecordingGenerator()
    result = await run_knowledge_qa(
        _graph(selector=RecordingSelector(fails=True), reviewer=reviewer, generator=generator),
        user_query="测试问题",
    )

    assert result.answerability is Answerability.FAILED
    assert result.errors[0].code == "evidence_selection_failed"
    assert reviewer.calls == 0 and generator.calls == 0
