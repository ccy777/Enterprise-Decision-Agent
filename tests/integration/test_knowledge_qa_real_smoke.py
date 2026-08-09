"""Opt-in real retrieval plus deterministic evidence-bound A2-3 smoke coverage."""

# ruff: noqa: RUF001

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from decision_agent.agents.answerability_reviewer import AnswerabilityDecision
from decision_agent.agents.evidence_selector import EvidenceSelection
from decision_agent.agents.grounded_answer import AnswerDraft
from decision_agent.retrieval.evidence_context import EvidenceItem
from decision_agent.retrieval.factory import build_enterprise_retrieval_pipeline
from decision_agent.workflows.knowledge_qa import (
    Answerability,
    build_knowledge_qa_graph,
    run_knowledge_qa,
)

pytestmark = pytest.mark.integration


@dataclass(frozen=True)
class EvidenceBoundCase:
    """Fixture facts used only when current selected Evidence explicitly contains them."""

    selection_fact: str
    required_fact: str
    answerability: str
    answer: str
    missing_information: str | None
    decision_reason: str


_CASES = (
    EvidenceBoundCase(
        selection_fact="A 型原装电池基础保修期为十二个月",
        required_fact="A 型原装电池基础保修期为十二个月",
        answerability="answerable",
        answer="产品 A 的原装电池保修期为 12 个月。",
        missing_information=None,
        decision_reason="选中的证据明确规定了 A 型原装电池的基础保修期。",
    ),
    EvidenceBoundCase(
        selection_fact="普通采购金额五万元及以上、二十万元以下，由采购总监审批",
        required_fact="普通采购金额五万元及以上、二十万元以下，由采购总监审批",
        answerability="answerable",
        answer="18 万元普通采购属于 5 万元及以上、20 万元以下区间，由采购总监审批。",
        missing_information=None,
        decision_reason="选中的证据明确规定了该金额区间的审批主体。",
    ),
    EvidenceBoundCase(
        selection_fact="数据访问须由部门负责人和信息安全办公室审批",
        required_fact="数据访问须由部门负责人和信息安全办公室审批",
        answerability="answerable",
        answer="访问 L3 数据须经部门负责人和信息安全办公室审批，两项批准缺一不可。",
        missing_information=None,
        decision_reason="选中的证据明确规定了 L3 数据访问的两项审批。",
    ),
    EvidenceBoundCase(
        selection_fact="维修流程包括初诊",
        required_fact="维修完成后的新增免费保修期限",
        answerability="unanswerable",
        answer="",
        missing_information="维修完成后的新增免费保修期限",
        decision_reason="选中的证据只规定维修流程，没有规定维修完成后的新增免费保修期限。",
    ),
)


class EvidenceBoundSelector:
    """Select only fixture facts that are present in this request's actual Evidence."""

    def __init__(self) -> None:
        self.case: EvidenceBoundCase | None = None

    async def select(
        self,
        *,
        user_query: str,
        evidence_context: str,
        retrieval_evidence: Sequence[EvidenceItem],
    ) -> EvidenceSelection:
        del user_query, evidence_context
        assert self.case is not None
        selected = [item for item in retrieval_evidence if self.case.selection_fact in item.content]
        return EvidenceSelection(
            selected_evidence_ids=[f"[{item.evidence_id}]" for item in selected],
            selection_reason="Selected fixture-direct Evidence."
            if selected
            else "No fixture-direct Evidence matched.",
        )


class EvidenceBoundReviewer:
    """Decide only from the fact visible in the selected Evidence subset."""

    def __init__(self) -> None:
        self.case: EvidenceBoundCase | None = None

    async def review(
        self,
        *,
        user_query: str,
        selected_evidence_context: str,
        selected_evidence: Sequence[EvidenceItem],
    ) -> AnswerabilityDecision:
        del user_query, selected_evidence_context
        assert self.case is not None
        required_fact_is_present = any(
            self.case.required_fact in item.content for item in selected_evidence
        )
        if self.case.answerability == "answerable":
            assert required_fact_is_present
            return AnswerabilityDecision(
                answerability="answerable",
                missing_information=None,
                decision_reason=self.case.decision_reason,
            )
        assert not required_fact_is_present
        return AnswerabilityDecision(
            answerability="unanswerable",
            missing_information=self.case.missing_information,
            decision_reason=self.case.decision_reason,
        )


class EvidenceBoundGenerator:
    """Generate only from the selected direct fact, never from question text or labels."""

    def __init__(self) -> None:
        self.calls = 0
        self.case: EvidenceBoundCase | None = None

    async def generate(
        self,
        *,
        user_query: str,
        selected_evidence_context: str,
        selected_evidence: Sequence[EvidenceItem],
        answerability: str,
        missing_information: str | None,
        decision_reason: str,
    ) -> AnswerDraft:
        del user_query, selected_evidence_context, missing_information, decision_reason
        assert answerability == "answerable"
        self.calls += 1
        assert self.case is not None and self.case.answerability == "answerable"
        matches = [item for item in selected_evidence if self.case.required_fact in item.content]
        if not matches:
            raise AssertionError("generator received no fixture-direct Evidence")
        citation = f"[{matches[0].evidence_id}]"
        return AnswerDraft(answer=f"{self.case.answer}{citation}", citations=[citation])


@pytest.mark.asyncio
async def test_real_retrieval_four_question_smoke() -> None:
    if os.getenv("RUN_KNOWLEDGE_QA_SMOKE") != "1":
        pytest.skip("set RUN_KNOWLEDGE_QA_SMOKE=1 to run cached real retrieval smoke")

    queries = (
        "产品 A 的原装电池保修期多久？",
        "普通采购申请金额为 18 万元，需要谁审批？",
        "访问 L3 数据需要经过哪些审批？",
        "产品 A 的维修完成后，公司承诺免费保修多少天？",
    )
    generator = EvidenceBoundGenerator()
    selector = EvidenceBoundSelector()
    reviewer = EvidenceBoundReviewer()
    pipeline = build_enterprise_retrieval_pipeline("datasets/enterprise_kb/m2c1")
    try:
        await pipeline.initialize()
        graph = build_knowledge_qa_graph(
            retrieval_pipeline=pipeline,
            evidence_selector=selector,
            answerability_reviewer=reviewer,
            answer_generator=generator,
        )
        results = []
        for query, case in zip(queries, _CASES, strict=True):
            selector.case = reviewer.case = generator.case = case
            results.append(await run_knowledge_qa(graph, user_query=query))
    finally:
        await pipeline.close()

    audit_output = [
        {
            "query": query,
            "raw_evidence": [
                {"evidence_id": item.evidence_id, "document_id": item.document_id}
                for item in result.retrieval_evidence
            ],
            "selected_evidence": [
                {"evidence_id": item.evidence_id, "document_id": item.document_id}
                for item in result.selected_evidence
            ],
            "answerability": result.answerability,
            "missing_information": result.missing_information,
            "decision_reason": result.decision_reason,
            "answer": result.answer,
            "citations": result.citations,
            "generator_called": result.answerability is Answerability.ANSWERABLE,
        }
        for query, result in zip(queries, results, strict=True)
    ]
    print(json.dumps(audit_output, ensure_ascii=False))

    assert [result.answerability for result in results] == [
        Answerability.ANSWERABLE,
        Answerability.ANSWERABLE,
        Answerability.ANSWERABLE,
        Answerability.UNANSWERABLE,
    ]
    assert "12 个月" in (results[0].answer or "")
    assert "采购总监" in (results[1].answer or "")
    assert "部门负责人" in (results[2].answer or "")
    assert "信息安全办公室" in (results[2].answer or "")
    q010 = results[3]
    assert q010.citations == []
    assert q010.answer is not None and not any(character.isdigit() for character in q010.answer)
    assert generator.calls == 3
