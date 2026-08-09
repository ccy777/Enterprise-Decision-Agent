"""Opt-in A2-3 real Retrieval, Reviewer, and Generator smoke coverage."""

# ruff: noqa: RUF001

from __future__ import annotations

import json
import os
from collections.abc import Sequence

import pytest

from decision_agent.agents.answerability_reviewer import OpenAICompatibleAnswerabilityReviewer
from decision_agent.agents.evidence_selector import OpenAICompatibleEvidenceSelector
from decision_agent.agents.grounded_answer import AnswerDraft, OpenAICompatibleAnswerGenerator
from decision_agent.config import Settings
from decision_agent.retrieval.evidence_context import EvidenceItem
from decision_agent.retrieval.factory import build_enterprise_retrieval_pipeline
from decision_agent.workflows.knowledge_qa import (
    Answerability,
    KnowledgeQAState,
    build_knowledge_qa_graph,
    run_knowledge_qa,
)

pytestmark = pytest.mark.integration

_QUERIES = (
    "产品 A 的原装电池保修期多久？",
    "普通采购申请金额为 18 万元，需要谁审批？",
    "访问 L3 数据需要经过哪些审批？",
    "产品 A 的维修完成后，公司承诺免费保修多少天？",
)


class CountingGenerator:
    """Record calls without exposing a provider response or configuration."""

    def __init__(self, delegate: OpenAICompatibleAnswerGenerator) -> None:
        self._delegate = delegate
        self.calls = 0

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
        self.calls += 1
        return await self._delegate.generate(
            user_query=user_query,
            selected_evidence_context=selected_evidence_context,
            selected_evidence=selected_evidence,
            answerability=answerability,
            missing_information=missing_information,
            decision_reason=decision_reason,
        )


def _contains_chinese(value: str | None) -> bool:
    return bool(value and any("\u4e00" <= character <= "\u9fff" for character in value))


def _safe_audit(state: KnowledgeQAState, *, generator_called: bool) -> dict[str, object]:
    return {
        "raw_evidence": [
            {"evidence_id": item.evidence_id, "document_id": item.document_id}
            for item in state.retrieval_evidence
        ],
        "selected_evidence": [
            {"evidence_id": item.evidence_id, "document_id": item.document_id}
            for item in state.selected_evidence
        ],
        "answerability": state.answerability,
        "decision_reason": state.decision_reason,
        "missing_information": state.missing_information,
        "answer": state.answer,
        "citations": state.citations,
        "generator_called": generator_called,
        "error_codes": [
            {"code": error.code, "subcode": error.details.get("subcode")} for error in state.errors
        ],
    }


@pytest.mark.asyncio
async def test_real_llm_four_question_smoke() -> None:
    if os.getenv("RUN_KNOWLEDGE_QA_REAL_LLM_SMOKE") != "1":
        pytest.skip("set RUN_KNOWLEDGE_QA_REAL_LLM_SMOKE=1 to run A2-3 Level 3 smoke")

    settings = Settings()
    pipeline = build_enterprise_retrieval_pipeline("datasets/enterprise_kb/m2c1")
    generator = CountingGenerator(OpenAICompatibleAnswerGenerator.from_settings(settings))
    try:
        await pipeline.initialize()
        graph = build_knowledge_qa_graph(
            retrieval_pipeline=pipeline,
            evidence_selector=OpenAICompatibleEvidenceSelector.from_settings(settings),
            answerability_reviewer=OpenAICompatibleAnswerabilityReviewer.from_settings(settings),
            answer_generator=generator,
        )
        results_with_generator_calls = []
        for query in _QUERIES:
            calls_before = generator.calls
            result = await run_knowledge_qa(graph, user_query=query)
            results_with_generator_calls.append((result, generator.calls > calls_before))
    finally:
        await pipeline.close()

    results = [result for result, _ in results_with_generator_calls]
    audits = [
        _safe_audit(result, generator_called=generator_called)
        for result, generator_called in results_with_generator_calls
    ]
    print(json.dumps(audits, ensure_ascii=False))

    battery, procurement, l3, q010 = results
    assert battery.answerability is Answerability.ANSWERABLE
    assert any(term in (battery.answer or "") for term in ("12 个月", "十二个月"))
    assert battery.citations
    assert procurement.answerability is Answerability.ANSWERABLE
    assert "采购总监" in (procurement.answer or "") and procurement.citations
    assert l3.answerability is Answerability.ANSWERABLE
    assert "部门负责人" in (l3.answer or "") and "信息安全办公室" in (l3.answer or "")
    assert l3.citations
    assert q010.answerability is Answerability.UNANSWERABLE
    assert q010.citations == []
    assert _contains_chinese(q010.decision_reason)
    assert _contains_chinese(q010.missing_information)
    assert q010.answer is not None and not any(character.isdigit() for character in q010.answer)
    assert generator.calls == 3
