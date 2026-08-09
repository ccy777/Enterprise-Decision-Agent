"""Opt-in 2-2 real-RAG plus real-Selector smoke without answer generation."""

# ruff: noqa: RUF001

from __future__ import annotations

import json
import os

import pytest

from decision_agent.agents.evidence_selector import (
    OpenAICompatibleEvidenceSelector,
    validate_evidence_selection,
)
from decision_agent.config import Settings
from decision_agent.retrieval.factory import build_enterprise_retrieval_pipeline

pytestmark = pytest.mark.integration

_CASES = (
    ("产品 A 的原装电池保修期多久？", "DOC-CS-001", "DOC-INV-001"),
    ("普通采购申请金额为 18 万元，需要谁审批？", "DOC-PROC-001", "DOC-FIN-001"),
    ("访问 L3 数据需要经过哪些审批？", "DOC-SEC-001", "DOC-HR-001"),
    ("产品 A 的维修完成后，公司承诺免费保修多少天？", None, None),
)


@pytest.mark.asyncio
async def test_real_selector_four_question_smoke() -> None:
    if os.getenv("RUN_EVIDENCE_SELECTOR_REAL_LLM_SMOKE") != "1":
        pytest.skip("set RUN_EVIDENCE_SELECTOR_REAL_LLM_SMOKE=1 to run 2-2 selector smoke")

    settings = Settings()
    selector = OpenAICompatibleEvidenceSelector.from_settings(settings)
    pipeline = build_enterprise_retrieval_pipeline("datasets/enterprise_kb/m2c1")
    try:
        await pipeline.initialize()
        audits = []
        for query, expected_document, excluded_document in _CASES:
            retrieval = await pipeline.retrieve(query)
            raw = retrieval.evidence_context.evidence_items
            selection = await selector.select(
                user_query=query,
                evidence_context=retrieval.evidence_context.rendered_context,
                retrieval_evidence=raw,
            )
            validation = validate_evidence_selection(
                evidence_ids=[item.evidence_id for item in raw], selection=selection
            )
            assert validation.validation_passed
            selected_ids = set(validation.normalized_selected_evidence_ids)
            selected = [item for item in raw if f"[{item.evidence_id}]" in selected_ids]
            audits.append(
                {
                    "query": query,
                    "raw_evidence": [
                        {"evidence_id": item.evidence_id, "document_id": item.document_id}
                        for item in raw
                    ],
                    "selected_evidence": [
                        {"evidence_id": item.evidence_id, "document_id": item.document_id}
                        for item in selected
                    ],
                    "selection_reason": selection.selection_reason,
                }
            )
            if expected_document is not None:
                assert any(item.document_id == expected_document for item in selected)
            if excluded_document is not None:
                assert all(item.document_id != excluded_document for item in selected)
    finally:
        await pipeline.close()

    print(json.dumps(audits, ensure_ascii=False))
