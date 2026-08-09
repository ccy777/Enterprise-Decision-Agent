"""Opt-in smoke test for the cached official BGE CrossEncoder model."""

import os

import pytest

from decision_agent.retrieval.reranking import (
    RerankCandidate,
    SentenceTransformerCrossEncoderReranker,
)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_bge_reranker_smoke() -> None:
    if os.getenv("RUN_BGE_RERANKER_INTEGRATION") != "1":
        pytest.skip("set RUN_BGE_RERANKER_INTEGRATION=1 to run the real BGE reranker")
    reranker = SentenceTransformerCrossEncoderReranker()
    results = await reranker.rerank(
        "产品A电池保修多久?",
        [
            RerankCandidate(
                document_id="product-a-battery-warranty",
                content="产品A的电池保修期为一年。",
                upstream_rank=1,
            ),
            RerankCandidate(
                document_id="product-b-battery-warranty",
                content="产品B的电池保修期为六个月。",
                upstream_rank=2,
            ),
        ],
    )
    assert len(results) == 2
    assert all(result.reranker_score == result.reranker_score for result in results)
