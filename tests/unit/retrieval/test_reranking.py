import asyncio
import math

import pytest

from decision_agent.exceptions import RerankerInferenceError, RetrievalValidationError
from decision_agent.retrieval.reranking import (
    RerankCandidate,
    SentenceTransformerCrossEncoderReranker,
)


class FakeCrossEncoder:
    def __init__(self, scores: list[float]) -> None:
        self.scores, self.calls, self.pairs = scores, 0, []

    def predict(self, pairs: list[tuple[str, str]], **kwargs: object) -> list[float]:
        self.calls += 1
        self.pairs = pairs
        return self.scores


def candidates() -> list[RerankCandidate]:
    return [
        RerankCandidate(document_id="b", content="second", upstream_rank=1, metadata={"a": 1}),
        RerankCandidate(document_id="a", content="first", upstream_rank=2),
    ]


@pytest.mark.asyncio
async def test_empty_candidates_do_not_load_model() -> None:
    called = False

    def factory(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        return FakeCrossEncoder([])

    reranker = SentenceTransformerCrossEncoderReranker(model_factory=factory)
    assert await reranker.rerank("query", []) == []
    assert not called


@pytest.mark.asyncio
async def test_batch_pairs_sorting_and_deep_copy() -> None:
    model = FakeCrossEncoder([0.1, 0.9])
    source = candidates()
    results = await SentenceTransformerCrossEncoderReranker(model=model).rerank("query", source)
    assert model.pairs == [("query", "second"), ("query", "first")]
    assert [item.document_id for item in results] == ["a", "b"]
    source[0].metadata["a"] = 2
    assert results[1].metadata == {"a": 1}


@pytest.mark.asyncio
async def test_ties_keep_upstream_order_and_top_k() -> None:
    model = FakeCrossEncoder([1.0, 1.0])
    results = await SentenceTransformerCrossEncoderReranker(model=model).rerank(
        "q", candidates(), top_k=1
    )
    assert [(item.final_rank, item.document_id) for item in results] == [(1, "b")]


@pytest.mark.asyncio
@pytest.mark.parametrize("scores", [[math.nan, 1.0], [math.inf, 1.0], [1.0]])
async def test_invalid_model_scores_are_rejected(scores: list[float]) -> None:
    with pytest.raises(RerankerInferenceError):
        await SentenceTransformerCrossEncoderReranker(model=FakeCrossEncoder(scores)).rerank(
            "q", candidates()
        )


@pytest.mark.asyncio
async def test_concurrent_first_requests_load_once() -> None:
    count = 0

    def factory(*args: object, **kwargs: object) -> FakeCrossEncoder:
        nonlocal count
        count += 1
        return FakeCrossEncoder([1.0, 0.0])

    reranker = SentenceTransformerCrossEncoderReranker(model_factory=factory)
    await asyncio.gather(reranker.rerank("q", candidates()), reranker.rerank("q", candidates()))
    assert count == 1


@pytest.mark.asyncio
async def test_invalid_candidate_contracts_are_rejected() -> None:
    reranker = SentenceTransformerCrossEncoderReranker(model=FakeCrossEncoder([1.0, 0.0]))
    with pytest.raises(RetrievalValidationError, match="consecutive"):
        await reranker.rerank("q", [candidates()[1]])
    with pytest.raises(RetrievalValidationError, match="query"):
        await reranker.rerank(" ", candidates())
