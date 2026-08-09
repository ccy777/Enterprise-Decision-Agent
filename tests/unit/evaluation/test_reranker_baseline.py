from pathlib import Path

import pytest

from decision_agent.evaluation.reranker_baseline import (
    RerankerBaselineConfig,
    run_reranker_baseline,
)
from decision_agent.retrieval.reranking import RerankCandidate, RerankedResult

ROOT = Path(__file__).resolve().parents[3]
RRF = ROOT / "artifacts/evaluation/m2b4_rrf_dense_bm25_k60.json"
CORPUS = ROOT / "datasets/retrieval/m2b2c_dense_corpus.jsonl"
QUERIES = ROOT / "datasets/retrieval/m2b2c_dense_queries.jsonl"


class EchoReranker:
    model_load_seconds = 0.0
    total_predict_seconds = 0.25

    async def rerank(
        self, query: str, candidates: list[RerankCandidate], *, top_k: int | None = None
    ) -> list[RerankedResult]:
        return [
            RerankedResult(
                final_rank=index,
                document_id=item.document_id,
                content=item.content,
                reranker_score=-item.upstream_rank,
                upstream_rank=item.upstream_rank,
                upstream_score=item.upstream_score,
            )
            for index, item in enumerate(candidates[:top_k], 1)
        ]


def config(tmp_path: Path, **kwargs: Path) -> RerankerBaselineConfig:
    return RerankerBaselineConfig(
        rrf_report_path=kwargs.get("rrf_report_path", RRF),
        corpus_path=kwargs.get("corpus_path", CORPUS),
        queries_path=kwargs.get("queries_path", QUERIES),
        output_path=tmp_path / "reranker.json",
    )


@pytest.mark.asyncio
async def test_baseline_preserves_candidate_sets_and_recomputes_metrics(tmp_path: Path) -> None:
    report = await run_reranker_baseline(config(tmp_path), EchoReranker())
    assert report.total_candidate_pairs == 90
    assert report.runtime_observations.total_rerank_seconds == 0.25
    assert report.runtime_observations.average_query_rerank_ms == pytest.approx(0.25 / 18 * 1000)
    assert report.runtime_observations.average_pair_rerank_ms == pytest.approx(0.25 / 90 * 1000)
    assert report.metrics == report.comparison.rrf_metrics
    assert report.source_rrf_report.relative_path.startswith("artifacts/")
    assert all(
        set(item.upstream_rrf_ranking) == set(item.ranked_document_ids)
        for item in report.query_results
    )
    assert str(ROOT).lower() not in (tmp_path / "reranker.json").read_text(encoding="utf-8").lower()


@pytest.mark.asyncio
async def test_baseline_rejects_missing_rrf_candidate_document(tmp_path: Path) -> None:
    text = RRF.read_text(encoding="utf-8").replace(
        '"product-b-battery-warranty"', '"missing-document"', 1
    )
    changed = tmp_path / "rrf.json"
    changed.write_text(text, encoding="utf-8")
    with pytest.raises(Exception, match="RRF"):
        await run_reranker_baseline(config(tmp_path, rrf_report_path=changed), EchoReranker())
