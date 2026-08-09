"""Versioned CrossEncoder reranking evaluation over immutable RRF candidates."""
# ruff: noqa: E501

from __future__ import annotations

import json
import math
import platform
import time
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from decision_agent.evaluation.dataset import compute_normalized_text_sha256, load_retrieval_dataset
from decision_agent.evaluation.reporting import write_json_report_atomically
from decision_agent.evaluation.retrieval_metrics import aggregate_metrics, compute_query_metrics
from decision_agent.evaluation.rrf_baseline import RRFBaselineReport
from decision_agent.exceptions import EvaluationValidationError
from decision_agent.retrieval.reranking import RerankCandidate, Reranker

EXPECTED_RRF_BASELINE = "m2b4-rrf-dense-bm25-k60"


class RerankerReportModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RerankerBaselineConfig(RerankerReportModel):
    rrf_report_path: Path
    corpus_path: Path
    queries_path: Path
    output_path: Path
    model_name: Literal["BAAI/bge-reranker-base"] = "BAAI/bge-reranker-base"
    model_revision: str | None = None
    device: Literal["cpu"] = "cpu"
    batch_size: int = Field(default=8, gt=0)
    input_depth: Literal[5] = 5
    output_top_k: Literal[5] = 5


class SourceAsset(RerankerReportModel):
    relative_path: str
    normalized_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def safe_path(self) -> SourceAsset:
        path = PurePosixPath(self.relative_path)
        if (
            Path(self.relative_path).is_absolute()
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in self.relative_path
        ):
            raise ValueError("source asset path must be a safe POSIX relative path")
        return self


class RerankerRankedResult(RerankerReportModel):
    document_id: str = Field(min_length=1)
    upstream_rank: int = Field(gt=0)
    upstream_rrf_score: float = Field(gt=0, allow_inf_nan=False)
    reranker_score: float = Field(allow_inf_nan=False)
    final_rank: int = Field(gt=0)
    is_relevant: bool


class RerankerQueryResult(RerankerReportModel):
    query_id: str
    query: str
    category: str
    relevant_document_ids: tuple[str, ...]
    upstream_rrf_ranking: tuple[str, ...]
    reranked_results: tuple[RerankerRankedResult, ...]
    metrics: dict[str, float]
    upstream_best_relevant_rank: int | None = None
    reranked_best_relevant_rank: int | None = None
    comparison_to_rrf: Literal["improved", "unchanged", "regressed"]

    @property
    def ranked_document_ids(self) -> tuple[str, ...]:
        return tuple(item.document_id for item in self.reranked_results)

    @model_validator(mode="after")
    def valid_ranking(self) -> RerankerQueryResult:
        if [item.final_rank for item in self.reranked_results] != list(
            range(1, len(self.reranked_results) + 1)
        ):
            raise ValueError("reranked ranks must be consecutive and start at one")
        ids = self.ranked_document_ids
        if len(ids) != len(set(ids)) or len(self.upstream_rrf_ranking) != len(
            set(self.upstream_rrf_ranking)
        ):
            raise ValueError("query rankings must have unique document IDs")
        if set(ids) != set(self.upstream_rrf_ranking):
            raise ValueError("reranking must preserve the upstream candidate set")
        relevant = set(self.relevant_document_ids)
        if any(
            item.is_relevant != (item.document_id in relevant) for item in self.reranked_results
        ):
            raise ValueError("relevance markers must match labels")
        return self


class RerankerRuntimeObservations(RerankerReportModel):
    model_load_seconds: float = Field(ge=0, allow_inf_nan=False)
    total_rerank_seconds: float = Field(ge=0, allow_inf_nan=False)
    average_query_rerank_ms: float = Field(ge=0, allow_inf_nan=False)
    average_pair_rerank_ms: float = Field(ge=0, allow_inf_nan=False)
    total_runtime_seconds: float = Field(ge=0, allow_inf_nan=False)


class RerankerComparison(RerankerReportModel):
    rrf_metrics: dict[str, float]
    reranker_metrics: dict[str, float]
    improved_query_ids: tuple[str, ...]
    regressed_query_ids: tuple[str, ...]
    unchanged_query_ids: tuple[str, ...]
    top1_fixed_query_ids: tuple[str, ...]
    top1_broken_query_ids: tuple[str, ...]


class RerankerBaselineReport(RerankerReportModel):
    evaluation_schema_version: str = "1.0"
    baseline_name: str = "m2b5-bge-reranker-base"
    retrieval_type: Literal["cross_encoder_reranker"] = "cross_encoder_reranker"
    model_name: str
    model_revision: str | None
    device: Literal["cpu"]
    score_semantics: Literal["raw_model_score_higher_is_more_relevant"] = (
        "raw_model_score_higher_is_more_relevant"
    )
    batch_size: int
    input_depth: int
    output_top_k: int
    source_rrf_report: SourceAsset
    corpus: SourceAsset
    queries: SourceAsset
    corpus_size: int
    query_count: int
    total_candidate_pairs: int
    metrics: dict[str, float]
    category_metrics: dict[str, dict[str, float]]
    query_results: tuple[RerankerQueryResult, ...]
    comparison: RerankerComparison
    runtime_observations: RerankerRuntimeObservations
    python_version: str
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def valid_report(self) -> RerankerBaselineReport:
        if (
            self.query_count != len(self.query_results)
            or self.total_candidate_pairs != self.query_count * self.input_depth
        ):
            raise ValueError("report counts are inconsistent")
        if self.comparison.reranker_metrics != self.metrics:
            raise ValueError("comparison metrics must match report metrics")
        expected = self.runtime_observations.total_rerank_seconds * 1000 / self.query_count
        if not math.isclose(
            self.runtime_observations.average_query_rerank_ms,
            expected,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("average query runtime must match total rerank time")
        return self


def _relative_path(path: Path) -> str:
    parts = list(path.parts)
    for name in ("artifacts", "datasets"):
        if name in [part.lower() for part in parts]:
            index = [part.lower() for part in parts].index(name)
            return Path(*parts[index:]).as_posix()
    return path.name


def _load_rrf(path: Path) -> RRFBaselineReport:
    try:
        return RRFBaselineReport.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValidationError, json.JSONDecodeError) as exc:
        raise EvaluationValidationError("failed to read validated RRF report") from exc


def _rank(ids: tuple[str, ...], relevant: tuple[str, ...]) -> int | None:
    relevant_ids = set(relevant)
    return next(
        (index for index, document_id in enumerate(ids, start=1) if document_id in relevant_ids),
        None,
    )


def _comparison(current: int | None, upstream: int | None) -> str:
    current_value, upstream_value = (
        math.inf if current is None else current,
        math.inf if upstream is None else upstream,
    )
    return (
        "improved"
        if current_value < upstream_value
        else "regressed"
        if current_value > upstream_value
        else "unchanged"
    )


async def run_reranker_baseline(
    config: RerankerBaselineConfig, reranker: Reranker
) -> RerankerBaselineReport:
    """Rerank fixed RRF candidates and atomically write an auditable report."""
    started = time.perf_counter()
    hashes = {
        "rrf": compute_normalized_text_sha256(config.rrf_report_path, dataset_name="RRF report"),
        "corpus": compute_normalized_text_sha256(config.corpus_path, dataset_name="corpus"),
        "queries": compute_normalized_text_sha256(config.queries_path, dataset_name="queries"),
    }
    rrf = _load_rrf(config.rrf_report_path)
    dataset = load_retrieval_dataset(config.corpus_path, config.queries_path)
    if (
        rrf.baseline_name != EXPECTED_RRF_BASELINE
        or rrf.corpus_sha256 != hashes["corpus"]
        or rrf.queries_sha256 != hashes["queries"]
    ):
        raise EvaluationValidationError("RRF report and evaluation datasets are incompatible")
    if rrf.corpus_size != len(dataset.corpus) or rrf.query_count != len(dataset.queries):
        raise EvaluationValidationError("RRF report dataset counts are incompatible")
    corpus = {document.document_id: document.text for document in dataset.corpus}
    queries = {query.query_id: query for query in dataset.queries}
    if set(queries) != {result.query_id for result in rrf.query_results}:
        raise EvaluationValidationError("RRF report query IDs differ from queries dataset")

    results: list[RerankerQueryResult] = []
    rerank_started = time.perf_counter()
    for upstream in rrf.query_results:
        query = queries[upstream.query_id]
        if (query.query, query.category, query.relevant_document_ids) != (
            upstream.query,
            upstream.category,
            upstream.relevant_document_ids,
        ):
            raise EvaluationValidationError(f"RRF query metadata differs: {query.query_id}")
        if len(upstream.ranked_results) != config.input_depth or [
            item.final_rank for item in upstream.ranked_results
        ] != list(range(1, config.input_depth + 1)):
            raise EvaluationValidationError(
                f"RRF ranking depth or ranks are invalid: {query.query_id}"
            )
        missing = [
            item.document_id for item in upstream.ranked_results if item.document_id not in corpus
        ]
        if missing:
            raise EvaluationValidationError(f"RRF document_id is absent from corpus: {missing[0]}")
        candidates = [
            RerankCandidate(
                document_id=item.document_id,
                content=corpus[item.document_id],
                upstream_rank=item.final_rank,
                upstream_score=item.fused_score,
            )
            for item in upstream.ranked_results
        ]
        ranked = await reranker.rerank(query.query, candidates, top_k=config.output_top_k)
        if len(ranked) != config.output_top_k:
            raise EvaluationValidationError("reranker returned an unexpected number of candidates")
        ranked_results = tuple(
            RerankerRankedResult(
                document_id=item.document_id,
                upstream_rank=item.upstream_rank,
                upstream_rrf_score=item.upstream_score or 0.0,
                reranker_score=item.reranker_score,
                final_rank=item.final_rank,
                is_relevant=item.document_id in query.relevant_document_ids,
            )
            for item in ranked
        )
        ids = tuple(item.document_id for item in ranked_results)
        metrics = compute_query_metrics(ids, query.relevant_document_ids, ks=(1, 3, 5), mrr_k=5)
        upstream_ids = tuple(item.document_id for item in upstream.ranked_results)
        results.append(
            RerankerQueryResult(
                query_id=query.query_id,
                query=query.query,
                category=query.category,
                relevant_document_ids=query.relevant_document_ids,
                upstream_rrf_ranking=upstream_ids,
                reranked_results=ranked_results,
                metrics=metrics,
                upstream_best_relevant_rank=_rank(upstream_ids, query.relevant_document_ids),
                reranked_best_relevant_rank=_rank(ids, query.relevant_document_ids),
                comparison_to_rrf=_comparison(
                    _rank(ids, query.relevant_document_ids),
                    _rank(upstream_ids, query.relevant_document_ids),
                ),
            )
        )
    measured_rerank_seconds = time.perf_counter() - rerank_started
    if any(
        compute_normalized_text_sha256(path, dataset_name=name) != hashes[key]
        for key, name, path in (
            ("rrf", "RRF report", config.rrf_report_path),
            ("corpus", "corpus", config.corpus_path),
            ("queries", "queries", config.queries_path),
        )
    ):
        raise EvaluationValidationError("source asset changed during reranker baseline execution")
    overall, categories = aggregate_metrics([(item.category, item.metrics) for item in results])
    groups = {
        name: tuple(item.query_id for item in results if item.comparison_to_rrf == name)
        for name in ("improved", "regressed", "unchanged")
    }
    top1_fixed = tuple(
        item.query_id
        for item in results
        if item.upstream_best_relevant_rank != 1 and item.reranked_best_relevant_rank == 1
    )
    top1_broken = tuple(
        item.query_id
        for item in results
        if item.upstream_best_relevant_rank == 1 and item.reranked_best_relevant_rank != 1
    )
    load_seconds = getattr(reranker, "model_load_seconds", 0.0) or 0.0
    rerank_seconds = getattr(reranker, "total_predict_seconds", measured_rerank_seconds)
    if not isinstance(rerank_seconds, (int, float)) or not math.isfinite(rerank_seconds):
        raise EvaluationValidationError("reranker predict timing must be a finite number")
    report = RerankerBaselineReport(
        model_name=config.model_name,
        model_revision=config.model_revision,
        device=config.device,
        batch_size=config.batch_size,
        input_depth=config.input_depth,
        output_top_k=config.output_top_k,
        source_rrf_report=SourceAsset(
            relative_path=_relative_path(config.rrf_report_path),
            normalized_text_sha256=hashes["rrf"],
        ),
        corpus=SourceAsset(
            relative_path=_relative_path(config.corpus_path),
            normalized_text_sha256=hashes["corpus"],
        ),
        queries=SourceAsset(
            relative_path=_relative_path(config.queries_path),
            normalized_text_sha256=hashes["queries"],
        ),
        corpus_size=len(dataset.corpus),
        query_count=len(dataset.queries),
        total_candidate_pairs=len(results) * config.input_depth,
        metrics=overall,
        category_metrics=categories,
        query_results=tuple(results),
        comparison=RerankerComparison(
            rrf_metrics=dict(rrf.metrics),
            reranker_metrics=overall,
            improved_query_ids=groups["improved"],
            regressed_query_ids=groups["regressed"],
            unchanged_query_ids=groups["unchanged"],
            top1_fixed_query_ids=top1_fixed,
            top1_broken_query_ids=top1_broken,
        ),
        runtime_observations=RerankerRuntimeObservations(
            model_load_seconds=load_seconds,
            total_rerank_seconds=rerank_seconds,
            average_query_rerank_ms=rerank_seconds * 1000 / len(results),
            average_pair_rerank_ms=rerank_seconds * 1000 / (len(results) * config.input_depth),
            total_runtime_seconds=time.perf_counter() - started,
        ),
        python_version=platform.python_version(),
        limitations=(
            "Metrics cover only the fixed 36-document/18-query synthetic dataset.",
            "The cross-encoder reranks only fixed RRF Top-5 candidates; it does not retrieve the corpus.",
            "CPU timings are one local observation, not a production benchmark.",
        ),
    )
    write_json_report_atomically(config.output_path, report.model_dump(mode="json"))
    return report
