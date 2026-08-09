"""Deterministic BM25 baseline over the versioned retrieval dataset."""

from __future__ import annotations

import json
import math
import platform
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from decision_agent.evaluation.dataset import (
    compute_normalized_text_sha256,
    load_retrieval_dataset,
)
from decision_agent.evaluation.dense_baseline import DenseBaselineReport
from decision_agent.evaluation.reporting import write_json_report_atomically
from decision_agent.evaluation.retrieval_metrics import aggregate_metrics, compute_query_metrics
from decision_agent.exceptions import EvaluationValidationError
from decision_agent.retrieval import (
    BM25Document,
    BM25Index,
    BM25Retriever,
    DeterministicChineseTokenizer,
)


class BM25ReportModel(BaseModel):
    """Strict JSON-safe base for BM25 evaluation artifacts."""

    model_config = ConfigDict(extra="forbid")


def _validate_metrics(metrics: dict[str, float]) -> None:
    if not metrics or any(
        not math.isfinite(value) or not 0 <= value <= 1 for value in metrics.values()
    ):
        raise ValueError("metric values must be finite numbers within [0, 1]")


class BM25BaselineConfig(BM25ReportModel):
    """Explicit filesystem and BM25 parameters for one offline run."""

    corpus_path: Path
    queries_path: Path
    dense_report_path: Path
    output_path: Path
    top_k: int = Field(default=5, gt=0)
    k1: float = Field(default=1.5, gt=0, allow_inf_nan=False)
    b: float = Field(default=0.75, ge=0, le=1, allow_inf_nan=False)


class BM25RankedResult(BM25ReportModel):
    """One explicit sparse rank with both storage and relevance identities."""

    rank: int = Field(gt=0)
    record_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    score: float = Field(gt=0, allow_inf_nan=False)
    is_relevant: bool


class BM25QueryResult(BM25ReportModel):
    """One query's actual BM25 ranking and reusable binary metrics."""

    query_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    category: str = Field(min_length=1)
    relevant_document_ids: tuple[str, ...] = Field(min_length=1)
    ranked_results: tuple[BM25RankedResult, ...]
    metrics: dict[str, float]

    @model_validator(mode="after")
    def validate_ranking(self) -> BM25QueryResult:
        """Keep ranks, identities, relevance markers, and metrics consistent."""
        _validate_metrics(self.metrics)
        if [item.rank for item in self.ranked_results] != list(
            range(1, len(self.ranked_results) + 1)
        ):
            raise ValueError("ranked result ranks must be consecutive and start at one")
        document_ids = [item.document_id for item in self.ranked_results]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("ranked result document IDs must be unique")
        relevant = set(self.relevant_document_ids)
        if any(item.is_relevant != (item.document_id in relevant) for item in self.ranked_results):
            raise ValueError("ranked result relevance markers must match labels")
        return self

    @property
    def ranked_document_ids(self) -> tuple[str, ...]:
        """Return document-level identities in rank order."""
        return tuple(item.document_id for item in self.ranked_results)


class BM25IndexObservations(BM25ReportModel):
    """Deterministic corpus statistics recorded alongside timings."""

    document_count: int = Field(gt=0)
    average_document_length: float = Field(gt=0, allow_inf_nan=False)
    vocabulary_size: int = Field(gt=0)


class BM25RuntimeObservations(BM25ReportModel):
    """Local process timings that are not production benchmarks."""

    index_build_seconds: float = Field(ge=0, allow_inf_nan=False)
    total_query_seconds: float = Field(ge=0, allow_inf_nan=False)
    average_query_ms: float = Field(ge=0, allow_inf_nan=False)


class DenseBM25Comparison(BM25ReportModel):
    """Top-1 outcome partition comparing fixed Dense and BM25 runs."""

    dense_baseline_name: str = Field(min_length=1)
    dense_metrics: dict[str, float]
    bm25_metrics: dict[str, float]
    dense_better_query_ids: tuple[str, ...]
    bm25_better_query_ids: tuple[str, ...]
    both_top1_correct_query_ids: tuple[str, ...]
    both_top1_failed_query_ids: tuple[str, ...]
    product_a_battery_warranty_fixed: bool

    @model_validator(mode="after")
    def validate_comparison_metrics(self) -> DenseBM25Comparison:
        _validate_metrics(self.dense_metrics)
        _validate_metrics(self.bm25_metrics)
        groups = (
            self.dense_better_query_ids,
            self.bm25_better_query_ids,
            self.both_top1_correct_query_ids,
            self.both_top1_failed_query_ids,
        )
        flattened = [query_id for group in groups for query_id in group]
        if len(flattened) != len(set(flattened)):
            raise ValueError("Dense/BM25 comparison groups must be disjoint")
        return self


class BM25BaselineReport(BM25ReportModel):
    """Versioned sparse baseline and objective Dense comparison."""

    evaluation_schema_version: str = "1.0"
    baseline_name: str
    retrieval_type: str = "bm25"
    dataset_name: str
    corpus_file: str
    queries_file: str
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    queries_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tokenizer_name: str
    tokenizer_version: str
    tokenizer_config: dict[str, str | list[int]]
    k1: float = Field(gt=0, allow_inf_nan=False)
    b: float = Field(ge=0, le=1, allow_inf_nan=False)
    top_k: int = Field(gt=0)
    corpus_size: int = Field(gt=0)
    query_count: int = Field(gt=0)
    index_observations: BM25IndexObservations
    metrics: dict[str, float]
    category_metrics: dict[str, dict[str, float]]
    query_results: tuple[BM25QueryResult, ...]
    dense_comparison: DenseBM25Comparison
    runtime_observations: BM25RuntimeObservations
    python_version: str
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def validate_report(self) -> BM25BaselineReport:
        """Validate counts, metrics, Top-K, and runtime arithmetic."""
        _validate_metrics(self.metrics)
        for category_metrics in self.category_metrics.values():
            _validate_metrics(category_metrics)
        if self.corpus_size != self.index_observations.document_count:
            raise ValueError("corpus_size must match index document_count")
        if self.query_count != len(self.query_results):
            raise ValueError("query_count must match query_results length")
        if any(len(result.ranked_results) > self.top_k for result in self.query_results):
            raise ValueError("query result length cannot exceed top_k")
        expected_average = self.runtime_observations.total_query_seconds * 1000 / self.query_count
        if not math.isclose(
            self.runtime_observations.average_query_ms,
            expected_average,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("average_query_ms must match total time and query_count")
        compared_ids = {
            query_id
            for group in (
                self.dense_comparison.dense_better_query_ids,
                self.dense_comparison.bm25_better_query_ids,
                self.dense_comparison.both_top1_correct_query_ids,
                self.dense_comparison.both_top1_failed_query_ids,
            )
            for query_id in group
        }
        if compared_ids != {result.query_id for result in self.query_results}:
            raise ValueError("Dense/BM25 comparison must cover every query exactly once")
        return self


def _load_dense_report(path: Path) -> DenseBaselineReport:
    if not path.exists():
        raise EvaluationValidationError("dense report file does not exist")
    if not path.is_file():
        raise EvaluationValidationError("dense report path must be a file")
    try:
        return DenseBaselineReport.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValidationError, json.JSONDecodeError) as exc:
        raise EvaluationValidationError("failed to read validated Dense baseline report") from exc


def _build_comparison(
    dense_report: DenseBaselineReport,
    query_results: tuple[BM25QueryResult, ...],
    bm25_metrics: dict[str, float],
) -> DenseBM25Comparison:
    dense_by_id = {result.query_id: result for result in dense_report.query_results}
    bm25_ids = {result.query_id for result in query_results}
    if set(dense_by_id) != bm25_ids:
        raise EvaluationValidationError("Dense and BM25 reports must contain identical query IDs")

    dense_better: list[str] = []
    bm25_better: list[str] = []
    both_correct: list[str] = []
    both_failed: list[str] = []
    for result in query_results:
        dense_hit = dense_by_id[result.query_id].metrics["hit_rate_at_1"] == 1.0
        bm25_hit = result.metrics["hit_rate_at_1"] == 1.0
        if dense_hit and bm25_hit:
            both_correct.append(result.query_id)
        elif dense_hit:
            dense_better.append(result.query_id)
        elif bm25_hit:
            bm25_better.append(result.query_id)
        else:
            both_failed.append(result.query_id)

    target_query = "query-product-a-battery-warranty"
    return DenseBM25Comparison(
        dense_baseline_name=dense_report.baseline_name,
        dense_metrics=dict(dense_report.metrics),
        bm25_metrics=dict(bm25_metrics),
        dense_better_query_ids=tuple(dense_better),
        bm25_better_query_ids=tuple(bm25_better),
        both_top1_correct_query_ids=tuple(both_correct),
        both_top1_failed_query_ids=tuple(both_failed),
        product_a_battery_warranty_fixed=(
            dense_by_id[target_query].metrics["hit_rate_at_1"] == 0.0
            and next(result for result in query_results if result.query_id == target_query).metrics[
                "hit_rate_at_1"
            ]
            == 1.0
        ),
    )


def run_bm25_baseline(config: BM25BaselineConfig) -> BM25BaselineReport:
    """Build, evaluate, compare, and atomically persist the BM25 baseline."""
    corpus_sha256 = compute_normalized_text_sha256(config.corpus_path, dataset_name="corpus")
    queries_sha256 = compute_normalized_text_sha256(config.queries_path, dataset_name="queries")
    dataset = load_retrieval_dataset(config.corpus_path, config.queries_path)
    dense_report = _load_dense_report(config.dense_report_path)
    if dense_report.corpus_sha256 != corpus_sha256 or dense_report.queries_sha256 != queries_sha256:
        raise EvaluationValidationError("Dense report and BM25 inputs must use identical datasets")

    tokenizer = DeterministicChineseTokenizer()
    documents = [
        BM25Document(
            record_id=f"m2b3-bm25-{document.document_id}",
            document_id=document.document_id,
            content=document.text,
            category=document.category,
            source=document.source,
            metadata={"evaluation_dataset": "m2b2c-dense-v1"},
        )
        for document in dataset.corpus
    ]
    index_started = time.perf_counter()
    index = BM25Index(documents, tokenizer=tokenizer, k1=config.k1, b=config.b)
    index_build_seconds = time.perf_counter() - index_started
    retriever = BM25Retriever(index)

    query_results: list[BM25QueryResult] = []
    query_started = time.perf_counter()
    for query in dataset.queries:
        hits = retriever.retrieve(query.query, top_k=config.top_k)
        ranked_document_ids = tuple(hit.document_id for hit in hits)
        metrics = compute_query_metrics(
            ranked_document_ids,
            query.relevant_document_ids,
            ks=(1, 3, 5),
            mrr_k=5,
        )
        query_results.append(
            BM25QueryResult(
                query_id=query.query_id,
                query=query.query,
                category=query.category,
                relevant_document_ids=query.relevant_document_ids,
                ranked_results=tuple(
                    BM25RankedResult(
                        rank=hit.rank,
                        record_id=hit.record_id,
                        document_id=hit.document_id,
                        score=hit.score,
                        is_relevant=hit.document_id in query.relevant_document_ids,
                    )
                    for hit in hits
                ),
                metrics=metrics,
            )
        )
    total_query_seconds = time.perf_counter() - query_started
    query_results_tuple = tuple(query_results)
    overall, categories = aggregate_metrics(
        [(result.category, result.metrics) for result in query_results_tuple]
    )

    if (
        compute_normalized_text_sha256(config.corpus_path, dataset_name="corpus") != corpus_sha256
        or compute_normalized_text_sha256(config.queries_path, dataset_name="queries")
        != queries_sha256
    ):
        raise EvaluationValidationError("evaluation dataset changed during BM25 baseline execution")

    report = BM25BaselineReport(
        evaluation_schema_version="1.0",
        baseline_name="m2b3-bm25-deterministic-chinese-v1",
        retrieval_type="bm25",
        dataset_name="m2b2c-dense-v1",
        corpus_file=config.corpus_path.name,
        queries_file=config.queries_path.name,
        corpus_sha256=corpus_sha256,
        queries_sha256=queries_sha256,
        tokenizer_name=tokenizer.name,
        tokenizer_version=tokenizer.version,
        tokenizer_config={
            "normalization": "NFKC",
            "english_case": "lower",
            "ascii_identifiers": "alphanumeric with internal -_./ separators",
            "chinese_ngrams": list(tokenizer.chinese_ngram_sizes),
        },
        k1=index.k1,
        b=index.b,
        top_k=config.top_k,
        corpus_size=len(dataset.corpus),
        query_count=len(dataset.queries),
        index_observations=BM25IndexObservations(
            document_count=index.document_count,
            average_document_length=index.average_document_length,
            vocabulary_size=index.vocabulary_size,
        ),
        metrics=overall,
        category_metrics=categories,
        query_results=query_results_tuple,
        dense_comparison=_build_comparison(dense_report, query_results_tuple, overall),
        runtime_observations=BM25RuntimeObservations(
            index_build_seconds=index_build_seconds,
            total_query_seconds=total_query_seconds,
            average_query_ms=total_query_seconds * 1000 / len(dataset.queries),
        ),
        python_version=platform.python_version(),
        limitations=(
            "Metrics describe only the versioned small synthetic Chinese dataset.",
            "Character n-grams do not provide semantic understanding or word segmentation.",
            "Timings are one in-process development run, not a production benchmark.",
            "This baseline excludes Dense fusion, reranking, and parent expansion.",
        ),
    )
    payload: dict[str, Any] = report.model_dump(mode="json")
    write_json_report_atomically(config.output_path, payload)
    return report
