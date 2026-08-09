"""Offline RRF baseline over versioned Dense and BM25 reports."""

from __future__ import annotations

import json
import math
import platform
import re
import time
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from decision_agent.evaluation.bm25_baseline import BM25BaselineReport, BM25QueryResult
from decision_agent.evaluation.dataset import compute_normalized_text_sha256
from decision_agent.evaluation.dense_baseline import DenseBaselineReport, DenseQueryResult
from decision_agent.evaluation.reporting import write_json_report_atomically
from decision_agent.evaluation.retrieval_metrics import aggregate_metrics, compute_query_metrics
from decision_agent.exceptions import EvaluationValidationError
from decision_agent.retrieval.fusion import (
    FusionCandidate,
    FusionContribution,
    reciprocal_rank_fusion,
)

DENSE_SOURCE_NAME = "dense"
BM25_SOURCE_NAME = "bm25"
EXPECTED_DENSE_BASELINE = "m2b2c-dense-bge-small-zh-v1.5"
EXPECTED_BM25_BASELINE = "m2b3-bm25-deterministic-chinese-v1"


class RRFReportModel(BaseModel):
    """Strict JSON-safe base for RRF evaluation artifacts."""

    model_config = ConfigDict(extra="forbid")


def _validate_metrics(metrics: dict[str, float]) -> None:
    if not metrics or any(
        not math.isfinite(value) or not 0 <= value <= 1 for value in metrics.values()
    ):
        raise ValueError("metric values must be finite numbers within [0, 1]")


class RRFBaselineConfig(RRFReportModel):
    """Fixed report inputs and untuned M2B-4 fusion parameters."""

    dense_report_path: Path
    bm25_report_path: Path
    output_path: Path
    rrf_k: Literal[60.0] = 60.0
    dense_weight: Literal[1.0] = 1.0
    bm25_weight: Literal[1.0] = 1.0
    fusion_input_depth: int = Field(default=5, gt=0)
    output_top_k: int = Field(default=5, gt=0)


class RRFSourceReport(RRFReportModel):
    """Audited identity of one immutable source report."""

    source_name: str = Field(min_length=1)
    relative_path: str = Field(min_length=1)
    baseline_name: str = Field(min_length=1)
    normalized_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_depth: int = Field(gt=0)
    weight: float = Field(gt=0, allow_inf_nan=False)

    @field_validator("relative_path")
    @classmethod
    def require_safe_relative_path(cls, value: str) -> str:
        """Reject absolute, parent-traversing, and platform-specific report paths."""
        path = PurePosixPath(value)
        if (
            Path(value).is_absolute()
            or path.is_absolute()
            or re.match(r"^[A-Za-z]:[\\/]", value)
            or ".." in path.parts
            or "\\" in value
        ):
            raise ValueError("source report path must be a safe POSIX relative path")
        return value


class RRFRankedResult(RRFReportModel):
    """One fused rank with auditable per-source contributions."""

    final_rank: int = Field(gt=0)
    document_id: str = Field(min_length=1)
    fused_score: float = Field(gt=0, allow_inf_nan=False)
    best_source_rank: int = Field(gt=0)
    matched_source_count: int = Field(gt=0)
    source_contributions: tuple[FusionContribution, ...] = Field(min_length=1)
    is_relevant: bool

    @model_validator(mode="after")
    def validate_contribution_summary(self) -> RRFRankedResult:
        """Reject internally inconsistent or duplicated source contributions."""
        names = [item.source_name for item in self.source_contributions]
        if len(names) != len(set(names)):
            raise ValueError("fused result source contributions must be unique")
        if self.matched_source_count != len(self.source_contributions):
            raise ValueError("matched_source_count must match source contributions")
        if self.best_source_rank != min(item.source_rank for item in self.source_contributions):
            raise ValueError("best_source_rank must match source contributions")
        total = math.fsum(item.contribution for item in self.source_contributions)
        if not math.isclose(self.fused_score, total, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("fused_score must equal the sum of source contributions")
        return self


class RRFQueryResult(RRFReportModel):
    """One query's fused ranking, metrics, and rank change versus Dense."""

    query_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    category: str = Field(min_length=1)
    relevant_document_ids: tuple[str, ...] = Field(min_length=1)
    ranked_results: tuple[RRFRankedResult, ...]
    metrics: dict[str, float]
    dense_best_relevant_rank: int | None = Field(default=None, gt=0)
    bm25_best_relevant_rank: int | None = Field(default=None, gt=0)
    rrf_best_relevant_rank: int | None = Field(default=None, gt=0)
    comparison_to_dense: Literal["improved", "unchanged", "regressed"]

    @model_validator(mode="after")
    def validate_query_result(self) -> RRFQueryResult:
        """Keep fused ranks, relevance flags, and metrics consistent."""
        _validate_metrics(self.metrics)
        if [item.final_rank for item in self.ranked_results] != list(
            range(1, len(self.ranked_results) + 1)
        ):
            raise ValueError("fused ranks must be consecutive and start at one")
        ids = [item.document_id for item in self.ranked_results]
        if len(ids) != len(set(ids)):
            raise ValueError("fused document IDs must be unique")
        relevant = set(self.relevant_document_ids)
        if any(item.is_relevant != (item.document_id in relevant) for item in self.ranked_results):
            raise ValueError("fused relevance markers must match labels")
        return self

    @property
    def ranked_document_ids(self) -> tuple[str, ...]:
        """Return fused document IDs in rank order."""
        return tuple(item.document_id for item in self.ranked_results)


class RRFComparison(RRFReportModel):
    """Dense/BM25/RRF metrics and query-level rank-change partitions."""

    dense_metrics: dict[str, float]
    bm25_metrics: dict[str, float]
    rrf_metrics: dict[str, float]
    improved_query_ids: tuple[str, ...]
    regressed_query_ids: tuple[str, ...]
    unchanged_query_ids: tuple[str, ...]

    @model_validator(mode="after")
    def validate_comparison(self) -> RRFComparison:
        for values in (self.dense_metrics, self.bm25_metrics, self.rrf_metrics):
            _validate_metrics(values)
        flattened = [
            query_id
            for group in (
                self.improved_query_ids,
                self.regressed_query_ids,
                self.unchanged_query_ids,
            )
            for query_id in group
        ]
        if len(flattened) != len(set(flattened)):
            raise ValueError("RRF comparison query groups must be disjoint")
        return self


class RRFRuntimeObservations(RRFReportModel):
    """Local fusion timings that are not production benchmarks."""

    total_fusion_seconds: float = Field(ge=0, allow_inf_nan=False)
    average_query_ms: float = Field(ge=0, allow_inf_nan=False)


class RRFBaselineReport(RRFReportModel):
    """Versioned Dense/BM25 Reciprocal Rank Fusion baseline."""

    evaluation_schema_version: str = "1.0"
    baseline_name: str = "m2b4-rrf-dense-bm25-k60"
    retrieval_type: Literal["rrf"] = "rrf"
    algorithm: Literal["reciprocal_rank_fusion"] = "reciprocal_rank_fusion"
    rrf_k: float = Field(gt=0, allow_inf_nan=False)
    source_names: tuple[str, ...] = Field(min_length=1)
    source_weights: dict[str, float]
    source_reports: tuple[RRFSourceReport, ...] = Field(min_length=1)
    corpus_file: str = Field(min_length=1)
    queries_file: str = Field(min_length=1)
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    queries_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_size: int = Field(gt=0)
    query_count: int = Field(gt=0)
    fusion_input_depth: int = Field(gt=0)
    output_top_k: int = Field(gt=0)
    metrics: dict[str, float]
    category_metrics: dict[str, dict[str, float]]
    query_results: tuple[RRFQueryResult, ...]
    comparison: RRFComparison
    runtime_observations: RRFRuntimeObservations
    python_version: str
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def validate_report(self) -> RRFBaselineReport:
        """Validate summary counts, metrics, partitions, and timing arithmetic."""
        _validate_metrics(self.metrics)
        for values in self.category_metrics.values():
            _validate_metrics(values)
        if self.query_count != len(self.query_results):
            raise ValueError("query_count must match query_results length")
        query_ids = [result.query_id for result in self.query_results]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("RRF report query IDs must be unique")
        if len(self.source_names) != len(set(self.source_names)):
            raise ValueError("RRF source names must be unique")
        if set(self.source_weights) != set(self.source_names):
            raise ValueError("RRF source weights must match source names")
        source_reports = {item.source_name: item for item in self.source_reports}
        if len(source_reports) != len(self.source_reports) or set(source_reports) != set(
            self.source_names
        ):
            raise ValueError("RRF source reports must uniquely match source names")
        for name, weight in self.source_weights.items():
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError("RRF source weights must be finite and positive")
            if not math.isclose(source_reports[name].weight, weight, rel_tol=0, abs_tol=0):
                raise ValueError("RRF source report weight must match source_weights")
        for result in self.query_results:
            for ranked in result.ranked_results:
                for contribution in ranked.source_contributions:
                    if contribution.source_name not in self.source_weights:
                        raise ValueError("fused contribution uses an unknown source")
                    expected = self.source_weights[contribution.source_name] / (
                        self.rrf_k + contribution.source_rank
                    )
                    if not math.isclose(
                        contribution.contribution,
                        expected,
                        rel_tol=1e-12,
                        abs_tol=1e-15,
                    ):
                        raise ValueError("fused contribution does not match the RRF formula")
        if self.comparison.rrf_metrics != self.metrics:
            raise ValueError("comparison RRF metrics must match report metrics")
        if any(len(result.ranked_results) > self.output_top_k for result in self.query_results):
            raise ValueError("fused result length cannot exceed output_top_k")
        expected_average = self.runtime_observations.total_fusion_seconds * 1000 / self.query_count
        if not math.isclose(
            self.runtime_observations.average_query_ms,
            expected_average,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("average_query_ms must match total time and query_count")
        compared = {
            query_id
            for group in (
                self.comparison.improved_query_ids,
                self.comparison.regressed_query_ids,
                self.comparison.unchanged_query_ids,
            )
            for query_id in group
        }
        if compared != {result.query_id for result in self.query_results}:
            raise ValueError("RRF comparison must cover every query exactly once")
        return self


def _load_report(path: Path, model_type: type[DenseBaselineReport] | type[BM25BaselineReport]):
    if not path.exists():
        raise EvaluationValidationError(f"source report does not exist: {path.name}")
    if not path.is_file():
        raise EvaluationValidationError(f"source report path must be a file: {path.name}")
    try:
        return model_type.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValidationError, json.JSONDecodeError) as exc:
        raise EvaluationValidationError(
            f"failed to read validated source report: {path.name}"
        ) from exc


def _relative_report_path(path: Path) -> str:
    parts = list(path.parts)
    try:
        index = [part.lower() for part in parts].index("artifacts")
    except ValueError:
        return path.name
    return Path(*parts[index:]).as_posix()


def _validate_source_compatibility(
    dense: DenseBaselineReport,
    bm25: BM25BaselineReport,
    *,
    input_depth: int,
) -> None:
    if dense.baseline_name != EXPECTED_DENSE_BASELINE:
        raise EvaluationValidationError("unexpected Dense baseline_name")
    if bm25.baseline_name != EXPECTED_BM25_BASELINE or bm25.retrieval_type != "bm25":
        raise EvaluationValidationError("unexpected BM25 baseline identity")
    common_pairs = (
        (dense.corpus_file, bm25.corpus_file, "corpus filename"),
        (dense.queries_file, bm25.queries_file, "queries filename"),
        (dense.corpus_sha256, bm25.corpus_sha256, "corpus SHA-256"),
        (dense.queries_sha256, bm25.queries_sha256, "queries SHA-256"),
        (dense.corpus_size, bm25.corpus_size, "corpus size"),
        (dense.query_count, bm25.query_count, "query count"),
    )
    for dense_value, bm25_value, label in common_pairs:
        if dense_value != bm25_value:
            raise EvaluationValidationError(f"source reports have different {label}")
    if dense.corpus_size != 36 or dense.query_count != 18:
        raise EvaluationValidationError("M2B-4 requires the fixed 36-document/18-query dataset")
    if dense.top_k < input_depth or bm25.top_k < input_depth:
        raise EvaluationValidationError("source report depth is smaller than fusion_input_depth")

    dense_query_ids = [result.query_id for result in dense.query_results]
    bm25_query_ids = [result.query_id for result in bm25.query_results]
    if len(dense_query_ids) != len(set(dense_query_ids)):
        raise EvaluationValidationError("Dense source report contains duplicate query IDs")
    if len(bm25_query_ids) != len(set(bm25_query_ids)):
        raise EvaluationValidationError("BM25 source report contains duplicate query IDs")
    dense_by_id = {result.query_id: result for result in dense.query_results}
    bm25_by_id = {result.query_id: result for result in bm25.query_results}
    if set(dense_by_id) != set(bm25_by_id):
        raise EvaluationValidationError("source report query ID sets differ")
    for query_id, dense_query in dense_by_id.items():
        bm25_query = bm25_by_id[query_id]
        pairs = (
            (dense_query.query, bm25_query.query, "query text"),
            (dense_query.category, bm25_query.category, "query category"),
            (
                dense_query.relevant_document_ids,
                bm25_query.relevant_document_ids,
                "relevance labels",
            ),
        )
        for dense_value, bm25_value, label in pairs:
            if dense_value != bm25_value:
                raise EvaluationValidationError(f"source reports differ in {label}: {query_id}")
        if (
            len(dense_query.ranked_results) < input_depth
            or len(bm25_query.ranked_results) < input_depth
        ):
            raise EvaluationValidationError(f"source query depth is too small: {query_id}")


def _best_relevant_rank(
    result: DenseQueryResult | BM25QueryResult | tuple[RRFRankedResult, ...],
    relevant_document_ids: tuple[str, ...],
) -> int | None:
    ranked = result if isinstance(result, tuple) else result.ranked_results
    relevant = set(relevant_document_ids)
    for item in ranked:
        rank = item.final_rank if isinstance(item, RRFRankedResult) else item.rank
        if item.document_id in relevant:
            return rank
    return None


def _compare_rank(rrf_rank: int | None, dense_rank: int | None) -> str:
    normalized_rrf = math.inf if rrf_rank is None else rrf_rank
    normalized_dense = math.inf if dense_rank is None else dense_rank
    if normalized_rrf < normalized_dense:
        return "improved"
    if normalized_rrf > normalized_dense:
        return "regressed"
    return "unchanged"


def run_rrf_baseline(config: RRFBaselineConfig) -> RRFBaselineReport:
    """Validate fixed reports, fuse their rankings, and atomically persist a report."""
    dense_hash = compute_normalized_text_sha256(
        config.dense_report_path, dataset_name="Dense source report"
    )
    bm25_hash = compute_normalized_text_sha256(
        config.bm25_report_path, dataset_name="BM25 source report"
    )
    dense = _load_report(config.dense_report_path, DenseBaselineReport)
    bm25 = _load_report(config.bm25_report_path, BM25BaselineReport)
    _validate_source_compatibility(dense, bm25, input_depth=config.fusion_input_depth)

    bm25_by_id = {result.query_id: result for result in bm25.query_results}
    query_results: list[RRFQueryResult] = []
    fusion_started = time.perf_counter()
    for dense_query in dense.query_results:
        bm25_query = bm25_by_id[dense_query.query_id]
        dense_candidates = [
            FusionCandidate(
                source_name=DENSE_SOURCE_NAME,
                rank=item.rank,
                document_id=item.document_id,
                source_score=item.score,
                provenance={"baseline": dense.baseline_name},
            )
            for item in dense_query.ranked_results[: config.fusion_input_depth]
        ]
        bm25_candidates = [
            FusionCandidate(
                source_name=BM25_SOURCE_NAME,
                rank=item.rank,
                document_id=item.document_id,
                record_id=item.record_id,
                source_score=item.score,
                provenance={"baseline": bm25.baseline_name},
            )
            for item in bm25_query.ranked_results[: config.fusion_input_depth]
        ]
        fused = reciprocal_rank_fusion(
            {DENSE_SOURCE_NAME: dense_candidates, BM25_SOURCE_NAME: bm25_candidates},
            rrf_k=config.rrf_k,
            source_weights={
                DENSE_SOURCE_NAME: config.dense_weight,
                BM25_SOURCE_NAME: config.bm25_weight,
            },
            top_k=config.output_top_k,
        )
        ranked = tuple(
            RRFRankedResult(
                final_rank=item.final_rank,
                document_id=item.document_id,
                fused_score=item.fused_score,
                best_source_rank=item.best_source_rank,
                matched_source_count=item.matched_source_count,
                source_contributions=item.source_contributions,
                is_relevant=item.document_id in dense_query.relevant_document_ids,
            )
            for item in fused
        )
        metrics = compute_query_metrics(
            tuple(item.document_id for item in ranked),
            dense_query.relevant_document_ids,
            ks=(1, 3, 5),
            mrr_k=5,
        )
        dense_rank = _best_relevant_rank(dense_query, dense_query.relevant_document_ids)
        bm25_rank = _best_relevant_rank(bm25_query, dense_query.relevant_document_ids)
        rrf_rank = _best_relevant_rank(ranked, dense_query.relevant_document_ids)
        query_results.append(
            RRFQueryResult(
                query_id=dense_query.query_id,
                query=dense_query.query,
                category=dense_query.category,
                relevant_document_ids=dense_query.relevant_document_ids,
                ranked_results=ranked,
                metrics=metrics,
                dense_best_relevant_rank=dense_rank,
                bm25_best_relevant_rank=bm25_rank,
                rrf_best_relevant_rank=rrf_rank,
                comparison_to_dense=_compare_rank(rrf_rank, dense_rank),
            )
        )
    total_fusion_seconds = time.perf_counter() - fusion_started
    query_results_tuple = tuple(query_results)
    overall, categories = aggregate_metrics(
        [(result.category, result.metrics) for result in query_results_tuple]
    )
    improved = tuple(
        result.query_id
        for result in query_results_tuple
        if result.comparison_to_dense == "improved"
    )
    regressed = tuple(
        result.query_id
        for result in query_results_tuple
        if result.comparison_to_dense == "regressed"
    )
    unchanged = tuple(
        result.query_id
        for result in query_results_tuple
        if result.comparison_to_dense == "unchanged"
    )

    if (
        compute_normalized_text_sha256(config.dense_report_path, dataset_name="Dense source report")
        != dense_hash
        or compute_normalized_text_sha256(
            config.bm25_report_path, dataset_name="BM25 source report"
        )
        != bm25_hash
    ):
        raise EvaluationValidationError("source report changed during RRF baseline execution")

    report = RRFBaselineReport(
        rrf_k=config.rrf_k,
        source_names=(DENSE_SOURCE_NAME, BM25_SOURCE_NAME),
        source_weights={
            DENSE_SOURCE_NAME: config.dense_weight,
            BM25_SOURCE_NAME: config.bm25_weight,
        },
        source_reports=(
            RRFSourceReport(
                source_name=DENSE_SOURCE_NAME,
                relative_path=_relative_report_path(config.dense_report_path),
                baseline_name=dense.baseline_name,
                normalized_text_sha256=dense_hash,
                input_depth=config.fusion_input_depth,
                weight=config.dense_weight,
            ),
            RRFSourceReport(
                source_name=BM25_SOURCE_NAME,
                relative_path=_relative_report_path(config.bm25_report_path),
                baseline_name=bm25.baseline_name,
                normalized_text_sha256=bm25_hash,
                input_depth=config.fusion_input_depth,
                weight=config.bm25_weight,
            ),
        ),
        corpus_file=dense.corpus_file,
        queries_file=dense.queries_file,
        corpus_sha256=dense.corpus_sha256,
        queries_sha256=dense.queries_sha256,
        corpus_size=dense.corpus_size,
        query_count=dense.query_count,
        fusion_input_depth=config.fusion_input_depth,
        output_top_k=config.output_top_k,
        metrics=overall,
        category_metrics=categories,
        query_results=query_results_tuple,
        comparison=RRFComparison(
            dense_metrics=dict(dense.metrics),
            bm25_metrics=dict(bm25.metrics),
            rrf_metrics=dict(overall),
            improved_query_ids=improved,
            regressed_query_ids=regressed,
            unchanged_query_ids=unchanged,
        ),
        runtime_observations=RRFRuntimeObservations(
            total_fusion_seconds=total_fusion_seconds,
            average_query_ms=total_fusion_seconds * 1000 / dense.query_count,
        ),
        python_version=platform.python_version(),
        limitations=(
            "Metrics describe only the fixed 36-document/18-query synthetic dataset.",
            "RRF uses rank positions only and cannot resolve semantic hard negatives by itself.",
            "Input depth is limited to the versioned Top-5 Dense and BM25 reports.",
            "Timings are one in-process report-fusion run, not a production benchmark.",
            "Cross-encoder reranking and parent expansion are not included.",
        ),
    )
    write_json_report_atomically(config.output_path, report.model_dump(mode="json"))
    return report
