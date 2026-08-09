"""Deterministic scoring and reporting for the enterprise retrieval pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from decision_agent.evaluation.enterprise_kb_dataset import (
    EXPECTED_ANSWERABLE_QUERY_COUNT,
    EXPECTED_QUERY_COUNT,
    EXPECTED_UNANSWERABLE_QUERY_COUNT,
)
from decision_agent.evaluation.enterprise_kb_ground_truth import RetrievalGroundTruthRecord
from decision_agent.evaluation.reporting import write_text_files_atomically
from decision_agent.evaluation.retrieval_metrics import compute_query_metrics
from decision_agent.exceptions import EvaluationValidationError
from decision_agent.retrieval.pipeline import RetrievalPipelineResult

SCHEMA_VERSION = "1.0"
BASELINE_ID = "m2c2a2-enterprise-real-retrieval-baseline-v1"
STAGE_NAMES = ("dense", "bm25", "rrf", "reranker", "parent", "evidence")
CHILD_STAGES = ("dense", "bm25", "rrf", "reranker")
FAILURE_TYPE_UNIVERSE = tuple(
    sorted(
        {
            *(f"{stage}_top1_miss" for stage in CHILD_STAGES),
            *(f"{stage}_no_relevant_at_5" for stage in CHILD_STAGES),
            *(f"{stage}_hard_negative_before_relevant" for stage in STAGE_NAMES),
            *(f"{stage}_overlap_before_relevant" for stage in STAGE_NAMES),
            "parent_no_relevant_at_5",
            "reranker_regressed_vs_rrf",
            "rrf_regressed_vs_dense",
            "rrf_regressed_vs_bm25",
            "unanswerable_returns_evidence",
        }
    )
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BenchmarkQueryInput(_StrictModel):
    """Query-only view that deliberately excludes answers and relevance labels."""

    query_id: str = Field(min_length=1)
    query: str = Field(min_length=1)


class MetricAggregate(_StrictModel):
    """Macro metric with an auditable sum-of-query-values numerator."""

    numerator: float = Field(ge=0, allow_inf_nan=False)
    denominator: int = Field(gt=0)
    value: float = Field(ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_ratio(self) -> MetricAggregate:
        if not math.isclose(self.value, self.numerator / self.denominator, abs_tol=1e-12):
            raise ValueError("metric value must equal numerator divided by denominator")
        return self


@dataclass(frozen=True, slots=True)
class EvaluatedRun:
    query_results: tuple[dict[str, Any], ...]
    failure_cases: tuple[dict[str, Any], ...]
    analysis: dict[str, Any]
    ranking_digest: str
    deterministic_digest: str


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise EvaluationValidationError(f"failed to read {path.name}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise EvaluationValidationError(f"{path.name} contains a blank line")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationValidationError(
                f"{path.name} line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(row, dict):
            raise EvaluationValidationError(f"{path.name} rows must be JSON objects")
        rows.append(row)
    return rows


def load_benchmark_query_inputs(path: Path) -> tuple[BenchmarkQueryInput, ...]:
    """Load only fields allowed to enter runtime retrieval, preserving file order."""
    queries = tuple(
        BenchmarkQueryInput.model_validate({key: row[key] for key in ("query_id", "query")})
        for row in _read_jsonl(path)
    )
    if len(queries) != EXPECTED_QUERY_COUNT:
        raise EvaluationValidationError(
            f"benchmark must contain exactly {EXPECTED_QUERY_COUNT} runtime queries"
        )
    if len({item.query_id for item in queries}) != len(queries):
        raise EvaluationValidationError("benchmark query IDs must be unique")
    return queries


def load_retrieval_ground_truth(path: Path) -> tuple[RetrievalGroundTruthRecord, ...]:
    records = tuple(RetrievalGroundTruthRecord.model_validate(row) for row in _read_jsonl(path))
    if len(records) != EXPECTED_QUERY_COUNT or (
        sum(item.answerable for item in records) != EXPECTED_ANSWERABLE_QUERY_COUNT
    ):
        raise EvaluationValidationError(
            "retrieval ground truth must contain exactly "
            f"{EXPECTED_ANSWERABLE_QUERY_COUNT}/{EXPECTED_UNANSWERABLE_QUERY_COUNT} queries"
        )
    for record in records:
        for level in ("child", "parent"):
            relevant = set(getattr(record, f"relevant_{level}_ids"))
            hard_negative = set(getattr(record, f"hard_negative_{level}_ids"))
            overlap = set(getattr(record, f"overlapping_{level}_ids"))
            if not overlap.issubset(relevant):
                raise EvaluationValidationError(
                    f"overlapping {level} IDs must remain in the relevant set"
                )
            if overlap & hard_negative:
                raise EvaluationValidationError(
                    f"overlapping {level} IDs must be excluded from pure hard negatives"
                )
    return records


def _first_relevant_rank(ranked: Sequence[str], relevant: Sequence[str], k: int) -> int | None:
    relevant_set = set(relevant)
    return next((rank for rank, item in enumerate(ranked[:k], 1) if item in relevant_set), None)


def _label(candidate_id: str, relevant: set[str], hard: set[str], overlap: set[str]) -> str:
    if candidate_id in overlap:
        return "overlapping"
    if candidate_id in relevant:
        return "relevant"
    if candidate_id in hard:
        return "hard_negative"
    return "unlabeled"


def _source(payload: Mapping[str, Any]) -> str | None:
    value = payload.get("source")
    if isinstance(value, str) and value.strip():
        return value
    for container_name in ("provenance", "metadata"):
        container = payload.get(container_name)
        if isinstance(container, Mapping):
            value = container.get("source")
            if isinstance(value, str) and value.strip():
                return value
    return None


def _parent_id(payload: Mapping[str, Any]) -> str | None:
    value = payload.get("parent_id")
    if isinstance(value, str) and value:
        return value
    for container_name in ("provenance", "metadata"):
        container = payload.get(container_name)
        if isinstance(container, Mapping):
            value = container.get("parent_id")
            if isinstance(value, str) and value:
                return value
    return None


def _child_candidates(
    values: Sequence[Any], *, stage: str, ground_truth: RetrievalGroundTruthRecord
) -> list[dict[str, Any]]:
    relevant = set(ground_truth.relevant_child_ids)
    hard = set(ground_truth.hard_negative_child_ids)
    overlap = set(ground_truth.overlapping_child_ids)
    candidates: list[dict[str, Any]] = []
    for value in values:
        payload = value.model_dump(mode="json")
        candidate_id = payload.get("candidate_id") or payload.get("record_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise EvaluationValidationError(f"{stage} candidate has no child identity")
        score_name = {
            "dense": "score",
            "bm25": "score",
            "rrf": "fused_score",
            "reranker": "reranker_score",
        }[stage]
        candidates.append(
            {
                "rank": payload.get("rank") or payload.get("final_rank"),
                "candidate_id": candidate_id,
                "child_id": candidate_id,
                "parent_id": _parent_id(payload),
                "document_id": payload["document_id"],
                "source": _source(payload),
                "score": payload[score_name],
                "upstream_rank": payload.get("upstream_rank") or payload.get("best_source_rank"),
                "contributions": payload.get("source_contributions", []),
                "relevance_label": _label(candidate_id, relevant, hard, overlap),
            }
        )
    return candidates


def _parent_candidates(
    values: Sequence[Any], ground_truth: RetrievalGroundTruthRecord
) -> list[dict[str, Any]]:
    relevant = set(ground_truth.relevant_parent_ids)
    hard = set(ground_truth.hard_negative_parent_ids)
    overlap = set(ground_truth.overlapping_parent_ids)
    candidates = []
    for value in values:
        payload = value.model_dump(mode="json")
        parent_id = payload["parent_id"]
        best_child = payload["matched_children"][0]
        candidates.append(
            {
                "rank": payload["final_rank"],
                "candidate_id": parent_id,
                "child_id": best_child["child_id"],
                "parent_id": parent_id,
                "document_id": payload["document_id"],
                "source": _source(payload),
                "score": best_child.get("reranker_score"),
                "upstream_rank": payload["best_child_rank"],
                "contributions": [],
                "relevance_label": _label(parent_id, relevant, hard, overlap),
            }
        )
    return candidates


def _evidence_references(result: RetrievalPipelineResult) -> list[dict[str, Any]]:
    return [reference.model_dump(mode="json") for reference in result.evidence_context.references]


def _metric_aggregate(per_query: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    if not per_query:
        raise EvaluationValidationError("metric aggregation requires answerable queries")
    names = tuple(per_query[0])
    result: dict[str, Any] = {}
    for name in names:
        numerator = math.fsum(item[name] for item in per_query)
        result[name] = MetricAggregate(
            numerator=numerator,
            denominator=len(per_query),
            value=numerator / len(per_query),
        ).model_dump(mode="json")
    return result


def _ranked_ids(query_result: Mapping[str, Any], stage: str) -> list[str]:
    if stage == "evidence":
        return [item["parent_id"] for item in query_result["evidence_references"]]
    return [item["candidate_id"] for item in query_result[f"{stage}_candidates"]]


def _stage_metrics(
    query_results: Sequence[dict[str, Any]], ground_truth: Sequence[RetrievalGroundTruthRecord]
) -> dict[str, Any]:
    gt_by_id = {item.query_id: item for item in ground_truth}
    output: dict[str, Any] = {}
    for stage in STAGE_NAMES:
        per_query = []
        for row in query_results:
            gt = gt_by_id[row["query_id"]]
            if not gt.answerable:
                continue
            relevant = gt.relevant_child_ids if stage in CHILD_STAGES else gt.relevant_parent_ids
            ks = (1, 3, 5, 10) if stage in ("dense", "bm25", "rrf") else (1, 3, 5)
            mrr_k = 10 if stage in ("dense", "bm25", "rrf") else 5
            metrics = compute_query_metrics(_ranked_ids(row, stage), relevant, ks=ks, mrr_k=mrr_k)
            if mrr_k == 10:
                metrics["mrr_at_5"] = compute_query_metrics(
                    _ranked_ids(row, stage), relevant, ks=(5,), mrr_k=5
                )["mrr_at_5"]
            per_query.append(metrics)
        output[stage] = {"query_count": len(per_query), "metrics": _metric_aggregate(per_query)}
    return output


def _exposure_statistics(
    query_results: Sequence[dict[str, Any]],
    ground_truth: Sequence[RetrievalGroundTruthRecord],
) -> dict[str, Any]:
    gt_by_id = {item.query_id: item for item in ground_truth}
    output: dict[str, Any] = {}
    for stage in STAGE_NAMES:
        rows = [row for row in query_results if gt_by_id[row["query_id"]].answerable]
        pure_top1: list[str] = []
        pure_top3: list[str] = []
        pure_top5: list[str] = []
        overlap_top1: list[str] = []
        overlap_top5: list[str] = []
        first_relevant_is_overlap: list[str] = []
        relevant_only_via_overlap: list[str] = []
        relevant_with_pure_hit: list[str] = []
        pure_before: list[str] = []
        no_relevant5: list[str] = []
        first_ranks: list[int] = []
        relevant_hit_count = 0
        overlapping_relevant_hit_count = 0
        for row in rows:
            gt = gt_by_id[row["query_id"]]
            ranked = _ranked_ids(row, stage)
            relevant = gt.relevant_child_ids if stage in CHILD_STAGES else gt.relevant_parent_ids
            hard = set(
                gt.hard_negative_child_ids if stage in CHILD_STAGES else gt.hard_negative_parent_ids
            )
            overlap = set(
                gt.overlapping_child_ids if stage in CHILD_STAGES else gt.overlapping_parent_ids
            )
            query_id = row["query_id"]
            if ranked[:1] and ranked[0] in hard:
                pure_top1.append(query_id)
            if hard.intersection(ranked[:3]):
                pure_top3.append(query_id)
            if hard.intersection(ranked[:5]):
                pure_top5.append(query_id)
            if ranked[:1] and ranked[0] in overlap:
                overlap_top1.append(query_id)
            if overlap.intersection(ranked[:5]):
                overlap_top5.append(query_id)
            first = _first_relevant_rank(ranked, relevant, len(ranked))
            if first is not None:
                first_ranks.append(first)
                if ranked[first - 1] in overlap:
                    first_relevant_is_overlap.append(query_id)
            relevant_hits = [
                candidate_id for candidate_id in ranked[:5] if candidate_id in relevant
            ]
            overlapping_hits = [
                candidate_id for candidate_id in relevant_hits if candidate_id in overlap
            ]
            relevant_hit_count += len(relevant_hits)
            overlapping_relevant_hit_count += len(overlapping_hits)
            if relevant_hits and len(relevant_hits) == len(overlapping_hits):
                relevant_only_via_overlap.append(query_id)
            if any(candidate_id not in overlap for candidate_id in relevant_hits):
                relevant_with_pure_hit.append(query_id)
            if _first_relevant_rank(ranked, relevant, 5) is None:
                no_relevant5.append(query_id)
            if (first is None or any(item in hard for item in ranked[: first - 1])) and any(
                item in hard for item in ranked[:5]
            ):
                pure_before.append(query_id)
        denominator = len(rows)
        output[stage] = {
            "answerable_query_count": denominator,
            "pure_hard_negative_top1": _count_rate(pure_top1, denominator),
            "pure_hard_negative_top3_exposure": _count_rate(pure_top3, denominator),
            "pure_hard_negative_top5_exposure": _count_rate(pure_top5, denominator),
            "overlap_top1": _count_rate(overlap_top1, denominator),
            "overlap_top5_exposure": _count_rate(overlap_top5, denominator),
            "first_relevant_is_overlap": _count_rate(first_relevant_is_overlap, denominator),
            "relevant_hit_only_via_overlap_at_5": _count_rate(
                relevant_only_via_overlap, denominator
            ),
            "relevant_hit_with_pure_relevant_at_5": _count_rate(
                relevant_with_pure_hit, denominator
            ),
            "overlap_share_of_relevant_hits_at_5": {
                "numerator": overlapping_relevant_hit_count,
                "denominator": relevant_hit_count,
                "value": (
                    0.0
                    if relevant_hit_count == 0
                    else overlapping_relevant_hit_count / relevant_hit_count
                ),
            },
            "pure_hard_negative_before_first_relevant": _count_rate(pure_before, denominator),
            "overlap_before_first_relevant": {
                **_count_rate((), denominator),
                "structural_zero": True,
                "informative": False,
                "reason": (
                    "Overlap IDs are a subset of relevant IDs, so an Overlap candidate is itself "
                    "a Relevant hit and cannot rank before the first Relevant hit."
                ),
            },
            "average_first_relevant_rank_when_found": (
                None if not first_ranks else math.fsum(first_ranks) / len(first_ranks)
            ),
            "first_relevant_found_count": len(first_ranks),
            "no_relevant_at_5_query_ids": no_relevant5,
        }
    return output


def _count_rate(ids: Sequence[str], denominator: int) -> dict[str, Any]:
    return {
        "count": len(ids),
        "denominator": denominator,
        "rate": len(ids) / denominator,
        "query_ids": list(ids),
    }


def _compare_stage(
    query_results: Sequence[dict[str, Any]],
    ground_truth: Sequence[RetrievalGroundTruthRecord],
    *,
    before: str,
    after: str,
) -> dict[str, Any]:
    gt_by_id = {item.query_id: item for item in ground_truth}
    hit = defaultdict(list)
    rr = defaultdict(list)
    changes: list[float] = []
    for row in query_results:
        gt = gt_by_id[row["query_id"]]
        if not gt.answerable:
            continue
        before_relevant = (
            gt.relevant_child_ids if before in CHILD_STAGES else gt.relevant_parent_ids
        )
        after_relevant = gt.relevant_child_ids if after in CHILD_STAGES else gt.relevant_parent_ids
        before_rank = _first_relevant_rank(_ranked_ids(row, before), before_relevant, 5)
        after_rank = _first_relevant_rank(_ranked_ids(row, after), after_relevant, 5)
        before_hit = before_rank == 1
        after_hit = after_rank == 1
        hit[
            "improved"
            if after_hit and not before_hit
            else "regressed"
            if before_hit and not after_hit
            else "unchanged"
        ].append(row["query_id"])
        before_rr = 0.0 if before_rank is None else 1 / before_rank
        after_rr = 0.0 if after_rank is None else 1 / after_rank
        rr[
            "improved"
            if after_rr > before_rr
            else "regressed"
            if after_rr < before_rr
            else "unchanged"
        ].append(row["query_id"])
        changes.append(float((before_rank or 6) - (after_rank or 6)))
    return {
        "before_stage": before,
        "after_stage": after,
        "before_relevance_granularity": "child" if before in CHILD_STAGES else "parent",
        "after_relevance_granularity": "child" if after in CHILD_STAGES else "parent",
        "comparison_scope": (
            "same_granularity_stage_comparison"
            if (before in CHILD_STAGES) == (after in CHILD_STAGES)
            else "cross_granularity_observation"
        ),
        "strict_same_metric_gain": (before in CHILD_STAGES) == (after in CHILD_STAGES),
        "hit_at_1": {
            name: {"count": len(hit[name]), "query_ids": hit[name]}
            for name in ("improved", "regressed", "unchanged")
        },
        "reciprocal_rank_at_5": {
            name: {"count": len(rr[name]), "query_ids": rr[name]}
            for name in ("improved", "regressed", "unchanged")
        },
        "average_first_relevant_rank_improvement": math.fsum(changes) / len(changes),
    }


def _reranker_movements(
    query_results: Sequence[dict[str, Any]], ground_truth: Sequence[RetrievalGroundTruthRecord]
) -> dict[str, Any]:
    gt_by_id = {item.query_id: item for item in ground_truth}
    movements = {
        name: []
        for name in (
            "relevant_improved",
            "relevant_regressed",
            "hard_negative_lowered",
            "hard_negative_raised",
            "overlap_changed",
        )
    }
    for row in query_results:
        gt = gt_by_id[row["query_id"]]
        if not gt.answerable:
            continue
        before = {item["candidate_id"]: item["rank"] for item in row["rrf_candidates"]}
        after = {item["candidate_id"]: item["rank"] for item in row["reranker_candidates"]}
        before_rel = _first_relevant_rank(list(before), gt.relevant_child_ids, 5)
        after_rel = _first_relevant_rank(list(after), gt.relevant_child_ids, 5)
        if (before_rel or 6) > (after_rel or 6):
            movements["relevant_improved"].append(row["query_id"])
        elif (before_rel or 6) < (after_rel or 6):
            movements["relevant_regressed"].append(row["query_id"])
        for label, ids in (
            ("hard_negative", gt.hard_negative_child_ids),
            ("overlap", gt.overlapping_child_ids),
        ):
            changed = False
            for item in ids:
                old, new = before.get(item), after.get(item)
                if old is None or new is None or old == new:
                    continue
                changed = True
                if label == "hard_negative":
                    movements[
                        "hard_negative_lowered" if new > old else "hard_negative_raised"
                    ].append(row["query_id"])
            if label == "overlap" and changed:
                movements["overlap_changed"].append(row["query_id"])
    return {
        key: {"count": len(set(ids)), "query_ids": sorted(set(ids))}
        for key, ids in movements.items()
    }


def _group_metrics(
    query_results: Sequence[dict[str, Any]],
    ground_truth: Sequence[RetrievalGroundTruthRecord],
    field: Literal["category", "query_type"],
) -> dict[str, Any]:
    gt_by_id = {item.query_id: item for item in ground_truth}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in query_results:
        groups[str(row[field])].append(row)
    output: dict[str, Any] = {}
    for name, rows in sorted(groups.items()):
        answerable = [row for row in rows if gt_by_id[row["query_id"]].answerable]
        stage_values: dict[str, Any] = {}
        for stage in ("rrf", "reranker", "parent"):
            metrics = []
            for row in answerable:
                gt = gt_by_id[row["query_id"]]
                relevant = (
                    gt.relevant_child_ids if stage in CHILD_STAGES else gt.relevant_parent_ids
                )
                metrics.append(
                    compute_query_metrics(_ranked_ids(row, stage), relevant, ks=(1, 3, 5), mrr_k=5)
                )
            stage_values[stage] = None if not metrics else _metric_aggregate(metrics)
        top1_exposure: dict[str, Any] = {}
        for stage in ("rrf", "reranker", "parent"):
            labels = [
                row[f"{stage}_candidates"][0]["relevance_label"]
                for row in answerable
                if row[f"{stage}_candidates"]
            ]
            top1_exposure[stage] = {
                "pure_hard_negative_count": labels.count("hard_negative"),
                "overlap_count": labels.count("overlapping"),
            }
        stage_values.update(
            {
                "query_count": len(rows),
                "answerable_query_count": len(answerable),
                "top1_exposure": top1_exposure,
            }
        )
        output[name] = stage_values
    return output


def _timing_distribution(query_results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    names = tuple(query_results[0]["stage_timings"])
    return {
        name: timing_summary([row["stage_timings"][name] for row in query_results])
        for name in names
    }


def timing_summary(values: Sequence[float]) -> dict[str, float]:
    """Return deterministic descriptive statistics with linearly interpolated p95."""
    if not values or not all(math.isfinite(value) and value >= 0 for value in values):
        raise EvaluationValidationError("timing values must be nonempty finite nonnegative values")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * 0.95
    lower = math.floor(position)
    upper = math.ceil(position)
    p95 = (
        ordered[lower]
        if lower == upper
        else ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    )
    return {
        "total": math.fsum(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": p95,
        "min": ordered[0],
        "max": ordered[-1],
    }


def _failures(
    query_results: Sequence[dict[str, Any]], ground_truth: Sequence[RetrievalGroundTruthRecord]
) -> tuple[dict[str, Any], ...]:
    gt_by_id = {item.query_id: item for item in ground_truth}
    failures: list[dict[str, Any]] = []
    for row in query_results:
        gt = gt_by_id[row["query_id"]]
        types: list[str] = []
        summaries: list[str] = []
        if not gt.answerable:
            if row["evidence_references"]:
                types.append("unanswerable_returns_evidence")
                summaries.append("Unanswerable query returned ranked evidence.")
        else:
            for stage in ("dense", "bm25", "rrf", "reranker"):
                rank = row["first_relevant_ranks"][stage]
                if rank != 1:
                    types.append(f"{stage}_top1_miss")
                    summaries.append(f"{stage} first relevant child rank is {rank}.")
                if rank is None or rank > 5:
                    types.append(f"{stage}_no_relevant_at_5")
            if (
                row["first_relevant_ranks"]["parent"] is None
                or row["first_relevant_ranks"]["parent"] > 5
            ):
                types.append("parent_no_relevant_at_5")
            rrf_rank = row["first_relevant_ranks"]["rrf"] or 6
            reranker_rank = row["first_relevant_ranks"]["reranker"] or 6
            if reranker_rank > rrf_rank:
                types.append("reranker_regressed_vs_rrf")
            dense_rank = row["first_relevant_ranks"]["dense"] or 11
            bm25_rank = row["first_relevant_ranks"]["bm25"] or 11
            if rrf_rank > dense_rank:
                types.append("rrf_regressed_vs_dense")
            if rrf_rank > bm25_rank:
                types.append("rrf_regressed_vs_bm25")
            for stage in STAGE_NAMES:
                candidates = (
                    row["evidence_references"]
                    if stage == "evidence"
                    else row[f"{stage}_candidates"]
                )
                relevant_rank = row["first_relevant_ranks"][stage] or 6
                labels = [
                    item.get("relevance_label") for item in candidates[: max(0, relevant_rank - 1)]
                ]
                if "hard_negative" in labels:
                    types.append(f"{stage}_hard_negative_before_relevant")
                if "overlapping" in labels:
                    types.append(f"{stage}_overlap_before_relevant")
        if types:
            failures.append(
                {
                    "query_id": row["query_id"],
                    "query": row["query"],
                    "category": row["category"],
                    "query_type": row["query_type"],
                    "answerable": row["answerable"],
                    "failure_types": sorted(set(types)),
                    "failure_summary": summaries,
                    "first_relevant_ranks": row["first_relevant_ranks"],
                    "top_candidates": {
                        stage: (_ranked_ids(row, stage)[:5]) for stage in STAGE_NAMES
                    },
                }
            )
    return tuple(failures)


def _evidence_analysis(
    query_results: Sequence[dict[str, Any]], ground_truth: Sequence[RetrievalGroundTruthRecord]
) -> dict[str, Any]:
    gt_by_id = {item.query_id: item for item in ground_truth}
    counts = [len(row["evidence_references"]) for row in query_results]
    chars = [row["evidence_diagnostics"]["total_included_chars"] for row in query_results]
    coverages = []
    for row in query_results:
        gt = gt_by_id[row["query_id"]]
        if gt.answerable:
            ids = set(_ranked_ids(row, "evidence"))
            coverages.append(
                len(ids.intersection(gt.relevant_parent_ids)) / len(gt.relevant_parent_ids)
            )
    return {
        "relevant_parent_coverage": MetricAggregate(
            numerator=math.fsum(coverages),
            denominator=len(coverages),
            value=math.fsum(coverages) / len(coverages),
        ).model_dump(mode="json"),
        "evidence_count_distribution": dict(sorted(Counter(counts).items())),
        "included_character_distribution": timing_summary(chars),
        "truncated_query_count": sum(
            row["evidence_diagnostics"]["truncated"] for row in query_results
        ),
    }


def build_ranking_digest(query_results: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        {
            "query_id": row["query_id"],
            "stages": {stage: _ranked_ids(row, stage) for stage in STAGE_NAMES},
        }
        for row in query_results
    ]
    return _sha256_json(payload)


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def evaluate_retrieval_results(
    query_inputs: Sequence[BenchmarkQueryInput],
    pipeline_results: Sequence[RetrievalPipelineResult],
    ground_truth: Sequence[RetrievalGroundTruthRecord],
) -> EvaluatedRun:
    """Score completed runtime results; relevance data enters only at this boundary."""
    if (
        len(query_inputs) != EXPECTED_QUERY_COUNT
        or len(pipeline_results) != EXPECTED_QUERY_COUNT
        or len(ground_truth) != EXPECTED_QUERY_COUNT
    ):
        raise EvaluationValidationError(
            f"formal evaluation requires exactly {EXPECTED_QUERY_COUNT} ordered queries"
        )
    gt_by_id = {item.query_id: item for item in ground_truth}
    if [item.query_id for item in query_inputs] != [item.query_id for item in ground_truth]:
        raise EvaluationValidationError("query inputs and ground truth order must match")
    query_results: list[dict[str, Any]] = []
    for query_input, result in zip(query_inputs, pipeline_results, strict=True):
        gt = gt_by_id[query_input.query_id]
        if query_input.query != gt.query:
            raise EvaluationValidationError(
                "runtime query text and ground truth query text disagree"
            )
        if result.query != query_input.query:
            raise EvaluationValidationError("pipeline result query order or text changed")
        stage_candidates = {
            stage: _child_candidates(
                getattr(
                    result,
                    {
                        "dense": "dense_results",
                        "bm25": "bm25_results",
                        "rrf": "fused_results",
                        "reranker": "reranked_child_results",
                    }[stage],
                ),
                stage=stage,
                ground_truth=gt,
            )
            for stage in CHILD_STAGES
        }
        stage_candidates["parent"] = _parent_candidates(result.expanded_parent_results, gt)
        references = _evidence_references(result)
        for reference in references:
            reference["candidate_id"] = reference["parent_id"]
            reference["relevance_label"] = _label(
                reference["parent_id"],
                set(gt.relevant_parent_ids),
                set(gt.hard_negative_parent_ids),
                set(gt.overlapping_parent_ids),
            )
        first_ranks = {
            stage: _first_relevant_rank(
                [item["candidate_id"] for item in stage_candidates[stage]],
                gt.relevant_child_ids if stage in CHILD_STAGES else gt.relevant_parent_ids,
                10 if stage in ("dense", "bm25", "rrf") else 5,
            )
            if gt.answerable
            else None
            for stage in (*CHILD_STAGES, "parent")
        }
        first_ranks["evidence"] = (
            _first_relevant_rank(
                [item["parent_id"] for item in references], gt.relevant_parent_ids, 5
            )
            if gt.answerable
            else None
        )
        query_results.append(
            {
                "query_id": query_input.query_id,
                "query": query_input.query,
                "category": gt.category,
                "query_type": gt.query_type,
                "answerable": gt.answerable,
                **{
                    f"{stage}_candidates": stage_candidates[stage]
                    for stage in (*CHILD_STAGES, "parent")
                },
                "evidence_references": references,
                "ground_truth_ids": {
                    "relevant_child_ids": list(gt.relevant_child_ids),
                    "hard_negative_child_ids": list(gt.hard_negative_child_ids),
                    "overlapping_child_ids": list(gt.overlapping_child_ids),
                    "relevant_parent_ids": list(gt.relevant_parent_ids),
                    "hard_negative_parent_ids": list(gt.hard_negative_parent_ids),
                    "overlapping_parent_ids": list(gt.overlapping_parent_ids),
                },
                "first_relevant_ranks": first_ranks,
                "stage_timings": result.stage_timings.model_dump(mode="json"),
                "evidence_diagnostics": {
                    "included_evidence_count": result.evidence_context.included_evidence_count,
                    "total_included_chars": result.evidence_context.total_included_chars,
                    "truncated": result.evidence_context.truncated,
                },
            }
        )
    failures, analysis = analyze_saved_query_results(query_results, ground_truth)
    ranking_digest = build_ranking_digest(query_results)
    deterministic_payload = {
        "rankings": ranking_digest,
        "stage_metrics": analysis["stage_metrics"],
        "hard_negative_and_overlap": analysis["hard_negative_and_overlap"],
        "stage_comparisons": analysis["stage_comparisons"],
        "reranker_movements": analysis["reranker_movements"],
        "category_metrics": analysis["category_metrics"],
        "query_type_metrics": analysis["query_type_metrics"],
        "failure_types": [
            {"query_id": row["query_id"], "failure_types": row["failure_types"]} for row in failures
        ],
    }
    return EvaluatedRun(
        tuple(query_results),
        failures,
        analysis,
        ranking_digest,
        _sha256_json(deterministic_payload),
    )


def analyze_saved_query_results(
    query_results: Sequence[dict[str, Any]],
    ground_truth: Sequence[RetrievalGroundTruthRecord],
    *,
    expected_query_count: int = EXPECTED_QUERY_COUNT,
    expected_answerable_query_count: int = EXPECTED_ANSWERABLE_QUERY_COUNT,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Recompute formal analysis from saved results without model inference.

    This is intentionally the same scoring path used immediately after a live
    Pipeline run.  It lets comparison reports derive both variants' metrics
    from their versioned Query Results rather than trusting summary fields.
    """
    if not 0 <= expected_answerable_query_count <= expected_query_count:
        raise ValueError("expected answerable query count must be within the query count")
    if len(query_results) != expected_query_count or len(ground_truth) != expected_query_count:
        raise EvaluationValidationError(
            f"formal analysis requires exactly {expected_query_count} ordered queries"
        )
    if sum(item.answerable for item in ground_truth) != expected_answerable_query_count:
        raise EvaluationValidationError("ground truth answerable query count is unexpected")
    expected_ids = [item.query_id for item in ground_truth]
    if [str(row.get("query_id")) for row in query_results] != expected_ids:
        raise EvaluationValidationError("saved query results and ground truth order must match")
    for row, gt in zip(query_results, ground_truth, strict=True):
        if row.get("query") != gt.query or row.get("answerable") != gt.answerable:
            raise EvaluationValidationError(
                "saved query result identity does not match ground truth"
            )
        for stage in STAGE_NAMES:
            ranked = _ranked_ids(row, stage)
            relevant = gt.relevant_child_ids if stage in CHILD_STAGES else gt.relevant_parent_ids
            expected_rank = (
                _first_relevant_rank(
                    ranked, relevant, 10 if stage in ("dense", "bm25", "rrf") else 5
                )
                if gt.answerable
                else None
            )
            if row["first_relevant_ranks"].get(stage) != expected_rank:
                raise EvaluationValidationError(
                    f"saved {stage} first relevant rank does not match Ground Truth"
                )

    failures = _failures(query_results, ground_truth)
    comparisons = {
        f"{after}_vs_{before}": _compare_stage(
            query_results, ground_truth, before=before, after=after
        )
        for before, after in (
            ("dense", "bm25"),
            ("dense", "rrf"),
            ("bm25", "rrf"),
            ("rrf", "reranker"),
            ("reranker", "parent"),
        )
    }
    analysis = {
        "query_count": expected_query_count,
        "answerable_query_count": expected_answerable_query_count,
        "unanswerable_query_count": expected_query_count - expected_answerable_query_count,
        "stage_metrics": _stage_metrics(query_results, ground_truth),
        "evidence_analysis": _evidence_analysis(query_results, ground_truth),
        "hard_negative_and_overlap": _exposure_statistics(query_results, ground_truth),
        "stage_comparisons": comparisons,
        "reranker_movements": _reranker_movements(query_results, ground_truth),
        "category_metrics": _group_metrics(query_results, ground_truth, "category"),
        "query_type_metrics": _group_metrics(query_results, ground_truth, "query_type"),
        "failure_case_counts": {
            kind: Counter(
                failure_type for row in failures for failure_type in row["failure_types"]
            )[kind]
            for kind in FAILURE_TYPE_UNIVERSE
        },
        "unanswerable_behavior": [
            {
                "query_id": row["query_id"],
                "child_candidate_counts": {
                    stage: len(row[f"{stage}_candidates"]) for stage in CHILD_STAGES
                },
                "parent_count": len(row["parent_candidates"]),
                "evidence_count": len(row["evidence_references"]),
                "top_evidence": (
                    {
                        key: row["evidence_references"][0].get(key)
                        for key in (
                            "evidence_id",
                            "parent_id",
                            "document_id",
                            "source",
                            "start_offset",
                            "end_offset",
                            "relevance_label",
                        )
                    }
                    if row["evidence_references"]
                    else None
                ),
                "top5_label_counts": {
                    stage: dict(
                        sorted(
                            Counter(
                                item["relevance_label"]
                                for item in (
                                    row["evidence_references"]
                                    if stage == "evidence"
                                    else row[f"{stage}_candidates"]
                                )[:5]
                            ).items()
                        )
                    )
                    for stage in STAGE_NAMES
                },
            }
            for row in query_results
            if not row["answerable"]
        ],
        "query_timing_distributions": _timing_distribution(query_results),
        "overlap_interpretation": (
            "Overlap chunks cover both relevant and hard-negative clauses; they are analyzed "
            "separately, remain Relevant, and are excluded from pure hard negatives. "
            "Overlap-before-first-Relevant is a structural zero, not a performance result."
        ),
    }
    return failures, analysis


def assert_deterministic_runs(run_a: EvaluatedRun, run_b: EvaluatedRun) -> None:
    if (
        run_a.ranking_digest != run_b.ranking_digest
        or run_a.deterministic_digest != run_b.deterministic_digest
    ):
        raise EvaluationValidationError("Run A and Run B deterministic retrieval outputs differ")


def stable_json(payload: Any, *, indent: int | None = 2) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=indent) + "\n"


def stable_jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )


def sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def write_baseline_artifacts(
    *,
    output_dir: Path,
    main_report: dict[str, Any],
    query_results: Sequence[Mapping[str, Any]],
    failure_cases: Sequence[Mapping[str, Any]],
    runtime_profile: dict[str, Any],
) -> dict[str, str]:
    query_text = stable_jsonl(query_results)
    failure_text = stable_jsonl(failure_cases)
    runtime_text = stable_json(runtime_profile)
    report = dict(main_report)
    report["generated_file_hashes"] = {
        "m2c2a2_query_results.jsonl": sha256_text(query_text),
        "m2c2a2_failure_cases.jsonl": sha256_text(failure_text),
    }
    report["artifact_hash_semantics"] = {
        "purpose": "Integrity hashes for the final files written by this invocation.",
        "cross_run_determinism_contract": "deterministic_ranking_digest",
        "runtime_dependent_artifacts": [
            "m2c2a2_query_results.jsonl",
            "m2c2a2_runtime_profile.json",
        ],
        "ranking_dependent_artifacts": ["m2c2a2_failure_cases.jsonl"],
        "main_report_byte_stability": (
            "Not guaranteed because it records the final runtime-dependent Query Results hash."
        ),
        "self_hash_included": False,
    }
    report_text = stable_json(report)
    files = {
        output_dir / "m2c2a2_retrieval_baseline.json": report_text,
        output_dir / "m2c2a2_query_results.jsonl": query_text,
        output_dir / "m2c2a2_failure_cases.jsonl": failure_text,
        output_dir / "m2c2a2_runtime_profile.json": runtime_text,
    }
    write_text_files_atomically(files)
    return {path.name: sha256_text(content) for path, content in files.items()}
