"""Offline validation and fair comparison for fixed-window and Clause-aware runs."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from decision_agent.evaluation.enterprise_kb_ground_truth import RetrievalGroundTruthRecord
from decision_agent.evaluation.enterprise_retrieval_baseline import (
    BASELINE_ID,
    STAGE_NAMES,
    analyze_saved_query_results,
    build_ranking_digest,
    sha256_text,
    stable_json,
    timing_summary,
)
from decision_agent.exceptions import EvaluationValidationError
from decision_agent.retrieval.pipeline import RetrievalPipelineConfig

FIXED_WINDOW_RANKING_DIGEST = "0acbcdea406fa6d9dc10d7818479a2ee9b0892fa1bd5241d68daf637d4b3b75e"
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
EMBEDDING_REVISION = "7999e1d3359715c523056ef9478215996d62a620"
RERANKER_MODEL = "BAAI/bge-reranker-base"
RERANKER_REVISION = "2cfc18c9415c912f9d8155881c133215df768a70"
COMPARISON_ID = "m2c2b2-fixed-window-vs-clause-aware-v1"
_COMPARISON_STAGES = ("dense", "bm25", "rrf", "reranker", "parent")
_HISTORICAL_QUERY_COUNT = 50
_HISTORICAL_ANSWERABLE_QUERY_COUNT = 46


@dataclass(frozen=True, slots=True)
class FixedWindowArtifacts:
    """Validated immutable fixed-window inputs for one fair comparison."""

    report: dict[str, Any]
    query_results: tuple[dict[str, Any], ...]
    runtime: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationValidationError(f"cannot read {path.name}") from exc
    if not isinstance(value, dict):
        raise EvaluationValidationError(f"{path.name} must be a JSON object")
    return value


def read_query_results(path: Path) -> tuple[dict[str, Any], ...]:
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise EvaluationValidationError(f"cannot read {path.name}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationValidationError(
                f"{path.name} line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(row, dict):
            raise EvaluationValidationError(f"{path.name} rows must be JSON objects")
        rows.append(row)
    return tuple(rows)


def chunk_length_statistics(*, parent_path: Path, child_path: Path) -> dict[str, dict[str, float]]:
    """Return content-length means from versioned chunks without touching retrieval."""
    result: dict[str, dict[str, float]] = {}
    for kind, path in (("parent", parent_path), ("child", child_path)):
        rows = read_query_results(path)
        lengths = [len(str(row.get("content", ""))) for row in rows]
        if not lengths or any(length <= 0 for length in lengths):
            raise EvaluationValidationError(f"{path.name} must contain nonempty chunk content")
        result[kind] = {
            "count": float(len(lengths)),
            "mean_content_chars": sum(lengths) / len(lengths),
        }
    return result


def _portable_file_hash(path: Path) -> str:
    try:
        return sha256_text(path.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise EvaluationValidationError(f"cannot read {path.name}") from exc


def _expected_models() -> dict[str, dict[str, Any]]:
    return {
        "embedding": {"name": EMBEDDING_MODEL, "revision": EMBEDDING_REVISION, "dimension": 512},
        "reranker": {"name": RERANKER_MODEL, "revision": RERANKER_REVISION},
    }


def validate_fixed_window_artifacts(
    *, report_path: Path, query_results_path: Path, runtime_path: Path
) -> FixedWindowArtifacts:
    """Reject a comparison unless the committed fixed-window baseline is exact."""
    report = _read_json(report_path)
    query_results = read_query_results(query_results_path)
    runtime = _read_json(runtime_path)
    if report.get("baseline_id") != BASELINE_ID:
        raise EvaluationValidationError("fixed report baseline_id is not M2C-2A-2")
    if report.get("ranking_digest") != FIXED_WINDOW_RANKING_DIGEST:
        raise EvaluationValidationError("fixed report ranking digest is unexpected")
    if report.get("models") != _expected_models():
        raise EvaluationValidationError("fixed report model names or revisions are unexpected")
    if report.get("fixed_config") != RetrievalPipelineConfig().model_dump(mode="json"):
        raise EvaluationValidationError("fixed report Pipeline configuration is unexpected")
    if report.get("counts") != {
        "document": 10,
        "parent": 30,
        "child": 87,
        "clause": 129,
        "query": 50,
        "answerable": 46,
        "unanswerable": 4,
    }:
        raise EvaluationValidationError("fixed report count contract is unexpected")
    if (
        len(query_results) != 50
        or build_ranking_digest(query_results) != FIXED_WINDOW_RANKING_DIGEST
    ):
        raise EvaluationValidationError(
            "fixed Query Results do not match the committed ranking digest"
        )
    hashes = report.get("generated_file_hashes")
    if not isinstance(hashes, Mapping) or hashes.get(
        "m2c2a2_query_results.jsonl"
    ) != _portable_file_hash(query_results_path):
        raise EvaluationValidationError("fixed Query Results hash does not match fixed report")
    if runtime.get("baseline_id") != BASELINE_ID:
        raise EvaluationValidationError("fixed runtime profile does not belong to M2C-2A-2")
    return FixedWindowArtifacts(report, query_results, runtime)


def _movement(
    fixed_rows: Sequence[dict[str, Any]], clause_rows: Sequence[dict[str, Any]], stage: str
) -> dict[str, Any]:
    hit = {key: [] for key in ("improved", "regressed", "unchanged")}
    reciprocal_rank = {key: [] for key in ("improved", "regressed", "unchanged")}
    for fixed, clause in zip(fixed_rows, clause_rows, strict=True):
        if not fixed["answerable"]:
            continue
        fixed_rank = fixed["first_relevant_ranks"][stage]
        clause_rank = clause["first_relevant_ranks"][stage]
        fixed_hit, clause_hit = fixed_rank == 1, clause_rank == 1
        hit[
            "improved"
            if clause_hit and not fixed_hit
            else "regressed"
            if fixed_hit and not clause_hit
            else "unchanged"
        ].append(fixed["query_id"])
        fixed_rr = 0.0 if fixed_rank is None else 1 / fixed_rank
        clause_rr = 0.0 if clause_rank is None else 1 / clause_rank
        reciprocal_rank[
            "improved"
            if clause_rr > fixed_rr
            else "regressed"
            if clause_rr < fixed_rr
            else "unchanged"
        ].append(fixed["query_id"])
    return {
        "stage": stage,
        "hit_at_1": {key: {"count": len(ids), "query_ids": ids} for key, ids in hit.items()},
        "reciprocal_rank_at_5": {
            key: {"count": len(ids), "query_ids": ids} for key, ids in reciprocal_rank.items()
        },
    }


def _stage_delta(fixed: Mapping[str, Any], clause: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in (
        "hit_rate_at_1",
        "hit_rate_at_3",
        "hit_rate_at_5",
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "mrr_at_5",
    ):
        fixed_value = fixed["metrics"][name]["value"]
        clause_value = clause["metrics"][name]["value"]
        result[name] = {
            "fixed_window": fixed_value,
            "clause_aware": clause_value,
            "delta": clause_value - fixed_value,
        }
    return result


def _exposure_delta(fixed: Mapping[str, Any], clause: Mapping[str, Any]) -> dict[str, Any]:
    names = (
        "pure_hard_negative_top1",
        "pure_hard_negative_top5_exposure",
        "overlap_top1",
        "overlap_top5_exposure",
        "no_relevant_at_5_query_ids",
    )
    result: dict[str, Any] = {}
    for name in names:
        fixed_value, clause_value = fixed[name], clause[name]
        if name == "no_relevant_at_5_query_ids":
            result[name] = {
                "fixed_window": {"count": len(fixed_value), "query_ids": fixed_value},
                "clause_aware": {"count": len(clause_value), "query_ids": clause_value},
            }
        else:
            result[name] = {
                "fixed_window": fixed_value,
                "clause_aware": clause_value,
                "delta_count": clause_value["count"] - fixed_value["count"],
            }
    return result


def _runtime_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    stage_names = tuple(rows[0]["stage_timings"])
    values = {name: [row["stage_timings"][name] for row in rows] for name in stage_names}
    values["parent_evidence_seconds"] = [
        row["stage_timings"]["parent_expansion_seconds"]
        + row["stage_timings"]["evidence_context_seconds"]
        for row in rows
    ]
    return {name: timing_summary(stage_values) for name, stage_values in values.items()}


def _case(row: Mapping[str, Any], stage: str) -> dict[str, Any]:
    key = "evidence_references" if stage == "evidence" else f"{stage}_candidates"
    return {
        "query_id": row["query_id"],
        "query": row["query"],
        "first_relevant_rank": row["first_relevant_ranks"][stage],
        "top5": [
            {
                "candidate_id": item.get("candidate_id", item.get("parent_id")),
                "relevance_label": item.get("relevance_label"),
            }
            for item in row[key][:5]
        ],
    }


def _representative_cases(
    fixed_rows: Sequence[dict[str, Any]],
    clause_rows: Sequence[dict[str, Any]],
    movements: Mapping[str, Any],
) -> dict[str, Any]:
    fixed_by_id = {row["query_id"]: row for row in fixed_rows}
    clause_by_id = {row["query_id"]: row for row in clause_rows}

    def paired(ids: Sequence[str], stage: str) -> dict[str, Any] | None:
        if not ids:
            return None
        query_id = ids[0]
        return {
            "stage": stage,
            "fixed_window": _case(fixed_by_id[query_id], stage),
            "clause_aware": _case(clause_by_id[query_id], stage),
        }

    improved = next(
        (
            paired(movements[stage]["reciprocal_rank_at_5"]["improved"]["query_ids"], stage)
            for stage in _COMPARISON_STAGES
            if movements[stage]["reciprocal_rank_at_5"]["improved"]["query_ids"]
        ),
        None,
    )
    regressed = next(
        (
            paired(movements[stage]["reciprocal_rank_at_5"]["regressed"]["query_ids"], stage)
            for stage in _COMPARISON_STAGES
            if movements[stage]["reciprocal_rank_at_5"]["regressed"]["query_ids"]
        ),
        None,
    )
    reranker_changed = next(
        (
            paired(movements["reranker"]["reciprocal_rank_at_5"][name]["query_ids"], "reranker")
            for name in ("improved", "regressed")
            if movements["reranker"]["reciprocal_rank_at_5"][name]["query_ids"]
        ),
        None,
    )
    unanswerable = next((row for row in clause_rows if not row["answerable"]), None)
    return {
        "improved": improved,
        "regressed": regressed,
        "reranker_changed": reranker_changed,
        "unanswerable_returns_evidence": None
        if unanswerable is None
        else _case(unanswerable, "evidence"),
    }


def build_chunking_comparison(
    *,
    fixed: FixedWindowArtifacts,
    fixed_ground_truth: Sequence[RetrievalGroundTruthRecord],
    clause_query_results: Sequence[dict[str, Any]],
    clause_ground_truth: Sequence[RetrievalGroundTruthRecord],
    clause_runtime: Mapping[str, Any],
    fixed_summary: Mapping[str, Any],
    clause_summary: Mapping[str, Any],
    fixed_chunk_lengths: Mapping[str, Any],
    clause_chunk_lengths: Mapping[str, Any],
    input_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Build a stable report from saved results, with only the chunk variant changed."""
    fixed_failures, fixed_analysis = analyze_saved_query_results(
        fixed.query_results,
        fixed_ground_truth,
        expected_query_count=_HISTORICAL_QUERY_COUNT,
        expected_answerable_query_count=_HISTORICAL_ANSWERABLE_QUERY_COUNT,
    )
    clause_failures, clause_analysis = analyze_saved_query_results(
        clause_query_results,
        clause_ground_truth,
        expected_query_count=_HISTORICAL_QUERY_COUNT,
        expected_answerable_query_count=_HISTORICAL_ANSWERABLE_QUERY_COUNT,
    )
    if [row["query_id"] for row in fixed.query_results] != [
        row["query_id"] for row in clause_query_results
    ]:
        raise EvaluationValidationError("fixed and Clause-aware Query Results order must match")
    if [item.query for item in fixed_ground_truth] != [item.query for item in clause_ground_truth]:
        raise EvaluationValidationError("fixed and Clause-aware Query text must match")
    if [item.answerable for item in fixed_ground_truth] != [
        item.answerable for item in clause_ground_truth
    ]:
        raise EvaluationValidationError("fixed and Clause-aware 46/4 contract must match")
    clause_contract = clause_runtime.get("comparison_contract")
    expected_config = RetrievalPipelineConfig().model_dump(mode="json")
    if (
        not isinstance(clause_contract, Mapping)
        or clause_contract.get("models") != _expected_models()
        or clause_contract.get("fixed_config") != expected_config
    ):
        raise EvaluationValidationError(
            "Clause-aware runtime model or Pipeline contract is unexpected"
        )
    movements = {
        stage: _movement(fixed.query_results, clause_query_results, stage)
        for stage in _COMPARISON_STAGES
    }
    fixed_runtime = _runtime_summary(fixed.query_results)
    clause_runtime_summary = _runtime_summary(clause_query_results)
    return {
        "schema_version": "1.0",
        "comparison_id": COMPARISON_ID,
        "comparison_contract": {
            "only_variable": "dataset_chunk_variant",
            "fixed_window_strategy_id": "fixed-window-v1",
            "clause_aware_strategy_id": "clause-aware-v1",
            "query_count": 50,
            "answerable_query_count": 46,
            "unanswerable_query_count": 4,
            "models": _expected_models(),
            "device": "cpu",
            "offline_only": True,
            "fixed_config": expected_config,
            "ground_truth_not_used_in_runtime_ranking": True,
        },
        "cross_variant_semantics": {
            "child": (
                "Child IDs differ by chunk variant, but both Ground Truth mappings originate from "
                "the same stable human-authored Clause labels; child-stage metrics therefore "
                "compare retrieval under the same Clause semantics."
            ),
            "parent_and_evidence": (
                "Parent boundaries, counts, and relevant_parent_ids differ by variant. Parent and "
                "Evidence metrics are evidence-coverage observations and are not same-identity "
                "candidate comparisons."
            ),
            "candidate_ids": "Candidate IDs are variant-local and are never compared for equality.",
        },
        "input_file_hashes": dict(sorted(input_hashes.items())),
        "ranking_digests": {
            "fixed_window": build_ranking_digest(fixed.query_results),
            "clause_aware": build_ranking_digest(clause_query_results),
        },
        "structural_comparison": {
            "fixed_window": {
                "parent_chunk_count": fixed_summary["parent_chunk_count"],
                "child_chunk_count": fixed_summary["child_chunk_count"],
                "parent_collision_query_count": fixed_summary["overlapping_parent_query_count"],
                "child_collision_query_count": fixed_summary["overlapping_child_query_count"],
                "no_independent_hard_negative_parent_query_count": fixed_summary[
                    "no_independent_hard_negative_parent_query_count"
                ],
                "no_independent_hard_negative_child_query_count": fixed_summary[
                    "no_independent_hard_negative_child_query_count"
                ],
                "cross_clause_child_count": None,
                "cross_section_child_count": None,
                "chunk_length_chars": fixed_chunk_lengths,
            },
            "clause_aware": {
                "parent_chunk_count": clause_summary["parent_chunk_count"],
                "child_chunk_count": clause_summary["child_chunk_count"],
                "parent_collision_query_count": clause_summary["overlapping_parent_query_count"],
                "child_collision_query_count": clause_summary["overlapping_child_query_count"],
                "no_independent_hard_negative_parent_query_count": clause_summary[
                    "no_independent_hard_negative_parent_query_count"
                ],
                "no_independent_hard_negative_child_query_count": clause_summary[
                    "no_independent_hard_negative_child_query_count"
                ],
                "cross_clause_child_count": clause_summary["cross_clause_child_count"],
                "cross_section_child_count": clause_summary["cross_section_child_count"],
                "chunk_length_chars": clause_chunk_lengths,
            },
        },
        "stage_metric_comparison": {
            stage: _stage_delta(
                fixed_analysis["stage_metrics"][stage], clause_analysis["stage_metrics"][stage]
            )
            for stage in STAGE_NAMES
        },
        "hard_negative_and_overlap_comparison": {
            stage: _exposure_delta(
                fixed_analysis["hard_negative_and_overlap"][stage],
                clause_analysis["hard_negative_and_overlap"][stage],
            )
            for stage in STAGE_NAMES
        },
        "parent_evidence_comparison": {
            "fixed_window": fixed_analysis["evidence_analysis"],
            "clause_aware": clause_analysis["evidence_analysis"],
        },
        "query_movements": movements,
        "representative_cases": _representative_cases(
            fixed.query_results, clause_query_results, movements
        ),
        "unanswerable_behavior": {
            "fixed_window": fixed_analysis["unanswerable_behavior"],
            "clause_aware": clause_analysis["unanswerable_behavior"],
        },
        "runtime_comparison": {
            "fixed_window_query_stage_seconds": fixed_runtime,
            "clause_aware_query_stage_seconds": clause_runtime_summary,
            "initialization_seconds": {
                "fixed_window": fixed.runtime["run_b"]["initialization"],
                "clause_aware": clause_runtime["initialization"],
            },
            "semantics": (
                "Both variants use the same model revisions, CPU, and Pipeline configuration, but "
                "the fixed result is a historical run and the Clause-aware result is one separate "
                "process observation. OS cache, process state, and CPU variation can affect "
                "values; this is not a controlled performance benchmark or a causal latency "
                "attribution."
            ),
        },
        "failure_case_counts": {
            "fixed_window": fixed_analysis["failure_case_counts"],
            "clause_aware": clause_analysis["failure_case_counts"],
        },
        "limitations": [
            (
                "The only experimental variable is the checked-in chunk variant; this does not "
                "establish production quality."
            ),
            (
                "Fixed and Clause-aware Ground Truth use their own runtime chunk identities, so "
                "Parent-versus-Child observations remain cross-granularity coverage observations."
            ),
            (
                "Runtime values are one local CPU observation and are not a production performance "
                "claim; more Child candidates and Cross-Encoder work are plausible factors but not "
                "independently proven causes."
            ),
            (
                "Unanswerable queries still exercise retrieval only and do not establish answer "
                "refusal behavior."
            ),
        ],
        "artifact_hash_semantics": {
            "self_hash_included": False,
            "timestamps_included": False,
            "absolute_paths_included": False,
        },
        "saved_result_failure_counts": {
            "fixed_window": len(fixed_failures),
            "clause_aware": len(clause_failures),
        },
    }


def stable_comparison_json(report: Mapping[str, Any]) -> str:
    """Serialize a comparison without a self-hash or runtime-generated timestamp."""
    return stable_json(dict(report))


__all__ = [
    "COMPARISON_ID",
    "FIXED_WINDOW_RANKING_DIGEST",
    "FixedWindowArtifacts",
    "build_chunking_comparison",
    "chunk_length_statistics",
    "read_query_results",
    "stable_comparison_json",
    "validate_fixed_window_artifacts",
]
