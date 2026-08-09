"""Offline-only fairness contracts for the M2C-2B-2 chunking comparison."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from decision_agent.evaluation.enterprise_chunking_comparison import (
    EMBEDDING_MODEL,
    EMBEDDING_REVISION,
    RERANKER_MODEL,
    RERANKER_REVISION,
    build_chunking_comparison,
    chunk_length_statistics,
    stable_comparison_json,
    validate_fixed_window_artifacts,
)
from decision_agent.evaluation.enterprise_kb_ground_truth import RetrievalGroundTruthRecord
from decision_agent.evaluation.enterprise_retrieval_baseline import (
    analyze_saved_query_results,
)
from decision_agent.exceptions import EvaluationValidationError
from decision_agent.retrieval.pipeline import RetrievalPipelineConfig

ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = ROOT / "datasets/enterprise_kb/m2c1"
ARTIFACT_ROOT = ROOT / "artifacts"
EVALUATION_ROOT = ARTIFACT_ROOT / "evaluation"


@pytest.fixture(scope="module")
def fixed():
    return validate_fixed_window_artifacts(
        report_path=EVALUATION_ROOT / "m2c2a2_retrieval_baseline.json",
        query_results_path=EVALUATION_ROOT / "m2c2a2_query_results.jsonl",
        runtime_path=EVALUATION_ROOT / "m2c2a2_runtime_profile.json",
    )


@pytest.fixture(scope="module")
def fixed_ground_truth(fixed):
    """Rehydrate the immutable 50-query Ground Truth embedded in its result artifact."""
    records = []
    for row in fixed.query_results:
        identifiers = row["ground_truth_ids"]
        records.append(
            RetrievalGroundTruthRecord(
                query_id=row["query_id"],
                query=row["query"],
                category=row["category"],
                query_type=row["query_type"],
                answerable=row["answerable"],
                relevant_clause_ids=(),
                relevant_parent_ids=identifiers["relevant_parent_ids"],
                relevant_child_ids=identifiers["relevant_child_ids"],
                hard_negative_clause_ids=(),
                hard_negative_parent_ids=identifiers["hard_negative_parent_ids"],
                hard_negative_child_ids=identifiers["hard_negative_child_ids"],
                overlapping_parent_ids=identifiers["overlapping_parent_ids"],
                overlapping_child_ids=identifiers["overlapping_child_ids"],
                expected_evidence_count=row["evidence_diagnostics"]["included_evidence_count"],
            )
        )
    assert len(records) == 50
    assert sum(record.answerable for record in records) == 46
    return tuple(records)


def _runtime_contract() -> dict[str, object]:
    return {
        "comparison_contract": {
            "models": {
                "embedding": {
                    "name": EMBEDDING_MODEL,
                    "revision": EMBEDDING_REVISION,
                    "dimension": 512,
                },
                "reranker": {"name": RERANKER_MODEL, "revision": RERANKER_REVISION},
            },
            "fixed_config": RetrievalPipelineConfig().model_dump(mode="json"),
        },
        "initialization": {"total_initialization_seconds": 0.0},
    }


def _summary(name: str) -> dict[str, object]:
    return json.loads((ARTIFACT_ROOT / "datasets" / name).read_text(encoding="utf-8"))


def _chunk_lengths() -> dict[str, object]:
    return chunk_length_statistics(
        parent_path=DATASET_ROOT / "generated/parent_chunks.jsonl",
        child_path=DATASET_ROOT / "generated/child_chunks.jsonl",
    )


def test_fixed_window_artifacts_are_immutable_and_fairness_validated(fixed) -> None:
    assert fixed.report["counts"]["query"] == 50
    assert len(fixed.query_results) == 50
    assert fixed.report["models"]["embedding"]["revision"] == EMBEDDING_REVISION


def test_saved_results_are_recomputed_from_ground_truth(fixed, fixed_ground_truth) -> None:
    _, analysis = analyze_saved_query_results(
        fixed.query_results,
        fixed_ground_truth,
        expected_query_count=50,
        expected_answerable_query_count=46,
    )
    assert analysis["stage_metrics"]["reranker"]["query_count"] == 46


def test_saved_results_reject_stale_first_relevant_rank(fixed, fixed_ground_truth) -> None:
    changed = copy.deepcopy(fixed.query_results)
    changed[0]["first_relevant_ranks"]["dense"] = 10
    with pytest.raises(EvaluationValidationError, match="first relevant rank"):
        analyze_saved_query_results(
            changed,
            fixed_ground_truth,
            expected_query_count=50,
            expected_answerable_query_count=46,
        )


def test_comparison_is_derived_from_query_results_and_partitions_queries(
    fixed, fixed_ground_truth
) -> None:
    report = build_chunking_comparison(
        fixed=fixed,
        fixed_ground_truth=fixed_ground_truth,
        clause_query_results=fixed.query_results,
        clause_ground_truth=fixed_ground_truth,
        clause_runtime=_runtime_contract(),
        fixed_summary=_summary("m2c1_parent_child_summary.json"),
        clause_summary=_summary("m2c2b1_clause_aware_summary.json"),
        fixed_chunk_lengths=_chunk_lengths(),
        clause_chunk_lengths=_chunk_lengths(),
        input_hashes={"fixed_results": "0" * 64, "clause_results": "1" * 64},
    )
    movements = report["query_movements"]
    assert sum(item["count"] for item in movements["reranker"]["hit_at_1"].values()) == 46
    assert report["stage_metric_comparison"]["dense"]["hit_rate_at_1"]["delta"] == 0.0
    assert report["comparison_contract"]["only_variable"] == "dataset_chunk_variant"
    assert "same stable human-authored Clause labels" in report["cross_variant_semantics"]["child"]
    assert "separate process observation" in report["runtime_comparison"]["semantics"]


def test_comparison_rejects_mismatched_pipeline_contract(fixed, fixed_ground_truth) -> None:
    runtime = _runtime_contract()
    runtime["comparison_contract"]["fixed_config"]["rrf_k"] = 61.0
    with pytest.raises(EvaluationValidationError, match="Pipeline contract"):
        build_chunking_comparison(
            fixed=fixed,
            fixed_ground_truth=fixed_ground_truth,
            clause_query_results=fixed.query_results,
            clause_ground_truth=fixed_ground_truth,
            clause_runtime=runtime,
            fixed_summary=_summary("m2c1_parent_child_summary.json"),
            clause_summary=_summary("m2c2b1_clause_aware_summary.json"),
            fixed_chunk_lengths=_chunk_lengths(),
            clause_chunk_lengths=_chunk_lengths(),
            input_hashes={},
        )


def test_comparison_rejects_changed_query_identity(fixed, fixed_ground_truth) -> None:
    changed = copy.deepcopy(fixed.query_results)
    changed[0]["query_id"] = "changed"
    with pytest.raises(EvaluationValidationError, match="order"):
        build_chunking_comparison(
            fixed=fixed,
            fixed_ground_truth=fixed_ground_truth,
            clause_query_results=changed,
            clause_ground_truth=fixed_ground_truth,
            clause_runtime=_runtime_contract(),
            fixed_summary=_summary("m2c1_parent_child_summary.json"),
            clause_summary=_summary("m2c2b1_clause_aware_summary.json"),
            fixed_chunk_lengths=_chunk_lengths(),
            clause_chunk_lengths=_chunk_lengths(),
            input_hashes={},
        )


def test_serialized_comparison_has_no_self_hash_timestamp_or_absolute_path(
    fixed, fixed_ground_truth
) -> None:
    report = build_chunking_comparison(
        fixed=fixed,
        fixed_ground_truth=fixed_ground_truth,
        clause_query_results=fixed.query_results,
        clause_ground_truth=fixed_ground_truth,
        clause_runtime=_runtime_contract(),
        fixed_summary=_summary("m2c1_parent_child_summary.json"),
        clause_summary=_summary("m2c2b1_clause_aware_summary.json"),
        fixed_chunk_lengths=_chunk_lengths(),
        clause_chunk_lengths=_chunk_lengths(),
        input_hashes={},
    )
    text = stable_comparison_json(report)
    assert "generated_at" not in text
    assert '"self_hash":' not in text
    assert str(ROOT) not in text


def test_real_runner_remains_opt_in_and_offline_only() -> None:
    script = (ROOT / "scripts/evaluate_m2c2b2_clause_aware_comparison.py").read_text(
        encoding="utf-8"
    )
    assert 'os.environ.get("HF_HUB_OFFLINE") != "1"' in script
    assert 'os.environ.get("TRANSFORMERS_OFFLINE") != "1"' in script
    assert "SentenceTransformerEmbeddingProvider" in script
    assert "SentenceTransformerCrossEncoderReranker" in script
    assert "Fake" not in script
