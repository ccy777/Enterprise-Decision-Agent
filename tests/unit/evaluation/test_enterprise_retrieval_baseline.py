from __future__ import annotations

import json
from pathlib import Path

import pytest

from decision_agent.evaluation.enterprise_retrieval_baseline import (
    EvaluatedRun,
    assert_deterministic_runs,
    build_ranking_digest,
    evaluate_retrieval_results,
    load_benchmark_query_inputs,
    load_retrieval_ground_truth,
    sha256_text,
    stable_json,
    timing_summary,
    write_baseline_artifacts,
)
from decision_agent.exceptions import EvaluationValidationError
from decision_agent.retrieval.evidence_context import (
    EvidenceContext,
    EvidenceItem,
    EvidenceReference,
)
from decision_agent.retrieval.fusion import FusedResult, FusionContribution
from decision_agent.retrieval.parent_expansion import MatchedChild, ParentExpansionResult
from decision_agent.retrieval.pipeline import (
    ChildRetrievalResult,
    RetrievalPipelineConfig,
    RetrievalPipelineResult,
    RetrievalStageTiming,
)
from decision_agent.retrieval.reranking import RerankedResult

ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "datasets/enterprise_kb/m2c1"
QUERY_PATH = DATASET / "query_blueprint.jsonl"
GT_PATH = DATASET / "generated/retrieval_ground_truth.jsonl"


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _fake_results():
    queries = load_benchmark_query_inputs(QUERY_PATH)
    ground_truth = load_retrieval_ground_truth(GT_PATH)
    child_rows = _rows(DATASET / "generated/child_chunks.jsonl")
    parent_rows = _rows(DATASET / "generated/parent_chunks.jsonl")
    children = {str(row["chunk_id"]): row for row in child_rows}
    parents = {str(row["chunk_id"]): row for row in parent_rows}
    child_ids = tuple(children)
    parent_ids = tuple(parents)
    results = []
    timing = RetrievalStageTiming(
        dense_query_seconds=0.01,
        bm25_query_seconds=0.02,
        rrf_seconds=0.001,
        reranker_seconds=0.03,
        parent_expansion_seconds=0.002,
        evidence_context_seconds=0.003,
        total_runtime_seconds=0.07,
    )
    for query, gt in zip(queries, ground_truth, strict=True):
        ordered_children = list(gt.relevant_child_ids)
        ordered_children.extend(item for item in child_ids if item not in ordered_children)
        ordered_children = ordered_children[:10]
        dense = tuple(
            ChildRetrievalResult(
                rank=rank,
                candidate_id=child_id,
                record_id=child_id,
                parent_id=str(children[child_id]["parent_id"]),
                document_id=str(children[child_id]["document_id"]),
                content=str(children[child_id]["content"]),
                score=1 / rank,
                source_name="dense",
                source=str(children[child_id]["source"]),
                metadata={"parent_id": str(children[child_id]["parent_id"])},
                provenance={"source": str(children[child_id]["source"])},
            )
            for rank, child_id in enumerate(ordered_children, 1)
        )
        bm25 = tuple(
            item.model_copy(update={"source_name": "bm25", "score": item.score / 2})
            for item in dense
        )
        fused = tuple(
            FusedResult(
                final_rank=rank,
                document_id=item.document_id,
                candidate_id=item.candidate_id,
                fused_score=1 / (60 + rank),
                best_source_rank=rank,
                matched_source_count=1,
                source_contributions=(
                    FusionContribution(
                        source_name="dense",
                        source_rank=rank,
                        contribution=1 / (60 + rank),
                        source_score=item.score,
                        candidate_id=item.candidate_id,
                        record_id=item.record_id,
                    ),
                ),
                record_id=item.record_id,
                content=item.content,
                metadata={"parent_id": item.parent_id},
                provenance={"source": item.source},
            )
            for rank, item in enumerate(dense, 1)
        )
        reranked = tuple(
            RerankedResult(
                final_rank=rank,
                document_id=item.document_id,
                candidate_id=item.candidate_id,
                content=item.content,
                reranker_score=1 / rank,
                upstream_rank=rank,
                record_id=item.record_id,
                upstream_score=item.fused_score,
                metadata=item.metadata,
                provenance=item.provenance,
            )
            for rank, item in enumerate(fused[:5], 1)
        )
        ordered_parents = list(gt.relevant_parent_ids)
        ordered_parents.extend(item for item in parent_ids if item not in ordered_parents)
        ordered_parents = ordered_parents[:5]
        expanded = []
        for rank, parent_id in enumerate(ordered_parents, 1):
            parent = parents[parent_id]
            child_id = next(
                item for item, row in children.items() if str(row["parent_id"]) == parent_id
            )
            child = children[child_id]
            matched = MatchedChild(
                child_id=child_id,
                parent_id=parent_id,
                document_id=str(parent["document_id"]),
                content=str(child["content"]),
                upstream_rank=rank,
                reranker_score=1 / rank,
                rrf_score=1 / (60 + rank),
                record_id=child_id,
                start_offset=int(child["start_offset"]),
                end_offset=int(child["end_offset"]),
                provenance={"source": str(child["source"])},
            )
            expanded.append(
                ParentExpansionResult(
                    final_rank=rank,
                    parent_id=parent_id,
                    document_id=str(parent["document_id"]),
                    parent_content=str(parent["content"]),
                    best_child_rank=rank,
                    matched_child_count=1,
                    matched_children=(matched,),
                    metadata={
                        "start_offset": int(parent["start_offset"]),
                        "end_offset": int(parent["end_offset"]),
                    },
                    provenance={"source": str(parent["source"])},
                )
            )
        evidence_items = tuple(
            EvidenceItem(
                evidence_id=f"E{item.final_rank}",
                final_rank=item.final_rank,
                parent_id=item.parent_id,
                document_id=item.document_id,
                content=item.parent_content,
                original_content_length=len(item.parent_content),
                included_content_length=len(item.parent_content),
                truncated=False,
                matched_child_count=1,
                best_child_rank=item.best_child_rank,
                matched_children=item.matched_children,
                metadata=item.metadata,
                provenance=item.provenance,
            )
            for item in expanded
        )
        references = tuple(
            EvidenceReference(
                evidence_id=item.evidence_id,
                parent_id=item.parent_id,
                document_id=item.document_id,
                source=str(parents[item.parent_id]["source"]),
                start_offset=int(parents[item.parent_id]["start_offset"]),
                end_offset=int(parents[item.parent_id]["end_offset"]),
            )
            for item in evidence_items
        )
        evidence = EvidenceContext(
            rendered_context="\n\n".join(
                f"[{item.evidence_id}]\n{item.content}" for item in evidence_items
            ),
            evidence_items=evidence_items,
            references=references,
            included_evidence_count=len(evidence_items),
            omitted_evidence_count=0,
            total_original_chars=sum(item.original_content_length for item in evidence_items),
            total_included_chars=sum(item.included_content_length for item in evidence_items),
            truncated=False,
        )
        results.append(
            RetrievalPipelineResult(
                query=query.query,
                dense_results=dense,
                bm25_results=bm25,
                fused_results=fused,
                reranked_child_results=reranked,
                expanded_parent_results=tuple(expanded),
                evidence_context=evidence,
                stage_timings=timing,
                total_runtime_seconds=timing.total_runtime_seconds,
                pipeline_config=RetrievalPipelineConfig(),
            )
        )
    return queries, tuple(results), ground_truth


@pytest.fixture(scope="module")
def evaluated() -> EvaluatedRun:
    return evaluate_retrieval_results(*_fake_results())


def test_query_only_loader_preserves_exact_60_order():
    queries = load_benchmark_query_inputs(QUERY_PATH)
    assert len(queries) == 60
    assert queries[0].query_id == "M2C1-Q001"
    assert queries[-1].query_id == "M2C1-Q060"


def test_query_only_loader_has_56_4_split():
    queries = load_benchmark_query_inputs(QUERY_PATH)
    assert set(queries[0].model_dump()) == {"query_id", "query"}
    ground_truth = load_retrieval_ground_truth(GT_PATH)
    assert sum(item.answerable for item in ground_truth) == 56
    assert sum(not item.answerable for item in ground_truth) == 4


def test_ground_truth_loader_validates_runtime_records():
    ground_truth = load_retrieval_ground_truth(GT_PATH)
    assert len(ground_truth) == 60
    assert all(not item.relevant_child_ids for item in ground_truth if not item.answerable)


def test_fake_pipeline_evaluator_produces_exactly_60_results(evaluated):
    assert len(evaluated.query_results) == 60


@pytest.mark.parametrize("stage", ["dense", "bm25", "rrf", "reranker", "parent", "evidence"])
def test_standard_metrics_use_only_56_answerable_queries(evaluated, stage):
    assert evaluated.analysis["stage_metrics"][stage]["query_count"] == 56
    assert all(
        metric["denominator"] == 56
        for metric in evaluated.analysis["stage_metrics"][stage]["metrics"].values()
    )


def test_unanswerable_queries_are_separate(evaluated):
    assert len(evaluated.analysis["unanswerable_behavior"]) == 4
    for item in evaluated.analysis["unanswerable_behavior"]:
        assert set(item["child_candidate_counts"]) == {"dense", "bm25", "rrf", "reranker"}
        assert item["parent_count"] > 0
        assert item["evidence_count"] > 0
        assert item["top_evidence"]["source"]


def test_reranker_does_not_fabricate_top10_metrics(evaluated):
    metrics = evaluated.analysis["stage_metrics"]["reranker"]["metrics"]
    assert "hit_rate_at_10" not in metrics
    assert "mrr_at_10" not in metrics


def test_dense_bm25_rrf_include_top10_metrics(evaluated):
    for stage in ("dense", "bm25", "rrf"):
        assert "hit_rate_at_10" in evaluated.analysis["stage_metrics"][stage]["metrics"]


def test_parent_and_evidence_use_parent_ground_truth(evaluated):
    assert evaluated.analysis["stage_metrics"]["parent"]["metrics"]["hit_rate_at_1"]["value"] == 1
    assert evaluated.analysis["stage_metrics"]["evidence"]["metrics"]["hit_rate_at_1"]["value"] == 1


def test_overlap_is_not_labeled_as_pure_hard_negative(evaluated):
    labels = {
        item["relevance_label"] for row in evaluated.query_results for item in row["rrf_candidates"]
    }
    assert "overlapping" in labels
    assert evaluated.analysis["overlap_interpretation"].startswith("Overlap chunks")
    for ground_truth in load_retrieval_ground_truth(GT_PATH):
        assert set(ground_truth.overlapping_child_ids) <= set(ground_truth.relevant_child_ids)
        assert not set(ground_truth.overlapping_child_ids) & set(
            ground_truth.hard_negative_child_ids
        )
        assert set(ground_truth.overlapping_parent_ids) <= set(ground_truth.relevant_parent_ids)
        assert not set(ground_truth.overlapping_parent_ids) & set(
            ground_truth.hard_negative_parent_ids
        )


def test_overlap_before_relevant_is_marked_noninformative_structural_zero(evaluated):
    for stage in ("dense", "bm25", "rrf", "reranker", "parent", "evidence"):
        diagnostic = evaluated.analysis["hard_negative_and_overlap"][stage][
            "overlap_before_first_relevant"
        ]
        assert diagnostic["count"] == 0
        assert diagnostic["structural_zero"] is True
        assert diagnostic["informative"] is False


def test_stage_comparisons_partition_all_answerable_queries(evaluated):
    comparison = evaluated.analysis["stage_comparisons"]["reranker_vs_rrf"]
    assert sum(value["count"] for value in comparison["hit_at_1"].values()) == 56


def test_parent_comparison_is_explicitly_cross_granularity(evaluated):
    comparison = evaluated.analysis["stage_comparisons"]["parent_vs_reranker"]
    assert comparison["comparison_scope"] == "cross_granularity_observation"
    assert comparison["before_relevance_granularity"] == "child"
    assert comparison["after_relevance_granularity"] == "parent"
    assert comparison["strict_same_metric_gain"] is False


def test_all_eight_categories_are_reported(evaluated):
    assert len(evaluated.analysis["category_metrics"]) == 8
    for group in evaluated.analysis["category_metrics"].values():
        assert set(group["top1_exposure"]) == {"rrf", "reranker", "parent"}


def test_all_five_query_types_are_reported(evaluated):
    assert len(evaluated.analysis["query_type_metrics"]) == 5
    assert evaluated.analysis["query_type_metrics"]["unanswerable"]["answerable_query_count"] == 0


def test_failure_counts_include_zero_occurrence_types(evaluated):
    assert "rrf_no_relevant_at_5" in evaluated.analysis["failure_case_counts"]
    assert evaluated.analysis["failure_case_counts"]["rrf_no_relevant_at_5"] == 0


def test_formal_runner_declares_real_models_and_offline_gate():
    script = (ROOT / "scripts/evaluate_enterprise_retrieval_baseline.py").read_text(
        encoding="utf-8"
    )
    assert "BAAI/bge-small-zh-v1.5" in script
    assert "BAAI/bge-reranker-base" in script
    assert 'os.environ.get("HF_HUB_OFFLINE") != "1"' in script
    assert '"same_python_process": True' in script
    assert '"run_b_is_independent_cold_start": False' in script
    assert "Fake" not in script


def test_ranking_digest_is_stable(evaluated):
    assert build_ranking_digest(evaluated.query_results) == evaluated.ranking_digest


def test_ranking_digest_detects_candidate_order_change(evaluated):
    changed = [dict(row) for row in evaluated.query_results]
    changed[0] = dict(changed[0])
    changed[0]["dense_candidates"] = list(reversed(changed[0]["dense_candidates"]))
    assert build_ranking_digest(changed) != evaluated.ranking_digest


def test_deterministic_run_comparison_accepts_identical(evaluated):
    assert_deterministic_runs(evaluated, evaluated)


def test_deterministic_run_comparison_rejects_digest_change(evaluated):
    changed = EvaluatedRun(
        evaluated.query_results,
        evaluated.failure_cases,
        evaluated.analysis,
        "0" * 64,
        evaluated.deterministic_digest,
    )
    with pytest.raises(EvaluationValidationError):
        assert_deterministic_runs(evaluated, changed)


def test_timing_summary_has_linear_p95():
    summary = timing_summary([0.0, 1.0, 2.0])
    assert summary["p95"] == pytest.approx(1.9)
    assert summary["median"] == 1.0


def test_timing_summary_rejects_invalid_values():
    with pytest.raises(EvaluationValidationError):
        timing_summary([])


def test_stable_json_contains_no_implicit_timestamp_or_path():
    text = stable_json({"schema_version": "1.0", "value": 1})
    assert "timestamp" not in text
    assert str(ROOT) not in text


def test_sha256_text_is_byte_stable():
    assert sha256_text("a\n") == sha256_text("a\n")
    assert sha256_text("a\n") != sha256_text("a\r\n")


def test_atomic_artifact_writer_emits_four_files_and_hashes(tmp_path, evaluated):
    hashes = write_baseline_artifacts(
        output_dir=tmp_path,
        main_report={"schema_version": "1.0"},
        query_results=evaluated.query_results,
        failure_cases=evaluated.failure_cases,
        runtime_profile={"measurement_scope": "local CPU"},
    )
    assert len(hashes) == 4
    assert len(list(tmp_path.iterdir())) == 4
    report = json.loads((tmp_path / "m2c2a2_retrieval_baseline.json").read_text("utf-8"))
    assert set(report["generated_file_hashes"]) == {
        "m2c2a2_query_results.jsonl",
        "m2c2a2_failure_cases.jsonl",
    }
    assert report["artifact_hash_semantics"]["self_hash_included"] is False
    assert (
        "m2c2a2_query_results.jsonl"
        in report["artifact_hash_semantics"]["runtime_dependent_artifacts"]
    )


def test_query_results_are_independently_metric_recalculable(evaluated):
    first = evaluated.query_results[0]
    assert first["ground_truth_ids"]["relevant_child_ids"]
    assert first["first_relevant_ranks"]["rrf"] == 1


def test_replacing_ground_truth_labels_does_not_change_candidate_order():
    queries, results, ground_truth = _fake_results()
    first = ground_truth[0].model_dump(mode="python")
    first.update(
        {
            "relevant_child_ids": (results[0].dense_results[-1].candidate_id,),
            "hard_negative_child_ids": (),
            "overlapping_child_ids": (),
            "relevant_parent_ids": (results[0].expanded_parent_results[-1].parent_id,),
            "hard_negative_parent_ids": (),
            "overlapping_parent_ids": (),
        }
    )
    changed_ground_truth = (
        type(ground_truth[0]).model_validate(first),
        *ground_truth[1:],
    )
    original = evaluate_retrieval_results(queries, results, ground_truth)
    relabeled = evaluate_retrieval_results(queries, results, changed_ground_truth)
    assert relabeled.ranking_digest == original.ranking_digest
    assert relabeled.analysis["stage_metrics"] != original.analysis["stage_metrics"]


def test_query_text_mismatch_fails_fast():
    queries, results, ground_truth = _fake_results()
    changed = list(results)
    changed[0] = changed[0].model_copy(update={"query": "changed"})
    with pytest.raises(EvaluationValidationError, match="query order"):
        evaluate_retrieval_results(queries, changed, ground_truth)


def test_formal_evaluator_rejects_non_60_input():
    queries, results, ground_truth = _fake_results()
    with pytest.raises(EvaluationValidationError, match="exactly 60"):
        evaluate_retrieval_results(queries[:-1], results[:-1], ground_truth[:-1])
