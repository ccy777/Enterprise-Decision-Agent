"""Offline tests for the versioned Dense/BM25 RRF baseline."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import decision_agent.evaluation.rrf_baseline as rrf_module
from decision_agent.evaluation.dataset import compute_normalized_text_sha256
from decision_agent.evaluation.retrieval_metrics import aggregate_metrics, compute_query_metrics
from decision_agent.evaluation.rrf_baseline import (
    RRFBaselineConfig,
    RRFBaselineReport,
    run_rrf_baseline,
)
from decision_agent.exceptions import EvaluationValidationError

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DENSE_REPORT = REPOSITORY_ROOT / "artifacts/evaluation/m2b2c_dense_bge_small_zh_v1_5.json"
BM25_REPORT = REPOSITORY_ROOT / "artifacts/evaluation/m2b3_bm25_baseline.json"
VERSIONED_RRF_REPORT = REPOSITORY_ROOT / "artifacts/evaluation/m2b4_rrf_dense_bm25_k60.json"


def make_config(tmp_path: Path, **updates: Any) -> RRFBaselineConfig:
    values: dict[str, Any] = {
        "dense_report_path": DENSE_REPORT,
        "bm25_report_path": BM25_REPORT,
        "output_path": tmp_path / "rrf.json",
    }
    values.update(updates)
    return RRFBaselineConfig(**values)


def write_mutated_report(
    source: Path,
    destination: Path,
    mutate: Callable[[dict[str, Any]], None],
) -> Path:
    payload = json.loads(source.read_text(encoding="utf-8"))
    mutate(payload)
    destination.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return destination


def load_cli() -> object:
    path = REPOSITORY_ROOT / "scripts/run_rrf_retrieval_baseline.py"
    spec = spec_from_file_location("run_rrf_retrieval_baseline_test", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("field", "value"),
    [("rrf_k", 61.0), ("dense_weight", 2.0), ("bm25_weight", 2.0)],
)
def test_versioned_config_rejects_tuned_parameters(
    tmp_path: Path, field: str, value: float
) -> None:
    with pytest.raises(ValidationError):
        make_config(tmp_path, **{field: value})


def test_runner_calls_production_rrf_core_for_every_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_fusion = rrf_module.reciprocal_rank_fusion
    calls: list[tuple[float, dict[str, float], int | None]] = []

    def observed_fusion(*args: Any, **kwargs: Any):
        calls.append((kwargs["rrf_k"], dict(kwargs["source_weights"]), kwargs["top_k"]))
        return real_fusion(*args, **kwargs)

    monkeypatch.setattr(rrf_module, "reciprocal_rank_fusion", observed_fusion)
    report = run_rrf_baseline(make_config(tmp_path))
    assert len(calls) == report.query_count == 18
    assert all(call == (60.0, {"dense": 1.0, "bm25": 1.0}, 5) for call in calls)


def test_report_records_fixed_algorithm_and_source_identity(tmp_path: Path) -> None:
    report = run_rrf_baseline(make_config(tmp_path))
    assert report.retrieval_type == "rrf"
    assert report.algorithm == "reciprocal_rank_fusion"
    assert report.rrf_k == 60
    assert report.source_names == ("dense", "bm25")
    assert report.source_weights == {"dense": 1.0, "bm25": 1.0}
    assert report.corpus_size == 36
    assert report.query_count == 18
    assert report.fusion_input_depth == report.output_top_k == 5


def test_source_reports_have_relative_paths_and_normalized_hashes(tmp_path: Path) -> None:
    report = run_rrf_baseline(make_config(tmp_path))
    by_name = {item.source_name: item for item in report.source_reports}
    assert by_name["dense"].relative_path == (
        "artifacts/evaluation/m2b2c_dense_bge_small_zh_v1_5.json"
    )
    assert by_name["bm25"].relative_path == "artifacts/evaluation/m2b3_bm25_baseline.json"
    assert by_name["dense"].normalized_text_sha256 == compute_normalized_text_sha256(DENSE_REPORT)
    assert by_name["bm25"].normalized_text_sha256 == compute_normalized_text_sha256(BM25_REPORT)


def test_query_source_contributions_preserve_scores_but_use_rank_formula(tmp_path: Path) -> None:
    report = run_rrf_baseline(make_config(tmp_path))
    merged = next(
        item for item in report.query_results[0].ranked_results if item.matched_source_count == 2
    )
    assert all(item.source_score is not None for item in merged.source_contributions)
    assert all(
        item.contribution == pytest.approx(1 / (60 + item.source_rank))
        for item in merged.source_contributions
    )
    assert merged.fused_score == pytest.approx(
        math.fsum(item.contribution for item in merged.source_contributions)
    )


def test_query_and_aggregate_metrics_are_independently_recomputable(tmp_path: Path) -> None:
    report = run_rrf_baseline(make_config(tmp_path))
    recomputed: list[tuple[str, dict[str, float]]] = []
    for result in report.query_results:
        metrics = compute_query_metrics(
            result.ranked_document_ids,
            result.relevant_document_ids,
            ks=(1, 3, 5),
            mrr_k=5,
        )
        assert result.metrics == metrics
        recomputed.append((result.category, metrics))
    overall, categories = aggregate_metrics(recomputed)
    assert report.metrics == overall
    assert report.category_metrics == categories


def test_comparison_groups_partition_all_queries(tmp_path: Path) -> None:
    report = run_rrf_baseline(make_config(tmp_path))
    groups = (
        report.comparison.improved_query_ids,
        report.comparison.regressed_query_ids,
        report.comparison.unchanged_query_ids,
    )
    flattened = [query_id for group in groups for query_id in group]
    assert len(flattened) == report.query_count
    assert len(flattened) == len(set(flattened))


def test_two_named_hard_negatives_record_actual_source_and_fused_ranks(tmp_path: Path) -> None:
    report = run_rrf_baseline(make_config(tmp_path))
    by_id = {result.query_id: result for result in report.query_results}
    warranty = by_id["query-product-a-battery-warranty"]
    inventory = by_id["query-inventory-product-a"]
    assert (warranty.dense_best_relevant_rank, warranty.bm25_best_relevant_rank) == (3, 2)
    assert warranty.rrf_best_relevant_rank == 2
    assert warranty.comparison_to_dense == "improved"
    assert (inventory.dense_best_relevant_rank, inventory.bm25_best_relevant_rank) == (1, 2)
    assert inventory.rrf_best_relevant_rank == 1
    assert inventory.comparison_to_dense == "unchanged"


def test_report_contains_no_absolute_paths_or_user_environment(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    run_rrf_baseline(config)
    text = config.output_path.read_text(encoding="utf-8").lower()
    assert str(REPOSITORY_ROOT).lower() not in text
    assert str(tmp_path).lower() not in text
    assert "lenovo" not in text
    assert "proxy" not in text


def test_historical_reports_are_not_modified(tmp_path: Path) -> None:
    dense_before = DENSE_REPORT.read_bytes()
    bm25_before = BM25_REPORT.read_bytes()
    run_rrf_baseline(make_config(tmp_path))
    assert DENSE_REPORT.read_bytes() == dense_before
    assert BM25_REPORT.read_bytes() == bm25_before


def test_missing_source_report_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(EvaluationValidationError, match="does not exist"):
        run_rrf_baseline(make_config(tmp_path, dense_report_path=tmp_path / "missing.json"))


@pytest.mark.parametrize("field", ["corpus_sha256", "queries_sha256"])
def test_source_dataset_hash_mismatch_is_rejected(tmp_path: Path, field: str) -> None:
    mutated = write_mutated_report(
        BM25_REPORT,
        tmp_path / "bm25.json",
        lambda payload: payload.__setitem__(field, "0" * 64),
    )
    with pytest.raises(EvaluationValidationError, match="SHA-256"):
        run_rrf_baseline(make_config(tmp_path, bm25_report_path=mutated))


def test_query_id_set_mismatch_is_rejected(tmp_path: Path) -> None:
    def mutate(payload: dict[str, Any]) -> None:
        payload["query_results"][0]["query_id"] = "replacement-query-id"

    mutated = write_mutated_report(DENSE_REPORT, tmp_path / "dense.json", mutate)
    with pytest.raises(EvaluationValidationError, match="query ID"):
        run_rrf_baseline(make_config(tmp_path, dense_report_path=mutated))


def test_duplicate_query_ids_are_rejected_before_mapping() -> None:
    dense = rrf_module._load_report(DENSE_REPORT, rrf_module.DenseBaselineReport)
    bm25 = rrf_module._load_report(BM25_REPORT, rrf_module.BM25BaselineReport)
    duplicate_id = dense.query_results[0].query_id
    dense_results = list(dense.query_results)
    bm25_results = list(bm25.query_results)
    dense_results[1] = dense_results[1].model_copy(update={"query_id": duplicate_id})
    bm25_results[1] = bm25_results[1].model_copy(update={"query_id": duplicate_id})
    dense = dense.model_copy(update={"query_results": tuple(dense_results)})
    bm25 = bm25.model_copy(update={"query_results": tuple(bm25_results)})
    with pytest.raises(EvaluationValidationError, match="duplicate query IDs"):
        rrf_module._validate_source_compatibility(dense, bm25, input_depth=5)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("query", "different query", "query text"),
        ("category", "different category", "query category"),
    ],
)
def test_query_text_or_category_mismatch_is_rejected(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    def mutate(payload: dict[str, Any]) -> None:
        payload["query_results"][0][field] = value

    mutated = write_mutated_report(BM25_REPORT, tmp_path / "bm25.json", mutate)
    with pytest.raises(EvaluationValidationError, match=message):
        run_rrf_baseline(make_config(tmp_path, bm25_report_path=mutated))


def test_relevance_mismatch_is_rejected_without_overwriting_labels(tmp_path: Path) -> None:
    def mutate(payload: dict[str, Any]) -> None:
        result = next(
            item for item in payload["query_results"] if len(item["relevant_document_ids"]) > 1
        )
        result["relevant_document_ids"] = list(reversed(result["relevant_document_ids"]))

    mutated = write_mutated_report(BM25_REPORT, tmp_path / "bm25.json", mutate)
    with pytest.raises(EvaluationValidationError, match="relevance labels"):
        run_rrf_baseline(make_config(tmp_path, bm25_report_path=mutated))


def test_invalid_source_rank_is_rejected_as_malformed_report(tmp_path: Path) -> None:
    def mutate(payload: dict[str, Any]) -> None:
        payload["query_results"][0]["ranked_results"][0]["rank"] = 2

    mutated = write_mutated_report(DENSE_REPORT, tmp_path / "dense.json", mutate)
    with pytest.raises(EvaluationValidationError, match="failed to read validated"):
        run_rrf_baseline(make_config(tmp_path, dense_report_path=mutated))


def test_non_finite_source_score_is_rejected_as_malformed_report(tmp_path: Path) -> None:
    def mutate(payload: dict[str, Any]) -> None:
        payload["query_results"][0]["ranked_results"][0]["score"] = math.inf

    mutated = write_mutated_report(DENSE_REPORT, tmp_path / "dense.json", mutate)
    with pytest.raises(EvaluationValidationError, match="failed to read validated"):
        run_rrf_baseline(make_config(tmp_path, dense_report_path=mutated))


def test_fusion_does_not_construct_embedding_or_milvus_clients(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import decision_agent.retrieval.embeddings as embeddings
    import decision_agent.retrieval.milvus_store as milvus_store

    monkeypatch.setattr(
        embeddings,
        "_default_sentence_transformer_factory",
        lambda **kwargs: pytest.fail("RRF must not construct an embedding model"),
    )
    monkeypatch.setattr(
        milvus_store,
        "MilvusClient",
        lambda **kwargs: pytest.fail("RRF must not construct a Milvus client"),
    )
    run_rrf_baseline(make_config(tmp_path))


def test_cli_defaults_use_versioned_reports_and_import_has_no_side_effect() -> None:
    before = VERSIONED_RRF_REPORT.read_bytes() if VERSIONED_RRF_REPORT.exists() else None
    cli = load_cli()
    args = cli.build_parser().parse_args([])
    assert args.dense_report == DENSE_REPORT
    assert args.bm25_report == BM25_REPORT
    assert args.output == VERSIONED_RRF_REPORT
    after = VERSIONED_RRF_REPORT.read_bytes() if VERSIONED_RRF_REPORT.exists() else None
    assert after == before


def test_versioned_report_is_independently_recomputable_when_present() -> None:
    if not VERSIONED_RRF_REPORT.exists():
        pytest.skip("versioned RRF report is generated later in this task")
    report = RRFBaselineReport.model_validate_json(VERSIONED_RRF_REPORT.read_text(encoding="utf-8"))
    recomputed = [
        (
            result.category,
            compute_query_metrics(
                result.ranked_document_ids,
                result.relevant_document_ids,
                ks=(1, 3, 5),
                mrr_k=5,
            ),
        )
        for result in report.query_results
    ]
    overall, categories = aggregate_metrics(recomputed)
    assert report.metrics == overall
    assert report.category_metrics == categories


def test_report_model_rejects_tampered_contribution_summary() -> None:
    payload = json.loads(VERSIONED_RRF_REPORT.read_text(encoding="utf-8"))
    payload["query_results"][0]["ranked_results"][0]["source_contributions"][0]["contribution"] *= 2
    with pytest.raises(ValidationError, match=r"fused_score|RRF formula"):
        RRFBaselineReport.model_validate(payload)


@pytest.mark.parametrize("path", ["C:/secret/report.json", "../report.json", "a\\report.json"])
def test_source_report_model_rejects_unsafe_paths(path: str) -> None:
    payload = json.loads(VERSIONED_RRF_REPORT.read_text(encoding="utf-8"))
    payload["source_reports"][0]["relative_path"] = path
    with pytest.raises(ValidationError, match="relative path"):
        RRFBaselineReport.model_validate(payload)
