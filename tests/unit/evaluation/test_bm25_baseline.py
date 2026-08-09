"""Offline tests for the BM25 baseline runner and versioned report."""

from __future__ import annotations

import math
import os
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

import decision_agent.evaluation.bm25_baseline as bm25_baseline
import decision_agent.evaluation.dataset as dataset_module
import decision_agent.evaluation.reporting as reporting_module
from decision_agent.evaluation.bm25_baseline import (
    BM25BaselineConfig,
    BM25BaselineReport,
    run_bm25_baseline,
)
from decision_agent.evaluation.retrieval_metrics import compute_query_metrics
from decision_agent.exceptions import EvaluationValidationError

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CORPUS_PATH = REPOSITORY_ROOT / "datasets/retrieval/m2b2c_dense_corpus.jsonl"
QUERIES_PATH = REPOSITORY_ROOT / "datasets/retrieval/m2b2c_dense_queries.jsonl"
DENSE_REPORT_PATH = REPOSITORY_ROOT / "artifacts/evaluation/m2b2c_dense_bge_small_zh_v1_5.json"


def make_config(tmp_path: Path) -> BM25BaselineConfig:
    return BM25BaselineConfig(
        corpus_path=CORPUS_PATH,
        queries_path=QUERIES_PATH,
        dense_report_path=DENSE_REPORT_PATH,
        output_path=tmp_path / "bm25-report.json",
    )


def load_cli() -> object:
    script_path = REPOSITORY_ROOT / "scripts/run_bm25_retrieval_baseline.py"
    spec = spec_from_file_location("run_bm25_retrieval_baseline_test", script_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_uses_versioned_dataset_without_embedding_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import decision_agent.retrieval.embeddings as embeddings

    monkeypatch.setattr(
        embeddings,
        "_default_sentence_transformer_factory",
        lambda **kwargs: pytest.fail("BM25 must not construct an embedding model"),
    )
    report = run_bm25_baseline(make_config(tmp_path))
    assert report.corpus_size == 36
    assert report.query_count == 18


def test_report_has_required_sparse_baseline_structure(tmp_path: Path) -> None:
    payload = run_bm25_baseline(make_config(tmp_path)).model_dump(mode="json")
    assert set(payload) == {
        "evaluation_schema_version",
        "baseline_name",
        "retrieval_type",
        "dataset_name",
        "corpus_file",
        "queries_file",
        "corpus_sha256",
        "queries_sha256",
        "tokenizer_name",
        "tokenizer_version",
        "tokenizer_config",
        "k1",
        "b",
        "top_k",
        "corpus_size",
        "query_count",
        "index_observations",
        "metrics",
        "category_metrics",
        "query_results",
        "dense_comparison",
        "runtime_observations",
        "python_version",
        "limitations",
    }


def test_report_uses_existing_normalized_dataset_hashes(tmp_path: Path) -> None:
    report = run_bm25_baseline(make_config(tmp_path))
    assert report.corpus_sha256 == dataset_module.compute_normalized_text_sha256(CORPUS_PATH)
    assert report.queries_sha256 == dataset_module.compute_normalized_text_sha256(QUERIES_PATH)


def test_query_results_use_document_id_for_relevance(tmp_path: Path) -> None:
    report = run_bm25_baseline(make_config(tmp_path))
    for result in report.query_results:
        assert all(item.record_id != item.document_id for item in result.ranked_results)
        assert all(
            item.is_relevant == (item.document_id in result.relevant_document_ids)
            for item in result.ranked_results
        )


def test_query_ranks_are_consecutive_and_scores_are_finite(tmp_path: Path) -> None:
    report = run_bm25_baseline(make_config(tmp_path))
    for result in report.query_results:
        assert [item.rank for item in result.ranked_results] == list(
            range(1, len(result.ranked_results) + 1)
        )
        assert all(math.isfinite(item.score) and item.score > 0 for item in result.ranked_results)


def test_query_metrics_reuse_existing_metric_contract(tmp_path: Path) -> None:
    report = run_bm25_baseline(make_config(tmp_path))
    for result in report.query_results:
        assert result.metrics == compute_query_metrics(
            result.ranked_document_ids,
            result.relevant_document_ids,
            ks=(1, 3, 5),
            mrr_k=5,
        )


def test_dense_comparison_partitions_every_query_once(tmp_path: Path) -> None:
    report = run_bm25_baseline(make_config(tmp_path))
    comparison = report.dense_comparison
    groups = (
        comparison.dense_better_query_ids,
        comparison.bm25_better_query_ids,
        comparison.both_top1_correct_query_ids,
        comparison.both_top1_failed_query_ids,
    )
    flattened = [query_id for group in groups for query_id in group]
    assert len(flattened) == report.query_count
    assert len(flattened) == len(set(flattened))


def test_warranty_fix_flag_is_derived_from_actual_rankings(tmp_path: Path) -> None:
    report = run_bm25_baseline(make_config(tmp_path))
    query = next(
        result
        for result in report.query_results
        if result.query_id == "query-product-a-battery-warranty"
    )
    dense_hit_at_one = report.dense_comparison.dense_metrics["hit_rate_at_1"] < 1.0
    assert report.dense_comparison.product_a_battery_warranty_fixed == (
        dense_hit_at_one and query.metrics["hit_rate_at_1"] == 1.0
    )


def test_report_contains_no_absolute_paths(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    run_bm25_baseline(config)
    text = config.output_path.read_text(encoding="utf-8")
    assert str(REPOSITORY_ROOT) not in text
    assert str(tmp_path) not in text


def test_dense_report_is_never_modified(tmp_path: Path) -> None:
    before = DENSE_REPORT_PATH.read_bytes()
    run_bm25_baseline(make_config(tmp_path))
    assert DENSE_REPORT_PATH.read_bytes() == before


def test_missing_input_fails_before_index_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path).model_copy(update={"corpus_path": tmp_path / "missing.jsonl"})
    monkeypatch.setattr(
        bm25_baseline,
        "BM25Index",
        lambda *args, **kwargs: pytest.fail("index must not be built before input validation"),
    )
    with pytest.raises(EvaluationValidationError, match="corpus file does not exist"):
        run_bm25_baseline(config)


def test_runner_uses_public_dataset_hash_function(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[Path, str]] = []
    real_hash = dataset_module.compute_normalized_text_sha256

    def observed_hash(path: Path, *, dataset_name: str) -> str:
        calls.append((path, dataset_name))
        return real_hash(path, dataset_name=dataset_name)

    monkeypatch.setattr(bm25_baseline, "compute_normalized_text_sha256", observed_hash)
    run_bm25_baseline(make_config(tmp_path))
    assert calls == [
        (CORPUS_PATH, "corpus"),
        (QUERIES_PATH, "queries"),
        (CORPUS_PATH, "corpus"),
        (QUERIES_PATH, "queries"),
    ]


def test_atomic_report_write_preserves_existing_file_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path)
    config.output_path.write_text("existing\n", encoding="utf-8")

    def fail_replace(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        raise OSError(f"replace failed for {Path(target).name}")

    monkeypatch.setattr(reporting_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        run_bm25_baseline(config)
    assert config.output_path.read_text(encoding="utf-8") == "existing\n"
    assert list(tmp_path.glob(f".{config.output_path.name}.*.tmp")) == []


def test_report_model_rejects_non_finite_metrics(tmp_path: Path) -> None:
    report = run_bm25_baseline(make_config(tmp_path))
    payload = report.model_dump(mode="python")
    payload["metrics"]["hit_rate_at_1"] = math.nan
    with pytest.raises(ValueError, match="metric"):
        BM25BaselineReport.model_validate(payload)


def test_cli_defaults_use_versioned_repository_paths() -> None:
    cli = load_cli()
    args = cli.build_parser().parse_args([])
    assert args.corpus == CORPUS_PATH
    assert args.queries == QUERIES_PATH
    assert args.dense_report == DENSE_REPORT_PATH
    assert args.output == REPOSITORY_ROOT / "artifacts/evaluation/m2b3_bm25_baseline.json"


def test_cli_import_does_not_execute_baseline(tmp_path: Path) -> None:
    cli = load_cli()
    assert hasattr(cli, "main")
    assert not (tmp_path / "bm25-report.json").exists()


def test_versioned_report_can_be_independently_recomputed_when_present() -> None:
    report_path = REPOSITORY_ROOT / "artifacts/evaluation/m2b3_bm25_baseline.json"
    if not report_path.exists():
        pytest.skip("versioned BM25 report is generated later in this task")
    report = BM25BaselineReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    assert report.corpus_sha256 == dataset_module.compute_normalized_text_sha256(CORPUS_PATH)
    assert report.queries_sha256 == dataset_module.compute_normalized_text_sha256(QUERIES_PATH)
    for result in report.query_results:
        assert result.metrics == compute_query_metrics(
            result.ranked_document_ids,
            result.relevant_document_ids,
            ks=(1, 3, 5),
            mrr_k=5,
        )
