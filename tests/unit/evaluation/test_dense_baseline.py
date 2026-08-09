"""Offline tests for dense baseline orchestration and reports."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

import decision_agent.evaluation.dataset as dataset_module
import decision_agent.evaluation.dense_baseline as dense_baseline
import decision_agent.evaluation.reporting as reporting_module
from decision_agent.evaluation.dense_baseline import (
    DenseBaselineConfig,
    DenseBaselineReport,
    run_dense_baseline,
)
from decision_agent.exceptions import EvaluationValidationError, RetrievalValidationError
from decision_agent.retrieval import DeterministicHashEmbeddingProvider


def write_dataset(corpus_path: Path, queries_path: Path) -> None:
    corpus = [
        {
            "document_id": "sales-east",
            "text": "华东区域产品销售下降",
            "category": "销售",
            "source": "synthetic-enterprise-baseline",
        },
        {
            "document_id": "leave-policy",
            "text": "员工年假制度",
            "category": "人力资源",
            "source": "synthetic-enterprise-baseline",
        },
    ]
    queries = [
        {
            "query_id": "query-sales",
            "query": "华东销售",
            "relevant_document_ids": ["sales-east"],
            "category": "销售",
        }
    ]
    corpus_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in corpus), encoding="utf-8"
    )
    queries_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in queries), encoding="utf-8"
    )


class ObservableProvider(DeterministicHashEmbeddingProvider):
    def __init__(self, *, dimension: int) -> None:
        super().__init__(dimension=dimension)
        self.document_calls = 0
        self.query_calls = 0

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_calls += 1
        return await super().embed_documents(texts)

    async def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return await super().embed_query(text)


class DatasetMutatingProvider(DeterministicHashEmbeddingProvider):
    def __init__(self, *, dimension: int, corpus_path: Path) -> None:
        super().__init__(dimension=dimension)
        self.corpus_path = corpus_path

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = await super().embed_documents(texts)
        self.corpus_path.write_text(
            self.corpus_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
        )
        return vectors


def make_config(tmp_path: Path, *, top_k: int = 2, dimension: int = 128) -> DenseBaselineConfig:
    corpus_path = tmp_path / "corpus.jsonl"
    queries_path = tmp_path / "queries.jsonl"
    write_dataset(corpus_path, queries_path)
    return DenseBaselineConfig(
        corpus_path=corpus_path,
        queries_path=queries_path,
        output_path=tmp_path / "report.json",
        model_name="offline/test-provider",
        dimension=dimension,
        device="cpu",
        top_k=top_k,
        local_files_only=True,
    )


def load_baseline_cli() -> object:
    script_path = Path(__file__).resolve().parents[3] / "scripts/run_dense_retrieval_baseline.py"
    spec = spec_from_file_location("run_dense_retrieval_baseline_test", script_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_runner_uses_project_dense_indexer_and_retriever_chain(tmp_path: Path) -> None:
    provider = ObservableProvider(dimension=128)

    report = await run_dense_baseline(make_config(tmp_path), embedding_provider=provider)

    assert provider.document_calls == 1
    assert provider.query_calls == 2
    assert report.query_results[0].ranked_document_ids[0] == "sales-east"


@pytest.mark.asyncio
async def test_report_has_stable_required_structure(tmp_path: Path) -> None:
    report = await run_dense_baseline(
        make_config(tmp_path),
        embedding_provider=DeterministicHashEmbeddingProvider(dimension=128),
    )

    payload = report.model_dump(mode="json")
    assert set(payload) == {
        "evaluation_schema_version",
        "baseline_name",
        "dataset_name",
        "corpus_file",
        "queries_file",
        "corpus_sha256",
        "queries_sha256",
        "model_name",
        "embedding_dimension",
        "device",
        "normalize_embeddings",
        "query_instruction",
        "top_k",
        "corpus_size",
        "query_count",
        "metrics",
        "category_metrics",
        "query_results",
        "runtime_observations",
        "dependency_versions",
        "limitations",
    }


@pytest.mark.asyncio
async def test_report_records_dataset_hashes_without_absolute_paths(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    report = await run_dense_baseline(
        config, embedding_provider=DeterministicHashEmbeddingProvider(dimension=128)
    )

    assert report.evaluation_schema_version == "1.0"
    assert report.dataset_name == "m2b2c-dense-v1"
    assert report.corpus_file == "corpus.jsonl"
    assert report.queries_file == "queries.jsonl"
    assert report.corpus_sha256 == dataset_module.compute_normalized_text_sha256(config.corpus_path)
    assert report.queries_sha256 == dataset_module.compute_normalized_text_sha256(
        config.queries_path
    )
    assert report.top_k == config.top_k
    assert str(tmp_path) not in config.output_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_query_results_have_explicit_consecutive_ranks_and_relevance(tmp_path: Path) -> None:
    report = await run_dense_baseline(
        make_config(tmp_path),
        embedding_provider=DeterministicHashEmbeddingProvider(dimension=128),
    )

    result = report.query_results[0]
    assert [item.rank for item in result.ranked_results] == list(
        range(1, len(result.ranked_results) + 1)
    )
    assert len({item.document_id for item in result.ranked_results}) == len(result.ranked_results)
    assert all(
        item.is_relevant == (item.document_id in result.relevant_document_ids)
        for item in result.ranked_results
    )
    assert result.ranked_document_ids == tuple(item.document_id for item in result.ranked_results)


@pytest.mark.asyncio
async def test_report_metrics_can_be_recomputed_without_production_metric_helpers(
    tmp_path: Path,
) -> None:
    report = await run_dense_baseline(
        make_config(tmp_path),
        embedding_provider=DeterministicHashEmbeddingProvider(dimension=128),
    )

    result = report.query_results[0]
    ranked = result.ranked_document_ids
    relevant = set(result.relevant_document_ids)
    expected = {
        "hit_rate_at_1": float(any(item in relevant for item in ranked[:1])),
        "recall_at_1": len(set(ranked[:1]) & relevant) / len(relevant),
        "hit_rate_at_3": float(any(item in relevant for item in ranked[:3])),
        "recall_at_3": len(set(ranked[:3]) & relevant) / len(relevant),
        "hit_rate_at_5": float(any(item in relevant for item in ranked[:5])),
        "recall_at_5": len(set(ranked[:5]) & relevant) / len(relevant),
        "mrr_at_5": next(
            (1.0 / rank for rank, item in enumerate(ranked[:5], start=1) if item in relevant),
            0.0,
        ),
    }
    assert result.metrics == expected


@pytest.mark.asyncio
async def test_written_report_omits_absolute_cache_path(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config = config.model_copy(update={"cache_folder": Path("C:/sensitive/user/cache")})

    await run_dense_baseline(
        config, embedding_provider=DeterministicHashEmbeddingProvider(dimension=128)
    )

    report_text = config.output_path.read_text(encoding="utf-8")
    assert "sensitive" not in report_text
    assert "cache_folder" not in report_text


@pytest.mark.asyncio
async def test_atomic_report_write_preserves_existing_file_when_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path)
    config.output_path.write_text("existing-report\n", encoding="utf-8")

    def fail_replace(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        raise OSError(f"replace failed for {Path(target).name}")

    monkeypatch.setattr(reporting_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        await run_dense_baseline(
            config, embedding_provider=DeterministicHashEmbeddingProvider(dimension=128)
        )

    assert config.output_path.read_text(encoding="utf-8") == "existing-report\n"
    assert list(tmp_path.glob(f".{config.output_path.name}.*.tmp")) == []


@pytest.mark.asyncio
async def test_dataset_change_during_run_is_rejected_before_report_write(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    provider = DatasetMutatingProvider(dimension=128, corpus_path=config.corpus_path)

    with pytest.raises(EvaluationValidationError, match="changed during baseline"):
        await run_dense_baseline(config, embedding_provider=provider)

    assert not config.output_path.exists()


@pytest.mark.asyncio
async def test_report_generation_uses_public_normalized_text_hash_function(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path)
    calls: list[tuple[Path, str]] = []

    def fake_hash(path: Path, *, dataset_name: str) -> str:
        calls.append((path, dataset_name))
        return "a" * 64 if dataset_name == "corpus" else "b" * 64

    monkeypatch.setattr(
        dense_baseline,
        "compute_normalized_text_sha256",
        fake_hash,
    )

    report = await run_dense_baseline(
        config, embedding_provider=DeterministicHashEmbeddingProvider(dimension=128)
    )

    assert report.corpus_sha256 == "a" * 64
    assert report.queries_sha256 == "b" * 64
    assert calls == [
        (config.corpus_path, "corpus"),
        (config.queries_path, "queries"),
        (config.corpus_path, "corpus"),
        (config.queries_path, "queries"),
    ]


@pytest.mark.asyncio
async def test_report_metrics_are_bounded(tmp_path: Path) -> None:
    report = await run_dense_baseline(
        make_config(tmp_path),
        embedding_provider=DeterministicHashEmbeddingProvider(dimension=128),
    )
    values = list(report.metrics.values())
    values.extend(
        value for metrics in report.category_metrics.values() for value in metrics.values()
    )
    assert all(0.0 <= value <= 1.0 for value in values)


@pytest.mark.asyncio
async def test_report_model_rejects_non_finite_or_out_of_range_metrics(tmp_path: Path) -> None:
    report = await run_dense_baseline(
        make_config(tmp_path),
        embedding_provider=DeterministicHashEmbeddingProvider(dimension=128),
    )
    payload = report.model_dump(mode="python")
    payload["metrics"]["hit_rate_at_1"] = float("nan")

    with pytest.raises(ValueError, match="metric values"):
        DenseBaselineReport.model_validate(payload)


@pytest.mark.asyncio
async def test_query_result_count_matches_dataset(tmp_path: Path) -> None:
    report = await run_dense_baseline(
        make_config(tmp_path),
        embedding_provider=DeterministicHashEmbeddingProvider(dimension=128),
    )
    assert report.query_count == 1
    assert len(report.query_results) == 1
    assert report.runtime_observations.average_query_ms == pytest.approx(
        report.runtime_observations.total_query_seconds * 1000 / report.query_count
    )


@pytest.mark.asyncio
async def test_top_k_never_exceeds_configuration(tmp_path: Path) -> None:
    report = await run_dense_baseline(
        make_config(tmp_path, top_k=1),
        embedding_provider=DeterministicHashEmbeddingProvider(dimension=128),
    )
    assert len(report.query_results[0].ranked_document_ids) == 1


@pytest.mark.asyncio
async def test_provider_configuration_dimension_mismatch_fails_before_indexing(
    tmp_path: Path,
) -> None:
    provider = ObservableProvider(dimension=64)
    with pytest.raises(RetrievalValidationError, match="dimension"):
        await run_dense_baseline(make_config(tmp_path, dimension=128), embedding_provider=provider)
    assert provider.document_calls == 0


def test_config_rejects_non_positive_top_k(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="greater than 0"):
        make_config(tmp_path, top_k=0)


def test_config_rejects_non_cpu_device(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    with pytest.raises(ValueError, match="CPU"):
        DenseBaselineConfig.model_validate({**config.model_dump(), "device": "cuda"})


def test_constructing_config_does_not_load_real_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import decision_agent.retrieval.embeddings as embeddings

    monkeypatch.setattr(
        embeddings,
        "_default_sentence_transformer_factory",
        lambda **kwargs: pytest.fail("real model factory must not run during configuration"),
    )
    make_config(tmp_path)


def test_corpus_document_id_and_vector_record_id_have_explicit_distinct_roles() -> None:
    chunk = dense_baseline._make_chunk(
        "sales-east", "synthetic text", "sales", "synthetic-enterprise-baseline"
    )

    assert chunk.document_id == "sales-east"
    assert chunk.chunk_id == "m2b2c-child-sales-east"


def test_cli_defaults_are_anchored_to_repository_root() -> None:
    cli = load_baseline_cli()
    repository_root = Path(__file__).resolve().parents[3]

    args = cli.build_parser().parse_args([])

    assert args.corpus == repository_root / "datasets/retrieval/m2b2c_dense_corpus.jsonl"
    assert args.queries == repository_root / "datasets/retrieval/m2b2c_dense_queries.jsonl"
    assert args.output == (
        repository_root / "artifacts/evaluation/m2b2c_dense_bge_small_zh_v1_5.json"
    )


def test_cli_missing_input_fails_before_model_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = load_baseline_cli()
    monkeypatch.setattr(
        dense_baseline,
        "SentenceTransformerEmbeddingProvider",
        lambda **kwargs: pytest.fail("model must not load before dataset validation"),
    )

    with pytest.raises(EvaluationValidationError, match="corpus file does not exist"):
        cli.main(
            [
                "--corpus",
                str(tmp_path / "missing-corpus.jsonl"),
                "--queries",
                str(tmp_path / "missing-queries.jsonl"),
                "--output",
                str(tmp_path / "report.json"),
            ]
        )


def test_versioned_report_matches_dataset_and_independent_metric_recalculation() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    corpus_path = repository_root / "datasets/retrieval/m2b2c_dense_corpus.jsonl"
    queries_path = repository_root / "datasets/retrieval/m2b2c_dense_queries.jsonl"
    report_path = repository_root / "artifacts/evaluation/m2b2c_dense_bge_small_zh_v1_5.json"
    report = DenseBaselineReport.model_validate_json(report_path.read_text(encoding="utf-8"))

    assert report.corpus_sha256 == dataset_module.compute_normalized_text_sha256(corpus_path)
    assert report.queries_sha256 == dataset_module.compute_normalized_text_sha256(queries_path)
    assert report.corpus_size == 36
    assert report.query_count == 18
    query_metrics: list[dict[str, float]] = []
    for result in report.query_results:
        relevant = set(result.relevant_document_ids)
        ranked = result.ranked_document_ids
        query_metrics.append(
            {
                "hit_rate_at_1": float(bool(set(ranked[:1]) & relevant)),
                "recall_at_1": len(set(ranked[:1]) & relevant) / len(relevant),
                "hit_rate_at_3": float(bool(set(ranked[:3]) & relevant)),
                "recall_at_3": len(set(ranked[:3]) & relevant) / len(relevant),
                "hit_rate_at_5": float(bool(set(ranked[:5]) & relevant)),
                "recall_at_5": len(set(ranked[:5]) & relevant) / len(relevant),
                "mrr_at_5": next(
                    (
                        1.0 / rank
                        for rank, document_id in enumerate(ranked[:5], start=1)
                        if document_id in relevant
                    ),
                    0.0,
                ),
            }
        )
    independently_aggregated = {
        metric_name: sum(metrics[metric_name] for metrics in query_metrics) / len(query_metrics)
        for metric_name in query_metrics[0]
    }
    assert report.metrics == pytest.approx(independently_aggregated)
