"""Reproducible dense retrieval baseline over the project's canonical services."""

from __future__ import annotations

import math
import platform
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from decision_agent.domain import ChildChunk
from decision_agent.evaluation.dataset import (
    compute_normalized_text_sha256,
    load_retrieval_dataset,
)
from decision_agent.evaluation.reporting import write_json_report_atomically
from decision_agent.evaluation.retrieval_metrics import aggregate_metrics, compute_query_metrics
from decision_agent.exceptions import EvaluationValidationError, RetrievalValidationError
from decision_agent.retrieval import (
    DenseIndexer,
    DenseRetriever,
    EmbeddingProvider,
    InMemoryVectorStore,
    SentenceTransformerEmbeddingProvider,
)

DEFAULT_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："  # noqa: RUF001


class EvaluationReportModel(BaseModel):
    """Strict JSON-safe base for generated evaluation artifacts."""

    model_config = ConfigDict(extra="forbid")


def _validate_metric_values(metrics: dict[str, float]) -> None:
    if not metrics or any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in metrics.values()
    ):
        raise ValueError("metric values must be finite numbers between zero and one")


class DenseBaselineConfig(EvaluationReportModel):
    """Explicit inputs for one reproducible dense retrieval run."""

    corpus_path: Path
    queries_path: Path
    output_path: Path
    model_name: str = Field(default="BAAI/bge-small-zh-v1.5", min_length=1)
    cache_folder: Path | None = None
    device: str = Field(default="cpu", min_length=1)
    dimension: int = Field(default=512, gt=0)
    batch_size: int = Field(default=32, gt=0)
    top_k: int = Field(default=5, gt=0)
    normalize_embeddings: bool = True
    query_instruction: str = Field(default=DEFAULT_QUERY_INSTRUCTION, min_length=1)
    local_files_only: bool = True

    @field_validator("device")
    @classmethod
    def require_cpu_device(cls, value: str) -> str:
        """Keep the M2B-2C benchmark within its explicitly authorized CPU scope."""
        if value.strip().lower() != "cpu":
            raise ValueError("M2B-2C baseline device must be CPU")
        return "cpu"


class RuntimeObservations(EvaluationReportModel):
    """Current-machine timings that are not production performance claims."""

    model_load_seconds: float = Field(ge=0)
    corpus_index_seconds: float = Field(ge=0)
    total_query_seconds: float = Field(ge=0)
    average_query_ms: float = Field(ge=0)


class DenseRankedResult(EvaluationReportModel):
    """One ranked document with an explicit rank and relevance marker."""

    rank: int = Field(gt=0)
    document_id: str = Field(min_length=1)
    score: float = Field(allow_inf_nan=False)
    is_relevant: bool


class DenseQueryResult(EvaluationReportModel):
    """Per-query ranking, labels, scores, and metrics."""

    query_id: str
    query: str
    category: str
    relevant_document_ids: tuple[str, ...]
    ranked_results: tuple[DenseRankedResult, ...]
    metrics: dict[str, float]

    @model_validator(mode="after")
    def validate_ranking(self) -> DenseQueryResult:
        """Reject ambiguous rankings before they become evaluation artifacts."""
        _validate_metric_values(self.metrics)
        expected_ranks = list(range(1, len(self.ranked_results) + 1))
        actual_ranks = [result.rank for result in self.ranked_results]
        if actual_ranks != expected_ranks:
            raise ValueError("ranked result ranks must be consecutive and start at one")
        document_ids = [result.document_id for result in self.ranked_results]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("ranked result document IDs must be unique")
        relevant_document_ids = set(self.relevant_document_ids)
        if any(
            result.is_relevant != (result.document_id in relevant_document_ids)
            for result in self.ranked_results
        ):
            raise ValueError("ranked result relevance markers must match relevance labels")
        return self

    @property
    def ranked_document_ids(self) -> tuple[str, ...]:
        """Return document IDs in rank order for metric recomputation."""
        return tuple(result.document_id for result in self.ranked_results)


class DenseBaselineReport(EvaluationReportModel):
    """Versionable dense baseline artifact without sensitive environment paths."""

    evaluation_schema_version: str = "1.0"
    baseline_name: str
    dataset_name: str
    corpus_file: str
    queries_file: str
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    queries_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_name: str
    embedding_dimension: int
    device: str
    normalize_embeddings: bool
    query_instruction: str
    top_k: int = Field(gt=0)
    corpus_size: int = Field(gt=0)
    query_count: int = Field(gt=0)
    metrics: dict[str, float]
    category_metrics: dict[str, dict[str, float]]
    query_results: tuple[DenseQueryResult, ...]
    runtime_observations: RuntimeObservations
    dependency_versions: dict[str, str]
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def validate_report_counts(self) -> DenseBaselineReport:
        """Keep report summary counts consistent with serialized query details."""
        _validate_metric_values(self.metrics)
        for category_values in self.category_metrics.values():
            _validate_metric_values(category_values)
        if self.query_count != len(self.query_results):
            raise ValueError("query_count must match query_results length")
        if any(len(result.ranked_results) > self.top_k for result in self.query_results):
            raise ValueError("query result length cannot exceed top_k")
        expected_average_ms = (
            self.runtime_observations.total_query_seconds * 1000 / self.query_count
        )
        if not math.isclose(
            self.runtime_observations.average_query_ms,
            expected_average_ms,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("average_query_ms must match total query time and query_count")
        return self


def _package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not-installed"


def _make_chunk(document_id: str, text: str, category: str, source: str) -> ChildChunk:
    return ChildChunk(
        chunk_id=f"m2b2c-child-{document_id}",
        parent_id=f"m2b2c-parent-{document_id}",
        document_id=document_id,
        document_version="m2b2c-v1",
        content=text,
        source=source,
        start_offset=0,
        end_offset=len(text),
        metadata={"category": category, "evaluation_dataset": "m2b2c-dense-v1"},
    )


async def run_dense_baseline(
    config: DenseBaselineConfig,
    *,
    embedding_provider: EmbeddingProvider | None = None,
) -> DenseBaselineReport:
    """Run the canonical dense chain and persist one reproducible JSON report."""
    corpus_sha256 = compute_normalized_text_sha256(config.corpus_path, dataset_name="corpus")
    queries_sha256 = compute_normalized_text_sha256(config.queries_path, dataset_name="queries")
    dataset = load_retrieval_dataset(config.corpus_path, config.queries_path)
    provider = embedding_provider or SentenceTransformerEmbeddingProvider(
        model_name=config.model_name,
        dimension=config.dimension,
        device=config.device,
        batch_size=config.batch_size,
        normalize=config.normalize_embeddings,
        query_instruction=config.query_instruction,
        cache_folder=str(config.cache_folder) if config.cache_folder is not None else None,
        local_files_only=config.local_files_only,
        trust_remote_code=False,
        show_progress=False,
    )
    if provider.dimension != config.dimension:
        raise RetrievalValidationError(
            "embedding provider dimension must match dense baseline configuration dimension"
        )

    store = InMemoryVectorStore(dimension=config.dimension)
    indexer = DenseIndexer(provider, store)
    retriever = DenseRetriever(provider, store)

    model_load_started = time.perf_counter()
    await provider.embed_query(dataset.queries[0].query)
    model_load_seconds = time.perf_counter() - model_load_started

    chunks = [
        _make_chunk(document.document_id, document.text, document.category, document.source)
        for document in dataset.corpus
    ]
    index_started = time.perf_counter()
    await indexer.index(chunks)
    corpus_index_seconds = time.perf_counter() - index_started

    query_results: list[DenseQueryResult] = []
    query_started = time.perf_counter()
    for query in dataset.queries:
        hits = await retriever.retrieve(query.query, top_k=config.top_k)
        ranked_document_ids = tuple(hit.document_id for hit in hits)
        metrics = compute_query_metrics(
            ranked_document_ids,
            query.relevant_document_ids,
            ks=(1, 3, 5),
            mrr_k=5,
        )
        query_results.append(
            DenseQueryResult(
                query_id=query.query_id,
                query=query.query,
                category=query.category,
                relevant_document_ids=query.relevant_document_ids,
                ranked_results=tuple(
                    DenseRankedResult(
                        rank=rank,
                        document_id=hit.document_id,
                        score=hit.score,
                        is_relevant=hit.document_id in query.relevant_document_ids,
                    )
                    for rank, hit in enumerate(hits, start=1)
                ),
                metrics=metrics,
            )
        )
    total_query_seconds = time.perf_counter() - query_started
    overall, categories = aggregate_metrics(
        [(result.category, result.metrics) for result in query_results]
    )
    if (
        compute_normalized_text_sha256(config.corpus_path, dataset_name="corpus") != corpus_sha256
        or compute_normalized_text_sha256(config.queries_path, dataset_name="queries")
        != queries_sha256
    ):
        raise EvaluationValidationError("evaluation dataset changed during baseline execution")
    report = DenseBaselineReport(
        evaluation_schema_version="1.0",
        baseline_name="m2b2c-dense-bge-small-zh-v1.5",
        dataset_name="m2b2c-dense-v1",
        corpus_file=config.corpus_path.name,
        queries_file=config.queries_path.name,
        corpus_sha256=corpus_sha256,
        queries_sha256=queries_sha256,
        model_name=config.model_name,
        embedding_dimension=config.dimension,
        device=config.device,
        normalize_embeddings=config.normalize_embeddings,
        query_instruction=config.query_instruction,
        top_k=config.top_k,
        corpus_size=len(dataset.corpus),
        query_count=len(dataset.queries),
        metrics=overall,
        category_metrics=categories,
        query_results=tuple(query_results),
        runtime_observations=RuntimeObservations(
            model_load_seconds=model_load_seconds,
            corpus_index_seconds=corpus_index_seconds,
            total_query_seconds=total_query_seconds,
            average_query_ms=total_query_seconds * 1000 / len(dataset.queries),
        ),
        dependency_versions={
            "python": platform.python_version(),
            "sentence-transformers": _package_version("sentence-transformers"),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
        },
        limitations=(
            "Metrics describe only the versioned small synthetic Chinese dataset.",
            "Timings are development-environment observations, not production benchmarks.",
            "This baseline uses dense retrieval and an in-memory vector store only.",
        ),
    )
    payload: dict[str, Any] = report.model_dump(mode="json")
    write_json_report_atomically(config.output_path, payload)
    return report
