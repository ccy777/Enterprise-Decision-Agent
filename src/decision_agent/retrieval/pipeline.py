"""Reusable enterprise retrieval pipeline over versioned parent/child chunks."""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import time
from collections.abc import Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from decision_agent.domain import ChildChunk, ParentChunk
from decision_agent.domain.models import Metadata
from decision_agent.exceptions import RetrievalValidationError
from decision_agent.observability.execution import (
    TraceSpanRecorder,
    complete_recorded_span,
    start_recorded_span,
)
from decision_agent.observability.models import TraceContext
from decision_agent.observability.stages import SpanStatus, TraceStage
from decision_agent.retrieval.bm25 import BM25Document, BM25Index, BM25Retriever
from decision_agent.retrieval.dense import DenseIndexer, DenseRetriever
from decision_agent.retrieval.evidence_context import EvidenceContext, EvidenceContextBuilder
from decision_agent.retrieval.fusion import FusedResult, FusionCandidate, reciprocal_rank_fusion
from decision_agent.retrieval.models import VectorUpsertResult
from decision_agent.retrieval.parent_expansion import (
    InMemoryParentChunkResolver,
    ParentChildCandidate,
    ParentExpander,
    ParentExpansionResult,
)
from decision_agent.retrieval.protocols import EmbeddingProvider, VectorStore
from decision_agent.retrieval.reranking import RerankCandidate, RerankedResult, Reranker


class PipelineModel(BaseModel):
    """Strict JSON-safe base for pipeline configuration and diagnostics."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class RetrievalPipelineConfig(PipelineModel):
    """Fixed, reviewable retrieval and evidence budgets."""

    config_version: str = Field(default="m2c2a1-v1", pattern=r"^m2c2a1-v[1-9][0-9]*$")
    dense_top_k: int = Field(default=10, gt=0, strict=True)
    bm25_top_k: int = Field(default=10, gt=0, strict=True)
    rrf_top_k: int = Field(default=10, gt=0, strict=True)
    rrf_k: float = Field(default=60.0, gt=0, allow_inf_nan=False)
    dense_weight: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    bm25_weight: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    reranker_candidate_count: int = Field(default=10, gt=0, strict=True)
    reranker_top_k: int = Field(default=5, gt=0, strict=True)
    parent_top_k: int = Field(default=5, gt=0, strict=True)
    evidence_max_count: int = Field(default=5, gt=0, strict=True)
    evidence_max_total_chars: int = Field(default=6000, gt=0, strict=True)
    evidence_max_chars_per_item: int = Field(default=1800, gt=0, strict=True)

    @model_validator(mode="after")
    def validate_stage_limits(self) -> RetrievalPipelineConfig:
        if self.reranker_candidate_count > self.rrf_top_k:
            raise ValueError("reranker candidate count cannot exceed RRF top_k")
        if self.reranker_top_k > self.reranker_candidate_count:
            raise ValueError("reranker top_k cannot exceed its candidate count")
        if self.parent_top_k > self.reranker_top_k:
            raise ValueError("parent top_k cannot exceed reranker top_k")
        if self.evidence_max_count > self.parent_top_k:
            raise ValueError("evidence count cannot exceed parent top_k")
        return self


class RetrievalInitializationTiming(PipelineModel):
    """One-time initialization latency, excluded from query latency."""

    data_load_seconds: float = Field(ge=0, allow_inf_nan=False)
    model_load_seconds: float = Field(ge=0, allow_inf_nan=False)
    dense_index_build_seconds: float = Field(ge=0, allow_inf_nan=False)
    bm25_index_build_seconds: float = Field(ge=0, allow_inf_nan=False)


class RetrievalStageTiming(PipelineModel):
    """Per-query stage latency with no initialization work included."""

    dense_query_seconds: float = Field(ge=0, allow_inf_nan=False)
    bm25_query_seconds: float = Field(ge=0, allow_inf_nan=False)
    rrf_seconds: float = Field(ge=0, allow_inf_nan=False)
    reranker_seconds: float = Field(ge=0, allow_inf_nan=False)
    parent_expansion_seconds: float = Field(ge=0, allow_inf_nan=False)
    evidence_context_seconds: float = Field(ge=0, allow_inf_nan=False)
    total_runtime_seconds: float = Field(ge=0, allow_inf_nan=False)


class ChildRetrievalResult(PipelineModel):
    """Normalized child-level hit emitted by one retrieval source."""

    rank: int = Field(gt=0, strict=True)
    candidate_id: str = Field(min_length=1)
    record_id: str = Field(min_length=1)
    parent_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    score: float = Field(allow_inf_nan=False)
    source_name: str = Field(min_length=1)
    source: str | None = None
    metadata: Metadata = Field(default_factory=dict)
    provenance: Metadata = Field(default_factory=dict)

    @field_validator("candidate_id", "record_id", "parent_id", "document_id", "content")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value cannot be blank")
        return value


class RetrievalPipelineResult(PipelineModel):
    """Auditable result of one complete retrieval-only request."""

    query: str = Field(min_length=1)
    dense_results: tuple[ChildRetrievalResult, ...]
    bm25_results: tuple[ChildRetrievalResult, ...]
    fused_results: tuple[FusedResult, ...]
    reranked_child_results: tuple[RerankedResult, ...]
    expanded_parent_results: tuple[ParentExpansionResult, ...]
    evidence_context: EvidenceContext
    stage_timings: RetrievalStageTiming
    total_runtime_seconds: float = Field(ge=0, allow_inf_nan=False)
    pipeline_config: RetrievalPipelineConfig

    @model_validator(mode="after")
    def validate_total_runtime(self) -> RetrievalPipelineResult:
        if not math.isclose(
            self.total_runtime_seconds,
            self.stage_timings.total_runtime_seconds,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("result and stage total runtimes must match")
        return self


class EnterpriseRetrievalPipeline:
    """Initialize once, then reuse a child-level hybrid retrieval chain."""

    def __init__(
        self,
        *,
        dataset_root: Path | str,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        reranker: Reranker,
        config: RetrievalPipelineConfig | None = None,
    ) -> None:
        self._dataset_root = Path(dataset_root)
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._reranker = reranker
        self._config = (config or RetrievalPipelineConfig()).model_copy(deep=True)
        if embedding_provider.dimension != vector_store.dimension:
            raise RetrievalValidationError("embedding and vector store dimensions must match")

        self._initialize_lock = asyncio.Lock()
        self._initialized = False
        self._closed = False
        self._initialization_timings: RetrievalInitializationTiming | None = None
        self._last_ingestion_result: VectorUpsertResult | None = None
        self._parents: tuple[ParentChunk, ...] = ()
        self._children: tuple[ChildChunk, ...] = ()
        self._children_by_id: dict[str, ChildChunk] = {}
        self._dense_retriever: DenseRetriever | None = None
        self._bm25_retriever: BM25Retriever | None = None
        self._parent_expander: ParentExpander | None = None
        self._evidence_builder: EvidenceContextBuilder | None = None

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def parent_count(self) -> int:
        return len(self._parents)

    @property
    def child_count(self) -> int:
        return len(self._children)

    @property
    def initialization_timings(self) -> RetrievalInitializationTiming | None:
        if self._initialization_timings is None:
            return None
        return self._initialization_timings.model_copy(deep=True)

    @property
    def last_ingestion_result(self) -> VectorUpsertResult | None:
        """Return the latest formal child-chunk upsert summary after initialization."""
        if self._last_ingestion_result is None:
            return None
        return self._last_ingestion_result.model_copy(deep=True)

    async def initialize(
        self,
        *,
        ingest_corpus: bool = True,
    ) -> RetrievalInitializationTiming:
        """Initialize one retrieval runtime, optionally performing the explicit corpus upsert."""
        if self._closed:
            raise RetrievalValidationError("retrieval pipeline is closed")
        if self._initialized:
            timing = self.initialization_timings
            if timing is None:  # pragma: no cover - defensive invariant
                raise RetrievalValidationError("retrieval pipeline timing is unavailable")
            return timing

        async with self._initialize_lock:
            if self._closed:
                raise RetrievalValidationError("retrieval pipeline is closed")
            if self._initialized:
                timing = self.initialization_timings
                if timing is None:  # pragma: no cover - defensive invariant
                    raise RetrievalValidationError("retrieval pipeline timing is unavailable")
                return timing

            started = time.perf_counter()
            parents, children = await asyncio.to_thread(_load_generated_chunks, self._dataset_root)
            data_load_seconds = time.perf_counter() - started

            started = time.perf_counter()
            await _initialize_dependency(self._embedding_provider)
            await _initialize_dependency(self._reranker)
            model_load_seconds = time.perf_counter() - started

            expected_ids = frozenset(child.chunk_id for child in children)
            ingestion_result: VectorUpsertResult | None = None
            if ingest_corpus:
                started = time.perf_counter()
                indexer = DenseIndexer(self._embedding_provider, self._vector_store)
                ingestion_result = await indexer.index(children)
                dense_index_build_seconds = time.perf_counter() - started
            else:
                dense_index_build_seconds = 0.0

            actual_ids = await self._vector_store.list_record_ids()
            if actual_ids != expected_ids:
                raise RetrievalValidationError(
                    "pipeline vector store must contain exactly the generated child corpus"
                )

            started = time.perf_counter()
            bm25_index = await asyncio.to_thread(BM25Index, _to_bm25_documents(children))
            bm25_index_build_seconds = time.perf_counter() - started

            self._parents = tuple(parent.model_copy(deep=True) for parent in parents)
            self._children = tuple(child.model_copy(deep=True) for child in children)
            self._children_by_id = {
                child.chunk_id: child.model_copy(deep=True) for child in self._children
            }
            self._dense_retriever = DenseRetriever(self._embedding_provider, self._vector_store)
            self._bm25_retriever = BM25Retriever(bm25_index)
            self._parent_expander = ParentExpander(InMemoryParentChunkResolver(self._parents))
            self._evidence_builder = EvidenceContextBuilder(
                max_total_chars=self._config.evidence_max_total_chars,
                max_evidence_chars=self._config.evidence_max_chars_per_item,
                max_evidence_count=self._config.evidence_max_count,
            )
            self._initialization_timings = RetrievalInitializationTiming(
                data_load_seconds=data_load_seconds,
                model_load_seconds=model_load_seconds,
                dense_index_build_seconds=dense_index_build_seconds,
                bm25_index_build_seconds=bm25_index_build_seconds,
            )
            self._last_ingestion_result = (
                None if ingestion_result is None else ingestion_result.model_copy(deep=True)
            )
            self._initialized = True
            return self._initialization_timings.model_copy(deep=True)

    async def retrieve(
        self,
        query: str,
        *,
        trace_recorder: TraceSpanRecorder | None = None,
        trace_parent_context: TraceContext | None = None,
        allowed_document_ids: frozenset[str] | None = None,
    ) -> RetrievalPipelineResult:
        """Run one query without rebuilding data, indexes, or models."""
        if self._closed:
            raise RetrievalValidationError("retrieval pipeline is closed")
        if not self._initialized:
            raise RetrievalValidationError("retrieval pipeline must be initialized")
        if not isinstance(query, str) or not query.strip():
            raise RetrievalValidationError("retrieval query cannot be empty or whitespace")
        dense_retriever, bm25_retriever, parent_expander, evidence_builder = (
            self._require_services()
        )

        total_started = time.perf_counter()
        retrieval_span = start_recorded_span(
            trace_recorder,
            stage=TraceStage.RETRIEVAL,
            component="retrieval",
            operation="hybrid_retrieve",
            parent_context=trace_parent_context,
            attributes={"requested_top_k": self._config.rrf_top_k},
        )
        try:
            started = time.perf_counter()
            raw_dense = await dense_retriever.retrieve(query, top_k=self._config.dense_top_k)
            dense_query_seconds = time.perf_counter() - started
            dense_results = tuple(
                self._normalize_child_result(
                    rank=rank,
                    child_id=result.record_id,
                    score=result.score,
                    source_name="dense",
                )
                for rank, result in enumerate(raw_dense, start=1)
            )
            dense_results = _filter_scope_results(dense_results, allowed_document_ids)

            started = time.perf_counter()
            raw_bm25 = await asyncio.to_thread(
                bm25_retriever.retrieve, query, top_k=self._config.bm25_top_k
            )
            bm25_query_seconds = time.perf_counter() - started
            bm25_results = tuple(
                self._normalize_child_result(
                    rank=result.rank,
                    child_id=result.candidate_id or result.record_id,
                    score=result.score,
                    source_name="bm25",
                )
                for result in raw_bm25
            )
            bm25_results = _filter_scope_results(bm25_results, allowed_document_ids)

            started = time.perf_counter()
            source_results: dict[str, Sequence[FusionCandidate]] = {}
            source_weights: dict[str, float] = {}
            for source_name, results, weight in (
                ("dense", dense_results, self._config.dense_weight),
                ("bm25", bm25_results, self._config.bm25_weight),
            ):
                if results:
                    source_results[source_name] = tuple(
                        FusionCandidate(
                            source_name=source_name,
                            rank=item.rank,
                            candidate_id=item.candidate_id,
                            record_id=item.record_id,
                            document_id=item.document_id,
                            source_score=item.score,
                            content=item.content,
                            metadata=item.model_copy(deep=True).metadata,
                            provenance=item.model_copy(deep=True).provenance,
                        )
                        for item in results
                    )
                    source_weights[source_name] = weight
            if source_results or allowed_document_ids is None:
                fused_results = tuple(
                    await asyncio.to_thread(
                        reciprocal_rank_fusion,
                        source_results,
                        rrf_k=self._config.rrf_k,
                        source_weights=source_weights,
                        top_k=self._config.rrf_top_k,
                    )
                )
            else:
                fused_results = ()
            rrf_seconds = time.perf_counter() - started
        except asyncio.CancelledError:
            complete_recorded_span(trace_recorder, retrieval_span, status=SpanStatus.CANCELLED)
            raise
        except Exception:
            complete_recorded_span(
                trace_recorder,
                retrieval_span,
                status=SpanStatus.FAILED,
                error_code="retrieval_failed",
                attributes={"success": False},
            )
            raise
        complete_recorded_span(
            trace_recorder,
            retrieval_span,
            status=SpanStatus.COMPLETED,
            attributes={
                "dense_candidate_count": len(dense_results),
                "sparse_candidate_count": len(bm25_results),
                "fused_candidate_count": len(fused_results),
                "retrieved_count": len(fused_results),
                "empty_result": not fused_results,
                "success": True,
            },
        )

        reranking_span = start_recorded_span(
            trace_recorder,
            stage=TraceStage.RERANKING,
            component="retrieval",
            operation="cross_encoder_rerank",
            parent_context=trace_parent_context,
            attributes={
                "input_candidate_count": min(
                    len(fused_results), self._config.reranker_candidate_count
                ),
                "requested_top_k": self._config.reranker_top_k,
            },
        )
        started = time.perf_counter()
        rerank_inputs = tuple(
            RerankCandidate(
                candidate_id=item.candidate_id,
                record_id=item.record_id,
                document_id=item.document_id,
                content=self._child(item.candidate_id or item.record_id or "").content,
                upstream_rank=item.final_rank,
                upstream_score=item.fused_score,
                metadata=item.model_copy(deep=True).metadata,
                provenance=item.model_copy(deep=True).provenance,
            )
            for item in fused_results[: self._config.reranker_candidate_count]
        )
        try:
            reranked_results = (
                tuple(
                    await self._reranker.rerank(
                        query,
                        rerank_inputs,
                        top_k=self._config.reranker_top_k,
                    )
                )
                if rerank_inputs
                else ()
            )
        except asyncio.CancelledError:
            complete_recorded_span(trace_recorder, reranking_span, status=SpanStatus.CANCELLED)
            raise
        except Exception:
            complete_recorded_span(
                trace_recorder,
                reranking_span,
                status=SpanStatus.FAILED,
                error_code="reranking_failed",
                attributes={"success": False},
            )
            raise
        reranker_seconds = time.perf_counter() - started
        complete_recorded_span(
            trace_recorder,
            reranking_span,
            status=SpanStatus.COMPLETED,
            attributes={"reranked_count": len(reranked_results), "success": True},
        )

        fused_by_id = {item.candidate_id or item.record_id or "": item for item in fused_results}
        parent_span = start_recorded_span(
            trace_recorder,
            stage=TraceStage.RETRIEVAL,
            component="retrieval",
            operation="parent_expansion",
            parent_context=trace_parent_context,
        )
        started = time.perf_counter()
        parent_candidates = tuple(
            self._to_parent_candidate(item, fused_by_id) for item in reranked_results
        )
        try:
            parent_results = tuple(
                await asyncio.to_thread(
                    parent_expander.expand,
                    parent_candidates,
                    top_k=self._config.parent_top_k,
                )
            )
        except asyncio.CancelledError:
            complete_recorded_span(trace_recorder, parent_span, status=SpanStatus.CANCELLED)
            raise
        except Exception:
            complete_recorded_span(
                trace_recorder,
                parent_span,
                status=SpanStatus.FAILED,
                error_code="parent_expansion_failed",
                attributes={"success": False},
            )
            raise
        parent_expansion_seconds = time.perf_counter() - started
        complete_recorded_span(
            trace_recorder,
            parent_span,
            status=SpanStatus.COMPLETED,
            attributes={
                "child_count": len(parent_candidates),
                "parent_count": len(parent_results),
                "expanded_count": len(parent_results),
                "success": True,
            },
        )

        evidence_span = start_recorded_span(
            trace_recorder,
            stage=TraceStage.EVIDENCE_SELECTION,
            component="retrieval",
            operation="build_evidence_context",
            parent_context=trace_parent_context,
            attributes={"candidate_evidence_count": len(parent_results)},
        )
        started = time.perf_counter()
        try:
            evidence_context = await asyncio.to_thread(evidence_builder.build, parent_results)
        except asyncio.CancelledError:
            complete_recorded_span(trace_recorder, evidence_span, status=SpanStatus.CANCELLED)
            raise
        except Exception:
            complete_recorded_span(
                trace_recorder,
                evidence_span,
                status=SpanStatus.FAILED,
                error_code="evidence_selection_failed",
                attributes={"success": False},
            )
            raise
        evidence_context_seconds = time.perf_counter() - started
        complete_recorded_span(
            trace_recorder,
            evidence_span,
            status=SpanStatus.COMPLETED,
            attributes={
                "selected_evidence_count": evidence_context.included_evidence_count,
                "empty_result": not evidence_context.evidence_items,
                "success": True,
            },
        )
        total_runtime_seconds = time.perf_counter() - total_started

        timings = RetrievalStageTiming(
            dense_query_seconds=dense_query_seconds,
            bm25_query_seconds=bm25_query_seconds,
            rrf_seconds=rrf_seconds,
            reranker_seconds=reranker_seconds,
            parent_expansion_seconds=parent_expansion_seconds,
            evidence_context_seconds=evidence_context_seconds,
            total_runtime_seconds=total_runtime_seconds,
        )
        return RetrievalPipelineResult(
            query=query,
            dense_results=dense_results,
            bm25_results=bm25_results,
            fused_results=fused_results,
            reranked_child_results=reranked_results,
            expanded_parent_results=parent_results,
            evidence_context=evidence_context,
            stage_timings=timings,
            total_runtime_seconds=total_runtime_seconds,
            pipeline_config=self._config.model_copy(deep=True),
        )

    async def close(self) -> None:
        """Release pipeline-owned snapshots; injected dependencies remain caller-owned."""
        async with self._initialize_lock:
            self._parents = ()
            self._children = ()
            self._children_by_id = {}
            self._dense_retriever = None
            self._bm25_retriever = None
            self._parent_expander = None
            self._evidence_builder = None
            self._initialized = False
            self._closed = True

    def _normalize_child_result(
        self, *, rank: int, child_id: str, score: float, source_name: str
    ) -> ChildRetrievalResult:
        child = self._child(child_id)
        return ChildRetrievalResult(
            rank=rank,
            candidate_id=child.chunk_id,
            record_id=child.chunk_id,
            parent_id=child.parent_id,
            document_id=child.document_id,
            content=child.content,
            score=score,
            source_name=source_name,
            source=child.source,
            metadata=child.model_copy(deep=True).metadata,
            provenance=_child_provenance(child),
        )

    def _to_parent_candidate(
        self, result: RerankedResult, fused_by_id: dict[str, FusedResult]
    ) -> ParentChildCandidate:
        child_id = result.candidate_id or result.record_id or ""
        child = self._child(child_id)
        fused = fused_by_id.get(child_id)
        if fused is None:
            raise RetrievalValidationError("reranker returned an unknown child candidate")
        if result.document_id != child.document_id:
            raise RetrievalValidationError("reranker changed the candidate document_id")
        return ParentChildCandidate(
            child_id=child.chunk_id,
            parent_id=child.parent_id,
            document_id=child.document_id,
            content=child.content,
            upstream_rank=result.final_rank,
            reranker_score=result.reranker_score,
            rrf_score=fused.fused_score,
            record_id=child.chunk_id,
            start_offset=child.start_offset,
            end_offset=child.end_offset,
            metadata=child.model_copy(deep=True).metadata,
            provenance=_child_provenance(child),
        )

    def _child(self, child_id: str) -> ChildChunk:
        child = self._children_by_id.get(child_id)
        if child is None:
            raise RetrievalValidationError("retrieval stage returned an unknown child ID")
        return child.model_copy(deep=True)

    def _require_services(
        self,
    ) -> tuple[DenseRetriever, BM25Retriever, ParentExpander, EvidenceContextBuilder]:
        services = (
            self._dense_retriever,
            self._bm25_retriever,
            self._parent_expander,
            self._evidence_builder,
        )
        if any(service is None for service in services):
            raise RetrievalValidationError("retrieval pipeline services are unavailable")
        return services  # type: ignore[return-value]


async def _initialize_dependency(dependency: Any) -> None:
    initializer = getattr(dependency, "initialize", None)
    if initializer is None:
        return
    outcome = initializer()
    if not inspect.isawaitable(outcome):
        raise RetrievalValidationError("dependency initialize must be asynchronous")
    await outcome


def _load_generated_chunks(dataset_root: Path) -> tuple[list[ParentChunk], list[ChildChunk]]:
    generated_root = (
        dataset_root
        if (dataset_root / "parent_chunks.jsonl").is_file()
        else dataset_root / "generated"
    )
    parent_rows = _read_jsonl(generated_root / "parent_chunks.jsonl")
    child_rows = _read_jsonl(generated_root / "child_chunks.jsonl")
    try:
        parents = [_parent_from_row(row) for row in parent_rows]
        children = [_child_from_row(row) for row in child_rows]
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        raise RetrievalValidationError("generated parent/child chunk schema is invalid") from exc
    if not parents or not children:
        raise RetrievalValidationError("generated parent and child chunks cannot be empty")
    if len({item.chunk_id for item in parents}) != len(parents):
        raise RetrievalValidationError("generated parent chunk IDs must be unique")
    if len({item.chunk_id for item in children}) != len(children):
        raise RetrievalValidationError("generated child chunk IDs must be unique")
    parent_by_id = {item.chunk_id: item for item in parents}
    for child in children:
        parent = parent_by_id.get(child.parent_id)
        if (
            parent is None
            or parent.document_id != child.document_id
            or parent.document_version != child.document_version
            or parent.source != child.source
            or child.start_offset is None
            or child.end_offset is None
            or child.start_offset < parent.start_offset
            or child.end_offset > parent.end_offset
        ):
            raise RetrievalValidationError("generated child has an invalid parent relation")
    return parents, children


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RetrievalValidationError(f"cannot read generated chunk file: {path.name}") from exc
    rows: list[dict[str, Any]] = []
    try:
        for line in lines:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError("JSONL row must be an object")
                rows.append(row)
    except (json.JSONDecodeError, TypeError) as exc:
        raise RetrievalValidationError(f"generated chunk file is invalid: {path.name}") from exc
    return rows


def _parent_from_row(row: dict[str, Any]) -> ParentChunk:
    _validate_generated_row(row)
    metadata = _runtime_metadata(row)
    return ParentChunk(
        chunk_id=row["chunk_id"],
        document_id=row["document_id"],
        document_version=row["document_version"],
        content=row["content"],
        block_ids=row["block_ids"],
        page_number=row.get("page_number"),
        source=row["source"],
        start_offset=row["start_offset"],
        end_offset=row["end_offset"],
        metadata=metadata,
    )


def _child_from_row(row: dict[str, Any]) -> ChildChunk:
    _validate_generated_row(row)
    return ChildChunk(
        chunk_id=row["chunk_id"],
        parent_id=row["parent_id"],
        document_id=row["document_id"],
        document_version=row["document_version"],
        content=row["content"],
        page_number=row.get("page_number"),
        source=row["source"],
        start_offset=row["start_offset"],
        end_offset=row["end_offset"],
        metadata=_runtime_metadata(row, parent_id=row["parent_id"]),
    )


def _validate_generated_row(row: dict[str, Any]) -> None:
    if row.get("schema_version") != "1.0":
        raise RetrievalValidationError("generated chunk schema_version must be 1.0")
    source = row.get("source")
    if not isinstance(source, str) or not source.strip():
        raise RetrievalValidationError("generated chunk source must be nonempty")
    if not isinstance(row.get("metadata"), dict) or not isinstance(row.get("provenance"), dict):
        raise RetrievalValidationError("generated chunk metadata and provenance must be objects")
    path = PurePosixPath(source)
    if (
        path.is_absolute()
        or PureWindowsPath(source).is_absolute()
        or "\\" in source
        or ".." in path.parts
    ):
        raise RetrievalValidationError("generated chunk source must be a relative POSIX path")


def _runtime_metadata(row: dict[str, Any], *, parent_id: str | None = None) -> Metadata:
    metadata = dict(row["metadata"])
    metadata.update(
        {
            "document_version": row["document_version"],
            "source": row["source"],
            "start_offset": row["start_offset"],
            "end_offset": row["end_offset"],
        }
    )
    if parent_id is not None:
        metadata["parent_id"] = parent_id
    if row.get("page_number") is not None:
        metadata["page_number"] = row["page_number"]
    provenance = row.get("provenance")
    if isinstance(provenance, dict):
        parser_name = provenance.get("parser_name")
        if isinstance(parser_name, str) and parser_name.strip():
            metadata["parser_name"] = parser_name
    return metadata


def _to_bm25_documents(children: Sequence[ChildChunk]) -> tuple[BM25Document, ...]:
    return tuple(
        BM25Document(
            record_id=child.chunk_id,
            candidate_id=child.chunk_id,
            document_id=child.document_id,
            content=child.content,
            category=str(child.metadata.get("category") or "enterprise_kb"),
            source=child.source or "",
            metadata=child.model_copy(deep=True).metadata,
        )
        for child in children
    )


def _filter_scope_results(
    results: tuple[ChildRetrievalResult, ...],
    allowed_document_ids: frozenset[str] | None,
) -> tuple[ChildRetrievalResult, ...]:
    """Apply the trusted application-layer document boundary before RRF or Evidence."""
    if allowed_document_ids is None:
        return results
    return tuple(
        result.model_copy(update={"rank": rank})
        for rank, result in enumerate(
            (item for item in results if item.document_id in allowed_document_ids),
            start=1,
        )
    )


def _child_provenance(child: ChildChunk) -> Metadata:
    provenance: Metadata = {
        "document_version": child.document_version,
        "parent_id": child.parent_id,
    }
    if child.source is not None:
        provenance["source"] = child.source
    if child.page_number is not None:
        provenance["page_number"] = child.page_number
    if child.start_offset is not None:
        provenance["start_offset"] = child.start_offset
    if child.end_offset is not None:
        provenance["end_offset"] = child.end_offset
    parser_name = child.metadata.get("parser_name")
    if isinstance(parser_name, str) and parser_name.strip():
        provenance["parser_name"] = parser_name
    return provenance
