"""Factories that construct the fixed formal enterprise retrieval pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from decision_agent.config import Settings
from decision_agent.retrieval.embeddings import SentenceTransformerEmbeddingProvider
from decision_agent.retrieval.in_memory_store import InMemoryVectorStore
from decision_agent.retrieval.milvus_store import MilvusVectorStore
from decision_agent.retrieval.pipeline import EnterpriseRetrievalPipeline, RetrievalPipelineConfig
from decision_agent.retrieval.protocols import EmbeddingProvider, VectorStore
from decision_agent.retrieval.reranking import (
    Reranker,
    SentenceTransformerCrossEncoderReranker,
)


class ProductionVectorStore(VectorStore, Protocol):
    """Vector-store lifecycle required by the production retrieval owner."""

    async def initialize(self) -> None:
        """Initialize the external vector store."""

    async def close(self) -> None:
        """Close the external vector store."""


@dataclass(frozen=True, slots=True)
class ProductionRetrievalDependencies:
    """Fixed construction seams for production retrieval external adapters."""

    embedding_provider_factory: Callable[[Settings], EmbeddingProvider]
    reranker_factory: Callable[[Settings], Reranker]
    vector_store_factory: Callable[[Settings], ProductionVectorStore]


class EnterpriseRetrievalRuntime:
    """Own the production retrieval pipeline and its external store lifecycle."""

    def __init__(
        self,
        *,
        pipeline: EnterpriseRetrievalPipeline,
        vector_store: ProductionVectorStore,
        embedding_provider: EmbeddingProvider,
        reranker: Reranker,
    ) -> None:
        self._pipeline = pipeline
        self._vector_store = vector_store
        self._embedding_provider = embedding_provider
        self._reranker = reranker

    @property
    def pipeline(self) -> EnterpriseRetrievalPipeline:
        """Return the formal pipeline consumed by the Knowledge graph."""
        return self._pipeline

    async def initialize(self) -> None:
        """Initialize a retrieval reader against an already-ingested formal corpus."""
        await self._vector_store.initialize()
        await self._pipeline.initialize(ingest_corpus=False)

    async def initialize_for_ingestion(self) -> None:
        """Explicitly initialize the formal corpus through the owned Milvus upsert path."""
        await self._vector_store.initialize()
        await self._pipeline.initialize(ingest_corpus=True)

    async def aclose(self) -> None:
        """Close the pipeline before its caller-owned Milvus store."""
        try:
            await self._pipeline.close()
        finally:
            await self._vector_store.close()


def build_enterprise_retrieval_pipeline(dataset_root: Path | str) -> EnterpriseRetrievalPipeline:
    """Build the unchanged formal BGE/BM25/RRF/reranker/parent-evidence pipeline.

    Initialization remains explicit so callers can construct one instance and reuse it for
    multiple queries. The pipeline's default configuration owns all formal retrieval parameters.
    """
    embedding = SentenceTransformerEmbeddingProvider(
        model_name="BAAI/bge-small-zh-v1.5", dimension=512, device="cpu"
    )
    reranker = SentenceTransformerCrossEncoderReranker(
        model_name="BAAI/bge-reranker-base", device="cpu"
    )
    return EnterpriseRetrievalPipeline(
        dataset_root=dataset_root,
        embedding_provider=embedding,
        vector_store=InMemoryVectorStore(dimension=embedding.dimension),
        reranker=reranker,
    )


def build_production_retrieval_runtime(
    settings: Settings,
    dependencies: ProductionRetrievalDependencies | None = None,
) -> EnterpriseRetrievalRuntime:
    """Construct the formal Milvus-backed retrieval runtime without performing I/O."""
    resolved = dependencies or _default_production_retrieval_dependencies()
    embedding = resolved.embedding_provider_factory(settings)
    reranker = resolved.reranker_factory(settings)
    vector_store = resolved.vector_store_factory(settings)
    pipeline = EnterpriseRetrievalPipeline(
        dataset_root=settings.knowledge_dataset_root or Path(),
        embedding_provider=embedding,
        vector_store=vector_store,
        reranker=reranker,
        config=RetrievalPipelineConfig(),
    )
    return EnterpriseRetrievalRuntime(
        pipeline=pipeline,
        vector_store=vector_store,
        embedding_provider=embedding,
        reranker=reranker,
    )


def _default_production_retrieval_dependencies() -> ProductionRetrievalDependencies:
    return ProductionRetrievalDependencies(
        embedding_provider_factory=SentenceTransformerEmbeddingProvider.from_settings,
        reranker_factory=_build_reranker_from_settings,
        vector_store_factory=MilvusVectorStore.from_settings,
    )


def _build_reranker_from_settings(settings: Settings) -> Reranker:
    return SentenceTransformerCrossEncoderReranker(
        model_name=settings.reranker_model_name,
        model_revision=settings.reranker_model_revision,
        device=settings.reranker_device,
        batch_size=settings.reranker_batch_size,
    )
