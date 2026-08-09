"""Dense indexing and retrieval services over typed async contracts."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from decision_agent.domain import ChildChunk
from decision_agent.exceptions import RetrievalValidationError
from decision_agent.retrieval.models import (
    VectorRecord,
    VectorSearchFilter,
    VectorSearchResult,
    VectorUpsertResult,
)
from decision_agent.retrieval.protocols import EmbeddingProvider, VectorStore


@dataclass(frozen=True, slots=True)
class DenseIndexer:
    """Embed canonical child chunks and idempotently upsert vector records."""

    embedding_provider: EmbeddingProvider
    vector_store: VectorStore

    def __post_init__(self) -> None:
        if self.embedding_provider.dimension != self.vector_store.dimension:
            raise RetrievalValidationError("embedding and vector store dimensions must match")

    async def index(self, chunks: Sequence[ChildChunk]) -> VectorUpsertResult:
        """Index an ordered chunk batch and preserve all retrieval provenance."""
        if not chunks:
            return VectorUpsertResult(attempted_count=0, inserted_count=0, updated_count=0)

        vectors = await self.embedding_provider.embed_documents([chunk.content for chunk in chunks])
        if len(vectors) != len(chunks):
            raise RetrievalValidationError(
                "embedding provider result count must match child chunk count"
            )

        for vector in vectors:
            self._validate_embedding_vector(vector)

        records = [
            VectorRecord(
                record_id=chunk.chunk_id,
                parent_id=chunk.parent_id,
                document_id=chunk.document_id,
                document_version=chunk.document_version,
                content=chunk.content,
                vector=vector,
                source=chunk.source,
                page_number=chunk.page_number,
                metadata=dict(chunk.metadata),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        return await self.vector_store.upsert(records)

    def _validate_embedding_vector(self, vector: Sequence[float]) -> None:
        if len(vector) != self.embedding_provider.dimension:
            raise RetrievalValidationError(
                "embedding vector dimension must match embedding provider dimension"
            )
        if not all(math.isfinite(value) for value in vector):
            raise RetrievalValidationError("embedding vector elements must be finite")
        if math.sqrt(sum(value * value for value in vector)) == 0.0:
            raise RetrievalValidationError("embedding vector cannot be zero")


@dataclass(frozen=True, slots=True)
class DenseRetriever:
    """Embed a query and return typed vector search results without answer generation."""

    embedding_provider: EmbeddingProvider
    vector_store: VectorStore

    def __post_init__(self) -> None:
        if self.embedding_provider.dimension != self.vector_store.dimension:
            raise RetrievalValidationError("embedding and vector store dimensions must match")

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: VectorSearchFilter | None = None,
    ) -> list[VectorSearchResult]:
        """Run the query-to-embedding-to-vector-search chain."""
        if not query.strip():
            raise RetrievalValidationError("retrieval query cannot be empty or whitespace")
        if top_k <= 0:
            raise RetrievalValidationError("top_k must be greater than zero")
        query_vector = await self.embedding_provider.embed_query(query)
        return await self.vector_store.search(query_vector, top_k, filters)
