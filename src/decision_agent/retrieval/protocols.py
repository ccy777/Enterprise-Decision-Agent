"""Async contracts for embedding providers and vector stores."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from decision_agent.retrieval.models import (
    VectorRecord,
    VectorSearchFilter,
    VectorSearchResult,
    VectorUpsertResult,
)


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Generate fixed-dimension vectors through an async boundary."""

    @property
    def dimension(self) -> int:
        """Return the number of components in every generated vector."""
        ...

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed an ordered batch and preserve input ordering."""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Embed one retrieval query."""
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Persist and search typed vector records through an async boundary."""

    @property
    def dimension(self) -> int:
        """Return the vector dimension accepted by the store."""
        ...

    async def upsert(self, records: Sequence[VectorRecord]) -> VectorUpsertResult:
        """Insert or replace records by stable record ID."""
        ...

    async def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        filters: VectorSearchFilter | None = None,
    ) -> list[VectorSearchResult]:
        """Return deterministic cosine-similarity results."""
        ...

    async def delete_by_document(self, document_id: str) -> int:
        """Delete all records for a document and return the actual count."""
        ...

    async def list_record_ids(self) -> frozenset[str]:
        """Return an immutable snapshot of queryable logical record IDs."""
        ...

    async def count(self) -> int:
        """Return the number of queryable logical records."""
        ...
