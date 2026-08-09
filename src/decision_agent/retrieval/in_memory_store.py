"""Pure-Python deterministic in-memory vector store substitute."""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import ValidationError

from decision_agent.exceptions import RetrievalValidationError
from decision_agent.retrieval.models import (
    VectorRecord,
    VectorSearchFilter,
    VectorSearchResult,
    VectorUpsertResult,
)


class InMemoryVectorStore:
    """Store vectors by ID and search with deterministic cosine similarity."""

    def __init__(self, *, dimension: int) -> None:
        if dimension <= 0:
            raise RetrievalValidationError("store dimension must be greater than zero")
        self._dimension = dimension
        self._records: dict[str, VectorRecord] = {}

    @property
    def dimension(self) -> int:
        """Return the vector dimension accepted by this store."""
        return self._dimension

    async def upsert(self, records: Sequence[VectorRecord]) -> VectorUpsertResult:
        """Validate a full batch, then atomically replace records by ID."""
        validated_records: list[VectorRecord] = []
        record_ids: set[str] = set()
        for record in records:
            try:
                validated = VectorRecord.model_validate(record.model_dump(mode="python"))
            except ValidationError as exc:
                raise RetrievalValidationError("vector record failed model validation") from exc
            if validated.record_id in record_ids:
                raise RetrievalValidationError(
                    f"duplicate record_id in one upsert batch: {validated.record_id}"
                )
            self._validate_vector(validated.vector, field_name="record vector")
            record_ids.add(validated.record_id)
            validated_records.append(validated.model_copy(deep=True))

        inserted_count = sum(record.record_id not in self._records for record in validated_records)
        updated_count = len(validated_records) - inserted_count
        next_records = dict(self._records)
        next_records.update({record.record_id: record for record in validated_records})
        self._records = next_records
        return VectorUpsertResult(
            attempted_count=len(validated_records),
            inserted_count=inserted_count,
            updated_count=updated_count,
        )

    async def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        filters: VectorSearchFilter | None = None,
    ) -> list[VectorSearchResult]:
        """Search by cosine similarity with stable record-ID tie breaking."""
        if top_k <= 0:
            raise RetrievalValidationError("top_k must be greater than zero")
        query = [float(value) for value in query_vector]
        query_norm = self._validate_vector(query, field_name="query vector")
        allowed_document_ids = filters.allowed_document_ids() if filters else None

        scored: list[tuple[float, str, VectorRecord]] = []
        for record_id, record in self._records.items():
            if allowed_document_ids is not None and record.document_id not in allowed_document_ids:
                continue
            record_norm = math.sqrt(sum(value * value for value in record.vector))
            dot_product = sum(
                query_value * record_value
                for query_value, record_value in zip(query, record.vector, strict=True)
            )
            score = dot_product / (query_norm * record_norm)
            scored.append((score, record_id, record))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [self._to_search_result(score, record) for score, _, record in scored[:top_k]]

    async def delete_by_document(self, document_id: str) -> int:
        """Delete document records and return the exact number removed."""
        if not document_id:
            raise RetrievalValidationError("document_id cannot be empty")
        record_ids = [
            record_id
            for record_id, record in self._records.items()
            if record.document_id == document_id
        ]
        for record_id in record_ids:
            del self._records[record_id]
        return len(record_ids)

    async def list_record_ids(self) -> frozenset[str]:
        """Return an immutable snapshot of the current logical primary keys."""
        return frozenset(self._records)

    async def count(self) -> int:
        """Return the current logical record count."""
        return len(await self.list_record_ids())

    def _validate_vector(self, vector: Sequence[float], *, field_name: str) -> float:
        if not vector:
            raise RetrievalValidationError(f"{field_name} cannot be empty")
        if len(vector) != self.dimension:
            raise RetrievalValidationError(
                f"{field_name} dimension {len(vector)} does not match store dimension "
                f"{self.dimension}"
            )
        if not all(math.isfinite(value) for value in vector):
            raise RetrievalValidationError(f"{field_name} elements must be finite")
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            raise RetrievalValidationError(f"{field_name} cannot be a zero vector")
        return norm

    @staticmethod
    def _to_search_result(score: float, record: VectorRecord) -> VectorSearchResult:
        return VectorSearchResult(
            record_id=record.record_id,
            score=score,
            content=record.content,
            parent_id=record.parent_id,
            document_id=record.document_id,
            document_version=record.document_version,
            source=record.source,
            page_number=record.page_number,
            metadata=dict(record.metadata),
        )
