"""Typed vector records, filters, and retrieval results."""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from decision_agent.domain.models import Metadata


class RetrievalModel(BaseModel):
    """Strict base for JSON-safe retrieval contracts."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class VectorRecord(RetrievalModel):
    """Canonical child-chunk record prepared for a vector store."""

    record_id: str = Field(min_length=1)
    parent_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    document_version: str = Field(min_length=1)
    content: str = Field(min_length=1)
    vector: list[float] = Field(min_length=1)
    source: str | None = None
    page_number: int | None = Field(default=None, ge=1)
    metadata: Metadata = Field(default_factory=dict)

    @field_validator("vector")
    @classmethod
    def validate_vector(cls, vector: list[float]) -> list[float]:
        """Reject NaN and infinite vector components."""
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("vector elements must be finite")
        return vector


class VectorSearchResult(RetrievalModel):
    """One immutable-by-contract vector search hit with provenance."""

    record_id: str = Field(min_length=1)
    score: float
    content: str = Field(min_length=1)
    parent_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    document_version: str = Field(min_length=1)
    source: str | None = None
    page_number: int | None = Field(default=None, ge=1)
    metadata: Metadata = Field(default_factory=dict)

    @field_validator("score")
    @classmethod
    def validate_score(cls, score: float) -> float:
        """Require a finite similarity score."""
        if not math.isfinite(score):
            raise ValueError("score must be finite")
        return score


class VectorUpsertResult(RetrievalModel):
    """Counts produced by one idempotent vector upsert operation."""

    attempted_count: int = Field(ge=0)
    inserted_count: int = Field(ge=0)
    updated_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_counts(self) -> VectorUpsertResult:
        """Ensure every attempted record is classified once."""
        if self.attempted_count != self.inserted_count + self.updated_count:
            raise ValueError("attempted_count must equal inserted_count plus updated_count")
        return self


class VectorSearchFilter(RetrievalModel):
    """Minimal document-scoped filtering supported by M2B-1."""

    document_id: str | None = Field(default=None, min_length=1)
    document_ids: tuple[str, ...] = ()

    @field_validator("document_ids")
    @classmethod
    def validate_document_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Reject blank or duplicate document identifiers."""
        if any(not value for value in values):
            raise ValueError("document_ids cannot contain blank values")
        if len(values) != len(set(values)):
            raise ValueError("document_ids must be unique")
        return values

    @model_validator(mode="after")
    def validate_scope(self) -> VectorSearchFilter:
        """Keep single- and multi-document filter forms unambiguous."""
        if self.document_id is not None and self.document_ids:
            raise ValueError("use either document_id or document_ids, not both")
        return self

    def allowed_document_ids(self) -> frozenset[str] | None:
        """Return the selected document IDs, or None when no filter is active."""
        if self.document_id is not None:
            return frozenset({self.document_id})
        if self.document_ids:
            return frozenset(self.document_ids)
        return None
