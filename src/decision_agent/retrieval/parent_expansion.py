"""Deterministic expansion from ranked child hits to parent evidence."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from decision_agent.domain import ParentChunk
from decision_agent.domain.models import Metadata
from decision_agent.exceptions import RetrievalValidationError


class ExpansionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ParentChildCandidate(ExpansionModel):
    child_id: str = Field(min_length=1)
    parent_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    upstream_rank: int = Field(gt=0, strict=True)
    reranker_score: float | None = Field(default=None, allow_inf_nan=False)
    rrf_score: float | None = Field(default=None, allow_inf_nan=False)
    record_id: str | None = Field(default=None, min_length=1)
    start_offset: int | None = Field(default=None, ge=0)
    end_offset: int | None = Field(default=None, ge=0)
    metadata: Metadata = Field(default_factory=dict)
    provenance: Metadata = Field(default_factory=dict)

    @field_validator("child_id", "parent_id", "document_id", "content")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value cannot be blank")
        return value


class MatchedChild(ParentChildCandidate):
    pass


class ParentExpansionResult(ExpansionModel):
    final_rank: int = Field(gt=0)
    parent_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    parent_content: str = Field(min_length=1)
    best_child_rank: int = Field(gt=0)
    matched_child_count: int = Field(gt=0)
    matched_children: tuple[MatchedChild, ...] = Field(min_length=1)
    metadata: Metadata = Field(default_factory=dict)
    provenance: Metadata = Field(default_factory=dict)


class ParentChunkResolver(Protocol):
    def resolve(self, parent_ids: Sequence[str]) -> Mapping[str, ParentChunk]: ...


class InMemoryParentChunkResolver:
    def __init__(self, parents: Sequence[ParentChunk]) -> None:
        self._parents = {parent.chunk_id: parent.model_copy(deep=True) for parent in parents}
        if len(self._parents) != len(parents):
            raise RetrievalValidationError("parent resolver cannot contain duplicate parent IDs")

    def resolve(self, parent_ids: Sequence[str]) -> Mapping[str, ParentChunk]:
        return {
            parent_id: self._parents[parent_id].model_copy(deep=True)
            for parent_id in parent_ids
            if parent_id in self._parents
        }


class ParentExpander:
    def __init__(self, resolver: ParentChunkResolver) -> None:
        self._resolver = resolver

    def expand(
        self, candidates: Sequence[ParentChildCandidate], *, top_k: int | None = None
    ) -> list[ParentExpansionResult]:
        if top_k is not None and (
            isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0
        ):
            raise RetrievalValidationError("parent expansion top_k must be a positive integer")
        copied = self._validate(candidates)
        if not copied:
            return []
        parent_ids = tuple(sorted({item.parent_id for item in copied}))
        resolved = self._resolver.resolve(parent_ids)
        if set(resolved) != set(parent_ids):
            raise RetrievalValidationError("parent resolver did not return every requested parent")
        if any(parent_id != parent.chunk_id for parent_id, parent in resolved.items()):
            raise RetrievalValidationError("parent resolver returned a mismatched parent ID")
        grouped: dict[str, list[ParentChildCandidate]] = {}
        for item in copied:
            parent = resolved[item.parent_id]
            if parent.chunk_id != item.parent_id or parent.document_id != item.document_id:
                raise RetrievalValidationError(
                    "resolved parent is inconsistent with child candidate"
                )
            grouped.setdefault(item.parent_id, []).append(item)
        unordered = []
        for parent_id, children in grouped.items():
            parent = resolved[parent_id]
            ordered = sorted(children, key=lambda child: child.upstream_rank)
            unordered.append(
                ParentExpansionResult(
                    final_rank=1,
                    parent_id=parent_id,
                    document_id=parent.document_id,
                    parent_content=parent.content,
                    best_child_rank=ordered[0].upstream_rank,
                    matched_child_count=len(ordered),
                    matched_children=tuple(
                        MatchedChild.model_validate(child.model_dump(mode="python"))
                        for child in ordered
                    ),
                    metadata=parent.model_copy(deep=True).metadata,
                    provenance={
                        "source": parent.source,
                        "document_version": parent.document_version,
                    },
                )
            )
        unordered.sort(
            key=lambda result: (result.best_child_rank, result.parent_id, result.document_id)
        )
        limit = len(unordered) if top_k is None else min(top_k, len(unordered))
        return [
            result.model_copy(update={"final_rank": rank}, deep=True)
            for rank, result in enumerate(unordered[:limit], 1)
        ]

    @staticmethod
    def _validate(candidates: Sequence[ParentChildCandidate]) -> list[ParentChildCandidate]:
        if candidates is None or isinstance(candidates, (str, bytes)):
            raise RetrievalValidationError("parent expansion candidates must be a sequence")
        try:
            copied = [
                ParentChildCandidate.model_validate(item.model_dump(mode="python"))
                for item in candidates
            ]
        except (AttributeError, ValidationError) as exc:
            raise RetrievalValidationError("parent expansion candidates are invalid") from exc
        if len({item.child_id for item in copied}) != len(copied):
            raise RetrievalValidationError(
                "parent expansion candidates contain duplicate child IDs"
            )
        if [item.upstream_rank for item in copied] != list(range(1, len(copied) + 1)):
            raise RetrievalValidationError(
                "parent expansion ranks must be consecutive and start at one"
            )
        if any(
            score is not None and not math.isfinite(score)
            for item in copied
            for score in (item.reranker_score, item.rrf_score)
        ):
            raise RetrievalValidationError("parent expansion scores must be finite")
        return copied
