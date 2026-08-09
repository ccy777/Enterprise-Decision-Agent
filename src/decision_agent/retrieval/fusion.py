"""Deterministic rank-only Reciprocal Rank Fusion."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from decision_agent.domain.models import Metadata
from decision_agent.exceptions import RetrievalValidationError


class FusionModel(BaseModel):
    """Strict base for fusion inputs and outputs."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class FusionCandidate(FusionModel):
    """One normalized candidate from an already ranked retrieval source."""

    source_name: str = Field(min_length=1)
    rank: int = Field(gt=0, strict=True)
    document_id: str = Field(min_length=1)
    candidate_id: str | None = Field(default=None, min_length=1)
    record_id: str | None = Field(default=None, min_length=1)
    source_score: float | None = Field(default=None, allow_inf_nan=False)
    content: str | None = None
    metadata: Metadata = Field(default_factory=dict)
    provenance: Metadata = Field(default_factory=dict)

    @field_validator("source_name", "document_id")
    @classmethod
    def reject_blank_identifiers(cls, value: str) -> str:
        """Reject whitespace-only source and document identities."""
        if not value.strip():
            raise ValueError("value cannot be blank")
        return value

    @field_validator("candidate_id")
    @classmethod
    def reject_blank_candidate_id(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("candidate_id cannot be blank")
        return value


class FusionContribution(FusionModel):
    """Auditable contribution made by one source to a fused document."""

    source_name: str = Field(min_length=1)
    source_rank: int = Field(gt=0, strict=True)
    contribution: float = Field(gt=0, allow_inf_nan=False)
    source_score: float | None = Field(default=None, allow_inf_nan=False)
    candidate_id: str | None = Field(default=None, min_length=1)
    record_id: str | None = Field(default=None, min_length=1)


class FusedResult(FusionModel):
    """One deterministic fused result with source-level diagnostics."""

    final_rank: int = Field(gt=0, strict=True)
    document_id: str = Field(min_length=1)
    candidate_id: str | None = Field(default=None, min_length=1)
    fused_score: float = Field(gt=0, allow_inf_nan=False)
    best_source_rank: int = Field(gt=0, strict=True)
    matched_source_count: int = Field(gt=0, strict=True)
    source_contributions: tuple[FusionContribution, ...] = Field(min_length=1)
    record_id: str | None = Field(default=None, min_length=1)
    content: str | None = None
    metadata: Metadata = Field(default_factory=dict)
    provenance: Metadata = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_contribution_summary(self) -> FusedResult:
        """Keep fused totals and source summaries internally consistent."""
        if self.matched_source_count != len(self.source_contributions):
            raise ValueError("matched_source_count must match source contributions")
        if self.best_source_rank != min(item.source_rank for item in self.source_contributions):
            raise ValueError("best_source_rank must match source contributions")
        contribution_sum = math.fsum(item.contribution for item in self.source_contributions)
        if not math.isclose(self.fused_score, contribution_sum, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("fused_score must equal the sum of source contributions")
        return self


def reciprocal_rank_fusion(
    source_results: Mapping[str, Sequence[FusionCandidate]],
    *,
    rrf_k: float = 60.0,
    source_weights: Mapping[str, float] | None = None,
    top_k: int | None = None,
) -> list[FusedResult]:
    """Fuse ranked sources using only rank positions and positive source weights.

    Final ordering is fused score descending, best source rank ascending,
    matched source count descending, document ID ascending, candidate ID ascending,
    then representative record ID ascending. ``candidate_id`` is the fusion identity
    when supplied; otherwise the historical document-level identity is preserved.
    When sources carry different payloads for one candidate,
    the representative is selected by source rank ascending, source name
    ascending, then record ID ascending; its payload is deep-copied.
    """
    if not source_results:
        raise RetrievalValidationError("RRF requires at least one source")
    if isinstance(rrf_k, bool) or not isinstance(rrf_k, (int, float)):
        raise RetrievalValidationError("rrf_k must be a finite positive number")
    if not math.isfinite(rrf_k) or rrf_k <= 0:
        raise RetrievalValidationError("rrf_k must be a finite positive number")
    if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0):
        raise RetrievalValidationError("top_k must be a positive integer")

    source_names = set(source_results)
    if any(not isinstance(name, str) or not name.strip() for name in source_names):
        raise RetrievalValidationError("source names cannot be blank")
    weights = (
        dict(source_weights) if source_weights is not None else {name: 1.0 for name in source_names}
    )
    if set(weights) != source_names:
        raise RetrievalValidationError("source weights must match source result names exactly")
    for name, weight in weights.items():
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise RetrievalValidationError(f"source weight for {name} must be finite and positive")
        if not math.isfinite(weight) or weight <= 0:
            raise RetrievalValidationError(f"source weight for {name} must be finite and positive")

    by_identity: dict[str, list[FusionCandidate]] = {}
    for source_name in sorted(source_results):
        candidates = source_results[source_name]
        if not candidates:
            raise RetrievalValidationError(f"RRF source {source_name} cannot be empty")
        validated: list[FusionCandidate] = []
        for candidate in candidates:
            try:
                copied = FusionCandidate.model_validate(candidate.model_dump(mode="python"))
            except (AttributeError, ValidationError) as exc:
                raise RetrievalValidationError(
                    f"RRF source {source_name} contains an invalid candidate"
                ) from exc
            if copied.source_name != source_name:
                raise RetrievalValidationError(
                    "candidate source_name must match its source mapping"
                )
            validated.append(copied.model_copy(deep=True))
        ranks = [candidate.rank for candidate in validated]
        if ranks != list(range(1, len(validated) + 1)):
            raise RetrievalValidationError(
                f"RRF source {source_name} ranks must be consecutive and start at one"
            )
        identities = [candidate.candidate_id or candidate.document_id for candidate in validated]
        if len(identities) != len(set(identities)):
            label = (
                "document_id"
                if all(candidate.candidate_id is None for candidate in validated)
                else "candidate identity"
            )
            raise RetrievalValidationError(
                f"RRF source {source_name} contains duplicate {label} values"
            )
        for candidate in validated:
            identity = candidate.candidate_id or candidate.document_id
            by_identity.setdefault(identity, []).append(candidate)

    unsorted: list[FusedResult] = []
    for identity, candidates in by_identity.items():
        document_ids = {candidate.document_id for candidate in candidates}
        if len(document_ids) != 1:
            raise RetrievalValidationError(
                "RRF candidates with one identity must share document_id"
            )
        ordered_candidates = sorted(
            candidates,
            key=lambda item: (item.rank, item.source_name, item.record_id or ""),
        )
        representative = ordered_candidates[0]
        contributions = tuple(
            FusionContribution(
                source_name=item.source_name,
                source_rank=item.rank,
                contribution=float(weights[item.source_name]) / (float(rrf_k) + item.rank),
                source_score=item.source_score,
                candidate_id=item.candidate_id,
                record_id=item.record_id,
            )
            for item in sorted(candidates, key=lambda item: item.source_name)
        )
        unsorted.append(
            FusedResult(
                final_rank=1,
                document_id=representative.document_id,
                candidate_id=identity if representative.candidate_id is not None else None,
                fused_score=math.fsum(item.contribution for item in contributions),
                best_source_rank=min(item.rank for item in candidates),
                matched_source_count=len(candidates),
                source_contributions=contributions,
                record_id=representative.record_id,
                content=representative.content,
                metadata=representative.model_copy(deep=True).metadata,
                provenance=representative.model_copy(deep=True).provenance,
            )
        )

    unsorted.sort(
        key=lambda item: (
            -item.fused_score,
            item.best_source_rank,
            -item.matched_source_count,
            item.document_id,
            item.candidate_id or "",
            item.record_id or "",
        )
    )
    limit = len(unsorted) if top_k is None else min(top_k, len(unsorted))
    return [
        item.model_copy(update={"final_rank": rank}, deep=True)
        for rank, item in enumerate(unsorted[:limit], start=1)
    ]
