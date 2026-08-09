"""Lazy, typed CrossEncoder reranking without retrieval-store dependencies."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from decision_agent.domain.models import Metadata
from decision_agent.exceptions import (
    RerankerInferenceError,
    RerankerModelLoadError,
    RetrievalValidationError,
)

_ModelFactory = Callable[..., Any]


def _default_cross_encoder_factory(**kwargs: Any) -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(**kwargs)


class RerankingModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class RerankCandidate(RerankingModel):
    """One upstream-ranked document eligible for a cross-encoder score."""

    document_id: str = Field(min_length=1)
    candidate_id: str | None = Field(default=None, min_length=1)
    content: str = Field(min_length=1)
    upstream_rank: int = Field(gt=0, strict=True)
    record_id: str | None = Field(default=None, min_length=1)
    upstream_score: float | None = Field(default=None, allow_inf_nan=False)
    metadata: Metadata = Field(default_factory=dict)
    provenance: Metadata = Field(default_factory=dict)

    @field_validator("document_id", "content")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value cannot be blank")
        return value

    @field_validator("candidate_id")
    @classmethod
    def reject_blank_candidate_id(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("candidate_id cannot be blank")
        return value


class RerankedResult(RerankingModel):
    """A scored candidate retaining upstream provenance."""

    final_rank: int = Field(gt=0, strict=True)
    document_id: str = Field(min_length=1)
    candidate_id: str | None = Field(default=None, min_length=1)
    content: str = Field(min_length=1)
    reranker_score: float = Field(allow_inf_nan=False)
    upstream_rank: int = Field(gt=0, strict=True)
    record_id: str | None = Field(default=None, min_length=1)
    upstream_score: float | None = Field(default=None, allow_inf_nan=False)
    metadata: Metadata = Field(default_factory=dict)
    provenance: Metadata = Field(default_factory=dict)


class Reranker(Protocol):
    async def rerank(
        self, query: str, candidates: Sequence[RerankCandidate], *, top_k: int | None = None
    ) -> list[RerankedResult]: ...


class SentenceTransformerCrossEncoderReranker:
    """CPU-first CrossEncoder adapter, loaded only for a nonempty request."""

    def __init__(
        self,
        *,
        model_name: str = "BAAI/bge-reranker-base",
        model_revision: str | None = None,
        device: str = "cpu",
        batch_size: int = 8,
        model: Any | None = None,
        model_factory: _ModelFactory | None = None,
    ) -> None:
        if not model_name.strip() or device.strip().lower() != "cpu" or batch_size <= 0:
            raise RetrievalValidationError(
                "reranker requires a nonempty model, CPU device, and batch size"
            )
        if model is not None and model_factory is not None:
            raise RetrievalValidationError("provide either a reranker model or model_factory")
        self.model_name, self.model_revision, self.device, self.batch_size = (
            model_name,
            model_revision,
            "cpu",
            batch_size,
        )
        self._injected_model, self._model = model, None
        self._model_factory = model_factory or _default_cross_encoder_factory
        self._ready = False
        self._model_load_seconds: float | None = None
        self._total_predict_seconds = 0.0
        self._model_lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()

    @property
    def model_load_seconds(self) -> float | None:
        return self._model_load_seconds

    @property
    def total_predict_seconds(self) -> float:
        """Return cumulative blocking model.predict time, excluding model construction."""
        return self._total_predict_seconds

    async def initialize(self) -> None:
        """Load the configured model without running prediction."""
        await self._get_model()

    async def _get_model(self) -> Any:
        if self._ready:
            return self._model
        async with self._model_lock:
            if self._ready:
                return self._model
            started = time.perf_counter()
            try:
                candidate = self._injected_model or await asyncio.to_thread(
                    self._model_factory,
                    model_name_or_path=self.model_name,
                    device=self.device,
                    revision=self.model_revision,
                )
            except Exception as exc:
                raise RerankerModelLoadError(
                    "failed to load configured cross-encoder model"
                ) from exc
            self._model = candidate
            self._model_load_seconds = time.perf_counter() - started
            self._ready = True
            return candidate

    async def rerank(
        self, query: str, candidates: Sequence[RerankCandidate], *, top_k: int | None = None
    ) -> list[RerankedResult]:
        if not isinstance(query, str) or not query.strip():
            raise RetrievalValidationError("reranker query cannot be empty or whitespace")
        if top_k is not None and (
            isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0
        ):
            raise RetrievalValidationError("reranker top_k must be a positive integer")
        copied = self._validate_candidates(candidates)
        if not copied:
            return []
        model = await self._get_model()
        pairs = [(query, item.content) for item in copied]
        try:
            async with self._inference_lock:
                predict_started = time.perf_counter()
                raw_scores = await asyncio.to_thread(
                    model.predict, pairs, batch_size=self.batch_size, show_progress_bar=False
                )
                self._total_predict_seconds += time.perf_counter() - predict_started
            scores = [float(value) for value in raw_scores]
        except (TypeError, ValueError) as exc:
            raise RerankerInferenceError("cross-encoder returned non-numeric scores") from exc
        except Exception as exc:
            raise RerankerInferenceError("cross-encoder prediction failed") from exc
        if len(scores) != len(copied) or not all(math.isfinite(score) for score in scores):
            raise RerankerInferenceError("cross-encoder returned invalid score output")
        ranked = sorted(
            zip(copied, scores, strict=True),
            key=lambda pair: (
                -pair[1],
                pair[0].upstream_rank,
                pair[0].document_id,
                pair[0].candidate_id or "",
                pair[0].record_id or "",
            ),
        )
        limit = len(ranked) if top_k is None else min(top_k, len(ranked))
        return [
            RerankedResult(
                final_rank=rank,
                document_id=item.document_id,
                candidate_id=item.candidate_id,
                content=item.content,
                reranker_score=score,
                upstream_rank=item.upstream_rank,
                record_id=item.record_id,
                upstream_score=item.upstream_score,
                metadata=item.model_copy(deep=True).metadata,
                provenance=item.model_copy(deep=True).provenance,
            )
            for rank, (item, score) in enumerate(ranked[:limit], start=1)
        ]

    @staticmethod
    def _validate_candidates(candidates: Sequence[RerankCandidate]) -> list[RerankCandidate]:
        if candidates is None or isinstance(candidates, (str, bytes)):
            raise RetrievalValidationError("reranker candidates must be a sequence")
        try:
            copied = [
                RerankCandidate.model_validate(item.model_dump(mode="python"))
                for item in candidates
            ]
        except (AttributeError, ValidationError) as exc:
            raise RetrievalValidationError("reranker candidates are invalid") from exc
        ids = [item.candidate_id or item.document_id for item in copied]
        if len(ids) != len(set(ids)):
            identity_name = (
                "document_id"
                if all(item.candidate_id is None for item in copied)
                else "candidate identity"
            )
            raise RetrievalValidationError(
                f"reranker candidates contain duplicate {identity_name} values"
            )
        if [item.upstream_rank for item in copied] != list(range(1, len(copied) + 1)):
            raise RetrievalValidationError(
                "reranker upstream ranks must be consecutive and start at one"
            )
        return copied
