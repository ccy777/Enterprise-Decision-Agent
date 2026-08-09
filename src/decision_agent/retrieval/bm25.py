"""Deterministic pure-Python BM25 indexing and sparse retrieval."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from decision_agent.domain.models import Metadata
from decision_agent.exceptions import RetrievalValidationError
from decision_agent.retrieval.tokenization import DeterministicChineseTokenizer, TextTokenizer


class SparseRetrievalModel(BaseModel):
    """Strict JSON-safe base for sparse retrieval contracts."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class BM25Document(SparseRetrievalModel):
    """One validated sparse-index document with retrieval provenance."""

    record_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    candidate_id: str | None = Field(default=None, min_length=1)
    content: str = Field(min_length=1)
    category: str = Field(min_length=1)
    source: str = Field(min_length=1)
    metadata: Metadata = Field(default_factory=dict)

    @field_validator("record_id", "document_id", "content", "category", "source")
    @classmethod
    def reject_blank_values(cls, value: str) -> str:
        """Reject identifiers and text containing only whitespace."""
        if not value.strip():
            raise ValueError("value cannot be blank")
        return value

    @field_validator("candidate_id")
    @classmethod
    def reject_blank_candidate_id(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("candidate_id cannot be blank")
        return value


class BM25SearchResult(SparseRetrievalModel):
    """One ranked sparse retrieval hit with an explicit BM25 score."""

    rank: int = Field(gt=0)
    record_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    candidate_id: str | None = Field(default=None, min_length=1)
    content: str = Field(min_length=1)
    score: float = Field(ge=0)
    category: str = Field(min_length=1)
    source: str = Field(min_length=1)
    metadata: Metadata = Field(default_factory=dict)

    @field_validator("score")
    @classmethod
    def require_finite_score(cls, value: float) -> float:
        """Reject NaN and infinity at the sparse retrieval boundary."""
        if not math.isfinite(value):
            raise ValueError("score must be finite")
        return value


class BM25Index:
    """Validated immutable-in-practice BM25 corpus statistics.

    For term ``t`` and document ``d`` the score contribution is::

        idf(t) * tf(t,d) * (k1 + 1)
        ---------------------------------------------
        tf(t,d) + k1 * (1 - b + b * dl(d) / avgdl)

    where ``idf(t) = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5))``.
    """

    def __init__(
        self,
        documents: Sequence[BM25Document],
        *,
        tokenizer: TextTokenizer | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not documents:
            raise RetrievalValidationError("BM25 corpus cannot be empty")
        if isinstance(k1, bool) or not isinstance(k1, (int, float)) or not math.isfinite(k1):
            raise RetrievalValidationError("BM25 k1 must be a finite number greater than zero")
        if k1 <= 0:
            raise RetrievalValidationError("BM25 k1 must be greater than zero")
        if isinstance(b, bool) or not isinstance(b, (int, float)) or not math.isfinite(b):
            raise RetrievalValidationError("BM25 b must be a finite number within [0, 1]")
        if not 0 <= b <= 1:
            raise RetrievalValidationError("BM25 b must be within [0, 1]")

        self._tokenizer = tokenizer or DeterministicChineseTokenizer()
        self._k1 = float(k1)
        self._b = float(b)
        validated_documents: list[BM25Document] = []
        frequencies: dict[str, Counter[str]] = {}
        document_lengths: dict[str, int] = {}
        document_frequency: Counter[str] = Counter()
        candidate_ids: set[str] = set()
        record_ids: set[str] = set()

        for candidate in documents:
            try:
                item = BM25Document.model_validate(candidate.model_dump(mode="python"))
            except (AttributeError, ValidationError) as exc:
                raise RetrievalValidationError("BM25 document failed model validation") from exc
            candidate_id = item.candidate_id or item.document_id
            if candidate_id in candidate_ids:
                label = "document_id" if item.candidate_id is None else "candidate_id"
                raise RetrievalValidationError(f"duplicate {label}: {candidate_id}")
            if item.record_id in record_ids:
                raise RetrievalValidationError(f"duplicate record_id: {item.record_id}")
            try:
                tokens = self._tokenizer.tokenize(item.content)
            except RetrievalValidationError as exc:
                raise RetrievalValidationError(
                    f"BM25 document {item.document_id} produced no searchable tokens"
                ) from exc
            term_frequencies = Counter(tokens)
            candidate_ids.add(candidate_id)
            record_ids.add(item.record_id)
            copied = item.model_copy(deep=True)
            validated_documents.append(copied)
            frequencies[candidate_id] = term_frequencies
            document_lengths[candidate_id] = len(tokens)
            document_frequency.update(term_frequencies.keys())

        self._documents = tuple(validated_documents)
        self._frequencies = frequencies
        self._document_lengths = document_lengths
        self._document_frequency = document_frequency
        self._average_document_length = sum(document_lengths.values()) / len(document_lengths)

    @property
    def tokenizer(self) -> TextTokenizer:
        """Return the tokenizer contract used for documents and queries."""
        return self._tokenizer

    @property
    def k1(self) -> float:
        """Return the term-frequency saturation parameter."""
        return self._k1

    @property
    def b(self) -> float:
        """Return the document-length normalization parameter."""
        return self._b

    @property
    def document_count(self) -> int:
        """Return corpus size N."""
        return len(self._documents)

    @property
    def average_document_length(self) -> float:
        """Return average corpus document length measured in tokens."""
        return self._average_document_length

    @property
    def vocabulary_size(self) -> int:
        """Return the number of unique corpus tokens."""
        return len(self._document_frequency)

    def documents(self) -> tuple[BM25Document, ...]:
        """Return deep copies so callers cannot mutate the index snapshot."""
        return tuple(document.model_copy(deep=True) for document in self._documents)

    def document_frequency(self, token: str) -> int:
        """Return how many corpus documents contain a token."""
        return self._document_frequency[token]

    def inverse_document_frequency(self, token: str) -> float:
        """Return the nonnegative Robertson-style smoothed IDF."""
        document_frequency = self.document_frequency(token)
        return math.log(
            1 + (self.document_count - document_frequency + 0.5) / (document_frequency + 0.5)
        )

    def score(self, query: str, document_id: str) -> float:
        """Score one query against a document ID or explicit candidate ID.

        The parameter name remains ``document_id`` for historical keyword-call
        compatibility. It denotes ``candidate_id`` when that field is configured.
        """
        tokens = self._tokenizer.tokenize(query)
        return self.score_tokens(tokens, document_id)

    def score_tokens(self, query_tokens: Sequence[str], document_id: str) -> float:
        """Score pre-tokenized terms against one configured candidate identity."""
        if document_id not in self._frequencies:
            raise RetrievalValidationError(f"unknown BM25 candidate identity: {document_id}")
        frequencies = self._frequencies[document_id]
        document_length = self._document_lengths[document_id]
        length_factor = 1 - self._b + self._b * document_length / self._average_document_length
        score = 0.0
        for token in query_tokens:
            term_frequency = frequencies[token]
            if term_frequency == 0:
                continue
            numerator = term_frequency * (self._k1 + 1)
            denominator = term_frequency + self._k1 * length_factor
            score += self.inverse_document_frequency(token) * numerator / denominator
        if not math.isfinite(score) or score < 0:
            raise RetrievalValidationError("BM25 score must be finite and nonnegative")
        return score


class BM25Retriever:
    """Search one fixed BM25 index without external I/O or mutable writes."""

    def __init__(self, index: BM25Index) -> None:
        self._index = index

    @property
    def index(self) -> BM25Index:
        """Return the fixed corpus index used by this retriever."""
        return self._index

    def retrieve(self, query: str, *, top_k: int) -> list[BM25SearchResult]:
        """Return positive-score results with stable score/ID ordering."""
        if not isinstance(query, str) or not query.strip():
            raise RetrievalValidationError("BM25 query cannot be empty or whitespace")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise RetrievalValidationError("top_k must be a positive integer")
        try:
            query_tokens = self._index.tokenizer.tokenize(query)
        except RetrievalValidationError as exc:
            raise RetrievalValidationError("BM25 query produced no searchable tokens") from exc

        scored: list[tuple[float, str, str, str, BM25Document]] = []
        for item in self._index.documents():
            candidate_id = item.candidate_id or item.document_id
            score = self._index.score_tokens(query_tokens, candidate_id)
            if score > 0:
                scored.append((score, item.document_id, candidate_id, item.record_id, item))
        scored.sort(key=lambda value: (-value[0], value[1], value[2], value[3]))

        return [
            BM25SearchResult(
                rank=rank,
                record_id=item.record_id,
                document_id=item.document_id,
                candidate_id=item.candidate_id,
                content=item.content,
                score=score,
                category=item.category,
                source=item.source,
                metadata=item.model_copy(deep=True).metadata,
            )
            for rank, (score, _, _, _, item) in enumerate(scored[:top_k], start=1)
        ]
