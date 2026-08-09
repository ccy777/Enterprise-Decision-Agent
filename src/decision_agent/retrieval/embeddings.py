"""Offline deterministic and lazy local semantic embedding providers."""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
import unicodedata
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from decision_agent.exceptions import (
    EmbeddingDimensionError,
    EmbeddingInferenceError,
    EmbeddingModelLoadError,
    RetrievalValidationError,
)

if TYPE_CHECKING:
    from decision_agent.config import Settings

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[\u3400-\u9fff]")
_ModelFactory = Callable[..., Any]


def _default_sentence_transformer_factory(**kwargs: Any) -> Any:
    """Import and construct the SDK model only on the first embedding call."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(**kwargs)


class SentenceTransformerEmbeddingProvider:
    """Lazy CPU-first adapter for a local Sentence-Transformer model."""

    def __init__(
        self,
        *,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        revision: str | None = None,
        dimension: int = 512,
        device: str = "cpu",
        batch_size: int = 32,
        normalize: bool = True,
        query_instruction: str = "为这个句子生成表示以用于检索相关文章：",  # noqa: RUF001
        cache_folder: str | None = None,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        show_progress: bool = False,
        model: Any | None = None,
        model_factory: _ModelFactory | None = None,
    ) -> None:
        if not model_name.strip():
            raise RetrievalValidationError("embedding model name cannot be empty")
        if dimension <= 0:
            raise RetrievalValidationError("embedding dimension must be greater than zero")
        if not device.strip():
            raise RetrievalValidationError("embedding device cannot be empty")
        if batch_size <= 0:
            raise RetrievalValidationError("embedding batch size must be greater than zero")
        if not query_instruction:
            raise RetrievalValidationError("embedding query instruction cannot be empty")
        if trust_remote_code:
            raise RetrievalValidationError("embedding trust_remote_code must remain false")
        if model is not None and model_factory is not None:
            raise RetrievalValidationError("provide either an embedding model or model_factory")

        self._model_name = model_name
        self._revision = revision
        self._dimension = dimension
        self._device = device
        self._batch_size = batch_size
        self._normalize = normalize
        self._query_instruction = query_instruction
        self._cache_folder = cache_folder
        self._local_files_only = local_files_only
        self._trust_remote_code = False
        self._show_progress = show_progress
        self._injected_model = model
        self._model: Any | None = None
        self._model_factory = model_factory or _default_sentence_transformer_factory
        self._model_ready = False
        self._model_lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        model: Any | None = None,
        model_factory: _ModelFactory | None = None,
    ) -> SentenceTransformerEmbeddingProvider:
        """Create an unloaded provider from validated application settings."""
        return cls(
            model_name=settings.embedding_model_name,
            revision=settings.embedding_model_revision,
            dimension=settings.embedding_dimension,
            device=settings.embedding_device,
            batch_size=settings.embedding_batch_size,
            normalize=settings.embedding_normalize,
            query_instruction=settings.embedding_query_instruction,
            cache_folder=settings.embedding_cache_folder,
            local_files_only=settings.embedding_local_files_only,
            trust_remote_code=settings.embedding_trust_remote_code,
            show_progress=settings.embedding_show_progress,
            model=model,
            model_factory=model_factory,
        )

    @property
    def dimension(self) -> int:
        """Return the configured output dimension without loading the model."""
        return self._dimension

    async def initialize(self) -> None:
        """Load and validate the configured model without encoding text."""
        await self._get_model()

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed an ordered document batch without applying the query instruction."""
        if texts is None or isinstance(texts, (str, bytes)):
            raise RetrievalValidationError("embedding documents must be a sequence of strings")
        documents = list(texts)
        if not documents:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in documents):
            raise RetrievalValidationError("embedding document cannot be empty or whitespace")
        return await self._encode(documents)

    async def embed_query(self, text: str) -> list[float]:
        """Embed one query with exactly one configured retrieval instruction."""
        if not isinstance(text, str) or not text.strip():
            raise RetrievalValidationError("embedding query cannot be empty or whitespace")
        instructed = (
            text if text.startswith(self._query_instruction) else self._query_instruction + text
        )
        return (await self._encode([instructed]))[0]

    async def _get_model(self) -> Any:
        if self._model_ready:
            return self._model
        async with self._model_lock:
            if self._model_ready:
                return self._model
            candidate = self._injected_model
            if candidate is None:
                try:
                    candidate = await asyncio.to_thread(
                        self._model_factory,
                        model_name_or_path=self._model_name,
                        device=self._device,
                        cache_folder=self._cache_folder,
                        local_files_only=self._local_files_only,
                        revision=self._revision,
                        trust_remote_code=self._trust_remote_code,
                    )
                except Exception as exc:
                    raise EmbeddingModelLoadError("failed to load local embedding model") from exc
            try:
                actual_dimension = await asyncio.to_thread(
                    candidate.get_sentence_embedding_dimension
                )
            except Exception as exc:
                raise EmbeddingModelLoadError(
                    "failed to read local embedding model dimension"
                ) from exc
            if actual_dimension != self.dimension:
                raise EmbeddingDimensionError(
                    f"embedding model actual dimension {actual_dimension} does not match "
                    f"configured dimension {self.dimension}"
                )
            self._model = candidate
            self._model_ready = True
            return self._model

    async def _encode(self, texts: list[str]) -> list[list[float]]:
        model = await self._get_model()
        async with self._inference_lock:
            try:
                raw_vectors = await asyncio.to_thread(
                    model.encode,
                    texts,
                    batch_size=self._batch_size,
                    show_progress_bar=self._show_progress,
                    convert_to_numpy=True,
                    convert_to_tensor=False,
                    normalize_embeddings=self._normalize,
                )
            except Exception as exc:
                raise EmbeddingInferenceError("local embedding inference failed") from exc
        return self._validate_vectors(raw_vectors, expected_count=len(texts))

    def _validate_vectors(self, raw_vectors: Any, *, expected_count: int) -> list[list[float]]:
        vectors = raw_vectors.tolist() if hasattr(raw_vectors, "tolist") else raw_vectors
        if not isinstance(vectors, Sequence) or isinstance(vectors, (str, bytes)):
            raise EmbeddingInferenceError("embedding output must be a two-dimensional sequence")
        if vectors and not self._is_vector_sequence(vectors[0]):
            raise EmbeddingInferenceError("embedding output must be a two-dimensional sequence")
        if len(vectors) != expected_count:
            raise EmbeddingInferenceError("embedding result count must match input count")

        validated: list[list[float]] = []
        for raw_vector in vectors:
            current = raw_vector.tolist() if hasattr(raw_vector, "tolist") else raw_vector
            if not self._is_vector_sequence(current):
                raise EmbeddingInferenceError("embedding output must be a two-dimensional sequence")
            if any(self._is_vector_sequence(value) for value in current):
                raise EmbeddingInferenceError("embedding output must be a two-dimensional sequence")
            try:
                vector = [float(value) for value in current]
            except (TypeError, ValueError, OverflowError) as exc:
                raise EmbeddingInferenceError("embedding vector elements must be numeric") from exc
            if len(vector) != self.dimension:
                raise EmbeddingDimensionError(
                    f"embedding output dimension {len(vector)} does not match {self.dimension}"
                )
            if not all(math.isfinite(value) for value in vector):
                raise EmbeddingInferenceError("embedding vector elements must be finite")
            norm = math.sqrt(sum(value * value for value in vector))
            if norm == 0.0:
                raise EmbeddingInferenceError("embedding output cannot be a zero vector")
            if self._normalize and not math.isclose(norm, 1.0, rel_tol=1e-4, abs_tol=1e-6):
                raise EmbeddingInferenceError("embedding output must be L2-normalized")
            validated.append(vector)
        return validated

    @staticmethod
    def _is_vector_sequence(value: Any) -> bool:
        return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


class DeterministicHashEmbeddingProvider:
    """Produce deterministic feature-hash vectors for offline testing only.

    This is not a semantic embedding model. It exists solely for interface
    validation, deterministic unit tests, and a local dense-retrieval baseline.
    Production environments must replace it with a real EmbeddingProvider.
    """

    def __init__(self, *, dimension: int = 128) -> None:
        if dimension <= 0:
            raise RetrievalValidationError("embedding dimension must be greater than zero")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """Return the configured output dimension."""
        return self._dimension

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed an ordered text batch without external services."""
        return [self._embed(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        """Embed one query without external services."""
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        tokens = self._tokenize(text)
        vector = [0.0] * self.dimension
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], byteorder="big") % self.dimension
            sign = 1.0 if digest[8] & 1 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(component * component for component in vector))
        if norm == 0.0:
            raise RetrievalValidationError("text tokens collapsed to a zero vector")
        return [component / norm for component in vector]

    @staticmethod
    def _tokenize(text: str) -> tuple[str, ...]:
        if not text.strip():
            raise RetrievalValidationError("embedding text cannot be empty or whitespace")
        normalized = unicodedata.normalize("NFKC", text).lower()
        tokens = tuple(_TOKEN_PATTERN.findall(normalized))
        if not tokens:
            raise RetrievalValidationError("embedding text contains no supported tokens")
        return tokens
