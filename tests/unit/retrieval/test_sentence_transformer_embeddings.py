"""Offline tests for the lazy local Sentence-Transformer provider."""

from __future__ import annotations

import asyncio
import importlib
import math
import sys
import threading
from collections.abc import Callable
from typing import Any

import pytest

from decision_agent.config import Settings
from decision_agent.domain import ChildChunk
from decision_agent.exceptions import (
    EmbeddingDimensionError,
    EmbeddingInferenceError,
    EmbeddingModelLoadError,
    RetrievalValidationError,
)
from decision_agent.retrieval import (
    DenseIndexer,
    DenseRetriever,
    EmbeddingProvider,
    InMemoryVectorStore,
    SentenceTransformerEmbeddingProvider,
)


class ArrayLike:
    """Small NumPy-like object used without importing NumPy in unit tests."""

    def __init__(self, value: Any) -> None:
        self.value = value

    def tolist(self) -> Any:
        return self.value


class ScalarLike:
    """Float-convertible scalar used to prove domain output conversion."""

    def __init__(self, value: float) -> None:
        self.value = value

    def __float__(self) -> float:
        return self.value


class FakeSentenceTransformer:
    def __init__(
        self,
        *,
        dimension: int = 3,
        output: Any | None = None,
        encode_error: Exception | None = None,
    ) -> None:
        self.dimension = dimension
        self.output = output
        self.encode_error = encode_error
        self.dimension_calls = 0
        self.encode_calls: list[tuple[list[str], dict[str, Any]]] = []

    def get_sentence_embedding_dimension(self) -> int:
        self.dimension_calls += 1
        return self.dimension

    def encode(self, inputs: list[str], **kwargs: Any) -> Any:
        self.encode_calls.append((list(inputs), dict(kwargs)))
        if self.encode_error is not None:
            raise self.encode_error
        if self.output is not None:
            return self.output
        vector = [1.0, 0.0, 0.0][: self.dimension]
        return [list(vector) for _ in inputs]


class DimensionReadFailureModel(FakeSentenceTransformer):
    """Model candidate that cannot complete lazy initialization."""

    def get_sentence_embedding_dimension(self) -> int:
        raise RuntimeError("dimension unavailable")


class ConcurrentEncodeProbeModel(FakeSentenceTransformer):
    """Record whether two worker threads enter encode at the same time."""

    def __init__(self) -> None:
        super().__init__()
        self.first_started = threading.Event()
        self.second_started = threading.Event()
        self.release_first = threading.Event()
        self._state_lock = threading.Lock()
        self._call_count = 0
        self._active_count = 0
        self.max_active_count = 0

    def encode(self, inputs: list[str], **kwargs: Any) -> Any:
        with self._state_lock:
            self._call_count += 1
            call_number = self._call_count
            self._active_count += 1
            self.max_active_count = max(self.max_active_count, self._active_count)
        try:
            if call_number == 1:
                self.first_started.set()
                assert self.release_first.wait(timeout=2)
            else:
                self.second_started.set()
            return super().encode(inputs, **kwargs)
        finally:
            with self._state_lock:
                self._active_count -= 1


def make_provider(
    model: FakeSentenceTransformer | None = None,
    *,
    model_factory: Callable[..., FakeSentenceTransformer] | None = None,
    dimension: int = 3,
    normalize: bool = True,
    batch_size: int = 8,
    query_instruction: str = "query: ",
    show_progress: bool = False,
) -> SentenceTransformerEmbeddingProvider:
    return SentenceTransformerEmbeddingProvider(
        model_name="test/model",
        dimension=dimension,
        device="cpu",
        batch_size=batch_size,
        normalize=normalize,
        query_instruction=query_instruction,
        cache_folder="test-cache",
        local_files_only=True,
        trust_remote_code=False,
        show_progress=show_progress,
        model=model,
        model_factory=model_factory,
    )


def make_chunk(chunk_id: str, content: str) -> ChildChunk:
    return ChildChunk(
        chunk_id=chunk_id,
        parent_id=f"parent-{chunk_id}",
        document_id="doc-embedding",
        document_version="v1",
        content=content,
        source="embedding.txt",
        page_number=1,
        start_offset=0,
        end_offset=len(content),
    )


def test_constructor_does_not_call_model_factory() -> None:
    calls = 0

    def factory(**kwargs: Any) -> FakeSentenceTransformer:
        nonlocal calls
        calls += 1
        return FakeSentenceTransformer()

    make_provider(model_factory=factory)

    assert calls == 0


@pytest.mark.asyncio
async def test_first_embedding_loads_model_once() -> None:
    calls = 0

    def factory(**kwargs: Any) -> FakeSentenceTransformer:
        nonlocal calls
        calls += 1
        return FakeSentenceTransformer()

    provider = make_provider(model_factory=factory)

    await provider.embed_query("first")

    assert calls == 1


@pytest.mark.asyncio
async def test_multiple_embeddings_reuse_one_model() -> None:
    calls = 0

    def factory(**kwargs: Any) -> FakeSentenceTransformer:
        nonlocal calls
        calls += 1
        return FakeSentenceTransformer()

    provider = make_provider(model_factory=factory)

    await provider.embed_query("first")
    await provider.embed_documents(["second"])

    assert calls == 1


@pytest.mark.asyncio
async def test_concurrent_first_calls_load_model_once_without_sleep() -> None:
    calls = 0
    started = threading.Event()
    release = threading.Event()

    def factory(**kwargs: Any) -> FakeSentenceTransformer:
        nonlocal calls
        calls += 1
        started.set()
        assert release.wait(timeout=2)
        return FakeSentenceTransformer()

    provider = make_provider(model_factory=factory)
    first = asyncio.create_task(provider.embed_query("first"))
    assert await asyncio.to_thread(started.wait, 2)
    second = asyncio.create_task(provider.embed_query("second"))
    release.set()

    await asyncio.gather(first, second)

    assert calls == 1


@pytest.mark.asyncio
async def test_model_factory_receives_supported_sdk_arguments() -> None:
    received: dict[str, Any] = {}

    def factory(**kwargs: Any) -> FakeSentenceTransformer:
        received.update(kwargs)
        return FakeSentenceTransformer()

    provider = SentenceTransformerEmbeddingProvider(
        model_name="test/model",
        revision="abc123",
        dimension=3,
        device="cpu",
        cache_folder="relative-cache",
        local_files_only=True,
        trust_remote_code=False,
        model_factory=factory,
    )

    await provider.embed_query("load")

    assert received == {
        "model_name_or_path": "test/model",
        "device": "cpu",
        "cache_folder": "relative-cache",
        "local_files_only": True,
        "revision": "abc123",
        "trust_remote_code": False,
    }


@pytest.mark.asyncio
async def test_from_settings_passes_configured_revision_to_model_factory() -> None:
    received: dict[str, Any] = {}

    def factory(**kwargs: Any) -> FakeSentenceTransformer:
        received.update(kwargs)
        return FakeSentenceTransformer(dimension=512, output=[[1.0, *([0.0] * 511)]])

    settings = Settings(
        app_name="Embedding Settings Unit",
        embedding_model_revision="7999e1d3359715c523056ef9478215996d62a620",
        _env_file=None,
    )
    provider = SentenceTransformerEmbeddingProvider.from_settings(settings, model_factory=factory)

    await provider.embed_query("load")

    assert received["revision"] == "7999e1d3359715c523056ef9478215996d62a620"


@pytest.mark.asyncio
async def test_optional_none_settings_are_passed_as_sdk_supported_none_values() -> None:
    received: dict[str, Any] = {}

    def factory(**kwargs: Any) -> FakeSentenceTransformer:
        received.update(kwargs)
        return FakeSentenceTransformer()

    provider = SentenceTransformerEmbeddingProvider(
        model_name="test/model",
        revision=None,
        dimension=3,
        device="cpu",
        cache_folder=None,
        local_files_only=True,
        trust_remote_code=False,
        model_factory=factory,
    )

    await provider.embed_query("load")

    assert received["revision"] is None
    assert received["cache_folder"] is None
    assert received["trust_remote_code"] is False


@pytest.mark.asyncio
async def test_model_load_failure_is_converted_and_preserves_cause() -> None:
    def factory(**kwargs: Any) -> FakeSentenceTransformer:
        raise RuntimeError("load failed")

    with pytest.raises(EmbeddingModelLoadError) as raised:
        await make_provider(model_factory=factory).embed_query("query")

    assert isinstance(raised.value.__cause__, RuntimeError)


@pytest.mark.asyncio
async def test_factory_failure_allows_next_embedding_call_to_retry() -> None:
    factory_calls = 0

    def factory(**kwargs: Any) -> FakeSentenceTransformer:
        nonlocal factory_calls
        factory_calls += 1
        if factory_calls == 1:
            raise RuntimeError("temporary load failure")
        return FakeSentenceTransformer()

    provider = make_provider(model_factory=factory)

    with pytest.raises(EmbeddingModelLoadError):
        await provider.embed_query("first attempt")
    result = await provider.embed_query("second attempt")

    assert result == [1.0, 0.0, 0.0]
    assert factory_calls == 2


@pytest.mark.asyncio
async def test_dimension_read_failure_discards_candidate_and_retries_factory() -> None:
    candidates = [DimensionReadFailureModel(), FakeSentenceTransformer()]
    factory_calls = 0

    def factory(**kwargs: Any) -> FakeSentenceTransformer:
        nonlocal factory_calls
        candidate = candidates[factory_calls]
        factory_calls += 1
        return candidate

    provider = make_provider(model_factory=factory)

    with pytest.raises(EmbeddingModelLoadError):
        await provider.embed_query("first attempt")
    result = await provider.embed_query("second attempt")

    assert result == [1.0, 0.0, 0.0]
    assert factory_calls == 2


@pytest.mark.asyncio
async def test_actual_model_dimension_mismatch_is_rejected_before_encode() -> None:
    model = FakeSentenceTransformer(dimension=4)

    with pytest.raises(EmbeddingDimensionError, match="actual dimension"):
        await make_provider(model).embed_query("query")

    assert model.encode_calls == []


@pytest.mark.asyncio
async def test_empty_document_batch_returns_without_loading_model() -> None:
    calls = 0

    def factory(**kwargs: Any) -> FakeSentenceTransformer:
        nonlocal calls
        calls += 1
        return FakeSentenceTransformer()

    assert await make_provider(model_factory=factory).embed_documents([]) == []
    assert calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["", " \n\t "])
async def test_empty_or_whitespace_document_is_rejected(text: str) -> None:
    with pytest.raises(RetrievalValidationError, match="document"):
        await make_provider(FakeSentenceTransformer()).embed_documents([text])


@pytest.mark.asyncio
async def test_none_document_batch_is_rejected() -> None:
    with pytest.raises(RetrievalValidationError, match="documents"):
        await make_provider(FakeSentenceTransformer()).embed_documents(None)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_document_batch_is_passed_to_encode_without_instruction() -> None:
    model = FakeSentenceTransformer()

    await make_provider(model).embed_documents(["first document", "第二份文档"])

    assert model.encode_calls[0][0] == ["first document", "第二份文档"]


@pytest.mark.asyncio
async def test_encode_receives_batch_normalization_and_output_arguments() -> None:
    model = FakeSentenceTransformer()

    await make_provider(model, batch_size=7, normalize=True).embed_documents(["document"])

    assert model.encode_calls[0][1] == {
        "batch_size": 7,
        "show_progress_bar": False,
        "convert_to_numpy": True,
        "convert_to_tensor": False,
        "normalize_embeddings": True,
    }


@pytest.mark.asyncio
async def test_show_progress_configuration_is_passed_explicitly() -> None:
    model = FakeSentenceTransformer()

    await make_provider(model, show_progress=True).embed_documents(["document"])

    assert model.encode_calls[0][1]["show_progress_bar"] is True


@pytest.mark.asyncio
async def test_array_and_scalar_outputs_become_plain_python_floats() -> None:
    model = FakeSentenceTransformer(
        output=ArrayLike([[ScalarLike(1.0), ScalarLike(0.0), ScalarLike(0.0)]])
    )

    vectors = await make_provider(model).embed_documents(["document"])

    assert vectors == [[1.0, 0.0, 0.0]]
    assert all(type(value) is float for value in vectors[0])


@pytest.mark.asyncio
async def test_output_dimension_mismatch_is_rejected() -> None:
    model = FakeSentenceTransformer(output=[[1.0, 0.0]])

    with pytest.raises(EmbeddingDimensionError, match="output dimension"):
        await make_provider(model).embed_documents(["document"])


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
async def test_non_finite_output_is_rejected(invalid: float) -> None:
    model = FakeSentenceTransformer(output=[[invalid, 0.0, 0.0]])

    with pytest.raises(EmbeddingInferenceError, match="finite"):
        await make_provider(model).embed_documents(["document"])


@pytest.mark.asyncio
async def test_zero_vector_output_is_rejected() -> None:
    model = FakeSentenceTransformer(output=[[0.0, 0.0, 0.0]])

    with pytest.raises(EmbeddingInferenceError, match="zero"):
        await make_provider(model).embed_documents(["document"])


@pytest.mark.asyncio
async def test_normalized_output_requires_unit_l2_norm() -> None:
    model = FakeSentenceTransformer(output=[[1.0, 1.0, 0.0]])

    with pytest.raises(EmbeddingInferenceError, match="L2-normalized"):
        await make_provider(model, normalize=True).embed_documents(["document"])


@pytest.mark.asyncio
async def test_non_normalized_output_allows_non_unit_nonzero_vector() -> None:
    model = FakeSentenceTransformer(output=[[2.0, 0.0, 0.0]])

    assert await make_provider(model, normalize=False).embed_documents(["document"]) == [
        [2.0, 0.0, 0.0]
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["", " \n\t "])
async def test_empty_or_whitespace_query_is_rejected(query: str) -> None:
    with pytest.raises(RetrievalValidationError, match="query"):
        await make_provider(FakeSentenceTransformer()).embed_query(query)


@pytest.mark.asyncio
async def test_query_instruction_is_concatenated_exactly_once() -> None:
    model = FakeSentenceTransformer()
    instruction = "为这个句子生成表示以用于检索相关文章："  # noqa: RUF001
    provider = make_provider(model, query_instruction=instruction)
    original_query = "产品电池保修多久"

    await provider.embed_query(original_query)
    await provider.embed_query(f"{instruction}已经带前缀")

    assert model.encode_calls[0][0] == [f"{instruction}{original_query}"]
    assert model.encode_calls[1][0] == [f"{instruction}已经带前缀"]
    assert original_query == "产品电池保修多久"


@pytest.mark.asyncio
async def test_query_returns_one_plain_vector() -> None:
    model = FakeSentenceTransformer(output=ArrayLike([[1.0, 0.0, 0.0]]))

    vector = await make_provider(model).embed_query("question")

    assert vector == [1.0, 0.0, 0.0]
    assert all(type(value) is float for value in vector)


@pytest.mark.asyncio
async def test_encode_error_is_converted_and_preserves_cause() -> None:
    model = FakeSentenceTransformer(encode_error=RuntimeError("encode failed"))

    with pytest.raises(EmbeddingInferenceError) as raised:
        await make_provider(model).embed_query("question")

    assert isinstance(raised.value.__cause__, RuntimeError)


@pytest.mark.asyncio
async def test_output_count_mismatch_is_rejected() -> None:
    model = FakeSentenceTransformer(output=[[1.0, 0.0, 0.0]])

    with pytest.raises(EmbeddingInferenceError, match="result count"):
        await make_provider(model).embed_documents(["first", "second"])


@pytest.mark.asyncio
async def test_one_dimensional_batch_output_is_rejected_explicitly() -> None:
    model = FakeSentenceTransformer(output=[1.0, 0.0, 0.0])

    with pytest.raises(EmbeddingInferenceError, match="two-dimensional"):
        await make_provider(model).embed_query("question")


@pytest.mark.asyncio
async def test_three_dimensional_batch_output_is_rejected_explicitly() -> None:
    model = FakeSentenceTransformer(output=[[[1.0, 0.0, 0.0]]])

    with pytest.raises(EmbeddingInferenceError, match="two-dimensional"):
        await make_provider(model).embed_query("question")


@pytest.mark.asyncio
async def test_concurrent_encode_calls_are_serialized_without_sleep() -> None:
    model = ConcurrentEncodeProbeModel()
    provider = make_provider(model)
    first = asyncio.create_task(provider.embed_query("first"))
    assert await asyncio.to_thread(model.first_started.wait, 2)
    second = asyncio.create_task(provider.embed_query("second"))

    second_entered_while_first_active = await asyncio.to_thread(model.second_started.wait, 0.2)
    model.release_first.set()
    await asyncio.gather(first, second)

    assert second_entered_while_first_active is False
    assert model.max_active_count == 1


def test_provider_satisfies_existing_embedding_protocol() -> None:
    assert isinstance(make_provider(FakeSentenceTransformer()), EmbeddingProvider)


@pytest.mark.asyncio
async def test_provider_runs_through_existing_dense_indexer_and_retriever() -> None:
    model = FakeSentenceTransformer(output=None)
    provider = make_provider(model)
    store = InMemoryVectorStore(dimension=3)
    indexer = DenseIndexer(provider, store)
    retriever = DenseRetriever(provider, store)

    await indexer.index([make_chunk("child-1", "document")])
    results = await retriever.retrieve("question", top_k=1)

    assert [result.record_id for result in results] == ["child-1"]


def test_provider_store_dimension_mismatch_fails_before_model_load() -> None:
    calls = 0

    def factory(**kwargs: Any) -> FakeSentenceTransformer:
        nonlocal calls
        calls += 1
        return FakeSentenceTransformer()

    provider = make_provider(model_factory=factory)
    store = InMemoryVectorStore(dimension=4)

    with pytest.raises(RetrievalValidationError, match="dimensions must match"):
        DenseIndexer(provider, store)
    with pytest.raises(RetrievalValidationError, match="dimensions must match"):
        DenseRetriever(provider, store)
    assert calls == 0


def test_reloading_embedding_module_does_not_import_sentence_transformers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import decision_agent.retrieval.embeddings as embedding_module

    monkeypatch.delitem(sys.modules, "sentence_transformers", raising=False)
    importlib.reload(embedding_module)

    assert "sentence_transformers" not in sys.modules
