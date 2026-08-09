"""Tests for dense indexing and retrieval service composition."""

import math
from collections.abc import Sequence

import pytest

from decision_agent.domain import ChildChunk
from decision_agent.exceptions import RetrievalValidationError
from decision_agent.retrieval import (
    DenseIndexer,
    DenseRetriever,
    DeterministicHashEmbeddingProvider,
    InMemoryVectorStore,
    VectorRecord,
    VectorSearchFilter,
    VectorUpsertResult,
)


def make_chunk(
    chunk_id: str,
    content: str,
    *,
    document_id: str = "doc-1",
    parent_id: str = "parent-1",
) -> ChildChunk:
    return ChildChunk(
        chunk_id=chunk_id,
        parent_id=parent_id,
        document_id=document_id,
        document_version="v2",
        content=content,
        page_number=3,
        source="reports/q2.txt",
        start_offset=0,
        end_offset=len(content),
        metadata={"department": "sales", "rank": 1},
    )


class WrongCountEmbeddingProvider:
    dimension = 4

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0]] * max(0, len(texts) - 1)

    async def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]


class FixedBatchEmbeddingProvider:
    dimension = 4

    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = vectors

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self.vectors

    async def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]


class ObservableVectorStore(InMemoryVectorStore):
    def __init__(self, *, dimension: int) -> None:
        super().__init__(dimension=dimension)
        self.upsert_calls = 0

    async def upsert(self, records: Sequence[VectorRecord]) -> VectorUpsertResult:
        self.upsert_calls += 1
        return await super().upsert(records)


@pytest.mark.asyncio
async def test_indexer_converts_child_chunks_to_vector_records() -> None:
    provider = DeterministicHashEmbeddingProvider(dimension=64)
    store = InMemoryVectorStore(dimension=64)
    result = await DenseIndexer(provider, store).index(
        [make_chunk("child-1", "east china product sales")]
    )
    hits = await store.search(await provider.embed_query("east china product sales"), 1)

    assert result.inserted_count == 1
    assert hits[0].record_id == "child-1"
    assert hits[0].content == "east china product sales"


@pytest.mark.asyncio
async def test_indexer_preserves_child_chunk_provenance() -> None:
    provider = DeterministicHashEmbeddingProvider(dimension=64)
    store = InMemoryVectorStore(dimension=64)
    chunk = make_chunk("child-1", "provenance test", document_id="doc-provenance")
    await DenseIndexer(provider, store).index([chunk])

    hit = (await store.search(await provider.embed_query(chunk.content), 1))[0]
    assert hit.parent_id == chunk.parent_id
    assert hit.document_id == chunk.document_id
    assert hit.document_version == chunk.document_version
    assert hit.source == chunk.source
    assert hit.page_number == chunk.page_number
    assert hit.metadata == chunk.metadata


@pytest.mark.asyncio
async def test_embedding_batch_count_mismatch_is_rejected() -> None:
    store = ObservableVectorStore(dimension=4)
    indexer = DenseIndexer(WrongCountEmbeddingProvider(), store)
    with pytest.raises(RetrievalValidationError, match="result count"):
        await indexer.index([make_chunk("a", "first"), make_chunk("b", "second")])
    assert store.upsert_calls == 0
    assert await store.count() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("vector", "message"),
    [
        ([1.0, 0.0, 0.0], "dimension"),
        ([math.nan, 0.0, 0.0, 0.0], "finite"),
        ([0.0, 0.0, 0.0, 0.0], "zero"),
    ],
)
async def test_invalid_provider_vector_never_calls_store_upsert(
    vector: list[float], message: str
) -> None:
    store = ObservableVectorStore(dimension=4)
    indexer = DenseIndexer(FixedBatchEmbeddingProvider([vector]), store)

    with pytest.raises(RetrievalValidationError, match=message):
        await indexer.index([make_chunk("a", "valid content")])

    assert store.upsert_calls == 0
    assert await store.count() == 0


@pytest.mark.asyncio
async def test_empty_chunk_batch_returns_zero_upsert_result() -> None:
    provider = DeterministicHashEmbeddingProvider(dimension=16)
    store = InMemoryVectorStore(dimension=16)
    result = await DenseIndexer(provider, store).index([])
    assert result.attempted_count == 0
    assert result.inserted_count == 0
    assert result.updated_count == 0
    assert await store.count() == 0


@pytest.mark.asyncio
async def test_query_embedding_search_chain_returns_ranked_results() -> None:
    provider = DeterministicHashEmbeddingProvider(dimension=256)
    store = InMemoryVectorStore(dimension=256)
    indexer = DenseIndexer(provider, store)
    retriever = DenseRetriever(provider, store)
    await indexer.index(
        [
            make_chunk("sales", "east china product sales declined"),
            make_chunk("policy", "employee annual leave policy"),
        ]
    )

    results = await retriever.retrieve("east china sales", top_k=2)

    assert results[0].record_id == "sales"
    assert results[0].score >= results[1].score


@pytest.mark.asyncio
async def test_retriever_applies_document_filter() -> None:
    provider = DeterministicHashEmbeddingProvider(dimension=64)
    store = InMemoryVectorStore(dimension=64)
    await DenseIndexer(provider, store).index(
        [
            make_chunk("a", "product sales", document_id="doc-a"),
            make_chunk("b", "product sales", document_id="doc-b"),
        ]
    )
    results = await DenseRetriever(provider, store).retrieve(
        "product sales", top_k=5, filters=VectorSearchFilter(document_id="doc-b")
    )
    assert [result.document_id for result in results] == ["doc-b"]


@pytest.mark.asyncio
async def test_blank_query_is_rejected_before_embedding() -> None:
    provider = DeterministicHashEmbeddingProvider(dimension=16)
    retriever = DenseRetriever(provider, InMemoryVectorStore(dimension=16))
    with pytest.raises(RetrievalValidationError, match="query"):
        await retriever.retrieve(" \t ", top_k=1)


@pytest.mark.asyncio
async def test_retriever_rejects_invalid_top_k() -> None:
    provider = DeterministicHashEmbeddingProvider(dimension=16)
    retriever = DenseRetriever(provider, InMemoryVectorStore(dimension=16))
    with pytest.raises(RetrievalValidationError, match="top_k"):
        await retriever.retrieve("valid query", top_k=0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "matching_content"),
    [
        ("sales decline", "regional sales decline analysis"),
        ("华东销售", "华东区域销售下降分析"),
    ],
)
async def test_english_and_chinese_retrieval_chains_run(query: str, matching_content: str) -> None:
    provider = DeterministicHashEmbeddingProvider(dimension=256)
    store = InMemoryVectorStore(dimension=256)
    await DenseIndexer(provider, store).index(
        [
            make_chunk("matching", matching_content),
            make_chunk("other", "employee vacation policy"),
        ]
    )
    results = await DenseRetriever(provider, store).retrieve(query, top_k=1)
    assert results[0].record_id == "matching"


@pytest.mark.asyncio
async def test_repeated_runs_produce_deterministic_results() -> None:
    provider = DeterministicHashEmbeddingProvider(dimension=128)
    store = InMemoryVectorStore(dimension=128)
    await DenseIndexer(provider, store).index(
        [make_chunk("b", "same token"), make_chunk("a", "same token")]
    )
    retriever = DenseRetriever(provider, store)
    first = await retriever.retrieve("same token", top_k=2)
    second = await retriever.retrieve("same token", top_k=2)
    assert first == second
    assert [result.record_id for result in first] == ["a", "b"]


def test_dense_services_reject_provider_store_dimension_mismatch() -> None:
    provider = DeterministicHashEmbeddingProvider(dimension=8)
    store = InMemoryVectorStore(dimension=16)
    with pytest.raises(RetrievalValidationError, match="dimensions must match"):
        DenseIndexer(provider, store)
    with pytest.raises(RetrievalValidationError, match="dimensions must match"):
        DenseRetriever(provider, store)
