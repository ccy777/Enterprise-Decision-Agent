"""Opt-in integration checks for the real local BGE embedding model."""

from __future__ import annotations

import math
import os
import time

import pytest

from decision_agent.config import Settings
from decision_agent.domain import ChildChunk
from decision_agent.retrieval import (
    DenseIndexer,
    DenseRetriever,
    InMemoryVectorStore,
    SentenceTransformerEmbeddingProvider,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_EMBEDDING_INTEGRATION") != "1",
        reason="set RUN_EMBEDDING_INTEGRATION=1 to load the configured local embedding model",
    ),
]


def cosine(left: list[float], right: list[float]) -> float:
    """Return a dot product for already normalized vectors."""
    return sum(a * b for a, b in zip(left, right, strict=True))


def make_chunk(chunk_id: str, content: str) -> ChildChunk:
    """Create one minimal canonical child chunk for the dense retrieval flow."""
    return ChildChunk(
        chunk_id=chunk_id,
        parent_id=f"parent-{chunk_id}",
        document_id="doc-real-embedding",
        document_version="v1",
        content=content,
        source="real-embedding-integration.txt",
        page_number=1,
        start_offset=0,
        end_offset=len(content),
    )


@pytest.mark.asyncio
async def test_real_bge_model_and_in_memory_retrieval_chain(
    record_property: pytest.FixtureRequest,
) -> None:
    """Load the configured BGE model only under explicit integration opt-in."""
    settings = Settings(app_name="Embedding Integration", _env_file=None)
    assert settings.embedding_model_name == "BAAI/bge-small-zh-v1.5"
    assert settings.embedding_dimension == 512
    provider = SentenceTransformerEmbeddingProvider.from_settings(settings)

    query = "产品电池保修多久？"  # noqa: RUF001
    related = "产品电池保修期为一年。"
    unrelated = "公司食堂周五供应面条。"

    load_started = time.perf_counter()
    query_vector = await provider.embed_query(query)
    model_load_seconds = time.perf_counter() - load_started
    encode_started = time.perf_counter()
    document_vectors = await provider.embed_documents([related, unrelated])
    batch_encode_seconds = time.perf_counter() - encode_started
    query_started = time.perf_counter()
    warm_query_vector = await provider.embed_query(query)
    query_encode_seconds = time.perf_counter() - query_started
    instructed_query_vector = await provider.embed_query(
        settings.embedding_query_instruction + query
    )
    related_similarity = cosine(query_vector, document_vectors[0])
    unrelated_similarity = cosine(query_vector, document_vectors[1])
    record_property("model_load_seconds", model_load_seconds)
    record_property("batch_encode_seconds", batch_encode_seconds)
    record_property("query_encode_seconds", query_encode_seconds)
    record_property("related_similarity", related_similarity)
    record_property("unrelated_similarity", unrelated_similarity)
    print(
        "real_bge_observations "
        f"load_and_first_query_seconds={model_load_seconds:.6f} "
        f"batch_encode_seconds={batch_encode_seconds:.6f} "
        f"warm_query_encode_seconds={query_encode_seconds:.6f} "
        f"dimension={len(query_vector)} "
        f"related_similarity={related_similarity:.6f} "
        f"unrelated_similarity={unrelated_similarity:.6f}"
    )

    assert type(query_vector) is list
    assert all(type(value) is float for value in query_vector)
    assert all(type(vector) is list for vector in document_vectors)
    assert len(query_vector) == 512
    assert all(len(vector) == 512 for vector in document_vectors)
    assert all(
        math.isfinite(value) for vector in [query_vector, *document_vectors] for value in vector
    )
    assert all(
        math.sqrt(sum(value * value for value in vector)) == pytest.approx(1.0)
        for vector in [query_vector, *document_vectors]
    )
    assert warm_query_vector == pytest.approx(query_vector, abs=1e-6)
    assert instructed_query_vector == pytest.approx(query_vector, abs=1e-6)
    assert related_similarity > unrelated_similarity

    store = InMemoryVectorStore(dimension=512)
    await DenseIndexer(provider, store).index(
        [make_chunk("related", related), make_chunk("unrelated", unrelated)]
    )
    results = await DenseRetriever(provider, store).retrieve(query, top_k=2)

    assert results[0].record_id == "related"
