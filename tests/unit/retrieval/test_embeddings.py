"""Tests for the deterministic local hash embedding provider."""

import inspect
import math

import pytest

from decision_agent.exceptions import RetrievalValidationError
from decision_agent.retrieval import DeterministicHashEmbeddingProvider, EmbeddingProvider


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


@pytest.mark.asyncio
async def test_same_text_produces_identical_vectors() -> None:
    provider = DeterministicHashEmbeddingProvider(dimension=64)
    first = await provider.embed_query("Product A sales")
    second = await provider.embed_query("Product A sales")
    assert first == second


@pytest.mark.asyncio
async def test_different_text_produces_different_vectors() -> None:
    provider = DeterministicHashEmbeddingProvider(dimension=64)
    assert await provider.embed_query("sales increase") != await provider.embed_query(
        "service complaint"
    )


@pytest.mark.asyncio
async def test_vector_dimension_matches_configuration() -> None:
    provider = DeterministicHashEmbeddingProvider(dimension=37)
    assert len(await provider.embed_query("dimension test")) == 37


@pytest.mark.asyncio
async def test_vectors_are_l2_normalized() -> None:
    vector = await DeterministicHashEmbeddingProvider(dimension=64).embed_query("normalized vector")
    assert math.sqrt(sum(value * value for value in vector)) == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_related_english_text_has_higher_similarity_than_unrelated_text() -> None:
    provider = DeterministicHashEmbeddingProvider(dimension=256)
    query = await provider.embed_query("east china product sales")
    related = await provider.embed_query("product sales in east china improved")
    unrelated = await provider.embed_query("employee holiday policy")
    assert cosine(query, related) > cosine(query, unrelated)


@pytest.mark.asyncio
async def test_related_chinese_text_has_higher_similarity_than_unrelated_text() -> None:
    provider = DeterministicHashEmbeddingProvider(dimension=256)
    query = await provider.embed_query("华东区域产品销售")
    related = await provider.embed_query("华东产品销售下降")
    unrelated = await provider.embed_query("员工休假制度")
    assert cosine(query, related) > cosine(query, unrelated)


@pytest.mark.asyncio
async def test_empty_text_is_rejected() -> None:
    with pytest.raises(RetrievalValidationError, match="empty or whitespace"):
        await DeterministicHashEmbeddingProvider().embed_query("")


@pytest.mark.asyncio
async def test_whitespace_text_is_rejected() -> None:
    with pytest.raises(RetrievalValidationError, match="empty or whitespace"):
        await DeterministicHashEmbeddingProvider().embed_query(" \n\t ")


def test_invalid_embedding_dimension_is_rejected() -> None:
    with pytest.raises(RetrievalValidationError, match="dimension"):
        DeterministicHashEmbeddingProvider(dimension=0)


@pytest.mark.asyncio
async def test_empty_document_batch_returns_empty_batch() -> None:
    assert await DeterministicHashEmbeddingProvider().embed_documents([]) == []


def test_provider_uses_stable_hashlib_instead_of_builtin_hash() -> None:
    source = inspect.getsource(DeterministicHashEmbeddingProvider)
    assert "hashlib.sha256" in source
    assert "hash(" not in source


def test_provider_satisfies_embedding_protocol() -> None:
    assert isinstance(DeterministicHashEmbeddingProvider(), EmbeddingProvider)
