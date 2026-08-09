"""Tests for deterministic in-memory cosine vector storage."""

import math

import pytest
from pydantic import ValidationError

from decision_agent.exceptions import RetrievalValidationError
from decision_agent.retrieval import (
    InMemoryVectorStore,
    VectorRecord,
    VectorSearchFilter,
    VectorSearchResult,
    VectorStore,
)


def make_record(
    record_id: str,
    vector: list[float],
    *,
    document_id: str = "doc-1",
    content: str | None = None,
    metadata: dict[str, str | int | float | bool | None] | None = None,
) -> VectorRecord:
    return VectorRecord(
        record_id=record_id,
        parent_id=f"parent-{record_id}",
        document_id=document_id,
        document_version="v1",
        content=content or f"content-{record_id}",
        vector=vector,
        source="report.txt",
        page_number=1,
        metadata=metadata if metadata is not None else {"record": record_id},
    )


@pytest.mark.asyncio
async def test_empty_store_search_returns_empty_list() -> None:
    assert await InMemoryVectorStore(dimension=2).search([1.0, 0.0], 3) == []


@pytest.mark.asyncio
async def test_upsert_updates_count_and_result_counts() -> None:
    store = InMemoryVectorStore(dimension=2)
    result = await store.upsert([make_record("a", [1.0, 0.0]), make_record("b", [0.0, 1.0])])
    assert result.attempted_count == 2
    assert result.inserted_count == 2
    assert result.updated_count == 0
    assert await store.count() == 2


@pytest.mark.asyncio
async def test_list_record_ids_returns_immutable_snapshot() -> None:
    store = InMemoryVectorStore(dimension=2)
    await store.upsert([make_record("a", [1.0, 0.0]), make_record("b", [0.0, 1.0])])

    record_ids = await store.list_record_ids()
    await store.upsert([make_record("c", [1.0, 1.0])])

    assert record_ids == frozenset({"a", "b"})
    assert isinstance(record_ids, frozenset)
    assert await store.list_record_ids() == frozenset({"a", "b", "c"})


@pytest.mark.asyncio
async def test_same_id_upsert_is_idempotent() -> None:
    store = InMemoryVectorStore(dimension=2)
    record = make_record("a", [1.0, 0.0])
    await store.upsert([record])
    result = await store.upsert([record])
    assert result.inserted_count == 0
    assert result.updated_count == 1
    assert await store.count() == 1


@pytest.mark.asyncio
async def test_same_id_upsert_replaces_content_and_vector() -> None:
    store = InMemoryVectorStore(dimension=2)
    await store.upsert([make_record("a", [1.0, 0.0], content="old")])
    await store.upsert([make_record("a", [0.0, 1.0], content="new")])
    results = await store.search([0.0, 1.0], 1)
    assert results[0].content == "new"
    assert results[0].score == pytest.approx(1.0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("invalid_vector", "message"),
    [
        ([], "model validation"),
        ([1.0], "dimension"),
        ([math.nan, 0.0], "model validation"),
        ([0.0, 0.0], "zero vector"),
    ],
)
async def test_invalid_record_makes_entire_upsert_batch_atomic(
    invalid_vector: list[float], message: str
) -> None:
    store = InMemoryVectorStore(dimension=2)
    await store.upsert([make_record("existing", [1.0, 0.0], content="original")])
    invalid_replacement = make_record(
        "existing", [0.0, 1.0], content="must-not-replace"
    ).model_copy(update={"vector": invalid_vector})

    with pytest.raises(RetrievalValidationError, match=message):
        await store.upsert([make_record("new-valid", [0.0, 1.0]), invalid_replacement])

    assert await store.count() == 1
    existing = await store.search([1.0, 0.0], 5)
    assert [result.record_id for result in existing] == ["existing"]
    assert existing[0].content == "original"


@pytest.mark.asyncio
async def test_duplicate_record_id_in_same_batch_is_rejected_atomically() -> None:
    store = InMemoryVectorStore(dimension=2)
    await store.upsert([make_record("existing", [1.0, 0.0])])

    with pytest.raises(RetrievalValidationError, match="duplicate record_id"):
        await store.upsert(
            [
                make_record("duplicate", [1.0, 0.0], content="first"),
                make_record("duplicate", [0.0, 1.0], content="second"),
            ]
        )

    assert await store.count() == 1
    results = await store.search([1.0, 0.0], 5)
    assert [result.record_id for result in results] == ["existing"]


@pytest.mark.asyncio
async def test_cosine_similarity_orders_results_high_to_low() -> None:
    store = InMemoryVectorStore(dimension=2)
    await store.upsert(
        [
            make_record("exact", [1.0, 0.0]),
            make_record("partial", [1.0, 1.0]),
            make_record("opposite", [-1.0, 0.0]),
        ]
    )
    results = await store.search([1.0, 0.0], 3)
    assert [result.record_id for result in results] == ["exact", "partial", "opposite"]


@pytest.mark.asyncio
async def test_equal_scores_use_record_id_as_stable_tie_breaker() -> None:
    store = InMemoryVectorStore(dimension=2)
    await store.upsert([make_record("b", [1.0, 0.0]), make_record("a", [1.0, 0.0])])
    results = await store.search([1.0, 0.0], 2)
    assert [result.record_id for result in results] == ["a", "b"]


@pytest.mark.asyncio
async def test_top_k_limits_result_count() -> None:
    store = InMemoryVectorStore(dimension=2)
    await store.upsert([make_record(str(index), [1.0, float(index + 1)]) for index in range(4)])
    assert len(await store.search([1.0, 0.0], 2)) == 2


@pytest.mark.asyncio
async def test_invalid_top_k_is_rejected() -> None:
    with pytest.raises(RetrievalValidationError, match="top_k"):
        await InMemoryVectorStore(dimension=2).search([1.0, 0.0], 0)


@pytest.mark.asyncio
async def test_record_dimension_mismatch_is_rejected() -> None:
    with pytest.raises(RetrievalValidationError, match="dimension"):
        await InMemoryVectorStore(dimension=2).upsert([make_record("a", [1.0])])


@pytest.mark.asyncio
async def test_query_dimension_mismatch_is_rejected() -> None:
    with pytest.raises(RetrievalValidationError, match="dimension"):
        await InMemoryVectorStore(dimension=2).search([1.0], 1)


@pytest.mark.parametrize("invalid_value", [math.nan, math.inf, -math.inf])
def test_vector_record_rejects_non_finite_values(invalid_value: float) -> None:
    with pytest.raises(ValidationError, match="finite"):
        make_record("invalid", [invalid_value, 0.0])


def test_vector_record_rejects_empty_vector() -> None:
    with pytest.raises(ValidationError):
        make_record("empty", [])


def test_search_result_rejects_non_finite_score() -> None:
    with pytest.raises(ValidationError, match="finite"):
        VectorSearchResult(
            record_id="a",
            score=math.nan,
            content="content",
            parent_id="parent-a",
            document_id="doc-a",
            document_version="v1",
        )


def test_search_filter_rejects_ambiguous_single_and_multiple_forms() -> None:
    with pytest.raises(ValidationError, match="either"):
        VectorSearchFilter(document_id="doc-a", document_ids=("doc-b",))


@pytest.mark.asyncio
async def test_store_rejects_non_finite_query_values() -> None:
    with pytest.raises(RetrievalValidationError, match="finite"):
        await InMemoryVectorStore(dimension=2).search([math.nan, 0.0], 1)


@pytest.mark.asyncio
async def test_single_document_filter_limits_results() -> None:
    store = InMemoryVectorStore(dimension=2)
    await store.upsert(
        [
            make_record("a", [1.0, 0.0], document_id="doc-a"),
            make_record("b", [1.0, 0.0], document_id="doc-b"),
        ]
    )
    results = await store.search([1.0, 0.0], 5, VectorSearchFilter(document_id="doc-b"))
    assert [result.document_id for result in results] == ["doc-b"]


@pytest.mark.asyncio
async def test_multiple_document_filter_limits_results() -> None:
    store = InMemoryVectorStore(dimension=2)
    await store.upsert(
        [
            make_record("a", [1.0, 0.0], document_id="doc-a"),
            make_record("b", [1.0, 0.0], document_id="doc-b"),
            make_record("c", [1.0, 0.0], document_id="doc-c"),
        ]
    )
    results = await store.search([1.0, 0.0], 5, VectorSearchFilter(document_ids=("doc-a", "doc-c")))
    assert [result.document_id for result in results] == ["doc-a", "doc-c"]


@pytest.mark.asyncio
async def test_delete_by_document_returns_actual_deleted_count() -> None:
    store = InMemoryVectorStore(dimension=2)
    await store.upsert(
        [
            make_record("a", [1.0, 0.0], document_id="doc-delete"),
            make_record("b", [0.0, 1.0], document_id="doc-delete"),
            make_record("c", [1.0, 1.0], document_id="doc-keep"),
        ]
    )
    assert await store.delete_by_document("doc-delete") == 2
    assert await store.delete_by_document("doc-delete") == 0
    assert await store.count() == 1


@pytest.mark.asyncio
async def test_store_and_results_do_not_expose_internal_metadata_references() -> None:
    store = InMemoryVectorStore(dimension=2)
    original_metadata = {"record": "a"}
    record = make_record("a", [1.0, 0.0], metadata=original_metadata)
    await store.upsert([record])
    original_metadata["record"] = "original-mutated"
    record.metadata["record"] = "caller-mutated"

    first = await store.search([1.0, 0.0], 1)
    assert first[0].metadata["record"] == "a"
    first[0].metadata["record"] = "result-mutated"
    second = await store.search([1.0, 0.0], 1)
    assert second[0].metadata["record"] == "a"


def test_invalid_store_dimension_is_rejected() -> None:
    with pytest.raises(RetrievalValidationError, match="dimension"):
        InMemoryVectorStore(dimension=0)


def test_store_satisfies_vector_store_protocol() -> None:
    assert isinstance(InMemoryVectorStore(dimension=2), VectorStore)
