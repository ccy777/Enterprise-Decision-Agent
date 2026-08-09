"""Tests for deterministic pure-Python BM25 indexing and retrieval."""

import math

import pytest
from pydantic import ValidationError

from decision_agent.exceptions import RetrievalValidationError
from decision_agent.retrieval.bm25 import (
    BM25Document,
    BM25Index,
    BM25Retriever,
    BM25SearchResult,
)


def document(
    document_id: str,
    content: str,
    *,
    record_id: str | None = None,
    metadata: dict[str, str | int | float | bool | None] | None = None,
) -> BM25Document:
    return BM25Document(
        record_id=record_id or f"record-{document_id}",
        document_id=document_id,
        content=content,
        category="test-category",
        source="synthetic-test",
        metadata=metadata or {"document": document_id},
    )


def test_empty_corpus_is_rejected() -> None:
    with pytest.raises(RetrievalValidationError, match="corpus"):
        BM25Index([])


def test_duplicate_document_id_is_rejected() -> None:
    with pytest.raises(RetrievalValidationError, match="duplicate document_id"):
        BM25Index([document("same", "alpha"), document("same", "beta", record_id="other")])


def test_duplicate_record_id_is_rejected() -> None:
    with pytest.raises(RetrievalValidationError, match="duplicate record_id"):
        BM25Index(
            [
                document("doc-a", "alpha", record_id="same"),
                document("doc-b", "beta", record_id="same"),
            ]
        )


@pytest.mark.parametrize("content", ["", " \t\n"])
def test_empty_or_whitespace_document_is_rejected(content: str) -> None:
    with pytest.raises(ValidationError, match="content"):
        document("doc", content)


def test_document_without_searchable_tokens_is_rejected() -> None:
    with pytest.raises(RetrievalValidationError, match="document doc"):
        BM25Index([document("doc", ",.!?---")])


@pytest.mark.parametrize("k1", [0.0, -0.1])
def test_non_positive_k1_is_rejected(k1: float) -> None:
    with pytest.raises(RetrievalValidationError, match="k1"):
        BM25Index([document("doc", "alpha")], k1=k1)


@pytest.mark.parametrize("b", [-0.01, 1.01])
def test_b_outside_unit_interval_is_rejected(b: float) -> None:
    with pytest.raises(RetrievalValidationError, match="b"):
        BM25Index([document("doc", "alpha")], b=b)


def test_document_frequency_counts_documents_not_occurrences() -> None:
    index = BM25Index([document("a", "alpha alpha"), document("b", "alpha beta")])
    assert index.document_frequency("alpha") == 2
    assert index.document_frequency("beta") == 1


def test_average_document_length_uses_token_counts() -> None:
    index = BM25Index([document("a", "alpha"), document("b", "alpha beta gamma")])
    assert index.average_document_length == pytest.approx(2.0)


def test_rare_term_idf_is_higher_than_common_term_idf() -> None:
    index = BM25Index(
        [
            document("a", "common rare"),
            document("b", "common other"),
            document("c", "common third"),
        ]
    )
    assert index.inverse_document_frequency("rare") > index.inverse_document_frequency("common")


def test_term_frequency_saturates_when_length_normalization_disabled() -> None:
    index = BM25Index(
        [document("once", "alpha"), document("twice", "alpha alpha")],
        b=0.0,
    )
    once = index.score("alpha", "once")
    twice = index.score("alpha", "twice")
    assert twice > once
    assert twice < 2 * once


def test_length_normalization_favors_shorter_document_for_same_term_frequency() -> None:
    index = BM25Index(
        [
            document("short", "alpha"),
            document("long", "alpha beta gamma delta epsilon"),
        ]
    )
    assert index.score("alpha", "short") > index.score("alpha", "long")


def test_unmatched_term_scores_zero() -> None:
    index = BM25Index([document("doc", "alpha beta")])
    assert index.score("missing", "doc") == 0.0


def test_all_index_scores_are_finite_and_nonnegative() -> None:
    index = BM25Index([document("a", "alpha beta"), document("b", "alpha gamma")])
    scores = [index.score("alpha beta", document_id) for document_id in ("a", "b")]
    assert all(math.isfinite(score) and score >= 0 for score in scores)


@pytest.mark.parametrize("query", ["", " \t\n"])
def test_empty_or_whitespace_query_is_rejected(query: str) -> None:
    retriever = BM25Retriever(BM25Index([document("doc", "alpha")]))
    with pytest.raises(RetrievalValidationError, match="query"):
        retriever.retrieve(query, top_k=1)


def test_query_without_searchable_tokens_is_rejected() -> None:
    retriever = BM25Retriever(BM25Index([document("doc", "alpha")]))
    with pytest.raises(RetrievalValidationError, match="query"):
        retriever.retrieve(",.!?---", top_k=1)


@pytest.mark.parametrize("top_k", [0, -1, True])
def test_non_positive_or_boolean_top_k_is_rejected(top_k: int) -> None:
    retriever = BM25Retriever(BM25Index([document("doc", "alpha")]))
    with pytest.raises(RetrievalValidationError, match="top_k"):
        retriever.retrieve("alpha", top_k=top_k)


def test_top_k_larger_than_corpus_returns_all_positive_matches() -> None:
    retriever = BM25Retriever(BM25Index([document("a", "alpha one"), document("b", "alpha two")]))
    assert len(retriever.retrieve("alpha", top_k=10)) == 2


def test_only_positive_scoring_results_are_returned() -> None:
    retriever = BM25Retriever(BM25Index([document("match", "alpha"), document("miss", "beta")]))
    assert [result.document_id for result in retriever.retrieve("alpha", top_k=10)] == ["match"]


def test_results_sort_by_descending_score() -> None:
    retriever = BM25Retriever(
        BM25Index([document("one", "alpha"), document("two", "alpha alpha")], b=0.0)
    )
    results = retriever.retrieve("alpha", top_k=2)
    assert [result.document_id for result in results] == ["two", "one"]
    assert results[0].score > results[1].score


def test_equal_scores_use_document_then_record_id_tie_breaking() -> None:
    retriever = BM25Retriever(BM25Index([document("b", "alpha"), document("a", "alpha")]))
    assert [result.document_id for result in retriever.retrieve("alpha", top_k=2)] == ["a", "b"]


def test_result_preserves_provenance() -> None:
    original = document("doc", "alpha", metadata={"rank": 1})
    result = BM25Retriever(BM25Index([original])).retrieve("alpha", top_k=1)[0]
    assert result.record_id == original.record_id
    assert result.document_id == original.document_id
    assert result.content == original.content
    assert result.category == original.category
    assert result.source == original.source
    assert result.metadata == original.metadata
    assert result.rank == 1


def test_caller_and_result_metadata_cannot_mutate_index_state() -> None:
    metadata = {"rank": 1}
    original = document("doc", "alpha", metadata=metadata)
    retriever = BM25Retriever(BM25Index([original]))
    metadata["rank"] = 2
    original.metadata["rank"] = 3
    first = retriever.retrieve("alpha", top_k=1)
    assert first[0].metadata == {"rank": 1}
    first[0].metadata["rank"] = 4
    assert retriever.retrieve("alpha", top_k=1)[0].metadata == {"rank": 1}


def test_repeated_retrieval_is_identical() -> None:
    retriever = BM25Retriever(BM25Index([document("a", "产品A Q2"), document("b", "产品B Q1")]))
    assert retriever.retrieve("产品A Q2", top_k=2) == retriever.retrieve("产品A Q2", top_k=2)


def test_index_does_not_modify_input_document_sequence() -> None:
    documents = [document("a", "alpha"), document("b", "beta")]
    snapshot = [item.model_copy(deep=True) for item in documents]
    BM25Index(documents)
    assert documents == snapshot


def test_product_identifier_changes_ranking() -> None:
    retriever = BM25Retriever(
        BM25Index([document("product-a", "产品A电池保修"), document("product-b", "产品B电池保修")])
    )
    assert retriever.retrieve("产品A电池保修", top_k=1)[0].document_id == "product-a"


def test_quarter_identifier_changes_ranking() -> None:
    retriever = BM25Retriever(
        BM25Index([document("q1", "2026 Q1 收入"), document("q2", "2026 Q2 收入")])
    )
    assert retriever.retrieve("2026 Q2 收入", top_k=1)[0].document_id == "q2"


def test_search_result_rejects_non_finite_score() -> None:
    with pytest.raises(ValidationError, match="finite"):
        BM25SearchResult(
            rank=1,
            record_id="record",
            document_id="document",
            content="content",
            score=math.inf,
            category="category",
            source="source",
        )
