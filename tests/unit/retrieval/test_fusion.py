"""Tests for deterministic rank-only Reciprocal Rank Fusion."""

import math

import pytest
from pydantic import ValidationError

from decision_agent.exceptions import RetrievalValidationError
from decision_agent.retrieval.fusion import FusionCandidate, reciprocal_rank_fusion


def candidate(
    source: str,
    rank: int,
    document_id: str,
    *,
    score: float | None = None,
    metadata: dict[str, str | int | float | bool | None] | None = None,
) -> FusionCandidate:
    return FusionCandidate(
        source_name=source,
        rank=rank,
        document_id=document_id,
        record_id=f"{source}-{document_id}",
        source_score=score,
        content=f"content-{document_id}",
        metadata=metadata or {"source": source},
        provenance={"retriever": source},
    )


@pytest.mark.parametrize("rrf_k", [0, -1, math.inf, math.nan, True])
def test_invalid_rrf_k_is_rejected(rrf_k: float) -> None:
    with pytest.raises(RetrievalValidationError, match="rrf_k"):
        reciprocal_rank_fusion({"dense": [candidate("dense", 1, "a")]}, rrf_k=rrf_k)


@pytest.mark.parametrize("weight", [0, -1, math.inf, math.nan, True])
def test_invalid_source_weight_is_rejected(weight: float) -> None:
    with pytest.raises(RetrievalValidationError, match="weight"):
        reciprocal_rank_fusion(
            {"dense": [candidate("dense", 1, "a")]},
            source_weights={"dense": weight},
        )


def test_empty_source_mapping_is_rejected() -> None:
    with pytest.raises(RetrievalValidationError, match="at least one source"):
        reciprocal_rank_fusion({})


def test_empty_source_list_is_rejected() -> None:
    with pytest.raises(RetrievalValidationError, match="cannot be empty"):
        reciprocal_rank_fusion({"dense": []})


def test_weight_names_must_exactly_match_source_names() -> None:
    with pytest.raises(RetrievalValidationError, match="match"):
        reciprocal_rank_fusion(
            {"dense": [candidate("dense", 1, "a")]},
            source_weights={"bm25": 1.0},
        )


def test_candidate_source_name_must_match_mapping() -> None:
    with pytest.raises(RetrievalValidationError, match="source_name"):
        reciprocal_rank_fusion({"dense": [candidate("bm25", 1, "a")]})


def test_single_source_preserves_rank_order() -> None:
    results = reciprocal_rank_fusion(
        {"dense": [candidate("dense", 1, "a"), candidate("dense", 2, "b")]}
    )
    assert [item.document_id for item in results] == ["a", "b"]
    assert [item.final_rank for item in results] == [1, 2]


def test_two_disjoint_sources_keep_all_documents() -> None:
    results = reciprocal_rank_fusion(
        {
            "dense": [candidate("dense", 1, "a")],
            "bm25": [candidate("bm25", 1, "b")],
        }
    )
    assert {item.document_id for item in results} == {"a", "b"}
    assert all(item.matched_source_count == 1 for item in results)


def test_same_document_is_merged_across_sources() -> None:
    results = reciprocal_rank_fusion(
        {
            "dense": [candidate("dense", 1, "a", score=0.8)],
            "bm25": [candidate("bm25", 1, "b"), candidate("bm25", 2, "a", score=4.2)],
        }
    )
    merged = next(item for item in results if item.document_id == "a")
    assert merged.matched_source_count == 2
    assert {item.source_name for item in merged.source_contributions} == {"dense", "bm25"}


def test_contributions_and_fused_score_follow_rrf_formula() -> None:
    result = reciprocal_rank_fusion(
        {
            "dense": [candidate("dense", 1, "a")],
            "bm25": [candidate("bm25", 1, "b"), candidate("bm25", 2, "a")],
        },
        rrf_k=60,
        source_weights={"dense": 1.0, "bm25": 2.0},
    )[0]
    expected = 1 / 61 + 2 / 62
    assert result.document_id == "a"
    assert result.fused_score == pytest.approx(expected)
    assert math.fsum(item.contribution for item in result.source_contributions) == pytest.approx(
        expected
    )


def test_source_scores_are_diagnostic_only() -> None:
    low = reciprocal_rank_fusion({"dense": [candidate("dense", 1, "a", score=-1000.0)]})
    high = reciprocal_rank_fusion({"dense": [candidate("dense", 1, "a", score=1000.0)]})
    assert low[0].fused_score == high[0].fused_score


@pytest.mark.parametrize(
    "ranks",
    [
        [1, 1],
        [1, 3],
        [2, 1],
    ],
)
def test_ranks_must_be_unique_consecutive_and_ordered(ranks: list[int]) -> None:
    items = [candidate("dense", rank, f"doc-{index}") for index, rank in enumerate(ranks)]
    with pytest.raises(RetrievalValidationError, match="consecutive"):
        reciprocal_rank_fusion({"dense": items})


def test_duplicate_document_in_one_source_is_rejected() -> None:
    with pytest.raises(RetrievalValidationError, match="duplicate document_id"):
        reciprocal_rank_fusion({"dense": [candidate("dense", 1, "a"), candidate("dense", 2, "a")]})


def test_equal_fused_scores_use_document_id_tie_breaker() -> None:
    results = reciprocal_rank_fusion(
        {
            "dense": [candidate("dense", 1, "b"), candidate("dense", 2, "a")],
            "bm25": [candidate("bm25", 1, "a"), candidate("bm25", 2, "b")],
        }
    )
    assert [item.document_id for item in results] == ["a", "b"]


def test_repeated_calls_are_identical_and_mapping_order_independent() -> None:
    dense = [candidate("dense", 1, "a"), candidate("dense", 2, "b")]
    bm25 = [candidate("bm25", 1, "b"), candidate("bm25", 2, "a")]
    first = reciprocal_rank_fusion({"dense": dense, "bm25": bm25})
    second = reciprocal_rank_fusion({"bm25": bm25, "dense": dense})
    assert first == second


def test_payload_conflict_uses_rank_then_source_name_and_deep_copies() -> None:
    dense = candidate("dense", 1, "a", metadata={"owner": "dense"})
    bm25 = candidate("bm25", 1, "a", metadata={"owner": "bm25"})
    dense.content = "dense payload"
    bm25.content = "bm25 payload"
    first = reciprocal_rank_fusion({"dense": [dense], "bm25": [bm25]})[0]
    second = reciprocal_rank_fusion({"bm25": [bm25], "dense": [dense]})[0]
    assert first == second
    assert first.content == "bm25 payload"
    assert first.metadata == {"owner": "bm25"}
    bm25.metadata["owner"] = "changed"
    assert first.metadata == {"owner": "bm25"}


def test_inputs_and_metadata_are_not_modified_or_exposed() -> None:
    metadata = {"position": 1}
    original = candidate("dense", 1, "a", metadata=metadata)
    snapshot = original.model_copy(deep=True)
    result = reciprocal_rank_fusion({"dense": [original]})[0]
    assert original == snapshot
    metadata["position"] = 2
    original.metadata["position"] = 3
    result.metadata["position"] = 4
    assert reciprocal_rank_fusion({"dense": [snapshot]})[0].metadata == {"position": 1}


@pytest.mark.parametrize("top_k", [0, -1, True])
def test_invalid_top_k_is_rejected(top_k: int) -> None:
    with pytest.raises(RetrievalValidationError, match="top_k"):
        reciprocal_rank_fusion({"dense": [candidate("dense", 1, "a")]}, top_k=top_k)


def test_top_k_larger_than_unique_candidate_count_returns_all() -> None:
    results = reciprocal_rank_fusion(
        {"dense": [candidate("dense", 1, "a"), candidate("dense", 2, "b")]}, top_k=10
    )
    assert len(results) == 2


def test_non_finite_source_score_is_rejected_by_candidate_contract() -> None:
    with pytest.raises(ValidationError):
        candidate("dense", 1, "a", score=math.inf)


@pytest.mark.parametrize("rank", [True, 1.0, "1"])
def test_candidate_rank_requires_a_strict_integer(rank: object) -> None:
    with pytest.raises(ValidationError):
        candidate("dense", rank, "a")  # type: ignore[arg-type]


def test_all_fused_scores_and_contributions_are_finite_positive_numbers() -> None:
    results = reciprocal_rank_fusion(
        {
            "dense": [candidate("dense", 1, "a"), candidate("dense", 2, "b")],
            "bm25": [candidate("bm25", 1, "b"), candidate("bm25", 2, "a")],
        }
    )
    values = [item.fused_score for item in results]
    values.extend(
        contribution.contribution for item in results for contribution in item.source_contributions
    )
    assert all(math.isfinite(value) and value > 0 for value in values)
