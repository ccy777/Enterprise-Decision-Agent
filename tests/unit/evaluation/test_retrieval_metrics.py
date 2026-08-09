"""Tests for binary relevance retrieval metrics and aggregation."""

import pytest

from decision_agent.evaluation.retrieval_metrics import aggregate_metrics, compute_query_metrics
from decision_agent.exceptions import EvaluationValidationError


def test_hit_rate_at_one_is_one_for_first_rank_hit() -> None:
    assert compute_query_metrics(["relevant"], ["relevant"], ks=(1,))["hit_rate_at_1"] == 1.0


def test_hit_rate_at_one_is_zero_for_miss() -> None:
    assert compute_query_metrics(["other"], ["relevant"], ks=(1,))["hit_rate_at_1"] == 0.0


def test_hit_rate_at_three_detects_later_hit() -> None:
    metrics = compute_query_metrics(["a", "b", "relevant"], ["relevant"], ks=(3,))
    assert metrics["hit_rate_at_3"] == 1.0


def test_recall_for_single_relevant_document() -> None:
    metrics = compute_query_metrics(["relevant"], ["relevant"], ks=(1,))
    assert metrics["recall_at_1"] == 1.0


def test_recall_for_multiple_relevant_documents() -> None:
    metrics = compute_query_metrics(["a", "b"], ["a", "b"], ks=(2,))
    assert metrics["recall_at_2"] == 1.0


def test_recall_reports_partial_multi_relevant_hit() -> None:
    metrics = compute_query_metrics(["a", "other"], ["a", "b"], ks=(2,))
    assert metrics["recall_at_2"] == 0.5


def test_k_larger_than_available_ranking_uses_all_available_results() -> None:
    metrics = compute_query_metrics(["a", "b"], ["a", "c"], ks=(5,), mrr_k=5)

    assert metrics == {
        "hit_rate_at_5": 1.0,
        "recall_at_5": 0.5,
        "mrr_at_5": 1.0,
    }


def test_mrr_is_one_for_first_rank_hit() -> None:
    metrics = compute_query_metrics(["relevant", "other"], ["relevant"], ks=(2,), mrr_k=2)
    assert metrics["mrr_at_2"] == 1.0


def test_mrr_is_one_third_for_third_rank_hit() -> None:
    metrics = compute_query_metrics(["a", "b", "relevant"], ["relevant"], ks=(3,), mrr_k=3)
    assert metrics["mrr_at_3"] == pytest.approx(1 / 3)


def test_mrr_is_zero_when_top_k_misses() -> None:
    metrics = compute_query_metrics(["a", "b"], ["relevant"], ks=(2,), mrr_k=2)
    assert metrics["mrr_at_2"] == 0.0


@pytest.mark.parametrize("invalid_k", [0, -1, True])
def test_non_positive_or_boolean_k_is_rejected(invalid_k: int) -> None:
    with pytest.raises(EvaluationValidationError, match="positive integer"):
        compute_query_metrics(["a"], ["a"], ks=(invalid_k,))


def test_duplicate_ranked_document_id_is_rejected() -> None:
    with pytest.raises(EvaluationValidationError, match="ranked document IDs"):
        compute_query_metrics(["a", "a"], ["a"], ks=(1,))


def test_empty_relevant_document_ids_are_rejected() -> None:
    with pytest.raises(EvaluationValidationError, match="relevant document IDs"):
        compute_query_metrics(["a"], [], ks=(1,))


def test_duplicate_relevant_document_ids_are_rejected() -> None:
    with pytest.raises(EvaluationValidationError, match="relevant document IDs"):
        compute_query_metrics(["a"], ["a", "a"], ks=(1,))


def test_macro_average_is_arithmetic_mean() -> None:
    overall, _ = aggregate_metrics(
        [("销售", {"hit_rate_at_1": 1.0}), ("销售", {"hit_rate_at_1": 0.0})]
    )
    assert overall == {"hit_rate_at_1": 0.5}


def test_category_metrics_are_aggregated_independently() -> None:
    _, categories = aggregate_metrics(
        [
            ("销售", {"hit_rate_at_1": 1.0}),
            ("销售", {"hit_rate_at_1": 0.0}),
            ("人力资源", {"hit_rate_at_1": 1.0}),
        ]
    )
    assert categories["销售"]["hit_rate_at_1"] == 0.5
    assert categories["人力资源"]["hit_rate_at_1"] == 1.0


def test_metric_values_are_always_bounded() -> None:
    metrics = compute_query_metrics(["x", "a", "b"], ["a", "b"], ks=(1, 3), mrr_k=3)
    assert all(0.0 <= value <= 1.0 for value in metrics.values())


def test_category_metrics_do_not_leak_across_unequal_category_sizes() -> None:
    overall, categories = aggregate_metrics(
        [
            ("category-a", {"recall_at_1": 0.0}),
            ("category-a", {"recall_at_1": 0.0}),
            ("category-a", {"recall_at_1": 0.0}),
            ("category-b", {"recall_at_1": 1.0}),
        ]
    )

    assert overall["recall_at_1"] == 0.25
    assert categories == {
        "category-a": {"recall_at_1": 0.0},
        "category-b": {"recall_at_1": 1.0},
    }
