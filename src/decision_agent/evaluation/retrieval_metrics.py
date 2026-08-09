"""Binary document-relevance metrics for deterministic retrieval evaluation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from decision_agent.exceptions import EvaluationValidationError


def _validate_k(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise EvaluationValidationError("K must be a positive integer")


def _validate_unique_ids(values: Sequence[str], *, field_name: str, allow_empty: bool) -> None:
    if not allow_empty and not values:
        raise EvaluationValidationError(f"{field_name} cannot be empty")
    if any(not isinstance(value, str) or not value for value in values):
        raise EvaluationValidationError(f"{field_name} must contain nonempty strings")
    if len(values) != len(set(values)):
        raise EvaluationValidationError(f"{field_name} must be unique")


def compute_query_metrics(
    ranked_document_ids: Sequence[str],
    relevant_document_ids: Sequence[str],
    *,
    ks: Sequence[int] = (1, 3, 5),
    mrr_k: int = 5,
) -> dict[str, float]:
    """Compute HitRate@K, Recall@K, and first-relevant MRR@K for one query."""
    _validate_unique_ids(ranked_document_ids, field_name="ranked document IDs", allow_empty=True)
    _validate_unique_ids(
        relevant_document_ids, field_name="relevant document IDs", allow_empty=False
    )
    if not ks:
        raise EvaluationValidationError("at least one K is required")
    for k in ks:
        _validate_k(k)
    _validate_k(mrr_k)

    relevant = set(relevant_document_ids)
    metrics: dict[str, float] = {}
    for k in ks:
        retrieved_relevant = relevant.intersection(ranked_document_ids[:k])
        metrics[f"hit_rate_at_{k}"] = float(bool(retrieved_relevant))
        metrics[f"recall_at_{k}"] = len(retrieved_relevant) / len(relevant)

    first_relevant_rank = next(
        (
            rank
            for rank, document_id in enumerate(ranked_document_ids[:mrr_k], start=1)
            if document_id in relevant
        ),
        None,
    )
    metrics[f"mrr_at_{mrr_k}"] = 0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank
    return metrics


def aggregate_metrics(
    category_metrics: Sequence[tuple[str, Mapping[str, float]]],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Return macro averages overall and independently for each category."""
    if not category_metrics:
        raise EvaluationValidationError("query metrics cannot be empty")
    metric_names = tuple(category_metrics[0][1])
    if not metric_names:
        raise EvaluationValidationError("query metric mapping cannot be empty")

    totals = dict.fromkeys(metric_names, 0.0)
    category_totals: dict[str, dict[str, float]] = defaultdict(
        lambda: dict.fromkeys(metric_names, 0.0)
    )
    category_counts: dict[str, int] = defaultdict(int)
    for category, metrics in category_metrics:
        if not category.strip():
            raise EvaluationValidationError("metric category cannot be empty")
        if tuple(metrics) != metric_names:
            raise EvaluationValidationError("all query metric mappings must have identical keys")
        for name, value in metrics.items():
            numeric = float(value)
            if not 0.0 <= numeric <= 1.0:
                raise EvaluationValidationError("retrieval metric values must be within [0, 1]")
            totals[name] += numeric
            category_totals[category][name] += numeric
        category_counts[category] += 1

    overall = {name: value / len(category_metrics) for name, value in totals.items()}
    per_category = {
        category: {name: value / category_counts[category] for name, value in metrics.items()}
        for category, metrics in sorted(category_totals.items())
    }
    return overall, per_category
