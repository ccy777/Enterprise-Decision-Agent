"""Offline verification for the public Retrieval Benchmark v2 evidence package."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PACKAGE_RELATIVE_PATH = Path("artifacts/public-evaluation/retrieval-v2")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _equivalent(actual: Any, expected: Any) -> bool:
    if isinstance(actual, float) or isinstance(expected, float):
        try:
            return math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _equivalent(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _equivalent(left, right) for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


def _ranked_ids(row: dict[str, Any], stage: str) -> list[str]:
    candidates = row[f"{stage}_candidates"]
    return [str(item["candidate_id"]) for item in candidates]


def _first_rank(
    ranked_ids: Sequence[str], relevant_ids: Sequence[str], *, limit: int = 5
) -> int | None:
    relevant = set(relevant_ids)
    for rank, candidate_id in enumerate(ranked_ids[:limit], start=1):
        if candidate_id in relevant:
            return rank
    return None


def _stage_metrics(rows: Sequence[dict[str, Any]], stage: str) -> dict[str, Any]:
    ranks = [row["recomputed_ranks"][stage] for row in rows if bool(row["answerable"])]
    denominator = len(ranks)
    hit_at_1 = sum(rank == 1 for rank in ranks)
    hit_at_5 = sum(rank is not None for rank in ranks)
    reciprocal_sum = sum(0.0 if rank is None else 1.0 / rank for rank in ranks)
    return {
        "answerable_denominator": denominator,
        "hit_at_1": {
            "numerator": hit_at_1,
            "denominator": denominator,
            "value": hit_at_1 / denominator if denominator else None,
        },
        "hit_at_5": {
            "numerator": hit_at_5,
            "denominator": denominator,
            "value": hit_at_5 / denominator if denominator else None,
        },
        "mrr_at_5": {
            "numerator": reciprocal_sum,
            "denominator": denominator,
            "value": reciprocal_sum / denominator if denominator else None,
        },
    }


def _slice_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rrf = _stage_metrics(rows, "rrf")
    reranker = _stage_metrics(rows, "reranker")
    rrf_hit = float(rrf["hit_at_1"]["value"])
    reranker_hit = float(reranker["hit_at_1"]["value"])
    rrf_mrr = float(rrf["mrr_at_5"]["value"])
    reranker_mrr = float(reranker["mrr_at_5"]["value"])
    return {
        "query_count": len(rows),
        "answerable_count": sum(bool(row["answerable"]) for row in rows),
        "unanswerable_count": sum(not bool(row["answerable"]) for row in rows),
        "rrf": rrf,
        "reranker": reranker,
        "hit_at_1_gain_percentage_points": (reranker_hit - rrf_hit) * 100,
        "mrr_at_5_gain_percentage_points": (reranker_mrr - rrf_mrr) * 100,
    }


def _records_with_recomputed_ranks(
    relevance: Sequence[dict[str, Any]], ranking: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], bool]:
    if [row.get("case_id") for row in relevance] != [row.get("case_id") for row in ranking]:
        return [], False

    output: list[dict[str, Any]] = []
    rank_fields_match = True
    shared_fields = (
        "case_id",
        "query",
        "answerable",
        "origin",
        "query_type",
        "evidence_mode",
        "expected_evidence_count",
    )
    for gold, result in zip(relevance, ranking, strict=True):
        if any(gold.get(field) != result.get(field) for field in shared_fields):
            return [], False
        relevant_ids = list(gold["adjudicated_relevant_child_ids"])
        if relevant_ids != list(result["adjudicated_relevant_child_ids"]):
            return [], False
        recomputed = {
            stage: (
                _first_rank(_ranked_ids(result, stage), relevant_ids)
                if bool(gold["answerable"])
                else None
            )
            for stage in ("rrf", "reranker")
        }
        rank_fields_match = rank_fields_match and (
            recomputed == result["adjudicated_first_relevant_ranks"]
        )
        output.append({**result, "recomputed_ranks": recomputed})
    return output, rank_fields_match


def verify(repository: Path) -> dict[str, Any]:
    package = repository / PACKAGE_RELATIVE_PATH
    manifest = _read_json(package / "manifest.json")
    metrics = _read_json(package / "metrics.json")
    relevance = _read_jsonl(package / "relevance_freeze.jsonl")
    ranking = _read_jsonl(package / "ranking_records.jsonl")
    rows, rank_fields_match = _records_with_recomputed_ranks(relevance, ranking)

    checks: dict[str, bool] = {}
    checks["artifact_hashes"] = all(
        _sha256(package / name) == expected
        for name, expected in manifest["artifact_hashes"].items()
    )
    checks["ordered_case_identity"] = bool(rows)
    checks["case_counts"] = (
        len(rows) == 200
        and sum(bool(row["answerable"]) for row in rows) == 160
        and sum(not bool(row["answerable"]) for row in rows) == 40
    )
    checks["evidence_modes"] = Counter(row["evidence_mode"] for row in rows) == Counter(
        {
            "single_window_sufficient": 135,
            "multi_evidence_required": 25,
            "unanswerable": 40,
        }
    )
    checks["answerable_relevance"] = all(
        row["adjudicated_relevant_child_ids"] for row in rows if bool(row["answerable"])
    )
    checks["unanswerable_relevance"] = all(
        not row["adjudicated_relevant_child_ids"] for row in rows if not bool(row["answerable"])
    )
    checks["multi_evidence_atoms"] = all(
        len(row["required_evidence"]) == int(row["expected_evidence_count"])
        for row in rows
        if row["evidence_mode"] == "multi_evidence_required"
    )
    checks["stored_rank_fields"] = rank_fields_match

    slices = {
        "v1_50_on_current_12_document_corpus": [
            row for row in rows if row["origin"] == "v1_frozen"
        ],
        "new_only_150": [row for row in rows if row["origin"] != "v1_frozen"],
        "v2_overall_200": rows,
        "single_window_sufficient_overall": [
            row for row in rows if row["evidence_mode"] == "single_window_sufficient"
        ],
        "multi_evidence_required_overall": [
            row for row in rows if row["evidence_mode"] == "multi_evidence_required"
        ],
    }
    for name, selected in slices.items():
        checks[f"metrics_{name}"] = _equivalent(
            _slice_metrics(selected), metrics["adjudicated_slices"][name]
        )

    overall = _slice_metrics(rows)
    checks["headline_metrics"] = _equivalent(overall, manifest["headline_metrics"])
    checks["identity_chain"] = (
        manifest["identities"]["candidate_sha256"] == metrics["candidate_sha256"]
        and manifest["identities"]["relevance_dataset_sha256"]
        == metrics["relevance_dataset_sha256"]
        and manifest["identities"]["ranking_digest"] == metrics["ranking_digest"]
    )

    return {
        "schema_version": "1.0",
        "verified": all(checks.values()),
        "checks": checks,
        "headline_metrics": overall,
        "identities": manifest["identities"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    result = verify(args.repository.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not result["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
