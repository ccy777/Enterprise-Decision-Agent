from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from decision_agent.evaluation.retrieval_evidence import verify_retrieval_evidence

_ROOT = Path(__file__).resolve().parents[3]


def test_frozen_retrieval_evidence_verifies_without_models() -> None:
    result = verify_retrieval_evidence(_ROOT)
    assert result.verified
    assert (result.query_count, result.answerable_query_count, result.unanswerable_query_count) == (
        50,
        46,
        4,
    )
    assert result.rrf_child_hit_at_1 == 39 / 46
    assert result.rrf_child_mrr_at_5 == pytest.approx(42.166666666666664 / 46)
    assert result.reranker_child_hit_at_1 == 43 / 46
    assert result.reranker_child_mrr_at_5 == 44.5 / 46
    assert result.rrf_child_hit_at_5 == result.reranker_child_hit_at_5 == 1.0
    assert result.verified_source_hash_count == 6
    assert result.verified_generated_hash_count == 2
    assert round(result.hit_at_1_gain_percentage_points, 2) == 8.70
    assert round(result.mrr_at_5_gain_percentage_points, 2) == 5.07


@pytest.mark.parametrize(
    "relative",
    [
        "artifacts/evaluation/m2c2a2_query_results.jsonl",
        "artifacts/evaluation/m2c2a2_retrieval_baseline.json",
        (
            "artifacts/evaluation/m2c2a2-retrieval-evidence-v1/source_snapshot/"
            "retrieval_ground_truth.jsonl"
        ),
        ("artifacts/evaluation/m2c2a2-retrieval-evidence-v1/source_snapshot/query_blueprint.jsonl"),
        "artifacts/evaluation/m2c2a2-retrieval-evidence-v1/manifest.json",
    ],
)
def test_retrieval_evidence_mutations_fail_closed(tmp_path: Path, relative: str) -> None:
    needed = [
        "artifacts/evaluation/m2c2a2_retrieval_baseline.json",
        "artifacts/evaluation/m2c2a2_query_results.jsonl",
        "artifacts/evaluation/m2c2a2_failure_cases.jsonl",
        "artifacts/evaluation/m2c2a2_runtime_profile.json",
        ("artifacts/evaluation/m2c2a2-retrieval-evidence-v1/source_snapshot/child_chunks.jsonl"),
        (
            "artifacts/evaluation/m2c2a2-retrieval-evidence-v1/source_snapshot/"
            "clause_chunk_map.jsonl"
        ),
        (
            "artifacts/evaluation/m2c2a2-retrieval-evidence-v1/source_snapshot/"
            "m2c1_parent_child_summary.json"
        ),
        ("artifacts/evaluation/m2c2a2-retrieval-evidence-v1/source_snapshot/parent_chunks.jsonl"),
        ("artifacts/evaluation/m2c2a2-retrieval-evidence-v1/source_snapshot/query_blueprint.jsonl"),
        (
            "artifacts/evaluation/m2c2a2-retrieval-evidence-v1/source_snapshot/"
            "retrieval_ground_truth.jsonl"
        ),
        "artifacts/evaluation/m2c2a2-retrieval-evidence-v1/manifest.json",
    ]
    for item in needed:
        target = tmp_path / item
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_ROOT / item, target)
    target = tmp_path / relative
    content = target.read_text(encoding="utf-8")
    if target.name == "manifest.json":
        content = content.replace('"schema_version": "1.0"', '"schema_version": "9.9"')
    else:
        content += " "
    target.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=r"hash mismatch|manifest mismatch"):
        verify_retrieval_evidence(tmp_path)
