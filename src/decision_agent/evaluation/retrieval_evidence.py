"""No-model verifier for the frozen 50-query M2C2-A retrieval evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from decision_agent.evaluation.enterprise_retrieval_baseline import build_ranking_digest


class RetrievalEvidenceVerification(BaseModel):
    """Closed verification result for the committed retrieval evidence bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1"
    verified: bool
    query_count: int = Field(ge=0)
    answerable_query_count: int = Field(ge=0)
    unanswerable_query_count: int = Field(ge=0)
    rrf_child_hit_at_1: float
    rrf_child_mrr_at_5: float
    reranker_child_hit_at_1: float
    reranker_child_mrr_at_5: float
    rrf_child_hit_at_5: float
    reranker_child_hit_at_5: float
    hit_at_1_gain_percentage_points: float
    mrr_at_5_gain_percentage_points: float
    deterministic_ranking_digest: str
    verified_source_hash_count: int = Field(ge=0)
    verified_generated_hash_count: int = Field(ge=0)


_EXPECTED_SOURCE_HASHES = {
    "child_chunks.jsonl": "fc1f6fdcc28f02158a8cdca12160403355777f48af7f0bb3d71f16e6b6fb77db",
    "clause_chunk_map.jsonl": "0264548723264f500c9ab146df012da9e2e792ea441e0006722b22d778768981",
    "m2c1_parent_child_summary.json": (
        "1ce62aeb6fb26a98bf9e20f6df7e64b86e3e0e2974024614e54f9200688dfd48"
    ),
    "parent_chunks.jsonl": "4c79261da76ac7cc7a1f479c2b8447865c35cf828602d2c5979244cbb4d56c47",
    "query_blueprint.jsonl": "473a2c70b8139609dc842c2ce9d4f761154522c18f194580b3cbb79bf6937ee6",
    "retrieval_ground_truth.jsonl": (
        "c12772c471d710dd7611871619a227133f79f893804bd77b8452fefc6de1574b"
    ),
}
_SNAPSHOT_PATHS = {
    "child_chunks.jsonl": (
        "artifacts/evaluation/m2c2a2-retrieval-evidence-v1/source_snapshot/child_chunks.jsonl"
    ),
    "clause_chunk_map.jsonl": (
        "artifacts/evaluation/m2c2a2-retrieval-evidence-v1/source_snapshot/clause_chunk_map.jsonl"
    ),
    "m2c1_parent_child_summary.json": (
        "artifacts/evaluation/m2c2a2-retrieval-evidence-v1/source_snapshot/"
        "m2c1_parent_child_summary.json"
    ),
    "parent_chunks.jsonl": (
        "artifacts/evaluation/m2c2a2-retrieval-evidence-v1/source_snapshot/parent_chunks.jsonl"
    ),
    "query_blueprint.jsonl": (
        "artifacts/evaluation/m2c2a2-retrieval-evidence-v1/source_snapshot/query_blueprint.jsonl"
    ),
    "retrieval_ground_truth.jsonl": (
        "artifacts/evaluation/m2c2a2-retrieval-evidence-v1/source_snapshot/"
        "retrieval_ground_truth.jsonl"
    ),
}
_EXPECTED_ARTIFACT_HASHES = {
    "m2c2a2_retrieval_baseline.json": (
        "2e3b75fdce011d9201bb7187572405ed9a362e1e2b4cc9659e68128664752278"
    ),
    "m2c2a2_query_results.jsonl": (
        "26558008740e62b22c6d72c0b826fa4bb7f1d4229d1a35e72ca03c816db5fa87"
    ),
    "m2c2a2_failure_cases.jsonl": (
        "b9f0be9bd6a68a4732bee1148b7f24d6967cdde4e46a266b1a6803631b39c0b3"
    ),
    "m2c2a2_runtime_profile.json": (
        "7f9191aeeb48bd120fcee40380ff5b00023b908f79b65359a2b380d80464bb0d"
    ),
}


def verify_retrieval_evidence(repository: Path) -> RetrievalEvidenceVerification:
    """Verify hashes, IDs, ranking digest, and published metrics without loading models."""
    repository = repository.resolve()
    artifact_dir = repository / "artifacts" / "evaluation"
    report_path = artifact_dir / "m2c2a2_retrieval_baseline.json"
    evidence_dir = artifact_dir / "m2c2a2-retrieval-evidence-v1"
    for name, expected_hash in _EXPECTED_ARTIFACT_HASHES.items():
        if _text_sha256(artifact_dir / name) != expected_hash:
            raise ValueError(f"retrieval formal artifact hash mismatch: {name}")
    report = _load_object(report_path)
    required_top_level = {
        "baseline_id",
        "dataset_id",
        "query_count",
        "answerable_query_count",
        "unanswerable_query_count",
        "stage_metrics",
        "source_hashes",
        "generated_file_hashes",
        "deterministic_ranking_digest",
        "ranking_digest",
        "determinism",
    }
    if not required_top_level <= set(report):
        raise ValueError("retrieval report is missing required evidence fields")
    if (
        report["baseline_id"] != "m2c2a2-enterprise-real-retrieval-baseline-v1"
        or report["query_count"] != 50
        or report["answerable_query_count"] != 46
        or report["unanswerable_query_count"] != 4
    ):
        raise ValueError("retrieval benchmark identity or counts changed")

    source_hashes = report["source_hashes"]
    if not isinstance(source_hashes, dict) or source_hashes != _EXPECTED_SOURCE_HASHES:
        raise ValueError("retrieval source hash set changed")
    for name, relative in _SNAPSHOT_PATHS.items():
        if _text_sha256(repository / relative) != source_hashes[name]:
            raise ValueError(f"retrieval source hash mismatch: {name}")

    manifest = _load_object(evidence_dir / "manifest.json")
    expected_manifest_artifacts = {
        f"artifacts/evaluation/{name}": digest for name, digest in _EXPECTED_ARTIFACT_HASHES.items()
    }
    expected_manifest_snapshots = {
        relative: source_hashes[name] for name, relative in _SNAPSHOT_PATHS.items()
    }
    if (
        manifest.get("schema_version") != "1.0"
        or manifest.get("evidence_id") != "m2c2a2-retrieval-evidence-v1"
        or manifest.get("formal_evaluation_commit") != "f5f540e84d0bb36cf640393f759a6b4e6f84ec38"
        or manifest.get("model_evaluation_rerun") is not False
        or manifest.get("source_data_classification")
        != "synthetic_enterprise_fixture_no_real_business_data"
        or manifest.get("formal_artifacts") != expected_manifest_artifacts
        or manifest.get("source_snapshot") != expected_manifest_snapshots
        or manifest.get("ranking_digest") != report["deterministic_ranking_digest"]
        or manifest.get("query_count") != 50
        or manifest.get("answerable_query_count") != 46
        or manifest.get("unanswerable_query_count") != 4
    ):
        raise ValueError("retrieval evidence manifest mismatch")

    generated_hashes = report["generated_file_hashes"]
    expected_generated = {
        "m2c2a2_failure_cases.jsonl",
        "m2c2a2_query_results.jsonl",
    }
    if not isinstance(generated_hashes, dict) or set(generated_hashes) != expected_generated:
        raise ValueError("retrieval generated hash set changed")
    for name, expected_hash in generated_hashes.items():
        if _text_sha256(artifact_dir / name) != expected_hash:
            raise ValueError(f"retrieval generated artifact hash mismatch: {name}")

    query_results = _load_jsonl(artifact_dir / "m2c2a2_query_results.jsonl")
    expected_ids = {f"M2C1-Q{index:03d}" for index in range(1, 51)}
    identifiers = [str(item.get("query_id")) for item in query_results]
    if len(query_results) != 50 or len(set(identifiers)) != 50 or set(identifiers) != expected_ids:
        raise ValueError("retrieval query IDs are not the exact frozen 50-case set")
    if (
        sum(bool(item.get("answerable")) for item in query_results) != 46
        or sum(not bool(item.get("answerable")) for item in query_results) != 4
    ):
        raise ValueError("retrieval query answerability counts changed")
    ranking_digest = build_ranking_digest(query_results)
    if (
        ranking_digest != report["deterministic_ranking_digest"]
        or ranking_digest != report["ranking_digest"]
        or report["determinism"]["run_a_ranking_digest"] != ranking_digest
        or report["determinism"]["run_b_ranking_digest"] != ranking_digest
    ):
        raise ValueError("retrieval deterministic ranking digest changed")

    metrics = report["stage_metrics"]
    rrf_hit_1 = _metric(metrics, "rrf", "hit_rate_at_1")
    rrf_mrr_5 = _metric(metrics, "rrf", "mrr_at_5")
    reranker_hit_1 = _metric(metrics, "reranker", "hit_rate_at_1")
    reranker_mrr_5 = _metric(metrics, "reranker", "mrr_at_5")
    rrf_hit_5 = _metric(metrics, "rrf", "hit_rate_at_5")
    reranker_hit_5 = _metric(metrics, "reranker", "hit_rate_at_5")
    expected_values = (
        (rrf_hit_1, 39 / 46),
        (rrf_mrr_5, 42.166666666666664 / 46),
        (reranker_hit_1, 43 / 46),
        (reranker_mrr_5, 44.5 / 46),
        (rrf_hit_5, 1.0),
        (reranker_hit_5, 1.0),
    )
    if any(actual != expected for actual, expected in expected_values):
        raise ValueError("retrieval headline metrics changed")
    hit_gain = (reranker_hit_1 - rrf_hit_1) * 100
    mrr_gain = (reranker_mrr_5 - rrf_mrr_5) * 100
    if round(hit_gain, 2) != 8.70 or round(mrr_gain, 2) != 5.07:
        raise ValueError("retrieval headline gains changed")

    return RetrievalEvidenceVerification(
        verified=True,
        query_count=50,
        answerable_query_count=46,
        unanswerable_query_count=4,
        rrf_child_hit_at_1=rrf_hit_1,
        rrf_child_mrr_at_5=rrf_mrr_5,
        reranker_child_hit_at_1=reranker_hit_1,
        reranker_child_mrr_at_5=reranker_mrr_5,
        rrf_child_hit_at_5=rrf_hit_5,
        reranker_child_hit_at_5=reranker_hit_5,
        hit_at_1_gain_percentage_points=hit_gain,
        mrr_at_5_gain_percentage_points=mrr_gain,
        deterministic_ranking_digest=ranking_digest,
        verified_source_hash_count=len(_SNAPSHOT_PATHS),
        verified_generated_hash_count=len(generated_hashes),
    )


def _metric(metrics: object, stage: str, name: str) -> float:
    try:
        item = metrics[stage]["metrics"][name]  # type: ignore[index]
        if item["denominator"] != 46 or item["value"] != item["numerator"] / 46:
            raise ValueError("retrieval metric counts do not reconcile")
        return float(item["value"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"retrieval metric is missing: {stage}.{name}") from exc


def _load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if any(not isinstance(value, dict) for value in values):
        raise ValueError(f"expected JSON objects: {path.name}")
    return values


def _text_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode("utf-8")).hexdigest()
