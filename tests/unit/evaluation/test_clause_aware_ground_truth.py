"""Structural-only M2C-2B-1 Clause-aware artifact and pipeline coverage."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import pytest

import decision_agent.evaluation.enterprise_clause_aware_ground_truth as clause_aware_module
from decision_agent.evaluation.enterprise_clause_aware_ground_truth import (
    SUMMARY_FILENAME,
    VARIANT_DIRECTORY,
    build_and_write_clause_aware_ground_truth,
    build_clause_aware_ground_truth,
)
from decision_agent.exceptions import EvaluationValidationError
from decision_agent.retrieval import (
    DeterministicHashEmbeddingProvider,
    EnterpriseRetrievalPipeline,
    InMemoryVectorStore,
)
from decision_agent.retrieval.reranking import RerankCandidate, RerankedResult

ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = ROOT / "datasets/enterprise_kb/m2c1"
VARIANT_ROOT = DATASET_ROOT / VARIANT_DIRECTORY


class _FakeReranker:
    async def initialize(self) -> None:
        return None

    async def rerank(
        self, query: str, candidates: Sequence[RerankCandidate], *, top_k: int | None = None
    ) -> list[RerankedResult]:
        del query
        limit = len(candidates) if top_k is None else min(len(candidates), top_k)
        return [
            RerankedResult(
                final_rank=index,
                candidate_id=item.candidate_id,
                record_id=item.record_id,
                document_id=item.document_id,
                content=item.content,
                reranker_score=float(limit - index + 1),
                upstream_rank=item.upstream_rank,
                upstream_score=item.upstream_score,
                metadata=item.model_copy(deep=True).metadata,
                provenance=item.model_copy(deep=True).provenance,
            )
            for index, item in enumerate(candidates[:limit], start=1)
        ]


@pytest.fixture(scope="module")
def formal_build():
    return build_clause_aware_ground_truth(DATASET_ROOT)


def test_formal_counts_and_answerability_are_preserved(formal_build) -> None:
    assert len(formal_build.parent_chunks) == 58
    assert len(formal_build.child_chunks) == 149
    assert len(formal_build.clause_chunk_map) == 149
    assert len(formal_build.retrieval_ground_truth) == 60
    assert sum(record.answerable for record in formal_build.retrieval_ground_truth) == 56


def test_every_clause_has_exactly_one_parent_and_child(formal_build) -> None:
    assert all(
        len(record.parent_ids) == len(record.child_ids) == 1
        for record in formal_build.clause_chunk_map
    )


def test_child_boundaries_and_clause_metadata_are_single_clause(formal_build) -> None:
    assert all(
        child.metadata["clause_ids"] and len(child.metadata["clause_ids"]) == 1
        for child in formal_build.child_chunks
    )
    assert formal_build.summary["cross_clause_child_count"] == 0
    assert formal_build.summary["cross_section_child_count"] == 0


def test_ground_truth_separates_overlap_from_pure_hard_negative(formal_build) -> None:
    assert all(
        not set(record.relevant_child_ids) & set(record.hard_negative_child_ids)
        and not set(record.overlapping_child_ids) & set(record.hard_negative_child_ids)
        for record in formal_build.retrieval_ground_truth
    )
    assert formal_build.summary["overlapping_child_query_count"] == 0


def test_parent_and_child_records_reconstruct_source_offsets(formal_build) -> None:
    source_by_document = {
        path.stem.split("_")[0]: path.read_bytes().decode("utf-8")
        for path in (DATASET_ROOT / "documents").glob("*.md")
    }
    for record in (*formal_build.parent_chunks, *formal_build.child_chunks):
        source = source_by_document[record.document_id]
        assert record.content == source[record.start_offset : record.end_offset]


def test_output_order_and_ids_are_deterministic(formal_build) -> None:
    rebuilt = build_clause_aware_ground_truth(DATASET_ROOT)
    assert formal_build.serialized_files() == rebuilt.serialized_files()
    assert len({record.chunk_id for record in formal_build.parent_chunks}) == 58
    assert len({record.chunk_id for record in formal_build.child_chunks}) == 149


def test_summary_compares_fixed_window_from_real_artifact(formal_build) -> None:
    comparison = formal_build.summary["fixed_window_comparison"]
    assert comparison["fixed_window_parent_chunk_count"] == 36
    assert comparison["fixed_window_child_chunk_count"] == 101
    assert comparison["clause_aware_parent_chunk_count"] == 58
    assert comparison["clause_aware_child_chunk_count"] == 149


def test_summary_hashes_match_generated_jsonl_bytes(formal_build) -> None:
    serialized = formal_build.serialized_files()
    for filename, expected in formal_build.summary["generated_file_hashes"].items():
        assert hashlib.sha256(serialized[filename].encode("utf-8")).hexdigest() == expected
    assert SUMMARY_FILENAME not in formal_build.summary["generated_file_hashes"]


def test_formal_variant_files_are_isolated_from_fixed_window_outputs(formal_build) -> None:
    del formal_build
    assert VARIANT_ROOT.is_dir()
    assert (DATASET_ROOT / "generated/parent_chunks.jsonl").is_file()
    assert (VARIANT_ROOT / "parent_chunks.jsonl").read_bytes() != (
        DATASET_ROOT / "generated/parent_chunks.jsonl"
    ).read_bytes()


@pytest.mark.asyncio
async def test_variant_runs_complete_fake_pipeline_without_ground_truth_ranking() -> None:
    embedding = DeterministicHashEmbeddingProvider(dimension=64)
    pipeline = EnterpriseRetrievalPipeline(
        dataset_root=VARIANT_ROOT,
        embedding_provider=embedding,
        vector_store=InMemoryVectorStore(dimension=64),
        reranker=_FakeReranker(),
    )
    await pipeline.initialize()
    result = await pipeline.retrieve("产品 A 电池保修期限")

    assert (pipeline.parent_count, pipeline.child_count) == (58, 149)
    assert result.reranked_child_results[0].candidate_id.startswith("child_")
    assert result.evidence_context.evidence_items[0].parent_id.startswith("parent_")
    assert result.evidence_context.references[0].source.startswith("documents/")
    await pipeline.close()


def test_atomic_variant_write_uses_only_variant_and_summary_targets(tmp_path: Path) -> None:
    output = tmp_path / SUMMARY_FILENAME
    variant_output = tmp_path / "variant"
    build = build_and_write_clause_aware_ground_truth(
        DATASET_ROOT,
        summary_output=output,
        variant_output_dir=variant_output,
    )

    assert output.read_text(encoding="utf-8") == build.serialized_files()[SUMMARY_FILENAME]
    assert (variant_output / "parent_chunks.jsonl").is_file()


def test_input_change_rejects_variant_publication(monkeypatch, tmp_path: Path) -> None:
    original_files = {
        path.name: path.read_bytes() for path in VARIANT_ROOT.iterdir() if path.is_file()
    }
    original_hashes = clause_aware_module._compute_input_hashes

    def changed_hashes(root: Path) -> dict[str, str]:
        hashes = original_hashes(root)
        hashes["query_blueprint.jsonl"] = "0" * 64
        return hashes

    monkeypatch.setattr(clause_aware_module, "_compute_input_hashes", changed_hashes)
    with pytest.raises(EvaluationValidationError, match="inputs changed"):
        build_and_write_clause_aware_ground_truth(
            DATASET_ROOT,
            summary_output=tmp_path / SUMMARY_FILENAME,
            variant_output_dir=tmp_path / "variant",
        )
    assert {
        path.name: path.read_bytes() for path in VARIANT_ROOT.iterdir() if path.is_file()
    } == original_files
