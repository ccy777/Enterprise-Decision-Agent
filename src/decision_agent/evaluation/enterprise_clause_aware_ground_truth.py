"""Build isolated Clause-aware M2C-2B-1 structural ground-truth artifacts."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from decision_agent.evaluation.enterprise_kb_ground_truth import (
    ChildChunkRecord,
    ClauseChunkMapRecord,
    ParentChunkRecord,
    RetrievalGroundTruthRecord,
    _compute_input_hashes,
    _serialize_jsonl,
    build_enterprise_kb_ground_truth,
)
from decision_agent.evaluation.reporting import write_text_files_atomically
from decision_agent.exceptions import EvaluationValidationError
from decision_agent.ingestion import ClauseAwareChunker

STRATEGY_ID = "clause-aware-v1"
VARIANT_DIRECTORY = Path("variants/clause_aware_v1")
SUMMARY_FILENAME = "m2c2b1_clause_aware_summary.json"
_GENERATED_FILENAMES = (
    "parent_chunks.jsonl",
    "child_chunks.jsonl",
    "clause_chunk_map.jsonl",
    "retrieval_ground_truth.jsonl",
)


@dataclass(frozen=True, slots=True)
class ClauseAwareGroundTruthBuild:
    """Versioned Clause-aware generated records plus a structural-only summary."""

    parent_chunks: tuple[ParentChunkRecord, ...]
    child_chunks: tuple[ChildChunkRecord, ...]
    clause_chunk_map: tuple[ClauseChunkMapRecord, ...]
    retrieval_ground_truth: tuple[RetrievalGroundTruthRecord, ...]
    summary: dict[str, Any]
    input_hashes: dict[str, str]

    def serialized_files(self) -> dict[str, str]:
        generated = {
            "parent_chunks.jsonl": _serialize_jsonl(self.parent_chunks),
            "child_chunks.jsonl": _serialize_jsonl(self.child_chunks),
            "clause_chunk_map.jsonl": _serialize_jsonl(self.clause_chunk_map),
            "retrieval_ground_truth.jsonl": _serialize_jsonl(self.retrieval_ground_truth),
        }
        generated[SUMMARY_FILENAME] = (
            json.dumps(self.summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        return generated


def _length_summary(
    records: tuple[ParentChunkRecord, ...] | tuple[ChildChunkRecord, ...],
) -> dict[str, float | int]:
    lengths = sorted(len(record.content) for record in records)
    if not lengths:
        raise EvaluationValidationError("Clause-aware chunk lengths cannot be empty")
    nearest_rank = lengths[max(0, math.ceil(len(lengths) * 0.95) - 1)]
    return {
        "min": lengths[0],
        "mean": sum(lengths) / len(lengths),
        "median": (
            lengths[len(lengths) // 2]
            if len(lengths) % 2
            else (lengths[len(lengths) // 2 - 1] + lengths[len(lengths) // 2]) / 2
        ),
        "p95": nearest_rank,
        "max": lengths[-1],
    }


def _cross_clause_child_count(
    children: tuple[ChildChunkRecord, ...], clauses: tuple[ClauseChunkMapRecord, ...]
) -> int:
    count = 0
    for child in children:
        overlapping = {
            clause.clause_id
            for clause in clauses
            if clause.document_id == child.document_id
            and any(
                span.block_id in child.provenance.block_ids
                and max(span.start_offset, child.start_offset)
                < min(span.end_offset, child.end_offset)
                for span in clause.content_spans
            )
        }
        if len(overlapping) > 1:
            count += 1
    return count


def _cross_section_child_count(children: tuple[ChildChunkRecord, ...], root: Path) -> int:
    section_boundaries: dict[str, tuple[int, ...]] = {}
    for child in children:
        if child.source not in section_boundaries:
            content = (root / child.source).read_bytes().decode("utf-8")
            section_boundaries[child.source] = tuple(
                match.start() for match in re.finditer(r"^##(?!#)[ \t]+", content, re.MULTILINE)
            )
    return sum(
        any(
            child.start_offset < boundary < child.end_offset
            for boundary in section_boundaries[child.source]
        )
        for child in children
    )


def _fixed_window_comparison(
    summary: dict[str, Any], root: Path, fixed_summary: Path | None
) -> dict[str, int]:
    fixed_path = fixed_summary or next(
        (
            parent / "artifacts/datasets/m2c1_parent_child_summary.json"
            for parent in root.parents
            if (parent / "artifacts/datasets/m2c1_parent_child_summary.json").is_file()
        ),
        None,
    )
    if fixed_path is None:
        raise EvaluationValidationError("cannot locate fixed-window structural summary")
    try:
        fixed = json.loads(fixed_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise EvaluationValidationError("cannot read fixed-window structural summary") from exc
    return {
        "fixed_window_parent_chunk_count": fixed["parent_chunk_count"],
        "clause_aware_parent_chunk_count": summary["parent_chunk_count"],
        "fixed_window_child_chunk_count": fixed["child_chunk_count"],
        "clause_aware_child_chunk_count": summary["child_chunk_count"],
        "fixed_window_parent_collision_query_count": fixed["overlapping_parent_query_count"],
        "clause_aware_parent_collision_query_count": summary["overlapping_parent_query_count"],
        "fixed_window_child_collision_query_count": fixed["overlapping_child_query_count"],
        "clause_aware_child_collision_query_count": summary["overlapping_child_query_count"],
        "fixed_window_no_independent_hard_negative_parent_query_count": fixed[
            "no_independent_hard_negative_parent_query_count"
        ],
        "clause_aware_no_independent_hard_negative_parent_query_count": summary[
            "no_independent_hard_negative_parent_query_count"
        ],
        "fixed_window_no_independent_hard_negative_child_query_count": fixed[
            "no_independent_hard_negative_child_query_count"
        ],
        "clause_aware_no_independent_hard_negative_child_query_count": summary[
            "no_independent_hard_negative_child_query_count"
        ],
    }


def build_clause_aware_ground_truth(
    root: str | Path, *, fixed_window_summary: str | Path | None = None
) -> ClauseAwareGroundTruthBuild:
    """Build the Clause-aware variant in memory without touching fixed outputs."""
    root_path = Path(root).resolve()
    chunker = ClauseAwareChunker(parent_chunk_size=800, child_chunk_size=300, chunk_overlap=50)
    base = build_enterprise_kb_ground_truth(
        root_path,
        chunker=chunker,
        dataset_id="m2c2b1-enterprise-kb-clause-aware-ground-truth-v1",
        chunker_identity="decision_agent.ingestion.ClauseAwareChunker",
        chunker_config={
            "strategy_id": STRATEGY_ID,
            "parent_max_chars": 800,
            "child_max_chars": 300,
            "child_overlap": 50,
        },
        stop_at_business_section=True,
        limitations=[
            "This structural variant maps Clause-aware Parent/Child IDs only; it does not "
            "run retrieval.",
            "Clause-aware boundaries can change collision sets without establishing retrieval "
            "quality.",
            "No HitRate, Recall, MRR, real model, Milvus, LLM, or production claim is "
            "produced in M2C-2B-1.",
        ],
    )
    summary = dict(base.summary)
    summary.update(
        {
            "strategy_id": STRATEGY_ID,
            "parent_length_chars": _length_summary(base.parent_chunks),
            "child_length_chars": _length_summary(base.child_chunks),
            "overlong_clause_count": sum(
                len(record.child_ids) > 1 for record in base.clause_chunk_map
            ),
            "split_clause_ids": [
                record.clause_id for record in base.clause_chunk_map if len(record.child_ids) > 1
            ],
            "cross_clause_child_count": _cross_clause_child_count(
                base.child_chunks, base.clause_chunk_map
            ),
            "cross_section_child_count": _cross_section_child_count(base.child_chunks, root_path),
            "length_percentile_method": "nearest-rank p95",
        }
    )
    summary["fixed_window_comparison"] = _fixed_window_comparison(
        summary,
        root_path,
        Path(fixed_window_summary).resolve() if fixed_window_summary is not None else None,
    )
    return ClauseAwareGroundTruthBuild(
        parent_chunks=base.parent_chunks,
        child_chunks=base.child_chunks,
        clause_chunk_map=base.clause_chunk_map,
        retrieval_ground_truth=base.retrieval_ground_truth,
        summary=summary,
        input_hashes=base.input_hashes,
    )


def build_and_write_clause_aware_ground_truth(
    root: str | Path,
    *,
    summary_output: str | Path,
    fixed_window_summary: str | Path | None = None,
    variant_output_dir: str | Path | None = None,
) -> ClauseAwareGroundTruthBuild:
    """Atomically publish Clause-aware files to an explicit isolated directory.

    Production callers keep the versioned ``variants/clause_aware_v1`` default.
    Tests that exercise publication must instead provide a temporary output directory,
    preserving the repository's checked-in generated artifacts as read-only fixtures.
    """
    root_path = Path(root).resolve()
    build = build_clause_aware_ground_truth(root_path, fixed_window_summary=fixed_window_summary)
    if _compute_input_hashes(root_path) != build.input_hashes:
        raise EvaluationValidationError("enterprise KB inputs changed during Clause-aware build")
    serialized = build.serialized_files()
    variant_root = (
        Path(variant_output_dir).resolve()
        if variant_output_dir is not None
        else root_path / VARIANT_DIRECTORY
    )
    outputs = {variant_root / name: serialized[name] for name in _GENERATED_FILENAMES}
    outputs[Path(summary_output).resolve()] = serialized[SUMMARY_FILENAME]
    write_text_files_atomically(outputs)
    return build


__all__ = [
    "STRATEGY_ID",
    "SUMMARY_FILENAME",
    "VARIANT_DIRECTORY",
    "ClauseAwareGroundTruthBuild",
    "build_and_write_clause_aware_ground_truth",
    "build_clause_aware_ground_truth",
]
