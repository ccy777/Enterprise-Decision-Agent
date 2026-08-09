"""Deterministic Parent/Child retrieval ground truth for the M2C-1 enterprise KB."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from decision_agent.domain import ChildChunk, DocumentBlock, ParentChunk
from decision_agent.domain.models import Metadata
from decision_agent.evaluation.dataset import compute_normalized_text_sha256
from decision_agent.evaluation.enterprise_kb_dataset import (
    EXPECTED_DOCUMENT_COUNT,
    EXPECTED_QUERY_COUNT,
    EnterpriseKBDataset,
    QueryBlueprint,
    load_and_validate_enterprise_kb,
)
from decision_agent.evaluation.reporting import write_text_files_atomically
from decision_agent.exceptions import EvaluationValidationError
from decision_agent.ingestion import ParentChildChunker, ParserRegistry
from decision_agent.ingestion.parsers import DEFAULT_MAX_FILE_SIZE
from decision_agent.ingestion.protocols import ChunkingStrategy

SCHEMA_VERSION = "1.0"
DATASET_ID = "m2c1-enterprise-kb-parent-child-ground-truth-v1"
PARENT_CHUNK_SIZE = 800
CHILD_CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
EXPECTED_CLAUSE_COUNT = 149

_CLAUSE_MARKER = re.compile(
    r"^条款 ID\N{FULLWIDTH COLON}(?P<clause_id>[A-Z0-9]+(?:-[A-Z0-9]+)+)[ \t]*(?=\r?$)",
    re.MULTILINE,
)
_BUSINESS_SECTION_HEADING = re.compile(r"^##(?!#)[ \t]+", re.MULTILINE)
_INPUT_FILENAMES = (
    "entity_dictionary.json",
    "business_fact_registry.json",
    "document_manifest.json",
    "query_blueprint.jsonl",
)
_GENERATED_FILENAMES = (
    "parent_chunks.jsonl",
    "child_chunks.jsonl",
    "clause_chunk_map.jsonl",
    "retrieval_ground_truth.jsonl",
)


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ChunkProvenance(_StrictFrozenModel):
    source: str
    block_ids: tuple[str, ...]
    page_number: int | None
    document_version: str
    parser_name: str


class ParentChunkRecord(_StrictFrozenModel):
    schema_version: str = SCHEMA_VERSION
    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    document_version: str = Field(min_length=1)
    content: str = Field(min_length=1)
    block_ids: tuple[str, ...] = Field(min_length=1)
    page_number: int | None
    source: str
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)
    metadata: Metadata
    provenance: ChunkProvenance

    @model_validator(mode="after")
    def validate_offsets(self) -> ParentChunkRecord:
        if self.end_offset <= self.start_offset:
            raise ValueError("end_offset must be greater than start_offset")
        return self


class ChildChunkRecord(_StrictFrozenModel):
    schema_version: str = SCHEMA_VERSION
    chunk_id: str = Field(min_length=1)
    parent_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    document_version: str = Field(min_length=1)
    content: str = Field(min_length=1)
    page_number: int | None
    source: str
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)
    metadata: Metadata
    provenance: ChunkProvenance

    @model_validator(mode="after")
    def validate_offsets(self) -> ChildChunkRecord:
        if self.end_offset <= self.start_offset:
            raise ValueError("end_offset must be greater than start_offset")
        return self


class ClauseContentSpan(_StrictFrozenModel):
    block_id: str = Field(min_length=1)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_offsets(self) -> ClauseContentSpan:
        if self.end_offset <= self.start_offset:
            raise ValueError("end_offset must be greater than start_offset")
        return self


class ClauseChunkMapRecord(_StrictFrozenModel):
    schema_version: str = SCHEMA_VERSION
    clause_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    source: str
    block_id: str = Field(min_length=1)
    marker_start: int = Field(ge=0)
    marker_end: int = Field(gt=0)
    content_start: int = Field(ge=0)
    content_end: int = Field(gt=0)
    content_spans: tuple[ClauseContentSpan, ...] = Field(min_length=1)
    clause_content: str = Field(min_length=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_ids: tuple[str, ...] = Field(min_length=1)
    child_ids: tuple[str, ...] = Field(min_length=1)
    relevant_parent_ids: tuple[str, ...] = Field(min_length=1)
    relevant_child_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_identity_and_offsets(self) -> ClauseChunkMapRecord:
        if self.marker_end <= self.marker_start:
            raise ValueError("marker_end must be greater than marker_start")
        if self.parent_ids != self.relevant_parent_ids:
            raise ValueError("parent ID aliases must agree")
        if self.child_ids != self.relevant_child_ids:
            raise ValueError("child ID aliases must agree")
        return self


class RetrievalGroundTruthRecord(_StrictFrozenModel):
    schema_version: str = SCHEMA_VERSION
    query_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    category: str = Field(min_length=1)
    query_type: str = Field(min_length=1)
    answerable: bool
    relevant_clause_ids: tuple[str, ...]
    relevant_parent_ids: tuple[str, ...]
    relevant_child_ids: tuple[str, ...]
    hard_negative_clause_ids: tuple[str, ...]
    hard_negative_parent_ids: tuple[str, ...]
    hard_negative_child_ids: tuple[str, ...]
    overlapping_parent_ids: tuple[str, ...]
    overlapping_child_ids: tuple[str, ...]
    expected_evidence_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_disjoint_chunk_sets(self) -> RetrievalGroundTruthRecord:
        if set(self.relevant_parent_ids) & set(self.hard_negative_parent_ids):
            raise ValueError("relevant and hard-negative parent IDs must be disjoint")
        if set(self.relevant_child_ids) & set(self.hard_negative_child_ids):
            raise ValueError("relevant and hard-negative child IDs must be disjoint")
        if not self.answerable and (
            self.relevant_clause_ids or self.relevant_parent_ids or self.relevant_child_ids
        ):
            raise ValueError("unanswerable queries cannot declare relevant evidence")
        return self


@dataclass(frozen=True, slots=True)
class EnterpriseKBGroundTruthBuild:
    parent_chunks: tuple[ParentChunkRecord, ...]
    child_chunks: tuple[ChildChunkRecord, ...]
    clause_chunk_map: tuple[ClauseChunkMapRecord, ...]
    retrieval_ground_truth: tuple[RetrievalGroundTruthRecord, ...]
    summary: dict[str, Any]
    input_hashes: dict[str, str]

    def serialized_files(self) -> dict[str, str]:
        """Return the five complete deterministic UTF-8 output payloads."""
        generated = {
            "parent_chunks.jsonl": _serialize_jsonl(self.parent_chunks),
            "child_chunks.jsonl": _serialize_jsonl(self.child_chunks),
            "clause_chunk_map.jsonl": _serialize_jsonl(self.clause_chunk_map),
            "retrieval_ground_truth.jsonl": _serialize_jsonl(self.retrieval_ground_truth),
        }
        generated["m2c1_parent_child_summary.json"] = (
            json.dumps(self.summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        return generated


@dataclass(frozen=True, slots=True)
class _ParsedDocument:
    document_id: str
    blocks: tuple[DocumentBlock, ...]
    parents: tuple[ParentChunk, ...]
    children: tuple[ChildChunk, ...]


@dataclass(frozen=True, slots=True)
class _Marker:
    clause_id: str
    block_index: int
    marker_start: int
    marker_end: int
    line_end: int


def _portable_source(filename: str) -> str:
    return PurePosixPath("documents", filename).as_posix()


def _compute_input_hashes(root: Path) -> dict[str, str]:
    paths = [root / filename for filename in _INPUT_FILENAMES]
    document_paths = sorted((root / "documents").glob("*.md"), key=lambda path: path.name)
    if len(document_paths) != EXPECTED_DOCUMENT_COUNT:
        raise EvaluationValidationError(
            f"expected {EXPECTED_DOCUMENT_COUNT} formal Markdown documents"
        )
    paths.extend(document_paths)
    return {
        path.relative_to(root).as_posix(): compute_normalized_text_sha256(
            path, dataset_name=path.name
        )
        for path in paths
    }


def _parse_and_chunk_documents(
    root: Path, dataset: EnterpriseKBDataset, *, chunker: ChunkingStrategy | None = None
) -> tuple[_ParsedDocument, ...]:
    parser_registry = ParserRegistry.default()
    active_chunker = chunker or ParentChildChunker(
        parent_chunk_size=PARENT_CHUNK_SIZE,
        child_chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    parsed_documents: list[_ParsedDocument] = []
    parent_ids: set[str] = set()
    child_ids: set[str] = set()

    for manifest_record in dataset.manifest.documents:
        source_path = root / "documents" / manifest_record.filename
        blocks = parser_registry.parse(
            str(source_path),
            document_id=manifest_record.document_id,
            document_version=manifest_record.version,
            metadata={
                "department": manifest_record.department,
                "category": manifest_record.category,
            },
        )
        portable_source = _portable_source(manifest_record.filename)
        normalized_blocks = tuple(
            block.model_copy(update={"source": portable_source}) for block in blocks
        )
        parents: list[ParentChunk] = []
        children: list[ChildChunk] = []
        for block in normalized_blocks:
            result = active_chunker.chunk(block)
            for parent in result.parents:
                if parent.chunk_id in parent_ids:
                    raise EvaluationValidationError(f"duplicate parent ID: {parent.chunk_id}")
                parent_ids.add(parent.chunk_id)
                parents.append(parent)
            for child in result.children:
                if child.chunk_id in child_ids:
                    raise EvaluationValidationError(f"duplicate child ID: {child.chunk_id}")
                child_ids.add(child.chunk_id)
                children.append(child)

        parent_id_set = {parent.chunk_id for parent in parents}
        for child in children:
            if child.parent_id not in parent_id_set:
                raise EvaluationValidationError(
                    f"child {child.chunk_id} references missing parent {child.parent_id}"
                )
        parsed_documents.append(
            _ParsedDocument(
                document_id=manifest_record.document_id,
                blocks=normalized_blocks,
                parents=tuple(parents),
                children=tuple(children),
            )
        )
    return tuple(parsed_documents)


def _line_content_start(content: str, marker_end: int) -> int:
    if marker_end < len(content) and content[marker_end] == "\r":
        marker_end += 1
    if marker_end < len(content) and content[marker_end] == "\n":
        marker_end += 1
    return marker_end


def _trim_segments(
    blocks: tuple[DocumentBlock, ...], segments: list[ClauseContentSpan]
) -> tuple[ClauseContentSpan, ...]:
    mutable = [[span.block_id, span.start_offset, span.end_offset] for span in segments]
    block_by_id = {block.block_id: block for block in blocks}
    while mutable:
        block_id, start, end = mutable[0]
        text = block_by_id[block_id].content[start:end]
        stripped = text.lstrip()
        if stripped:
            mutable[0][1] = end - len(stripped)
            break
        mutable.pop(0)
    while mutable:
        block_id, start, end = mutable[-1]
        text = block_by_id[block_id].content[start:end]
        stripped = text.rstrip()
        if stripped:
            mutable[-1][2] = start + len(stripped)
            break
        mutable.pop()
    return tuple(
        ClauseContentSpan(block_id=block_id, start_offset=start, end_offset=end)
        for block_id, start, end in mutable
        if end > start
    )


def _locate_clause_spans(
    document: _ParsedDocument, *, stop_at_business_section: bool = False
) -> tuple[tuple[_Marker, tuple[ClauseContentSpan, ...], str], ...]:
    markers: list[_Marker] = []
    for block_index, block in enumerate(document.blocks):
        for match in _CLAUSE_MARKER.finditer(block.content):
            markers.append(
                _Marker(
                    clause_id=match.group("clause_id"),
                    block_index=block_index,
                    marker_start=match.start(),
                    marker_end=match.end(),
                    line_end=_line_content_start(block.content, match.end()),
                )
            )
    if not markers:
        raise EvaluationValidationError(
            f"document {document.document_id} contains no Clause markers"
        )
    duplicate_ids = sorted(
        clause_id
        for clause_id, count in Counter(marker.clause_id for marker in markers).items()
        if count > 1
    )
    if duplicate_ids:
        raise EvaluationValidationError(f"duplicate Clause marker: {duplicate_ids[0]}")

    located: list[tuple[_Marker, tuple[ClauseContentSpan, ...], str]] = []
    for index, marker in enumerate(markers):
        next_marker = markers[index + 1] if index + 1 < len(markers) else None
        last_block_index = next_marker.block_index if next_marker else len(document.blocks) - 1
        segments: list[ClauseContentSpan] = []
        for block_index in range(marker.block_index, last_block_index + 1):
            block = document.blocks[block_index]
            start = marker.line_end if block_index == marker.block_index else 0
            end = (
                next_marker.marker_start
                if next_marker is not None and block_index == next_marker.block_index
                else len(block.content)
            )
            if stop_at_business_section and block_index == marker.block_index:
                section_boundaries = [
                    heading.start()
                    for heading in _BUSINESS_SECTION_HEADING.finditer(block.content)
                    if heading.start() > marker.marker_start
                ]
                if section_boundaries:
                    end = min(end, section_boundaries[0])
            if end > start:
                segments.append(
                    ClauseContentSpan(
                        block_id=block.block_id,
                        start_offset=start,
                        end_offset=end,
                    )
                )
        trimmed_segments = _trim_segments(document.blocks, segments)
        if not trimmed_segments:
            raise EvaluationValidationError(f"Clause {marker.clause_id} has empty content")
        block_by_id = {block.block_id: block for block in document.blocks}
        clause_content = "".join(
            block_by_id[span.block_id].content[span.start_offset : span.end_offset]
            for span in trimmed_segments
        )
        if not clause_content.strip():
            raise EvaluationValidationError(f"Clause {marker.clause_id} has blank content")
        located.append((marker, trimmed_segments, clause_content))
    return tuple(located)


def _overlaps(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return max(start_a, start_b) < min(end_a, end_b)


def _normalized_text_sha256(content: str) -> str:
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _map_clauses(
    documents: tuple[_ParsedDocument, ...], *, stop_at_business_section: bool = False
) -> tuple[ClauseChunkMapRecord, ...]:
    records: list[ClauseChunkMapRecord] = []
    all_clause_ids: set[str] = set()
    for document in documents:
        block_by_id = {block.block_id: block for block in document.blocks}
        parent_by_id = {parent.chunk_id: parent for parent in document.parents}
        for marker, spans, clause_content in _locate_clause_spans(
            document, stop_at_business_section=stop_at_business_section
        ):
            if marker.clause_id in all_clause_ids:
                raise EvaluationValidationError(f"duplicate Clause ID: {marker.clause_id}")
            all_clause_ids.add(marker.clause_id)
            parent_ids = tuple(
                parent.chunk_id
                for parent in document.parents
                if any(
                    span.block_id in parent.block_ids
                    and _overlaps(
                        span.start_offset,
                        span.end_offset,
                        parent.start_offset,
                        parent.end_offset,
                    )
                    for span in spans
                )
            )
            child_ids = tuple(
                child.chunk_id
                for child in document.children
                if any(
                    span.block_id in parent_by_id[child.parent_id].block_ids
                    and child.start_offset is not None
                    and child.end_offset is not None
                    and _overlaps(
                        span.start_offset,
                        span.end_offset,
                        child.start_offset,
                        child.end_offset,
                    )
                    for span in spans
                )
            )
            if not parent_ids:
                raise EvaluationValidationError(
                    f"Clause {marker.clause_id} maps to no Parent chunk"
                )
            if not child_ids:
                raise EvaluationValidationError(f"Clause {marker.clause_id} maps to no Child chunk")
            marker_block = document.blocks[marker.block_index]
            if any(
                block_by_id[span.block_id].document_id != document.document_id for span in spans
            ):
                raise EvaluationValidationError(
                    f"Clause {marker.clause_id} crosses document identities"
                )
            records.append(
                ClauseChunkMapRecord(
                    clause_id=marker.clause_id,
                    document_id=document.document_id,
                    source=marker_block.source or "",
                    block_id=marker_block.block_id,
                    marker_start=marker.marker_start,
                    marker_end=marker.marker_end,
                    content_start=spans[0].start_offset,
                    content_end=spans[-1].end_offset,
                    content_spans=spans,
                    clause_content=clause_content,
                    content_sha256=_normalized_text_sha256(clause_content),
                    parent_ids=parent_ids,
                    child_ids=child_ids,
                    relevant_parent_ids=parent_ids,
                    relevant_child_ids=child_ids,
                )
            )
    if len(records) != EXPECTED_CLAUSE_COUNT:
        raise EvaluationValidationError(
            f"expected {EXPECTED_CLAUSE_COUNT} Clause mappings, found {len(records)}"
        )
    return tuple(records)


def _stable_union(
    clause_ids: tuple[str, ...],
    clause_map: dict[str, ClauseChunkMapRecord],
    ordered_ids: tuple[str, ...],
    attribute: str,
) -> tuple[str, ...]:
    selected: set[str] = set()
    for clause_id in clause_ids:
        try:
            record = clause_map[clause_id]
        except KeyError as exc:
            raise EvaluationValidationError(f"unknown Clause ID: {clause_id}") from exc
        selected.update(getattr(record, attribute))
    return tuple(item_id for item_id in ordered_ids if item_id in selected)


def _build_query_ground_truth(
    queries: tuple[QueryBlueprint, ...],
    clauses: tuple[ClauseChunkMapRecord, ...],
    parent_order: tuple[str, ...],
    child_order: tuple[str, ...],
) -> tuple[RetrievalGroundTruthRecord, ...]:
    clause_by_id = {clause.clause_id: clause for clause in clauses}
    records: list[RetrievalGroundTruthRecord] = []
    for query in queries:
        relevant_parents = _stable_union(
            query.relevant_clause_ids, clause_by_id, parent_order, "parent_ids"
        )
        relevant_children = _stable_union(
            query.relevant_clause_ids, clause_by_id, child_order, "child_ids"
        )
        raw_hard_parents = _stable_union(
            query.hard_negative_clause_ids, clause_by_id, parent_order, "parent_ids"
        )
        raw_hard_children = _stable_union(
            query.hard_negative_clause_ids, clause_by_id, child_order, "child_ids"
        )
        relevant_parent_set = set(relevant_parents)
        relevant_child_set = set(relevant_children)
        overlapping_parents = tuple(
            item_id for item_id in raw_hard_parents if item_id in relevant_parent_set
        )
        overlapping_children = tuple(
            item_id for item_id in raw_hard_children if item_id in relevant_child_set
        )
        hard_parents = tuple(
            item_id for item_id in raw_hard_parents if item_id not in relevant_parent_set
        )
        hard_children = tuple(
            item_id for item_id in raw_hard_children if item_id not in relevant_child_set
        )
        if query.answerable and (not relevant_parents or not relevant_children):
            raise EvaluationValidationError(
                f"answerable query {query.query_id} maps to no relevant chunks"
            )
        records.append(
            RetrievalGroundTruthRecord(
                query_id=query.query_id,
                query=query.query,
                category=query.category,
                query_type=query.query_type,
                answerable=query.answerable,
                relevant_clause_ids=query.relevant_clause_ids,
                relevant_parent_ids=relevant_parents,
                relevant_child_ids=relevant_children,
                hard_negative_clause_ids=query.hard_negative_clause_ids,
                hard_negative_parent_ids=hard_parents,
                hard_negative_child_ids=hard_children,
                overlapping_parent_ids=overlapping_parents,
                overlapping_child_ids=overlapping_children,
                expected_evidence_count=query.expected_evidence_count,
            )
        )
    if len(records) != EXPECTED_QUERY_COUNT:
        raise EvaluationValidationError(
            f"expected {EXPECTED_QUERY_COUNT} query mappings, found {len(records)}"
        )
    return tuple(records)


def _parent_record(parent: ParentChunk, parser_name: str) -> ParentChunkRecord:
    provenance = ChunkProvenance(
        source=parent.source or "",
        block_ids=tuple(parent.block_ids),
        page_number=parent.page_number,
        document_version=parent.document_version,
        parser_name=parser_name,
    )
    return ParentChunkRecord(
        chunk_id=parent.chunk_id,
        document_id=parent.document_id,
        document_version=parent.document_version,
        content=parent.content,
        block_ids=tuple(parent.block_ids),
        page_number=parent.page_number,
        source=parent.source or "",
        start_offset=parent.start_offset,
        end_offset=parent.end_offset,
        metadata=dict(parent.metadata),
        provenance=provenance,
    )


def _child_record(child: ChildChunk, parent: ParentChunk, parser_name: str) -> ChildChunkRecord:
    if child.start_offset is None or child.end_offset is None:
        raise EvaluationValidationError(f"child {child.chunk_id} is missing offsets")
    provenance = ChunkProvenance(
        source=child.source or "",
        block_ids=tuple(parent.block_ids),
        page_number=child.page_number,
        document_version=child.document_version,
        parser_name=parser_name,
    )
    return ChildChunkRecord(
        chunk_id=child.chunk_id,
        parent_id=child.parent_id,
        document_id=child.document_id,
        document_version=child.document_version,
        content=child.content,
        page_number=child.page_number,
        source=child.source or "",
        start_offset=child.start_offset,
        end_offset=child.end_offset,
        metadata=dict(child.metadata),
        provenance=provenance,
    )


def _serialize_jsonl(records: tuple[BaseModel, ...]) -> str:
    return "".join(
        json.dumps(
            record.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for record in records
    )


def _count_distribution(values: list[int]) -> dict[str, int]:
    return {str(key): count for key, count in sorted(Counter(values).items())}


def _maximum_overlap(
    records: tuple[RetrievalGroundTruthRecord, ...], attribute: str
) -> dict[str, str | int] | None:
    candidates = [record for record in records if getattr(record, attribute)]
    if not candidates:
        return None
    maximum = max(candidates, key=lambda record: len(getattr(record, attribute)))
    return {
        "query_id": maximum.query_id,
        "overlap_count": len(getattr(maximum, attribute)),
    }


def _build_summary(
    *,
    dataset: EnterpriseKBDataset,
    documents: tuple[_ParsedDocument, ...],
    parents: tuple[ParentChunkRecord, ...],
    children: tuple[ChildChunkRecord, ...],
    clauses: tuple[ClauseChunkMapRecord, ...],
    queries: tuple[RetrievalGroundTruthRecord, ...],
    input_hashes: dict[str, str],
    generated_hashes: dict[str, str],
    dataset_id: str = DATASET_ID,
    chunker_identity: str = "decision_agent.ingestion.ParentChildChunker",
    chunker_config: dict[str, Any] | None = None,
    limitations: list[str] | None = None,
) -> dict[str, Any]:
    parent_overlap_ids = tuple(
        record.query_id for record in queries if record.overlapping_parent_ids
    )
    child_overlap_ids = tuple(record.query_id for record in queries if record.overlapping_child_ids)
    no_hard_parent_ids = tuple(
        record.query_id
        for record in queries
        if record.hard_negative_clause_ids and not record.hard_negative_parent_ids
    )
    no_hard_child_ids = tuple(
        record.query_id
        for record in queries
        if record.hard_negative_clause_ids and not record.hard_negative_child_ids
    )
    document_distribution = {
        document.document_id: {
            "document_block_count": len(document.blocks),
            "parent_chunk_count": len(document.parents),
            "child_chunk_count": len(document.children),
        }
        for document in documents
    }
    document_hashes = {
        key: value for key, value in input_hashes.items() if key.startswith("documents/")
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "document_count": len(documents),
        "total_source_chars": sum(len(block.content) for doc in documents for block in doc.blocks),
        "document_block_count": sum(len(document.blocks) for document in documents),
        "parent_chunk_count": len(parents),
        "child_chunk_count": len(children),
        "clause_count": len(clauses),
        "query_count": len(queries),
        "answerable_query_count": sum(record.answerable for record in queries),
        "unanswerable_query_count": sum(not record.answerable for record in queries),
        "relevant_parent_link_count": sum(len(record.relevant_parent_ids) for record in queries),
        "relevant_child_link_count": sum(len(record.relevant_child_ids) for record in queries),
        "hard_negative_parent_link_count": sum(
            len(record.hard_negative_parent_ids) for record in queries
        ),
        "hard_negative_child_link_count": sum(
            len(record.hard_negative_child_ids) for record in queries
        ),
        "overlapping_parent_query_count": len(parent_overlap_ids),
        "overlapping_child_query_count": len(child_overlap_ids),
        "overlapping_parent_query_ids": list(parent_overlap_ids),
        "overlapping_child_query_ids": list(child_overlap_ids),
        "multi_parent_clause_count": sum(len(record.parent_ids) > 1 for record in clauses),
        "multi_child_clause_count": sum(len(record.child_ids) > 1 for record in clauses),
        "no_independent_hard_negative_parent_query_count": len(no_hard_parent_ids),
        "no_independent_hard_negative_parent_query_ids": list(no_hard_parent_ids),
        "no_independent_hard_negative_child_query_count": len(no_hard_child_ids),
        "no_independent_hard_negative_child_query_ids": list(no_hard_child_ids),
        "max_parent_overlap_query": _maximum_overlap(queries, "overlapping_parent_ids"),
        "max_child_overlap_query": _maximum_overlap(queries, "overlapping_child_ids"),
        "clause_parent_count_distribution": _count_distribution(
            [len(record.parent_ids) for record in clauses]
        ),
        "clause_child_count_distribution": _count_distribution(
            [len(record.child_ids) for record in clauses]
        ),
        "document_chunk_distribution": document_distribution,
        "query_category_distribution": dict(
            sorted(Counter(record.category for record in queries).items())
        ),
        "query_type_distribution": dict(
            sorted(Counter(record.query_type for record in queries).items())
        ),
        "parser_identity": "decision_agent.ingestion.MarkdownDocumentParser",
        "parser_config": {
            "encoding": "utf-8-sig",
            "fallback_encodings": [],
            "heading_split": False,
            "max_file_size_bytes": DEFAULT_MAX_FILE_SIZE,
        },
        "chunker_identity": chunker_identity,
        "chunker_config": chunker_config
        or {
            "parent_chunk_size": PARENT_CHUNK_SIZE,
            "child_chunk_size": CHILD_CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        },
        "source_file_hashes": document_hashes,
        "query_blueprint_hash": input_hashes["query_blueprint.jsonl"],
        "entity_dictionary_hash": input_hashes["entity_dictionary.json"],
        "business_fact_registry_hash": input_hashes["business_fact_registry.json"],
        "document_manifest_hash": input_hashes["document_manifest.json"],
        "generated_file_hashes": generated_hashes,
        "limitations": limitations
        or [
            "Ground truth records character-overlap relevance only; it does not run retrieval.",
            "Character-window chunking can produce mixed chunks spanning multiple Clauses, which "
            "limits pure hard-negative analysis at Parent level; collisions are recorded, not "
            "silently removed.",
            "Multi-file rollback handles normal process exceptions, but it is not a durable "
            "transaction against abrupt process or operating-system failure.",
            "No HitRate, Recall, MRR, model, Milvus, or LLM result is produced in M2C-1B.",
        ],
        "manifest_document_order": [record.document_id for record in dataset.manifest.documents],
    }


def build_enterprise_kb_ground_truth(
    root: str | Path,
    *,
    chunker: ChunkingStrategy | None = None,
    dataset_id: str = DATASET_ID,
    chunker_identity: str = "decision_agent.ingestion.ParentChildChunker",
    chunker_config: dict[str, Any] | None = None,
    stop_at_business_section: bool = False,
    limitations: list[str] | None = None,
) -> EnterpriseKBGroundTruthBuild:
    """Build the complete deterministic M2C-1B ground truth in memory."""
    root_path = Path(root).resolve()
    input_hashes = _compute_input_hashes(root_path)
    dataset, _ = load_and_validate_enterprise_kb(root_path)
    documents = _parse_and_chunk_documents(root_path, dataset, chunker=chunker)
    if len(documents) != EXPECTED_DOCUMENT_COUNT:
        raise EvaluationValidationError(
            f"expected {EXPECTED_DOCUMENT_COUNT} parsed documents, found {len(documents)}"
        )

    parser_name = "MarkdownDocumentParser"
    parent_records: list[ParentChunkRecord] = []
    child_records: list[ChildChunkRecord] = []
    for document in documents:
        parent_by_id = {parent.chunk_id: parent for parent in document.parents}
        for parent in document.parents:
            parent_records.append(_parent_record(parent, parser_name))
        for child in document.children:
            parent = parent_by_id[child.parent_id]
            if child.document_id != parent.document_id:
                raise EvaluationValidationError(
                    f"child {child.chunk_id} has a different document_id than its parent"
                )
            child_records.append(_child_record(child, parent, parser_name))

    parents = tuple(parent_records)
    children = tuple(child_records)
    clauses = _map_clauses(documents, stop_at_business_section=stop_at_business_section)
    queries = _build_query_ground_truth(
        dataset.queries,
        clauses,
        tuple(record.chunk_id for record in parents),
        tuple(record.chunk_id for record in children),
    )
    generated_text = {
        "parent_chunks.jsonl": _serialize_jsonl(parents),
        "child_chunks.jsonl": _serialize_jsonl(children),
        "clause_chunk_map.jsonl": _serialize_jsonl(clauses),
        "retrieval_ground_truth.jsonl": _serialize_jsonl(queries),
    }
    generated_hashes = {
        filename: hashlib.sha256(content.encode("utf-8")).hexdigest()
        for filename, content in generated_text.items()
    }
    summary = _build_summary(
        dataset=dataset,
        documents=documents,
        parents=parents,
        children=children,
        clauses=clauses,
        queries=queries,
        input_hashes=input_hashes,
        generated_hashes=generated_hashes,
        dataset_id=dataset_id,
        chunker_identity=chunker_identity,
        chunker_config=chunker_config,
        limitations=limitations,
    )
    return EnterpriseKBGroundTruthBuild(
        parent_chunks=parents,
        child_chunks=children,
        clause_chunk_map=clauses,
        retrieval_ground_truth=queries,
        summary=summary,
        input_hashes=input_hashes,
    )


def build_and_write_enterprise_kb_ground_truth(
    root: str | Path,
    *,
    summary_output: str | Path,
) -> EnterpriseKBGroundTruthBuild:
    """Build, recheck inputs, and atomically publish all M2C-1B files."""
    root_path = Path(root).resolve()
    build = build_enterprise_kb_ground_truth(root_path)
    if _compute_input_hashes(root_path) != build.input_hashes:
        raise EvaluationValidationError("enterprise KB inputs changed during ground-truth build")

    serialized = build.serialized_files()
    generated_root = root_path / "generated"
    outputs = {generated_root / filename: serialized[filename] for filename in _GENERATED_FILENAMES}
    outputs[Path(summary_output).resolve()] = serialized["m2c1_parent_child_summary.json"]
    write_text_files_atomically(outputs)
    return build


__all__ = [
    "CHILD_CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "PARENT_CHUNK_SIZE",
    "ChildChunkRecord",
    "ClauseChunkMapRecord",
    "ClauseContentSpan",
    "EnterpriseKBGroundTruthBuild",
    "ParentChunkRecord",
    "RetrievalGroundTruthRecord",
    "build_and_write_enterprise_kb_ground_truth",
    "build_enterprise_kb_ground_truth",
]
