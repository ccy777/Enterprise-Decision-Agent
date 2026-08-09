from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

import decision_agent.evaluation.enterprise_kb_ground_truth as ground_truth_module
import decision_agent.evaluation.reporting as reporting
from decision_agent.domain import ChildChunk, DocumentBlock, ParentChunk
from decision_agent.evaluation.enterprise_kb_ground_truth import (
    ClauseContentSpan,
    build_and_write_enterprise_kb_ground_truth,
    build_enterprise_kb_ground_truth,
)
from decision_agent.exceptions import EvaluationValidationError
from decision_agent.ingestion import ParentChildChunker, ParserRegistry

ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = ROOT / "datasets/enterprise_kb/m2c1"
CORE_REVIEW_CLAUSE_IDS = {
    "CS-WARRANTY-A-BATTERY",
    "CS-WARRANTY-A-DEVICE",
    "SALES-DISCOUNT-SOUTH-REGULAR",
    "PROC-NORMAL-50K-200K",
    "CS-DOA-B-30D",
    "INV-B-EMERGENCY-50",
    "SEC-ACCESS-L3",
    "PROC-SCORE-RECTIFY-60-69",
}


@pytest.fixture(scope="module")
def formal_build():
    return build_enterprise_kb_ground_truth(DATASET_ROOT)


@pytest.fixture(scope="module")
def formal_blocks() -> dict[str, DocumentBlock]:
    manifest = json.loads((DATASET_ROOT / "document_manifest.json").read_text(encoding="utf-8"))
    registry = ParserRegistry.default()
    blocks: dict[str, DocumentBlock] = {}
    for document in manifest["documents"]:
        parsed = registry.parse(
            str(DATASET_ROOT / "documents" / document["filename"]),
            document_id=document["document_id"],
            document_version=document["version"],
            metadata={
                "department": document["department"],
                "category": document["category"],
            },
        )
        blocks.update((block.block_id, block) for block in parsed)
    return blocks


def _synthetic_document(*contents: str) -> ground_truth_module._ParsedDocument:
    blocks = tuple(
        DocumentBlock(
            block_id=f"block-{index}",
            document_id="DOC-TEST",
            document_version="1.0",
            content=content,
            block_index=index,
            source="documents/test.md",
        )
        for index, content in enumerate(contents)
    )
    return ground_truth_module._ParsedDocument(
        document_id="DOC-TEST", blocks=blocks, parents=(), children=()
    )


def _copy_dataset(tmp_path: Path) -> Path:
    destination = tmp_path / "m2c1"
    shutil.copytree(DATASET_ROOT, destination)
    return destination


def test_formal_builder_locates_exactly_149_clauses(formal_build) -> None:
    assert len(formal_build.clause_chunk_map) == 149
    assert len({record.clause_id for record in formal_build.clause_chunk_map}) == 149


def test_formal_clause_offsets_reproduce_exact_parser_content(formal_build, formal_blocks) -> None:
    clause_ids = {record.clause_id for record in formal_build.clause_chunk_map}
    assert clause_ids >= CORE_REVIEW_CLAUSE_IDS
    for record in formal_build.clause_chunk_map:
        block = formal_blocks[record.block_id]
        marker = block.content[record.marker_start : record.marker_end]
        clause_content = block.content[record.content_start : record.content_end]
        assert marker == f"条款 ID\N{FULLWIDTH COLON}{record.clause_id}"
        assert sum(parsed.content.count(marker) for parsed in formal_blocks.values()) == 1
        assert block.content[record.marker_end : record.content_start].strip() == ""
        assert "\n" in block.content[record.marker_end : record.content_start]
        assert clause_content == record.clause_content
        assert ground_truth_module._CLAUSE_MARKER.search(clause_content) is None
        assert ground_truth_module._normalized_text_sha256(clause_content) == (
            record.content_sha256
        )


def test_parent_and_child_offsets_reproduce_exact_parser_content(
    formal_build, formal_blocks
) -> None:
    for parent in formal_build.parent_chunks:
        assert len(parent.block_ids) == 1
        block = formal_blocks[parent.block_ids[0]]
        assert 0 <= parent.start_offset < parent.end_offset <= len(block.content)
        assert parent.content == block.content[parent.start_offset : parent.end_offset]
    for child in formal_build.child_chunks:
        assert len(child.provenance.block_ids) == 1
        block = formal_blocks[child.provenance.block_ids[0]]
        assert 0 <= child.start_offset < child.end_offset <= len(block.content)
        assert child.content == block.content[child.start_offset : child.end_offset]


def test_missing_clause_marker_fails() -> None:
    document = _synthetic_document("ordinary content")
    with pytest.raises(EvaluationValidationError, match="no Clause markers"):
        ground_truth_module._locate_clause_spans(document)


def test_duplicate_clause_marker_fails() -> None:
    document = _synthetic_document(
        "条款 ID\N{FULLWIDTH COLON}TEST-ONE\nfirst\n条款 ID\N{FULLWIDTH COLON}TEST-ONE\nsecond"
    )
    with pytest.raises(EvaluationValidationError, match="duplicate Clause marker"):
        ground_truth_module._locate_clause_spans(document)


def test_empty_clause_content_fails() -> None:
    document = _synthetic_document("条款 ID\N{FULLWIDTH COLON}TEST-ONE\n   \n")
    with pytest.raises(EvaluationValidationError, match="empty content"):
        ground_truth_module._locate_clause_spans(document)


def test_illegal_clause_span_order_fails() -> None:
    with pytest.raises(ValidationError, match="end_offset"):
        ClauseContentSpan(block_id="block", start_offset=4, end_offset=3)


def test_clause_can_continue_across_document_blocks() -> None:
    document = _synthetic_document(
        "条款 ID\N{FULLWIDTH COLON}TEST-ONE\r\nfirst part",
        " second part\r\n条款 ID\N{FULLWIDTH COLON}TEST-TWO\r\nsecond clause",
    )
    located = ground_truth_module._locate_clause_spans(document)
    assert located[0][2] == "first part second part"
    assert [span.block_id for span in located[0][1]] == ["block-0", "block-1"]
    assert located[1][2] == "second clause"


def test_every_clause_maps_to_parent_and_child(formal_build) -> None:
    assert all(record.parent_ids for record in formal_build.clause_chunk_map)
    assert all(record.child_ids for record in formal_build.clause_chunk_map)


def test_parent_and_child_document_identity_is_consistent(formal_build) -> None:
    parents = {record.chunk_id: record for record in formal_build.parent_chunks}
    assert all(
        child.document_id == parents[child.parent_id].document_id
        for child in formal_build.child_chunks
    )


def test_every_child_references_a_real_parent(formal_build) -> None:
    parent_ids = {record.chunk_id for record in formal_build.parent_chunks}
    assert all(child.parent_id in parent_ids for child in formal_build.child_chunks)


def test_runtime_chunk_ids_are_globally_unique(formal_build) -> None:
    parent_ids = [record.chunk_id for record in formal_build.parent_chunks]
    child_ids = [record.chunk_id for record in formal_build.child_chunks]
    assert len(parent_ids) == len(set(parent_ids))
    assert len(child_ids) == len(set(child_ids))


def test_generated_chunk_rows_revalidate_as_domain_models(formal_build) -> None:
    parent_payload = formal_build.parent_chunks[0].model_dump(
        exclude={"schema_version", "provenance"}
    )
    child_payload = formal_build.child_chunks[0].model_dump(
        exclude={"schema_version", "provenance"}
    )
    assert ParentChunk.model_validate(parent_payload).chunk_id
    assert ChildChunk.model_validate(child_payload).chunk_id


def test_single_clause_query_maps_to_that_clause_chunks(formal_build) -> None:
    query = next(
        record
        for record in formal_build.retrieval_ground_truth
        if len(record.relevant_clause_ids) == 1 and record.answerable
    )
    clause = next(
        record
        for record in formal_build.clause_chunk_map
        if record.clause_id == query.relevant_clause_ids[0]
    )
    assert query.relevant_parent_ids == clause.parent_ids
    assert query.relevant_child_ids == clause.child_ids


def test_multi_evidence_query_uses_stable_chunk_union(formal_build) -> None:
    query = next(
        record
        for record in formal_build.retrieval_ground_truth
        if len(record.relevant_clause_ids) > 1
    )
    clause_by_id = {record.clause_id: record for record in formal_build.clause_chunk_map}
    selected = {
        parent_id
        for clause_id in query.relevant_clause_ids
        for parent_id in clause_by_id[clause_id].parent_ids
    }
    global_order = [record.chunk_id for record in formal_build.parent_chunks]
    assert query.relevant_parent_ids == tuple(
        parent_id for parent_id in global_order if parent_id in selected
    )


def test_unanswerable_query_has_no_relevant_evidence(formal_build) -> None:
    unanswerable = [
        record for record in formal_build.retrieval_ground_truth if not record.answerable
    ]
    assert len(unanswerable) == 4
    assert all(
        not record.relevant_clause_ids
        and not record.relevant_parent_ids
        and not record.relevant_child_ids
        for record in unanswerable
    )


def test_all_declared_hard_negative_clauses_map_to_chunks(formal_build) -> None:
    clause_ids = {record.clause_id for record in formal_build.clause_chunk_map}
    assert all(
        set(record.hard_negative_clause_ids) <= clause_ids
        for record in formal_build.retrieval_ground_truth
    )


def test_parent_overlaps_are_recorded_and_removed_from_hard_negatives(formal_build) -> None:
    for record in formal_build.retrieval_ground_truth:
        assert not set(record.relevant_parent_ids) & set(record.hard_negative_parent_ids)
        assert not set(record.overlapping_parent_ids) & set(record.hard_negative_parent_ids)
    assert any(record.overlapping_parent_ids for record in formal_build.retrieval_ground_truth)


def test_child_overlaps_are_recorded_and_removed_from_hard_negatives(formal_build) -> None:
    for record in formal_build.retrieval_ground_truth:
        assert not set(record.relevant_child_ids) & set(record.hard_negative_child_ids)
        assert not set(record.overlapping_child_ids) & set(record.hard_negative_child_ids)
    assert any(record.overlapping_child_ids for record in formal_build.retrieval_ground_truth)


def test_all_query_collision_sets_match_real_positive_chunk_overlaps(formal_build) -> None:
    clause_by_id = {record.clause_id: record for record in formal_build.clause_chunk_map}
    parent_by_id = {record.chunk_id: record for record in formal_build.parent_chunks}
    child_by_id = {record.chunk_id: record for record in formal_build.child_chunks}
    parent_order = tuple(parent_by_id)
    child_order = tuple(child_by_id)

    def stable_union(clause_ids, attribute: str, order: tuple[str, ...]) -> tuple[str, ...]:
        selected = {
            chunk_id
            for clause_id in clause_ids
            for chunk_id in getattr(clause_by_id[clause_id], attribute)
        }
        return tuple(chunk_id for chunk_id in order if chunk_id in selected)

    def clause_overlaps_chunk(clause_id: str, chunk) -> bool:
        block_ids = chunk.block_ids if hasattr(chunk, "block_ids") else chunk.provenance.block_ids
        return any(
            span.block_id in block_ids
            and max(span.start_offset, chunk.start_offset) < min(span.end_offset, chunk.end_offset)
            for span in clause_by_id[clause_id].content_spans
        )

    for query in formal_build.retrieval_ground_truth:
        relevant_parents = stable_union(query.relevant_clause_ids, "parent_ids", parent_order)
        raw_hard_parents = stable_union(query.hard_negative_clause_ids, "parent_ids", parent_order)
        relevant_children = stable_union(query.relevant_clause_ids, "child_ids", child_order)
        raw_hard_children = stable_union(query.hard_negative_clause_ids, "child_ids", child_order)
        expected_parent_overlap = tuple(
            chunk_id for chunk_id in raw_hard_parents if chunk_id in relevant_parents
        )
        expected_child_overlap = tuple(
            chunk_id for chunk_id in raw_hard_children if chunk_id in relevant_children
        )
        assert query.relevant_parent_ids == relevant_parents
        assert query.relevant_child_ids == relevant_children
        assert query.overlapping_parent_ids == expected_parent_overlap
        assert query.overlapping_child_ids == expected_child_overlap
        assert query.hard_negative_parent_ids == tuple(
            chunk_id for chunk_id in raw_hard_parents if chunk_id not in relevant_parents
        )
        assert query.hard_negative_child_ids == tuple(
            chunk_id for chunk_id in raw_hard_children if chunk_id not in relevant_children
        )
        for chunk_id in expected_parent_overlap:
            chunk = parent_by_id[chunk_id]
            assert any(
                clause_overlaps_chunk(clause_id, chunk) for clause_id in query.relevant_clause_ids
            )
            assert any(
                clause_overlaps_chunk(clause_id, chunk)
                for clause_id in query.hard_negative_clause_ids
            )
        for chunk_id in expected_child_overlap:
            chunk = child_by_id[chunk_id]
            assert any(
                clause_overlaps_chunk(clause_id, chunk) for clause_id in query.relevant_clause_ids
            )
            assert any(
                clause_overlaps_chunk(clause_id, chunk)
                for clause_id in query.hard_negative_clause_ids
            )

    assert formal_build.summary["overlapping_parent_query_count"] == 45
    assert formal_build.summary["overlapping_child_query_count"] == 43
    assert formal_build.summary["no_independent_hard_negative_parent_query_count"] == 28
    assert formal_build.summary["no_independent_hard_negative_child_query_count"] == 14


def test_all_query_blueprint_fields_and_order_are_preserved(formal_build) -> None:
    blueprint_rows = [
        json.loads(line)
        for line in (DATASET_ROOT / "query_blueprint.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record.query_id for record in formal_build.retrieval_ground_truth] == [
        row["query_id"] for row in blueprint_rows
    ]
    for record, row in zip(formal_build.retrieval_ground_truth, blueprint_rows, strict=True):
        assert record.query == row["query"]
        assert record.category == row["category"]
        assert record.query_type == row["query_type"]
        assert record.answerable == row["answerable"]
        assert record.relevant_clause_ids == tuple(row["relevant_clause_ids"])
        assert record.hard_negative_clause_ids == tuple(row["hard_negative_clause_ids"])
        assert record.expected_evidence_count == row["expected_evidence_count"]
        for values in (
            record.relevant_parent_ids,
            record.relevant_child_ids,
            record.hard_negative_parent_ids,
            record.hard_negative_child_ids,
            record.overlapping_parent_ids,
            record.overlapping_child_ids,
        ):
            assert len(values) == len(set(values))


def test_query_and_clause_output_counts_are_exact(formal_build) -> None:
    assert len(formal_build.retrieval_ground_truth) == 60
    assert len(formal_build.clause_chunk_map) == 149


def test_output_order_is_manifest_then_offset_stable(formal_build) -> None:
    manifest_order = formal_build.summary["manifest_document_order"]
    document_index = {document_id: index for index, document_id in enumerate(manifest_order)}
    parent_keys = [
        (document_index[record.document_id], record.start_offset, record.chunk_id)
        for record in formal_build.parent_chunks
    ]
    assert parent_keys == sorted(parent_keys)
    child_keys = [
        (document_index[record.document_id], record.start_offset, record.chunk_id)
        for record in formal_build.child_chunks
    ]
    assert child_keys == sorted(child_keys)


def test_two_in_memory_builds_are_byte_identical(formal_build) -> None:
    rebuilt = build_enterprise_kb_ground_truth(DATASET_ROOT)
    assert rebuilt.serialized_files() == formal_build.serialized_files()


def test_input_hash_change_rejects_all_writes(tmp_path: Path, monkeypatch) -> None:
    dataset_root = _copy_dataset(tmp_path)
    summary_path = tmp_path / "summary.json"
    generated_root = dataset_root / "generated"
    original_outputs = {
        path.name: path.read_bytes() for path in generated_root.iterdir() if path.is_file()
    }
    original = ground_truth_module._compute_input_hashes
    call_count = 0

    def changing_hashes(root: Path) -> dict[str, str]:
        nonlocal call_count
        call_count += 1
        hashes = original(root)
        if call_count > 1:
            hashes["query_blueprint.jsonl"] = "0" * 64
        return hashes

    monkeypatch.setattr(ground_truth_module, "_compute_input_hashes", changing_hashes)
    with pytest.raises(EvaluationValidationError, match="inputs changed"):
        build_and_write_enterprise_kb_ground_truth(dataset_root, summary_output=summary_path)
    assert not summary_path.exists()
    assert {
        path.name: path.read_bytes() for path in generated_root.iterdir() if path.is_file()
    } == original_outputs


def test_atomic_commit_failure_restores_all_existing_files(tmp_path: Path, monkeypatch) -> None:
    targets = {
        tmp_path / "first.jsonl": b"old-first\n",
        tmp_path / "second.jsonl": b"old-second\n",
        tmp_path / "summary.json": b"old-summary\n",
    }
    for target, content in targets.items():
        target.write_bytes(content)
    real_replace = reporting.os.replace
    call_count = 0

    def fail_once(source: str | Path, destination: str | Path) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 5:
            raise OSError("simulated commit failure")
        real_replace(source, destination)

    monkeypatch.setattr(reporting.os, "replace", fail_once)
    with pytest.raises(OSError, match="simulated"):
        reporting.write_text_files_atomically(
            {target: f"new-{index}\n" for index, target in enumerate(targets)}
        )
    assert {target: target.read_bytes() for target in targets} == targets
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob("*.backup"))


def test_atomic_text_and_json_writers_emit_exact_utf8_lf_bytes(tmp_path: Path) -> None:
    text_path = tmp_path / "text.jsonl"
    report_path = tmp_path / "report.json"
    text_payload = "第一行\nsecond line\n"
    report_payload = {"message": "稳定输出", "values": [1, 2]}
    expected_report = (json.dumps(report_payload, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )

    reporting.write_text_atomically(text_path, text_payload)
    reporting.write_json_report_atomically(report_path, report_payload)

    assert text_path.read_bytes() == text_payload.encode("utf-8")
    assert report_path.read_bytes() == expected_report
    assert b"\r" not in text_path.read_bytes() + report_path.read_bytes()
    assert not text_path.read_bytes().startswith(b"\xef\xbb\xbf")
    assert (
        hashlib.sha256(report_path.read_bytes()).digest()
        == hashlib.sha256(expected_report).digest()
    )


def test_all_serialized_sources_are_portable_and_relative(formal_build) -> None:
    payload = "".join(formal_build.serialized_files().values())
    assert str(ROOT) not in payload
    assert "C:\\Users\\" not in payload
    assert all(
        record.source.startswith("documents/") and "\\" not in record.source
        for record in formal_build.parent_chunks
    )


def test_summary_statistics_are_independently_recomputable(formal_build) -> None:
    summary = formal_build.summary
    assert summary["parent_chunk_count"] == len(formal_build.parent_chunks)
    assert summary["child_chunk_count"] == len(formal_build.child_chunks)
    assert summary["relevant_parent_link_count"] == sum(
        len(record.relevant_parent_ids) for record in formal_build.retrieval_ground_truth
    )
    assert summary["hard_negative_child_link_count"] == sum(
        len(record.hard_negative_child_ids) for record in formal_build.retrieval_ground_truth
    )


def test_generated_hashes_match_exact_jsonl_bytes(formal_build) -> None:
    serialized = formal_build.serialized_files()
    assert set(formal_build.summary["generated_file_hashes"]) == {
        "parent_chunks.jsonl",
        "child_chunks.jsonl",
        "clause_chunk_map.jsonl",
        "retrieval_ground_truth.jsonl",
    }
    assert not any(
        "summary" in key.casefold() and "hash" in key.casefold() for key in formal_build.summary
    )
    for filename, expected_hash in formal_build.summary["generated_file_hashes"].items():
        assert hashlib.sha256(serialized[filename].encode("utf-8")).hexdigest() == expected_hash


def test_summary_source_hashes_use_the_normalized_disk_contract(formal_build) -> None:
    summary = formal_build.summary
    for relative_path, expected_hash in summary["source_file_hashes"].items():
        assert (
            ground_truth_module.compute_normalized_text_sha256(
                DATASET_ROOT / relative_path, dataset_name=relative_path
            )
            == expected_hash
        )
    for filename, key in {
        "query_blueprint.jsonl": "query_blueprint_hash",
        "entity_dictionary.json": "entity_dictionary_hash",
        "business_fact_registry.json": "business_fact_registry_hash",
        "document_manifest.json": "document_manifest_hash",
    }.items():
        assert (
            ground_truth_module.compute_normalized_text_sha256(
                DATASET_ROOT / filename, dataset_name=filename
            )
            == summary[key]
        )


def test_formal_builder_invokes_real_parser_and_chunker(monkeypatch) -> None:
    parser_calls = 0
    chunker_calls = 0
    real_parse = ParserRegistry.parse
    real_chunk = ParentChildChunker.chunk

    def counting_parse(self, *args, **kwargs):
        nonlocal parser_calls
        parser_calls += 1
        return real_parse(self, *args, **kwargs)

    def counting_chunk(self, *args, **kwargs):
        nonlocal chunker_calls
        chunker_calls += 1
        return real_chunk(self, *args, **kwargs)

    monkeypatch.setattr(ParserRegistry, "parse", counting_parse)
    monkeypatch.setattr(ParentChildChunker, "chunk", counting_chunk)
    build_enterprise_kb_ground_truth(DATASET_ROOT)
    assert parser_calls >= 12
    assert chunker_calls >= 12


def test_mixed_format_fixture_never_appears_in_formal_output(formal_build) -> None:
    payload = json.dumps(formal_build.summary, ensure_ascii=False)
    assert "mixed_format" not in payload
    assert ".pdf" not in payload
    assert ".txt" not in payload


def test_build_and_write_creates_only_expected_files(tmp_path: Path) -> None:
    dataset_root = _copy_dataset(tmp_path)
    summary_path = tmp_path / "artifacts/m2c1_parent_child_summary.json"
    build = build_and_write_enterprise_kb_ground_truth(dataset_root, summary_output=summary_path)
    assert sorted(path.name for path in (dataset_root / "generated").iterdir()) == [
        "child_chunks.jsonl",
        "clause_chunk_map.jsonl",
        "parent_chunks.jsonl",
        "retrieval_ground_truth.jsonl",
    ]
    assert summary_path.is_file()
    for filename, expected_hash in build.summary["generated_file_hashes"].items():
        output_bytes = (dataset_root / "generated" / filename).read_bytes()
        assert hashlib.sha256(output_bytes).hexdigest() == expected_hash
    assert not list(tmp_path.rglob("*.tmp"))
    assert not list(tmp_path.rglob("*.backup"))


def test_import_has_no_external_service_side_effects() -> None:
    assert callable(ground_truth_module.build_enterprise_kb_ground_truth)
