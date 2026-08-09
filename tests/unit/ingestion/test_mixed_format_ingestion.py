"""Real ParserRegistry-to-ParentChildChunker tests over static mixed-format fixtures."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from decision_agent.domain import DocumentBlock
from decision_agent.evaluation import load_and_validate_enterprise_kb
from decision_agent.evaluation.enterprise_kb_dataset import (
    EXPECTED_DOCUMENT_COUNT,
    EXPECTED_QUERY_COUNT,
)
from decision_agent.evaluation.enterprise_kb_ground_truth import EXPECTED_CLAUSE_COUNT
from decision_agent.exceptions import UnsupportedDocumentTypeError
from decision_agent.ingestion import (
    MarkdownDocumentParser,
    ParentChildChunker,
    ParserRegistry,
    PdfDocumentParser,
    TextDocumentParser,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ROOT = REPOSITORY_ROOT / "tests/fixtures/ingestion/mixed_format"
DATASET_ROOT = REPOSITORY_ROOT / "datasets/enterprise_kb/m2c1"
FIXTURES = {
    "txt": FIXTURE_ROOT / "service_maintenance_notice.txt",
    "md": FIXTURE_ROOT / "internal_security_guideline.md",
    "pdf": FIXTURE_ROOT / "equipment_operation_manual.pdf",
}
DOCUMENT_IDS = {key: f"mixed-format-{key}" for key in FIXTURES}


def relative_source(path: Path) -> str:
    """Keep fixture provenance portable instead of serializing a host-specific path."""
    return path.relative_to(REPOSITORY_ROOT).as_posix()


@pytest.fixture(autouse=True)
def use_explicit_repository_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve portable fixture sources from the explicit repository root."""
    monkeypatch.chdir(REPOSITORY_ROOT)


def parse_fixture(kind: str) -> list[DocumentBlock]:
    return ParserRegistry.default().parse(
        relative_source(FIXTURES[kind]),
        document_id=DOCUMENT_IDS[kind],
        document_version="1.0",
        metadata={"fixture": "mixed-format", "format": kind},
    )


@pytest.mark.parametrize(
    ("kind", "parser_type"),
    [
        ("txt", TextDocumentParser),
        ("md", MarkdownDocumentParser),
        ("pdf", PdfDocumentParser),
    ],
)
def test_registry_routes_each_static_fixture(kind: str, parser_type: type[object]) -> None:
    parser = ParserRegistry.default().parser_for(relative_source(FIXTURES[kind]))

    assert isinstance(parser, parser_type)


def test_registry_preserves_existing_unsupported_suffix_contract() -> None:
    with pytest.raises(UnsupportedDocumentTypeError, match="no document parser registered"):
        ParserRegistry.default().parser_for("unsupported.docx")


@pytest.mark.parametrize("kind", ["txt", "md", "pdf"])
def test_all_formats_produce_complete_nonempty_document_blocks(kind: str) -> None:
    blocks = parse_fixture(kind)

    assert blocks
    assert all(isinstance(block, DocumentBlock) for block in blocks)
    assert all(block.content.strip() for block in blocks)
    assert all(block.document_id == DOCUMENT_IDS[kind] for block in blocks)
    assert all(block.document_version == "1.0" for block in blocks)
    assert all(
        Path(block.source or "") == Path(relative_source(FIXTURES[kind])) for block in blocks
    )
    assert all(block.file_name == FIXTURES[kind].name for block in blocks)
    assert all(block.file_sha256 for block in blocks)
    assert all(block.block_content_sha256 for block in blocks)
    assert all(block.metadata == {"fixture": "mixed-format", "format": kind} for block in blocks)
    assert all(not Path(block.source or "").is_absolute() for block in blocks)
    assert all(str(REPOSITORY_ROOT) not in block.content for block in blocks)


def test_utf8_text_and_markdown_content_is_preserved() -> None:
    text_block = parse_fixture("txt")[0]
    markdown_block = parse_fixture("md")[0]

    assert "服务维护通知" in text_block.content
    assert "售后工单系统" in text_block.content
    assert "# 华衡智能内部测试环境安全指引" in markdown_block.content
    assert "ParserRegistry" in markdown_block.content


def test_pdf_extracts_stable_keywords_and_real_page_provenance() -> None:
    blocks = parse_fixture("pdf")

    assert len(blocks) == 2
    assert [block.page_number for block in blocks] == [1, 2]
    assert [block.block_index for block in blocks] == [0, 1]
    assert "LOCKOUT" in blocks[0].content
    assert "RESTART" in blocks[1].content
    assert all(block.parser_name == "PdfDocumentParser" for block in blocks)
    assert all(block.section is None for block in blocks)


def test_fixture_contents_are_distinct_bounded_and_synthetic() -> None:
    contents = {
        kind: "\n".join(block.content for block in parse_fixture(kind)) for kind in FIXTURES
    }

    assert len(set(contents.values())) == 3
    assert all(500 <= len(content) <= 1000 for content in contents.values())
    assert all(
        re.search(r"(?<!\d)1[3-9]\d{9}(?!\d)", content) is None for content in contents.values()
    )
    assert all(
        re.search(r"(?<!\d)\d{17}[0-9Xx](?!\d)", content) is None for content in contents.values()
    )


@pytest.mark.parametrize("kind", ["txt", "md", "pdf"])
def test_repeated_registry_parsing_is_deterministic(kind: str) -> None:
    first = parse_fixture(kind)
    second = parse_fixture(kind)

    assert first == second
    assert [block.model_dump(mode="json") for block in first] == [
        block.model_dump(mode="json") for block in second
    ]


@pytest.mark.parametrize("kind", ["txt", "md", "pdf"])
def test_all_formats_flow_through_real_parent_child_chunker(kind: str) -> None:
    chunker = ParentChildChunker(parent_chunk_size=400, child_chunk_size=180, chunk_overlap=30)
    blocks = parse_fixture(kind)

    first = [chunker.chunk(block) for block in blocks]
    second = [chunker.chunk(block) for block in blocks]

    assert first == second
    for block, result in zip(blocks, first, strict=True):
        assert result.parents
        assert result.children
        parent_ids = {parent.chunk_id for parent in result.parents}
        assert all(child.parent_id in parent_ids for child in result.children)
        assert all(parent.document_id == block.document_id for parent in result.parents)
        assert all(child.document_id == block.document_id for child in result.children)
        assert all(parent.source == block.source for parent in result.parents)
        assert all(child.source == block.source for child in result.children)
        assert all(parent.end_offset > parent.start_offset >= 0 for parent in result.parents)
        assert all(
            child.start_offset is not None
            and child.end_offset is not None
            and child.end_offset > child.start_offset >= 0
            for child in result.children
        )


def test_mixed_format_fixtures_are_isolated_from_formal_benchmark() -> None:
    dataset, statistics = load_and_validate_enterprise_kb(DATASET_ROOT)
    manifest_filenames = {document.filename for document in dataset.manifest.documents}
    fact_text = (DATASET_ROOT / "business_fact_registry.json").read_text(encoding="utf-8")
    query_text = (DATASET_ROOT / "query_blueprint.jsonl").read_text(encoding="utf-8")

    assert manifest_filenames.isdisjoint(path.name for path in FIXTURES.values())
    assert all(path.name not in fact_text for path in FIXTURES.values())
    assert all(path.name not in query_text for path in FIXTURES.values())
    assert statistics.document_count == EXPECTED_DOCUMENT_COUNT
    assert statistics.query_count == EXPECTED_QUERY_COUNT
    assert statistics.clause_count == EXPECTED_CLAUSE_COUNT


def test_manifest_remains_markdown_only() -> None:
    manifest = json.loads((DATASET_ROOT / "document_manifest.json").read_text(encoding="utf-8"))

    assert len(manifest["documents"]) == EXPECTED_DOCUMENT_COUNT
    assert all(document["filename"].endswith(".md") for document in manifest["documents"])
