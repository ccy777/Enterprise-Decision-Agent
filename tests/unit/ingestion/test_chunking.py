"""Tests for deterministic parent-child character chunking."""

import pytest

from decision_agent.domain import DocumentBlock
from decision_agent.ingestion import ChunkingStrategy, ParentChildChunker


def make_block(
    content: str,
    *,
    document_id: str = "doc-1",
    metadata: dict[str, str | int | float | bool | None] | None = None,
) -> DocumentBlock:
    return DocumentBlock(
        block_id="block-1",
        document_id=document_id,
        document_version="v1",
        content=content,
        page_number=2,
        source="reports/q2.txt",
        section="Overview",
        metadata=metadata or {"language": "mixed", "priority": 1},
    )


def test_empty_text_produces_no_chunks() -> None:
    result = ParentChildChunker(10, 5, 1).chunk(make_block(""))
    assert result.parents == ()
    assert result.children == ()


def test_whitespace_only_text_produces_no_chunks() -> None:
    result = ParentChildChunker(10, 5, 1).chunk(make_block("  \n\t  "))
    assert result.parents == ()
    assert result.children == ()


def test_short_text_produces_one_parent_and_one_child() -> None:
    result = ParentChildChunker(20, 10, 2).chunk(make_block("short"))
    assert [chunk.content for chunk in result.parents] == ["short"]
    assert [chunk.content for chunk in result.children] == ["short"]


def test_long_text_is_split_into_bounded_non_overlapping_parents() -> None:
    text = "abcdefghijklmnopqrstuvwxyz"
    result = ParentChildChunker(10, 5, 1).chunk(make_block(text))
    assert [(chunk.start_offset, chunk.end_offset) for chunk in result.parents] == [
        (0, 10),
        (10, 20),
        (20, 26),
    ]
    assert "".join(chunk.content for chunk in result.parents) == text


def test_chinese_text_is_chunked_by_unicode_characters() -> None:
    text = "华东区域产品销售持续增长"
    result = ParentChildChunker(8, 4, 1).chunk(make_block(text))
    assert result.parents[0].content == "华东区域产品销售"
    assert all(
        child.content == text[child.start_offset : child.end_offset] for child in result.children
    )


def test_english_text_preserves_exact_content() -> None:
    text = "Quarterly sales improved in East China."
    result = ParentChildChunker(50, 12, 3).chunk(make_block(text))
    assert result.parents[0].content == text
    assert all(
        child.content == text[child.start_offset : child.end_offset] for child in result.children
    )


def test_child_overlap_repeats_expected_source_span() -> None:
    result = ParentChildChunker(10, 5, 2).chunk(make_block("abcdefghij"))
    assert [(child.content, child.start_offset, child.end_offset) for child in result.children] == [
        ("abcde", 0, 5),
        ("defgh", 3, 8),
        ("ghij", 6, 10),
    ]


def test_every_child_references_an_existing_parent() -> None:
    result = ParentChildChunker(8, 4, 1).chunk(make_block("abcdefghijklmnop"))
    parent_ids = {parent.chunk_id for parent in result.parents}
    assert parent_ids
    assert all(child.parent_id in parent_ids for child in result.children)


def test_metadata_is_copied_to_parent_and_child_chunks() -> None:
    metadata = {"department": "sales", "confidential": False}
    result = ParentChildChunker(10, 5, 1).chunk(make_block("abcdefgh", metadata=metadata))
    assert result.parents[0].metadata == metadata
    assert all(child.metadata == metadata for child in result.children)
    assert result.parents[0].metadata is not metadata


def test_page_source_and_document_provenance_are_preserved() -> None:
    result = ParentChildChunker(10, 5, 1).chunk(make_block("abcdefgh"))
    for chunk in (*result.parents, *result.children):
        assert chunk.document_id == "doc-1"
        assert chunk.document_version == "v1"
        assert chunk.page_number == 2
        assert chunk.source == "reports/q2.txt"


def test_offsets_reconstruct_every_chunk_from_original_text() -> None:
    text = "  abcdefghijklmnop  "
    result = ParentChildChunker(9, 4, 1).chunk(make_block(text))
    for chunk in (*result.parents, *result.children):
        assert chunk.content == text[chunk.start_offset : chunk.end_offset]


def test_identical_input_and_configuration_produce_stable_ids() -> None:
    chunker = ParentChildChunker(10, 5, 1)
    first = chunker.chunk(make_block("abcdefghijk"))
    second = chunker.chunk(make_block("abcdefghijk"))
    assert [chunk.chunk_id for chunk in first.parents] == [
        chunk.chunk_id for chunk in second.parents
    ]
    assert [chunk.chunk_id for chunk in first.children] == [
        chunk.chunk_id for chunk in second.children
    ]


def test_different_document_ids_produce_different_chunk_ids() -> None:
    chunker = ParentChildChunker(10, 5, 1)
    first = chunker.chunk(make_block("same content", document_id="doc-1"))
    second = chunker.chunk(make_block("same content", document_id="doc-2"))
    assert first.parents[0].chunk_id != second.parents[0].chunk_id
    assert first.children[0].chunk_id != second.children[0].chunk_id


def test_configuration_changes_chunk_ids() -> None:
    block = make_block("abcdefghij")
    first = ParentChildChunker(10, 5, 1).chunk(block)
    second = ParentChildChunker(10, 5, 2).chunk(block)
    assert first.parents[0].chunk_id != second.parents[0].chunk_id


@pytest.mark.parametrize("parent_size,child_size", [(0, 0), (-1, 1), (10, 0), (10, -1)])
def test_chunk_sizes_must_be_positive(parent_size: int, child_size: int) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        ParentChildChunker(parent_size, child_size)


def test_child_size_must_not_exceed_parent_size() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        ParentChildChunker(parent_chunk_size=4, child_chunk_size=5)


@pytest.mark.parametrize("overlap", [-1, 5, 6])
def test_overlap_must_be_non_negative_and_smaller_than_child_size(overlap: int) -> None:
    with pytest.raises(ValueError, match="chunk_overlap"):
        ParentChildChunker(parent_chunk_size=10, child_chunk_size=5, chunk_overlap=overlap)


def test_parent_child_chunker_satisfies_strategy_protocol() -> None:
    assert isinstance(ParentChildChunker(10, 5, 1), ChunkingStrategy)
