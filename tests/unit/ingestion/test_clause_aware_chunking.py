"""High-value unit coverage for Clause-aware structural chunking."""
# ruff: noqa: RUF001

from __future__ import annotations

from decision_agent.domain import DocumentBlock
from decision_agent.ingestion import ClauseAwareChunker


def _block(content: str) -> DocumentBlock:
    return DocumentBlock(
        block_id="block-test",
        document_id="DOC-TEST",
        document_version="v1",
        content=content,
        source="documents/test.md",
        metadata={"category": "test"},
    )


def _chunker(*, parent: int = 160, child: int = 80, overlap: int = 20) -> ClauseAwareChunker:
    return ClauseAwareChunker(
        parent_chunk_size=parent, child_chunk_size=child, chunk_overlap=overlap
    )


def test_heading_paths_and_section_metadata_are_stable() -> None:
    block = _block("# 总标题\n## 第一节\n条款 ID：TEST-ONE\n甲\n## 第二节\n条款 ID：TEST-TWO\n乙\n")
    result = _chunker().chunk(block)

    assert [parent.metadata["section_path"] for parent in result.parents] == [
        "总标题 > 第一节",
        "总标题 > 第二节",
    ]
    assert [child.metadata["clause_ids"] for child in result.children] == [
        ("TEST-ONE",),
        ("TEST-TWO",),
    ]


def test_parent_does_not_cross_business_section() -> None:
    block = _block("# 标题\n## A\n条款 ID：TEST-ONE\n甲\n## B\n条款 ID：TEST-TWO\n乙\n")
    result = _chunker().chunk(block)

    assert all(
        "## A" not in parent.content or "## B" not in parent.content for parent in result.parents
    )


def test_parent_prefers_complete_adjacent_clauses() -> None:
    block = _block(
        "# 标题\n## A\n条款 ID：TEST-ONE\n" + "甲" * 40 + "\n条款 ID：TEST-TWO\n" + "乙" * 40
    )
    result = _chunker(parent=100, child=80).chunk(block)

    assert all(
        parent.content == block.content[parent.start_offset : parent.end_offset]
        for parent in result.parents
    )
    assert all(parent.metadata["clause_ids"] for parent in result.parents)


def test_child_never_crosses_clause_or_section() -> None:
    block = _block(
        "# 标题\n## A\n条款 ID：TEST-ONE\n甲\n条款 ID：TEST-TWO\n乙\n## B\n条款 ID：TEST-THREE\n丙"
    )
    result = _chunker().chunk(block)

    assert all(len(child.metadata["clause_ids"]) == 1 for child in result.children)
    assert all("## B" not in child.content for child in result.children[:-1])


def test_short_clause_produces_one_child_including_marker() -> None:
    block = _block("## A\n条款 ID：TEST-ONE\n短正文")
    result = _chunker().chunk(block)

    assert len(result.children) == 1
    assert result.children[0].content.startswith("条款 ID：TEST-ONE")


def test_overlong_clause_splits_only_inside_its_own_range() -> None:
    block = _block("## A\n条款 ID：TEST-LONG\n" + "甲" * 150 + "\n条款 ID：TEST-NEXT\n乙")
    result = _chunker(parent=300, child=60, overlap=10).chunk(block)
    long_children = [
        child for child in result.children if child.metadata["clause_ids"] == ("TEST-LONG",)
    ]

    assert len(long_children) > 1
    assert all("TEST-NEXT" not in child.content for child in long_children)
    assert all(
        child.start_offset >= block.content.index("条款 ID：TEST-LONG") for child in long_children
    )


def test_overlong_clause_overlap_is_bounded_by_clause_offsets() -> None:
    block = _block("## A\n条款 ID：TEST-LONG\n" + "甲" * 150)
    result = _chunker(parent=300, child=60, overlap=10).chunk(block)
    children = result.children

    assert all(child.end_offset <= len(block.content) for child in children)
    assert children[1].start_offset == children[0].end_offset - 10


def test_child_and_parent_offsets_reconstruct_original_content() -> None:
    block = _block("## A\n条款 ID：TEST-ONE\n甲\n条款 ID：TEST-TWO\n乙")
    result = _chunker().chunk(block)

    assert all(
        item.content == block.content[item.start_offset : item.end_offset]
        for item in (*result.parents, *result.children)
    )


def test_children_reference_existing_parents_with_matching_document_identity() -> None:
    block = _block("## A\n条款 ID：TEST-ONE\n甲\n条款 ID：TEST-TWO\n乙")
    result = _chunker().chunk(block)
    parents = {parent.chunk_id: parent for parent in result.parents}

    assert all(
        child.parent_id in parents and child.document_id == parents[child.parent_id].document_id
        for child in result.children
    )


def test_chunk_ids_are_unique_and_deterministic() -> None:
    block = _block("## A\n条款 ID：TEST-ONE\n甲\n条款 ID：TEST-TWO\n乙")
    first = _chunker().chunk(block)
    second = _chunker().chunk(block)

    assert first == second
    assert len({item.chunk_id for item in first.parents}) == len(first.parents)
    assert len({item.chunk_id for item in first.children}) == len(first.children)


def test_empty_block_produces_no_chunks() -> None:
    assert _chunker().chunk(_block(" \n")).parents == ()


def test_invalid_strategy_id_is_rejected() -> None:
    try:
        ClauseAwareChunker(
            parent_chunk_size=100,
            child_chunk_size=50,
            chunk_overlap=10,
            strategy_id="other",
        )
    except ValueError as exc:
        assert "strategy_id" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("invalid strategy_id must fail")
