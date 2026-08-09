# ruff: noqa: RUF001
import pytest

from decision_agent.domain import DocumentBlock
from decision_agent.exceptions import RetrievalValidationError
from decision_agent.ingestion import ParentChildChunker
from decision_agent.retrieval.evidence_context import EvidenceContextBuilder
from decision_agent.retrieval.parent_expansion import (
    InMemoryParentChunkResolver,
    MatchedChild,
    ParentChildCandidate,
    ParentExpander,
    ParentExpansionResult,
)

pytestmark = pytest.mark.offline_integration


def parent(
    identifier: str, rank: int, content: str = "abcdefghij", *, source: bool = True
) -> ParentExpansionResult:
    child = MatchedChild(
        child_id=f"c-{identifier}",
        parent_id=identifier,
        document_id="doc",
        content="child",
        upstream_rank=rank,
        start_offset=0,
        end_offset=5,
        metadata={"x": 1},
        provenance={"source": "child.txt"},
    )
    return ParentExpansionResult(
        final_rank=rank,
        parent_id=identifier,
        document_id="doc",
        parent_content=content,
        best_child_rank=rank,
        matched_child_count=1,
        matched_children=(child,),
        metadata={"page_number": 2, "start_offset": 0, "end_offset": 10},
        provenance={"source": "report.txt"} if source else {},
    )


def builder(
    total: int = 200, item: int = 100, count: int = 5, marker: str = "…[截断]"
) -> EvidenceContextBuilder:
    return EvidenceContextBuilder(
        max_total_chars=total,
        max_evidence_chars=item,
        max_evidence_count=count,
        truncation_marker=marker,
    )


def test_empty_context_is_stable() -> None:
    result = builder().build([])
    assert result.rendered_context == "" and result.evidence_items == () and not result.truncated


def test_multiple_items_preserve_order_ids_and_references() -> None:
    result = builder().build([parent("p1", 1), parent("p2", 2)])
    assert [item.parent_id for item in result.evidence_items] == ["p1", "p2"]
    assert [item.evidence_id for item in result.evidence_items] == ["E1", "E2"]
    assert [ref.evidence_id for ref in result.references] == ["E1", "E2"]
    assert result.rendered_context.index("[E1]") < result.rendered_context.index("[E2]")


def test_single_item_limit_and_total_statistics() -> None:
    result = builder(count=1).build([parent("p1", 1, "abc"), parent("p2", 2, "12345")])
    assert result.included_evidence_count == 1 and result.omitted_evidence_count == 1
    assert result.total_original_chars == 8 and result.total_included_chars == 3
    assert result.truncated


def test_per_item_truncation_includes_marker_within_limit() -> None:
    result = builder(item=6, marker="...").build([parent("p1", 1, "abcdefgh")])
    assert result.evidence_items[0].content == "abc..."
    assert result.evidence_items[0].included_content_length == 6


def test_total_budget_truncates_and_never_overflows() -> None:
    header_length = len("[E1]\n文档ID：doc\n来源：report.txt\n页码：2\n偏移：0-10\n内容：")
    result = builder(total=header_length + 6, marker="...").build([parent("p1", 1, "abcdefgh")])
    assert result.evidence_items[0].content == "abc..."
    assert len(result.rendered_context) == header_length + 6


def test_budget_that_cannot_fit_original_and_marker_omits_item() -> None:
    header_length = len("[E1]\n文档ID：doc\n来源：report.txt\n页码：2\n偏移：0-10\n内容：")
    result = builder(total=header_length + 3, marker="...").build([parent("p1", 1)])
    assert result.included_evidence_count == 0 and result.omitted_evidence_count == 1


def test_chinese_budget_counts_python_characters() -> None:
    result = builder(item=4, marker="…").build([parent("p1", 1, "甲乙丙丁戊")])
    assert result.evidence_items[0].content == "甲乙丙…" and result.total_included_chars == 4


def test_missing_source_and_page_are_not_fabricated() -> None:
    item = parent("p1", 1, source=False).model_copy(update={"metadata": {}})
    result = builder().build([item])
    assert result.references[0].source is None and result.references[0].page_number is None
    assert "来源：" not in result.rendered_context and "文档ID：doc" in result.rendered_context


def test_input_and_nested_output_are_deep_copied() -> None:
    source = [parent("p1", 1)]
    result = builder().build(source)
    source[0].metadata["page_number"] = 9
    result.evidence_items[0].metadata["page_number"] = 8
    result.evidence_items[0].matched_children[0].metadata["x"] = 7
    assert (
        source[0].metadata["page_number"] == 9 and source[0].matched_children[0].metadata["x"] == 1
    )


@pytest.mark.parametrize("parents", [[parent("p1", 1), parent("p1", 2)], [parent("p1", 2)]])
def test_duplicate_parent_or_nonconsecutive_rank_is_rejected(
    parents: list[ParentExpansionResult],
) -> None:
    with pytest.raises(RetrievalValidationError):
        builder().build(parents)


def test_blank_content_and_inconsistent_child_count_are_rejected() -> None:
    blank = parent("p1", 1).model_copy(update={"parent_content": " "})
    count = parent("p1", 1).model_copy(update={"matched_child_count": 2})
    with pytest.raises(RetrievalValidationError, match="blank"):
        builder().build([blank])
    with pytest.raises(RetrievalValidationError, match="count"):
        builder().build([count])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_total_chars": 0, "max_evidence_chars": 1, "max_evidence_count": 1},
        {"max_total_chars": 1, "max_evidence_chars": -1, "max_evidence_count": 1},
        {"max_total_chars": 1, "max_evidence_chars": 1, "max_evidence_count": True},
    ],
)
def test_invalid_limits_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(RetrievalValidationError):
        EvidenceContextBuilder(**kwargs)  # type: ignore[arg-type]


def test_same_input_is_deterministic() -> None:
    source = [parent("p1", 1), parent("p2", 2)]
    assert builder().build(source) == builder().build(source)


def test_real_offline_chunker_expander_context_chain() -> None:
    block = DocumentBlock(
        block_id="block",
        document_id="doc",
        document_version="v1",
        content="abcdefghijklmnopqrst",
        source="source.txt",
    )
    chunks = ParentChildChunker(10, 5, 1).chunk(block)
    first, second = chunks.parents
    selected = [
        chunks.children[0],
        chunks.children[1],
        next(child for child in chunks.children if child.parent_id == second.chunk_id),
    ]
    candidates = [
        ParentChildCandidate(
            child_id=child.chunk_id,
            parent_id=child.parent_id,
            document_id=child.document_id,
            content=child.content,
            upstream_rank=rank,
            start_offset=child.start_offset,
            end_offset=child.end_offset,
            provenance={"source": child.source},
        )
        for rank, child in enumerate(selected, 1)
    ]
    expanded = ParentExpander(InMemoryParentChunkResolver(chunks.parents)).expand(candidates)
    context = builder(total=500).build(expanded)
    assert [item.parent_id for item in context.evidence_items] == [first.chunk_id, second.chunk_id]
    assert context.evidence_items[0].matched_child_count == 2
    assert context.evidence_items[0].content == first.content
    assert context.references[0].source == "source.txt"
    assert "[E1]" in context.rendered_context and "[E2]" in context.rendered_context
