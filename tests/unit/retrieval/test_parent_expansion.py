import pytest

from decision_agent.domain import DocumentBlock, ParentChunk
from decision_agent.exceptions import RetrievalValidationError
from decision_agent.ingestion import ParentChildChunker
from decision_agent.retrieval.parent_expansion import (
    InMemoryParentChunkResolver,
    ParentChildCandidate,
    ParentExpander,
)


def parent(identifier: str, document: str = "doc") -> ParentChunk:
    return ParentChunk(
        chunk_id=identifier,
        document_id=document,
        document_version="v1",
        content=f"parent {identifier}",
        start_offset=0,
        end_offset=10,
    )


def child(
    identifier: str, parent_id: str, rank: int, document: str = "doc"
) -> ParentChildCandidate:
    return ParentChildCandidate(
        child_id=identifier,
        parent_id=parent_id,
        document_id=document,
        content=f"child {identifier}",
        upstream_rank=rank,
        metadata={"x": rank},
    )


def test_groups_children_and_preserves_order_and_copies() -> None:
    source = [child("a", "p1", 1), child("b", "p1", 2), child("c", "p2", 3)]
    results = ParentExpander(InMemoryParentChunkResolver([parent("p1"), parent("p2")])).expand(
        source
    )
    assert [
        (item.final_rank, item.parent_id, item.best_child_rank, item.matched_child_count)
        for item in results
    ] == [(1, "p1", 1, 2), (2, "p2", 3, 1)]
    source[0].metadata["x"] = 99
    assert results[0].matched_children[0].metadata == {"x": 1}


def test_empty_does_not_call_resolver() -> None:
    class Resolver:
        def resolve(self, parent_ids: object) -> object:
            raise AssertionError("called")

    assert ParentExpander(Resolver()).expand([]) == []


@pytest.mark.parametrize(
    "items", [[child("a", "p1", 2)], [child("a", "p1", 1), child("a", "p2", 2)]]
)
def test_invalid_child_ranks_or_duplicates_rejected(items: list[ParentChildCandidate]) -> None:
    with pytest.raises(RetrievalValidationError):
        ParentExpander(InMemoryParentChunkResolver([parent("p1"), parent("p2")])).expand(items)


def test_missing_parent_document_mismatch_and_top_k() -> None:
    expander = ParentExpander(InMemoryParentChunkResolver([parent("p1"), parent("p2")]))
    assert [
        item.parent_id
        for item in expander.expand([child("a", "p1", 1), child("b", "p2", 2)], top_k=1)
    ] == ["p1"]
    with pytest.raises(RetrievalValidationError, match="every requested"):
        ParentExpander(InMemoryParentChunkResolver([])).expand([child("a", "p1", 1)])
    with pytest.raises(RetrievalValidationError, match="inconsistent"):
        expander.expand([child("a", "p1", 1, "other")])


def test_batch_resolver_and_mismatched_mapping_key_are_rejected() -> None:
    class Resolver:
        calls = 0

        def resolve(self, parent_ids: object) -> object:
            self.calls += 1
            return {"p1": parent("wrong"), "p2": parent("p2")}

    resolver = Resolver()
    with pytest.raises(RetrievalValidationError, match="mismatched parent ID"):
        ParentExpander(resolver).expand([child("a", "p1", 1), child("b", "p2", 2)])
    assert resolver.calls == 1


def test_real_chunker_chain_recovers_parents_offsets_and_provenance() -> None:
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
        next(
            item
            for item in chunks.children
            if item.parent_id == first.chunk_id and item.chunk_id != chunks.children[0].chunk_id
        ),
        next(item for item in chunks.children if item.parent_id == second.chunk_id),
    ]
    candidates = [
        ParentChildCandidate(
            child_id=item.chunk_id,
            parent_id=item.parent_id,
            document_id=item.document_id,
            content=item.content,
            upstream_rank=rank,
            start_offset=item.start_offset,
            end_offset=item.end_offset,
            metadata=item.metadata,
            provenance={"source": item.source},
        )
        for rank, item in enumerate(selected, 1)
    ]
    results = ParentExpander(InMemoryParentChunkResolver(chunks.parents)).expand(candidates)
    assert [result.parent_content for result in results] == [first.content, second.content]
    assert results[0].matched_child_count == 2
    assert results[0].matched_children[0].start_offset == selected[0].start_offset
    assert results[0].provenance["source"] == "source.txt"
