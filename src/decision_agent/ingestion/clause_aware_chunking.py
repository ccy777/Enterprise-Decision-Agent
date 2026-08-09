"""Deterministic Markdown section and Clause-aware parent/child chunking."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

from decision_agent.domain import ChildChunk, DocumentBlock, ParentChunk
from decision_agent.domain.models import Metadata
from decision_agent.ingestion.chunking import ParentChildChunker
from decision_agent.ingestion.protocols import ChunkingResult

STRATEGY_ID = "clause-aware-v1"
_HEADING = re.compile(r"^(?P<marks>#{1,6})[ \t]+(?P<title>.*?)[ \t]*#*[ \t]*$", re.MULTILINE)
_CLAUSE_MARKER = re.compile(
    r"^条款 ID\N{FULLWIDTH COLON}\s*(?P<clause_id>[A-Z][A-Z0-9-]*)\s*$",
    re.MULTILINE,
)


@dataclass(frozen=True, slots=True)
class _Section:
    start: int
    end: int
    path: str
    nearest_heading: str


@dataclass(frozen=True, slots=True)
class _Clause:
    clause_id: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class ClauseAwareChunker(ParentChildChunker):
    """Respect Markdown business sections and Clause marker boundaries.

    Parent chunks collect complete adjacent Clauses within one ``##`` business
    section.  Children are Clause-local: ordinary Clauses yield one child, and
    only an overlong Clause uses the inherited bounded-overlap window logic.
    """

    strategy_id: str = STRATEGY_ID

    def __post_init__(self) -> None:
        ParentChildChunker.__post_init__(self)
        if self.strategy_id != STRATEGY_ID:
            raise ValueError(f"strategy_id must be {STRATEGY_ID}")

    def chunk(self, block: DocumentBlock) -> ChunkingResult:
        if not block.content.strip():
            return ChunkingResult(parents=(), children=())

        sections = tuple(self._sections(block.content))
        clauses = tuple(self._clauses(block.content, sections))
        parents: list[ParentChunk] = []
        children: list[ChildChunk] = []

        for section in sections:
            section_clauses = tuple(
                clause
                for clause in clauses
                if section.start <= clause.start and clause.end <= section.end
            )
            for start, end, clause_ids in self._parent_ranges(section, section_clauses):
                content = block.content[start:end]
                if not content.strip():
                    continue
                metadata = self._metadata(
                    block=block,
                    section=section,
                    clause_ids=clause_ids,
                )
                parent_id = self._stable_id(
                    kind="parent", block=block, start=start, end=end, content=content
                )
                parents.append(
                    ParentChunk(
                        chunk_id=parent_id,
                        document_id=block.document_id,
                        document_version=block.document_version,
                        content=content,
                        block_ids=[block.block_id],
                        page_number=block.page_number,
                        source=block.source,
                        start_offset=start,
                        end_offset=end,
                        metadata=metadata,
                    )
                )

        parent_by_range = {(parent.start_offset, parent.end_offset): parent for parent in parents}
        for section in sections:
            for clause in clauses:
                if not (section.start <= clause.start and clause.end <= section.end):
                    continue
                parent = next(
                    (
                        candidate
                        for candidate in parent_by_range.values()
                        if candidate.start_offset <= clause.start
                        and clause.end <= candidate.end_offset
                    ),
                    None,
                )
                if parent is None:  # pragma: no cover - construction invariant
                    raise ValueError(f"Clause {clause.clause_id} has no Parent chunk")
                for start, end in self._clause_windows(clause):
                    content = block.content[start:end]
                    children.append(
                        ChildChunk(
                            chunk_id=self._stable_id(
                                kind="child",
                                block=block,
                                start=start,
                                end=end,
                                content=content,
                                parent_id=parent.chunk_id,
                            ),
                            parent_id=parent.chunk_id,
                            document_id=block.document_id,
                            document_version=block.document_version,
                            content=content,
                            page_number=block.page_number,
                            source=block.source,
                            start_offset=start,
                            end_offset=end,
                            metadata=self._metadata(
                                block=block,
                                section=section,
                                clause_ids=(clause.clause_id,),
                            ),
                        )
                    )
        return ChunkingResult(parents=tuple(parents), children=tuple(children))

    def _sections(self, content: str) -> Iterator[_Section]:
        headings = list(_HEADING.finditer(content))
        business_level = min(
            (len(match.group("marks")) for match in headings if len(match.group("marks")) >= 2),
            default=1,
        )
        boundaries = [match for match in headings if len(match.group("marks")) == business_level]
        if not boundaries:
            yield _Section(0, len(content), "", "")
            return
        for index, heading in enumerate(boundaries):
            start = 0 if index == 0 else heading.start()
            end = boundaries[index + 1].start() if index + 1 < len(boundaries) else len(content)
            path = self._section_path(headings, heading.start())
            yield _Section(start, end, path, heading.group("title").strip())

    @staticmethod
    def _section_path(headings: list[re.Match[str]], position: int) -> str:
        stack: dict[int, str] = {}
        for heading in headings:
            if heading.start() > position:
                break
            level = len(heading.group("marks"))
            stack[level] = heading.group("title").strip()
            for nested in tuple(stack):
                if nested > level:
                    del stack[nested]
        return " > ".join(stack[level] for level in sorted(stack))

    @staticmethod
    def _clauses(content: str, sections: tuple[_Section, ...]) -> Iterator[_Clause]:
        markers = list(_CLAUSE_MARKER.finditer(content))
        if not markers:
            return
        for index, marker in enumerate(markers):
            section = next(item for item in sections if item.start <= marker.start() < item.end)
            next_marker_start = (
                markers[index + 1].start() if index + 1 < len(markers) else section.end
            )
            yield _Clause(
                clause_id=marker.group("clause_id"),
                start=marker.start(),
                end=min(next_marker_start, section.end),
            )

    def _parent_ranges(
        self, section: _Section, clauses: tuple[_Clause, ...]
    ) -> Iterator[tuple[int, int, tuple[str, ...]]]:
        if not clauses:
            yield section.start, section.end, ()
            return
        current_start = section.start
        current_end = section.start
        current_clause_ids: list[str] = []
        for clause in clauses:
            candidate_end = clause.end
            if (
                current_end > current_start
                and candidate_end - current_start > self.parent_chunk_size
            ):
                yield current_start, current_end, tuple(current_clause_ids)
                current_start = clause.start
                current_end = clause.end
                current_clause_ids = [clause.clause_id]
                continue
            current_end = candidate_end
            current_clause_ids.append(clause.clause_id)
        if current_end > current_start:
            yield current_start, current_end, tuple(current_clause_ids)
        if current_end < section.end:
            yield section.end if False else current_end, section.end, ()

    def _clause_windows(self, clause: _Clause) -> Iterator[tuple[int, int]]:
        length = clause.end - clause.start
        if length <= self.child_chunk_size:
            yield clause.start, clause.end
            return
        for local_start, local_end in self._windows(
            length=length, size=self.child_chunk_size, overlap=self.chunk_overlap
        ):
            yield clause.start + local_start, clause.start + local_end

    def _metadata(
        self,
        *,
        block: DocumentBlock,
        section: _Section,
        clause_ids: tuple[str, ...],
    ) -> Metadata:
        return {
            **dict(block.metadata),
            "strategy_id": self.strategy_id,
            "section_path": section.path,
            "nearest_heading": section.nearest_heading,
            "clause_ids": clause_ids,
        }
