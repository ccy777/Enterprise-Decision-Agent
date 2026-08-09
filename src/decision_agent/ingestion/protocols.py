"""Typed boundaries for document parsing and deterministic chunking."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from decision_agent.domain import ChildChunk, DocumentBlock, ParentChunk
from decision_agent.domain.models import Metadata


@dataclass(frozen=True, slots=True)
class ChunkingResult:
    """Parent and child chunks produced from one document block."""

    parents: tuple[ParentChunk, ...]
    children: tuple[ChildChunk, ...]


@runtime_checkable
class DocumentParser(Protocol):
    """Convert a source document into normalized document blocks."""

    def parse(
        self,
        source: str,
        *,
        document_id: str,
        document_version: str,
        metadata: Metadata | None = None,
    ) -> list[DocumentBlock]:
        """Parse one source into ordered canonical blocks."""
        ...


@runtime_checkable
class ChunkingStrategy(Protocol):
    """Convert a normalized block into linked parent and child chunks."""

    def chunk(self, block: DocumentBlock) -> ChunkingResult:
        """Split one block deterministically without external I/O."""
        ...
