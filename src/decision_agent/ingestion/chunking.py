"""Pure-Python deterministic parent-child character chunking."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass

from decision_agent.domain import ChildChunk, DocumentBlock, ParentChunk
from decision_agent.ingestion.protocols import ChunkingResult

_ID_VERSION = "parent-child-v1"


@dataclass(frozen=True, slots=True)
class ParentChildChunker:
    """Split document blocks into non-overlapping parents and overlapping children."""

    parent_chunk_size: int
    child_chunk_size: int
    chunk_overlap: int = 0

    def __post_init__(self) -> None:
        """Validate window sizes before processing any content."""
        if self.parent_chunk_size <= 0:
            raise ValueError("parent_chunk_size must be greater than zero")
        if self.child_chunk_size <= 0:
            raise ValueError("child_chunk_size must be greater than zero")
        if self.child_chunk_size > self.parent_chunk_size:
            raise ValueError("child_chunk_size must not exceed parent_chunk_size")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must not be negative")
        if self.chunk_overlap >= self.child_chunk_size:
            raise ValueError("chunk_overlap must be smaller than child_chunk_size")

    def chunk(self, block: DocumentBlock) -> ChunkingResult:
        """Create linked chunks while preserving provenance and global offsets."""
        if not block.content.strip():
            return ChunkingResult(parents=(), children=())

        parents: list[ParentChunk] = []
        children: list[ChildChunk] = []

        for parent_start, parent_end in self._windows(
            length=len(block.content), size=self.parent_chunk_size, overlap=0
        ):
            parent_content = block.content[parent_start:parent_end]
            if not parent_content.strip():
                continue

            parent_id = self._stable_id(
                kind="parent",
                block=block,
                start=parent_start,
                end=parent_end,
                content=parent_content,
            )
            parent = ParentChunk(
                chunk_id=parent_id,
                document_id=block.document_id,
                document_version=block.document_version,
                content=parent_content,
                block_ids=[block.block_id],
                page_number=block.page_number,
                source=block.source,
                start_offset=parent_start,
                end_offset=parent_end,
                metadata=dict(block.metadata),
            )
            parents.append(parent)

            for local_start, local_end in self._windows(
                length=len(parent_content),
                size=self.child_chunk_size,
                overlap=self.chunk_overlap,
            ):
                child_content = parent_content[local_start:local_end]
                if not child_content.strip():
                    continue
                child_start = parent_start + local_start
                child_end = parent_start + local_end
                children.append(
                    ChildChunk(
                        chunk_id=self._stable_id(
                            kind="child",
                            block=block,
                            start=child_start,
                            end=child_end,
                            content=child_content,
                            parent_id=parent_id,
                        ),
                        parent_id=parent_id,
                        document_id=block.document_id,
                        document_version=block.document_version,
                        content=child_content,
                        page_number=block.page_number,
                        source=block.source,
                        start_offset=child_start,
                        end_offset=child_end,
                        metadata=dict(block.metadata),
                    )
                )

        return ChunkingResult(parents=tuple(parents), children=tuple(children))

    @staticmethod
    def _windows(*, length: int, size: int, overlap: int) -> Iterator[tuple[int, int]]:
        """Yield bounded windows without duplicating the final short window."""
        start = 0
        while start < length:
            end = min(start + size, length)
            yield start, end
            if end == length:
                break
            start = end - overlap

    def _stable_id(
        self,
        *,
        kind: str,
        block: DocumentBlock,
        start: int,
        end: int,
        content: str,
        parent_id: str | None = None,
    ) -> str:
        """Build a portable content identity using canonical JSON and SHA-256."""
        payload = {
            "version": _ID_VERSION,
            "kind": kind,
            "document_id": block.document_id,
            "document_version": block.document_version,
            "block_id": block.block_id,
            "page_number": block.page_number,
            "source": block.source,
            "metadata": block.metadata,
            "start": start,
            "end": end,
            "content": content,
            "parent_id": parent_id,
            "parent_chunk_size": self.parent_chunk_size,
            "child_chunk_size": self.child_chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return f"{kind}_{hashlib.sha256(encoded).hexdigest()}"
