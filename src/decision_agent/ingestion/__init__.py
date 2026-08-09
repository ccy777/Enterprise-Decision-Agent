"""Deterministic document ingestion contracts and chunking services."""

from decision_agent.ingestion.chunking import ParentChildChunker
from decision_agent.ingestion.clause_aware_chunking import ClauseAwareChunker
from decision_agent.ingestion.parsers import (
    MarkdownDocumentParser,
    PdfDocumentParser,
    TextDocumentParser,
)
from decision_agent.ingestion.protocols import ChunkingResult, ChunkingStrategy, DocumentParser
from decision_agent.ingestion.registry import ParserRegistry

__all__ = [
    "ChunkingResult",
    "ChunkingStrategy",
    "ClauseAwareChunker",
    "DocumentParser",
    "MarkdownDocumentParser",
    "ParentChildChunker",
    "ParserRegistry",
    "PdfDocumentParser",
    "TextDocumentParser",
]
