"""Suffix-based routing for local document parsers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from decision_agent.domain import DocumentBlock
from decision_agent.domain.models import Metadata
from decision_agent.exceptions import UnsupportedDocumentTypeError
from decision_agent.ingestion.parsers import (
    DEFAULT_MAX_FILE_SIZE,
    MarkdownDocumentParser,
    PdfDocumentParser,
    TextDocumentParser,
)
from decision_agent.ingestion.protocols import DocumentParser


class RegisteredDocumentParser(DocumentParser, Protocol):
    """Parser contract enriched with suffix registration metadata."""

    supported_suffixes: frozenset[str]


class ParserRegistry:
    """Route a local source to exactly one parser using a case-insensitive suffix."""

    def __init__(self, parsers: Iterable[RegisteredDocumentParser]) -> None:
        self._parsers: dict[str, RegisteredDocumentParser] = {}
        for parser in parsers:
            self.register(parser)

    @classmethod
    def default(cls, *, max_file_size: int = DEFAULT_MAX_FILE_SIZE) -> ParserRegistry:
        """Create the M2A registry without reading files or creating clients."""
        return cls(
            [
                TextDocumentParser(max_file_size=max_file_size),
                MarkdownDocumentParser(max_file_size=max_file_size),
                PdfDocumentParser(max_file_size=max_file_size),
            ]
        )

    def register(self, parser: RegisteredDocumentParser) -> None:
        """Register every suffix exposed by a parser and reject ambiguity."""
        for suffix in parser.supported_suffixes:
            normalized = suffix.lower()
            if normalized in self._parsers:
                raise ValueError(f"parser already registered for suffix: {normalized}")
            self._parsers[normalized] = parser

    def parser_for(self, source: str) -> RegisteredDocumentParser:
        """Return the parser selected for a source suffix."""
        suffix = Path(source).suffix.lower()
        parser = self._parsers.get(suffix)
        if parser is None:
            raise UnsupportedDocumentTypeError(
                f"no document parser registered for suffix: {suffix or '<none>'}"
            )
        return parser

    def parse(
        self,
        source: str,
        *,
        document_id: str,
        document_version: str,
        metadata: Metadata | None = None,
    ) -> list[DocumentBlock]:
        """Route and parse one source through the canonical parser contract."""
        return self.parser_for(source).parse(
            source,
            document_id=document_id,
            document_version=document_version,
            metadata=metadata,
        )
