"""Offline parsers for local text, Markdown, and text-based PDF documents."""

from __future__ import annotations

import codecs
import hashlib
import json
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import FileNotDecryptedError, PdfReadError

from decision_agent.domain import DocumentBlock
from decision_agent.domain.models import Metadata
from decision_agent.exceptions import (
    DocumentDecodingError,
    DocumentParsingError,
    DocumentTooLargeError,
    InvalidDocumentSourceError,
    UnsupportedDocumentTypeError,
)

DEFAULT_MAX_FILE_SIZE = 10 * 1024 * 1024
_BLOCK_ID_VERSION = "document-block-v1"


class _LocalFileParser:
    """Shared validation, bounded reading, and provenance construction."""

    supported_suffixes: frozenset[str]
    mime_type: str
    parser_name: str

    def __init__(self, *, max_file_size: int = DEFAULT_MAX_FILE_SIZE) -> None:
        if max_file_size <= 0:
            raise ValueError("max_file_size must be greater than zero")
        self.max_file_size = max_file_size

    def _read_source(self, source: str) -> tuple[Path, bytes]:
        if "://" in source:
            raise InvalidDocumentSourceError("document source must be a local file path")

        path = Path(source)
        if not path.exists():
            raise InvalidDocumentSourceError(f"document file does not exist: {source}")
        if not path.is_file():
            raise InvalidDocumentSourceError(f"document source is not a regular file: {source}")

        suffix = path.suffix.lower()
        if suffix not in self.supported_suffixes:
            raise UnsupportedDocumentTypeError(
                f"{self.parser_name} does not support file suffix: {path.suffix or '<none>'}"
            )

        try:
            size = path.stat().st_size
        except OSError as exc:
            raise InvalidDocumentSourceError(f"cannot inspect document file: {source}") from exc
        if size > self.max_file_size:
            raise DocumentTooLargeError(
                f"document size {size} exceeds maximum {self.max_file_size} bytes"
            )

        try:
            with path.open("rb") as file:
                data = file.read(self.max_file_size + 1)
        except OSError as exc:
            raise InvalidDocumentSourceError(f"cannot read document file: {source}") from exc
        if len(data) > self.max_file_size:
            raise DocumentTooLargeError(
                f"document exceeds maximum {self.max_file_size} bytes while reading"
            )
        return path, data

    def _make_block(
        self,
        *,
        path: Path,
        file_sha256: str,
        document_id: str,
        document_version: str,
        block_index: int,
        page_number: int | None,
        content: str,
        metadata: Metadata | None,
    ) -> DocumentBlock:
        block_content_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
        block_id = self._stable_block_id(
            document_id=document_id,
            block_index=block_index,
            page_number=page_number,
            block_content_sha256=block_content_sha256,
        )
        return DocumentBlock(
            block_id=block_id,
            document_id=document_id,
            document_version=document_version,
            block_index=block_index,
            page_number=page_number,
            content=content,
            source=str(path),
            file_name=path.name,
            file_suffix=path.suffix.lower(),
            mime_type=self.mime_type,
            file_sha256=file_sha256,
            block_content_sha256=block_content_sha256,
            parser_name=self.parser_name,
            metadata=dict(metadata or {}),
        )

    def _stable_block_id(
        self,
        *,
        document_id: str,
        block_index: int,
        page_number: int | None,
        block_content_sha256: str,
    ) -> str:
        payload = {
            "schema_version": _BLOCK_ID_VERSION,
            "parser_name": self.parser_name,
            "document_id": document_id,
            "block_index": block_index,
            "page_number": page_number,
            "block_content_sha256": block_content_sha256,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return f"block_{hashlib.sha256(encoded).hexdigest()}"


class _TextFileParser(_LocalFileParser):
    """Shared strict text decoding for TXT and Markdown parsers."""

    def __init__(
        self,
        *,
        max_file_size: int = DEFAULT_MAX_FILE_SIZE,
        fallback_encodings: tuple[str, ...] = (),
    ) -> None:
        super().__init__(max_file_size=max_file_size)
        self.encodings = ("utf-8-sig", *fallback_encodings)
        for encoding in self.encodings:
            try:
                codecs.lookup(encoding)
            except LookupError as exc:
                raise ValueError(f"unknown text encoding: {encoding}") from exc

    def parse(
        self,
        source: str,
        *,
        document_id: str,
        document_version: str,
        metadata: Metadata | None = None,
    ) -> list[DocumentBlock]:
        path, data = self._read_source(source)
        content = self._decode(data, source)
        file_sha256 = hashlib.sha256(data).hexdigest()
        return [
            self._make_block(
                path=path,
                file_sha256=file_sha256,
                document_id=document_id,
                document_version=document_version,
                block_index=0,
                page_number=None,
                content=content,
                metadata=metadata,
            )
        ]

    def _decode(self, data: bytes, source: str) -> str:
        failures: list[UnicodeDecodeError] = []
        for encoding in self.encodings:
            try:
                return data.decode(encoding, errors="strict")
            except UnicodeDecodeError as exc:
                failures.append(exc)
        names = ", ".join(self.encodings)
        raise DocumentDecodingError(
            f"cannot decode document {source} using configured encodings: {names}"
        ) from failures[-1]


class TextDocumentParser(_TextFileParser):
    """Parse one local TXT file into one canonical document block."""

    supported_suffixes = frozenset({".txt"})
    mime_type = "text/plain"
    parser_name = "TextDocumentParser"


class MarkdownDocumentParser(_TextFileParser):
    """Parse one local Markdown file without heading-based splitting."""

    supported_suffixes = frozenset({".md", ".markdown"})
    mime_type = "text/markdown"
    parser_name = "MarkdownDocumentParser"


class PdfDocumentParser(_LocalFileParser):
    """Extract ordered pages from local, unencrypted, text-based PDFs."""

    supported_suffixes = frozenset({".pdf"})
    mime_type = "application/pdf"
    parser_name = "PdfDocumentParser"

    def parse(
        self,
        source: str,
        *,
        document_id: str,
        document_version: str,
        metadata: Metadata | None = None,
    ) -> list[DocumentBlock]:
        path, data = self._read_source(source)
        file_sha256 = hashlib.sha256(data).hexdigest()
        try:
            reader = PdfReader(BytesIO(data), strict=True)
            if reader.is_encrypted:
                raise DocumentParsingError("encrypted PDF documents are not supported")

            blocks: list[DocumentBlock] = []
            for block_index, page in enumerate(reader.pages):
                content = page.extract_text() or ""
                page_number = block_index + 1
                blocks.append(
                    self._make_block(
                        path=path,
                        file_sha256=file_sha256,
                        document_id=document_id,
                        document_version=document_version,
                        block_index=block_index,
                        page_number=page_number,
                        content=content,
                        metadata=metadata,
                    )
                )
            return blocks
        except DocumentParsingError:
            raise
        except (FileNotDecryptedError, PdfReadError, OSError, ValueError) as exc:
            raise DocumentParsingError(f"cannot parse PDF document: {source}") from exc
        except Exception as exc:
            raise DocumentParsingError(f"cannot extract text from PDF document: {source}") from exc
