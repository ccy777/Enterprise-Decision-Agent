"""Tests for deterministic suffix-based parser routing."""

from pathlib import Path

import pytest

from decision_agent.exceptions import UnsupportedDocumentTypeError
from decision_agent.ingestion import (
    MarkdownDocumentParser,
    ParserRegistry,
    PdfDocumentParser,
    TextDocumentParser,
)


@pytest.mark.parametrize(
    ("file_name", "parser_type"),
    [
        ("document.txt", TextDocumentParser),
        ("document.md", MarkdownDocumentParser),
        ("document.markdown", MarkdownDocumentParser),
        ("document.pdf", PdfDocumentParser),
    ],
)
def test_registry_routes_supported_suffixes(file_name: str, parser_type: type[object]) -> None:
    parser = ParserRegistry.default().parser_for(file_name)

    assert isinstance(parser, parser_type)


def test_registry_routes_suffixes_case_insensitively() -> None:
    registry = ParserRegistry.default()

    assert isinstance(registry.parser_for("REPORT.TXT"), TextDocumentParser)
    assert isinstance(registry.parser_for("README.MarkDown"), MarkdownDocumentParser)
    assert isinstance(registry.parser_for("REPORT.PDF"), PdfDocumentParser)


def test_registry_rejects_unsupported_suffix() -> None:
    with pytest.raises(UnsupportedDocumentTypeError, match="no document parser"):
        ParserRegistry.default().parser_for("report.docx")


def test_registry_parse_routes_and_preserves_identity(tmp_path: Path) -> None:
    path = tmp_path / "routed.txt"
    path.write_text("registry content", encoding="utf-8")

    blocks = ParserRegistry.default().parse(
        str(path),
        document_id="registry-doc",
        document_version="v3",
        metadata={"owner": "operations"},
    )

    assert blocks[0].document_id == "registry-doc"
    assert blocks[0].document_version == "v3"
    assert blocks[0].parser_name == "TextDocumentParser"
    assert blocks[0].metadata == {"owner": "operations"}


def test_registry_rejects_duplicate_suffix_registration() -> None:
    with pytest.raises(ValueError, match="already registered"):
        ParserRegistry([TextDocumentParser(), TextDocumentParser()])
