"""Offline tests for local text, Markdown, and PDF parsers."""

import hashlib
from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from decision_agent.exceptions import (
    DocumentDecodingError,
    DocumentParsingError,
    DocumentTooLargeError,
    InvalidDocumentSourceError,
    UnsupportedDocumentTypeError,
)
from decision_agent.ingestion import (
    DocumentParser,
    MarkdownDocumentParser,
    ParentChildChunker,
    PdfDocumentParser,
    TextDocumentParser,
)


def write_pdf(path: Path, pages: list[str | None], *, password: str | None = None) -> None:
    """Generate a minimal local PDF with optional text content streams."""
    writer = PdfWriter()
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_reference = writer._add_object(font)

    for text in pages:
        page = writer.add_blank_page(width=612, height=792)
        if text is not None:
            stream = DecodedStreamObject()
            stream.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("ascii"))
            page[NameObject("/Contents")] = writer._add_object(stream)
            page[NameObject("/Resources")] = DictionaryObject(
                {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_reference})}
            )

    if password is not None:
        writer.encrypt(password)
    writer.write(path)


def parse_text(path: Path, **kwargs: object):
    return TextDocumentParser(**kwargs).parse(str(path), document_id="doc-1", document_version="v1")


def test_utf8_txt_preserves_english_content(tmp_path: Path) -> None:
    path = tmp_path / "report.txt"
    path.write_bytes(b"Quarterly sales improved.")

    blocks = parse_text(path)

    assert len(blocks) == 1
    assert blocks[0].content == "Quarterly sales improved."


def test_utf8_sig_txt_removes_only_the_bom(tmp_path: Path) -> None:
    path = tmp_path / "bom.txt"
    path.write_bytes("BOM content".encode("utf-8-sig"))

    block = parse_text(path)[0]

    assert block.content == "BOM content"
    assert not block.content.startswith("\ufeff")


def test_utf8_txt_preserves_chinese_content(tmp_path: Path) -> None:
    path = tmp_path / "chinese.txt"
    path.write_text("华东区域销售分析", encoding="utf-8")

    assert parse_text(path)[0].content == "华东区域销售分析"


def test_markdown_content_is_not_heading_split_or_normalized(tmp_path: Path) -> None:
    content = "# Overview\n\n- Product A\n- Product B\n"
    path = tmp_path / "overview.markdown"
    path.write_bytes(content.encode("utf-8"))

    blocks = MarkdownDocumentParser().parse(str(path), document_id="doc-md", document_version="v2")

    assert len(blocks) == 1
    assert blocks[0].content == content


def test_text_block_contains_complete_provenance_and_metadata(tmp_path: Path) -> None:
    path = tmp_path / "facts.txt"
    data = b"facts"
    path.write_bytes(data)

    block = TextDocumentParser().parse(
        str(path),
        document_id="doc-facts",
        document_version="2026-07",
        metadata={"department": "sales", "approved": True},
    )[0]

    assert block.document_id == "doc-facts"
    assert block.document_version == "2026-07"
    assert block.block_index == 0
    assert block.page_number is None
    assert block.source == str(path)
    assert block.file_name == "facts.txt"
    assert block.file_suffix == ".txt"
    assert block.mime_type == "text/plain"
    assert block.file_sha256 == hashlib.sha256(data).hexdigest()
    assert block.block_content_sha256 == hashlib.sha256(b"facts").hexdigest()
    assert block.parser_name == "TextDocumentParser"
    assert block.metadata == {"department": "sales", "approved": True}


def test_block_id_is_stable_for_same_content_identity_and_version(tmp_path: Path) -> None:
    first_path = tmp_path / "first.txt"
    second_path = tmp_path / "second.txt"
    first_path.write_text("stable", encoding="utf-8")
    second_path.write_text("stable", encoding="utf-8")
    parser = TextDocumentParser()

    first = parser.parse(str(first_path), document_id="doc", document_version="v1")[0]
    second = parser.parse(str(second_path), document_id="doc", document_version="v1")[0]

    assert first.block_id == second.block_id


def test_document_version_change_preserves_stable_block_id(tmp_path: Path) -> None:
    path = tmp_path / "versioned.txt"
    path.write_bytes(b"stable block")
    parser = TextDocumentParser()

    first = parser.parse(str(path), document_id="doc", document_version="v1")[0]
    second = parser.parse(str(path), document_id="doc", document_version="v2")[0]

    assert first.document_version != second.document_version
    assert first.block_id == second.block_id


def test_different_document_ids_produce_different_block_ids(tmp_path: Path) -> None:
    path = tmp_path / "same.txt"
    path.write_text("same", encoding="utf-8")
    parser = TextDocumentParser()

    first = parser.parse(str(path), document_id="doc-1", document_version="v1")[0]
    second = parser.parse(str(path), document_id="doc-2", document_version="v1")[0]

    assert first.block_id != second.block_id


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(InvalidDocumentSourceError, match="does not exist"):
        parse_text(tmp_path / "missing.txt")


def test_directory_is_rejected_as_document_source(tmp_path: Path) -> None:
    directory = tmp_path / "folder.txt"
    directory.mkdir()

    with pytest.raises(InvalidDocumentSourceError, match="not a regular file"):
        parse_text(directory)


def test_network_url_is_rejected() -> None:
    with pytest.raises(InvalidDocumentSourceError, match="local file"):
        TextDocumentParser().parse(
            "https://example.com/report.txt", document_id="doc", document_version="v1"
        )


def test_concrete_parsers_satisfy_document_parser_protocol() -> None:
    assert isinstance(TextDocumentParser(), DocumentParser)
    assert isinstance(MarkdownDocumentParser(), DocumentParser)
    assert isinstance(PdfDocumentParser(), DocumentParser)


def test_parser_rejects_an_unsupported_suffix(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("notes", encoding="utf-8")

    with pytest.raises(UnsupportedDocumentTypeError, match="does not support"):
        parse_text(path)


def test_oversized_file_is_rejected_before_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "large.txt"
    path.write_bytes(b"12345")

    def fail_if_opened(*args: object, **kwargs: object) -> None:
        raise AssertionError("oversized file must not be opened")

    monkeypatch.setattr(Path, "open", fail_if_opened)
    with pytest.raises(DocumentTooLargeError, match="exceeds maximum"):
        parse_text(path, max_file_size=4)


def test_file_at_exact_size_limit_is_read_normally(tmp_path: Path) -> None:
    path = tmp_path / "exact.txt"
    path.write_bytes(b"1234")

    block = parse_text(path, max_file_size=4)[0]

    assert block.content == "1234"
    assert block.file_sha256 == hashlib.sha256(b"1234").hexdigest()


def test_undecodable_text_raises_explicit_error(tmp_path: Path) -> None:
    path = tmp_path / "damaged.txt"
    path.write_bytes(b"\xff\xfe\xfa")

    with pytest.raises(DocumentDecodingError, match="cannot decode"):
        parse_text(path)


def test_pdf_pages_are_returned_in_page_order(tmp_path: Path) -> None:
    path = tmp_path / "pages.pdf"
    write_pdf(path, ["First page", "Second page"])

    blocks = PdfDocumentParser().parse(str(path), document_id="pdf-1", document_version="v1")

    assert [block.page_number for block in blocks] == [1, 2]
    assert [block.block_index for block in blocks] == [0, 1]
    assert "First page" in blocks[0].content
    assert "Second page" in blocks[1].content


def test_pdf_blank_page_is_preserved(tmp_path: Path) -> None:
    path = tmp_path / "blank.pdf"
    write_pdf(path, ["Text page", None])

    blocks = PdfDocumentParser().parse(str(path), document_id="pdf-blank", document_version="v1")

    assert len(blocks) == 2
    assert blocks[1].page_number == 2
    assert blocks[1].content == ""
    assert blocks[1].block_content_sha256 == hashlib.sha256(b"").hexdigest()


def test_changing_one_pdf_page_only_changes_that_pages_block_identity(tmp_path: Path) -> None:
    path = tmp_path / "incremental.pdf"
    parser = PdfDocumentParser()
    write_pdf(path, ["Original first", "Stable second"])
    first = parser.parse(str(path), document_id="pdf-doc", document_version="v1")

    write_pdf(path, ["Changed first", "Stable second"])
    second = parser.parse(str(path), document_id="pdf-doc", document_version="v2")

    assert first[0].block_id != second[0].block_id
    assert first[0].block_content_sha256 != second[0].block_content_sha256
    assert first[1].block_id == second[1].block_id
    assert first[1].block_content_sha256 == second[1].block_content_sha256
    assert first[0].file_sha256 != second[0].file_sha256
    assert first[1].file_sha256 != second[1].file_sha256


def test_pdf_blocks_hold_independent_metadata_copies(tmp_path: Path) -> None:
    path = tmp_path / "metadata.pdf"
    write_pdf(path, ["First", "Second"])
    metadata = {"department": "operations", "approved": True}

    blocks = PdfDocumentParser().parse(
        str(path), document_id="pdf-meta", document_version="v1", metadata=metadata
    )

    assert blocks[0].metadata == metadata
    assert blocks[1].metadata == metadata
    assert blocks[0].metadata is not blocks[1].metadata
    assert blocks[0].metadata is not metadata


def test_encrypted_pdf_raises_explicit_error(tmp_path: Path) -> None:
    path = tmp_path / "encrypted.pdf"
    write_pdf(path, [None], password="secret")

    with pytest.raises(DocumentParsingError, match="encrypted"):
        PdfDocumentParser().parse(str(path), document_id="pdf-secret", document_version="v1")


def test_malformed_pdf_raises_explicit_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"not a valid PDF")

    with pytest.raises(DocumentParsingError, match="cannot parse"):
        PdfDocumentParser().parse(str(path), document_id="pdf-broken", document_version="v1")


def test_parser_output_can_flow_into_parent_child_chunker(tmp_path: Path) -> None:
    path = tmp_path / "pipeline.txt"
    path.write_text("abcdefghij", encoding="utf-8")
    block = parse_text(path)[0]

    result = ParentChildChunker(10, 5, 1).chunk(block)

    assert len(result.parents) == 1
    assert result.children
    assert all(child.document_id == block.document_id for child in result.children)
