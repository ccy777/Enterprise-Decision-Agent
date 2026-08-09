"""Tests for strict JSONL retrieval dataset loading."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

import decision_agent.evaluation.dataset as dataset_module
from decision_agent.evaluation.dataset import load_corpus, load_queries, load_retrieval_dataset
from decision_agent.exceptions import EvaluationValidationError


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def corpus_row(document_id: str = "doc-1", **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "document_id": document_id,
        "text": "产品电池保修期为一年。",
        "category": "产品与售后",
        "source": "synthetic-enterprise-baseline",
    }
    row.update(overrides)
    return row


def query_row(query_id: str = "query-1", **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "query_id": query_id,
        "query": "产品电池能保修多久？",  # noqa: RUF001
        "relevant_document_ids": ["doc-1"],
        "category": "产品与售后",
    }
    row.update(overrides)
    return row


def test_load_corpus_returns_typed_ordered_documents(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl"
    write_jsonl(path, [corpus_row("doc-1"), corpus_row("doc-2", text="第二条文档")])

    documents = load_corpus(path)

    assert [document.document_id for document in documents] == ["doc-1", "doc-2"]
    assert documents[0].text == "产品电池保修期为一年。"


def test_load_queries_returns_typed_ordered_queries(tmp_path: Path) -> None:
    path = tmp_path / "queries.jsonl"
    write_jsonl(path, [query_row("query-1"), query_row("query-2", query="第二个问题")])

    queries = load_queries(path, corpus_document_ids={"doc-1"})

    assert [query.query_id for query in queries] == ["query-1", "query-2"]


@pytest.mark.parametrize("loader_name", ["corpus", "queries"])
def test_empty_jsonl_file_is_rejected(tmp_path: Path, loader_name: str) -> None:
    path = tmp_path / f"{loader_name}.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(EvaluationValidationError, match="cannot be empty"):
        if loader_name == "corpus":
            load_corpus(path)
        else:
            load_queries(path, corpus_document_ids={"doc-1"})


def test_duplicate_document_id_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl"
    write_jsonl(path, [corpus_row(), corpus_row()])

    with pytest.raises(EvaluationValidationError, match="duplicate document_id"):
        load_corpus(path)


def test_whitespace_only_jsonl_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl"
    path.write_text(" \n\t\n", encoding="utf-8")

    with pytest.raises(EvaluationValidationError, match="line 1 cannot be blank"):
        load_corpus(path)


def test_duplicate_query_id_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "queries.jsonl"
    write_jsonl(path, [query_row(), query_row()])

    with pytest.raises(EvaluationValidationError, match="duplicate query_id"):
        load_queries(path, corpus_document_ids={"doc-1"})


@pytest.mark.parametrize("text", ["", " \n\t "])
def test_empty_corpus_text_is_rejected(tmp_path: Path, text: str) -> None:
    path = tmp_path / "corpus.jsonl"
    write_jsonl(path, [corpus_row(text=text)])

    with pytest.raises(EvaluationValidationError, match="corpus row 1"):
        load_corpus(path)


def test_empty_query_text_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "queries.jsonl"
    write_jsonl(path, [query_row(query="  ")])

    with pytest.raises(EvaluationValidationError, match="query row 1"):
        load_queries(path, corpus_document_ids={"doc-1"})


def test_empty_relevant_document_ids_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "queries.jsonl"
    write_jsonl(path, [query_row(relevant_document_ids=[])])

    with pytest.raises(EvaluationValidationError, match="query row 1"):
        load_queries(path, corpus_document_ids={"doc-1"})


def test_missing_relevant_document_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "queries.jsonl"
    write_jsonl(path, [query_row(relevant_document_ids=["missing-doc"])])

    with pytest.raises(EvaluationValidationError, match="unknown document_id"):
        load_queries(path, corpus_document_ids={"doc-1"})


def test_duplicate_relevant_document_id_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "queries.jsonl"
    write_jsonl(path, [query_row(relevant_document_ids=["doc-1", "doc-1"])])

    with pytest.raises(EvaluationValidationError, match="relevant_document_ids"):
        load_queries(path, corpus_document_ids={"doc-1"})


def test_malformed_jsonl_reports_line_number(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        json.dumps(corpus_row(), ensure_ascii=False) + "\n{bad json}\n", encoding="utf-8"
    )

    with pytest.raises(EvaluationValidationError, match="line 2"):
        load_corpus(path)


def test_wrong_jsonl_field_type_reports_field_without_input_value(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl"
    write_jsonl(path, [corpus_row(text=["not", "a", "string"])])

    with pytest.raises(EvaluationValidationError, match=r"corpus row 1: text:") as error:
        load_corpus(path)

    assert "not" not in str(error.value)


@pytest.mark.parametrize("field", ["category", "source"])
def test_blank_required_corpus_metadata_is_rejected(tmp_path: Path, field: str) -> None:
    path = tmp_path / "corpus.jsonl"
    write_jsonl(path, [corpus_row(**{field: " "})])

    with pytest.raises(EvaluationValidationError, match="corpus row 1"):
        load_corpus(path)


def test_retrieval_dataset_links_queries_to_loaded_corpus(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    queries_path = tmp_path / "queries.jsonl"
    write_jsonl(corpus_path, [corpus_row("doc-1"), corpus_row("doc-2")])
    write_jsonl(queries_path, [query_row(relevant_document_ids=["doc-2"])])

    dataset = load_retrieval_dataset(corpus_path, queries_path)

    assert len(dataset.corpus) == 2
    assert dataset.queries[0].relevant_document_ids == ("doc-2",)


def test_versioned_m2b2c_dataset_has_expected_integrity() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    dataset = load_retrieval_dataset(
        repository_root / "datasets/retrieval/m2b2c_dense_corpus.jsonl",
        repository_root / "datasets/retrieval/m2b2c_dense_queries.jsonl",
    )

    assert len(dataset.corpus) == 36
    assert len(dataset.queries) == 18
    assert Counter(document.category for document in dataset.corpus) == Counter(
        {category: 6 for category in {query.category for query in dataset.queries}}
    )
    assert Counter(query.category for query in dataset.queries) == Counter(
        {category: 3 for category in {query.category for query in dataset.queries}}
    )
    assert sum(len(query.relevant_document_ids) > 1 for query in dataset.queries) == 2
    assert {document.source for document in dataset.corpus} == {"synthetic-enterprise-baseline"}
    assert len({document.text for document in dataset.corpus}) == len(dataset.corpus)
    assert not (
        {document.text for document in dataset.corpus} & {query.query for query in dataset.queries}
    )


def test_normalized_text_hash_is_identical_for_lf_crlf_and_cr(tmp_path: Path) -> None:
    logical_text = '{"id":"one"}\n\n{"id":"two"}\n'
    paths = []
    for name, newline in (("lf", "\n"), ("crlf", "\r\n"), ("cr", "\r")):
        path = tmp_path / f"{name}.jsonl"
        path.write_bytes(logical_text.replace("\n", newline).encode("utf-8"))
        paths.append(path)

    hashes = {dataset_module.compute_normalized_text_sha256(path) for path in paths}

    assert len(hashes) == 1


def test_normalized_text_hash_changes_when_text_changes(tmp_path: Path) -> None:
    original = tmp_path / "original.txt"
    changed = tmp_path / "changed.txt"
    original.write_text(" value \n", encoding="utf-8", newline="")
    changed.write_text("value\n", encoding="utf-8", newline="")

    assert dataset_module.compute_normalized_text_sha256(
        original
    ) != dataset_module.compute_normalized_text_sha256(changed)


def test_normalized_text_hash_preserves_json_text_representation(tmp_path: Path) -> None:
    original = tmp_path / "original.jsonl"
    changed = tmp_path / "changed.jsonl"
    reordered = tmp_path / "reordered.jsonl"
    spaced = tmp_path / "spaced.jsonl"
    original.write_text('{"id":"one","value":1}\n', encoding="utf-8", newline="")
    changed.write_text('{"id":"one","value":2}\n', encoding="utf-8", newline="")
    reordered.write_text('{"value":1,"id":"one"}\n', encoding="utf-8", newline="")
    spaced.write_text('{"id": "one", "value": 1}\n', encoding="utf-8", newline="")

    hashes = {
        dataset_module.compute_normalized_text_sha256(path)
        for path in (original, changed, reordered, spaced)
    }

    assert len(hashes) == 4


def test_normalized_text_hash_preserves_trailing_newline_presence(tmp_path: Path) -> None:
    with_newline = tmp_path / "with-newline.jsonl"
    without_newline = tmp_path / "without-newline.jsonl"
    with_newline.write_text('{"id":"one"}\n', encoding="utf-8", newline="")
    without_newline.write_text('{"id":"one"}', encoding="utf-8", newline="")

    assert dataset_module.compute_normalized_text_sha256(
        with_newline
    ) != dataset_module.compute_normalized_text_sha256(without_newline)


def test_normalized_text_hash_preserves_blank_lines(tmp_path: Path) -> None:
    with_blank_line = tmp_path / "with-blank.jsonl"
    without_blank_line = tmp_path / "without-blank.jsonl"
    with_blank_line.write_text("first\n\nsecond\n", encoding="utf-8", newline="")
    without_blank_line.write_text("first\nsecond\n", encoding="utf-8", newline="")

    assert dataset_module.compute_normalized_text_sha256(
        with_blank_line
    ) != dataset_module.compute_normalized_text_sha256(without_blank_line)


def test_normalized_text_hash_wraps_invalid_utf8(tmp_path: Path) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_bytes(b'{"text":"\xff"}\n')

    with pytest.raises(EvaluationValidationError, match="UTF-8 text"):
        dataset_module.compute_normalized_text_sha256(path)
