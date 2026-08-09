"""Strict JSONL models and loaders for retrieval evaluation datasets."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from decision_agent.exceptions import EvaluationValidationError


class EvaluationModel(BaseModel):
    """Strict JSON-safe base for versioned evaluation records."""

    model_config = ConfigDict(extra="forbid")


class CorpusDocument(EvaluationModel):
    """One synthetic enterprise document used by the dense baseline."""

    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    category: str = Field(min_length=1)
    source: str = Field(min_length=1)

    @field_validator("document_id", "text", "category", "source")
    @classmethod
    def reject_blank_corpus_values(cls, value: str) -> str:
        """Reject values that satisfy length constraints using whitespace only."""
        if not value.strip():
            raise ValueError("value cannot be blank")
        return value


class RetrievalQuery(EvaluationModel):
    """One query and its binary document-level relevance labels."""

    query_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    relevant_document_ids: tuple[str, ...] = Field(min_length=1)
    category: str = Field(min_length=1)

    @field_validator("query_id", "query", "category")
    @classmethod
    def reject_blank_query_values(cls, value: str) -> str:
        """Reject blank identifiers, query text, and categories."""
        if not value.strip():
            raise ValueError("value cannot be blank")
        return value

    @field_validator("relevant_document_ids")
    @classmethod
    def validate_relevant_document_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Require unique, nonblank relevance labels."""
        if any(not value.strip() for value in values):
            raise ValueError("relevant_document_ids cannot contain blank values")
        if len(values) != len(set(values)):
            raise ValueError("relevant_document_ids must be unique")
        return values


class RetrievalDataset(EvaluationModel):
    """One validated corpus/query pair preserving file ordering."""

    corpus: tuple[CorpusDocument, ...] = Field(min_length=1)
    queries: tuple[RetrievalQuery, ...] = Field(min_length=1)


def compute_normalized_text_sha256(
    path: str | Path, *, dataset_name: str = "evaluation text"
) -> str:
    """Hash strict UTF-8 text after universal-newline normalization to LF."""
    file_path = Path(path)
    if not file_path.exists():
        raise EvaluationValidationError(f"{dataset_name} file does not exist")
    if not file_path.is_file():
        raise EvaluationValidationError(f"{dataset_name} path must be a file")
    try:
        with file_path.open("r", encoding="utf-8", newline=None) as file_handle:
            normalized_text = file_handle.read()
    except (OSError, UnicodeError) as exc:
        raise EvaluationValidationError(
            f"failed to read {dataset_name} as UTF-8 text for hashing"
        ) from exc
    return sha256(normalized_text.encode("utf-8")).hexdigest()


def read_jsonl_rows(path: str | Path, *, dataset_name: str) -> list[tuple[int, Any]]:
    """Read a nonempty UTF-8 JSONL file while preserving row numbers and order."""
    path = Path(path)
    if not path.exists():
        raise EvaluationValidationError(f"{dataset_name} file does not exist")
    if not path.is_file():
        raise EvaluationValidationError(f"{dataset_name} path must be a file")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise EvaluationValidationError(f"failed to read {dataset_name} JSONL") from exc
    if not lines:
        raise EvaluationValidationError(f"{dataset_name} JSONL cannot be empty")

    rows: list[tuple[int, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise EvaluationValidationError(
                f"{dataset_name} JSONL line {line_number} cannot be blank"
            )
        try:
            rows.append((line_number, json.loads(line)))
        except json.JSONDecodeError as exc:
            raise EvaluationValidationError(
                f"malformed {dataset_name} JSONL at line {line_number}"
            ) from exc
    return rows


def _validation_detail(exc: ValidationError) -> str:
    """Return a stable field/message summary without leaking input contents."""
    error = exc.errors(include_url=False, include_context=False, include_input=False)[0]
    location = ".".join(str(part) for part in error["loc"])
    return f"{location}: {error['msg']}" if location else str(error["msg"])


def load_corpus(path: str | Path) -> tuple[CorpusDocument, ...]:
    """Load a nonempty ordered corpus and reject duplicate document IDs."""
    documents: list[CorpusDocument] = []
    document_ids: set[str] = set()
    for line_number, row in read_jsonl_rows(path, dataset_name="corpus"):
        try:
            document = CorpusDocument.model_validate(row)
        except ValidationError as exc:
            raise EvaluationValidationError(
                f"invalid corpus row {line_number}: {_validation_detail(exc)}"
            ) from exc
        if document.document_id in document_ids:
            raise EvaluationValidationError(
                f"duplicate document_id in corpus: {document.document_id}"
            )
        document_ids.add(document.document_id)
        documents.append(document)
    return tuple(documents)


def load_queries(
    path: str | Path, *, corpus_document_ids: set[str] | frozenset[str]
) -> tuple[RetrievalQuery, ...]:
    """Load ordered queries and verify every relevance label against the corpus."""
    queries: list[RetrievalQuery] = []
    query_ids: set[str] = set()
    for line_number, row in read_jsonl_rows(path, dataset_name="queries"):
        try:
            query = RetrievalQuery.model_validate(row)
        except ValidationError as exc:
            raise EvaluationValidationError(
                f"invalid query row {line_number}: {_validation_detail(exc)}"
            ) from exc
        if query.query_id in query_ids:
            raise EvaluationValidationError(f"duplicate query_id in queries: {query.query_id}")
        missing = sorted(set(query.relevant_document_ids) - corpus_document_ids)
        if missing:
            raise EvaluationValidationError(
                f"query {query.query_id} references unknown document_id: {missing[0]}"
            )
        query_ids.add(query.query_id)
        queries.append(query)
    return tuple(queries)


def load_retrieval_dataset(corpus_path: str | Path, queries_path: str | Path) -> RetrievalDataset:
    """Load and cross-validate one versioned retrieval dataset pair."""
    corpus = load_corpus(corpus_path)
    queries = load_queries(
        queries_path,
        corpus_document_ids={document.document_id for document in corpus},
    )
    return RetrievalDataset(corpus=corpus, queries=queries)
