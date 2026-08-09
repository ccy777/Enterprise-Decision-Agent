"""Typed loading and offline integrity validation for the M2C-1 enterprise KB."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from decision_agent.evaluation.dataset import (
    compute_normalized_text_sha256,
    read_jsonl_rows,
)
from decision_agent.exceptions import EvaluationValidationError
from decision_agent.ingestion import ParentChildChunker, ParserRegistry

EXPECTED_DOCUMENT_COUNT = 12
EXPECTED_QUERY_COUNT = 60
EXPECTED_ANSWERABLE_QUERY_COUNT = 56
EXPECTED_UNANSWERABLE_QUERY_COUNT = 4
EXPECTED_CATEGORY_COUNTS = {
    "customer_service": 10,
    "sales": 8,
    "inventory": 6,
    "procurement": 8,
    "finance": 6,
    "hr": 5,
    "security_governance": 7,
    "enterprise_profile": 10,
}
EXPECTED_QUERY_TYPE_COUNTS = {
    "single_fact": 20,
    "multi_constraint": 15,
    "cross_section": 9,
    "multi_evidence": 12,
    "unanswerable": 4,
}
CLAUSE_PATTERN = re.compile(
    r"^条款 ID\N{FULLWIDTH COLON}([A-Z0-9]+(?:-[A-Z0-9]+)+)\s*$", re.MULTILINE
)
PHONE_PATTERN = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
IDENTITY_PATTERN = re.compile(r"(?<!\d)\d{17}[0-9Xx](?!\d)")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DictionaryEntity(_StrictModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    aliases: tuple[str, ...] = ()

    @field_validator("id", "name")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value cannot be blank")
        return value

    @field_validator("aliases")
    @classmethod
    def validate_aliases(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = [value.strip().casefold() for value in values]
        if any(not value for value in normalized):
            raise ValueError("aliases cannot contain blank values")
        if len(normalized) != len(set(normalized)):
            raise ValueError("aliases must be unique")
        return values


class EntityDictionary(_StrictModel):
    schema_version: Literal["1.0"]
    company: DictionaryEntity
    agents: tuple[DictionaryEntity, ...] = Field(min_length=1)
    departments: tuple[DictionaryEntity, ...] = Field(min_length=1)
    products: tuple[DictionaryEntity, ...] = Field(min_length=1)
    accessories: tuple[DictionaryEntity, ...] = Field(min_length=1)
    regions: tuple[DictionaryEntity, ...] = Field(min_length=1)
    customer_levels: tuple[DictionaryEntity, ...] = Field(min_length=1)
    suppliers: tuple[DictionaryEntity, ...] = Field(min_length=1)
    document_categories: tuple[DictionaryEntity, ...] = Field(min_length=1)
    query_categories: tuple[DictionaryEntity, ...] = Field(min_length=1)
    query_types: tuple[DictionaryEntity, ...] = Field(min_length=1)

    def collections(self) -> tuple[tuple[DictionaryEntity, ...], ...]:
        return (
            (self.company,),
            self.agents,
            self.departments,
            self.products,
            self.accessories,
            self.regions,
            self.customer_levels,
            self.suppliers,
            self.document_categories,
            self.query_categories,
            self.query_types,
        )


class BusinessFact(_StrictModel):
    fact_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    subject_id: str = Field(min_length=1)
    attribute: str = Field(min_length=1)
    value: str | int | float | bool
    unit: str = Field(min_length=1)
    applicable_conditions: tuple[str, ...] = Field(min_length=1)
    expected_document_ids: tuple[str, ...] = Field(min_length=1)
    expected_clause_ids: tuple[str, ...] = Field(min_length=1)


class BusinessFactRegistry(_StrictModel):
    schema_version: Literal["1.0"]
    facts: tuple[BusinessFact, ...] = Field(min_length=1)


class DocumentManifestRecord(_StrictModel):
    document_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    title: str = Field(min_length=1)
    department: str = Field(min_length=1)
    category: str = Field(min_length=1)
    version: Literal["1.0"]
    effective_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    status: Literal["active"]
    language: Literal["zh-CN"]
    confidentiality_level: Literal["L2", "L3"]


class DocumentManifest(_StrictModel):
    schema_version: Literal["1.0"]
    documents: tuple[DocumentManifestRecord, ...] = Field(min_length=1)


class QueryBlueprint(_StrictModel):
    query_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    category: str = Field(min_length=1)
    query_type: str = Field(min_length=1)
    answerable: bool
    difficulty: Literal["easy", "medium", "hard"]
    reference_answer: str | None
    relevant_clause_ids: tuple[str, ...]
    hard_negative_clause_ids: tuple[str, ...]
    expected_evidence_count: int = Field(ge=0)
    constraint_tags: tuple[str, ...] = Field(min_length=1)

    @field_validator("relevant_clause_ids", "hard_negative_clause_ids", "constraint_tags")
    @classmethod
    def require_unique_values(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() for value in values):
            raise ValueError("list values cannot be blank")
        if len(values) != len(set(values)):
            raise ValueError("list values must be unique")
        return values


class EnterpriseKBDataset(_StrictModel):
    entities: EntityDictionary
    facts: BusinessFactRegistry
    manifest: DocumentManifest
    queries: tuple[QueryBlueprint, ...]


class EnterpriseKBStatistics(_StrictModel):
    document_count: int
    entity_count: int
    fact_count: int
    clause_count: int
    query_count: int
    hard_negative_query_count: int
    unanswerable_query_count: int
    category_counts: dict[str, int]
    query_type_counts: dict[str, int]
    document_character_counts: dict[str, int]
    document_block_counts: dict[str, int]
    parent_chunk_counts: dict[str, int]
    child_chunk_counts: dict[str, int]
    normalized_sha256: dict[str, str]


def _validation_detail(exc: ValidationError) -> str:
    error = exc.errors(include_url=False, include_context=False, include_input=False)[0]
    location = ".".join(str(part) for part in error["loc"])
    return f"{location}: {error['msg']}" if location else str(error["msg"])


def _load_json(path: Path, model: type[_StrictModel], *, name: str) -> Any:
    if not path.is_file():
        raise EvaluationValidationError(f"{name} file does not exist")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationValidationError(f"failed to read {name} JSON") from exc
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        raise EvaluationValidationError(f"invalid {name}: {_validation_detail(exc)}") from exc


def _load_queries(path: Path) -> tuple[QueryBlueprint, ...]:
    queries: list[QueryBlueprint] = []
    for line_number, row in read_jsonl_rows(path, dataset_name="enterprise query blueprint"):
        try:
            queries.append(QueryBlueprint.model_validate(row))
        except ValidationError as exc:
            raise EvaluationValidationError(
                f"invalid query blueprint row {line_number}: {_validation_detail(exc)}"
            ) from exc
    return tuple(queries)


def _require_unique(values: list[str], *, label: str) -> None:
    duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
    if duplicates:
        raise EvaluationValidationError(f"duplicate {label}: {duplicates[0]}")


def _validate_entities(entities: EntityDictionary) -> set[str]:
    all_entities = [entity for collection in entities.collections() for entity in collection]
    _require_unique([entity.id for entity in all_entities], label="entity id")

    identities: dict[str, str] = {}
    for entity in all_entities:
        for identity in (entity.name, *entity.aliases):
            normalized = identity.strip().casefold()
            owner = identities.setdefault(normalized, entity.id)
            if owner != entity.id:
                raise EvaluationValidationError(
                    f"entity identity conflict between {owner} and {entity.id}"
                )
    return {entity.id for entity in all_entities}


def _validate_documents(
    root: Path, manifest: DocumentManifest, entities: EntityDictionary
) -> tuple[dict[str, set[str]], dict[str, int], dict[str, int], dict[str, int], dict[str, int]]:
    if len(manifest.documents) != EXPECTED_DOCUMENT_COUNT:
        raise EvaluationValidationError(
            f"document manifest must contain exactly {EXPECTED_DOCUMENT_COUNT} documents"
        )
    _require_unique([document.document_id for document in manifest.documents], label="document_id")
    _require_unique(
        [document.filename for document in manifest.documents], label="document filename"
    )

    department_ids = {entity.id for entity in entities.departments}
    category_ids = {entity.id for entity in entities.document_categories}
    parser = ParserRegistry.default()
    chunker = ParentChildChunker(parent_chunk_size=800, child_chunk_size=300, chunk_overlap=50)
    clauses_by_document: dict[str, set[str]] = {}
    character_counts: dict[str, int] = {}
    block_counts: dict[str, int] = {}
    parent_counts: dict[str, int] = {}
    child_counts: dict[str, int] = {}

    for document in manifest.documents:
        if document.department not in department_ids:
            raise EvaluationValidationError(
                f"manifest document {document.document_id} uses unknown department"
            )
        if document.category not in category_ids:
            raise EvaluationValidationError(
                f"manifest document {document.document_id} uses unknown category"
            )
        if Path(document.filename).name != document.filename:
            raise EvaluationValidationError("manifest filenames must be portable basenames")

        path = root / "documents" / document.filename
        if not path.is_file():
            raise EvaluationValidationError(
                f"manifest document file does not exist: {document.filename}"
            )
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise EvaluationValidationError(
                f"failed to read manifest document: {document.filename}"
            ) from exc
        expected_heading = f"# {document.document_id} {document.title}"
        if content.splitlines()[0] != expected_heading:
            raise EvaluationValidationError(
                f"document heading does not match manifest: {document.document_id}"
            )
        if PHONE_PATTERN.search(content) or IDENTITY_PATTERN.search(content):
            raise EvaluationValidationError(
                f"document contains apparent personal sensitive data: {document.document_id}"
            )

        character_count = len("".join(content.split()))
        if not 1800 <= character_count <= 3200:
            raise EvaluationValidationError(
                f"document character count is outside 1800-3200: {document.document_id}"
            )
        clauses = CLAUSE_PATTERN.findall(content)
        if len(clauses) < 10:
            raise EvaluationValidationError(
                f"document must contain at least 10 clause IDs: {document.document_id}"
            )
        _require_unique(clauses, label="clause id")

        blocks = parser.parse(
            str(path),
            document_id=document.document_id,
            document_version=document.version,
            metadata={"department": document.department, "category": document.category},
        )
        if not blocks or any(not block.content.strip() for block in blocks):
            raise EvaluationValidationError(
                f"parser returned empty document blocks: {document.document_id}"
            )
        chunk_results = [chunker.chunk(block) for block in blocks]
        parent_count = sum(len(result.parents) for result in chunk_results)
        child_count = sum(len(result.children) for result in chunk_results)
        if parent_count < 2 or child_count < 2:
            raise EvaluationValidationError(
                f"document does not produce multiple parent/child chunks: {document.document_id}"
            )

        clauses_by_document[document.document_id] = set(clauses)
        character_counts[document.document_id] = character_count
        block_counts[document.document_id] = len(blocks)
        parent_counts[document.document_id] = parent_count
        child_counts[document.document_id] = child_count

    all_clauses = [
        clause
        for document in manifest.documents
        for clause in clauses_by_document[document.document_id]
    ]
    _require_unique(all_clauses, label="clause id")
    return clauses_by_document, character_counts, block_counts, parent_counts, child_counts


def _validate_facts(
    registry: BusinessFactRegistry,
    entities: EntityDictionary,
    clauses_by_document: dict[str, set[str]],
) -> None:
    _require_unique([fact.fact_id for fact in registry.facts], label="fact_id")
    entity_ids = {entity.id for collection in entities.collections() for entity in collection}
    category_ids = {entity.id for entity in entities.query_categories}
    seen_values: dict[tuple[str, str, tuple[str, ...]], str] = {}
    for fact in registry.facts:
        if fact.category not in category_ids:
            raise EvaluationValidationError(f"fact {fact.fact_id} uses unknown category")
        if fact.subject_id not in entity_ids:
            raise EvaluationValidationError(f"fact {fact.fact_id} uses unknown subject_id")
        for document_id in fact.expected_document_ids:
            if document_id not in clauses_by_document:
                raise EvaluationValidationError(
                    f"fact {fact.fact_id} references unknown document_id: {document_id}"
                )
        available_clauses = set().union(
            *(clauses_by_document[document_id] for document_id in fact.expected_document_ids)
        )
        missing = sorted(set(fact.expected_clause_ids) - available_clauses)
        if missing:
            raise EvaluationValidationError(
                f"fact {fact.fact_id} references unknown clause_id: {missing[0]}"
            )
        key = (fact.subject_id, fact.attribute, tuple(sorted(fact.applicable_conditions)))
        serialized_value = json.dumps(fact.value, ensure_ascii=False, sort_keys=True)
        prior = seen_values.setdefault(key, serialized_value)
        if prior != serialized_value:
            raise EvaluationValidationError(
                f"conflicting fact value for {fact.subject_id}/{fact.attribute}"
            )


def _validate_queries(
    queries: tuple[QueryBlueprint, ...],
    entities: EntityDictionary,
    all_clause_ids: set[str],
) -> None:
    if len(queries) != EXPECTED_QUERY_COUNT:
        raise EvaluationValidationError(
            f"query blueprint must contain exactly {EXPECTED_QUERY_COUNT} queries"
        )
    _require_unique([query.query_id for query in queries], label="query_id")
    if Counter(query.category for query in queries) != Counter(EXPECTED_CATEGORY_COUNTS):
        raise EvaluationValidationError("query category distribution does not match M2C-1A")
    if Counter(query.query_type for query in queries) != Counter(EXPECTED_QUERY_TYPE_COUNTS):
        raise EvaluationValidationError("query type distribution does not match M2C-1A")

    category_ids = {entity.id for entity in entities.query_categories}
    query_type_ids = {entity.id for entity in entities.query_types}
    for query in queries:
        if query.category not in category_ids or query.query_type not in query_type_ids:
            raise EvaluationValidationError(
                f"query {query.query_id} uses an unknown category or query type"
            )
        referenced = set(query.relevant_clause_ids) | set(query.hard_negative_clause_ids)
        missing = sorted(referenced - all_clause_ids)
        if missing:
            raise EvaluationValidationError(
                f"query {query.query_id} references unknown clause_id: {missing[0]}"
            )
        if set(query.relevant_clause_ids) & set(query.hard_negative_clause_ids):
            raise EvaluationValidationError(
                f"query {query.query_id} overlaps relevant and hard-negative clauses"
            )
        if query.answerable:
            if not query.reference_answer or not query.reference_answer.strip():
                raise EvaluationValidationError(
                    f"answerable query {query.query_id} requires reference_answer"
                )
            if not query.relevant_clause_ids:
                raise EvaluationValidationError(
                    f"answerable query {query.query_id} requires relevant clauses"
                )
            if not 1 <= query.expected_evidence_count <= len(query.relevant_clause_ids):
                raise EvaluationValidationError(
                    f"query {query.query_id} expected_evidence_count is inconsistent"
                )
        elif (
            query.reference_answer is not None
            or query.relevant_clause_ids
            or query.expected_evidence_count != 0
        ):
            raise EvaluationValidationError(
                f"unanswerable query {query.query_id} has inconsistent answer fields"
            )
    if sum(bool(query.hard_negative_clause_ids) for query in queries) < 14:
        raise EvaluationValidationError("at least 14 queries must have hard negatives")
    if sum(not query.answerable for query in queries) != EXPECTED_UNANSWERABLE_QUERY_COUNT:
        raise EvaluationValidationError(
            f"query blueprint must contain exactly {EXPECTED_UNANSWERABLE_QUERY_COUNT} "
            "unanswerable queries"
        )


def load_and_validate_enterprise_kb(
    root: str | Path,
) -> tuple[EnterpriseKBDataset, EnterpriseKBStatistics]:
    """Load the complete M2C-1A package and return deterministic in-memory statistics."""
    root_path = Path(root)
    entities = _load_json(
        root_path / "entity_dictionary.json", EntityDictionary, name="entity dictionary"
    )
    facts = _load_json(
        root_path / "business_fact_registry.json",
        BusinessFactRegistry,
        name="business fact registry",
    )
    manifest = _load_json(
        root_path / "document_manifest.json", DocumentManifest, name="document manifest"
    )
    queries = _load_queries(root_path / "query_blueprint.jsonl")

    entity_ids = _validate_entities(entities)
    (
        clauses_by_document,
        character_counts,
        block_counts,
        parent_counts,
        child_counts,
    ) = _validate_documents(root_path, manifest, entities)
    _validate_facts(facts, entities, clauses_by_document)
    all_clause_ids = set().union(*clauses_by_document.values())
    _validate_queries(queries, entities, all_clause_ids)

    dataset = EnterpriseKBDataset(
        entities=entities,
        facts=facts,
        manifest=manifest,
        queries=queries,
    )
    files_to_hash = (
        root_path / "entity_dictionary.json",
        root_path / "business_fact_registry.json",
        root_path / "document_manifest.json",
        root_path / "query_blueprint.jsonl",
    )
    statistics = EnterpriseKBStatistics(
        document_count=len(manifest.documents),
        entity_count=len(entity_ids),
        fact_count=len(facts.facts),
        clause_count=len(all_clause_ids),
        query_count=len(queries),
        hard_negative_query_count=sum(bool(query.hard_negative_clause_ids) for query in queries),
        unanswerable_query_count=sum(not query.answerable for query in queries),
        category_counts=dict(sorted(Counter(query.category for query in queries).items())),
        query_type_counts=dict(sorted(Counter(query.query_type for query in queries).items())),
        document_character_counts=character_counts,
        document_block_counts=block_counts,
        parent_chunk_counts=parent_counts,
        child_chunk_counts=child_counts,
        normalized_sha256={
            path.name: compute_normalized_text_sha256(path, dataset_name=path.name)
            for path in files_to_hash
        },
    )
    return dataset, statistics
