"""Offline integrity tests for the fictional M2C-1A enterprise KB package."""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from pathlib import Path

import pytest

from decision_agent.evaluation import (
    EnterpriseKBDataset,
    EnterpriseKBStatistics,
    load_and_validate_enterprise_kb,
)
from decision_agent.exceptions import EvaluationValidationError
from decision_agent.ingestion import ParserRegistry

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = REPOSITORY_ROOT / "datasets/enterprise_kb/m2c1"


@pytest.fixture(scope="module")
def validated_package() -> tuple[EnterpriseKBDataset, EnterpriseKBStatistics]:
    return load_and_validate_enterprise_kb(DATASET_ROOT)


def copied_package(tmp_path: Path) -> Path:
    destination = tmp_path / "m2c1"
    shutil.copytree(DATASET_ROOT, destination)
    return destination


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_query_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_query_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_manifest_contains_all_registered_unique_documents(validated_package) -> None:
    dataset, statistics = validated_package
    records = dataset.manifest.documents

    assert statistics.document_count == 12
    assert len({record.document_id for record in records}) == 12
    assert len({record.filename for record in records}) == 12
    assert all((DATASET_ROOT / "documents" / record.filename).is_file() for record in records)
    assert {record.document_id for record in records} >= {"DOC-ORG-001", "DOC-AGENT-001"}


def test_document_headings_match_manifest(validated_package) -> None:
    dataset, _ = validated_package

    for record in dataset.manifest.documents:
        first_line = (
            (DATASET_ROOT / "documents" / record.filename)
            .read_text(encoding="utf-8")
            .splitlines()[0]
        )
        assert first_line == f"# {record.document_id} {record.title}"


def test_every_document_is_within_character_range(validated_package) -> None:
    _, statistics = validated_package

    assert all(1800 <= count <= 3200 for count in statistics.document_character_counts.values())


def test_entities_and_aliases_are_globally_unambiguous(validated_package) -> None:
    dataset, statistics = validated_package
    entities = [entity for collection in dataset.entities.collections() for entity in collection]
    identities = [
        value.casefold() for entity in entities for value in (entity.name, *entity.aliases)
    ]

    assert statistics.entity_count == 49
    assert len({entity.id for entity in entities}) == len(entities)
    assert len(identities) == len(set(identities))


def test_fact_ids_and_references_are_validated(validated_package) -> None:
    dataset, statistics = validated_package

    assert statistics.fact_count == 66
    assert len({fact.fact_id for fact in dataset.facts.facts}) == statistics.fact_count


def test_locked_boundary_facts_keep_exact_values_and_units(validated_package) -> None:
    dataset, _ = validated_package
    facts = {fact.fact_id: (fact.value, fact.unit) for fact in dataset.facts.facts}

    assert facts["FACT-PROC-BELOW-50K"] == ("部门经理", "role")
    assert facts["FACT-PROC-50K-200K"] == ("采购总监", "role")
    assert facts["FACT-PROC-200K-PLUS"] == ("财务负责人和总经理联合审批", "role")
    assert facts["FACT-FIN-BELOW-5K"] == ("部门负责人", "role")
    assert facts["FACT-FIN-5K-20K"] == ("财务经理", "role")
    assert facts["FACT-FIN-20K-PLUS"] == ("财务负责人", "role")
    assert facts["FACT-FIN-OVER-UPTO10"] == (
        "部门负责人和财务经理联合审批",
        "role",
    )
    assert facts["FACT-FIN-OVER-10"] == ("财务负责人和总经理联合审批", "role")
    assert facts["FACT-SUPPLIER-PREFERRED"] == (85, "point_minimum")
    assert facts["FACT-SUPPLIER-QUALIFIED"] == ("70-84", "point_range")
    assert facts["FACT-SUPPLIER-RECTIFY"] == ("60-69", "point_range")
    assert facts["FACT-SUPPLIER-SUSPEND"] == (60, "point_exclusive_maximum")


def test_clause_ids_are_global_and_in_expected_range(validated_package) -> None:
    _, statistics = validated_package

    assert 110 <= statistics.clause_count <= 150
    assert statistics.clause_count == 149


def test_query_count_and_ids_are_exact(validated_package) -> None:
    dataset, statistics = validated_package

    assert statistics.query_count == 60
    assert [query.query_id for query in dataset.queries] == [
        f"M2C1-Q{index:03d}" for index in range(1, 61)
    ]


def test_query_category_distribution_is_exact(validated_package) -> None:
    _, statistics = validated_package

    assert statistics.category_counts == {
        "customer_service": 10,
        "finance": 6,
        "hr": 5,
        "inventory": 6,
        "procurement": 8,
        "sales": 8,
        "security_governance": 7,
        "enterprise_profile": 10,
    }


def test_query_type_distribution_is_exact(validated_package) -> None:
    _, statistics = validated_package

    assert statistics.query_type_counts == {
        "cross_section": 9,
        "multi_constraint": 15,
        "multi_evidence": 12,
        "single_fact": 20,
        "unanswerable": 4,
    }


def test_unanswerable_contract_is_exact(validated_package) -> None:
    dataset, statistics = validated_package
    unanswerable = [query for query in dataset.queries if not query.answerable]

    assert statistics.unanswerable_query_count == 4
    assert all(query.reference_answer is None for query in unanswerable)
    assert all(not query.relevant_clause_ids for query in unanswerable)
    assert all(query.expected_evidence_count == 0 for query in unanswerable)


def test_hard_negatives_are_real_disjoint_clauses(validated_package) -> None:
    dataset, statistics = validated_package

    assert statistics.hard_negative_query_count == 53
    assert statistics.hard_negative_query_count >= 14
    assert all(
        set(query.relevant_clause_ids).isdisjoint(query.hard_negative_clause_ids)
        for query in dataset.queries
    )


def test_eight_core_hard_negative_scenarios_are_exact(validated_package) -> None:
    dataset, _ = validated_package
    queries = {query.query_id: query for query in dataset.queries}
    expected = {
        "M2C1-Q001": {
            "CS-WARRANTY-A-DEVICE",
            "CS-WARRANTY-B-BATTERY",
            "CS-WARRANTY-B-DEVICE",
        },
        "M2C1-Q011": {
            "SALES-DISCOUNT-EAST-REGULAR",
            "SALES-DISCOUNT-KEY-PLUS2",
            "SALES-DISCOUNT-STRATEGIC-15",
        },
        "M2C1-Q025": {
            "PROC-NORMAL-BELOW-50K",
            "PROC-NORMAL-200K-PLUS",
            "PROC-EMERGENCY-UPTO-80K",
        },
        "M2C1-Q005": {
            "CS-DOA-A-15D",
            "CS-RETURN-UNOPENED-7D",
            "CS-DOA-AFTER-WINDOW",
        },
        "M2C1-Q019": {
            "INV-B-SAFETY-80",
            "INV-A-EMERGENCY-80",
            "INV-B-REORDER-120",
        },
        "M2C1-Q045": {
            "SEC-ACCESS-L4",
            "SEC-DATA-L2",
            "SEC-LOG-GENERAL-180D",
            "SEC-LOG-L3L4-365D",
        },
        "M2C1-Q034": {"FIN-OVERBUDGET-OVER10", "FIN-EXPENSE-5K-20K"},
        "M2C1-Q026": {
            "PROC-SCORE-QUALIFIED-70-84",
            "PROC-SCORE-PREFERRED-85",
            "PROC-SCORE-SUSPEND-BELOW60",
        },
    }

    for query_id, hard_negatives in expected.items():
        assert set(queries[query_id].hard_negative_clause_ids) == hard_negatives


def test_every_domain_has_all_difficulty_levels(validated_package) -> None:
    dataset, _ = validated_package
    difficulties: dict[str, set[str]] = {}
    for query in dataset.queries:
        difficulties.setdefault(query.category, set()).add(query.difficulty)

    assert all(values == {"easy", "medium", "hard"} for values in difficulties.values())


def test_semantic_evidence_count_is_bounded_by_relevant_clauses(validated_package) -> None:
    dataset, _ = validated_package

    assert all(
        query.expected_evidence_count <= len(query.relevant_clause_ids) for query in dataset.queries
    )
    assert all(query.expected_evidence_count >= 1 for query in dataset.queries if query.answerable)
    assert sum(query.expected_evidence_count > 1 for query in dataset.queries) >= 8


def test_real_markdown_parser_and_chunker_validate_all_documents(validated_package) -> None:
    _, statistics = validated_package

    assert set(statistics.document_block_counts.values()) == {1}
    assert all(count >= 2 for count in statistics.parent_chunk_counts.values())
    assert all(count >= 2 for count in statistics.child_chunk_counts.values())


def test_package_contains_no_apparent_phone_or_identity_number() -> None:
    business_texts = [
        path.read_text(encoding="utf-8")
        for path in sorted((DATASET_ROOT / "documents").glob("*.md"))
    ]
    entities = read_json(DATASET_ROOT / "entity_dictionary.json")
    for collection in entities.values():
        records = collection if isinstance(collection, list) else [collection]
        for record in records:
            if isinstance(record, dict):
                business_texts.extend(
                    value
                    for key, value in record.items()
                    if key == "name" and isinstance(value, str)
                )
                business_texts.extend(record.get("aliases", []))

    facts = read_json(DATASET_ROOT / "business_fact_registry.json")
    for fact in facts["facts"]:
        business_texts.append(fact["attribute"])
        if isinstance(fact["value"], str):
            business_texts.append(fact["value"])
        business_texts.extend(fact["applicable_conditions"])

    manifest = read_json(DATASET_ROOT / "document_manifest.json")
    business_texts.extend(document["title"] for document in manifest["documents"])
    query_rows = read_query_rows(DATASET_ROOT / "query_blueprint.jsonl")
    business_texts.extend(row["query"] for row in query_rows)
    business_texts.extend(
        row["reference_answer"] for row in query_rows if isinstance(row["reference_answer"], str)
    )

    fixture_root = REPOSITORY_ROOT / "tests/fixtures/ingestion/mixed_format"
    registry = ParserRegistry.default()
    for index, path in enumerate(sorted(fixture_root.iterdir())):
        business_texts.extend(
            block.content
            for block in registry.parse(
                str(path),
                document_id=f"fixture-{index}",
                document_version="1.0",
            )
        )

    generated_root = DATASET_ROOT / "generated"
    generated_text_fields = {
        "parent_chunks.jsonl": "content",
        "child_chunks.jsonl": "content",
        "clause_chunk_map.jsonl": "clause_content",
        "retrieval_ground_truth.jsonl": "query",
    }
    for filename, field in generated_text_fields.items():
        business_texts.extend(
            json.loads(line)[field]
            for line in (generated_root / filename).read_text(encoding="utf-8").splitlines()
        )

    combined = "\n".join(business_texts)

    assert re.search(r"(?<!\d)1[3-9]\d{9}(?!\d)", combined) is None
    assert re.search(r"(?<!\d)\d{17}[0-9Xx](?!\d)", combined) is None
    assert re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", combined) is None


def test_repeated_loading_is_fully_deterministic() -> None:
    first_dataset, first_statistics = load_and_validate_enterprise_kb(DATASET_ROOT)
    second_dataset, second_statistics = load_and_validate_enterprise_kb(DATASET_ROOT)

    assert first_dataset == second_dataset
    assert first_statistics == second_statistics


def test_duplicate_clause_id_fails_fast(tmp_path: Path) -> None:
    root = copied_package(tmp_path)
    document = root / "documents/DOC-CS-002_return_repair_operations.md"
    document.write_text(
        document.read_text(encoding="utf-8") + "\n条款 ID\N{FULLWIDTH COLON}CS-RETURN-INTAKE\n",
        encoding="utf-8",
    )

    with pytest.raises(EvaluationValidationError, match="duplicate clause id"):
        load_and_validate_enterprise_kb(root)


def test_query_unknown_clause_fails_fast(tmp_path: Path) -> None:
    root = copied_package(tmp_path)
    path = root / "query_blueprint.jsonl"
    rows = read_query_rows(path)
    rows[0]["relevant_clause_ids"] = ["MISSING-CLAUSE"]
    write_query_rows(path, rows)

    with pytest.raises(EvaluationValidationError, match="unknown clause_id"):
        load_and_validate_enterprise_kb(root)


def test_wrong_query_distribution_fails_fast(tmp_path: Path) -> None:
    root = copied_package(tmp_path)
    path = root / "query_blueprint.jsonl"
    rows = read_query_rows(path)
    rows[0]["category"] = "sales"
    write_query_rows(path, rows)

    with pytest.raises(EvaluationValidationError, match="category distribution"):
        load_and_validate_enterprise_kb(root)


def test_inconsistent_answerable_fields_fail_fast(tmp_path: Path) -> None:
    root = copied_package(tmp_path)
    path = root / "query_blueprint.jsonl"
    rows = read_query_rows(path)
    rows[0]["reference_answer"] = None
    write_query_rows(path, rows)

    with pytest.raises(EvaluationValidationError, match="requires reference_answer"):
        load_and_validate_enterprise_kb(root)


def test_fact_unknown_clause_fails_fast(tmp_path: Path) -> None:
    root = copied_package(tmp_path)
    path = root / "business_fact_registry.json"
    payload = read_json(path)
    facts = payload["facts"]
    assert isinstance(facts, list)
    facts[0]["expected_clause_ids"] = ["MISSING-FACT-CLAUSE"]
    write_json(path, payload)

    with pytest.raises(EvaluationValidationError, match="unknown clause_id"):
        load_and_validate_enterprise_kb(root)


def test_alias_conflict_fails_fast(tmp_path: Path) -> None:
    root = copied_package(tmp_path)
    path = root / "entity_dictionary.json"
    payload = read_json(path)
    regions = payload["regions"]
    assert isinstance(regions, list)
    regions[1]["aliases"] = ["华东区域"]
    write_json(path, payload)

    with pytest.raises(EvaluationValidationError, match="identity conflict"):
        load_and_validate_enterprise_kb(root)


def test_missing_manifest_file_fails_fast(tmp_path: Path) -> None:
    root = copied_package(tmp_path)
    (root / "documents/DOC-CS-001_product_warranty_policy.md").unlink()

    with pytest.raises(EvaluationValidationError, match="file does not exist"):
        load_and_validate_enterprise_kb(root)


def test_query_categories_and_types_come_from_dictionary(validated_package) -> None:
    dataset, _ = validated_package
    category_ids = {entity.id for entity in dataset.entities.query_categories}
    query_type_ids = {entity.id for entity in dataset.entities.query_types}

    assert Counter(query.category for query in dataset.queries).keys() <= category_ids
    assert Counter(query.query_type for query in dataset.queries).keys() <= query_type_ids
