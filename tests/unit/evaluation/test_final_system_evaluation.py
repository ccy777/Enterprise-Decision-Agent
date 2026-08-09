from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

from decision_agent.evaluation.final_system_evaluation import (
    EvaluationMode,
    FinalCaseRecord,
    FinalEvaluationDataset,
    calculate_final_metrics,
    derive_final_metrics_from_frozen_sources,
)

_ROOT = Path(__file__).resolve().parents[3]
_DATASET = _ROOT / "datasets" / "agent_tasks" / "m9_final_eval_v1.json"


def _record(**updates: object) -> FinalCaseRecord:
    now = datetime.now(UTC)
    values: dict[str, object] = {
        "run_id": "m9-formal-1",
        "dataset_version": "m9-final-eval-v1",
        "code_commit": "a" * 40,
        "case_id": "m9-test-001",
        "scenario": "knowledge",
        "execution_mode": EvaluationMode.REAL_RUNTIME,
        "expected_answerable": True,
        "expected_final_status": "completed",
        "started_at": now,
        "completed_at": now,
        "route": "knowledge",
        "final_status": "completed",
        "latency_ms": 10.0,
        "provider_call_count": 2,
        "input_tokens": 20,
        "output_tokens": 10,
        "usage_complete": True,
        "citation_count": 1,
        "citation_prefix_counts": {"E": 1},
        "evidence_types": ("document",),
        "tool_categories": ("knowledge",),
        "answer_present": True,
        "abstained": False,
        "security_decision": "allowed",
        "checks": {
            "route": True,
            "final_status": True,
            "tools": True,
            "evidence": True,
            "citations": True,
            "facts": True,
            "answerability": True,
            "security": True,
        },
        "passed": True,
    }
    values.update(updates)
    return FinalCaseRecord.model_validate(values)


def test_frozen_dataset_is_valid_unique_and_covers_required_scenarios() -> None:
    dataset = FinalEvaluationDataset.model_validate_json(_DATASET.read_text(encoding="utf-8"))
    assert dataset.dataset_version == "m9-final-eval-v1"
    scenarios = {case.scenario for case in dataset.cases}
    assert {
        "controlled_mixed_inventory",
        "security_scope_denied",
        "mcp_failure",
        "provider_failure",
        "reviewer_rejection",
        "provider_budget_denied",
    } <= scenarios
    assert "unsupported_out_of_scope" in scenarios
    assert sum(case.execution_mode is EvaluationMode.REAL_RUNTIME for case in dataset.cases) == 9
    assert all(
        case.required_answer_fact_groups
        for case in dataset.cases
        if case.scenario
        in {"knowledge_single_fact", "knowledge_multi_evidence", "contradictory_premise"}
    )


def test_case_record_schema_rejects_payload_fields() -> None:
    values = _record().model_dump(mode="json")
    values["answer"] = "sensitive"
    with pytest.raises(ValueError):
        FinalCaseRecord.model_validate(values)


def test_metrics_use_actual_usage_and_do_not_guess_cost() -> None:
    report = calculate_final_metrics([_record()])
    assert report.pass_rate == 1.0
    assert report.rate_details["overall"].model_dump() == {
        "passed": 1,
        "total": 1,
        "numerator": 1,
        "denominator": 1,
        "value": 1.0,
    }
    assert report.input_tokens == 20
    assert report.output_tokens == 10
    assert report.cost_amount is None
    assert report.cost_unavailable_reason == "provider_pricing_not_frozen"


def test_metrics_withhold_partial_token_totals() -> None:
    report = calculate_final_metrics(
        [_record(input_tokens=None, output_tokens=None, usage_complete=False)]
    )
    assert report.usage_coverage_rate == 0.0
    assert report.input_tokens is None
    assert report.output_tokens is None


def test_empty_rate_samples_are_null_and_keep_zero_denominators_visible() -> None:
    report = calculate_final_metrics([_record(route="knowledge")])

    assert report.reviewer_accept_rate is None
    assert report.rate_details["workflow_reviewer_accept"].model_dump() == {
        "passed": 0,
        "total": 0,
        "numerator": 0,
        "denominator": 0,
        "value": None,
    }
    assert report.rate_details["planner_schema_check"].value is None


def test_formal_boundary_and_unanswerable_groups_keep_counts_and_real_failure() -> None:
    records = [_record(case_id=f"m9-formal-{index}") for index in range(8)] + [
        _record(
            case_id="m9-formal-failure",
            expected_answerable=False,
            answer_present=True,
            abstained=False,
            passed=False,
            failure_category="unanswerable_misclassified",
            checks={
                "route": True,
                "final_status": False,
                "tools": True,
                "evidence": True,
                "citations": False,
                "facts": True,
                "answerability": False,
                "security": True,
            },
        ),
        *[
            _record(
                case_id=f"m9-boundary-{index}",
                execution_mode=EvaluationMode.FROZEN_BOUNDARY,
                evidence_reference=f"boundary-{index}",
                expected_answerable=None,
                route=None,
                answer_present=False,
                abstained=True,
                provider_call_count=0,
                input_tokens=0,
                output_tokens=0,
                checks={"frozen_evidence": True},
            )
            for index in range(4)
        ],
    ]

    report = calculate_final_metrics(records)

    assert report.rate_details["formal_runtime"].value == 8 / 9
    assert report.rate_details["formal_runtime"].denominator == 9
    assert report.rate_details["deterministic_boundaries"].model_dump()["passed"] == 4
    assert report.rate_details["overall"].value == 12 / 13
    assert report.false_positive_count == 1


def test_serialized_record_has_no_question_answer_sql_or_rows() -> None:
    serialized = json.dumps(_record().model_dump(mode="json"), ensure_ascii=False).casefold()
    assert all(
        term not in serialized for term in ("question", "select ", "raw_rows", "provider_prompt")
    )


def test_record_retains_only_safe_span_failure_summaries() -> None:
    record = _record(
        passed=False,
        failure_category="workflow_reviewer_fail_closed",
        span_failures=("execute_plan_step:skill_execution_failed",),
    )
    serialized = record.model_dump_json()
    assert "execute_plan_step:skill_execution_failed" in serialized
    assert "traceback" not in serialized.casefold()


def test_frozen_final_artifacts_are_consistent_and_only_two_checks_are_adjudicated() -> None:
    artifact_dir = _ROOT / "artifacts" / "evaluation" / "m9-final-eval-v1"
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (artifact_dir / "case_records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    adjudications = json.loads((artifact_dir / "adjudications.json").read_text(encoding="utf-8"))
    assert len(records) == manifest["case_count"] == 13
    assert {record["run_id"] for record in records} == {manifest["run_id"]}
    assert {record["code_commit"] for record in records} == {manifest["code_commit"]}
    assert len(adjudications["adjudications"]) == 2
    assert {
        (item["case_id"], item["check"], item["original_value"], item["corrected_value"])
        for item in adjudications["adjudications"]
    } == {
        ("m9-knowledge-contradiction-004", "facts", False, True),
        ("m9-data-inventory-006", "facts", False, True),
    }
    dataset_hash = hashlib.sha256(_DATASET.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    assert dataset_hash == manifest["dataset_sha256"]
    for name, expected_hash in manifest["artifacts"].items():
        canonical = (artifact_dir / name).read_text(encoding="utf-8").encode("utf-8")
        assert hashlib.sha256(canonical).hexdigest() == expected_hash


def test_v101_derived_metrics_link_to_frozen_run_without_rewriting_it() -> None:
    artifact_dir = _ROOT / "artifacts" / "evaluation" / "m9-final-eval-v1"
    derived_dir = _ROOT / "artifacts" / "evaluation" / "m9-final-eval-v1-v1.0.1-derived"
    source_manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    derived_manifest = json.loads((derived_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((derived_dir / "metrics.json").read_text(encoding="utf-8"))

    assert derived_manifest["run_id"] == source_manifest["run_id"]
    assert derived_manifest["dataset_sha256"] == source_manifest["dataset_sha256"]
    assert derived_manifest["source_code_commit"] == source_manifest["code_commit"]
    assert derived_manifest["formal_provider_evaluation_rerun"] is False
    assert metrics["calculator_version"] == "1.0.1"
    assert metrics["rate_details"]["formal_runtime"]["passed"] == 8
    assert metrics["rate_details"]["formal_runtime"]["total"] == 9
    assert metrics["rate_details"]["deterministic_boundaries"]["value"] == 1.0
    assert metrics["rate_details"]["unanswerable"]["value"] == pytest.approx(2 / 3)
    assert metrics["false_positive_count"] == 1


def test_v102_strict_derivation_binds_all_frozen_sources_and_preserves_failure() -> None:
    artifact_dir = _ROOT / "artifacts" / "evaluation" / "m9-final-eval-v1"
    report, manifest = derive_final_metrics_from_frozen_sources(
        records_path=artifact_dir / "case_records.jsonl",
        dataset_path=_DATASET,
        adjudications_path=artifact_dir / "adjudications.json",
        source_manifest_path=artifact_dir / "manifest.json",
    )

    assert report.calculator_version == "1.0.2"
    assert report.rate_details["formal_runtime"].model_dump()["passed"] == 8
    assert report.rate_details["deterministic_boundaries"].denominator == 4
    assert report.rate_details["overall"].numerator == 12
    assert report.rate_details["unanswerable"].model_dump() == {
        "passed": 2,
        "total": 3,
        "numerator": 2,
        "denominator": 3,
        "value": pytest.approx(2 / 3),
    }
    assert report.false_positive_count == 1
    assert report.provider_call_count == 45
    assert report.input_tokens == 38_549
    assert report.output_tokens == 4_258
    assert report.latency_p50_ms == pytest.approx(9_594.0)
    assert report.latency_p95_ms == 24_250.0
    assert manifest["sole_true_failure_case_id"] == "m9-knowledge-unanswerable-003"
    assert manifest["formal_provider_evaluation_rerun"] is False
    assert manifest["hash_algorithm"] == "sha256"
    assert manifest["hash_semantics"] == "utf8_text_with_newlines_normalized_to_lf"
    assert manifest["source_manifest_self_hash_included"] is False
    assert manifest["source_dataset_sha256"] == manifest["dataset_sha256"]
    zero_rates = [metric for metric in report.rate_details.values() if metric.denominator == 0]
    assert len(zero_rates) == 3
    assert all(metric.value is None for metric in zero_rates)


@pytest.mark.parametrize(
    ("target", "mutation"),
    [
        ("manifest.json", "source_manifest"),
        ("case_records.jsonl", "artifact_modified"),
        ("case_records.jsonl", "duplicate_record"),
        ("case_records.jsonl", "missing_record"),
        ("case_records.jsonl", "extra_record"),
        ("case_records.jsonl", "run_id"),
        ("case_records.jsonl", "code_commit"),
        ("case_records.jsonl", "execution_mode"),
        ("case_records.jsonl", "waive_true_failure"),
        ("dataset.json", "dataset_version"),
        ("adjudications.json", "third_adjudication"),
        ("adjudications.json", "duplicate_adjudication"),
        ("adjudications.json", "wrong_original"),
        ("adjudications.json", "nonexistent_case"),
        ("adjudications.json", "nonexistent_check"),
        ("adjudications.json", "prohibited_field"),
        ("adjudications.json", "run_id"),
        ("adjudications.json", "dataset_hash"),
        ("adjudications.json", "code_commit"),
    ],
)
def test_v102_m9_integrity_mutations_fail_closed(
    tmp_path: Path, target: str, mutation: str
) -> None:
    source = _ROOT / "artifacts" / "evaluation" / "m9-final-eval-v1"
    frozen = tmp_path / "frozen"
    shutil.copytree(source, frozen)
    dataset = tmp_path / "dataset.json"
    shutil.copy2(_DATASET, dataset)
    manifest_path = frozen / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if target == "manifest.json":
        manifest["real_system_failure_count"] = 0
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif target == "dataset.json":
        document = json.loads(dataset.read_text(encoding="utf-8"))
        document["dataset_version"] = "m9-final-eval-v2"
        dataset.write_text(json.dumps(document), encoding="utf-8")
    elif target == "adjudications.json":
        path = frozen / target
        document = json.loads(path.read_text(encoding="utf-8"))
        if mutation in {"third_adjudication", "duplicate_adjudication"}:
            duplicate = dict(document["adjudications"][0])
            if mutation == "third_adjudication":
                duplicate["case_id"] = "m9-knowledge-unanswerable-003"
            document["adjudications"].append(duplicate)
        elif mutation == "wrong_original":
            document["adjudications"][0]["original_value"] = True
        elif mutation == "nonexistent_case":
            document["adjudications"][0]["case_id"] = "m9-missing-999"
        elif mutation == "nonexistent_check":
            document["adjudications"][0]["check"] = "missing_check"
        elif mutation == "prohibited_field":
            document["adjudications"][0]["check"] = "final_status"
        elif mutation == "run_id":
            document["run_id"] = "m9-other-run"
        elif mutation == "dataset_hash":
            document["dataset_sha256"] = "0" * 64
        elif mutation == "code_commit":
            document["code_commit"] = "0" * 40
        path.write_text(json.dumps(document), encoding="utf-8")
        manifest["artifacts"][target] = hashlib.sha256(
            path.read_text(encoding="utf-8").encode("utf-8")
        ).hexdigest()
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    else:
        path = frozen / target
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if mutation == "artifact_modified":
            records[0]["latency_ms"] += 1
        elif mutation == "duplicate_record":
            records.append(dict(records[0]))
        elif mutation == "missing_record":
            records.pop()
        elif mutation == "extra_record":
            extra = dict(records[0])
            extra["case_id"] = "m9-extra-999"
            records.append(extra)
        elif mutation == "run_id":
            records[0]["run_id"] = "m9-other-run"
        elif mutation == "code_commit":
            records[0]["code_commit"] = "0" * 40
        elif mutation == "execution_mode":
            records[0]["execution_mode"] = "frozen_boundary"
        else:
            record = next(
                item for item in records if item["case_id"] == "m9-knowledge-unanswerable-003"
            )
            record["checks"] = {key: True for key in record["checks"]}
            record["passed"] = True
            record["answer_present"] = False
            record["abstained"] = True
            record["failure_category"] = None
        path.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
            encoding="utf-8",
        )
        manifest["artifacts"][target] = hashlib.sha256(
            path.read_text(encoding="utf-8").encode("utf-8")
        ).hexdigest()
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError):
        derive_final_metrics_from_frozen_sources(
            records_path=frozen / "case_records.jsonl",
            dataset_path=dataset,
            adjudications_path=frozen / "adjudications.json",
            source_manifest_path=manifest_path,
        )
