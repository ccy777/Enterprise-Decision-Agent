"""Versioned, payload-free contracts and metrics for the M9 final evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from statistics import mean

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from decision_agent.observability.models import TraceAttribute

_SAFE_SPAN_FAILURE = re.compile(r"^[a-z][a-z0-9_]{0,63}:[a-z][a-z0-9_]{0,63}$")


class EvaluationMode(StrEnum):
    """How a fixed case obtains evidence without obscuring real-runtime coverage."""

    REAL_RUNTIME = "real_runtime"
    FROZEN_BOUNDARY = "frozen_boundary"


class FinalEvaluationCase(BaseModel):
    """One immutable case specification; questions never enter result artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    case_id: str = Field(pattern=r"^m9-[a-z0-9-]+$")
    scenario: str = Field(min_length=1, max_length=64)
    question: str = Field(min_length=1, repr=False)
    execution_mode: EvaluationMode
    expected_route: str | None = None
    answerable: bool
    expected_tool_categories: tuple[str, ...] = ()
    expected_evidence_types: tuple[str, ...] = ()
    required_citation_prefixes: tuple[str, ...] = ()
    required_answer_facts: tuple[str, ...] = Field(default=(), repr=False)
    required_answer_fact_groups: tuple[tuple[str, ...], ...] = Field(default=(), repr=False)
    expected_final_status: str
    expected_error_codes: tuple[str, ...] = ()
    expected_security_decision: str = Field(pattern=r"^(allowed|denied)$")
    evidence_reference: str | None = None
    tags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _mode_contract(self) -> FinalEvaluationCase:
        if self.execution_mode is EvaluationMode.REAL_RUNTIME and self.evidence_reference:
            raise ValueError("real runtime cases cannot use frozen evidence")
        if self.execution_mode is EvaluationMode.FROZEN_BOUNDARY and not self.evidence_reference:
            raise ValueError("frozen boundary cases require evidence_reference")
        if any(
            not group or any(not item.strip() for item in group)
            for group in self.required_answer_fact_groups
        ):
            raise ValueError("answer fact groups must contain nonblank alternatives")
        return self


class FinalEvaluationDataset(BaseModel):
    """Frozen M9 dataset manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: str = "1.0"
    dataset_version: str = Field(pattern=r"^m9-final-eval-v[0-9]+$")
    frozen_at: datetime
    cases: tuple[FinalEvaluationCase, ...] = Field(min_length=1)

    @field_validator("cases")
    @classmethod
    def _unique_case_ids(
        cls, value: tuple[FinalEvaluationCase, ...]
    ) -> tuple[FinalEvaluationCase, ...]:
        identifiers = [case.case_id for case in value]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("case_id values must be unique")
        return value


class FinalCaseRecord(BaseModel):
    """Redacted result record; content, SQL, rows, prompts and exceptions are forbidden."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    run_id: str
    dataset_version: str
    code_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    case_id: str
    scenario: str
    execution_mode: EvaluationMode
    expected_answerable: bool | None = None
    expected_final_status: str | None = None
    started_at: datetime
    completed_at: datetime
    route: str | None = None
    final_status: str
    error_code: str | None = None
    trace_id: str | None = None
    latency_ms: float = Field(ge=0, allow_inf_nan=False)
    provider_call_count: int = Field(ge=0)
    provider_operations: tuple[str, ...] = ()
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    usage_complete: bool
    stage_latency_ms: dict[str, float] = Field(default_factory=dict)
    citation_count: int = Field(ge=0)
    citation_prefix_counts: dict[str, int] = Field(default_factory=dict)
    evidence_types: tuple[str, ...] = ()
    tool_categories: tuple[str, ...] = ()
    answer_present: bool
    abstained: bool
    sensitive_leak_detected: bool = False
    security_decision: str = Field(pattern=r"^(allowed|denied|not_applicable)$")
    checks: dict[str, bool] = Field(default_factory=dict)
    span_failures: tuple[str, ...] = ()
    passed: bool
    failure_category: str | None = None
    evidence_reference: str | None = None

    @field_validator("citation_prefix_counts")
    @classmethod
    def _safe_prefix_counts(cls, value: dict[str, int]) -> dict[str, int]:
        if any(key not in {"D", "E", "G"} or count < 0 for key, count in value.items()):
            raise ValueError("citation prefix counts are closed and non-negative")
        return value

    @field_validator("span_failures")
    @classmethod
    def _safe_span_failures(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) > 32 or any(not _SAFE_SPAN_FAILURE.fullmatch(item) for item in value):
            raise ValueError("span failures must contain bounded fixed codes")
        return value


class RateMetric(BaseModel):
    """One rate with visible sample counts and honest empty-sample semantics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: int = Field(ge=0)
    total: int = Field(ge=0)
    numerator: int = Field(ge=0)
    denominator: int = Field(ge=0)
    value: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def _consistent_counts(self) -> RateMetric:
        if self.passed != self.numerator or self.total != self.denominator:
            raise ValueError("rate aliases must contain identical counts")
        if self.passed > self.total:
            raise ValueError("rate passed count cannot exceed total")
        expected = None if self.total == 0 else self.passed / self.total
        if self.value != expected:
            raise ValueError("rate value must be derived from its counts")
        return self


class FinalMetricReport(BaseModel):
    """Independently reproducible aggregate containing no source payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_version: str
    run_id: str
    code_commit: str
    total_case_count: int
    executed_case_count: int
    frozen_boundary_case_count: int
    passed_case_count: int
    failed_case_count: int
    calculator_version: str = "1.0.2"
    derived_from_frozen_run: bool = True
    rate_details: dict[str, RateMetric]
    metric_semantics: dict[str, str]
    deprecated_rate_fields: dict[str, str]
    false_positive_count: int
    pass_rate: float | None
    routing_accuracy: float | None
    scenario_accuracy: float | None
    planner_schema_valid_rate: float | None
    planner_skill_accuracy: float | None
    tool_selection_accuracy: float | None
    workflow_success_rate: float | None
    reviewer_accept_rate: float | None
    fail_closed_rate: float | None
    unauthorized_release_count: int
    retrieval_hit_rate: float | None
    parent_document_hit_rate: float | None
    evidence_precision: float | None
    evidence_recall: float | None
    evidence_selection_rate: float | None
    citation_compliance_rate: float | None
    citation_validity_rate: float | None
    citation_support_rate: float | None
    citation_coverage_rate: float | None
    answerability_accuracy: float | None
    unsupported_claim_rate: float | None
    data_tool_success_rate: float | None
    safe_query_success_rate: float | None
    expected_data_assertion_accuracy: float | None
    sql_guard_rejection_correctness: float | None
    mcp_lifecycle_success_rate: float | None
    unauthorized_data_access_count: int
    answer_correctness: float | None
    key_fact_coverage: float | None
    recommendation_alignment: float | None
    abstention_precision: float | None
    abstention_recall: float | None
    hallucination_rate: float | None
    sensitive_leak_count: int
    security_violation_count: int
    security_denial_success_rate: float | None
    provider_call_count: int
    provider_calls_per_request: float
    input_tokens: int | None
    output_tokens: int | None
    usage_coverage_rate: float | None
    successful_request_tokens: int | None
    failed_request_tokens: int | None
    provider_budget_exceeded_count: int
    cost_amount: float | None = None
    cost_currency: str | None = None
    cost_unavailable_reason: str | None = "provider_pricing_not_frozen"
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    retrieval_latency_mean_ms: float | None
    data_latency_mean_ms: float | None
    stage_latency_mean_ms: dict[str, float]
    unavailable_metric_reasons: dict[str, str]
    failure_counts: dict[str, int]


def calculate_final_metrics(records: Sequence[FinalCaseRecord]) -> FinalMetricReport:
    """Calculate metrics only from redacted case records."""
    if not records:
        raise ValueError("records must not be empty")
    run_ids = {record.run_id for record in records}
    versions = {record.dataset_version for record in records}
    commits = {record.code_commit for record in records}
    if len(run_ids) != 1 or len(versions) != 1 or len(commits) != 1:
        raise ValueError("records must share run_id, dataset_version and code_commit")

    executed = [r for r in records if r.execution_mode is EvaluationMode.REAL_RUNTIME]
    frozen = [r for r in records if r.execution_mode is EvaluationMode.FROZEN_BOUNDARY]
    if not executed:
        raise ValueError("at least one real runtime record is required")
    route_records = [r for r in executed if "route" in r.checks]
    citation_records = [r for r in executed if "citations" in r.checks]
    answerability_records = [r for r in executed if "answerability" in r.checks]
    denied = [r for r in executed if r.security_decision == "denied"]
    expected_failures = [r for r in records if r.expected_final_status == "failed"]
    answerable_rag = [
        r for r in executed if r.expected_answerable is True and r.route in {"knowledge", "mixed"}
    ]
    data_records = [r for r in executed if "data" in r.tool_categories]
    mixed_records = [r for r in executed if r.route == "mixed"]
    predicted_abstentions = [r for r in executed if r.abstained]
    expected_abstentions = [r for r in executed if r.expected_answerable is False]
    provider_calls = sum(record.provider_call_count for record in executed)
    usage_calls = sum(record.provider_call_count for record in executed if record.usage_complete)
    usage_complete = usage_calls == provider_calls
    failure_counts: dict[str, int] = {}
    for record in records:
        if record.failure_category:
            failure_counts[record.failure_category] = (
                failure_counts.get(record.failure_category, 0) + 1
            )
    latencies = sorted(record.latency_ms for record in executed)
    stage_names = sorted({stage for record in executed for stage in record.stage_latency_ms})
    all_usage_complete = all(record.usage_complete for record in executed)
    answerable_records = [r for r in executed if r.expected_answerable is True]
    rate_details = {
        "overall": _rate_metric(records, lambda record: record.passed),
        "formal_runtime": _rate_metric(executed, lambda record: record.passed),
        "deterministic_boundaries": _rate_metric(frozen, lambda record: record.passed),
        "routing_check": _check_rate_metric(route_records, "route"),
        "scenario_check": _check_rate_metric(route_records, "route"),
        "tool_selection_check": _check_rate_metric(executed, "tools"),
        "workflow_completion": _check_rate_metric(executed, "final_status"),
        "workflow_reviewer_accept": _check_rate_metric(mixed_records, "final_status"),
        "fail_closed": _rate_metric(
            expected_failures,
            lambda record: record.abstained or record.checks.get("frozen_evidence", False),
        ),
        "m9_final_case_evidence_check": _check_rate_metric(answerable_rag, "evidence"),
        "m9_final_case_citation_check": _check_rate_metric(citation_records, "citations"),
        "m9_final_case_citation_support_check": _rate_metric(
            citation_records,
            lambda record: (
                record.checks.get("citations", False) and record.checks.get("facts", False)
            ),
        ),
        "m9_final_case_fact_check": _check_rate_metric(answerable_records, "facts"),
        "answerability_check": _check_rate_metric(answerability_records, "answerability"),
        "unanswerable": _rate_metric(expected_abstentions, lambda record: record.abstained),
        "unsupported_claim": _rate_metric(
            expected_abstentions, lambda record: record.answer_present
        ),
        "data_tool_check": _check_rate_metric(data_records, "tools"),
        "safe_query_check": _check_rate_metric(data_records, "final_status"),
        "data_assertion_check": _check_rate_metric(data_records, "facts"),
        "mcp_lifecycle_check": _check_rate_metric(data_records, "tools"),
        "recommendation_alignment_check": _check_rate_metric(mixed_records, "facts"),
        "abstention_precision": _rate_metric(
            predicted_abstentions, lambda record: not record.expected_answerable
        ),
        "hallucination": _rate_metric(expected_abstentions, lambda record: record.answer_present),
        "security_boundary_check": _check_rate_metric(denied, "security"),
        "usage_coverage": _count_rate(usage_calls, provider_calls),
        "planner_schema_check": _count_rate(0, 0),
        "planner_skill_check": _count_rate(0, 0),
        "sql_guard_rejection_check": _count_rate(0, 0),
    }
    return FinalMetricReport(
        dataset_version=next(iter(versions)),
        run_id=next(iter(run_ids)),
        code_commit=next(iter(commits)),
        total_case_count=len(records),
        executed_case_count=len(executed),
        frozen_boundary_case_count=len(frozen),
        passed_case_count=sum(record.passed for record in records),
        failed_case_count=sum(not record.passed for record in records),
        rate_details=rate_details,
        metric_semantics={
            "independent_50_query_retrieval_baseline": (
                "separate frozen retrieval benchmark; not recalculated from M9 final cases"
            ),
            "m9_final_case_evidence_check": (
                "case-level boolean evidence assertion; not per-evidence precision or recall"
            ),
            "m9_final_case_citation_check": "case-level citation-format and scope assertion",
            "m9_final_case_fact_check": "case-level expected-fact assertion after adjudication",
            "answerability_check": "case-level answerability assertion",
            "data_assertion_check": "case-level deterministic data-fact assertion",
            "security_boundary_check": "case-level denied-request security assertion",
        },
        deprecated_rate_fields={
            "retrieval_hit_rate": "use rate_details.m9_final_case_evidence_check",
            "parent_document_hit_rate": "use rate_details.m9_final_case_evidence_check",
            "evidence_precision": "proxy only; use rate_details.m9_final_case_citation_check",
            "evidence_recall": "proxy only; use rate_details.m9_final_case_fact_check",
            "evidence_selection_rate": "use rate_details.m9_final_case_evidence_check",
        },
        false_positive_count=sum(record.answer_present for record in expected_abstentions),
        pass_rate=rate_details["overall"].value,
        routing_accuracy=rate_details["routing_check"].value,
        scenario_accuracy=rate_details["scenario_check"].value,
        planner_schema_valid_rate=None,
        planner_skill_accuracy=None,
        tool_selection_accuracy=rate_details["tool_selection_check"].value,
        workflow_success_rate=rate_details["workflow_completion"].value,
        reviewer_accept_rate=rate_details["workflow_reviewer_accept"].value,
        fail_closed_rate=rate_details["fail_closed"].value,
        unauthorized_release_count=sum(
            record.security_decision == "denied" and record.answer_present for record in executed
        ),
        retrieval_hit_rate=rate_details["m9_final_case_evidence_check"].value,
        parent_document_hit_rate=rate_details["m9_final_case_evidence_check"].value,
        evidence_precision=rate_details["m9_final_case_citation_check"].value,
        evidence_recall=rate_details["m9_final_case_fact_check"].value,
        evidence_selection_rate=rate_details["m9_final_case_evidence_check"].value,
        citation_compliance_rate=rate_details["m9_final_case_citation_check"].value,
        citation_validity_rate=rate_details["m9_final_case_citation_check"].value,
        citation_support_rate=rate_details["m9_final_case_citation_support_check"].value,
        citation_coverage_rate=_check_rate(answerable_rag, "citations"),
        answerability_accuracy=rate_details["answerability_check"].value,
        unsupported_claim_rate=rate_details["unsupported_claim"].value,
        data_tool_success_rate=rate_details["data_tool_check"].value,
        safe_query_success_rate=rate_details["safe_query_check"].value,
        expected_data_assertion_accuracy=rate_details["data_assertion_check"].value,
        sql_guard_rejection_correctness=None,
        mcp_lifecycle_success_rate=rate_details["mcp_lifecycle_check"].value,
        unauthorized_data_access_count=sum(
            record.security_decision == "denied" and "data" in record.tool_categories
            for record in executed
        ),
        answer_correctness=rate_details["m9_final_case_fact_check"].value,
        key_fact_coverage=rate_details["m9_final_case_fact_check"].value,
        recommendation_alignment=rate_details["recommendation_alignment_check"].value,
        abstention_precision=rate_details["abstention_precision"].value,
        abstention_recall=rate_details["unanswerable"].value,
        hallucination_rate=rate_details["hallucination"].value,
        sensitive_leak_count=sum(record.sensitive_leak_detected for record in records),
        security_violation_count=sum(
            1 for record in denied if not record.checks.get("security", False)
        ),
        security_denial_success_rate=rate_details["security_boundary_check"].value,
        provider_call_count=provider_calls,
        provider_calls_per_request=provider_calls / len(executed),
        input_tokens=(
            sum(record.input_tokens or 0 for record in executed) if usage_complete else None
        ),
        output_tokens=(
            sum(record.output_tokens or 0 for record in executed) if usage_complete else None
        ),
        usage_coverage_rate=rate_details["usage_coverage"].value,
        successful_request_tokens=(
            sum((r.input_tokens or 0) + (r.output_tokens or 0) for r in executed if r.passed)
            if all_usage_complete
            else None
        ),
        failed_request_tokens=(
            sum((r.input_tokens or 0) + (r.output_tokens or 0) for r in executed if not r.passed)
            if all_usage_complete
            else None
        ),
        provider_budget_exceeded_count=sum(
            record.error_code == "provider_budget_exceeded" for record in records
        ),
        latency_mean_ms=mean(latencies),
        latency_p50_ms=_percentile(latencies, 0.50),
        latency_p95_ms=_percentile(latencies, 0.95),
        retrieval_latency_mean_ms=_stage_mean(executed, {"retrieval", "reranking"}),
        data_latency_mean_ms=_stage_mean(executed, {"data_access"}),
        stage_latency_mean_ms={
            stage: mean(record.stage_latency_ms.get(stage, 0.0) for record in executed)
            for stage in stage_names
        },
        unavailable_metric_reasons={
            "planner_schema_valid_rate": "not_recorded_in_frozen_case_schema",
            "planner_skill_accuracy": "not_recorded_in_frozen_case_schema",
            "sql_guard_rejection_correctness": "no_dedicated_case_in_m9_final_eval_v1",
            "retrieval_latency_mean_ms": "not_recorded_in_frozen_case_schema",
            "data_latency_mean_ms": "not_recorded_in_frozen_case_schema",
            "stage_latency_mean_ms": "not_recorded_in_frozen_case_schema",
        },
        failure_counts=dict(sorted(failure_counts.items())),
    )


_APPROVED_ADJUDICATIONS = (
    {
        "case_id": "m9-knowledge-contradiction-004",
        "check": "facts",
        "original_value": False,
        "corrected_value": True,
        "category": "confirmed_literal_assertion_false_negative",
        "basis": "semantic_fact_group_match",
    },
    {
        "case_id": "m9-data-inventory-006",
        "check": "facts",
        "original_value": False,
        "corrected_value": True,
        "category": "confirmed_literal_assertion_false_negative",
        "basis": "semantic_identifier_or_name_match",
    },
)
_TRUE_FAILURE_CASE_ID = "m9-knowledge-unanswerable-003"
_EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "f299aa107bc659bcd87af1ce4914daa488848720a384eb97db3b0901649aa929"
)
_EXPECTED_SOURCE_ARTIFACT_HASHES = {
    "case_records.jsonl": "1e01570e9dd16ce55f81c01b6fb13b2fc25c9617ae91721ddf0d26ab7b9a7c54",
    "metrics.json": "7977dbb47b4b5e1fc00c5fccd312a1bfcd7ac96ac7b344f11624dfdcf1b21b93",
    "metrics_raw.json": "7435512bae0008196e25f3f8476bdbd9bd0f66f55c3599cea8309590ef6651b3",
    "adjudications.json": "53206de8a443cdd567d5267cd395e4a720f620ea5f6225aa48585110730c3a7e",
    "failure_records.jsonl": "212caed6240ea7c8798af3ed62bf9a9306d9043cfdb829528ae18b3b523c9bf6",
    "run_manifest.json": "870002c3a3b7d6ef845d27db62012165df2711f3e51edef9e530a28e2b455316",
}


def derive_final_metrics_from_frozen_sources(
    *,
    records_path: Path,
    dataset_path: Path,
    adjudications_path: Path,
    source_manifest_path: Path,
) -> tuple[FinalMetricReport, dict[str, object]]:
    """Strictly bind, validate, and derive v1.0.2 metrics from the frozen M9 run."""
    if _text_sha256(source_manifest_path) != _EXPECTED_SOURCE_MANIFEST_SHA256:
        raise ValueError("frozen source manifest hash mismatch")
    source_manifest = _load_object(source_manifest_path)
    required_manifest_fields = {
        "schema_version",
        "run_id",
        "dataset_version",
        "dataset_sha256",
        "code_commit",
        "case_count",
        "adjudication_policy",
        "real_system_failure_count",
        "artifacts",
        "redaction",
    }
    if set(source_manifest) != required_manifest_fields:
        raise ValueError("source manifest schema is not the frozen 1.0 contract")
    if (
        source_manifest["schema_version"] != "1.0"
        or source_manifest["run_id"] != "m9-formal-20260801T110101Z"
        or source_manifest["dataset_version"] != "m9-final-eval-v1"
        or source_manifest["dataset_sha256"]
        != "3cca357227d49985e5072e82d951a213b9a37e4abfe734eda68ef071ed8114bf"
        or source_manifest["code_commit"] != "6d4ad853bb1a841ca685a5f812ff930067dabb3d"
        or source_manifest["case_count"] != 13
        or source_manifest["adjudication_policy"] != "two_confirmed_literal_false_negatives_only"
        or source_manifest["real_system_failure_count"] != 1
    ):
        raise ValueError("source manifest identity or counts changed")
    artifact_hashes = source_manifest["artifacts"]
    if not isinstance(artifact_hashes, dict) or artifact_hashes != _EXPECTED_SOURCE_ARTIFACT_HASHES:
        raise ValueError("source manifest artifact hashes are invalid")
    for name, expected in artifact_hashes.items():
        artifact = source_manifest_path.parent / str(name)
        if not artifact.is_file() or _text_sha256(artifact) != expected:
            raise ValueError(f"frozen source artifact hash mismatch: {name}")
    if _text_sha256(dataset_path) != source_manifest["dataset_sha256"]:
        raise ValueError("frozen dataset hash mismatch")
    run_manifest = _load_object(source_manifest_path.parent / "run_manifest.json")
    if (
        run_manifest.get("schema_version") != "1.0"
        or run_manifest.get("run_kind") != "formal"
        or run_manifest.get("run_id") != source_manifest["run_id"]
        or run_manifest.get("dataset_version") != source_manifest["dataset_version"]
        or run_manifest.get("dataset_sha256") != source_manifest["dataset_sha256"]
        or run_manifest.get("code_commit") != source_manifest["code_commit"]
    ):
        raise ValueError("run manifest is not bound to the frozen evidence")

    dataset = FinalEvaluationDataset.model_validate_json(dataset_path.read_text(encoding="utf-8"))
    if dataset.dataset_version != source_manifest["dataset_version"]:
        raise ValueError("dataset version is not bound to the source manifest")
    case_by_id = {case.case_id: case for case in dataset.cases}
    raw_records = _load_jsonl(records_path)
    if len(raw_records) != source_manifest["case_count"] or len(raw_records) != len(dataset.cases):
        raise ValueError("record, dataset, and manifest case counts do not match")
    record_ids = [str(raw.get("case_id")) for raw in raw_records]
    if len(set(record_ids)) != len(record_ids) or set(record_ids) != set(case_by_id):
        raise ValueError("record IDs do not match the frozen dataset exactly")

    adjudication_document = _load_object(adjudications_path)
    if set(adjudication_document) != {
        "schema_version",
        "run_id",
        "code_commit",
        "dataset_sha256",
        "adjudications",
    }:
        raise ValueError("adjudication schema is not the frozen 1.0 contract")
    if (
        adjudication_document["schema_version"] != "1.0"
        or adjudication_document["run_id"] != source_manifest["run_id"]
        or adjudication_document["code_commit"] != source_manifest["code_commit"]
        or adjudication_document["dataset_sha256"] != source_manifest["dataset_sha256"]
    ):
        raise ValueError("adjudications are not bound to the frozen run")
    actual_adjudications = adjudication_document["adjudications"]
    if actual_adjudications != list(_APPROVED_ADJUDICATIONS):
        raise ValueError("only the two approved adjudications are permitted")

    corrections = {(item["case_id"], item["check"]): item for item in actual_adjudications}
    records: list[FinalCaseRecord] = []
    for raw_record in raw_records:
        raw = dict(raw_record)
        if (
            raw.get("run_id") != source_manifest["run_id"]
            or raw.get("dataset_version") != source_manifest["dataset_version"]
            or raw.get("code_commit") != source_manifest["code_commit"]
        ):
            raise ValueError("case record is not bound to the frozen run")
        case_id = str(raw["case_id"])
        case = case_by_id[case_id]
        if raw.get("scenario") != case.scenario or raw.get("execution_mode") != case.execution_mode:
            raise ValueError("case record does not match its dataset case")
        checks = dict(raw.get("checks", {}))
        for (corrected_case_id, check), adjudication in corrections.items():
            if corrected_case_id != case_id:
                continue
            if checks.get(check) is not adjudication["original_value"]:
                raise ValueError("adjudication original value does not match the frozen record")
            checks[check] = adjudication["corrected_value"]
        raw["expected_answerable"] = case.answerable
        raw["expected_final_status"] = case.expected_final_status
        raw["checks"] = checks
        raw["passed"] = all(checks.values())
        raw["failure_category"] = (
            None if raw["passed"] else raw.get("failure_category") or "assertion_failure"
        )
        records.append(FinalCaseRecord.model_validate(raw))

    true_failure = next(record for record in records if record.case_id == _TRUE_FAILURE_CASE_ID)
    if (
        true_failure.passed
        or true_failure.expected_answerable is not False
        or true_failure.abstained
        or not true_failure.answer_present
        or true_failure.failure_category != "assertion_failure"
        or any(key[0] == _TRUE_FAILURE_CASE_ID for key in corrections)
    ):
        raise ValueError("the sole true frozen failure was modified or waived")

    report = calculate_final_metrics(records)
    _verify_v102_metric_invariants(report)
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "calculator_version": "1.0.2",
        "hash_algorithm": "sha256",
        "hash_semantics": "utf8_text_with_newlines_normalized_to_lf",
        "source_manifest_self_hash_included": False,
        "derived_from_frozen_run": True,
        "run_id": report.run_id,
        "dataset_version": report.dataset_version,
        "dataset_sha256": _text_sha256(dataset_path),
        "source_dataset_sha256": _text_sha256(dataset_path),
        "source_code_commit": report.code_commit,
        "source_manifest_sha256": _text_sha256(source_manifest_path),
        "source_case_records_sha256": _text_sha256(records_path),
        "source_adjudications_sha256": _text_sha256(adjudications_path),
        "source_run_manifest_sha256": _text_sha256(
            source_manifest_path.parent / "run_manifest.json"
        ),
        "approved_adjudication_count": len(_APPROVED_ADJUDICATIONS),
        "sole_true_failure_case_id": _TRUE_FAILURE_CASE_ID,
        "formal_runtime": report.rate_details["formal_runtime"].model_dump(),
        "deterministic_boundaries": report.rate_details["deterministic_boundaries"].model_dump(),
        "overall": report.rate_details["overall"].model_dump(),
        "unanswerable": report.rate_details["unanswerable"].model_dump(),
        "false_positive_count": report.false_positive_count,
        "provider_call_count": report.provider_call_count,
        "input_tokens": report.input_tokens,
        "output_tokens": report.output_tokens,
        "latency_p50_ms": report.latency_p50_ms,
        "latency_p95_ms": report.latency_p95_ms,
        "formal_provider_evaluation_rerun": False,
    }
    return report, manifest


def _verify_v102_metric_invariants(report: FinalMetricReport) -> None:
    expected_rates = {
        "formal_runtime": (8, 9),
        "deterministic_boundaries": (4, 4),
        "overall": (12, 13),
        "unanswerable": (2, 3),
    }
    for name, (passed, total) in expected_rates.items():
        metric = report.rate_details[name]
        if (metric.numerator, metric.denominator, metric.value) != (
            passed,
            total,
            passed / total,
        ):
            raise ValueError(f"frozen metric invariant changed: {name}")
    if (
        report.false_positive_count != 1
        or report.provider_call_count != 45
        or report.input_tokens != 38_549
        or report.output_tokens != 4_258
        or report.latency_p50_ms != 9_593.999999997322
        or report.latency_p95_ms != 24_250.0
    ):
        raise ValueError("frozen call, token, latency, or false-positive evidence changed")
    for metric in report.rate_details.values():
        if metric.denominator == 0 and metric.value is not None:
            raise ValueError("zero-denominator rate must have a null value")


def _load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path.name}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not records or any(not isinstance(record, dict) for record in records):
        raise ValueError("case records must be nonempty JSON objects")
    return records


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def _rate(
    records: Sequence[FinalCaseRecord], predicate: Callable[[FinalCaseRecord], bool]
) -> float | None:
    return _rate_metric(records, predicate).value


def _check_rate(records: Sequence[FinalCaseRecord], check: str) -> float | None:
    return _check_rate_metric(records, check).value


def _rate_metric(
    records: Sequence[FinalCaseRecord], predicate: Callable[[FinalCaseRecord], bool]
) -> RateMetric:
    return _count_rate(sum(predicate(record) for record in records), len(records))


def _check_rate_metric(records: Sequence[FinalCaseRecord], check: str) -> RateMetric:
    return _rate_metric(records, lambda record: record.checks.get(check, False))


def _count_rate(passed: int, total: int) -> RateMetric:
    return RateMetric(
        passed=passed,
        total=total,
        numerator=passed,
        denominator=total,
        value=None if total == 0 else passed / total,
    )


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    index = max(0, math.ceil(quantile * len(values)) - 1)
    return values[index]


def _stage_mean(records: Sequence[FinalCaseRecord], stages: set[str]) -> float | None:
    values = [
        sum(record.stage_latency_ms[stage] for stage in stages if stage in record.stage_latency_ms)
        for record in records
        if any(stage in record.stage_latency_ms for stage in stages)
    ]
    return mean(values) if values else None


def trace_attributes(attributes: Iterable[TraceAttribute]) -> Mapping[str, object]:
    """Project already-safe trace attributes for evaluator internals."""
    return {item.key: item.value for item in attributes}
