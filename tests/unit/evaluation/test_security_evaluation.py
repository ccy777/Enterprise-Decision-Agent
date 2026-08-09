from __future__ import annotations

import json
from pathlib import Path

import pytest

from decision_agent.evaluation.security_evaluation import (
    CASE_HANDLERS,
    SecurityCaseObservation,
    SecurityEvaluationReport,
    evaluate_security_cases,
    load_security_cases,
    write_security_report,
)

_CASES = Path("datasets/security/m8c_d_security_cases.json")


def test_m8c_security_matrix_executes_exact_28_case_observations(tmp_path: Path) -> None:
    cases = load_security_cases(_CASES)
    assert len(cases) == len(CASE_HANDLERS) == 28
    assert {case.case_id for case in cases} == set(CASE_HANDLERS)

    report = evaluate_security_cases(cases)
    output = tmp_path / "security-summary.json"
    write_security_report(report, output)
    actual = SecurityEvaluationReport.model_validate_json(output.read_text(encoding="utf-8"))

    assert actual == report
    assert actual.passed == actual.total_cases == 28
    assert actual.failed == 0
    assert all(result.passed for result in actual.case_results)
    assert actual.unauthorized_release_count == actual.sensitive_leak_count == 0
    assert actual.provider_bypass_count == actual.tool_bypass_count == 0
    assert actual.audit_integrity_detection_rate == 1.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("expected_error_code", "wrong_code"),
        ("expected_provider_calls", 1),
        ("expected_tool_calls", 1),
        ("expected_release", True),
        ("sensitive_content_expected", True),
    ],
)
def test_security_expected_field_mutation_fails_closed(field: str, value: object) -> None:
    cases = list(load_security_cases(_CASES))
    index = next(i for i, case in enumerate(cases) if case.case_id == "tool_denied")
    cases[index] = cases[index].model_copy(update={field: value})

    with pytest.raises(ValueError):
        evaluate_security_cases(tuple(cases))


def test_missing_handler_and_missing_observation_fail_closed() -> None:
    cases = load_security_cases(_CASES)
    missing = dict(CASE_HANDLERS)
    missing.pop("tool_denied")
    with pytest.raises(ValueError, match="exact 28 IDs"):
        evaluate_security_cases(cases, handlers=missing)

    no_observation = dict(CASE_HANDLERS)
    no_observation["tool_denied"] = lambda _case, _root: None  # type: ignore[assignment]
    with pytest.raises(ValueError, match="no observation"):
        evaluate_security_cases(cases, handlers=no_observation)


def test_wrong_observation_fails_closed() -> None:
    cases = load_security_cases(_CASES)
    changed = dict(CASE_HANDLERS)

    def wrong(case, _root):
        return SecurityCaseObservation(
            case_id=case.case_id,
            observed_outcome="fail_closed",
            observed_error_code="wrong_code",
            observed_provider_calls=0,
            observed_tool_calls=0,
            observed_release=False,
            sensitive_content_detected=False,
        )

    changed["tool_denied"] = wrong
    with pytest.raises(ValueError, match="thresholds are not met"):
        evaluate_security_cases(cases, handlers=changed)


def test_sensitive_leak_observation_fails_closed() -> None:
    cases = load_security_cases(_CASES)
    changed = dict(CASE_HANDLERS)

    def leaked(case, _root):
        return SecurityCaseObservation(
            case_id=case.case_id,
            observed_outcome=case.expected_outcome,
            observed_error_code=case.expected_error_code,
            observed_provider_calls=case.expected_provider_calls,
            observed_tool_calls=case.expected_tool_calls,
            observed_release=case.expected_release,
            sensitive_content_detected=True,
        )

    changed["tool_denied"] = leaked
    with pytest.raises(ValueError, match="security bypass threshold exceeded"):
        evaluate_security_cases(cases, handlers=changed)


def test_security_report_rejects_extra_case() -> None:
    report = evaluate_security_cases(load_security_cases(_CASES))
    payload = report.model_dump()
    extra = dict(payload["case_results"][0])
    extra["case_id"] = "extra_security_case"
    payload["case_results"] = [*payload["case_results"], extra]
    payload["total_cases"] = payload["passed"] = 29

    with pytest.raises(ValueError, match="exact 28"):
        SecurityEvaluationReport.model_validate(payload)


@pytest.mark.parametrize("mutation", ["duplicate", "unknown", "release"])
def test_security_matrix_identity_mutations_are_rejected(tmp_path: Path, mutation: str) -> None:
    records = json.loads(_CASES.read_text(encoding="utf-8"))
    if mutation == "duplicate":
        records[-1]["case_id"] = records[0]["case_id"]
    elif mutation == "unknown":
        records[-1]["case_id"] = "unknown_security_case"
    else:
        records[-1]["expected_outcome"] = "allowed"
        records[-1]["expected_release"] = True
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(records), encoding="utf-8")

    with pytest.raises(ValueError):
        load_security_cases(path)
