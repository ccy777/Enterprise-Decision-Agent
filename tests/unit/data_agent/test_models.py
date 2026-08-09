"""Data Evidence provenance-contract tests."""

from __future__ import annotations

import pytest

from decision_agent.data.models import QueryAudit, SafeQueryResult
from decision_agent.data_agent.models import DataEvidence


def _result(*, error_code: str | None = None) -> SafeQueryResult:
    return SafeQueryResult(
        columns=["product_id"],
        rows=[["P100"]],
        row_count=1,
        truncated=False,
        elapsed_ms=1.0,
        accessed_tables=["products"],
        audit=QueryAudit(
            request_id="00000000-0000-0000-0000-000000000001",
            normalized_sql="SELECT product_id FROM products LIMIT 1",
            allowed=error_code is None,
            elapsed_ms=1.0,
            row_count=1,
        ),
        error_code=error_code,
    )


def test_data_evidence_preserves_only_successful_guarded_query_fields() -> None:
    evidence = DataEvidence.from_safe_query_result(evidence_id="D1", result=_result())
    assert evidence.evidence_id == "D1"
    assert evidence.normalized_sql == "SELECT product_id FROM products LIMIT 1"
    assert evidence.accessed_tables == ["products"]


def test_data_evidence_cannot_be_created_from_failed_safe_query() -> None:
    with pytest.raises(ValueError, match="successful guarded query"):
        DataEvidence.from_safe_query_result(
            evidence_id="D1",
            result=_result(error_code="write_statement_not_allowed"),
        )


def test_data_evidence_requires_an_accessed_business_table() -> None:
    result = _result()
    result.accessed_tables = []
    with pytest.raises(ValueError, match="successful guarded query"):
        DataEvidence.from_safe_query_result(evidence_id="D1", result=result)
