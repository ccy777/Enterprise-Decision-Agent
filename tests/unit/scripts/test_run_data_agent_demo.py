"""Public projection coverage for the Data Agent command-line entry point."""

from __future__ import annotations

import argparse
import json

import pytest
import scripts.run_data_agent_demo as demo
from scripts.run_data_agent_demo import project_public_result

from decision_agent.agents.data_query_planner import DataPlanStatus
from decision_agent.data_agent.models import DataEvidence
from decision_agent.domain import ErrorRecord
from decision_agent.workflows.data_agent import DataAgentState, DataAgentStatus


def test_public_projection_includes_only_audited_data_fields() -> None:
    state = DataAgentState(
        query="question",
        status=DataAgentStatus.ANSWERABLE_FINAL,
        plan_status=DataPlanStatus.READY,
        intent="query",
        planned_sql="SELECT product_id FROM products LIMIT 1",
        decision_reason="complete",
        answer="answer[D1]",
        citations=["[D1]"],
        data_evidence=(
            DataEvidence(
                evidence_id="D1",
                normalized_sql="SELECT product_id FROM products LIMIT 1",
                columns=["product_id"],
                rows=[["P100"]],
                row_count=1,
                truncated=False,
                accessed_tables=["products"],
                elapsed_ms=1.0,
            ),
        ),
    )
    result = project_public_result(state)
    assert set(result) == {
        "status",
        "intent",
        "answer",
        "citations",
        "missing_information",
        "decision_reason",
        "data_evidence",
    }
    assert "password" not in str(result).lower()


def test_failed_projection_exposes_only_controlled_error_summary() -> None:
    state = DataAgentState(
        query="question",
        status=DataAgentStatus.FAILED,
        errors=[ErrorRecord(code="data_query_planning_failed", message="safe", retryable=False)],
    )
    result = project_public_result(state)
    assert result["errors"] == [{"code": "data_query_planning_failed", "retryable": False}]
    assert "message" not in str(result["errors"])


@pytest.mark.asyncio
async def test_cli_unexpected_runtime_error_uses_safe_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def unexpected_failure(*args: object, **kwargs: object) -> DataAgentState:
        raise RuntimeError("credential=not-for-output")

    monkeypatch.setattr(demo, "Settings", lambda: object())
    monkeypatch.setattr(
        demo.OpenAICompatibleDataQueryPlanner,
        "from_settings",
        lambda settings: object(),
    )
    monkeypatch.setattr(demo, "EnterpriseDataMCPClient", lambda **kwargs: object())
    monkeypatch.setattr(
        demo.OpenAICompatibleDataAnswerGenerator,
        "from_settings",
        lambda settings: object(),
    )
    monkeypatch.setattr(demo, "run_data_agent", unexpected_failure)

    exit_code = await demo._run(argparse.Namespace(query="question"))
    output = capsys.readouterr().out
    assert exit_code == 2
    assert json.loads(output) == {
        "status": "failed",
        "errors": [{"code": "data_agent_runtime_failed"}],
    }
    assert "credential" not in output
