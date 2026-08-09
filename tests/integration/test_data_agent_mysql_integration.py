"""Opt-in Level 2: the Data Agent reaches real MySQL through stdio MCP."""

from __future__ import annotations

import os

import pytest

from decision_agent.agents.data_answer_generator import DataAnswerDraft
from decision_agent.agents.data_query_planner import DataQueryPlan
from decision_agent.config import Settings
from decision_agent.data.ground_truth import OperationsQuestion, load_operations_questions
from decision_agent.mcp_client import EnterpriseDataMCPClient
from decision_agent.workflows.data_agent import (
    DataAgentStatus,
    run_data_agent,
)

pytestmark = pytest.mark.integration


class QuestionSetPlanner:
    """Test-only adapter based on versioned verification SQL, never expected results."""

    def __init__(self, questions: list[OperationsQuestion]) -> None:
        self._sql_by_question = {
            question.question: question.verification_sql for question in questions
        }

    async def plan(
        self,
        *,
        user_query: str,
        enterprise_schema: dict[str, list[str]],
        business_definitions: dict[str, str],
    ) -> DataQueryPlan:
        assert enterprise_schema
        assert business_definitions
        return DataQueryPlan(
            status="ready",
            intent="\u7ecf\u8425\u6570\u636e\u67e5\u8be2",
            sql=self._sql_by_question[user_query],
            decision_reason="\u5df2\u6388\u6743\u7684\u7ecf\u8425\u6570\u636e\u53ef\u4ee5\u56de\u7b54\u8be5\u95ee\u9898\u3002",
        )


class UnsafePlanner:
    """Test-only planner used to prove a write never becomes data evidence."""

    async def plan(
        self,
        *,
        user_query: str,
        enterprise_schema: dict[str, list[str]],
        business_definitions: dict[str, str],
    ) -> DataQueryPlan:
        assert enterprise_schema
        assert business_definitions
        return DataQueryPlan(
            status="ready",
            intent="unsafe test",
            sql="DELETE FROM products",
            decision_reason="This test must be rejected by the remote SQLGuard.",
        )


class EvidenceBoundGenerator:
    """Test-only deterministic rendering that reads only successful D1 evidence."""

    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, *, user_query: str, data_evidence):  # type: ignore[no-untyped-def]
        self.calls += 1
        evidence = data_evidence[0]
        return DataAnswerDraft(
            answer=f"The query returned {evidence.row_count} rows.[{evidence.evidence_id}]",
            citations=[f"[{evidence.evidence_id}]"],
        )


def _configured_mcp_client_factory():  # type: ignore[no-untyped-def]
    if os.getenv("RUN_MYSQL_INTEGRATION") != "1":
        pytest.skip("set RUN_MYSQL_INTEGRATION=1 after starting docker compose mysql")
    settings = Settings()
    if settings.db_readonly_password is None:
        pytest.skip("configure DECISION_AGENT_DB_READONLY_PASSWORD in ignored .env")
    return lambda: EnterpriseDataMCPClient(timeout_seconds=settings.db_query_timeout_seconds + 2)


@pytest.mark.asyncio
async def test_eight_formal_questions_match_ground_truth_through_mcp_data_agent() -> None:
    questions = load_operations_questions()
    client_factory = _configured_mcp_client_factory()
    generator = EvidenceBoundGenerator()
    planner = QuestionSetPlanner(questions)

    outcomes = []
    for question in questions:
        state = await run_data_agent(
            query=question.question,
            planner=planner,
            enterprise_data_client_factory=client_factory,
            answer_generator=generator,
        )
        assert state.status is DataAgentStatus.ANSWERABLE_FINAL
        assert state.citations == ["[D1]"]
        assert state.data_evidence[0].normalized_sql
        actual = {
            "columns": state.data_evidence[0].columns,
            "rows": state.data_evidence[0].rows,
        }
        outcomes.append((question.question_id, actual == question.expected_result))

    assert generator.calls == len(questions)
    assert [question_id for question_id, passed in outcomes if not passed] == []


@pytest.mark.asyncio
async def test_write_sql_is_rejected_by_remote_sqlguard_without_evidence() -> None:
    client_factory = _configured_mcp_client_factory()
    generator = EvidenceBoundGenerator()
    state = await run_data_agent(
        query="unsafe test",
        planner=UnsafePlanner(),
        enterprise_data_client_factory=client_factory,
        answer_generator=generator,
    )

    assert state.status is DataAgentStatus.FAILED
    assert state.data_evidence == ()
    assert state.citations == []
    assert generator.calls == 0
    assert state.errors[0].code == "write_statement_not_allowed"
