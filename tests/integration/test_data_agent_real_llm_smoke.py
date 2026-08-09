"""Opt-in Level 3: one real LLM query and one unsupported query through MCP."""

from __future__ import annotations

import os
from collections.abc import Sequence

import pytest

from decision_agent.agents.data_answer_generator import (
    DataAnswerDraft,
    OpenAICompatibleDataAnswerGenerator,
)
from decision_agent.agents.data_query_planner import OpenAICompatibleDataQueryPlanner
from decision_agent.config import Settings
from decision_agent.data_agent.models import DataEvidence
from decision_agent.mcp_client import EnterpriseDataMCPClient
from decision_agent.mcp_client.contracts import (
    BusinessDefinitions,
    EnterpriseSchema,
    MCPQueryResult,
)
from decision_agent.workflows.data_agent import (
    DataAgentStatus,
    run_data_agent,
)

pytestmark = pytest.mark.integration

_QUERYABLE_QUESTION = (
    "2026\u5e745\u6708\u9500\u552e\u989d\u6700\u9ad8\u7684\u4ea7\u54c1\u662f\u4ec0\u4e48\uff1f"
)
_UNSUPPORTED_QUESTION = (
    "\u4f9b\u5e94\u5546 S300 \u5ef6\u671f\u4ea4\u4ed8\u7684\u5177\u4f53\u539f\u56e0"
    "\u662f\u4ec0\u4e48\uff1f"
)


class CountingEnterpriseDataMCPClient:
    """Test-only counter proving the workflow reaches the real MCP client."""

    def __init__(self, client: EnterpriseDataMCPClient) -> None:
        self._client = client
        self.execute_calls = 0

    async def __aenter__(self) -> CountingEnterpriseDataMCPClient:
        await self._client.__aenter__()
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self._client.__aexit__(exc_type, exc, traceback)

    async def get_enterprise_schema(self) -> EnterpriseSchema:
        return await self._client.get_enterprise_schema()

    async def get_business_definitions(self) -> BusinessDefinitions:
        return await self._client.get_business_definitions()

    async def execute_safe_query(self, sql: str) -> MCPQueryResult:
        self.execute_calls += 1
        return await self._client.execute_safe_query(sql)


class CountingEnterpriseDataMCPClientFactory:
    """Create an independent real MCP client for each smoke request."""

    def __init__(self, timeout_seconds: float) -> None:
        self._timeout_seconds = timeout_seconds
        self.clients: list[CountingEnterpriseDataMCPClient] = []

    def __call__(self) -> CountingEnterpriseDataMCPClient:
        client = CountingEnterpriseDataMCPClient(
            EnterpriseDataMCPClient(timeout_seconds=self._timeout_seconds)
        )
        self.clients.append(client)
        return client

    @property
    def execute_calls(self) -> int:
        return sum(client.execute_calls for client in self.clients)


class CountingAnswerGenerator:
    """Test-only counter around the real generator."""

    def __init__(self, generator: OpenAICompatibleDataAnswerGenerator) -> None:
        self._generator = generator
        self.calls = 0

    async def generate(
        self, *, user_query: str, data_evidence: Sequence[DataEvidence]
    ) -> DataAnswerDraft:
        self.calls += 1
        return await self._generator.generate(
            user_query=user_query,
            data_evidence=data_evidence,
        )


def _configured_settings() -> Settings:
    if os.getenv("RUN_DATA_AGENT_REAL_LLM_SMOKE") != "1":
        pytest.skip("set RUN_DATA_AGENT_REAL_LLM_SMOKE=1 to run the real Data Agent Level 3 smoke")
    settings = Settings()
    if (
        settings.llm_api_key is None
        or settings.llm_base_url is None
        or settings.llm_model_name is None
        or settings.db_readonly_password is None
    ):
        pytest.skip("configure ignored LLM and MySQL settings for the real smoke")
    return settings


@pytest.mark.asyncio
async def test_real_llm_minimal_data_agent_mcp_smoke() -> None:
    settings = _configured_settings()
    client_factory = CountingEnterpriseDataMCPClientFactory(
        timeout_seconds=settings.db_query_timeout_seconds + 2
    )
    answer_generator = CountingAnswerGenerator(
        OpenAICompatibleDataAnswerGenerator.from_settings(settings)
    )
    planner = OpenAICompatibleDataQueryPlanner.from_settings(settings)

    queryable = await run_data_agent(
        query=_QUERYABLE_QUESTION,
        planner=planner,
        enterprise_data_client_factory=client_factory,
        answer_generator=answer_generator,
    )
    assert queryable.status is DataAgentStatus.ANSWERABLE_FINAL
    assert queryable.citations == ["[D1]"]
    assert queryable.data_evidence[0].evidence_id == "D1"
    assert client_factory.execute_calls == 1
    assert answer_generator.calls == 1

    unsupported = await run_data_agent(
        query=_UNSUPPORTED_QUESTION,
        planner=planner,
        enterprise_data_client_factory=client_factory,
        answer_generator=answer_generator,
    )
    assert unsupported.status is DataAgentStatus.UNSUPPORTED
    assert unsupported.planned_sql is None
    assert unsupported.data_evidence == ()
    assert unsupported.citations == []
    assert client_factory.execute_calls == 1
    assert answer_generator.calls == 1
