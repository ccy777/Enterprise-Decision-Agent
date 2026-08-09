"""Opt-in one-pass real-LLM smoke for routing only; no agents or tools are invoked."""

# ruff: noqa: RUF001

from __future__ import annotations

import os

import pytest

from decision_agent.config import Settings
from decision_agent.routing.models import RequestRoute
from decision_agent.routing.request_router import OpenAICompatibleRequestRouter

pytestmark = pytest.mark.integration

_QUERY = "公司采购审批流程是什么？"
_EXPECTED_ROUTE = RequestRoute.KNOWLEDGE


@pytest.mark.asyncio
async def test_real_llm_routes_one_knowledge_request() -> None:
    """Call only the configured LLM router once; no downstream work is invoked."""
    if os.getenv("RUN_UNIFIED_REQUEST_ROUTER_REAL_LLM_SMOKE") != "1":
        pytest.skip("set RUN_UNIFIED_REQUEST_ROUTER_REAL_LLM_SMOKE=1 to run the real router smoke")
    settings = Settings()
    if (
        settings.llm_api_key is None
        or settings.llm_base_url is None
        or settings.llm_model_name is None
    ):
        pytest.skip("LLM settings are not fully configured")

    router = OpenAICompatibleRequestRouter.from_settings(settings)
    decision = await router.route(user_query=_QUERY)

    assert decision.route is _EXPECTED_ROUTE
    assert decision.knowledge_subquery is not None and decision.data_subquery is None
