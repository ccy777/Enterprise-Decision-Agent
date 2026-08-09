"""Unit contracts for bounded, non-executing routing decisions."""

# ruff: noqa: RUF001

from __future__ import annotations

import pytest
from pydantic import ValidationError

from decision_agent.routing.models import RequestRoute, RouterDecision


@pytest.mark.parametrize(
    ("route", "knowledge_subquery", "data_subquery"),
    [
        (RequestRoute.KNOWLEDGE, "查询采购审批制度", None),
        (RequestRoute.DATA, None, "查询五月销售额"),
        (RequestRoute.MIXED, "查询补货制度", "查询库存不足产品"),
        (RequestRoute.UNSUPPORTED, None, None),
    ],
)
def test_router_decision_accepts_only_the_required_subquery_combination(
    route: RequestRoute,
    knowledge_subquery: str | None,
    data_subquery: str | None,
) -> None:
    decision = RouterDecision(
        route=route,
        normalized_query="企业问题",
        decision_reason="路由依据。",
        knowledge_subquery=knowledge_subquery,
        data_subquery=data_subquery,
        missing_information=None,
        confidence=0.8,
    )
    assert decision.route is route


@pytest.mark.parametrize(
    "overrides",
    [
        {"route": "unknown"},
        {"route": "knowledge", "knowledge_subquery": None},
        {"route": "knowledge", "data_subquery": "不应存在"},
        {"route": "data", "data_subquery": None, "knowledge_subquery": "不应存在"},
        {"route": "mixed", "data_subquery": None},
        {"route": "unsupported", "knowledge_subquery": "不能执行"},
        {"route": "unsupported", "sql": "DELETE FROM products"},
        {"confidence": -0.01},
        {"confidence": 1.01},
        {"decision_reason": "   "},
    ],
)
def test_router_decision_rejects_invalid_or_executable_contract_fields(
    overrides: dict[str, object],
) -> None:
    payload: dict[str, object] = {
        "route": "knowledge",
        "normalized_query": "采购审批流程是什么？",
        "decision_reason": "问题询问制度。",
        "knowledge_subquery": "公司采购审批流程是什么？",
        "data_subquery": None,
        "missing_information": None,
        "confidence": 0.8,
    }
    payload.update(overrides)
    with pytest.raises(ValidationError):
        RouterDecision.model_validate(payload)
