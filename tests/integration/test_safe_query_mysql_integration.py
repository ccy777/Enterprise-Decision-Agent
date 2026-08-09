"""Opt-in verification against the local Docker MySQL development database."""

from __future__ import annotations

import os

import pytest
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from decision_agent.config import Settings
from decision_agent.data.executor import SQLAlchemyQueryExecutor
from decision_agent.data.ground_truth import (
    load_operations_questions,
    verify_operations_ground_truth,
)
from decision_agent.data.models import SafeQueryRequest
from decision_agent.data.safe_query_service import SafeQueryService

pytestmark = pytest.mark.integration


@pytest.fixture
def mysql_service() -> SafeQueryService:
    if os.getenv("RUN_MYSQL_INTEGRATION") != "1":
        pytest.skip("set RUN_MYSQL_INTEGRATION=1 after starting docker compose mysql")
    settings = Settings()
    if settings.db_readonly_password is None:
        pytest.skip("configure DECISION_AGENT_DB_READONLY_PASSWORD in ignored .env")
    return SafeQueryService.from_settings(settings)


@pytest.mark.asyncio
async def test_readonly_account_can_query_seed_business_data(
    mysql_service: SafeQueryService,
) -> None:
    result = await mysql_service.execute(
        SafeQueryRequest(sql="SELECT COUNT(*) AS product_count FROM products")
    )
    assert result.error_code is None
    assert result.rows == [[8]]


@pytest.mark.asyncio
async def test_per_product_latest_inventory_does_not_omit_an_older_snapshot(
    mysql_service: SafeQueryService,
) -> None:
    result = await mysql_service.execute(
        SafeQueryRequest(
            sql="WITH latest_inventory AS (SELECT s.product_id, s.snapshot_date "
            "FROM inventory_snapshots AS s JOIN (SELECT product_id, MAX(snapshot_date) "
            "AS snapshot_date FROM inventory_snapshots GROUP BY product_id) AS latest "
            "ON s.product_id = latest.product_id AND s.snapshot_date = latest.snapshot_date) "
            "SELECT product_id, snapshot_date FROM latest_inventory WHERE product_id = 'P500'"
        )
    )
    assert result.error_code is None
    assert result.rows == [["P500", "2026-05-31"]]


@pytest.mark.asyncio
async def test_formal_questions_match_seed_ground_truth(mysql_service: SafeQueryService) -> None:
    outcomes = await verify_operations_ground_truth(
        service=mysql_service,
        questions=load_operations_questions(),
    )
    assert [question_id for question_id, passed, _ in outcomes if not passed] == []


@pytest.mark.parametrize(
    "statement",
    [
        "INSERT INTO products(product_id) VALUES ('P999')",
        "UPDATE products SET product_name = 'forbidden' WHERE product_id = 'P100'",
        "DELETE FROM products WHERE product_id = 'P100'",
    ],
)
def test_database_account_rejects_write_at_permission_layer(statement: str) -> None:
    if os.getenv("RUN_MYSQL_INTEGRATION") != "1":
        pytest.skip("set RUN_MYSQL_INTEGRATION=1 after starting docker compose mysql")
    settings = Settings()
    if settings.db_readonly_password is None:
        pytest.skip("configure DECISION_AGENT_DB_READONLY_PASSWORD in ignored .env")
    executor = SQLAlchemyQueryExecutor.from_settings(settings)
    with pytest.raises(SQLAlchemyError), executor.engine.connect() as connection:
        connection.execute(text(statement))


@pytest.mark.asyncio
async def test_safe_error_result_does_not_expose_connection_details(
    mysql_service: SafeQueryService,
) -> None:
    result = await mysql_service.execute(SafeQueryRequest(sql="DELETE FROM products"))
    serialized = result.model_dump_json()
    assert result.error_code == "write_statement_not_allowed"
    assert "mysql+pymysql" not in serialized.lower()
    assert "password" not in serialized.lower()
