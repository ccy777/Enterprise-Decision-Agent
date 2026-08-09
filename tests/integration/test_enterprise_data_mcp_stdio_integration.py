"""Opt-in real stdio MCP verification against the local read-only MySQL service."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from decision_agent.data.models import SAFE_QUERY_SQL_MAX_LENGTH, SAFE_QUERY_SQL_MIN_LENGTH

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.asyncio
async def test_real_stdio_mcp_server_lists_and_calls_only_enterprise_data_tools() -> None:
    if os.getenv("RUN_MYSQL_INTEGRATION") != "1":
        pytest.skip("set RUN_MYSQL_INTEGRATION=1 after starting docker compose mysql")

    server = StdioServerParameters(
        command=sys.executable,
        args=[str(ROOT / "scripts" / "run_enterprise_data_mcp_server.py")],
        cwd=str(ROOT),
    )
    async with stdio_client(server) as (reader, writer), ClientSession(reader, writer) as client:
        await client.initialize()
        tools = await client.list_tools()
        tool_names = {tool.name for tool in tools.tools}
        assert tool_names == {
            "get_enterprise_schema",
            "get_business_definitions",
            "execute_safe_query",
        }
        execute_schema = next(
            tool.inputSchema for tool in tools.tools if tool.name == "execute_safe_query"
        )
        assert execute_schema["required"] == ["sql"]
        assert execute_schema["properties"]["sql"]["type"] == "string"
        assert execute_schema["properties"]["sql"]["minLength"] == SAFE_QUERY_SQL_MIN_LENGTH
        assert execute_schema["properties"]["sql"]["maxLength"] == SAFE_QUERY_SQL_MAX_LENGTH

        schema = await client.call_tool("get_enterprise_schema", {})
        assert schema.isError is False
        schema_payload = _payload(schema)
        assert "products" in schema_payload
        assert "product_name" in schema_payload["products"]

        definitions = await client.call_tool("get_business_definitions", {})
        assert definitions.isError is False
        assert "current_inventory" in _payload(definitions)

        allowed = await client.call_tool(
            "execute_safe_query",
            {"sql": "SELECT product_id FROM products ORDER BY product_id LIMIT 1"},
        )
        assert allowed.isError is False
        allowed_payload = _payload(allowed)
        assert allowed_payload["error_code"] is None
        assert allowed_payload["columns"] == ["product_id"]
        assert allowed_payload["rows"] == [["P100"]]
        assert allowed_payload["accessed_tables"] == ["products"]

        rejected = await client.call_tool("execute_safe_query", {"sql": "DELETE FROM products"})
        assert rejected.isError is False
        rejected_payload = _payload(rejected)
        assert rejected_payload["error_code"] == "write_statement_not_allowed"
        assert rejected_payload["normalized_sql"] is None
        assert rejected_payload["accessed_tables"] == []
        serialized = json.dumps(rejected_payload).lower()
        assert "password" not in serialized
        assert "mysql+pymysql" not in serialized
        assert "traceback" not in serialized


def _payload(result: object) -> object:
    content = result.content  # type: ignore[attr-defined]
    assert len(content) == 1
    return json.loads(content[0].text)
