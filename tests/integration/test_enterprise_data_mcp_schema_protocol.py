"""Opt-in official-Client checks for the public Enterprise Data MCP schemas."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import McpError

from decision_agent.data.models import SAFE_QUERY_SQL_MAX_LENGTH, SAFE_QUERY_SQL_MIN_LENGTH

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.asyncio
async def test_official_client_exposes_and_enforces_execute_safe_query_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "safe-query-service-called.txt"
    monkeypatch.setenv("MCP_TEST_CALL_COUNT_PATH", str(marker))
    server = StdioServerParameters(
        command=sys.executable,
        args=[str(ROOT / "tests" / "fixtures" / "enterprise_data_mcp_stub_server.py")],
        cwd=str(ROOT),
        env=dict(os.environ),
    )
    async with stdio_client(server) as (reader, writer), ClientSession(reader, writer) as client:
        await client.initialize()
        tools = await client.list_tools()
        tool = next(item for item in tools.tools if item.name == "execute_safe_query")
        sql_schema = tool.inputSchema["properties"]["sql"]
        assert sql_schema["type"] == "string"
        assert sql_schema["minLength"] == SAFE_QUERY_SQL_MIN_LENGTH
        assert sql_schema["maxLength"] == SAFE_QUERY_SQL_MAX_LENGTH
        assert "sql" in tool.inputSchema["required"]
        assert tool.outputSchema["additionalProperties"] is False
        assert {
            "columns",
            "rows",
            "row_count",
            "truncated",
            "normalized_sql",
            "accessed_tables",
            "elapsed_ms",
            "error_code",
        } <= set(tool.outputSchema["properties"])
        assert set(tool.outputSchema["required"]) == {"row_count", "truncated", "elapsed_ms"}

        for arguments in (
            {},
            {"sql": ""},
            {"sql": "x" * (SAFE_QUERY_SQL_MAX_LENGTH + 1)},
        ):
            message = await _protocol_error(client, arguments)
            assert "password" not in message.lower()
            assert "mysql+pymysql" not in message.lower()
            assert "traceback" not in message.lower()
        assert marker.exists() is False

        # FastMCP 1.28.1's generated argument model ignores unknown top-level keys.
        # The key cannot alter the declared SQL argument or create a second execution path.
        ignored_extra = await client.call_tool(
            "execute_safe_query",
            {"sql": "SELECT product_id FROM products", "unexpected": True},
        )
        assert ignored_extra.isError is False
        assert _payload(ignored_extra)["normalized_sql"] == (
            "SELECT product_id FROM products LIMIT 200"
        )
        assert marker.read_text(encoding="utf-8") == "1"

        result = await client.call_tool(
            "execute_safe_query", {"sql": "SELECT product_id FROM products"}
        )
        assert result.isError is False
        payload = _payload(result)
        assert payload == {
            "columns": ["product_id"],
            "rows": [["P100"]],
            "row_count": 1,
            "truncated": False,
            "normalized_sql": "SELECT product_id FROM products LIMIT 200",
            "accessed_tables": ["products"],
            "elapsed_ms": 1.0,
            "error_code": None,
        }
        assert marker.read_text(encoding="utf-8") == "2"


async def _protocol_error(client: ClientSession, arguments: dict[str, object]) -> str:
    try:
        result = await client.call_tool("execute_safe_query", arguments)
    except McpError as exc:
        return str(exc)
    assert result.isError is True
    return " ".join(item.text for item in result.content)


def _payload(result: object) -> object:
    content = result.content  # type: ignore[attr-defined]
    assert len(content) == 1
    return json.loads(content[0].text)
