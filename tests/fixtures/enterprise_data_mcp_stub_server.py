"""Test-only stdio server using a recording SafeQueryService substitute."""

from __future__ import annotations

import os
from pathlib import Path

from decision_agent.data.models import QueryAudit, SafeQueryResult
from decision_agent.mcp_server.enterprise_data_server import create_enterprise_data_server


class RecordingSafeQueryService:
    """Write one marker only when a protocol-valid call reaches the service boundary."""

    async def execute(self, request: object) -> SafeQueryResult:
        marker = Path(os.environ["MCP_TEST_CALL_COUNT_PATH"])
        call_count = int(marker.read_text(encoding="utf-8")) if marker.exists() else 0
        marker.write_text(str(call_count + 1), encoding="utf-8")
        return SafeQueryResult(
            columns=["product_id"],
            rows=[["P100"]],
            row_count=1,
            truncated=False,
            elapsed_ms=1.0,
            accessed_tables=["products"],
            audit=QueryAudit(
                request_id="00000000-0000-0000-0000-000000000001",
                normalized_sql="SELECT product_id FROM products LIMIT 200",
                allowed=True,
                accessed_tables=["products"],
                elapsed_ms=1.0,
                row_count=1,
            ),
        )


if __name__ == "__main__":
    create_enterprise_data_server(service_factory=RecordingSafeQueryService).run(transport="stdio")
