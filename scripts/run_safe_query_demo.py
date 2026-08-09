"""Run one guarded SQL query through the application read-only MySQL account."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from decision_agent.config import Settings
from decision_agent.data import SafeQueryRequest, SafeQueryService
from decision_agent.exceptions import ConfigurationError

_DEFAULT_SQL = "SELECT product_id, product_name FROM products ORDER BY product_id LIMIT 3"


async def _run(sql: str) -> int:
    try:
        service = SafeQueryService.from_settings(Settings())
    except ConfigurationError:
        print(json.dumps({"error_code": "database_unavailable"}, ensure_ascii=False))
        return 2
    result = await service.execute(SafeQueryRequest(sql=sql))
    print(
        json.dumps(
            {
                "columns": result.columns,
                "rows": result.rows,
                "row_count": result.row_count,
                "truncated": result.truncated,
                "elapsed_ms": result.elapsed_ms,
                "accessed_tables": result.accessed_tables,
                "error_code": result.error_code,
            },
            ensure_ascii=False,
            default=str,
        )
    )
    return 0 if result.error_code is None else 2


def main() -> int:
    """Parse a single SQL argument; SQLGuard remains the authority for safety."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sql", default=_DEFAULT_SQL, help="One read-only business SQL query.")
    args = parser.parse_args()
    return asyncio.run(_run(args.sql))


if __name__ == "__main__":
    sys.exit(main())
