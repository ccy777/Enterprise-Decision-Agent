"""Recompute the versioned operations question set through SafeQueryService."""

from __future__ import annotations

import asyncio
import json
import sys

from decision_agent.config import Settings
from decision_agent.data.ground_truth import (
    load_operations_questions,
    verify_operations_ground_truth,
)
from decision_agent.data.safe_query_service import SafeQueryService
from decision_agent.exceptions import ConfigurationError


async def _run() -> int:
    try:
        service = SafeQueryService.from_settings(Settings())
    except ConfigurationError:
        print(json.dumps({"error_code": "database_unavailable"}, ensure_ascii=False))
        return 2
    outcomes = await verify_operations_ground_truth(
        service=service,
        questions=load_operations_questions(),
    )
    failed = False
    for question_id, passed, actual in outcomes:
        print(
            json.dumps(
                {"question_id": question_id, "passed": passed, "actual": actual}, ensure_ascii=False
            )
        )
        failed = failed or not passed
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_run()))
