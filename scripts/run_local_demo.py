"""Run one explicit local demo through the formal configured Runtime."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from decision_agent.config import Settings
from decision_agent.demo import DemoCase, prepare_demo_settings, run_demo


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run one fixed localhost/CLI demo. This may call the configured Provider, Milvus, "
            "MCP and read-only MySQL and may incur Provider cost."
        )
    )
    parser.add_argument("case", choices=[case.value for case in DemoCase])
    arguments = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    try:
        settings = prepare_demo_settings(Settings(), repository_root=repository_root)
        response = asyncio.run(run_demo(DemoCase(arguments.case), settings=settings))
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        code = getattr(exc, "code", "local_demo_failed")
        print(json.dumps({"status": "failed", "error_code": code}), file=sys.stderr)
        return 1
    result = response.result
    print(
        json.dumps(
            {
                "request_id": response.request_id,
                "status": result.status.value,
                "route": result.route.value if result.route is not None else None,
                "skill": result.skill_name,
                "answer": result.answer,
                "citations": result.citations,
                "error_code": result.error_code,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.status.value == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
