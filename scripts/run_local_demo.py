"""Run one explicit local demo through the formal configured Runtime."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from decision_agent.config import Settings
from decision_agent.demo import DemoCase, prepare_demo_settings, run_demo

_TRACE_STATUS_LABELS = {
    "completed": "success",
    "failed": "failed",
    "cancelled": "cancelled",
    "skipped": "skipped",
    "unsupported": "unsupported",
    "not_requested": "not_requested",
}
_TRACE_ATTRIBUTE_LABELS = {
    "tool_name": "tool",
    "row_count": "row_count",
    "retrieved_count": "retrieval_count",
    "reranked_count": "rerank_count",
    "review_outcome": "reviewer_outcome",
}


def _configure_cli_output() -> None:
    """Keep Demo JSON readable when Windows stdout is redirected or captured."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="backslashreplace")


def _format_trace_summary(
    trace: object,
    *,
    route: str | None,
    skill_name: str | None,
) -> str:
    """Render only the approved, payload-free fields from a TraceSummary."""
    if trace is None:
        return "Trace Summary:\n\nunavailable"

    lines = ["Trace Summary:"]
    duration_ms = getattr(trace, "duration_ms", None)
    if route is not None:
        lines.append(f"route: {route}")
    if skill_name is not None:
        lines.append(f"skill_name: {skill_name}")
    if isinstance(duration_ms, (int, float)):
        lines.append(f"duration: {duration_ms:.0f} ms")

    stages = getattr(trace, "stages", ())
    for index, stage in enumerate(stages, start=1):
        stage_name = getattr(stage, "stage", "unknown")
        status = getattr(stage, "status", "unknown")
        display_status = _TRACE_STATUS_LABELS.get(str(status), str(status))
        lines.extend(("", f"{index}. {stage_name}", f"   status: {display_status}"))
        for attribute in getattr(stage, "attributes", ()):
            label = _TRACE_ATTRIBUTE_LABELS.get(getattr(attribute, "key", ""))
            if label is not None:
                lines.append(f"   {label}: {getattr(attribute, 'value', None)}")
    return "\n".join(lines)


def main() -> int:
    _configure_cli_output()
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
    print()
    print(
        _format_trace_summary(
            response.trace,
            route=result.route.value if result.route is not None else None,
            skill_name=result.skill_name,
        )
    )
    return 0 if result.status.value == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
