"""Run one bounded Data Agent request against the local read-only operations database."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence

from decision_agent.agents.data_answer_generator import OpenAICompatibleDataAnswerGenerator
from decision_agent.agents.data_query_planner import OpenAICompatibleDataQueryPlanner
from decision_agent.config import Settings
from decision_agent.exceptions import ConfigurationError
from decision_agent.mcp_client import EnterpriseDataMCPClient
from decision_agent.workflows.data_agent import (
    DataAgentState,
    DataAgentStatus,
    run_data_agent,
)


def _nonblank_query(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("query cannot be empty or whitespace")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Build the small public command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query", required=True, type=_nonblank_query, help="Enterprise operations question"
    )
    return parser


def project_public_result(state: DataAgentState) -> dict[str, object]:
    """Project state to audited public fields without provider or connection internals."""
    result: dict[str, object] = {
        "status": state.status,
        "intent": state.intent,
        "answer": state.answer,
        "citations": state.citations,
        "missing_information": state.missing_information,
        "decision_reason": state.decision_reason,
        "data_evidence": [item.model_dump() for item in state.data_evidence],
    }
    if state.status is DataAgentStatus.FAILED:
        result["errors"] = [
            {"code": error.code, "retryable": error.retryable} for error in state.errors
        ]
    return result


async def _run(args: argparse.Namespace) -> int:
    try:
        settings = Settings()
        state = await run_data_agent(
            query=args.query,
            planner=OpenAICompatibleDataQueryPlanner.from_settings(settings),
            enterprise_data_client_factory=lambda: EnterpriseDataMCPClient(
                timeout_seconds=settings.db_query_timeout_seconds + 2.0
            ),
            answer_generator=OpenAICompatibleDataAnswerGenerator.from_settings(settings),
        )
    except ConfigurationError:
        print(json.dumps({"status": "failed", "errors": [{"code": "configuration_unavailable"}]}))
        return 2
    except Exception:
        print(json.dumps({"status": "failed", "errors": [{"code": "data_agent_runtime_failed"}]}))
        return 2
    print(json.dumps(project_public_result(state), ensure_ascii=False, default=str, indent=2))
    return 0 if state.status is not DataAgentStatus.FAILED else 2


def main(argv: Sequence[str] | None = None) -> int:
    """Parse one question and print only the safe public result."""
    return asyncio.run(_run(build_parser().parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
