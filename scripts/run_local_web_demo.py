"""Serve the local Web workspace with a localhost-only, fixed-scope identity adapter."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from decision_agent.config import Settings
from decision_agent.demo import DemoCase
from decision_agent.demo.web import create_local_demo_app


def main() -> int:
    parser = argparse.ArgumentParser(
        description=("Run the Agent Web workspace on 127.0.0.1 with the frozen local scopes.")
    )
    parser.add_argument("case", choices=[case.value for case in DemoCase])
    parser.add_argument("--port", type=int, default=8000)
    arguments = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    app = create_local_demo_app(
        case=DemoCase(arguments.case),
        settings=Settings(),
        repository_root=repository_root,
    )
    uvicorn.run(app, host="127.0.0.1", port=arguments.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
