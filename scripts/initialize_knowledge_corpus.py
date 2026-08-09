"""Initialize the synthetic public knowledge corpus through the formal ingestion runtime."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from decision_agent.config import Settings
from decision_agent.retrieval.factory import (
    EnterpriseRetrievalRuntime,
    build_production_retrieval_runtime,
)

SettingsFactory = Callable[[], Settings]
RuntimeFactory = Callable[[Settings], EnterpriseRetrievalRuntime]


def build_parser() -> argparse.ArgumentParser:
    """Build the narrow public ingestion command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help=(
            "Synthetic knowledge dataset root. Defaults to DECISION_AGENT_KNOWLEDGE_DATASET_ROOT."
        ),
    )
    return parser


async def _run(
    args: argparse.Namespace,
    *,
    settings_factory: SettingsFactory = Settings,
    runtime_factory: RuntimeFactory = build_production_retrieval_runtime,
) -> int:
    """Run exactly one formal ingestion and emit only a bounded status projection."""
    runtime: EnterpriseRetrievalRuntime | None = None
    payload: dict[str, object]
    exit_code = 2
    try:
        settings = settings_factory()
        dataset_root = args.dataset_root or settings.knowledge_dataset_root
        if dataset_root is None or not Path(dataset_root).is_dir():
            payload = {"status": "failed", "error_code": "knowledge_dataset_missing"}
        else:
            settings = settings.model_copy(update={"knowledge_dataset_root": Path(dataset_root)})
            runtime = runtime_factory(settings)
            await runtime.initialize_for_ingestion()
            ingestion = runtime.pipeline.last_ingestion_result
            child_count = runtime.pipeline.child_count
            if (
                ingestion is None
                or child_count <= 0
                or ingestion.attempted_count != child_count
                or ingestion.inserted_count + ingestion.updated_count != child_count
            ):
                payload = {
                    "status": "failed",
                    "error_code": "knowledge_corpus_verification_failed",
                }
            else:
                payload = {
                    "status": "initialized",
                    "parent_count": runtime.pipeline.parent_count,
                    "child_count": child_count,
                    "processed_count": ingestion.attempted_count,
                    "inserted_count": ingestion.inserted_count,
                    "updated_count": ingestion.updated_count,
                }
                exit_code = 0
    except Exception:
        payload = {
            "status": "failed",
            "error_code": (
                "configuration_invalid"
                if runtime is None
                else "knowledge_corpus_initialization_failed"
            ),
        }

    if runtime is not None:
        try:
            await runtime.aclose()
        except Exception:
            if exit_code == 0:
                payload = {"status": "failed", "error_code": "resource_cleanup_failed"}
                exit_code = 2

    stream = sys.stdout if exit_code == 0 else sys.stderr
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), file=stream)
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and preserve a nonzero exit code for every failure."""
    return asyncio.run(_run(build_parser().parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
