"""Run the M2B-2C real BGE dense retrieval baseline."""

from __future__ import annotations

import argparse
import asyncio
import os
from collections.abc import Sequence
from pathlib import Path

from decision_agent.evaluation import DenseBaselineConfig, run_dense_baseline

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    """Build the minimal reproducible baseline command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=REPOSITORY_ROOT / "datasets/retrieval/m2b2c_dense_corpus.jsonl",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=REPOSITORY_ROOT / "datasets/retrieval/m2b2c_dense_queries.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts/evaluation/m2b2c_dense_bge_small_zh_v1_5.json",
    )
    parser.add_argument("--model-name", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument(
        "--cache-folder",
        type=Path,
        default=(
            Path(os.environ["DECISION_AGENT_EMBEDDING_CACHE_FOLDER"])
            if os.getenv("DECISION_AGENT_EMBEDDING_CACHE_FOLDER")
            else None
        ),
    )
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    parser.add_argument("--dimension", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, run the async canonical chain, and print measured metrics."""
    args = build_parser().parse_args(argv)
    config = DenseBaselineConfig(
        corpus_path=args.corpus,
        queries_path=args.queries,
        output_path=args.output,
        model_name=args.model_name,
        cache_folder=args.cache_folder,
        device=args.device,
        dimension=args.dimension,
        batch_size=args.batch_size,
        top_k=args.top_k,
        normalize_embeddings=True,
        local_files_only=args.local_files_only,
    )
    report = asyncio.run(run_dense_baseline(config))
    print(f"report_filename={args.output.name}")
    for name, value in report.metrics.items():
        print(f"{name}={value:.6f}")
    print(
        "runtime_seconds="
        f"load:{report.runtime_observations.model_load_seconds:.3f},"
        f"index:{report.runtime_observations.corpus_index_seconds:.3f},"
        f"queries:{report.runtime_observations.total_query_seconds:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
