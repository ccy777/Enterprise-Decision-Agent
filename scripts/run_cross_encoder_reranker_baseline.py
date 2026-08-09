"""Run the fixed BAAI CrossEncoder reranker baseline over the versioned RRF report."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path

from decision_agent.evaluation.reranker_baseline import (
    RerankerBaselineConfig,
    run_reranker_baseline,
)
from decision_agent.retrieval.reranking import SentenceTransformerCrossEncoderReranker

ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rrf-report",
        type=Path,
        default=ROOT / "artifacts/evaluation/m2b4_rrf_dense_bm25_k60.json",
    )
    parser.add_argument(
        "--corpus", type=Path, default=ROOT / "datasets/retrieval/m2b2c_dense_corpus.jsonl"
    )
    parser.add_argument(
        "--queries", type=Path, default=ROOT / "datasets/retrieval/m2b2c_dense_queries.jsonl"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/evaluation/m2b5_bge_reranker_base.json"
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    config = RerankerBaselineConfig(
        rrf_report_path=args.rrf_report,
        corpus_path=args.corpus,
        queries_path=args.queries,
        output_path=args.output,
    )
    report = await run_reranker_baseline(config, SentenceTransformerCrossEncoderReranker())
    print(f"report_filename={args.output.name}")
    for name, value in report.metrics.items():
        print(f"{name}={value:.6f}")
    print(f"model_load_seconds={report.runtime_observations.model_load_seconds:.6f}")
    print(f"total_rerank_seconds={report.runtime_observations.total_rerank_seconds:.6f}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(_run(build_parser().parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
