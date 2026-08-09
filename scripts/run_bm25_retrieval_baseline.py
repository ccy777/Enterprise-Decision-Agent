"""Run the deterministic M2B-3 BM25 sparse retrieval baseline."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from decision_agent.evaluation import BM25BaselineConfig, run_bm25_baseline

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    """Build the offline BM25 baseline command-line interface."""
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
        "--dense-report",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts/evaluation/m2b2c_dense_bge_small_zh_v1_5.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts/evaluation/m2b3_bm25_baseline.json",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the real BM25 chain and print measured metrics and timings."""
    args = build_parser().parse_args(argv)
    report = run_bm25_baseline(
        BM25BaselineConfig(
            corpus_path=args.corpus,
            queries_path=args.queries,
            dense_report_path=args.dense_report,
            output_path=args.output,
            top_k=args.top_k,
            k1=args.k1,
            b=args.b,
        )
    )
    print(f"report_filename={args.output.name}")
    for name, value in report.metrics.items():
        print(f"{name}={value:.6f}")
    print(
        "runtime_seconds="
        f"index:{report.runtime_observations.index_build_seconds:.6f},"
        f"queries:{report.runtime_observations.total_query_seconds:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
