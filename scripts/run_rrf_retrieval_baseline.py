"""Fuse the versioned Dense and BM25 reports with fixed RRF parameters."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from decision_agent.evaluation.rrf_baseline import RRFBaselineConfig, run_rrf_baseline

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    """Build the fully offline M2B-4 command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dense-report",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts/evaluation/m2b2c_dense_bge_small_zh_v1_5.json",
    )
    parser.add_argument(
        "--bm25-report",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts/evaluation/m2b3_bm25_baseline.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts/evaluation/m2b4_rrf_dense_bm25_k60.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run fixed equal-weight RRF and print measured metrics and timings."""
    args = build_parser().parse_args(argv)
    report = run_rrf_baseline(
        RRFBaselineConfig(
            dense_report_path=args.dense_report,
            bm25_report_path=args.bm25_report,
            output_path=args.output,
        )
    )
    print(f"report_filename={args.output.name}")
    for name, value in report.metrics.items():
        print(f"{name}={value:.6f}")
    print(f"fusion_seconds={report.runtime_observations.total_fusion_seconds:.6f}")
    print(f"average_query_ms={report.runtime_observations.average_query_ms:.6f}")
    print(f"improved_queries={','.join(report.comparison.improved_query_ids) or 'none'}")
    print(f"regressed_queries={','.join(report.comparison.regressed_query_ids) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
