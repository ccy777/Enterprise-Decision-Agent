"""Build deterministic M2C-1 Parent/Child retrieval ground truth."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from decision_agent.evaluation.enterprise_kb_ground_truth import (
    build_and_write_enterprise_kb_ground_truth,
)
from decision_agent.exceptions import EvaluationError

ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "datasets/enterprise_kb/m2c1",
        help="M2C-1 enterprise KB root (defaults to the repository dataset)",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=ROOT / "artifacts/datasets/m2c1_parent_child_summary.json",
        help="deterministic summary report path",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        build = build_and_write_enterprise_kb_ground_truth(
            args.dataset_root,
            summary_output=args.summary_output,
        )
    except EvaluationError as exc:
        print(f"ground-truth build failed: {exc}", file=sys.stderr)
        return 1

    print(f"documents={build.summary['document_count']}")
    print(f"parent_chunks={len(build.parent_chunks)}")
    print(f"child_chunks={len(build.child_chunks)}")
    print(f"clauses={len(build.clause_chunk_map)}")
    print(f"queries={len(build.retrieval_ground_truth)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
