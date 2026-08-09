"""Run the enterprise child-to-parent retrieval pipeline without answer generation."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path

from decision_agent.retrieval.factory import build_enterprise_retrieval_pipeline

ROOT = Path(__file__).resolve().parents[1]


def _nonblank_query(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("query cannot be empty or whitespace")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query", required=True, type=_nonblank_query, help="需要检索的企业知识问题"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "datasets/enterprise_kb/m2c1",
        help="包含 generated/ 的企业知识库数据目录",
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    pipeline = build_enterprise_retrieval_pipeline(args.dataset_root)
    try:
        initialization = await pipeline.initialize()
        result = await pipeline.retrieve(args.query)
    finally:
        await pipeline.close()

    print("=== 初始化耗时（秒） ===")  # noqa: RUF001
    for name, value in initialization.model_dump().items():
        print(f"{name}={value:.6f}")
    for label, candidates in (
        ("Dense", result.dense_results),
        ("BM25", result.bm25_results),
        ("RRF", result.fused_results),
        ("Reranker", result.reranked_child_results),
        ("Parent", result.expanded_parent_results),
    ):
        print(f"=== {label} ===")
        for candidate in candidates:
            payload = candidate.model_dump(mode="json")
            identity = payload.get("candidate_id") or payload.get("parent_id")
            score = next(
                (
                    payload[name]
                    for name in ("score", "fused_score", "reranker_score")
                    if payload.get(name) is not None
                ),
                None,
            )
            print(
                f"rank={payload.get('rank') or payload.get('final_rank')} "
                f"id={identity} document_id={payload.get('document_id')} score={score}"
            )
    print("=== Evidence Context ===")
    print(result.evidence_context.rendered_context)
    print("=== Query 阶段耗时（秒） ===")  # noqa: RUF001
    for name, value in result.stage_timings.model_dump().items():
        print(f"{name}={value:.6f}")
    print("当前只返回检索证据，不生成 LLM 答案")  # noqa: RUF001
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(_run(build_parser().parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
