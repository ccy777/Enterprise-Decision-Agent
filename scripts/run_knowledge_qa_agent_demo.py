"""Run the minimal real-retrieval LangGraph knowledge-QA workflow once."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

from decision_agent.agents.answerability_reviewer import OpenAICompatibleAnswerabilityReviewer
from decision_agent.agents.evidence_selector import OpenAICompatibleEvidenceSelector
from decision_agent.agents.grounded_answer import OpenAICompatibleAnswerGenerator
from decision_agent.config import Settings
from decision_agent.retrieval.factory import build_enterprise_retrieval_pipeline
from decision_agent.workflows.knowledge_qa import (
    Answerability,
    KnowledgeQAState,
    build_knowledge_qa_graph,
    run_knowledge_qa,
)

ROOT = Path(__file__).resolve().parents[1]


def _nonblank_query(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("query cannot be empty or whitespace")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, type=_nonblank_query, help="企业知识问题")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "datasets/enterprise_kb/m2c1",
        help="包含 generated/ 的企业知识库数据目录",
    )
    return parser


def project_public_result(state: KnowledgeQAState) -> dict[str, object]:
    """Expose only the validated answer contract and safe failure summaries."""
    result: dict[str, object] = {
        "answerability": state.answerability,
        "answer": state.answer,
        "citations": state.citations,
        "missing_information": state.missing_information,
        "decision_reason": state.decision_reason,
    }
    if state.answerability is Answerability.FAILED:
        result["errors"] = [
            {"code": error.code, "message": error.message, "retryable": error.retryable}
            for error in state.errors
        ]
    return result


async def _run(args: argparse.Namespace) -> int:
    pipeline = build_enterprise_retrieval_pipeline(args.dataset_root)
    try:
        await pipeline.initialize()
        settings = Settings()
        selector = OpenAICompatibleEvidenceSelector.from_settings(settings)
        reviewer = OpenAICompatibleAnswerabilityReviewer.from_settings(settings)
        generator = OpenAICompatibleAnswerGenerator.from_settings(settings)
        graph = build_knowledge_qa_graph(
            retrieval_pipeline=pipeline,
            evidence_selector=selector,
            answerability_reviewer=reviewer,
            answer_generator=generator,
        )
        result = await run_knowledge_qa(graph, user_query=args.query)
    finally:
        await pipeline.close()
    print(
        json.dumps(
            project_public_result(result),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(_run(build_parser().parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
