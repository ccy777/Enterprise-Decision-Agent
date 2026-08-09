"""Run the fixed M2C-2A-2 real-model retrieval baseline strictly offline."""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

from decision_agent.evaluation.enterprise_retrieval_baseline import (
    BASELINE_ID,
    SCHEMA_VERSION,
    EvaluatedRun,
    assert_deterministic_runs,
    evaluate_retrieval_results,
    load_benchmark_query_inputs,
    load_retrieval_ground_truth,
    sha256_text,
    timing_summary,
    write_baseline_artifacts,
)

ROOT = Path(__file__).resolve().parents[1]
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"
WARMUP_QUERY = "企业制度流程检索验证"
CLI_EXAMPLES = (
    ("M2C1-Q001", "产品 A 的原装电池保修期多久？"),  # noqa: RUF001
    ("M2C1-Q025", "普通采购申请金额为 18 万元，需要谁审批？"),  # noqa: RUF001
    ("M2C1-Q045", "访问 L3 数据需要经过哪些审批？"),  # noqa: RUF001
    ("M2C1-Q010", "产品 A 的维修完成后，公司承诺免费保修多少天？"),  # noqa: RUF001
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets/enterprise_kb/m2c1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/evaluation")
    parser.add_argument(
        "--hub-cache",
        type=Path,
        required=True,
        help="Hugging Face hub cache root containing both complete snapshots",
    )
    return parser


def _snapshot_revision(hub_cache: Path, model_name: str) -> str:
    model_dir = hub_cache / f"models--{model_name.replace('/', '--')}"
    ref = model_dir / "refs/main"
    try:
        revision = ref.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"offline cache ref is unavailable for {model_name}") from exc
    snapshot = model_dir / "snapshots" / revision
    if len(revision) != 40 or not snapshot.is_dir():
        raise RuntimeError(f"offline cache snapshot is invalid for {model_name}")
    return revision


def _portable_sha256(path: Path) -> str:
    return sha256_text(path.read_text(encoding="utf-8-sig"))


async def _run_once(
    *,
    label: str,
    dataset_root: Path,
    hub_cache: Path,
    embedding_revision: str,
    reranker_revision: str,
    include_cli_examples: bool,
) -> tuple[EvaluatedRun, dict[str, Any], list[dict[str, Any]]]:
    # Model-bound imports remain outside module import and default unit-test collection.
    from decision_agent.retrieval.embeddings import SentenceTransformerEmbeddingProvider
    from decision_agent.retrieval.in_memory_store import InMemoryVectorStore
    from decision_agent.retrieval.pipeline import (
        EnterpriseRetrievalPipeline,
        RetrievalPipelineConfig,
    )
    from decision_agent.retrieval.reranking import SentenceTransformerCrossEncoderReranker

    queries = load_benchmark_query_inputs(dataset_root / "query_blueprint.jsonl")
    config = RetrievalPipelineConfig()
    embedding = SentenceTransformerEmbeddingProvider(
        model_name=EMBEDDING_MODEL,
        revision=embedding_revision,
        dimension=512,
        device="cpu",
        cache_folder=str(hub_cache),
        local_files_only=True,
    )
    reranker = SentenceTransformerCrossEncoderReranker(
        model_name=RERANKER_MODEL,
        model_revision=reranker_revision,
        device="cpu",
    )

    initialization_started = time.perf_counter()
    started = time.perf_counter()
    await embedding.initialize()
    embedding_model_load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    await reranker.initialize()
    reranker_model_load_seconds = time.perf_counter() - started

    pipeline = EnterpriseRetrievalPipeline(
        dataset_root=dataset_root,
        embedding_provider=embedding,
        vector_store=InMemoryVectorStore(dimension=512),
        reranker=reranker,
        config=config,
    )
    initialization = await pipeline.initialize()
    total_initialization_seconds = time.perf_counter() - initialization_started
    try:
        warmup_started = time.perf_counter()
        await pipeline.retrieve(WARMUP_QUERY)
        warmup_seconds = time.perf_counter() - warmup_started
        results = [await pipeline.retrieve(query.query) for query in queries]
        cli_summaries: list[dict[str, Any]] = []
        cli_started = time.perf_counter()
        if include_cli_examples:
            for source_query_id, query in CLI_EXAMPLES:
                cli_result = await pipeline.retrieve(query)
                cli_summaries.append(
                    {
                        "source_query_id": source_query_id,
                        "query": query,
                        "stage_candidate_counts": {
                            "dense": len(cli_result.dense_results),
                            "bm25": len(cli_result.bm25_results),
                            "rrf": len(cli_result.fused_results),
                            "reranker": len(cli_result.reranked_child_results),
                            "parent": len(cli_result.expanded_parent_results),
                        },
                        "evidence_ids": [
                            item.evidence_id for item in cli_result.evidence_context.references
                        ],
                        "top_evidence": [
                            reference.model_dump(mode="json")
                            for reference in cli_result.evidence_context.references[:2]
                        ],
                        "retrieval_only_notice": "当前只返回检索证据，不生成 LLM 答案",  # noqa: RUF001
                    }
                )
        cli_validation_seconds = time.perf_counter() - cli_started
    finally:
        await pipeline.close()

    # Relevance labels are intentionally loaded only after all runtime retrieval calls finish.
    ground_truth = load_retrieval_ground_truth(
        dataset_root / "generated/retrieval_ground_truth.jsonl"
    )
    evaluated = evaluate_retrieval_results(queries, results, ground_truth)
    runtime = {
        "run_label": label,
        "run_semantics": {
            "process_position": (
                "current_process_first_observation"
                if label == "A"
                else "same_process_subsequent_observation"
            ),
            "same_python_process_as_other_run": True,
            "fresh_pipeline_instance": True,
            "fresh_embedding_provider_instance": True,
            "fresh_reranker_instance": True,
            "independent_cold_start_claim": False,
            "possible_cache_effects": [
                "operating_system_page_cache",
                "underlying_library_process_state",
            ],
            "model_load_boundary": (
                "Each timer surrounds the complete provider.initialize() call before Pipeline "
                "initialization; the Pipeline dependency-ready recheck is reported separately."
            ),
            "lazy_model_load_outside_timer": False,
        },
        "initialization": {
            "data_load_seconds": initialization.data_load_seconds,
            "embedding_model_load_seconds": embedding_model_load_seconds,
            "reranker_model_load_seconds": reranker_model_load_seconds,
            "dense_index_build_seconds": initialization.dense_index_build_seconds,
            "bm25_index_build_seconds": initialization.bm25_index_build_seconds,
            "dependency_ready_check_seconds": initialization.model_load_seconds,
            "total_initialization_seconds": total_initialization_seconds,
        },
        "warmup": {
            "query": WARMUP_QUERY,
            "seconds": warmup_seconds,
            "excluded_from_formal_queries": True,
        },
        "query_stage_distributions": evaluated.analysis["query_timing_distributions"],
        "cli_validation": {
            "query_count": len(cli_summaries),
            "seconds": cli_validation_seconds,
            "excluded_from_formal_50_query_timings": True,
        },
    }
    return evaluated, runtime, cli_summaries


async def _run(args: argparse.Namespace) -> int:
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 are required")
    hub_cache = args.hub_cache.resolve()
    embedding_revision = _snapshot_revision(hub_cache, EMBEDDING_MODEL)
    reranker_revision = _snapshot_revision(hub_cache, RERANKER_MODEL)

    run_a, runtime_a, _ = await _run_once(
        label="A",
        dataset_root=args.dataset_root,
        hub_cache=hub_cache,
        embedding_revision=embedding_revision,
        reranker_revision=reranker_revision,
        include_cli_examples=False,
    )
    gc.collect()
    run_b, runtime_b, cli_summaries = await _run_once(
        label="B",
        dataset_root=args.dataset_root,
        hub_cache=hub_cache,
        embedding_revision=embedding_revision,
        reranker_revision=reranker_revision,
        include_cli_examples=True,
    )
    assert_deterministic_runs(run_a, run_b)

    summary_path = ROOT / "artifacts/datasets/m2c1_parent_child_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = run_b.query_results[0]
    source_paths = {
        "query_blueprint.jsonl": args.dataset_root / "query_blueprint.jsonl",
        "parent_chunks.jsonl": args.dataset_root / "generated/parent_chunks.jsonl",
        "child_chunks.jsonl": args.dataset_root / "generated/child_chunks.jsonl",
        "clause_chunk_map.jsonl": args.dataset_root / "generated/clause_chunk_map.jsonl",
        "retrieval_ground_truth.jsonl": args.dataset_root
        / "generated/retrieval_ground_truth.jsonl",
        "m2c1_parent_child_summary.json": summary_path,
    }
    pipeline_path = ROOT / "src/decision_agent/retrieval/pipeline.py"
    from decision_agent.retrieval.pipeline import RetrievalPipelineConfig

    fixed_config = RetrievalPipelineConfig().model_dump(mode="json")
    report = {
        "schema_version": SCHEMA_VERSION,
        "baseline_id": BASELINE_ID,
        "dataset_id": summary["dataset_id"],
        "pipeline_version": fixed_config["config_version"],
        "pipeline_identity": {
            "source": "src/decision_agent/retrieval/pipeline.py",
            "source_sha256": _portable_sha256(pipeline_path),
            "config_sha256": sha256_text(
                json.dumps(fixed_config, sort_keys=True, separators=(",", ":"))
            ),
        },
        "models": {
            "embedding": {
                "name": EMBEDDING_MODEL,
                "revision": embedding_revision,
                "dimension": 512,
            },
            "reranker": {"name": RERANKER_MODEL, "revision": reranker_revision},
        },
        "device": "cpu",
        "offline_only": True,
        "fixed_config": fixed_config,
        "counts": {
            "document": summary["document_count"],
            "parent": summary["parent_chunk_count"],
            "child": summary["child_chunk_count"],
            "clause": summary["clause_count"],
            "query": summary["query_count"],
            "answerable": summary["answerable_query_count"],
            "unanswerable": summary["unanswerable_query_count"],
        },
        "source_hashes": {name: _portable_sha256(path) for name, path in source_paths.items()},
        "ground_truth_sha256": _portable_sha256(
            args.dataset_root / "generated/retrieval_ground_truth.jsonl"
        ),
        **{
            key: value
            for key, value in run_b.analysis.items()
            if key != "query_timing_distributions"
        },
        "ranking_digest": run_b.ranking_digest,
        "deterministic_ranking_digest": run_b.ranking_digest,
        "cli_evidence_summaries": cli_summaries,
        "determinism": {
            "run_a_ranking_digest": run_a.ranking_digest,
            "run_b_ranking_digest": run_b.ranking_digest,
            "run_a_deterministic_digest": run_a.deterministic_digest,
            "run_b_deterministic_digest": run_b.deterministic_digest,
            "candidate_order_and_scored_analysis_identical": True,
            "ignored_fields": ["timings", "runtime_profile", "floating_point_scores"],
            "artifact_hashes_are_not_ranking_determinism": True,
        },
        "limitations": [
            (
                "Metrics cover one fixed synthetic enterprise dataset and do not establish "
                "production quality."
            ),
            "Runtime values are one local CPU observation, not a production performance benchmark.",
            "Fixed character-window chunks can mix relevant and hard-negative clauses.",
            "Overlap chunks are reported separately and are not treated as pure negatives.",
            (
                "Parent-versus-Reranker is a cross-granularity evidence-coverage observation, "
                "not a strict same-metric model gain."
            ),
            (
                "Run B follows Run A in the same Python process and is not an independent cold "
                "start; operating-system and library cache effects may reduce load time."
            ),
            "This retrieval-only baseline does not call an LLM or assess answer quality.",
        ],
    }
    runtime_profile = {
        "schema_version": SCHEMA_VERSION,
        "baseline_id": BASELINE_ID,
        "measurement_scope": "本机 CPU 单次观测",
        "execution_contract": {
            "run_order": ["A", "B"],
            "same_python_process": True,
            "sequential": True,
            "fresh_pipeline_and_provider_instances_per_run": True,
            "run_b_is_independent_cold_start": False,
            "production_sla_claim": False,
        },
        "model_download_seconds": {"value": 29.733823, "excluded_from_all_formal_timings": True},
        "run_a": runtime_a,
        "run_b": runtime_b,
        "run_b_total_query_seconds_summary": timing_summary(
            [row["stage_timings"]["total_runtime_seconds"] for row in run_b.query_results]
        ),
    }
    hashes = write_baseline_artifacts(
        output_dir=args.output_dir,
        main_report=report,
        query_results=run_b.query_results,
        failure_cases=run_b.failure_cases,
        runtime_profile=runtime_profile,
    )
    print(f"ranking_digest_a={run_a.ranking_digest}")
    print(f"ranking_digest_b={run_b.ranking_digest}")
    print("rankings_identical=true")
    for filename, digest in sorted(hashes.items()):
        print(f"{filename} sha256={digest}")
    print(f"query_count={config['query_id'] and len(run_b.query_results)}")
    return 0


def main() -> int:
    return asyncio.run(_run(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
