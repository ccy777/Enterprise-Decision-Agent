"""Run one strict-offline Clause-aware evaluation and compare saved fixed results."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from decision_agent.evaluation.enterprise_chunking_comparison import (
    EMBEDDING_MODEL,
    EMBEDDING_REVISION,
    RERANKER_MODEL,
    RERANKER_REVISION,
    build_chunking_comparison,
    chunk_length_statistics,
    read_query_results,
    stable_comparison_json,
    validate_fixed_window_artifacts,
)
from decision_agent.evaluation.enterprise_retrieval_baseline import (
    evaluate_retrieval_results,
    load_benchmark_query_inputs,
    load_retrieval_ground_truth,
    sha256_text,
    stable_json,
    stable_jsonl,
)
from decision_agent.evaluation.reporting import write_text_files_atomically
from decision_agent.exceptions import EvaluationError

ROOT = Path(__file__).resolve().parents[1]
WARMUP_QUERY = "企业制度流程检索验证"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets/enterprise_kb/m2c1")
    parser.add_argument(
        "--variant-root",
        type=Path,
        default=ROOT / "datasets/enterprise_kb/m2c1/variants/clause_aware_v1",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/evaluation")
    parser.add_argument(
        "--hub-cache",
        type=Path,
        help="Hugging Face hub cache root containing both complete snapshots",
    )
    parser.add_argument(
        "--from-saved-results",
        action="store_true",
        help=(
            "Rebuild only the comparison JSON from saved Results and Runtime without model loading."
        ),
    )
    return parser


def _snapshot_revision(hub_cache: Path, model_name: str) -> str:
    model_dir = hub_cache / f"models--{model_name.replace('/', '--')}"
    try:
        revision = (model_dir / "refs/main").read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"offline cache ref is unavailable for {model_name}") from exc
    if (
        revision
        != {
            EMBEDDING_MODEL: EMBEDDING_REVISION,
            RERANKER_MODEL: RERANKER_REVISION,
        }[model_name]
        or not (model_dir / "snapshots" / revision).is_dir()
    ):
        raise RuntimeError(f"offline cache snapshot is not the required revision for {model_name}")
    return revision


def _portable_sha256(path: Path) -> str:
    return sha256_text(path.read_text(encoding="utf-8-sig"))


def _write_comparison_from_saved(args: argparse.Namespace) -> int:
    dataset_root, variant_root, output_dir = (
        args.dataset_root.resolve(),
        args.variant_root.resolve(),
        args.output_dir.resolve(),
    )
    fixed = validate_fixed_window_artifacts(
        report_path=output_dir / "m2c2a2_retrieval_baseline.json",
        query_results_path=output_dir / "m2c2a2_query_results.jsonl",
        runtime_path=output_dir / "m2c2a2_runtime_profile.json",
    )
    clause_results_path = output_dir / "m2c2b2_clause_aware_results.jsonl"
    clause_runtime_path = output_dir / "m2c2b2_clause_aware_runtime.json"
    clause_runtime = json.loads(clause_runtime_path.read_text(encoding="utf-8"))
    fixed_summary_path = ROOT / "artifacts/datasets/m2c1_parent_child_summary.json"
    clause_summary_path = ROOT / "artifacts/datasets/m2c2b1_clause_aware_summary.json"
    input_paths = {
        "fixed_report": output_dir / "m2c2a2_retrieval_baseline.json",
        "fixed_results": output_dir / "m2c2a2_query_results.jsonl",
        "fixed_runtime": output_dir / "m2c2a2_runtime_profile.json",
        "fixed_summary": fixed_summary_path,
        "fixed_parent_chunks": dataset_root / "generated/parent_chunks.jsonl",
        "fixed_child_chunks": dataset_root / "generated/child_chunks.jsonl",
        "clause_results": clause_results_path,
        "clause_runtime": clause_runtime_path,
        "clause_ground_truth": variant_root / "retrieval_ground_truth.jsonl",
        "clause_summary": clause_summary_path,
        "clause_parent_chunks": variant_root / "parent_chunks.jsonl",
        "clause_child_chunks": variant_root / "child_chunks.jsonl",
    }
    comparison = build_chunking_comparison(
        fixed=fixed,
        fixed_ground_truth=load_retrieval_ground_truth(
            dataset_root / "generated/retrieval_ground_truth.jsonl"
        ),
        clause_query_results=read_query_results(clause_results_path),
        clause_ground_truth=load_retrieval_ground_truth(
            variant_root / "retrieval_ground_truth.jsonl"
        ),
        clause_runtime=clause_runtime,
        fixed_summary=json.loads(fixed_summary_path.read_text(encoding="utf-8")),
        clause_summary=json.loads(clause_summary_path.read_text(encoding="utf-8")),
        fixed_chunk_lengths=chunk_length_statistics(
            parent_path=dataset_root / "generated/parent_chunks.jsonl",
            child_path=dataset_root / "generated/child_chunks.jsonl",
        ),
        clause_chunk_lengths=chunk_length_statistics(
            parent_path=variant_root / "parent_chunks.jsonl",
            child_path=variant_root / "child_chunks.jsonl",
        ),
        input_hashes={name: _portable_sha256(path) for name, path in input_paths.items()},
    )
    comparison_text = stable_comparison_json(comparison)
    path = output_dir / "m2c2b2_chunking_comparison.json"
    write_text_files_atomically({path: comparison_text})
    print(f"{path.name} sha256={sha256_text(comparison_text)}")
    print(f"clause_aware_ranking_digest={comparison['ranking_digests']['clause_aware']}")
    return 0


async def _run(args: argparse.Namespace) -> int:
    if args.from_saved_results:
        return _write_comparison_from_saved(args)
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 are required")
    if args.hub_cache is None:
        raise RuntimeError("--hub-cache is required for the real Clause-aware evaluation")
    hub_cache = args.hub_cache.resolve()
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    embedding_revision = _snapshot_revision(hub_cache, EMBEDDING_MODEL)
    reranker_revision = _snapshot_revision(hub_cache, RERANKER_MODEL)
    dataset_root, variant_root, output_dir = (
        args.dataset_root.resolve(),
        args.variant_root.resolve(),
        args.output_dir.resolve(),
    )
    fixed = validate_fixed_window_artifacts(
        report_path=output_dir / "m2c2a2_retrieval_baseline.json",
        query_results_path=output_dir / "m2c2a2_query_results.jsonl",
        runtime_path=output_dir / "m2c2a2_runtime_profile.json",
    )

    # Model-bound imports stay inside the opt-in CLI, never default unit-test collection.
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
    probe = await embedding.embed_documents(["离线模型门禁文本"])
    if len(probe) != 1 or len(probe[0]) != 512:
        raise RuntimeError("offline embedding model gate returned an unexpected dimension")
    pipeline = EnterpriseRetrievalPipeline(
        dataset_root=variant_root,
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
    finally:
        await pipeline.close()

    # Labels enter only after all 50 runtime calls have completed.
    clause_ground_truth = load_retrieval_ground_truth(variant_root / "retrieval_ground_truth.jsonl")
    evaluated = evaluate_retrieval_results(queries, results, clause_ground_truth)
    fixed_ground_truth = load_retrieval_ground_truth(
        dataset_root / "generated/retrieval_ground_truth.jsonl"
    )
    runtime = {
        "schema_version": "1.0",
        "run_id": "m2c2b2-clause-aware-real-retrieval-v1",
        "measurement_scope": "本机 CPU 单次观测",
        "comparison_contract": {
            "strategy_id": "clause-aware-v1",
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
            "fixed_config": config.model_dump(mode="json"),
            "formal_query_count": 50,
            "warmup_excluded_from_formal_queries": True,
            "ground_truth_not_used_in_runtime_ranking": True,
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
        "offline_model_gate": {"embedding_dimension": len(probe[0]), "passed": True},
        "warmup": {"query": WARMUP_QUERY, "seconds": warmup_seconds, "excluded": True},
        "query_stage_distributions": evaluated.analysis["query_timing_distributions"],
    }
    fixed_summary_path = ROOT / "artifacts/datasets/m2c1_parent_child_summary.json"
    clause_summary_path = ROOT / "artifacts/datasets/m2c2b1_clause_aware_summary.json"
    input_paths = {
        "fixed_report": output_dir / "m2c2a2_retrieval_baseline.json",
        "fixed_results": output_dir / "m2c2a2_query_results.jsonl",
        "fixed_runtime": output_dir / "m2c2a2_runtime_profile.json",
        "fixed_summary": fixed_summary_path,
        "fixed_parent_chunks": dataset_root / "generated/parent_chunks.jsonl",
        "fixed_child_chunks": dataset_root / "generated/child_chunks.jsonl",
        "clause_ground_truth": variant_root / "retrieval_ground_truth.jsonl",
        "clause_summary": clause_summary_path,
        "clause_parent_chunks": variant_root / "parent_chunks.jsonl",
        "clause_child_chunks": variant_root / "child_chunks.jsonl",
    }
    comparison = build_chunking_comparison(
        fixed=fixed,
        fixed_ground_truth=fixed_ground_truth,
        clause_query_results=evaluated.query_results,
        clause_ground_truth=clause_ground_truth,
        clause_runtime=runtime,
        fixed_summary=json.loads(fixed_summary_path.read_text(encoding="utf-8")),
        clause_summary=json.loads(clause_summary_path.read_text(encoding="utf-8")),
        fixed_chunk_lengths=chunk_length_statistics(
            parent_path=dataset_root / "generated/parent_chunks.jsonl",
            child_path=dataset_root / "generated/child_chunks.jsonl",
        ),
        clause_chunk_lengths=chunk_length_statistics(
            parent_path=variant_root / "parent_chunks.jsonl",
            child_path=variant_root / "child_chunks.jsonl",
        ),
        input_hashes={name: _portable_sha256(path) for name, path in input_paths.items()},
    )
    result_text = stable_jsonl(evaluated.query_results)
    runtime_text = stable_json(runtime)
    comparison_text = stable_comparison_json(comparison)
    files = {
        output_dir / "m2c2b2_clause_aware_results.jsonl": result_text,
        output_dir / "m2c2b2_clause_aware_runtime.json": runtime_text,
        output_dir / "m2c2b2_chunking_comparison.json": comparison_text,
    }
    write_text_files_atomically(files)
    for path, content in sorted(files.items()):
        print(f"{path.name} sha256={sha256_text(content)}")
    print(f"clause_aware_ranking_digest={evaluated.ranking_digest}")
    print("formal_query_count=50")
    return 0


def main() -> int:
    try:
        return asyncio.run(_run(build_parser().parse_args()))
    except (EvaluationError, RuntimeError) as exc:
        print(f"Clause-aware evaluation failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
