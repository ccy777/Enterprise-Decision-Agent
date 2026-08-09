"""Opt-in M8D-A2 production ingestion and hybrid retrieval acceptance."""

from __future__ import annotations

import math
import os
import time
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from decision_agent.config import Settings
from decision_agent.evaluation.enterprise_kb_dataset import (
    EnterpriseKBDataset,
    load_and_validate_enterprise_kb,
)
from decision_agent.observability import SpanStatus, TraceCollector, TraceContext, TraceStage
from decision_agent.retrieval.factory import (
    EnterpriseRetrievalRuntime,
    build_production_retrieval_runtime,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_M8D_A2_REAL_RETRIEVAL") != "1",
        reason="set RUN_M8D_A2_REAL_RETRIEVAL=1 for the provisioned local Milvus acceptance",
    ),
]

DATASET_ROOT = Path(__file__).resolve().parents[3] / "datasets/enterprise_kb/m2c1"


class _SpanIds:
    def __init__(self) -> None:
        self._value = 0

    def __call__(self) -> str:
        self._value += 1
        return f"m8d_span_{self._value}"


def _queries(dataset: EnterpriseKBDataset) -> Iterator[str]:
    yield next(
        query.query
        for query in dataset.queries
        if query.answerable and query.category == "customer_service"
    )
    yield next(
        query.query
        for query in dataset.queries
        if query.answerable and query.category == "inventory"
    )
    yield "量子纠缠实验室的激光器维护周期是什么？"  # noqa: RUF001


def _trace() -> tuple[TraceCollector, TraceContext, TraceContext]:
    collector = TraceCollector(
        context=TraceContext.create(request_id="m8d_retrieval", id_factory=lambda: "m8d_trace"),
        id_factory=_SpanIds(),
    )
    root = collector.start_span(stage=TraceStage.REQUEST, component="executor", operation="execute")
    tool = collector.start_span(
        stage=TraceStage.TOOL_EXECUTION,
        component="tool_calling",
        operation="execute_authorized_tool",
        parent_context=root,
    )
    return collector, root, tool


async def _run_retrieval_smoke(
    runtime: EnterpriseRetrievalRuntime,
    dataset: EnterpriseKBDataset,
) -> int:
    completed = 0
    for query in _queries(dataset):
        collector, root, tool = _trace()
        result = await runtime.pipeline.retrieve(
            query, trace_recorder=collector, trace_parent_context=tool
        )
        collector.complete_span(tool, status=SpanStatus.COMPLETED)
        collector.complete_span(root, status=SpanStatus.COMPLETED)
        trace = collector.finalize(final_status=SpanStatus.COMPLETED)
        operations = {span.operation: span for span in trace.spans}
        retrieval_operations = {
            "hybrid_retrieve",
            "cross_encoder_rerank",
            "parent_expansion",
            "build_evidence_context",
        }
        assert retrieval_operations <= set(operations)
        assert all(
            operations[operation].parent_span_id == tool.current_span_id
            for operation in retrieval_operations
        )
        serialized_trace = str(trace.model_dump(mode="json"))
        assert query not in serialized_trace
        assert result.dense_results and result.bm25_results and result.fused_results
        assert result.reranked_child_results and result.expanded_parent_results
        assert all(math.isfinite(item.score) for item in result.dense_results)
        assert all(math.isfinite(item.reranker_score) for item in result.reranked_child_results)
        evidence_ids = [item.evidence_id for item in result.evidence_context.evidence_items]
        assert evidence_ids and len(evidence_ids) == len(set(evidence_ids))
        assert all(
            item.content not in serialized_trace for item in result.evidence_context.evidence_items
        )
        assert all(
            reference.source and reference.parent_id
            for reference in result.evidence_context.references
        )
        completed += 1
    return completed


@pytest.mark.asyncio
async def test_real_milvus_child_ingestion_is_idempotent_and_hybrid_retrieval_runs() -> None:
    if os.getenv("HF_HUB_OFFLINE") != "1" or os.getenv("TRANSFORMERS_OFFLINE") != "1":
        pytest.fail("M8D-A2 requires strict Hugging Face and Transformers offline modes")

    started = time.perf_counter()
    settings = Settings(knowledge_dataset_root=DATASET_ROOT)
    target = urlsplit(settings.milvus_uri).hostname
    if target not in {"127.0.0.1", "localhost", "::1"}:
        pytest.skip("M8D-A2 requires the local loopback Milvus service")
    dataset, statistics = load_and_validate_enterprise_kb(settings.knowledge_dataset_root)

    first_runtime = build_production_retrieval_runtime(settings)
    try:
        await first_runtime._vector_store.initialize()  # type: ignore[attr-defined]
        before_count = await first_runtime._vector_store.count()  # type: ignore[attr-defined]
        await first_runtime.initialize_for_ingestion()
        first = first_runtime.pipeline.last_ingestion_result
        expected_count = first_runtime.pipeline.child_count
        assert first is not None and expected_count > 0
        assert statistics.document_count == 12
        assert first.attempted_count == expected_count
        assert first.inserted_count + first.updated_count == expected_count
        assert await first_runtime._vector_store.count() == expected_count  # type: ignore[attr-defined]
        if before_count == 0:
            assert first.inserted_count == expected_count and first.updated_count == 0
        else:
            assert before_count == expected_count and first.updated_count == expected_count
    finally:
        await first_runtime.aclose()

    runtime = build_production_retrieval_runtime(settings)
    try:
        await runtime.initialize_for_ingestion()
        second = runtime.pipeline.last_ingestion_result
        assert second is not None
        assert second.inserted_count == 0 and second.updated_count == expected_count
        assert await runtime._vector_store.count() == expected_count  # type: ignore[attr-defined]

        smoke_queries = await _run_retrieval_smoke(runtime, dataset)

        print(
            "m8d_a2 "
            f"dataset_version={dataset.manifest.schema_version} fixed_window_v1=true "
            f"documents={statistics.document_count} parents={runtime.pipeline.parent_count} "
            f"children={runtime.pipeline.child_count} duplicate_id_count=0 "
            "invalid_record_count=0 broken_parent_reference_count=0 "
            f"first_before={before_count} first_processed={first.attempted_count} "
            f"first_inserted={first.inserted_count} first_updated={first.updated_count} "
            "first_failed=0 "
            f"second_processed={second.attempted_count} second_inserted={second.inserted_count} "
            f"second_updated={second.updated_count} second_failed=0 "
            f"collection_rows={await runtime._vector_store.count()} "  # type: ignore[attr-defined]
            f"smoke_queries={smoke_queries} elapsed_seconds={time.perf_counter() - started:.3f}"
        )
    finally:
        await runtime.aclose()


@pytest.mark.asyncio
async def test_real_milvus_reader_reuses_existing_corpus_without_ingestion() -> None:
    if os.getenv("HF_HUB_OFFLINE") != "1" or os.getenv("TRANSFORMERS_OFFLINE") != "1":
        pytest.fail("M8D-A2 requires strict Hugging Face and Transformers offline modes")

    settings = Settings(knowledge_dataset_root=DATASET_ROOT)
    target = urlsplit(settings.milvus_uri).hostname
    if target not in {"127.0.0.1", "localhost", "::1"}:
        pytest.skip("M8D-A2 requires the local loopback Milvus service")
    dataset, _ = load_and_validate_enterprise_kb(settings.knowledge_dataset_root)

    runtime = build_production_retrieval_runtime(settings)
    try:
        await runtime.initialize()
        before_count = await runtime._vector_store.count()  # type: ignore[attr-defined]
        assert runtime.pipeline.last_ingestion_result is None
        smoke_queries = await _run_retrieval_smoke(runtime, dataset)
        assert await runtime._vector_store.count() == before_count == runtime.pipeline.child_count  # type: ignore[attr-defined]
        print(
            "m8d_a_reader "
            f"collection_rows={before_count} smoke_queries={smoke_queries} "
            "ingestion_triggered=false"
        )
    finally:
        await runtime.aclose()
