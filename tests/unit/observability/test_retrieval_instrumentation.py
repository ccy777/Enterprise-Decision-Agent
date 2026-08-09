"""Trace contracts for the concrete hybrid retrieval pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from decision_agent.exceptions import RetrievalValidationError
from decision_agent.observability import SpanStatus, TraceCollector, TraceContext, TraceStage
from decision_agent.retrieval import (
    DeterministicHashEmbeddingProvider,
    EnterpriseRetrievalPipeline,
    InMemoryVectorStore,
)
from decision_agent.retrieval.reranking import RerankCandidate, RerankedResult

ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = ROOT / "datasets/enterprise_kb/m2c1"


class _Ids:
    def __init__(self) -> None:
        self._value = 0

    def __call__(self) -> str:
        self._value += 1
        return f"span_{self._value}"


class _Reranker:
    def __init__(self) -> None:
        self.calls: list[tuple[RerankCandidate, ...]] = []

    async def initialize(self) -> None:
        return None

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        *,
        top_k: int | None = None,
    ) -> list[RerankedResult]:
        del query
        copied = tuple(candidates)
        self.calls.append(copied)
        limit = len(copied) if top_k is None else min(top_k, len(copied))
        return [
            RerankedResult(
                final_rank=rank,
                candidate_id=item.candidate_id,
                record_id=item.record_id,
                document_id=item.document_id,
                content=item.content,
                reranker_score=float(limit - rank + 1),
                upstream_rank=item.upstream_rank,
                upstream_score=item.upstream_score,
                metadata=item.metadata,
                provenance=item.provenance,
            )
            for rank, item in enumerate(copied[:limit], start=1)
        ]


class _EmptyDenseRetriever:
    async def retrieve(self, query: str, *, top_k: int) -> list[object]:
        del query, top_k
        return []


class _EmptyBm25Retriever:
    def retrieve(self, query: str, *, top_k: int) -> list[object]:
        del query, top_k
        return []


class _BrokenRecorder:
    def start_span(self, **_: object) -> TraceContext:
        raise RuntimeError("PRIVATE_RECORDER_FAILURE")

    def complete_span(self, *_: object, **__: object) -> None:
        raise RuntimeError("PRIVATE_RECORDER_FAILURE")


def _collector() -> tuple[TraceCollector, TraceContext, TraceContext]:
    collector = TraceCollector(
        context=TraceContext.create(request_id="request_1", id_factory=lambda: "trace_1"),
        utc_now=lambda: datetime(2026, 7, 28, tzinfo=UTC),
        monotonic=lambda: 10.0,
        id_factory=_Ids(),
    )
    root = collector.start_span(stage=TraceStage.REQUEST, component="executor", operation="execute")
    tool = collector.start_span(
        stage=TraceStage.TOOL_EXECUTION,
        component="tool_calling",
        operation="execute_authorized_tool",
        parent_context=root,
    )
    return collector, root, tool


def _finish(collector: TraceCollector, root: TraceContext, tool: TraceContext):
    collector.complete_span(tool, status=SpanStatus.COMPLETED)
    collector.complete_span(root, status=SpanStatus.COMPLETED)
    return collector.finalize(final_status=SpanStatus.COMPLETED)


def _attributes(span: object) -> dict[str, object]:
    return {attribute.key: attribute.value for attribute in span.attributes}  # type: ignore[attr-defined]


def _pipeline() -> tuple[EnterpriseRetrievalPipeline, _Reranker]:
    embedding = DeterministicHashEmbeddingProvider(dimension=128)
    reranker = _Reranker()
    return (
        EnterpriseRetrievalPipeline(
            dataset_root=DATASET_ROOT,
            embedding_provider=embedding,
            vector_store=InMemoryVectorStore(dimension=128),
            reranker=reranker,
        ),
        reranker,
    )


@pytest.mark.asyncio
async def test_hybrid_retrieval_stages_are_tool_children() -> None:
    pipeline, reranker = _pipeline()
    await pipeline.initialize()
    collector, root, tool = _collector()

    result = await pipeline.retrieve(
        "PRIVATE_RETRIEVAL_QUERY",
        trace_recorder=collector,
        trace_parent_context=tool,
    )
    trace = _finish(collector, root, tool)
    await pipeline.close()

    spans = {span.operation: span for span in trace.spans if span.operation != "execute"}
    retrieval = spans["hybrid_retrieve"]
    reranking = spans["cross_encoder_rerank"]
    expansion = spans["parent_expansion"]
    evidence = spans["build_evidence_context"]
    assert result.evidence_context.evidence_items
    assert len(reranker.calls) == 1
    assert (
        retrieval.parent_span_id
        == reranking.parent_span_id
        == expansion.parent_span_id
        == evidence.parent_span_id
        == tool.current_span_id
    )
    assert _attributes(retrieval) == {
        "requested_top_k": 10,
        "dense_candidate_count": 10,
        "sparse_candidate_count": 0,
        "fused_candidate_count": 10,
        "retrieved_count": 10,
        "empty_result": False,
        "success": True,
    }
    assert _attributes(reranking) == {
        "input_candidate_count": 10,
        "requested_top_k": 5,
        "reranked_count": 5,
        "success": True,
    }
    assert _attributes(expansion) == {
        "child_count": 5,
        "parent_count": 5,
        "expanded_count": 5,
        "success": True,
    }
    assert _attributes(evidence) == {
        "candidate_evidence_count": 5,
        "selected_evidence_count": 5,
        "empty_result": False,
        "success": True,
    }
    serialized = str(trace.model_dump(mode="json"))
    assert "PRIVATE_RETRIEVAL_QUERY" not in serialized
    assert "Evidence content" not in serialized


@pytest.mark.asyncio
async def test_empty_candidates_fail_before_reranking_without_a_fake_span() -> None:
    pipeline, reranker = _pipeline()
    await pipeline.initialize()
    pipeline._dense_retriever = _EmptyDenseRetriever()  # type: ignore[assignment]
    pipeline._bm25_retriever = _EmptyBm25Retriever()  # type: ignore[assignment]
    collector, root, tool = _collector()

    with pytest.raises(RetrievalValidationError, match="RRF requires at least one source"):
        await pipeline.retrieve(
            "PRIVATE_EMPTY_QUERY", trace_recorder=collector, trace_parent_context=tool
        )
    trace = _finish(collector, root, tool)
    await pipeline.close()

    retrieval = next(span for span in trace.spans if span.operation == "hybrid_retrieve")
    assert reranker.calls == []
    assert retrieval.status is SpanStatus.FAILED
    assert retrieval.error_code == "retrieval_failed"
    assert not [span for span in trace.spans if span.operation == "cross_encoder_rerank"]
    assert not [span for span in trace.spans if span.operation == "parent_expansion"]


@pytest.mark.asyncio
async def test_retrieval_recorder_failure_does_not_repeat_the_reranker() -> None:
    pipeline, reranker = _pipeline()
    await pipeline.initialize()

    result = await pipeline.retrieve("PRIVATE_QUERY", trace_recorder=_BrokenRecorder())
    await pipeline.close()

    assert result.evidence_context.evidence_items
    assert len(reranker.calls) == 1
