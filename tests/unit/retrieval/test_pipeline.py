"""Offline integration tests for the enterprise retrieval pipeline."""

from __future__ import annotations

import hashlib
import shutil
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from decision_agent.exceptions import RetrievalValidationError
from decision_agent.retrieval import (
    DeterministicHashEmbeddingProvider,
    EnterpriseRetrievalPipeline,
    InMemoryVectorStore,
    RetrievalPipelineConfig,
)
from decision_agent.retrieval.reranking import (
    RerankCandidate,
    RerankedResult,
    SentenceTransformerCrossEncoderReranker,
)

ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = ROOT / "datasets/enterprise_kb/m2c1"


class FakeEmbeddingProvider:
    """Count calls while delegating to the deterministic offline embedding."""

    def __init__(self, dimension: int = 128) -> None:
        self._inner = DeterministicHashEmbeddingProvider(dimension=dimension)
        self.initialize_count = 0
        self.document_calls: list[tuple[str, ...]] = []
        self.query_calls: list[str] = []

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    async def initialize(self) -> None:
        self.initialize_count += 1

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_calls.append(tuple(texts))
        return await self._inner.embed_documents(texts)

    async def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return await self._inner.embed_query(text)


class FakeReranker:
    """Return deterministic relative scores without loading a model."""

    def __init__(self) -> None:
        self.initialize_count = 0
        self.calls: list[tuple[str, tuple[RerankCandidate, ...], int | None]] = []

    async def initialize(self) -> None:
        self.initialize_count += 1

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        *,
        top_k: int | None = None,
    ) -> list[RerankedResult]:
        copied = tuple(candidate.model_copy(deep=True) for candidate in candidates)
        self.calls.append((query, copied, top_k))
        limit = len(copied) if top_k is None else min(top_k, len(copied))
        return [
            RerankedResult(
                final_rank=rank,
                candidate_id=item.candidate_id,
                record_id=item.record_id,
                document_id=item.document_id,
                content=item.content,
                reranker_score=float(len(copied) - index),
                upstream_rank=item.upstream_rank,
                upstream_score=item.upstream_score,
                metadata=item.model_copy(deep=True).metadata,
                provenance=item.model_copy(deep=True).provenance,
            )
            for rank, (index, item) in enumerate(enumerate(copied[:limit]), start=1)
        ]


class FakeCrossEncoderModel:
    def __init__(self) -> None:
        self.calls: list[list[tuple[str, str]]] = []

    def predict(
        self,
        pairs: list[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
    ) -> list[float]:
        del batch_size, show_progress_bar
        self.calls.append(pairs)
        return [2.0, 1.0]


class LogicalCorpusStore(InMemoryVectorStore):
    """Expose a logical primary-key snapshot independently of physical row statistics."""

    def __init__(
        self,
        *,
        dimension: int,
        physical_row_count: int = 522,
        physical_ids_factory: Callable[[frozenset[str]], Sequence[str]] | None = None,
    ) -> None:
        super().__init__(dimension=dimension)
        self.physical_row_count = physical_row_count
        self.physical_ids_factory = physical_ids_factory or tuple
        self.list_record_ids_calls = 0
        self.count_calls = 0
        self.last_physical_ids: tuple[str, ...] = ()

    async def list_record_ids(self) -> frozenset[str]:
        self.list_record_ids_calls += 1
        logical_ids = await super().list_record_ids()
        self.last_physical_ids = tuple(self.physical_ids_factory(logical_ids))
        return frozenset(self.last_physical_ids)

    async def count(self) -> int:
        self.count_calls += 1
        return self.physical_row_count


def make_pipeline(
    *,
    dataset_root: Path = DATASET_ROOT,
    config: RetrievalPipelineConfig | None = None,
    store: InMemoryVectorStore | None = None,
) -> tuple[
    EnterpriseRetrievalPipeline,
    FakeEmbeddingProvider,
    FakeReranker,
    InMemoryVectorStore,
]:
    embedding = FakeEmbeddingProvider()
    reranker = FakeReranker()
    resolved_store = store or InMemoryVectorStore(dimension=embedding.dimension)
    pipeline = EnterpriseRetrievalPipeline(
        dataset_root=dataset_root,
        embedding_provider=embedding,
        vector_store=resolved_store,
        reranker=reranker,
        config=config,
    )
    return pipeline, embedding, reranker, resolved_store


@pytest.fixture
async def initialized_pipeline():
    pipeline, embedding, reranker, store = make_pipeline()
    await pipeline.initialize()
    yield pipeline, embedding, reranker, store
    if not pipeline.closed:
        await pipeline.close()


def test_fixed_pipeline_config_and_validation() -> None:
    config = RetrievalPipelineConfig()

    assert config.model_dump() == {
        "config_version": "m2c2a1-v1",
        "dense_top_k": 10,
        "bm25_top_k": 10,
        "rrf_top_k": 10,
        "rrf_k": 60.0,
        "dense_weight": 1.0,
        "bm25_weight": 1.0,
        "reranker_candidate_count": 10,
        "reranker_top_k": 5,
        "parent_top_k": 5,
        "evidence_max_count": 5,
        "evidence_max_total_chars": 6000,
        "evidence_max_chars_per_item": 1800,
    }
    with pytest.raises(ValueError):
        RetrievalPipelineConfig(rrf_top_k=5, reranker_candidate_count=10)


@pytest.mark.asyncio
async def test_initialize_loads_formal_chunks_and_builds_indexes_once() -> None:
    pipeline, embedding, reranker, store = make_pipeline()

    first = await pipeline.initialize()
    second = await pipeline.initialize()

    assert pipeline.initialized is True
    assert (pipeline.parent_count, pipeline.child_count) == (36, 101)
    assert await store.count() == 101
    assert embedding.initialize_count == reranker.initialize_count == 1
    assert len(embedding.document_calls) == 1
    assert len(embedding.document_calls[0]) == 101
    assert first == second == pipeline.initialization_timings
    assert all(value >= 0 for value in first.model_dump().values())
    await pipeline.close()


@pytest.mark.asyncio
async def test_reader_initialization_does_not_embed_or_upsert_the_formal_corpus() -> None:
    seed_pipeline, _, _, store = make_pipeline()
    await seed_pipeline.initialize()
    await seed_pipeline.close()

    reader, embedding, reranker, _ = make_pipeline(store=store)
    await reader.initialize(ingest_corpus=False)
    first = await reader.retrieve("产品 A 原装电池保修期限")
    second = await reader.retrieve("采购合同审批要求")

    assert reader.last_ingestion_result is None
    assert embedding.document_calls == []
    assert len(embedding.query_calls) == 2
    assert reranker.initialize_count == 1
    assert first.evidence_context.evidence_items
    assert second.evidence_context.evidence_items
    assert await store.count() == 101
    await reader.close()


@pytest.mark.asyncio
async def test_pipeline_accepts_exact_logical_corpus_when_physical_stats_are_606() -> None:
    store = LogicalCorpusStore(dimension=128, physical_row_count=606)
    pipeline, _, _, _ = make_pipeline(store=store)

    await pipeline.initialize()

    assert pipeline.child_count == 101
    assert store.list_record_ids_calls == 1
    assert store.count_calls == 0
    assert len(store.last_physical_ids) == 101
    await pipeline.close()


@pytest.mark.asyncio
async def test_pipeline_rejects_missing_expected_record_id() -> None:
    def omit_one(record_ids: frozenset[str]) -> tuple[str, ...]:
        return tuple(sorted(record_ids)[1:])

    store = LogicalCorpusStore(dimension=128, physical_ids_factory=omit_one)
    pipeline, _, _, _ = make_pipeline(store=store)

    with pytest.raises(
        RetrievalValidationError,
        match=r"^pipeline vector store must contain exactly the generated child corpus$",
    ):
        await pipeline.initialize()
    assert pipeline.initialized is False


@pytest.mark.asyncio
async def test_pipeline_rejects_extra_stale_record_id() -> None:
    def add_stale(record_ids: frozenset[str]) -> tuple[str, ...]:
        return (*sorted(record_ids), "stale-record")

    store = LogicalCorpusStore(dimension=128, physical_ids_factory=add_stale)
    pipeline, _, _, _ = make_pipeline(store=store)

    with pytest.raises(
        RetrievalValidationError,
        match=r"^pipeline vector store must contain exactly the generated child corpus$",
    ):
        await pipeline.initialize()
    assert pipeline.initialized is False


@pytest.mark.asyncio
async def test_pipeline_rejects_same_count_with_different_record_ids() -> None:
    def replace_one(record_ids: frozenset[str]) -> tuple[str, ...]:
        expected = sorted(record_ids)
        return (*expected[1:], "different-record")

    store = LogicalCorpusStore(dimension=128, physical_ids_factory=replace_one)
    pipeline, _, _, _ = make_pipeline(store=store)

    with pytest.raises(
        RetrievalValidationError,
        match=r"^pipeline vector store must contain exactly the generated child corpus$",
    ):
        await pipeline.initialize()
    assert len(store.last_physical_ids) == 101
    assert pipeline.initialized is False


@pytest.mark.asyncio
async def test_pipeline_accepts_duplicate_physical_versions_with_exact_logical_ids() -> None:
    def repeat_six_times(record_ids: frozenset[str]) -> tuple[str, ...]:
        ordered = tuple(sorted(record_ids))
        return ordered * 6

    store = LogicalCorpusStore(dimension=128, physical_ids_factory=repeat_six_times)
    pipeline, _, _, _ = make_pipeline(store=store)

    await pipeline.initialize()

    assert len(store.last_physical_ids) == 606
    assert len(frozenset(store.last_physical_ids)) == 101
    assert store.count_calls == 0
    await pipeline.close()


@pytest.mark.asyncio
async def test_retrieve_requires_initialize() -> None:
    pipeline, _, _, _ = make_pipeline()

    with pytest.raises(RetrievalValidationError, match="must be initialized"):
        await pipeline.retrieve("产品 A 电池保修")


@pytest.mark.asyncio
async def test_close_is_terminal() -> None:
    pipeline, _, _, _ = make_pipeline()
    await pipeline.initialize()
    await pipeline.close()
    await pipeline.close()

    assert pipeline.closed is True
    assert pipeline.initialized is False
    with pytest.raises(RetrievalValidationError, match="closed"):
        await pipeline.initialize()
    with pytest.raises(RetrievalValidationError, match="closed"):
        await pipeline.retrieve("产品 A 电池保修")


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["", "   "])
async def test_blank_query_is_rejected(initialized_pipeline, query: str) -> None:
    pipeline, _, _, _ = initialized_pipeline

    with pytest.raises(RetrievalValidationError, match="cannot be empty"):
        await pipeline.retrieve(query)


@pytest.mark.asyncio
async def test_formal_data_runs_the_complete_offline_chain(initialized_pipeline) -> None:
    pipeline, embedding, reranker, _ = initialized_pipeline

    result = await pipeline.retrieve("产品 A 原装电池保修期限")

    assert result.query == "产品 A 原装电池保修期限"
    assert result.dense_results
    assert result.bm25_results
    assert result.fused_results
    assert result.reranked_child_results
    assert result.expanded_parent_results
    assert result.evidence_context.evidence_items
    assert len(embedding.query_calls) == 1
    assert len(reranker.calls) == 1


@pytest.mark.asyncio
async def test_document_scope_filters_dense_bm25_before_fusion_and_evidence(
    initialized_pipeline,
) -> None:
    pipeline, _, reranker, _ = initialized_pipeline

    result = await pipeline.retrieve(
        "这是什么企业",
        allowed_document_ids=frozenset({"DOC-ORG-001"}),
    )

    for stage in (
        result.dense_results,
        result.bm25_results,
        result.fused_results,
        result.reranked_child_results,
        result.expanded_parent_results,
    ):
        assert stage
        assert {item.document_id for item in stage} == {"DOC-ORG-001"}
    assert {item.document_id for item in result.evidence_context.evidence_items} == {"DOC-ORG-001"}
    assert {item.document_id for item in result.evidence_context.references} == {"DOC-ORG-001"}
    assert {candidate.document_id for candidate in reranker.calls[0][1]} == {"DOC-ORG-001"}


@pytest.mark.asyncio
async def test_empty_document_scope_returns_no_evidence_without_reranking(
    initialized_pipeline,
) -> None:
    pipeline, _, reranker, _ = initialized_pipeline

    result = await pipeline.retrieve(
        "这是什么企业",
        allowed_document_ids=frozenset({"DOC-NOT-AUTHORIZED"}),
    )

    assert result.dense_results == result.bm25_results == result.fused_results == ()
    assert result.reranked_child_results == result.expanded_parent_results == ()
    assert result.evidence_context.evidence_items == ()
    assert result.evidence_context.references == ()
    assert reranker.calls == []


@pytest.mark.asyncio
async def test_enterprise_profile_and_agent_queries_retrieve_their_authoritative_documents(
    initialized_pipeline,
) -> None:
    pipeline, _, _, _ = initialized_pipeline

    enterprise = await pipeline.retrieve("这是什么企业")
    capabilities = await pipeline.retrieve("这个 Agent 可以做什么")
    inventory = await pipeline.retrieve("产品 A 的安全库存线是多少")

    assert enterprise.evidence_context.references[0].document_id == "DOC-ORG-001"
    assert "DOC-AGENT-001" in {
        reference.document_id for reference in capabilities.evidence_context.references
    }
    assert inventory.evidence_context.references[0].document_id == "DOC-INV-001"


@pytest.mark.asyncio
async def test_dense_and_bm25_results_use_child_identity(initialized_pipeline) -> None:
    pipeline, _, _, _ = initialized_pipeline
    result = await pipeline.retrieve("产品 A 原装电池保修期限")

    for candidate in (*result.dense_results, *result.bm25_results):
        assert candidate.candidate_id.startswith("child_")
        assert candidate.record_id == candidate.candidate_id
        assert candidate.parent_id.startswith("parent_")
        assert candidate.document_id.startswith("DOC-")
        assert candidate.metadata["parent_id"] == candidate.parent_id
        assert candidate.metadata["source"] == candidate.source
        assert candidate.metadata["start_offset"] == candidate.provenance["start_offset"]
        assert candidate.metadata["parser_name"] == "MarkdownDocumentParser"


@pytest.mark.asyncio
async def test_multiple_queries_reuse_indexes_and_models(initialized_pipeline) -> None:
    pipeline, embedding, reranker, store = initialized_pipeline

    await pipeline.retrieve("产品 A 原装电池保修期限")
    await pipeline.retrieve("采购合同审批要求")

    assert len(embedding.document_calls) == 1
    assert len(embedding.query_calls) == 2
    assert embedding.initialize_count == reranker.initialize_count == 1
    assert len(reranker.calls) == 2
    assert await store.count() == 101


@pytest.mark.asyncio
async def test_rrf_keeps_source_rank_score_and_contribution(initialized_pipeline) -> None:
    pipeline, _, _, _ = initialized_pipeline
    result = await pipeline.retrieve("产品 A 原装电池保修期限")

    assert all(item.candidate_id for item in result.fused_results)
    for fused in result.fused_results:
        assert fused.matched_source_count == len(fused.source_contributions)
        assert {item.source_name for item in fused.source_contributions} <= {"dense", "bm25"}
        assert all(item.source_rank > 0 for item in fused.source_contributions)
        assert all(item.contribution > 0 for item in fused.source_contributions)


@pytest.mark.asyncio
async def test_reranker_receives_one_batch_of_authoritative_child_content(
    initialized_pipeline,
) -> None:
    pipeline, _, reranker, _ = initialized_pipeline
    result = await pipeline.retrieve("产品 A 原装电池保修期限")
    query, candidates, top_k = reranker.calls[0]
    fused_content = {item.candidate_id: item.content for item in result.fused_results}

    assert query == result.query
    assert len(candidates) == result.pipeline_config.reranker_candidate_count
    assert top_k == result.pipeline_config.reranker_top_k
    assert all(item.content == fused_content[item.candidate_id] for item in candidates)


@pytest.mark.asyncio
async def test_all_formal_children_survive_child_level_rrf_and_parent_aggregation() -> None:
    config = RetrievalPipelineConfig(
        dense_top_k=101,
        bm25_top_k=101,
        rrf_top_k=101,
        reranker_candidate_count=101,
        reranker_top_k=101,
        parent_top_k=36,
        evidence_max_count=5,
    )
    pipeline, _, _, _ = make_pipeline(config=config)
    await pipeline.initialize()

    result = await pipeline.retrieve("企业")

    assert len(result.fused_results) == 101
    assert len({item.candidate_id for item in result.fused_results}) == 101
    assert len(result.expanded_parent_results) == 36
    assert sum(item.matched_child_count for item in result.expanded_parent_results) == 101
    assert any(item.matched_child_count > 1 for item in result.expanded_parent_results)
    await pipeline.close()


@pytest.mark.asyncio
async def test_same_document_children_are_not_collapsed_by_rrf() -> None:
    config = RetrievalPipelineConfig(
        dense_top_k=101,
        bm25_top_k=101,
        rrf_top_k=101,
        reranker_candidate_count=101,
        reranker_top_k=101,
        parent_top_k=36,
    )
    pipeline, _, _, _ = make_pipeline(config=config)
    await pipeline.initialize()

    result = await pipeline.retrieve("企业")
    document_counts = Counter(item.document_id for item in result.fused_results)

    assert max(document_counts.values()) > 1
    repeated_document = next(key for key, count in document_counts.items() if count > 1)
    candidate_ids = {
        item.candidate_id for item in result.fused_results if item.document_id == repeated_document
    }
    assert len(candidate_ids) == document_counts[repeated_document]
    await pipeline.close()


@pytest.mark.asyncio
async def test_shared_reranker_accepts_distinct_children_from_one_document() -> None:
    model = FakeCrossEncoderModel()
    reranker = SentenceTransformerCrossEncoderReranker(model=model)
    candidates = [
        RerankCandidate(
            candidate_id=f"child_{index}",
            record_id=f"child_{index}",
            document_id="DOC-SAME",
            content=f"child content {index}",
            upstream_rank=index,
        )
        for index in (1, 2)
    ]

    results = await reranker.rerank("query", candidates)

    assert [item.candidate_id for item in results] == ["child_1", "child_2"]
    assert len(model.calls) == 1
    assert model.calls[0] == [("query", "child content 1"), ("query", "child content 2")]


@pytest.mark.asyncio
async def test_evidence_ids_and_references_trace_to_parent_and_children(
    initialized_pipeline,
) -> None:
    pipeline, _, _, _ = initialized_pipeline
    result = await pipeline.retrieve("产品 A 原装电池保修期限")
    context = result.evidence_context

    assert [item.evidence_id for item in context.evidence_items] == [
        f"E{index}" for index in range(1, context.included_evidence_count + 1)
    ]
    assert [item.evidence_id for item in context.evidence_items] == [
        item.evidence_id for item in context.references
    ]
    for item, reference in zip(context.evidence_items, context.references, strict=True):
        assert item.parent_id == reference.parent_id
        assert item.document_id == reference.document_id
        assert all(child.parent_id == item.parent_id for child in item.matched_children)
        assert reference.source is not None
        assert reference.start_offset is not None
        assert reference.end_offset is not None
        assert f"[{item.evidence_id}]" in context.rendered_context


@pytest.mark.asyncio
async def test_query_timings_are_complete_and_exclude_initialization(initialized_pipeline) -> None:
    pipeline, _, _, _ = initialized_pipeline
    initialization = pipeline.initialization_timings
    result = await pipeline.retrieve("产品 A 原装电池保修期限")

    assert initialization is not None
    assert set(result.stage_timings.model_dump()) == {
        "dense_query_seconds",
        "bm25_query_seconds",
        "rrf_seconds",
        "reranker_seconds",
        "parent_expansion_seconds",
        "evidence_context_seconds",
        "total_runtime_seconds",
    }
    assert all(value >= 0 for value in result.stage_timings.model_dump().values())
    assert "model_load_seconds" not in result.stage_timings.model_dump()
    assert result.total_runtime_seconds == result.stage_timings.total_runtime_seconds
    stage_total = sum(
        value
        for name, value in result.stage_timings.model_dump().items()
        if name != "total_runtime_seconds"
    )
    assert result.total_runtime_seconds >= stage_total


@pytest.mark.asyncio
async def test_result_mutation_does_not_change_pipeline_snapshot(initialized_pipeline) -> None:
    pipeline, _, _, _ = initialized_pipeline
    first = await pipeline.retrieve("产品 A 原装电池保修期限")
    original_category = first.dense_results[0].metadata.get("category")

    first.dense_results[0].metadata["category"] = "mutated"
    first.evidence_context.evidence_items[0].metadata["category"] = "mutated"
    second = await pipeline.retrieve("产品 A 原装电池保修期限")

    assert second.dense_results[0].metadata.get("category") == original_category
    assert second.evidence_context.evidence_items[0].metadata.get("category") != "mutated"


@pytest.mark.asyncio
async def test_runtime_ignores_missing_or_replaced_ground_truth(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    generated.mkdir()
    for name in ("parent_chunks.jsonl", "child_chunks.jsonl"):
        shutil.copyfile(DATASET_ROOT / "generated" / name, generated / name)
    pipeline, _, _, _ = make_pipeline(dataset_root=tmp_path)

    await pipeline.initialize()
    without_ground_truth = await pipeline.retrieve("产品 A 原装电池保修期限")

    assert without_ground_truth.evidence_context.included_evidence_count > 0
    assert not (generated / "retrieval_ground_truth.jsonl").exists()
    await pipeline.close()

    (generated / "retrieval_ground_truth.jsonl").write_text(
        "this is deliberately not JSONL", encoding="utf-8"
    )
    (generated / "clause_chunk_map.jsonl").write_text(
        "this is deliberately not JSONL", encoding="utf-8"
    )
    replaced_pipeline, _, _, _ = make_pipeline(dataset_root=tmp_path)
    await replaced_pipeline.initialize()
    with_replaced_ground_truth = await replaced_pipeline.retrieve("产品 A 原装电池保修期限")

    assert with_replaced_ground_truth.dense_results == without_ground_truth.dense_results
    assert with_replaced_ground_truth.bm25_results == without_ground_truth.bm25_results
    assert with_replaced_ground_truth.fused_results == without_ground_truth.fused_results
    assert (
        with_replaced_ground_truth.reranked_child_results
        == without_ground_truth.reranked_child_results
    )
    assert (
        with_replaced_ground_truth.expanded_parent_results
        == without_ground_truth.expanded_parent_results
    )
    assert with_replaced_ground_truth.evidence_context == without_ground_truth.evidence_context
    await replaced_pipeline.close()


@pytest.mark.asyncio
async def test_formal_generated_files_are_not_modified(initialized_pipeline) -> None:
    pipeline, _, _, _ = initialized_pipeline
    paths = [
        DATASET_ROOT / "generated/parent_chunks.jsonl",
        DATASET_ROOT / "generated/child_chunks.jsonl",
    ]
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]

    await pipeline.retrieve("产品 A 原装电池保修期限")

    assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths] == before


@pytest.mark.asyncio
async def test_missing_generated_data_fails_clearly(tmp_path: Path) -> None:
    pipeline, _, _, _ = make_pipeline(dataset_root=tmp_path)

    with pytest.raises(RetrievalValidationError, match=r"parent_chunks\.jsonl"):
        await pipeline.initialize()
    assert pipeline.initialized is False


def test_cli_import_and_parser_do_not_construct_models() -> None:
    from scripts.run_enterprise_retrieval_demo import build_parser

    args = build_parser().parse_args(["--query", "测试问题"])

    assert args.query == "测试问题"
    assert args.dataset_root == DATASET_ROOT
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["--query", "   "])
    assert exc_info.value.code == 2
