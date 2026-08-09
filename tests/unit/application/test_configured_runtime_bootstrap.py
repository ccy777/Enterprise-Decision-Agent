"""Offline coverage for the production-configured runtime composition root."""

from __future__ import annotations

import ast
import asyncio
import threading
from collections.abc import Sequence
from pathlib import Path

import pytest

import decision_agent.application.configured_runtime as configured_runtime
from decision_agent.agents.answerability_reviewer import (
    OpenAICompatibleAnswerabilityReviewer,
)
from decision_agent.agents.data_answer_generator import OpenAICompatibleDataAnswerGenerator
from decision_agent.agents.data_query_planner import OpenAICompatibleDataQueryPlanner
from decision_agent.agents.evidence_selector import OpenAICompatibleEvidenceSelector
from decision_agent.agents.grounded_answer import OpenAICompatibleAnswerGenerator
from decision_agent.application.bootstrap import (
    RuntimeBootstrapError,
    build_bootstrapped_runtime,
)
from decision_agent.application.configured_runtime import (
    ConfiguredProviderAdapters,
    ConfiguredRuntimeDependencies,
    create_configured_runtime_builder,
)
from decision_agent.application.executor import FormalRequestExecutor
from decision_agent.config import Environment, Settings
from decision_agent.coordination.coordinator import Coordinator
from decision_agent.memory.in_memory import InMemorySessionMemoryStore
from decision_agent.memory.redis_store import RedisSessionMemoryStore
from decision_agent.memory.summarization import ProviderRollingSummarizer
from decision_agent.observability import BestEffortTraceDispatcher, StructuredLoggingTraceSink
from decision_agent.retrieval.factory import (
    EnterpriseRetrievalRuntime,
    ProductionRetrievalDependencies,
    build_enterprise_retrieval_pipeline,
    build_production_retrieval_runtime,
)
from decision_agent.retrieval.in_memory_store import InMemoryVectorStore
from decision_agent.retrieval.milvus_store import MilvusVectorStore
from decision_agent.retrieval.reranking import RerankCandidate, RerankedResult
from decision_agent.routing.request_router import OpenAICompatibleRequestRouter
from decision_agent.security import DataClassification
from decision_agent.skills.enterprise_data_analysis import EnterpriseDataAnalysisSkill
from decision_agent.skills.enterprise_knowledge_qa import EnterpriseKnowledgeQASkill
from decision_agent.skills.inventory_risk_diagnosis import InventoryRiskDiagnosisSkill
from decision_agent.skills.native_runtime import (
    NativeToolCallingSkillExecutor,
    PreselectedAgentToolSkillExecutor,
)
from decision_agent.tool_calling.runtime import OpenAICompatibleNativeToolCallingModel
from decision_agent.tool_calling.tools import DataAgentTool, KnowledgeAgentTool

pytestmark = pytest.mark.offline_integration

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _isolate_settings_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for field_name in Settings.model_fields:
        monkeypatch.delenv(f"DECISION_AGENT_{field_name.upper()}", raising=False)


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "app_name": "Configured Runtime Unit",
        "environment": Environment.TEST,
        "llm_api_key": "unit-llm-secret",
        "llm_base_url": "https://provider.invalid/v1",
        "llm_model_name": "unit-model",
        "knowledge_dataset_root": ROOT / "datasets/enterprise_kb/m2c1",
        "db_readonly_password": "unit-db-secret",
        "memory_mode": "disabled",
    }
    values.update(overrides)
    return Settings(**values, _env_file=None)


class _ProviderBoundary:
    async def route(self, **_: object) -> object:
        raise AssertionError("Unit composition must not execute the Router")

    async def route_with_context(self, **_: object) -> object:
        raise AssertionError("Unit composition must not execute the Router")

    async def complete(self, **_: object) -> object:
        raise AssertionError("Unit composition must not call the Provider")

    async def complete_chat(self, **_: object) -> object:
        raise AssertionError("Unit composition must not call the Provider")

    async def select(self, **_: object) -> object:
        raise AssertionError("Unit composition must not execute Knowledge")

    async def review(self, **_: object) -> object:
        raise AssertionError("Unit composition must not execute Knowledge")

    async def generate(self, **_: object) -> object:
        raise AssertionError("Unit composition must not execute an Agent")

    async def plan(self, **_: object) -> object:
        raise AssertionError("Unit composition must not execute Data")


def _provider_bundle(native: _ProviderBoundary | None = None) -> ConfiguredProviderAdapters:
    shared = native or _ProviderBoundary()
    return ConfiguredProviderAdapters(
        router=shared,  # type: ignore[arg-type]
        native_model=shared,  # type: ignore[arg-type]
        evidence_selector=shared,  # type: ignore[arg-type]
        answerability_reviewer=shared,  # type: ignore[arg-type]
        answer_generator=shared,  # type: ignore[arg-type]
        data_query_planner=shared,  # type: ignore[arg-type]
        data_answer_generator=shared,  # type: ignore[arg-type]
    )


class _PipelineBoundary:
    async def retrieve(self, _: str) -> object:
        raise AssertionError("Unit composition must not retrieve")


class _KnowledgeRuntimeBoundary:
    def __init__(
        self,
        events: list[str],
        *,
        name: str = "knowledge",
        initialize_error: BaseException | None = None,
    ) -> None:
        self.pipeline = _PipelineBoundary()
        self.events = events
        self.name = name
        self.initialize_error = initialize_error
        self.initialize_calls = 0
        self.close_calls = 0

    async def initialize(self) -> None:
        self.initialize_calls += 1
        self.events.append(f"{self.name}:initialize")
        if self.initialize_error is not None:
            raise self.initialize_error

    async def aclose(self) -> None:
        self.close_calls += 1
        self.events.append(f"{self.name}:close")


class _MCPClientBoundary:
    def __init__(self, events: list[str], identity: int, *, fail_enter: bool = False) -> None:
        self.events = events
        self.identity = identity
        self.fail_enter = fail_enter
        self.enter_calls = 0
        self.exit_calls = 0

    async def __aenter__(self) -> _MCPClientBoundary:
        self.enter_calls += 1
        self.events.append(f"mcp:{self.identity}:enter")
        if self.fail_enter:
            raise RuntimeError("PRIVATE_MCP_FAILURE")
        return self

    async def __aexit__(self, *_: object) -> None:
        self.exit_calls += 1
        self.events.append(f"mcp:{self.identity}:exit")

    async def get_enterprise_schema(self) -> object:
        raise AssertionError("Preflight must not execute data operations")

    async def get_business_definitions(self) -> object:
        raise AssertionError("Preflight must not execute data operations")

    async def execute_safe_query(self, _: str) -> object:
        raise AssertionError("Preflight must not execute SQL")


class _RedisClientBoundary:
    def __init__(self, events: list[str], *, ping_error: RuntimeError | None = None) -> None:
        self.events = events
        self.ping_error = ping_error
        self.ping_calls = 0
        self.close_calls = 0
        self.ping_thread: int | None = None

    def ping(self) -> bool:
        self.ping_calls += 1
        self.ping_thread = threading.get_ident()
        self.events.append("redis:ping")
        if self.ping_error is not None:
            raise self.ping_error
        return True

    def close(self) -> None:
        self.close_calls += 1
        self.events.append("redis:close")


class _BoundaryFactories:
    def __init__(
        self,
        *,
        knowledge_error: BaseException | None = None,
        mcp_error: bool = False,
        redis_error: RuntimeError | None = None,
    ) -> None:
        self.events: list[str] = []
        self.provider_calls: list[Settings] = []
        self.knowledge_calls: list[Settings] = []
        self.mcp_calls: list[Settings] = []
        self.redis_calls: list[tuple[str, float]] = []
        self.knowledge_runtimes: list[_KnowledgeRuntimeBoundary] = []
        self.mcp_clients: list[_MCPClientBoundary] = []
        self.redis_clients: list[_RedisClientBoundary] = []
        self.bundle = _provider_bundle()
        self.knowledge_error = knowledge_error
        self.mcp_error = mcp_error
        self.redis_error = redis_error

    def providers(self, settings: Settings) -> ConfiguredProviderAdapters:
        self.provider_calls.append(settings)
        self.events.append("providers:create")
        return self.bundle

    def knowledge(self, settings: Settings) -> EnterpriseRetrievalRuntime:
        self.knowledge_calls.append(settings)
        self.events.append("knowledge:create")
        runtime = _KnowledgeRuntimeBoundary(
            self.events,
            name=f"knowledge:{len(self.knowledge_runtimes) + 1}",
            initialize_error=self.knowledge_error,
        )
        self.knowledge_runtimes.append(runtime)
        return runtime  # type: ignore[return-value]

    def mcp(self, settings: Settings) -> _MCPClientBoundary:
        self.mcp_calls.append(settings)
        client = _MCPClientBoundary(
            self.events,
            len(self.mcp_clients) + 1,
            fail_enter=self.mcp_error,
        )
        self.mcp_clients.append(client)
        return client

    def redis(self, url: str, timeout_seconds: float) -> _RedisClientBoundary:
        self.redis_calls.append((url, timeout_seconds))
        self.events.append("redis:create")
        client = _RedisClientBoundary(self.events, ping_error=self.redis_error)
        self.redis_clients.append(client)
        return client

    def dependencies(self) -> ConfiguredRuntimeDependencies:
        return ConfiguredRuntimeDependencies(
            provider_adapters_factory=self.providers,
            knowledge_runtime_factory=self.knowledge,
            enterprise_data_client_factory=self.mcp,
            redis_client_factory=self.redis,
        )


async def _build(
    factories: _BoundaryFactories,
    settings: Settings | None = None,
):
    builder = create_configured_runtime_builder(
        settings or _settings(),
        factories.dependencies(),
    )
    return await build_bootstrapped_runtime(builder)


def _runtime_parts(
    executor: FormalRequestExecutor,
) -> tuple[
    Coordinator,
    NativeToolCallingSkillExecutor,
    KnowledgeAgentTool,
    DataAgentTool,
    InventoryRiskDiagnosisSkill,
]:
    coordinator = executor._coordinator  # type: ignore[attr-defined]
    knowledge_skill = coordinator._registry.get("enterprise-knowledge-qa")  # type: ignore[attr-defined]
    data_skill = coordinator._registry.get("enterprise-data-analysis")  # type: ignore[attr-defined]
    mixed_skill = coordinator._registry.get("inventory-risk-diagnosis")  # type: ignore[attr-defined]
    native = knowledge_skill._runtime  # type: ignore[attr-defined]
    assert data_skill._runtime is native  # type: ignore[attr-defined]
    return (
        coordinator,
        native,
        native._knowledge_tool,  # type: ignore[attr-defined]
        native._data_tool,  # type: ignore[attr-defined]
        mixed_skill,
    )


def test_dependency_aggregates_have_only_the_fixed_typed_fields() -> None:
    assert tuple(ProductionRetrievalDependencies.__dataclass_fields__) == (
        "embedding_provider_factory",
        "reranker_factory",
        "vector_store_factory",
    )
    assert tuple(ConfiguredProviderAdapters.__dataclass_fields__) == (
        "router",
        "native_model",
        "evidence_selector",
        "answerability_reviewer",
        "answer_generator",
        "data_query_planner",
        "data_answer_generator",
    )
    assert tuple(ConfiguredRuntimeDependencies.__dataclass_fields__) == (
        "provider_adapters_factory",
        "knowledge_runtime_factory",
        "enterprise_data_client_factory",
        "redis_client_factory",
    )


def test_creating_configured_builder_defers_all_factories_and_io() -> None:
    factories = _BoundaryFactories()

    builder = create_configured_runtime_builder(_settings(), factories.dependencies())

    assert callable(builder)
    assert factories.events == []
    assert factories.provider_calls == factories.knowledge_calls == factories.mcp_calls == []
    assert factories.redis_calls == []


def test_default_builder_creation_is_deferred_and_uses_formal_factories() -> None:
    builder = create_configured_runtime_builder(_settings())
    defaults = configured_runtime._default_configured_runtime_dependencies()

    assert callable(builder)
    assert defaults.knowledge_runtime_factory is build_production_retrieval_runtime
    assert defaults.enterprise_data_client_factory.__name__ == "from_settings"
    assert defaults.redis_client_factory is configured_runtime._build_redis_client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "settings",
    [
        _settings(llm_api_key=None, llm_base_url=None, llm_model_name=None),
        _settings(knowledge_dataset_root=None),
        _settings(db_readonly_password=None),
    ],
    ids=["llm-trio", "dataset-root", "database-password"],
)
async def test_incomplete_runtime_configuration_fails_before_factories(
    settings: Settings,
) -> None:
    factories = _BoundaryFactories()

    with pytest.raises(RuntimeBootstrapError) as raised:
        await _build(factories, settings)

    assert raised.value.code == "bootstrap_configuration_invalid"
    assert str(raised.value) == "bootstrap_configuration_invalid"
    assert factories.events == []


@pytest.mark.asyncio
async def test_configuration_error_never_renders_secret_url_or_dataset_path() -> None:
    factories = _BoundaryFactories()
    settings = _settings(
        db_readonly_password=None,
        llm_api_key="SECRET_CONFIG_MARKER",
        llm_base_url="https://private.invalid/v1",
        knowledge_dataset_root=Path("PRIVATE_DATASET_MARKER"),
    )

    with pytest.raises(RuntimeBootstrapError) as raised:
        await _build(factories, settings)

    rendered = f"{raised.value!s} {raised.value!r}".lower()
    assert rendered == (
        "bootstrap_configuration_invalid runtimebootstraperror('bootstrap_configuration_invalid')"
    )
    assert all(
        marker not in rendered
        for marker in ("secret_config_marker", "private.invalid", "private_dataset_marker")
    )


def test_default_provider_factory_builds_each_formal_adapter_and_one_native_model() -> None:
    adapters = configured_runtime._build_provider_adapters(_settings())

    assert isinstance(adapters.router, OpenAICompatibleRequestRouter)
    assert isinstance(adapters.native_model, OpenAICompatibleNativeToolCallingModel)
    assert isinstance(adapters.evidence_selector, OpenAICompatibleEvidenceSelector)
    assert isinstance(adapters.answerability_reviewer, OpenAICompatibleAnswerabilityReviewer)
    assert isinstance(adapters.answer_generator, OpenAICompatibleAnswerGenerator)
    assert isinstance(adapters.data_query_planner, OpenAICompatibleDataQueryPlanner)
    assert isinstance(adapters.data_answer_generator, OpenAICompatibleDataAnswerGenerator)


@pytest.mark.asyncio
async def test_success_builds_formal_executor_coordinator_skills_and_tools() -> None:
    factories = _BoundaryFactories()
    runtime = await _build(factories)
    coordinator, native, knowledge_tool, data_tool, mixed_skill = _runtime_parts(runtime.executor)

    assert isinstance(runtime.executor, FormalRequestExecutor)
    assert runtime.executor.requires_security_context is True
    dispatcher = runtime.executor._trace_dispatcher  # type: ignore[attr-defined]
    assert isinstance(dispatcher, BestEffortTraceDispatcher)
    assert len(dispatcher._sinks) == 1  # type: ignore[attr-defined]
    assert isinstance(dispatcher._sinks[0], StructuredLoggingTraceSink)  # type: ignore[attr-defined]
    assert isinstance(coordinator, Coordinator)
    assert isinstance(native, NativeToolCallingSkillExecutor)
    assert isinstance(knowledge_tool, KnowledgeAgentTool)
    assert isinstance(data_tool, DataAgentTool)
    assert isinstance(
        coordinator._registry.get("enterprise-knowledge-qa"),  # type: ignore[attr-defined]
        EnterpriseKnowledgeQASkill,
    )
    assert isinstance(
        coordinator._registry.get("enterprise-data-analysis"),  # type: ignore[attr-defined]
        EnterpriseDataAnalysisSkill,
    )
    assert isinstance(mixed_skill, InventoryRiskDiagnosisSkill)
    await runtime.aclose()


@pytest.mark.asyncio
async def test_controlled_mixed_uses_preselected_router_owned_child_tools() -> None:
    factories = _BoundaryFactories()
    runtime = await _build(
        factories,
        _settings(controlled_workflow_enabled=True),
    )
    coordinator, _, _, data_tool, mixed_skill = _runtime_parts(runtime.executor)

    data_runtime = mixed_skill._data_skill._runtime  # type: ignore[attr-defined]
    knowledge_runtime = mixed_skill._knowledge_skill._runtime  # type: ignore[attr-defined]
    assert isinstance(data_runtime, PreselectedAgentToolSkillExecutor)
    assert knowledge_runtime is data_runtime
    planner_provider = coordinator._controlled_workflow._planner._provider  # type: ignore[attr-defined]
    assert planner_provider._classification is DataClassification.INTERNAL  # type: ignore[attr-defined]
    assert data_tool._answer_generator._payload_projector is not None  # type: ignore[attr-defined]
    await runtime.aclose()


@pytest.mark.asyncio
async def test_startup_order_and_native_model_reuse_are_exact() -> None:
    factories = _BoundaryFactories()
    runtime = await _build(factories)
    _, native, _, _, mixed_skill = _runtime_parts(runtime.executor)

    assert factories.events[:5] == [
        "providers:create",
        "knowledge:create",
        "knowledge:1:initialize",
        "mcp:1:enter",
        "mcp:1:exit",
    ]
    assert len(factories.provider_calls) == 1
    assert native._model._client is factories.bundle.native_model  # type: ignore[attr-defined]
    assert mixed_skill._synthesizer._client._client is factories.bundle.native_model  # type: ignore[attr-defined]
    await runtime.aclose()


@pytest.mark.asyncio
async def test_knowledge_cleanup_is_registered_before_initialize_failure() -> None:
    factories = _BoundaryFactories(knowledge_error=RuntimeError("PRIVATE_KNOWLEDGE_FAILURE"))

    with pytest.raises(RuntimeBootstrapError) as raised:
        await _build(factories)

    assert raised.value.code == "bootstrap_runtime_unavailable"
    assert factories.events == [
        "providers:create",
        "knowledge:create",
        "knowledge:1:initialize",
        "knowledge:1:close",
    ]
    assert factories.knowledge_runtimes[0].close_calls == 1


@pytest.mark.asyncio
async def test_mcp_preflight_is_temporary_and_data_factory_is_request_scoped() -> None:
    factories = _BoundaryFactories()
    runtime = await _build(factories)
    _, _, _, data_tool, _ = _runtime_parts(runtime.executor)

    preflight = factories.mcp_clients[0]
    assert (preflight.enter_calls, preflight.exit_calls) == (1, 1)
    assert not hasattr(preflight, "list_tools")
    request_client = data_tool._enterprise_data_client_factory()  # type: ignore[attr-defined]
    assert request_client is factories.mcp_clients[1]
    assert request_client is not preflight
    assert factories.mcp_calls == [factories.mcp_calls[0], factories.mcp_calls[0]]
    await runtime.aclose()


@pytest.mark.asyncio
async def test_mcp_preflight_failure_rolls_back_knowledge_without_executor() -> None:
    factories = _BoundaryFactories(mcp_error=True)

    with pytest.raises(RuntimeBootstrapError) as raised:
        await _build(factories)

    assert raised.value.code == "bootstrap_runtime_unavailable"
    assert factories.knowledge_runtimes[0].close_calls == 1
    assert factories.mcp_clients[0].enter_calls == 1
    assert factories.mcp_clients[0].exit_calls == 0


@pytest.mark.asyncio
async def test_disabled_memory_creates_neither_store_redis_nor_summarizer() -> None:
    factories = _BoundaryFactories()
    runtime = await _build(factories, _settings(memory_mode="disabled"))

    assert runtime.executor._memory_store is None  # type: ignore[attr-defined]
    assert runtime.executor._rolling_summary_service is None  # type: ignore[attr-defined]
    assert factories.redis_calls == []
    await runtime.aclose()


@pytest.mark.asyncio
async def test_in_memory_maps_policy_without_redis_or_summary() -> None:
    factories = _BoundaryFactories()
    runtime = await _build(
        factories,
        _settings(memory_mode="in_memory", memory_ttl_seconds=321, memory_max_turns=9),
    )
    store = runtime.executor._memory_store  # type: ignore[attr-defined]

    assert isinstance(store, InMemorySessionMemoryStore)
    assert store._policy.ttl_seconds == 321  # type: ignore[attr-defined]
    assert store._policy.max_turns == 9  # type: ignore[attr-defined]
    assert runtime.executor._rolling_summary_service is None  # type: ignore[attr-defined]
    assert factories.redis_calls == []
    await runtime.aclose()


@pytest.mark.asyncio
async def test_summary_enabled_reuses_native_provider_and_maps_policy() -> None:
    factories = _BoundaryFactories()
    runtime = await _build(
        factories,
        _settings(
            memory_mode="in_memory",
            memory_summary_enabled=True,
            memory_summary_trigger_turns=5,
            memory_summary_retain_recent_turns=2,
            memory_summary_max_source_chars=1234,
            memory_summary_max_summary_chars=345,
        ),
    )
    service = runtime.executor._rolling_summary_service  # type: ignore[attr-defined]
    summarizer = service._summarizer  # type: ignore[attr-defined]
    policy = service._policy  # type: ignore[attr-defined]

    assert isinstance(summarizer, ProviderRollingSummarizer)
    assert summarizer._provider is factories.bundle.native_model  # type: ignore[attr-defined]
    assert (
        policy.trigger_turns,
        policy.retain_recent_turns,
        policy.max_source_chars,
        policy.max_summary_chars,
    ) == (5, 2, 1234, 345)
    await runtime.aclose()


@pytest.mark.asyncio
async def test_redis_unwraps_url_only_at_factory_and_pings_in_worker_thread() -> None:
    factories = _BoundaryFactories()
    caller_thread = threading.get_ident()
    runtime = await _build(
        factories,
        _settings(
            memory_mode="redis",
            memory_redis_url="redis://cache.invalid:6379/0",
            memory_redis_timeout_seconds=2.5,
        ),
    )
    client = factories.redis_clients[0]
    store = runtime.executor._memory_store  # type: ignore[attr-defined]

    assert factories.redis_calls == [("redis://cache.invalid:6379/0", 2.5)]
    assert client.ping_calls == 1
    assert client.ping_thread is not None and client.ping_thread != caller_thread
    assert isinstance(store, RedisSessionMemoryStore)
    assert store._client is client  # type: ignore[attr-defined]
    await runtime.aclose()
    assert client.close_calls == 1


@pytest.mark.asyncio
async def test_redis_close_is_registered_before_ping_and_failure_has_no_fallback() -> None:
    factories = _BoundaryFactories(redis_error=RuntimeError("PRIVATE_REDIS_FAILURE"))

    with pytest.raises(RuntimeBootstrapError) as raised:
        await _build(
            factories,
            _settings(
                memory_mode="redis",
                memory_redis_url="redis://cache.invalid:6379/0",
            ),
        )

    assert raised.value.code == "bootstrap_runtime_unavailable"
    assert factories.redis_clients[0].close_calls == 1
    assert factories.knowledge_runtimes[0].close_calls == 1
    assert factories.events[-3:] == ["redis:ping", "redis:close", "knowledge:1:close"]


@pytest.mark.asyncio
async def test_shutdown_lifo_closes_redis_before_knowledge_once() -> None:
    factories = _BoundaryFactories()
    runtime = await _build(
        factories,
        _settings(
            memory_mode="redis",
            memory_redis_url="redis://cache.invalid:6379/0",
        ),
    )

    await runtime.aclose()
    await runtime.aclose()

    assert factories.events[-2:] == ["redis:close", "knowledge:1:close"]
    assert factories.redis_clients[0].close_calls == 1
    assert factories.knowledge_runtimes[0].close_calls == 1


@pytest.mark.asyncio
async def test_two_builder_invocations_have_isolated_resources() -> None:
    factories = _BoundaryFactories()
    builder = create_configured_runtime_builder(_settings(), factories.dependencies())

    first = await build_bootstrapped_runtime(builder)
    second = await build_bootstrapped_runtime(builder)
    await first.aclose()

    assert len(factories.knowledge_runtimes) == 2
    assert factories.knowledge_runtimes[0].close_calls == 1
    assert factories.knowledge_runtimes[1].close_calls == 0
    assert factories.mcp_clients[0] is not factories.mcp_clients[1]
    await second.aclose()
    assert factories.knowledge_runtimes[1].close_calls == 1


@pytest.mark.asyncio
async def test_cancelled_startup_rolls_back_and_propagates() -> None:
    factories = _BoundaryFactories(knowledge_error=asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await _build(factories)

    assert factories.knowledge_runtimes[0].close_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("control_error", [KeyboardInterrupt(), SystemExit()])
async def test_process_control_startup_errors_roll_back_and_propagate(
    control_error: BaseException,
) -> None:
    factories = _BoundaryFactories(knowledge_error=control_error)

    with pytest.raises(type(control_error)):
        await _build(factories)

    assert factories.knowledge_runtimes[0].close_calls == 1


class _EmbeddingBoundary:
    dimension = 512

    async def embed_documents(self, _: Sequence[str]) -> list[list[float]]:
        raise AssertionError("Production factory construction must not embed")

    async def embed_query(self, _: str) -> list[float]:
        raise AssertionError("Production factory construction must not embed")


class _RerankerBoundary:
    async def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        *,
        top_k: int | None = None,
    ) -> list[RerankedResult]:
        del query, candidates, top_k
        raise AssertionError("Production factory construction must not rerank")


class _VectorStoreBoundary:
    dimension = 512

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.initialize_calls = 0
        self.close_calls = 0

    async def initialize(self) -> None:
        self.initialize_calls += 1
        self.events.append("milvus:initialize")

    async def close(self) -> None:
        self.close_calls += 1
        self.events.append("milvus:close")


class _OwnedPipelineBoundary:
    def __init__(self, events: list[str], *, close_error: RuntimeError | None = None) -> None:
        self.events = events
        self.close_error = close_error

    async def initialize(self, *, ingest_corpus: bool = True) -> None:
        self.events.append(f"pipeline:initialize:{ingest_corpus}")

    async def close(self) -> None:
        self.events.append("pipeline:close")
        if self.close_error is not None:
            raise self.close_error


def test_production_retrieval_factory_is_io_free_and_supports_typed_dependencies() -> None:
    events: list[str] = []
    embedding = _EmbeddingBoundary()
    reranker = _RerankerBoundary()
    store = _VectorStoreBoundary(events)
    calls: list[str] = []
    dependencies = ProductionRetrievalDependencies(
        embedding_provider_factory=lambda _: calls.append("embedding") or embedding,
        reranker_factory=lambda _: calls.append("reranker") or reranker,
        vector_store_factory=lambda _: calls.append("store") or store,  # type: ignore[arg-type]
    )

    runtime = build_production_retrieval_runtime(_settings(), dependencies)

    assert calls == ["embedding", "reranker", "store"]
    assert events == []
    assert runtime._embedding_provider is embedding  # type: ignore[attr-defined]
    assert runtime._reranker is reranker  # type: ignore[attr-defined]
    assert runtime._vector_store is store  # type: ignore[attr-defined]
    assert runtime.pipeline._config.config_version == "m2c2a1-v1"  # type: ignore[attr-defined]


def test_default_production_retrieval_uses_formal_milvus_without_client_or_fallback() -> None:
    settings = _settings()
    runtime = build_production_retrieval_runtime(settings)

    assert isinstance(runtime._vector_store, MilvusVectorStore)  # type: ignore[attr-defined]
    assert runtime._vector_store._client is None  # type: ignore[attr-defined]
    assert not isinstance(runtime._vector_store, InMemoryVectorStore)  # type: ignore[attr-defined]
    assert runtime._embedding_provider._revision == settings.embedding_model_revision  # type: ignore[attr-defined]
    assert runtime._reranker.model_revision == settings.reranker_model_revision  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_retrieval_runtime_initializes_milvus_before_pipeline() -> None:
    events: list[str] = []
    store = _VectorStoreBoundary(events)
    pipeline = _OwnedPipelineBoundary(events)
    runtime = EnterpriseRetrievalRuntime(
        pipeline=pipeline,  # type: ignore[arg-type]
        vector_store=store,  # type: ignore[arg-type]
        embedding_provider=_EmbeddingBoundary(),
        reranker=_RerankerBoundary(),
    )

    await runtime.initialize()

    assert events == ["milvus:initialize", "pipeline:initialize:False"]


@pytest.mark.asyncio
async def test_retrieval_runtime_requires_explicit_ingestion_entry() -> None:
    events: list[str] = []
    store = _VectorStoreBoundary(events)
    pipeline = _OwnedPipelineBoundary(events)
    runtime = EnterpriseRetrievalRuntime(
        pipeline=pipeline,  # type: ignore[arg-type]
        vector_store=store,  # type: ignore[arg-type]
        embedding_provider=_EmbeddingBoundary(),
        reranker=_RerankerBoundary(),
    )

    await runtime.initialize_for_ingestion()

    assert events == ["milvus:initialize", "pipeline:initialize:True"]


@pytest.mark.asyncio
async def test_retrieval_runtime_closes_pipeline_before_milvus() -> None:
    events: list[str] = []
    store = _VectorStoreBoundary(events)
    runtime = EnterpriseRetrievalRuntime(
        pipeline=_OwnedPipelineBoundary(events),  # type: ignore[arg-type]
        vector_store=store,  # type: ignore[arg-type]
        embedding_provider=_EmbeddingBoundary(),
        reranker=_RerankerBoundary(),
    )

    await runtime.aclose()

    assert events == ["pipeline:close", "milvus:close"]


@pytest.mark.asyncio
async def test_retrieval_runtime_still_closes_milvus_when_pipeline_close_fails() -> None:
    events: list[str] = []
    store = _VectorStoreBoundary(events)
    runtime = EnterpriseRetrievalRuntime(
        pipeline=_OwnedPipelineBoundary(
            events,
            close_error=RuntimeError("PIPELINE_CLOSE_FAILURE"),
        ),  # type: ignore[arg-type]
        vector_store=store,  # type: ignore[arg-type]
        embedding_provider=_EmbeddingBoundary(),
        reranker=_RerankerBoundary(),
    )

    with pytest.raises(RuntimeError, match="PIPELINE_CLOSE_FAILURE"):
        await runtime.aclose()

    assert events == ["pipeline:close", "milvus:close"]


def test_baseline_retrieval_factory_remains_in_memory_and_unchanged() -> None:
    pipeline = build_enterprise_retrieval_pipeline("unchanged-baseline-root")

    assert isinstance(pipeline._vector_store, InMemoryVectorStore)  # type: ignore[attr-defined]
    assert pipeline._dataset_root == Path("unchanged-baseline-root")  # type: ignore[attr-defined]
    assert pipeline._config.config_version == "m2c2a1-v1"  # type: ignore[attr-defined]


def test_main_ast_has_one_deferred_bootstrapped_app_expression() -> None:
    source = (ROOT / "src/decision_agent/main.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    assignments = [
        node for node in module.body if isinstance(node, ast.Assign) and len(node.targets) == 1
    ]
    by_name = {
        target.id: node.value
        for node in assignments
        if isinstance((target := node.targets[0]), ast.Name)
    }

    assert set(by_name) == {"settings", "runtime_builder", "app"}
    assert isinstance(by_name["settings"], ast.Call)
    assert isinstance(by_name["settings"].func, ast.Name)
    assert by_name["settings"].func.id == "Settings"
    assert isinstance(by_name["runtime_builder"], ast.Call)
    assert isinstance(by_name["runtime_builder"].func, ast.Name)
    assert by_name["runtime_builder"].func.id == "create_configured_runtime_builder"
    assert isinstance(by_name["app"], ast.Call)
    assert isinstance(by_name["app"].func, ast.Name)
    assert by_name["app"].func.id == "create_bootstrapped_app"
    assert "create_app(Settings())" not in source
    assert (
        sum(
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "app" for target in node.targets)
            for node in module.body
        )
        == 1
    )
