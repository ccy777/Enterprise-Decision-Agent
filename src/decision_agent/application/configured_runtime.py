"""Production composition root for one fully configured formal runtime."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass
from functools import partial
from typing import Protocol, cast

from redis import Redis

from decision_agent.agent_workflow.providers import (
    OpenAICompatibleWorkflowPlanner,
    OpenAICompatibleWorkflowReviewer,
)
from decision_agent.agent_workflow.workflow import ControlledAgentWorkflow, ControlledWorkflowPolicy
from decision_agent.agents.answerability_reviewer import (
    AnswerabilityReviewer,
    OpenAICompatibleAnswerabilityReviewer,
)
from decision_agent.agents.data_answer_generator import (
    DataAnswerGenerator,
    OpenAICompatibleDataAnswerGenerator,
    project_data_answer_provider_payload,
)
from decision_agent.agents.data_query_planner import (
    DataQueryPlanner,
    OpenAICompatibleDataQueryPlanner,
)
from decision_agent.agents.evidence_selector import (
    EvidenceSelector,
    OpenAICompatibleEvidenceSelector,
)
from decision_agent.agents.grounded_answer import AnswerGenerator, OpenAICompatibleAnswerGenerator
from decision_agent.application.bootstrap import (
    BootstrapErrorCode,
    RuntimeBootstrapError,
    RuntimeBuilder,
)
from decision_agent.application.executor import FormalRequestExecutor
from decision_agent.application.runtime import (
    FormalMemoryConfiguration,
    build_formal_request_executor,
)
from decision_agent.config import Settings
from decision_agent.config.settings import Environment
from decision_agent.coordination.factory import build_default_coordinator
from decision_agent.mcp_client.enterprise_data_client import EnterpriseDataMCPClient
from decision_agent.memory.models import SessionMemoryPolicy
from decision_agent.memory.redis_store import RedisSessionMemoryStore
from decision_agent.memory.summarization import (
    ProviderRollingSummarizer,
    RollingSummaryPolicy,
)
from decision_agent.observability import (
    BestEffortTraceDispatcher,
    StructuredLoggingTraceSink,
)
from decision_agent.retrieval.factory import (
    EnterpriseRetrievalRuntime,
    build_production_retrieval_runtime,
)
from decision_agent.routing.request_router import (
    OpenAICompatibleRequestRouter,
    RequestRouter,
)
from decision_agent.security import (
    DataClassification,
    DefaultDenyAuthorizationPolicy,
    DeterministicProviderRedactor,
    GovernedChatCompletionClient,
    GovernedNativeToolCallingModel,
    GovernedProviderRole,
    InMemoryAuditSink,
    JsonlAuditSink,
    ProviderGovernance,
    ProviderPolicy,
    ProviderPolicyError,
    ProviderStage,
)
from decision_agent.skills.inventory_risk_synthesizer import (
    OpenAICompatibleChatCompletionClient,
    OpenAICompatibleInventoryRiskSynthesizer,
)
from decision_agent.skills.native_runtime import (
    NativeToolCallingSkillExecutor,
    PreselectedAgentToolSkillExecutor,
)
from decision_agent.tool_calling.runtime import (
    NativeToolCallingModel,
    OpenAICompatibleNativeToolCallingModel,
)
from decision_agent.tool_calling.tools import DataAgentTool, KnowledgeAgentTool
from decision_agent.workflows.data_agent import EnterpriseDataClient
from decision_agent.workflows.knowledge_qa import build_knowledge_qa_graph


class ConfiguredNativeModel(
    NativeToolCallingModel,
    OpenAICompatibleChatCompletionClient,
    Protocol,
):
    """Combined formal provider surface reused by native, mixed, and summary paths."""


class RedisClientProtocol(Protocol):
    """Synchronous Redis client methods used by startup and shutdown."""

    def ping(self) -> object:
        """Check startup reachability."""

    def close(self) -> object:
        """Release client resources."""


@dataclass(frozen=True, slots=True)
class ConfiguredProviderAdapters:
    """Fixed provider bundle for all formal runtime model roles."""

    router: RequestRouter
    native_model: ConfiguredNativeModel
    evidence_selector: EvidenceSelector
    answerability_reviewer: AnswerabilityReviewer
    answer_generator: AnswerGenerator
    data_query_planner: DataQueryPlanner
    data_answer_generator: DataAnswerGenerator


@dataclass(frozen=True, slots=True)
class ConfiguredRuntimeDependencies:
    """Fixed construction seams limited to external runtime boundaries."""

    provider_adapters_factory: Callable[[Settings], ConfiguredProviderAdapters]
    knowledge_runtime_factory: Callable[[Settings], EnterpriseRetrievalRuntime]
    enterprise_data_client_factory: Callable[[Settings], EnterpriseDataClient]
    redis_client_factory: Callable[[str, float], RedisClientProtocol]


def create_configured_runtime_builder(
    settings: Settings,
    dependencies: ConfiguredRuntimeDependencies | None = None,
) -> RuntimeBuilder:
    """Return one deferred production builder without invoking dependencies or I/O."""
    resolved = dependencies or _default_configured_runtime_dependencies()

    async def builder(stack: AsyncExitStack) -> FormalRequestExecutor:
        return await _build_configured_runtime(settings, stack, resolved)

    return builder


async def _build_configured_runtime(
    settings: Settings,
    stack: AsyncExitStack,
    dependencies: ConfiguredRuntimeDependencies,
) -> FormalRequestExecutor:
    _validate_complete_runtime_configuration(settings)

    if settings.audit_log_path is None and settings.environment is not Environment.TEST:
        raise RuntimeBootstrapError(BootstrapErrorCode.CONFIGURATION_INVALID)
    audit_sink = (
        InMemoryAuditSink()
        if settings.audit_log_path is None
        else JsonlAuditSink(settings.audit_log_path)
    )
    stack.callback(audit_sink.close)
    provider_governance = ProviderGovernance(
        policy=ProviderPolicy.controlled_mixed(),
        audit_sink=audit_sink,
        redactor=DeterministicProviderRedactor(),
    )

    providers = dependencies.provider_adapters_factory(settings)
    knowledge_runtime = dependencies.knowledge_runtime_factory(settings)
    stack.push_async_callback(knowledge_runtime.aclose)
    await knowledge_runtime.initialize()

    governed_providers = ConfiguredProviderAdapters(
        router=cast(
            RequestRouter,
            GovernedProviderRole(
                provider=providers.router,
                governance=provider_governance,
                stage=ProviderStage.ROUTING,
                classification=DataClassification.INTERNAL,
            ),
        ),
        native_model=providers.native_model,
        evidence_selector=cast(
            EvidenceSelector,
            GovernedProviderRole(
                provider=providers.evidence_selector,
                governance=provider_governance,
                stage=ProviderStage.EVIDENCE_SELECTION,
                classification=DataClassification.CONFIDENTIAL,
                payload_projector=_provider_role_projector(
                    external_keys=("user_query", "evidence_context"),
                    local_keys=(
                        "retrieval_evidence",
                        "trace_recorder",
                        "trace_parent_context",
                    ),
                    evidence_key="retrieval_evidence",
                ),
            ),
        ),
        answerability_reviewer=cast(
            AnswerabilityReviewer,
            GovernedProviderRole(
                provider=providers.answerability_reviewer,
                governance=provider_governance,
                stage=ProviderStage.ANSWERABILITY_REVIEW,
                classification=DataClassification.CONFIDENTIAL,
                payload_projector=_provider_role_projector(
                    external_keys=("user_query", "selected_evidence_context"),
                    local_keys=(
                        "selected_evidence",
                        "trace_recorder",
                        "trace_parent_context",
                    ),
                    evidence_key="selected_evidence",
                ),
            ),
        ),
        answer_generator=cast(
            AnswerGenerator,
            GovernedProviderRole(
                provider=providers.answer_generator,
                governance=provider_governance,
                stage=ProviderStage.KNOWLEDGE_ANSWER,
                classification=DataClassification.CONFIDENTIAL,
                payload_projector=_provider_role_projector(
                    external_keys=(
                        "user_query",
                        "selected_evidence_context",
                        "answerability",
                        "missing_information",
                        "decision_reason",
                    ),
                    local_keys=(
                        "selected_evidence",
                        "trace_recorder",
                        "trace_parent_context",
                    ),
                    evidence_key="selected_evidence",
                ),
            ),
        ),
        data_query_planner=cast(
            DataQueryPlanner,
            GovernedProviderRole(
                provider=providers.data_query_planner,
                governance=provider_governance,
                stage=ProviderStage.DATA_PLANNING,
                classification=DataClassification.INTERNAL,
            ),
        ),
        data_answer_generator=cast(
            DataAnswerGenerator,
            GovernedProviderRole(
                provider=providers.data_answer_generator,
                governance=provider_governance,
                stage=ProviderStage.DATA_ANSWER,
                classification=DataClassification.CONFIDENTIAL,
                payload_projector=project_data_answer_provider_payload,
            ),
        ),
    )

    knowledge_graph = build_knowledge_qa_graph(
        retrieval_pipeline=knowledge_runtime.pipeline,
        evidence_selector=governed_providers.evidence_selector,
        answerability_reviewer=governed_providers.answerability_reviewer,
        answer_generator=governed_providers.answer_generator,
    )
    knowledge_tool = KnowledgeAgentTool(graph=knowledge_graph)

    enterprise_data_client_factory = partial(
        dependencies.enterprise_data_client_factory,
        settings,
    )
    async with enterprise_data_client_factory():
        pass
    data_tool = DataAgentTool(
        planner=governed_providers.data_query_planner,
        enterprise_data_client_factory=enterprise_data_client_factory,
        answer_generator=governed_providers.data_answer_generator,
    )

    governed_native_model = GovernedNativeToolCallingModel(
        client=providers.native_model,
        governance=provider_governance,
        stage=ProviderStage.KNOWLEDGE_ANSWER,
    )
    native_executor = NativeToolCallingSkillExecutor(
        model=cast(ConfiguredNativeModel, governed_native_model),
        knowledge_tool=knowledge_tool,
        data_tool=data_tool,
    )
    mixed_synthesizer = OpenAICompatibleInventoryRiskSynthesizer(
        client=GovernedChatCompletionClient(
            client=providers.native_model,
            governance=provider_governance,
            stage=ProviderStage.INVENTORY_SYNTHESIS,
        ),
    )
    coordinator = build_default_coordinator(
        router=governed_providers.router,
        tool_calling_executor=native_executor,
        inventory_risk_synthesizer=mixed_synthesizer,
        inventory_child_executor=(
            PreselectedAgentToolSkillExecutor(
                knowledge_tool=knowledge_tool,
                data_tool=data_tool,
            )
            if settings.controlled_workflow_enabled
            else None
        ),
        controlled_workflow_builder=(
            (
                lambda registry: ControlledAgentWorkflow(
                    planner=OpenAICompatibleWorkflowPlanner(
                        provider=GovernedChatCompletionClient(
                            client=providers.native_model,
                            governance=provider_governance,
                            stage=ProviderStage.PLANNING,
                            classification=DataClassification.INTERNAL,
                        )
                    ),
                    reviewer=OpenAICompatibleWorkflowReviewer(
                        provider=GovernedChatCompletionClient(
                            client=providers.native_model,
                            governance=provider_governance,
                            stage=ProviderStage.WORKFLOW_REVIEW,
                        )
                    ),
                    registry=registry,
                    policy=ControlledWorkflowPolicy(enabled=True),
                )
            )
            if settings.controlled_workflow_enabled
            else None
        ),
    )

    memory = await _build_memory_configuration(settings, providers, stack, dependencies)
    executor = build_formal_request_executor(
        coordinator=coordinator,
        memory=memory,
        authorization_policy=DefaultDenyAuthorizationPolicy(),
        provider_governance=provider_governance,
    )
    return executor.with_trace_dispatcher(
        BestEffortTraceDispatcher(
            [StructuredLoggingTraceSink(logger=logging.getLogger("decision_agent.observability"))]
        )
    )


async def _build_memory_configuration(
    settings: Settings,
    providers: ConfiguredProviderAdapters,
    stack: AsyncExitStack,
    dependencies: ConfiguredRuntimeDependencies,
) -> FormalMemoryConfiguration:
    if settings.memory_mode == "disabled":
        return FormalMemoryConfiguration.disabled()

    policy = SessionMemoryPolicy(
        ttl_seconds=settings.memory_ttl_seconds,
        max_turns=settings.memory_max_turns,
    )
    summarizer = None
    summary_policy = None
    if settings.memory_summary_enabled:
        summarizer = ProviderRollingSummarizer(provider=providers.native_model)
        summary_policy = RollingSummaryPolicy(
            trigger_turns=settings.memory_summary_trigger_turns,
            retain_recent_turns=settings.memory_summary_retain_recent_turns,
            max_source_chars=settings.memory_summary_max_source_chars,
            max_summary_chars=settings.memory_summary_max_summary_chars,
        )

    if settings.memory_mode == "in_memory":
        return FormalMemoryConfiguration.in_memory(
            policy=policy,
            summarizer=summarizer,
            summary_policy=summary_policy,
        )

    redis_url = settings.memory_redis_url
    if redis_url is None:
        raise RuntimeBootstrapError(BootstrapErrorCode.CONFIGURATION_INVALID)
    redis_client = dependencies.redis_client_factory(
        redis_url.get_secret_value(),
        settings.memory_redis_timeout_seconds,
    )
    stack.push_async_callback(_close_redis_client, redis_client)
    await asyncio.to_thread(redis_client.ping)
    store = RedisSessionMemoryStore(client=redis_client, policy=policy)
    return FormalMemoryConfiguration.provided(
        store=store,
        summarizer=summarizer,
        summary_policy=summary_policy,
    )


def _provider_role_projector(
    *,
    external_keys: tuple[str, ...],
    local_keys: tuple[str, ...],
    evidence_key: str | None = None,
):
    """Separate actual Provider content from local-only Evidence and Trace objects."""

    allowed = frozenset((*external_keys, *local_keys))

    def project(
        args: tuple[object, ...], kwargs: dict[str, object]
    ) -> tuple[dict[str, object], dict[str, object], int]:
        if args or set(kwargs) - allowed or any(key not in kwargs for key in external_keys):
            raise ProviderPolicyError("provider_projection_failed")
        provider_kwargs = {key: kwargs[key] for key in external_keys}
        local_kwargs = {key: kwargs[key] for key in local_keys if key in kwargs}
        evidence = () if evidence_key is None else local_kwargs.get(evidence_key, ())
        if not isinstance(evidence, (list, tuple)):
            raise ProviderPolicyError("provider_projection_failed")
        return {"args": [], "kwargs": provider_kwargs}, local_kwargs, len(evidence)

    return project


async def _close_redis_client(client: RedisClientProtocol) -> None:
    await asyncio.to_thread(client.close)


def _validate_complete_runtime_configuration(settings: Settings) -> None:
    if (
        settings.llm_api_key is None
        or settings.llm_base_url is None
        or settings.llm_model_name is None
    ):
        raise RuntimeBootstrapError(BootstrapErrorCode.CONFIGURATION_INVALID)
    if settings.knowledge_dataset_root is None:
        raise RuntimeBootstrapError(BootstrapErrorCode.CONFIGURATION_INVALID)
    if settings.db_readonly_password is None:
        raise RuntimeBootstrapError(BootstrapErrorCode.CONFIGURATION_INVALID)


def _default_configured_runtime_dependencies() -> ConfiguredRuntimeDependencies:
    return ConfiguredRuntimeDependencies(
        provider_adapters_factory=_build_provider_adapters,
        knowledge_runtime_factory=build_production_retrieval_runtime,
        enterprise_data_client_factory=EnterpriseDataMCPClient.from_settings,
        redis_client_factory=_build_redis_client,
    )


def _build_provider_adapters(settings: Settings) -> ConfiguredProviderAdapters:
    native_model = OpenAICompatibleNativeToolCallingModel.from_settings(settings)
    return ConfiguredProviderAdapters(
        router=OpenAICompatibleRequestRouter.from_settings(settings),
        native_model=cast(ConfiguredNativeModel, native_model),
        evidence_selector=OpenAICompatibleEvidenceSelector.from_settings(settings),
        answerability_reviewer=OpenAICompatibleAnswerabilityReviewer.from_settings(settings),
        answer_generator=OpenAICompatibleAnswerGenerator.from_settings(settings),
        data_query_planner=OpenAICompatibleDataQueryPlanner.from_settings(settings),
        data_answer_generator=OpenAICompatibleDataAnswerGenerator.from_settings(settings),
    )


def _build_redis_client(url: str, timeout_seconds: float) -> RedisClientProtocol:
    return cast(
        RedisClientProtocol,
        Redis.from_url(
            url,
            socket_connect_timeout=timeout_seconds,
            socket_timeout=timeout_seconds,
            decode_responses=False,
        ),
    )
