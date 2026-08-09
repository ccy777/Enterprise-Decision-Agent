"""Explicit, side-effect-free composition for the formal request runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from decision_agent.application.executor import FormalRequestExecutor
from decision_agent.context.conversation_memory import ConversationMemoryProjector
from decision_agent.coordination import Coordinator
from decision_agent.memory import (
    InMemorySessionMemoryStore,
    RollingSummarizer,
    RollingSummaryPolicy,
    RollingSummaryService,
    SessionMemoryPolicy,
    SessionMemoryStore,
)
from decision_agent.security import AuthorizationPolicy, ProviderGovernance


class FormalMemoryMode(StrEnum):
    """The only supported memory composition modes for a formal runtime."""

    DISABLED = "disabled"
    IN_MEMORY = "in_memory"
    PROVIDED = "provided"


class FormalRuntimeConfigurationError(ValueError):
    """A stable, dependency-safe error for invalid runtime composition."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class FormalMemoryConfiguration:
    """Immutable caller-owned inputs for optional session-memory composition."""

    mode: FormalMemoryMode
    store: SessionMemoryStore | None = field(default=None, repr=False)
    in_memory_policy: SessionMemoryPolicy | None = field(default=None, repr=False)
    summarizer: RollingSummarizer | None = field(default=None, repr=False)
    summary_policy: RollingSummaryPolicy | None = field(default=None, repr=False)

    @classmethod
    def disabled(cls) -> FormalMemoryConfiguration:
        """Build a runtime configuration with session memory disabled."""
        return cls(mode=FormalMemoryMode.DISABLED)

    @classmethod
    def in_memory(
        cls,
        *,
        policy: SessionMemoryPolicy | None = None,
        summarizer: RollingSummarizer | None = None,
        summary_policy: RollingSummaryPolicy | None = None,
    ) -> FormalMemoryConfiguration:
        """Build a configuration that creates one fresh in-memory Store."""
        return cls(
            mode=FormalMemoryMode.IN_MEMORY,
            in_memory_policy=policy,
            summarizer=summarizer,
            summary_policy=summary_policy,
        )

    @classmethod
    def provided(
        cls,
        *,
        store: SessionMemoryStore | None,
        summarizer: RollingSummarizer | None = None,
        summary_policy: RollingSummaryPolicy | None = None,
    ) -> FormalMemoryConfiguration:
        """Build a configuration that reuses the caller-provided Store unchanged."""
        return cls(
            mode=FormalMemoryMode.PROVIDED,
            store=store,
            summarizer=summarizer,
            summary_policy=summary_policy,
        )


def build_formal_request_executor(
    *,
    coordinator: Coordinator,
    memory: FormalMemoryConfiguration | None = None,
    authorization_policy: AuthorizationPolicy | None = None,
    provider_governance: ProviderGovernance | None = None,
) -> FormalRequestExecutor:
    """Compose one uncached formal executor without performing external I/O."""
    if coordinator is None:
        raise FormalRuntimeConfigurationError("runtime_coordinator_required")

    configuration = FormalMemoryConfiguration.disabled() if memory is None else memory
    store = _resolve_store(configuration)
    summary_service = _build_summary_service(configuration, store)
    return FormalRequestExecutor(
        coordinator=coordinator,
        memory_store=store,
        memory_projector=ConversationMemoryProjector(),
        rolling_summary_service=summary_service,
        authorization_policy=authorization_policy,
        provider_governance=provider_governance,
    )


def _resolve_store(configuration: FormalMemoryConfiguration) -> SessionMemoryStore | None:
    if not isinstance(configuration, FormalMemoryConfiguration) or not isinstance(
        configuration.mode, FormalMemoryMode
    ):
        raise FormalRuntimeConfigurationError("runtime_memory_mode_invalid")

    mode = configuration.mode
    if mode is FormalMemoryMode.DISABLED:
        if configuration.store is not None:
            raise FormalRuntimeConfigurationError("runtime_memory_store_not_allowed")
        if configuration.in_memory_policy is not None:
            raise FormalRuntimeConfigurationError("runtime_memory_policy_not_allowed")
        return None
    if mode is FormalMemoryMode.IN_MEMORY:
        if configuration.store is not None:
            raise FormalRuntimeConfigurationError("runtime_memory_store_not_allowed")
        return InMemorySessionMemoryStore(
            **(
                {}
                if configuration.in_memory_policy is None
                else {"policy": configuration.in_memory_policy}
            )
        )
    if mode is FormalMemoryMode.PROVIDED:
        if configuration.in_memory_policy is not None:
            raise FormalRuntimeConfigurationError("runtime_memory_policy_not_allowed")
        if configuration.store is None:
            raise FormalRuntimeConfigurationError("runtime_memory_store_required")
        return configuration.store
    raise FormalRuntimeConfigurationError("runtime_memory_mode_invalid")


def _build_summary_service(
    configuration: FormalMemoryConfiguration,
    store: SessionMemoryStore | None,
) -> RollingSummaryService | None:
    if configuration.summarizer is None:
        if configuration.summary_policy is not None:
            raise FormalRuntimeConfigurationError("runtime_summary_policy_requires_summarizer")
        return None
    if store is None:
        raise FormalRuntimeConfigurationError("runtime_summary_requires_memory_store")
    if configuration.summary_policy is None:
        return RollingSummaryService(store=store, summarizer=configuration.summarizer)
    return RollingSummaryService(
        store=store,
        summarizer=configuration.summarizer,
        policy=configuration.summary_policy,
    )
