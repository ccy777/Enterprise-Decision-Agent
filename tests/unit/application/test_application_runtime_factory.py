"""Unit coverage for explicit formal runtime composition."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from decision_agent.application import (
    FormalMemoryConfiguration,
    FormalMemoryMode,
    FormalRequestExecutor,
    FormalRuntimeConfigurationError,
    build_formal_request_executor,
)
from decision_agent.memory import (
    InMemorySessionMemoryStore,
    RollingSummaryDraft,
    RollingSummaryPolicy,
    SessionMemoryPolicy,
)


class _NoIoCoordinator:
    def __init__(self) -> None:
        self.execute_calls = 0

    async def execute(self, **_: object) -> object:
        self.execute_calls += 1
        raise AssertionError("factory must not execute the coordinator")


class _RecordingStore(InMemorySessionMemoryStore):
    def __init__(self) -> None:
        super().__init__()
        self.read_calls = 0
        self.append_calls = 0
        self.compact_calls = 0
        self.clear_calls = 0

    def read(self, session_id: str):  # type: ignore[no-untyped-def]
        self.read_calls += 1
        return super().read(session_id)

    def append_turn(self, turn, *, expected_version: int):  # type: ignore[no-untyped-def]
        self.append_calls += 1
        return super().append_turn(turn, expected_version=expected_version)

    def compact(self, summary, compacted_turn_ids, *, expected_version: int):  # type: ignore[no-untyped-def]
        self.compact_calls += 1
        return super().compact(summary, compacted_turn_ids, expected_version=expected_version)

    def clear(self, session_id: str, *, expected_version: int):
        self.clear_calls += 1
        return super().clear(session_id, expected_version=expected_version)


class _RecordingSummarizer:
    def __init__(self) -> None:
        self.calls = 0

    def summarize(self, _: object) -> RollingSummaryDraft:
        self.calls += 1
        return RollingSummaryDraft(summary_text="safe summary")


def _executor(memory: FormalMemoryConfiguration | None = None) -> FormalRequestExecutor:
    return build_formal_request_executor(coordinator=_NoIoCoordinator(), memory=memory)  # type: ignore[arg-type]


def test_public_configuration_constructors_are_immutable_and_dependency_safe() -> None:
    store = _RecordingStore()
    summarizer = _RecordingSummarizer()
    disabled = FormalMemoryConfiguration.disabled()
    in_memory = FormalMemoryConfiguration.in_memory(summarizer=summarizer)
    provided = FormalMemoryConfiguration.provided(store=store, summarizer=summarizer)

    assert disabled.mode is FormalMemoryMode.DISABLED
    assert in_memory.mode is FormalMemoryMode.IN_MEMORY
    assert provided.mode is FormalMemoryMode.PROVIDED and provided.store is store
    with pytest.raises(FrozenInstanceError):
        disabled.mode = FormalMemoryMode.PROVIDED  # type: ignore[misc]
    with pytest.raises(TypeError):
        FormalMemoryConfiguration.disabled(unexpected=True)  # type: ignore[call-arg]
    rendered = repr(provided)
    assert "_RecordingStore" not in rendered and "_RecordingSummarizer" not in rendered
    assert {field.name for field in __import__("dataclasses").fields(provided)} == {
        "mode",
        "store",
        "in_memory_policy",
        "summarizer",
        "summary_policy",
    }


def test_disabled_factory_has_projector_but_no_store_or_summary_service() -> None:
    coordinator = _NoIoCoordinator()
    executor = build_formal_request_executor(coordinator=coordinator)  # type: ignore[arg-type]

    assert isinstance(executor, FormalRequestExecutor)
    assert executor._memory_store is None  # type: ignore[attr-defined]
    assert executor._rolling_summary_service is None  # type: ignore[attr-defined]
    assert executor._memory_projector is not None  # type: ignore[attr-defined]
    assert coordinator.execute_calls == 0


def test_in_memory_factory_creates_independent_stores_and_projectors() -> None:
    first = _executor(FormalMemoryConfiguration.in_memory())
    second = _executor(FormalMemoryConfiguration.in_memory())

    assert isinstance(first._memory_store, InMemorySessionMemoryStore)  # type: ignore[attr-defined]
    assert first._memory_store is not second._memory_store  # type: ignore[attr-defined]
    assert first._memory_projector is not second._memory_projector  # type: ignore[attr-defined]
    assert first._rolling_summary_service is None  # type: ignore[attr-defined]


def test_in_memory_factory_passes_the_supplied_policy() -> None:
    policy = SessionMemoryPolicy(ttl_seconds=61, max_turns=3)
    executor = _executor(FormalMemoryConfiguration.in_memory(policy=policy))

    assert executor._memory_store._policy is policy  # type: ignore[attr-defined]


def test_provided_store_is_reused_without_construction_io() -> None:
    store = _RecordingStore()
    executor = _executor(FormalMemoryConfiguration.provided(store=store))

    assert executor._memory_store is store  # type: ignore[attr-defined]
    assert executor._rolling_summary_service is None  # type: ignore[attr-defined]
    assert (store.read_calls, store.append_calls, store.compact_calls, store.clear_calls) == (
        0,
        0,
        0,
        0,
    )


@pytest.mark.parametrize(
    "configuration", [FormalMemoryConfiguration.in_memory(), FormalMemoryConfiguration.disabled()]
)
def test_build_with_no_summarizer_does_not_create_summary_service(
    configuration: FormalMemoryConfiguration,
) -> None:
    assert _executor(configuration)._rolling_summary_service is None  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "configuration",
    [
        FormalMemoryConfiguration.in_memory(summarizer=_RecordingSummarizer()),
        FormalMemoryConfiguration.provided(
            store=_RecordingStore(), summarizer=_RecordingSummarizer()
        ),
    ],
)
def test_summary_service_reuses_the_executor_store(
    configuration: FormalMemoryConfiguration,
) -> None:
    executor = _executor(configuration)
    summary_service = executor._rolling_summary_service  # type: ignore[attr-defined]

    assert summary_service is not None
    assert summary_service._store is executor._memory_store  # type: ignore[attr-defined]


def test_summary_policy_is_used_without_invoking_the_summarizer() -> None:
    summarizer = _RecordingSummarizer()
    policy = RollingSummaryPolicy(
        trigger_turns=4,
        retain_recent_turns=1,
        max_source_chars=300,
        max_summary_chars=60,
    )
    executor = _executor(
        FormalMemoryConfiguration.in_memory(summarizer=summarizer, summary_policy=policy)
    )

    assert executor._rolling_summary_service._policy is policy  # type: ignore[attr-defined]
    assert summarizer.calls == 0


@pytest.mark.parametrize(
    ("configuration", "code"),
    [
        (
            FormalMemoryConfiguration(
                mode=FormalMemoryMode.DISABLED, summarizer=_RecordingSummarizer()
            ),
            "runtime_summary_requires_memory_store",
        ),
        (
            FormalMemoryConfiguration(
                mode=FormalMemoryMode.IN_MEMORY, summary_policy=RollingSummaryPolicy()
            ),
            "runtime_summary_policy_requires_summarizer",
        ),
        (
            FormalMemoryConfiguration(mode=FormalMemoryMode.PROVIDED),
            "runtime_memory_store_required",
        ),
        (
            FormalMemoryConfiguration(mode=FormalMemoryMode.DISABLED, store=_RecordingStore()),
            "runtime_memory_store_not_allowed",
        ),
        (
            FormalMemoryConfiguration(mode=FormalMemoryMode.IN_MEMORY, store=_RecordingStore()),
            "runtime_memory_store_not_allowed",
        ),
        (
            FormalMemoryConfiguration(
                mode=FormalMemoryMode.PROVIDED,
                store=_RecordingStore(),
                in_memory_policy=SessionMemoryPolicy(),
            ),
            "runtime_memory_policy_not_allowed",
        ),
        (
            FormalMemoryConfiguration(
                mode=FormalMemoryMode.DISABLED, in_memory_policy=SessionMemoryPolicy()
            ),
            "runtime_memory_policy_not_allowed",
        ),
        (FormalMemoryConfiguration(mode="invalid"), "runtime_memory_mode_invalid"),  # type: ignore[arg-type]
    ],
)
def test_invalid_configurations_fail_closed_with_safe_codes(
    configuration: FormalMemoryConfiguration, code: str
) -> None:
    with pytest.raises(FormalRuntimeConfigurationError) as raised:
        _executor(configuration)

    assert raised.value.code == code
    assert (
        str(raised.value) == code
        and repr(raised.value) == f"FormalRuntimeConfigurationError('{code}')"
    )
    assert "Recording" not in repr(raised.value) and "secret" not in repr(raised.value).lower()


def test_missing_coordinator_fails_closed_with_safe_code() -> None:
    with pytest.raises(FormalRuntimeConfigurationError) as raised:
        build_formal_request_executor(coordinator=None)  # type: ignore[arg-type]

    assert raised.value.code == "runtime_coordinator_required"
    assert str(raised.value) == "runtime_coordinator_required"
