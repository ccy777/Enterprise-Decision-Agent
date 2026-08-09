"""Unit coverage for app-independent formal runtime lifecycle foundations."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from dataclasses import fields
from typing import cast

import pytest

from decision_agent.application.bootstrap import (
    BootstrapErrorCode,
    BootstrappedRuntime,
    BootstrapStatus,
    FormalRuntimeHandle,
    RuntimeBootstrapError,
    RuntimeSnapshot,
    build_bootstrapped_runtime,
)
from decision_agent.application.executor import FormalRequestExecutor


class _ExecutorDouble:
    def __init__(self, marker: str) -> None:
        self.marker = marker

    def __repr__(self) -> str:
        return f"_ExecutorDouble(secret={self.marker!r})"


class _RecordingResource:
    def __init__(
        self,
        name: str,
        events: list[str],
        *,
        close_error: BaseException | None = None,
    ) -> None:
        self.name = name
        self.events = events
        self.close_error = close_error
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1
        self.events.append(self.name)
        if self.close_error is not None:
            raise self.close_error


def _executor(marker: str = "EXECUTOR_SECRET_MARKER") -> FormalRequestExecutor:
    return cast(FormalRequestExecutor, _ExecutorDouble(marker))


def test_handle_starts_not_started_and_never_exposes_executor_before_ready() -> None:
    handle = FormalRuntimeHandle()

    assert handle.executor is None
    assert handle.snapshot() == RuntimeSnapshot(
        status=BootstrapStatus.NOT_STARTED,
        executor_available=False,
        error_code=None,
    )

    handle.mark_starting()

    assert handle.executor is None
    assert handle.snapshot().status is BootstrapStatus.STARTING


def test_handle_publishes_the_same_executor_only_in_ready_state() -> None:
    handle = FormalRuntimeHandle()
    executor = _executor()

    handle.mark_starting()
    handle.publish(executor)

    assert handle.executor is executor
    assert handle.snapshot() == RuntimeSnapshot(
        status=BootstrapStatus.READY,
        executor_available=True,
        error_code=None,
    )


def test_handle_failure_clears_executor_and_retains_only_safe_code() -> None:
    handle = FormalRuntimeHandle()
    handle.mark_starting()

    handle.fail(BootstrapErrorCode.RUNTIME_UNAVAILABLE)

    assert handle.executor is None
    assert handle.snapshot() == RuntimeSnapshot(
        status=BootstrapStatus.FAILED,
        executor_available=False,
        error_code="bootstrap_runtime_unavailable",
    )


def test_handle_shutdown_revokes_executor_before_stopped_state() -> None:
    handle = FormalRuntimeHandle()
    handle.mark_starting()
    handle.publish(_executor())

    handle.revoke_executor()

    assert handle.executor is None
    assert handle.snapshot().status is BootstrapStatus.READY

    handle.stop()

    assert handle.executor is None
    assert handle.snapshot().status is BootstrapStatus.STOPPED


def test_handle_stop_is_idempotent_and_clears_non_ready_state() -> None:
    handle = FormalRuntimeHandle()
    handle.mark_starting()

    handle.stop()
    handle.stop()

    assert handle.executor is None
    assert handle.snapshot().status is BootstrapStatus.STOPPED


def test_snapshot_shape_cannot_expose_executor_or_raw_exception() -> None:
    snapshot_fields = {item.name for item in fields(RuntimeSnapshot)}

    assert snapshot_fields == {"status", "executor_available", "error_code"}
    rendered = repr(FormalRuntimeHandle().snapshot())
    assert "_ExecutorDouble" not in rendered
    assert "EXECUTOR_SECRET_MARKER" not in rendered
    assert "exception" not in rendered.lower()


def test_handle_repr_never_contains_executor_repr() -> None:
    handle = FormalRuntimeHandle()
    handle.mark_starting()
    handle.publish(_executor("DO_NOT_RENDER_EXECUTOR"))

    rendered = repr(handle)

    assert "DO_NOT_RENDER_EXECUTOR" not in rendered
    assert "_ExecutorDouble" not in rendered
    assert "executor_available=True" in rendered


def test_two_handles_keep_status_and_executor_references_isolated() -> None:
    first, second = FormalRuntimeHandle(), FormalRuntimeHandle()
    first_executor, second_executor = _executor("FIRST"), _executor("SECOND")

    first.mark_starting()
    first.publish(first_executor)
    second.mark_starting()
    second.publish(second_executor)
    first.revoke_executor()
    first.stop()

    assert first.executor is None
    assert second.executor is second_executor
    assert first.snapshot().status is BootstrapStatus.STOPPED
    assert second.snapshot().status is BootstrapStatus.READY


def test_illegal_handle_transitions_fail_closed_with_safe_code() -> None:
    handle = FormalRuntimeHandle()

    with pytest.raises(RuntimeBootstrapError) as publish_error:
        handle.publish(_executor())
    with pytest.raises(RuntimeBootstrapError) as fail_error:
        handle.fail()

    assert publish_error.value.code == "bootstrap_configuration_invalid"
    assert fail_error.value.code == "bootstrap_configuration_invalid"
    assert handle.executor is None
    assert handle.snapshot().status is BootstrapStatus.NOT_STARTED


@pytest.mark.asyncio
async def test_aggregate_closes_resources_in_reverse_order_once() -> None:
    events: list[str] = []
    stack = AsyncExitStack()
    await stack.__aenter__()
    first = _RecordingResource("first", events)
    second = _RecordingResource("second", events)
    stack.push_async_callback(first.aclose)
    stack.push_async_callback(second.aclose)
    runtime = BootstrappedRuntime(executor=_executor(), resources=stack)

    await runtime.aclose()
    await runtime.aclose()

    assert events == ["second", "first"]
    assert (first.close_calls, second.close_calls) == (1, 1)
    assert runtime.closed is True


@pytest.mark.asyncio
async def test_aggregate_maps_close_failure_to_safe_typed_error() -> None:
    stack = AsyncExitStack()
    await stack.__aenter__()
    resource = _RecordingResource(
        "resource",
        [],
        close_error=RuntimeError("SECRET_SHUTDOWN_URL=https://private.example"),
    )
    stack.push_async_callback(resource.aclose)
    runtime = BootstrappedRuntime(executor=_executor(), resources=stack)

    with pytest.raises(RuntimeBootstrapError) as raised:
        await runtime.aclose()

    assert raised.value.code == "bootstrap_shutdown_failed"
    assert str(raised.value) == "bootstrap_shutdown_failed"
    assert "private" not in repr(raised.value).lower()
    assert runtime.closed is True
    await runtime.aclose()
    assert resource.close_calls == 1


@pytest.mark.asyncio
async def test_builder_success_returns_the_exact_formal_executor_reference() -> None:
    executor = _executor()

    async def builder(_: AsyncExitStack) -> FormalRequestExecutor:
        return executor

    runtime = await build_bootstrapped_runtime(builder)

    assert runtime.executor is executor
    await runtime.aclose()


@pytest.mark.asyncio
async def test_partial_startup_failure_rolls_back_registered_resources() -> None:
    events: list[str] = []

    async def builder(stack: AsyncExitStack) -> FormalRequestExecutor:
        first = _RecordingResource("first", events)
        second = _RecordingResource("second", events)
        stack.push_async_callback(first.aclose)
        stack.push_async_callback(second.aclose)
        raise ValueError("PRIMARY_STARTUP_FAILURE")

    with pytest.raises(RuntimeBootstrapError) as raised:
        await build_bootstrapped_runtime(builder)

    assert raised.value.code == "bootstrap_runtime_unavailable"
    assert events == ["second", "first"]
    assert isinstance(raised.value.__cause__, ValueError)
    assert "PRIMARY_STARTUP_FAILURE" in str(raised.value.__cause__)
    assert "PRIMARY_STARTUP_FAILURE" not in str(raised.value)


@pytest.mark.asyncio
async def test_cleanup_failure_does_not_replace_primary_startup_failure() -> None:
    async def builder(stack: AsyncExitStack) -> FormalRequestExecutor:
        resource = _RecordingResource(
            "resource",
            [],
            close_error=RuntimeError("SECONDARY_CLEANUP_FAILURE"),
        )
        stack.push_async_callback(resource.aclose)
        raise ValueError("PRIMARY_BUILDER_FAILURE")

    with pytest.raises(RuntimeBootstrapError) as raised:
        await build_bootstrapped_runtime(builder)

    assert raised.value.code == "bootstrap_runtime_unavailable"
    assert isinstance(raised.value.__cause__, ValueError)
    assert str(raised.value.__cause__) == "PRIMARY_BUILDER_FAILURE"
    assert "SECONDARY_CLEANUP_FAILURE" not in repr(raised.value)


@pytest.mark.asyncio
async def test_cancelled_builder_rolls_back_and_preserves_cancellation() -> None:
    events: list[str] = []

    async def builder(stack: AsyncExitStack) -> FormalRequestExecutor:
        resource = _RecordingResource("cancelled-resource", events)
        stack.push_async_callback(resource.aclose)
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await build_bootstrapped_runtime(builder)

    assert events == ["cancelled-resource"]


@pytest.mark.asyncio
async def test_builder_returning_no_executor_fails_closed() -> None:
    async def builder(_: AsyncExitStack) -> FormalRequestExecutor:
        return None  # type: ignore[return-value]

    with pytest.raises(RuntimeBootstrapError) as raised:
        await build_bootstrapped_runtime(builder)

    assert raised.value.code == "bootstrap_configuration_invalid"


def test_typed_bootstrap_error_renders_only_its_stable_code() -> None:
    error = RuntimeBootstrapError(BootstrapErrorCode.RUNTIME_UNAVAILABLE)
    error.__cause__ = RuntimeError("TOKEN=DO_NOT_RENDER")

    assert str(error) == "bootstrap_runtime_unavailable"
    assert repr(error) == "RuntimeBootstrapError('bootstrap_runtime_unavailable')"
    assert "DO_NOT_RENDER" not in repr(error)
