"""App-independent lifecycle contracts for publishing one formal runtime safely."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass
from enum import StrEnum

from decision_agent.application.executor import FormalRequestExecutor


class BootstrapStatus(StrEnum):
    """Minimal lifecycle states for one app-local formal runtime."""

    NOT_STARTED = "not_started"
    STARTING = "starting"
    READY = "ready"
    FAILED = "failed"
    STOPPED = "stopped"


class BootstrapErrorCode(StrEnum):
    """Stable, content-safe bootstrap failure categories."""

    CONFIGURATION_INVALID = "bootstrap_configuration_invalid"
    RUNTIME_UNAVAILABLE = "bootstrap_runtime_unavailable"
    SHUTDOWN_FAILED = "bootstrap_shutdown_failed"


class RuntimeBootstrapError(RuntimeError):
    """Typed bootstrap failure that never renders its underlying cause."""

    def __init__(self, code: BootstrapErrorCode) -> None:
        self.code = code.value
        super().__init__(self.code)


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    """Read-only lifecycle projection without runtime objects or failure details."""

    status: BootstrapStatus
    executor_available: bool
    error_code: str | None


class FormalRuntimeHandle:
    """Per-app mutable holder with one narrow, fail-closed executor read."""

    __slots__ = ("_error_code", "_executor", "_status")

    def __init__(self) -> None:
        self._status = BootstrapStatus.NOT_STARTED
        self._executor: FormalRequestExecutor | None = None
        self._error_code: str | None = None

    def __repr__(self) -> str:
        snapshot = self.snapshot()
        return (
            "FormalRuntimeHandle("
            f"status={snapshot.status!r}, "
            f"executor_available={snapshot.executor_available!r}, "
            f"error_code={snapshot.error_code!r})"
        )

    @property
    def executor(self) -> FormalRequestExecutor | None:
        """Return the executor only after a complete READY publication."""
        if self._status is not BootstrapStatus.READY:
            return None
        return self._executor

    def snapshot(self) -> RuntimeSnapshot:
        """Return a content-safe immutable state projection."""
        return RuntimeSnapshot(
            status=self._status,
            executor_available=self.executor is not None,
            error_code=self._error_code,
        )

    def mark_starting(self) -> None:
        """Begin the sole startup attempt for this handle."""
        self._require_status(BootstrapStatus.NOT_STARTED)
        self._status = BootstrapStatus.STARTING
        self._executor = None
        self._error_code = None

    def publish(self, executor: FormalRequestExecutor) -> None:
        """Atomically make one completely constructed executor available."""
        self._require_status(BootstrapStatus.STARTING)
        if executor is None:
            raise RuntimeBootstrapError(BootstrapErrorCode.CONFIGURATION_INVALID)
        self._executor = executor
        self._error_code = None
        self._status = BootstrapStatus.READY

    def fail(self, code: BootstrapErrorCode = BootstrapErrorCode.RUNTIME_UNAVAILABLE) -> None:
        """Fail the active startup attempt and discard any executor reference."""
        self._require_status(BootstrapStatus.STARTING)
        self._executor = None
        self._error_code = code.value
        self._status = BootstrapStatus.FAILED

    def revoke_executor(self) -> None:
        """Stop serving new requests before resource shutdown begins."""
        self._executor = None

    def stop(self) -> None:
        """Finalize shutdown; repeated calls remain safe."""
        self._executor = None
        if self._status is BootstrapStatus.STOPPED:
            return
        self._status = BootstrapStatus.STOPPED

    def _require_status(self, expected: BootstrapStatus) -> None:
        if self._status is not expected:
            raise RuntimeBootstrapError(BootstrapErrorCode.CONFIGURATION_INVALID)


RuntimeBuilder = Callable[[AsyncExitStack], Awaitable[FormalRequestExecutor]]


class BootstrappedRuntime:
    """Own one executor plus a private LIFO stack of caller-created resources."""

    __slots__ = ("_closed", "_executor", "_resources")

    def __init__(
        self,
        *,
        executor: FormalRequestExecutor,
        resources: AsyncExitStack,
    ) -> None:
        self._executor = executor
        self._resources = resources
        self._closed = False

    def __repr__(self) -> str:
        return f"BootstrappedRuntime(closed={self._closed!r})"

    @property
    def executor(self) -> FormalRequestExecutor:
        """Return the formal application boundary, never an internal resource."""
        return self._executor

    @property
    def closed(self) -> bool:
        return self._closed

    async def aclose(self) -> None:
        """Close registered resources once in reverse registration order."""
        if self._closed:
            return
        self._closed = True
        try:
            await self._resources.aclose()
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            raise RuntimeBootstrapError(BootstrapErrorCode.SHUTDOWN_FAILED) from exc


async def build_bootstrapped_runtime(builder: RuntimeBuilder) -> BootstrappedRuntime:
    """Run one injected builder and roll back every registered resource on failure."""
    if not callable(builder):
        raise RuntimeBootstrapError(BootstrapErrorCode.CONFIGURATION_INVALID)

    resources = AsyncExitStack()
    await resources.__aenter__()
    try:
        executor = await builder(resources)
        if executor is None:
            raise RuntimeBootstrapError(BootstrapErrorCode.CONFIGURATION_INVALID)
    except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
        await _rollback_preserving_primary_failure(resources)
        raise
    except RuntimeBootstrapError:
        await _rollback_preserving_primary_failure(resources)
        raise
    except Exception as exc:
        await _rollback_preserving_primary_failure(resources)
        raise RuntimeBootstrapError(BootstrapErrorCode.RUNTIME_UNAVAILABLE) from exc
    return BootstrappedRuntime(executor=executor, resources=resources)


async def _rollback_preserving_primary_failure(resources: AsyncExitStack) -> None:
    """Best-effort rollback whose failure cannot replace the startup failure."""
    try:
        await resources.aclose()
    except BaseException:
        # The active startup exception remains authoritative and is re-raised by the caller.
        return
