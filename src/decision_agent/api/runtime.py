"""FastAPI lifespan integration for one app-local formal runtime."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from contextlib import asynccontextmanager

from fastapi import FastAPI

from decision_agent.api.app import ReadinessCheck, create_app
from decision_agent.api.security import ApiSecurityContextResolver
from decision_agent.application.bootstrap import (
    BootstrapErrorCode,
    FormalRuntimeHandle,
    RuntimeBootstrapError,
    RuntimeBuilder,
    build_bootstrapped_runtime,
)
from decision_agent.config import Settings


def create_bootstrapped_app(
    settings: Settings,
    runtime_builder: RuntimeBuilder,
    readiness_checks: Mapping[str, ReadinessCheck] | None = None,
    *,
    security_context_resolver: ApiSecurityContextResolver | None = None,
) -> FastAPI:
    """Create an app whose lifespan owns one injected runtime builder."""
    handle = FormalRuntimeHandle()
    return create_app(
        settings,
        readiness_checks,
        runtime_handle=handle,
        lifespan=_runtime_lifespan(handle=handle, runtime_builder=runtime_builder),
        runtime_readiness_required=True,
        security_context_resolver=security_context_resolver,
    )


def _runtime_lifespan(
    *,
    handle: FormalRuntimeHandle,
    runtime_builder: RuntimeBuilder,
):
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        handle.mark_starting()
        runtime = None
        try:
            runtime = await build_bootstrapped_runtime(runtime_builder)
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            handle.fail(BootstrapErrorCode.RUNTIME_UNAVAILABLE)
            raise
        except RuntimeBootstrapError as exc:
            handle.fail(BootstrapErrorCode(exc.code))
        else:
            handle.publish(runtime.executor)

        try:
            yield
        finally:
            handle.revoke_executor()
            try:
                if runtime is not None:
                    await runtime.aclose()
            finally:
                handle.stop()

    return lifespan
