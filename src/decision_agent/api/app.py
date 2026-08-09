"""Minimal FastAPI application and dependency probes for M0."""

import logging
from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Response, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from decision_agent.api.routes import create_agent_router
from decision_agent.api.security import ApiSecurityContextResolver
from decision_agent.application import FormalRequestExecutor
from decision_agent.application.bootstrap import (
    BootstrapStatus,
    FormalRuntimeHandle,
)
from decision_agent.config import Settings

ReadinessCheck = Callable[[], bool]
AppLifespan = Callable[[FastAPI], AbstractAsyncContextManager[None]]
logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    """Process liveness response."""

    status: Literal["ok"] = "ok"


class ReadinessResponse(BaseModel):
    """Aggregate readiness response with honest dependency states."""

    status: Literal["ready", "not_ready"]
    dependencies: dict[str, bool]


def _mount_demo_ui(app: FastAPI) -> None:
    """Mount package-local demo assets without reading them during app construction."""
    web_root = Path(__file__).parent.parent / "web"

    @app.get("/", include_in_schema=False, response_class=FileResponse)
    async def demo_ui() -> FileResponse:
        return FileResponse(web_root / "index.html", media_type="text/html")

    app.mount(
        "/assets",
        StaticFiles(directory=web_root, check_dir=False),
        name="demo-assets",
    )


def create_app(
    settings: Settings,
    readiness_checks: Mapping[str, ReadinessCheck] | None = None,
    *,
    formal_request_executor: FormalRequestExecutor | None = None,
    runtime_handle: FormalRuntimeHandle | None = None,
    lifespan: AppLifespan | None = None,
    runtime_readiness_required: bool = False,
    security_context_resolver: ApiSecurityContextResolver | None = None,
) -> FastAPI:
    """Build the API from injected settings and lazy dependency checks."""
    checks = dict(readiness_checks or {})
    handle = runtime_handle or FormalRuntimeHandle()
    if runtime_handle is not None and formal_request_executor is not None:
        raise ValueError("provide either runtime_handle or formal_request_executor")
    if formal_request_executor is not None:
        handle.mark_starting()
        handle.publish(formal_request_executor)

    required_dependencies = list(settings.required_dependencies)
    if runtime_readiness_required and "agent_runtime" not in required_dependencies:
        required_dependencies.append("agent_runtime")

    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.include_router(
        create_agent_router(handle, security_context_resolver=security_context_resolver)
    )
    _mount_demo_ui(app)

    def dependency_is_ready(name: str) -> bool:
        if runtime_readiness_required and name == "agent_runtime":
            snapshot = handle.snapshot()
            return snapshot.status is BootstrapStatus.READY and snapshot.executor_available
        check = checks.get(name)
        if check is None:
            return False
        try:
            return bool(check())
        except Exception:
            logger.exception("Readiness check failed for dependency %s", name)
            return False

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse()

    @app.get("/ready", response_model=ReadinessResponse)
    async def ready(response: Response) -> ReadinessResponse:
        dependency_states = {name: dependency_is_ready(name) for name in required_dependencies}
        is_ready = all(dependency_states.values())
        if not is_ready:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(
            status="ready" if is_ready else "not_ready",
            dependencies=dependency_states,
        )

    return app
