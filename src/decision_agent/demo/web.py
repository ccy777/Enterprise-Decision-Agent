"""Localhost-only identity adapter and app factory for the Web workspace."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request

from decision_agent.api.runtime import create_bootstrapped_app
from decision_agent.application.configured_runtime import create_configured_runtime_builder
from decision_agent.config import Settings
from decision_agent.demo.local import (
    DemoCase,
    build_demo_security_context,
    prepare_demo_settings,
)
from decision_agent.security import SecurityAuthorizationError, SecurityContext, SecurityErrorCode

_LOOPBACK_CLIENTS = frozenset({"127.0.0.1", "::1"})


class LocalDemoSecurityContextResolver:
    """Grant one selected frozen Demo case to requests originating on loopback."""

    def __init__(self, case: DemoCase) -> None:
        self._case = case

    def resolve(self, *, request: Request, request_id: str) -> SecurityContext:
        client = request.client
        if client is None or client.host not in _LOOPBACK_CLIENTS:
            raise SecurityAuthorizationError(SecurityErrorCode.UNAUTHENTICATED)
        return build_demo_security_context(
            (self._case,),
            request_id=request_id,
            trace_id=f"local-web-demo-{uuid4().hex}",
            include_session_scope=True,
        )


def create_local_demo_app(
    *,
    case: DemoCase,
    settings: Settings,
    repository_root: Path,
) -> FastAPI:
    """Create the formal Runtime with a localhost-only, least-privilege Demo identity."""
    demo_settings = prepare_demo_settings(settings, repository_root=repository_root)
    return create_bootstrapped_app(
        demo_settings,
        create_configured_runtime_builder(demo_settings),
        security_context_resolver=LocalDemoSecurityContextResolver(case),
    )
