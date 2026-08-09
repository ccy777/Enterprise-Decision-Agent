"""Fail-closed API identity adapter seam for M8C-A."""

from __future__ import annotations

from typing import Protocol

from fastapi import Request

from decision_agent.security import SecurityAuthorizationError, SecurityContext, SecurityErrorCode


class ApiSecurityContextResolver(Protocol):
    """Resolve a context from a trusted upstream transport boundary only."""

    def resolve(self, *, request: Request, request_id: str) -> SecurityContext:
        """Return a verified context or raise a stable security denial."""


class RejectingApiSecurityContextResolver:
    """Production-safe default until a trusted external authentication adapter exists."""

    def resolve(self, *, request: Request, request_id: str) -> SecurityContext:
        del request, request_id
        raise SecurityAuthorizationError(SecurityErrorCode.UNAUTHENTICATED)
