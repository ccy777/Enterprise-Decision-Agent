"""FastAPI router adapting HTTP requests to the caller-owned formal executor."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from decision_agent.api.models import (
    AgentExecutionRequest,
    AgentExecutionResponse,
    ApiErrorResponse,
)
from decision_agent.api.security import (
    ApiSecurityContextResolver,
    RejectingApiSecurityContextResolver,
)
from decision_agent.application.bootstrap import FormalRuntimeHandle
from decision_agent.security import SecurityAuthorizationError, SecurityErrorCode

logger = logging.getLogger(__name__)


def create_agent_router(
    runtime_handle: FormalRuntimeHandle,
    *,
    security_context_resolver: ApiSecurityContextResolver | None = None,
) -> APIRouter:
    """Bind one app-local runtime holder without global mutable state."""
    router = APIRouter(prefix="/api/v1/agent", tags=["agent"])

    @router.post(
        "/execute",
        response_model=AgentExecutionResponse,
        responses={
            status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ApiErrorResponse},
            status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ApiErrorResponse},
            status.HTTP_401_UNAUTHORIZED: {"model": ApiErrorResponse},
            status.HTTP_403_FORBIDDEN: {"model": ApiErrorResponse},
        },
    )
    async def execute_agent(
        request: AgentExecutionRequest,
        http_request: Request,
    ) -> AgentExecutionResponse | JSONResponse:
        formal_request_executor = runtime_handle.executor
        if formal_request_executor is None:
            return _error_response(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                code="runtime_unavailable",
                message="The Agent runtime is unavailable.",
            )
        security_context = None
        if bool(getattr(formal_request_executor, "requires_security_context", False)):
            resolver = security_context_resolver or RejectingApiSecurityContextResolver()
            try:
                security_context = resolver.resolve(
                    request=http_request,
                    request_id=request.request_id,
                )
            except SecurityAuthorizationError as exc:
                return _security_error_response(exc.code)
            except Exception:
                return _security_error_response(SecurityErrorCode.SECURITY_CONTEXT_INVALID.value)
        try:
            response = await formal_request_executor.execute(
                request.to_formal_request(security_context=security_context)
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.error("Formal Agent execution failed")
            return _error_response(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                code="internal_execution_error",
                message="The Agent request could not be completed.",
            )
        return AgentExecutionResponse.from_formal_response(response)

    return router


def _error_response(*, status_code: int, code: str, message: str) -> JSONResponse:
    body = ApiErrorResponse(code=code, message=message)
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))


def _security_error_response(code: str) -> JSONResponse:
    status_code = (
        status.HTTP_401_UNAUTHORIZED
        if code == SecurityErrorCode.UNAUTHENTICATED.value
        else status.HTTP_403_FORBIDDEN
    )
    return _error_response(
        status_code=status_code,
        code=code,
        message="The Agent request is not authorized.",
    )
