"""Unit coverage for the stable FastAPI Agent execution boundary."""

from __future__ import annotations

import asyncio
from typing import cast

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from decision_agent.api import create_app
from decision_agent.api.models import AgentExecutionResponse
from decision_agent.application import (
    FormalRequest,
    FormalRequestExecutor,
    FormalResponse,
    MemoryContextStatus,
    MemoryPersistenceStatus,
    MemorySummarizationStatus,
)
from decision_agent.config import Environment, Settings
from decision_agent.coordination.models import CoordinatorResult, CoordinatorStatus
from decision_agent.routing.models import RequestRoute


class _RecordingExecutor:
    def __init__(
        self,
        response: FormalResponse | None = None,
        *,
        error: BaseException | None = None,
    ) -> None:
        self._response = response
        self._error = error
        self.calls: list[FormalRequest] = []

    async def execute(self, request: FormalRequest) -> FormalResponse:
        self.calls.append(request)
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


def _settings() -> Settings:
    return Settings(
        app_name="Module 6C-1 Test Agent",
        environment=Environment.TEST,
        required_dependencies=[],
        _env_file=None,
    )


def _completed_response(
    *,
    request_id: str = "api-request-1",
    answer: str = "API_RESPONSE_MARKER [E1]",
) -> FormalResponse:
    return FormalResponse(
        request_id=request_id,
        result=CoordinatorResult(
            status=CoordinatorStatus.COMPLETED,
            route=RequestRoute.KNOWLEDGE,
            skill_name="enterprise-knowledge-qa",
            answer=answer,
            citations=["[E1]"],
            coordinator_steps=("route_request", "execute_skill"),
            tool_steps=("run_knowledge_agent",),
        ),
        memory_context_status=MemoryContextStatus.PROJECTED,
        memory_persistence_status=MemoryPersistenceStatus.PERSISTED,
        memory_summarization_status=MemorySummarizationStatus.NOT_NEEDED,
    )


def _failed_response() -> FormalResponse:
    return FormalResponse(
        request_id="api-failed-1",
        result=CoordinatorResult(
            status=CoordinatorStatus.FAILED,
            route=RequestRoute.DATA,
            error_code="write_statement_not_allowed",
            coordinator_steps=("route_request", "execute_skill"),
            tool_steps=("run_data_agent",),
        ),
        memory_context_status=MemoryContextStatus.NOT_REQUESTED,
    )


def _app_with(executor: _RecordingExecutor | None):
    return create_app(
        _settings(),
        formal_request_executor=cast(FormalRequestExecutor | None, executor),
    )


def test_valid_request_maps_to_formal_request_and_reviewed_response() -> None:
    executor = _RecordingExecutor(_completed_response())

    with TestClient(_app_with(executor)) as client:
        response = client.post(
            "/api/v1/agent/execute",
            json={
                "request_id": "api-request-1",
                "session_id": "  session-6c  ",
                "query": "说明库存补货制度",
            },
        )

    assert response.status_code == 200
    assert executor.calls == [
        FormalRequest(
            request_id="api-request-1",
            session_id="session-6c",
            user_query="说明库存补货制度",
        )
    ]
    assert response.json() == {
        "request_id": "api-request-1",
        "status": "completed",
        "route": "knowledge",
        "skill": "enterprise-knowledge-qa",
        "answer": "API_RESPONSE_MARKER [E1]",
        "citations": ["[E1]"],
        "error_code": None,
        "memory_context_status": "projected",
        "memory_persistence_status": "persisted",
        "memory_summarization_status": "not_needed",
    }


def test_request_rejects_extra_fields_without_calling_executor() -> None:
    executor = _RecordingExecutor(_completed_response())

    with TestClient(_app_with(executor)) as client:
        response = client.post(
            "/api/v1/agent/execute",
            json={
                "request_id": "api-extra-1",
                "query": "合法问题",
                "route": "knowledge",
            },
        )

    assert response.status_code == 422
    assert executor.calls == []


@pytest.mark.parametrize("query", ["", "   "])
def test_request_rejects_empty_query_without_calling_executor(query: str) -> None:
    executor = _RecordingExecutor(_completed_response())

    with TestClient(_app_with(executor)) as client:
        response = client.post(
            "/api/v1/agent/execute",
            json={"request_id": "api-empty-1", "query": query},
        )

    assert response.status_code == 422
    assert executor.calls == []


@pytest.mark.parametrize(
    "session_id",
    ["a\u0085b", "a\u00a0b"],
    ids=["nel", "no-break-space"],
)
def test_agent_endpoint_rejects_disallowed_unicode_session_whitespace_before_execution(
    session_id: str,
) -> None:
    executor = _RecordingExecutor(_completed_response())

    with TestClient(_app_with(executor)) as client:
        response = client.post(
            "/api/v1/agent/execute",
            json={
                "request_id": "api-invalid-session-1",
                "session_id": session_id,
                "query": "Explain the inventory replenishment policy.",
            },
        )

    assert response.status_code == 422
    assert response.status_code != 500
    assert executor.calls == []


def test_runtime_unavailable_returns_stable_503() -> None:
    with TestClient(_app_with(None)) as client:
        response = client.post(
            "/api/v1/agent/execute",
            json={"request_id": "api-unavailable-1", "query": "合法问题"},
        )

    assert response.status_code == 503
    assert response.json() == {
        "code": "runtime_unavailable",
        "message": "The Agent runtime is unavailable.",
    }


def test_formal_failed_response_remains_http_200() -> None:
    executor = _RecordingExecutor(_failed_response())

    with TestClient(_app_with(executor)) as client:
        response = client.post(
            "/api/v1/agent/execute",
            json={"request_id": "api-failed-1", "query": "执行危险写操作"},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert response.json()["error_code"] == "write_statement_not_allowed"
    assert response.json()["answer"] is None
    assert response.json()["citations"] == []


def test_unexpected_exception_returns_safe_500_without_exception_content() -> None:
    secret_exception = RuntimeError(
        "https://secret.example.test mysql://user:password prompt=private"
    )
    executor = _RecordingExecutor(error=secret_exception)

    with TestClient(_app_with(executor)) as client:
        response = client.post(
            "/api/v1/agent/execute",
            json={"request_id": "api-error-1", "query": "合法问题"},
        )

    body = response.text.lower()
    assert response.status_code == 500
    assert response.json() == {
        "code": "internal_execution_error",
        "message": "The Agent request could not be completed.",
    }
    assert all(
        forbidden not in body
        for forbidden in ("secret.example", "mysql", "password", "prompt", "private")
    )


@pytest.mark.asyncio
async def test_cancelled_error_propagates_from_executor() -> None:
    executor = _RecordingExecutor(error=asyncio.CancelledError())
    transport = ASGITransport(app=_app_with(executor), raise_app_exceptions=True)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        with pytest.raises(asyncio.CancelledError):
            await client.post(
                "/api/v1/agent/execute",
                json={"request_id": "api-cancelled-1", "query": "合法问题"},
            )

    assert len(executor.calls) == 1


def test_two_apps_keep_distinct_executor_bindings() -> None:
    first = _RecordingExecutor(_completed_response(answer="FIRST_APP_MARKER [E1]"))
    second = _RecordingExecutor(_completed_response(answer="SECOND_APP_MARKER [E1]"))

    with (
        TestClient(_app_with(first)) as first_client,
        TestClient(_app_with(second)) as second_client,
    ):
        first_response = first_client.post(
            "/api/v1/agent/execute",
            json={"request_id": "api-request-1", "query": "第一个应用"},
        )
        second_response = second_client.post(
            "/api/v1/agent/execute",
            json={"request_id": "api-request-1", "query": "第二个应用"},
        )

    assert first_response.json()["answer"] == "FIRST_APP_MARKER [E1]"
    assert second_response.json()["answer"] == "SECOND_APP_MARKER [E1]"
    assert first.calls[0].user_query == "第一个应用"
    assert second.calls[0].user_query == "第二个应用"


def test_create_app_and_health_probes_do_not_call_executor() -> None:
    executor = _RecordingExecutor(_completed_response())
    app = _app_with(executor)
    assert executor.calls == []

    with TestClient(app) as client:
        health_response = client.get("/health")
        ready_response = client.get("/ready")

    assert health_response.status_code == 200
    assert ready_response.status_code == 200
    assert executor.calls == []


def test_openapi_exposes_only_reviewed_request_and_response_fields() -> None:
    with TestClient(_app_with(None)) as client:
        schema = client.get("/openapi.json").json()

    operation = schema["paths"]["/api/v1/agent/execute"]["post"]
    request_schema = schema["components"]["schemas"]["AgentExecutionRequest"]
    response_schema = schema["components"]["schemas"]["AgentExecutionResponse"]

    assert operation["responses"].keys() >= {"200", "422", "500", "503"}
    assert request_schema["additionalProperties"] is False
    assert set(request_schema["properties"]) == {"request_id", "session_id", "query"}
    assert set(response_schema["properties"]) == {
        "request_id",
        "status",
        "route",
        "skill",
        "answer",
        "citations",
        "error_code",
        "memory_context_status",
        "memory_persistence_status",
        "memory_summarization_status",
        "trace",
    }
    assert "trace" not in response_schema["required"]


def test_invalid_optional_trace_projection_does_not_break_the_business_response() -> None:
    formal_response = _completed_response().model_copy(update={"trace": object()})

    response = AgentExecutionResponse.from_formal_response(formal_response)

    assert response.trace is None
    assert response.answer == "API_RESPONSE_MARKER [E1]"
    assert response.status is CoordinatorStatus.COMPLETED
