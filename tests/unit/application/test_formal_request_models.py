from __future__ import annotations

import pytest
from pydantic import ValidationError

from decision_agent.application import (
    FormalRequest,
    FormalResponse,
    MemoryContextStatus,
    MemoryPersistenceStatus,
    MemorySummarizationStatus,
)
from decision_agent.coordination.models import CoordinatorResult, CoordinatorStatus
from decision_agent.routing.models import RequestRoute


class _SecretInvalidValue:
    def __repr__(self) -> str:
        return "USER_QUERY_SECRET_DO_NOT_LEAK"


def test_formal_request_validates_session_identity_without_leaking_query() -> None:
    request = FormalRequest(
        request_id="request-1", user_query="USER_QUERY_SECRET_DO_NOT_LEAK", session_id=" session-1 "
    )
    assert request.session_id == "session-1"
    assert "USER_QUERY_SECRET_DO_NOT_LEAK" not in repr(request)


@pytest.mark.parametrize("session_id", [" ", "bad\nvalue", "x" * 129])
def test_formal_request_rejects_invalid_session_id(session_id: str) -> None:
    with pytest.raises(ValueError):
        FormalRequest(request_id="request-1", user_query="query", session_id=session_id)


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "request_id": "request-1",
            "user_query": "query",
            "session_id": "SESSION_SECRET_DO_NOT_LEAK\x00",
        },
        {
            "request_id": "request-1",
            "user_query": _SecretInvalidValue(),
            "session_id": None,
        },
    ],
)
def test_formal_request_validation_errors_hide_sensitive_input(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValidationError) as raised:
        FormalRequest(**kwargs)
    marker = (
        "SESSION_SECRET_DO_NOT_LEAK"
        if kwargs["session_id"] is not None
        else "USER_QUERY_SECRET_DO_NOT_LEAK"
    )
    assert marker not in str(raised.value)
    assert marker not in repr(raised.value)


def test_formal_response_persistence_default_is_immutable_and_content_safe() -> None:
    response = FormalResponse(
        request_id="request-1",
        result=CoordinatorResult(
            status=CoordinatorStatus.COMPLETED,
            route=RequestRoute.KNOWLEDGE,
            skill_name="knowledge",
            answer="ANSWER_SECRET_DO_NOT_LEAK",
            tool_steps=("run",),
        ),
        memory_context_status=MemoryContextStatus.EMPTY,
    )
    assert response.memory_persistence_status is MemoryPersistenceStatus.NOT_REQUESTED
    assert response.memory_summarization_status is MemorySummarizationStatus.NOT_REQUESTED
    assert "ANSWER_SECRET_DO_NOT_LEAK" not in repr(response)
    with pytest.raises(ValidationError):
        response.memory_persistence_status = MemoryPersistenceStatus.PERSISTED  # type: ignore[misc]
    with pytest.raises(ValidationError):
        FormalResponse(
            request_id="request-1",
            result=response.result,
            memory_context_status=MemoryContextStatus.EMPTY,
            unexpected="value",
        )
