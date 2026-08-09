from __future__ import annotations

import pytest

from decision_agent.application.turn_identity import derive_turn_id


def test_turn_id_is_stable_versioned_and_does_not_expose_inputs() -> None:
    value = derive_turn_id("SESSION_SECRET_DO_NOT_LEAK", "REQUEST_SECRET_DO_NOT_LEAK")
    assert value == derive_turn_id("SESSION_SECRET_DO_NOT_LEAK", "REQUEST_SECRET_DO_NOT_LEAK")
    assert value.startswith("turn-v1-")
    assert len(value) == len("turn-v1-") + 64
    assert "SESSION_SECRET_DO_NOT_LEAK" not in value
    assert "REQUEST_SECRET_DO_NOT_LEAK" not in value


def test_turn_id_changes_when_either_identity_changes() -> None:
    baseline = derive_turn_id("session-a", "request-a")
    assert baseline != derive_turn_id("session-b", "request-a")
    assert baseline != derive_turn_id("session-a", "request-b")


@pytest.mark.parametrize("session_id, request_id", [(" ", "request"), ("session", " ")])
def test_turn_id_rejects_blank_identity_without_echoing_it(
    session_id: str, request_id: str
) -> None:
    with pytest.raises(ValueError) as raised:
        derive_turn_id(session_id, request_id)
    assert str(raised.value) in {"session_id must be non-empty", "request_id must be non-empty"}
