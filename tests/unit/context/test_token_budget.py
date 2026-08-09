from __future__ import annotations

import pytest
from pydantic import ValidationError

from decision_agent.context import ConservativeCharacterTokenEstimator, TokenBudget


def test_conservative_estimator_is_deterministic_for_empty_english_cjk_and_mixed_text() -> None:
    estimator = ConservativeCharacterTokenEstimator()

    assert estimator.estimate("") == 0
    assert estimator.estimate("abcd") == 1
    assert estimator.estimate("产品") == 2
    assert estimator.estimate("A产品") == 3
    assert estimator.estimate("A产品") == estimator.estimate("A产品")


def test_token_budget_exposes_immutable_available_capacity_and_boundary_fit() -> None:
    budget = TokenBudget(max_tokens=100, reserved_tokens=20)

    assert budget.available_tokens == 80
    assert budget.can_fit(70, 10)
    assert not budget.can_fit(70, 11)
    with pytest.raises((ValidationError, TypeError)):
        TokenBudget(max_tokens=100, available_tokens=50)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        TokenBudget(max_tokens=0)
    with pytest.raises(ValidationError):
        TokenBudget(max_tokens=10, reserved_tokens=10)
    with pytest.raises(ValueError):
        budget.can_fit(-1, 1)
    with pytest.raises(ValidationError):
        budget.max_tokens = 200  # type: ignore[misc]
