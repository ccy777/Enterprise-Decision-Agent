"""Conservative, fully offline token estimation and immutable budgets."""

from __future__ import annotations

import math
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TokenEstimator(Protocol):
    """Estimate tokens for text without requiring a model provider."""

    def estimate(self, text: str) -> int:
        """Return a deterministic non-negative token estimate."""


class ConservativeCharacterTokenEstimator:
    """Estimate tokens conservatively; this is not any model's exact tokenizer.

    ASCII characters are counted at four characters per token and non-ASCII
    characters at one character per token, then the total is rounded up.
    """

    def estimate(self, text: str) -> int:
        """Return a deterministic estimate without loading a tokenizer or model."""
        if not text:
            return 0
        ascii_count = sum(character.isascii() for character in text)
        non_ascii_count = len(text) - ascii_count
        return max(1, math.ceil(ascii_count / 4 + non_ascii_count))


class TokenBudget(BaseModel):
    """Immutable usable-token budget for one context-selection operation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_tokens: int = Field(gt=0)
    reserved_tokens: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _validate_reservation(self) -> TokenBudget:
        if self.reserved_tokens >= self.max_tokens:
            raise ValueError("reserved_tokens must be less than max_tokens")
        return self

    @property
    def available_tokens(self) -> int:
        """Return tokens available after the immutable reservation."""
        return self.max_tokens - self.reserved_tokens

    def can_fit(self, current_tokens: int, candidate_tokens: int) -> bool:
        """Return whether a non-negative candidate fits without mutating this budget."""
        if current_tokens < 0 or candidate_tokens < 0:
            raise ValueError("token counts must be non-negative")
        return current_tokens + candidate_tokens <= self.available_tokens
