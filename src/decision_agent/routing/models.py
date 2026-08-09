"""Strict public contracts for classifying one enterprise request."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RequestRoute(StrEnum):
    """The bounded capabilities available to the unified request router."""

    KNOWLEDGE = "knowledge"
    DATA = "data"
    MIXED = "mixed"
    UNSUPPORTED = "unsupported"


class RouterDecision(BaseModel):
    """Untrusted structured router output with no executable-tool fields."""

    model_config = ConfigDict(extra="forbid")

    route: RequestRoute
    normalized_query: str = Field(min_length=1, max_length=4_000)
    decision_reason: str = Field(min_length=1, max_length=600)
    knowledge_subquery: str | None = Field(default=None, max_length=2_000)
    data_subquery: str | None = Field(default=None, max_length=2_000)
    missing_information: str | None = Field(default=None, max_length=600)
    confidence: float = Field(ge=0, le=1)

    @field_validator(
        "normalized_query",
        "decision_reason",
        "knowledge_subquery",
        "data_subquery",
        "missing_information",
    )
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        """Reject whitespace-only values and keep the public decision concise."""
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("text fields cannot be blank")
        return normalized

    @model_validator(mode="after")
    def validate_route_contract(self) -> RouterDecision:
        """Require only the subqueries relevant to the selected route."""
        if self.route is RequestRoute.KNOWLEDGE:
            if self.knowledge_subquery is None or self.data_subquery is not None:
                raise ValueError("knowledge requires only knowledge_subquery")
        elif self.route is RequestRoute.DATA:
            if self.data_subquery is None or self.knowledge_subquery is not None:
                raise ValueError("data requires only data_subquery")
        elif self.route is RequestRoute.MIXED:
            if self.knowledge_subquery is None or self.data_subquery is None:
                raise ValueError("mixed requires knowledge_subquery and data_subquery")
        elif self.knowledge_subquery is not None or self.data_subquery is not None:
            raise ValueError("unsupported cannot contain subqueries")
        return self
