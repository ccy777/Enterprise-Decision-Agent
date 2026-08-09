"""Pure, bounded projection of persisted session memory into untrusted Context."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

from decision_agent.context.policies import DEFAULT_CONTEXT_BUDGET_CONFIG
from decision_agent.context.token_budget import ConservativeCharacterTokenEstimator, TokenEstimator
from decision_agent.memory.models import SessionMemorySnapshot, SessionTurn

_EVIDENCE_MARKER = re.compile(r"\[(?P<domain>[DE])(?P<number>[1-9]\d*)\]")


class ConversationMemoryProjection(BaseModel):
    """Validated untrusted prompt data, intentionally separate from a Store snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    content: str = Field(repr=False, min_length=1, max_length=20_000)
    estimated_tokens: int = Field(ge=1)

    @field_validator("content")
    @classmethod
    def _no_current_evidence_markers(cls, value: str) -> str:
        if _EVIDENCE_MARKER.search(value):
            raise ValueError("conversation memory cannot contain current evidence markers")
        return value


class ConversationMemoryProjector:
    """Render a summary plus a newest contiguous Turn suffix without truncation."""

    def __init__(
        self,
        *,
        estimator: TokenEstimator | None = None,
        token_allowance: int = DEFAULT_CONTEXT_BUDGET_CONFIG.router.available_tokens,
    ) -> None:
        if token_allowance <= 0:
            raise ValueError("token_allowance must be positive")
        self._estimator = estimator or ConservativeCharacterTokenEstimator()
        self._token_allowance = token_allowance

    def project(self, snapshot: SessionMemorySnapshot) -> ConversationMemoryProjection | None:
        """Return one complete candidate item, or ``None`` when it cannot safely fit."""
        if snapshot.summary is None and not snapshot.turns:
            return None
        summary_text = (
            None if snapshot.summary is None else _neutralize(snapshot.summary.summary_text)
        )
        selected_turns: tuple[SessionTurn, ...] = ()
        if summary_text is not None and not self._fits(summary_text, selected_turns):
            return None

        for turn in reversed(snapshot.turns):
            candidate = (turn, *selected_turns)
            if not self._fits(summary_text, candidate):
                if not selected_turns and summary_text is None:
                    return None
                break
            selected_turns = candidate

        content = _render(summary_text, selected_turns)
        estimated_tokens = self._estimator.estimate(content)
        if estimated_tokens > self._token_allowance:
            return None
        return ConversationMemoryProjection(content=content, estimated_tokens=estimated_tokens)

    def _fits(self, summary_text: str | None, turns: tuple[SessionTurn, ...]) -> bool:
        return self._estimator.estimate(_render(summary_text, turns)) <= self._token_allowance


def _neutralize(value: str) -> str:
    return _EVIDENCE_MARKER.sub(
        lambda match: f"[historical-{match.group('domain')}{match.group('number')}]", value
    )


def _render(summary_text: str | None, turns: tuple[SessionTurn, ...]) -> str:
    sections: list[str] = ["<UNTRUSTED_CONVERSATION_MEMORY>"]
    if summary_text is not None:
        sections.extend(("<PREVIOUS_SUMMARY>", summary_text, "</PREVIOUS_SUMMARY>"))
    for turn in turns:
        sections.extend(
            (
                "<HISTORICAL_TURN>",
                "<HISTORICAL_USER>",
                _neutralize(turn.user_text),
                "</HISTORICAL_USER>",
                "<HISTORICAL_ASSISTANT>",
                _neutralize(turn.assistant_text),
                "</HISTORICAL_ASSISTANT>",
                "</HISTORICAL_TURN>",
            )
        )
    sections.append("</UNTRUSTED_CONVERSATION_MEMORY>")
    return "\n".join(sections)
