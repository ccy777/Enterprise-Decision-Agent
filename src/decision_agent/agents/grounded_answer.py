"""Structured final-answer generation behind an injectable async contract."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from decision_agent.config import Settings
from decision_agent.exceptions import ConfigurationError, DecisionAgentError
from decision_agent.observability.execution import (
    TraceSpanRecorder,
    complete_recorded_span,
    start_recorded_span,
)
from decision_agent.observability.models import TraceContext
from decision_agent.observability.provider import (
    ProviderTraceMetadata,
    provider_failure_attributes,
    provider_response_attributes,
)
from decision_agent.observability.stages import SpanStatus, TraceStage
from decision_agent.providers import build_chat_completion_payload, extract_stopped_message_content
from decision_agent.retrieval.evidence_context import EvidenceItem


class AnswerGenerationError(DecisionAgentError):
    """Raised when a provider cannot produce the required structured final answer."""


class AnswerDraft(BaseModel):
    """Untrusted structured output before workflow-level citation validation."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    citations: list[str] = Field(default_factory=list)


class AnswerGenerator(Protocol):
    """Generate an answer only after the reviewer has determined it is answerable."""

    async def generate(
        self,
        *,
        user_query: str,
        selected_evidence_context: str,
        selected_evidence: Sequence[EvidenceItem],
        answerability: str,
        missing_information: str | None,
        decision_reason: str,
        trace_recorder: TraceSpanRecorder | None = None,
        trace_parent_context: TraceContext | None = None,
    ) -> AnswerDraft:
        """Return answer/citations only; workflow validates them before publishing."""


class OpenAICompatibleAnswerGenerator:
    """One minimal OpenAI-compatible JSON provider with no SDK import-time side effects."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model_name: str,
        timeout_seconds: float,
    ) -> None:
        required_values = (api_key, base_url, model_name)
        if not all(isinstance(value, str) and value.strip() for value in required_values):
            raise ConfigurationError("LLM API key, base URL, and model name are required")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._timeout_seconds = timeout_seconds

    @classmethod
    def from_settings(cls, settings: Settings) -> OpenAICompatibleAnswerGenerator:
        if (
            settings.llm_api_key is None
            or settings.llm_base_url is None
            or settings.llm_model_name is None
        ):
            raise ConfigurationError("LLM API key, base URL, and model name are required")
        return cls(
            api_key=settings.llm_api_key.get_secret_value(),
            base_url=settings.llm_base_url,
            model_name=settings.llm_model_name,
            timeout_seconds=settings.llm_timeout_seconds,
        )

    async def generate(
        self,
        *,
        user_query: str,
        selected_evidence_context: str,
        selected_evidence: Sequence[EvidenceItem],
        answerability: str,
        missing_information: str | None,
        decision_reason: str,
        trace_recorder: TraceSpanRecorder | None = None,
        trace_parent_context: TraceContext | None = None,
    ) -> AnswerDraft:
        del selected_evidence
        operation = "generate_grounded_answer"
        metadata = self.provider_trace_metadata()
        provider_span = start_recorded_span(
            trace_recorder,
            stage=TraceStage.PROVIDER_CALL,
            component="provider",
            operation=operation,
            parent_context=trace_parent_context,
            attributes=provider_failure_attributes(metadata=metadata, operation=operation),
        )
        try:
            payload = await asyncio.to_thread(
                self._post,
                user_query,
                selected_evidence_context,
                answerability,
                missing_information,
                decision_reason,
            )
        except asyncio.CancelledError:
            complete_recorded_span(
                trace_recorder,
                provider_span,
                status=SpanStatus.CANCELLED,
                attributes=provider_failure_attributes(metadata=metadata, operation=operation),
            )
            raise
        except (HTTPError, URLError, OSError, ValueError, ValidationError) as exc:
            complete_recorded_span(
                trace_recorder,
                provider_span,
                status=SpanStatus.FAILED,
                error_code="answer_generation_failed",
                attributes=provider_failure_attributes(metadata=metadata, operation=operation),
            )
            raise AnswerGenerationError("LLM returned no valid structured final answer") from exc
        except Exception:
            complete_recorded_span(
                trace_recorder,
                provider_span,
                status=SpanStatus.FAILED,
                error_code="answer_generation_failed",
                attributes=provider_failure_attributes(metadata=metadata, operation=operation),
            )
            raise
        complete_recorded_span(
            trace_recorder,
            provider_span,
            status=SpanStatus.COMPLETED,
            attributes=provider_response_attributes(
                metadata=metadata, operation=operation, payload=payload
            ),
        )
        try:
            return AnswerDraft.model_validate(json.loads(_extract_message_content(payload)))
        except (ValueError, ValidationError) as exc:
            raise AnswerGenerationError("LLM returned no valid structured final answer") from exc

    def provider_trace_metadata(self) -> ProviderTraceMetadata:
        """Return static adapter metadata; this adapter deliberately has no retry loop."""
        return ProviderTraceMetadata(
            provider="openai_compatible", model=self._model_name, retry_count=0
        )

    def _post(
        self,
        user_query: str,
        selected_evidence_context: str,
        answerability: str,
        missing_information: str | None,
        decision_reason: str,
    ) -> dict[str, Any]:
        body = json.dumps(
            build_chat_completion_payload(
                base_url=self._base_url,
                payload={
                    "model": self._model_name,
                    "temperature": 0,
                    "max_tokens": 600,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": "\n".join(
                                (
                                    "User question:",
                                    user_query,
                                    "",
                                    "Reviewer conclusion (already determined; do not change it):",
                                    answerability,
                                    "",
                                    "Reviewer missing information:",
                                    str(missing_information),
                                    "",
                                    "Reviewer audit reason:",
                                    decision_reason,
                                    "",
                                    "Selected Evidence Context:",
                                    selected_evidence_context,
                                )
                            ),
                        },
                    ],
                },
            ),
            ensure_ascii=False,
        ).encode("utf-8")
        request = Request(
            f"{self._base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlopen(request, timeout=self._timeout_seconds) as response:
            decoded = json.loads(response.read().decode("utf-8"))
        if not isinstance(decoded, dict):
            raise ValueError("LLM response must be an object")
        return decoded


def _extract_message_content(payload: dict[str, Any]) -> str:
    return extract_stopped_message_content(payload)


_SYSTEM_PROMPT = """You are an enterprise final-answer generator.
Output exactly one JSON object: no Markdown, no code fence, and no other text.

The reviewer has already determined that this request is answerable. Do not make or alter an
answerability decision. The JSON object must contain exactly these fields: answer, citations.
answer must be a nonempty string.

Use the same primary language as the user question for answer. A Chinese question must receive
Chinese output; an English question may receive English output. Evidence language does not
determine the output language.

Use only the supplied Selected Evidence. Do not use general knowledge, inference, or neighbouring
policies. answer must contain every citation inline in exact [E<number>] format.
Each citation in citations must occur in answer; every inline citation in answer must occur in
citations; do not put citations only in the citations field. citations must use actual Evidence
IDs. Do not output any field other than answer and citations.

Format-only example (not a business fact):
Evidence:
[E1] Employees entering a server room must wear an employee badge.
Correct JSON:
{
  "answer": "Employees entering a server room must wear an employee badge.[E1]",
  "citations": ["[E1]"]
}
"""
