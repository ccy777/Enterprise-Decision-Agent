"""Structured answerability review behind an injectable async contract."""

# ruff: noqa: RUF001

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Sequence
from typing import Any, Literal, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

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

_CHINESE_CHARACTER_PATTERN = re.compile(r"[\u4e00-\u9fff]")


class AnswerabilityReviewError(DecisionAgentError):
    """Raised when a provider cannot produce a valid answerability decision."""


class AnswerabilityDecision(BaseModel):
    """Untrusted structured review output before workflow-level language validation."""

    model_config = ConfigDict(extra="forbid")

    answerability: Literal["answerable", "unanswerable"]
    missing_information: str | None = None
    decision_reason: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_contract(self) -> AnswerabilityDecision:
        if not self.decision_reason.strip():
            raise ValueError("decision_reason must be nonempty")
        if self.answerability == "answerable" and self.missing_information is not None:
            raise ValueError("answerable decision cannot declare missing information")
        if self.answerability == "unanswerable" and (
            not isinstance(self.missing_information, str) or not self.missing_information.strip()
        ):
            raise ValueError("unanswerable decision requires missing information")
        return self


class AnswerabilityReviewValidationResult(BaseModel):
    """Pure reviewer-output validation result with controlled error codes."""

    model_config = ConfigDict(extra="forbid")

    validation_passed: bool
    validation_errors: list[str]


class AnswerabilityReviewer(Protocol):
    """Decide only whether selected Evidence sufficiently answers a user query."""

    async def review(
        self,
        *,
        user_query: str,
        selected_evidence_context: str,
        selected_evidence: Sequence[EvidenceItem],
        trace_recorder: TraceSpanRecorder | None = None,
        trace_parent_context: TraceContext | None = None,
    ) -> AnswerabilityDecision:
        """Return an answerability decision without generating an answer or citations."""


def validate_answerability_decision(
    *, user_query: str, decision: AnswerabilityDecision
) -> AnswerabilityReviewValidationResult:
    """Enforce the lightweight language guard for Chinese reviewer output."""
    errors: list[str] = []
    if _CHINESE_CHARACTER_PATTERN.search(user_query):
        if not _CHINESE_CHARACTER_PATTERN.search(decision.decision_reason):
            errors.append("reviewer_language_mismatch")
        if decision.answerability == "unanswerable" and not _CHINESE_CHARACTER_PATTERN.search(
            decision.missing_information or ""
        ):
            errors.append("reviewer_language_mismatch")
    return AnswerabilityReviewValidationResult(
        validation_passed=not errors,
        validation_errors=list(dict.fromkeys(errors)),
    )


class OpenAICompatibleAnswerabilityReviewer:
    """One minimal OpenAI-compatible JSON reviewer with no import-time client creation."""

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
    def from_settings(cls, settings: Settings) -> OpenAICompatibleAnswerabilityReviewer:
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

    async def review(
        self,
        *,
        user_query: str,
        selected_evidence_context: str,
        selected_evidence: Sequence[EvidenceItem],
        trace_recorder: TraceSpanRecorder | None = None,
        trace_parent_context: TraceContext | None = None,
    ) -> AnswerabilityDecision:
        del selected_evidence
        operation = "review_answerability"
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
            payload = await asyncio.to_thread(self._post, user_query, selected_evidence_context)
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
                error_code="answerability_review_failed",
                attributes=provider_failure_attributes(metadata=metadata, operation=operation),
            )
            raise AnswerabilityReviewError(
                "LLM returned no valid structured answerability decision"
            ) from exc
        except Exception:
            complete_recorded_span(
                trace_recorder,
                provider_span,
                status=SpanStatus.FAILED,
                error_code="answerability_review_failed",
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
            content = json.loads(_extract_message_content(payload))
            return AnswerabilityDecision.model_validate(content)
        except (ValueError, ValidationError) as exc:
            raise AnswerabilityReviewError(
                "LLM returned no valid structured answerability decision"
            ) from exc

    def provider_trace_metadata(self) -> ProviderTraceMetadata:
        """Return static adapter metadata; this adapter deliberately has no retry loop."""
        return ProviderTraceMetadata(
            provider="openai_compatible", model=self._model_name, retry_count=0
        )

    def _post(self, user_query: str, selected_evidence_context: str) -> dict[str, Any]:
        body = json.dumps(
            build_chat_completion_payload(
                base_url=self._base_url,
                payload={
                    "model": self._model_name,
                    "temperature": 0,
                    "max_tokens": 400,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": _build_user_prompt(
                                user_query=user_query,
                                selected_evidence_context=selected_evidence_context,
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


def _build_user_prompt(*, user_query: str, selected_evidence_context: str) -> str:
    """Add a request-level language reminder only when the query contains Chinese."""
    parts = [
        "User question:",
        user_query,
        "",
        "Selected Evidence Context:",
        selected_evidence_context,
    ]
    if _CHINESE_CHARACTER_PATTERN.search(user_query):
        parts.extend(
            (
                "",
                "Mandatory Chinese output requirement:",
                "Keep JSON field names in English. decision_reason must be a complete Simplified "
                "Chinese sentence. If answerability is unanswerable, missing_information must use "
                "Simplified Chinese. Do not write English sentences in these two fields.",
            )
        )
    return "\n".join(parts)


_SYSTEM_PROMPT = """You are an enterprise answerability reviewer.
Output exactly one JSON object: no Markdown, no code fence, and no other text.

Decide only whether the supplied Selected Evidence explicitly and directly contains every fact,
number, condition, subject, time, relation, or complete approval requirement requested by the user.
Do not generate a final answer, citations, selected-evidence IDs, hidden reasoning, or chain of
thought. Do not output tool calls or retry suggestions. Relevant Evidence is not necessarily
sufficient Evidence. Do not infer, use general knowledge, or substitute a related policy for a
missing fact.

When the question asks for documented policy rules or criteria that will be applied to
operating data in a later stage, decide answerability only for those requested rules or
criteria. Do not require the policy Evidence to contain current operating values, item
statuses, or query results from the separate data source.
当问题询问用于后续经营数据判断的制度、阈值、条件或处理流程时，只审查证据是否完整
支持这些制度内容，不得因为制度证据不包含当前经营数值而判为不可回答。

Output exactly these fields: answerability, missing_information, decision_reason.
answerability must be either "answerable" or "unanswerable"; never output "failed".
For answerable, missing_information must be null. For unanswerable, missing_information must name
the specific missing information. decision_reason must be a concise audit explanation, not hidden
reasoning. Use the same primary language as the user question.

For a Chinese user question, keep JSON field names in English. decision_reason must be a complete
Simplified Chinese sentence. For unanswerable, missing_information must use Simplified Chinese.
Do not write English sentences in either field.

Format-only examples:
{
  "answerability":"answerable",
  "missing_information":null,
  "decision_reason":"The selected Evidence directly states the required policy."
}
{
  "answerability":"unanswerable",
  "missing_information":"the requested effective date",
  "decision_reason":"The selected Evidence does not state the requested effective date."
}

Chinese format-only examples:
{
  "answerability":"answerable",
  "missing_information":null,
  "decision_reason":"选中的证据明确规定了所需条件。"
}
{
  "answerability":"unanswerable",
  "missing_information":"所请求的生效日期",
  "decision_reason":"选中的证据没有规定所请求的生效日期。"
}
"""
