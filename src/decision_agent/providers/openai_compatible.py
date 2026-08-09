"""Small, transport-neutral payload rules for direct Chat Completions requests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse


class ChatCompletionResponseError(ValueError):
    """Safe classification for an invalid ordinary structured Chat Completions response."""

    def __init__(self, code: str) -> None:
        super().__init__("OpenAI-compatible chat completion response is invalid")
        self.code = code


def build_chat_completion_payload(
    *, base_url: str, payload: Mapping[str, object]
) -> dict[str, object]:
    """Copy a generic payload and apply only the configured provider's direct-HTTP fields.

    ``extra_body`` is an OpenAI SDK convenience argument, not a Chat Completions JSON field.
    DeepSeek's direct HTTP API instead receives its non-thinking mode as a top-level field.
    Provider-specific reasoning fields are removed before the final body is returned so a caller
    cannot accidentally carry an SDK-only wrapper into a direct HTTP request.
    """
    body = dict(payload)
    for key in ("extra_body", "thinking", "reasoning_content"):
        body.pop(key, None)
    if urlparse(base_url).hostname == "api.deepseek.com":
        body["thinking"] = {"type": "disabled"}
    return body


def extract_stopped_message_content(payload: Mapping[str, Any]) -> str:
    """Return content only from a single normal completion without retaining provider output."""
    choices = payload.get("choices") if isinstance(payload, Mapping) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise ChatCompletionResponseError("missing_choice")
    choice = choices[0]
    finish_reason = choice.get("finish_reason")
    if finish_reason == "length":
        raise ChatCompletionResponseError("truncated")
    if finish_reason != "stop":
        raise ChatCompletionResponseError("invalid_finish_reason")
    message = choice.get("message")
    content = message.get("content") if isinstance(message, Mapping) else None
    if not isinstance(content, str) or not content.strip():
        raise ChatCompletionResponseError("empty_content")
    return content
