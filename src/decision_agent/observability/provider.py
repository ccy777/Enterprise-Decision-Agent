"""Safe, response-metadata-only projection for one provider-call span."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_SAFE_FINISH_REASONS = frozenset(
    {"stop", "length", "content_filter", "tool_calls", "function_call"}
)


@dataclass(frozen=True, slots=True)
class ProviderTraceMetadata:
    """Stable adapter-owned metadata; never derived from a raw response or repr."""

    provider: str | None
    model: str | None
    retry_count: int | None


def provider_response_attributes(
    *,
    metadata: ProviderTraceMetadata,
    operation: str,
    payload: Mapping[str, Any],
) -> dict[str, str | int | bool | None]:
    """Return a closed scalar projection without retaining provider content."""
    input_tokens, output_tokens, usage_available = _usage(payload)
    return {
        "provider": metadata.provider,
        "model": metadata.model,
        "operation": operation,
        "usage_available": usage_available,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "retry_count": metadata.retry_count,
        "finish_reason": _finish_reason(payload),
        "success": True,
    }


def provider_failure_attributes(
    *, metadata: ProviderTraceMetadata, operation: str
) -> dict[str, str | int | bool | None]:
    """Project safe adapter facts when a provider call did not return a payload."""
    return {
        "provider": metadata.provider,
        "model": metadata.model,
        "operation": operation,
        "usage_available": False,
        "input_tokens": None,
        "output_tokens": None,
        "retry_count": metadata.retry_count,
        "finish_reason": None,
        "success": False,
    }


def _usage(payload: Mapping[str, Any]) -> tuple[int | None, int | None, bool]:
    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        return None, None, False
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
    if not _valid_token_count(input_tokens) or not _valid_token_count(output_tokens):
        return None, None, False
    return input_tokens, output_tokens, True


def _valid_token_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _finish_reason(payload: Mapping[str, Any]) -> str | None:
    try:
        value = payload["choices"][0].get("finish_reason")
    except (KeyError, IndexError, TypeError, AttributeError):
        return None
    return value if isinstance(value, str) and value in _SAFE_FINISH_REASONS else None
