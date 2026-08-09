"""Payload-free, opt-in preflight for the configured Provider transport."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Protocol

from decision_agent.tool_calling.runtime import NativeToolCallingError


class ProviderTransport(Protocol):
    async def complete_chat(
        self, *, messages: list[dict[str, object]], response_format: dict[str, str]
    ) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class ProviderPreflightResult:
    component: str
    phase: str
    status: str
    error_code: str
    schema_valid: bool
    token_metadata_present: bool
    exception_type: str | None

    def safe_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


async def preflight_provider(transport: ProviderTransport) -> ProviderPreflightResult:
    """Issue one non-business JSON probe and project no response/configuration payload."""
    try:
        response = await transport.complete_chat(
            messages=[
                {
                    "role": "system",
                    "content": "Return exactly one JSON object with boolean field ok.",
                },
                {"role": "user", "content": "health"},
            ],
            response_format={"type": "json_object"},
        )
        choices = response.get("choices")
        content = choices[0]["message"]["content"] if isinstance(choices, list) else None
        parsed = json.loads(content) if isinstance(content, str) else None
        if not isinstance(parsed, dict):
            raise ValueError("invalid provider health schema")
        usage = response.get("usage")
        return ProviderPreflightResult(
            "provider", "preflight", "passed", "none", True, isinstance(usage, dict), None
        )
    except NativeToolCallingError as exc:
        return ProviderPreflightResult(
            "provider",
            "preflight",
            "failed",
            _map_transport_error(exc.code),
            False,
            False,
            type(exc).__name__,
        )
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return ProviderPreflightResult(
            "provider",
            "preflight",
            "failed",
            "provider_schema_incompatible",
            False,
            False,
            type(exc).__name__,
        )


def _map_transport_error(code: str) -> str:
    if code == "tool_calling_provider_http_error":
        return "provider_authentication_failed"
    if code == "tool_calling_provider_unavailable":
        return "provider_unreachable"
    return "provider_schema_incompatible"
