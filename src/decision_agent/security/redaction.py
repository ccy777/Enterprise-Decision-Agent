"""Deterministic, payload-free sensitive-content boundary checks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

from decision_agent.security.provider_policy import DataClassification, ProviderPolicyError

_PATTERNS = (
    re.compile(r"(?i)\b(?:api[_-]?key|token|password|secret)\s*[:=]\s*\S+"),
    re.compile(r"(?i)authorization\s*:\s*bearer\s+\S+"),
    re.compile(r"(?i)(?:mysql|postgres(?:ql)?|redis)://\S+"),
    re.compile(r"(?i)(?:[a-z]:\\|/)(?:[^\s<>:\\]+[\\/])+[^\s<>]+"),
    re.compile(r"(?i)traceback \(most recent call last\):|raw stderr"),
)
_SQL_PATTERN = re.compile(
    r"(?is)\bselect\b[\s\S]{0,2000}\bfrom\b|"
    r"^\s*(?:insert\s+into|update\s+\w+\s+set|delete\s+from|drop\s+\w+|alter\s+\w+)"
)
_IDENTITY_FIELD_NAMES = frozenset({"tenant_id", "user_id", "subject_id", "roles", "session_id"})
_FORBIDDEN_FIELD_NAMES = frozenset(
    {
        "authorization",
        "connection_string",
        "database_url",
        "password",
        "raw_error",
        "raw_rows",
        "rows",
        "secret",
        "sql",
        "token",
    }
)


@dataclass(frozen=True, slots=True)
class RedactionResult:
    text: str
    redacted_count: int
    sensitive_detected: bool


class ProviderRedactor(Protocol):
    """Injectable deterministic boundary used only by provider governance."""

    def sanitize_payload(
        self, value: Any, *, classification: DataClassification
    ) -> tuple[Any, int]: ...

    def ensure_safe_output(self, value: str, *, allow_sql: bool = False) -> None: ...


class DeterministicProviderRedactor:
    """Default server-owned implementation with no state or external dependency."""

    def sanitize_payload(
        self, value: Any, *, classification: DataClassification
    ) -> tuple[Any, int]:
        return sanitize_provider_payload(value, classification=classification)

    def ensure_safe_output(self, value: str, *, allow_sql: bool = False) -> None:
        ensure_safe_provider_output(value, allow_sql=allow_sql)


def sanitize_provider_text(value: str, *, classification: DataClassification) -> RedactionResult:
    """Replace bounded textual markers; Restricted/Secret input is never transformed through."""
    if classification >= DataClassification.RESTRICTED:
        raise ProviderPolicyError("provider_data_classification_forbidden")
    if any(pattern.search(value) for pattern in (*_PATTERNS, _SQL_PATTERN)):
        raise ProviderPolicyError("provider_sensitive_input_detected")
    return RedactionResult(text=value, redacted_count=0, sensitive_detected=False)


def sanitize_provider_payload(value: Any, *, classification: DataClassification) -> tuple[Any, int]:
    """Recursively deny identity/structured sensitive fields rather than string-replacing them."""
    if classification >= DataClassification.RESTRICTED:
        raise ProviderPolicyError("provider_data_classification_forbidden")
    if isinstance(value, str):
        result = sanitize_provider_text(value, classification=classification)
        return result.text, result.redacted_count
    if isinstance(value, list):
        sanitized = [
            sanitize_provider_payload(item, classification=classification) for item in value
        ]
        return [item for item, _ in sanitized], sum(count for _, count in sanitized)
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        count = 0
        for key, item in value.items():
            normalized_key = key.lower()
            if normalized_key in _FORBIDDEN_FIELD_NAMES:
                raise ProviderPolicyError("provider_sensitive_input_detected")
            if normalized_key in _IDENTITY_FIELD_NAMES:
                output[key] = "[REDACTED]"
                count += 1
            else:
                output[key], nested = sanitize_provider_payload(item, classification=classification)
                count += nested
        return output, count
    return value, 0


def ensure_safe_provider_output(value: str, *, allow_sql: bool = False) -> None:
    """Fail closed if a provider completion still contains prohibited text markers."""
    patterns = _PATTERNS if allow_sql else (*_PATTERNS, _SQL_PATTERN)
    if any(pattern.search(value) for pattern in patterns):
        raise ProviderPolicyError("provider_sensitive_output_detected")
