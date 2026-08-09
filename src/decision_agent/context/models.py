"""Immutable contracts for safe, policy-scoped prompt context."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from decision_agent.context.token_budget import TokenBudget

_EVIDENCE_CITATION = re.compile(r"^\[(?P<domain>[ED])(?P<number>[1-9]\d*)\]$")
_RESERVED_METADATA_KEYS = frozenset(
    {"kind", "source", "trust_level", "evidence_domain", "citation_ids", "system_instruction"}
)
_DERIVED_KINDS = frozenset(
    {
        "router_decision",
        "skill_instruction",
        "tool_result",
        "knowledge_evidence",
        "data_evidence",
        "structured_summary",
        "conversation_memory",
    }
)


class ContextKind(StrEnum):
    SYSTEM_INSTRUCTION = "system_instruction"
    USER_REQUEST = "user_request"
    ROUTER_DECISION = "router_decision"
    SKILL_INSTRUCTION = "skill_instruction"
    TOOL_RESULT = "tool_result"
    KNOWLEDGE_EVIDENCE = "knowledge_evidence"
    DATA_EVIDENCE = "data_evidence"
    STRUCTURED_SUMMARY = "structured_summary"
    CONVERSATION_MEMORY = "conversation_memory"


class ContextSource(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ROUTER = "router"
    SKILL = "skill"
    TOOL = "tool"
    AGENT = "agent"
    EXTERNAL = "external"


class TrustLevel(StrEnum):
    TRUSTED_SYSTEM = "trusted_system"
    VERIFIED_INTERNAL = "verified_internal"
    TRUSTED_TOOL_RESULT = "trusted_tool_result"
    UNTRUSTED_USER = "untrusted_user"
    UNTRUSTED_EXTERNAL = "untrusted_external"


class EvidenceDomain(StrEnum):
    KNOWLEDGE = "knowledge"
    DATA = "data"


class ContextDropReason(StrEnum):
    EXPIRED = "expired"
    KIND_NOT_ALLOWED = "kind_not_allowed"
    TRUST_NOT_ALLOWED = "trust_not_allowed"
    EVIDENCE_DOMAIN_NOT_ALLOWED = "evidence_domain_not_allowed"
    TOKEN_BUDGET_EXCEEDED = "token_budget_exceeded"


class _FrozenMapping(Mapping[str, Any]):
    """Private recursively immutable mapping that remains safe to deepcopy."""

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, Any]) -> None:
        self._values = dict(values)

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __deepcopy__(self, memo: dict[int, Any]) -> _FrozenMapping:
        memo[id(self)] = self
        return self


def _require_nonblank(value: str, field_name: str) -> str:
    if not value or not value.strip():
        raise ValueError(f"{field_name} must be non-empty")
    return value


def _require_aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def _deduplicate_strings(values: Any, field_name: str) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        _require_nonblank(value, field_name)
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def _freeze_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metadata floats must be finite JSON-safe values")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, nested_value in value.items():
            if not isinstance(key, str):
                raise ValueError("metadata keys must be strings")
            frozen[key] = _freeze_json(nested_value)
        return _FrozenMapping(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(nested_value) for nested_value in value)
    raise ValueError("metadata must contain only JSON-safe values")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(nested_value) for key, nested_value in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(nested_value) for nested_value in value]
    return value


class ContextProvenance(BaseModel):
    """Small, traceable provenance without prompts, credentials, or raw provider output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    producer: str
    source_item_ids: tuple[str, ...] = ()
    request_id: str
    trace_id: str | None = None
    generated_at: datetime

    @field_validator("producer", "request_id")
    @classmethod
    def _nonblank_required(cls, value: str) -> str:
        return _require_nonblank(value, "provenance field")

    @field_validator("trace_id")
    @classmethod
    def _nonblank_optional(cls, value: str | None) -> str | None:
        return None if value is None else _require_nonblank(value, "trace_id")

    @field_validator("source_item_ids", mode="before")
    @classmethod
    def _normalize_source_ids(cls, value: Any) -> tuple[str, ...]:
        return _deduplicate_strings(value, "source_item_id")

    @field_validator("generated_at")
    @classmethod
    def _aware_timestamp(cls, value: datetime) -> datetime:
        return _require_aware(value, "generated_at")


class ContextItem(BaseModel):
    """One immutable, explicitly classified context entry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: str
    kind: ContextKind
    content: str = Field(max_length=20_000, repr=False)
    source: ContextSource
    trust_level: TrustLevel
    provenance: ContextProvenance
    created_at: datetime
    expires_at: datetime | None = None
    estimated_tokens: int = Field(ge=0)
    evidence_domain: EvidenceDomain | None = None
    citation_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("item_id", "content")
    @classmethod
    def _nonblank_text(cls, value: str) -> str:
        return _require_nonblank(value, "context text")

    @field_validator("created_at")
    @classmethod
    def _aware_created_at(cls, value: datetime) -> datetime:
        return _require_aware(value, "created_at")

    @field_validator("expires_at")
    @classmethod
    def _aware_expires_at(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _require_aware(value, "expires_at")

    @field_validator("citation_ids", mode="before")
    @classmethod
    def _normalize_citations(cls, value: Any) -> tuple[str, ...]:
        return _deduplicate_strings(value, "citation_id")

    @field_validator("metadata", mode="before")
    @classmethod
    def _freeze_metadata(cls, value: Any) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise ValueError("metadata must be a mapping")
        reserved = _RESERVED_METADATA_KEYS.intersection(value)
        if reserved:
            raise ValueError("metadata contains reserved keys")
        frozen = _freeze_json(value)
        assert isinstance(frozen, Mapping)
        return frozen

    @field_serializer("metadata")
    def _serialize_metadata(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return _thaw_json(value)

    @model_validator(mode="after")
    def _validate_contract(self) -> ContextItem:
        # Pydantic normalizes a ``Mapping`` field to ``dict`` after field validators;
        # re-freeze the root so callers cannot mutate it through the public model.
        object.__setattr__(self, "metadata", _freeze_json(self.metadata))
        if self.expires_at is not None and self.expires_at < self.created_at:
            raise ValueError("expires_at must not be before created_at")
        if self.kind is ContextKind.SYSTEM_INSTRUCTION and self.source is not ContextSource.SYSTEM:
            raise ValueError("system_instruction must originate from system")
        if (
            self.trust_level is TrustLevel.TRUSTED_SYSTEM
            and self.source is not ContextSource.SYSTEM
        ):
            raise ValueError("trusted_system requires system source")
        if (
            self.source is ContextSource.SYSTEM
            and self.trust_level is not TrustLevel.TRUSTED_SYSTEM
        ):
            raise ValueError("system source must use trusted_system")
        if self.kind is ContextKind.USER_REQUEST and self.source is not ContextSource.USER:
            raise ValueError("user_request must originate from user")
        if self.source is ContextSource.USER and self.trust_level is TrustLevel.TRUSTED_SYSTEM:
            raise ValueError("user source cannot use trusted_system")
        if self.source is ContextSource.EXTERNAL and self.trust_level in {
            TrustLevel.TRUSTED_SYSTEM,
            TrustLevel.VERIFIED_INTERNAL,
        }:
            raise ValueError("external source cannot use internal trust")
        if self.kind is ContextKind.TOOL_RESULT and self.source not in {
            ContextSource.TOOL,
            ContextSource.AGENT,
        }:
            raise ValueError("tool_result must originate from tool or agent")

        evidence_kinds = {ContextKind.KNOWLEDGE_EVIDENCE, ContextKind.DATA_EVIDENCE}
        if self.kind in evidence_kinds:
            if self.source is ContextSource.SYSTEM or self.trust_level is TrustLevel.TRUSTED_SYSTEM:
                raise ValueError("evidence cannot use trusted system origin")
            expected_domain = (
                EvidenceDomain.KNOWLEDGE
                if self.kind is ContextKind.KNOWLEDGE_EVIDENCE
                else EvidenceDomain.DATA
            )
            expected_prefix = "E" if expected_domain is EvidenceDomain.KNOWLEDGE else "D"
            if self.evidence_domain is not expected_domain or not self.citation_ids:
                raise ValueError("evidence requires its domain and citations")
            if any(
                (match := _EVIDENCE_CITATION.fullmatch(citation)) is None
                or match.group("domain") != expected_prefix
                for citation in self.citation_ids
            ):
                raise ValueError("evidence citations must match their evidence domain")
        elif self.evidence_domain is not None or self.citation_ids:
            raise ValueError("non-evidence items cannot carry evidence domain or citations")

        if self.kind.value in _DERIVED_KINDS and not self.provenance.source_item_ids:
            raise ValueError("derived context items require source_item_ids")
        if self.item_id in self.provenance.source_item_ids:
            raise ValueError("context item cannot directly cite itself as a source")
        return self

    def is_expired(self, at: datetime) -> bool:
        """Return expiration at the supplied instant without reading system time."""
        _require_aware(at, "at")
        return self.expires_at is not None and at >= self.expires_at


class ContextPolicy(BaseModel):
    """Immutable allowlist and budget applied by one named workflow node."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_name: str
    allowed_kinds: frozenset[ContextKind]
    allowed_trust_levels: frozenset[TrustLevel]
    allowed_evidence_domains: frozenset[EvidenceDomain]
    token_budget: TokenBudget
    required_item_ids: tuple[str, ...] = ()

    @field_validator("node_name")
    @classmethod
    def _nonblank_node_name(cls, value: str) -> str:
        return _require_nonblank(value, "node_name")

    @field_validator("allowed_kinds", "allowed_trust_levels")
    @classmethod
    def _nonempty_sets(cls, value: frozenset[Any]) -> frozenset[Any]:
        if not value:
            raise ValueError("allowed policy sets must be non-empty")
        return value

    @field_validator("required_item_ids", mode="before")
    @classmethod
    def _normalize_required_ids(cls, value: Any) -> tuple[str, ...]:
        return _deduplicate_strings(value, "required_item_id")


class DroppedContextItem(BaseModel):
    """An ordinary candidate rejected by a stable, non-content reason."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: str
    reason: ContextDropReason

    @field_validator("item_id")
    @classmethod
    def _nonblank_item_id(cls, value: str) -> str:
        return _require_nonblank(value, "item_id")


class ContextSelectionResult(BaseModel):
    """Immutable deterministic selection result without transformed citations or items."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    selected_items: tuple[ContextItem, ...]
    selected_item_ids: tuple[str, ...]
    dropped_items: tuple[DroppedContextItem, ...]
    total_estimated_tokens: int = Field(ge=0)
    available_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_consistency(self) -> ContextSelectionResult:
        item_ids = tuple(item.item_id for item in self.selected_items)
        dropped_ids = tuple(item.item_id for item in self.dropped_items)
        if self.selected_item_ids != item_ids:
            raise ValueError("selected_item_ids must match selected_items")
        if len(set(item_ids)) != len(item_ids) or len(set(dropped_ids)) != len(dropped_ids):
            raise ValueError("selection item identifiers must be unique")
        if set(item_ids).intersection(dropped_ids):
            raise ValueError("an item cannot be selected and dropped")
        if self.total_estimated_tokens != sum(
            item.estimated_tokens for item in self.selected_items
        ):
            raise ValueError("total_estimated_tokens must match selected items")
        if self.total_estimated_tokens > self.available_tokens:
            raise ValueError("selected items exceed available_tokens")
        return self
