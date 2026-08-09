"""Closed allowlist and size controls for scalar trace attributes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from decision_agent.observability.models import TraceAttribute, TraceAttributeValue
from decision_agent.observability.stages import TraceStage

_COMMON_KEYS = frozenset(
    {
        "provider",
        "model",
        "retry_count",
        "input_tokens",
        "output_tokens",
        "tool_call_count",
        "result_count",
        "denied",
        "success",
    }
)
_STAGE_KEYS: dict[TraceStage, frozenset[str]] = {
    TraceStage.ROUTING: frozenset(
        {"route", "has_knowledge_subquery", "has_data_subquery", "latency_ms"}
    ),
    TraceStage.PROVIDER_CALL: frozenset(
        {"operation", "usage_available", "finish_reason", "success", "latency_ms"}
    ),
    TraceStage.SKILL_EXECUTION: frozenset(
        {
            "route",
            "skill_name",
            "execution_index",
            "selected_skill_count",
            "success",
            "result_status",
            "latency_ms",
        }
    ),
    TraceStage.TOOL_SELECTION: frozenset(
        {
            "tool_name",
            "authorized",
            "argument_validation",
            "denied",
            "tool_call_count",
            "selection_index",
            "query_source",
            "success",
            "latency_ms",
        }
    ),
    TraceStage.TOOL_EXECUTION: frozenset(
        {
            "tool_name",
            "authorized",
            "execution_index",
            "success",
            "result_status",
            "result_count",
            "result_truncated",
            "latency_ms",
        }
    ),
    TraceStage.RETRIEVAL: frozenset(
        {
            "requested_top_k",
            "dense_candidate_count",
            "sparse_candidate_count",
            "fused_candidate_count",
            "retrieved_count",
            "collection_name",
            "child_count",
            "parent_count",
            "expanded_count",
            "reranked_count",
            "empty_result",
            "success",
            "latency_ms",
        }
    ),
    TraceStage.RERANKING: frozenset(
        {
            "input_candidate_count",
            "requested_top_k",
            "reranked_count",
            "model",
            "success",
            "latency_ms",
        }
    ),
    TraceStage.EVIDENCE_SELECTION: frozenset(
        {
            "candidate_evidence_count",
            "selected_evidence_count",
            "answerable",
            "empty_result",
            "success",
            "result_status",
            "latency_ms",
        }
    ),
    TraceStage.DATA_ACCESS: frozenset(
        {
            "tool_name",
            "query_category",
            "authorized",
            "denied",
            "argument_validation",
            "row_count",
            "result_truncated",
            "timeout",
            "success",
            "result_status",
            "latency_ms",
        }
    ),
    TraceStage.ANSWER_GENERATION: frozenset(
        {
            "answer_type",
            "citation_count",
            "source_count",
            "success",
            "result_status",
            "latency_ms",
        }
    ),
    TraceStage.REVIEW: frozenset(
        {
            "reviewer_type",
            "review_passed",
            "review_outcome",
            "answerable",
            "success",
            "result_status",
            "latency_ms",
        }
    ),
    TraceStage.MEMORY_READ: frozenset(
        {"memory_requested", "result_count", "success", "latency_ms"}
    ),
    TraceStage.MEMORY_WRITE: frozenset({"written_item_count", "success", "latency_ms"}),
    TraceStage.MEMORY_SUMMARY: frozenset({"compacted_turn_count", "success", "latency_ms"}),
    TraceStage.RUNTIME_BOOTSTRAP: frozenset({"latency_ms"}),
    TraceStage.REQUEST: frozenset({"http_status", "latency_ms", "final_status", "session_present"}),
    TraceStage.COORDINATION: frozenset(
        {"route", "selected_skill_count", "skill_name", "success", "latency_ms"}
    ),
    TraceStage.RESPONSE_MAPPING: frozenset({"http_status", "latency_ms", "success"}),
    TraceStage.PLANNING: frozenset({"plan_version", "plan_step_count", "success", "latency_ms"}),
    TraceStage.AGENT_WORKFLOW: frozenset(
        {
            "plan_version",
            "plan_step_count",
            "repair_allowed",
            "success",
            "result_status",
            "latency_ms",
        }
    ),
    TraceStage.PLAN_STEP_EXECUTION: frozenset(
        {
            "execution_round",
            "skill_name",
            "skill_calls_used",
            "success",
            "result_status",
            "latency_ms",
        }
    ),
    TraceStage.WORKFLOW_REVIEW: frozenset(
        {
            "execution_round",
            "reviewer_outcome",
            "reason_code",
            "repair_attempts",
            "reviewer_calls_used",
            "budget_remaining",
            "success",
            "result_status",
            "latency_ms",
        }
    ),
}
_FORBIDDEN_KEYS = frozenset(
    {
        "prompt",
        "system_prompt",
        "messages",
        "query",
        "user_query",
        "rewritten_query",
        "sql",
        "raw_sql",
        "evidence",
        "context",
        "document_text",
        "tool_result",
        "rows",
        "response",
        "answer",
        "chain_of_thought",
        "reasoning",
        "api_key",
        "token",
        "authorization",
        "cookie",
        "password",
        "secret",
        "connection_string",
        "session_id",
        "local_path",
        "stack_trace",
    }
)


@dataclass(frozen=True, slots=True)
class AttributeLimits:
    """Independent per-span bounds; values are intentionally conservative."""

    max_attributes_per_span: int = 24
    max_attribute_value_length: int = 256

    def __post_init__(self) -> None:
        if self.max_attributes_per_span <= 0 or self.max_attribute_value_length <= 0:
            raise ValueError("attribute limits must be positive")


@dataclass(frozen=True, slots=True)
class AttributeSanitization:
    """Safe immutable attribute projection and count of omitted/truncated inputs."""

    attributes: tuple[TraceAttribute, ...]
    dropped_attribute_count: int


def sanitize_attributes(
    *,
    stage: TraceStage,
    values: Mapping[str, object] | None,
    limits: AttributeLimits,
) -> AttributeSanitization:
    """Keep only known scalar attributes without retaining rejected values."""
    if not values:
        return AttributeSanitization(attributes=(), dropped_attribute_count=0)

    allowed = _COMMON_KEYS | _STAGE_KEYS.get(stage, frozenset())
    attributes: list[TraceAttribute] = []
    dropped = 0
    seen: set[str] = set()
    for raw_key, raw_value in values.items():
        key = raw_key.strip().lower() if isinstance(raw_key, str) else ""
        if not key or key in seen or key in _FORBIDDEN_KEYS or key not in allowed:
            dropped += 1
            continue
        seen.add(key)
        if len(attributes) >= limits.max_attributes_per_span:
            dropped += 1
            continue
        value, value_dropped = _safe_scalar(raw_value, limits.max_attribute_value_length)
        if value_dropped and value is _REJECTED:
            dropped += 1
            continue
        dropped += int(value_dropped)
        attributes.append(TraceAttribute(key=key, value=value))
    return AttributeSanitization(attributes=tuple(attributes), dropped_attribute_count=dropped)


class _Rejected:
    pass


_REJECTED = _Rejected()


def _safe_scalar(value: object, max_length: int) -> tuple[TraceAttributeValue | _Rejected, bool]:
    if value is None or isinstance(value, (bool, int)):
        return value, False
    if isinstance(value, float):
        return (value, False) if math.isfinite(value) else (_REJECTED, True)
    if isinstance(value, str):
        if len(value) <= max_length:
            return value, False
        return value[:max_length], True
    return _REJECTED, True
