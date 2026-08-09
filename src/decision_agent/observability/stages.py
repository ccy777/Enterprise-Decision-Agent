"""Stable observability stages and terminal span statuses."""

from enum import StrEnum


class TraceStage(StrEnum):
    """Bounded, refactor-stable execution stages for request diagnostics."""

    REQUEST = "request"
    MEMORY_READ = "memory_read"
    ROUTING = "routing"
    PROVIDER_CALL = "provider_call"
    COORDINATION = "coordination"
    AGENT_WORKFLOW = "agent_workflow"
    PLANNING = "planning"
    PLAN_STEP_EXECUTION = "plan_step_execution"
    WORKFLOW_REVIEW = "workflow_review"
    SKILL_EXECUTION = "skill_execution"
    TOOL_SELECTION = "tool_selection"
    TOOL_EXECUTION = "tool_execution"
    DATA_ACCESS = "data_access"
    RETRIEVAL = "retrieval"
    RERANKING = "reranking"
    EVIDENCE_SELECTION = "evidence_selection"
    ANSWER_GENERATION = "answer_generation"
    REVIEW = "review"
    MEMORY_WRITE = "memory_write"
    MEMORY_SUMMARY = "memory_summary"
    RESPONSE_MAPPING = "response_mapping"
    RUNTIME_BOOTSTRAP = "runtime_bootstrap"


class SpanStatus(StrEnum):
    """Execution semantics; unsupported and not-requested are not failures."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"
    UNSUPPORTED = "unsupported"
    NOT_REQUESTED = "not_requested"


TERMINAL_SPAN_STATUSES = frozenset(SpanStatus) - {SpanStatus.RUNNING}
