"""Append-only, hash-chained audit facts separate from best-effort tracing."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from decision_agent.security.provider_policy import DataClassification, ProviderStage


class AuditEventType(StrEnum):
    PROVIDER_CALL_ALLOWED = "provider_call_allowed"
    PROVIDER_CALL_BLOCKED = "provider_call_blocked"
    PROVIDER_CALL_COMPLETED = "provider_call_completed"
    PROVIDER_CALL_FAILED = "provider_call_failed"
    RESPONSE_RELEASE_ALLOWED = "response_release_allowed"
    RESPONSE_RELEASE_BLOCKED = "response_release_blocked"
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"


class AuditComponent(StrEnum):
    APPLICATION = "application"
    PROVIDER = "provider"
    RUNTIME = "runtime"


class AuditAction(StrEnum):
    COMPLETE = "complete"
    EGRESS = "egress"
    RESPONSE_RELEASE = "response_release"
    WORKFLOW = "workflow"


class AuditOutcome(StrEnum):
    ALLOWED = "allowed"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    STARTED = "started"


class AuditEvent(BaseModel):
    """Closed-schema compliance event; it deliberately carries no business payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str = Field(min_length=1, max_length=80)
    timestamp: datetime
    request_id: str = Field(min_length=1, max_length=128)
    trace_id: str = Field(min_length=1, max_length=128)
    principal_type: str = Field(min_length=1, max_length=24)
    tenant_hash: str = Field(min_length=16, max_length=64)
    event_type: AuditEventType
    component: AuditComponent
    action: AuditAction
    resource_type: str | None = Field(default=None, max_length=48)
    resource_id_hash: str | None = Field(default=None, min_length=16, max_length=64)
    authorization_decision: str | None = Field(default=None, pattern="^(allowed|denied)$")
    policy_id: str = Field(min_length=1, max_length=80)
    policy_version: str = Field(min_length=1, max_length=40)
    outcome: AuditOutcome
    error_code: str | None = Field(default=None, pattern=r"^[a-z0-9_]{1,80}$")
    latency_ms: int | None = Field(default=None, ge=0, le=3_600_000)
    provider_call_count: int | None = Field(default=None, ge=0, le=32)
    provider_stage: ProviderStage | None = None
    tool_category: str | None = Field(default=None, max_length=48)
    data_classification: DataClassification | None = None
    redaction_status: str | None = Field(default=None, max_length=32)
    previous_event_hash: str = Field(default="", max_length=64)
    event_hash: str = Field(default="", max_length=64)

    def with_chain_hash(self, previous_event_hash: str) -> AuditEvent:
        value = self.model_copy(
            update={"previous_event_hash": previous_event_hash, "event_hash": ""}
        )
        return value.model_copy(update={"event_hash": _event_hash(value)})


class AuditChainError(RuntimeError):
    """Safe local signal; callers convert critical failures to fixed public codes."""


class JsonlAuditSink:
    """Locked append-only JSONL writer with synchronous integrity verification."""

    def __init__(self, path: Path, *, fsync: bool = True) -> None:
        self._path = path
        self._anchor_path = path.with_suffix(path.suffix + ".anchor")
        self._fsync = fsync
        self._lock = threading.Lock()
        self._closed = False
        if self._anchor_path.exists() and not path.exists():
            raise AuditChainError("audit_chain_invalid")
        self._previous_hash = self.verify(path) if path.exists() else ""

    def append(self, event: AuditEvent) -> AuditEvent:
        with self._lock:
            if self._closed:
                raise AuditChainError("audit_sink_closed")
            self._path.parent.mkdir(parents=True, exist_ok=True)
            chained = event.with_chain_hash(self._previous_hash)
            line = _canonical(chained.model_dump(mode="json")) + "\n"
            try:
                with self._path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(line)
                    handle.flush()
                    if self._fsync:
                        os.fsync(handle.fileno())
                self._write_anchor(chained.event_hash)
            except OSError as exc:
                raise AuditChainError("audit_write_failed") from exc
            self._previous_hash = chained.event_hash
            return chained

    def close(self) -> None:
        with self._lock:
            self._closed = True

    def _write_anchor(self, event_hash: str) -> None:
        """Persist the committed chain tip so tail deletion is detectable."""
        temporary = self._anchor_path.with_suffix(self._anchor_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(event_hash)
            handle.flush()
            if self._fsync:
                os.fsync(handle.fileno())
        os.replace(temporary, self._anchor_path)

    @staticmethod
    def verify(path: Path) -> str:
        previous = ""
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise AuditChainError("audit_sink_unavailable") from exc
        if raw and not raw.endswith("\n"):
            raise AuditChainError("audit_integrity_verification_failed")
        lines = raw.splitlines()
        for line in lines:
            try:
                event = AuditEvent.model_validate_json(line)
            except Exception as exc:
                raise AuditChainError("audit_integrity_verification_failed") from exc
            if event.previous_event_hash != previous or event.event_hash != _event_hash(
                event.model_copy(update={"event_hash": ""})
            ):
                raise AuditChainError("audit_integrity_verification_failed")
            previous = event.event_hash
        anchor_path = path.with_suffix(path.suffix + ".anchor")
        if lines and not anchor_path.exists():
            raise AuditChainError("audit_chain_invalid")
        if anchor_path.exists():
            try:
                anchor = anchor_path.read_text(encoding="utf-8").strip()
            except OSError as exc:
                raise AuditChainError("audit_sink_unavailable") from exc
            if anchor != previous:
                raise AuditChainError("audit_chain_invalid")
        return previous


def new_audit_event(**values: object) -> AuditEvent:
    """Construct one event with a UTC timestamp while retaining a closed field set."""
    return AuditEvent(timestamp=datetime.now(UTC), **values)


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _event_hash(event: AuditEvent) -> str:
    return hashlib.sha256(
        _canonical(event.model_dump(mode="json", exclude={"event_hash"})).encode()
    ).hexdigest()
