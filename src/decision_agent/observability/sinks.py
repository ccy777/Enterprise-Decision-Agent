"""Safe request-trace sinks and best-effort fan-out."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from decision_agent.observability.collector import TraceLimits
from decision_agent.observability.models import RequestTrace

_EVENT_NAME = "decision_agent.request_trace"
_FALLBACK_ERROR_CODE = "observability_sink_failed"
_FALLBACK_LOGGER = logging.getLogger("decision_agent.observability")


class TraceSinkError(RuntimeError):
    """Safe sink-boundary error without retaining underlying exception details."""


class TraceSink(Protocol):
    """Emit only completed immutable request traces."""

    def emit(self, trace: RequestTrace) -> None:
        """Emit one safe completed trace."""


@dataclass(frozen=True, slots=True)
class TraceEmissionResult:
    """Safe best-effort fan-out summary, separate from business result semantics."""

    attempted_sink_count: int
    succeeded_sink_count: int
    failed_sink_count: int


class InMemoryTraceSink:
    """Thread-safe bounded sink retaining latest completed traces in insertion order."""

    def __init__(self, *, max_traces: int = 256) -> None:
        if max_traces <= 0:
            raise ValueError("max_traces must be positive")
        self._max_traces = max_traces
        self._lock = threading.Lock()
        self._traces: list[RequestTrace] = []

    def emit(self, trace: RequestTrace) -> None:
        if not isinstance(trace, RequestTrace):
            raise TraceSinkError("observability_sink_invalid_trace")
        with self._lock:
            self._traces.append(trace)
            overflow = len(self._traces) - self._max_traces
            if overflow > 0:
                del self._traces[:overflow]

    def snapshot(self) -> tuple[RequestTrace, ...]:
        """Return an immutable insertion-ordered snapshot."""
        with self._lock:
            return tuple(self._traces)

    def find(self, trace_id: str) -> RequestTrace | None:
        """Return one immutable trace by opaque identifier."""
        with self._lock:
            return next((trace for trace in self._traces if trace.trace_id == trace_id), None)


class StructuredLoggingTraceSink:
    """Emit one bounded JSON line through an injected standard-library logger."""

    def __init__(
        self,
        *,
        logger: logging.Logger,
        limits: TraceLimits | None = None,
    ) -> None:
        self._logger = logger
        self._limits = limits or TraceLimits()

    def emit(self, trace: RequestTrace) -> None:
        if not isinstance(trace, RequestTrace):
            raise TraceSinkError("observability_sink_invalid_trace")
        try:
            payload = _full_payload(trace)
            serialized = _serialize(payload)
            if len(serialized.encode("utf-8")) > self._limits.max_serialized_trace_bytes:
                serialized = _serialize(_summary_payload(trace))
            self._logger.info(serialized)
        except (OSError, TypeError, ValueError) as exc:
            raise TraceSinkError(_FALLBACK_ERROR_CODE) from exc


class BestEffortTraceDispatcher:
    """Fan out to independent sinks without letting telemetry change business outcomes."""

    def __init__(
        self,
        sinks: Iterable[TraceSink],
        *,
        fallback_logger: logging.Logger | None = None,
    ) -> None:
        self._sinks = tuple(sinks)
        self._fallback_logger = fallback_logger or _FALLBACK_LOGGER

    def emit(self, trace: RequestTrace) -> TraceEmissionResult:
        succeeded = 0
        failed = 0
        for sink in self._sinks:
            try:
                sink.emit(trace)
            except (TraceSinkError, OSError, TypeError, ValueError):
                failed += 1
                self._log_safe_failure(trace, sink)
            else:
                succeeded += 1
        return TraceEmissionResult(
            attempted_sink_count=len(self._sinks),
            succeeded_sink_count=succeeded,
            failed_sink_count=failed,
        )

    def _log_safe_failure(self, trace: RequestTrace, sink: TraceSink) -> None:
        payload = {
            "event": _FALLBACK_ERROR_CODE,
            "trace_id": trace.trace_id,
            "sink_type": _sink_type_name(sink),
            "error_code": _FALLBACK_ERROR_CODE,
        }
        try:
            self._fallback_logger.warning(_serialize(payload))
        except (OSError, TypeError, ValueError):
            return


def _full_payload(trace: RequestTrace) -> dict[str, object]:
    return {
        "event": _EVENT_NAME,
        "trace_id": trace.trace_id,
        "request_id": trace.request_id,
        "session_present": trace.session_present,
        "started_at": trace.started_at.isoformat(),
        "completed_at": trace.completed_at.isoformat(),
        "duration_ms": trace.duration_ms,
        "final_status": trace.final_status.value,
        "route": trace.route,
        "skill_name": trace.skill_name,
        "error_code": trace.error_code,
        "span_count": trace.span_count,
        "dropped_span_count": trace.dropped_span_count,
        "dropped_attribute_count": trace.dropped_attribute_count,
        "sink_failure_count": trace.sink_failure_count,
        "trace_truncated": False,
        "spans": [
            {
                "span_id": span.span_id,
                "parent_span_id": span.parent_span_id,
                "stage": span.stage.value,
                "component": span.component,
                "operation": span.operation,
                "started_at": span.started_at.isoformat(),
                "completed_at": span.completed_at.isoformat(),
                "duration_ms": span.duration_ms,
                "status": span.status.value,
                "error_code": span.error_code,
                "attributes": {attribute.key: attribute.value for attribute in span.attributes},
                "dropped_attribute_count": span.dropped_attribute_count,
            }
            for span in trace.spans
        ],
    }


def _summary_payload(trace: RequestTrace) -> dict[str, object]:
    return {
        "event": _EVENT_NAME,
        "trace_id": trace.trace_id,
        "request_id": trace.request_id,
        "final_status": trace.final_status.value,
        "span_count": trace.span_count,
        "dropped_span_count": trace.dropped_span_count,
        "dropped_attribute_count": trace.dropped_attribute_count,
        "sink_failure_count": trace.sink_failure_count,
        "trace_truncated": True,
    }


def _serialize(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _sink_type_name(sink: TraceSink) -> str:
    name = type(sink).__name__.lower()
    return "".join(character for character in name if character.isalnum() or character == "_")[:64]
