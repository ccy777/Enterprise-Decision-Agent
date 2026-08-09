"""Small synchronous store protocol and content-safe memory errors."""

from __future__ import annotations

from typing import Protocol

from decision_agent.memory.models import SessionMemorySnapshot, SessionSummary, SessionTurn


class SessionMemoryError(Exception):
    """Base class for deterministic session-memory failures."""


class SessionVersionConflictError(SessionMemoryError):
    """Raised when a write would overwrite a newer session version."""

    def __init__(self, *, session_id: str, expected_version: int, actual_version: int) -> None:
        super().__init__(
            "session memory version conflict: "
            f"session_id={session_id}, expected_version={expected_version}, "
            f"actual_version={actual_version}"
        )


class SessionTurnConflictError(SessionMemoryError):
    """Raised when a reused turn or request identifier has different content."""

    def __init__(self, *, session_id: str, turn_id: str, request_id: str) -> None:
        super().__init__(
            "session memory turn conflict: "
            f"session_id={session_id}, turn_id={turn_id}, request_id={request_id}"
        )


class SessionMemoryContentionError(SessionMemoryError):
    """Raised after bounded Redis transaction retries are exhausted."""

    def __init__(self, *, session_id: str, retries: int) -> None:
        super().__init__(f"session memory contention: session_id={session_id}, retries={retries}")


class SessionMemoryUnavailableError(SessionMemoryError):
    """Raised when Redis cannot safely perform a memory operation."""

    def __init__(self, *, operation: str) -> None:
        super().__init__(f"session memory unavailable: operation={operation}")


class SessionMemoryCorruptionError(SessionMemoryError):
    """Raised when a persisted Redis value cannot satisfy the memory schema."""

    def __init__(self, *, session_id: str) -> None:
        super().__init__(f"session memory corruption: session_id={session_id}")


class SessionCompactionPrefixError(SessionMemoryError):
    """Raised when compaction does not select a strict oldest-turn prefix."""

    def __init__(self, *, session_id: str, compacted_turn_count: int) -> None:
        super().__init__(
            "session memory compaction prefix invalid: "
            f"session_id={session_id}, compacted_turn_count={compacted_turn_count}"
        )


class SessionSummaryLineageConflictError(SessionMemoryError):
    """Raised when a candidate summary does not continue the stored summary lineage."""

    def __init__(self, *, session_id: str, summary_id: str) -> None:
        super().__init__(
            "session memory summary lineage conflict: "
            f"session_id={session_id}, summary_id={summary_id}"
        )


class SessionSummaryIdentityConflictError(SessionMemoryError):
    """Raised when a summary ID is reused for unequal summary content or lineage."""

    def __init__(self, *, session_id: str, summary_id: str) -> None:
        super().__init__(
            "session memory summary identity conflict: "
            f"session_id={session_id}, summary_id={summary_id}"
        )


class SessionMemoryStore(Protocol):
    """Synchronous optimistic-concurrency contract for completed session turns."""

    def read(self, session_id: str) -> SessionMemorySnapshot:
        """Read a session without extending its lifetime."""

    def append_turn(self, turn: SessionTurn, *, expected_version: int) -> SessionMemorySnapshot:
        """Atomically append one successful turn or return an idempotent retry."""

    def clear(self, session_id: str, *, expected_version: int) -> SessionMemorySnapshot:
        """Atomically clear a session when its version matches."""

    def compact(
        self,
        summary: SessionSummary,
        compacted_turn_ids: tuple[str, ...],
        *,
        expected_version: int,
    ) -> SessionMemorySnapshot:
        """Atomically save a summary while removing a strict oldest-turn prefix."""


def validate_compaction(
    current: SessionMemorySnapshot,
    summary: SessionSummary,
    compacted_turn_ids: tuple[str, ...],
    *,
    expected_version: int,
) -> bool:
    """Validate compact semantics and return whether it is an idempotent retry.

    The function is shared by local and Redis stores so only their persistence mechanics differ.
    """
    existing = current.summary
    if existing is not None:
        if existing == summary:
            return True
        if existing.summary_id == summary.summary_id:
            raise SessionSummaryIdentityConflictError(
                session_id=current.session_id, summary_id=summary.summary_id
            )
    if expected_version != current.version:
        raise SessionVersionConflictError(
            session_id=current.session_id,
            expected_version=expected_version,
            actual_version=current.version,
        )
    if summary.session_id != current.session_id or summary.source_version != expected_version:
        raise SessionSummaryLineageConflictError(
            session_id=current.session_id, summary_id=summary.summary_id
        )
    if (
        not isinstance(compacted_turn_ids, tuple)
        or not compacted_turn_ids
        or len(set(compacted_turn_ids)) != len(compacted_turn_ids)
        or len(compacted_turn_ids) >= len(current.turns)
        or any(
            not isinstance(turn_id, str) or not turn_id.strip() for turn_id in compacted_turn_ids
        )
    ):
        raise SessionCompactionPrefixError(
            session_id=current.session_id, compacted_turn_count=len(compacted_turn_ids)
        )
    if existing is None:
        lineage_matches = summary.previous_summary_id is None
        count_matches = summary.covered_turn_count == len(compacted_turn_ids)
    else:
        lineage_matches = summary.previous_summary_id == existing.summary_id
        count_matches = summary.covered_turn_count == existing.covered_turn_count + len(
            compacted_turn_ids
        )
    if not lineage_matches or not count_matches:
        raise SessionSummaryLineageConflictError(
            session_id=current.session_id, summary_id=summary.summary_id
        )
    if (
        tuple(turn.turn_id for turn in current.turns[: len(compacted_turn_ids)])
        != compacted_turn_ids
    ):
        raise SessionCompactionPrefixError(
            session_id=current.session_id, compacted_turn_count=len(compacted_turn_ids)
        )
    if summary.covered_through_turn_id != compacted_turn_ids[-1]:
        raise SessionSummaryLineageConflictError(
            session_id=current.session_id, summary_id=summary.summary_id
        )
    return False
