"""One-read Session Memory facade ahead of the existing Coordinator."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime

from decision_agent.application.models import (
    FormalRequest,
    FormalResponse,
    MemoryContextStatus,
    MemoryPersistenceStatus,
    MemorySummarizationStatus,
)
from decision_agent.application.turn_identity import derive_turn_id
from decision_agent.context.conversation_memory import ConversationMemoryProjector
from decision_agent.coordination import Coordinator
from decision_agent.coordination.models import CoordinatorResult, CoordinatorStatus
from decision_agent.exceptions import DecisionAgentError
from decision_agent.memory import (
    RollingSummaryGenerationError,
    RollingSummaryInputTooLarge,
    RollingSummaryOutputInvalid,
    RollingSummaryService,
    RollingSummaryStatus,
    SessionMemorySnapshot,
    SessionMemoryStore,
    SessionTurn,
    SessionTurnConflictError,
    SessionVersionConflictError,
)
from decision_agent.memory.store import SessionMemoryError
from decision_agent.observability import (
    BestEffortTraceDispatcher,
    SpanStatus,
    TraceCollector,
    TraceContext,
    TraceExecution,
    TraceStage,
    build_trace_summary,
    new_trace_id,
)
from decision_agent.observability.models import RequestTrace
from decision_agent.security import (
    AuthorizationPolicy,
    ProviderGovernance,
    ProviderPolicyError,
    SecurityAuthorizationError,
    SecurityContext,
    SecurityErrorCode,
)

_REQUEST_EXECUTION_FAILURE_CODE = "formal_request_execution_failed"
_RESPONSE_MAPPING_FAILURE_CODE = "response_mapping_failed"
TraceCollectorFactory = Callable[[FormalRequest], TraceCollector]


class SessionMemoryReadError(DecisionAgentError):
    """Stable, content-safe failure for any configured Store read problem."""

    code = "session_memory_read_failed"

    def __init__(self) -> None:
        super().__init__("Session memory could not be read")


class FormalRequestExecutor:
    """Read session memory once before dispatching and persist successful turns.

    The executor owns the sole Store reference in this stage.  It never constructs a
    provider, router, Context Runtime, or Redis client.
    """

    def __init__(
        self,
        *,
        coordinator: Coordinator,
        memory_store: SessionMemoryStore | None,
        memory_projector: ConversationMemoryProjector,
        rolling_summary_service: RollingSummaryService | None = None,
        clock: Callable[[], datetime] | None = None,
        trace_collector_factory: TraceCollectorFactory | None = None,
        trace_dispatcher: BestEffortTraceDispatcher | None = None,
        authorization_policy: AuthorizationPolicy | None = None,
        provider_governance: ProviderGovernance | None = None,
    ) -> None:
        if coordinator is None or memory_projector is None:
            raise ValueError("coordinator and memory_projector must not be None")
        self._coordinator = coordinator
        self._memory_store = memory_store
        self._memory_projector = memory_projector
        self._rolling_summary_service = rolling_summary_service
        self._clock = clock or (lambda: datetime.now(UTC))
        self._trace_collector_factory = trace_collector_factory or _default_trace_collector
        self._trace_dispatcher = trace_dispatcher
        self._authorization_policy = authorization_policy
        self._provider_governance = provider_governance

    @property
    def requires_security_context(self) -> bool:
        """Expose only whether this composition root requires M8C authorization."""
        return self._authorization_policy is not None

    async def execute(self, request: FormalRequest) -> FormalResponse:
        """Execute one request with zero or one synchronous Store reads."""
        trace = TraceExecution.start(
            collector_factory=lambda: self._trace_collector_factory(request),
            dispatcher=self._trace_dispatcher,
        )
        root_span = trace.start_span(
            stage=TraceStage.REQUEST,
            component="application",
            operation="execute_formal_request",
            attributes={"session_present": request.session_id is not None},
        )
        final_status = SpanStatus.FAILED
        final_error_code: str | None = _REQUEST_EXECUTION_FAILURE_CODE
        response: FormalResponse | None = None
        request_trace: RequestTrace | None = None
        try:
            scoped_request, scope_denial = self._bind_session_scope(request)
            denial = self._security_denial(request) or scope_denial
            if denial is not None:
                response = self._authorization_failure_response(request, denial, trace, root_span)
            elif scoped_request.session_id is None or self._memory_store is None:
                trace.memory_not_requested(parent_context=root_span)
                response = await self._execute_without_memory(scoped_request, trace, root_span)
            else:
                response = await self._execute_with_memory(scoped_request, trace, root_span)
            final_status, final_error_code = _trace_terminal_status(response)
        except asyncio.CancelledError:
            final_status = SpanStatus.CANCELLED
            final_error_code = None
            trace.cancel_active_span()
            raise
        except (KeyboardInterrupt, SystemExit):
            trace.cancel_active_span()
            raise
        except Exception as exc:
            final_status = SpanStatus.FAILED
            final_error_code = _safe_execution_error_code(exc)
            raise
        finally:
            request_trace = trace.finish(
                root_span=root_span,
                final_status=final_status,
                error_code=final_error_code,
                route=(
                    response.result.route.value
                    if response is not None and response.result.route is not None
                    else None
                ),
                skill_name=response.result.skill_name if response is not None else None,
            )
        if response is None:  # pragma: no cover - defensive execution invariant
            raise RuntimeError("formal response is unavailable")
        response = _attach_trace_summary(response, request_trace)
        return self._release_response(request=request, response=response)

    def _release_response(
        self, *, request: FormalRequest, response: FormalResponse
    ) -> FormalResponse:
        """Prevent an answer from crossing the formal boundary without durable approval."""
        if self._provider_governance is None or request.security_context is None:
            return response
        allowed = (
            response.result.status is CoordinatorStatus.COMPLETED
            and response.result.answer is not None
            and bool(response.result.citations)
        )
        try:
            self._provider_governance.audit_for_request(
                request_id=request.request_id,
                trace_id=request.security_context.trace_id,
                security_context=request.security_context,
                event_type="response_release_allowed" if allowed else "response_release_blocked",
                outcome="allowed" if allowed else "blocked",
                error_code=None if allowed else "response_release_blocked",
            )
        except Exception:
            return self._release_blocked_response(request)
        if allowed:
            return response
        return self._release_blocked_response(request)

    @staticmethod
    def _release_blocked_response(request: FormalRequest) -> FormalResponse:
        return FormalResponse(
            request_id=request.request_id,
            result=CoordinatorResult(
                status=CoordinatorStatus.FAILED,
                coordinator_steps=("response_release",),
                error_code="response_release_blocked",
            ),
            memory_context_status=MemoryContextStatus.NOT_REQUESTED,
            memory_persistence_status=MemoryPersistenceStatus.NOT_REQUESTED,
            memory_summarization_status=MemorySummarizationStatus.NOT_REQUESTED,
        )

    async def _execute_with_memory(
        self,
        request: FormalRequest,
        trace: TraceExecution,
        request_span: TraceContext | None,
    ) -> FormalResponse:
        memory_span = trace.start_span(
            stage=TraceStage.MEMORY_READ,
            component="application",
            operation="load_conversation_memory",
            parent_context=request_span,
            attributes={"memory_requested": True},
        )
        read_failed = False
        try:
            snapshot = await asyncio.to_thread(self._memory_store.read, request.session_id)
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            trace.complete_span(memory_span, status=SpanStatus.CANCELLED)
            raise
        except Exception:
            read_failed = True
        if read_failed:
            trace.complete_span(
                memory_span,
                status=SpanStatus.FAILED,
                error_code=SessionMemoryReadError.code,
            )
            # Raise outside ``except`` so no Store exception remains on the public error.
            raise SessionMemoryReadError()

        trace.complete_span(
            memory_span,
            status=SpanStatus.COMPLETED,
            attributes={"result_count": len(snapshot.turns), "success": True},
        )

        projection = self._memory_projector.project(snapshot)
        result = await self._execute_coordinator(
            request=request,
            conversation_memory=projection,
            trace=trace,
            request_span=request_span,
        )
        if projection is None:
            status = (
                MemoryContextStatus.EMPTY
                if snapshot.summary is None and not snapshot.turns
                else MemoryContextStatus.OMITTED_BY_BUDGET
            )
        elif result.memory_context_selected:
            status = MemoryContextStatus.PROJECTED
        else:
            status = MemoryContextStatus.OMITTED_BY_BUDGET
        persistence_status, post_append_snapshot = await self._persist_successful_turn(
            request=request,
            result=result,
            snapshot=snapshot,
            trace=trace,
            request_span=request_span,
        )
        summarization_status = await self._summarize_persisted_turn(
            request=request,
            persistence_status=persistence_status,
            post_append_snapshot=post_append_snapshot,
            trace=trace,
            request_span=request_span,
        )
        return self._build_formal_response(
            request_id=request.request_id,
            result=result,
            memory_context_status=status,
            memory_persistence_status=persistence_status,
            memory_summarization_status=summarization_status,
            trace=trace,
            request_span=request_span,
        )

    async def _execute_without_memory(
        self,
        request: FormalRequest,
        trace: TraceExecution,
        request_span: TraceContext | None,
    ) -> FormalResponse:
        """Execute without configured or requested session-memory behavior."""
        _record_memory_not_requested(
            trace,
            stage=TraceStage.MEMORY_WRITE,
            operation="persist_conversation_memory",
            parent_context=request_span,
        )
        _record_memory_not_requested(
            trace,
            stage=TraceStage.MEMORY_SUMMARY,
            operation="summarize_conversation_memory",
            parent_context=request_span,
        )
        result = await self._execute_coordinator(
            request=request,
            conversation_memory=None,
            trace=trace,
            request_span=request_span,
        )
        return self._build_formal_response(
            request_id=request.request_id,
            result=result,
            memory_context_status=MemoryContextStatus.NOT_REQUESTED,
            memory_persistence_status=MemoryPersistenceStatus.NOT_REQUESTED,
            memory_summarization_status=MemorySummarizationStatus.NOT_REQUESTED,
            trace=trace,
            request_span=request_span,
        )

    def _build_formal_response(
        self,
        *,
        request_id: str,
        result: CoordinatorResult,
        memory_context_status: MemoryContextStatus,
        memory_persistence_status: MemoryPersistenceStatus,
        memory_summarization_status: MemorySummarizationStatus,
        trace: TraceExecution,
        request_span: TraceContext | None,
    ) -> FormalResponse:
        """Build only the Executor's FormalResponse envelope, not HTTP serialization."""
        mapping_span = trace.start_span(
            stage=TraceStage.RESPONSE_MAPPING,
            component="application",
            operation="build_formal_response",
            parent_context=request_span,
        )
        try:
            response = FormalResponse(
                request_id=request_id,
                result=result,
                memory_context_status=memory_context_status,
                memory_persistence_status=memory_persistence_status,
                memory_summarization_status=memory_summarization_status,
            )
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            trace.complete_span(mapping_span, status=SpanStatus.CANCELLED)
            raise
        except Exception:
            trace.complete_span(
                mapping_span,
                status=SpanStatus.FAILED,
                error_code=_RESPONSE_MAPPING_FAILURE_CODE,
            )
            raise
        trace.complete_span(
            mapping_span,
            status=SpanStatus.COMPLETED,
            attributes={"success": True},
        )
        return response

    async def _persist_successful_turn(
        self,
        *,
        request: FormalRequest,
        result: CoordinatorResult,
        snapshot: SessionMemorySnapshot,
        trace: TraceExecution,
        request_span: TraceContext | None,
    ) -> tuple[MemoryPersistenceStatus, SessionMemorySnapshot | None]:
        """Append one completed, nonblank answer without retrying a failed write."""
        answer = result.answer
        if result.status is not CoordinatorStatus.COMPLETED or answer is None or not answer.strip():
            return MemoryPersistenceStatus.SKIPPED, None

        # A session ID and Store are guaranteed by the preceding read stage.
        assert request.session_id is not None
        assert self._memory_store is not None
        write_span: TraceContext | None = None
        try:
            turn_id = derive_turn_id(request.session_id, request.request_id)
            existing_turn = next(
                (stored_turn for stored_turn in snapshot.turns if stored_turn.turn_id == turn_id),
                None,
            )
            turn = SessionTurn(
                session_id=request.session_id,
                turn_id=turn_id,
                request_id=request.request_id,
                user_text=request.user_query,
                assistant_text=answer,
                created_at=self._utc_now() if existing_turn is None else existing_turn.created_at,
            )
            write_span = trace.start_span(
                stage=TraceStage.MEMORY_WRITE,
                component="memory",
                operation="persist_conversation_memory",
                parent_context=request_span,
            )
            post_append_snapshot = await asyncio.to_thread(
                self._memory_store.append_turn,
                turn,
                expected_version=snapshot.version,
            )
        except asyncio.CancelledError:
            trace.complete_span(write_span, status=SpanStatus.CANCELLED)
            raise
        except (KeyboardInterrupt, SystemExit):
            trace.complete_span(write_span, status=SpanStatus.CANCELLED)
            raise
        except SessionVersionConflictError:
            trace.complete_span(
                write_span,
                status=SpanStatus.FAILED,
                error_code="memory_write_failed",
                attributes={"success": False},
            )
            return MemoryPersistenceStatus.VERSION_CONFLICT, None
        except SessionTurnConflictError:
            trace.complete_span(
                write_span,
                status=SpanStatus.FAILED,
                error_code="memory_write_failed",
                attributes={"success": False},
            )
            return MemoryPersistenceStatus.IDEMPOTENCY_CONFLICT, None
        except Exception:
            trace.complete_span(
                write_span,
                status=SpanStatus.FAILED,
                error_code="memory_write_failed",
                attributes={"success": False},
            )
            return MemoryPersistenceStatus.STORE_FAILURE, None
        trace.complete_span(
            write_span,
            status=SpanStatus.COMPLETED,
            attributes={"written_item_count": 1, "success": True},
        )
        return MemoryPersistenceStatus.PERSISTED, post_append_snapshot

    async def _summarize_persisted_turn(
        self,
        *,
        request: FormalRequest,
        persistence_status: MemoryPersistenceStatus,
        post_append_snapshot: SessionMemorySnapshot | None,
        trace: TraceExecution,
        request_span: TraceContext | None,
    ) -> MemorySummarizationStatus:
        """Attempt one post-append summary without changing persistence outcome."""
        if persistence_status is not MemoryPersistenceStatus.PERSISTED:
            return MemorySummarizationStatus.SKIPPED
        if self._rolling_summary_service is None:
            _record_memory_not_requested(
                trace,
                stage=TraceStage.MEMORY_SUMMARY,
                operation="summarize_conversation_memory",
                parent_context=request_span,
            )
            return MemorySummarizationStatus.NOT_REQUESTED

        assert request.session_id is not None
        assert post_append_snapshot is not None
        summary_span = trace.start_span(
            stage=TraceStage.MEMORY_SUMMARY,
            component="memory",
            operation="summarize_conversation_memory",
            parent_context=request_span,
        )
        try:
            outcome = await asyncio.to_thread(
                self._rolling_summary_service.compact_snapshot_if_needed,
                request.session_id,
                post_append_snapshot,
            )
            if outcome.status is RollingSummaryStatus.NOT_REQUIRED:
                trace.complete_span(summary_span, status=SpanStatus.NOT_REQUESTED)
                return MemorySummarizationStatus.NOT_NEEDED
            trace.complete_span(
                summary_span,
                status=SpanStatus.COMPLETED,
                attributes={"compacted_turn_count": outcome.compacted_turn_count, "success": True},
            )
            return MemorySummarizationStatus.COMPACTED
        except asyncio.CancelledError:
            trace.complete_span(summary_span, status=SpanStatus.CANCELLED)
            raise
        except (KeyboardInterrupt, SystemExit):
            trace.complete_span(summary_span, status=SpanStatus.CANCELLED)
            raise
        except SessionVersionConflictError:
            trace.complete_span(
                summary_span,
                status=SpanStatus.FAILED,
                error_code="memory_summary_failed",
                attributes={"success": False},
            )
            return MemorySummarizationStatus.VERSION_CONFLICT
        except (RollingSummaryGenerationError, RollingSummaryOutputInvalid):
            trace.complete_span(
                summary_span,
                status=SpanStatus.FAILED,
                error_code="memory_summary_failed",
                attributes={"success": False},
            )
            return MemorySummarizationStatus.PROVIDER_FAILURE
        except (RollingSummaryInputTooLarge, SessionMemoryError):
            trace.complete_span(
                summary_span,
                status=SpanStatus.FAILED,
                error_code="memory_summary_failed",
                attributes={"success": False},
            )
            return MemorySummarizationStatus.STORE_FAILURE
        except Exception:
            trace.complete_span(
                summary_span,
                status=SpanStatus.FAILED,
                error_code="memory_summary_failed",
                attributes={"success": False},
            )
            return MemorySummarizationStatus.STORE_FAILURE

    def _utc_now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return value.astimezone(UTC)

    def with_trace_dispatcher(
        self,
        trace_dispatcher: BestEffortTraceDispatcher,
    ) -> FormalRequestExecutor:
        """Return an equivalent Executor with one composition-root-owned trace dispatcher."""
        if trace_dispatcher is None:
            raise ValueError("trace_dispatcher must not be None")
        return FormalRequestExecutor(
            coordinator=self._coordinator,
            memory_store=self._memory_store,
            memory_projector=self._memory_projector,
            rolling_summary_service=self._rolling_summary_service,
            clock=self._clock,
            trace_collector_factory=self._trace_collector_factory,
            trace_dispatcher=trace_dispatcher,
            authorization_policy=self._authorization_policy,
            provider_governance=self._provider_governance,
        )

    async def _execute_coordinator(
        self,
        *,
        request: FormalRequest,
        conversation_memory: object | None,
        trace: TraceExecution,
        request_span: TraceContext | None,
    ) -> CoordinatorResult:
        """Pass M8C context only when this executor is explicitly security-enabled."""
        kwargs: dict[str, object] = {
            "user_query": request.user_query,
            "request_id": request.request_id,
            "trace_recorder": trace,
            "trace_parent_context": request_span,
        }
        if conversation_memory is not None:
            kwargs["conversation_memory"] = conversation_memory
        if self._authorization_policy is not None:
            assert request.security_context is not None
            kwargs["security_context"] = request.security_context
            kwargs["authorization_policy"] = self._authorization_policy
        if self._provider_governance is None:
            if self._authorization_policy is not None:
                raise ProviderPolicyError("provider_policy_missing")
            return await self._coordinator.execute(**kwargs)  # type: ignore[arg-type]
        if request.security_context is None:
            raise ProviderPolicyError("provider_policy_missing")
        with self._provider_governance.bind_request(
            request_id=request.request_id,
            trace_id=request.security_context.trace_id,
            security_context=request.security_context,
        ):
            self._provider_governance.audit(event_type="workflow_started", outcome="started")
            result = await self._coordinator.execute(**kwargs)  # type: ignore[arg-type]
            completed = result.status is CoordinatorStatus.COMPLETED
            event_type = "workflow_completed" if completed else "workflow_failed"
            self._provider_governance.audit(
                event_type=event_type,
                outcome="completed" if completed else "failed",
                error_code=None if completed else result.error_code,
            )
            return result

    def _security_denial(self, request: FormalRequest) -> SecurityErrorCode | None:
        if self._authorization_policy is None:
            return None
        context = request.security_context
        if context is None:
            return SecurityErrorCode.UNAUTHENTICATED
        if not isinstance(context, SecurityContext):
            return SecurityErrorCode.SECURITY_CONTEXT_INVALID
        if context.request_id != request.request_id:
            return SecurityErrorCode.SECURITY_CONTEXT_INVALID
        if not context.principal.tenant_id.strip():
            return SecurityErrorCode.TENANT_CONTEXT_MISSING
        return None

    def _bind_session_scope(
        self, request: FormalRequest
    ) -> tuple[FormalRequest, SecurityErrorCode | None]:
        """Derive a Store key before any read; caller labels never key memory directly."""
        if self._authorization_policy is None or request.session_id is None:
            return request, None
        context = request.security_context
        if not isinstance(context, SecurityContext):
            return request, SecurityErrorCode.SECURITY_CONTEXT_INVALID
        try:
            self._authorization_policy.require_session_scope(context)
            scoped_key = context.scoped_session_key(request.session_id)
        except SecurityAuthorizationError as exc:
            return request, SecurityErrorCode(exc.code)
        return request.model_copy(update={"session_id": scoped_key}), None

    def _authorization_failure_response(
        self,
        request: FormalRequest,
        code: SecurityErrorCode,
        trace: TraceExecution,
        request_span: TraceContext | None,
    ) -> FormalResponse:
        """Return a terminal safe failure before memory, Router, or external dependencies."""
        return self._build_formal_response(
            request_id=request.request_id,
            result=CoordinatorResult(
                status=CoordinatorStatus.FAILED,
                coordinator_steps=("authorize_request",),
                error_code=code.value,
            ),
            memory_context_status=MemoryContextStatus.NOT_REQUESTED,
            memory_persistence_status=MemoryPersistenceStatus.NOT_REQUESTED,
            memory_summarization_status=MemorySummarizationStatus.NOT_REQUESTED,
            trace=trace,
            request_span=request_span,
        )


def _default_trace_collector(request: FormalRequest) -> TraceCollector:
    """Create one server-owned, request-local collector without retaining request content."""
    context = TraceContext.create(
        request_id=request.request_id,
        session_present=request.session_id is not None,
    )
    return TraceCollector(context=context, id_factory=new_trace_id)


def _trace_terminal_status(response: FormalResponse) -> tuple[SpanStatus, str | None]:
    if response.result.status is CoordinatorStatus.COMPLETED:
        return SpanStatus.COMPLETED, None
    if response.result.status is CoordinatorStatus.UNSUPPORTED:
        return SpanStatus.UNSUPPORTED, None
    return SpanStatus.FAILED, response.result.error_code or _REQUEST_EXECUTION_FAILURE_CODE


def _safe_execution_error_code(error: Exception) -> str:
    if isinstance(error, DecisionAgentError):
        return error.code
    return _REQUEST_EXECUTION_FAILURE_CODE


def _record_memory_not_requested(
    trace: TraceExecution,
    *,
    stage: TraceStage,
    operation: str,
    parent_context: TraceContext | None,
) -> None:
    span = trace.start_span(
        stage=stage,
        component="memory",
        operation=operation,
        parent_context=parent_context,
    )
    trace.complete_span(span, status=SpanStatus.NOT_REQUESTED)


def _attach_trace_summary(
    response: FormalResponse,
    request_trace: RequestTrace | None,
) -> FormalResponse:
    if request_trace is None:
        return response
    try:
        return response.model_copy(update={"trace": build_trace_summary(request_trace)})
    except Exception:
        return response
