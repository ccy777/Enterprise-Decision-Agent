from __future__ import annotations

from datetime import UTC, datetime

import pytest

from decision_agent.context import (
    ContextProjectionError,
    ConversationMemoryProjection,
    EvidenceDomain,
    RequestContextRuntime,
)
from decision_agent.context.runtime import SkillContext
from decision_agent.routing.models import RequestRoute, RouterDecision
from decision_agent.skills.enterprise_data_analysis import EnterpriseDataAnalysisSkill
from decision_agent.tool_calling.models import NativeToolCallingStatus, ToolCallingResult

NOW = datetime(2026, 7, 24, tzinfo=UTC)


def _decision() -> RouterDecision:
    return RouterDecision(
        route=RequestRoute.DATA,
        normalized_query="query",
        decision_reason="test",
        knowledge_subquery=None,
        data_subquery="query",
        missing_information=None,
        confidence=1,
    )


class _SpyRuntime:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def execute(self, *, user_query: str, decision: RouterDecision) -> ToolCallingResult:
        self.queries.append(user_query)
        return ToolCallingResult(
            status=NativeToolCallingStatus.COMPLETED,
            route=decision.route,
            selected_tool="run_data_agent",
            tool_call_id="call",
            answer="answer [D1]",
            citations=["[D1]"],
            steps=2,
        )


class _MemorySpyRuntime(_SpyRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.memories: list[str] = []

    async def execute_with_memory(
        self, *, user_query: str, decision: RouterDecision, conversation_memory: str
    ) -> ToolCallingResult:
        self.queries.append(user_query)
        self.memories.append(conversation_memory)
        return await self.execute(user_query=user_query, decision=decision)


class _NoCitationRuntime(_SpyRuntime):
    async def execute(self, *, user_query: str, decision: RouterDecision) -> ToolCallingResult:
        self.queries.append(user_query)
        return ToolCallingResult(
            status=NativeToolCallingStatus.COMPLETED,
            route=decision.route,
            selected_tool="run_data_agent",
            tool_call_id="call",
            answer="clarification required",
            citations=[],
            steps=2,
        )


def _context() -> tuple[RequestContextRuntime, str]:
    runtime = RequestContextRuntime(request_id="data", created_at=NOW)
    return runtime, runtime.user_request("ORIGINAL_DATA_QUERY").item_id


@pytest.mark.asyncio
async def test_selected_data_query_drives_formal_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, user_id = _context()
    spy = _SpyRuntime()
    monkeypatch.setattr(
        RequestContextRuntime,
        "project",
        staticmethod(
            lambda _: type(
                "P",
                (),
                {"skill": lambda self, **__: SkillContext("SELECTED_DATA_QUERY", "instruction")},
            )()
        ),
    )
    result = await EnterpriseDataAnalysisSkill(runtime=spy).execute_with_context(
        user_query="ORIGINAL_DATA_QUERY",
        decision=_decision(),
        context_runtime=runtime,
        user_item_id=user_id,
    )
    assert result.citations == ["[D1]"] and spy.queries == ["SELECTED_DATA_QUERY"]


@pytest.mark.asyncio
async def test_data_projection_failure_blocks_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, user_id = _context()
    spy = _SpyRuntime()
    monkeypatch.setattr(
        RequestContextRuntime,
        "project",
        staticmethod(lambda _: (_ for _ in ()).throw(ContextProjectionError("safe"))),
    )
    with pytest.raises(ContextProjectionError):
        await EnterpriseDataAnalysisSkill(runtime=spy).execute_with_context(
            user_query="ORIGINAL_DATA_QUERY",
            decision=_decision(),
            context_runtime=runtime,
            user_item_id=user_id,
        )
    assert spy.queries == []


def test_data_policy_rejects_knowledge_evidence() -> None:
    runtime, user_id = _context()
    user = runtime.get(user_id)
    assert user is not None
    instruction = runtime.skill_instruction("data", "instruction", source_item_id=user_id)
    summary = runtime.verified_summary("knowledge-summary", "knowledge", source_item_ids=(user_id,))
    evidence = runtime.evidence(
        "knowledge-evidence",
        "KNOWLEDGE_EVIDENCE_DO_NOT_FORWARD",
        domain=EvidenceDomain.KNOWLEDGE,
        citation_ids=("[E1]",),
        source_item_ids=(summary.item_id,),
    )
    selection = runtime.select_for_data(user_item=user, instruction_item=instruction, at=NOW)
    assert evidence.item_id not in selection.selected_item_ids and "KNOWLEDGE_EVIDENCE" not in str(
        runtime.diagnostics
    )


@pytest.mark.asyncio
async def test_data_citation_is_not_renumbered() -> None:
    runtime, user_id = _context()
    spy = _SpyRuntime()
    result = await EnterpriseDataAnalysisSkill(runtime=spy).execute_with_context(
        user_query="ignored", decision=_decision(), context_runtime=runtime, user_item_id=user_id
    )
    assert result.citations == ["[D1]"]


@pytest.mark.asyncio
async def test_citationless_terminal_result_is_not_registered_as_evidence() -> None:
    runtime, user_id = _context()
    result = await EnterpriseDataAnalysisSkill(runtime=_NoCitationRuntime()).execute_with_context(
        user_query="ignored",
        decision=_decision(),
        context_runtime=runtime,
        user_item_id=user_id,
    )
    assert result.citations == []


@pytest.mark.asyncio
async def test_selected_conversation_memory_reaches_data_runtime() -> None:
    runtime, user_id = _context()
    projection = ConversationMemoryProjection(
        content="<UNTRUSTED_CONVERSATION_MEMORY>history</UNTRUSTED_CONVERSATION_MEMORY>",
        estimated_tokens=1,
    )
    runtime.add_conversation_memory(projection)
    spy = _MemorySpyRuntime()
    await EnterpriseDataAnalysisSkill(runtime=spy).execute_with_context(
        user_query="ignored", decision=_decision(), context_runtime=runtime, user_item_id=user_id
    )
    assert spy.memories == [projection.content]
