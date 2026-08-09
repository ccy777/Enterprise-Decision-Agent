from __future__ import annotations

from datetime import UTC, datetime

import pytest

from decision_agent.context import (
    ContextBudgetConfig,
    ContextProjectionError,
    ContextTokenBudgetExceededError,
    ConversationMemoryProjection,
    EvidenceDomain,
    RequestContextRuntime,
    TokenBudget,
)
from decision_agent.context.runtime import SkillContext
from decision_agent.routing.models import RequestRoute, RouterDecision
from decision_agent.skills.enterprise_knowledge_qa import EnterpriseKnowledgeQASkill
from decision_agent.tool_calling.models import NativeToolCallingStatus, ToolCallingResult

NOW = datetime(2026, 7, 24, tzinfo=UTC)


def _decision() -> RouterDecision:
    return RouterDecision(
        route=RequestRoute.KNOWLEDGE,
        normalized_query="query",
        decision_reason="test",
        knowledge_subquery="query",
        data_subquery=None,
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
            selected_tool="run_knowledge_agent",
            tool_call_id="call",
            answer="answer [E1]",
            citations=["[E1]"],
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


def _context() -> tuple[RequestContextRuntime, str]:
    runtime = RequestContextRuntime(request_id="knowledge", created_at=NOW)
    return runtime, runtime.user_request("ORIGINAL_KNOWLEDGE_QUERY").item_id


@pytest.mark.asyncio
async def test_selected_knowledge_query_drives_formal_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, user_id = _context()
    spy = _SpyRuntime()
    monkeypatch.setattr(
        RequestContextRuntime,
        "project",
        staticmethod(
            lambda _: type(
                "P",
                (),
                {
                    "skill": lambda self, **__: SkillContext(
                        "SELECTED_KNOWLEDGE_QUERY", "instruction"
                    )
                },
            )()
        ),
    )
    result = await EnterpriseKnowledgeQASkill(runtime=spy).execute_with_context(
        user_query="ORIGINAL_KNOWLEDGE_QUERY",
        decision=_decision(),
        context_runtime=runtime,
        user_item_id=user_id,
    )
    assert result.citations == ["[E1]"] and spy.queries == ["SELECTED_KNOWLEDGE_QUERY"]


@pytest.mark.asyncio
async def test_knowledge_projection_failure_blocks_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, user_id = _context()
    spy = _SpyRuntime()
    monkeypatch.setattr(
        RequestContextRuntime,
        "project",
        staticmethod(lambda _: (_ for _ in ()).throw(ContextProjectionError("safe"))),
    )
    with pytest.raises(ContextProjectionError):
        await EnterpriseKnowledgeQASkill(runtime=spy).execute_with_context(
            user_query="ORIGINAL_KNOWLEDGE_QUERY",
            decision=_decision(),
            context_runtime=runtime,
            user_item_id=user_id,
        )
    assert spy.queries == []


@pytest.mark.asyncio
async def test_tiny_real_budget_fails_closed_before_knowledge_agent() -> None:
    runtime = RequestContextRuntime(
        request_id="budget",
        created_at=NOW,
        budget_config=ContextBudgetConfig(knowledge=TokenBudget(max_tokens=2, reserved_tokens=1)),
    )
    user_id = runtime.user_request("required request cannot fit").item_id
    spy = _SpyRuntime()

    with pytest.raises(ContextTokenBudgetExceededError) as error:
        await EnterpriseKnowledgeQASkill(runtime=spy).execute_with_context(
            user_query="required request cannot fit",
            decision=_decision(),
            context_runtime=runtime,
            user_item_id=user_id,
        )

    assert spy.queries == []
    assert "required request cannot fit" not in str(error.value)


def test_knowledge_policy_rejects_data_evidence() -> None:
    runtime, user_id = _context()
    user = runtime.get(user_id)
    assert user is not None
    instruction = runtime.skill_instruction("knowledge", "instruction", source_item_id=user_id)
    summary = runtime.verified_summary("data-summary", "data", source_item_ids=(user_id,))
    evidence = runtime.evidence(
        "data-evidence",
        "DATA_EVIDENCE_DO_NOT_FORWARD",
        domain=EvidenceDomain.DATA,
        citation_ids=("[D1]",),
        source_item_ids=(summary.item_id,),
    )
    selection = runtime.select_for_knowledge(user_item=user, instruction_item=instruction, at=NOW)
    assert evidence.item_id not in selection.selected_item_ids and "DATA_EVIDENCE" not in str(
        runtime.diagnostics
    )


@pytest.mark.asyncio
async def test_knowledge_citation_is_not_renumbered() -> None:
    runtime, user_id = _context()
    spy = _SpyRuntime()
    result = await EnterpriseKnowledgeQASkill(runtime=spy).execute_with_context(
        user_query="ignored", decision=_decision(), context_runtime=runtime, user_item_id=user_id
    )
    assert result.citations == ["[E1]"]


@pytest.mark.asyncio
async def test_selected_conversation_memory_reaches_knowledge_runtime() -> None:
    runtime, user_id = _context()
    projection = ConversationMemoryProjection(
        content="<UNTRUSTED_CONVERSATION_MEMORY>history</UNTRUSTED_CONVERSATION_MEMORY>",
        estimated_tokens=1,
    )
    runtime.add_conversation_memory(projection)
    spy = _MemorySpyRuntime()
    await EnterpriseKnowledgeQASkill(runtime=spy).execute_with_context(
        user_query="ignored", decision=_decision(), context_runtime=runtime, user_item_id=user_id
    )
    assert spy.memories == [projection.content]
