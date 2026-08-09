"""Coordinator Level 2 flow with only Fake Router, Skills, and native runtime."""

from __future__ import annotations

import asyncio

import pytest

from decision_agent.context import ContextProjectionError, ConversationMemoryProjection
from decision_agent.context.runtime import CoordinatorContext, RouterContext
from decision_agent.coordination.coordinator import Coordinator
from decision_agent.coordination.factory import build_default_coordinator
from decision_agent.coordination.models import CoordinatorStatus, SkillResult, SkillStatus
from decision_agent.routing.models import RequestRoute, RouterDecision
from decision_agent.skills.contracts import SkillDefinition
from decision_agent.skills.inventory_risk_diagnosis import InventoryRiskDiagnosisSkill
from decision_agent.skills.inventory_risk_synthesizer import (
    InventoryRiskSynthesisInput,
    InventoryRiskSynthesisResult,
)
from decision_agent.skills.registry import SkillRegistry
from decision_agent.tool_calling.models import NativeToolCallingStatus, ToolCallingResult


def decision(route: RequestRoute) -> RouterDecision:
    return RouterDecision(
        route=route,
        normalized_query="query",
        decision_reason="reason",
        knowledge_subquery="knowledge"
        if route in {RequestRoute.KNOWLEDGE, RequestRoute.MIXED}
        else None,
        data_subquery="data" if route in {RequestRoute.DATA, RequestRoute.MIXED} else None,
        missing_information=None,
        confidence=0.9,
    )


class FakeRouter:
    def __init__(self, result: RouterDecision) -> None:
        self.result = result
        self.calls = 0

    async def route(self, *, user_query: str) -> RouterDecision:
        self.calls += 1
        return self.result


class FakeInventoryRiskSynthesizer:
    async def synthesize(
        self, input_data: InventoryRiskSynthesisInput
    ) -> InventoryRiskSynthesisResult:
        return InventoryRiskSynthesisResult(
            risk_summary="库存风险。[D1]",
            policy_basis="补货制度。[E1]",
            recommended_actions=("安排补货。",),
            citations=("[D1]", "[E1]"),
        )


class FakeSkill:
    def __init__(
        self,
        route: RequestRoute,
        result: SkillResult | None = None,
        *,
        name: str | None = None,
    ) -> None:
        self.definition = SkillDefinition(
            name=name or f"{route}-skill",
            version="1",
            description="fake",
            supported_route=route,
            input_contract=("q",),
            allowed_tools=(
                "run_knowledge_agent" if route is RequestRoute.KNOWLEDGE else "run_data_agent",
            ),
            steps=("run",),
            output_contract=("answer",),
            failure_codes=("failed",),
        )
        self.result = result
        self.calls = 0

    async def execute(self, *, user_query: str, decision: RouterDecision) -> SkillResult:
        self.calls += 1
        return self.result or SkillResult(
            status=SkillStatus.COMPLETED,
            skill_name=self.definition.name,
            skill_version="1",
            route=decision.route,
            answer="answer[E1]" if decision.route is RequestRoute.KNOWLEDGE else "answer[D1]",
            citations=["[E1]" if decision.route is RequestRoute.KNOWLEDGE else "[D1]"],
            executed_steps=("run",),
            selected_tool=self.definition.allowed_tools[0],
        )


@pytest.mark.asyncio
async def test_coordinator_passes_projected_request_and_decision_to_context_skill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = RouterDecision(
        route=RequestRoute.KNOWLEDGE,
        normalized_query="ORIGINAL_ROUTER_SUBQUERY",
        decision_reason="reason",
        knowledge_subquery="ORIGINAL_ROUTER_SUBQUERY",
        data_subquery=None,
        missing_information=None,
        confidence=0.9,
    )
    selected = RouterDecision(
        route=RequestRoute.KNOWLEDGE,
        normalized_query="SELECTED_ROUTER_SUBQUERY",
        decision_reason="reason",
        knowledge_subquery="SELECTED_ROUTER_SUBQUERY",
        data_subquery=None,
        missing_information=None,
        confidence=0.9,
    )

    class ContextRouter(FakeRouter):
        async def route_with_context(
            self, *, user_query: str, selected_items: tuple[object, ...]
        ) -> RouterDecision:
            return original

    class SpySkill(FakeSkill):
        def __init__(self) -> None:
            super().__init__(RequestRoute.KNOWLEDGE)
            self.context_calls: list[tuple[str, RouterDecision]] = []
            self.legacy_calls = 0

        async def execute_with_context(
            self,
            *,
            user_query: str,
            decision: RouterDecision,
            context_runtime: object,
            user_item_id: str,
        ) -> SkillResult:
            self.context_calls.append((user_query, decision))
            return await self.execute(user_query=user_query, decision=decision)

        async def execute(self, *, user_query: str, decision: RouterDecision) -> SkillResult:
            self.legacy_calls += 1
            return await super().execute(user_query=user_query, decision=decision)

    import decision_agent.coordination.coordinator as coordinator_module

    class Projection:
        def user_request(self, **_: object) -> str:
            return "SELECTED_COORDINATOR_QUERY"

        def router(self, **_: object) -> RouterContext:
            return RouterContext("SELECTED_COORDINATOR_QUERY")

        def coordinator(self, **_: object) -> CoordinatorContext:
            return CoordinatorContext(user_request="SELECTED_COORDINATOR_QUERY", decision=selected)

    monkeypatch.setattr(
        coordinator_module.RequestContextRuntime, "project", staticmethod(lambda _: Projection())
    )
    skill = SpySkill()
    registry = SkillRegistry()
    registry.register(skill)
    result = await Coordinator(router=ContextRouter(original), registry=registry).execute(
        user_query="ORIGINAL_COORDINATOR_QUERY"
    )
    assert result.status is CoordinatorStatus.COMPLETED
    assert skill.context_calls == [("SELECTED_COORDINATOR_QUERY", selected)]


@pytest.mark.asyncio
async def test_context_aware_type_error_does_not_fallback_to_legacy_execute() -> None:
    class Router(FakeRouter):
        async def route_with_context(self, **_: object) -> RouterDecision:
            return self.result

    class Skill(FakeSkill):
        def __init__(self) -> None:
            super().__init__(RequestRoute.KNOWLEDGE)
            self.context_calls = 0
            self.legacy_calls = 0

        async def execute_with_context(self, **_: object) -> SkillResult:
            self.context_calls += 1
            raise TypeError("context implementation error")

        async def execute(self, *, user_query: str, decision: RouterDecision) -> SkillResult:
            self.legacy_calls += 1
            return await super().execute(user_query=user_query, decision=decision)

    skill = Skill()
    registry = SkillRegistry()
    registry.register(skill)
    result = await Coordinator(
        router=Router(decision(RequestRoute.KNOWLEDGE)), registry=registry
    ).execute(user_query="query")
    assert (
        result.error_code == "skill_execution_failed"
        and skill.context_calls == 1
        and skill.legacy_calls == 0
    )


@pytest.mark.asyncio
async def test_coordinator_creates_an_isolated_runtime_per_request() -> None:
    class Skill(FakeSkill):
        def __init__(self) -> None:
            super().__init__(RequestRoute.KNOWLEDGE)
            self.runtimes: list[object] = []

        async def execute_with_context(self, **kwargs: object) -> SkillResult:
            self.runtimes.append(kwargs["context_runtime"])
            return await self.execute(
                user_query=kwargs["user_query"],
                decision=kwargs["decision"],  # type: ignore[arg-type]
            )

    skill = Skill()
    registry = SkillRegistry()
    registry.register(skill)
    coordinator = Coordinator(
        router=FakeRouter(decision(RequestRoute.KNOWLEDGE)), registry=registry
    )
    await coordinator.execute(user_query="first request")
    await coordinator.execute(user_query="second request")

    first, second = skill.runtimes
    assert first is not second
    assert first.request_id != second.request_id  # type: ignore[attr-defined]
    assert second.get(f"{first.request_id}:user-request") is None  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_coordinator_projection_error_blocks_both_skill_entrypoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import decision_agent.coordination.coordinator as coordinator_module

    class Router(FakeRouter):
        async def route_with_context(self, **_: object) -> RouterDecision:
            return self.result

    class Skill(FakeSkill):
        def __init__(self) -> None:
            super().__init__(RequestRoute.KNOWLEDGE)
            self.context_calls = 0
            self.legacy_calls = 0

        async def execute_with_context(self, **_: object) -> SkillResult:
            self.context_calls += 1
            raise AssertionError("context skill must not run")

        async def execute(self, *, user_query: str, decision: RouterDecision) -> SkillResult:
            self.legacy_calls += 1
            raise AssertionError("legacy skill must not run")

    monkeypatch.setattr(
        coordinator_module.RequestContextRuntime,
        "project",
        staticmethod(lambda _: (_ for _ in ()).throw(ContextProjectionError("safe"))),
    )
    skill = Skill()
    registry = SkillRegistry()
    registry.register(skill)
    result = await Coordinator(
        router=Router(decision(RequestRoute.KNOWLEDGE)), registry=registry
    ).execute(user_query="SECRET_QUERY")
    assert result.error_code == "coordinator_router_failed"
    assert skill.context_calls == 0 and skill.legacy_calls == 0


def inventory_skill() -> InventoryRiskDiagnosisSkill:
    return InventoryRiskDiagnosisSkill(
        data_skill=FakeSkill(RequestRoute.DATA, name="enterprise-data-analysis"),
        knowledge_skill=FakeSkill(RequestRoute.KNOWLEDGE, name="enterprise-knowledge-qa"),
        synthesizer=FakeInventoryRiskSynthesizer(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("route", [RequestRoute.KNOWLEDGE, RequestRoute.DATA])
async def test_coordinator_executes_only_registered_matching_skill(route: RequestRoute) -> None:
    router = FakeRouter(decision(route))
    registry = SkillRegistry()
    skill = FakeSkill(route)
    registry.register(skill)
    result = await Coordinator(router=router, registry=registry).execute(user_query="query")
    assert (
        result.status is CoordinatorStatus.COMPLETED
        and result.skill_name == skill.definition.name
        and skill.calls == 1
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route", "status", "calls"),
    [
        (RequestRoute.UNSUPPORTED, CoordinatorStatus.UNSUPPORTED, 0),
    ],
)
async def test_unsupported_short_circuits(route, status, calls) -> None:
    router = FakeRouter(decision(route))
    registry = SkillRegistry()
    skill = FakeSkill(RequestRoute.KNOWLEDGE)
    registry.register(skill)
    result = await Coordinator(router=router, registry=registry).execute(user_query="query")
    assert result.status is status and skill.calls == calls


@pytest.mark.asyncio
async def test_legacy_router_marks_memory_consumed_only_after_context_skill_receives_it() -> None:
    class ContextSkill(FakeSkill):
        def __init__(self) -> None:
            super().__init__(RequestRoute.KNOWLEDGE)
            self.conversation_memory: str | None = None

        async def execute_with_context(
            self,
            *,
            user_query: str,
            decision: RouterDecision,
            context_runtime: object,
            user_item_id: str,
        ) -> SkillResult:
            instruction = context_runtime.skill_instruction(  # type: ignore[attr-defined]
                "knowledge-instruction", "safe", source_item_id=user_item_id
            )
            selection = context_runtime.select_for_knowledge(  # type: ignore[attr-defined]
                user_item=context_runtime.get(user_item_id),  # type: ignore[attr-defined]
                instruction_item=instruction,
                at=context_runtime.created_at,  # type: ignore[attr-defined]
            )
            selected = context_runtime.project(selection).skill(  # type: ignore[attr-defined]
                user_item_id=user_item_id, instruction_item_id=instruction.item_id
            )
            self.conversation_memory = selected.conversation_memory
            return await self.execute(user_query=user_query, decision=decision)

    memory = ConversationMemoryProjection(
        content="<UNTRUSTED_CONVERSATION_MEMORY>history</UNTRUSTED_CONVERSATION_MEMORY>",
        estimated_tokens=1,
    )
    skill = ContextSkill()
    registry = SkillRegistry()
    registry.register(skill)
    result = await Coordinator(
        router=FakeRouter(decision(RequestRoute.KNOWLEDGE)), registry=registry
    ).execute(user_query="query", conversation_memory=memory)
    assert skill.conversation_memory == memory.content
    assert result.memory_context_selected is True


@pytest.mark.asyncio
async def test_mixed_without_registered_skill_is_safe_failure() -> None:
    result = await Coordinator(
        router=FakeRouter(decision(RequestRoute.MIXED)), registry=SkillRegistry()
    ).execute(user_query="query")
    assert result.error_code == "skill_route_not_registered"


@pytest.mark.asyncio
async def test_unrelated_mixed_request_has_no_matching_skill() -> None:
    registry = SkillRegistry()
    registry.register(inventory_skill())

    result = await Coordinator(
        router=FakeRouter(decision(RequestRoute.MIXED)), registry=registry
    ).execute(user_query="供应商履约情况与采购制度")

    assert result.status is CoordinatorStatus.FAILED
    assert result.error_code == "no_matching_skill"


@pytest.mark.asyncio
async def test_inventory_mixed_request_selects_inventory_skill() -> None:
    registry = SkillRegistry()
    registry.register(inventory_skill())

    result = await Coordinator(
        router=FakeRouter(decision(RequestRoute.MIXED)), registry=registry
    ).execute(user_query="库存风险与补货制度")

    assert result.status is CoordinatorStatus.COMPLETED
    assert result.error_code is None
    assert result.answer is not None
    assert result.citations == ["[D1]", "[E1]"]
    assert "selected_tool" not in result.model_dump()


@pytest.mark.asyncio
async def test_missing_or_invalid_skill_result_fails_closed() -> None:
    missing = await Coordinator(
        router=FakeRouter(decision(RequestRoute.DATA)), registry=SkillRegistry()
    ).execute(user_query="query")
    assert (
        missing.status is CoordinatorStatus.FAILED
        and missing.error_code == "skill_route_not_registered"
    )


@pytest.mark.asyncio
async def test_cancelled_router_is_not_swallowed() -> None:
    class CancelledRouter:
        async def route(self, *, user_query: str) -> RouterDecision:
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await Coordinator(router=CancelledRouter(), registry=SkillRegistry()).execute(
            user_query="query"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [None, {}, "not-a-decision"])
async def test_invalid_router_output_fails_closed(invalid: object) -> None:
    class InvalidRouter:
        async def route(self, *, user_query: str) -> RouterDecision:
            return invalid  # type: ignore[return-value]

    result = await Coordinator(router=InvalidRouter(), registry=SkillRegistry()).execute(
        user_query="query"
    )
    assert (
        result.status is CoordinatorStatus.FAILED and result.error_code == "invalid_router_decision"
    )


def test_default_factory_registers_three_trusted_skills() -> None:
    coordinator = build_default_coordinator(
        router=FakeRouter(decision(RequestRoute.KNOWLEDGE)),
        tool_calling_executor=object(),  # type: ignore[arg-type]
        inventory_risk_synthesizer=FakeInventoryRiskSynthesizer(),
    )
    assert isinstance(coordinator, Coordinator)
    assert coordinator._registry.names == (  # type: ignore[attr-defined]
        "enterprise-knowledge-qa",
        "enterprise-data-analysis",
        "inventory-risk-diagnosis",
    )


def test_default_factory_requires_explicit_synthesizer() -> None:
    with pytest.raises(TypeError):
        build_default_coordinator(  # type: ignore[call-arg]
            router=FakeRouter(decision(RequestRoute.KNOWLEDGE)),
            tool_calling_executor=object(),
        )


@pytest.mark.asyncio
async def test_default_factory_executes_inventory_mixed_without_external_dependencies() -> None:
    class FakeRuntime:
        async def execute(self, *, user_query: str, decision: RouterDecision) -> ToolCallingResult:
            route = decision.route
            return ToolCallingResult(
                status=NativeToolCallingStatus.COMPLETED,
                route=route,
                selected_tool=(
                    "run_data_agent" if route is RequestRoute.DATA else "run_knowledge_agent"
                ),
                tool_call_id="fake",
                answer="可信答案" + ("[D1]" if route is RequestRoute.DATA else "[E1]"),
                citations=["[D1]" if route is RequestRoute.DATA else "[E1]"],
                steps=2,
            )

    coordinator = build_default_coordinator(
        router=FakeRouter(decision(RequestRoute.MIXED)),
        tool_calling_executor=FakeRuntime(),
        inventory_risk_synthesizer=FakeInventoryRiskSynthesizer(),
    )
    result = await coordinator.execute(user_query="库存风险与补货制度")
    assert result.status is CoordinatorStatus.COMPLETED
    assert result.skill_name == "inventory-risk-diagnosis"
