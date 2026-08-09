"""Explicit, single-request ContextManager integration runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

from decision_agent.context.conversation_memory import ConversationMemoryProjection
from decision_agent.context.manager import ContextManager
from decision_agent.context.models import (
    ContextItem,
    ContextKind,
    ContextProvenance,
    ContextSelectionResult,
    ContextSource,
    EvidenceDomain,
    TrustLevel,
)
from decision_agent.context.policies import (
    DEFAULT_CONTEXT_BUDGET_CONFIG,
    ContextBudgetConfig,
    coordinator_policy,
    data_policy,
    knowledge_policy,
    mixed_synthesis_policy,
    router_policy,
)
from decision_agent.context.token_budget import ConservativeCharacterTokenEstimator, TokenEstimator
from decision_agent.routing.models import RouterDecision


@dataclass(frozen=True)
class ContextDiagnostic:
    """Safe internal selection metadata that never retains context content."""

    node_name: str
    selected_item_ids: tuple[str, ...]
    dropped_item_ids: tuple[tuple[str, str], ...]
    total_estimated_tokens: int
    available_tokens: int


class ContextProjectionError(ValueError):
    """A selected context set cannot safely provide a required typed input."""


@dataclass(frozen=True)
class CoordinatorContext:
    user_request: str
    decision: RouterDecision


@dataclass(frozen=True)
class RouterContext:
    user_request: str
    conversation_memory: str | None = None


@dataclass(frozen=True)
class SkillContext:
    user_request: str
    instruction: str
    conversation_memory: str | None = None


@dataclass(frozen=True)
class MixedSynthesisContext:
    original_request: str
    data_subquery: str
    data_answer: str
    data_citations: tuple[str, ...]
    knowledge_subquery: str
    knowledge_answer: str
    knowledge_citations: tuple[str, ...]


class SelectedContextProjection:
    """Typed authority boundary over a single policy selection result only."""

    def __init__(self, selection: ContextSelectionResult) -> None:
        self._items = selection.selected_items

    def coordinator(self, *, user_item_id: str, decision_item_id: str) -> CoordinatorContext:
        user = self._require(user_item_id, ContextKind.USER_REQUEST)
        decision = self._require(decision_item_id, ContextKind.ROUTER_DECISION)
        try:
            parsed = RouterDecision.model_validate_json(decision.content)
        except Exception as exc:
            raise ContextProjectionError("selected router decision is invalid") from exc
        return CoordinatorContext(user_request=user.content, decision=parsed)

    def user_request(self, *, user_item_id: str) -> str:
        return self._require(user_item_id, ContextKind.USER_REQUEST).content

    def router(self, *, user_item_id: str) -> RouterContext:
        user = self._require(user_item_id, ContextKind.USER_REQUEST)
        return RouterContext(
            user_request=user.content,
            conversation_memory=self._optional_memory(),
        )

    def skill(self, *, user_item_id: str, instruction_item_id: str) -> SkillContext:
        user = self._require(user_item_id, ContextKind.USER_REQUEST)
        instruction = self._require(instruction_item_id, ContextKind.SKILL_INSTRUCTION)
        return SkillContext(
            user_request=user.content,
            instruction=instruction.content,
            conversation_memory=self._optional_memory(),
        )

    def mixed(
        self, *, user_item_id: str, data_summary_id: str, knowledge_summary_id: str
    ) -> MixedSynthesisContext:
        user = self._require(user_item_id, ContextKind.USER_REQUEST)
        data = self._summary(data_summary_id, EvidenceDomain.DATA)
        knowledge = self._summary(knowledge_summary_id, EvidenceDomain.KNOWLEDGE)
        return MixedSynthesisContext(
            original_request=user.content,
            data_subquery=data["subquery"],
            data_answer=data["answer"],
            data_citations=data["citations"],
            knowledge_subquery=knowledge["subquery"],
            knowledge_answer=knowledge["answer"],
            knowledge_citations=knowledge["citations"],
        )

    def _require(self, item_id: str, kind: ContextKind) -> ContextItem:
        matches = [item for item in self._items if item.item_id == item_id]
        if len(matches) != 1 or matches[0].kind is not kind:
            raise ContextProjectionError("selected required context item is invalid")
        return matches[0]

    def _optional_memory(self) -> str | None:
        matches = [item for item in self._items if item.kind is ContextKind.CONVERSATION_MEMORY]
        if len(matches) > 1:
            raise ContextProjectionError("selected conversation memory is invalid")
        return None if not matches else matches[0].content

    def _summary(self, item_id: str, domain: EvidenceDomain) -> dict[str, object]:
        item = self._require(item_id, ContextKind.STRUCTURED_SUMMARY)
        try:
            value = json.loads(item.content)
        except json.JSONDecodeError as exc:
            raise ContextProjectionError("selected summary is invalid") from exc
        if not isinstance(value, dict):
            raise ContextProjectionError("selected summary is invalid")
        answer, subquery, citations = (
            value.get("answer"),
            value.get("subquery"),
            value.get("citations"),
        )
        pattern = "[D" if domain is EvidenceDomain.DATA else "[E"
        if (
            not isinstance(answer, str)
            or not answer.strip()
            or not isinstance(subquery, str)
            or not subquery.strip()
            or not isinstance(citations, list)
            or not citations
            or any(
                not isinstance(value, str)
                or not value.startswith(pattern)
                or not value.endswith("]")
                for value in citations
            )
        ):
            raise ContextProjectionError("selected summary is invalid")
        return {"answer": answer, "subquery": subquery, "citations": tuple(citations)}


class RequestContextRuntime:
    """Own Context items and policies for one request; never shared across requests."""

    def __init__(
        self,
        *,
        request_id: str,
        created_at: datetime,
        estimator: TokenEstimator | None = None,
        budget_config: ContextBudgetConfig = DEFAULT_CONTEXT_BUDGET_CONFIG,
    ) -> None:
        if not request_id.strip():
            raise ValueError("request_id must be non-empty")
        if created_at.tzinfo is None or created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        self.request_id = request_id
        self.created_at = created_at
        self._estimator = estimator or ConservativeCharacterTokenEstimator()
        self._budget_config = budget_config
        self._manager = ContextManager()
        self._diagnostics: list[ContextDiagnostic] = []

    @property
    def diagnostics(self) -> tuple[ContextDiagnostic, ...]:
        return tuple(self._diagnostics)

    def add(self, item: ContextItem) -> ContextItem:
        self._manager.add(item)
        return item

    def add_many(self, *items: ContextItem) -> None:
        for item in items:
            self.add(item)

    def get(self, item_id: str) -> ContextItem | None:
        return self._manager.get(item_id)

    def user_request(self, content: str) -> ContextItem:
        return self._add_item(
            label="user-request",
            kind=ContextKind.USER_REQUEST,
            content=content,
            source=ContextSource.USER,
            trust_level=TrustLevel.UNTRUSTED_USER,
        )

    def add_conversation_memory(self, projection: ConversationMemoryProjection) -> ContextItem:
        """Add one externally sourced, untrusted memory candidate without Store access."""
        if not isinstance(projection, ConversationMemoryProjection):
            raise ValueError("conversation memory projection is invalid")
        return self._add_item(
            label="conversation-memory",
            kind=ContextKind.CONVERSATION_MEMORY,
            content=projection.content,
            source=ContextSource.EXTERNAL,
            trust_level=TrustLevel.UNTRUSTED_EXTERNAL,
            source_item_ids=("session-memory-projection",),
        )

    def system_instruction(self, label: str, content: str) -> ContextItem:
        return self._add_item(
            label=label,
            kind=ContextKind.SYSTEM_INSTRUCTION,
            content=content,
            source=ContextSource.SYSTEM,
            trust_level=TrustLevel.TRUSTED_SYSTEM,
        )

    def router_decision(self, decision: RouterDecision, *, source_item_id: str) -> ContextItem:
        return self._add_item(
            label="router-decision",
            kind=ContextKind.ROUTER_DECISION,
            content=decision.model_dump_json(),
            source=ContextSource.ROUTER,
            trust_level=TrustLevel.VERIFIED_INTERNAL,
            source_item_ids=(source_item_id,),
        )

    def skill_instruction(self, label: str, content: str, *, source_item_id: str) -> ContextItem:
        return self._add_item(
            label=label,
            kind=ContextKind.SKILL_INSTRUCTION,
            content=content,
            source=ContextSource.SKILL,
            trust_level=TrustLevel.VERIFIED_INTERNAL,
            source_item_ids=(source_item_id,),
        )

    def verified_summary(
        self, label: str, content: str, *, source_item_ids: tuple[str, ...]
    ) -> ContextItem:
        return self._add_item(
            label=label,
            kind=ContextKind.STRUCTURED_SUMMARY,
            content=content,
            source=ContextSource.AGENT,
            trust_level=TrustLevel.VERIFIED_INTERNAL,
            source_item_ids=source_item_ids,
        )

    def verified_answer_summary(
        self,
        label: str,
        *,
        answer: str,
        subquery: str,
        citations: tuple[str, ...],
        source_item_ids: tuple[str, ...],
    ) -> ContextItem:
        return self.verified_summary(
            label,
            json.dumps(
                {"answer": answer, "subquery": subquery, "citations": list(citations)},
                ensure_ascii=False,
            ),
            source_item_ids=source_item_ids,
        )

    @staticmethod
    def project(selection: ContextSelectionResult) -> SelectedContextProjection:
        return SelectedContextProjection(selection)

    def evidence(
        self,
        label: str,
        content: str,
        *,
        domain: EvidenceDomain,
        citation_ids: tuple[str, ...],
        source_item_ids: tuple[str, ...],
    ) -> ContextItem:
        return self._add_item(
            label=label,
            kind=(
                ContextKind.KNOWLEDGE_EVIDENCE
                if domain is EvidenceDomain.KNOWLEDGE
                else ContextKind.DATA_EVIDENCE
            ),
            content=content,
            source=ContextSource.AGENT,
            trust_level=TrustLevel.VERIFIED_INTERNAL,
            source_item_ids=source_item_ids,
            evidence_domain=domain,
            citation_ids=citation_ids,
        )

    def select_for_router(
        self, *, user_item: ContextItem, instruction_item: ContextItem, at: datetime
    ) -> ContextSelectionResult:
        return self._select(
            router_policy(
                user_item_id=user_item.item_id,
                instruction_item_id=instruction_item.item_id,
                budget_config=self._budget_config,
            ),
            at,
        )

    def select_for_coordinator(
        self, *, user_item: ContextItem, decision_item: ContextItem, at: datetime
    ) -> ContextSelectionResult:
        return self._select(
            coordinator_policy(
                user_item_id=user_item.item_id,
                decision_item_id=decision_item.item_id,
                budget_config=self._budget_config,
            ),
            at,
        )

    def select_for_knowledge(
        self, *, user_item: ContextItem, instruction_item: ContextItem, at: datetime
    ) -> ContextSelectionResult:
        return self._select(
            knowledge_policy(
                user_item_id=user_item.item_id,
                instruction_item_id=instruction_item.item_id,
                budget_config=self._budget_config,
            ),
            at,
        )

    def select_for_data(
        self, *, user_item: ContextItem, instruction_item: ContextItem, at: datetime
    ) -> ContextSelectionResult:
        return self._select(
            data_policy(
                user_item_id=user_item.item_id,
                instruction_item_id=instruction_item.item_id,
                budget_config=self._budget_config,
            ),
            at,
        )

    def select_for_mixed_synthesis(
        self,
        *,
        user_item: ContextItem,
        data_summary: ContextItem,
        knowledge_summary: ContextItem,
        at: datetime,
    ) -> ContextSelectionResult:
        return self._select(
            mixed_synthesis_policy(
                user_item_id=user_item.item_id,
                data_summary_item_id=data_summary.item_id,
                knowledge_summary_item_id=knowledge_summary.item_id,
                budget_config=self._budget_config,
            ),
            at,
        )

    def _add_item(
        self,
        *,
        label: str,
        kind: ContextKind,
        content: str,
        source: ContextSource,
        trust_level: TrustLevel,
        source_item_ids: tuple[str, ...] = (),
        evidence_domain: EvidenceDomain | None = None,
        citation_ids: tuple[str, ...] = (),
    ) -> ContextItem:
        item = ContextItem(
            item_id=f"{self.request_id}:{label}",
            kind=kind,
            content=content,
            source=source,
            trust_level=trust_level,
            provenance=ContextProvenance(
                producer="context-runtime",
                request_id=self.request_id,
                generated_at=self.created_at,
                source_item_ids=source_item_ids,
            ),
            created_at=self.created_at,
            estimated_tokens=self._estimator.estimate(content),
            evidence_domain=evidence_domain,
            citation_ids=citation_ids,
        )
        return self.add(item)

    def _select(self, policy, at: datetime) -> ContextSelectionResult:  # type: ignore[no-untyped-def]
        result = self._manager.select(policy, at=at)
        self._diagnostics.append(
            ContextDiagnostic(
                node_name=policy.node_name,
                selected_item_ids=result.selected_item_ids,
                dropped_item_ids=tuple(
                    (item.item_id, item.reason.value) for item in result.dropped_items
                ),
                total_estimated_tokens=result.total_estimated_tokens,
                available_tokens=result.available_tokens,
            )
        )
        return result
