"""Server-owned, least-privilege local demo over the formal configured runtime."""

# ruff: noqa: RUF001

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from tempfile import gettempdir
from types import MappingProxyType
from uuid import uuid4

from decision_agent.application import FormalRequest, FormalRequestExecutor, FormalResponse
from decision_agent.application.bootstrap import RuntimeBuilder
from decision_agent.application.configured_runtime import create_configured_runtime_builder
from decision_agent.config import Settings
from decision_agent.security import (
    DataScope,
    KnowledgeScope,
    SecurityContext,
    SessionScope,
    build_security_context,
    make_system_principal,
)

_DEMO_TENANT = "local-demo-tenant"
_DEMO_SUBJECT = "local-demo-principal"
_DEMO_ROLE = "local_demo_reader"
_KNOWLEDGE_NAMESPACE = "enterprise_kb"


class DemoCase(StrEnum):
    """Closed set of server-owned demo requests; arbitrary prompts are not accepted."""

    KNOWLEDGE = "knowledge"
    DATA = "data"
    MIXED = "mixed"


@dataclass(frozen=True, slots=True)
class DemoSpecification:
    question: str
    scenario: str
    workflow: str
    skill: str
    tools: frozenset[str]
    document_ids: frozenset[str]
    data_resources: frozenset[str]


DEMO_SPECIFICATIONS: Mapping[DemoCase, DemoSpecification] = MappingProxyType(
    {
        DemoCase.KNOWLEDGE: DemoSpecification(
            question="华衡智能科技有限公司的业务定位和本项目演示范围是什么？",
            scenario="knowledge",
            workflow="direct",
            skill="enterprise-knowledge-qa",
            tools=frozenset({"run_knowledge_agent"}),
            document_ids=frozenset({"DOC-ORG-001", "DOC-AGENT-001"}),
            data_resources=frozenset(),
        ),
        DemoCase.DATA: DemoSpecification(
            question="截至2026年6月30日，哪些产品低于安全库存？",
            scenario="data",
            workflow="direct",
            skill="enterprise-data-analysis",
            tools=frozenset({"run_data_agent"}),
            document_ids=frozenset(),
            data_resources=frozenset({"products", "inventory_snapshots"}),
        ),
        DemoCase.MIXED: DemoSpecification(
            question=(
                "请仅依据当前库存数据与《DOC-INV-001 库存安全线与补货管理办法》，"
                "分析截至2026年6月30日的库存风险；分别给出风险概览、制度依据和行动建议，"
                "并同时标注数据与知识引用。"
            ),
            scenario="mixed",
            workflow="controlled_mixed",
            skill="inventory-risk-diagnosis",
            tools=frozenset({"run_data_agent", "run_knowledge_agent"}),
            document_ids=frozenset({"DOC-INV-001"}),
            data_resources=frozenset(
                {"products", "inventory_snapshots", "purchase_orders", "suppliers"}
            ),
        ),
    }
)


def prepare_demo_settings(
    settings: Settings,
    *,
    repository_root: Path,
    audit_root: Path | None = None,
) -> Settings:
    """Force the demo audit sink outside the repository and enable the fixed workflow."""
    repository = repository_root.resolve()
    external_root = (audit_root or Path(gettempdir()) / "enterprise-decision-agent-audit").resolve()
    if external_root == repository or repository in external_root.parents:
        raise ValueError("demo audit root must be outside the repository")
    external_root.mkdir(parents=True, exist_ok=True)
    audit_path = external_root / f"local-demo-{uuid4().hex}.jsonl"
    return settings.model_copy(
        update={"audit_log_path": audit_path, "controlled_workflow_enabled": True}
    )


def build_demo_security_context(
    cases: Iterable[DemoCase],
    *,
    request_id: str,
    trace_id: str,
    include_session_scope: bool = False,
) -> SecurityContext:
    """Build immutable grants for a closed set of server-owned Demo cases."""
    selected_cases = tuple(dict.fromkeys(cases))
    if not selected_cases:
        raise ValueError("at least one demo case is required")
    specifications = tuple(DEMO_SPECIFICATIONS[case] for case in selected_cases)
    principal = make_system_principal(
        subject_id=_DEMO_SUBJECT,
        tenant_id=_DEMO_TENANT,
        roles=frozenset({_DEMO_ROLE}),
    )
    data_resources = frozenset(
        resource for specification in specifications for resource in specification.data_resources
    )
    document_ids = frozenset(
        document_id
        for specification in specifications
        for document_id in specification.document_ids
    )
    data_scope = (
        DataScope(
            tenant_id=_DEMO_TENANT,
            allowed_domains=frozenset({"enterprise_operations"}),
            allowed_resources=data_resources,
            allowed_query_capabilities=frozenset({"read"}),
        )
        if data_resources
        else None
    )
    knowledge_scope = (
        KnowledgeScope(
            tenant_id=_DEMO_TENANT,
            allowed_namespaces=frozenset({_KNOWLEDGE_NAMESPACE}),
            allowed_document_ids=document_ids,
        )
        if document_ids
        else None
    )
    return build_security_context(
        principal=principal,
        request_id=request_id,
        trace_id=trace_id,
        allowed_scenarios=frozenset(specification.scenario for specification in specifications),
        allowed_workflows=frozenset(specification.workflow for specification in specifications),
        allowed_skills=frozenset(specification.skill for specification in specifications),
        allowed_tools=frozenset(
            tool for specification in specifications for tool in specification.tools
        ),
        data_scope=data_scope,
        knowledge_scope=knowledge_scope,
        session_scope=(
            SessionScope(tenant_id=_DEMO_TENANT, subject_id=_DEMO_SUBJECT)
            if include_session_scope
            else None
        ),
    )


def build_demo_request(case: DemoCase, *, request_id: str, trace_id: str) -> FormalRequest:
    """Build a fixed request and immutable grants; caller input cannot enlarge either scope."""
    specification = DEMO_SPECIFICATIONS[case]
    context = build_demo_security_context(
        (case,),
        request_id=request_id,
        trace_id=trace_id,
    )
    return FormalRequest(
        request_id=request_id,
        user_query=specification.question,
        security_context=context,
    )


async def run_demo(
    case: DemoCase,
    *,
    settings: Settings,
    runtime_builder_factory: Callable[[Settings], RuntimeBuilder] = (
        create_configured_runtime_builder
    ),
) -> FormalResponse:
    """Execute exactly one fixed case through the configured formal runtime."""
    request_id = f"local-demo-{uuid4().hex}"
    request = build_demo_request(case, request_id=request_id, trace_id=f"trace-{uuid4().hex}")
    async with AsyncExitStack() as stack:
        executor = await runtime_builder_factory(settings)(stack)
        if not isinstance(executor, FormalRequestExecutor):
            raise TypeError("configured runtime did not return FormalRequestExecutor")
        return await executor.execute(request)
