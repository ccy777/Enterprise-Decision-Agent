"""Run the fixed M9 suite through the configured formal Runtime."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from decision_agent.application import FormalRequest
from decision_agent.application.bootstrap import build_bootstrapped_runtime
from decision_agent.application.configured_runtime import create_configured_runtime_builder
from decision_agent.config import Settings
from decision_agent.evaluation.final_system_evaluation import (
    EvaluationMode,
    FinalCaseRecord,
    FinalEvaluationCase,
    FinalEvaluationDataset,
    calculate_final_metrics,
    trace_attributes,
)
from decision_agent.observability import BestEffortTraceDispatcher, InMemoryTraceSink, TraceStage
from decision_agent.security import (
    DataScope,
    KnowledgeScope,
    SecurityErrorCode,
    build_security_context,
    make_system_principal,
)

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DATASET = _ROOT / "datasets" / "agent_tasks" / "m9_final_eval_v1.json"
_DOCUMENT_MANIFEST = _ROOT / "datasets" / "enterprise_kb" / "m2c1" / "document_manifest.json"
_CHILD_CHUNKS = _ROOT / "datasets" / "enterprise_kb" / "m2c1" / "generated" / "child_chunks.jsonl"
_CITATION = re.compile(r"^\[([DEG])\d+\]$")
_DATA_RESOURCES = frozenset(
    {
        "products",
        "sales_orders",
        "sales_order_items",
        "inventory_snapshots",
        "suppliers",
        "purchase_orders",
    }
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--case-id")
    parser.add_argument("--run-kind", choices=("smoke", "formal"), required=True)
    return parser.parse_args()


def _load_dataset(path: Path) -> FinalEvaluationDataset:
    return FinalEvaluationDataset.model_validate_json(path.read_text(encoding="utf-8"))


def _head_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _document_ids() -> frozenset[str]:
    manifest = json.loads(_DOCUMENT_MANIFEST.read_text(encoding="utf-8"))
    return frozenset(item["document_id"] for item in manifest["documents"])


def _security_context(case: FinalEvaluationCase, request_id: str, trace_id: str):
    tenant = "m9-evaluation-tenant"
    principal = make_system_principal(
        subject_id="m9-final-evaluator",
        tenant_id=tenant,
        roles=frozenset({"evaluation_runner"}),
    )
    knowledge_scope = None
    if case.scenario != "security_scope_denied":
        knowledge_scope = KnowledgeScope(
            tenant_id=tenant,
            allowed_namespaces=frozenset({"enterprise_kb"}),
            allowed_document_ids=_document_ids(),
            scope_version="m9-final-eval-v1",
        )
    return build_security_context(
        principal=principal,
        request_id=request_id,
        trace_id=trace_id,
        allowed_scenarios=frozenset({"knowledge", "data", "mixed"}),
        allowed_workflows=frozenset({"direct", "controlled_mixed"}),
        allowed_skills=frozenset(
            {"enterprise-knowledge-qa", "enterprise-data-analysis", "inventory-risk-diagnosis"}
        ),
        allowed_tools=frozenset({"run_knowledge_agent", "run_data_agent"}),
        data_scope=DataScope(
            tenant_id=tenant,
            allowed_domains=frozenset({"enterprise_operations"}),
            allowed_resources=_DATA_RESOURCES,
            allowed_query_capabilities=frozenset({"read"}),
            scope_version="m9-final-eval-v1",
        ),
        knowledge_scope=knowledge_scope,
    )


async def _execute_case(
    executor: object,
    sink: InMemoryTraceSink,
    case: FinalEvaluationCase,
    *,
    run_id: str,
    dataset_version: str,
    code_commit: str,
) -> FinalCaseRecord:
    request_id = f"{run_id}-{case.case_id}"
    security_trace_id = str(uuid.uuid4())
    started = datetime.now(UTC)
    response = await executor.execute(  # type: ignore[attr-defined]
        FormalRequest(
            request_id=request_id,
            user_query=case.question,
            security_context=_security_context(case, request_id, security_trace_id),
        )
    )
    completed = datetime.now(UTC)
    trace = next(trace for trace in reversed(sink.snapshot()) if trace.request_id == request_id)
    provider_spans = [span for span in trace.spans if span.stage is TraceStage.PROVIDER_CALL]
    usage = [trace_attributes(span.attributes) for span in provider_spans]
    usage_complete = all(item.get("usage_available") is True for item in usage)
    input_tokens = sum(int(item["input_tokens"]) for item in usage) if usage_complete else None
    output_tokens = sum(int(item["output_tokens"]) for item in usage) if usage_complete else None
    citations = response.result.citations
    prefix_counts: dict[str, int] = {}
    for citation in citations:
        match = _CITATION.fullmatch(citation)
        if match:
            prefix = match.group(1)
            prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
    tool_categories: set[str] = set()
    if any(span.stage is TraceStage.RETRIEVAL for span in trace.spans):
        tool_categories.add("knowledge")
    if any(span.stage is TraceStage.DATA_ACCESS for span in trace.spans):
        tool_categories.update({"data", "mcp"})
    if prefix_counts.get("E", 0):
        tool_categories.add("knowledge")
    if prefix_counts.get("D", 0):
        tool_categories.update({"data", "mcp"})
    if trace.route == "knowledge" and len(provider_spans) > 1:
        tool_categories.add("knowledge")
    if trace.route == "data" and len(provider_spans) > 1:
        tool_categories.update({"data", "mcp"})
    evidence_types = tuple(
        evidence
        for prefix, evidence in (("D", "sql"), ("E", "document"), ("G", "graph"))
        if prefix_counts.get(prefix, 0)
    )
    answer = response.result.answer
    actual_error_codes = {code for code in (response.result.error_code, trace.error_code) if code}
    security_codes = {item.value for item in SecurityErrorCode}
    security_decision = "denied" if actual_error_codes & security_codes else "allowed"
    checks = {
        "route": trace.route == case.expected_route,
        "final_status": response.result.status.value == case.expected_final_status,
        "error_code": (not case.expected_error_codes and not actual_error_codes)
        or bool(actual_error_codes & set(case.expected_error_codes)),
        "tools": set(case.expected_tool_categories) <= tool_categories,
        "evidence": set(case.expected_evidence_types) <= set(evidence_types),
        "citations": all(
            prefix_counts.get(prefix, 0) > 0 for prefix in case.required_citation_prefixes
        )
        and (case.answerable or not citations),
        "facts": all(
            fact.casefold() in (answer or "").casefold() for fact in case.required_answer_facts
        )
        and all(
            any(alternative.casefold() in (answer or "").casefold() for alternative in group)
            for group in case.required_answer_fact_groups
        ),
        "answerability": bool(answer and answer.strip()) is case.answerable,
        "security": security_decision == case.expected_security_decision,
    }
    passed = all(checks.values())
    span_failures = tuple(
        f"{span.operation}:{span.error_code}" for span in trace.spans if span.error_code is not None
    )
    primary_error = trace.error_code or response.result.error_code
    return FinalCaseRecord(
        run_id=run_id,
        dataset_version=dataset_version,
        code_commit=code_commit,
        case_id=case.case_id,
        scenario=case.scenario,
        execution_mode=case.execution_mode,
        started_at=started,
        completed_at=completed,
        route=trace.route,
        final_status=response.result.status.value,
        error_code=primary_error,
        trace_id=trace.trace_id,
        latency_ms=trace.duration_ms,
        provider_call_count=len(provider_spans),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        usage_complete=usage_complete,
        citation_count=len(citations),
        citation_prefix_counts=prefix_counts,
        evidence_types=evidence_types,
        tool_categories=tuple(sorted(tool_categories)),
        answer_present=bool(answer and answer.strip()),
        abstained=not bool(answer and answer.strip()),
        security_decision=security_decision,
        checks=checks,
        span_failures=span_failures,
        passed=passed,
        failure_category=None if passed else (primary_error or "assertion_failure"),
    )


def _frozen_record(
    case: FinalEvaluationCase,
    *,
    run_id: str,
    dataset_version: str,
    code_commit: str,
    timestamp: datetime,
) -> FinalCaseRecord:
    reference_exists = (_ROOT / str(case.evidence_reference)).is_file()
    checks = {"frozen_evidence": reference_exists}
    return FinalCaseRecord(
        run_id=run_id,
        dataset_version=dataset_version,
        code_commit=code_commit,
        case_id=case.case_id,
        scenario=case.scenario,
        execution_mode=case.execution_mode,
        started_at=timestamp,
        completed_at=timestamp,
        route=case.expected_route,
        final_status=case.expected_final_status,
        error_code=case.expected_error_codes[0] if case.expected_error_codes else None,
        latency_ms=0,
        provider_call_count=0,
        usage_complete=True,
        citation_count=0,
        answer_present=False,
        abstained=True,
        security_decision=case.expected_security_decision,
        checks=checks,
        passed=reference_exists,
        failure_category=None if reference_exists else "frozen_evidence_missing",
        evidence_reference=case.evidence_reference,
    )


async def _run(arguments: argparse.Namespace) -> int:
    dataset_path = arguments.dataset.resolve()
    dataset = _load_dataset(dataset_path)
    if _head_commit() != arguments.code_commit:
        raise RuntimeError("evaluation_commit_mismatch")
    selected = [
        case for case in dataset.cases if not arguments.case_id or case.case_id == arguments.case_id
    ]
    if not selected:
        raise RuntimeError("evaluation_case_not_found")
    if arguments.run_kind == "formal" and arguments.case_id:
        raise RuntimeError("formal_run_must_use_fixed_suite")
    if arguments.run_kind == "smoke" and len(selected) != 1:
        raise RuntimeError("smoke_run_requires_one_case")

    run_id = f"m9-{arguments.run_kind}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_started = datetime.now(UTC)
    settings = Settings()
    real_cases = [case for case in selected if case.execution_mode is EvaluationMode.REAL_RUNTIME]
    sink = InMemoryTraceSink(max_traces=max(1, len(real_cases)))
    records: list[FinalCaseRecord] = []
    if real_cases:
        if not settings.controlled_workflow_enabled:
            raise RuntimeError("controlled_workflow_not_enabled")
        runtime = await build_bootstrapped_runtime(create_configured_runtime_builder(settings))
        executor = runtime.executor.with_trace_dispatcher(BestEffortTraceDispatcher([sink]))
        try:
            for case in real_cases:
                records.append(
                    await _execute_case(
                        executor,
                        sink,
                        case,
                        run_id=run_id,
                        dataset_version=dataset.dataset_version,
                        code_commit=arguments.code_commit,
                    )
                )
        finally:
            await runtime.aclose()
    timestamp = datetime.now(UTC)
    records.extend(
        _frozen_record(
            case,
            run_id=run_id,
            dataset_version=dataset.dataset_version,
            code_commit=arguments.code_commit,
            timestamp=timestamp,
        )
        for case in selected
        if case.execution_mode is EvaluationMode.FROZEN_BOUNDARY
    )
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "case_records.jsonl").write_text(
        "".join(record.model_dump_json() + "\n" for record in records), encoding="utf-8"
    )
    metrics = calculate_final_metrics(records)
    (output_dir / "metrics.json").write_text(
        metrics.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    dataset_bytes = dataset_path.read_bytes()
    child_count = sum(
        1 for line in _CHILD_CHUNKS.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    manifest = {
        "schema_version": "1.0",
        "run_id": run_id,
        "run_kind": arguments.run_kind,
        "dataset_version": dataset.dataset_version,
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "code_commit": arguments.code_commit,
        "started_at": run_started.isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "runtime_path": [
            "configured_runtime",
            "formal_request_executor",
            "router",
            "coordinator",
            "controlled_workflow",
            "rag_or_mcp",
            "evidence",
            "reviewer",
            "release",
            "trace_and_audit",
        ],
        "models": {
            "provider": "openai_compatible",
            "provider_model": settings.llm_model_name,
            "embedding_model": settings.embedding_model_name,
            "embedding_revision": settings.embedding_model_revision,
            "embedding_local_files_only": settings.embedding_local_files_only,
            "reranker_model": settings.reranker_model_name,
            "reranker_revision": settings.reranker_model_revision,
            "model_cache_offline": bool(os.environ.get("HF_HUB_OFFLINE") == "1"),
        },
        "corpus": {
            "collection": settings.milvus_collection,
            "database": settings.milvus_database,
            "schema_contract": "decision_agent_child_chunk_v1",
            "validated_row_count": child_count,
            "validation": "runtime_exact_record_id_match",
            "ingestion_performed": False,
        },
        "redaction": {
            "question": "omitted",
            "answer": "omitted",
            "sql_and_rows": "omitted",
            "provider_payloads_and_prompts": "omitted",
            "raw_exceptions_and_stderr": "omitted",
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "passed": metrics.failed_case_count == 0,
                "case_count": len(records),
            },
            separators=(",", ":"),
        )
    )
    return 0 if metrics.failed_case_count == 0 else 2


def main() -> int:
    try:
        return asyncio.run(_run(_args()))
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        code = (
            str(exc)
            if str(exc).startswith(("evaluation_", "controlled_", "formal_"))
            else "m9_evaluation_failed"
        )
        print(json.dumps({"error_code": code}, separators=(",", ":")), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
