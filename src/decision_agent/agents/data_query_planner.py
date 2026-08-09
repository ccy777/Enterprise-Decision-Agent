"""Structured, OpenAI-compatible planning for one safe enterprise data query."""

# ruff: noqa: E501, RUF001

from __future__ import annotations

import asyncio
import json
import re
from enum import StrEnum
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from decision_agent.config import Settings
from decision_agent.exceptions import ConfigurationError, DecisionAgentError
from decision_agent.observability import (
    SpanStatus,
    TraceContext,
    TraceSpanRecorder,
    TraceStage,
    complete_recorded_span,
    start_recorded_span,
)
from decision_agent.observability.provider import (
    ProviderTraceMetadata,
    provider_failure_attributes,
    provider_response_attributes,
)
from decision_agent.providers import build_chat_completion_payload, extract_stopped_message_content

_CHINESE_PATTERN = re.compile(r"[\u4e00-\u9fff]")
_SAFE_FINISH_REASONS = frozenset(
    {"stop", "length", "content_filter", "tool_calls", "function_call"}
)


class DataPlanStatus(StrEnum):
    """Bounded outcomes for a single data-query planning decision."""

    READY = "ready"
    NEEDS_CLARIFICATION = "needs_clarification"
    UNSUPPORTED = "unsupported"


class DataQueryPlan(BaseModel):
    """Untrusted planner output; SQL remains subject to SQLGuard."""

    model_config = ConfigDict(extra="forbid")

    status: DataPlanStatus
    intent: str = Field(min_length=1, max_length=240)
    sql: str | None = Field(default=None, max_length=20_000)
    decision_reason: str = Field(min_length=1, max_length=500)
    missing_information: str | None = Field(default=None, max_length=300)

    @model_validator(mode="after")
    def validate_contract(self) -> DataQueryPlan:
        if self.status is DataPlanStatus.READY:
            if not self.sql or not self.sql.strip() or self.missing_information is not None:
                raise ValueError("ready plan requires sql and no missing_information")
        elif self.sql is not None:
            raise ValueError("non-ready plan cannot contain sql")
        if self.status is DataPlanStatus.NEEDS_CLARIFICATION and (
            not self.missing_information or not self.missing_information.strip()
        ):
            raise ValueError("needs_clarification requires missing_information")
        return self


class DataPlanValidationResult(BaseModel):
    """Safe result of code-level plan validation."""

    model_config = ConfigDict(extra="forbid")

    validation_passed: bool
    validation_errors: list[str]


class DataQueryPlanner(Protocol):
    """Create a single safe SQL plan without answering the business question."""

    async def plan(
        self,
        *,
        user_query: str,
        enterprise_schema: dict[str, list[str]],
        business_definitions: dict[str, str],
    ) -> DataQueryPlan:
        """Return a strict planning decision only."""


class DataQueryPlanningError(DecisionAgentError):
    """Raised when an external planner has no valid public structured result."""

    def __init__(
        self, subcode: str, *, details: dict[str, str | bool | int | None] | None = None
    ) -> None:
        super().__init__("Data query planning could not be completed")
        self.subcode = subcode
        self.details = details or {}


def validate_data_query_plan(*, user_query: str, plan: DataQueryPlan) -> DataPlanValidationResult:
    """Enforce the small language contract without attempting language detection frameworks."""
    errors: list[str] = []
    if _CHINESE_PATTERN.search(user_query):
        natural_fields = [plan.intent, plan.decision_reason]
        if plan.missing_information is not None:
            natural_fields.append(plan.missing_information)
        if not all(_CHINESE_PATTERN.search(value) for value in natural_fields):
            errors.append("data_planner_language_mismatch")
    return DataPlanValidationResult(validation_passed=not errors, validation_errors=errors)


class OpenAICompatibleDataQueryPlanner:
    """Minimal JSON planner using the project's existing OpenAI-compatible settings."""

    def __init__(
        self, *, api_key: str, base_url: str, model_name: str, timeout_seconds: float
    ) -> None:
        if not all(
            isinstance(value, str) and value.strip() for value in (api_key, base_url, model_name)
        ):
            raise ConfigurationError("LLM API key, base URL, and model name are required")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._timeout_seconds = timeout_seconds

    @classmethod
    def from_settings(cls, settings: Settings) -> OpenAICompatibleDataQueryPlanner:
        if (
            settings.llm_api_key is None
            or settings.llm_base_url is None
            or settings.llm_model_name is None
        ):
            raise ConfigurationError("LLM API key, base URL, and model name are required")
        return cls(
            api_key=settings.llm_api_key.get_secret_value(),
            base_url=settings.llm_base_url,
            model_name=settings.llm_model_name,
            timeout_seconds=settings.llm_timeout_seconds,
        )

    async def plan(
        self,
        *,
        user_query: str,
        enterprise_schema: dict[str, list[str]],
        business_definitions: dict[str, str],
    ) -> DataQueryPlan:
        return await self.plan_with_trace(
            user_query=user_query,
            enterprise_schema=enterprise_schema,
            business_definitions=business_definitions,
            trace_recorder=None,
            trace_parent_context=None,
        )

    async def plan_with_trace(
        self,
        *,
        user_query: str,
        enterprise_schema: dict[str, list[str]],
        business_definitions: dict[str, str],
        trace_recorder: TraceSpanRecorder | None,
        trace_parent_context: TraceContext | None,
    ) -> DataQueryPlan:
        operation = "plan_data_query"
        metadata = self.provider_trace_metadata()
        provider_span = start_recorded_span(
            trace_recorder,
            stage=TraceStage.PROVIDER_CALL,
            component="provider",
            operation=operation,
            parent_context=trace_parent_context,
            attributes=provider_failure_attributes(metadata=metadata, operation=operation),
        )
        try:
            payload = await asyncio.to_thread(
                self._post, user_query, enterprise_schema, business_definitions
            )
        except asyncio.CancelledError:
            complete_recorded_span(
                trace_recorder,
                provider_span,
                status=SpanStatus.CANCELLED,
                attributes=provider_failure_attributes(metadata=metadata, operation=operation),
            )
            raise
        except DataQueryPlanningError as exc:
            complete_recorded_span(
                trace_recorder,
                provider_span,
                status=SpanStatus.FAILED,
                error_code=exc.subcode,
                attributes=provider_failure_attributes(metadata=metadata, operation=operation),
            )
            raise
        except HTTPError as exc:
            complete_recorded_span(
                trace_recorder,
                provider_span,
                status=SpanStatus.FAILED,
                error_code="data_query_provider_http_error",
                attributes=provider_failure_attributes(metadata=metadata, operation=operation),
            )
            raise DataQueryPlanningError(
                "data_query_provider_http_error", details={"http_status": exc.code}
            ) from exc
        except (URLError, OSError, TimeoutError) as exc:
            complete_recorded_span(
                trace_recorder,
                provider_span,
                status=SpanStatus.FAILED,
                error_code="data_query_provider_unavailable",
                attributes=provider_failure_attributes(metadata=metadata, operation=operation),
            )
            raise DataQueryPlanningError("data_query_provider_unavailable") from exc
        except ValueError as exc:
            complete_recorded_span(
                trace_recorder,
                provider_span,
                status=SpanStatus.FAILED,
                error_code="data_query_provider_invalid_response",
                attributes=provider_failure_attributes(metadata=metadata, operation=operation),
            )
            raise DataQueryPlanningError("data_query_provider_invalid_response") from exc
        complete_recorded_span(
            trace_recorder,
            provider_span,
            status=SpanStatus.COMPLETED,
            attributes=provider_response_attributes(
                metadata=metadata,
                operation=operation,
                payload=payload,
            ),
        )
        try:
            metadata = _response_metadata(payload)
            if metadata["response_empty"]:
                raise DataQueryPlanningError("data_plan_response_empty", details=metadata)
            if metadata["token_limit_reached"]:
                raise DataQueryPlanningError("data_plan_output_truncated", details=metadata)
            if metadata["finish_reason"] != "stop":
                raise DataQueryPlanningError(
                    "data_query_provider_invalid_response", details=metadata
                )
            content = _extract_message_content(payload)
            try:
                decoded = json.loads(content)
            except json.JSONDecodeError as exc:
                raise DataQueryPlanningError(
                    "data_plan_json_parse_failed", details=metadata
                ) from exc
            try:
                return DataQueryPlan.model_validate(decoded)
            except ValidationError as exc:
                raise DataQueryPlanningError(
                    "data_plan_schema_validation_failed", details=metadata
                ) from exc
        except DataQueryPlanningError:
            raise
        except ValueError as exc:
            raise DataQueryPlanningError("data_query_provider_invalid_response") from exc

    def provider_trace_metadata(self) -> ProviderTraceMetadata:
        """Return static adapter metadata; this HTTP adapter has no retry loop."""
        return ProviderTraceMetadata(
            provider="openai_compatible",
            model=self._model_name,
            retry_count=0,
        )

    def _post(
        self,
        user_query: str,
        enterprise_schema: dict[str, list[str]],
        business_definitions: dict[str, str],
    ) -> dict[str, Any]:
        body = json.dumps(
            build_chat_completion_payload(
                base_url=self._base_url,
                payload={
                    "model": self._model_name,
                    "temperature": 0,
                    "max_tokens": 1500,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {
                            "role": "system",
                            "content": _build_system_prompt(
                                enterprise_schema, business_definitions
                            ),
                        },
                        {"role": "user", "content": f"User question:\n{user_query}"},
                    ],
                },
            ),
            ensure_ascii=False,
        ).encode("utf-8")
        request = Request(
            f"{self._base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlopen(request, timeout=self._timeout_seconds) as response:
            try:
                decoded = json.loads(response.read().decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise DataQueryPlanningError("data_query_provider_invalid_response") from exc
        if not isinstance(decoded, dict):
            raise DataQueryPlanningError("data_query_provider_invalid_response")
        return decoded


def _extract_message_content(payload: dict[str, Any]) -> str:
    return extract_stopped_message_content(payload)


def _response_metadata(payload: dict[str, Any]) -> dict[str, str | bool | None]:
    """Extract bounded completion metadata without retaining provider response content."""
    try:
        choice = payload["choices"][0]
        finish_reason = choice.get("finish_reason")
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise ValueError("LLM response lacks completion metadata") from exc
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise ValueError("LLM finish_reason must be a string or null")
    safe_finish_reason = (
        finish_reason
        if finish_reason in _SAFE_FINISH_REASONS
        else "unknown"
        if finish_reason is not None
        else None
    )
    if safe_finish_reason != "stop":
        return {
            "finish_reason": safe_finish_reason,
            "token_limit_reached": safe_finish_reason == "length",
            "response_empty": False,
        }
    try:
        message = choice["message"]
        content = message.get("content")
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("LLM response lacks completion message") from exc
    return {
        "finish_reason": safe_finish_reason,
        "token_limit_reached": safe_finish_reason == "length",
        "response_empty": not isinstance(content, str) or not content.strip(),
    }


_SYSTEM_PROMPT = """You are an enterprise data-query planner. Output exactly one JSON object and no Markdown.
You plan one query only; never answer the business question or provide a business result. Return exactly
status, intent, sql, decision_reason, missing_information. No hidden reasoning, answer field, citation,
tool call, retry, or extra field.

status is ready, needs_clarification, or unsupported.
ready requires one SQL string and missing_information null. needs_clarification requires sql null and a
specific missing_information. unsupported requires sql null. Use needs_clarification only when no single
authorized SELECT can retrieve the requested raw data because a required scope is absent. A current/latest
snapshot and a request across all authorized rows already define their scope. A qualitative metric that a
later analysis stage will interpret is not a reason to ask for clarification. Use unsupported for
non-operational-database facts or dangerous requests. Use the same primary language as the user question
for intent, decision_reason, and missing_information.

A request for the current database snapshot across all authorized rows already has a defined period and
object scope. Do not ask for a reporting period or one specific object in that case. This includes current
inventory-risk screening across the authorized inventory catalog; use the supplied business definitions to
plan the one read-only snapshot query.

When a broader analysis or policy step will interpret the data later, query the authorized raw fields needed
for that interpretation. Do not require every qualitative risk label or recommendation to be defined as a
database metric, and do not answer or infer that later-stage interpretation yourself.

For a current or latest inventory request, the supplied current_inventory definition resolves the period.
If the schema supplies inventory quantities and product safety-stock fields, status must be ready: query the
current per-product raw values needed by later analysis. Do not request a separate date range, product, or
risk-metric definition.
对于“当前库存”或“最新库存”问题，current_inventory 已经定义统计时点；只要授权 Schema
包含库存数量和产品安全库存字段，status 必须为 ready。查询后续分析所需的当前逐产品原始字段，
不得再要求日期区间、指定产品或额外的风险指标定义。

JSON shape example only (not a business answer or a required SQL pattern):
{"status":"ready","intent":"query a defined metric","sql":"SELECT allowed_column FROM allowed_table",
"decision_reason":"the requested inputs are defined","missing_information":null}

Only create one SELECT or WITH ... SELECT. UNION/UNION ALL is allowed only when each branch follows
these rules. Never use *, writes, DDL, SET, locking reads, file output, dangerous functions, system schemas,
or unlisted tables/columns. SQL is untrusted and will be checked again by SQLGuard.

Do not infer causes, forecasts, policies, or facts absent from the supplied tables."""


def _build_system_prompt(
    enterprise_schema: dict[str, list[str]], business_definitions: dict[str, str]
) -> str:
    """Attach only MCP-supplied authorization and business context to the stable planner rules."""
    return (
        f"{_SYSTEM_PROMPT}\n\nAuthorized schema from Enterprise Data MCP:\n"
        f"{json.dumps(enterprise_schema, ensure_ascii=False, sort_keys=True)}\n\n"
        "Business definitions from Enterprise Data MCP:\n"
        f"{json.dumps(business_definitions, ensure_ascii=False, sort_keys=True)}"
    )
