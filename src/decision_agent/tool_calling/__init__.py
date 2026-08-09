"""Controlled native-tool-calling runtime over existing high-level Agents."""

from decision_agent.tool_calling.models import (
    AgentToolResult,
    NativeToolCallingStatus,
    ToolCallingResult,
)
from decision_agent.tool_calling.runtime import (
    NativeToolCallingError,
    OpenAICompatibleNativeToolCallingModel,
    run_native_tool_calling,
)
from decision_agent.tool_calling.tools import DataAgentTool, KnowledgeAgentTool

__all__ = [
    "AgentToolResult",
    "DataAgentTool",
    "KnowledgeAgentTool",
    "NativeToolCallingError",
    "NativeToolCallingStatus",
    "OpenAICompatibleNativeToolCallingModel",
    "ToolCallingResult",
    "run_native_tool_calling",
]
