"""Controlled public error codes for Enterprise Data MCP client failures."""

from __future__ import annotations

from decision_agent.exceptions import DecisionAgentError


class EnterpriseDataMCPError(DecisionAgentError):
    """A safe MCP-client failure that never retains transport or credential details."""

    def __init__(self, code: str) -> None:
        super().__init__("Enterprise Data MCP request could not be completed")
        self.code = code
