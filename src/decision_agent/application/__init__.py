"""Internal formal request execution contracts.

This package deliberately does not expose an HTTP or CLI endpoint.  It provides the
application boundary that a future transport may adapt without constructing session
memory dependencies itself.
"""

from decision_agent.application.executor import FormalRequestExecutor, SessionMemoryReadError
from decision_agent.application.models import (
    FormalRequest,
    FormalResponse,
    MemoryContextStatus,
    MemoryPersistenceStatus,
    MemorySummarizationStatus,
)
from decision_agent.application.runtime import (
    FormalMemoryConfiguration,
    FormalMemoryMode,
    FormalRuntimeConfigurationError,
    build_formal_request_executor,
)

__all__ = [
    "FormalMemoryConfiguration",
    "FormalMemoryMode",
    "FormalRequest",
    "FormalRequestExecutor",
    "FormalResponse",
    "FormalRuntimeConfigurationError",
    "MemoryContextStatus",
    "MemoryPersistenceStatus",
    "MemorySummarizationStatus",
    "SessionMemoryReadError",
    "build_formal_request_executor",
]
