"""Production ASGI entry point with deferred lifespan-owned runtime composition."""

from decision_agent.api.runtime import create_bootstrapped_app
from decision_agent.application.configured_runtime import create_configured_runtime_builder
from decision_agent.config import Settings

settings = Settings()
runtime_builder = create_configured_runtime_builder(settings)
app = create_bootstrapped_app(settings, runtime_builder)
