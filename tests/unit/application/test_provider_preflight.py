import pytest

from decision_agent.application.provider_preflight import preflight_provider
from decision_agent.tool_calling.runtime import NativeToolCallingError


class _Transport:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error

    async def complete_chat(self, **_: object):
        if self.error:
            raise self.error
        return self.response


@pytest.mark.asyncio
async def test_provider_preflight_projects_safe_success() -> None:
    result = await preflight_provider(_Transport({"choices": [{"message": {"content": "{}"}}]}))
    assert result.status == "passed"
    assert result.safe_json().startswith('{"component":"provider"')


@pytest.mark.asyncio
async def test_provider_preflight_maps_transport_and_schema_failures() -> None:
    unavailable = await preflight_provider(
        _Transport(error=NativeToolCallingError("tool_calling_provider_unavailable"))
    )
    invalid = await preflight_provider(_Transport({"choices": []}))
    assert unavailable.error_code == "provider_unreachable"
    assert invalid.error_code == "provider_schema_incompatible"
