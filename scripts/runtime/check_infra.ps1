[CmdletBinding()]
param(
    [switch]$RequireEmpty
)

$ErrorActionPreference = "Stop"
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$python = Join-Path $repositoryRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "project_virtual_environment_missing"
}

Set-Location $repositoryRoot
$healthCheck = @'
import asyncio
from urllib.parse import urlsplit

from decision_agent.config import Settings
from decision_agent.retrieval import MilvusVectorStore
from pymilvus import MilvusClient


async def main() -> None:
    settings = Settings()
    target = urlsplit(settings.milvus_uri).hostname
    if target not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError("milvus_settings_target_not_local")

    store = MilvusVectorStore.from_settings(settings)
    try:
        await store.initialize()
        record_count = await store.count()
    finally:
        await store.close()

    client = MilvusClient(
        uri=settings.milvus_uri,
        token=(settings.milvus_token.get_secret_value() if settings.milvus_token else None),
        db_name=settings.milvus_database,
        timeout=settings.milvus_timeout_seconds,
    )
    try:
        server_version = client.get_server_version()
        collection_exists = client.has_collection(collection_name=settings.milvus_collection)
    finally:
        client.close()

    if not collection_exists:
        raise RuntimeError("milvus_collection_not_found")
    if __REQUIRE_EMPTY__ and record_count != 0:
        raise RuntimeError("milvus_collection_not_empty")

    print("settings_parse=passed")
    print("milvus_connection=passed")
    print(f"milvus_server_version={server_version}")
    print("collection_exists=true")
    print("collection_schema_and_hnsw_validation=passed")
    print(f"milvus_collection_logical_count={record_count}")
    print(f"collection_empty={str(record_count == 0).lower()}")
    print("milvus_store_close=passed")


asyncio.run(main())
'@
$healthCheck = $healthCheck.Replace(
    "__REQUIRE_EMPTY__",
    $RequireEmpty.ToString()
)

$encodedHealthCheck = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($healthCheck))
$pythonCommand = "import base64; exec(compile(base64.b64decode('$encodedHealthCheck'), '<m8d-check>', 'exec'))"
& $python -I -c $pythonCommand
if ($LASTEXITCODE -ne 0) {
    throw "milvus_store_initialization_failed"
}
