[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $repositoryRoot

docker compose stop etcd minio milvus
if ($LASTEXITCODE -ne 0) {
    throw "milvus_infrastructure_stop_failed"
}

Write-Output "milvus_infrastructure=stopped_volumes_preserved"
