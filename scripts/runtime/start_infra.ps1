[CmdletBinding()]
param(
    [ValidateRange(30, 600)]
    [int]$TimeoutSeconds = 180
)

$ErrorActionPreference = "Stop"
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $repositoryRoot

docker compose up --detach etcd minio milvus
if ($LASTEXITCODE -ne 0) {
    throw "milvus_infrastructure_start_failed"
}

$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
while ((Get-Date) -lt $deadline) {
    $rawStatus = docker compose ps --format json
    if ($LASTEXITCODE -ne 0) {
        throw "milvus_infrastructure_status_unavailable"
    }

    $records = @(
        $rawStatus -split "`r?`n" |
            Where-Object { $_.Trim() } |
            ForEach-Object { $_ | ConvertFrom-Json }
    )
    $healthyServices = @("etcd", "minio", "milvus")
    $allHealthy = $true
    foreach ($service in $healthyServices) {
        $record = $records | Where-Object { $_.Service -eq $service } | Select-Object -First 1
        if ($null -eq $record -or $record.State -ne "running" -or $record.Health -ne "healthy") {
            $allHealthy = $false
            break
        }
    }
    if ($allHealthy) {
        Write-Output "milvus_infrastructure=healthy"
        exit 0
    }
    Start-Sleep -Seconds 5
}

throw "milvus_infrastructure_healthcheck_timeout"
