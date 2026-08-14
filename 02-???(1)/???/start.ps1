param([string]$Node = "node")

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$root = Split-Path -Parent $PSScriptRoot
New-Item -ItemType Directory -Force -Path `
    (Join-Path $root "runtime\logs\web-client"),(Join-Path $root "runtime\cache\python"),(Join-Path $root "runtime\cache\pytest"),(Join-Path $root "runtime\history"),(Join-Path $root "runtime\pids") | Out-Null
& (Join-Path $root "scripts\housekeep-runtime.ps1")

if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot ".env"))) {
    Write-Warning "Missing web-client/.env. Copy .env.example to .env and configure the retrieval API address."
}

& $Node server.js
