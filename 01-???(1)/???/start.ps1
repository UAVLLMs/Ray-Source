param(
    [string]$ListenHost = "127.0.0.1",
    [int]$Port = 8014,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$root = Split-Path -Parent $PSScriptRoot
$runtime = Join-Path $root "runtime"
$runtimeCache = Join-Path $runtime "cache"
$runtimeHistory = Join-Path $runtime "history"
New-Item -ItemType Directory -Force -Path `
    (Join-Path $runtime "logs\retrieval-service"),(Join-Path $runtimeCache "python"),(Join-Path $runtimeCache "pytest"),$runtimeHistory,(Join-Path $runtime "pids") | Out-Null
& (Join-Path $root "scripts\housekeep-runtime.ps1")

$previousPycache = $env:PYTHONPYCACHEPREFIX
$previousTracePath = $env:CHAT_API_TRACE_PATH
$env:PYTHONPYCACHEPREFIX = Join-Path $runtimeCache "python"
$env:CHAT_API_TRACE_PATH = Join-Path $runtimeHistory "chat-api.trace.jsonl"
try {
    & $Python -m uvicorn api_server:app --host $ListenHost --port $Port --workers 1
} finally {
    $env:PYTHONPYCACHEPREFIX = $previousPycache
    $env:CHAT_API_TRACE_PATH = $previousTracePath
}
