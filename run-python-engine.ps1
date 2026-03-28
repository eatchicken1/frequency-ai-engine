param(
  [switch]$Reload
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSCommandPath
$venvPython = Join-Path $root '.venv-frequency\Scripts\python.exe'

if (-not (Test-Path $venvPython)) {
  throw "Project venv not found: $venvPython"
}

Set-Location $root
$env:RUNTIME_ALLOW_INSECURE = 'true'
if (-not $env:REDIS_URL) {
  $env:REDIS_URL = 'redis://:123456@127.0.0.1:6379/0'
}

if ($Reload) {
  & $venvPython -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
}
else {
  & $venvPython -m uvicorn app.main:app --host 0.0.0.0 --port 8000
}
