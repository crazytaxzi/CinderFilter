param(
    [Parameter(Mandatory = $true)][string]$PythonPath,
    [Parameter(Mandatory = $true)][string]$AppRoot,
    [Parameter(Mandatory = $true)][int]$ParentPid,
    [Parameter(Mandatory = $true)][ValidateSet("cu128", "cu130")][string]$WheelTag
)

$ErrorActionPreference = "Stop"
$Host.UI.RawUI.WindowTitle = "CinderFilter CUDA Runtime Installer"

Write-Host ""
Write-Host "CinderFilter CUDA Runtime Installer" -ForegroundColor Cyan
Write-Host "Waiting for CinderFilter to release PyTorch files..." -ForegroundColor DarkGray
Wait-Process -Id $ParentPid -ErrorAction SilentlyContinue

if (-not (Test-Path $PythonPath)) {
    throw "The CinderFilter virtual-environment Python executable was not found: $PythonPath"
}

$indexUrl = "https://download.pytorch.org/whl/$WheelTag"
Write-Host ""
Write-Host "Installing official PyTorch 2.11.0 $WheelTag runtime..." -ForegroundColor Cyan
Write-Host "This is a large one-time download. Do not close this window." -ForegroundColor Yellow

& $PythonPath -m pip install --upgrade --force-reinstall `
    torch==2.11.0 torchaudio==2.11.0 `
    --index-url $indexUrl

if ($LASTEXITCODE -ne 0) {
    throw "PyTorch CUDA installation failed with exit code $LASTEXITCODE."
}

Write-Host ""
Write-Host "Verifying CUDA from the CinderFilter environment..." -ForegroundColor Cyan
$verification = & $PythonPath -c @"
import json
import torch
result = {
    "torch": torch.__version__,
    "build_cuda": torch.version.cuda,
    "available": bool(torch.cuda.is_available()),
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
print(json.dumps(result))
"@

if ($LASTEXITCODE -ne 0) {
    throw "PyTorch installed, but verification could not run."
}

try {
    $result = $verification | ConvertFrom-Json
} catch {
    throw "PyTorch verification returned unreadable output: $verification"
}

if (-not $result.available) {
    Write-Host ""
    Write-Host "The CUDA wheel installed, but PyTorch still cannot access the GPU." -ForegroundColor Red
    Write-Host "Detected Torch build CUDA: $($result.build_cuda)" -ForegroundColor Yellow
    Write-Host "Update the NVIDIA driver and reboot Windows, then launch CinderFilter again." -ForegroundColor Yellow
    Read-Host "Press Enter to close"
    exit 1
}

Write-Host ""
Write-Host "CUDA READY: $($result.device)" -ForegroundColor Green
Write-Host "PyTorch $($result.torch), CUDA build $($result.build_cuda)" -ForegroundColor Green

$stateDir = Join-Path $env:LOCALAPPDATA "CinderFilter"
New-Item -ItemType Directory -Path $stateDir -Force | Out-Null
@{
    installed_unix = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    wheel_tag = $WheelTag
    torch = $result.torch
    build_cuda = $result.build_cuda
    device = $result.device
} | ConvertTo-Json | Set-Content -Path (Join-Path $stateDir "gpu-runtime.json") -Encoding UTF8

$launcher = Join-Path $AppRoot "START_CINDERFILTER_V2_PITCH.bat"
if (Test-Path $launcher) {
    Write-Host ""
    Write-Host "Relaunching CinderFilter..." -ForegroundColor Cyan
    Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "`"$launcher`"" -WorkingDirectory $AppRoot
} else {
    Write-Host "CUDA is ready. Launch CinderFilter normally." -ForegroundColor Green
}
