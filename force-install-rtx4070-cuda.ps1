$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$Host.UI.RawUI.WindowTitle = "CinderFilter RTX 4070 CUDA Force Installer"

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "CinderFilter's virtual environment is missing. Run START_CINDERFILTER.bat once, then run this installer again."
}

Write-Host ""
Write-Host "CinderFilter RTX 4070 CUDA Force Installer" -ForegroundColor Cyan
Write-Host "This replaces the current PyTorch build with the official CUDA 12.8 build." -ForegroundColor Yellow
Write-Host "The download is large. Do not close this window." -ForegroundColor Yellow
Write-Host ""

$nvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if ($nvidiaSmi) {
    Write-Host "NVIDIA driver sees:" -ForegroundColor Cyan
    & $nvidiaSmi.Source --query-gpu=name,driver_version,memory.total --format=csv,noheader
    Write-Host ""
} else {
    Write-Host "WARNING: nvidia-smi was not found. The CUDA wheel will still be installed," -ForegroundColor Yellow
    Write-Host "but GPU verification may fail until the NVIDIA driver is installed or updated." -ForegroundColor Yellow
    Write-Host ""
}

Write-Host "Removing existing CPU or mismatched PyTorch packages..." -ForegroundColor Cyan
& $python -m pip uninstall -y torch torchaudio torchvision

Write-Host ""
Write-Host "Installing PyTorch 2.11.0 + CUDA 12.8..." -ForegroundColor Cyan
& $python -m pip install --upgrade --force-reinstall --no-cache-dir `
    torch==2.11.0 torchaudio==2.11.0 `
    --index-url https://download.pytorch.org/whl/cu128

if ($LASTEXITCODE -ne 0) {
    throw "CUDA PyTorch installation failed with exit code $LASTEXITCODE."
}

Write-Host ""
Write-Host "Checking package consistency..." -ForegroundColor Cyan
& $python -m pip check
if ($LASTEXITCODE -ne 0) {
    Write-Host "pip reported a dependency warning. Continuing to the CUDA hardware verification." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Verifying the RTX GPU from CinderFilter's Python environment..." -ForegroundColor Cyan
$verification = & $python -c @"
import json
import torch
available = bool(torch.cuda.is_available())
name = torch.cuda.get_device_name(0) if available else None
props = torch.cuda.get_device_properties(0) if available else None
print(json.dumps({
    "torch": torch.__version__,
    "build_cuda": torch.version.cuda,
    "available": available,
    "device": name,
    "vram_gb": round(props.total_memory / 1024**3, 2) if props else None,
    "capability": list(torch.cuda.get_device_capability(0)) if available else None,
}))
"@

if ($LASTEXITCODE -ne 0) {
    throw "PyTorch installed, but the verification command failed."
}

try {
    $result = $verification | Select-Object -Last 1 | ConvertFrom-Json
} catch {
    throw "CUDA verification returned unreadable output: $verification"
}

if (-not $result.available) {
    Write-Host ""
    Write-Host "The CUDA-enabled PyTorch wheel is installed, but Windows/PyTorch still cannot access the GPU." -ForegroundColor Red
    Write-Host "Torch: $($result.torch)   CUDA build: $($result.build_cuda)" -ForegroundColor Yellow
    if ($nvidiaSmi) {
        Write-Host "NVIDIA diagnostic output:" -ForegroundColor Yellow
        & $nvidiaSmi.Source
    }
    Write-Host ""
    Write-Host "Update the NVIDIA Game Ready or Studio driver, reboot Windows, then run this installer again." -ForegroundColor Yellow
    Read-Host "Press Enter to close"
    exit 1
}

if ($result.device -notmatch "4070") {
    Write-Host "WARNING: CUDA works, but the selected device reports as '$($result.device)', not an RTX 4070." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "CUDA READY" -ForegroundColor Green
Write-Host "GPU: $($result.device)" -ForegroundColor Green
Write-Host "VRAM: $($result.vram_gb) GB" -ForegroundColor Green
Write-Host "PyTorch: $($result.torch)" -ForegroundColor Green
Write-Host "CUDA build: $($result.build_cuda)" -ForegroundColor Green
Write-Host "Compute capability: $($result.capability -join '.')" -ForegroundColor Green

$stateDir = Join-Path $env:LOCALAPPDATA "CinderFilter"
New-Item -ItemType Directory -Path $stateDir -Force | Out-Null
$settingsPath = Join-Path $stateDir "settings.json"
$settings = @{}
if (Test-Path $settingsPath) {
    try {
        $loaded = Get-Content $settingsPath -Raw | ConvertFrom-Json
        foreach ($property in $loaded.PSObject.Properties) {
            $settings[$property.Name] = $property.Value
        }
    } catch {
        Write-Host "Existing settings file was unreadable; creating a clean one." -ForegroundColor Yellow
    }
}
$settings["v2_device"] = "CUDA"
$settings | ConvertTo-Json -Depth 8 | Set-Content -Path $settingsPath -Encoding UTF8

@{
    installed_unix = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    forced = $true
    wheel_tag = "cu128"
    torch = $result.torch
    build_cuda = $result.build_cuda
    device = $result.device
    vram_gb = $result.vram_gb
} | ConvertTo-Json | Set-Content -Path (Join-Path $stateDir "gpu-runtime.json") -Encoding UTF8

$launcher = Join-Path $PSScriptRoot "START_CINDERFILTER_V2_PITCH.bat"
if (Test-Path $launcher) {
    Write-Host ""
    Write-Host "Relaunching CinderFilter with CUDA selected..." -ForegroundColor Cyan
    Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "`"$launcher`"" -WorkingDirectory $PSScriptRoot
} else {
    Write-Host "CUDA is ready. Launch CinderFilter normally." -ForegroundColor Green
}
