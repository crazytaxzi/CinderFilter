param(
    [string]$AppRoot = "",
    [int]$ParentPid = 0
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($AppRoot)) {
    $AppRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
} else {
    $AppRoot = (Resolve-Path -LiteralPath $AppRoot).Path
}
Set-Location -LiteralPath $AppRoot
$Host.UI.RawUI.WindowTitle = "CinderFilter CUDA Runtime"
if ($ParentPid -gt 0) { Wait-Process -Id $ParentPid -ErrorAction SilentlyContinue }

$python = Join-Path $AppRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "The main .venv is missing. Run START_CINDERFILTER.bat first."
}
if (-not (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue)) {
    throw "The NVIDIA driver was not detected. Install or update the NVIDIA driver first."
}

Write-Host "Installing the official PyTorch 2.11 CUDA 12.8 runtime..." -ForegroundColor Cyan
& $python -m pip install --upgrade --force-reinstall --no-cache-dir `
    torch==2.11.0 torchaudio==2.11.0 `
    --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw "CUDA PyTorch installation failed." }

$result = & $python -c "import json,torch; print(json.dumps({'ok':torch.cuda.is_available(),'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"
if ($LASTEXITCODE -ne 0) { throw "CUDA verification command failed." }
$info = $result | Select-Object -Last 1 | ConvertFrom-Json
if (-not $info.ok) { throw "CUDA PyTorch installed, but PyTorch still cannot access the NVIDIA GPU." }

Write-Host "CUDA READY: $($info.device)" -ForegroundColor Green
Write-Host "PyTorch $($info.torch), CUDA $($info.cuda)" -ForegroundColor Green
Start-Process -FilePath (Join-Path $AppRoot "START_CINDERFILTER.bat") -WorkingDirectory $AppRoot
