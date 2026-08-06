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
$Host.UI.RawUI.WindowTitle = "CinderFilter CUDA Main Noise Engine"
if ($ParentPid -gt 0) { Wait-Process -Id $ParentPid -ErrorAction SilentlyContinue }

function Test-Python311([string]$Command, [string[]]$Prefix) {
    & $Command @Prefix -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3,11) else 1)" 2>$null
    return ($LASTEXITCODE -eq 0)
}

$command = $null
$prefix = @()
$py = Get-Command py.exe -ErrorAction SilentlyContinue
if ($py -and (Test-Python311 $py.Source @("-3.11"))) {
    $command = $py.Source; $prefix = @("-3.11")
}
if (-not $command) {
    $candidate = Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe"
    if ((Test-Path -LiteralPath $candidate) -and (Test-Python311 $candidate @())) { $command = $candidate }
}
if (-not $command) {
    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if (-not $winget) { throw "Python 3.11 is required for the DeepFilterNet CUDA sidecar." }
    Write-Host "Installing Python 3.11 side-by-side..." -ForegroundColor Cyan
    & $winget.Source install --exact --id Python.Python.3.11 --scope user --silent `
        --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) { throw "Python 3.11 installation failed." }
    $candidate = Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe"
    if ((Test-Path -LiteralPath $candidate) -and (Test-Python311 $candidate @())) { $command = $candidate }
}
if (-not $command) { throw "Python 3.11 could not be located after installation." }
if (-not (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue)) {
    throw "The NVIDIA driver was not detected."
}

$venv = Join-Path $AppRoot ".venv_cuda_noise"
$python = Join-Path $venv "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    Write-Host "Creating isolated CUDA denoiser environment..." -ForegroundColor Cyan
    & $command @prefix -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw "Could not create .venv_cuda_noise." }
}

& $python -m pip install --upgrade --disable-pip-version-check pip setuptools
if ($LASTEXITCODE -ne 0) { throw "pip bootstrap failed." }

Write-Host "Installing CUDA PyTorch..." -ForegroundColor Cyan
& $python -m pip install --upgrade --force-reinstall --no-cache-dir `
    torch==2.11.0 torchaudio==2.11.0 `
    --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw "CUDA PyTorch installation failed." }

Write-Host "Installing the pinned DeepFilterNet3 runtime..." -ForegroundColor Cyan
& $python -m pip install --upgrade --force-reinstall --no-cache-dir `
    DeepFilterNet==0.5.6 DeepFilterLib==0.5.6 --no-deps
if ($LASTEXITCODE -ne 0) { throw "DeepFilterNet binary installation failed." }
& $python -m pip install --upgrade --force-reinstall --no-cache-dir `
    "numpy==1.26.4" "packaging==23.2" "appdirs==1.4.4" `
    "loguru>=0.7,<1" "requests>=2.27,<3" "sympy>=1.6" "typing-extensions>=4.10,<5"
if ($LASTEXITCODE -ne 0) { throw "DeepFilterNet dependency installation failed." }
& $python -m pip uninstall -y wheel | Out-Null
& $python -m pip check
if ($LASTEXITCODE -ne 0) { throw "The CUDA denoiser environment has dependency conflicts." }

$worker = Join-Path $AppRoot "cuda_noise_worker.py"
if (-not (Test-Path -LiteralPath $worker)) { throw "Missing cuda_noise_worker.py." }
Write-Host "Running 20-window RTX acceptance test..." -ForegroundColor Cyan
$output = & $python $worker --self-test --atten 45 --chunk-samples 24000
if ($LASTEXITCODE -ne 0) { throw "The CUDA DeepFilterNet3 acceptance test failed." }
$info = $output | Select-Object -Last 1 | ConvertFrom-Json
if (-not $info.ok) { throw "The CUDA worker produced invalid audio." }
if ([double]$info.p95_rtf -ge 1.0) { throw "CUDA is functional but missed real time: p95 RTF $($info.p95_rtf)." }

$state = Join-Path $env:LOCALAPPDATA "CinderFilter"
New-Item -ItemType Directory -Path $state -Force | Out-Null
$info | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $state "cuda-noise-engine.json") -Encoding UTF8
Write-Host "CUDA MAIN NOISE REDUCER READY: $($info.device), p95 RTF $($info.p95_rtf)" -ForegroundColor Green
Start-Process -FilePath (Join-Path $AppRoot "START_CINDERFILTER.bat") -WorkingDirectory $AppRoot
