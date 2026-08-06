$ErrorActionPreference = "Stop"
$AppRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $AppRoot
$Host.UI.RawUI.WindowTitle = "CinderFilter CUDA Noise Engine Repair"

Write-Host ""
Write-Host "CinderFilter CUDA Noise Engine Repair" -ForegroundColor Cyan
Write-Host "Fixing DeepFilterNet 0.5.6 compatibility without redownloading PyTorch." -ForegroundColor Yellow
Write-Host ""

$venvPython = Join-Path $AppRoot ".venv_cuda_noise\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "The CUDA noise environment does not exist. Run INSTALL_CUDA_NOISE_ENGINE.bat instead."
}

Write-Host "Pinning DeepFilterNet-compatible runtime packages..." -ForegroundColor Cyan
& $venvPython -m pip install --upgrade --force-reinstall --no-cache-dir `
    "numpy==1.26.4" "packaging==23.2" `
    "appdirs==1.4.4" "loguru>=0.7,<1" "requests>=2.27,<3" `
    "sympy>=1.6" "typing-extensions>=4.10,<5"
if ($LASTEXITCODE -ne 0) { throw "Dependency repair failed." }

Write-Host "Ensuring DeepFilterNet 0.5.6 binaries are present..." -ForegroundColor Cyan
& $venvPython -m pip install --upgrade --force-reinstall --no-cache-dir `
    DeepFilterNet==0.5.6 DeepFilterLib==0.5.6 --no-deps
if ($LASTEXITCODE -ne 0) { throw "DeepFilterNet binary repair failed." }

Write-Host "Removing the optional wheel CLI package..." -ForegroundColor Cyan
Write-Host "It is not needed to run CinderFilter and conflicts with DeepFilterNet's packaging pin." -ForegroundColor DarkGray
& $venvPython -m pip uninstall -y wheel
if ($LASTEXITCODE -ne 0) { throw "Could not remove the conflicting wheel package." }

Write-Host "Checking installed runtime-package consistency..." -ForegroundColor Cyan
& $venvPython -m pip check
if ($LASTEXITCODE -ne 0) {
    throw "The repaired CUDA noise environment still has incompatible runtime packages."
}

$worker = Join-Path $AppRoot "cuda_noise_worker.py"
if (-not (Test-Path -LiteralPath $worker)) { throw "Missing patched CUDA worker: $worker" }

Write-Host ""
Write-Host "Running RTX CUDA + DeepFilterNet3 self-test..." -ForegroundColor Cyan
$verification = & $venvPython $worker --self-test --atten 45
if ($LASTEXITCODE -ne 0) {
    throw "CUDA DeepFilterNet3 self-test still failed. Read the worker error above."
}
Write-Host $verification -ForegroundColor Green

try {
    $result = $verification | Select-Object -Last 1 | ConvertFrom-Json
} catch {
    throw "CUDA self-test returned unreadable output: $verification"
}
if (-not $result.ok) { throw "CUDA self-test completed but did not return valid audio." }

$stateDir = Join-Path $env:LOCALAPPDATA "CinderFilter"
New-Item -ItemType Directory -Path $stateDir -Force | Out-Null
$settingsPath = Join-Path $stateDir "settings.json"
$settings = @{}
if (Test-Path -LiteralPath $settingsPath) {
    try {
        $loaded = Get-Content $settingsPath -Raw | ConvertFrom-Json
        foreach ($property in $loaded.PSObject.Properties) {
            $settings[$property.Name] = $property.Value
        }
    } catch { }
}
$settings["noise_backend"] = "CUDA"
$settings["noise_cuda_preset"] = "Low Latency"
$settings | ConvertTo-Json -Depth 8 | Set-Content -Path $settingsPath -Encoding UTF8

Write-Host ""
Write-Host "CUDA MAIN NOISE REDUCER REPAIRED" -ForegroundColor Green
Write-Host "GPU: $($result.device)" -ForegroundColor Green
Write-Host "Torch: $($result.torch) / CUDA $($result.cuda)" -ForegroundColor Green

$launcher = Join-Path $AppRoot "START_CINDERFILTER_V2_PITCH.bat"
if (Test-Path -LiteralPath $launcher) {
    Write-Host "Relaunching CinderFilter..." -ForegroundColor Cyan
    Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "`"$launcher`"" -WorkingDirectory $AppRoot
}
