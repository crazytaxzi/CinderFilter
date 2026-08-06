$ErrorActionPreference = "Stop"
$AppRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $AppRoot
$Host.UI.RawUI.WindowTitle = "CinderFilter Lookahead-Safe CUDA Repair and Verification"

Write-Host ""
Write-Host "CinderFilter Lookahead-Safe CUDA Repair" -ForegroundColor Cyan
Write-Host "This keeps the installed CUDA PyTorch build and replaces the invalid zero-lookahead worker." -ForegroundColor Yellow
Write-Host ""

$python = Join-Path $AppRoot ".venv_cuda_noise\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "The CUDA noise environment is missing. Run INSTALL_CUDA_NOISE_ENGINE.bat first."
}

Write-Host "Pinning the DeepFilterNet 0.5.6 runtime..." -ForegroundColor Cyan
& $python -m pip install --upgrade --force-reinstall --no-cache-dir `
    "numpy==1.26.4" "packaging==23.2" `
    "appdirs==1.4.4" "loguru>=0.7,<1" "requests>=2.27,<3" `
    "sympy>=1.6" "typing-extensions>=4.10,<5"
if ($LASTEXITCODE -ne 0) { throw "DeepFilter runtime dependency repair failed." }

& $python -m pip install --upgrade --force-reinstall --no-cache-dir `
    DeepFilterNet==0.5.6 DeepFilterLib==0.5.6 --no-deps
if ($LASTEXITCODE -ne 0) { throw "DeepFilterNet binary repair failed." }

& $python -m pip uninstall -y wheel | Out-Null

Write-Host "Checking runtime consistency..." -ForegroundColor Cyan
& $python -m pip check
if ($LASTEXITCODE -ne 0) {
    throw "The isolated CUDA noise environment still has a genuine dependency conflict."
}

$worker = Join-Path $AppRoot "cuda_noise_worker_stable.py"
if (-not (Test-Path -LiteralPath $worker)) {
    throw "Missing lookahead-safe CUDA worker: $worker"
}

Write-Host ""
Write-Host "Running 20 consecutive 500 ms lookahead-aware CUDA windows..." -ForegroundColor Cyan
$verification = & $python $worker --self-test --atten 45 --chunk-samples 24000
if ($LASTEXITCODE -ne 0) {
    throw "The lookahead-safe CUDA self-test failed. The full traceback is shown above."
}
Write-Host $verification -ForegroundColor Green
try {
    $result = $verification | Select-Object -Last 1 | ConvertFrom-Json
} catch {
    throw "The CUDA worker returned unreadable verification output: $verification"
}
if (-not $result.ok) { throw "The CUDA worker returned invalid audio during verification." }
if ($result.p95_rtf -ge 1.0) {
    throw "CUDA is functional but too slow at the tested size: p95 RTF $($result.p95_rtf)."
}
if ($result.df_lookahead_frames -lt 1) {
    throw "The verifier expected a non-zero DF lookahead model but reported $($result.df_lookahead_frames)."
}

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
$settings["noise_cuda_preset"] = "Balanced"
$settings | ConvertTo-Json -Depth 8 | Set-Content -Path $settingsPath -Encoding UTF8

@{
    verified_unix = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    device = $result.device
    torch = $result.torch
    cuda = $result.cuda
    median_rtf = $result.median_rtf
    p95_rtf = $result.p95_rtf
    streaming_mode = $result.streaming_mode
    df_lookahead_frames = $result.df_lookahead_frames
    conv_lookahead_frames = $result.conv_lookahead_frames
    past_context_ms = $result.past_context_ms
    future_context_ms = $result.future_context_ms
    fixed_latency_ms = $result.latency_ms
} | ConvertTo-Json | Set-Content -Path (Join-Path $stateDir "cuda-noise-stable.json") -Encoding UTF8

Write-Host ""
Write-Host "LOOKAHEAD-SAFE CUDA MAIN NOISE REDUCER VERIFIED" -ForegroundColor Green
Write-Host "GPU: $($result.device)" -ForegroundColor Green
Write-Host "Model lookahead: DF $($result.df_lookahead_frames) frames / Conv $($result.conv_lookahead_frames) frames" -ForegroundColor Green
Write-Host "Context: $($result.past_context_ms) ms past / $($result.future_context_ms) ms future" -ForegroundColor Green
Write-Host "Median RTF: $($result.median_rtf)   P95 RTF: $($result.p95_rtf)" -ForegroundColor Green
Write-Host "Saved mode: CUDA / Balanced. CPU fallback is blocked in explicit CUDA mode." -ForegroundColor Green

$launcher = Join-Path $AppRoot "START_CINDERFILTER_V2_PITCH.bat"
if (Test-Path -LiteralPath $launcher) {
    Write-Host "Relaunching CinderFilter..." -ForegroundColor Cyan
    Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "`"$launcher`"" -WorkingDirectory $AppRoot
}
