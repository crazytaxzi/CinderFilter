$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "CinderFilter's Python environment is missing. Run START_CINDERFILTER.bat once first."
}

Write-Host "Launching CinderFilter with the lookahead-safe CUDA noise engine..." -ForegroundColor Green
& $venvPython "$PSScriptRoot\cinderfilter_voice_lock_v2_pitch_saved_gpu_noise_stable.py"
