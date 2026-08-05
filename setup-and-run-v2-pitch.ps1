$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "CinderFilter's Python environment is missing. Run START_CINDERFILTER.bat once first."
}

Write-Host "Launching CinderFilter with selectable CUDA main noise reduction..." -ForegroundColor Green
& $venvPython "$PSScriptRoot\cinderfilter_voice_lock_v2_pitch_saved_gpu_noise.py"
