$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Find-Python312 {
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "$env:ProgramFiles\Python312\python.exe"
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) { return $candidate }
    }

    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        try {
            $version = & $pythonCommand.Source -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
            if ($version -eq "3.12") { return $pythonCommand.Source }
        } catch { }
    }

    return $null
}

$python312 = Find-Python312
if (-not $python312) {
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "Python 3.12 is required and winget is unavailable. Install Python 3.12 x64, then run this file again."
    }

    Write-Host "Installing Python 3.12 for CinderFilter..." -ForegroundColor Cyan
    & $winget.Source install --exact --id Python.Python.3.12 --scope user `
        --accept-package-agreements --accept-source-agreements --silent
    if ($LASTEXITCODE -ne 0) {
        throw "Python installation failed with exit code $LASTEXITCODE."
    }
    $python312 = Find-Python312
    if (-not $python312) {
        throw "Python 3.12 installed, but its executable could not be located. Reopen this launcher."
    }
}

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "Creating the CinderFilter environment..." -ForegroundColor Cyan
    & $python312 -m venv .venv
}

$requirementsHash = (Get-FileHash "$PSScriptRoot\requirements.txt" -Algorithm SHA256).Hash
$marker = Join-Path $PSScriptRoot ".venv\.requirements-sha256"
$installedHash = if (Test-Path $marker) { (Get-Content $marker -Raw).Trim() } else { "" }

if ($installedHash -ne $requirementsHash) {
    Write-Host "Installing the weighted AI audio engine..." -ForegroundColor Cyan
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install --requirement "$PSScriptRoot\requirements.txt"
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency installation failed."
    }
    Set-Content -Path $marker -Value $requirementsHash -NoNewline
}

Write-Host "Launching CinderFilter..." -ForegroundColor Green
& $venvPython "$PSScriptRoot\cinderfilter_threadsafe.py"
