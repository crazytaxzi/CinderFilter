param([switch]$Console)

$ErrorActionPreference = "Stop"
$AppRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $AppRoot
$Host.UI.RawUI.WindowTitle = "CinderFilter Setup"

function Find-Python {
    $py = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($py) {
        foreach ($version in @("-3.12", "-3.11")) {
            & $py.Source $version -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" 2>$null
            if ($LASTEXITCODE -eq 0) { return @($py.Source, $version) }
        }
    }
    $python = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($python) {
        & $python.Source -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) { return @($python.Source) }
    }
    throw "Python 3.11 or newer is required. Install Python from python.org, then run START_CINDERFILTER.bat again."
}

$venv = Join-Path $AppRoot ".venv"
$python = Join-Path $venv "Scripts\python.exe"
$pythonw = Join-Path $venv "Scripts\pythonw.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $found = Find-Python
    $command = $found[0]
    $prefix = @()
    if ($found.Count -gt 1) { $prefix = @($found[1]) }
    Write-Host "Creating CinderFilter environment..." -ForegroundColor Cyan
    & $command @prefix -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw "Could not create .venv." }
}

$requirements = Join-Path $AppRoot "requirements.txt"
$marker = Join-Path $venv ".cinderfilter-requirements.sha256"
$hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $requirements).Hash
$needsInstall = -not (Test-Path -LiteralPath $marker)
if (-not $needsInstall) {
    $needsInstall = ((Get-Content -LiteralPath $marker -Raw).Trim() -ne $hash)
}

& $python -c "import PySide6, numpy, sounddevice, deepfilternet_rs, speechbrain, psutil" 2>$null
if ($LASTEXITCODE -ne 0) { $needsInstall = $true }

& $python -c "import torch; raise SystemExit(0 if torch.__version__.startswith('2.11.') and torch.cuda.is_available() else 1)" 2>$null
$torchReady = ($LASTEXITCODE -eq 0)
if (-not $torchReady) {
    $nvidia = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    if ($nvidia) {
        Write-Host "Installing PyTorch 2.11 CUDA 12.8 runtime..." -ForegroundColor Cyan
        & $python -m pip install --upgrade --disable-pip-version-check `
            torch==2.11.0 torchaudio==2.11.0 `
            --index-url https://download.pytorch.org/whl/cu128
    } else {
        Write-Host "NVIDIA driver not detected; installing the CPU PyTorch runtime." -ForegroundColor Yellow
        & $python -m pip install --upgrade --disable-pip-version-check torch==2.11.0 torchaudio==2.11.0
    }
    if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed." }
}

if ($needsInstall) {
    Write-Host "Installing or updating CinderFilter dependencies..." -ForegroundColor Cyan
    & $python -m pip install --upgrade --disable-pip-version-check -r $requirements
    if ($LASTEXITCODE -ne 0) { throw "CinderFilter dependency installation failed." }
    Set-Content -LiteralPath $marker -Value $hash -Encoding ASCII
}

& $python -m pip check
if ($LASTEXITCODE -ne 0) { throw "The main CinderFilter environment has dependency conflicts." }

if ($Console) {
    & $python (Join-Path $AppRoot "main.py")
    exit $LASTEXITCODE
}
if (-not (Test-Path -LiteralPath $pythonw)) { $pythonw = $python }
Start-Process -FilePath $pythonw -ArgumentList ('"' + (Join-Path $AppRoot "main.py") + '"') -WorkingDirectory $AppRoot
