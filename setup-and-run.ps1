param([switch]$Console)

$ErrorActionPreference = "Stop"
$AppRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $AppRoot
$Host.UI.RawUI.WindowTitle = "CinderFilter Setup"

function Invoke-Native {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$Arguments = @(),
        [switch]$Quiet
    )

    # Windows PowerShell 5.1 converts some native stderr output into
    # NativeCommandError records. Probes are allowed to return nonzero; callers
    # inspect the exit code explicitly instead of letting ErrorActionPreference
    # terminate the setup script.
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        if ($Quiet) {
            & $FilePath @Arguments *> $null
        } else {
            & $FilePath @Arguments 2>&1 | ForEach-Object { Write-Host $_ }
        }
        return [int]$LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousPreference
    }
}

function Find-Python {
    $py = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($py) {
        foreach ($version in @("-3.12", "-3.11")) {
            $code = Invoke-Native -FilePath $py.Source -Arguments @(
                $version,
                "-c",
                "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
            ) -Quiet
            if ($code -eq 0) { return @($py.Source, $version) }
        }
    }

    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        $code = Invoke-Native -FilePath $pythonCommand.Source -Arguments @(
            "-c",
            "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
        ) -Quiet
        if ($code -eq 0) { return @($pythonCommand.Source) }
    }

    throw "Python 3.11 or newer is required. Install Python from python.org, then run START_CINDERFILTER.bat again."
}

$venv = Join-Path $AppRoot ".venv"
$python = Join-Path $venv "Scripts\python.exe"
$pythonw = Join-Path $venv "Scripts\pythonw.exe"

if (-not (Test-Path -LiteralPath $python)) {
    $found = Find-Python
    $command = $found[0]
    $arguments = @()
    if ($found.Count -gt 1) { $arguments += $found[1] }
    $arguments += @("-m", "venv", $venv)

    Write-Host "Creating CinderFilter environment..." -ForegroundColor Cyan
    $code = Invoke-Native -FilePath $command -Arguments $arguments
    if ($code -ne 0) { throw "Could not create .venv." }
}

$requirements = Join-Path $AppRoot "requirements.txt"
$marker = Join-Path $venv ".cinderfilter-requirements.sha256"
$hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $requirements).Hash
$needsInstall = -not (Test-Path -LiteralPath $marker)
if (-not $needsInstall) {
    $needsInstall = ((Get-Content -LiteralPath $marker -Raw).Trim() -ne $hash)
}

# Torch is installed separately so the NVIDIA CUDA wheel can come from the
# official PyTorch CUDA index. A failed probe on a fresh environment is normal.
$torchProbe = Invoke-Native -FilePath $python -Arguments @(
    "-c",
    "import torch; raise SystemExit(0 if torch.__version__.startswith('2.11.') and torch.cuda.is_available() else 1)"
) -Quiet
$torchReady = ($torchProbe -eq 0)

if (-not $torchReady) {
    $nvidia = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    if ($nvidia) {
        Write-Host "Installing PyTorch 2.11 CUDA 12.8 runtime..." -ForegroundColor Cyan
        $code = Invoke-Native -FilePath $python -Arguments @(
            "-m", "pip", "install", "--upgrade", "--disable-pip-version-check",
            "torch==2.11.0", "torchaudio==2.11.0",
            "--index-url", "https://download.pytorch.org/whl/cu128"
        )
    } else {
        Write-Host "NVIDIA driver not detected; installing the CPU PyTorch runtime." -ForegroundColor Yellow
        $code = Invoke-Native -FilePath $python -Arguments @(
            "-m", "pip", "install", "--upgrade", "--disable-pip-version-check",
            "torch==2.11.0", "torchaudio==2.11.0"
        )
    }
    if ($code -ne 0) { throw "PyTorch installation failed." }
}

if ($needsInstall) {
    Write-Host "Installing or updating CinderFilter dependencies..." -ForegroundColor Cyan
    $code = Invoke-Native -FilePath $python -Arguments @(
        "-m", "pip", "install", "--upgrade", "--disable-pip-version-check",
        "-r", $requirements
    )
    if ($code -ne 0) { throw "CinderFilter dependency installation failed." }
}

Write-Host "Validating the CinderFilter runtime..." -ForegroundColor Cyan
$importProbe = Invoke-Native -FilePath $python -Arguments @(
    "-c",
    "import PySide6, numpy, sounddevice, deepfilternet_rs, speechbrain, psutil, torch, torchaudio; print('Runtime imports OK'); print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
)
if ($importProbe -ne 0) {
    throw "CinderFilter runtime validation failed. The complete Python import error is printed above."
}

$checkCode = Invoke-Native -FilePath $python -Arguments @("-m", "pip", "check")
if ($checkCode -ne 0) {
    throw "The main CinderFilter environment has dependency conflicts."
}

Set-Content -LiteralPath $marker -Value $hash -Encoding ASCII

if ($Console) {
    $code = Invoke-Native -FilePath $python -Arguments @((Join-Path $AppRoot "main.py"))
    exit $code
}

if (-not (Test-Path -LiteralPath $pythonw)) { $pythonw = $python }
Start-Process -FilePath $pythonw -ArgumentList ('"' + (Join-Path $AppRoot "main.py") + '"') -WorkingDirectory $AppRoot
