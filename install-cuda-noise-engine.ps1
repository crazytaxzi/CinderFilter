param(
    [string]$AppRoot = "",
    [int]$ParentPid = 0
)

$ErrorActionPreference = "Stop"

# The installer always lives in the CinderFilter folder. Derive that path
# directly instead of trusting fragile batch-file quoting. If AppRoot was
# supplied by the GUI, accept it only when it resolves to a real directory.
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($AppRoot)) {
    $AppRoot = $ScriptRoot
} else {
    try {
        $AppRoot = (Resolve-Path -LiteralPath $AppRoot -ErrorAction Stop).Path
    } catch {
        Write-Host "Ignoring malformed AppRoot argument; using installer folder." -ForegroundColor Yellow
        $AppRoot = $ScriptRoot
    }
}
Set-Location -LiteralPath $AppRoot
$Host.UI.RawUI.WindowTitle = "CinderFilter CUDA Main Noise Engine Installer"

Write-Host ""
Write-Host "CinderFilter CUDA Main Noise Engine" -ForegroundColor Cyan
Write-Host "Installing the official DeepFilterNet3 PyTorch backend for the RTX GPU." -ForegroundColor Cyan
Write-Host "This is a large one-time download." -ForegroundColor Yellow
Write-Host ""

if ($ParentPid -gt 0) {
    Write-Host "Waiting for CinderFilter to release its audio and Python files..." -ForegroundColor DarkGray
    Wait-Process -Id $ParentPid -ErrorAction SilentlyContinue
}

function Test-Python311 {
    param([string]$Command, [string[]]$PrefixArgs)
    try {
        $args = @($PrefixArgs) + @("-c", "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)")
        & $Command @args | Out-Null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

$pythonCommand = $null
$pythonPrefix = @()
$py = Get-Command py.exe -ErrorAction SilentlyContinue
if ($py -and (Test-Python311 $py.Source @("-3.11"))) {
    $pythonCommand = $py.Source
    $pythonPrefix = @("-3.11")
}

if (-not $pythonCommand) {
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "$env:ProgramFiles\Python311\python.exe"
    )
    foreach ($candidate in $candidates) {
        if ((Test-Path $candidate) -and (Test-Python311 $candidate @())) {
            $pythonCommand = $candidate
            $pythonPrefix = @()
            break
        }
    }
}

if (-not $pythonCommand) {
    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "Python 3.11 is required for the official Windows DeepFilterNet library, and winget is unavailable."
    }
    Write-Host "Installing Python 3.11 side-by-side with the existing Python 3.12..." -ForegroundColor Cyan
    & $winget.Source install --exact --id Python.Python.3.11 --scope user `
        --accept-package-agreements --accept-source-agreements --silent
    if ($LASTEXITCODE -ne 0) {
        throw "Python 3.11 installation failed with exit code $LASTEXITCODE."
    }
    $py = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($py -and (Test-Python311 $py.Source @("-3.11"))) {
        $pythonCommand = $py.Source
        $pythonPrefix = @("-3.11")
    } else {
        $candidate = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
        if ((Test-Path $candidate) -and (Test-Python311 $candidate @())) {
            $pythonCommand = $candidate
            $pythonPrefix = @()
        }
    }
    if (-not $pythonCommand) {
        throw "Python 3.11 installed, but CinderFilter could not locate it. Reboot Windows and run this installer again."
    }
}

$venv = Join-Path $AppRoot ".venv_cuda_noise"
$venvPython = Join-Path $venv "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "Creating the isolated CUDA noise-engine environment..." -ForegroundColor Cyan
    $venvArgs = @($pythonPrefix) + @("-m", "venv", $venv)
    & $pythonCommand @venvArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the CUDA noise-engine virtual environment."
    }
}

Write-Host "Updating pip..." -ForegroundColor Cyan
& $venvPython -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw "pip update failed." }

Write-Host ""
Write-Host "Installing PyTorch 2.11.0 with CUDA 12.8..." -ForegroundColor Cyan
& $venvPython -m pip install --upgrade --force-reinstall --no-cache-dir `
    torch==2.11.0 torchaudio==2.11.0 `
    --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) {
    throw "CUDA PyTorch installation failed."
}

Write-Host ""
Write-Host "Installing the official DeepFilterNet3 runtime..." -ForegroundColor Cyan
# Install the official model and CPython 3.11 Windows library without letting pip
# replace the CUDA-enabled Torch build from the previous step.
& $venvPython -m pip install --upgrade --no-cache-dir `
    DeepFilterNet==0.5.6 DeepFilterLib==0.5.6 --no-deps
if ($LASTEXITCODE -ne 0) {
    throw "DeepFilterNet installation failed."
}
& $venvPython -m pip install --upgrade --force-reinstall --no-cache-dir `
    "numpy==1.26.4" "packaging==23.2" `
    "appdirs==1.4.4" "loguru>=0.7,<1" "requests>=2.27,<3" `
    "sympy>=1.6" "typing-extensions>=4.10,<5"
if ($LASTEXITCODE -ne 0) {
    throw "DeepFilterNet support dependency installation failed."
}

Write-Host "Checking DeepFilterNet dependency compatibility..." -ForegroundColor Cyan
& $venvPython -m pip check
if ($LASTEXITCODE -ne 0) {
    throw "The CUDA noise environment still has incompatible packages."
}

Write-Host ""
Write-Host "Verifying CUDA and preloading DeepFilterNet3..." -ForegroundColor Cyan
Write-Host "The first model load may download the model checkpoint." -ForegroundColor Yellow
$worker = Join-Path $AppRoot "cuda_noise_worker.py"
if (-not (Test-Path $worker)) {
    throw "Missing CUDA worker file: $worker"
}
$verification = & $venvPython $worker --self-test --atten 45
if ($LASTEXITCODE -ne 0) {
    throw "CUDA DeepFilterNet3 self-test failed. See the error above."
}
Write-Host $verification -ForegroundColor Green

try {
    $result = $verification | Select-Object -Last 1 | ConvertFrom-Json
} catch {
    throw "CUDA noise-engine self-test returned unreadable output: $verification"
}
if (-not $result.ok) {
    throw "CUDA noise-engine self-test did not produce valid audio."
}

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
        Write-Host "Existing settings were unreadable; creating clean settings." -ForegroundColor Yellow
    }
}
$settings["noise_backend"] = "CUDA"
$settings["noise_cuda_preset"] = "Low Latency"
$settings | ConvertTo-Json -Depth 8 | Set-Content -Path $settingsPath -Encoding UTF8

@{
    installed_unix = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    model = "DeepFilterNet3"
    device = $result.device
    torch = $result.torch
    cuda = $result.cuda
    vram_gb = $result.vram_gb
    python = "3.11"
} | ConvertTo-Json | Set-Content -Path (Join-Path $stateDir "cuda-noise-engine.json") -Encoding UTF8

Write-Host ""
Write-Host "CUDA MAIN NOISE REDUCER READY" -ForegroundColor Green
Write-Host "GPU: $($result.device)" -ForegroundColor Green
Write-Host "Backend: DeepFilterNet3 / PyTorch CUDA $($result.cuda)" -ForegroundColor Green
Write-Host "The CPU Rust denoiser will no longer be created when Noise Engine is set to CUDA." -ForegroundColor Green

$launcher = Join-Path $AppRoot "START_CINDERFILTER_V2_PITCH.bat"
if (Test-Path $launcher) {
    Write-Host ""
    Write-Host "Relaunching CinderFilter..." -ForegroundColor Cyan
    Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "`"$launcher`"" -WorkingDirectory $AppRoot
}
