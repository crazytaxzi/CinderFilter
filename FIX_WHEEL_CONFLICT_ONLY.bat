@echo off
setlocal
cd /d "%~dp0"
set "PY=%~dp0.venv_cuda_noise\Scripts\python.exe"
if not exist "%PY%" (
    echo CUDA noise environment is missing.
    pause
    exit /b 1
)
"%PY%" -m pip uninstall -y wheel
if errorlevel 1 (
    echo Could not remove wheel.
    pause
    exit /b 1
)
"%PY%" -m pip check
pause
endlocal
