@echo off
setlocal
cd /d "%~dp0"
title CinderFilter Stable CUDA Noise Engine Verification
set "PY=%~dp0.venv_cuda_noise\Scripts\python.exe"
if not exist "%PY%" (
    echo The CUDA noise environment is missing.
    echo Run INSTALL_CUDA_NOISE_ENGINE.bat first.
    pause
    exit /b 1
)

echo Removing the optional wheel package if it is still causing a packaging conflict...
"%PY%" -m pip uninstall -y wheel >nul 2>&1

echo Checking runtime package consistency...
"%PY%" -m pip check
if errorlevel 1 (
    echo.
    echo The CUDA environment still has a real dependency conflict.
    pause
    exit /b 1
)

echo.
echo Running 20 consecutive stateful CUDA chunks on the installed RTX GPU...
"%PY%" "%~dp0cuda_noise_worker_stable.py" --self-test --atten 45 --chunk-samples 24000
if errorlevel 1 (
    echo.
    echo Stable CUDA verification failed. The full traceback is above.
    pause
    exit /b 1
)

echo.
echo Stable CUDA noise engine verified.
pause
endlocal
