@echo off
setlocal
cd /d "%~dp0"
title CinderFilter CUDA Noise Engine Repair
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0repair-cuda-noise-engine.ps1"
if errorlevel 1 (
    echo.
    echo CUDA noise-engine repair failed. Read the error above.
    pause
)
endlocal
