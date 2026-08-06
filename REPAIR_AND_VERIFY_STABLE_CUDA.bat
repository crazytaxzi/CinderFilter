@echo off
setlocal
cd /d "%~dp0"
title CinderFilter Lookahead-Safe CUDA Repair
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0REPAIR_AND_VERIFY_STABLE_CUDA.ps1"
if errorlevel 1 (
    echo.
    echo Lookahead-safe CUDA repair or verification failed. Read the exact error above.
    pause
)
endlocal
