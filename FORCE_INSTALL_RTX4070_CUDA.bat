@echo off
setlocal
cd /d "%~dp0"
title CinderFilter RTX 4070 CUDA Force Installer
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0force-install-rtx4070-cuda.ps1"
if errorlevel 1 (
    echo.
    echo CUDA installation failed. Read the error above.
    pause
)
endlocal
