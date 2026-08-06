@echo off
setlocal
cd /d "%~dp0"
title CinderFilter CUDA Main Noise Engine Installer
rem Do not pass %%~dp0 as AppRoot: its trailing backslash can corrupt quoted arguments.
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0install-cuda-noise-engine.ps1" -ParentPid 0
if errorlevel 1 (
    echo.
    echo CUDA main noise-engine installation failed. Read the error above.
    pause
)
endlocal
