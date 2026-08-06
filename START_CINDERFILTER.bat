@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup-and-run.ps1"
if errorlevel 1 (
    echo.
    echo CinderFilter could not start. Read the error above.
    pause
)
endlocal
