@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup-and-run-v2-pitch.ps1"
if errorlevel 1 pause
endlocal
