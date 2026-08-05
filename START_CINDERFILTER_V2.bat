@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup-and-run-v2.ps1"
if errorlevel 1 pause
endlocal
