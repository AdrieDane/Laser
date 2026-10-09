@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    py -3.12 -m venv .venv || py -3 -m venv .venv || goto :error
)
.venv\Scripts\python.exe -m pip install --upgrade pip || goto :error
.venv\Scripts\python.exe -m pip install -e . || goto :error
echo Setup complete. Run "Launch LASER Standard.bat".
exit /b 0

:error
echo Setup failed. Install Python 3.10-3.13 and try again.
pause
exit /b 1
