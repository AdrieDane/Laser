@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    py -3.12 -m venv .venv || py -3 -m venv .venv || goto :error
    .venv\Scripts\python.exe -m pip install --upgrade pip || goto :error
    .venv\Scripts\python.exe -m pip install -e . || goto :error
)

.venv\Scripts\python.exe -m streamlit run app\app.py
exit /b %errorlevel%

:error
echo LASER Standard Lipidomics could not be started.
pause
exit /b 1
