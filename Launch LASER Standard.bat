@echo off
setlocal
cd /d "%~dp0"

if not defined LASER_STANDARD_HOST set "LASER_STANDARD_HOST=0.0.0.0"
if not defined LASER_STANDARD_PORT set "LASER_STANDARD_PORT=8085"

if not exist ".venv\Scripts\python.exe" (
    py -3.14 -m venv .venv || py -3.13 -m venv .venv || py -3.12 -m venv .venv || py -3 -m venv .venv || goto :error
    .venv\Scripts\python.exe -m pip install --upgrade pip || goto :error
)

.venv\Scripts\python.exe -m pip install -e . || goto :error
.venv\Scripts\python.exe -m streamlit run app\app.py ^
    --server.address "%LASER_STANDARD_HOST%" ^
    --server.port "%LASER_STANDARD_PORT%" ^
    --server.headless true ^
    --browser.gatherUsageStats false
exit /b %errorlevel%

:error
echo LASER Standard Lipidomics could not be started.
pause
exit /b 1
