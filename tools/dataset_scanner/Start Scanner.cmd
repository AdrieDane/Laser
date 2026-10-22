@echo off
setlocal
pushd "%~dp0"
if not exist "runtime\python.exe" (
  echo The portable runtime is missing. Extract the COMPLETE LASER_Dataset_Scanner_Windows.zip first.
  echo Do not open this launcher from inside the ZIP file.
  pause
  popd
  exit /b 1
)
"runtime\python.exe" -I "%~dp0scanner.py"
if errorlevel 1 (
  echo.
  echo Scanner stopped with an error. Keep the message above for troubleshooting.
  pause
)
popd
endlocal
