@echo off
echo =======================================================
echo    IndicFakeSpeech Deepfake Audio Detector           
echo =======================================================
echo.
echo Please ensure Python 3.11+ is installed.
echo Press Ctrl+C to stop the server at any time.
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in system PATH.
    pause
    exit /b
)

echo [INFO] Looking for dependencies...
python -m pip install -r requirements.txt >nul 2>&1

echo [INFO] Starting Flask Server...
python app.py

pause
