@echo off
REM Quick start script for Windows

echo ========================================
echo Urban Issue Detection - Quick Start
echo ========================================
echo.

REM Check if .env exists
if not exist ".env" (
    echo [INFO] Creating .env file from template...
    copy env.template .env
    echo.
    echo [INFO] Edit .env and add your NGROK_AUTHTOKEN for remote access
    echo        Get token at: https://dashboard.ngrok.com/get-started/your-authtoken
    echo.
    pause
)

echo [INFO] Starting services...
echo.

REM Start Docker Compose
docker-compose up

REM If user pressed Ctrl+C, clean up
echo.
echo [INFO] Stopping services...
docker-compose down
