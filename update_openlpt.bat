@echo off
title OpenLPT Updater
echo ==========================================
echo       OpenLPT Auto-Updater
echo ==========================================
echo.
cd /d "D:\0.Code\OpenLPTGUI\OpenLPT"

echo [0/4] Restoring environment variables...
set "GIT_ASKPASS=c:\Users\tan_s\AppData\Local\Programs\Antigravity\resources\app\extensions\git\dist\askpass.sh"

echo.
echo NOTE: If the process pauses below, please type your password (SSH passphrase or Git account password) and press Enter.
echo.
echo [1/4] Pulling latest code from git...
git pull
if %errorlevel% neq 0 (
    echo [Error] Git pull failed.
    pause
    exit /b %errorlevel%
)

echo.
echo [2/4] Activating Conda Environment 'OpenLPT'...
call conda activate OpenLPT
if %errorlevel% neq 0 (
    echo [Warning] Failed to activate 'OpenLPT'. Trying to proceed with current env...
)

echo.
echo [3/4] Update dependencies with Mamba...
call mamba install -c conda-forge --file requirements.txt -y
if %errorlevel% neq 0 (
    echo [Error] Mamba install failed.
    pause
    exit /b %errorlevel%
)

echo.
echo [4/4] Re-installing OpenLPT package...
pip install . --no-build-isolation
if %errorlevel% neq 0 (
    echo [Error] Pip install failed.
    pause
    exit /b %errorlevel%
)

echo.
echo ==========================================
echo       Update Successful! 
echo ==========================================
echo Please restart OpenLPT manually.
pause
