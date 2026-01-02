@echo off
title OpenLPT Installer
echo ==========================================
echo       OpenLPT One-Click Installer
echo ==========================================
echo.
echo This script will:
echo 1. Create a Conda environment named 'OpenLPT'
echo 2. Install all dependencies
echo 3. Install the OpenLPT package
echo.


:: Using clang compiler from conda-forge, no Visual Studio needed!

cd /d "%~dp0"

echo.
where conda >nul 2>nul
if errorlevel 1 (
    echo [!] Conda/Mamba not found.
    echo [!] Should I automatically download and install Miniforge3?
    echo     (Includes Conda + Mamba, installs to %UserProfile%\Miniforge3)
    echo.
    pause
    
    echo.
    echo [0/3] Downloading Miniforge3 Installer...
    :: Curl is available on Win 10 1803+
    curl -L -o miniforge_installer.exe "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe"
    
    echo Installing Miniforge3 (Active user only, Silent)...
    start /wait "" miniforge_installer.exe /InstallationType=JustMe /RegisterPython=0 /S /D=%UserProfile%\Miniforge3
    del miniforge_installer.exe
    
    echo Initializing Conda...
    call "%UserProfile%\Miniforge3\Scripts\activate.bat"
    call conda init cmd.exe
    
    echo Miniforge3 installed!
)

echo.
echo [1/3] Creating Conda Environment 'OpenLPT'...
call conda create -n OpenLPT python=3.10 -y
if errorlevel 1 (
    echo [Warning] Environment might already exist or conda failed. Trying to proceed...
)

echo.
echo [2/3] Activating Environment...
call conda activate OpenLPT
if errorlevel 1 (
    echo [Error] Failed to activate 'OpenLPT' environment.
    echo Please make sure you have initialized conda for your shell ^(conda init cmd.exe^).
    pause
    exit /b 1
)

echo.
echo [3/3] Installing Dependencies...
:: Try mamba first, fallback to conda if missing (optional logic, sticking to user pref for mamba)
call mamba install -c conda-forge --file requirements.txt -y
if errorlevel 1 (
    echo [Error] Mamba install failed.
    echo NOTE: This project requires mamba. Please install miniforge or mambaforge.
    pause
    exit /b 1
)

echo.
echo [3.5/4] Installing Windows C++ Compiler (MinGW-w64)...
call mamba install -c conda-forge m2w64-toolchain -y
if errorlevel 1 (
    echo [Warning] Failed to install MinGW compiler. Build might fail.
)

echo.
echo [4/4] Installing OpenLPT...



:: Fresh terminal will handle VS environment detection automatically

:: Create a helper script to run pip install in a fresh environment
set "PIP_SCRIPT=%TEMP%\openlpt_pip_install_%RANDOM%.bat"
echo @echo off > "%PIP_SCRIPT%"
echo cd /d "%~dp0" >> "%PIP_SCRIPT%"
echo call conda activate OpenLPT >> "%PIP_SCRIPT%"
echo set "CMAKE_GENERATOR=Ninja" >> "%PIP_SCRIPT%"
echo set "CMAKE_GENERATOR_INSTANCE=" >> "%PIP_SCRIPT%"
echo set "CMAKE_GENERATOR_PLATFORM=" >> "%PIP_SCRIPT%"
echo set "CMAKE_GENERATOR_TOOLSET=" >> "%PIP_SCRIPT%"
echo set "CC=gcc" >> "%PIP_SCRIPT%"
echo set "CXX=g++" >> "%PIP_SCRIPT%"
echo echo. >> "%PIP_SCRIPT%"
echo echo [INFO] Running pip install in fresh environment... >> "%PIP_SCRIPT%"
echo if exist build rmdir /s /q build >> "%PIP_SCRIPT%"
echo if exist openlpt.egg-info rmdir /s /q openlpt.egg-info >> "%PIP_SCRIPT%"
echo pip install . --no-build-isolation >> "%PIP_SCRIPT%"
echo if errorlevel 1 ( >> "%PIP_SCRIPT%"
echo     echo [Error] Pip install failed. >> "%PIP_SCRIPT%"
echo     pause >> "%PIP_SCRIPT%"
echo     exit /b 1 >> "%PIP_SCRIPT%"
echo ) >> "%PIP_SCRIPT%"
echo echo. >> "%PIP_SCRIPT%"
echo echo ========================================== >> "%PIP_SCRIPT%"
echo echo       Installation Complete! >> "%PIP_SCRIPT%"
echo echo       Launching OpenLPT GUI... >> "%PIP_SCRIPT%"
echo echo ========================================== >> "%PIP_SCRIPT%"
echo python GUI.py >> "%PIP_SCRIPT%"
echo del "%%~f0" >> "%PIP_SCRIPT%"

echo.
echo [INFO] Spawning fresh terminal for compilation...
echo       (This ensures a clean environment with latest VS paths)
echo.
start "" cmd /c "%PIP_SCRIPT%"

echo [INFO] Installation continues in the new window.
echo       You can close this window.
exit /b 0

echo.
echo ==========================================
echo       Installation Complete!
echo       Launching OpenLPT GUI...
echo ==========================================
python GUI.py
