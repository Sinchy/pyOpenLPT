@echo off
title OpenLPT Installer
echo ==========================================
echo       OpenLPT One-Click Installer
echo ==========================================
echo.
echo This script will:
echo 1. Install Visual Studio Build Tools (if needed)
echo 2. Create a Conda environment named 'OpenLPT'
echo 3. Install all dependencies
echo 4. Install the OpenLPT package
echo.

cd /d "%~dp0"

:: ============================================
:: STEP 0: Visual Studio Build Tools Check
:: ============================================

echo.
echo [0/4] Checking for Visual Studio Build Tools...

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VS_INSTALLER=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vs_installer.exe"
set "HAS_VS="
set "VCVARS="

:: Check if vswhere exists
if not exist "%VSWHERE%" (
    echo [INFO] Visual Studio Installer not found. Will install Build Tools...
    goto :InstallVS
)

:: Check for proper C++ tools installation
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2^>nul`) do (
    set "HAS_VS=%%i"
)

if defined HAS_VS (
    :: Verify vcvarsall.bat exists
    if exist "%HAS_VS%\VC\Auxiliary\Build\vcvarsall.bat" (
        echo [OK] Visual Studio C++ Tools found at: "%HAS_VS%"
        set "VCVARS=%HAS_VS%\VC\Auxiliary\Build\vcvarsall.bat"
        goto :VSCheckDone
    )
)

echo [WARNING] Visual Studio found but C++ tools incomplete.
goto :InstallVS

:: ============================================
:: Install Visual Studio Build Tools
:: ============================================
:InstallVS
echo.
echo ================================================================
echo         Installing Visual Studio Build Tools 2022
echo ================================================================
echo.

:: Download VS Build Tools installer
echo [INFO] Downloading Visual Studio Build Tools installer...
set "VS_INSTALLER_URL=https://aka.ms/vs/17/release/vs_buildtools.exe"
set "VS_INSTALLER_PATH=%TEMP%\vs_buildtools.exe"

curl -L -o "%VS_INSTALLER_PATH%" "%VS_INSTALLER_URL%"
if errorlevel 1 (
    echo [ERROR] Failed to download VS Build Tools installer.
    echo Please download manually from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
    pause
    exit /b 1
)

echo.
echo [INFO] Installing Visual Studio Build Tools with C++ components...
echo        This may take 5-15 minutes. Please wait...
echo.
echo        Components being installed:
echo        - MSVC v143 C++ x64/x86 build tools
echo        - Windows 11 SDK
echo        - C++ CMake tools
echo.

:: Install with required components
"%VS_INSTALLER_PATH%" --wait --passive --norestart ^
    --add Microsoft.VisualStudio.Workload.VCTools ^
    --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 ^
    --add Microsoft.VisualStudio.Component.Windows11SDK.22621 ^
    --add Microsoft.VisualStudio.Component.VC.CMake.Project ^
    --includeRecommended

if errorlevel 1 (
    echo [WARNING] VS installer returned an error, but installation may have succeeded.
    echo           Continuing to verify...
)

:: Clean up installer
del "%VS_INSTALLER_PATH%" 2>nul

:: Re-check after installation
echo.
echo [INFO] Verifying installation...
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"

if not exist "%VSWHERE%" (
    echo [ERROR] VS installation failed. vswhere not found.
    pause
    exit /b 1
)

for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2^>nul`) do (
    set "HAS_VS=%%i"
)

if not defined HAS_VS (
    echo [ERROR] Visual Studio Build Tools installation failed.
    echo Please install manually from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
    echo Make sure to select "Desktop development with C++" workload.
    pause
    exit /b 1
)

if exist "%HAS_VS%\VC\Auxiliary\Build\vcvarsall.bat" (
    echo [SUCCESS] Visual Studio Build Tools installed successfully!
    set "VCVARS=%HAS_VS%\VC\Auxiliary\Build\vcvarsall.bat"
) else (
    echo [ERROR] vcvarsall.bat not found after installation.
    pause
    exit /b 1
)

:VSCheckDone
echo.

:: ============================================
:: STEP 1: Check/Install Conda
:: ============================================

echo [1/4] Checking for Conda/Mamba...
where mamba >nul 2>nul
if errorlevel 1 (
    where conda >nul 2>nul
    if errorlevel 1 (
        echo [INFO] Conda not found. Installing Miniforge3...
        
        curl -L -o miniforge_installer.exe "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe"
        
        echo Installing Miniforge3...
        start /wait "" miniforge_installer.exe /InstallationType=JustMe /RegisterPython=0 /S /D=%UserProfile%\Miniforge3
        del miniforge_installer.exe
        
        echo Initializing Conda...
        call "%UserProfile%\Miniforge3\Scripts\activate.bat"
        call conda init cmd.exe
        
        echo [SUCCESS] Miniforge3 installed!
    )
)

:: ============================================
:: STEP 2: Create Conda Environment
:: ============================================

echo.
echo [2/4] Creating Conda Environment 'OpenLPT'...
call conda create -n OpenLPT python=3.10 -y
if errorlevel 1 (
    echo [Warning] Environment might already exist. Continuing...
)

echo.
echo [3/4] Activating Environment and Installing Dependencies...
call conda activate OpenLPT
if errorlevel 1 (
    echo [Error] Failed to activate 'OpenLPT' environment.
    pause
    exit /b 1
)

:: Install dependencies
call mamba install -c conda-forge --file requirements.txt -y
if errorlevel 1 (
    echo [Error] Mamba install failed.
    pause
    exit /b 1
)

:: ============================================
:: STEP 4: Build and Install OpenLPT
:: ============================================

echo.
echo [4/4] Building and Installing OpenLPT...
echo.

:: Activate Visual Studio environment
echo [INFO] Activating Visual Studio environment...
call "%VCVARS%" x64
if errorlevel 1 (
    echo [WARNING] vcvarsall.bat returned an error, but continuing...
)

:: Set CMake to use NMake (works reliably with MSVC)
set "CMAKE_GENERATOR=NMake Makefiles"
set "CMAKE_GENERATOR_INSTANCE="
set "CMAKE_GENERATOR_PLATFORM="
set "CMAKE_GENERATOR_TOOLSET="
set "CMAKE_BUILD_TYPE=Release"

:: Clean previous build
if exist build rmdir /s /q build
if exist openlpt.egg-info rmdir /s /q openlpt.egg-info

echo.
echo [INFO] Running pip install...
pip install . --no-build-isolation
if errorlevel 1 (
    echo.
    echo [Error] Pip install failed.
    pause
    exit /b 1
)

echo.
echo ==========================================
echo       Installation Complete!
echo ==========================================
echo.
echo To use OpenLPT:
echo   1. Open a new terminal
echo   2. Run: conda activate OpenLPT
echo   3. Run: python GUI.py
echo.
echo Launching OpenLPT GUI...
python GUI.py

pause
