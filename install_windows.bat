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
pause

:: --- Visual Studio Build Tools Check ---
echo.
echo [0.5/4] Checking for Visual Studio Build Tools...
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "HAS_VS="

if exist "%VSWHERE%" (
    for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Workload.NativeDesktop -property installationPath`) do (
        set "HAS_VS=%%i"
    )
)

if defined HAS_VS (
    echo [OK] Visual Studio C++ Tools found at: "%HAS_VS%"
) else (
    echo.
    echo [ERROR] Visual Studio Build Tools with "Desktop development with C++" not found!
    echo         This is REQUIRED to compile the OpenLPT C++ extensions.
    echo.
    
    :: Check for Winget
    where winget ^>nul 2^>nul
    if not errorlevel 1 (
        echo [INFO] Winget found. Attempting AUTO-INSTALLATION/UPDATE...
        echo        ^(This will open a prompt asking for permission^)
        echo.
        echo Running: winget install Microsoft.VisualStudio.2022.BuildTools...
        
        winget install --id Microsoft.VisualStudio.2022.BuildTools --exact --scope machine --override "--add Microsoft.VisualStudio.Workload.NativeDesktop --includeRecommended --passive --norestart"
        
        echo.
        echo [INFO] Winget finished. Re-checking installation...
        
        :: RE-CHECK: Did it actually work?
        set "HAS_VS_RECHECK="
        if exist "%VSWHERE%" (
            for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Workload.NativeDesktop -property installationPath`) do (
                set "HAS_VS_RECHECK=%%i"
            )
        )
        
        if defined HAS_VS_RECHECK (
            echo [SUCCESS] Visual Studio Build Tools verified. Proceeding...
            goto :EndVSCheck
        ) else (
            echo.
            echo [WARNING] Automatic setup finished, but C++ Tools are STILL missing.
            
            :: LAST RESORT: Check if we can just "Modify" the existing installation directly
            :: (Winget sometimes refuses to update if the base package is present)
            set "VS_PARTIAL_PATH="
            if exist "%VSWHERE%" (
                 for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -property installationPath`) do (
                    set "VS_PARTIAL_PATH=%%i"
                )
            )
            
            if defined VS_PARTIAL_PATH (
                echo [INFO] Found existing VS installation at: "%VS_PARTIAL_PATH%"
                echo        Attempting to forcefully ADD the C++ workload via vs_installer.exe...
                echo.
                
                :: Try to find the installer executable (standard location)
                if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vs_installer.exe" (
                    start /wait "" "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vs_installer.exe" modify --installPath "%VS_PARTIAL_PATH%" --add Microsoft.VisualStudio.Workload.NativeDesktop --passive --norestart
                    
                    :: One final check...
                    if exist "%VSWHERE%" (
                        for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Workload.NativeDesktop -property installationPath`) do (
                           echo [SUCCESS] Workload added successfully!
                           goto :EndVSCheck
                        )
                    )
                )
            )
            
            echo.
            echo [ERROR] Could not auto-repair. Please fix manually.
            echo           ^>^>^> PLEASE FIX MANUALLY ^<^<^<
            echo.
            goto :ManualInstallVS
        )
    ) else (
        goto :ManualInstallVS
    )
)
goto :EndVSCheck

:ManualInstallVS
echo.
echo ================================================================
echo                   MANUAL ACTION REQUIRED
echo ================================================================
echo Please download and install "Visual Studio Build Tools":
echo URL: https://visualstudio.microsoft.com/visual-cpp-build-tools/
echo.
echo 1. Run the installer.
echo 2. Check "Desktop development with C++" workload.
echo 3. Click Keep/Install.
echo 4. After installation, RUN THIS SCRIPT AGAIN.
echo ================================================================
pause
exit /b 1

:EndVSCheck
:: ---------------------------------------

cd /d "%~dp0"

echo.
where conda >nul 2>nul
if %errorlevel% neq 0 (
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
if %errorlevel% neq 0 (
    echo [Warning] Environment might already exist or conda failed. Trying to proceed...
)

echo.
echo [2/3] Activating Environment...
call conda activate OpenLPT
if %errorlevel% neq 0 (
    echo [Error] Failed to activate 'OpenLPT' environment.
    echo Please make sure you have initialized conda for your shell (conda init cmd.exe).
    pause
    exit /b %errorlevel%
)

echo.
echo [3/3] Installing Dependencies...
:: Try mamba first, fallback to conda if missing (optional logic, sticking to user pref for mamba)
call mamba install -c conda-forge --file requirements.txt -y
if %errorlevel% neq 0 (
    echo [Error] Mamba install failed.
    echo NOTE: This project requires mamba. Please install miniforge or mambaforge.
    pause
    exit /b %errorlevel%
)

echo.
echo [4/4] Installing OpenLPT...
pip install . --no-build-isolation
if %errorlevel% neq 0 (
    echo [Error] Pip install failed.
    pause
    exit /b %errorlevel%
)

echo.
echo ==========================================
echo       Installation Complete!
echo       Launching OpenLPT GUI...
echo ==========================================
python GUI.py
pause
