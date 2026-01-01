import sys
import os
import platform
import subprocess
from pathlib import Path

def run_auto_update(project_root: Path):
    """
    Generate an update script and execute it in a new terminal window.
    Then exit the current application.
    """
    system = platform.system()
    project_root = project_root.resolve()
    
    if system == "Windows":
        _run_windows_update(project_root)
    elif system == "Darwin":
        _run_mac_update(project_root)
    else:
        print(f"[AutoUpdate] Check not implemented for {system}")

def _run_windows_update(root: Path):
    """
    Create batch file and run it.
    """
    script_path = root / "update_openlpt.bat"
    
    # Capture relevant environment variables to preserve SSH/Git auth
    # "start" command spawns a fresh cmd which might lose session env vars (like SSH_AUTH_SOCK)
    env_setup = []
    for k, v in os.environ.items():
        if k.startswith(('SSH_', 'GIT_')):
            env_setup.append(f'set "{k}={v}"')
            
    env_block = "\n".join(env_setup)
    
    # Batch script content
    # Note: 'call conda activate' is required for batch files
    content = f"""@echo off
title OpenLPT Updater
echo ==========================================
echo       OpenLPT Auto-Updater
echo ==========================================
echo.
cd /d "{root}"

echo [0/4] Restoring environment variables...
{env_block}

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
"""
    try:
        with open(script_path, "w") as f:
            f.write(content)
            
        print(f"[AutoUpdate] Created script at {script_path}")
        
        # Execute in new window and exit app
        # start "Title" "script"
        os.system(f'start "OpenLPT Updater" "{script_path}"')
        
        print("[AutoUpdate] Exiting application to allow update...")
        sys.exit(0)
        
    except Exception as e:
        print(f"[AutoUpdate] Failed to start update: {e}")

def _run_mac_update(root: Path):
    """
    Create shell script and run it via Terminal.app
    """
    script_path = root / "update_openlpt.command"
    
    content = f"""#!/bin/bash
echo "=========================================="
echo "      OpenLPT Auto-Updater"
echo "=========================================="
echo ""
cd "{root}"

echo ""
echo "NOTE: If the process pauses below, please type your password (SSH passphrase or Git account password) and press Enter."
echo ""
echo "[1/4] Pulling latest code from git..."
git pull
if [ $? -ne 0 ]; then
    echo "[Error] Git pull failed."
    read -p "Press enter to exit"
    exit 1
fi

echo ""
echo "[2/4] Activating Conda Environment 'OpenLPT'..."
# Try to find conda hook
CONDA_BASE=$(conda info --base)
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate OpenLPT
else
    echo "[Warning] Could not find conda.sh. Assuming environment is correct or manual activation needed."
fi

echo ""
echo "[3/4] Update dependencies with Mamba..."
mamba install -c conda-forge --file requirements.txt -y
if [ $? -ne 0 ]; then
    echo "[Error] Mamba install failed."
    read -p "Press enter to exit"
    exit 1
fi

echo ""
echo "[4/4] Re-installing OpenLPT package..."
pip install . --no-build-isolation
if [ $? -ne 0 ]; then
    echo "[Error] Pip install failed."
    read -p "Press enter to exit"
    exit 1
fi

echo ""
echo "=========================================="
echo "      Update Successful!"
echo "=========================================="
echo "Please restart OpenLPT manually."
read -p "Press enter to close..."
"""
    try:
        with open(script_path, "w") as f:
            f.write(content)
        
        # Make executable
        os.chmod(script_path, 0o755)
        
        print(f"[AutoUpdate] Created script at {script_path}")
        
        # Open in Terminal
        subprocess.call(["open", str(script_path)])
        
        print("[AutoUpdate] Exiting application to allow update...")
        sys.exit(0)
        
    except Exception as e:
        print(f"[AutoUpdate] Failed to start update: {e}")
