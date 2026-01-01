#!/usr/bin/env python3
"""
Create Desktop Shortcut for OpenLPT GUI
Works on Windows and macOS.
"""

import os
import sys
import platform
from pathlib import Path

def get_desktop_path():
    """Get the path to the user's desktop."""
    if platform.system() == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders")
            path_str = winreg.QueryValueEx(key, "Desktop")[0]
            return Path(path_str)
        except Exception:
            return Path(os.environ["USERPROFILE"]) / "Desktop"
    else:
        return Path.home() / "Desktop"

def create_windows_shortcut(target_script, icon_path):
    """Create a Windows .lnk shortcut using PowerShell."""
    # We will let PowerShell resolve the Desktop path dynamically to avoid unicode/path issues
    # passed from Python.
    
    python_exe = sys.executable
    target_path = Path(target_script).resolve()
    working_dir = target_path.parent.parent # Project root
    
    # Verify icon and ensure it's .ico for Windows
    if not isinstance(icon_path, Path):
        icon_path = Path(icon_path)
    
    # Try to get/convert to .ico
    icon_path = ensure_ico_for_windows(icon_path)
    
    icon_str = str(icon_path.resolve()) if icon_path.exists() else ""
    
    # PowerShell script content
    # We use WScript.Shell SpecialFolders to get the true Desktop path (handles OneDrive etc)
    ps_script = f"""
    $WshShell = New-Object -ComObject WScript.Shell
    $DesktopPath = $WshShell.SpecialFolders("Desktop")
    $ShortcutPath = Join-Path $DesktopPath "OpenLPT.lnk"
    
    if (Test-Path $ShortcutPath) {{
        Write-Host "Shortcut already exists at $ShortcutPath"
        exit 2
    }}
    
    $Shortcut = $WshShell.CreateShortcut($ShortcutPath)
    $Shortcut.TargetPath = "{python_exe}"
    $Shortcut.Arguments = '"{target_path}"'
    $Shortcut.WorkingDirectory = "{working_dir}"
    $Shortcut.Description = "OpenLPT 3D Particle Tracking"
    """
    
    if icon_str:
        ps_script += f'$Shortcut.IconLocation = "{icon_str}"\n'
        
    ps_script += """
    $Shortcut.Save()
    Write-Host "Shortcut created at $ShortcutPath"
    """
    
    ps_file = working_dir / "create_shortcut.ps1"
    try:
        # PowerShell likes UTF-8 with BOM
        with open(ps_file, "w", encoding="utf-8-sig") as f:
            f.write(ps_script)
        
        cmd = f'powershell -NoProfile -ExecutionPolicy Bypass -File "{ps_file}"'
        
        # We can't easily check for existence in Python since we don't know the exact path resolved by PS
        # So we trust the PS exit code.
        ret = os.system(cmd)
        
        if ret == 2:
             # Exit code 2 means shortcut exists (as we defined in PS script)
             return 2
        elif ret != 0:
            print(f"[Shortcut] PowerShell execution failed with code {ret}")
            # Check for non-ascii path
            if not str(desktop).isascii():
                return -2
            return -1
            
    except Exception as e:
        print(f"[Shortcut] Failed to run PowerShell script: {e}")
        return False
    finally:
        if ps_file.exists():
            os.remove(ps_file)
            
    # Return 1 for success
    return 1

def ensure_ico_for_windows(png_path):
    """
    Ensure an .ico file exists for Windows shortcut.
    If only .png exists, try to convert it using Pillow (if available).
    Returns path to .ico file or original path if conversion fails/unnecessary.
    """
    if not isinstance(png_path, Path):
        png_path = Path(png_path)
        
    if png_path.suffix.lower() == '.ico' and png_path.exists():
        return png_path
        
    ico_path = png_path.with_suffix('.ico')
    if ico_path.exists():
        return ico_path
        
    # Attempt conversion
    try:
        from PIL import Image
        img = Image.open(png_path)
        img.save(ico_path, format='ICO', sizes=[(256, 256)])
        print(f"[Shortcut] Converted icon to {ico_path}")
        return ico_path
    except ImportError:
        print("[Shortcut] Warning: PIL/Pillow not installed. Cannot convert icon to .ico for Windows.")
    except Exception as e:
        print(f"[Shortcut] Icon conversion failed: {e}")
        
    return png_path

def create_mac_shortcut(target_script, icon_path):
    """Create a macOS .command file (executable script)."""
    desktop = get_desktop_path()
    shortcut_path = desktop / "OpenLPT.command"
    
    if shortcut_path.exists():
        return False
        
    target_path = Path(target_script).resolve()
    working_dir = target_path.parent.parent
    
    script_content = f"""#!/bin/bash
cd "{working_dir}"
"{sys.executable}" "{target_path}"
"""
    
    try:
        with open(shortcut_path, "w") as f:
            f.write(script_content)
        # Make executable
        os.chmod(shortcut_path, 0o755)
    except Exception as e:
        print(f"Failed to create Mac shortcut: {e}")
        return -1
        
    return 1 if shortcut_path.exists() else -1

def check_and_create_shortcut():
    """
    Check if desktop shortcut exists. If not, create it.
    Returns: True if a new shortcut was created, False otherwise.
    """
    try:
        # Locate gui/main.py relative to this file
        current_dir = Path(__file__).parent
        target_script = current_dir / "main.py"
        
        if not target_script.exists():
            print(f"[Shortcut] Warning: Could not find main.py at {target_script}")
            return False

        # Locate Icon
        icon_path = current_dir / "assets" / "icon.png"
        
        # Create based on OS
        system = platform.system()
        if system == "Windows":
            return create_windows_shortcut(target_script, icon_path)
        elif system == "Darwin": # macOS
            return create_mac_shortcut(target_script, icon_path)
        else:
            return False
            
    except Exception as e:
        print(f"[Shortcut] Error: {e}")
        return -1

if __name__ == "__main__":
    if check_and_create_shortcut():
        print("Shortcut created successfully.")
    else:
        print("Shortcut already exists or failed to create.")
