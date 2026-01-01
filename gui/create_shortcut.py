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
    """
    Create a macOS App Bundle (.app) instead of a simple .command file.
    This allows for a native icon and better UX.
    """
    desktop = get_desktop_path()
    app_name = "OpenLPT.app"
    app_path = desktop / app_name
    
    # If app bundle exists, verify or skip
    if app_path.exists():
        return 2 # Already exists code
        
    target_path = Path(target_script).resolve()
    # Assume target_path is .../gui/main.py, we want project root .../
    working_dir = target_path.parent.parent
    
    # App Bundle Structure
    contents_dir = app_path / "Contents"
    macos_dir = contents_dir / "MacOS"
    resources_dir = contents_dir / "Resources"
    
    try:
        os.makedirs(macos_dir, exist_ok=True)
        os.makedirs(resources_dir, exist_ok=True)
        
        # 1. Info.plist
        info_plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>OpenLPTLauncher</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundleIdentifier</key>
    <string>com.openlpt.gui</string>
    <key>CFBundleName</key>
    <string>OpenLPT</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>2.1.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.13</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>"""
        
        with open(contents_dir / "Info.plist", "w", encoding="utf-8") as f:
            f.write(info_plist)
            
        # 2. Launcher Script
        # We need to ensure we use the same python interpreter that is running this script
        # and set the working directory correctly.
        launcher_script = f"""#!/bin/bash
EXEC="{sys.executable}"
SCRIPT="{target_path}"
DIR="{working_dir}"

cd "$DIR"
"$EXEC" "$SCRIPT"
"""
        launcher_path = macos_dir / "OpenLPTLauncher"
        with open(launcher_path, "w", encoding="utf-8") as f:
            f.write(launcher_script)
        
        # Make executable
        os.chmod(launcher_path, 0o755)
        
        # 3. Handle Icon (png -> icns)
        # We try to use standard macOS tools (sips, iconutil) to create .icns
        if icon_path.exists():
            try:
                # Create a temporary iconset directory
                iconset_dir = resources_dir / "AppIcon.iconset"
                os.makedirs(iconset_dir, exist_ok=True)
                
                # We need specific sizes for iconutil
                # 16, 32, 64, 128, 256, 512, 1024 (and 2x versions)
                # For simplicity, we'll generate a few key sizes.
                sizes = [16, 32, 64, 128, 256, 512, 1024]
                
                for s in sizes:
                    out_name = f"icon_{s}x{s}.png"
                    out_path = iconset_dir / out_name
                    # sips -z H W input --out output
                    ret = os.system(f'sips -z {s} {s} "{icon_path}" --out "{out_path}" > /dev/null 2>&1')
                    
                    if s > 512: continue # Skip 2x for smaller ones to save time, or do 2x logic if needed
                    
                    # 2x version
                    out_name_2x = f"icon_{s}x{s}@2x.png"
                    out_path_2x = iconset_dir / out_name_2x
                    s2 = s * 2
                    os.system(f'sips -z {s2} {s2} "{icon_path}" --out "{out_path_2x}" > /dev/null 2>&1')

                # Convert iconset to icns
                icns_path = resources_dir / "AppIcon.icns"
                ret = os.system(f'iconutil -c icns "{iconset_dir}" -o "{icns_path}" > /dev/null 2>&1')
                
                # Cleanup iconset
                import shutil
                shutil.rmtree(iconset_dir, ignore_errors=True)
                
                if ret != 0:
                     print("[Shortcut] Warning: Failed to convert icon using iconutil.")
                     
            except Exception as e:
                print(f"[Shortcut] Icon generation failed: {e}")
                
    except Exception as e:
        print(f"Failed to create Mac App Bundle: {e}")
        return -1
        
    return 1 if app_path.exists() else -1

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
