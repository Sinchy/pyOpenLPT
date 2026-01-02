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
    """Create a Windows .lnk shortcut using win32com (preferred) or PowerShell fallback."""
    
    python_exe = sys.executable
    target_path = Path(target_script).resolve()
    working_dir = target_path.parent.parent  # Project root
    
    # Verify icon and ensure it's .ico for Windows
    if not isinstance(icon_path, Path):
        icon_path = Path(icon_path)
    
    # Try to get/convert to .ico
    icon_path = ensure_ico_for_windows(icon_path)
    icon_str = str(icon_path.resolve()) if icon_path.exists() else ""
    
    # Get desktop path
    desktop = get_desktop_path()
    shortcut_path = desktop / "OpenLPT.lnk"
    
    if shortcut_path.exists():
        print(f"Shortcut already exists at {shortcut_path}")
        return 2
    
    # Try win32com first (best Unicode support)
    try:
        import win32com.client
        shell = win32com.client.Dispatch("WScript.Shell")
        shortcut = shell.CreateShortcut(str(shortcut_path))
        shortcut.TargetPath = python_exe
        shortcut.Arguments = f'"{target_path}"'
        shortcut.WorkingDirectory = str(working_dir)
        shortcut.Description = "OpenLPT 3D Particle Tracking"
        if icon_str:
            shortcut.IconLocation = icon_str
        shortcut.Save()
        print(f"Shortcut created at {shortcut_path}")
        return 1
    except ImportError:
        print("[Shortcut] win32com not available, trying PowerShell fallback...")
    except Exception as e:
        print(f"[Shortcut] win32com failed: {e}, trying PowerShell fallback...")
    
    # PowerShell fallback with UTF-8 BOM and proper encoding
    ps_script = f'''
# Ensure UTF-8 encoding
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$ShortcutPath = "{shortcut_path}"

if (Test-Path $ShortcutPath) {{
    Write-Host "Shortcut already exists at $ShortcutPath"
    exit 2
}}

$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = "{python_exe}"
$Shortcut.Arguments = '"{target_path}"'
$Shortcut.WorkingDirectory = "{working_dir}"
$Shortcut.Description = "OpenLPT 3D Particle Tracking"
'''
    
    if icon_str:
        ps_script += f'$Shortcut.IconLocation = "{icon_str}"\n'
        
    ps_script += '''
$Shortcut.Save()
Write-Host "Shortcut created at $ShortcutPath"
'''
    
    ps_file = working_dir / "create_shortcut.ps1"
    try:
        # Write with UTF-8 BOM for PowerShell
        with open(ps_file, "w", encoding="utf-8-sig") as f:
            f.write(ps_script)
        
        # Run PowerShell with UTF-8 code page
        cmd = f'chcp 65001 >nul && powershell -NoProfile -ExecutionPolicy Bypass -File "{ps_file}"'
        ret = os.system(cmd)
        
        if ret == 2:
            return 2
        elif ret != 0:
            print(f"[Shortcut] PowerShell execution failed with code {ret}")
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
                
                # 16, 32, 128, 256, 512
                # Apple strictly defines the required iconset names
                icon_definitions = [
                    (16, "icon_16x16.png"),
                    (32, "icon_16x16@2x.png"),
                    (32, "icon_32x32.png"),
                    (64, "icon_32x32@2x.png"),
                    (128, "icon_128x128.png"),
                    (256, "icon_128x128@2x.png"),
                    (256, "icon_256x256.png"),
                    (512, "icon_256x256@2x.png"),
                    (512, "icon_512x512.png"),
                    (1024, "icon_512x512@2x.png")
                ]
                
                # Try using PIL first for better transparency handling
                has_pil = False
                try:
                    from PIL import Image
                    src_img = Image.open(icon_path)
                    has_pil = True
                except ImportError:
                    pass

                for size, name in icon_definitions:
                    out_path = iconset_dir / name
                    
                    if has_pil:
                        # Resize with ANTIALIAS/LANCZOS and save preserving alpha
                        try:
                            # Use LANCZOS if available (Pillow 2.7+), else ANTIALIAS
                            resample = getattr(Image, 'Resampling', Image).LANCZOS
                            resized = src_img.resize((size, size), resample=resample)
                            resized.save(out_path, format="PNG")
                            continue
                        except Exception as e:
                            print(f"[Shortcut] PIL resize failed for {name}: {e}, falling back to sips")
                    
                    # Fallback to sips if PIL missing or failed
                    os.system(f'sips -z {size} {size} "{icon_path}" --out "{out_path}" > /dev/null 2>&1')

                # Convert iconset to icns
                icns_path = resources_dir / "AppIcon.icns"
                ret = os.system(f'iconutil -c icns "{iconset_dir}" -o "{icns_path}"')
                
                # Cleanup iconset
                import shutil
                shutil.rmtree(iconset_dir, ignore_errors=True)
                
                if ret != 0:
                     print("[Shortcut] Warning: iconutil failed. Trying fallback to simple PNG copy.")
                     # Fallback: Copy PNG as AppIcon.png (some macOS versions support this via plist)
                     # We already set CFBundleIconFile to AppIcon, so AppIcon.png might work.
                     import shutil
                     shutil.copy(icon_path, resources_dir / "AppIcon.png")
                     
            except Exception as e:
                print(f"[Shortcut] Icon generation failed: {e}")
                
    except Exception as e:
        print(f"Failed to create Mac App Bundle: {e}")
        return -1
    
    # Force Finder to verify the new app bundle (clears icon cache)
    if app_path.exists():
        os.system(f'touch "{app_path}"')
        
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
