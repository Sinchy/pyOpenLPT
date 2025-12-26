"""
Update Checker for OpenLPT GUI

Checks GitHub releases for new versions and notifies users.
"""

import threading
from typing import Optional, Dict, Any

# Version info - read from central version file
try:
    import sys
    from pathlib import Path
    # Add project root to path if needed
    root_dir = Path(__file__).resolve().parent.parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    from _version import __version__ as CURRENT_VERSION
except ImportError:
    CURRENT_VERSION = "1.0.0"  # Fallback

GITHUB_REPO = "Sinchy/pyOpenLPT"


def parse_version(version_str: str) -> tuple:
    """Parse version string like '1.2.3' or 'v1.2.3' to tuple (1, 2, 3)."""
    version_str = version_str.lstrip('v').strip()
    try:
        parts = version_str.split('.')
        return tuple(int(p) for p in parts[:3])
    except (ValueError, AttributeError):
        return (0, 0, 0)


def compare_versions(current: str, latest: str) -> bool:
    """Return True if latest > current."""
    return parse_version(latest) > parse_version(current)


def check_for_updates() -> Dict[str, Any]:
    """
    Check GitHub for newer releases.
    
    Returns:
        dict with keys:
            - available: bool, True if update available
            - current: str, current version
            - latest: str, latest version (if available)
            - url: str, URL to release page (if available)
            - notes: str, release notes excerpt (if available)
    """
    result = {
        "available": False,
        "current": CURRENT_VERSION,
        "latest": None,
        "url": None,
        "notes": None,
        "error": None
    }
    
    try:
        import requests
        
        url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
        response = requests.get(url, timeout=5, headers={
            "Accept": "application/vnd.github.v3+json"
        })
        
        if response.status_code == 200:
            data = response.json()
            latest_version = data.get("tag_name", "").lstrip("v")
            
            if latest_version and compare_versions(CURRENT_VERSION, latest_version):
                result["available"] = True
                result["latest"] = latest_version
                result["url"] = data.get("html_url", f"https://github.com/{GITHUB_REPO}/releases")
                
                # Get first 500 chars of release notes
                notes = data.get("body", "")
                if notes and len(notes) > 500:
                    notes = notes[:500] + "..."
                result["notes"] = notes
                
        elif response.status_code == 404:
            # No releases yet
            pass
        else:
            result["error"] = f"GitHub API returned status {response.status_code}"
            
    except ImportError:
        result["error"] = "requests module not installed"
    except Exception as e:
        result["error"] = str(e)
    
    return result


def check_for_updates_async(callback: callable) -> None:
    """
    Check for updates in background thread.
    
    Args:
        callback: Function to call with result dict when done
    """
    def _worker():
        result = check_for_updates()
        callback(result)
    
    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()


def get_current_version() -> str:
    """Return current version string."""
    return CURRENT_VERSION
