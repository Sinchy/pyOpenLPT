"""
OpenLPT GUI - Main Entry Point
"""

import sys
from pathlib import Path

# Add gui folder to path so that relative imports in app.py work
gui_path = Path(__file__).parent / "gui"
sys.path.insert(0, str(gui_path))

from gui.app import main

if __name__ == "__main__":
    main()
