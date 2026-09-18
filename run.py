"""
Convenience launcher for the Resume Screening System.

Usage:
    python3 run.py
"""

import sys
import subprocess
from pathlib import Path


def main():
    root_dir = Path(__file__).resolve().parent
    app_file = root_dir / "app.py"
    
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_file)] + sys.argv[1:]
    print("Starting Resume Screening System...")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nApplication stopped.")


if __name__ == "__main__":
    main()
