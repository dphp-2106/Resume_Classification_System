"""
SAMATRIX RESUMEFORGE 2026 — Master Root Runner
Run from repository root:
    python run_pipeline.py         # Runs complete end-to-end pipeline
    python run_pipeline.py --app   # Launches interactive Streamlit demo
"""

import sys
import os
import subprocess

_ROOT = os.path.dirname(os.path.abspath(__file__))
_CLASSIFIER_DIR = os.path.join(_ROOT, 'resume_classifier')

if __name__ == '__main__':
    args = sys.argv[1:]
    if '--app' in args or '--demo' in args:
        print("[*] Launching Streamlit Web App...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], cwd=_CLASSIFIER_DIR)
    else:
        print("=" * 70)
        print("  SAMATRIX RESUMEFORGE 2026 — Master Execution")
        print("=" * 70)
        print(f"  Root: {_ROOT}")
        print(f"  Running end-to-end pipeline in resume_classifier/...")
        print()
        subprocess.run([sys.executable, "run_all.py"], cwd=_CLASSIFIER_DIR)
