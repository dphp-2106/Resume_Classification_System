"""
SAMATRIX RESUMEFORGE 2026
run_all.py — Master Orchestration Script

Runs the entire pipeline end-to-end:
    Step 1: EDA
    Step 2: Preprocessing
    Step 3: Classical ML Training
    Step 4: Deep Learning (Word2Vec + MLP)
    Step 5: Error Analysis
    Step 6: Prediction test

Usage:
    python run_all.py

Then run the demo:
    streamlit run app.py
"""

import sys, io, os, subprocess, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Ensure working directory is always resume_classifier
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from config import OUTPUTS_DIR, MODELS_DIR, EDA_DIR, PLOTS_DIR, ERROR_DIR

STEPS = [
    ("01_EDA.py",           "EDA & Visualizations"),
    ("02_preprocessing.py", "Text Preprocessing (cache)"),
    ("03_classical_ml.py",  "Classical ML Training"),
    ("04_deep_learning.py", "Word2Vec + MLP Training"),
    ("05_error_analysis.py","Error Analysis"),
    ("06_predict.py",       "Final Prediction Test"),
]

os.makedirs('outputs', exist_ok=True)
os.makedirs('outputs/models', exist_ok=True)
os.makedirs('outputs/eda', exist_ok=True)
os.makedirs('outputs/plots', exist_ok=True)
os.makedirs('outputs/error_analysis', exist_ok=True)

print("=" * 70)
print("  SAMATRIX RESUMEFORGE 2026 — Full Pipeline Execution")
print("=" * 70)
print(f"  Running from: {os.getcwd()}")
print(f"  Python: {sys.version.split()[0]}")
print()

start_total = time.time()

for script, description in STEPS:
    print(f"\n{'─'*70}")
    print(f"  STEP: {description}")
    print(f"  Script: {script}")
    print(f"{'─'*70}")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script],
        capture_output=False,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"\n  [OK] Completed in {elapsed:.1f}s")
    else:
        print(f"\n  [ERROR] Step failed (exit code {result.returncode})")
        print(f"  Continuing with next step...")

total_time = time.time() - start_total
print(f"\n{'='*70}")
print(f"  PIPELINE COMPLETE in {total_time/60:.1f} minutes")
print(f"{'='*70}")
print()
print("  Output files generated:")
for root, dirs, files in os.walk('outputs'):
    level = root.replace('outputs', '').count(os.sep)
    indent = ' ' * 4 * (level + 1)
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 4 * (level + 2)
    for file in files:
        size = os.path.getsize(os.path.join(root, file))
        size_str = f"{size/1024:.0f}KB" if size > 1024 else f"{size}B"
        print(f"{subindent}{file}  ({size_str})")

print()
print("  To launch the demo app:")
print("    streamlit run app.py")
print()
