"""Run the complete SAMATRIX RESUMEFORGE 2026 pipeline in order."""
import subprocess
import sys

STAGES = [
    ("Data quality checks", "src.data_loading"),
    ("EDA", "src.eda"),
    ("TF-IDF classical baselines", "src.train_classical"),
    ("Word2Vec + Dense MLP", "src.train_w2v_mlp"),
    ("Embedding + BiLSTM", "src.train_lstm"),
    ("Evaluation", "src.evaluate"),
    ("Error analysis", "src.error_analysis"),
    ("Final pipeline demo", "src.pipeline"),
]

for name, mod in STAGES:
    print("\n" + "=" * 78)
    print(f"STAGE: {name}  ({mod})")
    print("=" * 78)
    subprocess.run([sys.executable, "-m", mod], check=True)

print("\nALL STAGES COMPLETE - see resumeforge/outputs/")
