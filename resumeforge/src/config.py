"""SAMATRIX RESUMEFORGE 2026 — Resume Classification
End-to-end NLP pipeline: Data -> EDA -> Preprocessing -> Features -> ML/DL -> Evaluation -> Demo
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(r"D:\samatrix_hackathon")
DATASET_DIR = ROOT / "Dataset-20261005T050929Z-1-001" / "Dataset"
RESUME_CSV = DATASET_DIR / "Resume" / "Resume.csv"
RESUME_XLSX = DATASET_DIR / "Resume" / "Resume.xlsx"
PDF_DIRS = {
    cat: DATASET_DIR / "data" / "data" / cat
    for cat in ["INFORMATION-TECHNOLOGY", "PUBLIC-RELATIONS", "SALES", "TEACHER"]
}

PROJECT_DIR = ROOT / "resumeforge"
OUTPUT_DIR = PROJECT_DIR / "outputs"
EDA_DIR = OUTPUT_DIR / "eda"
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_DIR = OUTPUT_DIR / "reports"
for _d in (OUTPUT_DIR, EDA_DIR, MODEL_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- constants
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15          # of the training remainder -> overall ~70/15/15
MIN_WORDS_TO_KEEP = 3    # documents with fewer words are dropped (justified in report)

# classes sorted for reproducibility
CLASSES = [
    "ACCOUNTANT", "AGRICULTURE", "ADVOCATE", "APPAREL", "ARTS", "AVIATION",
    "BANKING", "BPO", "BUSINESS-DEVELOPMENT", "CHEF", "CONSTRUCTION",
    "CONSULTANT", "DESIGNER", "DIGITAL-MEDIA", "ENGINEERING", "FINANCE",
    "FITNESS", "HEALTHCARE", "HR", "INFORMATION-TECHNOLOGY", "PUBLIC-RELATIONS",
    "SALES", "TEACHER", "AUTOMOBILE",
]
