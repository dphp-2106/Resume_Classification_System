"""
Shared configuration — paths and constants.
All scripts import from here.
"""
import os

# Root of the repo (parent of resume_classifier/)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# Data paths
DATA_PATH = os.path.join(_ROOT, 'Dataset-20261005T050929Z-1-001', 'Dataset', 'Resume', 'Resume.csv')
RESUME_PDF_DIR = os.path.join(_ROOT, 'Dataset-20261005T050929Z-1-001', 'Dataset', 'data', 'data')

# Output directories (relative to resume_classifier/)
OUTPUTS_DIR = os.path.join(_HERE, 'outputs')
EDA_DIR     = os.path.join(OUTPUTS_DIR, 'eda')
MODELS_DIR  = os.path.join(OUTPUTS_DIR, 'models')
PLOTS_DIR   = os.path.join(OUTPUTS_DIR, 'plots')
ERROR_DIR   = os.path.join(OUTPUTS_DIR, 'error_analysis')

CLEANED_CSV = os.path.join(OUTPUTS_DIR, 'cleaned_resumes.csv')

# Create all output dirs
for d in [OUTPUTS_DIR, EDA_DIR, MODELS_DIR, PLOTS_DIR, ERROR_DIR]:
    os.makedirs(d, exist_ok=True)
