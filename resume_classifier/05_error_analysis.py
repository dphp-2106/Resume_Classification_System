"""
SAMATRIX RESUMEFORGE 2026
Step 5: Error Analysis
Inspect wrongly classified resumes and understand failure modes.
"""

import sys, io, os, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
from config import CLEANED_CSV, MODELS_DIR, PLOTS_DIR, ERROR_DIR

import numpy as np
import pandas as pd
import joblib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score

plt.rcParams.update({
    'axes.facecolor': '#0f0f1a', 'figure.facecolor': '#0f0f1a',
    'text.color': '#e0e0e0', 'axes.labelcolor': '#e0e0e0',
    'xtick.color': '#e0e0e0', 'ytick.color': '#e0e0e0',
    'axes.edgecolor': '#333355',
})

print("=" * 65)
print("  SAMATRIX RESUMEFORGE 2026 — Error Analysis")
print("=" * 65)

# Load data and models
df = pd.read_csv(CLEANED_CSV)
df['cleaned_text'] = df['cleaned_text'].fillna('')
le = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.pkl'))
y = le.transform(df['Category'])
classes = le.classes_

# Recreate test split (same seed as training)
indices = np.arange(len(df))
idx_train, idx_test = train_test_split(indices, test_size=0.15, random_state=42, stratify=y)
idx_train, idx_val  = train_test_split(idx_train, test_size=0.1765, random_state=42, stratify=y[idx_train])

X_test_text = df['cleaned_text'].iloc[idx_test].values
y_test = y[idx_test]
df_test = df.iloc[idx_test].copy()

# Load best classical model
best_clf = joblib.load(os.path.join(MODELS_DIR, 'best_classical_model.pkl'))
best_vec = joblib.load(os.path.join(MODELS_DIR, 'best_classical_vectorizer.pkl'))

X_test_vec = best_vec.transform(X_test_text)
y_pred = best_clf.predict(X_test_vec)

# Confidence scores (if supported)
try:
    y_proba = best_clf.predict_proba(X_test_vec)
    confidence = y_proba.max(axis=1)
    has_proba = True
except AttributeError:
    try:
        y_scores = best_clf.decision_function(X_test_vec)
        confidence = y_scores.max(axis=1)
        has_proba = False
    except:
        confidence = np.ones(len(y_pred))
        has_proba = False

# ═══════════════════════════════════════════════════════════════════
# Error DataFrame
# ═══════════════════════════════════════════════════════════════════
df_test = df_test.reset_index(drop=True)
df_test['true_label']  = le.inverse_transform(y_test)
df_test['pred_label']  = le.inverse_transform(y_pred)
df_test['correct']     = y_test == y_pred
df_test['confidence']  = confidence
df_test['text_preview'] = df_test['Resume_str'].str[:300].str.replace('\n', ' ')

errors = df_test[~df_test['correct']].copy()
print(f"\n  Total test samples: {len(df_test)}")
print(f"  Correct:   {df_test['correct'].sum()} ({df_test['correct'].mean()*100:.1f}%)")
print(f"  Incorrect: {len(errors)} ({len(errors)/len(df_test)*100:.1f}%)")

# Save full error table
error_cols = ['ID', 'true_label', 'pred_label', 'confidence', 'text_preview']
errors[error_cols].sort_values('confidence', ascending=False).to_csv(
    os.path.join(ERROR_DIR, 'errors.csv'), index=False)
print(f"\n  Error table saved to outputs/error_analysis/errors.csv")

# ═══════════════════════════════════════════════════════════════════
# Top confusion pairs
# ═══════════════════════════════════════════════════════════════════
print("\n[2] Top confused category pairs:")
confusion_pairs = errors.groupby(['true_label', 'pred_label']).size().reset_index(name='count')
confusion_pairs = confusion_pairs.sort_values('count', ascending=False).head(15)
print(confusion_pairs.to_string(index=False))
confusion_pairs.to_csv(os.path.join(ERROR_DIR, 'confusion_pairs.csv'), index=False)

# ═══════════════════════════════════════════════════════════════════
# Error rate per class
# ═══════════════════════════════════════════════════════════════════
print("\n[3] Error rate per class:")
class_stats = df_test.groupby('true_label')['correct'].agg(['sum', 'count'])
class_stats.columns = ['correct', 'total']
class_stats['error_rate'] = 1 - class_stats['correct'] / class_stats['total']
class_stats = class_stats.sort_values('error_rate', ascending=False)
print(class_stats.to_string())

fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor('#0f0f1a')
ax.set_facecolor('#0f0f1a')
colors = ['#FF6584' if r > 0.1 else '#43C6AC' for r in class_stats['error_rate']]
bars = ax.barh(class_stats.index, class_stats['error_rate'] * 100,
               color=colors, edgecolor='none')
ax.axvline(10, color='#FFD200', linewidth=2, linestyle='--', label='10% threshold')
ax.set_xlabel('Error Rate (%)', fontsize=12)
ax.set_title('Error Rate per Category', fontsize=14, fontweight='bold', color='#e0e0e0')
ax.legend(fontsize=9, framealpha=0.2, labelcolor='#e0e0e0', facecolor='#1a1a2e')
for bar, rate in zip(bars, class_stats['error_rate']):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f'{rate*100:.1f}%', va='center', fontsize=9, color='#e0e0e0')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '14_error_rate_per_class.png'), dpi=130,
            bbox_inches='tight', facecolor='#0f0f1a')
plt.close()

# ═══════════════════════════════════════════════════════════════════
# Confidence distribution: correct vs incorrect
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('#0f0f1a')
ax.set_facecolor('#0f0f1a')
correct_conf = df_test[df_test['correct']]['confidence']
error_conf   = df_test[~df_test['correct']]['confidence']
ax.hist(correct_conf, bins=30, alpha=0.7, color='#43C6AC', label='Correct', edgecolor='none')
ax.hist(error_conf,   bins=30, alpha=0.7, color='#FF6584', label='Incorrect', edgecolor='none')
ax.axvline(correct_conf.mean(), color='#43C6AC', linewidth=2, linestyle='--',
           label=f'Correct mean: {correct_conf.mean():.2f}')
ax.axvline(error_conf.mean(), color='#FF6584', linewidth=2, linestyle='--',
           label=f'Error mean: {error_conf.mean():.2f}')
ax.set_title('Confidence Score Distribution: Correct vs Incorrect', fontsize=13,
             fontweight='bold', color='#e0e0e0')
ax.set_xlabel('Confidence Score', fontsize=12)
ax.legend(fontsize=9, framealpha=0.3, labelcolor='#e0e0e0', facecolor='#1a1a2e')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '15_confidence_distribution.png'), dpi=130,
            bbox_inches='tight', facecolor='#0f0f1a')
plt.close()

# ═══════════════════════════════════════════════════════════════════
# Sample error inspection
# ═══════════════════════════════════════════════════════════════════
print("\n[4] Sample misclassified resumes:")
print("-" * 65)
for _, row in errors.head(5).iterrows():
    print(f"  True:  {row['true_label']}")
    print(f"  Pred:  {row['pred_label']}")
    print(f"  Conf:  {row['confidence']:.3f}")
    print(f"  Text:  {row['text_preview'][:200]}...")
    print()

# ═══════════════════════════════════════════════════════════════════
# Root cause analysis
# ═══════════════════════════════════════════════════════════════════
print("[5] Root cause analysis:")
print("  Likely causes:")
print("  - Similar vocabularies: HR vs BUSINESS-DEVELOPMENT, FINANCE vs BANKING")
print("  - Generic resumes with no domain-specific keywords")
print("  - Very short resumes (insufficient signal)")
print("  - Imbalanced classes: BPO (22), AUTOMOBILE (36), AGRICULTURE (63)")

short_errors = errors[errors['clean_word_count'] < 100]
print(f"\n  Short resume errors (< 100 words): {len(short_errors)} / {len(errors)}")

imbalanced_cats = ['BPO', 'AUTOMOBILE', 'AGRICULTURE']
imb_errors = errors[errors['true_label'].isin(imbalanced_cats)]
print(f"  Errors in imbalanced classes: {len(imb_errors)} / {len(errors)}")

print("\n" + "=" * 65)
print("  ERROR ANALYSIS COMPLETE")
print("=" * 65)
