"""
SAMATRIX RESUMEFORGE 2026
Step 3: Model Training — Classical ML Baselines (TF-IDF)
  - Logistic Regression
  - Random Forest
  - SVM (Linear)
  - XGBoost
  - LightGBM
  - Multinomial Naive Bayes

Saves models, vectorizers, label encoder, and metrics.
"""

import sys, io, os, warnings, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
from config import DATA_PATH, CLEANED_CSV, MODELS_DIR, PLOTS_DIR

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb
import lightgbm as lgb

# Setup
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

plt.rcParams.update({
    'axes.facecolor': '#0f0f1a', 'figure.facecolor': '#0f0f1a',
    'text.color': '#e0e0e0', 'axes.labelcolor': '#e0e0e0',
    'xtick.color': '#e0e0e0', 'ytick.color': '#e0e0e0',
    'axes.edgecolor': '#333355',
})

print("=" * 65)
print("  SAMATRIX RESUMEFORGE 2026 — Classical ML Training")
print("=" * 65)

# ═══════════════════════════════════════════════════════════════════
# Load and preprocess
# ═══════════════════════════════════════════════════════════════════
print("\n[1] Loading and preprocessing data...")

# Try to load cached cleaned version
cleaned_path = CLEANED_CSV
if os.path.exists(cleaned_path):
    df = pd.read_csv(cleaned_path)
    print(f"    Loaded pre-cleaned data: {df.shape}")
else:
    from preprocessing_utils import clean_resume
    df_raw = pd.read_csv(DATA_PATH)
    df = df_raw.copy()
    df['cleaned_text'] = df['Resume_str'].apply(clean_resume)
    df['clean_word_count'] = df['cleaned_text'].str.split().str.len()
    df.to_csv(CLEANED_CSV, index=False)
    print(f"    Preprocessed and saved: {df.shape}")

# Load and encode labels
le = LabelEncoder()
y = le.fit_transform(df['Category'])
X_text = df['cleaned_text'].fillna('').values
classes = le.classes_

print(f"    Classes ({len(classes)}): {classes[:5]} ...")
print(f"    Label mapping saved.")

# Save label encoder
joblib.dump(le, os.path.join(MODELS_DIR, 'label_encoder.pkl'))

# ═══════════════════════════════════════════════════════════════════
# Stratified train/val/test split (70/15/15)
# ═══════════════════════════════════════════════════════════════════
print("\n[2] Stratified split (70/15/15)...")
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.15, random_state=42, stratify=y
)
X_train_text, X_val_text, y_train, y_val = train_test_split(
    X_train_text, y_train, test_size=0.1765, random_state=42, stratify=y_train
)
print(f"    Train: {len(X_train_text)}, Val: {len(X_val_text)}, Test: {len(X_test_text)}")

# ═══════════════════════════════════════════════════════════════════
# TF-IDF Vectorization (fit only on train!)
# ═══════════════════════════════════════════════════════════════════
print("\n[3] TF-IDF Vectorization...")
tfidf_unigram = TfidfVectorizer(
    ngram_range=(1, 1), max_features=15000,
    min_df=2, max_df=0.95, sublinear_tf=True
)
tfidf_bigram = TfidfVectorizer(
    ngram_range=(1, 2), max_features=20000,
    min_df=2, max_df=0.95, sublinear_tf=True
)

X_train_uni = tfidf_unigram.fit_transform(X_train_text)
X_val_uni   = tfidf_unigram.transform(X_val_text)
X_test_uni  = tfidf_unigram.transform(X_test_text)

X_train_bi = tfidf_bigram.fit_transform(X_train_text)
X_val_bi   = tfidf_bigram.transform(X_val_text)
X_test_bi  = tfidf_bigram.transform(X_test_text)

print(f"    Unigram vocab: {X_train_uni.shape[1]}")
print(f"    Bigram vocab:  {X_train_bi.shape[1]}")

joblib.dump(tfidf_unigram, os.path.join(MODELS_DIR, 'tfidf_unigram.pkl'))
joblib.dump(tfidf_bigram, os.path.join(MODELS_DIR, 'tfidf_bigram.pkl'))

# ═══════════════════════════════════════════════════════════════════
# Model Zoo
# ═══════════════════════════════════════════════════════════════════
models = {
    'Logistic Regression (uni)': (LogisticRegression(max_iter=1000, C=5.0, solver='lbfgs',
                                                      n_jobs=1),
                                   X_train_uni, X_val_uni, X_test_uni, 'tfidf_unigram'),
    'Logistic Regression (bi)':  (LogisticRegression(max_iter=1000, C=5.0, solver='lbfgs',
                                                      n_jobs=1),
                                   X_train_bi, X_val_bi, X_test_bi, 'tfidf_bigram'),
    'LinearSVC (bi)':            (LinearSVC(C=1.0, max_iter=2000),
                                   X_train_bi, X_val_bi, X_test_bi, 'tfidf_bigram'),
    'Naive Bayes (uni)':         (MultinomialNB(alpha=0.1),
                                   X_train_uni, X_val_uni, X_test_uni, 'tfidf_unigram'),
    'XGBoost (bi)':              (xgb.XGBClassifier(n_estimators=80, max_depth=4,
                                                     learning_rate=0.15, n_jobs=4,
                                                     eval_metric='mlogloss',
                                                     verbosity=0),
                                   X_train_bi, X_val_bi, X_test_bi, 'tfidf_bigram'),
    'LightGBM (bi)':             (lgb.LGBMClassifier(n_estimators=80, learning_rate=0.15,
                                                      num_leaves=31, n_jobs=4,
                                                      class_weight='balanced', verbose=-1),
                                   X_train_bi, X_val_bi, X_test_bi, 'tfidf_bigram'),
}

results = {}
print("\n[4] Training models...")
print("-" * 65)

for name, (clf, X_tr, X_vl, X_ts, vec_name) in models.items():
    print(f"\n  >> Training: {name}")
    clf.fit(X_tr, y_train)

    y_val_pred = clf.predict(X_vl)
    y_test_pred = clf.predict(X_ts)

    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1  = f1_score(y_val, y_val_pred, average='macro')
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1  = f1_score(y_test, y_test_pred, average='macro')

    print(f"     Val  Acc={val_acc:.4f}  Macro-F1={val_f1:.4f}")
    print(f"     Test Acc={test_acc:.4f}  Macro-F1={test_f1:.4f}")

    results[name] = {
        'model': clf,
        'vectorizer': vec_name,
        'val_acc': val_acc, 'val_f1': val_f1,
        'test_acc': test_acc, 'test_f1': test_f1,
        'y_test_pred': y_test_pred
    }

    # Save model
    safe_name = name.replace(' ', '_').replace('(', '').replace(')', '')
    joblib.dump(clf, os.path.join(MODELS_DIR, f'{safe_name}.pkl'))

print("\n" + "-" * 65)

# ═══════════════════════════════════════════════════════════════════
# Comparison table
# ═══════════════════════════════════════════════════════════════════
print("\n[5] Results Summary:")
summary_rows = []
for name, r in results.items():
    summary_rows.append({
        'Model': name,
        'Val Acc': f"{r['val_acc']:.4f}",
        'Val Macro-F1': f"{r['val_f1']:.4f}",
        'Test Acc': f"{r['test_acc']:.4f}",
        'Test Macro-F1': f"{r['test_f1']:.4f}",
    })
summary_df = pd.DataFrame(summary_rows).sort_values('Test Macro-F1', ascending=False)
print(summary_df.to_string(index=False))
summary_df.to_csv(os.path.join(os.path.dirname(MODELS_DIR), 'model_comparison.csv'), index=False)

# ═══════════════════════════════════════════════════════════════════
# Save best model
# ═══════════════════════════════════════════════════════════════════
best_name = max(results, key=lambda k: results[k]['test_f1'])
print(f"\n  Best model: {best_name}")
best_clf = results[best_name]['model']
best_vec_name = results[best_name]['vectorizer']
best_vec = tfidf_bigram if 'bigram' in best_vec_name else tfidf_unigram

joblib.dump(best_clf, os.path.join(MODELS_DIR, 'best_classical_model.pkl'))
joblib.dump(best_vec, os.path.join(MODELS_DIR, 'best_classical_vectorizer.pkl'))

# Save best model name for later
with open(os.path.join(MODELS_DIR, 'best_model_info.json'), 'w') as f:
    json.dump({'name': best_name, 'vectorizer': best_vec_name,
               'test_acc': results[best_name]['test_acc'],
               'test_f1': results[best_name]['test_f1']}, f, indent=2)

# ═══════════════════════════════════════════════════════════════════
# Confusion Matrix for best model
# ═══════════════════════════════════════════════════════════════════
print("\n[6] Plotting confusion matrix for best model...")
y_pred_best = results[best_name]['y_test_pred']
cm = confusion_matrix(y_test, y_pred_best)
cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(18, 15))
fig.patch.set_facecolor('#0f0f1a')
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='viridis',
            xticklabels=classes, yticklabels=classes, ax=ax,
            linewidths=0.5, linecolor='#0f0f1a',
            annot_kws={'size': 8})
ax.set_title(f'Normalized Confusion Matrix — {best_name}', fontsize=14,
             fontweight='bold', color='#e0e0e0', pad=15)
ax.set_ylabel('True Label', fontsize=12)
ax.set_xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '10_confusion_matrix_best.png'), dpi=130,
            bbox_inches='tight', facecolor='#0f0f1a')
plt.close()
print("    Saved confusion matrix")

# ═══════════════════════════════════════════════════════════════════
# Model comparison bar chart
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.patch.set_facecolor('#0f0f1a')
colors = ['#6C63FF', '#FF6584', '#43C6AC', '#F7971E', '#FFD200', '#FC5C7D']

model_names = [r['Model'] for _, r in summary_df.iterrows()]
val_f1s = [float(r['Val Macro-F1']) for _, r in summary_df.iterrows()]
test_f1s = [float(r['Test Macro-F1']) for _, r in summary_df.iterrows()]

for ax, vals, title in zip(axes, [val_f1s, test_f1s], ['Validation Macro-F1', 'Test Macro-F1']):
    ax.set_facecolor('#0f0f1a')
    bars = ax.barh(model_names, vals, color=colors[:len(model_names)], edgecolor='none')
    ax.set_xlim(0, 1.05)
    ax.set_title(title, fontsize=13, fontweight='bold', color='#e0e0e0')
    ax.set_xlabel('Macro-F1 Score', fontsize=11)
    ax.axvline(0.9, color='#FFD200', linewidth=1.5, linestyle='--', alpha=0.7)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{v:.4f}', va='center', fontsize=9, color='#e0e0e0')

plt.suptitle('Classical ML Model Comparison', fontsize=15, fontweight='bold', color='#6C63FF')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '11_model_comparison.png'), dpi=130,
            bbox_inches='tight', facecolor='#0f0f1a')
plt.close()
print("    Saved model comparison chart")

# ═══════════════════════════════════════════════════════════════════
# Per-class metrics for best model
# ═══════════════════════════════════════════════════════════════════
print("\n[7] Per-class metrics report:")
X_test_vec = best_vec.transform(X_test_text)
report = classification_report(y_test, y_pred_best,
                               target_names=classes, output_dict=True)
report_df = pd.DataFrame(report).T.iloc[:-3]
print(report_df[['precision', 'recall', 'f1-score', 'support']].round(3).to_string())
report_df.to_csv(os.path.join(os.path.dirname(MODELS_DIR), 'per_class_metrics.csv'))

print("\n" + "=" * 65)
print("  CLASSICAL ML TRAINING COMPLETE")
print(f"  Best: {best_name} — Test Macro-F1: {results[best_name]['test_f1']:.4f}")
print("=" * 65)
