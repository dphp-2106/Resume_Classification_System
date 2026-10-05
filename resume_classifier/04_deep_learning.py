"""
SAMATRIX RESUMEFORGE 2026
Step 4: Deep Learning — Word2Vec + LSTM/GRU Neural Classifier
"""

import sys, io, os, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
from config import DATA_PATH, CLEANED_CSV, MODELS_DIR, PLOTS_DIR

import pandas as pd
import numpy as np
import joblib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import seaborn as sns

from gensim.models import Word2Vec

plt.rcParams.update({
    'axes.facecolor': '#0f0f1a', 'figure.facecolor': '#0f0f1a',
    'text.color': '#e0e0e0', 'axes.labelcolor': '#e0e0e0',
    'xtick.color': '#e0e0e0', 'ytick.color': '#e0e0e0',
    'axes.edgecolor': '#333355',
})

os.makedirs('outputs/models', exist_ok=True)
os.makedirs('outputs/plots', exist_ok=True)

print("=" * 65)
print("  SAMATRIX RESUMEFORGE 2026 — Word2Vec + Neural Classifier")
print("=" * 65)

# ═══════════════════════════════════════════════════════════════════
# Load cleaned data
# ═══════════════════════════════════════════════════════════════════
cleaned_path = CLEANED_CSV
if os.path.exists(cleaned_path):
    df = pd.read_csv(cleaned_path)
else:
    from preprocessing_utils import clean_resume
    DATA_PATH_LOCAL = DATA_PATH
    df = pd.read_csv(DATA_PATH_LOCAL)
    df['cleaned_text'] = df['Resume_str'].apply(clean_resume)

df['cleaned_text'] = df['cleaned_text'].fillna('')

le = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.pkl'))
y = le.transform(df['Category'])
classes = le.classes_
num_classes = len(classes)

tokenized = [text.split() for text in df['cleaned_text']]

# ═══════════════════════════════════════════════════════════════════
# Train/Val/Test split
# ═══════════════════════════════════════════════════════════════════
print("\n[1] Stratified split (70/15/15)...")
indices = np.arange(len(df))
idx_train, idx_test = train_test_split(indices, test_size=0.15, random_state=42, stratify=y)
idx_train, idx_val = train_test_split(idx_train, test_size=0.1765, random_state=42, stratify=y[idx_train])

print(f"    Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")

# ═══════════════════════════════════════════════════════════════════
# Word2Vec (trained on TRAIN ONLY)
# ═══════════════════════════════════════════════════════════════════
print("\n[2] Training Word2Vec on training corpus...")
train_tokens = [tokenized[i] for i in idx_train]

W2V_DIM = 200
w2v_model = Word2Vec(
    sentences=train_tokens,
    vector_size=W2V_DIM,
    window=10,
    min_count=2,
    workers=4,
    epochs=15,
    sg=1,   # Skip-gram
    hs=0,
    negative=10
)
w2v_model.save(os.path.join(MODELS_DIR, 'word2vec.model'))
print(f"    Word2Vec vocab size: {len(w2v_model.wv)}")


def doc_to_vector(tokens, model, dim=W2V_DIM):
    """Mean-pool word vectors. Returns zero vector for unknown docs."""
    vectors = []
    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(dim)


# ═══════════════════════════════════════════════════════════════════
# Create document vectors (mean pooling)
# ═══════════════════════════════════════════════════════════════════
print("\n[3] Creating document vectors (mean pooling)...")
all_vectors = np.array([doc_to_vector(tok, w2v_model) for tok in tokenized])

X_train_w2v = all_vectors[idx_train]
X_val_w2v   = all_vectors[idx_val]
X_test_w2v  = all_vectors[idx_test]

y_train = y[idx_train]
y_val   = y[idx_val]
y_test  = y[idx_test]

print(f"    Vector shape: {X_train_w2v.shape}")

# ═══════════════════════════════════════════════════════════════════
# Neural Classifier (Dense Network on W2V mean vectors)
# ═══════════════════════════════════════════════════════════════════
print("\n[4] Training Neural Dense Classifier (W2V + MLP)...")

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_w2v)
X_val_scaled   = scaler.transform(X_val_w2v)
X_test_scaled  = scaler.transform(X_test_w2v)

mlp = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128),
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size=64,
    learning_rate='adaptive',
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=15,
    random_state=42,
    verbose=False
)
mlp.fit(X_train_scaled, y_train)

y_val_pred_mlp  = mlp.predict(X_val_scaled)
y_test_pred_mlp = mlp.predict(X_test_scaled)

val_acc_mlp  = accuracy_score(y_val, y_val_pred_mlp)
val_f1_mlp   = f1_score(y_val, y_val_pred_mlp, average='macro')
test_acc_mlp = accuracy_score(y_test, y_test_pred_mlp)
test_f1_mlp  = f1_score(y_test, y_test_pred_mlp, average='macro')

print(f"  W2V + MLP  Val:  Acc={val_acc_mlp:.4f}  Macro-F1={val_f1_mlp:.4f}")
print(f"  W2V + MLP  Test: Acc={test_acc_mlp:.4f}  Macro-F1={test_f1_mlp:.4f}")

joblib.dump(mlp, os.path.join(MODELS_DIR, 'w2v_mlp_classifier.pkl'))
joblib.dump(scaler, os.path.join(MODELS_DIR, 'w2v_scaler.pkl'))

# ═══════════════════════════════════════════════════════════════════
# Also test W2V with LightGBM
# ═══════════════════════════════════════════════════════════════════
print("\n[5] Training W2V + LightGBM...")
import lightgbm as lgb

lgbm_w2v = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                                num_leaves=63, n_jobs=1,
                                class_weight='balanced', verbose=-1)
lgbm_w2v.fit(X_train_scaled, y_train)

y_val_pred_lgb  = lgbm_w2v.predict(X_val_scaled)
y_test_pred_lgb = lgbm_w2v.predict(X_test_scaled)

val_f1_lgb  = f1_score(y_val, y_val_pred_lgb, average='macro')
test_f1_lgb = f1_score(y_test, y_test_pred_lgb, average='macro')

print(f"  W2V + LightGBM  Val:  Macro-F1={val_f1_lgb:.4f}")
print(f"  W2V + LightGBM  Test: Macro-F1={test_f1_lgb:.4f}")

joblib.dump(lgbm_w2v, os.path.join(MODELS_DIR, 'w2v_lgbm_classifier.pkl'))

# ═══════════════════════════════════════════════════════════════════
# Save DL results
# ═══════════════════════════════════════════════════════════════════
dl_results = {
    'W2V + MLP': {
        'val_acc': val_acc_mlp, 'val_f1': val_f1_mlp,
        'test_acc': test_acc_mlp, 'test_f1': test_f1_mlp
    },
    'W2V + LightGBM': {
        'val_f1': val_f1_lgb, 'test_f1': test_f1_lgb
    }
}
with open(os.path.join(MODELS_DIR, 'dl_results.json'), 'w') as f:
    json.dump(dl_results, f, indent=2)

# ═══════════════════════════════════════════════════════════════════
# Confusion matrix for best DL model
# ═══════════════════════════════════════════════════════════════════
best_dl_pred = y_test_pred_mlp if test_f1_mlp >= test_f1_lgb else y_test_pred_lgb
best_dl_name = 'W2V + MLP' if test_f1_mlp >= test_f1_lgb else 'W2V + LightGBM'

cm = confusion_matrix(y_test, best_dl_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(18, 15))
fig.patch.set_facecolor('#0f0f1a')
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='magma',
            xticklabels=classes, yticklabels=classes, ax=ax,
            linewidths=0.5, linecolor='#0f0f1a', annot_kws={'size': 8})
ax.set_title(f'Normalized Confusion Matrix — {best_dl_name}', fontsize=14,
             fontweight='bold', color='#e0e0e0', pad=15)
ax.set_ylabel('True Label', fontsize=12)
ax.set_xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '12_confusion_matrix_dl.png'), dpi=130,
            bbox_inches='tight', facecolor='#0f0f1a')
plt.close()

# ═══════════════════════════════════════════════════════════════════
# Word2Vec visualization — t-SNE of word vectors
# ═══════════════════════════════════════════════════════════════════
print("\n[6] Word2Vec visualization (selected words)...")
from sklearn.manifold import TSNE

tech_words = [
    'python', 'java', 'sql', 'accounting', 'nurse', 'teaching',
    'chef', 'sales', 'management', 'engineering', 'finance',
    'design', 'marketing', 'hr', 'analytics', 'network',
    'database', 'budget', 'patient', 'student', 'cooking',
    'insurance', 'software', 'hardware', 'legal', 'consulting'
]
valid_words = [w for w in tech_words if w in w2v_model.wv]

if len(valid_words) > 5:
    vectors_2d_words = np.array([w2v_model.wv[w] for w in valid_words])
    tsne = TSNE(n_components=2, perplexity=min(5, len(valid_words)-1),
                random_state=42, max_iter=1000)
    coords = tsne.fit_transform(vectors_2d_words)

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#0f0f1a')
    scatter = ax.scatter(coords[:, 0], coords[:, 1],
                         c=range(len(valid_words)), cmap='plasma', s=200, alpha=0.8)
    for i, word in enumerate(valid_words):
        ax.annotate(word, (coords[i, 0], coords[i, 1]),
                    fontsize=11, color='#e0e0e0',
                    xytext=(5, 5), textcoords='offset points')
    ax.set_title('Word2Vec Embeddings — t-SNE Projection', fontsize=14,
                 fontweight='bold', color='#e0e0e0')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '13_word2vec_tsne.png'), dpi=130,
                bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    print("    Saved t-SNE visualization")

print("\n" + "=" * 65)
print(f"  W2V + MLP  Test Macro-F1: {test_f1_mlp:.4f}")
print(f"  W2V + LGBM Test Macro-F1: {test_f1_lgb:.4f}")
print("  DEEP LEARNING TRAINING COMPLETE")
print("=" * 65)
