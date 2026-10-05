"""Neural approach #1: Word2Vec embeddings + Dense neural network.

Word2Vec is a *representation* method (per the guidelines) - it is trained on
the TRAIN split only, then combined with a Dense MLP classifier.
Document vector = mean-pooling of in-vocabulary word vectors.
"""
import json
import sys
from pathlib import Path

import gensim
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score, precision_score, recall_score)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import MODEL_DIR, REPORT_DIR, RANDOM_STATE  # noqa: E402
from src.data_loading import load_data  # noqa: E402
from src.preprocessing import preprocess  # noqa: E402
from src.train_classical import clean_df  # noqa: E402

W2V_SIZE = 100
W2V_WINDOW = 5
W2V_MIN_COUNT = 2
W2V_EPOCHS = 15


def doc_vector(tokens, model):
    """Mean-pool word vectors; zero vector if no in-vocab token."""
    vecs = [model.wv[t] for t in tokens if t in model.wv]
    if not vecs:
        return np.zeros(W2V_SIZE)
    return np.mean(vecs, axis=0)


def main():
    df = clean_df(load_data())

    # reuse the exact split created by train_classical (no leakage)
    blob = joblib.load(MODEL_DIR / "best_classical.joblib")
    tr_idx, va_idx, te_idx = (pd.Index(blob["train_index"]),
                              pd.Index(blob["val_index"]),
                              pd.Index(blob["test_index"]))
    y_train, y_val, y_test = df.loc[tr_idx, "Category"], \
        df.loc[va_idx, "Category"], df.loc[te_idx, "Category"]

    tok_train = [preprocess(t, "standard").split() for t in df.loc[tr_idx, "text"]]
    tok_val = [preprocess(t, "standard").split() for t in df.loc[va_idx, "text"]]
    tok_test = [preprocess(t, "standard").split() for t in df.loc[te_idx, "text"]]

    # ---- Word2Vec trained on TRAIN ONLY
    w2v = gensim.models.Word2Vec(
        sentences=tok_train, vector_size=W2V_SIZE, window=W2V_WINDOW,
        min_count=W2V_MIN_COUNT, workers=4, epochs=W2V_EPOCHS,
        seed=RANDOM_STATE)
    print(f"Word2Vec: vocab={len(w2v.wv)} trained on {len(tok_train)} train resumes")

    Xtr = np.array([doc_vector(t, w2v) for t in tok_train])
    Xva = np.array([doc_vector(t, w2v) for t in tok_val])
    Xte = np.array([doc_vector(t, w2v) for t in tok_test])
    print("doc vectors:", Xtr.shape, Xva.shape, Xte.shape)

    # ---- Dense NN (MLP) with class-balanced sample weights
    # (labels are integer-encoded: MLPClassifier + early_stopping +
    #  sample_weight cannot handle raw string labels)
    le = LabelEncoder().fit(df["Category"])
    Ytr, Yva, Yte = le.transform(y_train), le.transform(y_val), \
        le.transform(y_test)
    sw = compute_sample_weight("balanced", Ytr)
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400,
                        random_state=RANDOM_STATE, early_stopping=True,
                        validation_fraction=0.15, n_iter_no_change=15)
    mlp.fit(Xtr, Ytr, sample_weight=sw)
    out = {"model": "Word2Vec(mean-pool) + Dense MLP (128,64)",
           "w2v_vocab": int(len(w2v.wv))}
    for split, X, y in [("val", Xva, y_val), ("test", Xte, y_test)]:
        pred = le.inverse_transform(mlp.predict(X))
        out[split] = {
            "accuracy": round(accuracy_score(y, pred), 4),
            "precision_macro": round(precision_score(y, pred, average="macro", zero_division=0), 4),
            "recall_macro": round(recall_score(y, pred, average="macro", zero_division=0), 4),
            "f1_macro": round(f1_score(y, pred, average="macro", zero_division=0), 4),
            "f1_weighted": round(f1_score(y, pred, average="weighted", zero_division=0), 4),
        }
        print(split, out[split])

    (REPORT_DIR / "w2v_mlp_results.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")

    # dump val+test predictions for error analysis
    preds = []
    for split, X, y, idxs in [("val", Xva, y_val, va_idx),
                                ("test", Xte, y_test, te_idx)]:
        p = mlp.predict_proba(X)
        preds.append(pd.DataFrame({
            "id": df.loc[idxs, "ID"].values,
            "true": y.values,
            "pred": le.inverse_transform(mlp.predict(X)),
            "confidence": p.max(axis=1).round(4),
            "split": split,
        }))
    pd.concat(preds).to_csv(REPORT_DIR / "preds_w2v_mlp.csv", index=False)
    print("saved ->", REPORT_DIR / "preds_w2v_mlp.csv")

    joblib.dump({
        "w2v": w2v, "mlp": mlp, "classes": le.classes_,
        "preprocess_mode": "standard",
    }, MODEL_DIR / "w2v_mlp.joblib")
    print(f"saved -> {MODEL_DIR / 'w2v_mlp.joblib'}")
    return out


if __name__ == "__main__":
    main()
