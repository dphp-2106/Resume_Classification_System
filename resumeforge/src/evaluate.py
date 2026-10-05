"""Final evaluation (Stage 10): compare every model on the held-out TEST set.

Outputs:
  outputs/reports/model_comparison.csv      - test metrics for all models
  outputs/eda/10_confusion_matrix_best.png  - confusion matrix heatmap
  outputs/reports/per_class_metrics.csv     - per-class P/R/F1 of the best model
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import EDA_DIR, MODEL_DIR, REPORT_DIR  # noqa: E402
from src.data_loading import load_data  # noqa: E402
from src.preprocessing import preprocess  # noqa: E402
from src.train_classical import clean_df  # noqa: E402

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight"})


def _f1s(y_true, pred):
    return {
        "accuracy": round(accuracy_score(y_true, pred), 4),
        "precision_macro": round(precision_score(y_true, pred, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_true, pred, average="macro", zero_division=0), 4),
        "f1_macro": round(f1_score(y_true, pred, average="macro", zero_division=0), 4),
        "f1_weighted": round(f1_score(y_true, pred, average="weighted", zero_division=0), 4),
    }


def main():
    df = clean_df(load_data())
    blob = joblib.load(MODEL_DIR / "best_classical.joblib")
    te_idx = pd.Index(blob["test_index"])
    y_test = df.loc[te_idx, "Category"]

    rows = []

    # ---- classical (best config)
    cls_pred = pd.read_csv(REPORT_DIR / "preds_classical_test.csv")
    cls_pred = cls_pred[cls_pred["split"] == "test"].sort_values("id")
    rows.append({"model": f"TF-IDF + {blob['model_name']} "
                          f"({blob['preprocess_mode']}, {blob['ngram']})",
                 **_f1s(cls_pred["true"], cls_pred["pred"])})

    # ---- Word2Vec + MLP
    w2v_pred = pd.read_csv(REPORT_DIR / "preds_w2v_mlp.csv")
    w2v_pred = w2v_pred[w2v_pred["split"] == "test"].sort_values("id")
    rows.append({"model": "Word2Vec(mean) + Dense MLP",
                 **_f1s(w2v_pred["true"], w2v_pred["pred"])})

    # ---- BiLSTM
    lstm_pred = pd.read_csv(REPORT_DIR / "preds_bilstm_test.csv")
    lstm_pred = lstm_pred.sort_values("id")
    rows.append({"model": "Embedding + BiLSTM",
                 **_f1s(lstm_pred["true"], lstm_pred["pred"])})

    comp = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    comp.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    pd.set_option("display.width", 200)
    print("=" * 78)
    print("TEST-SET COMPARISON (sorted by macro-F1)")
    print("=" * 78)
    print(comp.to_string(index=False))

    # ---- confusion matrix + per-class metrics of the best model
    best_name = comp.iloc[0]["model"]
    if "TF-IDF" in best_name:
        y_true, y_pred = cls_pred["true"], cls_pred["pred"]
    elif "Word2Vec" in best_name:
        y_true, y_pred = w2v_pred["true"], w2v_pred["pred"]
    else:
        y_true, y_pred = lstm_pred["true"], lstm_pred["pred"]

    le = LabelEncoder().fit(df["Category"])
    cm = confusion_matrix(y_true, y_pred, labels=le.classes_)
    fig, ax = plt.subplots(figsize=(15, 13))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix - TEST set - {best_name}")
    ax.tick_params(labelsize=8)
    fig.savefig(EDA_DIR / "10_confusion_matrix_best.png")
    plt.close(fig)

    report = classification_report(y_true, y_pred, labels=le.classes_,
                                    target_names=le.classes_,
                                    output_dict=True, zero_division=0)
    per_class = pd.DataFrame(report).T.iloc[:len(le.classes_)][
        ["precision", "recall", "f1-score", "support"]]
    per_class.index.name = "class"
    per_class.to_csv(REPORT_DIR / "per_class_metrics.csv")
    print("\nPer-class metrics (best model):")
    print(per_class.round(3).to_string())

    summary = {
        "best_model": best_name,
        "test_metrics": comp.iloc[0].to_dict(),
        "worst_f1_classes": per_class["f1-score"].sort_values().head(5).round(3).to_dict(),
        "top_confusions": _top_confusions(y_true, y_pred, le, 5),
    }
    (REPORT_DIR / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("\nTop confusions:", summary["top_confusions"])
    return comp


def _top_confusions(y_true, y_pred, le, k):
    cm = confusion_matrix(y_true, y_pred, labels=le.classes_)
    np.fill_diagonal(cm, 0)
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                pairs.append((le.classes_[i], le.classes_[j], int(cm[i, j])))
    pairs.sort(key=lambda x: -x[2])
    return [f"{a} -> {b} ({n})" for a, b, n in pairs[:k]]


if __name__ == "__main__":
    main()
