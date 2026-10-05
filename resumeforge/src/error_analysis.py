"""Error analysis (Stage 11): inspect the wrongly classified resumes.

Outputs:
  outputs/reports/errors_best_model.csv   - every wrong prediction + preview
  outputs/reports/error_analysis.json     - grouped findings
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import MODEL_DIR, REPORT_DIR  # noqa: E402
from src.data_loading import load_data  # noqa: E402
from src.train_classical import clean_df  # noqa: E402


def load_best_preds():
    """Pick the model with the highest test macro-F1 and return its preds."""
    comp = pd.read_csv(REPORT_DIR / "model_comparison.csv")
    best = comp.iloc[0]["model"]
    if "TF-IDF" in best:
        p = pd.read_csv(REPORT_DIR / "preds_classical_test.csv")
    elif "Word2Vec" in best:
        p = pd.read_csv(REPORT_DIR / "preds_w2v_mlp.csv")
    else:
        p = pd.read_csv(REPORT_DIR / "preds_bilstm_test.csv")
    p = p[p["split"] == "test"].copy()
    return best, p


def main():
    df = clean_df(load_data()).set_index("ID")
    best_name, preds = load_best_preds()

    wrong = preds[preds["true"] != preds["pred"]].copy()
    wrong["text_preview"] = [
        " ".join(str(df.loc[i, "text"]).split())[:220] + "..."
        for i in wrong["id"]]
    wrong["correct"] = False
    right = preds[preds["true"] == preds["pred"]].copy()
    right["correct"] = True
    allp = pd.concat([wrong, right])
    allp["n_words"] = [
        len(str(df.loc[i, "text"]).split()) for i in allp["id"]]
    wrong = allp[~allp["correct"]]
    right = allp[allp["correct"]]

    # --- pattern analysis ----------------------------------------------
    analysis = {
        "best_model": best_name,
        "test_n": int(len(preds)),
        "wrong_n": int(len(wrong)),
        "error_rate": round(len(wrong) / len(preds), 4),
        "mean_confidence_wrong": round(
            float(wrong["confidence"].mean()), 4),
        "mean_confidence_right": round(
            float(right["confidence"].mean()), 4),
        "mean_words_wrong": round(float(wrong["n_words"].mean()), 1),
        "mean_words_right": round(float(right["n_words"].mean()), 1),
        "errors_below_0_6_confidence": int(
            (wrong["confidence"] < 0.6).sum()),
    }

    # confusion pairs
    pairs = (wrong.groupby(["true", "pred"]).size()
             .reset_index(name="n")
             .sort_values("n", ascending=False))
    analysis["top_confusion_pairs"] = [
        {"actual": r["true"], "predicted": r["pred"], "count": int(r["n"])}
        for _, r in pairs.head(8).iterrows()]

    # per-class error rate (which classes suffer most)
    err_rate = (wrong.groupby("true").size()
                / preds.groupby("true").size()).fillna(0)
    analysis["worst_error_rate_classes"] = {
        k: round(float(v), 3)
        for k, v in err_rate.sort_values(ascending=False).head(6).items()}

    # minority classes
    analysis["minority_class_support"] = {
        k: int(v) for k, v in
        preds.groupby("true").size().sort_values().head(5).items()}

    wrong.sort_values(["true", "pred", "confidence"],
                      ascending=[True, True, False]).to_csv(
        REPORT_DIR / "errors_best_model.csv", index=False)
    (REPORT_DIR / "error_analysis.json").write_text(
        json.dumps(analysis, indent=2, default=str), encoding="utf-8")

    print("=" * 78)
    print(f"ERROR ANALYSIS - {best_name}")
    print("=" * 78)
    print(f"test={analysis['test_n']}  wrong={analysis['wrong_n']} "
          f"({analysis['error_rate']*100:.1f}%)")
    print(f"mean confidence: wrong={analysis['mean_confidence_wrong']} "
          f"right={analysis['mean_confidence_right']}")
    print(f"mean words:      wrong={analysis['mean_words_wrong']} "
          f"right={analysis['mean_words_right']}")
    print(f"errors with confidence<0.6: {analysis['errors_below_0_6_confidence']}")
    print("\nTop confusion pairs (actual -> predicted : count):")
    for c in analysis["top_confusion_pairs"]:
        print(f"  {c['actual']:24s} -> {c['predicted']:24s} : {c['count']}")
    print("\nWorst error-rate classes:")
    for k, v in analysis["worst_error_rate_classes"].items():
        print(f"  {k:24s} {v*100:.1f}% of its test resumes misclassified")

    print("\nSample wrong predictions (top 8 by confusion pair):")
    shown = set()
    for _, r in pairs.head(4).iterrows():
        sub = wrong[(wrong["true"] == r["true"]) &
                    (wrong["pred"] == r["pred"])].head(2)
        for _, w in sub.iterrows():
            print("-" * 78)
            print(f"ID {int(w['id'])} | actual={w['true']} | "
                  f"predicted={w['pred']} | conf={w['confidence']}")
            print(f"  {w['text_preview']}")
    return analysis


if __name__ == "__main__":
    main()
