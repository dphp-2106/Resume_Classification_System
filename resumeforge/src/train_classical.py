"""Classical ML baselines with TF-IDF (Stages 6-8).

Compares:
  preprocessing modes : light / standard(stopwords) / lemma
  ngram ranges        : (1,1) unigrams  vs  (1,2) unigram+bigram
  models              : MultinomialNB / LogisticRegression / LinearSVC
Selection rule: best macro-F1 on the VALIDATION set (test stays untouched).
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import MODEL_DIR, REPORT_DIR, RANDOM_STATE, TEST_SIZE, VAL_SIZE  # noqa: E402
from src.data_loading import load_data  # noqa: E402
from src.preprocessing import preprocess  # noqa: E402

warnings.filterwarnings("ignore")

MODES = ["light", "standard", "lemma"]
NGRAMS = [(1, 1), (1, 2)]


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the 1 empty resume and exact duplicate texts (justified)."""
    df = df[df["text"].str.split().str.len() >= 3].copy()
    df = df.drop_duplicates(subset=["text"], keep="first").copy()
    return df


def build_models():
    return {
        "MultinomialNB": MultinomialNB(),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "LinearSVC": LinearSVC(
            class_weight="balanced", random_state=RANDOM_STATE),
    }


def main():
    df = clean_df(load_data())
    print(f"cleaned dataset: {len(df)} resumes, {df['Category'].nunique()} classes")

    # ---- stratified split 70/15/15 (split BEFORE fitting any vectorizer)
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        df.index, df["Category"], test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=df["Category"])
    rel_val = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=rel_val, random_state=RANDOM_STATE,
        stratify=y_tmp)
    print(f"split -> train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    raw = {part: df.loc[idxs, "text"].tolist()
           for part, idxs in [("train", X_train), ("val", X_val), ("test", X_test)]}

    rows = []
    best = {"val_macro_f1": -1}
    texts_by_mode = {}

    for mode in MODES:
        texts = texts_by_mode.setdefault(mode, {
            part: [preprocess(df.loc[idx, "text"], mode) for idx in idxs]
            for part, idxs in [("train", X_train), ("val", X_val), ("test", X_test)]})
        for ng in NGRAMS:
            # token_pattern=r"\S+" : the preprocessor already emitted clean
            # tokens (incl. C++, C#, .NET); re-tokenising with the default
            # pattern would silently drop them
            vec = TfidfVectorizer(ngram_range=ng, min_df=2, max_df=0.9,
                                  sublinear_tf=True, token_pattern=r"\S+")
            Xtr = vec.fit_transform(texts["train"])   # fit on TRAIN ONLY
            Xva = vec.transform(texts["val"])
            Xte = vec.transform(texts["test"])

            for name, model in build_models().items():
                model.fit(Xtr, y_train)
                for part, X, y in [("val", Xva, y_val), ("test", Xte, y_test)]:
                    pred = model.predict(X)
                    rows.append({
                        "preprocess": mode, "ngram": str(ng), "model": name,
                        "split": part,
                        "accuracy": round(accuracy_score(y, pred), 4),
                        "precision_macro": round(precision_score(y, pred, average="macro", zero_division=0), 4),
                        "recall_macro": round(recall_score(y, pred, average="macro", zero_division=0), 4),
                        "f1_macro": round(f1_score(y, pred, average="macro", zero_division=0), 4),
                        "f1_weighted": round(f1_score(y, pred, average="weighted", zero_division=0), 4),
                    })
                # track best on VALIDATION only
                pred = model.predict(Xva)
                vf1 = f1_score(y_val, pred, average="macro", zero_division=0)
                if vf1 > best["val_macro_f1"]:
                    best = {
                        "val_macro_f1": round(float(vf1), 4),
                        "preprocess": mode, "ngram": str(ng), "model": name,
                        "vectorizer": vec, "fitted_model": model,
                        "train_index": X_train.tolist(),
                        "val_index": X_val.tolist(),
                        "test_index": X_test.tolist(),
                    }

    results = pd.DataFrame(rows)
    results.to_csv(REPORT_DIR / "classical_results.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nAll runs (test metrics):")
    print(results[results["split"] == "test"].sort_values("f1_macro", ascending=False).to_string(index=False))

    print(f"\nBEST on validation: {best['model']} | {best['preprocess']} | "
          f"ngram={best['ngram']} | val macro-F1={best['val_macro_f1']}")

    # persist the best classical config (vectorizer + model + split indices)
    import joblib
    joblib.dump({
        "preprocess_mode": best["preprocess"],
        "ngram": tuple(int(x) for x in best["ngram"].strip("()").split(",")),
        "model_name": best["model"],
        "vectorizer": best["vectorizer"],
        "model": best["fitted_model"],
        "train_index": best["train_index"],
        "val_index": best["val_index"],
        "test_index": best["test_index"],
        "val_macro_f1": best["val_macro_f1"],
    }, MODEL_DIR / "best_classical.joblib")
    print(f"saved -> {MODEL_DIR / 'best_classical.joblib'}")

    # class-wise TF-IDF vocabulary plot (EDA - train split only)
    from sklearn.preprocessing import LabelEncoder
    from src.eda import plot_class_tfidf_terms
    le = LabelEncoder().fit(df["Category"])
    plot_class_tfidf_terms(
        best["vectorizer"].transform(texts["train"]),
        np.array(best["vectorizer"].get_feature_names_out()),
        None, le.transform(y_train))

    # top terms per class for interpretability (from best vectorizer/model)
    try:
        vec = best["vectorizer"]
        feat = np.array(vec.get_feature_names_out())
        if hasattr(best["fitted_model"], "coef_"):
            top_terms = {}
            for i, cls in enumerate(best["fitted_model"].classes_):
                w = np.asarray(best["fitted_model"].coef_[i]).ravel()
                top_terms[cls] = feat[np.argsort(w)[-15:][::-1]].tolist()
            (REPORT_DIR / "tfidf_top_terms_per_class.json").write_text(
                json.dumps(top_terms, indent=2), encoding="utf-8")
            print("\nTop TF-IDF terms per class (Linear model coefficients):")
            for cls, terms in top_terms.items():
                print(f"  {cls:24s}: {', '.join(terms[:10])}")
    except Exception as e:
        print("term inspection skipped:", e)

    # dump val+test predictions of the best config for error analysis
    texts = texts_by_mode[best["preprocess"]]

    def _proba(model, X):
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)
        d = model.decision_function(X)          # LinearSVC
        proba = np.exp(d - d.max(axis=1, keepdims=True))
        return proba / proba.sum(axis=1, keepdims=True)

    preds = []
    for part, idxs in [("val", best["val_index"]), ("test", best["test_index"])]:
        if part == "val":
            X = best["vectorizer"].transform(texts["val"])
        else:
            X = best["vectorizer"].transform(texts["test"])
        p = _proba(best["fitted_model"], X)
        y_true = df.loc[idxs, "Category"].values
        pred = best["fitted_model"].predict(X)
        preds.append(pd.DataFrame({
            "id": df.loc[idxs, "ID"].values,
            "true": y_true,
            "pred": pred,
            "confidence": p.max(axis=1).round(4),
            "split": part,
        }))
    pd.concat(preds).to_csv(REPORT_DIR / "preds_classical_test.csv", index=False)
    print("saved ->", REPORT_DIR / "preds_classical_test.csv")

    return results, best


if __name__ == "__main__":
    main()
