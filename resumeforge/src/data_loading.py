"""Data gathering + data-quality checks (Stage 2 & 3 of the approach guide).

Produces:
  outputs/reports/data_quality_report.json  - every check, counts and percentages
  console summary
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import REPORT_DIR, RESUME_CSV  # noqa: E402


def load_data(path: str | Path = RESUME_CSV) -> pd.DataFrame:
    """Load the resume dataset and add derived length columns."""
    df = pd.read_csv(path)
    df["text"] = df["Resume_str"].fillna("").astype(str)
    df["n_char"] = df["text"].str.len()
    df["n_word"] = df["text"].str.split().str.len()
    return df


def quality_checks(df: pd.DataFrame) -> dict:
    """Run every data-quality check required by the guidelines."""
    rep: dict = {}

    rep["n_rows"] = int(len(df))
    rep["n_columns"] = int(df.shape[1])
    rep["columns"] = list(df.columns)

    # --- missing values ------------------------------------------------
    rep["missing_values"] = {c: int(df[c].isna().sum()) for c in df.columns}

    # --- empty / whitespace-only resumes --------------------------------
    empty_mask = df["text"].str.strip().eq("")
    rep["empty_resumes"] = {
        "count": int(empty_mask.sum()),
        "pct": round(100 * empty_mask.mean(), 2),
        "ids": df.loc[empty_mask, "ID"].astype(int).tolist(),
    }

    # --- duplicates ------------------------------------------------------
    rep["duplicates"] = {
        "duplicate_ids": int(df["ID"].duplicated().sum()),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "duplicate_resume_text": int(df["text"].duplicated().sum()),
        "duplicate_text_within_same_class": int(
            df.duplicated(subset=["Category", "text"]).sum()
        ),
    }
    dup_ids = df.loc[df["text"].duplicated(keep=False), "ID"].astype(int).tolist()
    rep["duplicates"]["duplicate_ids_list"] = dup_ids

    # --- noisy text signals ----------------------------------------------
    rep["noise_signals"] = {
        "contains_html_tag": int(
            df["text"].str.contains(r"<[a-zA-Z/][^>]*>", regex=True).sum()
        ),
        "contains_html_entity": int(
            df["text"].str.contains(r"&(amp|lt|gt|nbsp|quot|#\d+);", regex=True).sum()
        ),
        "contains_url": int(df["text"].str.contains(r"https?://\S+", regex=True).sum()),
        "contains_email": int(
            df["text"].str.contains(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
                                   regex=True).sum()
        ),
        "contains_phone": int(
            df["text"].str.contains(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
                                    regex=True).sum()
        ),
        "non_ascii_heavy_rows": int(
            df["text"].apply(lambda t: sum(ord(ch) > 127 for ch in t) > 0.05 * max(len(t), 1)).sum()
        ),
    }

    # --- class distribution ----------------------------------------------
    vc = df["Category"].value_counts()
    rep["class_distribution"] = {k: int(v) for k, v in vc.items()}
    rep["n_classes"] = int(df["Category"].nunique())
    rep["class_balance"] = {
        "minority_class": vc.idxmin(),
        "minority_count": int(vc.min()),
        "majority_class": vc.idxmax(),
        "majority_count": int(vc.max()),
        "imbalance_ratio": round(vc.max() / vc.min(), 2),
    }

    # --- text-length statistics ------------------------------------------
    rep["text_length"] = {
        "char": df["n_char"].describe().round(1).to_dict(),
        "word": df["n_word"].describe().round(1).to_dict(),
    }

    # --- label consistency -----------------------------------------------
    rep["label_consistency"] = {
        "unique_labels": sorted(df["Category"].unique().tolist()),
        "labels_match_expected_24": bool(df["Category"].nunique() == 24),
    }

    # --- leakage check: identical text across DIFFERENT classes ----------
    cross = df.groupby("text")["Category"].nunique()
    rep["leakage_check"] = {
        "texts_appearing_in_multiple_classes": int((cross > 1).sum()),
        "note": "0 => no identical resume is labelled with two different classes",
    }
    return rep


def main() -> pd.DataFrame:
    df = load_data()
    rep = quality_checks(df)
    (REPORT_DIR / "data_quality_report.json").write_text(
        json.dumps(rep, indent=2, default=str), encoding="utf-8"
    )

    print("=" * 70)
    print("DATA QUALITY REPORT  (SAMATRIX RESUMEFORGE 2026)")
    print("=" * 70)
    print(f"rows={rep['n_rows']}  columns={rep['n_columns']}  classes={rep['n_classes']}")
    print(f"missing values      : {rep['missing_values']}")
    print(f"empty resumes       : {rep['empty_resumes']['count']} "
          f"({rep['empty_resumes']['pct']}%) ids={rep['empty_resumes']['ids']}")
    print(f"duplicates          : text={rep['duplicates']['duplicate_resume_text']} "
          f"(same-class pairs), ids={rep['duplicates']['duplicate_ids_list']}")
    print(f"noise: html={rep['noise_signals']['contains_html_tag']} "
          f"entity={rep['noise_signals']['contains_html_entity']} "
          f"url={rep['noise_signals']['contains_url']} "
          f"email={rep['noise_signals']['contains_email']} "
          f"phone={rep['noise_signals']['contains_phone']}")
    print(f"imbalance: {rep['class_balance']}")
    print(f"leakage: texts in multiple classes = "
          f"{rep['leakage_check']['texts_appearing_in_multiple_classes']}")
    print(f"\nword-count stats:\n{pd.DataFrame(rep['text_length']['word']).T}")
    print(f"\nreport saved -> {REPORT_DIR / 'data_quality_report.json'}")
    return df


if __name__ == "__main__":
    main()
