"""Exploratory Data Analysis (Stage 4) - the most important part.

Answers, with plots saved under outputs/eda:
  1. What does a resume look like?      -> sample lengths, top words, wordcloud
  2. How are classes distributed?        -> class bar chart, imbalance ratio
  3. Which words/phrases distinguish?    -> n-grams, class-wise TF-IDF terms
  4. Where is the data noisy?            -> empty/duplicate/contact-info bars
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CLASSES, EDA_DIR, REPORT_DIR  # noqa: E402
from src.data_loading import load_data, quality_checks  # noqa: E402
from src.preprocessing import preprocess  # noqa: E402

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight"})

try:
    from wordcloud import WordCloud
    _WC_OK = True
except Exception:
    _WC_OK = False


def plot_class_distribution(df: pd.DataFrame):
    vc = df["Category"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#c0392b" if v < 70 else "#2c6fbb" for v in vc.values]
    ax.bar(vc.index, vc.values, color=colors)
    ax.axhline(vc.mean(), ls="--", c="gray", lw=1, label=f"mean={vc.mean():.0f}")
    ax.set_title("Resume count per category (red = minority class < 70)")
    ax.set_ylabel("resumes")
    ax.set_xticks(range(len(vc)))
    ax.set_xticklabels(vc.index, rotation=60, ha="right", fontsize=8)
    ax.legend()
    fig.savefig(EDA_DIR / "01_class_distribution.png")
    plt.close(fig)
    return vc


def plot_text_length(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(df["n_char"], bins=50, color="#2c6fbb", edgecolor="white")
    axes[0].set_title("Resume length - characters")
    axes[0].set_xlabel("characters")
    axes[1].hist(df["n_word"], bins=50, color="#27ae60", edgecolor="white")
    axes[1].set_title("Resume length - words")
    axes[1].set_xlabel("words")
    fig.suptitle("Text-length distribution (all 2,484 resumes)")
    fig.savefig(EDA_DIR / "02_text_length_hist.png")
    plt.close(fig)

    # per-class word-count boxplot (sorted by median)
    order = df.groupby("Category")["n_word"].median().sort_values().index
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=df, x="Category", y="n_word", order=order,
                hue="Category", legend=False, palette="viridis", ax=ax)
    ax.set_xticklabels(order, rotation=60, ha="right", fontsize=8)
    ax.set_title("Words per resume by category (sorted by median)")
    fig.savefig(EDA_DIR / "03_text_length_by_class.png")
    plt.close(fig)


def plot_top_words(tokens_by_doc: list[list[str]]):
    counter = Counter(t for doc in tokens_by_doc for t in doc)
    top = counter.most_common(30)
    words, counts = zip(*top)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(words[::-1], counts[::-1], color="#8e44ad")
    ax.set_title("Top 30 words after light cleaning (stopwords kept for view)")
    ax.set_xlabel("frequency")
    fig.savefig(EDA_DIR / "04_top_words.png")
    plt.close(fig)
    return counter


def plot_wordclouds(tokens_by_doc: list[list[str]], df: pd.DataFrame):
    if not _WC_OK:
        return
    text_all = " ".join(" ".join(d) for d in tokens_by_doc)
    wc = WordCloud(width=1100, height=500, background_color="white",
                   max_words=200, collocations=False)
    wc.generate(text_all)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word cloud - full corpus")
    fig.savefig(EDA_DIR / "05_wordcloud_overall.png")
    plt.close(fig)

    # class-wise for the 4 largest + 2 smallest classes
    vc = df["Category"].value_counts()
    picks = list(vc.index[:4]) + ["BPO", "AUTOMOBILE"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, cat in zip(axes.ravel(), picks):
        doc = " ".join(" ".join(tokens_by_doc[i]) for i in df.index[df["Category"] == cat])
        wc = WordCloud(width=600, height=400, background_color="white",
                       max_words=120, collocations=False)
        wc.generate(doc)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"{cat} (n={int((df['Category'] == cat).sum())})")
    fig.suptitle("Class-wise word clouds")
    fig.savefig(EDA_DIR / "06_wordcloud_classwise.png")
    plt.close(fig)


def _ngram_counts(tokens_by_doc: list[list[str]], n: int, top_k: int = 25):
    c = Counter()
    for doc in tokens_by_doc:
        c.update(zip(*[doc[i:] for i in range(n)]))
    return c.most_common(top_k)


def plot_ngrams(tokens_by_doc: list[list[str]]):
    for n in (1, 2, 3):
        items = _ngram_counts(tokens_by_doc, n, 25)
        labels = [" ".join(g) for g, _ in items][::-1]
        vals = [v for _, v in items][::-1]
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.barh(labels, vals, color=["#16a085", "#d35400", "#2c3e50"][n - 1])
        ax.set_title(f"Top 25 {n}-grams (stopwords removed)")
        ax.set_xlabel("frequency")
        fig.savefig(EDA_DIR / f"07_top_{n}grams.png")
        plt.close(fig)


def plot_class_tfidf_terms(tfidf_matrix, feature_names, df, labels_enc, top_k: int = 12):
    """Class-wise vocabulary: mean TF-IDF weight per term per class."""
    fig, axes = plt.subplots(6, 4, figsize=(18, 22))
    for ci, cat in enumerate(CLASSES):
        idx = np.where(labels_enc == ci)[0]
        mean_tfidf = np.asarray(tfidf_matrix[idx].mean(axis=0)).ravel()
        top = np.argsort(mean_tfidf)[-top_k:][::-1]
        ax = axes.ravel()[ci]
        ax.barh([feature_names[i] for i in top][::-1], mean_tfidf[top][::-1],
                color="#2c6fbb")
        ax.set_title(f"{cat}  (n={len(idx)})", fontsize=10)
        ax.tick_params(labelsize=8)
    fig.suptitle("Class-distinguishing terms: mean TF-IDF weight per class", y=1.0)
    fig.tight_layout()
    fig.savefig(EDA_DIR / "08_class_tfidf_terms.png")
    plt.close(fig)


def plot_data_quality(rep: dict):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    # missing / empty / duplicates
    cats = ["missing\nvalues", "empty\nresumes", "duplicate\ntexts",
            "dup. ids", "html\ntags", "html\nentities"]
    vals = [
        sum(rep["missing_values"].values()),
        rep["empty_resumes"]["count"],
        rep["duplicates"]["duplicate_resume_text"],
        rep["duplicates"]["duplicate_ids"],
        rep["noise_signals"]["contains_html_tag"],
        rep["noise_signals"]["contains_html_entity"],
    ]
    axes[0].bar(cats, vals, color="#c0392b")
    axes[0].set_title("Data-quality issues (counts)")
    for i, v in enumerate(vals):
        axes[0].text(i, v, str(v), ha="center", va="bottom")
    # contact info prevalence
    cats2 = ["URLs", "emails", "phones"]
    vals2 = [rep["noise_signals"]["contains_url"],
             rep["noise_signals"]["contains_email"],
             rep["noise_signals"]["contains_phone"]]
    axes[1].bar(cats2, vals2, color="#7f8c8d")
    axes[1].set_title("Contact-info occurrences (non-discriminative, removed)")
    for i, v in enumerate(vals2):
        axes[1].text(i, v, str(v), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "09_data_quality.png")
    plt.close(fig)


def main():
    df = load_data()
    rep = quality_checks(df)
    (REPORT_DIR / "data_quality_report.json").write_text(
        json.dumps(rep, indent=2, default=str), encoding="utf-8")

    # tokens for word-level EDA: 'standard' mode (stopwords removed)
    tokens_by_doc = [preprocess(t, "standard").split() for t in df["text"]]

    plot_class_distribution(df)
    plot_text_length(df)
    counter = plot_top_words(tokens_by_doc)
    plot_wordclouds(tokens_by_doc, df)
    plot_ngrams(tokens_by_doc)
    plot_data_quality(rep)

    # quick numeric summary for the report
    summary = {
        "top_30_words": [w for w, _ in counter.most_common(30)],
        "top_bigrams": [" ".join(g) for g, _ in _ngram_counts(tokens_by_doc, 2, 15)],
        "top_trigrams": [" ".join(g) for g, _ in _ngram_counts(tokens_by_doc, 3, 15)],
        "median_words": float(df["n_word"].median()),
        "median_chars": float(df["n_char"].median()),
    }
    (REPORT_DIR / "eda_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print("EDA plots written to", EDA_DIR)
    for f in sorted(EDA_DIR.glob("*.png")):
        print("  -", f.name)
    print("\nTop 15 bigrams:", summary["top_bigrams"][:15])
    return df, rep


if __name__ == "__main__":
    main()
