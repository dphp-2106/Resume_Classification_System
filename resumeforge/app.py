"""Streamlit demo - SAMATRIX RESUMEFORGE 2026.

Run:  streamlit run app.py
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.pipeline import ResumeClassifier  # noqa: E402

st.set_page_config(page_title="ResumeForge 2026",
                   page_icon="📄", layout="wide")

st.title("📄 SAMATRIX ResumeForge 2026")
st.markdown("""
**Resume Classification System** — end-to-end NLP pipeline.
Raw resume text → preprocessing → features → model → predicted category
(24 job categories).
""")


@st.cache_resource
def get_classifier():
    return ResumeClassifier()


st.sidebar.header("Model")
model_choice = st.sidebar.radio(
    "Choose model",
    ["best (TF-IDF + classical)",
     "TF-IDF + classical",
     "Word2Vec + Dense MLP",
     "Embedding + BiLSTM"],
)
MODEL_MAP = {
    "best (TF-IDF + classical)": "best",
    "TF-IDF + classical": "classical",
    "Word2Vec + Dense MLP": "w2v_mlp",
    "Embedding + BiLSTM": "bilstm",
}

st.sidebar.header("Or try a sample")
if st.sidebar.button("Load a random held-out resume"):
    from src.data_loading import load_data
    from src.train_classical import clean_df
    blob = joblib.load(Path("outputs/models/best_classical.joblib"))
    df = clean_df(load_data())
    te = pd.Index(blob["test_index"])
    idx = te[len(st.session_state.get("n", 0)) % len(te)]
    st.session_state["n"] = st.session_state.get("n", 0) + 1
    st.session_state["text"] = df.loc[idx, "text"]
    st.session_state["true"] = df.loc[idx, "Category"]

text = st.text_area("Paste resume text here (or upload a .txt):",
                    height=260,
                    value=st.session_state.get("text", ""))

uploaded = st.file_uploader("Upload a .txt resume", type=["txt"])
if uploaded:
    text = uploaded.read().decode("utf-8", errors="ignore")

if st.button("🔮 Classify"):
    if not text.strip():
        st.warning("Please paste or upload resume text first.")
    else:
        clf = get_classifier()
        with st.spinner("Preprocessing → vectorising → predicting..."):
            out = clf.predict(text, model=MODEL_MAP[model_choice])
        col1, col2 = st.columns([1, 2])
        col1.metric("Predicted category", out["predicted"])
        col2.metric("Confidence", f"{out['confidence']:.1%}")
        if st.session_state.get("true"):
            mark = "✅" if out["predicted"] == st.session_state["true"] \
                else "❌"
            st.info(f"{mark} Held-out true label: "
                    f"{st.session_state['true']}")
        st.subheader("Top-3 categories")
        chart = pd.DataFrame(out["top3"],
                             columns=["category", "probability"])
        st.bar_chart(chart.set_index("category"))
        with st.expander("How does it work?"):
            st.markdown("""
            1. **Preprocess** - lowercase, strip HTML/URLs/emails/phones,
               keep technical tokens (C++, .NET), remove stopwords.
            2. **Features** - TF-IDF unigram+bigram (or Word2Vec mean-pool /
               BiLSTM sequence).
            3. **Model** - class-balanced LinearSVC chosen by macro-F1 on the
               validation set (never the test set).
            """)

st.sidebar.markdown("---")
st.sidebar.caption("24 categories • trained on 2,481 resumes "
                   "(1 empty + 2 duplicates removed)")
