"""
SAMATRIX RESUMEFORGE 2026
Step 6: Final Prediction Pipeline
Inference with the trained model — identical preprocessing as training.
"""

import sys, io, os, re, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
from config import MODELS_DIR

import numpy as np
import joblib
import json
from gensim.models import Word2Vec
from preprocessing_utils import clean_resume

# ════════════════════════════════════════════════════════════════════
# Load pipeline artifacts
# ════════════════════════════════════════════════════════════════════
_MODELS_DIR = 'outputs/models'
_ARTIFACTS_LOADED = False

def _load_artifacts():
    global le, best_clf, logreg_clf, best_vec, w2v_model, mlp, scaler, _ARTIFACTS_LOADED
    le         = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.pkl'))
    best_clf   = joblib.load(os.path.join(MODELS_DIR, 'best_classical_model.pkl'))
    best_vec   = joblib.load(os.path.join(MODELS_DIR, 'best_classical_vectorizer.pkl'))
    
    try:
        logreg_clf = joblib.load(os.path.join(MODELS_DIR, 'Logistic_Regression_bi.pkl'))
    except Exception:
        logreg_clf = None

    try:
        from gensim.models import Word2Vec
        w2v_model = Word2Vec.load(os.path.join(MODELS_DIR, 'word2vec.model'))
        mlp       = joblib.load(os.path.join(MODELS_DIR, 'w2v_mlp_classifier.pkl'))
        scaler    = joblib.load(os.path.join(MODELS_DIR, 'w2v_scaler.pkl'))
        _has_dl = True
    except Exception:
        w2v_model = mlp = scaler = None
        _has_dl = False

    _ARTIFACTS_LOADED = True
    return _has_dl


def predict_resume(raw_text: str, ensemble: bool = True):
    """
    Predict resume category from raw text using Calibrated Adaptive Ensemble.
    Combines linear models (zero-bias immune), LightGBM (high capacity), and Word2Vec (dense semantic).

    Args:
        raw_text: Raw resume text (not pre-cleaned)
        ensemble: If True, uses adaptive ensemble; if False, uses classical model only.

    Returns:
        dict with keys:
          - predicted_category: str
          - confidence: float
          - top_5: list of (category, score) tuples
          - method: str
    """
    if not _ARTIFACTS_LOADED:
        _load_artifacts()

    # Step 1: Preprocessing (identical to training)
    cleaned = clean_resume(raw_text, lemmatize=True)
    if not cleaned.strip():
        return {
            'predicted_category': 'UNKNOWN',
            'confidence': 0.0,
            'top_5': [],
            'method': 'none',
            'warning': 'Resume text is too short or empty after cleaning.'
        }

    words = cleaned.split()
    word_count = len(words)

    # Step 2: Classical TF-IDF predictions
    X_tfidf = best_vec.transform([cleaned])
    try:
        proba_tree = best_clf.predict_proba(X_tfidf)[0]
    except AttributeError:
        scores = best_clf.decision_function(X_tfidf)[0]
        scores -= scores.max()
        proba_tree = np.exp(scores) / np.exp(scores).sum()

    if logreg_clf is not None:
        try:
            proba_linear = logreg_clf.predict_proba(X_tfidf)[0]
        except AttributeError:
            scores = logreg_clf.decision_function(X_tfidf)[0]
            scores -= scores.max()
            proba_linear = np.exp(scores) / np.exp(scores).sum()
    else:
        proba_linear = proba_tree

    # Step 3: W2V + MLP prediction (if available)
    proba_dl = None
    if ensemble and w2v_model is not None and mlp is not None:
        vecs = [w2v_model.wv[t] for t in words if t in w2v_model.wv]
        if vecs:
            doc_vec = np.mean(vecs, axis=0).reshape(1, -1)
            doc_vec_scaled = scaler.transform(doc_vec)
            proba_dl = mlp.predict_proba(doc_vec_scaled)[0]

    # Step 4: Calibrated Adaptive Ensemble
    if ensemble and proba_dl is not None:
        if word_count < 80:
            # Short text: linear model + dense semantics dominate (immune to sparse default splits)
            proba_final = 0.50 * proba_linear + 0.35 * proba_dl + 0.15 * proba_tree
            method = 'Adaptive Ensemble (Linear + W2V-MLP + LightGBM)'
        else:
            # Full resume: high-capacity LightGBM + Linear + W2V-MLP
            proba_final = 0.45 * proba_tree + 0.35 * proba_linear + 0.20 * proba_dl
            method = 'Adaptive Ensemble (LightGBM + Linear + W2V-MLP)'
    elif ensemble and logreg_clf is not None:
        proba_final = 0.55 * proba_tree + 0.45 * proba_linear
        method = 'Classical Ensemble (LightGBM + LogisticRegression)'
    else:
        proba_final = proba_tree
        method = 'Classical LightGBM'

    # Step 5: Decode top predictions
    predicted_idx = np.argmax(proba_final)
    predicted_cat = le.classes_[predicted_idx]
    confidence = float(proba_final[predicted_idx])

    top_5_idx = np.argsort(proba_final)[::-1][:5]
    top_5 = [(le.classes_[i], float(proba_final[i])) for i in top_5_idx]

    return {
        'predicted_category': predicted_cat,
        'confidence': confidence,
        'top_5': top_5,
        'method': method
    }


# ════════════════════════════════════════════════════════════════════
# Batch prediction
# ════════════════════════════════════════════════════════════════════
def predict_batch(texts: list, ensemble: bool = True):
    """Predict multiple resumes at once."""
    return [predict_resume(t, ensemble=ensemble) for t in texts]


# ════════════════════════════════════════════════════════════════════
# CLI test
# ════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 65)
    print("  SAMATRIX RESUMEFORGE 2026 — Prediction Pipeline Test")
    print("=" * 65)

    _has_dl = _load_artifacts()
    print(f"  Classical model loaded: YES")
    print(f"  Deep Learning (W2V+MLP): {'YES' if _has_dl else 'NO (run 04_deep_learning.py first)'}")
    print(f"  Label encoder: {len(le.classes_)} classes")

    # Test resumes
    test_resumes = {
        "Software Engineer": """
        Experienced software engineer with 5 years of experience in Python, Java, and AWS.
        Proficient in machine learning, TensorFlow, Docker, Kubernetes. Built REST APIs
        and microservices. BS Computer Science. Skills: SQL, MongoDB, React, Git, Linux.
        Worked on NLP projects including text classification and sentiment analysis.
        """,

        "Doctor / Healthcare": """
        Medical doctor with 8 years experience in internal medicine and patient care.
        MBBS from State Medical University. Skilled in diagnosis, treatment planning,
        EMR systems. Experience in ICU, ER, outpatient consultations. Completed
        residency in cardiology. Published research in cardiology journals.
        """,

        "Financial Analyst": """
        CFA Level 2 candidate with 4 years experience in equity research and financial
        modeling. Proficient in Excel, Bloomberg, Python for data analysis. Strong
        understanding of DCF valuation, financial statements, SEC filings.
        Managed portfolio of $50M assets. MBA Finance from top-tier business school.
        """,

        "HR Manager": """
        Human Resources professional with 7 years experience in talent acquisition,
        performance management, and employee relations. SHRM-CP certified.
        Implemented HRIS systems, onboarding programs. Expertise in labor law compliance,
        compensation benchmarking, diversity initiatives. Managed team of 5 HR specialists.
        """,

        "Executive Chef": """
        Executive Chef with 12 years experience in fine dining restaurants. Expert in
        French cuisine, menu development, kitchen management. Managed team of 20 cooks.
        Certified by Culinary Institute of America. Expertise in food costing, inventory
        management, HACCP compliance. Featured in Food & Wine magazine.
        """
    }

    print("\n[Prediction Results]")
    print("-" * 65)
    for title, resume_text in test_resumes.items():
        result = predict_resume(resume_text, ensemble=_has_dl)
        print(f"\n  Input type: {title}")
        print(f"  Predicted:  {result['predicted_category']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Method:     {result['method']}")
        print(f"  Top-3: {[(c, f'{s:.3f}') for c, s in result['top_5'][:3]]}")

    print("\n" + "=" * 65)
    print("  PREDICTION PIPELINE TEST COMPLETE")
    print("=" * 65)
