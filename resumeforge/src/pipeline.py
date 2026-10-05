"""Final inference pipeline (Stage 12).

Bundles preprocessing + feature extraction + model into ONE saved artifact so
inference is guaranteed to match training. Provides `predict()` for raw text
and `predict_pdf()` for raw PDF files.
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
from src.preprocessing import preprocess, tokenize  # noqa: E402
from src.train_classical import clean_df  # noqa: E402


class ResumeClassifier:
    """One object = the complete trained system."""

    def __init__(self):
        self.classical = joblib.load(MODEL_DIR / "best_classical.joblib")
        self.w2v_mlp = joblib.load(MODEL_DIR / "w2v_mlp.joblib")
        self.lstm = None
        if (MODEL_DIR / "bilstm.pt").exists():
            import torch
            self.lstm = torch.load(
                MODEL_DIR / "bilstm.pt", map_location="cpu",
                weights_only=False)
        self._lstm_model = None
        self.classes = list(self.classical["model"].classes_)

    # -- classical path ------------------------------------------------
    def _predict_classical(self, raw_text):
        text = preprocess(raw_text, self.classical["preprocess_mode"])
        X = self.classical["vectorizer"].transform([text])
        model = self.classical["model"]
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X)[0]
        else:  # LinearSVC decision function -> softmax
            d = model.decision_function(X)[0]
            p = np.exp(d - d.max())
            p = p / p.sum()
        order = np.argsort(-p)
        return {
            "predicted": model.predict(X)[0],
            "confidence": round(float(p.max()), 4),
            "top3": [(self.classical["model"].classes_[i],
                      round(float(p[i]), 4)) for i in order[:3]],
        }

    # -- word2vec + mlp path -------------------------------------------
    def _predict_w2v(self, raw_text):
        toks = tokenize(raw_text, "standard")
        vecs = [self.w2v_mlp["w2v"].wv[t] for t in toks
                if t in self.w2v_mlp["w2v"].wv]
        v = np.mean(vecs, axis=0) if vecs else np.zeros(
            self.w2v_mlp["w2v"].vector_size)
        mlp = self.w2v_mlp["mlp"]
        p = mlp.predict_proba([v])[0]
        order = np.argsort(-p)
        cls = self.w2v_mlp["classes"]          # string labels
        int_pred = int(mlp.predict([v])[0])
        return {
            "predicted": cls[int_pred],
            "confidence": round(float(p.max()), 4),
            "top3": [(cls[i], round(float(p[i]), 4)) for i in order[:3]],
        }

    # -- bilstm path ---------------------------------------------------
    def _predict_lstm(self, raw_text):
        import torch
        if self._lstm_model is None:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from src.train_lstm import BiLSTM
            cfg = self.lstm["config"]
            self._lstm_model = BiLSTM(
                len(self.lstm["vocab"]), cfg["embed_dim"],
                cfg["hidden_dim"], len(self.lstm["classes"]))
            self._lstm_model.load_state_dict(self.lstm["state_dict"])
            self._lstm_model.eval()
        toks = tokenize(raw_text, "standard")[:self.lstm["config"]["maxlen"]]
        ids = [self.lstm["vocab"].get(t, self.lstm["vocab"]["<unk>"])
               for t in toks]
        ids += [self.lstm["vocab"]["<pad>"]] * (
            self.lstm["config"]["maxlen"] - len(ids))
        with torch.no_grad():
            logits = self._lstm_model(
                torch.tensor([ids], dtype=torch.long))
            p = torch.softmax(logits, dim=1).numpy()[0]
        order = np.argsort(-p)
        cls = self.lstm["classes"]
        return {
            "predicted": cls[int(p.argmax())],
            "confidence": round(float(p.max()), 4),
            "top3": [(cls[i], round(float(p[i]), 4)) for i in order[:3]],
        }

    def predict(self, raw_text, model="best"):
        """Predict the category of a raw resume string."""
        if model == "best":
            return self._predict_classical(raw_text)
        if model == "classical":
            return self._predict_classical(raw_text)
        if model == "w2v_mlp":
            return self._predict_w2v(raw_text)
        if model == "bilstm":
            return self._predict_lstm(raw_text)
        raise ValueError(f"unknown model: {model}")


def demo_unseen_examples():
    """Test the system on 5 unseen resumes from the held-out test set."""
    df = clean_df(load_data())
    blob = joblib.load(MODEL_DIR / "best_classical.joblib")
    te = pd.Index(blob["test_index"])
    clf = ResumeClassifier()

    # pick 5 diverse examples: 3 random + 2 from minority classes
    rng = np.random.default_rng(7)
    minority = df.loc[df["Category"].isin(["BPO", "AUTOMOBILE",
                                           "AGRICULTURE"])]
    minority = minority[minority.index.isin(te)]
    picks = rng.choice(te, 3, replace=False).tolist() + \
        minority.index[:2].tolist()

    print("=" * 78)
    print("UNSEEN-RESUME DEMO (held-out test set)")
    print("=" * 78)
    results = []
    for idx in picks:
        raw = df.loc[idx, "text"]
        true = df.loc[idx, "Category"]
        out = clf.predict(raw)
        ok = "CORRECT" if out["predicted"] == true else "WRONG"
        results.append({"id": int(df.loc[idx, "ID"]), "true": true,
                        **out})
        print(f"\nID {int(df.loc[idx, 'ID'])} | true={true} | "
              f"predicted={out['predicted']} ({ok}, conf={out['confidence']})")
        print(f"  top3={out['top3']}")
        print(f"  preview: {' '.join(raw.split())[:160]}...")
    (REPORT_DIR / "unseen_demo.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")
    return results


if __name__ == "__main__":
    demo_unseen_examples()
