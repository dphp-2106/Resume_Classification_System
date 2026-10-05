"""Neural approach #2: Embedding + BiLSTM (PyTorch).

Token sequences from the SAME preprocessed text and split as the other
models. Vocabulary and embedding are learned from the TRAIN split only.
Class-balanced cross-entropy handles the minority classes (BPO=22 ...).
"""
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score, precision_score, recall_score)
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import MODEL_DIR, REPORT_DIR, RANDOM_STATE  # noqa: E402
from src.data_loading import load_data  # noqa: E402
from src.preprocessing import preprocess  # noqa: E402
from src.train_classical import clean_df  # noqa: E402

EMBED_DIM = 96
HIDDEN_DIM = 128
NUM_LAYERS = 1
BATCH_SIZE = 64
EPOCHS = 30
LR = 2e-3
MAXLEN = 256          # proven stable config; median=565 tokens so
                      # truncation is a real limitation (noted)
MIN_FREQ = 3
PATIENCE = 10         # early stop on val macro-F1

torch.manual_seed(RANDOM_STATE)


class ResumeDataset(Dataset):
    def __init__(self, seqs, labels=None):
        self.seqs = seqs
        self.labels = labels

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        if self.labels is None:
            return torch.tensor(self.seqs[i], dtype=torch.long)
        return (torch.tensor(self.seqs[i], dtype=torch.long),
                self.labels[i])


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes,
                 pad_idx=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=NUM_LAYERS,
                            batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(2 * hidden_dim, n_classes)

    def forward(self, x):
        e = self.emb(x)
        out, _ = self.lstm(e)          # (B, T, 2H)
        out = self.drop(out)
        return self.head(out.mean(dim=1))   # mean pooling over time


def build_vocab(token_lists):
    counts = Counter(t for doc in token_lists for t in doc)
    words = ["<pad>", "<unk>"] + [w for w, c in counts.most_common()
                                    if c >= MIN_FREQ]
    return {w: i for i, w in enumerate(words)}


def encode(tokens, vocab, maxlen=MAXLEN):
    ids = [vocab.get(t, vocab["<unk>"]) for t in tokens][:maxlen]
    if len(ids) < maxlen:
        ids += [vocab["<pad>"]] * (maxlen - len(ids))
    return ids


@torch.no_grad()
def predict_proba(model, loader, device, n_classes):
    model.eval()
    probs = []
    for x in loader:
        x = x.to(device)
        logits = model(x)
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(probs)


def main():
    device = torch.device("cpu")
    df = clean_df(load_data())

    blob = joblib.load(MODEL_DIR / "best_classical.joblib")
    tr_idx, va_idx, te_idx = (pd.Index(blob["train_index"]),
                              pd.Index(blob["val_index"]),
                              pd.Index(blob["test_index"]))
    y_train, y_val, y_test = (df.loc[tr_idx, "Category"],
                              df.loc[va_idx, "Category"],
                              df.loc[te_idx, "Category"])

    tok_train = [preprocess(t, "standard").split() for t in df.loc[tr_idx, "text"]]
    tok_val = [preprocess(t, "standard").split() for t in df.loc[va_idx, "text"]]
    tok_test = [preprocess(t, "standard").split() for t in df.loc[te_idx, "text"]]

    lens = [len(t) for t in tok_train]
    print(f"token length: median={int(np.median(lens))} "
          f"p90={int(np.percentile(lens, 90))} max={max(lens)} "
          f"-> MAXLEN={MAXLEN} (keeps summary/skills at the head)")

    vocab = build_vocab(tok_train)
    print(f"vocab: {len(vocab)} (min_freq={MIN_FREQ})")

    le = LabelEncoder().fit(df["Category"])
    Ytr = le.transform(y_train)
    Yva = le.transform(y_val)
    Yte = le.transform(y_test)

    Xtr = [encode(t, vocab) for t in tok_train]
    Xva = [encode(t, vocab) for t in tok_val]
    Xte = [encode(t, vocab) for t in tok_test]

    loaders = {
        "train": DataLoader(ResumeDataset(Xtr, Ytr), batch_size=BATCH_SIZE,
                            shuffle=True),
        "val": DataLoader(ResumeDataset(Xva), batch_size=BATCH_SIZE),
        "test": DataLoader(ResumeDataset(Xte), batch_size=BATCH_SIZE),
    }

    model = BiLSTM(len(vocab), EMBED_DIM, HIDDEN_DIM, len(le.classes_)).to(device)
    class_counts = np.bincount(Ytr, minlength=len(le.classes_))
    weights = torch.tensor(len(Ytr) / (len(le.classes_) * class_counts),
                           dtype=torch.float)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # ---- training loop
    best_val_f1, best_state, bad_epochs = -1.0, None, 0
    for epoch in range(EPOCHS):
        model.train()
        tot, n = 0.0, 0
        for x, yb in loaders["train"]:
            x = x.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * x.size(0)
            n += x.size(0)
        # validation
        probs = predict_proba(model, loaders["val"], device, len(le.classes_))
        pred = probs.argmax(1)
        vf1 = f1_score(Yva, pred, average="macro")
        print(f"epoch {epoch+1:2d} | train loss {tot/n:.4f} | val macro-F1 {vf1:.4f}")
        if vf1 > best_val_f1:
            best_val_f1 = vf1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"early stop at epoch {epoch+1} "
                      f"(best val macro-F1 {best_val_f1:.4f})")
                break

    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "vocab": vocab,
                "classes": le.classes_.tolist(),
                "config": {"embed_dim": EMBED_DIM, "hidden_dim": HIDDEN_DIM,
                           "maxlen": MAXLEN, "min_freq": MIN_FREQ}},
               MODEL_DIR / "bilstm.pt")

    # ---- final metrics on val + test
    out = {"model": "Embedding + BiLSTM", "val_macro_f1_best": round(float(best_val_f1), 4)}
    for split, loader, Y in [("val", loaders["val"], Yva),
                             ("test", loaders["test"], Yte)]:
        probs = predict_proba(model, loader, device, len(le.classes_))
        pred = probs.argmax(1)
        out[split] = {
            "accuracy": round(accuracy_score(Y, pred), 4),
            "precision_macro": round(precision_score(Y, pred, average="macro", zero_division=0), 4),
            "recall_macro": round(recall_score(Y, pred, average="macro", zero_division=0), 4),
            "f1_macro": round(f1_score(Y, pred, average="macro", zero_division=0), 4),
            "f1_weighted": round(f1_score(Y, pred, average="weighted", zero_division=0), 4),
        }
        print(split, out[split])
        if split == "test":
            pd.DataFrame({"id": df.loc[te_idx, "ID"].values,
                          "true": y_test.values,
                          "pred": le.inverse_transform(pred),
                          "confidence": probs.max(1).round(4),
                          "split": "test"}).to_csv(
                REPORT_DIR / "preds_bilstm_test.csv", index=False)

    (REPORT_DIR / "bilstm_results.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(classification_report(Yte, predict_proba(model, loaders["test"],
                                                   device, len(le.classes_)).argmax(1),
                                target_names=le.classes_, digits=3, zero_division=0))
    return out


if __name__ == "__main__":
    main()
