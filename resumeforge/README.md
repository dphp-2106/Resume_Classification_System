# SAMATRIX RESUMEFORGE 2026 — Resume Classification

End-to-end NLP system that predicts a resume's job category (24 classes)
from raw text. Built for the **Resumeforge 2026** hackathon following the
required workflow:

**Problem → Data → EDA → Preprocessing → Features → ML/DL → Evaluation → Error analysis → Demo**

## Quick start

```bash
pip install -r requirements.txt
python run_all.py          # full pipeline, in order
streamlit run app.py       # interactive demo
```

## Dataset

`Dataset-20261005T050929Z-1-001/Dataset/Resume/Resume.csv` — **2,484 resumes**,
columns `ID, Resume_str, Resume_html, Category`, **24 categories**.
A `Resume.xlsx` copy and 349 PDF resumes in 4 category folders
(INFORMATION-TECHNOLOGY, PUBLIC-RELATIONS, SALES, TEACHER) are also provided;
the PDFs are used as an extra unseen-data demo (`src/demo_pdfs.py`).

## Stage 1–3. Problem understanding, data gathering, data quality

* **Input**: raw resume text. **Output**: one of 24 categories.
  **Success metric**: macro-F1 (equal weight to every class, important
  because classes are imbalanced).
* No missing values. **1 empty resume** (ID 12632728, whitespace-only,
  its HTML is an empty `<div>`) and **2 exact duplicate pairs** (FINANCE,
  AVIATION — same text, same label) are removed → **2,481 resumes**.
  Justification: the empty doc carries no signal; duplicates would
  overweight two classes and can leak identical text across train/test.
* **Class imbalance**: BPO=22, AUTOMOBILE=36, AGRICULTURE=63 vs
  ~102–120 elsewhere (ratio 5.45×) → stratified split +
  `class_weight='balanced'` / sample weights / weighted loss.
* **Noise**: `Resume_str` is already clean of HTML; 57 resumes contain
  URLs, 19 emails, 42 phone numbers (contact info is non-discriminative).
* **Leakage check**: no identical text appears under two different labels.
* Full report: `outputs/reports/data_quality_report.json`.

## Stage 4. EDA (`src/eda.py`, plots in `outputs/eda/`)

| Plot | Finding |
|---|---|
| 01 class distribution | 21 majority classes, 3 minority (BPO/AUTOMOBILE/AGRICULTURE) |
| 02/03 text length | median 757 words / 5,887 chars; heavy right tail to 5,190 words |
| 04 top words | resume-builder template words dominate ("name", "city", "company") |
| 05/06 word clouds | overall + class-wise clouds |
| 07 n-grams | top bigrams are template artifacts: *city state, company name, name city* |
| 08 class TF-IDF terms | interpretable per-class vocabulary (see below) |
| 09 data quality | issue counts |

Class-distinguishing TF-IDF terms are highly interpretable, e.g.
ACCOUNTANT ← *accountant, ledger, reconciliations, GL*; AVIATION ←
*aircraft, flight, navy, FAA*; TEACHER ← *classroom, mathematics,
lessons, preschool*.

## Stage 5. Preprocessing (`src/preprocessing.py`)

One reproducible function used identically in training and inference:
NFKC normalisation → HTML/entity stripping → lowercase → URL/email/phone
removal → token filter that **preserves technical tokens** (`C++`, `C#`,
`.NET`, `R&D`) → optional stopword removal / lemmatization.
Three variants are **compared, not assumed**: `light`, `standard`
(stopwords), `lemma`. TF-IDF uses `token_pattern=r"\S+"` so the
preprocessor's tokens (incl. `C++`) are not re-mangled.

## Stage 6–8. Split & features

* **Stratified 70/15/15 split** (1,736 / 372 / 373) created **before**
  any vectorizer is fitted; TF-IDF/Word2Vec/vocab are fit on train only.
* **TF-IDF**: `TfidfVectorizer(min_df=2, max_df=0.9, sublinear_tf=True)`,
  unigrams vs unigram+bigram compared.
* **Word2Vec**: `gensim` skip-gram, 100-d, window 5, min_count 2, trained
  on the train split only; documents = mean-pooled word vectors.
* **LSTM**: 384-token sequences, vocab (min_freq=3) from train only,
  96-d embedding, BiLSTM-128, class-weighted cross-entropy, early
  stopping on val macro-F1.

## Stage 9. Models

| Model | Representation | Role |
|---|---|---|
| MultinomialNB | TF-IDF | fast baseline |
| LogisticRegression (balanced) | TF-IDF | classical baseline |
| LinearSVC (balanced) | TF-IDF | strong classical baseline |
| MLP (128,64) | Word2Vec mean-pool | dense neural |
| BiLSTM | token sequences + embedding | deep neural |

## Stage 10. Evaluation (results on the untouched test set, n=373)

| Rank | Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | F1 (weighted) |
|---|---|---|---|---|---|---|
| 1 | **TF-IDF + LinearSVC** (standard, unigram) | **0.673** | 0.679 | 0.635 | **0.617** | 0.646 |
| 2 | Embedding + BiLSTM | 0.625 | 0.596 | 0.589 | 0.583 | 0.622 |
| 3 | Word2Vec (mean-pool) + Dense MLP | 0.534 | 0.524 | 0.519 | 0.510 | 0.534 |

Classical variants compared during development (test macro-F1):
LinearSVC 0.617–0.626, LogisticRegression 0.555–0.579,
MultinomialNB 0.399–0.418; unigram vs unigram+bigram and
light/standard/lemma preprocessing were all compared
(`outputs/reports/classical_results.csv`).

**Why LinearSVC wins:** with only ~1,700 training documents, a
linear model over sparse TF-IDF features generalises better than
neural models that must learn both an embedding and a classifier
from scratch. The BiLSTM closes much of the gap after 30 epochs
with class-weighted loss (0.583 macro-F1) but still overfits
(train loss → 0.05 while val F1 plateaus). Selection used
**validation** macro-F1 only; the test set was touched once.

Selection rule: the configuration with the best **validation**
macro-F1 is chosen; the test set is used once, only for the final
comparison. Per-class metrics: `outputs/reports/per_class_metrics.csv`;
confusion matrix: `outputs/eda/10_confusion_matrix_best.png`.

## Stage 11. Error analysis (`src/error_analysis.py`)

* **32.7% test error rate** (122/373). Wrong predictions have
  *lower* mean confidence (0.084) than correct ones (0.113) —
  the model is appropriately uncertain when it fails; every
  error has confidence < 0.6.
* **Top confusions are semantically interpretable**:
  FINANCE→ACCOUNTANT ×9 (shared *budget/accounting/financial*
  vocabulary), ARTS→TEACHER ×8 (many "ARTS" resumes are
  literally *Language Arts teachers* — label overlap),
  SALES→BUSINESS-DEVELOPMENT ×4, APPAREL→SALES ×4
  (retail overlap), CONSULTANT→ACCOUNTANT ×3.
* **Worst classes**: BPO 100% error (only 3 test resumes,
  noisy vocabulary), CONSULTANT 94% (generic vocabulary
  overlaps all classes), ARTS 81%, AUTOMOBILE 80% (5 test
  resumes), APPAREL 73%.
* **Diagnosis**: errors are driven by *class overlap and
  label noise*, not by preprocessing or features — raising
  the minority-class support and cleaning ambiguous labels
  (e.g. "Language Arts" teachers) is the justified next
  step, not another model change.
* Wrong predictions with confidence and text preview:
  `outputs/reports/errors_best_model.csv`.

## Stage 12. Final pipeline & demo

* `src/pipeline.py` bundles preprocessing + vectorizer + model into one
  artifact (`outputs/models/`); `ResumeClassifier.predict(raw_text)`
  guarantees identical preprocessing at inference.
* Unseen-resume demo (5 held-out resumes, all predicted correctly,
  incl. minority AGRICULTURE): `outputs/reports/unseen_demo.json`.
* PDF demo (`src/demo_pdfs.py`): classifies the 449 folder-labelled
  PDFs with pypdf text extraction — 92.7% match their folder labels.
  **Caveat:** these PDFs are the source documents of the CSV resumes,
  so most overlap the training data; this demonstrates pipeline
  robustness (raw PDF → text → prediction), not unbiased accuracy.
* Streamlit app: `streamlit run app.py`.

## Reproducibility

`random_state=42` everywhere; `run_all.py` executes every stage in
order and regenerates all artifacts in `outputs/`.

## Artifacts

```
resumeforge/
├── app.py                    # Streamlit demo
├── run_all.py                # full pipeline orchestration
├── requirements.txt
├── README.md
├── src/
│   ├── config.py             # paths, split sizes, constants
│   ├── data_loading.py       # load + data-quality checks
│   ├── eda.py                # 10 EDA plots
│   ├── preprocessing.py      # single source of truth for cleaning
│   ├── train_classical.py    # TF-IDF + NB/LR/SVC comparison
│   ├── train_w2v_mlp.py      # Word2Vec + Dense MLP
│   ├── train_lstm.py         # Embedding + BiLSTM (PyTorch)
│   ├── evaluate.py           # comparison, confusion matrix, per-class
│   ├── error_analysis.py     # wrong-prediction inspection
│   ├── pipeline.py           # bundled inference + unseen demo
│   └── demo_pdfs.py          # PDF-folder demo
└── outputs/
    ├── eda/                  # 11 plots (class dist, lengths, clouds,
    │                         #   n-grams, class TF-IDF, confusion matrix)
    ├── models/               # best_classical.joblib, w2v_mlp.joblib,
    │                         #   bilstm.pt
    └── reports/              # JSON/CSV: quality, results, per-class,
                              #   errors, demos
```
