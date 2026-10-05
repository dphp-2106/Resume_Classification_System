"""Reproducible text preprocessing (Stage 5).

Design decisions (each justified in README / report):
  * lowercase            -> consistent case, merges "Python"/"PYTHON"
  * strip HTML           -> Resume_html holds markup; Resume_str is mostly clean,
                            but the function must be safe on raw text too
  * remove URLs, emails, phones -> contact details are near-random per resume and
                            do not discriminate between categories (verified: only
                            ~2% of rows contain them)
  * keep symbols inside tokens  -> preserves technical tokens: C++, C#, .NET, R&D
  * keep tokens with at least one alphanumeric char -> drops stray punctuation
  * stopword removal     -> OPTIONAL: we train variants with/without and report
  * lemmatization        -> OPTIONAL: compared against raw tokens, not assumed
The same function is used at training time and inference time (single source of
truth imported by every trainer and by the demo app).
"""
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:  # nltk is optional at import time, required for 'standard'/'lemma' modes
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.stem import WordNetLemmatizer
    _NLTK_OK = True
except Exception:  # pragma: no cover
    _NLTK_OK = False

_HTML_RE = re.compile(r"<[^>]+>")
_ENTITY_RE = re.compile(r"&(amp|lt|gt|nbsp|quot|#\d+);")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")
_WS_RE = re.compile(r"\s+")
# token: letters/digits and internal symbols like + # . & - (C++, C#, .NET, R&D)
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9+#.&_\-/]*")

_ENGLISH_STOP = None
_LEMMATIZER = None


def _ensure_nltk():
    global _ENGLISH_STOP, _LEMMATIZER
    if not _NLTK_OK:
        raise ImportError("nltk is required for 'standard'/'lemma' preprocessing")
    if _ENGLISH_STOP is None:
        try:
            _ENGLISH_STOP = set(nltk_stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            _ENGLISH_STOP = set(nltk_stopwords.words("english"))
    if _LEMMATIZER is None:
        try:
            from nltk.corpus import wordnet  # noqa: F401
            _LEMMATIZER = WordNetLemmatizer()
        except LookupError:
            nltk.download("wordnet", quiet=True)
            _LEMMATIZER = WordNetLemmatizer()
    return _ENGLISH_STOP, _LEMMATIZER


def preprocess(text: str, mode: str = "standard") -> str:
    """Clean one resume. Returns whitespace-joined tokens.

    mode:
      'raw'      -> only whitespace/HTML/entity normalisation (diagnostic)
      'light'    -> + lowercase, contact-info removal, token filter
      'standard' -> light + stopword removal   (default for TF-IDF)
      'lemma'    -> standard + lemmatization   (compared variant)
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _HTML_RE.sub(" ", text)
    text = _ENTITY_RE.sub(" ", text)
    text = text.replace("\xa0", " ")

    if mode == "raw":
        return _WS_RE.sub(" ", text).strip()

    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _PHONE_RE.sub(" ", text)

    tokens = [t for t in _TOKEN_RE.findall(text) if any(ch.isalnum() for ch in t)]

    if mode in ("standard", "lemma"):
        stops, lemma = _ensure_nltk()
        tokens = [t for t in tokens if t not in stops and len(t) > 1]
        if mode == "lemma":
            tokens = [lemma.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def tokenize(text: str, mode: str = "standard") -> list[str]:
    """Preprocess and split into tokens (used by Word2Vec / LSTM pipelines)."""
    return preprocess(text, mode=mode).split()
