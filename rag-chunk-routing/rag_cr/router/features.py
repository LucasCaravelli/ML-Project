from __future__ import annotations

import re
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

_TFIDF_MAX_FEATURES = 10_000
_MINILM_CACHE: dict[str, object] = {}  # module-level cache: model_name → SentenceTransformer
_TFIDF_NGRAM_RANGE = (1, 2)
_MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_LEN_NORM = 30.0
_NER_NORM = 5.0
_QUESTION_WORDS = ["who", "what", "when", "where", "why", "how", "which"]
_TYPES = ["factoid", "multihop", "synthesis"]


class FeatureExtractor(Protocol):
    """Uniform fit/transform contract shared by every candidate feature set."""

    name: str

    def fit(self, queries: list[str], types: list[str] | None = None) -> None: ...

    def transform(self, queries: list[str], types: list[str] | None = None) -> np.ndarray: ...

    def fit_transform(self, queries: list[str], types: list[str] | None = None) -> np.ndarray: ...


class TfidfExtractor:
    """TF-IDF bag-of-ngrams extractor backed by sklearn's TfidfVectorizer."""

    name = "tfidf"

    def __init__(
        self,
        max_features: int = _TFIDF_MAX_FEATURES,
        ngram_range: tuple[int, int] = _TFIDF_NGRAM_RANGE,
    ) -> None:
        self._vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

    def fit(self, queries: list[str], types: list[str] | None = None) -> None:
        """Fit the TF-IDF vocabulary on training queries."""
        self._vec.fit(queries)

    def transform(self, queries: list[str], types: list[str] | None = None) -> np.ndarray:
        """Transform queries into a dense TF-IDF matrix."""
        return self._vec.transform(queries).toarray().astype(np.float32)

    def fit_transform(self, queries: list[str], types: list[str] | None = None) -> np.ndarray:
        """Fit and transform in one pass."""
        return self._vec.fit_transform(queries).toarray().astype(np.float32)


class MiniLMExtractor:
    """Sentence-embedding extractor using all-MiniLM-L6-v2 (frozen weights)."""

    name = "minilm"

    def __init__(self, model_name: str = _MINILM_MODEL) -> None:
        self._model_name = model_name
        self._model = None  # loaded lazily on first transform call

    def _load(self) -> None:
        if self._model is None:
            if self._model_name not in _MINILM_CACHE:
                from sentence_transformers import SentenceTransformer

                _MINILM_CACHE[self._model_name] = SentenceTransformer(self._model_name)
            self._model = _MINILM_CACHE[self._model_name]

    def fit(self, queries: list[str], types: list[str] | None = None) -> None:
        """No-op: MiniLM weights are frozen."""

    def transform(self, queries: list[str], types: list[str] | None = None) -> np.ndarray:
        """Encode queries into (n, 384) float32 embeddings."""
        self._load()
        return self._model.encode(queries, show_progress_bar=False).astype(np.float32)  # type: ignore[union-attr]

    def fit_transform(self, queries: list[str], types: list[str] | None = None) -> np.ndarray:
        """Fit (no-op) and transform."""
        return self.transform(queries, types)


def _question_word_onehot(question: str) -> list[float]:
    """8-dim one-hot over {who,what,when,where,why,how,which,other}."""
    raw = question.lower().split()[0] if question.strip() else ""
    m = re.match(r"[a-z]+", raw)
    first = m.group() if m else ""
    vec = [float(first == w) for w in _QUESTION_WORDS]
    vec.append(float(first not in _QUESTION_WORDS))
    return vec


def _ner_count(question: str) -> float:
    """Heuristic count of capitalised multi-word spans (not at sentence start), normalised."""
    tokens = question.split()
    count = 0
    in_span = False
    for i, tok in enumerate(tokens):
        if i == 0:
            in_span = False
            continue
        if tok and tok[0].isupper():
            if not in_span:
                count += 1
                in_span = True
        else:
            in_span = False
    return count / _NER_NORM


def _type_onehot(qtype: str) -> list[float]:
    """3-dim one-hot over {factoid, multihop, synthesis}."""
    return [float(qtype == t) for t in _TYPES]


class HandcraftedExtractor:
    """13-dimensional deterministic feature extractor (no fitting needed)."""

    name = "handcrafted"

    def fit(self, queries: list[str], types: list[str] | None = None) -> None:
        """No-op."""

    def transform(self, queries: list[str], types: list[str] | None = None) -> np.ndarray:
        """Return (n, 13) feature matrix: length + question_word + ner + type."""
        rows: list[list[float]] = []
        for i, q in enumerate(queries):
            qtype = types[i] if types else "factoid"
            row = (
                [len(q.split()) / _LEN_NORM]
                + _question_word_onehot(q)
                + [_ner_count(q)]
                + _type_onehot(qtype)
            )
            rows.append(row)
        return np.array(rows, dtype=np.float32)

    def fit_transform(self, queries: list[str], types: list[str] | None = None) -> np.ndarray:
        """Fit (no-op) and transform."""
        return self.transform(queries, types)


class ConcatExtractor:
    """Horizontally concatenates outputs from a list of feature extractors."""

    name = "concat"

    def __init__(self, extractors: list) -> None:
        self._extractors = extractors

    def fit(self, queries: list[str], types: list[str] | None = None) -> None:
        """Fit each constituent extractor on the same queries."""
        for ext in self._extractors:
            ext.fit(queries, types)

    def transform(self, queries: list[str], types: list[str] | None = None) -> np.ndarray:
        """Transform and horizontally stack outputs of all extractors."""
        parts = [ext.transform(queries, types) for ext in self._extractors]
        return np.hstack(parts)

    def fit_transform(self, queries: list[str], types: list[str] | None = None) -> np.ndarray:
        """fit_transform each extractor and horizontally stack."""
        parts = [ext.fit_transform(queries, types) for ext in self._extractors]
        return np.hstack(parts)


def build_feature_extractor(name: str, config=None) -> FeatureExtractor:
    """Construct a named feature extractor (tfidf, minilm, handcrafted)."""
    if name == "tfidf":
        return TfidfExtractor()
    if name == "minilm":
        return MiniLMExtractor()
    if name == "handcrafted":
        return HandcraftedExtractor()
    raise ValueError(f"Unknown feature extractor: {name!r}. Valid: tfidf, minilm, handcrafted")
