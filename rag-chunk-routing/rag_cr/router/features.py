# TODO: Implement the candidate feature extractors (TF-IDF, MiniLM sentence embeddings, handcrafted query features) behind a uniform fit/transform interface.

from __future__ import annotations

from typing import Protocol

import numpy as np


class FeatureExtractor(Protocol):
    """Uniform fit/transform contract shared by every candidate feature set."""

    name: str

    def fit(self, queries: list[str]) -> None: ...

    def transform(self, queries: list[str]) -> np.ndarray: ...


def build_feature_extractor(name: str) -> FeatureExtractor:
    """Construct a named feature extractor (tfidf, minilm, handcrafted)."""
    raise NotImplementedError
