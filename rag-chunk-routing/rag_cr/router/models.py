# TODO: Implement thin wrappers around the candidate classifiers (logistic regression, linear SVM, LightGBM) with a uniform fit/predict/save/load interface.

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class RouterModel(Protocol):
    """Uniform fit/predict/save/load contract for candidate classifiers."""

    name: str

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None: ...

    def predict(self, features: np.ndarray) -> np.ndarray: ...

    def predict_proba(self, features: np.ndarray) -> np.ndarray: ...

    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> "RouterModel": ...


def build_router_model(name: str) -> RouterModel:
    """Construct a named router model (logreg, svm_linear, lightgbm)."""
    raise NotImplementedError
