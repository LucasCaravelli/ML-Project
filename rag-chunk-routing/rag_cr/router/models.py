"""Classifier wrappers (Logistic, Linear-SVM, LightGBM) with calibrated-probability output."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

_LOGREG_C = 1.0
_LOGREG_MAX_ITER = 1000
_SVM_C = 1.0
_LGB_N_ESTIMATORS = 200
_LGB_NUM_LEAVES = 31
_SVM_CALIB_CV = 3


class RouterModel(Protocol):
    """Uniform fit/predict/save/load contract for candidate classifiers.

    Implementations expose:

    - ``fit(features, labels)``: train on a feature matrix + integer label vector.
    - ``predict(features)``: return per-example argmax labels.
    - ``predict_proba(features)``: return calibrated per-class probabilities.
    - ``save(path)``: persist the trained classifier to disk (pickle).
    - ``load(path)`` (classmethod): reconstruct a previously saved classifier.

    The ``name`` attribute identifies the model in CV-grid result tables.
    """

    name: str

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None: ...

    def predict(self, features: np.ndarray) -> np.ndarray: ...

    def predict_proba(self, features: np.ndarray) -> np.ndarray: ...

    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> RouterModel: ...


class LogisticRouter:
    """Logistic regression classifier with balanced class weights."""

    name = "logreg"

    def __init__(
        self, C: float = _LOGREG_C, max_iter: int = _LOGREG_MAX_ITER, seed: int = 42,
    ) -> None:
        from sklearn.linear_model import LogisticRegression

        self._clf = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=seed,
            class_weight="balanced",
        )

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Fit logistic regression on (features, labels)."""
        self._clf.fit(features, labels)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self._clf.predict(features)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return (n, n_classes) probability matrix ordered by sorted class labels."""
        return self._clf.predict_proba(features)

    @property
    def classes_(self) -> np.ndarray:
        """Sorted unique class labels seen during fit."""
        return self._clf.classes_

    def save(self, path: Path) -> None:
        """Pickle this model to path."""
        import pickle

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> LogisticRouter:
        """Load a pickled LogisticRouter from path."""
        import pickle

        with path.open("rb") as f:
            return pickle.load(f)


class SVMRouter:
    """Calibrated linear SVM classifier with balanced class weights.

    Uses CalibratedClassifierCV (cv folds) when enough samples are available per
    class; falls back to softmax over the SVM decision scores otherwise.
    """

    name = "svm_linear"

    def __init__(self, C: float = _SVM_C, seed: int = 42) -> None:
        self._C = C
        self._seed = seed
        self._base = None   # LinearSVC, always fitted
        self._calib = None  # CalibratedClassifierCV, fitted when possible

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Fit LinearSVC; add Platt calibration when min class count allows it."""
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.svm import LinearSVC

        _, counts = np.unique(labels, return_counts=True)
        min_count = int(counts.min())
        cv = min(_SVM_CALIB_CV, min_count)

        base = LinearSVC(C=self._C, class_weight="balanced", random_state=self._seed)
        base.fit(features, labels)
        self._base = base

        if cv >= 2:
            new_base = LinearSVC(C=self._C, class_weight="balanced", random_state=self._seed)
            self._calib = CalibratedClassifierCV(new_base, cv=cv)
            self._calib.fit(features, labels)
        else:
            self._calib = None

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if self._calib is not None:
            return self._calib.predict(features)
        return self._base.predict(features)  # type: ignore[union-attr]

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return (n, n_classes) probability matrix (calibrated or softmax fallback)."""
        if self._calib is not None:
            return self._calib.predict_proba(features)
        scores = self._base.decision_function(features)  # type: ignore[union-attr]
        shifted = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    @property
    def classes_(self) -> np.ndarray:
        """Sorted unique class labels seen during fit."""
        if self._calib is not None:
            return self._calib.classes_
        return self._base.classes_  # type: ignore[union-attr]

    def save(self, path: Path) -> None:
        """Pickle this model to path."""
        import pickle

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> SVMRouter:
        """Load a pickled SVMRouter from path."""
        import pickle

        with path.open("rb") as f:
            return pickle.load(f)


class LightGBMRouter:
    """LightGBM gradient-boosted classifier with balanced class weights."""

    name = "lightgbm"

    def __init__(
        self,
        n_estimators: int = _LGB_N_ESTIMATORS,
        num_leaves: int = _LGB_NUM_LEAVES,
        seed: int = 42,
    ) -> None:
        from lightgbm import LGBMClassifier

        self._clf = LGBMClassifier(
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            random_state=seed,
            class_weight="balanced",
            verbose=-1,
        )

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Fit LightGBM on (features, labels)."""
        self._clf.fit(features, labels)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict class labels, suppressing the benign feature-name mismatch warning."""
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
            return self._clf.predict(features)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return (n, n_classes) probability matrix."""
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
            return self._clf.predict_proba(features)

    @property
    def classes_(self) -> np.ndarray:
        """Sorted unique class labels seen during fit."""
        return self._clf.classes_

    def save(self, path: Path) -> None:
        """Pickle this model to path."""
        import pickle

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> LightGBMRouter:
        """Load a pickled LightGBMRouter from path."""
        import pickle

        with path.open("rb") as f:
            return pickle.load(f)


def build_router_model(name: str, config=None, seed: int = 42) -> RouterModel:
    """Construct a named router model (logreg, svm_linear, lightgbm)."""
    if name == "logreg":
        return LogisticRouter(seed=seed)
    if name == "svm_linear":
        return SVMRouter(seed=seed)
    if name == "lightgbm":
        return LightGBMRouter(seed=seed)
    raise ValueError(f"Unknown router model: {name!r}. Valid: logreg, svm_linear, lightgbm")
