"""Unit tests for rag_cr.router.predict.predict_with_threshold."""

from __future__ import annotations

import numpy as np
import pytest


class _FakeClassifier:
    """Minimal stub with predict / predict_proba / classes_."""

    def __init__(self, classes: list[int], proba: np.ndarray) -> None:
        self.classes_ = np.array(classes)
        self._proba = proba

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self._proba, axis=1)]

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self._proba


# classes: [128, 256, 512], majority = 128
CLASSES = [128, 256, 512]
# Row 0: 128 wins (0.8, 0.1, 0.1)  -> p_non_maj = 0.2
# Row 1: 256 wins (0.3, 0.5, 0.2)  -> p_non_maj = 0.7
# Row 2: 512 wins (0.1, 0.2, 0.7)  -> p_non_maj = 0.9
# Row 3: 128 wins (0.6, 0.2, 0.2)  -> p_non_maj = 0.4
PROBA = np.array([
    [0.8, 0.1, 0.1],
    [0.3, 0.5, 0.2],
    [0.1, 0.2, 0.7],
    [0.6, 0.2, 0.2],
], dtype=np.float32)
FEATURES = np.zeros((4, 1), dtype=np.float32)  # features are ignored by _FakeClassifier


@pytest.fixture
def clf() -> _FakeClassifier:
    return _FakeClassifier(CLASSES, PROBA)


class TestPredictWithThreshold:
    def test_none_threshold_matches_predict(self, clf: _FakeClassifier) -> None:
        from rag_cr.router.predict import predict_with_threshold

        result = predict_with_threshold(clf, FEATURES, majority_class=128, threshold=None)
        expected = clf.predict(FEATURES)
        np.testing.assert_array_equal(result, expected)

    def test_threshold_one_always_returns_majority(self, clf: _FakeClassifier) -> None:
        from rag_cr.router.predict import predict_with_threshold

        result = predict_with_threshold(clf, FEATURES, majority_class=128, threshold=1.0)
        np.testing.assert_array_equal(result, np.array([128, 128, 128, 128]))

    def test_threshold_zero_matches_non_majority_argmax(self, clf: _FakeClassifier) -> None:
        from rag_cr.router.predict import predict_with_threshold

        # threshold=0.0 -> every example overrides (p_non_maj always > 0)
        result = predict_with_threshold(clf, FEATURES, majority_class=128, threshold=0.0)
        # best among {256, 512} for each row
        # row 0: 256(0.1) vs 512(0.1) -> first wins -> 256
        # row 1: 256(0.5) vs 512(0.2) -> 256
        # row 2: 256(0.2) vs 512(0.7) -> 512
        # row 3: 256(0.2) vs 512(0.2) -> first wins -> 256
        expected = np.array([256, 256, 512, 256])
        np.testing.assert_array_equal(result, expected)

    def test_intermediate_threshold_mixed_behaviour(self, clf: _FakeClassifier) -> None:
        from rag_cr.router.predict import predict_with_threshold

        # threshold=0.5 -> override only rows where p_non_maj > 0.5
        # row 0: p_non_maj=0.2 -> 128 (no override)
        # row 1: p_non_maj=0.7 -> override -> 256
        # row 2: p_non_maj=0.9 -> override -> 512
        # row 3: p_non_maj=0.4 -> 128 (no override)
        result = predict_with_threshold(clf, FEATURES, majority_class=128, threshold=0.5)
        expected = np.array([128, 256, 512, 128])
        np.testing.assert_array_equal(result, expected)

    def test_majority_class_not_in_classes_raises(self, clf: _FakeClassifier) -> None:
        from rag_cr.router.predict import predict_with_threshold

        with pytest.raises(ValueError, match="majority_class=999"):
            predict_with_threshold(clf, FEATURES, majority_class=999, threshold=0.5)

    def test_threshold_out_of_range_raises(self, clf: _FakeClassifier) -> None:
        from rag_cr.router.predict import predict_with_threshold

        with pytest.raises(ValueError, match=r"threshold must be in"):
            predict_with_threshold(clf, FEATURES, majority_class=128, threshold=1.5)
        with pytest.raises(ValueError, match=r"threshold must be in"):
            predict_with_threshold(clf, FEATURES, majority_class=128, threshold=-0.1)
