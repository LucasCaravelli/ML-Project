"""Threshold-based prediction override for the chunk-size router."""

from __future__ import annotations

import numpy as np


def predict_with_threshold(
    classifier,
    features: np.ndarray,
    majority_class: int,
    threshold: float | None,
) -> np.ndarray:
    """Argmax prediction with a P(non-majority) > threshold override.

    If threshold is None, returns classifier.predict(features) (argmax).
    Otherwise, for each example:
      p = classifier.predict_proba(features)
      p_non_majority = 1 - p[:, idx(majority_class)]
      pred = argmax_{c != majority_class} p[:, c] if p_non_majority > threshold
             else majority_class
    """
    if threshold is None:
        return classifier.predict(features)

    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in [0.0, 1.0], got {threshold}")

    classes = np.asarray(classifier.classes_)
    majority_indices = np.where(classes == majority_class)[0]
    if majority_indices.size == 0:
        raise ValueError(
            f"majority_class={majority_class!r} not in classifier.classes_={list(classes)}"
        )
    maj_idx = int(majority_indices[0])

    proba = classifier.predict_proba(features)
    n = proba.shape[0]
    p_non_majority = 1.0 - proba[:, maj_idx]

    non_maj_mask = np.ones(len(classes), dtype=bool)
    non_maj_mask[maj_idx] = False
    non_maj_indices = np.where(non_maj_mask)[0]

    predictions = np.full(n, majority_class, dtype=classes.dtype)
    override = p_non_majority > threshold
    if override.any():
        non_maj_proba = proba[np.ix_(np.where(override)[0], non_maj_indices)]
        best_non_maj = non_maj_indices[np.argmax(non_maj_proba, axis=1)]
        predictions[override] = classes[best_non_maj]

    return predictions
