from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from ..io import read_jsonl
from .features import ConcatExtractor, FeatureExtractor, build_feature_extractor
from .models import RouterModel, build_router_model


@dataclass(frozen=True)
class RouterTrainingResult:
    """Summary of a completed router training run."""

    feature_set: str
    model_name: str
    cv_score: float
    model_path: Path


def load_router_data(
    splits_dir: Path,
    oracle_labels_path: Path,
    split: str,
    router_chunk_sizes: list[int],
) -> tuple[list[str], list[str], list[int]]:
    """Return (questions, types, labels) for a split, excluding oracle_best_size not in router_chunk_sizes."""
    split_rows = read_jsonl(splits_dir / f"{split}.jsonl")
    oracle_rows = read_jsonl(oracle_labels_path)

    label_by_id: dict[str, int] = {r["qa_id"]: r["best_size"] for r in oracle_rows}

    questions: list[str] = []
    types: list[str] = []
    labels: list[int] = []

    for row in split_rows:
        qa_id = row["qa_id"]
        best_size = label_by_id.get(qa_id)
        if best_size is None or best_size not in router_chunk_sizes:
            continue
        questions.append(row["question"])
        types.append(row.get("type", "factoid"))
        labels.append(best_size)

    return questions, types, labels


def build_extractor(feature_set: str, config: Any = None) -> FeatureExtractor:
    """Instantiate the right extractor from a feature-set name string."""
    return build_feature_extractor(feature_set, config)


def build_classifier(classifier_name: str, config: Any = None, seed: int = 42) -> RouterModel:
    """Instantiate the right classifier from a name string."""
    return build_router_model(classifier_name, config, seed)


def _score_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute accuracy, balanced accuracy, and macro F1."""
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def run_cv_grid(
    questions: list[str],
    types: list[str],
    labels: list[int],
    config: Any,
    seed: int,
) -> pd.DataFrame:
    """Run 3×3 feature×classifier grid with StratifiedKFold; return per-fold metrics DataFrame."""
    cv = StratifiedKFold(n_splits=config.router.cv_folds, shuffle=True, random_state=seed)
    y = np.array(labels)
    records: list[dict[str, Any]] = []

    for fs in config.router.feature_sets:
        for clf_name in config.router.model_names:
            for fold, (train_idx, val_idx) in enumerate(cv.split(questions, y)):
                train_q = [questions[i] for i in train_idx]
                train_t = [types[i] for i in train_idx]
                val_q = [questions[i] for i in val_idx]
                val_t = [types[i] for i in val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                ext = build_extractor(fs)
                X_train = ext.fit_transform(train_q, train_t)
                X_val = ext.transform(val_q, val_t)

                clf = build_classifier(clf_name, config, seed)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_val)

                scores = _score_predictions(y_val, y_pred)
                records.append(
                    {
                        "feature_set": fs,
                        "classifier": clf_name,
                        "fold": fold,
                        **scores,
                    }
                )

    return pd.DataFrame(records)


def select_best_on_val(
    train_questions: list[str],
    train_types: list[str],
    train_labels: list[int],
    val_questions: list[str],
    val_types: list[str],
    val_labels: list[int],
    config: Any,
    seed: int,
) -> tuple[FeatureExtractor, RouterModel, dict[str, Any]]:
    """Refit each grid cell on full train, score on val, return best extractor + classifier + metadata."""
    y_train = np.array(train_labels)
    y_val = np.array(val_labels)

    best_f1 = -1.0
    best_ext: FeatureExtractor | None = None
    best_clf: RouterModel | None = None
    best_meta: dict[str, Any] = {}

    for fs in config.router.feature_sets:
        for clf_name in config.router.model_names:
            ext = build_extractor(fs)
            X_train = ext.fit_transform(train_questions, train_types)
            X_val = ext.transform(val_questions, val_types)

            clf = build_classifier(clf_name, config, seed)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)

            scores = _score_predictions(y_val, y_pred)
            if scores["macro_f1"] > best_f1:
                best_f1 = scores["macro_f1"]
                best_ext = ext
                best_clf = clf
                best_meta = {
                    "feature_set": fs,
                    "classifier_name": clf_name,
                    "val_macro_f1": scores["macro_f1"],
                    "val_balanced_accuracy": scores["balanced_accuracy"],
                    "val_accuracy": scores["accuracy"],
                }

    assert best_ext is not None and best_clf is not None
    return best_ext, best_clf, best_meta


def train_router(
    qa_path: str | Path,
    oracle_labels_path: str | Path,
    feature_sets: list[str],
    model_names: list[str],
    cv_folds: int,
    out_dir: str | Path,
) -> RouterTrainingResult:
    """Train every (feature × model) combination and return the selected one.

    Legacy high-level entry point; prefer the individual functions for fine-grained control.
    """
    import pickle

    from ..config import load_config

    config = load_config()
    seed = config.project.seed
    splits_dir = Path(qa_path).parent if Path(qa_path).is_file() else Path(qa_path)
    labels_path = Path(oracle_labels_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    router_sizes = config.router.feature_sets  # kept for compat; sizes come from chunking
    chunk_sizes = [s for s in config.chunking.sizes if s != 1024]

    train_q, train_t, train_l = load_router_data(splits_dir, labels_path, "train", chunk_sizes)
    val_q, val_t, val_l = load_router_data(splits_dir, labels_path, "val", chunk_sizes)

    ext, clf, meta = select_best_on_val(train_q, train_t, train_l, val_q, val_t, val_l, config, seed)

    model_path = out / "best.pkl"
    with model_path.open("wb") as f:
        pickle.dump({"extractor": ext, "classifier": clf, "config": meta}, f)

    return RouterTrainingResult(
        feature_set=meta["feature_set"],
        model_name=meta["classifier_name"],
        cv_score=meta["val_macro_f1"],
        model_path=model_path,
    )
