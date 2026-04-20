# TODO: Implement the cross-validated training loop across all (feature set x model) combinations, with probability calibration and selection of the best configuration.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RouterTrainingResult:
    feature_set: str
    model_name: str
    cv_score: float
    model_path: Path


def train_router(
    qa_path: str | Path,
    oracle_labels_path: str | Path,
    feature_sets: list[str],
    model_names: list[str],
    cv_folds: int,
    out_dir: str | Path,
) -> RouterTrainingResult:
    """Train every (feature x model) combination and return the selected one."""
    raise NotImplementedError
