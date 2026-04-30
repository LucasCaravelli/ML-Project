from __future__ import annotations

import math
import random
from collections import defaultdict

from .types import QAPair


SPLIT_NAMES = ("train", "val", "test")


def make_splits(
    qa_pairs: list[QAPair],
    ratios: dict[str, float],
    seed: int,
    stratify_by: str = "type",
) -> dict[str, list[QAPair]]:
    """Stratified train/val/test split, deterministic given seed.

    Within each stratum, sort by qa_id (so input order doesn't matter), shuffle
    with a Random(seed) instance, then take floor(n*train_ratio) for train,
    floor(n*val_ratio) for val, and the remainder for test. The remainder
    going to test absorbs all rounding so every input row lands in exactly
    one split.
    """
    missing = set(SPLIT_NAMES) - ratios.keys()
    if missing:
        raise ValueError(f"ratios is missing keys: {sorted(missing)}")
    if not math.isclose(sum(ratios[k] for k in SPLIT_NAMES), 1.0, abs_tol=1e-6):
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")

    by_stratum: dict[str, list[QAPair]] = defaultdict(list)
    for qa in qa_pairs:
        by_stratum[qa[stratify_by]].append(qa)

    rng = random.Random(seed)
    splits: dict[str, list[QAPair]] = {name: [] for name in SPLIT_NAMES}

    for stratum in sorted(by_stratum):
        group = sorted(by_stratum[stratum], key=lambda q: q["qa_id"])
        rng.shuffle(group)

        n = len(group)
        n_train = int(n * ratios["train"])
        n_val = int(n * ratios["val"])

        splits["train"].extend(group[:n_train])
        splits["val"].extend(group[n_train : n_train + n_val])
        splits["test"].extend(group[n_train + n_val :])

    return splits
