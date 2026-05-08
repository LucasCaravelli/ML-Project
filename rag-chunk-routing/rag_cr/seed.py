"""Deterministic seeding across Python, NumPy, and PyTorch."""

from __future__ import annotations

import os
import random


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (if installed) for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)

    try:
        import torch
    except (ImportError, OSError):
        pass
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
