"""Shared utilities for experiment scripts (git hash, oracle-gap reading, gap-closure metric)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def git_hash() -> str:
    """Return the current git commit hash, or 'unknown' on failure."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def read_oracle_gap(artifacts_dir: Path) -> tuple[float | None, float | None]:
    """Return (oracle_f1_mean, best_baseline_f1) from artifacts/baselines/oracle_gap.json."""
    path = artifacts_dir / "baselines" / "oracle_gap.json"
    if not path.exists():
        return None, None
    data = json.loads(path.read_text())
    return data.get("oracle_f1_mean"), data.get("best_baseline_f1")


def gap_closure(
    router_f1: float, best_baseline_f1: float | None, oracle_f1: float | None
) -> float | None:
    """Compute (router - baseline) / (oracle - baseline), or None if reference data missing."""
    if best_baseline_f1 is None or oracle_f1 is None:
        return None
    denom = oracle_f1 - best_baseline_f1
    if abs(denom) < 1e-9:
        return None
    return (router_f1 - best_baseline_f1) / denom
