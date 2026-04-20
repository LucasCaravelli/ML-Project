# TODO: Implement the oracle labeler that, for each validated question, evaluates every chunk size and records the best one.

from __future__ import annotations

from .types import OracleLabel


def compute_oracle_labels(
    qa_path: str,
    artifacts_dir: str,
    sizes: list[int],
) -> list[OracleLabel]:
    """Evaluate every chunk size per question and record the winner."""
    raise NotImplementedError
