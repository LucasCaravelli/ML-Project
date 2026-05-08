"""Oracle-label derivation from the per-(question, chunk-size) eval grid."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from .types import OracleLabel


def label_from_grid(grid_rows: Iterable[dict[str, Any]]) -> list[OracleLabel]:
    """Turn scored eval-grid rows into one OracleLabel per qa_id.

    Tie-break order (README:97-99): highest F1, then highest EM, then
    smallest chunk_size. Each input row must contain qa_id, chunk_size, em,
    and f1; extra fields are ignored.
    """
    by_qa: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in grid_rows:
        by_qa[row["qa_id"]].append(row)

    labels: list[OracleLabel] = []
    for qa_id in sorted(by_qa):
        rows = by_qa[qa_id]
        # Encode the tie-break as a single sort key so `min` returns the winner.
        # Negate f1 and em so HIGHER values become smaller tuple elements (min
        # picks the smallest); leave chunk_size positive so SMALLER sizes win
        # the final tiebreak. This implements: max F1, then max EM, then min size.
        winner = min(
            rows,
            key=lambda r: (-float(r["f1"]), -float(r["em"]), int(r["chunk_size"])),
        )
        scores_by_size = {int(r["chunk_size"]): float(r["f1"]) for r in rows}
        labels.append(
            OracleLabel(
                qa_id=qa_id,
                best_size=int(winner["chunk_size"]),
                scores_by_size=scores_by_size,
            )
        )
    return labels
