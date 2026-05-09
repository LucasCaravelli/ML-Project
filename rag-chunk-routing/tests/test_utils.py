from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_cr.utils import gap_closure, read_oracle_gap


@pytest.mark.parametrize(
    "router_f1,baseline,oracle,expected",
    [
        (0.5, 0.4, 0.6, 0.5),    # standard: halfway
        (0.6, 0.4, 0.6, 1.0),    # router at oracle
        (0.4, 0.4, 0.6, 0.0),    # router at baseline
        (0.3, 0.4, 0.6, -0.5),   # router below baseline
        (0.5, None, 0.6, None),  # baseline missing
        (0.5, 0.4, None, None),  # oracle missing
        (0.5, 0.4, 0.4, None),   # zero denominator
    ],
)
def test_gap_closure(router_f1, baseline, oracle, expected):
    result = gap_closure(router_f1, baseline, oracle)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


def test_read_oracle_gap_returns_none_pair_when_missing(tmp_path: Path):
    assert read_oracle_gap(tmp_path) == (None, None)


def test_read_oracle_gap_reads_expected_keys(tmp_path: Path):
    gap_dir = tmp_path / "baselines"
    gap_dir.mkdir(parents=True)
    (gap_dir / "oracle_gap.json").write_text(
        json.dumps({"oracle_f1_mean": 0.62, "best_baseline_f1": 0.48}),
        encoding="utf-8",
    )
    oracle, baseline = read_oracle_gap(tmp_path)
    assert oracle == pytest.approx(0.62)
    assert baseline == pytest.approx(0.48)


def test_read_oracle_gap_handles_missing_keys(tmp_path: Path):
    gap_dir = tmp_path / "baselines"
    gap_dir.mkdir(parents=True)
    (gap_dir / "oracle_gap.json").write_text("{}", encoding="utf-8")
    assert read_oracle_gap(tmp_path) == (None, None)
