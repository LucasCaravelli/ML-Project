from __future__ import annotations

from collections import Counter

import pytest

from rag_cr.splits import SPLIT_NAMES, make_splits
from rag_cr.types import QAPair


def _qa(qa_id: str, qtype: str) -> QAPair:
    return {
        "qa_id": qa_id,
        "question": f"Q{qa_id}",
        "answer": f"A{qa_id}",
        "source_chunk_id": "512-000000",
        "type": qtype,
        "validated": True,
    }


@pytest.fixture
def qa_pairs() -> list[QAPair]:
    rows: list[QAPair] = []
    for i in range(120):
        rows.append(_qa(f"qa_{i:04d}", "factoid"))
    for i in range(120, 240):
        rows.append(_qa(f"qa_{i:04d}", "multihop"))
    for i in range(240, 360):
        rows.append(_qa(f"qa_{i:04d}", "synthesis"))
    return rows


RATIOS = {"train": 0.6, "val": 0.2, "test": 0.2}


def test_determinism(qa_pairs: list[QAPair]) -> None:
    a = make_splits(qa_pairs, RATIOS, seed=42)
    b = make_splits(qa_pairs, RATIOS, seed=42)
    for name in SPLIT_NAMES:
        assert [q["qa_id"] for q in a[name]] == [q["qa_id"] for q in b[name]]


def test_input_order_does_not_matter(qa_pairs: list[QAPair]) -> None:
    a = make_splits(qa_pairs, RATIOS, seed=42)
    b = make_splits(list(reversed(qa_pairs)), RATIOS, seed=42)
    for name in SPLIT_NAMES:
        assert sorted(q["qa_id"] for q in a[name]) == sorted(q["qa_id"] for q in b[name])


def test_disjoint_and_complete(qa_pairs: list[QAPair]) -> None:
    splits = make_splits(qa_pairs, RATIOS, seed=42)
    seen: set[str] = set()
    total = 0
    for name in SPLIT_NAMES:
        ids = {q["qa_id"] for q in splits[name]}
        assert seen.isdisjoint(ids), f"split {name} overlaps with earlier splits"
        seen |= ids
        total += len(splits[name])
    assert total == len(qa_pairs)
    assert seen == {q["qa_id"] for q in qa_pairs}


def test_stratification_preserves_type_ratios(qa_pairs: list[QAPair]) -> None:
    splits = make_splits(qa_pairs, RATIOS, seed=42)
    for name, ratio in RATIOS.items():
        counts = Counter(q["type"] for q in splits[name])
        for qtype in ("factoid", "multihop", "synthesis"):
            expected = 120 * ratio
            assert abs(counts[qtype] - expected) <= 1, (
                f"{name}/{qtype}: {counts[qtype]} not within ±1 of {expected}"
            )


def test_ratios_must_sum_to_one(qa_pairs: list[QAPair]) -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        make_splits(qa_pairs, {"train": 0.5, "val": 0.2, "test": 0.2}, seed=42)


def test_ratios_must_have_all_three_keys(qa_pairs: list[QAPair]) -> None:
    with pytest.raises(ValueError, match="missing keys"):
        make_splits(qa_pairs, {"train": 0.8, "test": 0.2}, seed=42)


def test_different_seeds_produce_different_assignments(qa_pairs: list[QAPair]) -> None:
    a = make_splits(qa_pairs, RATIOS, seed=42)
    b = make_splits(qa_pairs, RATIOS, seed=43)
    a_test_ids = [q["qa_id"] for q in a["test"]]
    b_test_ids = [q["qa_id"] for q in b["test"]]
    assert a_test_ids != b_test_ids
