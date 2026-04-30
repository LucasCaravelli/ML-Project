from __future__ import annotations

from rag_cr.oracle import label_from_grid


def _row(qa_id: str, size: int, f1: float, em: float = 0.0) -> dict:
    return {"qa_id": qa_id, "chunk_size": size, "f1": f1, "em": em}


def test_picks_highest_f1():
    rows = [
        _row("q1", 128, 0.5),
        _row("q1", 256, 0.8),
        _row("q1", 512, 0.7),
        _row("q1", 1024, 0.6),
    ]
    labels = label_from_grid(rows)
    assert len(labels) == 1
    assert labels[0]["qa_id"] == "q1"
    assert labels[0]["best_size"] == 256
    assert labels[0]["scores_by_size"] == {128: 0.5, 256: 0.8, 512: 0.7, 1024: 0.6}


def test_tiebreak_f1_tied_higher_em_wins():
    rows = [
        _row("q1", 128, f1=0.7, em=0.0),
        _row("q1", 256, f1=0.7, em=1.0),
        _row("q1", 512, f1=0.7, em=0.0),
    ]
    labels = label_from_grid(rows)
    assert labels[0]["best_size"] == 256


def test_tiebreak_f1_and_em_tied_smaller_size_wins():
    rows = [
        _row("q1", 1024, f1=0.7, em=1.0),
        _row("q1", 512, f1=0.7, em=1.0),
        _row("q1", 256, f1=0.7, em=1.0),
        _row("q1", 128, f1=0.7, em=1.0),
    ]
    labels = label_from_grid(rows)
    assert labels[0]["best_size"] == 128


def test_f1_strictly_dominates_em():
    rows = [
        _row("q1", 128, f1=0.8, em=0.0),
        _row("q1", 256, f1=0.7, em=1.0),
    ]
    labels = label_from_grid(rows)
    assert labels[0]["best_size"] == 128


def test_multiple_qa_ids_grouped_independently():
    rows = [
        _row("q1", 128, 0.9),
        _row("q1", 256, 0.5),
        _row("q2", 128, 0.3),
        _row("q2", 512, 0.6),
    ]
    labels = label_from_grid(rows)
    by_id = {label["qa_id"]: label for label in labels}
    assert by_id["q1"]["best_size"] == 128
    assert by_id["q2"]["best_size"] == 512


def test_output_schema_matches_oracle_label():
    rows = [_row("q1", 128, 0.5)]
    labels = label_from_grid(rows)
    label = labels[0]
    assert set(label.keys()) == {"qa_id", "best_size", "scores_by_size"}
    assert isinstance(label["best_size"], int)
    assert isinstance(label["scores_by_size"], dict)


def test_extra_fields_in_grid_rows_are_ignored():
    rows = [
        {"qa_id": "q1", "chunk_size": 128, "f1": 0.9, "em": 1.0,
         "split": "test", "type": "factoid", "prediction": "...", "passages": []},
        {"qa_id": "q1", "chunk_size": 256, "f1": 0.5, "em": 0.0,
         "split": "test", "type": "factoid", "prediction": "...", "passages": []},
    ]
    labels = label_from_grid(rows)
    assert labels[0]["best_size"] == 128
