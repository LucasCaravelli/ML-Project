"""Tests for run_type_router.py and make_router_figures.py."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _write_eval_grid(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _write_oracle_gap(path: Path, oracle_f1: float, baseline_f1: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "oracle_f1_mean": oracle_f1,
        "best_baseline_f1": baseline_f1,
        "per_type": {
            "factoid":  {"best_baseline_f1": baseline_f1},
            "multihop": {"best_baseline_f1": baseline_f1 - 0.05},
            "synthesis":{"best_baseline_f1": baseline_f1 - 0.10},
        },
    }))


CHUNK_SIZES = [128, 256, 512]

# Minimal eval-grid fixture: 6 test questions (2 per type)
EVAL_GRID_ROWS = [
    # factoid: best=128
    {"qa_id": "q0", "split": "test", "type": "factoid",  "chunk_size": 128, "f1": 0.8},
    {"qa_id": "q0", "split": "test", "type": "factoid",  "chunk_size": 256, "f1": 0.5},
    {"qa_id": "q0", "split": "test", "type": "factoid",  "chunk_size": 512, "f1": 0.4},
    {"qa_id": "q1", "split": "test", "type": "factoid",  "chunk_size": 128, "f1": 0.6},
    {"qa_id": "q1", "split": "test", "type": "factoid",  "chunk_size": 256, "f1": 0.3},
    {"qa_id": "q1", "split": "test", "type": "factoid",  "chunk_size": 512, "f1": 0.2},
    # multihop: best=256
    {"qa_id": "q2", "split": "test", "type": "multihop", "chunk_size": 128, "f1": 0.1},
    {"qa_id": "q2", "split": "test", "type": "multihop", "chunk_size": 256, "f1": 0.3},
    {"qa_id": "q2", "split": "test", "type": "multihop", "chunk_size": 512, "f1": 0.2},
    {"qa_id": "q3", "split": "test", "type": "multihop", "chunk_size": 128, "f1": 0.1},
    {"qa_id": "q3", "split": "test", "type": "multihop", "chunk_size": 256, "f1": 0.4},
    {"qa_id": "q3", "split": "test", "type": "multihop", "chunk_size": 512, "f1": 0.2},
    # synthesis: best=512
    {"qa_id": "q4", "split": "test", "type": "synthesis","chunk_size": 128, "f1": 0.05},
    {"qa_id": "q4", "split": "test", "type": "synthesis","chunk_size": 256, "f1": 0.07},
    {"qa_id": "q4", "split": "test", "type": "synthesis","chunk_size": 512, "f1": 0.15},
    {"qa_id": "q5", "split": "test", "type": "synthesis","chunk_size": 128, "f1": 0.03},
    {"qa_id": "q5", "split": "test", "type": "synthesis","chunk_size": 256, "f1": 0.04},
    {"qa_id": "q5", "split": "test", "type": "synthesis","chunk_size": 512, "f1": 0.12},
]


# ---------------------------------------------------------------------------
# _gap_closure
# ---------------------------------------------------------------------------

class TestGapClosure:
    def _fn(self, router_f1, baseline_f1, oracle_f1):
        from experiments.run_type_router import _gap_closure
        return _gap_closure(router_f1, baseline_f1, oracle_f1)

    def test_perfect_router(self):
        assert self._fn(0.3, 0.2, 0.3) == pytest.approx(1.0)

    def test_no_improvement(self):
        assert self._fn(0.2, 0.2, 0.3) == pytest.approx(0.0)

    def test_negative_gap(self):
        result = self._fn(0.1, 0.2, 0.3)
        assert result == pytest.approx(-1.0)

    def test_halfway(self):
        assert self._fn(0.25, 0.2, 0.3) == pytest.approx(0.5)

    def test_none_when_baseline_missing(self):
        assert self._fn(0.25, None, 0.3) is None

    def test_none_when_oracle_missing(self):
        assert self._fn(0.25, 0.2, None) is None

    def test_none_when_zero_denominator(self):
        assert self._fn(0.3, 0.3, 0.3) is None


# ---------------------------------------------------------------------------
# _read_oracle_gap
# ---------------------------------------------------------------------------

class TestReadOracleGap:
    def test_reads_values(self, tmp_path):
        from experiments.run_type_router import _read_oracle_gap

        gap_file = tmp_path / "baselines" / "oracle_gap.json"
        _write_oracle_gap(gap_file, oracle_f1=0.30, baseline_f1=0.21)

        oracle, baseline = _read_oracle_gap(tmp_path)
        assert oracle == pytest.approx(0.30)
        assert baseline == pytest.approx(0.21)

    def test_returns_none_when_missing(self, tmp_path):
        from experiments.run_type_router import _read_oracle_gap

        oracle, baseline = _read_oracle_gap(tmp_path)
        assert oracle is None
        assert baseline is None


# ---------------------------------------------------------------------------
# main() — end-to-end logic (no LLM/GPU required)
# ---------------------------------------------------------------------------

class TestTypeRouterMain:
    @pytest.fixture
    def artifacts_dir(self, tmp_path):
        ad = tmp_path / "artifacts"
        oracle_dir = ad / "oracle"
        oracle_dir.mkdir(parents=True)
        _write_eval_grid(oracle_dir / "eval_grid.jsonl", EVAL_GRID_ROWS)
        _write_oracle_gap(ad / "baselines" / "oracle_gap.json", 0.35, 0.21)
        return ad

    @pytest.fixture
    def results_dir(self, tmp_path):
        rd = tmp_path / "results"
        rd.mkdir()
        return rd

    @pytest.fixture
    def fake_config(self, artifacts_dir, results_dir):
        class Paths:
            pass

        class Chunking:
            sizes = CHUNK_SIZES

        class Config:
            paths = Paths()
            chunking = Chunking()

        cfg = Config()
        cfg.paths.artifacts_dir = artifacts_dir
        cfg.paths.results_dir = results_dir
        return cfg

    def _run(self, fake_config):
        from experiments.run_type_router import main
        main(fake_config, "configs/base.yaml")

    def test_run_produces_output_files(self, fake_config, results_dir):
        self._run(fake_config)
        runs = list((results_dir / "runs").glob("*_type_router"))
        assert len(runs) == 1
        run_dir = runs[0]
        assert (run_dir / "metrics.json").exists()
        assert (run_dir / "predictions.jsonl").exists()

    def test_predictions_cover_all_test_questions(self, fake_config, results_dir):
        self._run(fake_config)
        run_dir = next((results_dir / "runs").glob("*_type_router"))
        from rag_cr.io import read_jsonl
        preds = read_jsonl(run_dir / "predictions.jsonl")
        qa_ids = {r["qa_id"] for r in preds}
        expected = {"q0", "q1", "q2", "q3", "q4", "q5"}
        assert qa_ids == expected

    def test_predicted_sizes_in_chunk_sizes(self, fake_config, results_dir):
        self._run(fake_config)
        run_dir = next((results_dir / "runs").glob("*_type_router"))
        from rag_cr.io import read_jsonl
        for row in read_jsonl(run_dir / "predictions.jsonl"):
            assert row["predicted_chunk_size"] in CHUNK_SIZES

    def test_factoid_routed_to_128(self, fake_config, results_dir):
        self._run(fake_config)
        run_dir = next((results_dir / "runs").glob("*_type_router"))
        from rag_cr.io import read_jsonl
        factoid_preds = [
            r["predicted_chunk_size"] for r in read_jsonl(run_dir / "predictions.jsonl")
            if r["question_type"] == "factoid"
        ]
        assert all(s == 128 for s in factoid_preds)

    def test_multihop_routed_to_256(self, fake_config, results_dir):
        self._run(fake_config)
        run_dir = next((results_dir / "runs").glob("*_type_router"))
        from rag_cr.io import read_jsonl
        preds = [
            r["predicted_chunk_size"] for r in read_jsonl(run_dir / "predictions.jsonl")
            if r["question_type"] == "multihop"
        ]
        assert all(s == 256 for s in preds)

    def test_synthesis_routed_to_512(self, fake_config, results_dir):
        self._run(fake_config)
        run_dir = next((results_dir / "runs").glob("*_type_router"))
        from rag_cr.io import read_jsonl
        preds = [
            r["predicted_chunk_size"] for r in read_jsonl(run_dir / "predictions.jsonl")
            if r["question_type"] == "synthesis"
        ]
        assert all(s == 512 for s in preds)

    def test_mean_f1_matches_manual_computation(self, fake_config, results_dir):
        self._run(fake_config)
        run_dir = next((results_dir / "runs").glob("*_type_router"))
        metrics = json.loads((run_dir / "metrics.json").read_text())

        # factoid → 128: q0=0.8, q1=0.6 → mean 0.7
        # multihop → 256: q2=0.3, q3=0.4 → mean 0.35
        # synthesis → 512: q4=0.15, q5=0.12 → mean 0.135
        expected = float(np.mean([0.8, 0.6, 0.3, 0.4, 0.15, 0.12]))
        assert metrics["mean_f1"] == pytest.approx(expected, abs=1e-6)

    def test_metrics_keys_present(self, fake_config, results_dir):
        self._run(fake_config)
        run_dir = next((results_dir / "runs").glob("*_type_router"))
        metrics = json.loads((run_dir / "metrics.json").read_text())
        for key in ("mean_f1", "gap_closure_fraction", "n_test_examples",
                    "policy", "type_best_sizes"):
            assert key in metrics, f"Missing key: {key}"

    def test_policy_field_is_type_aware(self, fake_config, results_dir):
        self._run(fake_config)
        run_dir = next((results_dir / "runs").glob("*_type_router"))
        metrics = json.loads((run_dir / "metrics.json").read_text())
        assert metrics["policy"] == "type_aware"

    def test_n_test_examples_correct(self, fake_config, results_dir):
        self._run(fake_config)
        run_dir = next((results_dir / "runs").glob("*_type_router"))
        metrics = json.loads((run_dir / "metrics.json").read_text())
        assert metrics["n_test_examples"] == 6

    def test_train_rows_ignored(self, fake_config, artifacts_dir, results_dir):
        """Rows with split != 'test' must not affect routing or scoring."""
        extra = [
            {"qa_id": "train_q", "split": "train", "type": "factoid",
             "chunk_size": 256, "f1": 0.99},
        ]
        oracle_dir = artifacts_dir / "oracle"
        grid_path = oracle_dir / "eval_grid.jsonl"
        with grid_path.open("a") as f:
            for row in extra:
                f.write(json.dumps(row) + "\n")

        self._run(fake_config)
        run_dir = next((results_dir / "runs").glob("*_type_router"))
        metrics = json.loads((run_dir / "metrics.json").read_text())
        assert metrics["n_test_examples"] == 6  # unchanged

    def test_missing_eval_grid_raises(self, fake_config, artifacts_dir):
        (artifacts_dir / "oracle" / "eval_grid.jsonl").unlink()
        with pytest.raises(FileNotFoundError):
            from experiments.run_type_router import main
            main(fake_config, "configs/base.yaml")


# ---------------------------------------------------------------------------
# make_router_figures — _latest helper
# ---------------------------------------------------------------------------

class TestLatestHelper:
    def test_returns_most_recent(self, tmp_path):
        from experiments.make_router_figures import _latest

        results_dir = tmp_path / "results"
        for ts in ["20260101_000000", "20260201_000000", "20260301_000000"]:
            run = results_dir / "runs" / f"{ts}_router"
            run.mkdir(parents=True)
            (run / "metrics.json").write_text("{}")

        found = _latest(results_dir, "router", "metrics.json")
        assert found is not None
        assert "20260301_000000" in str(found)

    def test_returns_none_when_no_runs(self, tmp_path):
        from experiments.make_router_figures import _latest

        results_dir = tmp_path / "results"
        results_dir.mkdir()
        assert _latest(results_dir, "router", "metrics.json") is None

    def test_ignores_other_tags(self, tmp_path):
        from experiments.make_router_figures import _latest

        results_dir = tmp_path / "results"
        run = results_dir / "runs" / "20260101_000000_type_router"
        run.mkdir(parents=True)
        (run / "metrics.json").write_text("{}")

        assert _latest(results_dir, "router", "metrics.json") is None


# ---------------------------------------------------------------------------
# make_router_figures — figure generation (smoke tests, no display)
# ---------------------------------------------------------------------------

class TestRouterFigures:
    @pytest.fixture(autouse=True)
    def no_display(self):
        import matplotlib
        matplotlib.use("Agg")

    @pytest.fixture
    def setup_dirs(self, tmp_path):
        artifacts_dir = tmp_path / "artifacts"
        results_dir = tmp_path / "results"

        oracle_dir = artifacts_dir / "oracle"
        oracle_dir.mkdir(parents=True)
        _write_eval_grid(oracle_dir / "eval_grid.jsonl", EVAL_GRID_ROWS)
        _write_oracle_gap(
            artifacts_dir / "baselines" / "oracle_gap.json",
            oracle_f1=0.35, baseline_f1=0.21,
        )

        # Fake router run
        router_run = results_dir / "runs" / "20260101_000000_router"
        router_run.mkdir(parents=True)
        (router_run / "metrics.json").write_text(json.dumps({"mean_f1": 0.17}))
        preds = [
            {"qa_id": "q0", "question_type": "factoid",  "f1": 0.6},
            {"qa_id": "q1", "question_type": "factoid",  "f1": 0.5},
            {"qa_id": "q2", "question_type": "multihop", "f1": 0.2},
            {"qa_id": "q3", "question_type": "multihop", "f1": 0.1},
            {"qa_id": "q4", "question_type": "synthesis","f1": 0.05},
            {"qa_id": "q5", "question_type": "synthesis","f1": 0.03},
        ]
        with (router_run / "predictions.jsonl").open("w") as f:
            for r in preds:
                f.write(json.dumps(r) + "\n")

        # Fake type_router run
        type_run = results_dir / "runs" / "20260101_000001_type_router"
        type_run.mkdir(parents=True)
        (type_run / "metrics.json").write_text(json.dumps({"mean_f1": 0.229}))
        with (type_run / "predictions.jsonl").open("w") as f:
            for r in preds:
                f.write(json.dumps(r) + "\n")

        out_dir = results_dir / "figures"
        out_dir.mkdir(parents=True)
        return artifacts_dir, results_dir, out_dir

    def test_comparison_figure_created(self, setup_dirs):
        from experiments.make_router_figures import _figure_router_comparison
        artifacts_dir, results_dir, out_dir = setup_dirs
        _figure_router_comparison(out_dir, artifacts_dir, results_dir)
        assert (out_dir / "fig_router_comparison.pdf").exists()
        assert (out_dir / "fig_router_comparison.png").exists()

    def test_per_type_figure_created(self, setup_dirs):
        from experiments.make_router_figures import _figure_router_per_type
        artifacts_dir, results_dir, out_dir = setup_dirs
        _figure_router_per_type(out_dir, artifacts_dir, results_dir)
        assert (out_dir / "fig_router_per_type.pdf").exists()
        assert (out_dir / "fig_router_per_type.png").exists()

    def test_comparison_figure_skips_gracefully_without_router_run(self, tmp_path):
        from experiments.make_router_figures import _figure_router_comparison

        artifacts_dir = tmp_path / "artifacts"
        results_dir = tmp_path / "results"
        out_dir = results_dir / "figures"
        out_dir.mkdir(parents=True)
        _write_oracle_gap(
            artifacts_dir / "baselines" / "oracle_gap.json",
            oracle_f1=0.35, baseline_f1=0.21,
        )
        # No router run — must not raise, just skip
        _figure_router_comparison(out_dir, artifacts_dir, results_dir)
        assert not (out_dir / "fig_router_comparison.pdf").exists()

    def test_comparison_figure_skips_gracefully_without_oracle_gap(self, tmp_path):
        from experiments.make_router_figures import _figure_router_comparison

        results_dir = tmp_path / "results"
        out_dir = results_dir / "figures"
        out_dir.mkdir(parents=True)
        # No oracle_gap.json — must not raise
        _figure_router_comparison(out_dir, tmp_path / "artifacts", results_dir)
        assert not (out_dir / "fig_router_comparison.pdf").exists()
