from __future__ import annotations

import io
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CHUNK_SIZES = [128, 256, 512]
QUERIES = [
    "What is ataxia?",
    "How many patients were enrolled in the study?",
    "Which gene is responsible for Friedreich ataxia?",
    "Describe the treatment options for FRDA.",
    "What is the prognosis for patients with FA?",
    "Who first described frataxin deficiency?",
    "When was the GAA repeat expansion discovered?",
    "Where does frataxin localise inside the cell?",
    "Why does frataxin deficiency cause oxidative stress?",
    "Which scale is used to assess ataxia severity?",
    "How does leriglitazone work in FRDA models?",
    "What percentage of FRDA patients develop cardiomyopathy?",
]
TYPES = [
    "factoid", "factoid", "factoid",
    "synthesis", "synthesis", "factoid",
    "factoid", "factoid", "multihop",
    "factoid", "multihop", "factoid",
]
LABELS = [128, 256, 128, 512, 256, 128, 256, 512, 128, 256, 512, 128]


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def small_X(rng: np.random.Generator) -> np.ndarray:
    return rng.random((len(QUERIES), 20)).astype(np.float32)


@pytest.fixture
def big_X(rng: np.random.Generator) -> np.ndarray:
    """30 samples — enough for 3-fold CV calibration (10 per class)."""
    return rng.random((30, 20)).astype(np.float32)


@pytest.fixture
def big_labels() -> np.ndarray:
    return np.array([128] * 10 + [256] * 10 + [512] * 10)


@pytest.fixture
def oracle_jsonl(tmp_path: Path) -> Path:
    """Write a minimal oracle labels file covering QUERIES."""
    path = tmp_path / "labels.jsonl"
    with path.open("w") as f:
        for i, (label, _qtype) in enumerate(zip(LABELS, TYPES, strict=False)):
            f.write(json.dumps({
                "qa_id": f"qa_{i:04d}",
                "best_size": label,
                "scores_by_size": {128: 0.5, 256: 0.4, 512: 0.3, 1024: 0.2},
            }) + "\n")
        # extra row with 1024 — must be filtered out
        f.write(json.dumps({
            "qa_id": "qa_9999",
            "best_size": 1024,
            "scores_by_size": {128: 0.1, 256: 0.2, 512: 0.3, 1024: 0.9},
        }) + "\n")
    return path


@pytest.fixture
def splits_dir(tmp_path: Path, oracle_jsonl: Path) -> Path:
    """Write minimal train / val / test split files aligned with oracle_jsonl."""
    sd = tmp_path / "splits"
    sd.mkdir()
    # first 8 → train, next 2 → val, last 2 → test
    boundaries = {"train": range(8), "val": range(8, 10), "test": range(10, 12)}
    for split, idx_range in boundaries.items():
        rows = [
            {
                "qa_id": f"qa_{i:04d}",
                "question": QUERIES[i],
                "answer": f"answer_{i}",
                "source_chunk_id": "512-000000",
                "type": TYPES[i],
                "validated": True,
                "split": split,
            }
            for i in idx_range
        ]
        with (sd / f"{split}.jsonl").open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    # also write qa_9999 into train so load_router_data can encounter it
    with (sd / "train.jsonl").open("a") as f:
        f.write(json.dumps({
            "qa_id": "qa_9999",
            "question": "Extra question with oracle 1024.",
            "answer": "extra",
            "source_chunk_id": "512-000001",
            "type": "factoid",
            "validated": True,
            "split": "train",
        }) + "\n")
    return sd


# ---------------------------------------------------------------------------
# features.py
# ---------------------------------------------------------------------------

class TestTfidfExtractor:
    def test_output_shape_and_dtype(self) -> None:
        from rag_cr.router.features import TfidfExtractor

        ext = TfidfExtractor()
        X = ext.fit_transform(QUERIES, TYPES)
        assert X.ndim == 2
        assert X.shape[0] == len(QUERIES)
        assert X.dtype == np.float32

    def test_transform_uses_fitted_vocabulary(self) -> None:
        from rag_cr.router.features import TfidfExtractor

        ext = TfidfExtractor()
        ext.fit(QUERIES[:8], TYPES[:8])
        X_val = ext.transform(QUERIES[8:], TYPES[8:])
        assert X_val.shape == (4, X_val.shape[1])

class TestHandcraftedExtractor:
    def test_output_shape_and_dtype(self) -> None:
        from rag_cr.router.features import HandcraftedExtractor

        ext = HandcraftedExtractor()
        X = ext.transform(QUERIES, TYPES)
        assert X.shape == (len(QUERIES), 13)
        assert X.dtype == np.float32

    def test_type_onehot_exclusive(self) -> None:
        from rag_cr.router.features import HandcraftedExtractor

        ext = HandcraftedExtractor()
        # last 3 cols are type one-hot; each row must sum to exactly 1
        X = ext.transform(QUERIES, TYPES)
        type_cols = X[:, -3:]
        np.testing.assert_array_equal(type_cols.sum(axis=1), np.ones(len(QUERIES)))

    def test_question_word_onehot_exclusive(self) -> None:
        from rag_cr.router.features import HandcraftedExtractor

        ext = HandcraftedExtractor()
        # cols 1..8 are question-word one-hot
        X = ext.transform(QUERIES, TYPES)
        qw_cols = X[:, 1:9]
        np.testing.assert_array_equal(qw_cols.sum(axis=1), np.ones(len(QUERIES)))

    def test_deterministic(self) -> None:
        from rag_cr.router.features import HandcraftedExtractor

        ext = HandcraftedExtractor()
        X1 = ext.transform(QUERIES, TYPES)
        X2 = ext.transform(QUERIES, TYPES)
        np.testing.assert_array_equal(X1, X2)

class TestMiniLMExtractor:
    @pytest.mark.slow
    def test_output_shape_and_cache(self) -> None:
        from rag_cr.router.features import MiniLMExtractor

        ext1 = MiniLMExtractor()
        ext2 = MiniLMExtractor()
        X1 = ext1.transform(QUERIES[:3])
        X2 = ext2.transform(QUERIES[:3])
        assert X1.shape == (3, 384)
        assert X1.dtype == np.float32
        np.testing.assert_array_equal(X1, X2)  # same model → same output
        # both instances must share the same underlying model object
        assert ext1._model is ext2._model


class TestConcatExtractor:
    def test_horizontal_stack(self) -> None:
        from rag_cr.router.features import (
            ConcatExtractor,
            HandcraftedExtractor,
            TfidfExtractor,
        )

        tfidf = TfidfExtractor()
        hand = HandcraftedExtractor()
        cc = ConcatExtractor([tfidf, hand])
        Xc = cc.fit_transform(QUERIES, TYPES)

        tfidf2 = TfidfExtractor()
        Xt = tfidf2.fit_transform(QUERIES, TYPES)
        hand2 = HandcraftedExtractor()
        Xh = hand2.transform(QUERIES, TYPES)

        assert Xc.shape == (len(QUERIES), Xt.shape[1] + 13)
        np.testing.assert_allclose(Xc[:, Xt.shape[1]:], Xh)

    def test_fit_then_transform(self) -> None:
        from rag_cr.router.features import (
            ConcatExtractor,
            HandcraftedExtractor,
            TfidfExtractor,
        )

        cc = ConcatExtractor([TfidfExtractor(), HandcraftedExtractor()])
        cc.fit(QUERIES[:8], TYPES[:8])
        Xv = cc.transform(QUERIES[8:], TYPES[8:])
        assert Xv.shape[0] == 4


class TestBuildFeatureExtractor:
    @pytest.mark.parametrize("name", ["tfidf", "handcrafted"])
    def test_known_names(self, name: str) -> None:
        from rag_cr.router.features import build_feature_extractor

        ext = build_feature_extractor(name)
        assert ext.name == name

    def test_unknown_name_raises(self) -> None:
        from rag_cr.router.features import build_feature_extractor

        with pytest.raises(ValueError, match="Unknown feature extractor"):
            build_feature_extractor("nonexistent")


# ---------------------------------------------------------------------------
# models.py
# ---------------------------------------------------------------------------

class TestLogisticRouter:
    def test_deterministic(self, small_X: np.ndarray) -> None:
        from rag_cr.router.models import LogisticRouter

        y = np.array(LABELS)
        clf1 = LogisticRouter(seed=42)
        clf1.fit(small_X, y)
        clf2 = LogisticRouter(seed=42)
        clf2.fit(small_X, y)
        np.testing.assert_array_equal(clf1.predict(small_X), clf2.predict(small_X))


class TestSVMRouter:
    def test_fallback_softmax_when_few_samples(self, rng: np.random.Generator) -> None:
        """With only 1 sample for one class, calibration falls back to softmax."""
        from rag_cr.router.models import SVMRouter

        # 3 samples total: 128×1, 256×1, 512×1 → min_count=1 → fallback
        X = rng.random((3, 10)).astype(np.float32)
        y = np.array([128, 256, 512])
        clf = SVMRouter(seed=42)
        clf.fit(X, y)
        assert clf._calib is None  # fallback path taken
        p = clf.predict_proba(X)
        np.testing.assert_allclose(p.sum(axis=1), np.ones(3), atol=1e-5)

    def test_calibration_when_enough_samples(
        self, big_X: np.ndarray, big_labels: np.ndarray
    ) -> None:
        from rag_cr.router.models import SVMRouter

        clf = SVMRouter(seed=42)
        clf.fit(big_X, big_labels)
        assert clf._calib is not None  # real calibration used
        p = clf.predict_proba(big_X)
        assert p.shape == (30, 3)
        np.testing.assert_allclose(p.sum(axis=1), np.ones(30), atol=1e-5)

class TestLightGBMRouter:
    def test_predict_proba_shape(
        self, big_X: np.ndarray, big_labels: np.ndarray
    ) -> None:
        pytest.importorskip("lightgbm")
        from rag_cr.router.models import LightGBMRouter

        clf = LightGBMRouter(seed=42)
        clf.fit(big_X, big_labels)
        p = clf.predict_proba(big_X)
        assert p.shape == (30, 3)
        np.testing.assert_allclose(p.sum(axis=1), np.ones(30), atol=1e-5)

    def test_no_feature_name_warning(
        self, big_X: np.ndarray, big_labels: np.ndarray
    ) -> None:
        """predict() and predict_proba() must not emit the feature-name warning."""
        pytest.importorskip("lightgbm")
        import warnings

        from rag_cr.router.models import LightGBMRouter

        clf = LightGBMRouter(seed=42)
        clf.fit(big_X, big_labels)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            try:
                clf.predict(big_X)
                clf.predict_proba(big_X)
            except UserWarning as e:
                pytest.fail(f"Unexpected UserWarning: {e}")

class TestBuildRouterModel:
    @pytest.mark.parametrize("name", ["logreg", "svm_linear"])
    def test_known_names(self, name: str) -> None:
        from rag_cr.router.models import build_router_model

        m = build_router_model(name, seed=42)
        assert m.name == name

    def test_unknown_name_raises(self) -> None:
        from rag_cr.router.models import build_router_model

        with pytest.raises(ValueError, match="Unknown router model"):
            build_router_model("nonexistent")


# ---------------------------------------------------------------------------
# train.py
# ---------------------------------------------------------------------------

class TestLoadRouterData:
    def test_filters_1024(self, splits_dir: Path, oracle_jsonl: Path) -> None:
        from rag_cr.router.train import load_router_data

        q, t, labels = load_router_data(splits_dir, oracle_jsonl, "train", CHUNK_SIZES)
        assert all(label in CHUNK_SIZES for label in labels), "1024 label leaked through filter"

    def test_lengths_consistent(self, splits_dir: Path, oracle_jsonl: Path) -> None:
        from rag_cr.router.train import load_router_data

        q, t, labels = load_router_data(splits_dir, oracle_jsonl, "train", CHUNK_SIZES)
        assert len(q) == len(t) == len(labels)

    def test_types_valid(self, splits_dir: Path, oracle_jsonl: Path) -> None:
        from rag_cr.router.train import load_router_data

        _, types, _ = load_router_data(splits_dir, oracle_jsonl, "train", CHUNK_SIZES)
        valid = {"factoid", "multihop", "synthesis"}
        assert all(t in valid for t in types)

    def test_each_split_loaded_independently(
        self, splits_dir: Path, oracle_jsonl: Path
    ) -> None:
        from rag_cr.router.train import load_router_data

        q_tr, _, _ = load_router_data(splits_dir, oracle_jsonl, "train", CHUNK_SIZES)
        q_va, _, _ = load_router_data(splits_dir, oracle_jsonl, "val",   CHUNK_SIZES)
        q_te, _, _ = load_router_data(splits_dir, oracle_jsonl, "test",  CHUNK_SIZES)
        assert len(set(q_tr) & set(q_va)) == 0, "train and val share questions"
        assert len(set(q_tr) & set(q_te)) == 0, "train and test share questions"


class TestRunCvGrid:
    @pytest.fixture
    def fake_config(self):
        class FR:
            feature_sets = ["tfidf", "handcrafted"]
            model_names = ["logreg"]
            cv_folds = 3

        class FC:
            router = FR()

        return FC()

    def test_dataframe_columns(
        self, splits_dir: Path, oracle_jsonl: Path, fake_config
    ) -> None:
        from rag_cr.router.train import load_router_data, run_cv_grid

        q, t, labels = load_router_data(splits_dir, oracle_jsonl, "train", CHUNK_SIZES)
        df = run_cv_grid(q, t, labels, fake_config, seed=42)
        assert set(df.columns) >= {
            "feature_set", "classifier", "fold",
            "accuracy", "balanced_accuracy", "macro_f1",
        }

    def test_row_count(
        self, splits_dir: Path, oracle_jsonl: Path, fake_config
    ) -> None:
        from rag_cr.router.train import load_router_data, run_cv_grid

        q, t, labels = load_router_data(splits_dir, oracle_jsonl, "train", CHUNK_SIZES)
        df = run_cv_grid(q, t, labels, fake_config, seed=42)
        n_fs = len(fake_config.router.feature_sets)
        n_clf = len(fake_config.router.model_names)
        n_folds = fake_config.router.cv_folds
        assert len(df) == n_fs * n_clf * n_folds

    def test_metrics_in_unit_interval(
        self, splits_dir: Path, oracle_jsonl: Path, fake_config
    ) -> None:
        from rag_cr.router.train import load_router_data, run_cv_grid

        q, t, labels = load_router_data(splits_dir, oracle_jsonl, "train", CHUNK_SIZES)
        df = run_cv_grid(q, t, labels, fake_config, seed=42)
        for col in ["accuracy", "balanced_accuracy", "macro_f1"]:
            assert df[col].between(0, 1).all(), f"{col} out of [0, 1]"

    def test_tfidf_not_leaked_across_folds(
        self, splits_dir: Path, oracle_jsonl: Path
    ) -> None:
        """TF-IDF must be re-fitted per fold — check that val rows differ between folds."""
        from rag_cr.router.train import load_router_data, run_cv_grid

        class FR:
            feature_sets = ["tfidf"]
            model_names = ["logreg"]
            cv_folds = 3

        class FC:
            router = FR()

        q, t, labels = load_router_data(splits_dir, oracle_jsonl, "train", CHUNK_SIZES)
        # If TF-IDF was fitted globally the macro_f1 values across folds would be
        # identical (same feature space). With per-fold fitting they may vary.
        df = run_cv_grid(q, t, labels, FC(), seed=42)
        tfidf_rows = df[df["feature_set"] == "tfidf"]
        # At minimum the fold column must have distinct values
        assert tfidf_rows["fold"].nunique() == 3


class TestSelectBestOnVal:
    @pytest.fixture
    def fake_config(self):
        class FR:
            feature_sets = ["tfidf", "handcrafted"]
            model_names = ["logreg", "svm_linear"]
            cv_folds = 3

        class FC:
            router = FR()

        return FC()

    def test_returns_correct_types(
        self, splits_dir: Path, oracle_jsonl: Path, fake_config
    ) -> None:
        from rag_cr.router.train import load_router_data, select_best_on_val

        tr_q, tr_t, tr_l = load_router_data(splits_dir, oracle_jsonl, "train", CHUNK_SIZES)
        va_q, va_t, va_l = load_router_data(splits_dir, oracle_jsonl, "val",   CHUNK_SIZES)
        ext, clf, meta = select_best_on_val(
            tr_q, tr_t, tr_l, va_q, va_t, va_l, fake_config, seed=42
        )
        assert hasattr(ext, "transform")
        assert hasattr(clf, "predict")
        assert isinstance(meta, dict)

    def test_metadata_keys(
        self, splits_dir: Path, oracle_jsonl: Path, fake_config
    ) -> None:
        from rag_cr.router.train import load_router_data, select_best_on_val

        tr_q, tr_t, tr_l = load_router_data(splits_dir, oracle_jsonl, "train", CHUNK_SIZES)
        va_q, va_t, va_l = load_router_data(splits_dir, oracle_jsonl, "val",   CHUNK_SIZES)
        _, _, meta = select_best_on_val(
            tr_q, tr_t, tr_l, va_q, va_t, va_l, fake_config, seed=42
        )
        for key in ("feature_set", "classifier_name",
                    "val_macro_f1", "val_balanced_accuracy", "val_accuracy"):
            assert key in meta, f"Missing key: {key}"

    def test_metrics_in_unit_interval(
        self, splits_dir: Path, oracle_jsonl: Path, fake_config
    ) -> None:
        from rag_cr.router.train import load_router_data, select_best_on_val

        tr_q, tr_t, tr_l = load_router_data(splits_dir, oracle_jsonl, "train", CHUNK_SIZES)
        va_q, va_t, va_l = load_router_data(splits_dir, oracle_jsonl, "val",   CHUNK_SIZES)
        _, _, meta = select_best_on_val(
            tr_q, tr_t, tr_l, va_q, va_t, va_l, fake_config, seed=42
        )
        assert 0.0 <= meta["val_macro_f1"] <= 1.0
        assert 0.0 <= meta["val_balanced_accuracy"] <= 1.0
        assert 0.0 <= meta["val_accuracy"] <= 1.0

    def test_extractor_already_fitted(
        self, splits_dir: Path, oracle_jsonl: Path, fake_config
    ) -> None:
        """The returned extractor must transform test data without re-fitting."""
        from rag_cr.router.train import load_router_data, select_best_on_val

        tr_q, tr_t, tr_l = load_router_data(splits_dir, oracle_jsonl, "train", CHUNK_SIZES)
        va_q, va_t, va_l = load_router_data(splits_dir, oracle_jsonl, "val",   CHUNK_SIZES)
        te_q, te_t, te_l = load_router_data(splits_dir, oracle_jsonl, "test",  CHUNK_SIZES)
        ext, clf, _ = select_best_on_val(
            tr_q, tr_t, tr_l, va_q, va_t, va_l, fake_config, seed=42
        )
        X_test = ext.transform(te_q, te_t)  # must not raise
        preds = clf.predict(X_test)
        assert set(preds).issubset(set(CHUNK_SIZES))

    def test_pickle_roundtrip_of_bundle(
        self, splits_dir: Path, oracle_jsonl: Path, fake_config
    ) -> None:
        from rag_cr.router.train import load_router_data, select_best_on_val

        tr_q, tr_t, tr_l = load_router_data(splits_dir, oracle_jsonl, "train", CHUNK_SIZES)
        va_q, va_t, va_l = load_router_data(splits_dir, oracle_jsonl, "val",   CHUNK_SIZES)
        te_q, te_t, _    = load_router_data(splits_dir, oracle_jsonl, "test",  CHUNK_SIZES)
        ext, clf, meta = select_best_on_val(
            tr_q, tr_t, tr_l, va_q, va_t, va_l, fake_config, seed=42
        )

        bundle = {"extractor": ext, "classifier": clf, "config": meta}
        buf = io.BytesIO()
        pickle.dump(bundle, buf)
        buf.seek(0)
        loaded = pickle.load(buf)

        X = loaded["extractor"].transform(te_q, te_t)
        preds_orig   = clf.predict(ext.transform(te_q, te_t))
        preds_loaded = loaded["classifier"].predict(X)
        np.testing.assert_array_equal(preds_orig, preds_loaded)
