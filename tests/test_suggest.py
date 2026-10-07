"""The Suggester's load: a stale or missing model reports itself instead of going quiet."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


class _ConstantModel:
    classes_ = np.array(["x", "y"])

    def predict_proba(self, X):
        return np.array([[0.2, 0.8]])


def _stub_model_files(cfg, columns, feature_columns=()):
    """A feature parquet plus a model payload, without training anything."""
    import joblib
    import pandas as pd

    from trackanno.learn.suggest import MODEL_FILE
    from trackanno.measure.clipfeatures import CLIP_FEATURES_FILE, FEATURE_VERSION

    work = cfg.paths.work_dir
    work.mkdir(parents=True, exist_ok=True)
    frame = {"clip_id": ["a", "b"], "movie_id": ["m", "m"],
             "track_id": [1, 2], "f_start": [0, 0],
             "_feature_version": [FEATURE_VERSION, FEATURE_VERSION]}
    for i, name in enumerate(feature_columns):
        frame[name] = [float(i), float(i) + 1.0]
    pd.DataFrame(frame).to_parquet(work / CLIP_FEATURES_FILE, index=False)
    joblib.dump({"columns": list(columns), "models": {"state": _ConstantModel()},
                 "feature_version": FEATURE_VERSION},
                work / MODEL_FILE)


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch):
    from trackanno.config import load_config
    from trackanno.project import create_project

    monkeypatch.setenv("TRACKANNO_STATE_DIR", str(tmp_path / "state"))
    return load_config(create_project(tmp_path / "stale"))


def test_a_model_fitted_on_missing_columns_says_so(cfg) -> None:
    from trackanno.learn.suggest import Suggester

    _stub_model_files(cfg, columns=["speed_mean", "wave_amp_asym"],
                      feature_columns=["speed_mean"])
    s = Suggester(cfg)
    assert not s.ready
    assert "wave_amp_asym" in s.problem, s.problem
    assert "no model yet" not in s.status_line(), s.status_line()
    assert s.status_line() == s.problem


def test_reloading_after_a_retrain_recovers(cfg) -> None:
    from trackanno.learn.suggest import Suggester

    _stub_model_files(cfg, columns=["speed_mean", "wave_amp_asym"],
                      feature_columns=["speed_mean"])
    s = Suggester(cfg)
    _stub_model_files(cfg, columns=["speed_mean"], feature_columns=["speed_mean"])
    assert s.load() and s.ready
    assert s.problem == ""
    assert s.suggest("b") == {"state": ("y", 0.8)}


def test_an_absent_model_is_reported_as_absent(cfg) -> None:
    from trackanno.learn.suggest import MODEL_FILE, Suggester

    _stub_model_files(cfg, columns=["speed_mean"], feature_columns=["speed_mean"])
    s = Suggester(cfg)
    (cfg.paths.work_dir / MODEL_FILE).unlink()
    assert not s.load()
    assert "no model has been trained" in s.problem, s.problem
