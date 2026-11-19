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


def _layout(per_movie: dict[str, dict[str, int]]):
    """``(y, groups)`` for ``{movie: {class: clips}}``."""
    y, groups = [], []
    for movie, counts in per_movie.items():
        for name, n in counts.items():
            y += [name] * n
            groups += [movie] * n
    return np.array(y), np.array(groups)


def _training_sides_hold(y, folds) -> bool:
    return all(set(y[train]) == set(y) for train, _ in folds)


def test_every_training_fold_holds_a_class_found_in_two_of_three_movies() -> None:
    from trackanno.learn.suggest import _movie_folds

    y, groups = _layout({
        "m0": {"a": 30, "b": 63, "c": 24},
        "m1": {"a": 23, "b": 45, "c": 5},
        "m2": {"a": 71, "b": 77},
    })
    folds = _movie_folds(y, groups)
    assert len(folds) == 3, "each movie is held out alone"
    assert _training_sides_hold(y, folds)
    held_out = np.sort(np.concatenate([test for _, test in folds]))
    assert held_out.tolist() == list(range(len(y))), "every clip is scored exactly once"


def test_folds_keep_every_class_when_no_two_way_split_could() -> None:
    from trackanno.learn.suggest import _movie_folds

    # Each class sits in two of three movies, so any two movies held out lose one.
    y, groups = _layout({
        "m0": {"a": 10, "c": 10},
        "m1": {"a": 10, "b": 10},
        "m2": {"b": 10, "c": 10},
    })
    folds = _movie_folds(y, groups)
    assert len(folds) == 3
    assert _training_sides_hold(y, folds)


def test_many_movies_share_stratified_folds_that_keep_a_rare_class() -> None:
    from trackanno.learn.suggest import MAX_FOLDS, _movie_folds

    per_movie = {f"m{i}": {"a": 20 + i, "b": 15 + i} for i in range(8)}
    per_movie["m0"]["c"] = 6
    per_movie["m1"]["c"] = 4
    y, groups = _layout(per_movie)
    folds = _movie_folds(y, groups)
    assert len(folds) == MAX_FOLDS
    assert _training_sides_hold(y, folds)
    for train, test in folds:
        assert not set(groups[train]) & set(groups[test]), "a movie is on one side only"


def test_a_class_confined_to_one_movie_leaves_no_folds() -> None:
    from trackanno.learn.suggest import _movie_folds

    y, groups = _layout({"m0": {"a": 10, "b": 10}, "m1": {"a": 10}, "m2": {"a": 10}})
    assert _movie_folds(y, groups) == []
    y, groups = _layout({"m0": {"a": 10, "b": 10}})
    assert _movie_folds(y, groups) == []


def _tracked(n_tracks: int, clips_per_track: int, n_movies: int, agree: float, seed: int = 0):
    """Two-class labels and predictions agreeing at rate ``agree``, with movie and track ids."""
    rng = np.random.default_rng(seed)
    n = n_tracks * clips_per_track
    y = rng.choice(np.array(["a", "b"]), size=n)
    flip = rng.random(n) >= agree
    pred = np.where(flip, np.where(y == "a", "b", "a"), y)
    tracks = np.repeat(np.arange(n_tracks), clips_per_track)
    movies = np.array([f"m{t % n_movies}" for t in tracks])
    return y, pred, movies, tracks


def test_the_kappa_bound_sits_below_kappa_and_narrows_with_more_tracks() -> None:
    from sklearn.metrics import cohen_kappa_score

    from trackanno.learn.suggest import _kappa_lower

    gaps = []
    for n_tracks in (20, 320):
        y, pred, movies, tracks = _tracked(n_tracks, 3, n_movies=2, agree=0.8)
        kappa = cohen_kappa_score(y, pred, labels=["a", "b"])
        low = _kappa_lower(y, pred, movies, tracks, ["a", "b"])
        assert low < kappa, f"{low} vs {kappa}"
        assert low == _kappa_lower(y, pred, movies, tracks, ["a", "b"]), "the bound is seeded"
        gaps.append(kappa - low)
    assert gaps[1] < gaps[0] / 2, f"gaps {gaps}"


def test_the_kappa_bound_does_not_clear_zero_on_chance_agreement() -> None:
    from trackanno.learn.suggest import _kappa_lower

    y, pred, movies, tracks = _tracked(60, 3, n_movies=3, agree=0.5)
    assert _kappa_lower(y, pred, movies, tracks, ["a", "b"]) <= 0.0


def test_the_kappa_bound_resamples_tracks_not_clips() -> None:
    from trackanno.learn.suggest import _kappa_lower

    # The same clips, as few long tracks or many short ones: fewer tracks, wider bound.
    y, pred, movies, _ = _tracked(240, 1, n_movies=2, agree=0.8)
    order = np.argsort(movies, kind="stable")
    y, pred, movies = y[order], pred[order], movies[order]
    correct = (y == pred)
    # Sort within each movie so whole tracks are right or wrong together.
    within = np.lexsort((correct, movies))
    y, pred, movies = y[within], pred[within], movies[within]
    long_tracks = np.arange(len(y)) // 20
    short_tracks = np.arange(len(y))
    assert (_kappa_lower(y, pred, movies, long_tracks, ["a", "b"])
            < _kappa_lower(y, pred, movies, short_tracks, ["a", "b"]))


def test_one_class_in_every_resample_gives_no_bound() -> None:
    from trackanno.learn.suggest import _kappa_lower

    y = np.array(["a"] * 12)
    pred = np.array(["a"] * 12)
    movies = np.array(["m0"] * 6 + ["m1"] * 6)
    assert _kappa_lower(y, pred, movies, np.arange(12), ["a", "b"]) == float("-inf")
