"""The window length a model records at training, and the mismatch guard it enables."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from fixtures import make_project


GROUP = "behaviour"                       # the default project's single-choice group
CLASSES = ("swimming", "immotile")


class StubModel:
    """A fitted classifier without the fit; ``load`` only needs it to exist."""

    classes_ = np.array(CLASSES)

    def predict_proba(self, X):
        p = np.full(len(np.asarray(X, dtype=float)), 0.9)
        return np.column_stack([p, 1.0 - p])


def _copy_project(root: Path, name: str) -> "object":
    """A copy of the shared fixture's prepared tables, so training does not touch it."""
    from trackanno.config import load_config
    from trackanno.project import write_config

    fixture = make_project()
    project = Path(root) / name
    project.mkdir(parents=True, exist_ok=True)
    config_path = project / "config.yaml"
    shutil.copy(fixture, config_path)
    write_config(config_path, {
        "paths.work_dir": str(project / "work").replace("\\", "/"),
        "paths.csv": str(project / "annotations.csv").replace("\\", "/"),
    })
    cfg = load_config(config_path)
    work = cfg.paths.work_dir
    work.mkdir(parents=True, exist_ok=True)

    manifests = []
    for movie_dir in sorted(p for p in load_config(fixture).paths.work_dir.iterdir()
                            if p.is_dir()):
        if not (movie_dir / "clips.parquet").exists():
            continue
        (work / movie_dir.name).mkdir(parents=True, exist_ok=True)
        for fname in ("clips.parquet", "tracks.parquet", "cache.json",
                      "instance_features.npy"):
            if (movie_dir / fname).exists():
                shutil.copy(movie_dir / fname, work / movie_dir.name / fname)
        manifests.append(pd.read_parquet(movie_dir / "clips.parquet"))
    return cfg, pd.concat(manifests, ignore_index=True)


def _stub_model_project(root: Path, name: str, trained_window: int | None):
    """A project whose model payload records ``trained_window`` (``None``: no key)."""
    import joblib

    from trackanno.measure.clipfeatures import CLIP_FEATURES_FILE, FEATURE_VERSION
    from trackanno.learn.suggest import MODEL_FILE

    cfg, manifest = _copy_project(root, name)
    work = cfg.paths.work_dir

    pd.DataFrame({
        "clip_id": manifest["clip_id"].astype(str),
        "movie_id": manifest["movie_id"].astype(str),
        "track_id": manifest["track_id"].astype(int),
        "f_start": manifest["f_start"].astype(int),
        "speed_mean": np.arange(len(manifest), dtype=float),
        "path_length": np.linspace(10.0, 20.0, len(manifest)),
        "_feature_version": FEATURE_VERSION,
    }).to_parquet(work / CLIP_FEATURES_FILE, index=False)

    payload = {
        "columns": ["speed_mean", "path_length"],
        "models": {GROUP: StubModel()},
        "feature_version": FEATURE_VERSION,
    }
    if trained_window is not None:
        payload["trained_window"] = int(trained_window)
    joblib.dump(payload, work / MODEL_FILE)
    return cfg


# --------------------------------------------------------------------------- #
# what train() records
# --------------------------------------------------------------------------- #
def test_a_trained_model_records_the_window_it_was_trained_at(root: Path) -> None:
    """Training rows are clips.frames long, so that is the length the model records."""
    from trackanno.annotate.store import AnnotationStore
    from trackanno.learn.suggest import Suggester, train

    cfg, manifest = _copy_project(root, "trained")

    # One movie cannot be cross-validated, so train stops early but still writes the payload.
    store = AnnotationStore(cfg.paths.csv, cfg)
    for i, row in enumerate(manifest.to_dict("records")):
        store.record(row, {GROUP: CLASSES[i % len(CLASSES)]})

    train(cfg, rebuild_features=True, verbose=False)

    suggester = Suggester(cfg)
    assert suggester.trained_window == cfg.clips.frames == 50, suggester.trained_window
    assert suggester.window_problem(cfg.clips.frames) == "", (
        suggester.window_problem(cfg.clips.frames))
    assert suggester.window_problem(np.int64(cfg.clips.frames)) == "", (
        "an int-like length compares by value")


def test_the_recorded_length_survives_a_save_load_round_trip(root: Path) -> None:
    import joblib

    from trackanno.config import load_config
    from trackanno.learn.suggest import MODEL_FILE, Suggester

    cfg = _stub_model_project(root, "roundtrip", trained_window=87)

    on_disk = joblib.load(cfg.paths.work_dir / MODEL_FILE)
    assert on_disk.get("trained_window") == 87, on_disk.get("trained_window")

    suggester = Suggester(load_config(cfg.paths.work_dir.parent / "config.yaml"))
    assert suggester.ready, suggester.problem
    assert suggester.trained_window == 87, suggester.trained_window
    assert suggester.window_problem(87) == ""
    assert suggester.window_problem(cfg.clips.frames) != "", (
        "the project's own clip length is the mismatch")


# --------------------------------------------------------------------------- #
# the mismatch
# --------------------------------------------------------------------------- #
def test_a_mismatched_window_is_reported_with_both_numbers(root: Path) -> None:
    from trackanno.learn.suggest import Suggester

    cfg = _stub_model_project(root, "mismatch", trained_window=50)
    suggester = Suggester(cfg)
    message = suggester.window_problem(100)

    assert "50" in message and "100" in message, message
    assert "value" in message.lower() and "window length changes" in message.lower(), (
        f"the message names the mechanism: {message}")
    assert sum(c in message for c in ("path_length", "net_displacement", "speed_std")) >= 2, (
        f"the message names window-dependent columns: {message}")
    assert "retrain" in message.lower() and "restore" in message.lower(), message
    assert "25" in suggester.window_problem(25), "a shorter window is a mismatch too"
    assert suggester.ready, suggester.problem


# --------------------------------------------------------------------------- #
# a model without a recorded window
# --------------------------------------------------------------------------- #
def test_a_model_with_no_recorded_window_loads_and_says_so(root: Path) -> None:
    """An unrecorded length is reported as unchecked, never raised."""
    from trackanno.learn.suggest import Suggester

    cfg = _stub_model_project(root, "legacy", trained_window=None)

    suggester = Suggester(cfg)
    assert suggester.ready, suggester.problem
    assert suggester.trained_window is None, suggester.trained_window
    assert suggester.problem == "", suggester.problem

    message = suggester.window_problem(cfg.clips.frames)
    assert "does not record" in message.lower() or "unrecorded" in message.lower(), message
    assert "could not be checked" in message.lower(), message
    assert "was trained on" not in message.lower(), message
    assert str(cfg.clips.frames) in message, message
    assert suggester.window_problem(1) != "" and suggester.window_problem(9999) != ""


def test_the_helper_answers_rather_than_raising(root: Path) -> None:
    """``window_problem`` returns a sentence even when no model is loaded."""
    from trackanno.config import load_config
    from trackanno.project import create_project, write_config
    from trackanno.learn.suggest import Suggester

    project = Path(root) / "empty"
    config_path = create_project(project)
    write_config(config_path, {
        "paths.work_dir": str(project / "work").replace("\\", "/"),
    })
    cfg = load_config(config_path)
    cfg.paths.work_dir.mkdir(parents=True, exist_ok=True)

    suggester = Suggester(cfg)
    assert not suggester.ready
    answer = suggester.window_problem(50)
    assert isinstance(answer, str) and answer != ""
