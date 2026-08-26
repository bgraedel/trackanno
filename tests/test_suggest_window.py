"""The window length a model was trained at, and the skew it closes.

`FEATURE_VERSION` says what the columns MEAN; it says nothing about how many
frames each row was measured over. Window length changes the feature VALUES
without touching a single column name - `path_length` over 100 frames is simply
twice the `path_length` over 50 - so a model fitted on the annotation grid and
applied to a differently-windowed feature table passes every check
`Suggester.load` makes and then predicts confidently on the wrong scale.

These are claims about the guard, not about the model: whether the length is
recorded, whether it survives the file, whether the mismatch is reported in
words someone can act on, and whether a model saved before any of this existed
still loads.
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from fixtures import make_project
from harness import check

FIXTURES = Path(tempfile.mkdtemp(prefix="trackanno-suggestwin-fixtures-"))
atexit.register(shutil.rmtree, FIXTURES, True)

GROUP = "behaviour"                       # the default project's single-choice group
CLASSES = ("swimming", "immotile")


class StubModel:
    """A fitted classifier without the fit; `load` only needs it to exist."""

    classes_ = np.array(CLASSES)

    def predict_proba(self, X):
        p = np.full(len(np.asarray(X, dtype=float)), 0.9)
        return np.column_stack([p, 1.0 - p])


def _copy_project(root: Path, name: str) -> "object":
    """A project carrying the fixture's prepared tables, and nothing else.

    Copied rather than used in place: `fixtures.make_project` caches one project
    per flavour for the whole session, so training into it would leave a model
    and a feature table behind for every other suite.
    """
    from trackanno.config import load_config
    from trackanno.project import write_config

    fixture = make_project(FIXTURES)
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
    """A loadable project whose model payload records `trained_window`.

    ``None`` writes the payload `train` used to write - no key at all - which is
    what a model saved before this change looks like on disk.
    """
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
    """The rows `train` fits come from clip_features.parquet, which is built over
    the annotation clip manifest, whose windows are `clips.frames` long. That is
    the number the model has to carry, and asked about its own window length it
    must report no problem at all."""
    from trackanno.annotate.store import AnnotationStore
    from trackanno.learn.suggest import Suggester, train

    cfg, manifest = _copy_project(root, "trained")

    # Enough labels in both classes to get past the class-count bail; the fit
    # itself is not the point here and a one-movie project cannot be
    # leave-movie-out cross-validated anyway, so `train` stops before any
    # gradient booster runs. The payload is still written.
    store = AnnotationStore(cfg.paths.csv, cfg)
    for i, row in enumerate(manifest.to_dict("records")):
        store.record(row, {GROUP: CLASSES[i % len(CLASSES)]})

    train(cfg, rebuild_features=True, verbose=False)

    suggester = Suggester(cfg)
    check(
        "the model records the window length its training rows were built at",
        suggester.trained_window == cfg.clips.frames,
        f"recorded {suggester.trained_window!r}, clips.frames is {cfg.clips.frames}",
    )
    check(
        "that length is the annotation clip length, not the stride or the dense grid",
        suggester.trained_window == 50,
        f"got {suggester.trained_window!r}",
    )
    check(
        "a model asked about its own window length reports no problem",
        suggester.window_problem(cfg.clips.frames) == "",
        f"got {suggester.window_problem(cfg.clips.frames)!r}",
    )
    check(
        "an int-like length compares equal rather than by type",
        suggester.window_problem(np.int64(cfg.clips.frames)) == "",
    )


def test_the_recorded_length_survives_a_save_load_round_trip(root: Path) -> None:
    """The length is only a guard if it is still there after the model has been
    written to disk and read back, for a value nothing else in the project would
    have guessed."""
    import joblib

    from trackanno.config import load_config
    from trackanno.learn.suggest import MODEL_FILE, Suggester

    cfg = _stub_model_project(root, "roundtrip", trained_window=87)

    on_disk = joblib.load(cfg.paths.work_dir / MODEL_FILE)
    check(
        "the window length is written into the model payload",
        on_disk.get("trained_window") == 87,
        f"payload has {on_disk.get('trained_window')!r}",
    )

    # Reloaded through a freshly parsed config, so nothing survives in memory.
    suggester = Suggester(load_config(cfg.paths.work_dir.parent / "config.yaml"))
    check("the model still loads", suggester.ready, suggester.problem)
    check(
        "the recorded length comes back unchanged",
        suggester.trained_window == 87,
        f"got {suggester.trained_window!r}",
    )
    check(
        "and it is the length the reloaded model accepts",
        suggester.window_problem(87) == "",
    )
    check(
        "while the project's own clip length is now the mismatching one",
        suggester.window_problem(cfg.clips.frames) != "",
    )


# --------------------------------------------------------------------------- #
# the mismatch
# --------------------------------------------------------------------------- #
def test_a_mismatched_window_is_reported_with_both_numbers(root: Path) -> None:
    """The whole point is that neither number is visible anywhere else: the
    columns line up, the feature version matches, and only the message can say
    what was fitted and what is being predicted on."""
    from trackanno.learn.suggest import Suggester

    cfg = _stub_model_project(root, "mismatch", trained_window=50)
    suggester = Suggester(cfg)
    message = suggester.window_problem(100)

    check("a mismatched window length is a problem", message != "")
    check(
        "the message names the length the model was trained at",
        "50" in message, message,
    )
    check(
        "the message names the length the features were built at",
        "100" in message, message,
    )
    check(
        "the message names the mechanism, not just the numbers",
        "value" in message.lower() and "window length changes" in message.lower(),
        message,
    )
    check(
        "the message names columns whose values actually move with the window",
        sum(c in message for c in ("path_length", "net_displacement", "speed_std")) >= 2,
        message,
    )
    check(
        "the message says what to do about it",
        "retrain" in message.lower() and "restore" in message.lower(),
        message,
    )
    check(
        "a longer window is a problem in the same way a shorter one is",
        suggester.window_problem(25) != "" and "25" in suggester.window_problem(25),
    )
    check(
        "a model that loaded fine is still the one being asked",
        suggester.ready, suggester.problem,
    )


# --------------------------------------------------------------------------- #
# the model that predates the guard
# --------------------------------------------------------------------------- #
def test_a_model_with_no_recorded_window_loads_and_says_so(root: Path) -> None:
    """Every model anyone already has was saved without this key. Refusing them
    would mean a trackanno upgrade silently invalidates a project's model, so an
    unrecorded length is reported as unchecked - a warning the caller can pass
    on - and never raised."""
    from trackanno.learn.suggest import Suggester

    cfg = _stub_model_project(root, "legacy", trained_window=None)

    suggester = Suggester(cfg)          # must not raise
    check(
        "a model saved before the length was recorded still loads",
        suggester.ready, suggester.problem,
    )
    check(
        "loading it does not invent a length",
        suggester.trained_window is None, f"got {suggester.trained_window!r}",
    )
    check(
        "nothing about the load is reported as a problem",
        suggester.problem == "", suggester.problem,
    )

    message = suggester.window_problem(cfg.clips.frames)
    check("an unrecorded length is reported rather than passed", message != "")
    check(
        "the message says the length is unrecorded",
        "does not record" in message.lower() or "unrecorded" in message.lower(),
        message,
    )
    check(
        "the message says it could not be checked, not that it is wrong",
        "could not be checked" in message.lower()
        and "was trained on" not in message.lower(),
        message,
    )
    check(
        "the message still names the length that was asked about",
        str(cfg.clips.frames) in message, message,
    )
    check(
        "asking is safe for any length, since none of them can be compared",
        suggester.window_problem(1) != "" and suggester.window_problem(9999) != "",
    )


def test_the_helper_answers_rather_than_raising(root: Path) -> None:
    """`predict.py` calls this to decide between refusing and warning, so it has
    to return a sentence in every case it can be handed - including a
    Suggester that never loaded a model at all."""
    from trackanno.config import load_config
    from trackanno.project import create_project, write_config
    from trackanno.learn.suggest import Suggester

    project = Path(root) / "empty"
    config_path = create_project(project)
    write_config(config_path, {
        "paths.work_dir": str(project / "work").replace("\\", "/"),
    })
    cfg = load_config(config_path)
    (cfg.paths.work_dir).mkdir(parents=True, exist_ok=True)

    suggester = Suggester(cfg)
    check("there is no model to check", not suggester.ready)
    check(
        "and asking about a window length answers instead of raising",
        isinstance(suggester.window_problem(50), str)
        and suggester.window_problem(50) != "",
    )
