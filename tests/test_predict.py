"""Labelling every clip with the model, and exporting movies coloured by it.

Every check here is about provenance. A prediction has to stay recognisable as a
prediction at each stage: in the file it is written to (never the annotation
CSV), in the row that says whether a person judged the clip, and in the pixels
of the exported movie. The rest guards the two ways a bulk run can quietly go
wrong - labelling a whole project with a model the report gated off, and leaving
a half-written parquet behind under the real name.

One file, two kinds of row. With ``predict.decode: none`` - the shipped default -
a row is a CLIP or a dense WINDOW and everything below the "what it writes"
heading is a claim about that path, unchanged. With a decode a row is a BOUT,
cut where the per-frame posterior says the behaviour changed, and the claims
about those live under "decoded bouts". The two are deliberately not merged: the
whole argument for the decode is that it is OPTIONAL, and the way to hold this
code to that is a set of assertions the default has to keep passing verbatim.

Nothing here needs a GPU, a display or the repository's own data: the project,
the clip manifest and the "trained" model are all built by this file.
"""

from __future__ import annotations

import atexit
import hashlib
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from fixtures import make_project
from harness import check

# The clip manifest under test is a real one, written by the real clip cache
# builder, so f_end and the clip ids are exactly what a prepared project has.
FIXTURES = Path(tempfile.mkdtemp(prefix="trackanno-predict-fixtures-"))
atexit.register(shutil.rmtree, FIXTURES, True)

GROUP = "behaviour"                       # the default project's single-choice group
CLASSES = ("swimming", "immotile")

# The planted movie the boundary claims are measured on: one track, one change of
# behaviour, at a frame that is a multiple of neither the window length nor the
# stride, so neither grid can land on it by luck.
CUT, PLANTED_FRAMES = 137, 300
BIG, SMALL = 20.0, 10.0                   # box side while moving, and once stopped
LENGTH = 50                               # predict.window_frames in those cases


def pump(app, predicate, timeout: float = 60.0) -> bool:
    """Spin the event loop until a worker thread's signals have landed."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# a project with a "trained" model in it
# --------------------------------------------------------------------------- #
class StubModel:
    """A fitted classifier without the fit.

    Deterministic and instant: these tests are about what happens to the
    predictions afterwards, not about whether a gradient booster can learn
    anything from four clips. The first feature column decides the class, so
    both classes appear and the confidences differ between them.
    """

    classes_ = np.array(CLASSES)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        swimming = np.nan_to_num(X[:, 0]) % 2.0 < 1.0
        p = np.where(swimming, 0.9, 0.25)
        return np.column_stack([p, 1.0 - p])


class AreaModel:
    """A model that is EXACTLY right about the fixture built by `plant_transition`.

    That fixture's animal fills a 20x20 box while it is moving and a 10x10 one
    once it has stopped, and ``area_mean`` is the plain mean of the per-frame box
    area over the window - no smoothing, no derivative - so a window's
    ``area_mean`` is an affine function of the FRACTION of its frames spent
    moving. Inverting that affine map is what this does, which makes it the
    perfect classifier for that movie: any boundary error the test then measures
    belongs to the window GRID and not to the model's noise. Same standard as
    ``tests/test_bouts.py``, one layer further out.
    """

    classes_ = np.array(CLASSES)

    def predict_proba(self, X):
        area = np.asarray(X, dtype=float)[:, 1]        # area_mean, by position
        moving = np.clip(
            (area - SMALL ** 2) / (BIG ** 2 - SMALL ** 2), 0.001, 0.999
        )
        return np.column_stack([moving, 1.0 - moving])


class Recorder:
    """A reporter that keeps every line, for the claims about what a run SAYS."""

    def __init__(self) -> None:
        self.logged: list[tuple[str, str]] = []

    def log(self, message, level="info"):
        self.logged.append((str(message), str(level)))

    def stage(self, label, total=None):
        pass

    def advance(self, n=1):
        pass

    def raise_if_cancelled(self):
        pass

    def said(self, *needles: str) -> bool:
        return any(all(n in message for n in needles) for message, _ in self.logged)


def _report_entry(group: str, reliable: bool) -> dict:
    """The fields of GroupReport that anything downstream actually reads."""
    return {
        "group": group,
        "n_labels": 40,
        "n_classes": 2,
        "classes": list(CLASSES),
        "macro_f1": 0.81 if reliable else 0.12,
        "accuracy": 0.84 if reliable else 0.5,
        "baseline_accuracy": 0.5,
        "kappa": 0.62 if reliable else 0.01,
        "kappa_lower": 0.31 if reliable else -0.20,
        "reliable": bool(reliable),
        "reason": "ok" if reliable else
                  "agreement is not distinguishable from chance (kappa 0.01)",
        "confusion": [[10, 2], [3, 9]],
        "classes_dropped": {},
    }


def build_case(
    root: Path,
    name: str,
    *,
    model: bool = True,
    report: bool = True,
    reliable: bool = True,
    groups: dict | None = None,
    labels: int = 0,
    coverage: str = "manifest",
    model_class: type = StubModel,
    trained_window: int | None = None,
):
    """A project with a clip manifest, clip features and a stubbed-in model.

    ``groups`` overrides the {group: reliable} map when a case needs a second
    group; ``labels`` is how many clips to hand label through the real
    AnnotationStore, so the CSV under test is one a session could have written.

    ``coverage`` is pinned into the project rather than left at the shipped
    default, because most of the cases below are assertions about the manifest
    path specifically - how many rows there are, where f_end came from, which
    clip got which class. The dense path gets its own case.

    ``trained_window`` defaults to None, which writes the payload a model saved
    before the length was recorded has: no key at all. That is what most cases
    below want, since none of them are about the skew guard and an unrecorded
    length is only ever a warning.
    """
    import joblib

    from trackanno.clipfeatures import CLIP_FEATURES_FILE, FEATURE_VERSION
    from trackanno.config import load_config
    from trackanno.project import write_config
    from trackanno.store import AnnotationStore
    from trackanno.suggest import MODEL_FILE, REPORT_FILE

    fixture = make_project(FIXTURES)
    project = Path(root) / name
    project.mkdir(parents=True, exist_ok=True)
    config_path = project / "config.yaml"
    shutil.copy(fixture, config_path)
    write_config(config_path, {
        "paths.work_dir": str(project / "work").replace("\\", "/"),
        "paths.csv": str(project / "annotations.csv").replace("\\", "/"),
        "predict.coverage": coverage,
    })
    cfg = load_config(config_path)
    work = cfg.paths.work_dir
    work.mkdir(parents=True, exist_ok=True)

    # Only the tables are copied, not the pixels: predicting reads features,
    # never the clip cache, and a test that copied 50 MB per case to prove that
    # would be proving something else. tracks.parquet and cache.json come along
    # because the dense grid is built from those two alone.
    manifests = []
    for movie_dir in sorted(p for p in load_config(fixture).paths.work_dir.iterdir()
                            if p.is_dir()):
        source = movie_dir / "clips.parquet"
        if not source.exists():
            continue
        (work / movie_dir.name).mkdir(parents=True, exist_ok=True)
        for name in ("clips.parquet", "tracks.parquet", "cache.json"):
            if (movie_dir / name).exists():
                shutil.copy(movie_dir / name, work / movie_dir.name / name)
        manifests.append(pd.read_parquet(source))
    manifest = pd.concat(manifests, ignore_index=True)

    features = pd.DataFrame({
        "clip_id": manifest["clip_id"].astype(str),
        "movie_id": manifest["movie_id"].astype(str),
        "track_id": manifest["track_id"].astype(int),
        "f_start": manifest["f_start"].astype(int),
        "speed_mean": np.arange(len(manifest), dtype=float),
        "area_mean": np.linspace(10.0, 20.0, len(manifest)),
        # Both artefacts carry the stamp, as build_clip_features and train do:
        # Suggester.load refuses an unstamped pair rather than assuming it fits.
        "_feature_version": FEATURE_VERSION,
    })
    features.to_parquet(work / CLIP_FEATURES_FILE, index=False)

    wanted = groups if groups is not None else {GROUP: reliable}
    if model:
        payload = {"columns": ["speed_mean", "area_mean"],
                   "models": {g: model_class() for g in wanted},
                   "feature_version": FEATURE_VERSION}
        if trained_window is not None:
            payload["trained_window"] = int(trained_window)
        joblib.dump(payload, work / MODEL_FILE)
    if report:
        import json

        (work / REPORT_FILE).write_text(
            json.dumps({g: _report_entry(g, ok) for g, ok in wanted.items()}, indent=2),
            encoding="utf-8",
        )

    if labels:
        store = AnnotationStore(cfg.paths.csv, cfg)
        for i, row in enumerate(manifest.head(labels).to_dict("records")):
            store.record(row, {GROUP: CLASSES[i % len(CLASSES)]})
    return cfg, manifest


def planted_case(root: Path, name: str, stride: int, **settings):
    """A project whose one track changes behaviour at frame `CUT`, and nothing else.

    The fixture movie is 120 frames of two tracks that never do anything
    different, which is enough to check that bouts are well formed and useless
    for checking WHERE a boundary lands. This replaces its ``tracks.parquet``
    with a single 300-frame track whose box shrinks from 20x20 to 10x10 at
    ``CUT``, so ``area_mean`` reports the fraction of each window spent moving
    and :class:`AreaModel` reads that fraction back exactly.

    The rest of the project is left alone. The dense pass reads
    ``tracks.parquet`` and ``cache.json`` and nothing else - no pixels, no clip
    cache - so a movie invented here is as real to it as a prepared one, and
    ``cache.json``'s stale 120-frame count is harmless because
    ``build_dense_features`` takes the larger of that and the last detected
    frame.
    """
    import joblib

    from trackanno.config import load_config
    from trackanno.project import write_config
    from trackanno.suggest import MODEL_FILE

    cfg, _ = build_case(root, name, coverage="dense", model_class=AreaModel,
                        trained_window=LENGTH)
    work = cfg.paths.work_dir
    movie = next(p for p in sorted(work.iterdir())
                 if p.is_dir() and (p / "tracks.parquet").exists())
    frames = np.arange(PLANTED_FRAMES)
    side = np.where(frames < CUT, BIG, SMALL)
    left = 20.0 + 0.5 * frames                      # drifting, so speed_* exist too
    pd.DataFrame({
        "frame": frames, "track_id": 1,
        "x1": left, "y1": 40.0, "x2": left + side, "y2": 40.0 + side,
        "conf": 0.9,
    }).to_parquet(movie / "tracks.parquet", index=False)

    payload = joblib.load(work / MODEL_FILE)
    payload["models"] = {g: AreaModel() for g in payload["models"]}
    joblib.dump(payload, work / MODEL_FILE)

    write_config(cfg.path, {
        "predict.window_frames": LENGTH,
        "predict.window_stride": stride,
        "predict.decode": "min_bout",
        "predict.min_bout_frames": 20,
        **settings,
    })
    return load_config(cfg.path), movie.name


def bout_edge(frame: pd.DataFrame, label: str) -> int:
    """The first frame the decode calls ``label``, or -1 if it never does."""
    rows = frame[frame[f"{GROUP}_pred"] == label]
    return int(rows["f_start"].min()) if len(rows) else -1


# --------------------------------------------------------------------------- #
# refusing to run
# --------------------------------------------------------------------------- #
def test_an_unusable_model_is_refused_with_a_reason(root: Path) -> None:
    """A bulk run is the suggester's mistake at 1000x scale: whatever the report
    gated off must not be labelled anyway, and the refusal has to say which of
    the several possible reasons applies or nobody can act on it."""
    from trackanno.predict import PREDICTIONS_FILE, predict_all, prediction_blocker

    cfg, _ = build_case(root, "no-model", model=False, report=False)
    try:
        predict_all(cfg)
        check("an absent model is refused", False, "it ran anyway")
    except RuntimeError as exc:
        check("an absent model is refused", "no model has been trained" in str(exc), str(exc))
    check("...and the blocker says the same",
          "no model has been trained" in prediction_blocker(cfg), prediction_blocker(cfg))

    cfg, _ = build_case(root, "no-report", report=False)
    try:
        predict_all(cfg)
        check("a model with no report is refused", False, "it ran anyway")
    except RuntimeError as exc:
        check("a model with no report is refused", "suggest_report.json" in str(exc), str(exc))

    cfg, _ = build_case(root, "silent", reliable=False)
    try:
        predict_all(cfg)
        check("a model the report gated off is refused", False, "it ran anyway")
    except RuntimeError as exc:
        check("a model the report gated off is refused",
              "not distinguishable from chance" in str(exc), str(exc))
        check("...and it names the group", GROUP in str(exc), str(exc))

    check("no predictions file was left behind by any refusal",
          not (cfg.paths.work_dir / PREDICTIONS_FILE).exists())


def test_an_unreliable_group_is_skipped_not_silently_predicted(root: Path) -> None:
    """The gate is per group. A project where one group cleared the bar and one
    did not must get predictions for the first and nothing at all for the second
    - and be told which, because a missing column is otherwise indistinguishable
    from a group nobody configured."""
    from trackanno.predict import predict_all, verdict

    cfg, _ = build_case(root, "mixed", groups={GROUP: True, "shape": False})
    reporter = Recorder()
    logged = reporter.logged

    frame = predict_all(cfg, reporter=reporter)
    check("the reliable group is predicted", f"{GROUP}_pred" in frame.columns)
    check("the gated-off group is not", "shape_pred" not in frame.columns,
          str(list(frame.columns)))
    check("the skip is reported, with the reason",
          any("shape" in m and "chance" in m and level == "warning"
              for m, level in logged),
          str(logged))
    decision = verdict(cfg)
    check("the verdict separates the two", decision.usable == [GROUP]
          and set(decision.skipped) == {"shape"}, str(decision))
    check("...and does not block the run", decision.problem == "", decision.problem)


# --------------------------------------------------------------------------- #
# what it writes
# --------------------------------------------------------------------------- #
def test_one_row_per_clip_with_the_expected_columns(root: Path) -> None:
    """The file is meant to be joined on clip_id, so a clip the model had no
    opinion about still needs its row; the alternative is every consumer having
    to work out for itself which clips are missing and why."""
    from trackanno.clipfeatures import CLIP_FEATURES_FILE
    from trackanno.predict import PREDICTIONS_FILE, predict_all

    cfg, manifest = build_case(root, "columns")
    frame = predict_all(cfg)
    features = pd.read_parquet(cfg.paths.work_dir / CLIP_FEATURES_FILE)

    check("one row per clip in clip_features.parquet", len(frame) == len(features),
          f"{len(frame)} vs {len(features)}")
    check("clip ids match, in order",
          list(frame["clip_id"]) == list(features["clip_id"].astype(str)))
    for column in ("clip_id", "movie_id", "track_id", "f_start", "f_end",
                   f"{GROUP}_pred", f"{GROUP}_conf", "human_labelled"):
        check(f"it carries {column}", column in frame.columns, str(list(frame.columns)))

    ends = dict(zip(manifest["clip_id"].astype(str), manifest["f_end"].astype(int)))
    check("f_end comes from the clip manifest, not a guess",
          all(int(r["f_end"]) == ends[r["clip_id"]] for r in frame.to_dict("records")))
    check("every clip got a class", frame[f"{GROUP}_pred"].isin(CLASSES).all(),
          str(frame[f"{GROUP}_pred"].unique()))
    check("confidences are probabilities",
          bool(((frame[f"{GROUP}_conf"] > 0.0) & (frame[f"{GROUP}_conf"] <= 1.0)).all()))
    check("both classes appear, so the stub is not degenerate",
          frame[f"{GROUP}_pred"].nunique() == 2)

    on_disk = pd.read_parquet(cfg.paths.work_dir / PREDICTIONS_FILE)
    check("the file on disk is what was returned", on_disk.equals(frame))
    check("it lives in the work dir, beside the model",
          (cfg.paths.work_dir / PREDICTIONS_FILE).exists())


def test_predictions_never_reach_the_annotation_csv(root: Path) -> None:
    """The CSV is the ground truth the next model trains on. A predicted value
    written into it is indistinguishable from a hand label on the next read, so
    the model would train on its own output and the truth would be gone with no
    way to recover it."""
    from trackanno.predict import PREDICTIONS_FILE, predict_all

    cfg, _ = build_case(root, "provenance", labels=3)
    before = digest(cfg.paths.csv)
    rows_before = pd.read_csv(cfg.paths.csv)

    predict_all(cfg)
    check("the annotation CSV is byte-for-byte unchanged",
          digest(cfg.paths.csv) == before)

    rows_after = pd.read_csv(cfg.paths.csv)
    check("no rows were added", len(rows_after) == len(rows_before) == 3,
          f"{len(rows_before)} -> {len(rows_after)}")
    check("no prediction column was added to it",
          not [c for c in rows_after.columns if c.endswith("_pred")],
          str(list(rows_after.columns)))
    check("the predictions live in their own file",
          (cfg.paths.work_dir / PREDICTIONS_FILE).exists())

    # ...and running it again, with the labelled clips included, still does not.
    predict_all(cfg, include_labelled=True)
    check("a second run does not touch it either", digest(cfg.paths.csv) == before)


def test_human_labelled_marks_the_clips_a_person_judged(root: Path) -> None:
    """The one column that lets a consumer merge this file with the CSV without
    losing track of which values a person stands behind."""
    from trackanno.predict import predict_all

    cfg, manifest = build_case(root, "flags", labels=3)
    labelled = set(manifest["clip_id"].astype(str).head(3))

    frame = predict_all(cfg).set_index("clip_id")
    check("human_labelled is true exactly for the hand labels",
          set(frame.index[frame["human_labelled"]]) == labelled,
          str(sorted(frame.index[frame["human_labelled"]])))
    check("hand-labelled clips get no prediction by default",
          (frame.loc[sorted(labelled), f"{GROUP}_pred"] == "").all(),
          str(frame.loc[sorted(labelled), f"{GROUP}_pred"].tolist()))
    check("...and no confidence either",
          frame.loc[sorted(labelled), f"{GROUP}_conf"].isna().all())
    rest = [c for c in frame.index if c not in labelled]
    check("every other clip does get one",
          (frame.loc[rest, f"{GROUP}_pred"] != "").all())

    frame = predict_all(cfg, include_labelled=True).set_index("clip_id")
    check("include_labelled predicts them too",
          (frame.loc[sorted(labelled), f"{GROUP}_pred"] != "").all())
    check("...and still marks them as a person's",
          bool(frame.loc[sorted(labelled), "human_labelled"].all()))


def test_dense_coverage_reaches_every_frame(root: Path) -> None:
    """The clip grid is a sampling chosen for a human's attention, so scoring
    only those clips leaves most of every movie unlabelled here and unpainted in
    the render. The dense pass covers every frame of every track instead - and
    in the SHIPPED DEFAULT it has to do it exactly once per frame, because the
    renderer paints range(f_start, f_end) per track and a frame in two windows
    means one prediction silently overwrites the other.

    Every claim below belongs to that default - `window_stride: auto`, so the
    windows tile, and `decode: none`, so a row is still a window. None of them
    survives an overlapping stride: one frame is then in several windows, the
    short tail window is dropped, and a row stops being a window at all. The
    right response to that is NOT to loosen these into something both modes can
    satisfy. Holding the default to the invariant it has always had is the
    optionality guarantee; the other mode makes its own claims further down.
    """
    from trackanno.clipfeatures import CLIP_FEATURES_FILE
    from trackanno.densepredict import DENSE_FEATURES_FILE
    from trackanno.predict import predict_all

    cfg, manifest = build_case(root, "dense", labels=1, coverage="dense")
    work = cfg.paths.work_dir

    # The annotation grid this project actually has: clips.window_mode 'first',
    # one window per movie, every later window of every track uncut.
    features = pd.read_parquet(work / CLIP_FEATURES_FILE)
    first = features[features["f_start"] == 0]
    first.to_parquet(work / CLIP_FEATURES_FILE, index=False)

    sparse = predict_all(cfg, coverage="manifest")
    # No coverage argument: this is the call the Model page's button makes, so
    # the config default is what has to reach it.
    dense = predict_all(cfg)

    check("the manifest path scores only the clips that were cut",
          len(sparse) == len(first), f"{len(sparse)} vs {len(first)}")
    check("the dense path scores more than that", len(dense) > len(sparse),
          f"{len(dense)} vs {len(sparse)}")
    check("...and writes its own feature table, not clip_features.parquet",
          (work / DENSE_FEATURES_FILE).exists()
          and len(pd.read_parquet(work / CLIP_FEATURES_FILE)) == len(first))

    tracks = pd.read_parquet(work / str(dense["movie_id"].iloc[0]) / "tracks.parquet")
    uncovered = doubled = detected = 0
    for track_id, tdf in tracks.groupby("track_id"):
        windows = dense[dense["track_id"] == int(track_id)][["f_start", "f_end"]].to_numpy()
        for frame in tdf["frame"].to_numpy():
            hits = int(((windows[:, 0] <= frame) & (frame < windows[:, 1])).sum())
            detected += 1
            uncovered += int(hits == 0)
            doubled += int(hits > 1)
    check("every detected frame falls in a window", uncovered == 0, f"{uncovered} missed")
    check("...and in exactly one", doubled == 0, f"{doubled} covered twice")
    check("the movie's tail window is not dropped",
          int(dense["f_end"].max()) == int(tracks["frame"].max()) + 1,
          f"{int(dense['f_end'].max())} vs {int(tracks['frame'].max()) + 1}")

    # Same id scheme, so the hand labels still join and the annotated windows
    # are recognisably the same windows.
    check("a dense window that coincides with a cut clip keeps its clip_id",
          set(sparse["clip_id"]).issubset(set(dense["clip_id"])),
          str(sorted(set(sparse["clip_id"]) - set(dense["clip_id"]))[:3]))
    labelled = set(pd.read_csv(cfg.paths.csv)["clip_id"].astype(str))
    check("human_labelled still lands on the annotator's rows",
          set(dense.loc[dense["human_labelled"], "clip_id"]) == labelled,
          str(sorted(dense.loc[dense['human_labelled'], 'clip_id'])))

    check("dense rows say how many detections they had",
          "n_present" in dense.columns and (dense["n_present"] > 0).all())
    check("...and why one could not be predicted, rather than going missing",
          "skip_reason" in dense.columns)
    blank = dense[dense[f"{GROUP}_pred"] == ""]
    check("every blank row is explained by a hand label or a skip_reason",
          bool((blank["human_labelled"] | (blank["skip_reason"] != "")).all()),
          str(blank[["clip_id", "skip_reason", "human_labelled"]].to_dict("records")))

    # And nothing the annotation flow depends on moved.
    check("the clip manifests are untouched",
          all(pd.read_parquet(work / m / "clips.parquet").equals(
              manifest[manifest["movie_id"] == m].reset_index(drop=True))
              for m in manifest["movie_id"].unique()))


def test_a_window_too_thin_to_measure_is_a_row_not_a_hole(root: Path) -> None:
    """A dropped row and a row the model declined to label look identical once
    the file is read back, so a window that cannot be featurised has to survive
    into the output carrying the reason."""
    from trackanno.densepredict import build_dense_features
    from trackanno.predict import predict_all
    from trackanno.project import write_config
    from trackanno.config import load_config

    cfg, _ = build_case(root, "thin", coverage="dense")
    # min_frames above anything the fixture's tracks can supply, so every window
    # is skipped and the answer must still be a full table.
    write_config(cfg.path, {"predict.min_frames": 999})
    cfg = load_config(cfg.path)

    dense = build_dense_features(cfg)
    check("no window survives the bar", (dense["skip_reason"] != "").all())
    check("...and every one of them is still a row", len(dense) > 0)
    check("the reason names the setting that stopped it",
          dense["skip_reason"].str.contains("predict.min_frames").all(),
          str(dense["skip_reason"].iloc[0]))

    out = predict_all(cfg)
    check("nothing is predicted", (out[f"{GROUP}_pred"] == "").all())
    check("...and nothing is claimed to be", out[f"{GROUP}_conf"].isna().all())
    check("the rows are all there", len(out) == len(dense))
    check("the model's columns survived the all-skipped table",
          all(c in pd.read_parquet(cfg.paths.work_dir / "dense_clip_features.parquet").columns
              for c in ("speed_mean", "area_mean")))


# --------------------------------------------------------------------------- #
# what one row is
# --------------------------------------------------------------------------- #
def test_a_predictions_table_says_which_of_three_things_a_row_is(root: Path) -> None:
    """`coverage` and `decode` are two settings, so there are three schemas.

    A manifest run writes one row per annotation CLIP, a dense run one row per
    WINDOW off the predictor's own grid, and a decode re-cuts those windows into
    BOUTS. Only the file can be asked which it holds: `predict.decode` can be
    edited after a run, and a consumer reading it would describe a run that has
    not happened.
    """
    from trackanno.config import load_config
    from trackanno.predict import RowKind, predict_all, row_kind, summarise
    from trackanno.project import write_config
    from trackanno.render import prediction_unit

    manifest, _ = build_case(root, "kind-clips", coverage="manifest")
    clips = predict_all(manifest)

    grid, _ = build_case(root, "kind-windows", coverage="dense")
    windows = predict_all(grid)

    decoded, _ = build_case(root, "kind-bouts", coverage="dense")
    write_config(decoded.path, {
        "predict.window_stride": 10,
        "predict.decode": "min_bout",
        "predict.min_bout_frames": 20,
    })
    bouts = predict_all(load_config(decoded.path))

    check("a manifest run writes annotation clips",
          row_kind(clips) is RowKind.CLIP, str(list(clips.columns)))
    check("a dense run writes windows off the predictor's own grid",
          row_kind(windows) is RowKind.DENSE_WINDOW, str(list(windows.columns)))
    check("a decode writes bouts",
          row_kind(bouts) is RowKind.BOUT, str(list(bouts.columns)))

    for frame, noun in ((clips, "clips"), (windows, "dense windows"),
                        (bouts, "bouts")):
        summary = summarise(frame)
        check(f"...and the summary counts them in {noun}",
              summary.startswith(f"{len(frame)} {noun}"), summary)

    # The third value is what the noun needs and the export's two words cannot
    # give it: a clip and a dense window are both a fixed span the predictor was
    # handed, so prediction_unit calls them the same thing.
    check("the export separates only the bouts",
          [prediction_unit(f) for f in (clips, windows, bouts)]
          == ["predicted windows", "predicted windows", "predicted bouts"])
    check("...and only the bouts are disjoint by construction",
          [row_kind(f).bouts for f in (clips, windows, bouts)]
          == [False, False, True])


def test_only_one_function_asks_what_a_prediction_row_is() -> None:
    """Four consumers used to probe `bout_id` for themselves.

    Rename the column and `predict.row_kind` follows while the rest silently
    answer "no": the timeline draws window lanes over bout data, the renderer
    drops its disjointness guard, and the summary calls bouts clips. Nothing
    raises, so the count of probes is the guard.
    """
    import inspect

    from trackanno.predict import row_kind
    from trackanno.render import prediction_unit

    src = Path(__file__).resolve().parents[1] / "src" / "trackanno"
    hits = [
        f"{path.relative_to(src).as_posix()}:{n}"
        for path in sorted(src.rglob("*.py"))
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if '"bout_id" in' in line or "'bout_id' in" in line
    ]
    check("the sentinel column is tested in exactly one place", len(hits) == 1,
          str(hits))
    check("...which is predict.row_kind, in the module that writes the schema",
          hits[0].startswith("predict.py:")
          and '"bout_id" in' in inspect.getsource(row_kind), str(hits))
    check("...and render.prediction_unit formats that answer rather than "
          "re-deriving it", "row_kind" in inspect.getsource(prediction_unit))


# --------------------------------------------------------------------------- #
# decoded bouts
# --------------------------------------------------------------------------- #
def test_writing_the_default_stride_out_by_hand_changes_the_file_not_at_all(
    root: Path,
) -> None:
    """`window_stride: auto` is documented as "the window length", and the whole
    argument for adding the setting was that a project which never touches it
    keeps the file it already had. So the two spellings must agree BYTE FOR BYTE,
    not merely on the numbers: a window boundary that moved would move every
    length-dependent feature value with it, silently, and a rewritten parquet is
    the only place that would show."""
    from trackanno.config import load_config
    from trackanno.densepredict import window_length
    from trackanno.predict import PREDICTIONS_FILE, predict_all
    from trackanno.project import write_config

    auto, _ = build_case(root, "stride-auto", coverage="dense")
    spelled, _ = build_case(root, "stride-spelled", coverage="dense")
    write_config(spelled.path, {"predict.window_stride": window_length(spelled)})
    spelled = load_config(spelled.path)

    check("the two configs really do resolve to the same grid",
          (spelled.predict.window_stride, spelled.predict.resolved_stride(spelled.clips))
          == (window_length(spelled), auto.predict.resolved_stride(auto.clips)),
          f"{spelled.predict.window_stride!r} vs auto")

    first = predict_all(auto)
    other = predict_all(spelled)
    check("spelling the stride out gives the same rows",
          first.equals(other), f"{first.shape} vs {other.shape}")
    check("...in the same columns, in the same order",
          list(first.columns) == list(other.columns), str(list(other.columns)))
    check("...and the same bytes on disk",
          digest(auto.paths.work_dir / PREDICTIONS_FILE)
          == digest(spelled.paths.work_dir / PREDICTIONS_FILE))

    # Re-running is byte-identical too, so the check above is a claim about the
    # inputs rather than about parquet happening to be deterministic today.
    before = digest(auto.paths.work_dir / PREDICTIONS_FILE)
    predict_all(auto)
    check("a second identical run rewrites the same bytes",
          digest(auto.paths.work_dir / PREDICTIONS_FILE) == before)

    check("a tiling run still writes windows, not bouts",
          "bout_id" not in first.columns and "clip_id" in first.columns,
          str(list(first.columns)))
    check("...and carries nothing a decode would have added",
          not {"group", "human_overlap_frac", "n_covering_mean"} & set(first.columns),
          str(list(first.columns)))


def test_a_decode_writes_bouts_where_it_used_to_write_windows(root: Path) -> None:
    """A bout is not a window: it starts where the behaviour started, runs as
    long as the behaviour did, and is the row the renderer paints.

    So the contract it has to keep is the renderer's - sorted, disjoint,
    gapless over the frames the windows covered, per track - and the contract it
    must NOT keep is the clip grid's. `clip_id` is derived from f_start alone, so
    a bout beginning on an annotated clip's f_start would inherit that clip's
    whole identity while spanning a different number of frames; the column is
    absent rather than wrong, so a downstream merge on it raises instead of
    quietly attributing a 300-frame bout to a 51-frame hand label.
    """
    from trackanno.config import load_config
    from trackanno.densepredict import DENSE_FEATURES_FILE
    from trackanno.predict import predict_all
    from trackanno.project import write_config

    cfg, _ = build_case(root, "bouts", coverage="dense")
    windows = predict_all(cfg)
    write_config(cfg.path, {
        "predict.window_stride": 10,
        "predict.decode": "min_bout",
        "predict.min_bout_frames": 20,
    })
    cfg = load_config(cfg.path)
    bouts = predict_all(cfg)

    dense = pd.read_parquet(cfg.paths.work_dir / DENSE_FEATURES_FILE)
    check("the overlapping grid really did score more windows than the tiling one",
          len(dense) > len(windows), f"{len(dense)} windows vs {len(windows)} rows")
    check("but a row of the answer is no longer one of them",
          len(bouts) < len(dense), f"{len(bouts)} rows for {len(dense)} windows")

    check("every bout carries a bout_id", "bout_id" in bouts.columns
          and bouts["bout_id"].astype(str).ne("").all())
    check("...naming both ends, so it cannot collide with a clip_id",
          all(row["bout_id"].endswith(f"f{row['f_start']:06d}-{row['f_end']:06d}")
              for row in bouts.to_dict("records")),
          str(bouts["bout_id"].head(2).tolist()))
    check("there is no clip_id column at all, so a join on one raises",
          "clip_id" not in bouts.columns, str(list(bouts.columns)))
    check("the row says which group's question it answers",
          set(bouts["group"]) == {GROUP}, str(set(bouts["group"])))
    check("n_frames is the span, not a nominal window length",
          bool((bouts["n_frames"] == bouts["f_end"] - bouts["f_start"]).all()))

    for track_id, group in bouts.groupby("track_id"):
        spans = group.sort_values("f_start")[["f_start", "f_end"]].to_numpy()
        covered = windows[windows["track_id"] == track_id]
        check(f"track {track_id}: the bouts are sorted and disjoint",
              bool((spans[:, 0] < spans[:, 1]).all())
              and list(spans[1:, 0]) == list(spans[:-1, 1]), str(spans))
        check(f"track {track_id}: ...and gapless over the frames the windows covered",
              (int(spans[0, 0]), int(spans[-1, 1]))
              == (int(covered["f_start"].min()), int(covered["f_end"].max())),
              f"{spans[0, 0]}..{spans[-1, 1]} vs "
              f"{covered['f_start'].min()}..{covered['f_end'].max()}")

    # Both decoders, because 'viterbi' is a different function reached by the
    # same branch and a bout table that only min_bout can produce is half tested.
    write_config(cfg.path, {"predict.decode": "viterbi", "predict.self_transition": 0.98})
    viterbi = predict_all(load_config(cfg.path))
    check("viterbi produces bouts too, over the same frames",
          "bout_id" in viterbi.columns
          and viterbi.groupby("track_id")["f_end"].max().equals(
              bouts.groupby("track_id")["f_end"].max()),
          str(viterbi[["track_id", "f_start", "f_end"]].to_dict("records")))

    # A decode at the TILING stride is a legal config and has to do what it says
    # rather than being quietly ignored because the windows happen not to overlap.
    write_config(cfg.path, {
        "predict.window_stride": "auto", "predict.decode": "min_bout",
        "predict.min_bout_frames": 20,
    })
    tiled = predict_all(load_config(cfg.path))
    check("a decode at the tiling stride still cuts bouts rather than being ignored",
          "bout_id" in tiled.columns and "clip_id" not in tiled.columns,
          str(list(tiled.columns)))
    check("...and every frame the windows covered is in exactly one of them",
          int(tiled["n_frames"].sum()) == int((windows["f_end"] - windows["f_start"]).sum()),
          f"{int(tiled['n_frames'].sum())} vs "
          f"{int((windows['f_end'] - windows['f_start']).sum())}")


def test_include_labelled_is_inert_under_a_decode_and_says_so(root: Path) -> None:
    """Withholding a hand-labelled window from a decode does not withhold it.

    Under an overlapping stride the neighbouring windows paint the same frames
    from almost the same features, so the label leaks whatever the flag says; and
    the hole it punches in the posterior re-cuts the bouts on BOTH sides of it,
    so the flag changes frames nobody annotated. Suppressing every overlapping
    window instead would punch a bigger hole. So the flag is inert here, and the
    provenance moves onto the row as `human_overlap_frac` - which is the
    fraction of the bout's frames a person actually decided, and is checkable
    against the one clip this project has a label for.
    """
    from trackanno.config import load_config
    from trackanno.predict import predict_all
    from trackanno.project import write_config

    cfg, _ = build_case(root, "inert", labels=1, coverage="dense")
    write_config(cfg.path, {
        "predict.window_stride": 10,
        "predict.decode": "min_bout",
        "predict.min_bout_frames": 20,
    })
    cfg = load_config(cfg.path)

    quiet = Recorder()
    withheld = predict_all(cfg, reporter=quiet)
    loud = Recorder()
    included = predict_all(cfg, reporter=loud, include_labelled=True)

    check("the flag changes not one row", withheld.equals(included),
          f"{withheld.shape} vs {included.shape}")
    check("passing it says so, rather than looking like it worked",
          loud.said("--include-labelled has no effect", "human_overlap_frac"),
          str(loud.logged))
    check("...and it says why, in terms of the posterior it would damage",
          loud.said("no posterior") and loud.said("nearly the same features"),
          str([m for m, _ in loud.logged if "include-labelled" in m]))
    check("not passing it says nothing at all",
          not quiet.said("--include-labelled"), str(quiet.logged))

    labelled = pd.read_csv(cfg.paths.csv).iloc[0]
    movie, track = str(labelled["movie_id"]), int(labelled["track_id"])
    lo, hi = int(labelled["f_start"]), int(labelled["f_end"])
    check("the fixture really does have exactly one hand-labelled clip",
          len(pd.read_csv(cfg.paths.csv)) == 1 and hi > lo, f"{lo}..{hi}")

    mine = withheld[(withheld["movie_id"] == movie) & (withheld["track_id"] == track)]
    for row in mine.to_dict("records"):
        overlap = max(0, min(row["f_end"], hi) - max(row["f_start"], lo))
        check(f"human_overlap_frac is {overlap}/{row['n_frames']} on bout "
              f"{row['f_start']}-{row['f_end']}",
              abs(row["human_overlap_frac"] - overlap / row["n_frames"]) < 1e-12,
              f"{row['human_overlap_frac']}")
    share = mine["human_overlap_frac"]
    check("at least one bout really does straddle the labelled clip, so the "
          "fraction above is not a column of zeros and ones",
          bool(((share > 0.0) & (share < 1.0)).any()), str(share.tolist()))
    check("and no bout claims more of itself than it has",
          bool(((share >= 0.0) & (share <= 1.0)).all()), str(share.tolist()))

    others = withheld[withheld["track_id"] != track]
    check("a track nobody labelled overlaps nothing",
          bool((others["human_overlap_frac"] == 0.0).all())
          and not bool(others["human_labelled"].any()),
          str(others["human_overlap_frac"].unique()))
    check("human_labelled is exactly 'the fraction is above zero'",
          bool((withheld["human_labelled"]
                == (withheld["human_overlap_frac"] > 0.0)).all()))


def test_a_bout_boundary_lands_nearer_the_truth_than_a_tile_boundary(root: Path) -> None:
    """The reason the whole feature exists, measured end to end.

    One track that stops moving at frame 137, 50-frame windows, and a model that
    is exactly right about every window it is shown. The tiling grid cannot
    answer 137: every window is either wholly before or wholly after, and the one
    straddling the change is 74% "moving" and votes that way for all fifty of its
    frames, so the best it can say is 150. At stride 10 five windows speak for
    each frame and their mean crosses over within one stride of the truth. The
    model is the same in both runs, so the whole difference is the grid.
    """
    from trackanno.predict import predict_all

    tiled_cfg, _ = planted_case(root, "planted-tiled", LENGTH)
    dense_cfg, _ = planted_case(root, "planted-dense", 10)
    tiled = predict_all(tiled_cfg)
    dense = predict_all(dense_cfg)

    check("the planted movie decodes into exactly two bouts either way",
          len(tiled) == 2 and len(dense) == 2,
          f"tiled {tiled[['f_start', 'f_end', f'{GROUP}_pred']].to_dict('records')}, "
          f"dense {dense[['f_start', 'f_end', f'{GROUP}_pred']].to_dict('records')}")
    check("...in the planted order, moving then still",
          list(tiled[f"{GROUP}_pred"]) == list(dense[f"{GROUP}_pred"]) == list(CLASSES),
          str(list(dense[f"{GROUP}_pred"])))

    tiled_edge = bout_edge(tiled, CLASSES[1])
    dense_edge = bout_edge(dense, CLASSES[1])
    check("the tiling grid can only answer a multiple of the window length",
          tiled_edge % LENGTH == 0, str(tiled_edge))
    check("...so it puts the change at 150 when the truth is 137",
          tiled_edge == 150, str(tiled_edge))
    check("the overlapping grid places it within one stride of the truth",
          abs(dense_edge - CUT) <= 10, f"{dense_edge} vs {CUT}")
    check("which is closer than the tiling grid manages",
          abs(dense_edge - CUT) < abs(tiled_edge - CUT),
          f"dense {dense_edge}, tiled {tiled_edge}, truth {CUT}")
    check("both cover the whole track, so the gain is not paid for in coverage",
          int(tiled["n_frames"].sum()) == int(dense["n_frames"].sum()) == PLANTED_FRAMES,
          f"{int(tiled['n_frames'].sum())} / {int(dense['n_frames'].sum())}")
    check("a bout in the interior rests on more than one window, and says how many",
          float(dense["n_covering_mean"].max()) > 1.0
          and float(tiled["n_covering_mean"].max()) == 1.0,
          f"{dense['n_covering_mean'].tolist()} vs {tiled['n_covering_mean'].tolist()}")


def test_the_saved_posterior_round_trips(root: Path) -> None:
    """`save_posteriors` is for arguing with a decode, which needs the numbers the
    decode saw - so the file has to be the posterior the bouts were cut from, not
    a recomputation of it.

    One row per (movie, track, frame), wide over the classes: the posterior is a
    function of the frame, so that is its natural key, and a long form would
    repeat that key once per class and multiply an already-large file. Class
    columns are namespaced by group (`{group}__p__{class}`), so two groups
    sharing a class name cannot collide, and parquet being columnar means reading
    one group's answer costs only that group's columns.
    """
    from trackanno.predict import POSTERIORS_FILE, predict_all

    plain_cfg, _ = planted_case(root, "post-off", 10)
    predict_all(plain_cfg)
    check("nothing is written unless it is asked for",
          not (plain_cfg.paths.work_dir / POSTERIORS_FILE).exists())

    cfg, movie_id = planted_case(root, "post-on", 10,
                                 **{"predict.save_posteriors": True})
    bouts = predict_all(cfg)
    post = pd.read_parquet(cfg.paths.work_dir / POSTERIORS_FILE)

    check("the file is there under exactly that name",
          (cfg.paths.work_dir / POSTERIORS_FILE).exists())
    check("one row per (movie, track, frame) over the frames the windows covered",
          len(post) == PLANTED_FRAMES
          and list(post["frame"]) == list(range(PLANTED_FRAMES))
          and set(post["movie_id"]) == {movie_id} and set(post["track_id"]) == {1},
          f"{len(post)} rows, frames {post['frame'].min()}..{post['frame'].max()}")
    columns = [f"{GROUP}__p__{name}" for name in CLASSES]
    check("one column per class, namespaced by the group that owns it",
          all(c in post.columns for c in columns), str(list(post.columns)))
    check("a row of a posterior is still a posterior",
          np.allclose(post[columns].to_numpy().sum(axis=1), 1.0, atol=1e-12))

    # The closed form for this grid: a 50-frame window every 10 frames puts every
    # interior frame in exactly five of them, and the first frame in one.
    check("n_covering is length/stride in the interior",
          bool((post["n_covering"].to_numpy()[LENGTH: PLANTED_FRAMES - LENGTH]
                == LENGTH // 10).all()),
          str(sorted(set(post["n_covering"].to_numpy()[LENGTH:PLANTED_FRAMES - LENGTH]))))
    check("...and 1 at the very first frame, where only one window can reach",
          int(post["n_covering"].iloc[0]) == 1)

    # The round trip that matters: a bout's confidence is defined as the mean of
    # its own class's posterior over its own frames, so the two files have to
    # agree on that number or one of them is describing a different run.
    for row in bouts.to_dict("records"):
        column = f"{GROUP}__p__{row[f'{GROUP}_pred']}"
        mean = float(post[column].to_numpy()[row["f_start"]: row["f_end"]].mean())
        check(f"bout {row['f_start']}-{row['f_end']} carries the mean of the "
              "posterior it was cut from",
              abs(mean - row[f"{GROUP}_conf"]) < 1e-12,
              f"{mean} vs {row[f'{GROUP}_conf']}")
    edge = bout_edge(bouts, CLASSES[1])
    check("the posterior crosses over exactly where the bouts were cut",
          post[columns[0]].iloc[edge - 1] > 0.5 >= post[columns[0]].iloc[edge],
          f"{post[columns[0]].iloc[edge - 1]} -> {post[columns[0]].iloc[edge]}")


def test_the_summary_counts_a_frame_once_however_many_rows_cover_it(root: Path) -> None:
    """"covering N track-frames" used to be `(f_end - f_start).sum()`, which is a
    frame count only while the rows are disjoint.

    A bout table carries one row per (group, bout), so a project asking two
    questions of the same movie would claim twice the frames its movies contain -
    and this is a number nobody can check, because it is only ever compared with
    itself from one run to the next. It is now the union per (movie, track), so
    two groups' answers over the same frames collapse to those frames.
    """
    from trackanno.config import load_config
    from trackanno.predict import predict_all, summarise
    from trackanno.project import write_config

    cfg, _ = build_case(root, "counts", coverage="dense",
                        groups={GROUP: True, "shape": True})
    write_config(cfg.path, {
        "predict.window_stride": 10,
        "predict.decode": "min_bout",
        "predict.min_bout_frames": 20,
    })
    cfg = load_config(cfg.path)
    bouts = predict_all(cfg)

    tracks = pd.concat([
        pd.read_parquet(d / "tracks.parquet")
        for d in sorted(cfg.paths.work_dir.iterdir())
        if d.is_dir() and (d / "tracks.parquet").exists()
    ])
    detected = int(len(tracks))
    naive = int((bouts["f_end"] - bouts["f_start"]).sum())

    check("both groups really did get their own bouts over the same frames",
          set(bouts["group"]) == {GROUP, "shape"} and naive == 2 * detected,
          f"{sorted(set(bouts['group']))}, {naive} row-frames for {detected} detections")
    summary = summarise(bouts)
    check("the summary counts each of those frames once",
          f"covering {detected} distinct track-frames" in summary, summary)
    check("...which is not what summing the rows would have said",
          f"covering {naive} " not in summary, summary)
    check("and it says a row is a bout, over how many groups",
          f"{len(bouts)} bouts over 2 category group(s)" in summary, summary)


def test_an_overlapping_annotation_grid_has_a_way_through(root: Path) -> None:
    """`predict.window_frames: auto` is two promises, and overlapping clips break
    the tie between them.

    `auto` is min(clips.frames, clips.stride): the longest dense window that is
    BOTH the length the model was fitted at (clips.frames - clip_features.parquet
    is cut over the annotation manifest) AND aligned with that manifest's own
    starts (a multiple of clips.stride). Those are the same number only while the
    clips tile. Set clips.stride below clips.frames and the `min` keeps the
    alignment and drops the trained length, so every dense row is measured at
    clips.stride frames and scored by a model fitted at clips.frames - silently
    before the trained-window stamp existed, and as a mid-run refusal after it,
    which left the project simply unable to run a dense pass at all.

    So `auto` is refused at load and the number is asked for, with both coherent
    answers named. The point of this test is that both of them WORK: an
    overlapping annotation grid is a legitimate project, and neither route asks
    anyone to relabel anything.
    """
    from trackanno.config import ConfigError, load_config
    from trackanno.predict import predict_all
    from trackanno.project import write_config

    FRAMES, STRIDE = 51, 50            # the one-frame overlap that started this
    cfg, _ = build_case(root, "overlap-auto", coverage="dense", labels=2,
                        trained_window=FRAMES)
    before = digest(cfg.paths.csv)
    overlapping = {"clips.frames": FRAMES, "clips.stride": STRIDE}

    try:
        # write_config validates before it commits, so an unresolvable config
        # cannot be written from the GUI either - the refusal arrives while the
        # setting is being changed rather than at the next start-up.
        write_config(cfg.path, dict(overlapping))
        check("auto over an overlapping clip grid is refused", False, "it resolved")
    except ConfigError as exc:
        message = str(exc)
        check("auto over an overlapping clip grid is refused", True)
        check("...naming the key that fixes it, which is the whole complaint",
              "predict.window_frames" in message, message)
        check("...and both numbers that disagree, spelled out",
              f"clips.stride is {STRIDE}" in message
              and f"clips.frames is {FRAMES}" in message, message)
        check("...and the number to write, not just that a number is wanted",
              f"predict.window_frames: {FRAMES}" in message, message)
        check("...saying labels are not at stake, since that is the fear a "
              "refusal at load raises", "label" in message, message)
    untouched = load_config(cfg.path)
    check("...and the config that could not be resolved was never committed, so "
          "the project on disk still tiles",
          untouched.clips.stride == untouched.clips.frames,
          f"{untouched.clips.frames}/{untouched.clips.stride}")

    # Route 1: tile at the trained length.
    write_config(cfg.path, dict(overlapping, **{"predict.window_frames": FRAMES}))
    tiled = load_config(cfg.path)
    check("writing the length out makes the project load again",
          tiled.predict.resolved_window(tiled.clips) == FRAMES,
          repr(tiled.predict.window_frames))
    reporter = Recorder()
    windows = predict_all(tiled, reporter=reporter)
    check("...and a dense run completes", len(windows) > 0)
    check("...with nothing said about a window length",
          not reporter.said("window length"), str(reporter.logged))
    check("...still writing windows, because the grid still tiles",
          "bout_id" not in windows.columns, str(list(windows.columns)))

    # Route 2: the trained length AND the annotation grid's starts, which is
    # overlap, which is what a decode is for.
    write_config(cfg.path, {"predict.window_stride": STRIDE,
                            "predict.decode": "min_bout",
                            "predict.min_bout_frames": 20})
    overlapped = load_config(cfg.path)
    check("the overlapping route loads too",
          (overlapped.predict.resolved_window(overlapped.clips),
           overlapped.predict.resolved_stride(overlapped.clips)) == (FRAMES, STRIDE))
    bouts = predict_all(overlapped)
    check("...and its run completes, as bouts", len(bouts) > 0
          and "bout_id" in bouts.columns, str(list(bouts.columns)))
    check("...whose provenance is the frame-based share, not a clip_id join",
          "human_overlap_frac" in bouts.columns, str(list(bouts.columns)))

    check("no route touched a single hand label", digest(cfg.paths.csv) == before)

    # And the two routes really do trade the thing the refusal says they trade,
    # which is the whole reason it asks rather than choosing. Read off the grid
    # function itself, so the claim does not depend on where the fixture's tracks
    # happen to start and stop.
    from trackanno.densepredict import window_grid, window_length, window_stride

    long_movie = 10 * FRAMES
    tiled_starts = {s for s, _ in
                    window_grid(long_movie, window_length(tiled), window_stride(tiled))}
    over_starts = {s for s, _ in
                   window_grid(long_movie, window_length(overlapped),
                               window_stride(overlapped))}
    check("route 1 gives up the annotation grid's starts, as the refusal says",
          STRIDE not in tiled_starts, str(sorted(tiled_starts)))
    check("route 2 keeps them, as the refusal says",
          {0, STRIDE, 2 * STRIDE} <= over_starts, str(sorted(over_starts)))

    # The number `auto` used to pick is still available, and still wrong - which
    # is the evidence that the guard exposed a broken configuration rather than
    # breaking a working one. The refusal now names the key to edit.
    write_config(cfg.path, {"predict.window_frames": STRIDE,
                            "predict.window_stride": "auto",
                            "predict.decode": "none", "predict.min_bout_frames": 0})
    try:
        predict_all(load_config(cfg.path))
        check("what auto used to resolve to is still train/serve skew", False, "it ran")
    except RuntimeError as exc:
        message = str(exc)
        check("what auto used to resolve to is still train/serve skew", True)
        check("...refused by the length the model recorded, naming the key",
              "predict.window_frames" in message, message)
        check("...and the number to write there",
              f"predict.window_frames: {FRAMES}" in message, message)

    # A project whose clips tile is untouched: `auto` still resolves, and to the
    # same number it always did.
    plain, _ = build_case(root, "overlap-none", coverage="dense")
    check("a tiling clip grid still resolves auto, as every project on disk does",
          plain.predict.window_frames == "auto"
          and plain.predict.resolved_window(plain.clips) == plain.clips.frames,
          repr(plain.predict.window_frames))

    # And the refusal is scoped to the coverage the key governs: manifest
    # coverage scores clip_features.parquet, whose length is clips.frames
    # whatever the dense grid would have said.
    manifest_cfg, _ = build_case(root, "overlap-manifest", coverage="manifest")
    write_config(manifest_cfg.path, dict(overlapping))
    check("a manifest-coverage project with overlapping clips still loads",
          load_config(manifest_cfg.path).clips.stride == STRIDE)


def test_a_min_bout_floor_that_cannot_bind_is_refused(root: Path) -> None:
    """`min_bout_frames: 1` is `0` wearing a different number.

    `bouts.decode_min_bout` returns the plain argmax whenever the floor is 1 or
    less, because the shortest run a decode can produce is already one frame
    long: a floor of 1 excludes nothing. 0 was refused under `decode: min_bout`
    for exactly that reason - "the decode has no shortest bout to enforce and
    would hand back the labels it was given, unchanged" - and 1 was accepted, so
    the second spelling of "do nothing" got through under a decode that
    announces it is doing something.
    """
    from trackanno.config import ConfigError, load_config
    from trackanno.project import write_config

    cfg, _ = build_case(root, "floor", coverage="dense")
    # A floor that binds first, so the two spellings of "no floor" below are
    # edits to a working project rather than its initial state - and so there is
    # something for the refusal to leave intact.
    write_config(cfg.path, {"predict.decode": "min_bout", "predict.min_bout_frames": 5})
    check("a floor that can actually bind loads",
          load_config(cfg.path).predict.min_bout_frames == 5)

    for floor in (0, 1):
        try:
            # write_config validates before it commits, so this is the refusal a
            # settings edit in the GUI gets, not only the one a hand-edited file
            # gets on the next load.
            write_config(cfg.path, {"predict.min_bout_frames": floor})
            check(f"min_bout_frames {floor} is refused under decode: min_bout",
                  False, "it loaded")
        except ConfigError as exc:
            message = str(exc)
            check(f"min_bout_frames {floor} is refused under decode: min_bout", True)
            check("...quoting the number that was written",
                  f"is {floor}" in message, message)
            check("...and saying what the decode would do instead of it",
                  "unchanged" in message, message)
        check("...leaving the working value in the file",
              load_config(cfg.path).predict.min_bout_frames == 5)
    check("...with the reason 1 is not different from 0 stated, since accepting "
          "one and refusing the other is the bug",
          "excludes nothing" in message, message)
    check("...and the smallest floor that binds",
          "2 or more" in message, message)

    # viterbi is a different question and keeps its own answer: there the floor
    # is only read to derive self_transition, so a floor of 1 with a probability
    # written out by hand is unused rather than inert-but-pretending.
    write_config(cfg.path, {"predict.decode": "viterbi",
                            "predict.min_bout_frames": 1,
                            "predict.self_transition": 0.95})
    check("viterbi with its own self_transition does not need a binding floor",
          load_config(cfg.path).predict.resolved_self_transition() == 0.95)


def test_a_model_fitted_at_another_window_length_is_refused(root: Path) -> None:
    """Window length changes what the features are WORTH, not what they are
    called, so a model fitted at 50 frames and handed 25-frame rows passes every
    check the loader makes and then predicts confidently on a scale it has never
    seen. That is worth refusing over. A model that never recorded its length is
    a different case - every model saved before the stamp existed is one - and
    refusing those would mean an upgrade silently invalidates them, so it is a
    warning and the run goes on."""
    from trackanno.config import load_config
    from trackanno.densepredict import DENSE_FEATURES_FILE
    from trackanno.predict import predict_all
    from trackanno.project import write_config

    cfg, _ = build_case(root, "skew-ok", trained_window=50)
    reporter = Recorder()
    predict_all(cfg, reporter=reporter)
    check("a model asked for the length it was trained at says nothing",
          not reporter.said("window"), str(reporter.logged))

    cfg, _ = build_case(root, "skew-dense", coverage="dense", trained_window=50)
    write_config(cfg.path, {"predict.window_frames": 25})
    cfg = load_config(cfg.path)
    try:
        predict_all(cfg)
        check("a mismatched window length is refused", False, "it ran anyway")
    except RuntimeError as exc:
        check("a mismatched window length is refused", "50" in str(exc), str(exc))
        check("...naming the length the features would have been built at",
              "25" in str(exc), str(exc))
        check("...and the mechanism, not just the two numbers",
              "window length changes" in str(exc).lower(), str(exc))
    check("the refusal comes before the expensive part, so no dense table is built",
          not (cfg.paths.work_dir / DENSE_FEATURES_FILE).exists())

    # The dense grid's own window, not the clip length: a project whose annotation
    # clips are 50 frames long can still export a 25-frame dense grid, and it is
    # the grid the model is about to be shown that has to match.
    cfg, _ = build_case(root, "skew-manifest", trained_window=50)
    write_config(cfg.path, {"predict.window_frames": 25})
    reporter = Recorder()
    frame = predict_all(load_config(cfg.path), reporter=reporter)
    check("the same setting on the manifest path is not a mismatch, because those "
          "rows are clips.frames long whatever the dense grid says",
          len(frame) > 0 and not reporter.said("window length"),
          str(reporter.logged))

    cfg, _ = build_case(root, "skew-legacy")
    reporter = Recorder()
    frame = predict_all(cfg, reporter=reporter)
    check("a model that never recorded its window length still runs", len(frame) > 0)
    check("...but the run says the check could not be made",
          any("does not record" in m and level == "warning"
              for m, level in reporter.logged), str(reporter.logged))


def test_the_window_gate_makes_one_decision_for_both_entry_points(root: Path) -> None:
    """A labelling run and the annotation queue ask the same question about the
    same model file, and they must not be able to answer it differently.

    This is a train/serve-skew gate. Two copies of the three-way decision agree
    only until one of them is edited: tighten the bulk path alone and selection
    keeps ranking every candidate on a scale the model never saw, at normal
    confidence, with an ORDER as its only output and nothing in it to look
    wrong. So both go through ``Suggester.refuse_window_mismatch``, and what the
    caller supplies is the trailing sentence naming its own numbers.
    """
    import warnings

    from trackanno.config import load_config
    from trackanno.predict import predict_all
    from trackanno.project import write_config
    from trackanno.suggest import Suggester

    def selecting(name: str, trained_window: int | None):
        """A project both entry points can run over: a model on disk and a
        selection rule, so `selection_batch` does not return [] before it asks."""
        case, manifest = build_case(root, name, trained_window=trained_window)
        write_config(case.path, {"suggest.select_rule": "margin"})
        return load_config(case.path), manifest

    def refusal(fn, *args) -> str:
        try:
            fn(*args)
        except RuntimeError as exc:
            return str(exc)
        return ""

    # A model fitted at 25 frames in a project whose clips are 50 long. Manifest
    # coverage scores clip_features.parquet, so both entry points are reading
    # the SAME length off the same project - only the sentence about where that
    # number came from is the caller's.
    cfg, manifest = selecting("one-gate", 25)
    suggester = Suggester(cfg)
    problem = suggester.window_problem(int(cfg.clips.frames))
    check("the model and the project really do disagree", problem != "", problem)

    predicting = refusal(predict_all, cfg)
    ranking = refusal(suggester.selection_batch, manifest, set(), 10)
    check("a labelling run refuses the mismatch", predicting != "")
    check("...and so does a selection batch, off the same model file", ranking != "")
    check("...both saying the same thing about the model",
          predicting.startswith(problem) and ranking.startswith(problem),
          f"{predicting}\n--\n{ranking}")
    check("...and differing only in the numbers each one owns",
          predicting != ranking
          and "manifest coverage scores" in predicting
          and "every candidate in this batch" in ranking,
          f"{predicting}\n--\n{ranking}")

    # The other two states of the same decision, which is where a second copy
    # would drift: a model that never recorded its length is a warning on both
    # paths, and a matching one is silent on both.
    legacy, legacy_manifest = selecting("one-gate-legacy", None)
    quiet = Suggester(legacy)
    reporter = Recorder()
    rows = predict_all(legacy, reporter=reporter)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        batch = quiet.selection_batch(legacy_manifest, set(), 5)

    warned = [m for m, level in reporter.logged if level == "warning"
              and "does not record" in m]
    check("an unrecorded length stops neither path",
          len(rows) > 0 and len(batch) > 0, f"{len(rows)} rows, {len(batch)} clips")
    check("...and both warn about it", len(warned) == 1 and len(caught) == 1,
          f"{warned} / {[str(w.message) for w in caught]}")
    check("...in the same words, because it is one sentence with two channels",
          warned[0] == str(caught[0].message), warned[0])

    matched, matched_manifest = selecting("one-gate-matched", int(legacy.clips.frames))
    agreeing = Suggester(matched)
    reporter = Recorder()
    predict_all(matched, reporter=reporter)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        batch = agreeing.selection_batch(matched_manifest, set(), 5)
    check("and a model asked for the length it was trained at is silent on both",
          not reporter.said("window") and not caught and len(batch) > 0,
          f"{reporter.logged} / {[str(w.message) for w in caught]}")


def test_a_failed_write_leaves_no_part_file_and_no_damage(root: Path) -> None:
    """predictions.parquet is read as authoritative by everything downstream, so
    a run that dies mid-write must leave the previous file intact rather than a
    truncated parquet under the real name - and must not litter the work dir
    with .part files that the next StagedWrite would have to clean up."""
    from trackanno.predict import PREDICTIONS_FILE, _write_atomic, predict_all

    cfg, _ = build_case(root, "atomic")
    good = predict_all(cfg)
    target = cfg.paths.work_dir / PREDICTIONS_FILE
    before = digest(target)

    # A column pyarrow cannot encode: a real serialisation failure, raised at
    # exactly the point where the staged file already exists.
    broken = pd.DataFrame({"clip_id": ["a", "b"], "junk": [object(), object()]})
    raised = ""
    try:
        _write_atomic(broken, cfg.paths.work_dir)
    except Exception as exc:                       # noqa: BLE001 - any failure will do
        raised = type(exc).__name__
    check("a broken frame is not written silently", bool(raised), "it succeeded")

    check("no .part file is left behind",
          not list(cfg.paths.work_dir.glob("*.part")),
          str([p.name for p in cfg.paths.work_dir.glob("*.part")]))
    check("the previous predictions are untouched", digest(target) == before)
    check("...and still readable", len(pd.read_parquet(target)) == len(good))


def test_staleness_notices_a_retrain(root: Path) -> None:
    """Retraining and forgetting to re-label is the way these files go wrong, and
    nothing in an exported movie reveals which model drew the dashed boxes."""
    from trackanno.predict import predict_all, staleness
    from trackanno.suggest import MODEL_FILE

    cfg, _ = build_case(root, "stale")
    predict_all(cfg)
    check("fresh predictions are not stale", staleness(cfg) == "", staleness(cfg))

    model_path = cfg.paths.work_dir / MODEL_FILE
    os.utime(model_path, (time.time() + 60, time.time() + 60))
    check("a newer model makes them stale", "retrained" in staleness(cfg), staleness(cfg))


# --------------------------------------------------------------------------- #
# the export
# --------------------------------------------------------------------------- #
def test_a_prediction_never_looks_like_a_hand_label(root: Path) -> None:
    """Colour cannot carry the difference - the categories already own the
    colours. A predicted box is dashed and captioned with a ~ and its
    confidence, and where both exist the hand label wins the box while the
    model's answer stays visible in the caption."""
    from trackanno.overlay import _draw_box, _draw_dashed_box
    from trackanno.render import build_frame_marks

    cfg, manifest = build_case(root, "marks")
    row = manifest.iloc[0]
    annotations = pd.DataFrame([{
        "clip_id": row["clip_id"], "movie_id": row["movie_id"],
        "track_id": int(row["track_id"]), "f_start": 0, "f_end": 4,
        GROUP: "swimming",
    }])
    predictions = pd.DataFrame([{
        "clip_id": row["clip_id"], "movie_id": row["movie_id"],
        "track_id": int(row["track_id"]), "f_start": 0, "f_end": 4,
        f"{GROUP}_pred": "immotile", f"{GROUP}_conf": 0.73,
    }])
    tid = int(row["track_id"])

    marks = build_frame_marks(annotations, predictions, cfg, GROUP, "labels")
    check("'my labels' is exactly the labels", not marks[0][tid].predicted
          and marks[0][tid].text == "swimming", str(marks[0][tid]))

    marks = build_frame_marks(annotations, predictions, cfg, GROUP, "predictions")
    mark = marks[0][tid]
    check("'model predictions' is marked as predicted", mark.predicted)
    check("...and carries the confidence", "0.73" in mark.text, mark.text)
    check("...and is visibly not a label", mark.text.startswith("~"), mark.text)

    marks = build_frame_marks(annotations, predictions, cfg, GROUP, "both")
    mark = marks[0][tid]
    check("'both' keeps the hand label solid", not mark.predicted)
    check("...and shows the disagreement",
          "swimming" in mark.text and "~immotile" in mark.text, mark.text)

    # A track the annotator never reached keeps the dashed, predicted mark.
    other = predictions.copy()
    other["track_id"] = tid + 1
    marks = build_frame_marks(annotations, pd.concat([predictions, other]), cfg, GROUP, "both")
    check("an unlabelled track stays a prediction", marks[0][tid + 1].predicted)

    # And the difference survives into pixels.
    solid = np.zeros((40, 40, 3), np.uint8)
    dashed = np.zeros((40, 40, 3), np.uint8)
    _draw_box(solid, 5, 5, 34, 34, (255, 255, 255), 1)
    _draw_dashed_box(dashed, 5, 5, 34, 34, (255, 255, 255), 1)
    check("a dashed box draws fewer pixels than a solid one",
          0 < int((dashed > 0).sum()) < int((solid > 0).sum()),
          f"{int((dashed > 0).sum())} vs {int((solid > 0).sum())}")


def test_the_exported_pixels_show_which_is_which(root: Path) -> None:
    """The distinction has to survive all the way into the file that leaves the
    project. Same track, same category, same colour: the only difference between
    the two renders is that one value came from a person and one from the model,
    and the dashed outline is what carries that."""
    import tifffile

    from test_pipeline import SyntheticMovie

    from trackanno.overlay import hex_to_rgb
    from trackanno.render import render_movie

    cfg, _ = build_case(root, "pixels")
    # Captions off, so the comparison below is about the box and nothing else.
    cfg.render.show_category = False
    cfg.render.show_track_id = False

    movie = SyntheticMovie()
    tracks = movie.tracks()
    tracks = tracks[tracks["frame"] < 4]
    row = {"clip_id": "c", "movie_id": movie.id, "track_id": 1, "f_start": 0, "f_end": 4}
    annotations = pd.DataFrame([{**row, GROUP: "swimming"}])
    predictions = pd.DataFrame([{**row, f"{GROUP}_pred": "swimming", f"{GROUP}_conf": 0.9}])

    out = Path(root) / "pixels"
    solid = render_movie(cfg, movie, tracks, annotations, GROUP, out / "labels.tif",
                         max_frames=4, write_tif=True, source="labels")
    dashed = render_movie(cfg, movie, tracks, annotations, GROUP, out / "predicted.tif",
                          max_frames=4, write_tif=True, predictions=predictions,
                          source="predictions")

    green = np.array(hex_to_rgb("#4C9F70"), dtype=np.uint8)
    n_solid = int((tifffile.imread(str(solid)) == green).all(axis=-1).sum())
    n_dashed = int((tifffile.imread(str(dashed)) == green).all(axis=-1).sum())
    check("a hand label is drawn in its category colour", n_solid > 0)
    check("a prediction is drawn in the same colour", n_dashed > 0)
    check("...but with a broken outline, so the two never look alike",
          n_dashed < n_solid, f"{n_dashed} vs {n_solid} pixels")


def test_the_export_refuses_a_missing_predictions_file(root: Path) -> None:
    """Rendering nothing is indistinguishable from "the model had no opinion
    about this movie", so an absent predictions file has to stop the export and
    say where the file comes from."""
    from trackanno.predict import PREDICTIONS_FILE, load_predictions, predict_all
    from trackanno.render import render_all

    cfg, _ = build_case(root, "no-predictions", labels=2)
    try:
        render_all(cfg, source="predictions")
        check("rendering by predictions refuses without them", False, "it ran")
    except FileNotFoundError as exc:
        check("rendering by predictions refuses without them",
              "Label all clips with the model" in str(exc), str(exc))
    try:
        render_all(cfg, source="both")
        check("...and so does 'both'", False, "it ran")
    except FileNotFoundError:
        check("...and so does 'both'", True)

    # A group the model was not allowed to label is its own failure, and saying
    # "no predictions" for it would send the user to re-run a pass they already
    # ran.
    predict_all(cfg)
    check("the predictions are there now",
          (cfg.paths.work_dir / PREDICTIONS_FILE).exists())
    try:
        load_predictions(cfg, "quality")
        check("an unpredicted group is refused by name", False, "it passed")
    except ValueError as exc:
        check("an unpredicted group is refused by name",
              "quality" in str(exc) and GROUP in str(exc), str(exc))


def test_the_export_page_offers_the_three_sources(app, root: Path) -> None:
    """The page must not let a run start that can only produce blank movies, and
    the refusal has to name the file and how to make it."""
    from qtpy.QtWidgets import QMessageBox

    from trackanno.gui.hub import ExportPage, Hub
    from trackanno.predict import predict_all

    cfg, _ = build_case(root, "export-page", labels=2)
    hub = Hub(str(cfg.path))
    hub.show()
    app.processEvents()
    page = hub.page(ExportPage)

    check("the source selector offers three sources", page.source.count() == 3,
          str([page.source.itemText(i) for i in range(page.source.count())]))
    check("it defaults to your own labels", page.source_value() == "labels")
    check("...which keeps the export unblocked", page.blocker() == "", page.blocker())

    # Both, because a refusal takes whichever route fits: the page's own gate
    # speaks through information(), a bad predictions file through warning().
    said: list[tuple[str, str]] = []
    originals = (QMessageBox.warning, QMessageBox.information)
    for name in ("warning", "information"):
        setattr(QMessageBox, name, staticmethod(
            lambda parent, title, text, *a, **k: said.append((title, text))
        ))
    try:
        page.source.setCurrentIndex(1)          # model predictions
        app.processEvents()
        check("with no predictions the page blocks itself",
              "no predictions yet" in page.blocker(), page.blocker())
        check("...and says how to make them",
              "Label all clips with the model" in page.blocker(), page.blocker())
        check("the movie list says so too rather than sitting empty",
              page.movies.count() == 1
              and "no predictions" in page.movies.item(0).text(),
              page.movies.item(0).text() if page.movies.count() else "<empty>")

        check("...and pressing render refuses instead of starting",
              page._build_work() is None)
        check("the refusal names the file and how to make it",
              any("Label all clips with the model" in text for _, text in said),
              str(said))

        # With predictions present the same page runs, and lists the movies.
        predict_all(cfg)
        page.on_project()
        app.processEvents()
        check("predictions unblock the page", page.blocker() == "", page.blocker())
        check("and it lists the predicted movies", page.movies.count() > 0)
        check("the option is spelled out to the viewer",
              "dashed" in page.source_note.text(), page.source_note.text())
        work = page._build_work()
        check("the run is accepted now", work is not None)
    finally:
        QMessageBox.warning, QMessageBox.information = originals
        hub.close()
        app.processEvents()


def test_the_model_page_labels_every_clip(app, root: Path) -> None:
    """The button is gated on a reliable model, runs off the UI thread, and
    reports what it did. It must never write to the annotation CSV, however
    many times it is pressed."""
    from trackanno.gui.hub import Hub, ModelPage
    from trackanno.predict import PREDICTIONS_FILE

    # First a project whose model the report gated off: the button must be dead.
    cfg, _ = build_case(root, "page-silent", reliable=False, labels=3)
    hub = Hub(str(cfg.path))
    hub.show()
    app.processEvents()
    page = hub.page(ModelPage)
    check("a gated-off model leaves the button disabled",
          not page.predict_button.isEnabled())
    check("...and the page says why",
          "reliable enough" in page.predict_note.text(), page.predict_note.text())
    hub.close()
    app.processEvents()

    cfg, manifest = build_case(root, "page-live", labels=2)
    before = digest(cfg.paths.csv)
    hub = Hub(str(cfg.path))
    hub.show()
    app.processEvents()
    page = hub.page(ModelPage)
    check("a reliable model enables the button", page.predict_button.isEnabled())

    page.predict_button.click()
    check("the job runs and finishes", pump(app, lambda: not page.tasks.busy))
    check("...off the UI thread, through the task panel",
          "clips" in page.tasks.log.toPlainText(), page.tasks.log.toPlainText()[:200])

    target = cfg.paths.work_dir / PREDICTIONS_FILE
    check("every clip was labelled",
          target.exists() and len(pd.read_parquet(target)) == len(manifest))
    check("the annotation CSV is untouched", digest(cfg.paths.csv) == before)

    log = page.tasks.log.toPlainText()
    check("the summary counts the classes", "swimming" in log and "immotile" in log,
          log[-400:])
    check("...and reports a mean confidence", "mean confidence" in log, log[-400:])
    check("the hand labels are counted as the annotator's",
          "2 of them already labelled by hand" in log, log[-400:])

    page.shutdown()
    hub.close()
    app.processEvents()


def test_a_movie_with_no_detector_embeddings_is_refused_not_filled_with_nan(
    root: Path,
) -> None:
    """Found by running the real project: a movie missing instance_features.npy
    used to write every cnn* column as NaN and predict on them anyway.

    The classifier is NaN-native, so nothing objected. On three real movies that
    silenced an entire behaviour class - 197 windows of `frozen` became 0 - while
    mean confidence stayed at 0.83 and the log said nothing at all. The two
    checks beside this one already refused a MISALIGNED and a WRONG-WIDTH
    embeddings file; absent was the third case and the only silent one.
    """
    import joblib
    from sklearn.decomposition import PCA

    from trackanno.clipfeatures import PCA_FILE
    from trackanno.densepredict import FEATURES_FILE, build_dense_features

    cfg, _ = build_case(root, "noembed", coverage="dense")
    work = cfg.paths.work_dir

    # A saved PCA is what says "the training features contain a cnn* block", so
    # it is the whole trigger. The fixture's movies carry no embeddings, which
    # is exactly the state being refused.
    pca = PCA(n_components=2).fit(np.arange(60, dtype=float).reshape(20, 3))
    joblib.dump(pca, work / PCA_FILE)
    present = [d / FEATURES_FILE for d in work.iterdir() if d.is_dir()]
    check("the fixture supplies no embeddings to begin with",
          not any(p.exists() for p in present))

    try:
        build_dense_features(cfg)
        check("a movie that cannot supply cnn* columns is refused", False)
    except RuntimeError as exc:
        message = str(exc)
        check("a movie that cannot supply cnn* columns is refused", True)
        check("...and the message names the missing file",
              FEATURES_FILE in message, message)
        check("...and the file whose presence made it required",
              PCA_FILE in message, message)
        check("...and says what would otherwise happen to those columns",
              "NaN" in message, message)
        check("...and names which movie to re-prepare",
              any(d.name in message for d in work.iterdir() if d.is_dir()), message)
