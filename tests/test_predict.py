"""Labelling every clip with the model, and exporting movies coloured by it.

Every check here is about provenance. A prediction has to stay recognisable as a
prediction at each stage: in the file it is written to (never the annotation
CSV), in the row that says whether a person judged the clip, and in the pixels
of the exported movie. The rest guards the two ways a bulk run can quietly go
wrong - labelling a whole project with a model the report gated off, and leaving
a half-written parquet behind under the real name.

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
):
    """A project with a clip manifest, clip features and a stubbed-in model.

    ``groups`` overrides the {group: reliable} map when a case needs a second
    group; ``labels`` is how many clips to hand label through the real
    AnnotationStore, so the CSV under test is one a session could have written.

    ``coverage`` is pinned into the project rather than left at the shipped
    default, because most of the cases below are assertions about the manifest
    path specifically - how many rows there are, where f_end came from, which
    clip got which class. The dense path gets its own case.
    """
    import joblib

    from trackanno.clipfeatures import CLIP_FEATURES_FILE
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
    })
    features.to_parquet(work / CLIP_FEATURES_FILE, index=False)

    wanted = groups if groups is not None else {GROUP: reliable}
    if model:
        joblib.dump(
            {"columns": ["speed_mean", "area_mean"],
             "models": {g: StubModel() for g in wanted}},
            work / MODEL_FILE,
        )
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
    logged: list[tuple[str, str]] = []

    class Recorder:
        def log(self, message, level="info"):
            logged.append((message, level))

        def stage(self, label, total=None):
            pass

        def advance(self, n=1):
            pass

        def raise_if_cancelled(self):
            pass

    frame = predict_all(cfg, reporter=Recorder())
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
    has to do it exactly once per frame, because the renderer paints
    range(f_start, f_end) per track and a frame in two windows means one
    prediction silently overwrites the other."""
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
