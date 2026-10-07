"""Bulk prediction into decoded bouts, and exporting movies coloured by it."""

from __future__ import annotations

import hashlib
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fixtures import (
    CLASSES, CUT, GROUP, LENGTH, PLANTED_FRAMES, build_case, planted_case, pump,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# a project with a "trained" model in it
# --------------------------------------------------------------------------- #


class Recorder:
    """A reporter that keeps every logged line."""

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


def bout_edge(frame: pd.DataFrame, label: str) -> int:
    """The first frame the decode calls ``label``, or -1 if it never does."""
    rows = frame[frame[f"{GROUP}_pred"] == label]
    return int(rows["f_start"].min()) if len(rows) else -1


def _strided(root: Path, name: str, **kwargs):
    """`build_case` at stride 10 with a 20-frame bout floor."""
    from trackanno.config import load_config
    from trackanno.project import write_config

    cfg, manifest = build_case(root, name, **kwargs)
    write_config(cfg.path, {"predict.window_stride": 10, "predict.min_bout_frames": 20})
    return load_config(cfg.path), manifest


# --------------------------------------------------------------------------- #
# refusing to run
# --------------------------------------------------------------------------- #
def test_an_absent_model_is_refused(root: Path) -> None:
    from trackanno.infer.predict import PREDICTIONS_FILE, predict_all, prediction_blocker

    cfg, _ = build_case(root, "no-model", model=False, report=False)
    with pytest.raises(RuntimeError, match="no model has been trained"):
        predict_all(cfg)
    assert "no model has been trained" in prediction_blocker(cfg)
    assert not (cfg.paths.work_dir / PREDICTIONS_FILE).exists()


def test_a_model_with_no_report_is_refused(root: Path) -> None:
    from trackanno.infer.predict import PREDICTIONS_FILE, predict_all

    cfg, _ = build_case(root, "no-report", report=False)
    with pytest.raises(RuntimeError, match="suggest_report.json"):
        predict_all(cfg)
    assert not (cfg.paths.work_dir / PREDICTIONS_FILE).exists()


def test_a_model_the_report_gated_off_is_refused_by_group(root: Path) -> None:
    from trackanno.infer.predict import PREDICTIONS_FILE, predict_all

    cfg, _ = build_case(root, "silent", reliable=False)
    with pytest.raises(RuntimeError, match="not distinguishable from chance") as exc:
        predict_all(cfg)
    assert GROUP in str(exc.value), "the refusal names the group"
    assert not (cfg.paths.work_dir / PREDICTIONS_FILE).exists()


def test_an_unreliable_group_is_skipped_not_silently_predicted(root: Path) -> None:
    """The gate is per group, and the skipped group is reported with its reason."""
    from trackanno.infer.predict import predict_all, verdict

    cfg, _ = build_case(root, "mixed", groups={GROUP: True, "shape": False})
    reporter = Recorder()
    logged = reporter.logged

    frame = predict_all(cfg, reporter=reporter)
    assert f"{GROUP}_pred" in frame.columns, "the reliable group is predicted"
    assert "shape_pred" not in frame.columns, list(frame.columns)
    assert any("shape" in m and "chance" in m and level == "warning"
               for m, level in logged), f"the skip is reported with its reason: {logged}"
    decision = verdict(cfg)
    assert decision.usable == [GROUP] and set(decision.skipped) == {"shape"}, decision
    assert decision.problem == "", "a partial skip does not block the run"


# --------------------------------------------------------------------------- #
# what it writes
# --------------------------------------------------------------------------- #
def test_one_row_per_bout_with_the_expected_columns(root: Path) -> None:
    from trackanno.infer.predict import PREDICTIONS_FILE, predict_all

    cfg, _ = build_case(root, "columns")
    frame = predict_all(cfg)

    for column in ("bout_id", "movie_id", "track_id", "f_start", "f_end", "n_frames",
                   "group", f"{GROUP}_pred", f"{GROUP}_conf", "human_labelled",
                   "human_overlap_frac", "n_covering_mean", "skip_reason"):
        assert column in frame.columns, f"missing {column}: {list(frame.columns)}"
    given = frame[frame[f"{GROUP}_pred"] != ""]
    assert given[f"{GROUP}_pred"].isin(CLASSES).all(), frame[f"{GROUP}_pred"].unique()
    assert ((given[f"{GROUP}_conf"] > 0.0) & (given[f"{GROUP}_conf"] <= 1.0)).all(), (
        "confidences are probabilities")

    on_disk = pd.read_parquet(cfg.paths.work_dir / PREDICTIONS_FILE)
    assert on_disk.equals(frame), "the file on disk is what was returned"


def test_predictions_never_reach_the_annotation_csv(root: Path) -> None:
    """The CSV is ground truth; predictions go to their own file."""
    from trackanno.infer.predict import PREDICTIONS_FILE, predict_all

    cfg, _ = build_case(root, "provenance", labels=3)
    before = digest(cfg.paths.csv)
    rows_before = pd.read_csv(cfg.paths.csv)

    predict_all(cfg)
    assert digest(cfg.paths.csv) == before, "the annotation CSV is byte-for-byte unchanged"

    rows_after = pd.read_csv(cfg.paths.csv)
    assert len(rows_after) == len(rows_before) == 3
    assert not [c for c in rows_after.columns if c.endswith("_pred")], (
        f"a prediction column reached the CSV: {list(rows_after.columns)}")
    assert (cfg.paths.work_dir / PREDICTIONS_FILE).exists()

    predict_all(cfg)
    assert digest(cfg.paths.csv) == before, "a second run does not touch it either"


def test_bouts_cover_every_detected_frame_once(root: Path) -> None:
    """Every detected frame is in exactly one bout per group; clip manifests are untouched."""
    from trackanno.infer.densepredict import DENSE_FEATURES_FILE
    from trackanno.infer.predict import predict_all

    cfg, manifest = build_case(root, "dense", labels=1)
    work = cfg.paths.work_dir
    bouts = predict_all(cfg)
    assert (work / DENSE_FEATURES_FILE).exists(), "the dense pass writes its own feature table"

    for movie_id in bouts["movie_id"].unique():
        tracks = pd.read_parquet(work / str(movie_id) / "tracks.parquet")
        mine = bouts[bouts["movie_id"] == movie_id]
        uncovered = doubled = 0
        for track_id, tdf in tracks.groupby("track_id"):
            spans = mine[mine["track_id"] == int(track_id)][["f_start", "f_end"]].to_numpy()
            for frame in tdf["frame"].to_numpy():
                hits = int(((spans[:, 0] <= frame) & (frame < spans[:, 1])).sum())
                uncovered += int(hits == 0)
                doubled += int(hits > 1)
        assert uncovered == 0, f"{movie_id}: {uncovered} detected frames in no bout"
        assert doubled == 0, f"{movie_id}: {doubled} frames covered twice"
        assert int(mine["f_end"].max()) == int(tracks["frame"].max()) + 1, (
            f"{movie_id}: the tail is dropped")

    for m in manifest["movie_id"].unique():
        assert pd.read_parquet(work / m / "clips.parquet").equals(
            manifest[manifest["movie_id"] == m].reset_index(drop=True)), m


def test_a_window_too_thin_to_measure_is_a_row_not_a_hole(root: Path) -> None:
    """An unfeaturisable window survives as a row carrying the reason."""
    from trackanno.infer.densepredict import build_dense_features
    from trackanno.infer.predict import predict_all

    cfg, _ = build_case(root, "thin")
    cfg.predict.min_frames = 999           # above anything the fixture's tracks supply

    dense = build_dense_features(cfg)
    assert len(dense) > 0, "every window is still a row"
    assert (dense["skip_reason"] != "").all(), "no window survives the bar"
    assert dense["skip_reason"].str.contains("a window needs 999").all(), (
        dense["skip_reason"].iloc[0])

    out = predict_all(cfg)
    assert (out[f"{GROUP}_pred"] == "").all(), "nothing is predicted"
    assert out[f"{GROUP}_conf"].isna().all(), "no confidence is claimed"
    assert (out["skip_reason"] != "").all(), "every bout says why"
    columns = pd.read_parquet(cfg.paths.work_dir / "dense_clip_features.parquet").columns
    assert {"speed_mean", "area_mean"} <= set(columns), (
        "the model's columns survive the all-skipped table")


# --------------------------------------------------------------------------- #
# decoded bouts
# --------------------------------------------------------------------------- #
def test_writing_the_default_stride_out_by_hand_changes_the_file_not_at_all(
    root: Path,
) -> None:
    """`window_stride: auto` and its value spelled out write the same bytes, run after run."""
    from trackanno.config import load_config
    from trackanno.infer.densepredict import window_length
    from trackanno.infer.predict import PREDICTIONS_FILE, predict_all
    from trackanno.project import write_config

    auto, _ = build_case(root, "stride-auto")
    spelled, _ = build_case(root, "stride-spelled")
    write_config(spelled.path, {"predict.window_stride": window_length(spelled)})
    spelled = load_config(spelled.path)

    assert (spelled.predict.resolved_stride(spelled.clips)
            == auto.predict.resolved_stride(auto.clips) == auto.clips.frames)

    first = predict_all(auto)
    other = predict_all(spelled)
    assert first.equals(other), f"{first.shape} vs {other.shape}"
    assert (digest(auto.paths.work_dir / PREDICTIONS_FILE)
            == digest(spelled.paths.work_dir / PREDICTIONS_FILE))

    before = digest(auto.paths.work_dir / PREDICTIONS_FILE)
    predict_all(auto)
    assert digest(auto.paths.work_dir / PREDICTIONS_FILE) == before, (
        "a second identical run rewrites the same bytes")


@pytest.fixture(scope="module")
def strided_bouts(root: Path):
    """Bouts and dense windows of a stride-10 run."""
    from trackanno.infer.densepredict import DENSE_FEATURES_FILE
    from trackanno.infer.predict import predict_all

    cfg, _ = _strided(root, "bouts")
    bouts = predict_all(cfg)
    windows = pd.read_parquet(cfg.paths.work_dir / DENSE_FEATURES_FILE)
    return bouts, windows


def test_overlapping_windows_are_cut_into_fewer_bouts(strided_bouts) -> None:
    bouts, windows = strided_bouts
    assert len(bouts) < len(windows), f"{len(bouts)} bouts for {len(windows)} windows"
    assert set(bouts["group"]) == {GROUP}
    assert (bouts["n_frames"] == bouts["f_end"] - bouts["f_start"]).all(), (
        "n_frames is the span, not a nominal window length")


def test_a_bout_id_names_both_ends_and_there_is_no_clip_id(strided_bouts) -> None:
    """A join on clip_id raises instead of attributing a bout to a clip."""
    bouts, _ = strided_bouts
    assert "bout_id" in bouts.columns and bouts["bout_id"].astype(str).ne("").all()
    assert all(row["bout_id"].endswith(f"f{row['f_start']:06d}-{row['f_end']:06d}")
               for row in bouts.to_dict("records")), bouts["bout_id"].head(2).tolist()
    assert "clip_id" not in bouts.columns, list(bouts.columns)


def test_bouts_are_sorted_disjoint_and_gapless_per_track(strided_bouts) -> None:
    bouts, windows = strided_bouts
    for track_id, group in bouts.groupby("track_id"):
        spans = group.sort_values("f_start")[["f_start", "f_end"]].to_numpy()
        covered = windows[windows["track_id"] == track_id]
        assert (spans[:, 0] < spans[:, 1]).all(), f"track {track_id}: {spans}"
        assert list(spans[1:, 0]) == list(spans[:-1, 1]), f"track {track_id}: {spans}"
        assert ((int(spans[0, 0]), int(spans[-1, 1]))
                == (int(covered["f_start"].min()), int(covered["f_end"].max()))), (
            f"track {track_id}: bouts do not span the frames the windows covered")


def test_human_overlap_is_measured_in_frames(root: Path) -> None:
    """Each bout reports the fraction of its frames a person labelled."""
    from trackanno.infer.predict import predict_all

    cfg, _ = _strided(root, "overlap-frac", labels=1)
    bouts = predict_all(cfg)

    labelled = pd.read_csv(cfg.paths.csv).iloc[0]
    movie, track = str(labelled["movie_id"]), int(labelled["track_id"])
    lo, hi = int(labelled["f_start"]), int(labelled["f_end"])
    assert len(pd.read_csv(cfg.paths.csv)) == 1 and hi > lo, f"{lo}..{hi}"

    mine = bouts[(bouts["movie_id"] == movie) & (bouts["track_id"] == track)]
    for row in mine.to_dict("records"):
        overlap = max(0, min(row["f_end"], hi) - max(row["f_start"], lo))
        assert abs(row["human_overlap_frac"] - overlap / row["n_frames"]) < 1e-12, (
            f"bout {row['f_start']}-{row['f_end']}: {row['human_overlap_frac']} "
            f"vs {overlap}/{row['n_frames']}")
    share = mine["human_overlap_frac"]
    assert ((share > 0.0) & (share < 1.0)).any(), (
        f"no bout straddles the labelled clip: {share.tolist()}")

    others = bouts[bouts["track_id"] != track]
    assert (others["human_overlap_frac"] == 0.0).all() and not others["human_labelled"].any(), (
        "a track nobody labelled overlaps nothing")
    assert (bouts["human_labelled"] == (bouts["human_overlap_frac"] > 0.0)).all()


@pytest.fixture(scope="module")
def planted_runs(root: Path):
    """The planted movie decoded on the tiling grid and at stride 10."""
    from trackanno.infer.predict import predict_all

    tiled_cfg, _ = planted_case(root, "planted-tiled", LENGTH)
    dense_cfg, _ = planted_case(root, "planted-dense", 10)
    return predict_all(tiled_cfg), predict_all(dense_cfg)


def test_the_planted_movie_decodes_into_two_bouts_either_way(planted_runs) -> None:
    tiled, dense = planted_runs
    shown = ["f_start", "f_end", f"{GROUP}_pred"]
    assert len(tiled) == 2 and len(dense) == 2, (
        f"tiled {tiled[shown].to_dict('records')}, dense {dense[shown].to_dict('records')}")
    assert list(tiled[f"{GROUP}_pred"]) == list(dense[f"{GROUP}_pred"]) == list(CLASSES)
    assert int(tiled["n_frames"].sum()) == int(dense["n_frames"].sum()) == PLANTED_FRAMES, (
        "both cover the whole track")


def test_a_bout_boundary_lands_nearer_the_truth_than_a_tile_boundary(planted_runs) -> None:
    """Same exact model on both grids, so the whole difference is the grid."""
    tiled, dense = planted_runs
    tiled_edge = bout_edge(tiled, CLASSES[1])
    dense_edge = bout_edge(dense, CLASSES[1])
    assert tiled_edge % LENGTH == 0, tiled_edge
    assert tiled_edge == 150, f"the tiling grid answers 150 for a cut at 137: {tiled_edge}"
    assert abs(dense_edge - CUT) <= 10, f"{dense_edge} not within one stride of {CUT}"
    assert abs(dense_edge - CUT) < abs(tiled_edge - CUT)


def test_interior_bouts_report_how_many_windows_cover_them(planted_runs) -> None:
    tiled, dense = planted_runs
    assert float(dense["n_covering_mean"].max()) > 1.0, dense["n_covering_mean"].tolist()
    assert float(tiled["n_covering_mean"].max()) == 1.0, tiled["n_covering_mean"].tolist()


def test_the_summary_counts_a_frame_once_however_many_rows_cover_it(root: Path) -> None:
    """Two groups over the same frames count those frames once."""
    from trackanno.infer.predict import predict_all, summarise

    cfg, _ = _strided(root, "counts", groups={GROUP: True, "shape": True})
    bouts = predict_all(cfg)

    tracks = pd.concat([
        pd.read_parquet(d / "tracks.parquet")
        for d in sorted(cfg.paths.work_dir.iterdir())
        if d.is_dir() and (d / "tracks.parquet").exists()
    ])
    detected = int(len(tracks))
    naive = int((bouts["f_end"] - bouts["f_start"]).sum())

    assert set(bouts["group"]) == {GROUP, "shape"} and naive == 2 * detected, (
        f"{sorted(set(bouts['group']))}, {naive} row-frames for {detected} detections")
    summary = summarise(bouts)
    assert f"covering {detected} distinct track-frames" in summary, summary
    assert f"covering {naive} " not in summary, summary
    assert f"{len(bouts)} bouts over 2 category group(s)" in summary, summary


def test_an_overlapping_annotation_grid_still_predicts(root: Path) -> None:
    """The dense window is clips.frames whatever the clip stride."""
    from trackanno.config import load_config
    from trackanno.infer.predict import predict_all
    from trackanno.project import write_config

    frames, stride = 51, 50
    cfg, _ = build_case(root, "overlap-clips", labels=2, trained_window=frames)
    before = digest(cfg.paths.csv)
    write_config(cfg.path, {"clips.frames": frames, "clips.stride": stride})
    cfg = load_config(cfg.path)
    bouts = predict_all(cfg)
    assert len(bouts) > 0 and "bout_id" in bouts.columns

    write_config(cfg.path, {"predict.window_stride": stride})
    assert len(predict_all(load_config(cfg.path))) > 0, "an overlapping dense grid runs too"
    assert digest(cfg.paths.csv) == before, "no run touched a hand label"


def test_a_window_stride_past_the_window_is_refused(root: Path) -> None:
    from trackanno.config import ConfigError
    from trackanno.project import create_project, write_config

    path = create_project(root / "stride-past")
    with pytest.raises(ConfigError, match="predict.window_stride"):
        write_config(path, {"clips.frames": 51, "predict.window_stride": 52})


def test_a_floor_of_zero_keeps_every_run(root: Path) -> None:
    """`min_bout_frames` 0 and 1 both keep the plain argmax runs."""
    from trackanno.config import load_config
    from trackanno.infer.predict import predict_all
    from trackanno.project import write_config

    cfg, _ = build_case(root, "floor")
    write_config(cfg.path, {"predict.window_stride": 10, "predict.min_bout_frames": 0})
    zero = predict_all(load_config(cfg.path))
    write_config(cfg.path, {"predict.min_bout_frames": 1})
    one = predict_all(load_config(cfg.path))
    assert zero.equals(one), "0 and 1 cut the same bouts"
    write_config(cfg.path, {"predict.min_bout_frames": 40})
    floored = predict_all(load_config(cfg.path))
    assert len(floored) <= len(zero), f"{len(floored)} vs {len(zero)}"


def test_a_negative_bout_floor_is_refused(root: Path) -> None:
    from trackanno.config import ConfigError
    from trackanno.project import create_project, write_config

    path = create_project(root / "negative-floor")
    with pytest.raises(ConfigError):
        write_config(path, {"predict.min_bout_frames": -1})


# --------------------------------------------------------------------------- #
# the window-length gate
# --------------------------------------------------------------------------- #
def test_a_model_at_the_project_window_length_runs_silently(root: Path) -> None:
    from trackanno.infer.predict import predict_all

    cfg, _ = build_case(root, "skew-ok", trained_window=50)
    reporter = Recorder()
    predict_all(cfg, reporter=reporter)
    assert not reporter.said("frame windows"), reporter.logged


def test_a_model_fitted_at_another_window_length_is_refused(root: Path) -> None:
    """The refusal names both lengths and the setting, before any dense table is built."""
    from trackanno.config import load_config
    from trackanno.infer.densepredict import DENSE_FEATURES_FILE
    from trackanno.infer.predict import predict_all
    from trackanno.project import write_config

    cfg, _ = build_case(root, "skew-dense", trained_window=50)
    write_config(cfg.path, {"clips.frames": 25, "clips.stride": 25,
                            "clips.min_clip_frames": 20})
    cfg = load_config(cfg.path)
    with pytest.raises(RuntimeError) as exc:
        predict_all(cfg)
    message = str(exc.value)
    assert "50" in message and "25" in message, message
    assert "clips.frames" in message, message
    assert not (cfg.paths.work_dir / DENSE_FEATURES_FILE).exists()


def test_a_model_that_never_recorded_its_length_runs_with_a_warning(root: Path) -> None:
    from trackanno.infer.predict import predict_all

    cfg, _ = build_case(root, "skew-legacy")
    reporter = Recorder()
    frame = predict_all(cfg, reporter=reporter)
    assert len(frame) > 0
    assert any("does not record" in m and level == "warning"
               for m, level in reporter.logged), reporter.logged


def _selecting(root: Path, name: str, trained_window: int | None):
    """A case with a selection rule, so both entry points reach the window gate."""
    from trackanno.config import load_config
    from trackanno.project import write_config

    case, manifest = build_case(root, name, trained_window=trained_window)
    write_config(case.path, {"suggest.select_rule": "margin"})
    return load_config(case.path), manifest


def test_prediction_and_selection_refuse_a_window_mismatch_alike(root: Path) -> None:
    """Both entry points share one gate; only the trailing sentence is the caller's."""
    from trackanno.infer.predict import predict_all
    from trackanno.learn.suggest import Suggester

    def refusal(fn, *args) -> str:
        try:
            fn(*args)
        except RuntimeError as exc:
            return str(exc)
        return ""

    cfg, manifest = _selecting(root, "one-gate", 25)
    suggester = Suggester(cfg)
    problem = suggester.window_problem(int(cfg.clips.frames))
    assert problem != "", "the model and the project disagree"

    predicting = refusal(predict_all, cfg)
    ranking = refusal(suggester.selection_batch, manifest, set(), 10)
    assert predicting != "", "a labelling run refuses the mismatch"
    assert ranking != "", "a selection batch refuses it too"
    assert predicting.startswith(problem) and ranking.startswith(problem), (
        f"{predicting}\n--\n{ranking}")
    assert predicting != ranking, f"{predicting}\n--\n{ranking}"
    assert "The dense window is clips.frames" in predicting, predicting
    assert "frames are clips.frames" in ranking, ranking


def test_prediction_and_selection_warn_alike_on_an_unrecorded_length(root: Path) -> None:
    from trackanno.infer.predict import predict_all
    from trackanno.learn.suggest import Suggester

    legacy, manifest = _selecting(root, "one-gate-legacy", None)
    reporter = Recorder()
    rows = predict_all(legacy, reporter=reporter)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        batch = Suggester(legacy).selection_batch(manifest, set(), 5)

    warned = [m for m, level in reporter.logged
              if level == "warning" and "does not record" in m]
    assert len(rows) > 0 and len(batch) > 0, f"{len(rows)} rows, {len(batch)} clips"
    assert len(warned) == 1 and len(caught) == 1, (
        f"{warned} / {[str(w.message) for w in caught]}")
    assert warned[0] == str(caught[0].message), "both paths warn in the same words"


def test_prediction_and_selection_are_silent_on_a_matching_length(root: Path) -> None:
    from trackanno.infer.predict import predict_all
    from trackanno.learn.suggest import Suggester

    matched, manifest = _selecting(root, "one-gate-matched", LENGTH)
    assert int(matched.clips.frames) == LENGTH
    reporter = Recorder()
    predict_all(matched, reporter=reporter)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        batch = Suggester(matched).selection_batch(manifest, set(), 5)
    assert not reporter.said("frame windows"), reporter.logged
    assert not caught, [str(w.message) for w in caught]
    assert len(batch) > 0


# --------------------------------------------------------------------------- #
# the file on disk
# --------------------------------------------------------------------------- #
def test_a_failed_write_leaves_no_part_file_and_no_damage(root: Path) -> None:
    from trackanno.infer.predict import PREDICTIONS_FILE, _write_atomic, predict_all

    cfg, _ = build_case(root, "atomic")
    good = predict_all(cfg)
    target = cfg.paths.work_dir / PREDICTIONS_FILE
    before = digest(target)

    # A column pyarrow cannot encode fails after the staged file exists.
    broken = pd.DataFrame({"clip_id": ["a", "b"], "junk": [object(), object()]})
    with pytest.raises(Exception):  # noqa: B017 - any serialisation failure will do
        _write_atomic(broken, cfg.paths.work_dir)

    assert not list(cfg.paths.work_dir.glob("*.part")), (
        [p.name for p in cfg.paths.work_dir.glob("*.part")])
    assert digest(target) == before, "the previous predictions are untouched"
    assert len(pd.read_parquet(target)) == len(good)


def test_staleness_notices_a_retrain(root: Path) -> None:
    from trackanno.infer.predict import predict_all, staleness
    from trackanno.learn.suggest import MODEL_FILE

    cfg, _ = build_case(root, "stale")
    predict_all(cfg)
    assert staleness(cfg) == "", staleness(cfg)

    model_path = cfg.paths.work_dir / MODEL_FILE
    os.utime(model_path, (time.time() + 60, time.time() + 60))
    assert "retrained" in staleness(cfg), staleness(cfg)


# --------------------------------------------------------------------------- #
# the export
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def marks_case(root: Path):
    """One clip both hand-labelled swimming and predicted immotile."""
    cfg, manifest = build_case(root, "marks")
    row = manifest.iloc[0]
    span = {"clip_id": row["clip_id"], "movie_id": row["movie_id"],
            "track_id": int(row["track_id"]), "f_start": 0, "f_end": 4}
    annotations = pd.DataFrame([{**span, GROUP: "swimming"}])
    predictions = pd.DataFrame([{**span, f"{GROUP}_pred": "immotile",
                                 f"{GROUP}_conf": 0.73}])
    return cfg, annotations, predictions, int(row["track_id"])


def test_the_labels_source_draws_exactly_the_labels(marks_case) -> None:
    from trackanno.view.render import build_frame_marks

    cfg, annotations, predictions, tid = marks_case
    mark = build_frame_marks(annotations, predictions, cfg, GROUP, "labels")[0][tid]
    assert not mark.predicted and mark.text == "swimming", mark


def test_a_predicted_mark_is_flagged_and_carries_its_confidence(marks_case) -> None:
    from trackanno.view.render import build_frame_marks

    cfg, annotations, predictions, tid = marks_case
    mark = build_frame_marks(annotations, predictions, cfg, GROUP, "predictions")[0][tid]
    assert mark.predicted
    assert "0.73" in mark.text, mark.text
    assert mark.text.startswith("~"), mark.text


def test_both_keeps_the_hand_label_solid_and_shows_the_disagreement(marks_case) -> None:
    from trackanno.view.render import build_frame_marks

    cfg, annotations, predictions, tid = marks_case
    mark = build_frame_marks(annotations, predictions, cfg, GROUP, "both")[0][tid]
    assert not mark.predicted
    assert "swimming" in mark.text and "~immotile" in mark.text, mark.text

    other = predictions.copy()
    other["track_id"] = tid + 1
    marks = build_frame_marks(annotations, pd.concat([predictions, other]), cfg, GROUP, "both")
    assert marks[0][tid + 1].predicted, "an unlabelled track stays a prediction"


def test_a_dashed_box_draws_fewer_pixels_than_a_solid_one() -> None:
    from trackanno.view.overlay import _draw_box, _draw_dashed_box

    solid = np.zeros((40, 40, 3), np.uint8)
    dashed = np.zeros((40, 40, 3), np.uint8)
    _draw_box(solid, 5, 5, 34, 34, (255, 255, 255), 1)
    _draw_dashed_box(dashed, 5, 5, 34, 34, (255, 255, 255), 1)
    assert 0 < int((dashed > 0).sum()) < int((solid > 0).sum())


def test_the_exported_pixels_show_which_is_which(root: Path, monkeypatch) -> None:
    """Same track, category and colour: only the dashed outline tells a prediction apart."""
    import imageio.v2 as imageio
    from fixtures import SyntheticMovie

    from trackanno.view.overlay import hex_to_rgb
    from trackanno.view.render import render_movie

    cfg, _ = build_case(root, "pixels")

    movie = SyntheticMovie()
    tracks = movie.tracks()
    tracks = tracks[tracks["frame"] < 4]
    row = {"clip_id": "c", "movie_id": movie.id, "track_id": 1, "f_start": 0, "f_end": 4}
    annotations = pd.DataFrame([{**row, GROUP: "swimming"}])
    predictions = pd.DataFrame([{**row, f"{GROUP}_pred": "swimming", f"{GROUP}_conf": 0.9}])

    # Frames as handed to the encoder, since mp4 compression would blur the colour.
    written: dict[str, list] = {}

    class Capture:
        def __init__(self, path, **_kwargs):
            self.frames = written.setdefault(Path(path).name, [])

        def append_data(self, frame):
            self.frames.append(np.array(frame))

        def close(self):
            pass

    out = Path(root) / "pixels"
    monkeypatch.setattr(imageio, "get_writer", Capture)
    monkeypatch.setattr("trackanno.view.render.SHOW_CATEGORY", False)
    monkeypatch.setattr("trackanno.view.render.SHOW_TRACK_ID", False)
    render_movie(cfg, movie, tracks, annotations, GROUP, out / "labels.mp4",
                 max_frames=4, source="labels")
    render_movie(cfg, movie, tracks, annotations, GROUP, out / "predicted.mp4",
                 max_frames=4, predictions=predictions, source="predictions")

    green = np.array(hex_to_rgb("#4C9F70"), dtype=np.uint8)
    n_solid = int((np.stack(written["labels.mp4"]) == green).all(axis=-1).sum())
    n_dashed = int((np.stack(written["predicted.mp4"]) == green).all(axis=-1).sum())
    assert n_solid > 0, "a hand label is drawn in its category colour"
    assert n_dashed > 0, "a prediction is drawn in the same colour"
    assert n_dashed < n_solid, f"{n_dashed} vs {n_solid} pixels"


@pytest.fixture(scope="module")
def unpredicted(root: Path):
    """A labelled project that has not been predicted."""
    cfg, _ = build_case(root, "no-predictions", labels=2)
    return cfg


@pytest.mark.parametrize("source", ["predictions", "both"])
def test_the_export_refuses_a_missing_predictions_file(unpredicted, source) -> None:
    from trackanno.view.render import render_all

    with pytest.raises(FileNotFoundError, match="Label all clips with the model"):
        render_all(unpredicted, source=source)


def test_an_unpredicted_group_is_refused_by_name(root: Path) -> None:
    from trackanno.infer.predict import load_predictions, predict_all

    cfg, _ = build_case(root, "unpredicted-group")
    predict_all(cfg)
    with pytest.raises(ValueError) as exc:
        load_predictions(cfg, "quality")
    assert "quality" in str(exc.value) and GROUP in str(exc.value), str(exc.value)


def test_the_export_page_offers_the_three_sources(app, root: Path, monkeypatch) -> None:
    """With no predictions the page blocks itself and says how to make them."""
    from qtpy.QtWidgets import QMessageBox

    from trackanno.gui.hub import ExportPage, Hub
    from trackanno.infer.predict import predict_all

    cfg, _ = build_case(root, "export-page", labels=2)
    hub = Hub(str(cfg.path))
    hub.show()
    app.processEvents()
    page = hub.page(ExportPage)

    assert page.source.count() == 3, (
        [page.source.itemText(i) for i in range(page.source.count())])
    assert page.source_value() == "labels"
    assert page.blocker() == "", page.blocker()

    # The page's gate speaks through information(), a bad file through warning().
    said: list[tuple[str, str]] = []
    for name in ("warning", "information"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(
            lambda parent, title, text, *a, **k: said.append((title, text))
        ))
    try:
        page.source.setCurrentIndex(1)          # model predictions
        app.processEvents()
        assert "no predictions yet" in page.blocker(), page.blocker()
        assert "Label all clips with the model" in page.blocker(), page.blocker()
        assert page.movies.count() == 1, "the movie list explains itself rather than sit empty"
        assert "no predictions" in page.movies.item(0).text(), page.movies.item(0).text()

        assert page._build_work() is None, "pressing render refuses instead of starting"
        assert any("Label all clips with the model" in text for _, text in said), said

        predict_all(cfg)
        page.on_project()
        app.processEvents()
        assert page.blocker() == "", page.blocker()
        assert page.movies.count() > 0, "it lists the predicted movies"
        assert "dashed" in page.source_note.text(), page.source_note.text()
        assert page._build_work() is not None, "the run is accepted"
    finally:
        hub.close()
        app.processEvents()


def test_the_model_page_disables_labelling_for_a_gated_off_model(app, root: Path) -> None:
    from trackanno.gui.hub import Hub, ModelPage

    cfg, _ = build_case(root, "page-silent", reliable=False, labels=3)
    hub = Hub(str(cfg.path))
    hub.show()
    app.processEvents()
    try:
        page = hub.page(ModelPage)
        assert not page.predict_button.isEnabled()
        assert "reliable enough" in page.predict_note.text(), page.predict_note.text()
    finally:
        hub.close()
        app.processEvents()


def test_the_model_page_labels_every_clip(app, root: Path) -> None:
    """The button runs off the UI thread, reports a summary, and never writes the CSV."""
    from trackanno.gui.hub import Hub, ModelPage
    from trackanno.infer.predict import PREDICTIONS_FILE

    cfg, manifest = build_case(root, "page-live", labels=2)
    before = digest(cfg.paths.csv)
    hub = Hub(str(cfg.path))
    hub.show()
    app.processEvents()
    page = hub.page(ModelPage)
    assert page.predict_button.isEnabled(), "a reliable model enables the button"

    page.predict_button.click()
    assert pump(app, lambda: not page.tasks.busy), "the job runs and finishes"
    log = page.tasks.log.toPlainText()
    assert "bouts" in log, log[:200]

    target = cfg.paths.work_dir / PREDICTIONS_FILE
    assert target.exists()
    assert set(pd.read_parquet(target)["track_id"]) == set(manifest["track_id"]), (
        "every track was labelled")
    assert digest(cfg.paths.csv) == before, "the annotation CSV is untouched"

    assert "swimming" in log and "immotile" in log, log[-400:]
    assert "mean confidence" in log, log[-400:]
    assert "overlapping a clip a person labelled" in log, log[-400:]

    page.shutdown()
    hub.close()
    app.processEvents()


def test_a_movie_with_no_detector_embeddings_is_refused_not_filled_with_nan(
    root: Path,
) -> None:
    from sklearn.decomposition import PCA

    from trackanno.config import load_config
    from trackanno.infer.densepredict import build_dense_features
    from trackanno.measure import embeddings as emb
    from trackanno.project import write_config

    cfg, _ = build_case(root, "noembed")
    write_config(cfg.path, {"suggest.embeddings": [{"name": "det", "source": "detector"}]})
    cfg = load_config(cfg.path)
    work = cfg.paths.work_dir
    emb.save_bases(work / emb.BASES_FILE,
                   {"det": PCA(n_components=2).fit(np.arange(60.0).reshape(20, 3))})
    present = [d / emb.DETECTOR_FILE for d in work.iterdir() if d.is_dir()]
    assert not any(p.exists() for p in present), "the fixture supplies no embeddings"

    with pytest.raises(RuntimeError) as exc:
        build_dense_features(cfg)
    message = str(exc.value)
    assert emb.DETECTOR_FILE in message, message
    assert any(d.name in message for d in work.iterdir() if d.is_dir()), (
        f"the message does not name the movie to re-prepare: {message}")


def test_bouts_stop_at_the_first_and_last_detection(root: Path) -> None:
    """Windows reach past a track's ends; its bouts do not."""
    from trackanno.infer.predict import predict_all

    cfg, movie = planted_case(root, "clipped", stride=LENGTH)
    path = cfg.paths.work_dir / movie / "tracks.parquet"
    tracks = pd.read_parquet(path)
    tracks[(tracks["frame"] >= 7) & (tracks["frame"] <= 262)].to_parquet(path, index=False)
    bouts = predict_all(cfg)
    assert int(bouts["f_start"].min()) == 7
    assert int(bouts["f_end"].max()) == 263
    assert (bouts["n_frames"] == bouts["f_end"] - bouts["f_start"]).all()
