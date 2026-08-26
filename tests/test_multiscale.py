"""Scoring one model at several window lengths, and refusing to when it cannot.

Window length is a commitment about the timescale of behaviour, and a single one
is a guess: a transition shorter than the window is invisible inside it, and a
bout much longer than the window is cut into independent verdicts that can
disagree. ``predict.window_scales`` builds the dense grid at several lengths and
lets them all reach the same per-frame average - the trick
``predict.window_stride`` plays with a window's OFFSET, played with its
DURATION.

Three things have to be true for that to be worth anything, and they are what
this suite is about.

* **It is off, and off means unchanged.** Not "nearly", not "equivalent" -
  byte for byte the same predictions and the same dense table, because a
  changed window boundary moves every length-dependent feature value with it
  and a rewritten parquet is the only place that would show.
* **The scales are combined the way the code says they are.** Each scale
  carries the same weight, which is a decision and not an accident of the
  stride arithmetic, so it is checked against a closed form on constructed data
  rather than against a second implementation.
* **The incoherent combination is refused, loudly, twice.** A model fitted on
  columns whose values scale with the window length cannot be scored at several
  lengths; the config refuses the intent and a labelling run refuses the model
  actually on disk. The second is not redundant, because retraining is what
  moves a project between them.

The project, the planted movie and the "trained" model all come from
``test_predict``, deliberately: the boundary claims there and the ones here are
about the same grid, and a second copy of the fixture would be a second thing to
keep in step.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from harness import check
from test_predict import (
    BIG, LENGTH, PLANTED_FRAMES, SMALL, build_case, planted_case,
)

GROUP = "behaviour"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scaled_case(root: Path, name: str, scales, **settings):
    """A dense, decoding project scored at ``scales``, and its config reloaded.

    Every key goes in ONE ``write_config`` call because the loader refuses the
    combination rather than the keys: several scales need a decode and a
    length-invariant feature scope, so writing them one at a time would be
    refused halfway through with the file in a state nobody asked for.
    """
    from trackanno.config import load_config
    from trackanno.project import write_config

    cfg, _ = build_case(root, name, coverage="dense", trained_window=LENGTH)
    write_config(cfg.path, {
        "predict.window_frames": LENGTH,
        "predict.window_stride": "auto",
        "predict.window_scales": list(scales),
        "predict.decode": "min_bout",
        "predict.min_bout_frames": 20,
        "suggest.feature_scope": "length_invariant",
        **settings,
    })
    return load_config(cfg.path)


# --------------------------------------------------------------------------- #
# off, and off means unchanged
# --------------------------------------------------------------------------- #
def test_a_fresh_project_scores_at_exactly_one_scale(root: Path) -> None:
    """The shipped template, not a config a test wrote.

    Multi-scale scoring changes what a prediction is averaged from and costs a
    feature build per scale, so a project created today has to keep behaving the
    way it did before any of this existed - and a default that drifts does not
    announce itself the way a broken feature does.
    """
    from trackanno.config import load_config
    from trackanno.infer.densepredict import scale_grid, window_length, window_stride
    from trackanno.project import create_project

    cfg = load_config(create_project(Path(root) / "fresh"))

    check("a new project lists no extra scales",
          cfg.predict.window_scales == [], repr(cfg.predict.window_scales))
    check("...so it resolves to exactly one, the one it already had",
          cfg.predict.resolved_scales(cfg.clips)
          == [cfg.predict.resolved_window(cfg.clips)],
          repr(cfg.predict.resolved_scales(cfg.clips)))
    check("...and the grid it plans is the single grid it always planned",
          scale_grid(cfg) == [(window_length(cfg), window_stride(cfg))],
          repr(scale_grid(cfg)))
    check("the model is fitted on every column its feature groups select",
          cfg.suggest.feature_scope == "all", repr(cfg.suggest.feature_scope))


def test_writing_the_single_scale_out_by_hand_changes_the_file_not_at_all(
    root: Path,
) -> None:
    """``window_scales: []`` and ``window_scales: [window_frames]`` are one run.

    This is the guarantee the whole setting rests on. The empty list takes the
    single-grid path and the spelled-out one takes the multi-scale path with one
    entry in it, so if those two ever diverge the divergence is silent: the
    columns are the same, the row count is the same, and only the VALUES move -
    which is exactly the failure mode a window boundary shifting by a frame has.
    Hence bytes, not shapes.
    """
    from trackanno.config import load_config
    from trackanno.infer.densepredict import DENSE_FEATURES_FILE
    from trackanno.infer.predict import PREDICTIONS_FILE, predict_all
    from trackanno.project import write_config

    absent = _scaled_case(root, "scales-absent", [], **{
        # One scale needs no length-invariant scope; leaving it at the shipped
        # default is what makes this the DEFAULT run rather than a nearby one.
        "suggest.feature_scope": "all",
    })
    spelled = _scaled_case(root, "scales-spelled", [LENGTH], **{
        "suggest.feature_scope": "all",
    })

    check("the two configs really do resolve to the same one grid",
          absent.predict.resolved_scales(absent.clips)
          == spelled.predict.resolved_scales(spelled.clips) == [LENGTH],
          f"{absent.predict.resolved_scales(absent.clips)} vs "
          f"{spelled.predict.resolved_scales(spelled.clips)}")
    check("...while the file really does say different things",
          (absent.predict.window_scales, spelled.predict.window_scales)
          == ([], [LENGTH]))

    first = predict_all(absent)
    other = predict_all(spelled)
    check("spelling the single scale out gives the same rows",
          first.equals(other), f"{first.shape} vs {other.shape}")
    check("...in the same columns, in the same order",
          list(first.columns) == list(other.columns), str(list(other.columns)))
    check("...and the same predictions.parquet, byte for byte",
          digest(absent.paths.work_dir / PREDICTIONS_FILE)
          == digest(spelled.paths.work_dir / PREDICTIONS_FILE))
    check("...and the same dense feature table, byte for byte",
          digest(absent.paths.work_dir / DENSE_FEATURES_FILE)
          == digest(spelled.paths.work_dir / DENSE_FEATURES_FILE))

    # So the equality above is a claim about the inputs rather than about
    # parquet happening to be deterministic today.
    before = digest(absent.paths.work_dir / PREDICTIONS_FILE)
    predict_all(absent)
    check("a second identical run rewrites the same bytes",
          digest(absent.paths.work_dir / PREDICTIONS_FILE) == before)

    dense = pd.read_parquet(absent.paths.work_dir / DENSE_FEATURES_FILE)
    check("a single-scale dense table carries no scale suffix on any clip_id",
          not dense["clip_id"].astype(str).str.contains("__w").any(),
          str(dense["clip_id"].head(2).tolist()))
    check("...and every clip_id in it is still unique",
          dense["clip_id"].is_unique)

    # And the write path itself round-trips, so a GUI or a script that writes
    # the key back cannot turn [] into something else.
    write_config(absent.path, {"predict.window_scales": []})
    check("an empty list survives being written back out",
          load_config(absent.path).predict.window_scales == [],
          repr(load_config(absent.path).predict.window_scales))


# --------------------------------------------------------------------------- #
# how the scales are combined
# --------------------------------------------------------------------------- #
def test_each_scale_carries_the_same_weight_in_the_posterior(root: Path) -> None:
    """Three scales over one frame average to the plain mean of the three.

    The closed form, on constructed data: give every window of the 40-frame
    scale posterior 0.9, every 20-frame window 0.6 and every 10-frame window
    0.3, and a frame covered once by each must come out at exactly
    (0.9 + 0.6 + 0.3) / 3 = 0.6. Nothing here re-implements the average - it is
    ``bouts.frame_posterior``'s uniform mean - so what is being tested is the
    thing this branch actually decides: that the three scales reach that mean
    with equal counts.

    The contrast at the end is the alternative that was rejected. Score all
    three lengths on ONE stride and the covering counts go 4 : 2 : 1 in favour
    of the longest window, which then carries 4/7 of the answer on its own - so
    a short scale added to see a brief event would be outvoted by the window too
    blunt to see it. That failure is what the stride scaling exists to prevent,
    and it is quantified here rather than asserted.
    """
    from trackanno.infer.bouts import frame_posterior
    from trackanno.infer.densepredict import (
        plan_scaled_windows, scale_grid, window_grid,
    )

    n_frames = 120
    cfg = _scaled_case(root, "weights", [10, 20, 40],
                       **{"predict.window_frames": 40})
    grid = scale_grid(cfg)
    check("the base scale leads and the rest follow in ascending order",
          [scale for scale, _ in grid] == [40, 10, 20], repr(grid))
    check("at the tiling stride every scale tiles too",
          all(step == scale for scale, step in grid), repr(grid))
    check("...so every scale covers a frame the same number of times",
          len({scale / step for scale, step in grid}) == 1, repr(grid))

    tracks = pd.DataFrame({"frame": np.arange(n_frames), "track_id": 1})
    plan = plan_scaled_windows(tracks, n_frames, grid)[1]
    check("the plan really does hold windows of three different lengths",
          {end - start for start, end, _ in plan} == {10, 20, 40},
          str(sorted({end - start for start, end, _ in plan})))
    check("...and one window per scale per frame, 12 + 6 + 3 of them",
          len(plan) == n_frames // 10 + n_frames // 20 + n_frames // 40,
          f"{len(plan)} windows")

    # One posterior per scale, constant within it, so the mean has a closed form.
    by_scale = {40: 0.9, 20: 0.6, 10: 0.3}
    spans = [(start, end) for start, end, _ in plan]
    proba = np.array([[by_scale[scale], 1.0 - by_scale[scale]]
                      for _, _, scale in plan])
    posterior, n_covering = frame_posterior(spans, proba, 0, n_frames)

    expected = sum(by_scale.values()) / len(by_scale)
    check("every frame is covered by exactly one window of each scale",
          bool((n_covering == len(by_scale)).all()),
          str(sorted(set(n_covering.tolist()))))
    check(f"...so every frame's posterior is the plain mean of the three, {expected}",
          bool(np.allclose(posterior[:, 0], expected, rtol=0, atol=1e-12)),
          str(np.unique(np.round(posterior[:, 0], 15))))
    check("...and the complementary class is the mean of the complements",
          bool(np.allclose(posterior[:, 1], 1.0 - expected, rtol=0, atol=1e-12)))
    check("no scale is worth more than another, so no scale can outvote one",
          bool(np.allclose(posterior[:, 0].max(), posterior[:, 0].min(),
                           rtol=0, atol=1e-12)))

    # The rejected alternative, computed rather than described: the same three
    # lengths walked at the shortest scale's stride.
    common = [(scale, 10) for scale in (40, 20, 10)]
    flat_spans, flat_proba = [], []
    for scale, step in common:
        for start, end in window_grid(n_frames, scale, step):
            flat_spans.append((start, end))
            flat_proba.append([by_scale[scale], 1.0 - by_scale[scale]])
    flat, flat_covering = frame_posterior(
        flat_spans, np.array(flat_proba), 0, n_frames
    )
    middle = 60                     # an interior frame, away from either edge
    counts: dict[int, int] = {}
    for start, end in flat_spans:
        if start <= middle < end:
            counts[end - start] = counts.get(end - start, 0) + 1
    check("on one common stride the long window covers a frame four times over",
          counts == {40: 4, 20: 2, 10: 1}, str(counts))
    biased = sum(by_scale[scale] * n for scale, n in counts.items()) / sum(counts.values())
    check(f"...so that frame's posterior would be {biased:.4f}, not {expected}",
          abs(flat[middle, 0] - biased) < 1e-12
          and abs(flat[middle, 0] - expected) > 0.1,
          f"{flat[middle, 0]} vs {biased} / {expected}")
    check("...with the longest scale carrying 4 of the 7 votes on its own",
          int(flat_covering[middle]) == 7 and counts[40] == 4,
          f"{int(flat_covering[middle])} covering windows")


def test_the_stride_of_a_scale_follows_its_length(root: Path) -> None:
    """The weighting is set by one function, so it is checked against arithmetic.

    ``length / stride`` is how many windows of a scale reach a frame, and equal
    weight means that ratio is the same at every scale. At the tiling stride
    that is exact by construction; away from it the ratio is preserved wherever
    the multiplication comes out whole, and the docstring says so rather than
    claiming exactness it does not have.
    """
    from trackanno.infer.densepredict import scale_stride

    check("a tiling base grid tiles at every scale",
          [scale_stride(n, 50, 50) for n in (25, 50, 100)] == [25, 50, 100])
    check("...whatever the scale, including one that does not divide the base",
          scale_stride(37, 50, 50) == 37)
    check("an overlapping base grid keeps its overlap ratio at every scale",
          [scale_stride(n, 40, 20) for n in (10, 20, 40, 80)] == [5, 10, 20, 40])
    check("...so length / stride is the same number at each of them",
          {n / scale_stride(n, 40, 20) for n in (10, 20, 40, 80)} == {2.0})
    check("a stride never rounds down to zero, however short the scale",
          scale_stride(1, 400, 1) == 1)


# --------------------------------------------------------------------------- #
# what the bouts look like
# --------------------------------------------------------------------------- #
def test_scales_of_different_lengths_still_cut_disjoint_gapless_bouts(
    root: Path,
) -> None:
    """The renderer's contract does not care how the posterior was made.

    It paints ``range(f_start, f_end)`` per track and lets whichever row came
    last win, so bouts have to be sorted, disjoint and gapless over the frames
    the windows covered - exactly as they are on one scale. Windows of several
    LENGTHS are the case where that could quietly stop being true: they start
    and end in places no single grid would, and a hole or an overlap between two
    bouts would show up as a mis-coloured frame in an exported movie rather than
    as an error.
    """
    from trackanno.infer.densepredict import DENSE_FEATURES_FILE
    from trackanno.infer.predict import predict_all

    cfg, _ = planted_case(root, "planted-scales", LENGTH, **{
        "predict.window_scales": [25, LENGTH, 100],
        "suggest.feature_scope": "length_invariant",
    })
    bouts = predict_all(cfg)
    dense = pd.read_parquet(cfg.paths.work_dir / DENSE_FEATURES_FILE)
    spans = dense["f_end"] - dense["f_start"]

    check("the dense table really was built at three lengths",
          set(spans.unique()) == {25, LENGTH, 100},
          str(sorted(spans.unique())))
    check("...one grid's worth of each over the planted track",
          [int((spans == n).sum()) for n in (25, LENGTH, 100)]
          == [PLANTED_FRAMES // 25, PLANTED_FRAMES // LENGTH, PLANTED_FRAMES // 100],
          str([int((spans == n).sum()) for n in (25, 50, 100)]))
    check("every dense row still has an id of its own",
          dense["clip_id"].is_unique)
    check("...with the base scale's ids left as the annotation grid's are",
          int((~dense["clip_id"].astype(str).str.contains("__w")).sum())
          == PLANTED_FRAMES // LENGTH,
          str(dense["clip_id"].head(3).tolist()))
    check("...and every other scale's naming its own length",
          set(dense.loc[spans == 100, "clip_id"].astype(str).str[-7:]) == {"__w0100"},
          str(dense.loc[spans == 100, "clip_id"].head(2).tolist()))

    check("a row of the answer is a bout, not a window",
          "bout_id" in bouts.columns and "clip_id" not in bouts.columns,
          str(list(bouts.columns)))
    for track_id, group in bouts.groupby("track_id"):
        edges = group.sort_values("f_start")[["f_start", "f_end"]].to_numpy()
        covered = dense[dense["track_id"] == track_id]
        check(f"track {track_id}: the bouts are sorted and disjoint",
              bool((edges[:, 0] < edges[:, 1]).all())
              and list(edges[1:, 0]) == list(edges[:-1, 1]), str(edges))
        check(f"track {track_id}: ...and gapless over every frame any scale reached",
              (int(edges[0, 0]), int(edges[-1, 1]))
              == (int(covered["f_start"].min()), int(covered["f_end"].max())),
              f"{edges[0, 0]}..{edges[-1, 1]} vs "
              f"{covered['f_start'].min()}..{covered['f_end'].max()}")
    check("no frame of the track is spoken for twice or not at all",
          int(bouts["n_frames"].sum()) == PLANTED_FRAMES,
          f"{int(bouts['n_frames'].sum())} of {PLANTED_FRAMES}")
    check("a bout at the interior rests on more windows than one at the edge",
          bouts["n_covering_mean"].max() >= bouts["n_covering_mean"].min() > 0,
          str(bouts["n_covering_mean"].tolist()))


def test_an_event_shorter_than_the_window_is_invisible_until_a_shorter_scale_sees_it(
    root: Path,
) -> None:
    """The claim the whole setting rests on, on constructed data.

    The animal stops for 30 frames inside a 50-frame tiling grid. No window of
    that grid is ever more than 40% stopped, so the majority of every window is
    swimming and a model reading the fraction back EXACTLY - ``AreaModel``, which
    inverts the affine map from box area to time spent moving - reports swimming
    everywhere. The event is not missed because the model is weak; it is missed
    because a 50-frame window cannot represent it. No stride helps: a shorter
    stride moves where a boundary is reported, not which behaviours exist.

    Adding a 25-frame scale is what makes it representable, and the reported
    stop is the one the 25-frame grid can express. Adding a 10-frame scale as
    well pins its start on the frame.

    The negative result is asserted too, because it is what stops this being
    read as a boundary-localisation feature: the planted transition in
    ``test_predict`` is reported at frame 150 at one scale and at 150 with three,
    since every scale's tiling grid still cuts at its own window edges.
    ``predict.window_stride`` is the knob for WHERE; this is the knob for WHAT.
    """
    from trackanno.config import load_config
    from trackanno.infer.predict import predict_all

    stop_lo, stop_hi = 130, 160                 # 30 frames, well inside one window

    def plant(cfg) -> None:
        work = cfg.paths.work_dir
        movie = next(p for p in sorted(work.iterdir())
                     if p.is_dir() and (p / "tracks.parquet").exists())
        frames = np.arange(PLANTED_FRAMES)
        side = np.where((frames >= stop_lo) & (frames < stop_hi), SMALL, BIG)
        left = 20.0 + 0.5 * frames              # drifting, so speed_* exist too
        pd.DataFrame({
            "frame": frames, "track_id": 1,
            "x1": left, "y1": 40.0, "x2": left + side, "y2": 40.0 + side,
            "conf": 0.9,
        }).to_parquet(movie / "tracks.parquet", index=False)

    def stops(name: str, **updates) -> list[tuple[int, int]]:
        cfg, _ = planted_case(root, name, LENGTH, **updates)
        plant(cfg)
        out = predict_all(load_config(cfg.path))
        found = out[out[f"{GROUP}_pred"] == "immotile"][["f_start", "f_end"]]
        return [(int(a), int(b)) for a, b in found.to_numpy()]

    one = stops("short-one")
    check("a 30-frame stop is invisible to a 50-frame grid, however it is decoded",
          one == [], str(one))

    two = stops("short-two", **{
        "predict.window_scales": [25, LENGTH],
        "suggest.feature_scope": "length_invariant",
    })
    check("adding a 25-frame scale finds it", len(two) == 1, str(two))
    check("...and places it on the 25-frame grid that could express it",
          two == [(125, 150)], str(two))

    three = stops("short-three", **{
        "predict.window_scales": [25, LENGTH, 100],
        "suggest.feature_scope": "length_invariant",
    })
    check("a 100-frame scale on top does not drown the 25-frame one out",
          three == two, f"{three} vs {two}")

    four = stops("short-four", **{
        "predict.window_scales": [10, 25, LENGTH, 100],
        "suggest.feature_scope": "length_invariant",
        "predict.min_bout_frames": 10,
    })
    check("a 10-frame scale pins the start of the stop on the frame it happened",
          four and four[0][0] == stop_lo, str(four))


# --------------------------------------------------------------------------- #
# refusing the incoherent combination
# --------------------------------------------------------------------------- #
def test_the_config_refuses_scales_it_cannot_make_sense_of(root: Path) -> None:
    """Four refusals, all at load time, all about a pairing rather than a value.

    None of these is catchable later. A grid that leaves out its own base
    length, a scale listed twice, several scales with no decode, several scales
    against a model that will be fitted on length-dependent columns - each
    produces a perfectly well-formed parquet full of numbers that are wrong for
    a reason nothing in the file records.
    """
    from trackanno.config import ConfigError, load_config
    from trackanno.project import write_config

    cfg, _ = build_case(root, "refusals", coverage="dense", trained_window=LENGTH)
    base = {
        "predict.window_frames": LENGTH,
        "predict.decode": "min_bout",
        "predict.min_bout_frames": 20,
        "suggest.feature_scope": "length_invariant",
    }

    def refused(updates: dict) -> str:
        """The refusal `write_config` re-raises from its trial load, or "".

        Through `write_config` rather than `load_config` because that is the
        path the GUI's autosave takes, and the guarantee that matters is that a
        config which would not load is never written: an empty string here means
        the file on disk now says something this suite thinks is incoherent.
        """
        try:
            write_config(cfg.path, {**base, **updates})
        except ConfigError as exc:
            return str(exc)
        return ""

    message = refused({"predict.window_scales": [25, 100]})
    check("a scale list that leaves out predict.window_frames is refused",
          message != "")
    check("...and the message names the length that is missing and why it matters",
          str(LENGTH) in message and "window_frames" in message, message)

    message = refused({"predict.window_scales": [25, LENGTH, 25]})
    check("a scale listed twice is refused rather than deduplicated",
          "twice" in message, message)
    check("...because a repeat is a weighting, and the message says so",
          "weight" in message and "outvote" in message, message)

    message = refused({"predict.window_scales": [25, LENGTH],
                                    "predict.decode": "none"})
    check("several scales with no decode are refused",
          "decode" in message, message)
    check("...on the grounds that the exported colour would follow row order",
          "row order" in message, message)

    message = refused({"predict.window_scales": [25, LENGTH],
                                "suggest.feature_scope": "all"})
    check("several scales against a length-dependent feature scope are refused",
          "feature_scope" in message, message)
    check("...and the message names a column whose value would be on two scales",
          "path_length" in message, message)
    check("...and says the model on disk is checked separately",
          "already on disk" in message, message)

    # The whole point of the refusals is that the legal combination loads.
    write_config(cfg.path, {**base, "predict.window_scales": [25, LENGTH, 100]})
    ok = load_config(cfg.path)
    check("the coherent combination loads and resolves base-first",
          ok.predict.resolved_scales(ok.clips) == [LENGTH, 25, 100],
          repr(ok.predict.resolved_scales(ok.clips)))
    check("...and an unknown feature scope is refused on its own",
          bool(refused({"suggest.feature_scope": "invariant"})))


def test_a_model_fitted_on_length_dependent_columns_is_refused(root: Path) -> None:
    """The second refusal, and the one that is about a FACT rather than a plan.

    ``load_config`` can only read what the project intends to fit next. A model
    trained before the scope was changed - or trained with the scope and then
    retrained without it - loads, matches on feature version and column names,
    and would be scored at three window lengths at unchanged confidence. So the
    labelling run asks the payload what it was actually fitted on, and the
    message has to NAME the offending columns: "your features scale with the
    window" is not something anyone can act on.
    """
    import joblib

    from trackanno.measure.clipfeatures import CLIP_FEATURES_FILE, FEATURE_VERSION
    from trackanno.infer.predict import predict_all
    from trackanno.learn.suggest import MODEL_FILE, Suggester

    cfg = _scaled_case(root, "dependent", [25, LENGTH])
    work = cfg.paths.work_dir

    # A model exactly like the fixture's, plus one column that scales with the
    # window - which is the only difference between a legal multi-scale model
    # and an illegal one.
    features = pd.read_parquet(work / CLIP_FEATURES_FILE)
    features["path_length"] = np.linspace(10.0, 20.0, len(features))
    features["_feature_version"] = FEATURE_VERSION
    features.to_parquet(work / CLIP_FEATURES_FILE, index=False)
    payload = joblib.load(work / MODEL_FILE)
    payload["columns"] = ["speed_mean", "area_mean", "path_length"]
    joblib.dump(payload, work / MODEL_FILE)

    suggester = Suggester(cfg)
    check("the model loads without complaint, which is the problem",
          suggester.ready, suggester.problem)
    check("...and passes the window-length check, because its base scale matches",
          suggester.window_problem(LENGTH) == "", suggester.window_problem(LENGTH))

    message = suggester.scale_problem([LENGTH, 25])
    check("but two scales are a problem for it", message != "")
    check("...and the message names the offending column",
          "path_length" in message, message)
    check("...and does not accuse the columns that are fine",
          "area_mean" not in message and "speed_mean" not in message, message)
    check("...and says what to do about it",
          "length_invariant" in message and "retrain" in message.lower(), message)
    check("one scale is never a problem, whatever the columns",
          suggester.scale_problem([LENGTH]) == ""
          and suggester.scale_problem([LENGTH, LENGTH]) == "",
          suggester.scale_problem([LENGTH]))

    try:
        predict_all(cfg)
        check("a labelling run refuses it", False, "it labelled the project anyway")
    except RuntimeError as exc:
        check("a labelling run refuses it", "path_length" in str(exc), str(exc))
        check("...naming the scales it was asked for",
              "25" in str(exc) and str(LENGTH) in str(exc), str(exc))

    # The same project with the same config, once the model is fitted on the
    # columns it should have been: the refusal has to be about the model.
    payload["columns"] = ["speed_mean", "area_mean"]
    joblib.dump(payload, work / MODEL_FILE)
    out = predict_all(cfg)
    check("dropping the length-dependent column is what unblocks the run",
          len(out) > 0 and "bout_id" in out.columns, str(list(out.columns)))


# --------------------------------------------------------------------------- #
# the subset itself
# --------------------------------------------------------------------------- #
def test_the_length_invariant_subset_selects_what_it_claims(root: Path) -> None:
    """The predicate, against the real feature schema, column by column.

    It is an ALLOWLIST, and the reason is the last claim here: a column nobody
    has classified must come out length-DEPENDENT. Under a denylist a feature
    added to the schema next year would match no "scales with the window"
    pattern, be scored at three lengths, and nothing would raise - which is the
    exact failure this whole mechanism exists to prevent.
    """
    from trackanno.measure.clipfeatures import feature_schema
    from trackanno.learn.suggest import (
        is_length_invariant, length_dependent_columns, length_invariant_columns,
        scoped_columns,
    )

    schema = list(feature_schema())
    invariant = set(length_invariant_columns(schema))
    dependent = set(length_dependent_columns(schema))

    check("the two halves partition the schema and share nothing",
          invariant | dependent == set(schema) and not (invariant & dependent),
          f"{len(invariant)} + {len(dependent)} of {len(schema)}")

    # Measured on a window pinned to spectral.window, so the requested length
    # does not enter their definition at all.
    spectral = [c for c in schema if c.startswith(("beat_", "wave_"))]
    check(f"all {len(spectral)} beat_* and wave_* columns are invariant",
          set(spectral) <= invariant,
          str(sorted(set(spectral) - invariant)))

    for column in ("speed_mean", "speed_median", "area_mean", "immotile_frac",
                   "conf_mean", "body_len_mean", "flagellum_complete_frac",
                   "shape_elongation", "body_speed_median"):
        check(f"{column} is invariant: averaging more frames does not scale a mean",
              column in invariant and is_length_invariant(column))

    for column, why in (
        ("path_length", "extensive - twice the window, twice the path"),
        ("net_displacement", "extensive"),
        ("radius_gyration", "extensive"),
        ("straightness", "a ratio of two extensive quantities"),
        ("body_displacement_per_len", "derived from an extensive quantity"),
        ("speed_std", "a dispersion grows with the span it is measured over"),
        ("area_std", "a dispersion"),
        ("flagellum_len_cv", "a dispersion in disguise"),
        ("speed_max", "an order statistic rises with the number of samples"),
        ("speed_p90", "an order statistic"),
        ("turn_lag_frames", "the lag is min(n_frames // 4, 25), so it IS the length"),
        ("turn_abs_mean", "a mean of turns measured over that lag"),
        ("turn_n_valid", "a count"),
        ("present_frac", "its denominator is the window"),
    ):
        check(f"{column} is length-dependent: {why}",
              column in dependent and not is_length_invariant(column))

    check("cnn*_mean is invariant, being the mean of a window's embeddings",
          is_length_invariant("cnn0_mean") and is_length_invariant("cnn31_mean"))
    check("...while cnn*_std is not, a dispersion being a dispersion",
          not is_length_invariant("cnn0_std"))
    check("a column nobody has classified is treated as length-dependent",
          not is_length_invariant("some_feature_added_next_year"),
          "an allowlist is what makes the unclassified case the safe one")

    check("scoped_columns('all') is the identity",
          scoped_columns(schema, "all") == schema)
    check("scoped_columns('length_invariant') keeps order and drops nothing else",
          scoped_columns(schema, "length_invariant")
          == [c for c in schema if c in invariant])
    check("an unknown scope raises rather than falling through to 'all'",
          _raises(lambda: scoped_columns(schema, "typo"), ValueError))


def _raises(call, kind) -> bool:
    try:
        call()
    except kind:
        return True
    return False


def test_fitting_on_the_subset_saves_a_model_that_can_be_multi_scaled(
    root: Path,
) -> None:
    """The named selection has to reach the saved payload, not just the fit.

    ``Suggester.scale_problem`` reads the model file's ``columns`` and nothing
    else, so a run that narrowed the inputs somewhere deeper - per group, say -
    would fit on the subset and then record a wider input space, and the check
    would refuse a model that was actually fine. The two have to be the same
    list.
    """
    from trackanno.config import load_config
    from trackanno.project import write_config
    from trackanno.annotate.store import AnnotationStore
    from trackanno.learn.suggest import (
        MODEL_FILE, Suggester, length_dependent_columns, train,
    )

    scratch = Path(tempfile.mkdtemp(dir=root, prefix="subset-"))
    fixture_root = Path(tempfile.mkdtemp(dir=root, prefix="subset-fixtures-"))
    from fixtures import make_project

    fixture = make_project(fixture_root)
    project = scratch / "project"
    project.mkdir(parents=True, exist_ok=True)
    config_path = project / "config.yaml"
    shutil.copy(fixture, config_path)
    write_config(config_path, {
        "paths.work_dir": str(project / "work").replace("\\", "/"),
        "paths.csv": str(project / "annotations.csv").replace("\\", "/"),
        "suggest.feature_scope": "length_invariant",
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
        for fname in ("clips.parquet", "tracks.parquet", "cache.json"):
            if (movie_dir / fname).exists():
                shutil.copy(movie_dir / fname, work / movie_dir.name / fname)
        manifests.append(pd.read_parquet(movie_dir / "clips.parquet"))
    manifest = pd.concat(manifests, ignore_index=True)

    store = AnnotationStore(cfg.paths.csv, cfg)
    for i, row in enumerate(manifest.to_dict("records")):
        store.record(row, {GROUP: ("swimming", "immotile")[i % 2]})

    train(cfg, rebuild_features=True, verbose=False)

    import joblib

    columns = list(joblib.load(work / MODEL_FILE)["columns"])
    check("the fit kept some columns", bool(columns))
    check("...and every one of them survives a change of window length",
          not length_dependent_columns(columns),
          str(length_dependent_columns(columns)))
    check("...including the ones a length-invariant model is mostly made of",
          {"speed_mean", "area_mean"} <= set(columns),
          str(sorted(columns)[:8]))

    suggester = Suggester(cfg)
    check("a model fitted this way may be scored at several scales",
          suggester.scale_problem([50, 25, 100]) == "",
          suggester.scale_problem([50, 25, 100]))

    # And the same project fitted without the scope may not, which is what says
    # the subset did the work rather than the fixture happening to be narrow.
    write_config(config_path, {"suggest.feature_scope": "all"})
    train(load_config(config_path), rebuild_features=False, verbose=False)
    wide = list(joblib.load(work / MODEL_FILE)["columns"])
    check("the unscoped fit really is wider",
          len(wide) > len(columns), f"{len(wide)} vs {len(columns)}")
    check("...and carries columns that scale with the window",
          bool(length_dependent_columns(wide)))
    check("...so that model is refused at several scales",
          Suggester(load_config(config_path)).scale_problem([50, 25]) != "")
