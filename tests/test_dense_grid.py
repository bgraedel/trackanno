"""Checks for the dense window grid and the plan laid over it.

The grid used to have one shape - stride equals length, every frame in exactly
one window - and one invariant that said so. ``predict.window_stride`` gives it
a second shape, and the two disagree about almost everything worth asserting: a
short tail window is emitted under a tiling stride and dropped under an
overlapping one, a frame belongs to one window or to ``length // stride`` of
them, and the count of "detected frames covered" is a sum in one mode and a
union in the other. So the claims here come in pairs, and three of them are
about the change itself rather than about either mode:

* the OPTIONALITY guarantee - ``stride is None`` must reproduce the grid and the
  plan exactly as they were before strides existed, short tail included, because
  every project on disk was built by that code and a silent change of window
  boundaries is a silent change of every feature value in the table;
* the index bug - the plan used to turn a frame number into a grid index by
  dividing it by ``length``, which is only an index while the grid tiles. Under
  overlap both bounds land short, and the failure is missing rows rather than an
  exception, so the check is that a track's windows actually span its lifetime;
* that :func:`trackanno.infer.densepredict.count_windows` counts the windows
  the plan lays down.

Nothing here reads pixels, features or a model: the grid is arithmetic, and the
project built below is a config plus one ``tracks.parquet``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from harness import check
from trackanno.config import load_config
from trackanno.infer.densepredict import (
    count_windows, plan_dense_windows, window_grid, window_length, window_stride,
)

MOVIE = "grid_movie"
LENGTH = 60
STRIDE = 20                      # divides LENGTH, so the interior count is exact
N_FRAMES = 250                   # a multiple of neither, so both tails are tested
WHOLE, MIDGRID, SHORT = 1, 2, 3  # the three tracks, by what they are there to show


def _tracks() -> pd.DataFrame:
    """One movie's detections: a track spanning it, one living mid-grid, one
    shorter than a single window.

    ``WHOLE`` carries a gap. A gap is not a hole in the coverage sense - no
    window is missing, the animal was - and every count below has to keep the
    two apart.
    """
    whole = [f for f in range(N_FRAMES) if not 100 <= f < 110]
    midgrid = list(range(63, 182))
    short = list(range(10, 31))
    frames = whole + midgrid + short
    ids = [WHOLE] * len(whole) + [MIDGRID] * len(midgrid) + [SHORT] * len(short)
    return pd.DataFrame({"track_id": ids, "frame": frames})


TRACKS = _tracks()


@pytest.fixture(scope="module")
def cfg(root):
    """A real config with an overlapping stride, plus that ``tracks.parquet``.

    Written with the real ``write_config``, so ``load_config`` is what accepts
    the stride.
    """
    from trackanno.project import create_project, write_config

    path = create_project(root / "grid")
    write_config(path, {
        "clips.frames": LENGTH,
        "clips.stride": LENGTH,
        "predict.window_stride": STRIDE,
        "predict.min_bout_frames": 5,
    })
    loaded = load_config(path)
    movie = loaded.movie_dir(MOVIE)
    movie.mkdir(parents=True, exist_ok=True)
    TRACKS.to_parquet(movie / "tracks.parquet", index=False)
    return loaded


def old_tiling(n_frames: int, length: int) -> list[tuple[int, int]]:
    """The grid exactly as it was written before ``stride`` existed."""
    return [(start, min(start + length, n_frames))
            for start in range(0, n_frames, length)]


def old_plan(frames: np.ndarray, grid: list, length: int) -> list[tuple[int, int]]:
    """The windows the replaced code chose: a frame divided by the length, used
    as an index into the grid. Correct only while ``grid[i]`` starts at
    ``i * length``."""
    return grid[int(frames.min()) // length: int(frames.max()) // length + 1]


def hits_per_frame(windows, n_frames: int) -> np.ndarray:
    """How many of ``windows`` contain each frame of ``[0, n_frames)``."""
    hits = np.zeros(n_frames, dtype=int)
    for f_start, f_end in windows:
        hits[f_start:f_end] += 1
    return hits


# --------------------------------------------------------------------------- #
# the guarantee that nothing moved
# --------------------------------------------------------------------------- #
def test_no_stride_is_the_grid_that_was_there_before() -> None:
    """Every project on disk was built by the tiling code, and its window
    boundaries are baked into every feature value in its dense table. So the
    default is not "a stride equal to the length" in spirit: it has to be the
    same list of tuples, in the same order, tail window and all."""
    for n_frames in (250, 240, 1, 59, 61, 600):
        grid = window_grid(n_frames, LENGTH)
        check(f"the tiling at n_frames={n_frames} is unchanged",
              grid == old_tiling(n_frames, LENGTH), f"{grid[:3]}...{grid[-2:]}")
        check(f"...and stride=None says so explicitly at n_frames={n_frames}",
              grid == window_grid(n_frames, LENGTH, None))
        check(f"...as does stride==length at n_frames={n_frames}",
              grid == window_grid(n_frames, LENGTH, LENGTH))

    grid = window_grid(N_FRAMES, LENGTH)
    check("the short tail window is still emitted",
          grid[-1] == (240, 250), str(grid[-1]))
    check("...and the tiling still covers every frame exactly once",
          bool((hits_per_frame(grid, N_FRAMES) == 1).all()))

    plan = plan_dense_windows(TRACKS, N_FRAMES, LENGTH)
    check("the plan without a stride is the plan with stride=length",
          plan == plan_dense_windows(TRACKS, N_FRAMES, LENGTH, LENGTH))
    for track_id, windows in plan.items():
        frames = TRACKS.loc[TRACKS["track_id"] == track_id, "frame"].to_numpy()
        check(f"track {track_id} gets the windows the old division chose",
              windows == old_plan(frames, grid, LENGTH), str(windows))


# --------------------------------------------------------------------------- #
# the overlapping grid
# --------------------------------------------------------------------------- #
def test_an_overlapping_grid_emits_only_full_windows() -> None:
    """Window length changes the feature VALUES - path_length,
    net_displacement, straightness, every *_std and present_frac all scale with
    the span - so a short window is scored by a model fitted at another length.
    Tiling produces one such row per track and always has; overlap would produce
    one per stride at every track's end, feeding the posterior exactly where
    tracks finish. They are dropped instead, and the last full window is moved
    flush with the end so dropping them costs no coverage."""
    grid = window_grid(N_FRAMES, LENGTH, STRIDE)
    spans = [e - s for s, e in grid]
    check("every overlapping window is full length",
          set(spans) == {LENGTH}, f"{sorted(set(spans))}")
    check("there are more of them than the tiling had",
          len(grid) > len(old_tiling(N_FRAMES, LENGTH)),
          f"{len(grid)} vs {len(old_tiling(N_FRAMES, LENGTH))}")
    check("the first window still starts at frame 0", grid[0][0] == 0)
    check("the last one ends flush with the movie",
          grid[-1][1] == N_FRAMES, str(grid[-1]))

    steps = [b[0] - a[0] for a, b in zip(grid, grid[1:])]
    check("the starts are one stride apart, except the flush one",
          set(steps[:-1]) == {STRIDE} and 0 < steps[-1] <= STRIDE, str(steps[-3:]))
    hits = hits_per_frame(grid, N_FRAMES)
    check("no frame is left without a window", int(hits.min()) >= 1,
          f"{int((hits == 0).sum())} frames uncovered")


def test_windows_per_frame_in_the_interior_is_length_over_stride() -> None:
    """What an overlapping stride buys: a transition is pinned to within one
    stride because that many windows speak for each frame, and each is offset
    from the last by a stride. The edges see fewer - there is no window starting
    before frame 0 - so the claim is about the interior."""
    for stride in (LENGTH, LENGTH // 2, LENGTH // 3, LENGTH // 4):
        hits = hits_per_frame(window_grid(N_FRAMES, LENGTH, stride), N_FRAMES)
        interior = hits[LENGTH:N_FRAMES - LENGTH]
        check(f"stride {stride}: every interior frame is in {LENGTH // stride} window(s)",
              bool((interior == LENGTH // stride).all()),
              f"{sorted(set(interior.tolist()))}")


# --------------------------------------------------------------------------- #
# the plan
# --------------------------------------------------------------------------- #
def test_the_plan_searches_the_grid_instead_of_dividing_by_the_length() -> None:
    """The plan turned a frame number into a grid index by dividing it by the
    window length. That is an index only while the grid tiles: under an
    overlapping stride the grid is length/stride times longer, both bounds name
    windows far short of the ones they meant to, and the track loses coverage at
    its tail. Nothing raises - the rows are simply not there - so the check is
    that the returned windows span the track's whole lifetime."""
    grid = window_grid(N_FRAMES, LENGTH, STRIDE)
    frames = TRACKS.loc[TRACKS["track_id"] == MIDGRID, "frame"].to_numpy()
    first, last = int(frames.min()), int(frames.max())
    windows = plan_dense_windows(TRACKS, N_FRAMES, LENGTH, STRIDE)[MIDGRID]

    check("the plan reaches back over the track's first detection",
          windows[0][0] <= first < windows[0][1], f"{windows[0]} vs frame {first}")
    check("...and past its last one", windows[-1][1] > last,
          f"{windows[-1]} vs frame {last}")
    hits = hits_per_frame(windows, N_FRAMES)
    check("every detected frame of the track is in a window",
          bool((hits[frames] >= 1).all()),
          f"{int((hits[frames] == 0).sum())} of {frames.size} missed")
    check("the windows are a contiguous run of the grid, gaps in the track and all",
          windows == grid[grid.index(windows[0]): grid.index(windows[-1]) + 1])
    check("no window is emitted beyond the track's lifetime",
          windows[0][1] > first and windows[-1][0] <= last)

    # The bug this replaces, stated so a return to it is a failure and not a
    # slow afternoon: dividing by the length picks windows 1..3 of a 11-window
    # grid and stops 60-odd frames before the track does.
    stale = old_plan(frames, grid, LENGTH)
    check("the division it replaced would have stopped short of the track's end",
          stale[-1][1] < last, f"the old code ended at {stale[-1][1]}, track ends {last}")


def test_a_track_shorter_than_one_window() -> None:
    """A track shorter than the window has no full window of its own. Under
    either stride it is carried by the movie's grid - the windows are cut over
    the movie, not over the track - so it gets a full-length window that happens
    to hold few of its frames, and n_present is what says so. Only a MOVIE
    shorter than one window has no full window to give, and there the short
    window is emitted anyway: a miscalibrated row can be filtered, a missing one
    is indistinguishable from a model that declined to label."""
    frames = TRACKS.loc[TRACKS["track_id"] == SHORT, "frame"].to_numpy()
    for stride in (None, STRIDE):
        windows = plan_dense_windows(TRACKS, N_FRAMES, LENGTH, stride)[SHORT]
        label = "tiling" if stride is None else f"stride {stride}"
        check(f"{label}: the short track still gets a window", len(windows) >= 1)
        check(f"{label}: ...a full-length one, cut over the movie",
              all(e - s == LENGTH for s, e in windows), str(windows))
        hits = hits_per_frame(windows, N_FRAMES)
        check(f"{label}: ...covering all {frames.size} of its detections",
              bool((hits[frames] >= 1).all()))

    short_movie = TRACKS[TRACKS["track_id"] == SHORT]
    for stride in (None, STRIDE):
        label = "tiling" if stride is None else f"stride {stride}"
        grid = window_grid(31, LENGTH, stride)
        check(f"{label}: a movie shorter than a window gets one short window",
              grid == [(0, 31)], str(grid))
        plan = plan_dense_windows(short_movie, 31, LENGTH, stride)
        check(f"{label}: ...and no track disappears from a movie that short",
              plan == {SHORT: [(0, 31)]}, str(plan))


# --------------------------------------------------------------------------- #
# the count
# --------------------------------------------------------------------------- #
def test_the_window_count_is_the_plan(cfg) -> None:
    """count_windows walks the same plan the feature build does."""
    check("the config resolves to the length and stride it was written with",
          (window_length(cfg), window_stride(cfg)) == (LENGTH, STRIDE),
          f"{window_length(cfg)}, {window_stride(cfg)}")
    plan = plan_dense_windows(TRACKS, N_FRAMES, LENGTH, STRIDE)
    n_windows, n_tracks = count_windows(cfg)
    check("the count is the plan's windows",
          n_windows == sum(len(w) for w in plan.values()), str(n_windows))
    check("...over every track", n_tracks == 3, str(n_tracks))
    hits = np.zeros(N_FRAMES, dtype=int)
    for windows in plan.values():
        for f_start, f_end in windows:
            hits[f_start:f_end] = 1
    check("no detected frame is left without a window",
          bool(hits[TRACKS["frame"].to_numpy()].all()))


# --------------------------------------------------------------------------- #
# the two tables have to agree about what a feature IS
# --------------------------------------------------------------------------- #
def test_the_embedding_delta_is_measured_on_the_frame_grid() -> None:
    """`emb_*_delta` is a mean over ADJACENT-FRAME steps; gaps are not bridged."""
    from trackanno.measure.clipfeatures import grid_delta
    from trackanno.measure.embeddings import summarise

    span = 10
    offsets = np.array([0, 1, 2, 7, 8])
    values = np.array([[0.0, 5.0], [1.0, 4.0], [3.0, 4.0], [10.0, 0.0], [14.0, -1.0]])
    # Adjacent pairs are (0,1), (1,2) and (7,8); (2,7) is a gap and contributes
    # nothing. Column 0: (1 + 2 + 4) / 3. Column 1: (1 + 0 + 1) / 3.
    want = np.array([7.0 / 3.0, 2.0 / 3.0])
    # What differencing the present rows alone would say, gap bridged: column 0
    # gains a step of 7 and column 1 one of 4.
    bridged = np.abs(np.diff(values, axis=0)).mean(axis=0)

    got = grid_delta(values, offsets, span)
    check("grid_delta averages the adjacent-frame steps and only those",
          np.allclose(got, want), f"{got} vs {want}")
    check("...which is not what bridging the gap would give",
          not np.allclose(bridged, want), f"bridged {bridged}")

    block = summarise("x", values, offsets, span)
    check("the embedding summary forms its delta the same way",
          np.isclose(block["emb_x_0_delta"], want[0])
          and np.isclose(block["emb_x_1_delta"], want[1]),
          f"{block['emb_x_0_delta']}, {block['emb_x_1_delta']} vs {want}")
    check("...and its mean and std are over the present rows",
          np.isclose(block["emb_x_0_mean"], values[:, 0].mean())
          and np.isclose(block["emb_x_0_std"], values[:, 0].std()),
          f"{block['emb_x_0_mean']}, {block['emb_x_0_std']}")

    # A missing vector is a NaN row: skipped, not bridged across.
    holed = np.vstack([values[:2], [[np.nan, np.nan]], values[2:]])
    holed_block = summarise("x", holed, np.array([0, 1, 5, 2, 7, 8]), span)
    check("a NaN row changes nothing",
          all(np.isclose(holed_block[k], block[k]) for k in block), str(holed_block))
    empty = summarise("x", np.full((3, 2), np.nan), np.array([0, 1, 2]), span)
    check("a window with no vector at all is NaN, not zero",
          all(np.isnan(v) for v in empty.values()), str(empty))

    # A window with no adjacent pair at all has measured no change, which is 0
    # rather than undefined: the quantity is a sum of steps and there were none.
    lonely = grid_delta(values[:2], np.array([0, 5]), span)
    check("no adjacent pair means zero change, not NaN",
          np.allclose(lonely, 0.0), str(lonely))


def test_border_dropped_rows_do_not_break_feature_alignment(root) -> None:
    """instance_features.npy is row-aligned with the UNFILTERED tracks table.

    `build_dense_features` drops the detections at the frame edge before it
    plans, so checking the feature count against the FILTERED table refuses a
    perfectly healthy project the moment `tracking.border_margin_px` costs it a
    row. The row LABELS survive the filter, so the lookups into the array were
    always right; only the count they were checked against was wrong.
    """
    import json

    from trackanno.infer.densepredict import build_dense_features
    from trackanno.project import create_project, write_config

    height, width, n_frames = 200, 200, 120
    path = create_project(root / "border")
    write_config(path, {
        "clips.frames": 30,
        "clips.stride": 30,
        "clips.min_clip_frames": 20,
        "tracking.border_margin_px": 20,
        "tracking.border_max_frac": 0.5,
        "suggest.embeddings": [{"name": "det", "source": "detector"}],
    })
    border_cfg = load_config(path)
    movie = border_cfg.movie_dir(MOVIE)
    movie.mkdir(parents=True, exist_ok=True)
    (movie / "cache.json").write_text(json.dumps({"movie": {
        "n_frames": n_frames, "fps": 100.0, "height": height, "width": width,
    }}), encoding="utf-8")

    # One track well inside the frame, one hugging the left edge in every frame.
    frames = np.arange(n_frames)
    boxes = {WHOLE: (80.0, 80.0, 120.0, 120.0), MIDGRID: (0.0, 0.0, 40.0, 40.0)}
    rows = []
    for track_id, (x1, y1, x2, y2) in boxes.items():
        rows.append(pd.DataFrame({
            "track_id": track_id, "frame": frames,
            "x1": x1 + frames * 0.1, "y1": y1, "x2": x2 + frames * 0.1, "y2": y2,
            "conf": 0.8,
        }))
    tracks = pd.concat(rows, ignore_index=True)
    tracks.to_parquet(movie / "tracks.parquet", index=False)
    # Aligned with the file on disk, as prepare.py writes it: one row per
    # detection, border track included.
    vectors = np.random.default_rng(0).normal(size=(len(tracks), 8)).astype(np.float32)
    np.save(movie / "instance_features.npy", vectors)
    from trackanno.measure import embeddings as emb

    border_cfg.paths.work_dir.mkdir(parents=True, exist_ok=True)
    emb.save_bases(border_cfg.paths.work_dir / emb.BASES_FILE,
                   emb.fit_bases(border_cfg, {"det": {MOVIE: vectors}}))

    out = build_dense_features(border_cfg)
    check("a project whose border rule drops rows still builds",
          len(out) > 0, f"{len(out)} rows")
    # This one is at the edge on every frame, so nothing of it is left to plan.
    check("...and a track that is nothing but border rows is absent from it",
          set(out["track_id"]) == {WHOLE}, str(sorted(set(out["track_id"]))))
    check("...and carries the embedding columns",
          "emb_det_0_mean" in out.columns
          and out.loc[out["skip_reason"] == "", "emb_det_0_mean"].notna().all())
