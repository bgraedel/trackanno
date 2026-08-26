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
* that :func:`trackanno.infer.densepredict.coverage` has a caller at all. It is the
  function that would have caught the index bug and it had been dead code
  through two changes of the grid.

Nothing here reads pixels, features or a model: the grid is arithmetic, and the
project built below is a config plus one ``tracks.parquet``, which is the least
``coverage`` can be given.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from harness import check
from trackanno.config import load_config
from trackanno.infer.densepredict import (
    coverage, plan_dense_windows, window_grid, window_length, window_stride,
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
    the stride: an overlapping one is only legal beside a decode, and a fixture
    that bypassed the loader could assert against a config no user can have.
    """
    from trackanno.project import create_project, write_config

    path = create_project(root / "grid")
    write_config(path, {
        "predict.window_frames": LENGTH,
        "predict.window_stride": STRIDE,
        "predict.decode": "min_bout",
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


def dense_like(plan: dict) -> pd.DataFrame:
    """The columns ``coverage`` reads out of a dense feature table."""
    rows = [{"movie_id": MOVIE, "track_id": int(t), "f_start": int(s), "f_end": int(e)}
            for t, windows in plan.items() for s, e in windows]
    return pd.DataFrame(rows)


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
# the check that had no caller
# --------------------------------------------------------------------------- #
def test_coverage_reports_what_is_true_in_both_modes(cfg) -> None:
    """coverage() is the function that would have caught the index bug, and it
    was in __all__ and called from nowhere while the grid changed under it. Its
    claim was "exactly once", which an overlapping stride makes false by design;
    what stays true in both modes is that no detected frame may be left
    uncovered, and that the number of windows speaking for a frame is 1 while
    tiling and length/stride under overlap."""
    check("the config resolves to the length and stride it was written with",
          (window_length(cfg), window_stride(cfg)) == (LENGTH, STRIDE),
          f"{window_length(cfg)}, {window_stride(cfg)}")

    tiled = coverage(cfg, dense_like(plan_dense_windows(TRACKS, N_FRAMES, LENGTH)))
    row = tiled.iloc[0]
    check("tiling leaves no detected frame uncovered", int(row["uncovered"]) == 0,
          f"{int(row['uncovered'])} of {int(row['detected_frames'])}")
    check("...and speaks for each of them exactly once",
          int(row["windows_per_frame_max"]) == 1
          and float(row["windows_per_frame_mean"]) == 1.0,
          f"max {row['windows_per_frame_max']}, mean {row['windows_per_frame_mean']}")
    check("...over every detection the movie has",
          int(row["detected_frames"]) == len(TRACKS), str(int(row["detected_frames"])))

    over = coverage(cfg, dense_like(plan_dense_windows(TRACKS, N_FRAMES, LENGTH, STRIDE)))
    row = over.iloc[0]
    check("an overlapping stride leaves no detected frame uncovered either",
          int(row["uncovered"]) == 0,
          f"{int(row['uncovered'])} of {int(row['detected_frames'])}")
    # length/stride in the interior; one more at the very end, where the window
    # moved flush with the last frame overlaps its neighbour by more than a
    # stride. That is the price of dropping the short tail, and it is bounded.
    check("...and no frame is in more windows than the grid can put there",
          int(row["windows_per_frame_max"]) == LENGTH // STRIDE + 1,
          str(int(row["windows_per_frame_max"])))
    check("...with most frames in that many, not near one",
          float(row["windows_per_frame_mean"]) > LENGTH // STRIDE - 1,
          str(float(row["windows_per_frame_mean"])))
    check("...so the overlap it reports is real and is what the decode has to resolve",
          int(row["covered_once"]) < int(row["detected_frames"]),
          f"{int(row['covered_once'])} of {int(row['detected_frames'])}")
    check("both modes agree on how many detections there are",
          int(row["detected_frames"]) == int(tiled.iloc[0]["detected_frames"]))
