"""The dense window grid, the plan laid over it, and the window count."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
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
    """A track spanning the movie (with a gap), one mid-grid, one shorter than a window."""
    whole = [f for f in range(N_FRAMES) if not 100 <= f < 110]
    midgrid = list(range(63, 182))
    short = list(range(10, 31))
    frames = whole + midgrid + short
    ids = [WHOLE] * len(whole) + [MIDGRID] * len(midgrid) + [SHORT] * len(short)
    return pd.DataFrame({"track_id": ids, "frame": frames})


TRACKS = _tracks()


@pytest.fixture(scope="module")
def cfg(root):
    """A real config with an overlapping stride, plus the tracks table."""
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


def reference_tiling(n_frames: int, length: int) -> list[tuple[int, int]]:
    """Back-to-back windows from frame 0, short tail included."""
    return [(start, min(start + length, n_frames))
            for start in range(0, n_frames, length)]


def divided_plan(frames: np.ndarray, grid: list, length: int) -> list[tuple[int, int]]:
    """Windows chosen by frame // length, which is a grid index only while the grid tiles."""
    return grid[int(frames.min()) // length: int(frames.max()) // length + 1]


def hits_per_frame(windows, n_frames: int) -> np.ndarray:
    """How many of ``windows`` contain each frame of ``[0, n_frames)``."""
    hits = np.zeros(n_frames, dtype=int)
    for f_start, f_end in windows:
        hits[f_start:f_end] += 1
    return hits


# --------------------------------------------------------------------------- #
# the tiling grid
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_frames", [250, 240, 1, 59, 61, 600])
def test_no_stride_is_the_tiling_grid(n_frames: int) -> None:
    grid = window_grid(n_frames, LENGTH)
    assert grid == reference_tiling(n_frames, LENGTH), f"{grid[:3]}...{grid[-2:]}"
    assert grid == window_grid(n_frames, LENGTH, None)
    assert grid == window_grid(n_frames, LENGTH, LENGTH)


def test_the_tiling_keeps_its_short_tail_and_covers_each_frame_once() -> None:
    grid = window_grid(N_FRAMES, LENGTH)
    assert grid[-1] == (240, 250), grid[-1]
    assert (hits_per_frame(grid, N_FRAMES) == 1).all()


def test_the_plan_without_a_stride_is_the_divided_plan() -> None:
    grid = window_grid(N_FRAMES, LENGTH)
    plan = plan_dense_windows(TRACKS, N_FRAMES, LENGTH)
    assert plan == plan_dense_windows(TRACKS, N_FRAMES, LENGTH, LENGTH)
    for track_id, windows in plan.items():
        frames = TRACKS.loc[TRACKS["track_id"] == track_id, "frame"].to_numpy()
        assert windows == divided_plan(frames, grid, LENGTH), f"track {track_id}: {windows}"


# --------------------------------------------------------------------------- #
# the overlapping grid
# --------------------------------------------------------------------------- #
def test_an_overlapping_grid_emits_only_full_windows() -> None:
    """Short windows are dropped and the last full one sits flush with the end."""
    grid = window_grid(N_FRAMES, LENGTH, STRIDE)
    spans = [e - s for s, e in grid]
    assert set(spans) == {LENGTH}, sorted(set(spans))
    assert len(grid) > len(reference_tiling(N_FRAMES, LENGTH))
    assert grid[0][0] == 0
    assert grid[-1][1] == N_FRAMES, grid[-1]

    steps = [b[0] - a[0] for a, b in zip(grid, grid[1:])]
    assert set(steps[:-1]) == {STRIDE} and 0 < steps[-1] <= STRIDE, steps[-3:]
    hits = hits_per_frame(grid, N_FRAMES)
    assert int(hits.min()) >= 1, f"{int((hits == 0).sum())} frames uncovered"


@pytest.mark.parametrize("stride", [LENGTH, LENGTH // 2, LENGTH // 3, LENGTH // 4])
def test_windows_per_frame_in_the_interior_is_length_over_stride(stride: int) -> None:
    hits = hits_per_frame(window_grid(N_FRAMES, LENGTH, stride), N_FRAMES)
    interior = hits[LENGTH:N_FRAMES - LENGTH]
    assert (interior == LENGTH // stride).all(), sorted(set(interior.tolist()))


# --------------------------------------------------------------------------- #
# the plan
# --------------------------------------------------------------------------- #
def test_the_plan_spans_the_track_under_an_overlapping_stride() -> None:
    """The plan searches the grid rather than dividing by the length."""
    grid = window_grid(N_FRAMES, LENGTH, STRIDE)
    frames = TRACKS.loc[TRACKS["track_id"] == MIDGRID, "frame"].to_numpy()
    first, last = int(frames.min()), int(frames.max())
    windows = plan_dense_windows(TRACKS, N_FRAMES, LENGTH, STRIDE)[MIDGRID]

    assert windows[0][0] <= first < windows[0][1], f"{windows[0]} vs frame {first}"
    assert windows[-1][1] > last, f"{windows[-1]} vs frame {last}"
    hits = hits_per_frame(windows, N_FRAMES)
    assert (hits[frames] >= 1).all(), (
        f"{int((hits[frames] == 0).sum())} of {frames.size} detections missed")
    assert windows == grid[grid.index(windows[0]): grid.index(windows[-1]) + 1], (
        "the windows are a contiguous run of the grid")
    assert windows[0][1] > first and windows[-1][0] <= last, (
        "a window is emitted beyond the track's lifetime")

    # Dividing by the length would stop short here, so this case exercises the search.
    assert divided_plan(frames, grid, LENGTH)[-1][1] < last


@pytest.mark.parametrize("stride", [None, STRIDE])
def test_a_track_shorter_than_one_window_gets_a_full_window(stride) -> None:
    """Windows are cut over the movie, so a short track rides a full-length one."""
    frames = TRACKS.loc[TRACKS["track_id"] == SHORT, "frame"].to_numpy()
    windows = plan_dense_windows(TRACKS, N_FRAMES, LENGTH, stride)[SHORT]
    assert len(windows) >= 1
    assert all(e - s == LENGTH for s, e in windows), windows
    hits = hits_per_frame(windows, N_FRAMES)
    assert (hits[frames] >= 1).all(), f"not all {frames.size} detections covered"


@pytest.mark.parametrize("stride", [None, STRIDE])
def test_a_movie_shorter_than_one_window_gets_one_short_window(stride) -> None:
    short_movie = TRACKS[TRACKS["track_id"] == SHORT]
    assert window_grid(31, LENGTH, stride) == [(0, 31)]
    assert plan_dense_windows(short_movie, 31, LENGTH, stride) == {SHORT: [(0, 31)]}


# --------------------------------------------------------------------------- #
# the count
# --------------------------------------------------------------------------- #
def test_the_window_count_is_the_plan(cfg) -> None:
    """count_windows walks the same plan the feature build does."""
    assert (window_length(cfg), window_stride(cfg)) == (LENGTH, STRIDE)
    plan = plan_dense_windows(TRACKS, N_FRAMES, LENGTH, STRIDE)
    n_windows, n_tracks = count_windows(cfg)
    assert n_windows == sum(len(w) for w in plan.values()), n_windows
    assert n_tracks == 3, n_tracks
    hits = np.zeros(N_FRAMES, dtype=int)
    for windows in plan.values():
        for f_start, f_end in windows:
            hits[f_start:f_end] = 1
    assert hits[TRACKS["frame"].to_numpy()].all(), "a detected frame has no window"


# --------------------------------------------------------------------------- #
# embedding summaries
# --------------------------------------------------------------------------- #
SPAN = 10
OFFSETS = np.array([0, 1, 2, 7, 8])
VALUES = np.array([[0.0, 5.0], [1.0, 4.0], [3.0, 4.0], [10.0, 0.0], [14.0, -1.0]])
# Adjacent pairs (0,1), (1,2), (7,8); the (2,7) gap contributes nothing.
DELTA = np.array([7.0 / 3.0, 2.0 / 3.0])


def test_grid_delta_averages_adjacent_frame_steps_only() -> None:
    from trackanno.measure.clipfeatures import grid_delta

    got = grid_delta(VALUES, OFFSETS, SPAN)
    assert np.allclose(got, DELTA), f"{got} vs {DELTA}"
    bridged = np.abs(np.diff(VALUES, axis=0)).mean(axis=0)
    assert not np.allclose(bridged, DELTA), "the case does not tell bridging apart"


def test_grid_delta_without_an_adjacent_pair_is_zero() -> None:
    from trackanno.measure.clipfeatures import grid_delta

    lonely = grid_delta(VALUES[:2], np.array([0, 5]), SPAN)
    assert np.allclose(lonely, 0.0), lonely


def test_the_embedding_summary_uses_the_same_delta() -> None:
    from trackanno.measure.embeddings import summarise

    block = summarise("x", VALUES, OFFSETS, SPAN)
    assert np.isclose(block["emb_x_0_delta"], DELTA[0]), block["emb_x_0_delta"]
    assert np.isclose(block["emb_x_1_delta"], DELTA[1]), block["emb_x_1_delta"]
    assert np.isclose(block["emb_x_0_mean"], VALUES[:, 0].mean())
    assert np.isclose(block["emb_x_0_std"], VALUES[:, 0].std())


def test_a_nan_embedding_row_is_skipped_not_bridged() -> None:
    from trackanno.measure.embeddings import summarise

    block = summarise("x", VALUES, OFFSETS, SPAN)
    holed = np.vstack([VALUES[:2], [[np.nan, np.nan]], VALUES[2:]])
    holed_block = summarise("x", holed, np.array([0, 1, 5, 2, 7, 8]), SPAN)
    assert all(np.isclose(holed_block[k], block[k]) for k in block), holed_block
    empty = summarise("x", np.full((3, 2), np.nan), np.array([0, 1, 2]), SPAN)
    assert all(np.isnan(v) for v in empty.values()), f"no vector at all is not NaN: {empty}"


def test_border_dropped_rows_do_not_break_feature_alignment(root) -> None:
    """instance_features.npy aligns with the unfiltered tracks, not the border-filtered ones."""
    import json

    from trackanno.infer.densepredict import build_dense_features
    from trackanno.measure import embeddings as emb
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
    vectors = np.random.default_rng(0).normal(size=(len(tracks), 8)).astype(np.float32)
    np.save(movie / "instance_features.npy", vectors)

    border_cfg.paths.work_dir.mkdir(parents=True, exist_ok=True)
    emb.save_bases(border_cfg.paths.work_dir / emb.BASES_FILE,
                   emb.fit_bases(border_cfg, {"det": {MOVIE: vectors}}))

    out = build_dense_features(border_cfg)
    assert len(out) > 0, "a project whose border rule drops rows still builds"
    assert set(out["track_id"]) == {WHOLE}, (
        f"an all-border track is planned: {sorted(set(out['track_id']))}")
    assert "emb_det_0_mean" in out.columns
    assert out.loc[out["skip_reason"] == "", "emb_det_0_mean"].notna().all()
