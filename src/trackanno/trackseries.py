"""Per-frame series for one track, sliced and reduced into window features.

:func:`build_series` walks a track once and returns a per-FRAME table;
:func:`aggregate_window` slices it to any half-open frame range and collapses
that to a scalar dict. The window is a free parameter, not a property of the
clip manifest, and the per-frame ``eigh`` behind ``shape_*`` is paid once per
track rather than once per overlapping window.

The hard contract: a window's features are a property of the WINDOW and nothing
else, so aggregating a range out of a whole-track series must equal aggregating
it out of a series built for that range alone. The gates are part of that -
where there is too little to measure the feature is NaN rather than computed
anyway, and that NaN pattern is itself informative to a NaN-native model.

``beat_*`` and ``wave_*`` are ABSENT rather than NaN: they need the whole 2-D
body-frame displacement matrix and a resolvability floor that is a function of
window LENGTH, so they change meaning when the window does.
:func:`trackanno.clipfeatures.window_features` joins them back on.

Frames are addressed by ABSOLUTE frame number: the table is indexed
``f_start .. f_start + n_frames - 1`` and ``aggregate_window`` takes absolute
bounds.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .clipfeatures import (
    IMMOTILE_PX_PER_SEC,
    LOCALISATION_NOISE_PX,
    _heading_lag,
    _interp_nan,
    _on_grid,
    _path_length,
    _safe,
)
from .config import Config

__all__ = [
    "build_series", "aggregate_window", "feature_names",
    "SERIES_SPEC", "SERIES_COLUMNS", "WINDOW_FUNCTIONALS",
]


# --------------------------------------------------------------------------- #
# the contract
# --------------------------------------------------------------------------- #
# Each column maps to the reducers that collapse it over a window; the feature
# name is column + the reducer's suffix, so this table IS the feature list for
# every point quantity. Quantities defined by the window's ENDPOINTS live in
# `WINDOW_FUNCTIONALS`. An empty reducer tuple means no feature of its own.
SERIES_SPEC: dict[str, tuple[str, ...]] = {
    # -- detection box ------------------------------------------------------- #
    "cx": (),                       # box centre, the trajectory everything else hangs off
    "cy": (),
    "box_w": (),
    "box_h": (),
    "area": ("mean", "std"),
    "aspect": ("mean", "std"),
    # -- trajectory ---------------------------------------------------------- #
    "speed": ("mean", "median", "p90", "std", "max"),
    "immotile": ("frac",),
    # -- quality ------------------------------------------------------------- #
    "present": ("frac",),
    "conf": ("mean", "std"),
    "kpt_conf": ("mean",),
    "kpt_conf_n": (),
    # -- rotation-invariant shape, from the keypoint cloud ------------------- #
    "shape_extent": ("mean", "std"),
    "shape_pc1": ("mean",),
    "shape_pc2": ("mean",),
    "shape_elong": ("mean",),
    # -- pose lengths -------------------------------------------------------- #
    "body_len": ("mean", "std"),
    "flagellum_len": ("mean", "std"),
    "flagellum_complete": ("frac",),
    # -- cell body, not the box ---------------------------------------------- #
    "anchor_x": (),
    "anchor_y": (),
    "body_speed": ("mean", "median", "p90"),
    # -- orientation --------------------------------------------------------- #
    "body_axis_ang": (),            # Base -> Head
    "tip_ang": (),                  # Base -> most distal node
}

SERIES_COLUMNS: tuple[str, ...] = tuple(SERIES_SPEC)

# Functionals of the WINDOW, not reductions of a per-frame column. Listed so the
# schema is fixed whether or not any could be computed (the `setdefault` sweep
# in `aggregate_window`): never a missing column, always NaN.
WINDOW_FUNCTIONALS: tuple[str, ...] = (
    "path_length", "net_displacement", "straightness", "radius_gyration",
    "turn_lag_frames", "turn_abs_mean", "turn_std", "turn_abs_p90", "turn_n_valid",
    "flagellum_len_cv",
    "body_path_length", "body_net_displacement", "body_straightness",
    "body_displacement_per_len", "body_rotation_speed",
    "tip_angle_std", "tip_angular_speed",
)

# Reducers see the window's FINITE values only, compacted once by `_reduce`, so
# these are the plain ones. `mean` and `frac` are the same callable on purpose:
# "frac" averages an INDICATOR, whose own NaN pattern chooses the denominator.
# `present` is defined at every frame, so present_frac is over the whole window;
# `immotile` is NaN where no step could be measured, so immotile_frac is over
# measurable steps.
_REDUCERS: dict[str, tuple[str, object]] = {
    "mean": ("_mean", np.mean),
    "std": ("_std", np.std),                          # ddof=0, as everywhere in clipfeatures
    "median": ("_median", np.median),
    "p90": ("_p90", lambda v: np.percentile(v, 90)),
    "max": ("_max", np.max),
    "frac": ("_frac", np.mean),
}

# One existing feature has no reducer suffix. Renaming it would change the input
# schema of every model already trained against clip_features.parquet.
_FEATURE_NAMES: dict[tuple[str, str], str] = {
    ("shape_elong", "mean"): "shape_elongation",
}

# `kpt_conf_mean` pools over keypoint-frames, not frames: `nanmean` of the whole
# (frames x keypoints) grid, since averaging per-frame means would weight a
# frame with one keypoint like a frame with eight. The COUNT is carried too.
_POOLED_WEIGHT: dict[str, str] = {"kpt_conf": "kpt_conf_n"}

# Columns holding a BACKWARD difference: sample i is the step from frame i-1 into
# frame i. Drop the first in-window sample, which straddles the left edge, or a
# window's speed depends on a frame outside it and two overlapping windows both
# claim the shared step. That leaves `[f_start, f_end)` holding exactly the n-1
# steps interior to it.
_BACKWARD_DIFF: tuple[str, ...] = ("speed", "immotile", "body_speed")


def feature_names() -> tuple[str, ...]:
    """Every key :func:`aggregate_window` returns, in a fixed order.

    A schema, not a convenience: a model trained on one project and applied to
    another must see the same columns in both, NaN ones included.
    """
    names = [
        _FEATURE_NAMES.get((col, name), col + _REDUCERS[name][0])
        for col, reducers in SERIES_SPEC.items()
        for name in reducers
    ]
    return tuple(names) + WINDOW_FUNCTIONALS


# --------------------------------------------------------------------------- #
# series
# --------------------------------------------------------------------------- #
def build_series(
    sub: pd.DataFrame, cfg: Config, fps: float, f_start: int, n_frames: int
) -> pd.DataFrame:
    """Per-frame quantities for one track, on the full frame grid.

    ``sub`` is one track's rows; ``f_start`` and ``n_frames`` define the grid,
    normally the track's whole span. Rows outside it are ignored and frames with
    no row become NaN, as in :func:`~trackanno.clipfeatures._on_grid`. Rates are
    per SECOND, so :func:`aggregate_window` must be handed the same ``fps``.

    ``present`` and ``flagellum_complete`` are 0.0/1.0, not NaN/1.0: a pose track
    answers both at every frame, so 0.0 means NO. They go all-NaN only when the
    project has no keypoints, which keeps ``flagellum_complete_frac`` NaN on a
    ``task: detect`` project.
    """
    sub = sub.sort_values("frame")
    dt = 1.0 / max(fps, 1e-6)
    n_frames = int(n_frames)
    f_start = int(f_start)
    nan = np.full(n_frames, np.nan)

    cols: dict[str, np.ndarray] = {}

    n_kpts = cfg.n_keypoints
    kx = [f"kp_x_{k}" for k in range(n_kpts)]
    ky = [f"kp_y_{k}" for k in range(n_kpts)]
    kc = [f"kp_c_{k}" for k in range(n_kpts)]
    has_pose = n_kpts >= 3 and all(c in sub.columns for c in kx + ky)
    has_kpt_conf = has_pose and all(c in sub.columns for c in kc)

    # One `_on_grid` call for every column at once: the gridding rule is the same
    # for all of them, and splitting it per block costs a take and cast each time.
    wanted = ["x1", "y1", "x2", "y2", "conf"]
    if has_pose:
        wanted += kx + ky + (kc if has_kpt_conf else [])
    grid = _on_grid(sub, wanted, n_frames, f_start)

    box = grid[:, :4]
    x1, y1, x2, y2 = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
    cols["cx"], cols["cy"] = (x1 + x2) / 2, (y1 + y2) / 2
    cols["box_w"], cols["box_h"] = x2 - x1, y2 - y1
    cols["area"] = cols["box_w"] * cols["box_h"]
    # READ `area` AND `aspect` WITH CARE: the box is AXIS-ALIGNED, so they move
    # when the cell TURNS even though its shape has not changed (extent L at
    # angle t gives a box L|cos t| x L|sin t|). 98.8% of aspect_mean's variance
    # is orientation, so it is an orientation readout, not a shape one;
    # area_mean is size-dominated (unique-to-size R2 0.76 vs orientation 0.18)
    # but still contaminated. They stay because `detection.task: detect` has no
    # keypoints; where keypoints exist, prefer the shape_* columns below.
    cols["aspect"] = np.where(
        cols["box_h"] > 1e-6, cols["box_w"] / np.maximum(cols["box_h"], 1e-6), np.nan
    )

    cols["speed"] = _backward_step(cols["cx"], cols["cy"]) / dt
    finite_speed = np.isfinite(cols["speed"])
    cols["immotile"] = np.where(
        finite_speed, (cols["speed"] < IMMOTILE_PX_PER_SEC).astype(float), np.nan
    )

    # Built from the frame index, not from "is x1 finite": presence is about the
    # ROW existing, and an unwritable column would otherwise stop it counting.
    present = np.zeros(n_frames)
    offsets = sub["frame"].to_numpy(dtype=int) - f_start
    inside = (offsets >= 0) & (offsets < n_frames)
    present[offsets[inside]] = 1.0
    cols["present"] = present
    cols["conf"] = grid[:, 4]

    if not has_pose:
        for name in ("kpt_conf", "kpt_conf_n", "shape_extent", "shape_pc1", "shape_pc2",
                     "shape_elong", "body_len", "flagellum_len", "flagellum_complete",
                     "anchor_x", "anchor_y", "body_speed", "body_axis_ang", "tip_ang"):
            cols[name] = nan.copy()
        return _frame(cols, f_start, n_frames, fps, has_pose=False)

    # Contiguous copies: the eigen loop's per-frame indexing is slower on a
    # strided view of the combined grid.
    px = np.ascontiguousarray(grid[:, 5:5 + n_kpts])
    py = np.ascontiguousarray(grid[:, 5 + n_kpts:5 + 2 * n_kpts])
    conf = (
        np.ascontiguousarray(grid[:, 5 + 2 * n_kpts:])
        if has_kpt_conf else np.ones_like(px)
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN rows are expected
        cols["kpt_conf"] = np.nanmean(conf, axis=1)
        cols["kpt_conf_n"] = np.isfinite(conf).sum(axis=1).astype(float)

        # -- rotation-invariant size and elongation -------------------------- #
        # `np.cov`'s ddof=1 and the ascending-eigenvalue convention of `eigh`,
        # spelled out. The per-frame `eigh` is the expensive part of the feature
        # set; running it once per TRACK is most of what this module buys.
        extent = np.full(n_frames, np.nan)
        pc1 = np.full(n_frames, np.nan)
        pc2 = np.full(n_frames, np.nan)
        finite_kpt = np.isfinite(px) & np.isfinite(py)
        measurable = np.flatnonzero(finite_kpt.sum(axis=1) >= 3)
        for i in measurable:
            ok = finite_kpt[i]
            xs, ys = px[i, ok], py[i, ok]
            extent[i] = float(np.hypot(xs[:, None] - xs, ys[:, None] - ys).max())
            pts = np.stack([xs - xs.mean(), ys - ys.mean()], axis=1)
            # `np.cov(pts.T)` spelled out - same maths - because np.cov spends 12
            # of its 28 us per call on argument handling for a 2x2.
            cov = np.array(pts.T)
            cov -= cov.mean(axis=1)[:, None]
            cov = cov @ cov.T
            cov *= 1.0 / (xs.size - 1)
            _, axes = np.linalg.eigh(cov)
            proj = pts @ axes                      # columns ascend by eigenvalue
            minor, major = proj[:, 0], proj[:, 1]
            pc2[i] = float(minor.max() - minor.min())    # np.ptp, without its wrapper
            pc1[i] = float(major.max() - major.min())
        cols["shape_extent"], cols["shape_pc1"], cols["shape_pc2"] = extent, pc1, pc2
        # Ratio per FRAME, then averaged: the mean of a ratio is not the ratio of
        # the means, and it is the per-frame elongation that describes the cell.
        cols["shape_elong"] = pc1 / np.where(pc2 > 1e-6, pc2, np.nan)

        # body: node 0 -> node 1 (Head -> Base for this skeleton)
        cols["body_len"] = np.hypot(px[:, 0] - px[:, 1], py[:, 0] - py[:, 1])

        # Only frames where the WHOLE chain is present: nansum would drop the
        # segments either side of a missing node and read it as a length change.
        seg = np.hypot(np.diff(px[:, 1:], axis=1), np.diff(py[:, 1:], axis=1))
        complete = np.isfinite(seg).all(axis=1)
        cols["flagellum_len"] = np.where(complete, seg.sum(axis=1), np.nan)
        cols["flagellum_complete"] = complete.astype(float)

        # -- translocation, anchored on the CELL BODY ------------------------ #
        # The box centroid jitters as fast for a tethered cell thrashing its
        # flagellum as for a swimmer; the Head-Base midpoint does not.
        cols["anchor_x"] = np.nanmean(px[:, :2], axis=1)
        cols["anchor_y"] = np.nanmean(py[:, :2], axis=1)
        cols["body_speed"] = _backward_step(cols["anchor_x"], cols["anchor_y"]) / dt

        # Base -> Head. `isfinite` on this angle already implies all four
        # coordinates were finite (arctan2 propagates NaN).
        cols["body_axis_ang"] = np.arctan2(py[:, 0] - py[:, 1], px[:, 0] - px[:, 1])
        cols["tip_ang"] = np.arctan2(py[:, -1] - py[:, 1], px[:, -1] - px[:, 1])

    return _frame(cols, f_start, n_frames, fps, has_pose=True)


def _backward_step(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """One-frame displacement aligned so sample i is the step INTO frame i.

    Length matches the frame grid, NaN at the first sample. See `_BACKWARD_DIFF`
    for why the alignment has to be backwards.
    """
    out = np.full(x.shape[0], np.nan)
    if x.shape[0] >= 2:
        out[1:] = np.hypot(np.diff(x), np.diff(y))
    return out


def _frame(
    cols: dict[str, np.ndarray], f_start: int, n_frames: int, fps: float, has_pose: bool
) -> pd.DataFrame:
    index = pd.RangeIndex(f_start, f_start + n_frames, name="frame")
    frame = pd.DataFrame({c: cols[c] for c in SERIES_COLUMNS}, index=index, dtype=float)
    frame.attrs.update({"fps": float(fps), "has_pose": bool(has_pose)})
    return frame


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def aggregate_window(
    series: pd.DataFrame, f_start: int, f_end: int, fps: float, cfg: Config
) -> dict[str, float]:
    """Collapse ``series`` over the half-open frame range to a feature dict.

    Every kinematic feature for those frames, minus the ``beat_*``/``wave_*``
    spectral block (see the module docstring). ``fps`` must be the value the
    series was built with, since the speed columns are already per second. Where
    there is too little to measure - fewer than one measurable step, two present
    body anchors, eight finite tip angles - this returns NaN.
    """
    f_start, f_end = int(f_start), int(f_end)
    if f_end <= f_start:
        raise ValueError(f"empty window [{f_start}, {f_end})")
    win = _window(series, f_start, f_end)
    n_frames = f_end - f_start
    dt = 1.0 / max(fps, 1e-6)

    out: dict[str, float] = {}
    with warnings.catch_warnings():
        # All-NaN slices are normal for an undetected stretch, and every reducer
        # below is nan-aware precisely so that is not an error.
        warnings.simplefilter("ignore", RuntimeWarning)
        _reduce(win, out)
        _trajectory(win, out, n_frames, dt)
        _pose(win, out, n_frames, dt, cfg)

    for key in WINDOW_FUNCTIONALS:
        out.setdefault(key, np.nan)
    return out


def _window(series: pd.DataFrame, f_start: int, f_end: int) -> dict[str, np.ndarray]:
    """Slice every column to ``[f_start, f_end)``, addressed by absolute frame.

    Positional, not ``reindex``: the index is a contiguous frame range by
    construction, and hashing frame numbers per window would cost more than every
    reducer together. One ``to_numpy`` on the whole frame, not ``series[c]`` per
    column (460x slower).
    """
    n_rows = len(series.index)
    if n_rows == 0:
        raise ValueError(f"window [{f_start}, {f_end}) asked of an empty series")
    origin = int(series.index[0])
    if int(series.index[-1]) - origin + 1 != n_rows:
        raise ValueError(
            "series index is not a contiguous frame range; positional slicing "
            "would silently return the wrong frames"
        )
    i0, i1 = f_start - origin, f_end - origin
    if i0 < 0 or i1 > n_rows:
        raise ValueError(
            f"window [{f_start}, {f_end}) is not inside the series "
            f"[{origin}, {origin + n_rows}); build_series was given too narrow a "
            "frame range, and padding it here would report frames that were never "
            "looked at as undetected"
        )
    block = series.to_numpy()[i0:i1]
    win = {name: block[:, j] for j, name in enumerate(series.columns)}
    for name in _BACKWARD_DIFF:
        # Copy before blanking: the slice is a view onto the caller's frame.
        edge = win[name].copy()
        edge[0] = np.nan
        win[name] = edge
    return win


def _reduce(win: dict[str, np.ndarray], out: dict[str, float]) -> None:
    """Apply SERIES_SPEC. Everything here is a point quantity by construction.

    Finite values are compacted once per column and the plain reducers run on
    that: nan-aware reducers rebuild the same mask per feature, and
    ``np.nanpercentile`` costs three times its plain counterpart. Compacting once
    also fixes the summation order, so a reduction does not depend on where the
    NaNs sit.
    """
    for col, reducers in SERIES_SPEC.items():
        if not reducers:
            continue
        values = win[col]
        finite = np.isfinite(values)
        good = values[finite]
        weight = win[_POOLED_WEIGHT[col]] if col in _POOLED_WEIGHT else None
        for name in reducers:
            suffix, fn = _REDUCERS[name]
            key = _FEATURE_NAMES.get((col, name), col + suffix)
            if good.size == 0:
                # `np.max` of an empty array raises, and reporting a number for a
                # window with nothing in it would be worse than either.
                out[key] = np.nan
            elif weight is not None and name == "mean":
                w = weight[finite]
                n = float(w.sum())
                out[key] = _safe(float((good * w).sum()) / n if n > 0 else np.nan)
            else:
                out[key] = _safe(fn(good))


def _trajectory(
    win: dict[str, np.ndarray], out: dict[str, float], n_frames: int, dt: float
) -> None:
    """Path, spread and turning of the box centroid over this window."""
    cx, cy = win["cx"], win["cy"]
    # A gate on measurable STEPS, not on present frames. Speed keeps the
    # per-frame grid while `_path_length` bridges gaps, so a window holding two
    # detections either side of a gap does have a path length - and is still
    # refused, that bridge being a guess about the frames the tracker lost.
    if not np.isfinite(win["speed"]).any():
        return

    path, net = _path_length(cx, cy)
    out["path_length"] = _safe(path)
    out["net_displacement"] = _safe(net)
    out["straightness"] = _safe(net / path if path > 1e-6 else np.nan)
    out["radius_gyration"] = _safe(
        np.sqrt(np.nanmean((cx - np.nanmean(cx)) ** 2 + (cy - np.nanmean(cy)) ** 2))
    )

    # Heading needs a baseline longer than the localisation jitter, and the lag
    # that provides it depends on how fast THIS window's cell moved, so it is
    # re-derived per window rather than stored per frame.
    lag = _heading_lag(cx, cy, n_frames)
    out["turn_lag_frames"] = float(lag)
    hx, hy = cx[lag:] - cx[:-lag], cy[lag:] - cy[:-lag]
    moved = np.hypot(hx, hy) > LOCALISATION_NOISE_PX
    heading = np.where(moved, np.arctan2(hy, hx), np.nan)
    turn = heading[lag:] - heading[:-lag] if heading.size > lag else np.empty(0)
    turn = (turn + np.pi) % (2 * np.pi) - np.pi
    turn = turn[np.isfinite(turn)]
    span = lag * dt
    if turn.size >= 2:
        out["turn_abs_mean"] = _safe(np.abs(turn).mean() / span)
        out["turn_std"] = _safe(turn.std() / span)
        out["turn_abs_p90"] = _safe(np.percentile(np.abs(turn), 90) / span)
    out["turn_n_valid"] = float(turn.size)


def _pose(
    win: dict[str, np.ndarray], out: dict[str, float], n_frames: int, dt: float, cfg: Config
) -> None:
    """Window functionals that need the keypoints."""
    mean_arc = out.get("flagellum_len_mean", np.nan)
    out["flagellum_len_cv"] = _safe(
        out.get("flagellum_len_std", np.nan) / mean_arc
        if np.isfinite(mean_arc) and mean_arc > 1e-6 else np.nan
    )

    anchor_x, anchor_y = win["anchor_x"], win["anchor_y"]
    present = np.flatnonzero(np.isfinite(anchor_x) & np.isfinite(anchor_y))
    if np.isfinite(win["body_speed"]).any() and present.size >= 2:
        # Same endpoints for path and net, so body_straightness cannot exceed 1.
        path, net = _path_length(anchor_x, anchor_y)
        out["body_net_displacement"] = _safe(net)
        out["body_path_length"] = _safe(path)
        out["body_straightness"] = _safe(net / path if path > 1e-6 else np.nan)
        body_len = out.get("body_len_mean", np.nan)
        out["body_displacement_per_len"] = _safe(
            net / body_len if np.isfinite(body_len) and body_len > 1e-6 else np.nan
        )

    axis = win["body_axis_ang"]
    valid = np.isfinite(axis)
    if n_frames >= 2 and cfg.n_keypoints >= 3 and valid.sum() >= 2:
        # Unwrap on the gap-filled grid, then keep only steps whose BOTH
        # endpoints were really detected: a gap must neither break the unwrap nor
        # contribute a bridged rotation. The fill is local to the window, which
        # makes this a window functional rather than a series column.
        ang = np.unwrap(_interp_nan(axis[:, None])[:, 0])
        turn = np.diff(ang)[valid[1:] & valid[:-1]]
        if turn.size:
            out["body_rotation_speed"] = _safe(np.nanmean(np.abs(turn)) / dt)

    # Compacted, not gridded, and therefore gap-bridging: the unwrap walks
    # consecutive samples, so a NaN in the middle would end it there and leave
    # the rest of the window on a different 2*pi branch.
    tip = win["tip_ang"]
    tip = tip[np.isfinite(tip)]
    if tip.size >= 8:
        tip = np.unwrap(tip)
        out["tip_angle_std"] = _safe(np.std(tip))
        out["tip_angular_speed"] = _safe(np.mean(np.abs(np.diff(tip))) / dt)
