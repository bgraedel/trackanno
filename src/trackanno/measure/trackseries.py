"""Per-frame series for one track (:func:`build_series`) and its reduction over any window.

A window's features depend only on its frames; ``beat_*`` and ``wave_*`` are added by clipfeatures.
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
from ..config import Config

__all__ = [
    "build_series", "aggregate_window", "feature_names",
    "SERIES_SPEC", "SERIES_COLUMNS", "WINDOW_FUNCTIONALS",
]


# --------------------------------------------------------------------------- #
# the contract
# --------------------------------------------------------------------------- #
# Column -> reducers over a window; feature name is column + reducer suffix.
SERIES_SPEC: dict[str, tuple[str, ...]] = {
    # -- detection box ------------------------------------------------------- #
    "cx": (),                       # box centre
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

# Functionals of the whole window; NaN when not computable.
WINDOW_FUNCTIONALS: tuple[str, ...] = (
    "path_length", "net_displacement", "straightness", "radius_gyration",
    "turn_lag_frames", "turn_abs_mean", "turn_std", "turn_abs_p90", "turn_n_valid",
    "flagellum_len_cv",
    "body_path_length", "body_net_displacement", "body_straightness",
    "body_displacement_per_len", "body_rotation_speed",
    "tip_angle_std", "tip_angular_speed",
)

# Reducers see finite values only; "frac" is the mean of an indicator.
_REDUCERS: dict[str, tuple[str, object]] = {
    "mean": ("_mean", np.mean),
    "std": ("_std", np.std),                          # ddof=0
    "median": ("_median", np.median),
    "p90": ("_p90", lambda v: np.percentile(v, 90)),
    "max": ("_max", np.max),
    "frac": ("_frac", np.mean),
}

# Feature names that do not follow column + suffix.
_FEATURE_NAMES: dict[tuple[str, str], str] = {
    ("shape_elong", "mean"): "shape_elongation",
}

# `kpt_conf_mean` is weighted by the per-frame keypoint count.
_POOLED_WEIGHT: dict[str, str] = {"kpt_conf": "kpt_conf_n"}

# Backward differences: the first in-window sample straddles the edge and is dropped.
_BACKWARD_DIFF: tuple[str, ...] = ("speed", "immotile", "body_speed")


def feature_names() -> tuple[str, ...]:
    """Every key :func:`aggregate_window` returns, in a fixed order."""
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
    """Per-frame quantities for one track on a frame grid starting at ``f_start``, NaN where undetected.

    Rates are per second, so :func:`aggregate_window` needs the same ``fps``.
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

    wanted = ["x1", "y1", "x2", "y2", "conf"]
    if has_pose:
        wanted += kx + ky + (kc if has_kpt_conf else [])
    grid = _on_grid(sub, wanted, n_frames, f_start)

    box = grid[:, :4]
    x1, y1, x2, y2 = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
    cols["cx"], cols["cy"] = (x1 + x2) / 2, (y1 + y2) / 2
    cols["box_w"], cols["box_h"] = x2 - x1, y2 - y1
    cols["area"] = cols["box_w"] * cols["box_h"]
    # Axis-aligned box: area and aspect also change with orientation; prefer shape_* with keypoints.
    cols["aspect"] = np.where(
        cols["box_h"] > 1e-6, cols["box_w"] / np.maximum(cols["box_h"], 1e-6), np.nan
    )

    cols["speed"] = _backward_step(cols["cx"], cols["cy"]) / dt
    finite_speed = np.isfinite(cols["speed"])
    cols["immotile"] = np.where(
        finite_speed, (cols["speed"] < IMMOTILE_PX_PER_SEC).astype(float), np.nan
    )

    # Presence means a row exists, whatever its values.
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

    # Contiguous copies for faster per-frame indexing.
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
            # np.cov(pts.T) without its per-call overhead.
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
        cols["shape_elong"] = pc1 / np.where(pc2 > 1e-6, pc2, np.nan)

        # body: node 0 -> node 1 (Head -> Base for this skeleton)
        cols["body_len"] = np.hypot(px[:, 0] - px[:, 1], py[:, 0] - py[:, 1])

        # Only frames with the whole chain present.
        seg = np.hypot(np.diff(px[:, 1:], axis=1), np.diff(py[:, 1:], axis=1))
        complete = np.isfinite(seg).all(axis=1)
        cols["flagellum_len"] = np.where(complete, seg.sum(axis=1), np.nan)
        cols["flagellum_complete"] = complete.astype(float)

        # -- translocation of the Head-Base midpoint ------------------------- #
        cols["anchor_x"] = np.nanmean(px[:, :2], axis=1)
        cols["anchor_y"] = np.nanmean(py[:, :2], axis=1)
        cols["body_speed"] = _backward_step(cols["anchor_x"], cols["anchor_y"]) / dt

        # Base -> Head.
        cols["body_axis_ang"] = np.arctan2(py[:, 0] - py[:, 1], px[:, 0] - px[:, 1])
        cols["tip_ang"] = np.arctan2(py[:, -1] - py[:, 1], px[:, -1] - px[:, 1])

    return _frame(cols, f_start, n_frames, fps, has_pose=True)


def _backward_step(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """One-frame displacement where sample i is the step into frame i; NaN at the first."""
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
    """Collapse ``series`` over ``[f_start, f_end)`` to a feature dict, NaN where unmeasurable."""
    f_start, f_end = int(f_start), int(f_end)
    if f_end <= f_start:
        raise ValueError(f"empty window [{f_start}, {f_end})")
    win = _window(series, f_start, f_end)
    n_frames = f_end - f_start
    dt = 1.0 / max(fps, 1e-6)

    out: dict[str, float] = {}
    with warnings.catch_warnings():
        # All-NaN slices are normal for undetected stretches.
        warnings.simplefilter("ignore", RuntimeWarning)
        _reduce(win, out)
        _trajectory(win, out, n_frames, dt)
        _pose(win, out, n_frames, dt, cfg)

    for key in WINDOW_FUNCTIONALS:
        out.setdefault(key, np.nan)
    return out


def _window(series: pd.DataFrame, f_start: int, f_end: int) -> dict[str, np.ndarray]:
    """Slice every column to ``[f_start, f_end)`` by absolute frame, positionally."""
    n_rows = len(series.index)
    if n_rows == 0:
        raise ValueError(f"window [{f_start}, {f_end}) asked of an empty series")
    origin = int(series.index[0])
    if int(series.index[-1]) - origin + 1 != n_rows:
        raise ValueError("series index is not a contiguous frame range")
    i0, i1 = f_start - origin, f_end - origin
    if i0 < 0 or i1 > n_rows:
        raise ValueError(
            f"window [{f_start}, {f_end}) is not inside the series "
            f"[{origin}, {origin + n_rows}); build the series over a wider range"
        )
    block = series.to_numpy()[i0:i1]
    win = {name: block[:, j] for j, name in enumerate(series.columns)}
    for name in _BACKWARD_DIFF:
        # Copy: the slice is a view onto the caller's frame.
        edge = win[name].copy()
        edge[0] = np.nan
        win[name] = edge
    return win


def _reduce(win: dict[str, np.ndarray], out: dict[str, float]) -> None:
    """Apply SERIES_SPEC to each column's finite values."""
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
    # Needs at least one measurable one-frame step.
    if not np.isfinite(win["speed"]).any():
        return

    path, net = _path_length(cx, cy)
    out["path_length"] = _safe(path)
    out["net_displacement"] = _safe(net)
    out["straightness"] = _safe(net / path if path > 1e-6 else np.nan)
    out["radius_gyration"] = _safe(
        np.sqrt(np.nanmean((cx - np.nanmean(cx)) ** 2 + (cy - np.nanmean(cy)) ** 2))
    )

    # Heading lag is chosen per window to clear the localisation noise.
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
        # Unwrap on the filled grid; keep only steps between two detected frames.
        ang = np.unwrap(_interp_nan(axis[:, None])[:, 0])
        turn = np.diff(ang)[valid[1:] & valid[:-1]]
        if turn.size:
            out["body_rotation_speed"] = _safe(np.nanmean(np.abs(turn)) / dt)

    # Unwrap detected samples only; rate steps only between adjacent frames.
    tip = win["tip_ang"]
    idx = np.flatnonzero(np.isfinite(tip))
    if idx.size >= 8:
        ang = np.unwrap(tip[idx])
        out["tip_angle_std"] = _safe(np.std(ang))
        step = np.diff(ang)[np.diff(idx) == 1]
        if step.size:
            out["tip_angular_speed"] = _safe(np.mean(np.abs(step)) / dt)
