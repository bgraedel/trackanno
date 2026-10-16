"""Per-frame series for one track (:func:`build_series`) and its reduction over any window.

A window's features depend only on its frames; ``wave_*`` are added by clipfeatures.
Speed, rotation and turning compare means over ``KINEMATIC_LAG_S``, since per frame
keypoint jitter swamps them.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .clipfeatures import (
    IMMOTILE_PX_PER_SEC,
    KINEMATIC_LAG_S,
    LOCALISATION_NOISE_PX,
    _on_grid,
    _safe,
)
from ..config import Config
from .wavespectra import segment_arcs

__all__ = [
    "build_series", "aggregate_window", "feature_names", "kinematic_lag",
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
    "anchor_x": (),                 # Head-Base midpoint
    "anchor_y": (),
    "body_speed": ("mean", "median", "p90"),
    "body_immotile": ("frac",),
    # -- orientation --------------------------------------------------------- #
    "body_axis_ang": (),            # Base -> Head, in the lab
    "tip_ang": (),                  # Base -> Tip, relative to Head -> Base
}

SERIES_COLUMNS: tuple[str, ...] = tuple(SERIES_SPEC)

# Functionals of the whole window; NaN when not computable.
WINDOW_FUNCTIONALS: tuple[str, ...] = (
    "path_length", "net_displacement", "straightness", "radius_gyration",
    "turn_abs_mean", "turn_std", "turn_abs_p90", "turn_n_valid",
    "flagellum_len_cv",
    "body_path_length", "body_net_displacement", "body_straightness", "body_radius_gyration",
    "body_turn_abs_mean", "body_turn_std", "body_turn_abs_p90", "body_turn_n_valid",
    "body_displacement_per_len", "body_rotation_speed",
    "tip_angle_std",
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

# Sample i spans frames (i - 2*lag, i], so a window's first 2*lag - 1 samples reach before it.
_LAGGED: tuple[str, ...] = ("speed", "immotile", "body_speed", "body_immotile")


def feature_names() -> tuple[str, ...]:
    """Every key :func:`aggregate_window` returns, in a fixed order."""
    names = [
        _FEATURE_NAMES.get((col, name), col + _REDUCERS[name][0])
        for col, reducers in SERIES_SPEC.items()
        for name in reducers
    ]
    return tuple(names) + WINDOW_FUNCTIONALS


def kinematic_lag(fps: float) -> int:
    """Frames in ``KINEMATIC_LAG_S`` at ``fps``, at least one."""
    return max(1, int(round(KINEMATIC_LAG_S * float(fps))))


def _lagged_change(values: np.ndarray, lag: int) -> np.ndarray:
    """Change per frame from the mean over frames ``(i-2*lag, i-lag]`` to that over ``(i-lag, i]``.

    A row with any NaN is missing; a mean needs half its frames and is timed at the frames it has.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    n, d = values.shape
    out = np.full((n, d), np.nan)
    if n < 2 * lag:
        return out
    ok = np.isfinite(values).all(axis=1)
    filled = np.where(ok[:, None], values, 0.0)
    m = n - lag + 1                                   # block j covers frames j .. j+lag-1
    total, count, when = np.zeros((m, d)), np.zeros(m), np.zeros(m)
    # Summed in a fixed order, so a window gets the same bits in any longer series.
    for k in range(lag):
        total += filled[k:k + m]
        count += ok[k:k + m]
        when += k * ok[k:k + m]
    count[count < (lag + 1) // 2] = np.nan
    mean = total / count[:, None]
    centre = when / count                             # mean frame offset within the block
    span = lag + centre[lag:] - centre[:-lag]
    out[2 * lag - 1:] = (mean[lag:] - mean[:-lag]) / span[:, None]
    return out


def _wrap(angle: np.ndarray) -> np.ndarray:
    """Angles folded into [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


# --------------------------------------------------------------------------- #
# series
# --------------------------------------------------------------------------- #
def build_series(
    sub: pd.DataFrame, cfg: Config, fps: float, f_start: int, n_frames: int
) -> pd.DataFrame:
    """Per-frame quantities for one track on a frame grid starting at ``f_start``, NaN where undetected.

    Rates are per second over :func:`kinematic_lag`, so :func:`aggregate_window` needs this ``fps``.
    """
    sub = sub.sort_values("frame")
    dt = 1.0 / max(fps, 1e-6)
    lag = kinematic_lag(fps)
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

    cols["speed"] = _lagged_speed(cols["cx"], cols["cy"], lag) / dt
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
                     "anchor_x", "anchor_y", "body_speed", "body_immotile", "body_axis_ang",
                     "tip_ang"):
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
        dx, dy = np.diff(px[:, 1:], axis=1), np.diff(py[:, 1:], axis=1)
        seg = segment_arcs(np.hypot(dx, dy), np.unwrap(np.arctan2(dy, dx), axis=1))
        complete = np.isfinite(seg).all(axis=1)
        cols["flagellum_len"] = np.where(complete, seg.sum(axis=1), np.nan)
        cols["flagellum_complete"] = complete.astype(float)

        # -- translocation of the Head-Base midpoint, NaN unless both are seen -- #
        cols["anchor_x"] = (px[:, 0] + px[:, 1]) / 2
        cols["anchor_y"] = (py[:, 0] + py[:, 1]) / 2
        cols["body_speed"] = _lagged_speed(cols["anchor_x"], cols["anchor_y"], lag) / dt
        # The box sways with a beating flagellum; the body says whether the cell moves.
        cols["body_immotile"] = np.where(
            np.isfinite(cols["body_speed"]),
            (cols["body_speed"] < IMMOTILE_PX_PER_SEC).astype(float), np.nan)

        cols["body_axis_ang"] = np.arctan2(py[:, 0] - py[:, 1], px[:, 0] - px[:, 1])
        # The flagellum against the body, so the cell's own rotation cancels.
        tip_lab = np.arctan2(py[:, -1] - py[:, 1], px[:, -1] - px[:, 1])
        cols["tip_ang"] = _wrap(tip_lab - cols["body_axis_ang"] - np.pi)

    return _frame(cols, f_start, n_frames, fps, has_pose=True)


def _lagged_speed(x: np.ndarray, y: np.ndarray, lag: int) -> np.ndarray:
    """Speed in px per frame from :func:`_lagged_change` of the position."""
    v = _lagged_change(np.column_stack([x, y]), lag)
    return np.hypot(v[:, 0], v[:, 1])


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
    built = series.attrs.get("fps")
    if built is not None and not np.isclose(float(built), float(fps)):
        raise ValueError(f"series built at {built} fps cannot be read at {fps} fps")
    lag = kinematic_lag(fps)
    win = _window(series, f_start, f_end, lag)
    dt = 1.0 / max(fps, 1e-6)

    out: dict[str, float] = {}
    with warnings.catch_warnings():
        # All-NaN slices are normal for undetected stretches.
        warnings.simplefilter("ignore", RuntimeWarning)
        _reduce(win, out)
        _trajectory(win, out, dt, lag)
        _pose(win, out, dt, lag)

    for key in WINDOW_FUNCTIONALS:
        out.setdefault(key, np.nan)
    return out


def _window(series: pd.DataFrame, f_start: int, f_end: int, lag: int) -> dict[str, np.ndarray]:
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
    for name in _LAGGED:
        # Copy: the slice is a view onto the caller's frame.
        edge = win[name].copy()
        edge[:2 * lag - 1] = np.nan
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


def _path(x: np.ndarray, y: np.ndarray, speed: np.ndarray, dt: float, lag: int,
          out: dict[str, float], prefix: str) -> bool:
    """Net displacement, path length and straightness; False with under two positions."""
    present = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
    if present.size < 2:
        return False
    first, last = present[0], present[-1]
    net = float(np.hypot(x[last] - x[first], y[last] - y[first]))
    out[prefix + "net_displacement"] = _safe(net)
    speed = speed[np.isfinite(speed)]
    if speed.size:
        # Lagged speed over the span seen: summed frame steps would add up the jitter.
        path = float(speed.mean()) * float(last - first) * dt
        out[prefix + "path_length"] = _safe(path)
        # Needs the motion a heading needs: a mean move per lag clear of the noise.
        if float(speed.mean()) * lag * dt > 3.0 * LOCALISATION_NOISE_PX:
            out[prefix + "straightness"] = _safe(min(net / path, 1.0))
    return True


def _turning(x: np.ndarray, y: np.ndarray, dt: float, lag: int,
             out: dict[str, float], prefix: str) -> None:
    """Spread and turning of a position over the window."""
    out[prefix + "radius_gyration"] = _safe(
        np.sqrt(np.nanmean((x - np.nanmean(x)) ** 2 + (y - np.nanmean(y)) ** 2))
    )
    # Heading over one lag where the move clears the noise; turn over the next lag.
    v = _lagged_change(np.column_stack([x, y]), lag)
    moved = np.hypot(v[:, 0], v[:, 1]) * lag > 3.0 * LOCALISATION_NOISE_PX
    heading = np.where(moved, np.arctan2(v[:, 1], v[:, 0]), np.nan)
    turn = _wrap(heading[lag:] - heading[:-lag])
    turn = turn[np.isfinite(turn)]
    span = lag * dt
    if turn.size >= 2:
        out[prefix + "turn_abs_mean"] = _safe(np.abs(turn).mean() / span)
        out[prefix + "turn_std"] = _safe(turn.std() / span)
        out[prefix + "turn_abs_p90"] = _safe(np.percentile(np.abs(turn), 90) / span)
    out[prefix + "turn_n_valid"] = float(turn.size)


def _trajectory(win: dict[str, np.ndarray], out: dict[str, float], dt: float, lag: int) -> None:
    """Path, spread and turning of the box centroid over this window."""
    if _path(win["cx"], win["cy"], win["speed"], dt, lag, out, ""):
        _turning(win["cx"], win["cy"], dt, lag, out, "")


def _pose(win: dict[str, np.ndarray], out: dict[str, float], dt: float, lag: int) -> None:
    """Window functionals that need the keypoints."""
    mean_arc = out.get("flagellum_len_mean", np.nan)
    out["flagellum_len_cv"] = _safe(
        out.get("flagellum_len_std", np.nan) / mean_arc
        if np.isfinite(mean_arc) and mean_arc > 1e-6 else np.nan
    )

    if _path(win["anchor_x"], win["anchor_y"], win["body_speed"], dt, lag, out, "body_"):
        _turning(win["anchor_x"], win["anchor_y"], dt, lag, out, "body_")
        net = out["body_net_displacement"]
        body_len = out.get("body_len_mean", np.nan)
        out["body_displacement_per_len"] = _safe(
            net / body_len if np.isfinite(body_len) and body_len > 1e-6 else np.nan
        )

    # Rotation over the lag as for speed, on the unwrapped detected samples.
    axis = win["body_axis_ang"]
    idx = np.flatnonzero(np.isfinite(axis))
    if idx.size >= 2:
        ang = np.full(axis.shape, np.nan)
        ang[idx] = np.unwrap(axis[idx])
        rate = _lagged_change(ang, lag)[:, 0]
        rate = rate[np.isfinite(rate)]
        if rate.size:
            out["body_rotation_speed"] = _safe(np.abs(rate).mean() / dt)

    # The swing of Base -> Tip against the body; its rate is the beat's, in the wave columns.
    tip = win["tip_ang"]
    idx = np.flatnonzero(np.isfinite(tip))
    if idx.size >= 8:
        out["tip_angle_std"] = _safe(np.std(np.unwrap(tip[idx])))
