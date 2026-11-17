"""Beat measurements per sliding window, computed as the flagella-analysis-pipeline computes them.

Each window goes through ``processFile.m`` of Fochler et al. (2025) with the smoothing tolerance in
pixels (the spatialScale^2 fix): the flagellum is a spline through Base, Flag1..Flag5 and Tip, smoothed,
put in the body frame, resampled to 200 points over 0.9 L and Fourier transformed in time. Lengths are
in px, frequencies in Hz. ``wave_direction`` +1 means the wave travels Base to Tip.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, PchipInterpolator

from .clipfeatures import BEAT_F_MIN_HZ, BEAT_MIN_CYCLES, _interp_nan, _on_grid, _safe

__all__ = [
    "spectral_series", "window_values", "WindowSource", "track_spectral_series",
    "flow_series", "segment_arcs", "smooth_rows", "linear_smooth", "frame_curves", "body_frame_curves",
    "analyse_window",
    "SERIES_COLUMNS", "DEFAULT_WINDOW", "DEFAULT_HOP", "DEFAULT_FLOW_FRAMES",
]

# 0.75 s at 100 fps.
DEFAULT_WINDOW = 75
# 50 ms at 100 fps.
DEFAULT_HOP = 5
# Frames one flow estimate is read over.
DEFAULT_FLOW_FRAMES = 5
MIN_WINDOW_FRAMES = 8

# processFile.m and batchProcess.m.
N_SAMPLES = 200                  # points along the analysed flagellum
LENGTH_SHARE = 0.9               # analysed share of the length; shorter frames are bad
BAND_HZ = 1.0                    # amplitude is summed over +-1 Hz of the peak
XY_TOLERANCE = 0.2               # spaps tolerance per px of length, in px^2
ANGLE_TOLERANCE = 0.01           # spaps tolerance of the angle profile, per sample spacing
MIN_GOOD_FRAMES = 30
BRIDGE_FRAMES = 2                # bad frames bridged when good frames lie this close on both sides
MAX_BRIDGED = 0.2                # more bridged frames than this marks a bad window
# Points the spline through the keypoints is smoothed on (the tracer gives about one per px).
DENSE_POINTS = 128

SERIES_COLUMNS = (
    # -- as the pipeline writes them -------------------------------------------
    "wave_freq_hz", "wave_freq_proximal_hz", "wave_freq_central_hz", "wave_freq_distal_hz",
    "wave_amp_mean", "wave_amp_asym", "wave_amp_para", "wave_amp_integral",
    "wave_phase_asym", "wave_phase_para", "wave_phase_rsq", "wave_phase_cycles",
    "wave_static_mean", "wave_freq_dominance", "wave_fit_error", "wave_xy_fit_error",
    "wave_disp_amp_max", "wave_angular_amp", "wave_length_px", "wave_bad_window",
    # -- read off the same analysis --------------------------------------------
    "wave_phase_slope", "wave_wavelength_px", "wave_wavelength_ratio", "wave_direction",
    "wave_speed_px_s", "wave_static_asym", "wave_static_para", "wave_disp_amp_mean",
    "wave_disp_amp_tip", "wave_excess", "wave_censored", "wave_resolvable_hz",
    "wave_interp_frac", "wave_window_frames",
    # -- direction without a spectrum, from the keypoint chords ---------------
    "wave_flow", "wave_flow_coherence", "wave_flow_direction",
    # -- stability across windows ------------------------------------------------
    "wave_freq_drift_hz", "wave_freq_track_sd_hz",
)

# Aggregated by mean (a share) rather than median.
_FRACTION_COLUMNS = ("wave_censored", "wave_bad_window")


# --------------------------------------------------------------------------- #
# smoothing: the spaps family, solved for many rows at once
# --------------------------------------------------------------------------- #
_BASIS: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _basis(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Eigen-decomposition of spaps' problem on n unit-spaced sites.

    The cubic smoothing spline minimises sum w (y - f)^2 + lam f' K f with K = Q R^-1 Q' (Reinsch)
    and trapezoid weights w; in z = sqrt(w) f it diagonalises once for every row.
    """
    if n not in _BASIS:
        q = np.zeros((n, n - 2))
        r = np.zeros((n - 2, n - 2))
        for j in range(n - 2):
            q[j, j], q[j + 1, j], q[j + 2, j] = 1.0, -2.0, 1.0
            r[j, j] = 2.0 / 3.0
            if j + 1 < n - 2:
                r[j, j + 1] = r[j + 1, j] = 1.0 / 6.0
        k = q @ np.linalg.solve(r, q.T)
        w = np.ones(n)
        w[0] = w[-1] = 0.5
        root = np.sqrt(w)
        lam, vec = np.linalg.eigh(k / root[:, None] / root[None, :])
        # The constant and linear modes are free; round-off leaves them slightly off zero.
        _BASIS[n] = (np.where(lam > 1e-10 * lam.max(), lam, 0.0), vec, root)
    return _BASIS[n]


def smooth_rows(values: np.ndarray, budget) -> np.ndarray:
    """spaps(x, y, tol) for rows sampled at unit spacing: the smoothest cubic spline values whose
    trapezoid-weighted squared residual equals ``budget``, or the weighted line if that is within it.
    """
    values = np.asarray(values, dtype=float)
    rows, n = values.shape
    if n < 3 or rows == 0:
        return values.copy()
    lam, vec, root = _basis(n)
    coef = (values * root) @ vec
    power = coef ** 2
    budget = np.broadcast_to(np.asarray(budget, dtype=float), (rows,))
    # The residual rises with log mu to the line fit; rows within budget of the line take it.
    # Otherwise safeguarded Newton on log mu inside a bracket that always holds the root, each
    # row until its residual is within 1e-10 of the budget or of rounding.
    line = (power * (lam > 0)).sum(axis=1)
    lo, hi = np.full(rows, -60.0), np.full(rows, 60.0)
    t = np.zeros(rows)
    log_lam = np.log(np.where(lam > 0, lam, 1.0))[None, :]
    done = line <= budget
    tol = 1e-10 * budget + 1e-13 * line
    for _ in range(40):
        g = 1.0 / (1.0 + np.exp(np.clip(-(t[:, None] + log_lam), -700.0, 700.0)))
        g = np.where(lam[None, :] > 0, g, 0.0)
        resid = (g * g * power).sum(axis=1) - budget
        done |= np.abs(resid) <= tol
        if done.all():
            break
        lo = np.where(resid < 0, t, lo)
        hi = np.where(resid >= 0, t, hi)
        slope = (2.0 * g * g * (1.0 - g) * power).sum(axis=1)
        newton = t - resid / np.where(slope > 1e-300, slope, np.nan)
        inside = np.isfinite(newton) & (newton >= lo) & (newton <= hi)
        t = np.where(done, t, np.where(inside, newton, 0.5 * (lo + hi)))
    mu = np.where((line <= budget)[:, None], np.inf, np.exp(t)[:, None] * lam[None, :])
    with np.errstate(invalid="ignore"):
        shrink = np.where(lam[None, :] > 0, 1.0 / (1.0 + mu), 1.0)
    return ((coef * shrink) @ vec.T) / root


def linear_smooth(values: np.ndarray, cutoff) -> np.ndarray:
    """Rows smoothed by a cubic smoothing spline of fixed strength, on unit-spaced samples.

    A wave of ``cutoff`` samples keeps about half its amplitude, longer ones more; ``cutoff`` is
    one number or one per row. Smoothing x and y alike leaves a rotated curve rotated.
    """
    values = np.asarray(values, dtype=float)
    if values.shape[1] < 3 or values.shape[0] == 0:
        return values.copy()
    lam, vec, root = _basis(values.shape[1])
    cutoff = np.asarray(cutoff, dtype=float)
    if cutoff.ndim:
        cutoff = cutoff[:, None]
    shrink = 1.0 / (1.0 + lam / (2.0 * np.pi / cutoff) ** 4)
    return (((values * root) @ vec) * shrink @ vec.T) / root


# --------------------------------------------------------------------------- #
# helpers shared with the flow readout and trackseries
# --------------------------------------------------------------------------- #
def _slice_starts(n: int, m: int, hop: int) -> np.ndarray:
    """Starts of the windows lying wholly inside ``n`` frames whose centres are multiples of ``hop``."""
    mid = m // 2
    p_first = -(-mid // hop)
    p_last = (n - m + mid) // hop
    if p_last < p_first:
        return np.empty(0, dtype=int)
    return np.arange(p_first, p_last + 1, dtype=int) * hop - mid


def _centred_sum(values: np.ndarray, half: int) -> np.ndarray:
    """Sliding sum over ``[i - half, i + half]`` along axis 0, clipped at the ends."""
    n = values.shape[0]
    zero = np.zeros((1,) + values.shape[1:], dtype=float)
    cumulative = np.concatenate([zero, np.cumsum(values, axis=0)], axis=0)
    idx = np.arange(n)
    lo = np.clip(idx - half, 0, n)
    hi = np.clip(idx + half + 1, 0, n)
    return cumulative[hi] - cumulative[lo]


def segment_arcs(chords: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Arc length of each segment, taking it as a circular arc through its two nodes.

    A segment turns by the mean of the chord-angle changes at its ends; a chord of turning tau
    spans an arc ``(tau / 2) / sin(tau / 2)`` times longer.
    """
    chords = np.asarray(chords, dtype=float)
    if chords.shape[-1] < 2:
        return chords.copy()
    turn = np.diff(angles, axis=-1)
    tau = np.empty_like(chords)
    tau[..., 1:-1] = 0.5 * (turn[..., :-1] + turn[..., 1:])
    tau[..., 0], tau[..., -1] = turn[..., 0], turn[..., -1]
    half = np.minimum(np.abs(tau) / 2.0, 3.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        gain = np.where(half > 1e-9, half / np.sin(np.where(half > 1e-9, half, 1.0)), 1.0)
    return chords * gain


def _chord_angles(px: np.ndarray, py: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Chord tangent angles of Base..Tip against Head to Base, continuous in time at the base."""
    ox, oy = px[:, 1], py[:, 1]
    axis = np.arctan2(oy - py[:, 0], ox - px[:, 0])
    c, s = np.cos(-axis)[:, None], np.sin(-axis)[:, None]
    rx, ry = px[:, 1:] - ox[:, None], py[:, 1:] - oy[:, None]
    fx, fy = rx * c - ry * s, rx * s + ry * c
    dx, dy = np.diff(fx, axis=1), np.diff(fy, axis=1)
    raw = np.arctan2(dy, dx)
    base = raw[:, :1].copy()
    finite = np.isfinite(base[:, 0])
    if finite.sum() >= 2:
        base[finite, 0] = np.unwrap(base[finite, 0])
    steps = np.angle(np.exp(1j * np.diff(raw, axis=1)))
    psi = np.concatenate([base, base + np.cumsum(steps, axis=1)], axis=1)
    seg = segment_arcs(np.hypot(dx, dy), psi)
    return psi, np.cumsum(seg, axis=1) - seg / 2.0


def flow_series(
    psi: np.ndarray, arc_px: np.ndarray, complete: np.ndarray,
    frames: int = DEFAULT_FLOW_FRAMES,
) -> np.ndarray:
    """Per-frame wave direction as the correlation ``-<psi_t psi_s>`` in [-1, 1].

    Sign as ``wave_direction`` (+1 Base to Tip); NaN where the stencil is mostly interpolated.
    """
    psi = np.asarray(psi, dtype=float)
    n = psi.shape[0]
    out = np.full(n, np.nan)
    half = max(1, int(frames) // 2)
    if psi.ndim != 2 or n < 2 * half + 1 or psi.shape[1] < 2:
        return out
    finite = np.isfinite(psi)
    filled = np.where(finite, psi, 0.0)
    seen = _centred_sum(finite.astype(float), half)
    with np.errstate(invalid="ignore", divide="ignore"):
        static = np.where(seen > 0, _centred_sum(filled, half) / np.maximum(seen, 1), np.nan)
    psi_t = np.gradient(psi, axis=0)
    step = np.diff(arc_px, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        psi_s = np.diff(psi - static, axis=1) / np.where(np.abs(step) > 1e-9, step, np.nan)
    psi_t = 0.5 * (psi_t[:, :-1] + psi_t[:, 1:])
    good = np.isfinite(psi_t) & np.isfinite(psi_s)
    a, b = np.where(good, psi_t, 0.0), np.where(good, psi_s, 0.0)
    cross = _centred_sum((a * b).sum(axis=1), half)
    aa = _centred_sum((a * a).sum(axis=1), half)
    bb = _centred_sum((b * b).sum(axis=1), half)
    pairs = _centred_sum(good.sum(axis=1).astype(float), half)
    detected = _centred_sum(np.asarray(complete, dtype=float), half)
    span = _centred_sum(np.ones(n), half)
    with np.errstate(invalid="ignore", divide="ignore"):
        local_fill = 1.0 - detected / np.maximum(span, 1.0)
        denominator = np.sqrt(aa * bb)
        out = np.where((denominator > 1e-12) & (pairs >= 4) & (local_fill <= MAX_BRIDGED),
                       -cross / np.where(denominator > 0, denominator, np.nan), np.nan)
    out = np.clip(out, -1.0, 1.0)
    out[:half] = np.nan
    out[n - half:] = np.nan
    return out


def _read_at(values: np.ndarray, centres: np.ndarray, reach: int) -> np.ndarray:
    """Median of a per-frame series over ``+-reach`` frames of each centre."""
    out = np.full(centres.size, np.nan)
    n = values.shape[0]
    for i, centre in enumerate(centres):
        lo, hi = max(0, int(centre) - reach), min(n, int(centre) + reach + 1)
        block = values[lo:hi]
        block = block[np.isfinite(block)]
        if block.size:
            out[i] = float(np.median(block))
    return out


def _local_spread(values: np.ndarray, radius: int) -> np.ndarray:
    """``1.4826 * MAD`` of ``values`` over ``+-radius`` neighbours; NaN below three finite ones."""
    pad = np.full(int(radius), np.nan)
    block = np.lib.stride_tricks.sliding_window_view(
        np.concatenate([pad, values, pad]), 2 * int(radius) + 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(block, axis=1)
        mad = np.nanmedian(np.abs(block - median[:, None]), axis=1)
    return np.where(np.isfinite(block).sum(axis=1) >= 3, 1.4826 * mad, np.nan)


# --------------------------------------------------------------------------- #
# per frame: the smoothed flagellum in the body frame (processFile.m lines 37-73)
# --------------------------------------------------------------------------- #
def frame_curves(px: np.ndarray, py: np.ndarray, cutoff_segments: float | None = None,
                 cutoff_px: float | None = None, whole_trace: bool = True) -> dict[str, np.ndarray]:
    """Smoothed flagellum per frame in the body frame: x, y, arc (frames, DENSE_POINTS) and lengths.

    The flagellum is a cubic spline through Base..Tip by node index; the body centre is the
    Head-Base midpoint. By default it is smoothed as the pipeline does (``spaps``, axis to the
    smoothed base). With ``cutoff_segments`` it is smoothed by :func:`linear_smooth` with that
    cutoff, in segments between keypoints, and the axis runs to the Base keypoint itself;
    ``cutoff_px`` does the same with a cutoff in pixels, for a densely sampled centerline.

    ``overall`` is the whole trace for the pipeline's second bad-frame test: Head-Base plus the
    flagellum, standing in for the skeleton through body and flagellum. ``whole_trace=False``
    makes it the flagellum itself, so that test repeats the flagellum one and drops out: a
    centerline has no traced body (processFileCenterline.m, which drops the test too).
    """
    n_frames = px.shape[0]
    n = DENSE_POINTS
    xs = np.full((n_frames, n), np.nan)
    ys = np.full((n_frames, n), np.nan)
    valid = np.isfinite(px).all(axis=1) & np.isfinite(py).all(axis=1)
    rows = np.flatnonzero(valid)
    if rows.size:
        # One spline for every frame, by node index: the nodes sit about evenly along the flagellum.
        k = px.shape[1] - 1
        u = np.arange(k, dtype=float)
        fine = np.linspace(0.0, k - 1.0, 4 * n)
        cx = CubicSpline(u, px[rows, 1:], axis=1)(fine)
        cy = CubicSpline(u, py[rows, 1:], axis=1)(fine)
        # Resampled evenly in arclength, as the tracer's pixels are.
        arc = np.concatenate([np.zeros((rows.size, 1)),
                              np.cumsum(np.hypot(np.diff(cx, axis=1), np.diff(cy, axis=1)), axis=1)], 1)
        target = arc[:, -1:] * np.linspace(0.0, 1.0, n)[None, :]
        for j, r in enumerate(rows):
            xs[r] = np.interp(target[j], arc[j], cx[j])
            ys[r] = np.interp(target[j], arc[j], cy[j])
        degenerate = arc[:, -1] <= 1e-9
        valid[rows[degenerate]] = False
        rows = rows[~degenerate]
    out = {"x": np.full((n_frames, n), np.nan), "y": np.full((n_frames, n), np.nan),
           "arc": np.full((n_frames, n), np.nan), "length": np.full(n_frames, np.nan),
           "overall": np.full(n_frames, np.nan)}
    if rows.size == 0:
        return out
    cutoff = None if cutoff_segments is None else cutoff_segments * (n - 1) / (px.shape[1] - 2)
    if cutoff_px is not None:
        # Each frame's curve has n points over its own length.
        length = np.hypot(np.diff(xs[rows], axis=1), np.diff(ys[rows], axis=1)).sum(axis=1)
        cutoff = cutoff_px * (n - 1) / length
    framed = body_frame_curves(xs[rows], ys[rows], 0.5 * (px[rows, 0] + px[rows, 1]),
                               0.5 * (py[rows, 0] + py[rows, 1]), cutoff=cutoff,
                               axis_to_base=cutoff is not None)
    for key in ("x", "y", "arc", "length"):
        out[key][rows] = framed[key]
    body = np.hypot(px[rows, 0] - px[rows, 1], py[rows, 0] - py[rows, 1])
    chain = np.hypot(np.diff(px[rows, 1:], axis=1), np.diff(py[rows, 1:], axis=1)).sum(axis=1)
    out["overall"][rows] = body + chain if whole_trace else out["length"][rows]
    return out


def body_frame_curves(x: np.ndarray, y: np.ndarray, centre_x: np.ndarray, centre_y: np.ndarray,
                      cutoff: float | None = None,
                      axis_to_base: bool = False) -> dict[str, np.ndarray]:
    """Flagella (frames, points; base to tip, even in arclength, px) smoothed and in the body frame.

    Smoothed as spaps with tolerance 0.2 L px^2, or by :func:`linear_smooth` with ``cutoff``
    points (one number, or one per frame), then moved so the smoothed base is the origin and the axis from the body centre to
    it is +x. With ``axis_to_base`` the axis runs to the unsmoothed base instead, which does not
    sway with the smoothing of the beat.
    """
    n = x.shape[1]
    if cutoff is None:
        # spaps tolerance 0.2 L (px) with trapezoid weights of spacing L / (n - 1): 0.2 (n - 1) per unit.
        smoothed = smooth_rows(np.vstack([x, y]), XY_TOLERANCE * (n - 1))
    else:
        cutoff = np.asarray(cutoff, dtype=float)
        smoothed = linear_smooth(np.vstack([x, y]),
                                 np.concatenate([cutoff, cutoff]) if cutoff.ndim else cutoff)
    sx, sy = smoothed[:x.shape[0]], smoothed[x.shape[0]:]
    bx, by = sx[:, :1], sy[:, :1]
    ax, ay = (x[:, 0], y[:, 0]) if axis_to_base else (bx[:, 0], by[:, 0])
    ang = np.arctan2(ay - centre_y, ax - centre_x)
    c, s = np.cos(-ang)[:, None], np.sin(-ang)[:, None]
    rx, ry = sx - bx, sy - by
    fx, fy = c * rx - s * ry, s * rx + c * ry
    arc = np.concatenate([np.zeros((x.shape[0], 1)),
                          np.cumsum(np.hypot(np.diff(fx, axis=1), np.diff(fy, axis=1)), axis=1)], 1)
    return {"x": fx, "y": fy, "arc": arc, "length": arc[:, -1]}


# --------------------------------------------------------------------------- #
# per window: the Fourier analysis (processFile.m from line 74)
# --------------------------------------------------------------------------- #
def _movmean_omitnan(v: np.ndarray, k: int) -> np.ndarray:
    """MATLAB movmean(v, k, 'omitnan'): an even k looks one further back than ahead."""
    n = v.size
    before, after = k // 2, (k - 1) // 2
    finite = np.isfinite(v)
    csum = np.concatenate([[0.0], np.cumsum(np.where(finite, v, 0.0))])
    ccnt = np.concatenate([[0], np.cumsum(finite)])
    idx = np.arange(n)
    lo, hi = np.clip(idx - before, 0, n), np.clip(idx + after + 1, 0, n)
    count = ccnt[hi] - ccnt[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, (csum[hi] - csum[lo]) / np.maximum(count, 1), np.nan)


def _good_frame_range(bad: np.ndarray) -> tuple[np.ndarray, int, bool]:
    """computeGoodFrameRange.m: the largest block of good or bridged frames, bridged count, bad flag."""
    good = ~bad
    n = good.size
    left = np.array([good[max(0, i - BRIDGE_FRAMES): i + 1].any() for i in range(n)])
    right = np.array([good[i: min(n, i + BRIDGE_FRAMES + 1)].any() for i in range(n)])
    allowable = left & right
    added = good ^ allowable
    good = good | allowable
    padded = np.concatenate([[0], good.astype(int), [0]])
    zeros = np.flatnonzero(padded == 0)
    block = int(np.argmax(np.diff(zeros)))
    frames = np.arange(zeros[block], zeros[block + 1] - 1)
    if frames.size == 0:
        return frames, 0, True
    bridged = int(added[frames].sum())
    return frames, bridged, bridged > MAX_BRIDGED * frames.size


def _fill_pchip(v: np.ndarray, bad: np.ndarray) -> np.ndarray:
    """fillmissing(v, 'pchip', 1) over the bad rows, extrapolating at the ends."""
    if not bad.any():
        return v
    t = np.arange(v.shape[0])
    ok = ~bad
    out = v.copy()
    if ok.sum() >= 2:
        out[bad] = PchipInterpolator(t[ok], v[ok], axis=0, extrapolate=True)(t[bad])
    return out


def _legendre(data: np.ndarray) -> np.ndarray:
    """legendreCoeffs.m: (2n+1) trapz(xs, data * l_n) on equally spaced [0, 1]."""
    xs = np.linspace(0.0, 1.0, data.shape[-1])
    ell = np.stack([np.ones_like(xs), 2 * xs - 1, 6 * xs ** 2 - 6 * xs + 1])
    return np.array([1.0, 3.0, 5.0]) * np.trapezoid(data[None, :] * ell, xs, axis=-1)


def _cumulative_xy(angles: np.ndarray, ds: float) -> tuple[np.ndarray, np.ndarray]:
    """anglesToSpatial.m: positions from the base by the trapezoid rule."""
    zero = np.zeros((angles.shape[0], 1))
    c, s = np.cos(angles), np.sin(angles)
    x = np.concatenate([zero, np.cumsum(0.5 * (c[:, 1:] + c[:, :-1]) * ds, axis=1)], axis=1)
    y = np.concatenate([zero, np.cumsum(0.5 * (s[:, 1:] + s[:, :-1]) * ds, axis=1)], axis=1)
    return x, y


_DIRECTIONS = np.linspace(0.0, np.pi, 64, endpoint=False)
_NO_WAVE_RAD = 1e-6              # phase range along s below which a window has no wave


def _trapz_weights(s: np.ndarray) -> np.ndarray:
    """Weights w with w @ f equal to trapz(s, f)."""
    d = np.diff(s)
    w = np.zeros_like(s)
    w[:-1] += d / 2.0
    w[1:] += d / 2.0
    return w


def _spatial_amplitudes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """computeSpatialAmplitudes.m: half the largest distance between any two frames, per point.

    The largest spread along 64 directions, within 0.03% of the largest pairwise distance.
    """
    proj = (x[:, :, None] * np.cos(_DIRECTIONS) + y[:, :, None] * np.sin(_DIRECTIONS))
    return (proj.max(axis=0) - proj.min(axis=0)).max(axis=1) / 2.0


def _phase_line(phase: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    """Slope of a least-squares line through the phase, and its R^2 (1 only for an exactly flat phase)."""
    slope, intercept = np.polyfit(s, phase, 1)
    if np.all(phase == phase.mean()):
        return float(slope), 1.0
    resid = phase - (slope * s + intercept)
    spread = phase - phase.mean()
    return float(slope), float(1.0 - (resid ** 2).sum() / (spread ** 2).sum())


_ROTATION_STEP = 0.02           # rad, grid of the rotation search in _xy_fit_error


def _xy_fit_error(x, y, xr, yr, s, t) -> float:
    """computeQualityOfFitXY.m: best rotation of the reconstruction per frame, then the overall error."""
    norm = np.sqrt(x ** 2 + y ** 2).sum(axis=1)
    norm = np.where(norm > 0, norm, 1.0)
    w = _trapz_weights(s)

    def score(theta):
        c, sn = np.cos(theta)[:, None], np.sin(theta)[:, None]
        xt, yt = c * xr - sn * yr, sn * xr + c * yr
        return np.sqrt((xt - x) ** 2 + (yt - y) ** 2) @ w / norm

    # fminunc from 0 ends in the minimum downhill of no rotation: walk the grid from 0 to it,
    # then refine it by golden section.
    grid = np.arange(-np.pi, np.pi + 1e-9, _ROTATION_STEP)
    here = np.full(x.shape[0], int(np.argmin(np.abs(grid))))
    f_here = score(grid[here])
    for _ in range(grid.size):
        left = np.maximum(here - 1, 0)
        right = np.minimum(here + 1, grid.size - 1)
        f_left, f_right = score(grid[left]), score(grid[right])
        down_left, down_right = f_left < f_here, f_right < f_here
        to_left = down_left & ~(down_right & (f_right < f_left))
        to_right = down_right & ~to_left
        if not (to_left | to_right).any():
            break
        here = np.where(to_left, left, np.where(to_right, right, here))
        f_here = np.where(to_left, f_left, np.where(to_right, f_right, f_here))
    a, b = grid[here] - _ROTATION_STEP, grid[here] + _ROTATION_STEP
    g = (np.sqrt(5.0) - 1.0) / 2.0
    c1, c2 = b - g * (b - a), a + g * (b - a)
    f1, f2 = score(c1), score(c2)
    for _ in range(25):
        left = f1 < f2
        b, a = np.where(left, c2, b), np.where(left, a, c1)
        c1, c2 = b - g * (b - a), a + g * (b - a)
        f1, f2 = score(c1), score(c2)
    theta = 0.5 * (a + b)
    c, sn = np.cos(theta)[:, None], np.sin(theta)[:, None]
    xt, yt = c * xr - sn * yr, sn * xr + c * yr
    err = np.trapezoid(np.trapezoid((x - xt) ** 2 + (y - yt) ** 2, s, axis=1), t)
    ref = np.trapezoid(np.trapezoid(x ** 2 + y ** 2, s, axis=1), t)
    return float(np.sqrt(err / ref)) if ref > 0 else np.nan


def analyse_window(curves: dict[str, np.ndarray], lo: int, hi: int, fps: float) -> dict:
    """The pipeline's outputs for frames [lo, hi), as ``wave_*`` columns."""
    out: dict[str, float] = {}
    lengths = curves["length"][lo:hi]
    overall = curves["overall"][lo:hi]
    n_frames = hi - lo
    if np.isfinite(lengths).sum() < 2:
        return out
    k = max(1, int(np.floor(n_frames / 10.0 + 0.5)))
    flag_len = float(np.nanmax(_movmean_omitnan(lengths, k)))
    overall_len = float(np.nanmax(_movmean_omitnan(overall, k)))
    with np.errstate(invalid="ignore"):
        bad = ~(lengths >= LENGTH_SHARE * flag_len) | ~(overall >= LENGTH_SHARE * overall_len)
    frames, bridged, bad_window = _good_frame_range(bad)
    if frames.size < MIN_GOOD_FRAMES:
        bad_window = True
    if frames.size < MIN_WINDOW_FRAMES:
        return out
    s = np.linspace(0.0, LENGTH_SHARE * flag_len, N_SAMPLES)
    x = np.full((frames.size, N_SAMPLES), np.nan)
    y = np.full((frames.size, N_SAMPLES), np.nan)
    for j, f in enumerate(frames + lo):
        if np.isfinite(curves["length"][f]):
            arc = curves["arc"][f]
            x[j] = np.interp(s, arc, curves["x"][f], left=np.nan, right=np.nan)
            y[j] = np.interp(s, arc, curves["y"][f], left=np.nan, right=np.nan)
    bad = bad[frames] | ~np.isfinite(x).all(axis=1)
    if (~bad).sum() < 2:
        return out
    x[bad], y[bad] = np.nan, np.nan
    x, y = _fill_pchip(x, bad), _fill_pchip(y, bad)
    angles = np.unwrap(np.arctan2(np.gradient(y, s, axis=1), np.gradient(x, s, axis=1)), axis=1)
    angles = smooth_rows(angles, ANGLE_TOLERANCE)
    ds = s[1] - s[0]
    n = angles.shape[0]
    xs, ys = _cumulative_xy(angles, ds)

    # fourierSpectra.m: no window, an odd count padded with one zero frame.
    L = n + (n % 2)
    freqs = fps * np.arange(L // 2 + 1) / L
    spec = np.fft.fft(angles, n=L, axis=0)
    amp = np.abs(spec / L)[: L // 2 + 1]
    amp[1:-1] *= 2.0
    w = int(np.floor(BAND_HZ / (freqs[1] - freqs[0])))
    if amp.shape[0] <= w + 1:
        return out
    csum = np.vstack([np.zeros((1, N_SAMPLES)), np.cumsum(amp, axis=0)])
    idx = np.arange(amp.shape[0])
    a_lo, a_hi = np.clip(idx - w, 0, None), np.clip(idx + w + 1, None, amp.shape[0])
    windowed = (csum[a_hi] - csum[a_lo]) / (a_hi - a_lo)[:, None] * (2 * w + 1)
    weights = _trapz_weights(s)
    summed = amp @ weights
    wsummed = windowed @ weights

    def peak(cols: np.ndarray) -> int:
        part = (windowed[:, cols] @ _trapz_weights(s[cols]))[w + 1:]
        # Ties (a flagellum that does not move) go to the first bin, as MATLAB's max does.
        top = part.max()
        return w + 1 + int(np.argmax(part >= top - 1e-12 * max(abs(top), 1e-300)))

    k0 = peak(np.ones(N_SAMPLES, bool))
    third = s.max() / 3.0
    proximal, distal = s < third, s >= 2 * third
    profile = windowed[k0]
    phase = np.unwrap(np.angle(spec[k0]))
    slope, rsq = _phase_line(phase, s)
    amp_c, phase_c = _legendre(profile), _legendre(phase)
    static_summed = np.trapezoid(windowed[0], s)

    # Single-frequency reconstruction: DC, the band around the peak, its mirror and the low bins.
    keep = np.zeros(L, bool)
    keep[0] = True
    keep[k0 + np.arange(-w, w + 1)] = True
    keep[(L - k0 + np.arange(-w, w + 1)) % L] = True
    keep[1: w + 1] = True
    keep[L - 1 - np.arange(w)] = True
    rec = np.real(np.fft.ifft(np.where(keep[:, None], spec, 0.0), axis=0))[:n]
    t = np.arange(n) / fps
    err = np.trapezoid(np.trapezoid((angles - rec) ** 2, s, axis=1), t)
    ref = np.trapezoid(np.trapezoid(angles ** 2, s, axis=1), t)
    xr, yr = _cumulative_xy(rec, ds)
    spatial = _spatial_amplitudes(xs, ys)
    band = wsummed[w + 1:]
    baseline = float(np.median(band))

    out.update({
        "wave_freq_hz": freqs[k0],
        "wave_freq_proximal_hz": freqs[peak(proximal)],
        "wave_freq_central_hz": freqs[peak(~(proximal | distal))],
        "wave_freq_distal_hz": freqs[peak(distal)],
        "wave_amp_mean": amp_c[0], "wave_amp_asym": amp_c[1], "wave_amp_para": amp_c[2],
        "wave_amp_integral": wsummed[k0],
        "wave_phase_asym": phase_c[1], "wave_phase_para": phase_c[2],
        "wave_phase_rsq": rsq,
        "wave_phase_cycles": (phase.max() - phase.min()) / (2 * np.pi),
        "wave_static_mean": summed[0] / flag_len,
        "wave_freq_dominance": wsummed[k0] / (summed.sum() - static_summed),
        "wave_fit_error": float(np.sqrt(err / ref)) if ref > 0 else np.nan,
        "wave_xy_fit_error": _xy_fit_error(xs, ys, xr, yr, s, t),
        "wave_disp_amp_max": spatial.max(),
        "wave_angular_amp": ((angles.max(axis=0) - angles.min(axis=0)) / 2.0).max(),
        "wave_length_px": flag_len,
        "wave_bad_window": float(bad_window),
        "wave_phase_slope": slope,
        "wave_disp_amp_mean": spatial.mean(), "wave_disp_amp_tip": spatial[-1],
        "wave_excess": band.max() / baseline if baseline > 0 else np.nan,
        "wave_interp_frac": bridged / frames.size,
        "wave_window_frames": float(n),
    })
    # The time-mean angle profile, sign chosen so its mean is positive (a mirror image flips it).
    static_c = _legendre(angles.mean(axis=0))
    flip = -1.0 if static_c[0] < 0 else 1.0
    out["wave_static_asym"], out["wave_static_para"] = flip * static_c[1], flip * static_c[2]
    # A flagellum the smoothing has made straight has no wave along it; its slope is rounding noise.
    if abs(slope) > 1e-12 and np.ptp(phase) > _NO_WAVE_RAD:
        wavelength = 2.0 * np.pi / abs(slope)
        out["wave_wavelength_px"] = wavelength
        out["wave_wavelength_ratio"] = wavelength / flag_len
        out["wave_direction"] = -float(np.sign(slope))
        out["wave_speed_px_s"] = out["wave_direction"] * wavelength * freqs[k0]
    floor = max(BEAT_F_MIN_HZ, BEAT_MIN_CYCLES / (n / fps))
    out["wave_resolvable_hz"] = floor
    out["wave_censored"] = float(freqs[k0] < floor)
    return out


# --------------------------------------------------------------------------- #
# the series
# --------------------------------------------------------------------------- #
def spectral_series(
    px: np.ndarray, py: np.ndarray, fps: float, window: int = DEFAULT_WINDOW,
    hop: int = DEFAULT_HOP, flow_frames: int = DEFAULT_FLOW_FRAMES, whole_trace: bool = True,
) -> pd.DataFrame:
    """Beat measurements (:data:`SERIES_COLUMNS`) per sliding window, indexed by centre frame.

    ``px``/``py`` are (n_frames, n_keypoints) on the full frame grid, NaN where undetected:
    Head, Base, then the flagellum to its tip. ``whole_trace`` as for :func:`frame_curves`.
    """
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    empty = pd.DataFrame(columns=list(SERIES_COLUMNS), dtype=float)
    empty.index.name = "frame"
    if px.ndim != 2 or px.shape != py.shape or px.shape[1] < 4:
        return empty
    n_frames = px.shape[0]
    fps = max(float(fps), 1e-6)
    window, hop = int(max(MIN_WINDOW_FRAMES, window)), int(max(1, hop))
    if n_frames < MIN_WINDOW_FRAMES:
        return empty
    m = int(min(window, n_frames))
    starts = _slice_starts(n_frames, m, hop)
    if starts.size == 0:
        starts = np.array([(n_frames - m) // 2])
    centres = starts + m // 2

    curves = frame_curves(px, py, whole_trace=whole_trace)
    rows = [analyse_window(curves, int(c), int(c) + m, fps) for c in starts]
    out = {c: np.array([r.get(c, np.nan) for r in rows], dtype=float) for c in SERIES_COLUMNS}

    # Direction without a spectrum, from the keypoint chords over gaps filled in time.
    complete = np.isfinite(px).all(axis=1) & np.isfinite(py).all(axis=1)
    psi, arc = _chord_angles(_interp_nan(px), _interp_nan(py))
    flow = flow_series(psi, arc, complete, flow_frames)
    reach = max(int(flow_frames) // 2, int(hop) // 2)
    out["wave_flow"] = _read_at(flow, centres, reach)
    out["wave_flow_coherence"] = _read_at(np.abs(flow), centres, reach)
    direction = np.sign(out["wave_flow"])
    out["wave_flow_direction"] = np.where(direction == 0.0, np.nan, direction)

    # Frequency stability over windows that measured one.
    steady = np.where((out["wave_censored"] == 1.0) | (out["wave_bad_window"] == 1.0), np.nan,
                      out["wave_freq_hz"])
    out["wave_freq_drift_hz"] = _local_spread(steady, max(1, m // hop))
    good = steady[np.isfinite(steady)]
    if good.size >= 3:
        out["wave_freq_track_sd_hz"] = np.full(centres.size, float(good.std(ddof=1)))
    frame = pd.DataFrame({c: out[c] for c in SERIES_COLUMNS}, index=pd.Index(centres, name="frame"))
    frame.attrs["window"], frame.attrs["hop"] = m, hop
    return frame.replace([np.inf, -np.inf], np.nan)


def track_spectral_series(
    sub: pd.DataFrame, n_keypoints: int, fps: float, window: int = DEFAULT_WINDOW,
    hop: int = DEFAULT_HOP, n_frames: int | None = None, f_start: int | None = None,
    flow_frames: int = DEFAULT_FLOW_FRAMES, whole_trace: bool = True,
) -> pd.DataFrame:
    """:func:`spectral_series` from one track's rows, laid on the full frame grid first."""
    sub = sub.sort_values("frame")
    frames = sub["frame"].to_numpy(dtype=int)
    if frames.size == 0:
        return spectral_series(np.empty((0, 0)), np.empty((0, 0)), fps, window, hop, flow_frames,
                               whole_trace)
    if f_start is None:
        f_start = int(frames.min())
    if n_frames is None:
        n_frames = int(frames.max()) - int(f_start) + 1
    kx = [f"kp_x_{k}" for k in range(n_keypoints)]
    ky = [f"kp_y_{k}" for k in range(n_keypoints)]
    px = _on_grid(sub, kx, int(n_frames), int(f_start))
    py = _on_grid(sub, ky, int(n_frames), int(f_start))
    series = spectral_series(px, py, fps, window, hop, flow_frames, whole_trace)
    series.index = pd.Index(series.index.to_numpy(dtype=int) + int(f_start), name="frame")
    return series


# --------------------------------------------------------------------------- #
# reading the series out over one annotation window
# --------------------------------------------------------------------------- #
class WindowSource:
    """:func:`window_values` with the per-track work done once."""

    __slots__ = ("centres", "values", "columns", "is_fraction")

    def __init__(self, series: pd.DataFrame | None):
        present = [] if series is None else [c for c in SERIES_COLUMNS if c in series.columns]
        self.columns = present
        if series is None or len(series) == 0:
            self.centres = np.empty(0, dtype=float)
            self.values = np.empty((0, len(present)), dtype=float)
        else:
            self.centres = series.index.to_numpy(dtype=float)
            self.values = series[present].to_numpy(dtype=float)
        self.is_fraction = np.array([c in _FRACTION_COLUMNS for c in present], dtype=bool)

    def at(self, f_start: int, f_end: int) -> dict[str, float]:
        """Median of the rows centred in ``[f_start, f_end)``, else of the nearest row.

        Directions and speed are recomputed from the median signed values.
        """
        out: dict[str, float] = {c: np.nan for c in SERIES_COLUMNS}
        out["n_windows"] = 0.0
        out["centre_offset"] = np.nan
        if self.centres.size == 0:
            return out
        target = (float(f_start) + float(f_end)) / 2.0
        inside = (self.centres >= float(f_start)) & (self.centres < float(f_end))
        rows = (np.flatnonzero(inside) if inside.any()
                else np.array([int(np.argmin(np.abs(self.centres - target)))]))
        out["n_windows"] = float(rows.size)
        out["centre_offset"] = float(np.median(self.centres[rows]) - target)
        block = self.values[rows]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pooled = np.where(np.isnan(block).all(axis=0), np.nan,
                              np.where(self.is_fraction, np.nanmean(block, axis=0),
                                       np.nanmedian(block, axis=0)))
        for column, value in zip(self.columns, pooled):
            if np.isfinite(value):
                out[column] = float(value)
        slope = out.get("wave_phase_slope", np.nan)
        out["wave_direction"] = _safe(
            -float(np.sign(slope)) if np.isfinite(slope) and slope != 0.0 else np.nan)
        flow = out.get("wave_flow", np.nan)
        out["wave_flow_direction"] = _safe(
            float(np.sign(flow)) if np.isfinite(flow) and flow != 0.0 else np.nan)
        out["wave_speed_px_s"] = _safe(
            out["wave_direction"] * out.get("wave_wavelength_px", np.nan) * out.get("wave_freq_hz", np.nan))
        return out


def window_values(series: pd.DataFrame, f_start: int, f_end: int) -> dict[str, float]:
    """One window of one track's spectral series. :class:`WindowSource` for many."""
    return WindowSource(series).at(f_start, f_end)
