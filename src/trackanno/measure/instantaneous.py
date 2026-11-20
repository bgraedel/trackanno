"""Beat readouts per frame, from the ridge of a Morlet wavelet transform of the tangent angle.

The flagellum is the spline through the keypoints, smoothed with a fixed cutoff of one segment between
keypoints, or of ``SMOOTHING_UM`` along a dense centerline (the same in x and y, so the cell's
orientation does not matter), and its tangent angle is taken
per frame at ``N_POINTS`` along 0.9 of that frame's length. After a 1 Hz high-pass, each point's angle goes
through a Morlet transform (omega0 = 6) on a log grid of frequencies. The ridge is the frequency path
through the power summed along s that pays ``JUMP_COST`` per octave it moves. At the ridge each frame
has an amplitude and a phase profile along s, read as the pipeline reads its spectrum at the peak; the
frequency is the phase advance per frame there.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from . import wavespectra as ws
from .clipfeatures import BEAT_F_MIN_HZ, _on_grid

__all__ = [
    "beat_series", "beat_from_curves", "track_beat_series", "window_values", "angle_field",
    "SERIES_COLUMNS", "CLIP_COLUMNS", "CONFIDENT_EXCESS",
]

N_POINTS = 100                   # points along 0.9 L
SMOOTHING_SEGMENTS = 1.0         # smoothing cutoff for keypoints, in segments between them
SMOOTHING_UM = 2.0               # smoothing cutoff for a centerline (about one pose segment)
OMEGA0 = 6.0                     # Morlet: the window spans about one period per radian of omega0
F_LOW = 1.5                      # Hz, lowest ridge frequency
F_HIGH_SHARE = 0.35              # of the frame rate, highest ridge frequency
N_FREQS = 48
JUMP_COST = 4.0                  # per octave, in units of log power
MAX_GAP = 5                      # frames without a flagellum bridged in time
NO_BEAT_RAD = 1e-9               # ridge amplitude below which a frame has no beat
# Ridge power over the median power across frequencies; above it a frame's beat is trusted
# (validated with jitter: frequency within 10% in 99.95% of beating frames, direction right in 99%;
# still cells stay below 22). Slow beats read lower, so it only sets inst_confident_frac.
CONFIDENT_EXCESS = 30.0

SERIES_COLUMNS = (
    "inst_freq_hz", "inst_excess", "inst_amp_mean", "inst_amp_asym", "inst_amp_para",
    "inst_phase_asym", "inst_phase_para", "inst_phase_slope", "inst_wavelength_px",
    "inst_direction", "inst_angle_mean", "inst_curvature_mean",
)
# Per clip: the median of each series column over its frames inside the cone of influence, or over
# all its frames when none is inside (all frames always for the shape columns), and the spreads and
# confident share.
CLIP_COLUMNS = SERIES_COLUMNS + ("inst_freq_sd_hz", "inst_angle_sd", "inst_confident_frac")
_SHAPE_COLUMNS = ("inst_angle_mean", "inst_curvature_mean", "inst_excess")


def angle_field(curves: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Tangent angle (frames, N_POINTS) over 0.9 of each frame's length, and its spacing in px.

    Rows are NaN where the frame has no curve.
    """
    length = curves["length"]
    n_frames = length.size
    psi = np.full((n_frames, N_POINTS), np.nan)
    rows = np.flatnonzero(np.isfinite(length) & (length > 0))
    if rows.size == 0:
        return psi, np.full(n_frames, np.nan)
    u = np.linspace(0.0, 1.0, N_POINTS)
    xs = np.empty((rows.size, N_POINTS))
    ys = np.empty((rows.size, N_POINTS))
    for j, f in enumerate(rows):
        s = u * ws.LENGTH_SHARE * length[f]
        xs[j] = np.interp(s, curves["arc"][f], curves["x"][f])
        ys[j] = np.interp(s, curves["arc"][f], curves["y"][f])
    psi[rows] = np.unwrap(np.arctan2(np.gradient(ys, axis=1), np.gradient(xs, axis=1)), axis=1)
    return psi, ws.LENGTH_SHARE * length / (N_POINTS - 1)


def _segments(valid: np.ndarray) -> list[tuple[int, int]]:
    """Runs of valid frames, gaps of up to MAX_GAP frames bridged."""
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > MAX_GAP + 1)
    return list(zip(np.r_[idx[0], idx[breaks + 1]], np.r_[idx[breaks], idx[-1]] + 1))


def _filters(nfft: int, fps: float, freqs: np.ndarray) -> np.ndarray:
    """Morlet filters in frequency, scaled so a cosine of amplitude A reads A at its own frequency."""
    nu = np.fft.fftfreq(nfft, d=1.0 / fps)
    h = np.exp(-0.5 * ((nu[None, :] - freqs[:, None]) / (freqs[:, None] / OMEGA0)) ** 2)
    h[:, nu <= 0] = 0.0
    return 2.0 * h


def _ridge(logp: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """Column per row maximising the summed ``logp`` less ``cost[from, to]`` per step (Viterbi)."""
    n, k = logp.shape
    score = logp[0].copy()
    back = np.zeros((n, k), dtype=int)
    cols = np.arange(k)
    for t in range(1, n):
        cand = score[:, None] - cost
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], cols] + logp[t]
    path = np.zeros(n, dtype=int)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def _legendre_rows(values: np.ndarray) -> np.ndarray:
    """``wavespectra._legendre`` of every row: (rows, 3)."""
    xs = np.linspace(0.0, 1.0, values.shape[1])
    ell = np.stack([np.ones_like(xs), 2 * xs - 1, 6 * xs ** 2 - 6 * xs + 1])
    w = ws._trapz_weights(xs)
    return (values * w) @ ell.T * np.array([1.0, 3.0, 5.0])


def beat_from_curves(curves: dict[str, np.ndarray], fps: float) -> pd.DataFrame:
    """:data:`SERIES_COLUMNS` per frame from smoothed body-frame curves (see :func:`beat_series`)."""
    psi, ds = angle_field(curves)
    n_frames = psi.shape[0]
    out = {c: np.full(n_frames, np.nan) for c in SERIES_COLUMNS}
    valid = np.isfinite(psi).all(axis=1)
    if valid.any():
        mean = np.nanmean(psi[valid], axis=1)
        sign = -1.0 if np.median(mean) < 0 else 1.0       # a mirror image flips every angle
        out["inst_angle_mean"][valid] = sign * mean
        out["inst_curvature_mean"][valid] = np.abs(np.diff(psi[valid], axis=1)).mean(axis=1) / ds[valid]
    fps = float(fps)
    freqs = np.geomspace(F_LOW, F_HIGH_SHARE * fps, N_FREQS)
    cost = JUMP_COST * np.abs(np.log2(freqs[:, None] / freqs[None, :]))
    sos = butter(2, BEAT_F_MIN_HZ, btype="highpass", fs=fps, output="sos")
    w = ws._trapz_weights(np.linspace(0.0, 1.0, N_POINTS))
    sx = np.linspace(0.0, 1.0, N_POINTS)
    for a, b in _segments(valid):
        n = b - a
        if n < 8:
            continue
        seg = psi[a:b].copy()
        gap = ~valid[a:b]
        if gap.any():
            t = np.arange(n)
            for k in range(N_POINTS):
                seg[gap, k] = np.interp(t[gap], t[~gap], seg[~gap, k])
        pad = min(n - 1, int(fps))
        seg = sosfiltfilt(sos, seg, axis=0, padtype="odd", padlen=min(n - 1, 3 * pad))
        x = np.concatenate([seg[pad:0:-1], seg, seg[-2:-pad - 2:-1]])
        nfft = int(2 ** np.ceil(np.log2(x.shape[0])))
        spec = np.fft.fft(x, n=nfft, axis=0)
        h = _filters(nfft, fps, freqs)
        power = np.empty((n, freqs.size))
        for j in range(freqs.size):
            power[:, j] = (np.abs(np.fft.ifft(spec * h[j][:, None], axis=0)[pad:pad + n]) ** 2) @ w
        path = _ridge(np.log(np.maximum(power, 1e-300)), cost)
        here = np.zeros((n, N_POINTS), dtype=complex)
        ahead = np.zeros((n, N_POINTS), dtype=complex)
        for j in np.unique(path):
            wj = np.fft.ifft(spec * h[j][:, None], axis=0)[pad:pad + n]
            sel = path == j
            here[sel] = wj[sel]
            ahead[sel] = np.r_[wj[1:], wj[-1:]][sel]
        advance = np.angle((ahead * np.conj(here)) @ w)
        advance[-1] = advance[-2] if n > 1 else advance[-1]
        amp = np.abs(here)
        phase = np.unwrap(np.angle(here), axis=1)
        slope = np.polyfit(sx, phase.T, 1)[0] / (ds[a:b] * (N_POINTS - 1))
        a_c, p_c = _legendre_rows(amp), _legendre_rows(phase)
        # Frames the wavelet reaches past the segment's ends (its cone of influence) are not trusted.
        reach = np.sqrt(2.0) * OMEGA0 / (2.0 * np.pi * freqs[path]) * fps
        t = np.arange(n)
        inside = (t >= reach) & (t <= n - 1 - reach)
        median = np.median(power, axis=1)
        silent = np.sqrt(power[np.arange(n), path]) < NO_BEAT_RAD   # nothing moves: no beat to read
        with np.errstate(divide="ignore", invalid="ignore"):
            block = {
                "inst_freq_hz": advance * fps / (2.0 * np.pi),
                "inst_excess": np.where(inside, power[np.arange(n), path] / median, np.nan),
                "inst_amp_mean": a_c[:, 0], "inst_amp_asym": a_c[:, 1], "inst_amp_para": a_c[:, 2],
                "inst_phase_asym": p_c[:, 1], "inst_phase_para": p_c[:, 2],
                "inst_phase_slope": slope,
                "inst_wavelength_px": np.where(slope != 0, 2.0 * np.pi / np.abs(slope), np.nan),
                "inst_direction": -np.sign(slope),
            }
        for col, values in block.items():
            values = np.where(gap | silent, np.nan, values)
            out[col][a:b] = values
    frame = pd.DataFrame(out, index=pd.RangeIndex(n_frames, name="frame"))
    return frame.replace([np.inf, -np.inf], np.nan)


def beat_series(px: np.ndarray, py: np.ndarray, fps: float,
                um_per_px: float | None = None) -> pd.DataFrame:
    """:data:`SERIES_COLUMNS` per frame; ``px``/``py`` as for :func:`wavespectra.spectral_series`.

    With ``um_per_px`` the nodes are a dense centerline, smoothed over ``SMOOTHING_UM``."""
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    if px.ndim != 2 or px.shape != py.shape or px.shape[1] < 4 or px.shape[0] == 0:
        return pd.DataFrame(columns=list(SERIES_COLUMNS), dtype=float)
    if um_per_px:
        curves = ws.frame_curves(px, py, cutoff_px=SMOOTHING_UM / um_per_px)
    else:
        curves = ws.frame_curves(px, py, cutoff_segments=SMOOTHING_SEGMENTS)
    return beat_from_curves(curves, fps)


def track_beat_series(sub: pd.DataFrame, n_keypoints: int, fps: float,
                      n_frames: int | None = None, f_start: int | None = None,
                      um_per_px: float | None = None) -> pd.DataFrame:
    """:func:`beat_series` from one track's rows, laid on the full frame grid first."""
    sub = sub.sort_values("frame")
    frames = sub["frame"].to_numpy(dtype=int)
    if frames.size == 0:
        return pd.DataFrame(columns=list(SERIES_COLUMNS), dtype=float)
    f_start = int(frames.min()) if f_start is None else int(f_start)
    n_frames = int(frames.max()) - f_start + 1 if n_frames is None else int(n_frames)
    px = _on_grid(sub, [f"kp_x_{k}" for k in range(n_keypoints)], n_frames, f_start)
    py = _on_grid(sub, [f"kp_y_{k}" for k in range(n_keypoints)], n_frames, f_start)
    series = beat_series(px, py, fps, um_per_px)
    series.index = pd.Index(series.index.to_numpy(dtype=int) + f_start, name="frame")
    return series


def window_values(series: pd.DataFrame | None, f_start: int, f_end: int) -> dict[str, float]:
    """:data:`CLIP_COLUMNS` over frames [f_start, f_end) of a track's :func:`beat_series`."""
    out = {c: np.nan for c in CLIP_COLUMNS}
    if series is None or len(series) == 0:
        return out
    rows = series.loc[(series.index >= f_start) & (series.index < f_end)]
    if len(rows) == 0:
        return out
    excess = rows["inst_excess"].to_numpy(dtype=float)
    seen = np.isfinite(excess)          # inside the cone of influence
    confident = seen & (excess >= CONFIDENT_EXCESS)
    out["inst_confident_frac"] = float(confident.sum() / seen.sum()) if seen.any() else np.nan
    # A clip wholly outside the cone still reads a weaker beat from all its frames.
    used = seen if seen.any() else np.ones(len(rows), dtype=bool)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for col in SERIES_COLUMNS:
            values = rows[col].to_numpy(dtype=float)
            values = values if col in _SHAPE_COLUMNS else values[used]
            if np.isfinite(values).any():
                out[col] = float(np.nanmedian(values))
        freq = rows["inst_freq_hz"].to_numpy(dtype=float)[used]
        if np.isfinite(freq).sum() >= 2:
            out["inst_freq_sd_hz"] = float(np.nanstd(freq))
        angle = rows["inst_angle_mean"].to_numpy(dtype=float)
        if np.isfinite(angle).sum() >= 2:
            out["inst_angle_sd"] = float(np.nanstd(angle))
    return out
