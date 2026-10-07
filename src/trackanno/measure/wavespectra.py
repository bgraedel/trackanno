"""Sliding-window tangent-angle wave spectra per track, after Geyer, Howard and Sartori (2022).

Frequencies in Hz, lengths in px; ``wave_direction`` +1 means the wave travels Base to Tip.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.signal import ShortTimeFFT, detrend

from .clipfeatures import (
    BEAT_F_MIN_HZ, BEAT_MIN_CYCLES, MAX_INTERP_FRAC,
    _interp_nan, _on_grid, _shifted_legendre, _safe,
)

__all__ = [
    "spectral_series", "window_values", "WindowSource", "track_spectral_series",
    "flow_series",
    "SERIES_COLUMNS", "DEFAULT_WINDOW", "DEFAULT_HOP", "DEFAULT_FLOW_FRAMES",
]

# 0.75 s at 100 fps.
DEFAULT_WINDOW = 75
# 50 ms at 100 fps.
DEFAULT_HOP = 5

# Below this the transform has fewer bins than the band it is searched over.
MIN_WINDOW_FRAMES = 8
# An order-2 Legendre fit needs at least three segments.
MIN_SEGMENTS = 3

# Frames one flow estimate is read over.
DEFAULT_FLOW_FRAMES = 5

SERIES_COLUMNS = (
    # -- the beat -----------------------------------------------------------
    "wave_freq_hz", "wave_resolvable_hz", "wave_excess", "wave_fundamental_frac",
    "wave_censored",
    # -- amplitude profile --------------------------------------------------
    "wave_amp_mean", "wave_amp_asym", "wave_amp_para",
    # -- phase profile, i.e. the direction ----------------------------------
    "wave_phase_slope", "wave_phase_rsq", "wave_phase_asym", "wave_phase_para",
    "wave_wavelength_px", "wave_wavelength_ratio", "wave_direction",
    "wave_speed_px_s",
    # -- the same direction, read without a spectrum -------------------------
    "wave_flow", "wave_flow_coherence", "wave_flow_direction",
    # -- static (time-averaged) shape ---------------------------------------
    "wave_static_cos", "wave_static_sin", "wave_static_asym", "wave_static_para",
    # -- geometry and quality ------------------------------------------------
    "wave_length_px", "wave_interp_frac", "wave_window_frames",
    # -- stability across windows ----------------------------------------------
    "wave_freq_drift_hz", "wave_freq_track_sd_hz",
)

# Aggregated by mean (a share) rather than median.
_FRACTION_COLUMNS = ("wave_censored",)


# --------------------------------------------------------------------------- #
# sliding aggregates
# --------------------------------------------------------------------------- #
def _sliding_sum(values: np.ndarray, m: int, starts: np.ndarray) -> np.ndarray:
    """Sum of ``values[c : c + m]`` for every start ``c``, via a cumulative sum."""
    csum = np.concatenate([np.zeros((1,) + values.shape[1:], values.dtype),
                           np.cumsum(values, axis=0)])
    return csum[starts + m] - csum[starts]


def _sliding_mean(values: np.ndarray, m: int, starts: np.ndarray) -> np.ndarray:
    """Mean of ``values[c : c + m]`` for every start ``c``. See :func:`_sliding_sum`."""
    return _sliding_sum(values, m, starts) / float(m)


def _detrend_coeffs(
    signal: np.ndarray, m: int, starts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-window OLS intercept and slope against the local tick, from sliding sums.

    Keep float64: the sliding-sum subtraction loses about ``log10(n / m)`` digits.
    """
    n = signal.shape[0]
    tick = np.arange(n, dtype=float)[:, None]
    s_x = _sliding_sum(signal, m, starts)
    s_tx = _sliding_sum(tick * signal, m, starts) - starts[:, None].astype(float) * s_x
    t_bar = (m - 1) / 2.0
    s_tt = m * (m * m - 1) / 12.0            # sum (t - t_bar)^2 for t = 0..m-1
    beta = (s_tx - t_bar * s_x) / s_tt
    return s_x / float(m) - beta * t_bar, beta


def _slice_starts(n: int, m: int, hop: int, mid: int) -> np.ndarray:
    """Start frames of the fully interior slices on :class:`ShortTimeFFT`'s grid."""
    p_first = -(-mid // hop)                    # ceil(mid / hop)
    p_last = (n - m + mid) // hop               # last p with start + m <= n
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


def flow_series(
    psi: np.ndarray,
    arc_px: np.ndarray,
    complete: np.ndarray,
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

    # Remove the local static bend before d/ds; d/dt ignores it anyway.
    finite = np.isfinite(psi)
    filled = np.where(finite, psi, 0.0)
    seen = _centred_sum(finite.astype(float), half)
    with np.errstate(invalid="ignore", divide="ignore"):
        static = np.where(seen > 0, _centred_sum(filled, half) / np.maximum(seen, 1), np.nan)

    psi_t = np.gradient(psi, axis=0)
    step = np.diff(arc_px, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        psi_s = np.diff(psi - static, axis=1) / np.where(np.abs(step) > 1e-9, step, np.nan)
    # onto the grid the arclength difference lives on
    psi_t = 0.5 * (psi_t[:, :-1] + psi_t[:, 1:])

    good = np.isfinite(psi_t) & np.isfinite(psi_s)
    a, b = np.where(good, psi_t, 0.0), np.where(good, psi_s, 0.0)
    cross = _centred_sum((a * b).sum(axis=1), half)
    aa = _centred_sum((a * a).sum(axis=1), half)
    bb = _centred_sum((b * b).sum(axis=1), half)
    pairs = _centred_sum(good.sum(axis=1).astype(float), half)

    # Interpolation is judged over the stencil, not the spectral window.
    detected = _centred_sum(np.asarray(complete, dtype=float), half)
    span = _centred_sum(np.ones(n), half)
    with np.errstate(invalid="ignore", divide="ignore"):
        local_fill = 1.0 - detected / np.maximum(span, 1.0)
        denominator = np.sqrt(aa * bb)
        out = np.where(
            (denominator > 1e-12) & (pairs >= 4) & (local_fill <= MAX_INTERP_FRAC),
            -cross / np.where(denominator > 0, denominator, np.nan),
            np.nan,
        )
    # Clip rounding past +-1.
    out = np.clip(out, -1.0, 1.0)
    # Partial stencils at the ends are a different measurement.
    out[:half] = np.nan
    out[n - half:] = np.nan
    return out


def _read_at(values: np.ndarray, centres: np.ndarray, reach: int) -> np.ndarray:
    """Median of a per-frame series over ``+-reach`` frames of each centre."""
    out = np.full(centres.size, np.nan)
    n = values.shape[0]
    for i, centre in enumerate(centres):
        lo, hi = max(0, int(centre) - reach), min(n, int(centre) + reach + 1)
        if hi <= lo:
            continue
        block = values[lo:hi]
        block = block[np.isfinite(block)]
        if block.size:
            out[i] = float(np.median(block))
    return out


def _direct_spectrum(
    signal: np.ndarray, m: int, starts: np.ndarray, fps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The sliding transform computed directly for explicit window starts."""
    block = np.stack([signal[c:c + m] for c in starts])          # (n_win, m, n_node)
    block = detrend(block, axis=1, type="linear") * np.hanning(m)[:, None]
    freqs = np.fft.rfftfreq(m, d=1.0 / max(fps, 1e-6))
    return freqs, np.moveaxis(np.fft.rfft(block, axis=1), 0, -1), np.asarray(starts, dtype=int)


def _sliding_spectrum(
    signal: np.ndarray, m: int, hop: int, fps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One STFT for the whole record: ``(freqs, S, starts)``, S of shape (n_freq, n_node, n_window).

    Equals a per-slice ``rfft(detrended * hanning)``; the linear detrend is applied analytically.
    """
    n = signal.shape[0]
    m = min(m, n)
    win = np.hanning(m)
    sft = ShortTimeFFT(win, hop=hop, fs=max(fps, 1e-6),
                       fft_mode="onesided", phase_shift=None)
    starts = _slice_starts(n, m, hop, sft.m_num_mid)
    if starts.size == 0:
        # No legal slice centre: centre one window by hand.
        return _direct_spectrum(signal, m, np.array([(n - m) // 2]), fps)
    p0 = int((starts[0] + sft.m_num_mid) // hop)
    spectrum = sft.stft(signal, p0=p0, p1=p0 + starts.size, axis=0)
    alpha, beta = _detrend_coeffs(signal, m, starts)             # (n_win, n_node)
    w_0 = np.fft.rfft(win)
    w_t = np.fft.rfft(win * np.arange(m, dtype=float))
    spectrum = (spectrum
                - w_0[:, None, None] * alpha.T[None, :, :]
                - w_t[:, None, None] * beta.T[None, :, :])
    return sft.f, spectrum, starts


def _legendre_batch(profiles: np.ndarray, s: np.ndarray, order: int = 2) -> np.ndarray:
    """Least-squares fit of ``(n_window, n_segment)`` profiles to shifted Legendre polynomials."""
    basis = np.moveaxis(_shifted_legendre(s, order), 0, -1)   # (n_win, n_seg, order+1)
    good = np.isfinite(profiles)
    filled = np.where(good, profiles, 0.0)
    gram = np.einsum("wsk,wsl->wkl", basis, basis)
    rhs = np.einsum("wsk,ws->wk", basis, filled)
    coeffs = np.einsum("wkl,wl->wk", np.linalg.pinv(gram), rhs)
    # A profile with holes gets no fit.
    coeffs[~good.all(axis=1)] = np.nan
    return coeffs


def _line_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise least-squares slope and R^2 (``phaseLinearityRsq``) of ``y`` against ``x``."""
    n = x.shape[1]
    xm, ym = x.mean(axis=1, keepdims=True), y.mean(axis=1, keepdims=True)
    dx, dy = x - xm, y - ym
    sxx = (dx * dx).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = np.where(sxx > 1e-12, (dx * dy).sum(axis=1) / np.where(sxx > 1e-12, sxx, 1.0),
                         np.nan)
        resid = dy - slope[:, None] * dx
        ss_res = (resid ** 2).sum(axis=1)
        ss_tot = (dy * dy).sum(axis=1)
        rsq = np.where(ss_tot > 1e-12, 1.0 - ss_res / np.where(ss_tot > 1e-12, ss_tot, 1.0),
                       np.nan)
    return slope, rsq


def _local_spread(values: np.ndarray, radius: int) -> np.ndarray:
    """``1.4826 * MAD`` of ``values`` over ``+-radius`` neighbours; NaN below three finite ones."""
    pad = np.full(int(radius), np.nan)
    block = np.lib.stride_tricks.sliding_window_view(
        np.concatenate([pad, values, pad]), 2 * int(radius) + 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN rows are expected
        median = np.nanmedian(block, axis=1)
        mad = np.nanmedian(np.abs(block - median[:, None]), axis=1)
    return np.where(np.isfinite(block).sum(axis=1) >= 3, 1.4826 * mad, np.nan)


# --------------------------------------------------------------------------- #
# the series
# --------------------------------------------------------------------------- #
def spectral_series(
    px: np.ndarray,
    py: np.ndarray,
    fps: float,
    window: int = DEFAULT_WINDOW,
    hop: int = DEFAULT_HOP,
    flow_frames: int = DEFAULT_FLOW_FRAMES,
) -> pd.DataFrame:
    """Wave descriptors (:data:`SERIES_COLUMNS`) per sliding window, indexed by centre frame.

    ``px``/``py`` are (n_frames, n_keypoints) on the full frame grid, NaN where undetected.
    """
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    empty = pd.DataFrame(columns=list(SERIES_COLUMNS), dtype=float)
    empty.index.name = "frame"
    if px.ndim != 2 or px.shape != py.shape or px.shape[1] < 4:
        return empty
    n_frames = px.shape[0]
    window = int(max(MIN_WINDOW_FRAMES, window))
    hop = int(max(1, hop))
    if n_frames < MIN_WINDOW_FRAMES:
        return empty

    # -- 1. per-frame geometry, computed ONCE ------------------------------- #
    # Fill gaps in time; the transform assumes uniform sampling.
    complete = np.isfinite(px).all(axis=1) & np.isfinite(py).all(axis=1)
    px, py = _interp_nan(px), _interp_nan(py)

    ox, oy = px[:, 1], py[:, 1]                       # Base = origin
    axis = np.arctan2(py[:, 0] - oy, px[:, 0] - ox)   # Base -> Head
    cos, sin = np.cos(-axis), np.sin(-axis)
    fx = (px[:, 1:] - ox[:, None]) * cos[:, None] - (py[:, 1:] - oy[:, None]) * sin[:, None]
    fy = (px[:, 1:] - ox[:, None]) * sin[:, None] + (py[:, 1:] - oy[:, None]) * cos[:, None]

    dx, dy = np.diff(fx, axis=1), np.diff(fy, axis=1)
    seg = np.hypot(dx, dy)
    n_seg = seg.shape[1]
    if n_seg < MIN_SEGMENTS:
        return empty
    arc = np.cumsum(seg, axis=1)
    length = arc[:, -1]

    # Unwrap along arclength first, then time.
    psi = np.unwrap(np.arctan2(dy, dx), axis=1)

    # Treat a zero-length flagellum as undetected.
    degenerate = ~(np.isfinite(psi).all(axis=1) & np.isfinite(seg).all(axis=1)
                   & np.isfinite(length) & (length > 1e-9))
    complete = complete & ~degenerate
    if degenerate.any():
        psi[degenerate] = np.nan
        psi = _interp_nan(psi)
        seg[degenerate] = np.nan
        seg = _interp_nan(seg)
        arc = np.cumsum(seg, axis=1)
        length = np.where(degenerate, np.nan, length)
        length = _interp_nan(length[:, None])[:, 0]
        length = np.where(np.isfinite(length) & (length > 1e-9), length, np.nan)

    with np.errstate(invalid="ignore", divide="ignore"):
        node_s = (arc - seg / 2.0) / length[:, None]        # per-frame, normalised
    node_s = np.where(np.isfinite(node_s), node_s, 0.0)

    # For the circular mean of the raw angle, taken before the time unwrap.
    tangent = np.exp(1j * psi)
    psi = np.unwrap(psi, axis=0)                            # now along time

    # -- 2. one transform for the whole track -------------------------------- #
    freqs, spectrum, starts = _sliding_spectrum(psi, window, hop, fps)
    if starts.size == 0:
        return empty
    m = int(min(window, n_frames))
    n_win = starts.size
    centres = starts + m // 2

    out = {c: np.full(n_win, np.nan) for c in SERIES_COLUMNS}
    out["wave_window_frames"] = np.full(n_win, float(m))

    # -- 2b. direction, read without the spectrum ---------------------------- #
    # Before the band check, so rows without a resolvable beat still get it.
    flow = flow_series(psi, arc - seg / 2.0, complete, flow_frames)
    reach = max(int(flow_frames) // 2, int(hop) // 2)
    out["wave_flow"] = _read_at(flow, centres, reach)
    out["wave_flow_coherence"] = _read_at(np.abs(flow), centres, reach)
    out["wave_flow_direction"] = np.sign(out["wave_flow"])
    out["wave_flow_direction"][out["wave_flow_direction"] == 0.0] = np.nan

    # -- 3. per-window geometry, from sliding sums --------------------------- #
    s = _sliding_mean(node_s, m, starts)                    # (n_win, n_seg)
    mean_length = _sliding_mean(length[:, None], m, starts)[:, 0]
    mean_dir = _sliding_mean(tangent, m, starts)            # (n_win, n_seg) complex
    filled_frac = 1.0 - _sliding_mean(complete[:, None].astype(float), m, starts)[:, 0]
    out["wave_interp_frac"] = filled_frac

    # MAX_INTERP_FRAC applies per window.
    usable = (filled_frac <= MAX_INTERP_FRAC)
    usable &= ((1.0 - filled_frac) * m >= MIN_WINDOW_FRAMES)
    usable &= np.isfinite(mean_length) & (mean_length > 1e-9)
    usable &= np.isfinite(s).all(axis=1)
    out["wave_length_px"] = np.where(usable, mean_length, np.nan)

    # -- 4. static mode ------------------------------------------------------ #
    psi_0 = np.unwrap(np.angle(mean_dir), axis=1)           # continuous along arclength
    resultant = np.angle(mean_dir.mean(axis=1))
    out["wave_static_cos"] = np.where(usable, np.cos(resultant), np.nan)
    out["wave_static_sin"] = np.where(usable, np.sin(resultant), np.nan)
    static_c = _legendre_batch(psi_0, s)
    # Gradients of the profile, so the undecidable constant offset cancels.
    out["wave_static_asym"] = np.where(usable, static_c[:, 1], np.nan)
    out["wave_static_para"] = np.where(usable, static_c[:, 2], np.nan)

    # -- 5. the fundamental -------------------------------------------------- #
    # Band-limited as in `_beat_estimate`.
    duration = m / max(fps, 1e-6)
    f_floor = max(BEAT_F_MIN_HZ, BEAT_MIN_CYCLES / duration)
    out["wave_resolvable_hz"] = np.where(usable, f_floor, np.nan)
    band = np.flatnonzero(freqs >= f_floor)
    if band.size < 3 or not np.isfinite(spectrum).all():
        return _finish(out, centres, hop, m)

    power = (np.abs(spectrum) ** 2).sum(axis=1)             # (n_freq, n_win), pooled
    band_power = power[band]                                # (n_band, n_win)
    local = np.argmax(band_power, axis=0)
    peak = band[local]
    win_idx = np.arange(n_win)
    peak_power = band_power[local, win_idx]
    baseline = np.median(band_power, axis=0)
    total = band_power.sum(axis=0)
    # A window with no variation has no beat.
    with np.errstate(invalid="ignore", divide="ignore"):
        moving = total > np.finfo(float).eps * max(1.0, float(np.nanmax(total, initial=0.0)))
    usable = usable & moving

    with np.errstate(invalid="ignore", divide="ignore"):
        excess = np.where(baseline > 0, peak_power / baseline, np.nan)
        frac = np.where(total > 0, peak_power / total, np.nan)
    out["wave_excess"] = np.where(usable, excess, np.nan)
    out["wave_fundamental_frac"] = np.where(usable, frac, np.nan)
    # Peak on the band floor: the true frequency is at or below it.
    censored = (local == 0)
    out["wave_censored"] = np.where(usable, censored.astype(float), np.nan)
    out["wave_freq_hz"] = np.where(usable, freqs[peak], np.nan)
    measured = usable

    peak_spec = spectrum[peak, :, win_idx]                  # (n_win, n_seg)
    # Hanning halves the coherent gain, hence the extra factor of two.
    amplitude = np.abs(peak_spec) * (2.0 / m) * 2.0
    phase = np.unwrap(np.angle(peak_spec), axis=1)

    amp_c = _legendre_batch(amplitude, s)
    a0 = amp_c[:, 0]
    scale = np.where(np.isfinite(a0) & (np.abs(a0) > 1e-9), a0, np.nan)
    out["wave_amp_mean"] = np.where(measured, a0, np.nan)
    # Normalised by the mean, as in the paper.
    out["wave_amp_asym"] = np.where(measured, amp_c[:, 1] / scale, np.nan)
    out["wave_amp_para"] = np.where(measured, amp_c[:, 2] / scale, np.nan)

    phase_c = _legendre_batch(phase, s)
    out["wave_phase_asym"] = np.where(usable, phase_c[:, 1], np.nan)
    out["wave_phase_para"] = np.where(usable, phase_c[:, 2], np.nan)

    # -- 6. wavelength and direction from the phase gradient ----------------- #
    # Fitted against arclength in px.
    arc_px = s * mean_length[:, None]
    slope, rsq = _line_fit(arc_px, phase)                   # rad per px
    slope = np.where(usable, slope, np.nan)
    out["wave_phase_slope"] = slope
    out["wave_phase_rsq"] = np.where(usable, rsq, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        wavelength = np.where(np.abs(slope) > 1e-9, 2.0 * np.pi / np.abs(slope), np.nan)
    out["wave_wavelength_px"] = wavelength
    out["wave_wavelength_ratio"] = wavelength / mean_length
    # +1 means Base -> Tip.
    direction = -np.sign(slope)
    direction = np.where(np.isfinite(wavelength) & (direction != 0.0), direction, np.nan)
    out["wave_direction"] = direction
    out["wave_speed_px_s"] = direction * wavelength * out["wave_freq_hz"]

    # -- 7. beat stability ---------------------------------------------------- #
    # Local and whole-track spread, over uncensored windows only.
    freq = np.where(out["wave_censored"] == 1.0, np.nan, out["wave_freq_hz"])
    out["wave_freq_drift_hz"] = _local_spread(freq, max(1, m // hop))
    good = freq[np.isfinite(freq)]
    if good.size >= 3:
        out["wave_freq_track_sd_hz"] = np.full(n_win, float(good.std(ddof=1)))
    return _finish(out, centres, hop, m)


def _finish(out: dict, centres: np.ndarray, hop: int, m: int) -> pd.DataFrame:
    """Assemble the series, in a fixed column order, indexed by centre frame."""
    frame = pd.DataFrame({c: np.asarray(out[c], dtype=float) for c in SERIES_COLUMNS},
                         index=pd.Index(centres, name="frame"))
    frame.attrs["window"] = int(m)
    frame.attrs["hop"] = int(hop)
    return frame.replace([np.inf, -np.inf], np.nan)


def track_spectral_series(
    sub: pd.DataFrame,
    n_keypoints: int,
    fps: float,
    window: int = DEFAULT_WINDOW,
    hop: int = DEFAULT_HOP,
    n_frames: int | None = None,
    f_start: int | None = None,
    flow_frames: int = DEFAULT_FLOW_FRAMES,
) -> pd.DataFrame:
    """:func:`spectral_series` from one track's rows, laid on the full frame grid first."""
    sub = sub.sort_values("frame")
    frames = sub["frame"].to_numpy(dtype=int)
    if frames.size == 0:
        return spectral_series(np.empty((0, 0)), np.empty((0, 0)), fps, window, hop,
                               flow_frames)
    if f_start is None:
        f_start = int(frames.min())
    if n_frames is None:
        n_frames = int(frames.max()) - int(f_start) + 1
    kx = [f"kp_x_{k}" for k in range(n_keypoints)]
    ky = [f"kp_y_{k}" for k in range(n_keypoints)]
    px = _on_grid(sub, kx, int(n_frames), int(f_start))
    py = _on_grid(sub, ky, int(n_frames), int(f_start))
    series = spectral_series(px, py, fps, window, hop, flow_frames)
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
                              np.where(self.is_fraction,
                                       np.nanmean(block, axis=0),
                                       np.nanmedian(block, axis=0)))
        for column, value in zip(self.columns, pooled):
            if np.isfinite(value):
                out[column] = float(value)

        slope = out.get("wave_phase_slope", np.nan)
        out["wave_direction"] = _safe(
            -float(np.sign(slope)) if np.isfinite(slope) and slope != 0.0 else np.nan)
        # From the signed median, never a median of +-1 values.
        flow = out.get("wave_flow", np.nan)
        out["wave_flow_direction"] = _safe(
            float(np.sign(flow)) if np.isfinite(flow) and flow != 0.0 else np.nan
        )
        freq = out.get("wave_freq_hz", np.nan)
        wavelength = out.get("wave_wavelength_px", np.nan)
        out["wave_speed_px_s"] = _safe(out["wave_direction"] * wavelength * freq)
        return out


def window_values(series: pd.DataFrame, f_start: int, f_end: int) -> dict[str, float]:
    """One window of one track's spectral series. :class:`WindowSource` for many."""
    return WindowSource(series).at(f_start, f_end)
