"""Sliding-window flagellar wave spectra, computed once per TRACK.

The Geyer, Howard and Sartori (2022) beat decomposition needs more frames than a
clip has: at 25 frames / 100 fps the FFT bin is 4 Hz and the ``BEAT_MIN_CYCLES``
floor lands at 12 Hz, above most annotated beats. So the window is centred on the
clip but LONGER than it, and a clip's features are a median over the windows
whose centres fall inside it. One STFT covers the whole track, the body frame,
tangent angle, unwraps and arclength being per-FRAME quantities.

``DEFAULT_WINDOW = 75`` is where the two demands meet: shorter cannot resolve the
beat, longer straddles a reversal. A hop finer than the clip stride locates a
reversal to within a few frames and lets ``wave_freq_drift_hz`` (local) and
``wave_freq_track_sd_hz`` (whole-track) report wander a whole-record FFT would
hide in its peak width.

Rules, applied window by window:

* Fill the time axis, never compact it (``_interp_nan``, mirroring
  ``interpInTime.m``); compacting scales the frequency by the reciprocal of the
  surviving fraction. ``MAX_INTERP_FRAC`` applies PER WINDOW.
* The static mode is a circular mean of the RAW tangent angle; a mean of the
  time-unwrapped one inherits an arbitrary 2*pi branch.
* ``wave_direction = -sign(slope)``, so **+1 means Base -> Tip**: a Base->Tip
  wave ``psi = A sin(wt - ks)``, k > 0, carries phase ``-ks`` at +f and so a
  NEGATIVE gradient along arclength. Verified against the real labels.
* The amplitude carries the Hanning coherent-gain factor ``(2/n) * 2``.
* Nothing is blanked on a significance threshold; ``wave_excess`` and
  ``wave_censored`` are columns standing beside the values, as ``processFile.m``
  reports ``domFreq`` beside ``domFreqDominance``. ``wave_phase_rsq``
  (``phaseLinearityRsq``) is what separates a wave from a random walk; condition
  on that rather than on excess.

Shared primitives come from :mod:`trackanno.measure.clipfeatures`.
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
    "spectral_series", "window_values", "track_spectral_series",
    "SERIES_COLUMNS", "DEFAULT_WINDOW", "DEFAULT_HOP",
]

# 0.75 s at 100 fps.
DEFAULT_WINDOW = 75
# 50 ms at 100 fps: five readouts per 25-frame clip, so a reversal can be placed
# inside a clip.
DEFAULT_HOP = 5

# Below this the transform has fewer bins than the band it is searched over.
MIN_WINDOW_FRAMES = 8
# Fewer than three segments and the order-2 Legendre fit is not overdetermined,
# so its "shape" is just interpolation between the points it was given.
MIN_SEGMENTS = 3

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
    # -- static (time-averaged) shape ---------------------------------------
    "wave_static_cos", "wave_static_sin", "wave_static_asym", "wave_static_para",
    # -- geometry and quality ------------------------------------------------
    "wave_length_px", "wave_interp_frac", "wave_window_frames",
    # -- stability, which only a SLIDING estimate can report -----------------
    "wave_freq_drift_hz", "wave_freq_track_sd_hz",
)

# Aggregated as a FRACTION rather than a level: the mean of "was this window
# censored" is the share of the clip that was, where a median collapses to 0/1.
_FRACTION_COLUMNS = ("wave_censored",)


# --------------------------------------------------------------------------- #
# sliding aggregates
# --------------------------------------------------------------------------- #
def _sliding_sum(values: np.ndarray, m: int, starts: np.ndarray) -> np.ndarray:
    """Sum of ``values[c : c + m]`` for every start ``c``, in one pass.

    A cumulative sum rather than a strided view, which at hop 1 materialises
    ``window`` copies of the whole track. Complex input is accepted, for the
    circular mean of the tangent angle.
    """
    csum = np.concatenate([np.zeros((1,) + values.shape[1:], values.dtype),
                           np.cumsum(values, axis=0)])
    return csum[starts + m] - csum[starts]


def _sliding_mean(values: np.ndarray, m: int, starts: np.ndarray) -> np.ndarray:
    """Mean of ``values[c : c + m]`` for every start ``c``. See :func:`_sliding_sum`."""
    return _sliding_sum(values, m, starts) / float(m)


def _detrend_coeffs(
    signal: np.ndarray, m: int, starts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-window OLS intercept and slope of ``signal`` against the local tick.

    Both come out of two cumulative sums, because the design matrix ``[1, t]`` is
    the same for every window: with ``t`` the local index ``0..m-1`` and ``j`` the
    global frame,

        ``sum_k k * x[c+k] = sum_j j * x[j] - c * sum_j x[j]``

    over the window, and both sums on the right slide. Returns
    ``(n_window, n_node)``.

    Keep this in float64: that subtraction cancels two terms of order
    ``j * m * |x|`` into a difference of order ``m^2 * |x|``, burning about
    ``log10(n / m)`` digits - more than float32 has to give past a few hundred
    frames.
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
    """Window start frames on :class:`ShortTimeFFT`'s own slice grid.

    ``ShortTimeFFT`` places slice ``p`` over ``x[p*hop - mid : p*hop - mid + m]``
    with ``mid = m // 2``, so the slice CENTRES are the multiples of ``hop``.
    Keep only fully-interior slices: one hanging off an end is zero-padded, which
    is not a shorter window but a window with a step discontinuity that the
    full-length taper leaves in and the FFT reads as broadband power.
    """
    p_first = -(-mid // hop)                    # ceil(mid / hop)
    p_last = (n - m + mid) // hop               # last p with start + m <= n
    if p_last < p_first:
        return np.empty(0, dtype=int)
    return np.arange(p_first, p_last + 1, dtype=int) * hop - mid


def _direct_spectrum(
    signal: np.ndarray, m: int, starts: np.ndarray, fps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The same transform, for an explicit handful of window starts.

    Used only where :class:`ShortTimeFFT`'s slice grid cannot express the wanted
    window. Runs ``scipy.signal.detrend``, what ``stft_detrend`` calls
    internally, so this path and the sliding one agree exactly.
    """
    block = np.stack([signal[c:c + m] for c in starts])          # (n_win, m, n_node)
    block = detrend(block, axis=1, type="linear") * np.hanning(m)[:, None]
    freqs = np.fft.rfftfreq(m, d=1.0 / max(fps, 1e-6))
    return freqs, np.moveaxis(np.fft.rfft(block, axis=1), 0, -1), np.asarray(starts, dtype=int)


def _sliding_spectrum(
    signal: np.ndarray, m: int, hop: int, fps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One STFT for the whole record: ``(freqs, S, starts)``.

    ``S`` has shape ``(n_freq, n_node, n_window)`` and must stay bit-for-bit what
    ``np.fft.rfft(detrended_slice * np.hanning(m), axis=0)`` returns for each
    slice: the amplitude calibration and the phase convention downstream assume
    exactly that transform, and the tests pin it against a per-slice ``rfft``.

    * ``phase_shift=None``. SciPy's default phase reference is the window
      MIDPOINT (``mfft // 2``), not its first sample, offsetting every bin by
      ``exp(2j*pi*q*mid/m)``. That offset cancels out of the phase GRADIENT but
      not out of the absolute phase, so it would escape into whatever reads
      these columns next.
    * The linear detrend is not optional: an unwrapped tangent angle from an
      imperfectly tracked flagellum is a random walk, which piles its power at
      the bottom of the band and masquerades as a slow beat. It is applied
      ANALYTICALLY, being linear in the samples and so commuting with the
      transform:

          ``rfft(w * (x - a - b t)) = rfft(w * x) - a rfft(w) - b rfft(w t)``

      with ``rfft(w)`` and ``rfft(w * t)`` fixed vectors and ``a``, ``b`` from
      sliding sums (:func:`_detrend_coeffs`). Exact to floating-point noise.
    """
    n = signal.shape[0]
    m = min(m, n)
    win = np.hanning(m)
    sft = ShortTimeFFT(win, hop=hop, fs=max(fps, 1e-6),
                       fft_mode="onesided", phase_shift=None)
    starts = _slice_starts(n, m, hop, sft.m_num_mid)
    if starts.size == 0:
        # Slice CENTRES are multiples of the hop, so a record only a little
        # longer than the window can have no legal centre (76 frames at window 75
        # hop 5 needs 37 or 38 and is offered 35 or 40). Centre one window by
        # hand rather than dropping every short track.
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
    """Least-squares projection onto shifted Legendre polynomials, batched.

    Least squares rather than the analytic integral because ``s`` is a handful of
    tracked keypoints, not a densely sampled arclength.

    ``profiles`` and ``s`` are both ``(n_window, n_segment)``. The arclength grid
    is per-window because a flagellum foreshortening as it turns out of plane
    really does move its normalised nodes. Normal equations with a pseudo-inverse
    give the same minimum-norm answer as a per-window ``lstsq``, in one call.
    """
    basis = np.moveaxis(_shifted_legendre(s, order), 0, -1)   # (n_win, n_seg, order+1)
    good = np.isfinite(profiles)
    filled = np.where(good, profiles, 0.0)
    gram = np.einsum("wsk,wsl->wkl", basis, basis)
    rhs = np.einsum("wsk,ws->wk", basis, filled)
    coeffs = np.einsum("wkl,wl->wk", np.linalg.pinv(gram), rhs)
    # Never fit a window from the surviving points of a holed profile: the
    # residual would not be comparable with its neighbours'. Report the absence.
    coeffs[~good.all(axis=1)] = np.nan
    return coeffs


def _line_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Batched least-squares slope and R^2 of ``y`` against ``x``, row by row.

    ``R^2`` is the point of this, not the slope. A phase profile that is not
    LINEAR in arclength is not a travelling wave: its fitted slope collapses
    towards zero and the wavelength blows up. This is the reference pipeline's
    ``phaseLinearityRsq``, and it is what separates those blow-ups where
    ``wave_excess`` and ``wave_fundamental_frac`` cannot.
    """
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
    """Robust spread of ``values`` among its ``+-radius`` neighbours in the series.

    ``1.4826 * MAD``, so a single window locked onto a harmonic does not become
    the answer. NaN below three finite neighbours, since a spread estimated from
    two points is a difference. Padded with NaN rather than reflected: near the
    ends of a track there genuinely are fewer neighbours.
    """
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
) -> pd.DataFrame:
    """Beat-waveform descriptors for every sliding window of one track.

    Parameters
    ----------
    px, py
        ``(n_frames, n_keypoints)`` keypoint coordinates on the FULL frame grid,
        NaN where undetected - what :func:`clipfeatures._on_grid` produces. Node
        0 is Head, node 1 is Base, nodes 2.. are the flagellum. Passing a
        compacted array (missing frames dropped) is the one mistake that will not
        raise: it scales every reported frequency by the reciprocal of the
        surviving fraction.
    fps
        Frames per second. Every rate out of this function is per SECOND.
    window
        Window length in frames; see the module docstring for the default. A
        track shorter than this gets a single whole-track window rather than
        nothing, with ``wave_window_frames`` recording what was used and
        ``wave_resolvable_hz`` what that costs.
    hop
        Frames between successive window centres.

    Returns
    -------
    One row per window, indexed by the window's CENTRE frame (index name
    ``frame``), carrying :data:`SERIES_COLUMNS`. Empty (with the right columns
    and dtypes) when the track cannot support even one window.

    Notes
    -----
    NaN means "undefined" throughout and is never filled with 0.0: a direction of
    0.0 is a standing wave, and a frequency of 0.0 is DC.

    The unwrap along TIME is done once over the whole track, not per window. That
    is exact: ``np.unwrap`` builds its correction from consecutive differences,
    so within any window the two agree up to a per-node constant, which the
    per-window linear detrend removes.
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
    # FILL the time axis, never compact it: np.hanning(m), rfftfreq(m, 1/fps) and
    # BEAT_MIN_CYCLES all assume uniform sampling at 1/fps.
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

    # Unwrap along the arclength FIRST. Time alone is not enough: the node at
    # which the wave crosses +-pi moves as the wave travels, so a fixed node
    # jumps by 2*pi between frames and every amplitude comes out near 2*pi.
    psi = np.unwrap(np.arctan2(dy, dx), axis=1)

    # A zero-length flagellum (a collapsed detection) divides by zero in the
    # normalised arclength and puts inf into a cumulative sum, poisoning every
    # window downstream. Treat it exactly like an undetected frame.
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

    # exp(1j * psi) is EXACTLY exp(1j * raw_psi): the arclength unwrap only adds
    # multiples of 2*pi, which the exponential does not see. This is the circular
    # mean of the RAW angle the static mode requires, taken BEFORE the time
    # unwrap, whose 2*pi branch would otherwise leak into it.
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

    # -- 3. per-window geometry, from sliding sums --------------------------- #
    s = _sliding_mean(node_s, m, starts)                    # (n_win, n_seg)
    mean_length = _sliding_mean(length[:, None], m, starts)[:, 0]
    mean_dir = _sliding_mean(tangent, m, starts)            # (n_win, n_seg) complex
    filled_frac = 1.0 - _sliding_mean(complete[:, None].astype(float), m, starts)[:, 0]
    out["wave_interp_frac"] = filled_frac

    # Above MAX_INTERP_FRAC filled frames a spectral estimate describes the
    # interpolator, not the cell (the 20% badFileFlag in
    # computeGoodFrameRange.m). Applied PER WINDOW: one bad patch costs the
    # windows over it, not the track.
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
    # NOTE psi_0 never reaches the spectrum: it is constant in time within a
    # window and the detrend removes the mean, so `psi - psi_0` and `psi` have
    # the SAME transform at every non-DC bin. Hence the STFT runs on psi.

    # -- 5. the fundamental -------------------------------------------------- #
    # Band-limited exactly as `_beat_estimate` is: a raw argmax on a red spectrum
    # always returns the lowest bin, and every shape descriptor is read AT that
    # bin.
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
    # A window whose tangent angle does not vary has no beat, so whichever bin
    # the argmax lands in is floating-point noise. This is a DEGENERACY test, not
    # a significance one: keep it scaled by the window's own power, so a cell
    # that is still but jittery does vary and IS reported.
    with np.errstate(invalid="ignore", divide="ignore"):
        moving = total > np.finfo(float).eps * max(1.0, float(np.nanmax(total, initial=0.0)))
    usable = usable & moving

    with np.errstate(invalid="ignore", divide="ignore"):
        excess = np.where(baseline > 0, peak_power / baseline, np.nan)
        frac = np.where(total > 0, peak_power / total, np.nan)
    out["wave_excess"] = np.where(usable, excess, np.nan)
    out["wave_fundamental_frac"] = np.where(usable, frac, np.nan)
    # Censored: an argmax on the bottom edge of the band measures the estimator
    # running out of band, not that frequency - the real oscillation is at or
    # below the floor. Flag it rather than blanking it.
    censored = (local == 0)
    out["wave_censored"] = np.where(usable, censored.astype(float), np.nan)
    # Reported, not gated: a fixed excess threshold passes much of pure red
    # noise. `wave_phase_rsq` is what separates a wave from a random walk, so
    # condition on that rather than on excess.
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
    # Normalised by the mean, as the paper does, so shape is separate from size.
    out["wave_amp_asym"] = np.where(measured, amp_c[:, 1] / scale, np.nan)
    out["wave_amp_para"] = np.where(measured, amp_c[:, 2] / scale, np.nan)

    phase_c = _legendre_batch(phase, s)
    out["wave_phase_asym"] = np.where(usable, phase_c[:, 1], np.nan)
    out["wave_phase_para"] = np.where(usable, phase_c[:, 2], np.nan)

    # -- 6. wavelength and direction from the phase gradient ----------------- #
    # Fitted against PHYSICAL arclength, which keeps the wavelength in pixels and
    # so makes lambda/L comparable between cells of different length.
    arc_px = s * mean_length[:, None]
    slope, rsq = _line_fit(arc_px, phase)                   # rad per px
    slope = np.where(usable, slope, np.nan)
    out["wave_phase_slope"] = slope
    out["wave_phase_rsq"] = np.where(usable, rsq, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        wavelength = np.where(np.abs(slope) > 1e-9, 2.0 * np.pi / np.abs(slope), np.nan)
    out["wave_wavelength_px"] = wavelength
    out["wave_wavelength_ratio"] = wavelength / mean_length
    # +1 means Base -> Tip. See the module docstring; verified against the labels.
    direction = -np.sign(slope)
    direction = np.where(np.isfinite(wavelength) & (direction != 0.0), direction, np.nan)
    out["wave_direction"] = direction
    out["wave_speed_px_s"] = direction * wavelength * out["wave_freq_hz"]

    # -- 7. beat stability ---------------------------------------------------- #
    # Two scales. `wave_freq_drift_hz` is the spread within +-one window length -
    # how unsteady the beat is HERE; `wave_freq_track_sd_hz` is the spread over
    # the whole track - whether this cell holds a frequency at all.
    #
    # Both measured only on windows that resolved a frequency: a censored window
    # reports the band edge, which does not move, so including them makes every
    # wandering cell look steady.
    freq = out["wave_freq_hz"]
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
) -> pd.DataFrame:
    """:func:`spectral_series` straight off one track's rows in ``tracks.parquet``.

    Missing frames are absent ROWS in that file, so the keypoints must be laid on
    the FULL frame grid first; packed rows would scale every reported frequency.
    ``f_start`` and ``n_frames`` default to the track's own extent, so the index
    is in real frame numbers.
    """
    sub = sub.sort_values("frame")
    frames = sub["frame"].to_numpy(dtype=int)
    if frames.size == 0:
        return spectral_series(np.empty((0, 0)), np.empty((0, 0)), fps, window, hop)
    if f_start is None:
        f_start = int(frames.min())
    if n_frames is None:
        n_frames = int(frames.max()) - int(f_start) + 1
    kx = [f"kp_x_{k}" for k in range(n_keypoints)]
    ky = [f"kp_y_{k}" for k in range(n_keypoints)]
    px = _on_grid(sub, kx, int(n_frames), int(f_start))
    py = _on_grid(sub, ky, int(n_frames), int(f_start))
    series = spectral_series(px, py, fps, window, hop)
    series.index = pd.Index(series.index.to_numpy(dtype=int) + int(f_start), name="frame")
    return series


# --------------------------------------------------------------------------- #
# reading the series out over one annotation window
# --------------------------------------------------------------------------- #
def window_values(series: pd.DataFrame, f_start: int, f_end: int) -> dict[str, float]:
    """Collapse the series over one annotation window into a feature row.

    Uses every row whose window CENTRE falls in ``[f_start, f_end)`` - half-open,
    so adjacent clips never both claim the same row. When none do (a clip shorter
    than the hop, or one at a track end where the window could not be centred)
    the single nearest row by centre distance is used, and never more than one.
    The shift stays visible: ``n_windows`` is 1 and ``centre_offset`` gives the
    distance from the middle of the requested range, so a strict caller can
    filter on it.

    Combined by median, ignoring NaN, except:

    * ``wave_direction``, recomputed as ``-sign(median wave_phase_slope)``. A
      median of an even number of +-1 values ties at exactly 0.0, which is no
      direction at all but reads as one; the slope has no such tie.
    * ``wave_censored``, a MEAN - the share of this window's analysis positions
      that ran out of band, which beats a 0/1 median.

    Median rather than mean, because these columns have heavy tails by
    construction: ``wave_wavelength_ratio`` blows up whenever a phase profile is
    not linear, and one such window would drag a mean anywhere.

    Returns plain floats, NaN for anything undefined, plus ``n_windows`` and
    ``centre_offset``. Never fills with 0.0.
    """
    out: dict[str, float] = {c: np.nan for c in SERIES_COLUMNS}
    out["n_windows"] = 0.0
    out["centre_offset"] = np.nan
    if series is None or len(series) == 0:
        return out

    centres = series.index.to_numpy(dtype=float)
    target = (float(f_start) + float(f_end)) / 2.0
    inside = (centres >= float(f_start)) & (centres < float(f_end))
    if inside.any():
        rows = series.loc[inside]
    else:
        rows = series.iloc[[int(np.argmin(np.abs(centres - target)))]]

    out["n_windows"] = float(len(rows))
    used = rows.index.to_numpy(dtype=float)
    out["centre_offset"] = float(np.median(used) - target)
    for column in SERIES_COLUMNS:
        if column not in rows:
            continue
        values = rows[column].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        if column in _FRACTION_COLUMNS:
            out[column] = float(values.mean())
        else:
            out[column] = float(np.median(values))

    slope = out.get("wave_phase_slope", np.nan)
    out["wave_direction"] = _safe(-float(np.sign(slope)) if np.isfinite(slope) and slope != 0.0
                                  else np.nan)
    # Recompute the speed from the three aggregated numbers: the median of a
    # column whose sign flipped partway makes {+400, -400} a speed of 0.
    freq, wavelength = out.get("wave_freq_hz", np.nan), out.get("wave_wavelength_px", np.nan)
    out["wave_speed_px_s"] = _safe(out["wave_direction"] * wavelength * freq)
    return out
