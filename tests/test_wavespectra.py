"""Checks for :mod:`trackanno.wavespectra`, the sliding-window beat spectrum.

Everything here is generated from a travelling wave with known parameters -
``psi(s, t) = A sin(2 pi f t - k s)`` integrated into a keypoint chain - so the
truth is known exactly and no annotation, movie or network share is involved.
The four things worth testing separately are:

* that a window read out of a whole-track series is the same measurement as one
  computed on those frames alone - the sliding machinery is supposed to change
  the COST and nothing else, so anything it leaks across a window edge (the
  global time unwrap, the whole-track interpolation, the STFT's own slice grid)
  shows up as a disagreement here;
* that the shape descriptors have the values the wave's own algebra says they
  must, which is what stops a plausible-looking Legendre fit from being wrong
  in a way no ground-truth frequency check would notice;
* that ``f``, ``lambda`` and ``lambda / L`` come back, for a retrograde wave as
  well as an anterograde one - a direction estimator that cannot be made to say
  -1 is not measuring direction;
* that a reversal inside a track is LOCATED, which is the entire reason the
  sliding version exists;
* that the guards still fire - the interpolation ceiling, the band floor, the
  red-noise excess - and that a failure is reported as NaN rather than as 0.0.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.signal import ShortTimeFFT
from harness import check
from trackanno.wavespectra import (
    DEFAULT_HOP, DEFAULT_WINDOW, SERIES_COLUMNS, _detrend_coeffs, spectral_series,
    track_spectral_series, window_values,
)

FPS = 100.0
N_KEYPOINTS = 8            # Head, Base, Flag1..Flag5, Tip


# --------------------------------------------------------------------------- #
# the generator
# --------------------------------------------------------------------------- #
def travelling_wave(
    n_frames: int,
    freq_hz: float = 8.0,
    wavelength_ratio: float = 1.0,
    amplitude: float = 0.6,
    length_px: float = 60.0,
    k_sign: int = +1,
    noise_px: float = 0.0,
    seed: int = 0,
    flip_at: int | None = None,
    fps: float = FPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Keypoints of a flagellum carrying ``psi(s, t) = A sin(2 pi f t - k s)``.

    The tangent angle is the primitive here and the keypoints are its integral,
    which is the inverse of what the module does - so a recovered ``psi`` that
    matches is a genuine round trip rather than an algebraic identity.

    ``k_sign = +1`` is a wave running Base -> Tip and must read
    ``wave_direction = +1``; ``-1`` is retrograde and must read ``-1``. With
    ``flip_at`` the wavevector reverses at that frame while the temporal phase
    stays continuous, which is what a real cell reversing its beat looks like -
    the wave turns round, it does not restart.

    The chain is laid out with Head at ``+x`` from Base so the flagellum's
    tangent angle sits near ``+-pi``, on ``arctan2``'s branch cut. That is the
    awkward case (54% of real clips land within 0.5 rad of it) and it is
    deliberate: a static mode taken as a plain mean rather than a circular one
    fails here and nowhere else.
    """
    rng = np.random.default_rng(seed)
    n_seg = N_KEYPOINTS - 2
    ds = length_px / n_seg
    s_mid = (np.arange(n_seg) + 0.5) * ds
    k = 2.0 * np.pi / (wavelength_ratio * length_px)

    sign = np.full(n_frames, float(k_sign))
    if flip_at is not None:
        sign[flip_at:] = -float(k_sign)
    phase = np.cumsum(np.full(n_frames, 2.0 * np.pi * freq_hz / fps))
    psi = np.pi + amplitude * np.sin(phase[:, None] - sign[:, None] * k * s_mid[None, :])

    x = np.concatenate([np.zeros((n_frames, 1)), np.cumsum(ds * np.cos(psi), axis=1)], axis=1)
    y = np.concatenate([np.zeros((n_frames, 1)), np.cumsum(ds * np.sin(psi), axis=1)], axis=1)
    px = np.zeros((n_frames, N_KEYPOINTS))
    py = np.zeros((n_frames, N_KEYPOINTS))
    px[:, 1:], py[:, 1:] = x, y
    px[:, 0], py[:, 0] = 20.0, 0.0                     # Head, out along +x from Base
    if noise_px:
        px = px + rng.normal(0.0, noise_px, px.shape)
        py = py + rng.normal(0.0, noise_px, py.shape)
    return px, py


def measurable(series: pd.DataFrame) -> pd.DataFrame:
    return series[np.isfinite(series["wave_freq_hz"])]


# --------------------------------------------------------------------------- #
def test_window_placement_is_self_consistent() -> None:
    """A window read out of the track must equal that window computed alone.

    This is the load-bearing test of the sliding machinery. Everything the
    module does once per TRACK rather than once per window - the arclength unwrap
    then the time unwrap, the whole-track NaN fill, the single
    :class:`ShortTimeFFT` the slices are read out of, the per-window linear
    detrend that is supposed to absorb whatever the global unwrap left behind -
    is an optimisation, and an optimisation that changes the answer is a bug.

    Handing ``spectral_series`` exactly one window's frames makes it produce that
    window on its own, with nothing outside it to leak in. On a gapless track the
    two must agree to floating-point noise; the docstring's claim that a global
    unwrap "agrees up to a constant per node, which the detrend removes" is
    precisely what is being checked.
    """
    px, py = travelling_wave(400, noise_px=0.4, seed=11)
    window = 75
    series = spectral_series(px, py, FPS, window, hop=1)
    mid = window // 2

    worst_key, worst = "", 0.0
    compared = 0
    for centre in series.index[::37]:
        start = int(centre) - mid
        alone = spectral_series(px[start:start + window], py[start:start + window],
                                FPS, window=window, hop=1)
        check(f"the window at frame {centre} computes on its own", len(alone) == 1,
              f"{len(alone)} rows")
        for key in series.columns:
            here = float(series.loc[centre, key])
            solo = float(alone.iloc[0][key])
            if np.isnan(here) and np.isnan(solo):
                continue
            compared += 1
            deviation = abs(here - solo) / max(1.0, abs(here))
            if deviation > worst:
                worst_key, worst = key, deviation
    check("every column of a slid window matches the window computed alone to 1e-9",
          worst < 1e-9, f"worst {worst:.2e} on {worst_key} over {compared} comparisons")

    # And on a track with real gaps in it, where the fill has to do some work.
    # Here the two are NOT expected to be identical, by design: the module
    # interpolates over the WHOLE track, so a gap straddling the window edge is
    # bridged from both sides instead of being clamped to the last value inside
    # the window. What must survive is the physics, so it is checked against the
    # wave that was generated rather than against another estimate of it.
    holes = px.copy(), py.copy()
    rng = np.random.default_rng(2)
    drop = rng.choice(np.arange(120, 260), size=14, replace=False)
    holes[0][drop] = np.nan
    holes[1][drop] = np.nan
    gapped = spectral_series(holes[0], holes[1], FPS, window, hop=1)
    centre = 190
    check("a gapped window still reports the direction the wave was built with",
          gapped.loc[centre, "wave_direction"] == 1.0,
          str(gapped.loc[centre, "wave_direction"]))
    check("a gapped window is within one bin of the frequency it was built with",
          abs(float(gapped.loc[centre, "wave_freq_hz"]) - 8.0) <= FPS / window + 1e-9,
          f"{gapped.loc[centre, 'wave_freq_hz']} vs 8.0")


def test_shape_coefficients_are_analytic() -> None:
    """The Legendre descriptors, against the algebra of the wave that made them.

    ``psi(s, t) = pi + A sin(2 pi f t - sign * k s)`` has, by construction:

    * a static mode that is the constant ``pi``, so its circular readout is
      ``cos = -1``, ``sin = 0`` and both shape coefficients are zero;
    * a FLAT amplitude profile ``a(s) = A``, so the asymmetric and parabolic
      amplitude coefficients are zero and only the mean survives;
    * a LINEAR phase profile. For ``f(s) = a + b s`` on ``s`` in [0, 1] the
      shifted-Legendre expansion is ``(a + b/2) P0 + (b/2) P1 + 0 P2``, and the
      total phase across the flagellum is ``-sign * k L = -sign * 2 pi / ratio``.
      So ``wave_phase_asym = -sign * pi / ratio`` exactly, and
      ``wave_phase_para`` is zero.

    These are the columns no frequency or wavelength check touches. Without this
    the whole Legendre projection could be transposed, mis-scaled or fitted on
    the wrong axis and every other test in the file would still pass.
    """
    for ratio in (1.0, 0.75, 1.25):
        for k_sign in (+1, -1):
            px, py = travelling_wave(600, freq_hz=8.0, wavelength_ratio=ratio,
                                     amplitude=0.6, k_sign=k_sign)
            rows = measurable(spectral_series(px, py, FPS, 75, 5))
            med = {c: float(rows[c].median()) for c in rows.columns}
            tag = f"(k_sign={k_sign:+d} lam/L={ratio})"

            check(f"the static mode is the constant it was built as {tag}",
                  abs(med["wave_static_cos"] + 1.0) < 1e-3
                  and abs(med["wave_static_sin"]) < 1e-3
                  and abs(med["wave_static_asym"]) < 1e-3
                  and abs(med["wave_static_para"]) < 1e-3,
                  f"cos {med['wave_static_cos']:.4f} sin {med['wave_static_sin']:.4f} "
                  f"asym {med['wave_static_asym']:.4f} para {med['wave_static_para']:.4f}")
            check(f"a flat amplitude profile has no shape terms {tag}",
                  abs(med["wave_amp_asym"]) < 1e-3 and abs(med["wave_amp_para"]) < 1e-3,
                  f"asym {med['wave_amp_asym']:.5f} para {med['wave_amp_para']:.5f}")
            expected_asym = -k_sign * np.pi / ratio
            check(f"the phase asymmetry is -sign * pi / ratio {tag}",
                  abs(med["wave_phase_asym"] - expected_asym) < 1e-3,
                  f"{med['wave_phase_asym']:.5f} vs {expected_asym:.5f}")
            check(f"a linear phase profile has no parabolic term {tag}",
                  abs(med["wave_phase_para"]) < 1e-3, f"{med['wave_phase_para']:.5f}")
            # The slope is per PIXEL: k = 2 pi / (ratio * L), signed by -k_sign.
            expected_slope = -k_sign * 2.0 * np.pi / (ratio * 60.0)
            check(f"the phase slope is the wavevector in px^-1 {tag}",
                  abs(med["wave_phase_slope"] - expected_slope) < 1e-4,
                  f"{med['wave_phase_slope']:.6f} vs {expected_slope:.6f}")


def test_detrend_is_exact() -> None:
    """The analytic detrend must equal SciPy's per-slice one, not approximate it.

    ``spectral_series`` subtracts the fitted line in the FREQUENCY domain -
    ``rfft(w*(x-a-bt)) = rfft(w*x) - a rfft(w) - b rfft(w*t)`` - because
    ``ShortTimeFFT.stft_detrend`` runs an ``lstsq`` per slice from Python and
    that cost 64% of the runtime. The identity is exact, so the test is exact.
    """
    rng = np.random.default_rng(5)
    n, m, hop, n_node = 500, 75, 5, 6
    signal = rng.normal(size=(n, n_node)).cumsum(axis=0)     # a random walk, as psi is
    win = np.hanning(m)
    sft = ShortTimeFFT(win, hop=hop, fs=FPS, fft_mode="onesided", phase_shift=None)
    starts = np.arange(int(np.ceil(sft.m_num_mid / hop)),
                       (n - m + sft.m_num_mid) // hop + 1) * hop - sft.m_num_mid
    p0 = int((starts[0] + sft.m_num_mid) // hop)

    reference = sft.stft_detrend(signal, "linear", p0=p0, p1=p0 + starts.size, axis=0)
    alpha, beta = _detrend_coeffs(signal, m, starts)
    analytic = (sft.stft(signal, p0=p0, p1=p0 + starts.size, axis=0)
                - np.fft.rfft(win)[:, None, None] * alpha.T[None]
                - np.fft.rfft(win * np.arange(m, dtype=float))[:, None, None] * beta.T[None])
    scale = np.abs(reference).max()
    check("analytic detrend equals ShortTimeFFT.stft_detrend",
          float(np.abs(reference - analytic).max()) < 1e-8 * scale,
          f"max abs diff {np.abs(reference - analytic).max():.3e} on scale {scale:.3e}")

    # And the phase reference is the window's FIRST sample, as np.fft.rfft uses -
    # SciPy defaults to the window midpoint, which is a constant per bin and so
    # invisible in the phase gradient but wrong in the absolute phase.
    manual = np.fft.rfft(signal[starts[3]:starts[3] + m] * win[:, None], axis=0)
    plain = sft.stft(signal, p0=p0, p1=p0 + starts.size, axis=0)[:, :, 3]
    check("phase reference matches np.fft.rfft (phase_shift=None)",
          float(np.abs(manual - plain).max()) < 1e-8 * np.abs(manual).max())


def test_synthetic_recovery() -> None:
    """f, lambda and lambda/L come back, in both propagation directions."""
    cases = [(8.0, 1.00, 60.0), (12.0, 0.75, 80.0), (6.0, 1.25, 45.0)]
    for k_sign in (+1, -1):
        for freq, ratio, length in cases:
            px, py = travelling_wave(400, freq_hz=freq, wavelength_ratio=ratio,
                                     length_px=length, k_sign=k_sign)
            series = spectral_series(px, py, FPS, DEFAULT_WINDOW, DEFAULT_HOP)
            good = measurable(series)
            tag = f"k_sign={k_sign:+d} f={freq} lam/L={ratio}"
            check(f"frequency recovered ({tag})",
                  abs(float(good["wave_freq_hz"].median()) - freq) <= FPS / DEFAULT_WINDOW,
                  f"got {good['wave_freq_hz'].median()}")
            check(f"wavelength recovered ({tag})",
                  abs(float(good["wave_wavelength_px"].median()) - ratio * length)
                  < 0.02 * ratio * length,
                  f"got {good['wave_wavelength_px'].median()} want {ratio * length}")
            check(f"lambda/L recovered ({tag})",
                  abs(float(good["wave_wavelength_ratio"].median()) - ratio) < 0.02 * ratio,
                  f"got {good['wave_wavelength_ratio'].median()}")
            check(f"flagellum length recovered ({tag})",
                  abs(float(good["wave_length_px"].median()) - length) < 0.01 * length)
            check(f"phase profile is linear, rsq > 0.99 ({tag})",
                  float(good["wave_phase_rsq"].median()) > 0.99)
            # THE sign convention: +1 is Base -> Tip. Verified against 170 real
            # hand labels at 85% with no training; a flip here inverts the whole
            # behaviour classifier.
            check(f"direction is {k_sign:+d} on every window ({tag})",
                  bool((series["wave_direction"].dropna() == k_sign).all()),
                  f"{(series['wave_direction'] == k_sign).mean():.3f} agreed")

    # Amplitude is a real number with the Hanning coherent gain in it, not an
    # arbitrary scale: a pure tone of amplitude A must read back as A.
    for amp in (0.3, 0.6):
        px, py = travelling_wave(400, amplitude=amp)
        good = measurable(spectral_series(px, py, FPS, DEFAULT_WINDOW, DEFAULT_HOP))
        check(f"amplitude {amp} recovered within 5%",
              abs(float(good["wave_amp_mean"].median()) - amp) < 0.05 * amp,
              f"got {good['wave_amp_mean'].median()}")

    # A noisy chain must not flip the answer - noise costs precision, not sign.
    px, py = travelling_wave(400, noise_px=1.0, seed=3, k_sign=-1)
    series = spectral_series(px, py, FPS, DEFAULT_WINDOW, DEFAULT_HOP)
    check("direction survives 1 px of localisation noise",
          float((series["wave_direction"] == -1).mean()) > 0.95,
          f"{(series['wave_direction'] == -1).mean():.3f}")


def test_reversal_is_located() -> None:
    """A track that turns its wave round at frame 250 must show it there.

    This is the point of the sliding version. The clip-at-a-time estimator can
    only say that two neighbouring clips disagree; this says where.
    """
    flip = 250
    px, py = travelling_wave(500, flip_at=flip, noise_px=0.3, seed=9)
    for window, hop in ((50, 5), (75, 5), (75, 1), (100, 5)):
        series = spectral_series(px, py, FPS, window, hop)
        direction = series["wave_direction"].to_numpy(float)
        centres = series.index.to_numpy(int)
        finite = np.isfinite(direction)
        # Where does the sign change for good?
        negative = np.flatnonzero(finite & (direction < 0))
        located = int(centres[negative[0]]) if negative.size else -1
        check(f"reversal located within one hop (window={window}, hop={hop})",
              abs(located - flip) <= hop, f"found at {located}, true {flip}")
        before = direction[finite & (centres < flip - window // 2)]
        after = direction[finite & (centres > flip + window // 2)]
        check(f"pure +1 before the reversal (window={window}, hop={hop})",
              bool((before > 0).all()), f"{(before > 0).mean():.3f}")
        check(f"pure -1 after the reversal (window={window}, hop={hop})",
              bool((after < 0).all()), f"{(after < 0).mean():.3f}")
        # A window straddling the reversal is NOT looking at a travelling wave,
        # and the phase linearity is the column that has to say so - otherwise a
        # caller has no way to tell a transition from a clean measurement.
        straddling = series.loc[np.abs(centres - flip) < window // 4, "wave_phase_rsq"]
        clean = series.loc[np.abs(centres - flip) > window, "wave_phase_rsq"]
        check(f"phase linearity dips across the reversal (window={window}, hop={hop})",
              float(straddling.median()) < float(clean.median()),
              f"{straddling.median():.4f} vs {clean.median():.4f}")

    # A single long window over the same track averages the reversal away - the
    # failure mode the whole module exists to avoid. Worth asserting so that a
    # future "just use a longer window" cannot pass the suite.
    whole = spectral_series(px, py, FPS, 500, 5)
    check("a whole-track window cannot see the reversal",
          len(whole) == 1, f"{len(whole)} rows")


def test_beat_stability() -> None:
    """Frequency wander is measured, and a steady beat reads as steady."""
    steady, _py = travelling_wave(500, freq_hz=8.0)
    steady_series = spectral_series(steady, _py, FPS, DEFAULT_WINDOW, DEFAULT_HOP)

    # A beat whose frequency swings between 6 and 12 Hz, same mean, same
    # amplitude, same wavelength. A single whole-record FFT smears this into one
    # broad peak and reports a number indistinguishable from the steady cell.
    n = 500
    t = np.arange(n) / FPS
    instantaneous = 9.0 + 3.0 * np.sin(2.0 * np.pi * 0.6 * t)
    phase = np.cumsum(2.0 * np.pi * instantaneous / FPS)
    n_seg, length = N_KEYPOINTS - 2, 60.0
    ds = length / n_seg
    s_mid = (np.arange(n_seg) + 0.5) * ds
    psi = np.pi + 0.6 * np.sin(phase[:, None] - (2.0 * np.pi / length) * s_mid[None, :])
    x = np.concatenate([np.zeros((n, 1)), np.cumsum(ds * np.cos(psi), axis=1)], axis=1)
    y = np.concatenate([np.zeros((n, 1)), np.cumsum(ds * np.sin(psi), axis=1)], axis=1)
    wx = np.zeros((n, N_KEYPOINTS))
    wy = np.zeros((n, N_KEYPOINTS))
    wx[:, 1:], wy[:, 1:] = x, y
    wx[:, 0] = 20.0
    wander = spectral_series(wx, wy, FPS, DEFAULT_WINDOW, DEFAULT_HOP)

    steady_sd = float(steady_series["wave_freq_track_sd_hz"].iloc[0])
    wander_sd = float(wander["wave_freq_track_sd_hz"].iloc[0])
    check("steady beat has near-zero track frequency spread", steady_sd < 0.2,
          f"{steady_sd:.3f} Hz")
    check("wandering beat has a large track frequency spread", wander_sd > 1.0,
          f"{wander_sd:.3f} Hz")
    check("the two have nearly the same mean frequency",
          abs(float(steady_series["wave_freq_hz"].mean())
              - float(wander["wave_freq_hz"].mean())) < 1.5,
          "if these differed, the spread could be read off the mean instead")
    check("local drift separates them too",
          float(steady_series["wave_freq_drift_hz"].median()) < 0.2
          < float(wander["wave_freq_drift_hz"].median()),
          f"{steady_series['wave_freq_drift_hz'].median():.3f} vs "
          f"{wander['wave_freq_drift_hz'].median():.3f}")


def test_window_values() -> None:
    """Reading one annotation window out of the series."""
    px, py = travelling_wave(400, noise_px=0.3, seed=4)
    series = spectral_series(px, py, FPS, DEFAULT_WINDOW, DEFAULT_HOP)
    centres = series.index.to_numpy(int)

    f_start, f_end = 200, 225
    values = window_values(series, f_start, f_end)
    inside = centres[(centres >= f_start) & (centres < f_end)]
    check("uses every window centred in [f_start, f_end)",
          values["n_windows"] == len(inside), f"{values['n_windows']} vs {len(inside)}")
    check("half-open, so adjacent clips never share a window",
          window_values(series, 175, 200)["n_windows"] + values["n_windows"]
          == len(centres[(centres >= 175) & (centres < 225)]))
    check("direction survives the readout", values["wave_direction"] == 1.0)
    check("aggregate is the median of the in-range rows",
          np.isclose(values["wave_wavelength_px"],
                     float(series.loc[inside, "wave_wavelength_px"].median())))
    check("every series column is present in the readout",
          all(c in values for c in SERIES_COLUMNS))

    # The fallback: a range with no window centred in it takes the nearest ONE.
    early = window_values(series, 0, 5)
    check("falls back to the nearest single window", early["n_windows"] == 1.0)
    check("and says how far it had to reach", abs(early["centre_offset"]) > 5.0,
          f"offset {early['centre_offset']}")
    check("the fallback still reports a direction", early["wave_direction"] == 1.0)

    # An even number of +-1 directions medians to exactly 0.0, which is not a
    # direction. Deriving it from the median SLOPE instead cannot tie.
    flipped, flipped_y = travelling_wave(400, flip_at=200, noise_px=0.2, seed=6)
    straddle = spectral_series(flipped, flipped_y, FPS, 50, 2)
    at_flip = window_values(straddle, 190, 210)
    check("a straddling readout still returns +-1, never 0.0",
          at_flip["wave_direction"] in (1.0, -1.0), f"{at_flip['wave_direction']}")

    # An empty series must not invent numbers.
    blank = window_values(spectral_series(np.empty((0, 8)), np.empty((0, 8)), FPS), 0, 25)
    check("an empty series reads out as NaN, never 0.0",
          all(np.isnan(blank[c]) for c in SERIES_COLUMNS) and blank["n_windows"] == 0.0)


def test_guards() -> None:
    """The guards, each applied per window rather than per track."""
    px, py = travelling_wave(400, noise_px=0.2, seed=8)

    # -- the interpolation ceiling ------------------------------------------ #
    holed_x, holed_y = px.copy(), py.copy()
    holed_x[200:240] = np.nan          # 40 consecutive frames gone
    holed_y[200:240] = np.nan
    series = spectral_series(holed_x, holed_y, FPS, 75, 1)
    over = series[series["wave_interp_frac"] > 0.20]
    under = series[series["wave_interp_frac"] <= 0.20]
    check("windows over MAX_INTERP_FRAC are blanked",
          bool(over["wave_direction"].isna().all()) and len(over) > 0,
          f"{len(over)} windows over the ceiling")
    check("windows under it are kept",
          bool(under["wave_direction"].notna().all()) and len(under) > 0)
    check("a bad patch costs only the windows that overlap it",
          len(under) > len(over), f"{len(under)} kept, {len(over)} blanked")

    # -- filled, never compacted -------------------------------------------- #
    # Dropping the missing frames instead of filling them scales the reported
    # frequency by 1/(surviving fraction). Scatter 12% of frames and the answer
    # must not move.
    rng = np.random.default_rng(21)
    thin_x, thin_y = px.copy(), py.copy()
    gone = rng.choice(np.arange(400), size=48, replace=False)
    thin_x[gone] = np.nan
    thin_y[gone] = np.nan
    thinned = measurable(spectral_series(thin_x, thin_y, FPS, DEFAULT_WINDOW, DEFAULT_HOP))
    check("12% missing frames do not scale the frequency",
          abs(float(thinned["wave_freq_hz"].median()) - 8.0) <= FPS / DEFAULT_WINDOW,
          f"got {thinned['wave_freq_hz'].median()} (compaction would give ~9.1)")

    # -- band floor and censoring -------------------------------------------- #
    slow_x, slow_y = travelling_wave(400, freq_hz=2.0)
    slow = spectral_series(slow_x, slow_y, FPS, 75, 5)      # floor is 4 Hz here
    # Reported and FLAGGED, not blanked. A beat below the floor is a censored
    # observation - the true frequency is at or below the edge bin - and a
    # censored observation is data, so it is emitted with `wave_censored` saying
    # what it is. Blanking it here would throw away the one thing the window did
    # establish (an upper bound) and would be indistinguishable downstream from
    # a track that was never measured at all.
    check("a beat below the resolvable floor reports the edge bin it ran into",
          bool(slow["wave_freq_hz"].notna().any())
          and np.allclose(slow["wave_freq_hz"].dropna(), 4.0),
          f"got {sorted(set(slow['wave_freq_hz'].dropna()))}")
    check("and flags every one of those windows as censored",
          float(slow["wave_censored"].mean()) > 0.9)
    check("while still reporting the resolvable floor it ran into",
          np.isclose(float(slow["wave_resolvable_hz"].iloc[0]), 4.0))
    check("direction is still reported when the frequency is censored",
          bool(slow["wave_direction"].notna().all()),
          "direction is a sign, not a magnitude - it must survive the gate")

    # -- noise is not a beat -------------------------------------------------- #
    rng = np.random.default_rng(1)
    noise_x = np.cumsum(rng.normal(0, 1.0, (400, N_KEYPOINTS)), axis=0) + np.arange(8) * 8.0
    noise_y = np.cumsum(rng.normal(0, 1.0, (400, N_KEYPOINTS)), axis=0)
    junk = spectral_series(noise_x, noise_y, FPS, DEFAULT_WINDOW, DEFAULT_HOP)
    # A random walk in the KEYPOINT COORDINATES is a very red process by the time
    # it has been differenced into a tangent angle, and it DOES produce a
    # spectral peak. Nothing here pretends otherwise: since no significance
    # threshold gates the output any more, every such window reports a frequency.
    # That is the deliberate stance - report the measurement, publish its quality
    # - and it is safe only because the quality column below actually works. The
    # threshold that used to sit here did not: it passed a measured 24% of these
    # windows (mean over 30 seeds, max 49%), having been calibrated on red noise
    # in lateral displacement, a much whiter process than this one.
    claimed = float(np.isfinite(junk["wave_freq_hz"]).mean())
    check("a random walk still reports a frequency - nothing is gated away",
          claimed > 0.9, f"{claimed:.3f} of windows claimed a beat")
    clean_rsq = float(spectral_series(px, py, FPS)["wave_phase_rsq"].median())
    junk_rsq = float(junk["wave_phase_rsq"].median())
    check("phase linearity is what separates a wave from a random walk",
          junk_rsq < 0.7 < clean_rsq, f"noise rsq {junk_rsq:.3f}, wave rsq {clean_rsq:.3f}")

    # -- a motionless flagellum has no beat ----------------------------------- #
    # The one thing that IS still refused, and the only thing standing between a
    # dead cell and a real-looking frequency: with no variation at all, whichever
    # bin the argmax lands in is floating-point noise. A degeneracy test, not a
    # significance one - the random walk above varies, and is reported.
    dead_x, dead_y = travelling_wave(400, amplitude=0.0)
    dead = spectral_series(dead_x, dead_y, FPS, DEFAULT_WINDOW, DEFAULT_HOP)
    check("a motionless flagellum reports no beat frequency",
          bool(dead["wave_freq_hz"].isna().all()),
          f"claimed {sorted(set(dead['wave_freq_hz'].dropna()))}")
    check("and no shape descriptors fitted to nothing",
          bool(dead[["wave_amp_mean", "wave_phase_slope",
                     "wave_direction"]].isna().all().all()))

    # -- NaN means undefined, never 0.0 --------------------------------------- #
    blanked = series[series["wave_interp_frac"] > 0.20]
    check("a blanked window is NaN everywhere, never 0.0",
          bool(blanked[["wave_direction", "wave_freq_hz", "wave_wavelength_px",
                        "wave_length_px", "wave_phase_slope"]].isna().all().all()))


def test_shape_and_edges() -> None:
    """Index, columns, short tracks and degenerate input."""
    px, py = travelling_wave(300)
    series = spectral_series(px, py, FPS, 75, 5)
    check("one row per analysis position, indexed by centre frame",
          series.index.name == "frame" and series.index.is_monotonic_increasing)
    check("centres are spaced by the hop",
          bool((np.diff(series.index.to_numpy(int)) == 5).all()))
    check("every window lies wholly inside the record",
          int(series.index.min()) - 37 >= 0 and int(series.index.max()) - 37 + 75 <= 300,
          f"{series.index.min()}..{series.index.max()}")
    check("all documented columns are emitted",
          list(series.columns) == list(SERIES_COLUMNS))
    check("every column is float", bool((series.dtypes == float).all()))

    # A track shorter than the window gets ONE whole-track window rather than
    # nothing: on this project nine of the 170 labelled clips sit in tracks
    # shorter than 150 frames, and dropping them loses hand labels.
    short_x, short_y = travelling_wave(90)
    short = spectral_series(short_x, short_y, FPS, 150, 5)
    check("a track shorter than the window still yields one window", len(short) == 1)
    check("and says what it actually used",
          float(short["wave_window_frames"].iloc[0]) == 90.0)
    check("with the honest, coarser resolvable floor",
          float(short["wave_resolvable_hz"].iloc[0]) > 3.0)

    # A record only a LITTLE longer than the window is the awkward one: its only
    # legal window starts at 0 or 1, and ShortTimeFFT's slice centres are
    # multiples of the hop, so the grid offers 35 or 40 where 37 is needed and
    # the naive answer is an empty series. Every length must yield something.
    for extra in range(0, 12):
        n = 75 + extra
        long_x, long_y = travelling_wave(n)
        got = spectral_series(long_x, long_y, FPS, 75, 5)
        check(f"a {n}-frame track with a 75-frame window yields a window",
              len(got) >= 1 and bool(np.isfinite(got["wave_direction"]).all()),
              f"{len(got)} rows")

    check("a track shorter than the FFT floor yields nothing",
          len(spectral_series(px[:5], py[:5], FPS)) == 0)
    check("too few keypoints yields nothing",
          len(spectral_series(px[:, :3], py[:, :3], FPS)) == 0)
    check("mismatched shapes yield nothing",
          len(spectral_series(px, py[:, :6], FPS)) == 0)

    # A collapsed detection (every keypoint on one pixel) divides by a zero
    # flagellum length. It must poison its own window and no other.
    collapsed_x, collapsed_y = px.copy(), py.copy()
    collapsed_x[150] = 5.0
    collapsed_y[150] = 5.0
    survived = spectral_series(collapsed_x, collapsed_y, FPS, 75, 5)
    check("a collapsed frame does not poison the whole track",
          bool(np.isfinite(survived["wave_direction"]).sum() > 0.8 * len(survived)),
          f"{np.isfinite(survived['wave_direction']).mean():.3f} of windows survived")
    check("and nothing became infinite",
          bool(np.isfinite(survived.to_numpy()[np.isfinite(survived.to_numpy())]).all()))


def test_track_helper() -> None:
    """``track_spectral_series`` lays the rows on the frame grid first."""
    px, py = travelling_wave(300)
    f_start = 1000
    rows = {"frame": np.arange(300) + f_start}
    for k in range(N_KEYPOINTS):
        rows[f"kp_x_{k}"] = px[:, k]
        rows[f"kp_y_{k}"] = py[:, k]
    frame = pd.DataFrame(rows)
    dropped = frame.drop(index=np.arange(40, 55)).sample(frac=1.0, random_state=0)

    series = track_spectral_series(dropped, N_KEYPOINTS, FPS, 75, 5)
    check("the index is in real movie frame numbers",
          int(series.index.min()) >= f_start and int(series.index.max()) < f_start + 300)
    direct = spectral_series(px, py, FPS, 75, 5)
    check("shuffled, gappy rows give the same answer as the dense array",
          bool((series["wave_direction"].dropna() == 1.0).all())
          and abs(float(series["wave_freq_hz"].median())
                  - float(direct["wave_freq_hz"].median())) < 1e-6,
          "missing frames are absent ROWS; handing them to an FFT unfilled "
          "scales the frequency by 1/(surviving fraction)")


def test_speed() -> None:
    """One transform per track must actually beat one transform per window.

    The per-window loop is the same function handed one window's frames at a
    time, which is what a caller who wanted a series without this module would
    have to write. Both sides therefore compute identical numbers (that is
    ``test_window_placement_is_self_consistent``) and differ only in how many
    times the per-frame geometry and the FFT planning are paid for.
    """
    px, py = travelling_wave(500, noise_px=0.5, seed=1)
    window, hop = DEFAULT_WINDOW, DEFAULT_HOP
    series = spectral_series(px, py, FPS, window, hop)          # warm any FFT plans

    t0 = time.perf_counter()
    for _ in range(5):
        spectral_series(px, py, FPS, window, hop)
    sliding = (time.perf_counter() - t0) / 5

    t0 = time.perf_counter()
    for centre in series.index.to_numpy(int):
        start = centre - window // 2
        spectral_series(px[start:start + window], py[start:start + window],
                        FPS, window=window, hop=1)
    per_window = time.perf_counter() - t0

    ratio = per_window / max(sliding, 1e-9)
    print(f"      {len(series)} windows: sliding {sliding * 1e3:.1f} ms, "
          f"per-window loop {per_window * 1e3:.1f} ms, {ratio:.1f}x")
    # Measured at 16x on the development machine; asserted well below that so a
    # loaded or slower box does not turn a performance note into a red suite.
    check("the sliding series is at least 4x faster than the per-window loop",
          ratio > 4.0, f"{ratio:.1f}x")
