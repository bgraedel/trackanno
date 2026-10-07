"""Tests for :mod:`trackanno.measure.wavespectra`, on synthetic travelling waves."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest
from scipy.signal import ShortTimeFFT
from trackanno.measure.wavespectra import (
    DEFAULT_FLOW_FRAMES, DEFAULT_HOP, DEFAULT_WINDOW, SERIES_COLUMNS,
    _detrend_coeffs, spectral_series, track_spectral_series, window_values,
)

FPS = 100.0
N_KEYPOINTS = 8            # Head, Base, Flag1..Flag5, Tip
N_SEG = N_KEYPOINTS - 2


def chain(psi: np.ndarray, length_px: float = 60.0) -> tuple[np.ndarray, np.ndarray]:
    """Keypoints whose segment tangent angles are ``psi`` (frames x segments)."""
    n_frames = psi.shape[0]
    ds = length_px / N_SEG
    x = np.concatenate([np.zeros((n_frames, 1)), np.cumsum(ds * np.cos(psi), axis=1)], axis=1)
    y = np.concatenate([np.zeros((n_frames, 1)), np.cumsum(ds * np.sin(psi), axis=1)], axis=1)
    px = np.zeros((n_frames, N_KEYPOINTS))
    py = np.zeros((n_frames, N_KEYPOINTS))
    px[:, 1:], py[:, 1:] = x, y
    px[:, 0], py[:, 0] = 20.0, 0.0                     # Head, out along +x from Base
    return px, py


def s_mid(length_px: float = 60.0) -> np.ndarray:
    return (np.arange(N_SEG) + 0.5) * length_px / N_SEG


def travelling_wave(
    n_frames: int,
    freq_hz: float = 8.0,
    wavelength_ratio: float = 1.0,
    amplitude: float = 0.6,
    length_px: float = 60.0,
    k_sign: int = +1,
    noise_px: float = 0.0,
    seed: int = 0,
    flip_at: int | tuple | None = None,
    fps: float = FPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Keypoints of ``psi(s, t) = pi + A sin(2 pi f t - k s)``; ``k_sign=+1`` runs Base to Tip.

    The angle sits near the arctan2 branch cut on purpose; ``flip_at`` reverses
    the wavevector at those frames with a continuous temporal phase.
    """
    rng = np.random.default_rng(seed)
    k = 2.0 * np.pi / (wavelength_ratio * length_px)
    sign = np.full(n_frames, float(k_sign))
    for i, at in enumerate(np.atleast_1d(flip_at) if flip_at is not None else ()):
        sign[int(at):] = float(k_sign) * (-1.0) ** (i + 1)
    phase = np.cumsum(np.full(n_frames, 2.0 * np.pi * freq_hz / fps))
    psi = np.pi + amplitude * np.sin(
        phase[:, None] - sign[:, None] * k * s_mid(length_px)[None, :])
    px, py = chain(psi, length_px)
    if noise_px:
        px = px + rng.normal(0.0, noise_px, px.shape)
        py = py + rng.normal(0.0, noise_px, py.shape)
    return px, py


def wandering_wave(n_frames: int, mean_hz: float, swing_hz: float):
    """A wave whose frequency swings sinusoidally around ``mean_hz`` at 0.6 Hz."""
    t = np.arange(n_frames) / FPS
    instantaneous = mean_hz + swing_hz * np.sin(2.0 * np.pi * 0.6 * t)
    phase = np.cumsum(2.0 * np.pi * instantaneous / FPS)
    psi = np.pi + 0.6 * np.sin(phase[:, None] - (2.0 * np.pi / 60.0) * s_mid()[None, :])
    return chain(psi)


def measurable(series: pd.DataFrame) -> pd.DataFrame:
    return series[np.isfinite(series["wave_freq_hz"])]


# --------------------------------------------------------------------------- #
# sliding windows
# --------------------------------------------------------------------------- #
def test_a_slid_window_equals_that_window_computed_alone() -> None:
    """Per-track work (unwrap, fill, one STFT) must not change any window's answer."""
    px, py = travelling_wave(400, noise_px=0.4, seed=11)
    window = 75
    series = spectral_series(px, py, FPS, window, hop=1)
    mid = window // 2

    worst_key, worst, compared = "", 0.0, 0
    for centre in series.index[::37]:
        start = int(centre) - mid
        alone = spectral_series(px[start:start + window], py[start:start + window],
                                FPS, window=window, hop=1)
        assert len(alone) == 1, f"window at frame {centre}: {len(alone)} rows"
        for key in series.columns:
            here = float(series.loc[centre, key])
            solo = float(alone.iloc[0][key])
            if np.isnan(here) and np.isnan(solo):
                continue
            compared += 1
            deviation = abs(here - solo) / max(1.0, abs(here))
            if deviation > worst:
                worst_key, worst = key, deviation
    assert worst < 1e-9, f"worst {worst:.2e} on {worst_key} over {compared} comparisons"


def test_a_gapped_window_still_recovers_the_wave() -> None:
    """Gaps are bridged from the whole track, so check against the generated wave."""
    px, py = travelling_wave(400, noise_px=0.4, seed=11)
    window = 75
    drop = np.random.default_rng(2).choice(np.arange(120, 260), size=14, replace=False)
    px[drop] = np.nan
    py[drop] = np.nan
    gapped = spectral_series(px, py, FPS, window, hop=1)
    assert gapped.loc[190, "wave_direction"] == 1.0
    assert abs(float(gapped.loc[190, "wave_freq_hz"]) - 8.0) <= FPS / window + 1e-9


@pytest.mark.parametrize("ratio", [1.0, 0.75, 1.25])
@pytest.mark.parametrize("k_sign", [+1, -1])
def test_shape_coefficients_are_analytic(ratio: float, k_sign: int) -> None:
    """Legendre descriptors match the wave's algebra: constant static mode, flat amplitude, linear phase."""
    px, py = travelling_wave(600, freq_hz=8.0, wavelength_ratio=ratio,
                             amplitude=0.6, k_sign=k_sign)
    rows = measurable(spectral_series(px, py, FPS, 75, 5))
    med = {c: float(rows[c].median()) for c in rows.columns}

    for key, want in (("wave_static_cos", -1.0), ("wave_static_sin", 0.0),
                      ("wave_static_asym", 0.0), ("wave_static_para", 0.0),
                      ("wave_amp_asym", 0.0), ("wave_amp_para", 0.0),
                      ("wave_phase_asym", -k_sign * np.pi / ratio),
                      ("wave_phase_para", 0.0)):
        assert abs(med[key] - want) < 1e-3, f"{key}: {med[key]:.5f} vs {want:.5f}"
    # The slope is per pixel: k = 2 pi / (ratio * L), signed by -k_sign.
    expected_slope = -k_sign * 2.0 * np.pi / (ratio * 60.0)
    assert abs(med["wave_phase_slope"] - expected_slope) < 1e-4


def test_analytic_detrend_equals_scipys() -> None:
    """The frequency-domain detrend is an exact identity with stft_detrend."""
    rng = np.random.default_rng(5)
    n, m, hop, n_node = 500, 75, 5, 6
    signal = rng.normal(size=(n, n_node)).cumsum(axis=0)
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
    assert float(np.abs(reference - analytic).max()) < 1e-8 * scale

    # The phase reference is the window's first sample, as in np.fft.rfft.
    manual = np.fft.rfft(signal[starts[3]:starts[3] + m] * win[:, None], axis=0)
    plain = sft.stft(signal, p0=p0, p1=p0 + starts.size, axis=0)[:, :, 3]
    assert float(np.abs(manual - plain).max()) < 1e-8 * np.abs(manual).max()


# --------------------------------------------------------------------------- #
# recovery
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k_sign", [+1, -1])
@pytest.mark.parametrize("freq,ratio,length", [(8.0, 1.00, 60.0), (12.0, 0.75, 80.0),
                                               (6.0, 1.25, 45.0)])
def test_frequency_wavelength_and_direction_come_back(freq, ratio, length, k_sign) -> None:
    px, py = travelling_wave(400, freq_hz=freq, wavelength_ratio=ratio,
                             length_px=length, k_sign=k_sign)
    series = spectral_series(px, py, FPS, DEFAULT_WINDOW, DEFAULT_HOP)
    good = measurable(series)
    assert abs(float(good["wave_freq_hz"].median()) - freq) <= FPS / DEFAULT_WINDOW
    assert abs(float(good["wave_wavelength_px"].median()) - ratio * length) < 0.02 * ratio * length
    assert abs(float(good["wave_wavelength_ratio"].median()) - ratio) < 0.02 * ratio
    assert abs(float(good["wave_length_px"].median()) - length) < 0.01 * length
    assert float(good["wave_phase_rsq"].median()) > 0.99
    # The sign convention: +1 is Base to Tip.
    assert bool((series["wave_direction"].dropna() == k_sign).all()), (
        f"{(series['wave_direction'] == k_sign).mean():.3f} agreed")


@pytest.mark.parametrize("amp", [0.3, 0.6])
def test_amplitude_reads_back_as_built(amp: float) -> None:
    """The Hanning coherent gain is divided out, so a tone of amplitude A reads A."""
    px, py = travelling_wave(400, amplitude=amp)
    good = measurable(spectral_series(px, py, FPS, DEFAULT_WINDOW, DEFAULT_HOP))
    assert abs(float(good["wave_amp_mean"].median()) - amp) < 0.05 * amp


def test_noise_does_not_flip_the_direction() -> None:
    px, py = travelling_wave(400, noise_px=1.0, seed=3, k_sign=-1)
    series = spectral_series(px, py, FPS, DEFAULT_WINDOW, DEFAULT_HOP)
    assert float((series["wave_direction"] == -1).mean()) > 0.95


# --------------------------------------------------------------------------- #
# reversals and flow direction
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("window,hop", [(50, 5), (75, 5), (75, 1), (100, 5)])
def test_reversal_is_located(window: int, hop: int) -> None:
    flip = 250
    px, py = travelling_wave(500, flip_at=flip, noise_px=0.3, seed=9)
    series = spectral_series(px, py, FPS, window, hop)
    direction = series["wave_direction"].to_numpy(float)
    centres = series.index.to_numpy(int)
    finite = np.isfinite(direction)
    negative = np.flatnonzero(finite & (direction < 0))
    located = int(centres[negative[0]]) if negative.size else -1
    assert abs(located - flip) <= hop, f"found at {located}, true {flip}"
    assert bool((direction[finite & (centres < flip - window // 2)] > 0).all())
    assert bool((direction[finite & (centres > flip + window // 2)] < 0).all())
    # A window straddling the reversal is not one travelling wave, and rsq says so.
    straddling = series.loc[np.abs(centres - flip) < window // 4, "wave_phase_rsq"]
    clean = series.loc[np.abs(centres - flip) > window, "wave_phase_rsq"]
    assert float(straddling.median()) < float(clean.median())


def test_a_whole_track_window_cannot_see_the_reversal() -> None:
    px, py = travelling_wave(500, flip_at=250, noise_px=0.3, seed=9)
    assert len(spectral_series(px, py, FPS, 500, 5)) == 1


@pytest.mark.parametrize("k_sign", [+1, -1])
def test_flow_direction_needs_no_resolvable_beat(k_sign: int) -> None:
    """A 2.5 Hz beat is below the window's floor; the flow columns still read it."""
    px, py = travelling_wave(400, freq_hz=2.5, k_sign=k_sign)
    series = spectral_series(px, py, FPS)
    flow = series["wave_flow_direction"].dropna()
    assert float(series["wave_censored"].mean()) > 0.5
    assert len(flow) > 0 and bool((flow == k_sign).all()), (
        f"{(flow == k_sign).mean():.3f} agreed over {len(flow)} windows")
    assert float(series["wave_flow_coherence"].median()) > 0.8


@pytest.mark.parametrize("window", [50, 150, 250])
def test_the_flow_direction_does_not_depend_on_the_spectral_window(window: int) -> None:
    px, py = travelling_wave(500, flip_at=250, noise_px=0.3, seed=9)
    reference = spectral_series(px, py, FPS, DEFAULT_WINDOW, DEFAULT_HOP)
    series = spectral_series(px, py, FPS, window, DEFAULT_HOP)
    shared = reference.index.intersection(series.index)
    assert len(shared) > 10
    for column in ("wave_flow", "wave_flow_coherence", "wave_flow_direction"):
        assert np.allclose(reference.loc[shared, column], series.loc[shared, column],
                           equal_nan=True), column
    # The resolvable band does move with the window, which is the contrast.
    assert not np.allclose(reference.loc[shared, "wave_resolvable_hz"],
                           series.loc[shared, "wave_resolvable_hz"], equal_nan=True)


@pytest.mark.parametrize("window", [50, 75, 150])
def test_flow_locates_a_reversal_within_a_hop(window: int) -> None:
    flip = 250
    px, py = travelling_wave(500, flip_at=flip, noise_px=0.3, seed=9)
    series = spectral_series(px, py, FPS, window, DEFAULT_HOP)
    centres = series.index.to_numpy(int)
    value = series["wave_flow_direction"].to_numpy(float)
    finite = np.isfinite(value)
    negative = np.flatnonzero(finite & (value < 0))
    located = int(centres[negative[0]]) if negative.size else -1
    assert abs(located - flip) <= DEFAULT_HOP, f"found at {located}, true {flip}"
    wrong = finite & (value != np.where(centres < flip, 1.0, -1.0))
    assert int(wrong.sum()) * DEFAULT_HOP <= 2 * DEFAULT_FLOW_FRAMES


def test_both_directions_see_a_bout_shorter_than_the_window() -> None:
    """A 30-frame reversal on a clean wave is seen by the spectral and the flow column."""
    px, py = travelling_wave(500, flip_at=(250, 280), noise_px=0.3, seed=9)
    series = spectral_series(px, py, FPS, DEFAULT_WINDOW, DEFAULT_HOP)
    centres = series.index.to_numpy(int)
    inside = (centres >= 250 + DEFAULT_FLOW_FRAMES) & (centres < 280 - DEFAULT_FLOW_FRAMES)
    for column in ("wave_direction", "wave_flow_direction"):
        value = series[column].to_numpy(float)[inside]
        assert float((value < 0).mean()) > 0.5, f"{column}: {(value < 0).mean():.3f}"


def test_a_standing_wave_has_no_flow_coherence() -> None:
    """``psi = A sin(wt) cos(ks)`` does not travel, so its coherence is low."""
    n_frames = 400
    phase = 2.0 * np.pi * 8.0 * np.arange(n_frames) / FPS
    psi = np.pi + 0.6 * np.sin(phase)[:, None] * np.cos(
        (2.0 * np.pi / 60.0) * s_mid())[None, :]
    standing = float(spectral_series(*chain(psi), FPS)["wave_flow_coherence"].median())
    travelling = float(spectral_series(*travelling_wave(n_frames), FPS)[
        "wave_flow_coherence"].median())
    assert standing < 0.2
    assert standing < 0.5 * travelling


def test_flow_survives_the_readout() -> None:
    px, py = travelling_wave(300, k_sign=-1)
    series = spectral_series(px, py, FPS)
    values = window_values(series, 100, 125)
    assert values["wave_flow_direction"] == -1.0
    # Taken from the signed coefficient, so it never ties at zero.
    assert np.sign(values["wave_flow"]) == values["wave_flow_direction"]
    assert 0.0 <= values["wave_flow_coherence"] <= 1.0

    empty = window_values(series.iloc[0:0], 0, 25)
    assert np.isnan(empty["wave_flow_direction"]) and np.isnan(empty["wave_flow"])


# --------------------------------------------------------------------------- #
# beat stability
# --------------------------------------------------------------------------- #
def test_a_wandering_beat_has_a_larger_spread_than_a_steady_one() -> None:
    steady = spectral_series(*travelling_wave(500, freq_hz=8.0), FPS,
                             DEFAULT_WINDOW, DEFAULT_HOP)
    wander = spectral_series(*wandering_wave(500, 9.0, 3.0), FPS,
                             DEFAULT_WINDOW, DEFAULT_HOP)
    assert float(steady["wave_freq_track_sd_hz"].iloc[0]) < 0.2
    assert float(wander["wave_freq_track_sd_hz"].iloc[0]) > 1.0
    # Same mean frequency, so the spread is not read off the mean.
    assert abs(float(steady["wave_freq_hz"].mean()) - float(wander["wave_freq_hz"].mean())) < 1.5
    assert (float(steady["wave_freq_drift_hz"].median()) < 0.2
            < float(wander["wave_freq_drift_hz"].median()))


def test_stability_is_undefined_when_every_window_is_censored() -> None:
    """A 1 to 3 Hz beat against a 4 Hz floor reports NaN spread and drift, not 0.0."""
    series = spectral_series(*wandering_wave(500, 2.0, 1.0), FPS, 75, DEFAULT_HOP)
    measured = series["wave_censored"].dropna()
    assert len(measured) > 0 and bool((measured == 1.0).all())
    assert np.isclose(float(series["wave_resolvable_hz"].iloc[0]), 4.0)
    assert np.allclose(series["wave_freq_hz"].dropna(), 4.0)
    assert bool(series["wave_freq_track_sd_hz"].isna().all())
    assert bool(series["wave_freq_drift_hz"].isna().all())


# --------------------------------------------------------------------------- #
# window readout
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def noisy_series() -> pd.DataFrame:
    px, py = travelling_wave(400, noise_px=0.3, seed=4)
    return spectral_series(px, py, FPS, DEFAULT_WINDOW, DEFAULT_HOP)


def test_window_values_reads_the_windows_centred_in_the_range(noisy_series) -> None:
    centres = noisy_series.index.to_numpy(int)
    values = window_values(noisy_series, 200, 225)
    inside = centres[(centres >= 200) & (centres < 225)]
    assert values["n_windows"] == len(inside)
    # Half-open, so adjacent clips never share a window.
    assert (window_values(noisy_series, 175, 200)["n_windows"] + values["n_windows"]
            == len(centres[(centres >= 175) & (centres < 225)]))
    assert values["wave_direction"] == 1.0
    assert np.isclose(values["wave_wavelength_px"],
                      float(noisy_series.loc[inside, "wave_wavelength_px"].median()))
    assert all(c in values for c in SERIES_COLUMNS)


def test_window_values_falls_back_to_the_nearest_window(noisy_series) -> None:
    early = window_values(noisy_series, 0, 5)
    assert early["n_windows"] == 1.0
    assert abs(early["centre_offset"]) > 5.0
    assert early["wave_direction"] == 1.0


def test_a_straddling_readout_never_ties_at_zero() -> None:
    """An even count of +-1 medians to 0.0; the median slope cannot tie."""
    px, py = travelling_wave(400, flip_at=200, noise_px=0.2, seed=6)
    at_flip = window_values(spectral_series(px, py, FPS, 50, 2), 190, 210)
    assert at_flip["wave_direction"] in (1.0, -1.0)


def test_an_empty_series_reads_out_as_nan() -> None:
    blank = window_values(spectral_series(np.empty((0, 8)), np.empty((0, 8)), FPS), 0, 25)
    assert all(np.isnan(blank[c]) for c in SERIES_COLUMNS)
    assert blank["n_windows"] == 0.0


# --------------------------------------------------------------------------- #
# guards
# --------------------------------------------------------------------------- #
def test_the_interpolation_ceiling_blanks_only_overlapping_windows() -> None:
    px, py = travelling_wave(400, noise_px=0.2, seed=8)
    px[200:240] = np.nan
    py[200:240] = np.nan
    series = spectral_series(px, py, FPS, 75, 1)
    over = series[series["wave_interp_frac"] > 0.20]
    under = series[series["wave_interp_frac"] <= 0.20]
    assert len(over) > 0 and bool(over["wave_direction"].isna().all())
    assert len(under) > len(over) and bool(under["wave_direction"].notna().all())
    # Blanked is NaN everywhere, never 0.0.
    assert bool(over[["wave_direction", "wave_freq_hz", "wave_wavelength_px",
                      "wave_length_px", "wave_phase_slope"]].isna().all().all())


def test_missing_frames_are_filled_not_compacted() -> None:
    """Dropping 12% of frames must not scale the frequency by 1/0.88."""
    px, py = travelling_wave(400, noise_px=0.2, seed=8)
    gone = np.random.default_rng(21).choice(np.arange(400), size=48, replace=False)
    px[gone] = np.nan
    py[gone] = np.nan
    thinned = measurable(spectral_series(px, py, FPS, DEFAULT_WINDOW, DEFAULT_HOP))
    assert abs(float(thinned["wave_freq_hz"].median()) - 8.0) <= FPS / DEFAULT_WINDOW


def test_a_beat_below_the_floor_is_reported_and_censored() -> None:
    """Censored windows report the 4 Hz edge bin and keep their direction."""
    slow = spectral_series(*travelling_wave(400, freq_hz=2.0), FPS, 75, 5)
    assert bool(slow["wave_freq_hz"].notna().any())
    assert np.allclose(slow["wave_freq_hz"].dropna(), 4.0)
    assert float(slow["wave_censored"].mean()) > 0.9
    assert np.isclose(float(slow["wave_resolvable_hz"].iloc[0]), 4.0)
    assert bool(slow["wave_direction"].notna().all())


def test_phase_linearity_separates_a_wave_from_a_random_walk() -> None:
    """Nothing is gated on significance; the rsq column is what tells noise apart."""
    rng = np.random.default_rng(1)
    noise_x = np.cumsum(rng.normal(0, 1.0, (400, N_KEYPOINTS)), axis=0) + np.arange(8) * 8.0
    noise_y = np.cumsum(rng.normal(0, 1.0, (400, N_KEYPOINTS)), axis=0)
    junk = spectral_series(noise_x, noise_y, FPS, DEFAULT_WINDOW, DEFAULT_HOP)
    assert float(np.isfinite(junk["wave_freq_hz"]).mean()) > 0.9
    px, py = travelling_wave(400, noise_px=0.2, seed=8)
    clean_rsq = float(spectral_series(px, py, FPS)["wave_phase_rsq"].median())
    junk_rsq = float(junk["wave_phase_rsq"].median())
    assert junk_rsq < 0.7 < clean_rsq, f"noise {junk_rsq:.3f}, wave {clean_rsq:.3f}"


def test_a_motionless_flagellum_has_no_beat() -> None:
    dead = spectral_series(*travelling_wave(400, amplitude=0.0), FPS,
                           DEFAULT_WINDOW, DEFAULT_HOP)
    assert bool(dead["wave_freq_hz"].isna().all())
    assert bool(dead[["wave_amp_mean", "wave_phase_slope", "wave_direction"]].isna().all().all())


# --------------------------------------------------------------------------- #
# shape and edges
# --------------------------------------------------------------------------- #
def test_series_layout() -> None:
    px, py = travelling_wave(300)
    series = spectral_series(px, py, FPS, 75, 5)
    assert series.index.name == "frame" and series.index.is_monotonic_increasing
    assert bool((np.diff(series.index.to_numpy(int)) == 5).all())
    # Every window lies wholly inside the record.
    assert int(series.index.min()) - 37 >= 0 and int(series.index.max()) - 37 + 75 <= 300
    assert list(series.columns) == list(SERIES_COLUMNS)
    assert bool((series.dtypes == float).all())


def test_a_track_shorter_than_the_window_yields_one_window() -> None:
    short = spectral_series(*travelling_wave(90), FPS, 150, 5)
    assert len(short) == 1
    assert float(short["wave_window_frames"].iloc[0]) == 90.0
    assert float(short["wave_resolvable_hz"].iloc[0]) > 3.0


@pytest.mark.parametrize("extra", range(12))
def test_a_track_just_longer_than_the_window_yields_a_window(extra: int) -> None:
    """The slice grid is in multiples of the hop, so some lengths are awkward."""
    got = spectral_series(*travelling_wave(75 + extra), FPS, 75, 5)
    assert len(got) >= 1 and bool(np.isfinite(got["wave_direction"]).all())


def test_degenerate_input_yields_nothing() -> None:
    px, py = travelling_wave(300)
    assert len(spectral_series(px[:5], py[:5], FPS)) == 0, "shorter than the FFT floor"
    assert len(spectral_series(px[:, :3], py[:, :3], FPS)) == 0, "too few keypoints"
    assert len(spectral_series(px, py[:, :6], FPS)) == 0, "mismatched shapes"


def test_a_collapsed_detection_poisons_only_its_own_windows() -> None:
    px, py = travelling_wave(300)
    px[150] = 5.0
    py[150] = 5.0
    survived = spectral_series(px, py, FPS, 75, 5)
    assert np.isfinite(survived["wave_direction"]).sum() > 0.8 * len(survived)
    values = survived.to_numpy()
    assert bool(np.isfinite(values[~np.isnan(values)]).all()), "nothing became infinite"


def test_track_helper_lays_rows_on_the_frame_grid() -> None:
    """Shuffled rows with missing frames give the same answer as the dense array."""
    px, py = travelling_wave(300)
    f_start = 1000
    rows = {"frame": np.arange(300) + f_start}
    for k in range(N_KEYPOINTS):
        rows[f"kp_x_{k}"] = px[:, k]
        rows[f"kp_y_{k}"] = py[:, k]
    dropped = pd.DataFrame(rows).drop(index=np.arange(40, 55)).sample(frac=1.0, random_state=0)

    series = track_spectral_series(dropped, N_KEYPOINTS, FPS, 75, 5)
    assert int(series.index.min()) >= f_start and int(series.index.max()) < f_start + 300
    direct = spectral_series(px, py, FPS, 75, 5)
    assert bool((series["wave_direction"].dropna() == 1.0).all())
    assert abs(float(series["wave_freq_hz"].median())
               - float(direct["wave_freq_hz"].median())) < 1e-6


def test_one_transform_per_track_beats_one_per_window() -> None:
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
    # Measured near 16x; asserted well below so a loaded machine stays green.
    assert ratio > 4.0, f"{ratio:.1f}x"
