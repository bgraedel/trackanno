"""Tests for :mod:`trackanno.measure.wavespectra`: the MATLAB pipeline's beat analysis per window."""

from __future__ import annotations

import golden_beats as gb
import numpy as np
import pandas as pd
import pytest
from trackanno.measure import wavespectra as ws
from trackanno.measure.wavespectra import (
    DEFAULT_FLOW_FRAMES, DEFAULT_HOP, DEFAULT_WINDOW, SERIES_COLUMNS,
    spectral_series, track_spectral_series, window_values,
)

FPS = 100.0
N_KEYPOINTS = 8            # Head, Base, Flag1..Flag5, Tip
N_SEG = N_KEYPOINTS - 2
SUB = 40                   # integration steps per segment of the continuous curve
FLOOR_75 = 2.5 / 0.75      # lowest frequency a 75-frame window counts as resolved
BIN_75 = FPS / 76          # bin width of a 75-frame window, padded to 76 as MATLAB does


def chain(psi: np.ndarray, length_px: float = 60.0) -> tuple[np.ndarray, np.ndarray]:
    """Keypoints evenly spaced along the curve whose tangent angle is ``psi`` (frames x N_SEG*SUB)."""
    n_frames = psi.shape[0]
    ds = length_px / psi.shape[1]
    x = np.concatenate([np.zeros((n_frames, 1)), np.cumsum(ds * np.cos(psi), axis=1)], axis=1)
    y = np.concatenate([np.zeros((n_frames, 1)), np.cumsum(ds * np.sin(psi), axis=1)], axis=1)
    step = psi.shape[1] // N_SEG
    px = np.zeros((n_frames, N_KEYPOINTS))
    py = np.zeros((n_frames, N_KEYPOINTS))
    px[:, 1:], py[:, 1:] = x[:, ::step], y[:, ::step]
    px[:, 0], py[:, 0] = 20.0, 0.0                     # Head, out along +x from Base
    return px, py


def s_mid(length_px: float = 60.0) -> np.ndarray:
    """Arc positions (px) of the continuous curve's integration steps."""
    return (np.arange(N_SEG * SUB) + 0.5) * length_px / (N_SEG * SUB)


def travelling_wave(n_frames: int, freq_hz: float = 8.0, wavelength_ratio: float = 1.0,
                    amplitude: float = 0.6, length_px: float = 60.0, k_sign: int = +1,
                    noise_px: float = 0.0, seed: int = 0, flip_at=None):
    """Keypoints of ``psi = pi + A sin(2 pi f t - k s)``; ``k_sign=+1`` runs Base to Tip.

    ``flip_at`` reverses the wavevector at those frames with a continuous temporal phase.
    """
    rng = np.random.default_rng(seed)
    k = 2.0 * np.pi / (wavelength_ratio * length_px)
    sign = np.full(n_frames, float(k_sign))
    for i, at in enumerate(np.atleast_1d(flip_at) if flip_at is not None else ()):
        sign[int(at):] = float(k_sign) * (-1.0) ** (i + 1)
    phase = np.cumsum(np.full(n_frames, 2.0 * np.pi * freq_hz / FPS))
    psi = np.pi + amplitude * np.sin(phase[:, None] - sign[:, None] * k * s_mid(length_px)[None, :])
    px, py = chain(psi, length_px)
    if noise_px:
        px = px + rng.normal(0.0, noise_px, px.shape)
        py = py + rng.normal(0.0, noise_px, py.shape)
    return px, py


def wandering_wave(n_frames: int, mean_hz: float, swing_hz: float):
    """A wave whose frequency swings sinusoidally around ``mean_hz`` at 0.6 Hz."""
    t = np.arange(n_frames) / FPS
    phase = np.cumsum(2.0 * np.pi * (mean_hz + swing_hz * np.sin(2.0 * np.pi * 0.6 * t)) / FPS)
    psi = np.pi + 0.6 * np.sin(phase[:, None] - (2.0 * np.pi / 60.0) * s_mid()[None, :])
    return chain(psi)


# --------------------------------------------------------------------------- #
# the same numbers as MATLAB
# --------------------------------------------------------------------------- #
# processFile.m of github.com/mar5bar/flagella-analysis-pipeline (00ef0a0, with the smoothing
# tolerance scaled by spatialScale^2), run in MATLAB R2024b on tests/golden_beats.trace_lines at
# spatialScale 1. A spline through 61 flagellum nodes stands in for the traced pixels here.
MATLAB = {
    "tip_to_base_75": {"wave_freq_hz": 21.052632, "wave_freq_proximal_hz": 21.052632, "wave_freq_central_hz": 21.052632, "wave_freq_distal_hz": 21.052632, "wave_amp_mean": 0.361998, "wave_amp_asym": 0.030081, "wave_amp_para": -0.123352, "wave_amp_integral": 48.187681, "wave_phase_asym": 3.742589, "wave_phase_para": 0.217634, "wave_phase_rsq": 0.996001, "wave_phase_cycles": 1.109445, "wave_static_mean": 0.041401, "wave_freq_dominance": 0.408141, "wave_fit_error": 0.320049, "wave_xy_fit_error": 0.025202, "wave_disp_amp_max": 15.441762, "wave_angular_amp": 0.447135, "wave_length_px": 147.906493},  # noqa: E501
    "tip_to_base_300": {"wave_freq_hz": 21.333333, "wave_freq_proximal_hz": 21.333333, "wave_freq_central_hz": 21.333333, "wave_freq_distal_hz": 21.333333, "wave_amp_mean": 0.52279, "wave_amp_asym": 0.042401, "wave_amp_para": -0.179029, "wave_amp_integral": 69.555833, "wave_phase_asym": 3.741422, "wave_phase_para": 0.22846, "wave_phase_rsq": 0.995849, "wave_phase_cycles": 1.110484, "wave_static_mean": 0.041943, "wave_freq_dominance": 0.702354, "wave_fit_error": 0.074491, "wave_xy_fit_error": 0.005647, "wave_disp_amp_max": 15.44541, "wave_angular_amp": 0.447238, "wave_length_px": 147.830299},  # noqa: E501
    "base_to_tip_75": {"wave_freq_hz": 5.263158, "wave_freq_proximal_hz": 5.263158, "wave_freq_central_hz": 5.263158, "wave_freq_distal_hz": 5.263158, "wave_amp_mean": 0.513228, "wave_amp_asym": 0.241098, "wave_amp_para": -0.02023, "wave_amp_integral": 69.118004, "wave_phase_asym": -1.201268, "wave_phase_para": -0.078697, "wave_phase_rsq": 0.99585, "wave_phase_cycles": 0.359456, "wave_static_mean": 0.290143, "wave_freq_dominance": 0.339137, "wave_fit_error": 0.389756, "wave_xy_fit_error": 0.058187, "wave_disp_amp_max": 58.923169, "wave_angular_amp": 0.864198, "wave_length_px": 149.636825},  # noqa: E501
    "base_to_tip_300": {"wave_freq_hz": 5.666667, "wave_freq_proximal_hz": 5.666667, "wave_freq_central_hz": 5.666667, "wave_freq_distal_hz": 5.666667, "wave_amp_mean": 0.82577, "wave_amp_asym": 0.341308, "wave_amp_para": -0.051893, "wave_amp_integral": 110.814534, "wave_phase_asym": -1.201408, "wave_phase_para": -0.122653, "wave_phase_rsq": 0.992752, "wave_phase_cycles": 0.362853, "wave_static_mean": 0.309448, "wave_freq_dominance": 0.927654, "wave_fit_error": 0.052329, "wave_xy_fit_error": 0.008077, "wave_disp_amp_max": 58.81185, "wave_angular_amp": 0.864473, "wave_length_px": 149.106005},  # noqa: E501
    "small_75": {"wave_freq_hz": 11.842105, "wave_freq_proximal_hz": 11.842105, "wave_freq_central_hz": 11.842105, "wave_freq_distal_hz": 11.842105, "wave_amp_mean": 0.05684, "wave_amp_asym": 0.005815, "wave_amp_para": 0.004661, "wave_amp_integral": 7.667444, "wave_phase_asym": 1.096049, "wave_phase_para": -0.019567, "wave_phase_rsq": 0.999655, "wave_phase_cycles": 0.340604, "wave_static_mean": 0.001021, "wave_freq_dominance": 0.242066, "wave_fit_error": 0.670755, "wave_xy_fit_error": 0.007552, "wave_disp_amp_max": 8.304689, "wave_angular_amp": 0.100854, "wave_length_px": 149.883235},  # noqa: E501
    "small_300": {"wave_freq_hz": 12.333333, "wave_freq_proximal_hz": 12.333333, "wave_freq_central_hz": 12.333333, "wave_freq_distal_hz": 12.333333, "wave_amp_mean": 0.128722, "wave_amp_asym": 0.012074, "wave_amp_para": 0.008919, "wave_amp_integral": 17.363559, "wave_phase_asym": 1.083743, "wave_phase_para": -0.012467, "wave_phase_rsq": 0.999819, "wave_phase_cycles": 0.338181, "wave_static_mean": 0.000172, "wave_freq_dominance": 0.551747, "wave_fit_error": 0.155471, "wave_xy_fit_error": 0.001743, "wave_disp_amp_max": 8.305602, "wave_angular_amp": 0.10135, "wave_length_px": 149.879958},  # noqa: E501
}
FREQUENCIES = ("wave_freq_hz", "wave_freq_proximal_hz", "wave_freq_central_hz", "wave_freq_distal_hz")


@pytest.mark.parametrize("key", sorted(MATLAB))
def test_every_output_matches_matlab(key: str) -> None:
    """Frequencies exactly; the rest within 2.5% (the spline through nodes, not pixels).

    The amplitude profile's shape coefficients are compared on the scale of its mean.
    """
    case, frames = key.rsplit("_", 1)
    px, py = gb.keypoints(case, int(frames))
    got = spectral_series(px, py, FPS, window=int(frames), hop=int(frames)).iloc[0]
    expected = MATLAB[key]
    for column, want in expected.items():
        if column in FREQUENCIES:
            assert got[column] == pytest.approx(want, abs=1e-5), column
        elif column in ("wave_amp_asym", "wave_amp_para"):
            assert got[column] == pytest.approx(want, abs=0.01 * expected["wave_amp_mean"]), column
        else:
            assert got[column] == pytest.approx(want, rel=0.025, abs=1e-4), column


@pytest.mark.parametrize("case,direction", [("tip_to_base", -1.0), ("base_to_tip", 1.0)])
def test_the_phase_slope_gives_the_direction(case: str, direction: float) -> None:
    px, py = gb.keypoints(case, 300)
    got = spectral_series(px, py, FPS, window=300, hop=300).iloc[0]
    assert got["wave_direction"] == direction
    assert np.sign(got["wave_phase_asym"]) == -direction


# --------------------------------------------------------------------------- #
# the MATLAB helpers, one by one
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n,budget", [(128, 25.4), (200, 0.01)])
def test_smoothing_spends_the_tolerance_like_spaps(n: int, budget: float) -> None:
    """The trapezoid-weighted residual equals the tolerance, unless the weighted line fits within it."""
    rng = np.random.default_rng(0)
    rows = np.cumsum(rng.normal(size=(300, n)), axis=1) * rng.uniform(0.01, 3.0, (300, 1))
    smooth = ws.smooth_rows(rows, budget)
    w = np.ones(n)
    w[0] = w[-1] = 0.5
    resid = ((rows - smooth) ** 2 * w).sum(axis=1)
    x = np.arange(n, dtype=float)
    design = np.stack([np.ones(n), x], axis=1) * np.sqrt(w)[:, None]
    coef, *_ = np.linalg.lstsq(design, (rows * np.sqrt(w)).T, rcond=None)
    line = (((rows * np.sqrt(w)).T - design @ coef) ** 2).sum(axis=0)
    assert np.allclose(resid, np.minimum(line, budget), rtol=1e-4, atol=1e-8)


def test_the_good_frame_range_bridges_short_gaps_only() -> None:
    bad = np.zeros(60, bool)
    bad[[10, 11]] = True                     # bridged: good frames within 2 on both sides
    bad[30:36] = True                        # breaks the record
    frames, bridged, flagged = ws._good_frame_range(bad)
    assert frames[0] == 0 and frames[-1] == 29 and bridged == 2 and not flagged
    bad = np.zeros(40, bool)
    bad[1::3] = True                         # a third of the frames bridged
    assert ws._good_frame_range(bad)[2]


def test_movmean_follows_matlab_for_an_even_window() -> None:
    """movmean(v, 4) averages one more frame behind than ahead, shrinking at the ends."""
    v = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    got = ws._movmean_omitnan(v, 4)
    assert np.allclose(got, [1.5, 1.5, 7 / 3, 11 / 3, 5.0, 5.0])


def test_legendre_coefficients_are_legendrecoeffs() -> None:
    xs = np.linspace(0.0, 1.0, 200)
    got = ws._legendre(0.5 + 0.2 * (2 * xs - 1) - 0.1 * (6 * xs ** 2 - 6 * xs + 1))
    assert np.allclose(got, [0.5, 0.2, -0.1], atol=2e-4)


# --------------------------------------------------------------------------- #
# behaviour of the MATLAB analysis on keypoints
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("freq", [5.3, 8.0, 12.45, 21.7])
def test_frequency_is_the_peak_bin(freq: float) -> None:
    """MATLAB reads no finer than the bin: 100/76 Hz in a 75-frame window."""
    series = spectral_series(*travelling_wave(200, freq_hz=freq), FPS, 75, 25)
    got = series["wave_freq_hz"].dropna().to_numpy()
    assert len(got) and np.all(np.abs(got - freq) <= BIN_75 / 2 + 1e-9)
    assert np.allclose(got / BIN_75, np.round(got / BIN_75))


def test_a_mirror_image_measures_the_same() -> None:
    """y -> -y flips every angle; no output may depend on which side the cell is seen from."""
    px, py = travelling_wave(200, freq_hz=9.3, wavelength_ratio=0.9, noise_px=0.3, seed=4)
    here, there = spectral_series(px, py, FPS, 75, 25), spectral_series(px, -py, FPS, 75, 25)
    for column in SERIES_COLUMNS:
        a, b = here[column].to_numpy(), there[column].to_numpy()
        # A mirror adds pi to every phase; legendreCoeffs' trapz leaks a constant into phi2.
        atol = 1e-3 if column == "wave_phase_para" else 1e-9
        assert np.allclose(a, b, rtol=1e-6, atol=atol, equal_nan=True), column


def test_a_motionless_flagellum_reports_matlabs_first_bin() -> None:
    """With nothing to choose between, MATLAB's max takes the first searched bin, at zero amplitude."""
    dead = spectral_series(*travelling_wave(200, amplitude=0.0), FPS, 75, 25)
    assert np.allclose(dead["wave_freq_hz"], BIN_75)
    assert np.allclose(dead["wave_amp_mean"], 0.0, atol=1e-9)
    assert bool(dead["wave_direction"].isna().all())


def test_a_slow_beat_is_flagged_censored_with_matlabs_frequency() -> None:
    slow = spectral_series(*travelling_wave(300, freq_hz=2.0), FPS, 75, 25)
    assert bool((slow["wave_censored"] == 1.0).all())
    assert bool((slow["wave_freq_hz"] < FLOOR_75).all())
    assert np.isclose(float(slow["wave_resolvable_hz"].iloc[0]), FLOOR_75)


def test_noise_does_not_flip_the_direction() -> None:
    px, py = travelling_wave(300, noise_px=1.0, seed=3, k_sign=-1)
    series = spectral_series(px, py, FPS, DEFAULT_WINDOW, 10)
    assert float((series["wave_direction"] == -1).mean()) > 0.95


@pytest.mark.parametrize("window", [50, 75])
def test_reversal_is_located(window: int) -> None:
    flip = 250
    px, py = travelling_wave(500, flip_at=flip, noise_px=0.3, seed=9)
    series = spectral_series(px, py, FPS, window, 5)
    direction = series["wave_direction"].to_numpy(float)
    centres = series.index.to_numpy(int)
    finite = np.isfinite(direction)
    negative = np.flatnonzero(finite & (direction < 0))
    located = int(centres[negative[0]]) if negative.size else -1
    assert abs(located - flip) <= window // 4, f"found at {located}, true {flip}"
    assert bool((direction[finite & (centres < flip - window // 2)] > 0).all())
    assert bool((direction[finite & (centres > flip + window // 2)] < 0).all())


def test_the_quality_outputs_separate_a_wave_from_a_random_walk() -> None:
    """Dominance, excess and the xy fit error (Fochler et al. reject above 0.2) tell noise apart.

    Phase linearity does not: a spline through seven nodes makes any phase profile smooth.
    """
    rng = np.random.default_rng(1)
    noise_x = np.cumsum(rng.normal(0, 1.0, (200, N_KEYPOINTS)), axis=0) + np.arange(8) * 8.0
    noise_y = np.cumsum(rng.normal(0, 1.0, (200, N_KEYPOINTS)), axis=0)
    junk = spectral_series(noise_x, noise_y, FPS, DEFAULT_WINDOW, 25).median()
    px, py = travelling_wave(200, noise_px=0.2, seed=8)
    wave = spectral_series(px, py, FPS, DEFAULT_WINDOW, 25).median()
    assert junk["wave_freq_dominance"] < 0.25 < wave["wave_freq_dominance"]
    assert junk["wave_excess"] < 3.0 < wave["wave_excess"]
    assert wave["wave_xy_fit_error"] < 0.1 < junk["wave_xy_fit_error"]


# --------------------------------------------------------------------------- #
# gaps, as the pipeline treats untraced frames
# --------------------------------------------------------------------------- #
def test_single_missing_frames_are_bridged() -> None:
    px, py = travelling_wave(200, noise_px=0.2, seed=8)
    for frame in (60, 66, 80, 95, 101):
        px[frame], py[frame] = np.nan, np.nan
    series = spectral_series(px, py, FPS, 75, 25)
    full = spectral_series(*travelling_wave(200, noise_px=0.2, seed=8), FPS, 75, 25)
    assert bool((series["wave_interp_frac"] > 0).any())
    assert bool((series["wave_bad_window"] == 0).all())
    assert np.allclose(series["wave_freq_hz"], full["wave_freq_hz"])


def test_a_long_gap_marks_the_window_bad() -> None:
    """A gap longer than the bridge splits the window; the pipeline flags it under 30 good frames."""
    px, py = travelling_wave(300, noise_px=0.2, seed=8)
    px[130:170], py[130:170] = np.nan, np.nan
    series = spectral_series(px, py, FPS, 75, 5)
    centres = series.index.to_numpy(int)
    straddling = series[(centres > 130 + 15) & (centres < 170 - 15)]
    assert len(straddling) and bool((straddling["wave_bad_window"] == 1.0).all())
    clear = series[(centres + 38 < 130) | (centres - 38 > 170)]
    assert len(clear) and bool((clear["wave_bad_window"] == 0.0).all())


# --------------------------------------------------------------------------- #
# flow direction, read without a spectrum
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k_sign", [+1, -1])
def test_flow_direction_needs_no_resolvable_beat(k_sign: int) -> None:
    """A 2.5 Hz beat is below the window's floor; the flow columns still read it."""
    px, py = travelling_wave(300, freq_hz=2.5, k_sign=k_sign)
    series = spectral_series(px, py, FPS, DEFAULT_WINDOW, 10)
    flow = series["wave_flow_direction"].dropna()
    assert float(series["wave_censored"].mean()) > 0.5
    assert len(flow) > 0 and bool((flow == k_sign).all())
    assert float(series["wave_flow_coherence"].median()) > 0.8


def test_flow_locates_a_reversal_within_a_hop() -> None:
    flip = 250
    px, py = travelling_wave(500, flip_at=flip, noise_px=0.3, seed=9)
    series = spectral_series(px, py, FPS, 75, DEFAULT_HOP)
    centres = series.index.to_numpy(int)
    value = series["wave_flow_direction"].to_numpy(float)
    finite = np.isfinite(value)
    negative = np.flatnonzero(finite & (value < 0))
    located = int(centres[negative[0]]) if negative.size else -1
    assert abs(located - flip) <= DEFAULT_HOP, f"found at {located}, true {flip}"
    wrong = finite & (value != np.where(centres < flip, 1.0, -1.0))
    assert int(wrong.sum()) * DEFAULT_HOP <= 2 * DEFAULT_FLOW_FRAMES


def test_a_standing_wave_has_no_flow_coherence() -> None:
    """``psi = A sin(wt) cos(ks)`` does not travel, so its coherence is low."""
    phase = 2.0 * np.pi * 8.0 * np.arange(200) / FPS
    psi = np.pi + 0.6 * np.sin(phase)[:, None] * np.cos((2.0 * np.pi / 60.0) * s_mid())[None, :]
    standing = float(spectral_series(*chain(psi), FPS, 75, 25)["wave_flow_coherence"].median())
    travelling = float(spectral_series(*travelling_wave(200), FPS, 75, 25)["wave_flow_coherence"].median())
    assert standing < 0.2 and standing < 0.5 * travelling


# --------------------------------------------------------------------------- #
# stability and readout
# --------------------------------------------------------------------------- #
def test_a_wandering_beat_has_a_larger_spread_than_a_steady_one() -> None:
    steady = spectral_series(*travelling_wave(400, freq_hz=8.0), FPS, DEFAULT_WINDOW, 10)
    wander = spectral_series(*wandering_wave(400, 9.0, 3.0), FPS, DEFAULT_WINDOW, 10)
    assert float(steady["wave_freq_track_sd_hz"].iloc[0]) < 0.2
    assert float(wander["wave_freq_track_sd_hz"].iloc[0]) > 1.0


def test_stability_is_undefined_when_every_window_is_censored() -> None:
    series = spectral_series(*wandering_wave(300, 2.0, 1.0), FPS, 75, 10)
    assert bool((series["wave_censored"] == 1.0).all())
    assert bool(series["wave_freq_track_sd_hz"].isna().all())
    assert bool(series["wave_freq_drift_hz"].isna().all())


@pytest.fixture(scope="module")
def noisy_series() -> pd.DataFrame:
    px, py = travelling_wave(300, noise_px=0.3, seed=4)
    return spectral_series(px, py, FPS, DEFAULT_WINDOW, DEFAULT_HOP)


def test_window_values_reads_the_windows_centred_in_the_range(noisy_series) -> None:
    centres = noisy_series.index.to_numpy(int)
    values = window_values(noisy_series, 150, 175)
    inside = centres[(centres >= 150) & (centres < 175)]
    assert values["n_windows"] == len(inside)
    assert (window_values(noisy_series, 125, 150)["n_windows"] + values["n_windows"]
            == len(centres[(centres >= 125) & (centres < 175)]))
    assert values["wave_direction"] == 1.0
    assert np.isclose(values["wave_amp_mean"], float(noisy_series.loc[inside, "wave_amp_mean"].median()))
    assert all(c in values for c in SERIES_COLUMNS)


def test_window_values_falls_back_to_the_nearest_window(noisy_series) -> None:
    early = window_values(noisy_series, 0, 5)
    assert early["n_windows"] == 1.0 and abs(early["centre_offset"]) > 5.0


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
# shape and edges
# --------------------------------------------------------------------------- #
def test_series_layout() -> None:
    series = spectral_series(*travelling_wave(150), FPS, 75, 5)
    assert series.index.name == "frame" and series.index.is_monotonic_increasing
    assert bool((np.diff(series.index.to_numpy(int)) == 5).all())
    assert int(series.index.min()) - 37 >= 0 and int(series.index.max()) - 37 + 75 <= 150
    assert list(series.columns) == list(SERIES_COLUMNS)
    assert bool((series.dtypes == float).all())


def test_a_track_shorter_than_the_window_yields_one_window() -> None:
    short = spectral_series(*travelling_wave(90), FPS, 150, 5)
    assert len(short) == 1 and float(short["wave_window_frames"].iloc[0]) == 90.0
    assert np.isclose(float(short["wave_resolvable_hz"].iloc[0]), 2.5 / 0.9)


def test_degenerate_input_yields_nothing() -> None:
    px, py = travelling_wave(100)
    assert len(spectral_series(px[:5], py[:5], FPS)) == 0
    assert len(spectral_series(px[:, :3], py[:, :3], FPS)) == 0
    assert len(spectral_series(px, py[:, :6], FPS)) == 0


def test_track_helper_lays_rows_on_the_frame_grid() -> None:
    """Shuffled rows with missing frames give the same answer as the dense array."""
    px, py = travelling_wave(200)
    f_start = 1000
    rows = {"frame": np.arange(200) + f_start}
    for k in range(N_KEYPOINTS):
        rows[f"kp_x_{k}"], rows[f"kp_y_{k}"] = px[:, k], py[:, k]
    shuffled = pd.DataFrame(rows).sample(frac=1.0, random_state=0)
    series = track_spectral_series(shuffled, N_KEYPOINTS, FPS, 75, 25)
    direct = spectral_series(px, py, FPS, 75, 25)
    assert list(series.index) == [int(c) + f_start for c in direct.index]
    assert np.allclose(series.to_numpy(), direct.to_numpy(), equal_nan=True)


def test_segment_arcs_are_exact_for_a_circle() -> None:
    radius, turn = 10.0, 0.9
    angles = np.arange(6) * turn
    chords = np.full(6, 2.0 * radius * np.sin(turn / 2.0))
    assert np.allclose(ws.segment_arcs(chords, angles), radius * turn)


# --------------------------------------------------------------------------- #
# found by the independent verification
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("angle", [0.3, 0.7, 1.2])
def test_the_fit_rotation_is_not_bounded(angle: float) -> None:
    s = np.linspace(0.0, 60.0, 200)
    x = np.tile(s, (5, 1))
    y = np.zeros_like(x)
    c, sn = np.cos(angle), np.sin(angle)
    xr, yr = c * x - sn * y, sn * x + c * y          # the reconstruction, turned by ``angle``
    assert ws._xy_fit_error(x, y, xr, yr, s, np.arange(5) / FPS) == pytest.approx(0.0, abs=1e-4)


def test_phase_linearity_is_one_only_for_an_exactly_flat_phase() -> None:
    s = np.linspace(0.0, 1.0, 200)
    assert ws._phase_line(np.full(200, 0.5), s)[1] == 1.0
    noisy = 0.5 + 1e-6 * np.random.default_rng(0).normal(size=200)
    assert ws._phase_line(noisy, s)[1] < 0.1


def test_a_flagellum_smoothed_straight_has_no_direction() -> None:
    px, py = travelling_wave(150, freq_hz=12.0, amplitude=0.004)
    s = spectral_series(px, py, FPS, 75, 75)
    assert s["wave_direction"].isna().all() and s["wave_wavelength_px"].isna().all()


def test_a_centerline_judges_frames_on_the_flagellum_alone() -> None:
    """whole_trace=False drops the pipeline's whole-trace test, as processFileCenterline.m does:
    a Head that jumps toward the Base no longer rejects frames whose flagellum is whole."""
    px, py = travelling_wave(300)
    short = px.copy(), py.copy()
    for a in short:  # every third frame, the Head lands most of the way to the Base
        a[::3, 0] = a[::3, 1] + 0.1 * (a[::3, 0] - a[::3, 1])
    whole = spectral_series(*short, FPS, 300, 300).iloc[0]
    alone = spectral_series(*short, FPS, 300, 300, whole_trace=False).iloc[0]
    intact = spectral_series(px, py, FPS, 300, 300).iloc[0]
    assert whole["wave_interp_frac"] > 0.2, "with a body in the trace, those frames are bad"
    assert alone["wave_interp_frac"] == intact["wave_interp_frac"] == 0.0
    assert alone["wave_window_frames"] == intact["wave_window_frames"] == 300
    # The angles still move: the axis runs from the body centre to the SMOOTHED base, and a
    # centre that close to the base turns that small offset into a large angle.
