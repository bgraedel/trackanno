"""Per-frame beat readouts: the wavelet ridge of the tangent angle."""

from __future__ import annotations

import golden_beats as gb
import numpy as np
import pandas as pd
import pytest
from test_wavespectra import FPS, travelling_wave
from trackanno.measure import instantaneous as inst
from trackanno.measure import wavespectra as ws

CORE = slice(30, -30)          # away from the ends of the track


@pytest.mark.parametrize("case", list(gb.CASES))
def test_a_steady_beat_reads_its_frequency_every_frame(case: str) -> None:
    px, py = gb.keypoints(case, 300)
    freq = inst.beat_series(px, py, gb.FPS)["inst_freq_hz"].to_numpy()[CORE]
    assert np.allclose(freq, gb.CASES[case]["freq"], rtol=0.03)


@pytest.mark.parametrize("case", list(gb.CASES))
def test_amplitude_is_the_cosine_amplitude(case: str) -> None:
    px, py = gb.keypoints(case, 300)
    got = inst.beat_series(px, py, gb.FPS)["inst_amp_mean"].iloc[CORE].median()
    u = np.linspace(0.0, ws.LENGTH_SHARE, 200)
    assert got == pytest.approx(gb.CASES[case]["amp"](u).mean(), rel=0.03)


@pytest.mark.parametrize("case", list(gb.CASES))
def test_phase_profile_is_the_true_one(case: str) -> None:
    """The pipeline's smoothing flattens a small beat's phase; the per-frame readout does not."""
    px, py = gb.keypoints(case, 300)
    frames = inst.beat_series(px, py, gb.FPS).iloc[CORE]
    spec = gb.CASES[case]
    slope = -spec["direction"] * 2.0 * np.pi / spec["wavelength"]
    assert frames["inst_direction"].eq(spec["direction"]).all()
    assert frames["inst_phase_slope"].median() == pytest.approx(slope, rel=0.02)
    # A linear phase over 0.9 L has Legendre coefficient 1 equal to half its rise.
    rise = slope * ws.LENGTH_SHARE * gb.LENGTH
    assert frames["inst_phase_asym"].median() == pytest.approx(rise / 2.0, rel=0.03)


def test_a_switch_is_followed_within_a_few_frames() -> None:
    fast_x, fast_y = gb.keypoints("tip_to_base", 300)
    slow_x, slow_y = gb.keypoints("base_to_tip", 300)
    px, py = np.vstack([fast_x, slow_x]), np.vstack([fast_y, slow_y])
    s = inst.beat_series(px, py, gb.FPS)
    freq, direction = s["inst_freq_hz"].to_numpy(), s["inst_direction"].to_numpy()
    assert np.allclose(freq[50:285], 21.3, rtol=0.05)
    assert np.allclose(freq[330:550], 5.7, rtol=0.05)
    assert (direction[50:285] == -1).all() and (direction[330:550] == 1).all()


def test_a_switch_is_wrong_for_fewer_frames_than_the_window() -> None:
    fast_x, fast_y = gb.keypoints("tip_to_base", 300)
    slow_x, slow_y = gb.keypoints("base_to_tip", 300)
    px, py = np.vstack([fast_x, slow_x]), np.vstack([fast_y, slow_y])
    truth = np.where(np.arange(600) < 300, 21.3, 5.7)
    ridge = inst.beat_series(px, py, gb.FPS)["inst_freq_hz"].to_numpy()
    windows = ws.spectral_series(px, py, gb.FPS, window=75, hop=1)
    windowed = np.full(600, np.nan)
    windowed[windows.index.to_numpy()] = windows["wave_freq_hz"].to_numpy()
    near = slice(200, 400)
    wrong_ridge = np.sum(np.abs(ridge[near] / truth[near] - 1) >= 0.1)
    wrong_window = np.sum(np.abs(windowed[near] / truth[near] - 1) >= 0.1)
    assert wrong_ridge <= 3 and wrong_ridge < wrong_window


def test_a_still_flagellum_is_never_confident() -> None:
    px, py = gb.keypoints("small", 300)
    px, py = np.repeat(px[:1], 300, axis=0), np.repeat(py[:1], 300, axis=0)
    rng = np.random.default_rng(0)
    px, py = px + rng.normal(0, 0.3, px.shape), py + rng.normal(0, 0.3, py.shape)
    s = inst.beat_series(px, py, gb.FPS)
    assert not (s["inst_excess"] >= inst.CONFIDENT_EXCESS).any()
    clip = inst.window_values(s, 50, 100)
    assert clip["inst_confident_frac"] == 0.0 and np.isfinite(clip["inst_curvature_mean"])


def test_a_mirror_image_measures_the_same() -> None:
    px, py = travelling_wave(300, freq_hz=9.3, wavelength_ratio=0.9, noise_px=0.3, seed=4)
    a, b = inst.beat_series(px, py, FPS), inst.beat_series(px, -py, FPS)
    for col in ("inst_freq_hz", "inst_excess", "inst_amp_mean", "inst_phase_slope",
                "inst_direction", "inst_angle_mean", "inst_curvature_mean"):
        assert np.allclose(a[col], b[col], rtol=1e-6, atol=1e-9, equal_nan=True), col


@pytest.mark.parametrize("k_sign", [+1, -1])
def test_the_direction_follows_the_wave(k_sign: int) -> None:
    px, py = travelling_wave(300, freq_hz=12.0, wavelength_ratio=1.0, k_sign=k_sign, noise_px=0.3)
    s = inst.beat_series(px, py, FPS).iloc[CORE]
    assert (s["inst_direction"] == k_sign).mean() > 0.95


def test_short_gaps_are_bridged_and_long_ones_split_the_track() -> None:
    px, py = travelling_wave(400, freq_hz=12.0)
    px, py = px.copy(), py.copy()
    px[100:103] = np.nan
    px[250:270] = np.nan
    s = inst.beat_series(px, py, FPS)
    freq = s["inst_freq_hz"].to_numpy()
    assert np.isnan(freq[100:103]).all() and np.isnan(freq[250:270]).all()
    assert np.allclose(freq[40:95], 12.0, rtol=0.03) and np.allclose(freq[108:240], 12.0, rtol=0.03)
    assert np.allclose(freq[290:370], 12.0, rtol=0.03)


def test_window_values_take_every_frame_inside_the_cone() -> None:
    """Frames outside the cone of influence (no excess) are left out; confidence is only counted."""
    series = pd.DataFrame({c: np.nan for c in inst.SERIES_COLUMNS}, index=pd.RangeIndex(6, name="frame"))
    series["inst_excess"] = [100.0, 100.0, 1.0, 1.0, np.nan, np.nan]
    series["inst_freq_hz"] = [10.0, 12.0, 4.0, 4.0, 40.0, 40.0]
    series["inst_angle_mean"] = np.linspace(0.0, 0.5, 6)
    out = inst.window_values(series, 0, 6)
    assert out["inst_freq_hz"] == 7.0
    assert out["inst_confident_frac"] == pytest.approx(0.5)
    assert out["inst_freq_sd_hz"] == pytest.approx(np.std([10, 12, 4, 4]))
    assert out["inst_angle_mean"] == pytest.approx(0.25)
    assert set(out) == set(inst.CLIP_COLUMNS)


def test_a_slow_beat_below_the_confidence_cut_keeps_its_frequency() -> None:
    """Slow beats read a lower excess than fast ones of the same size."""
    px, py = travelling_wave(400, freq_hz=3.5, amplitude=0.2, noise_px=0.8, seed=2)
    s = inst.beat_series(px, py, FPS)
    clip = inst.window_values(s, 150, 250)
    assert clip["inst_confident_frac"] == 0.0
    assert clip["inst_freq_hz"] == pytest.approx(3.5, rel=0.02)


def test_empty_and_short_input() -> None:
    assert all(np.isnan(v) for v in inst.window_values(None, 0, 10).values())
    s = inst.beat_series(np.empty((0, 8)), np.empty((0, 8)), FPS)
    assert list(s.columns) == list(inst.SERIES_COLUMNS) and len(s) == 0
    px, py = travelling_wave(6)
    assert inst.beat_series(px, py, FPS)["inst_freq_hz"].isna().all()


def test_track_series_is_on_the_movie_frames() -> None:
    px, py = travelling_wave(200, freq_hz=12.0)
    rows = pd.DataFrame({"frame": np.arange(200) + 500})
    for k in range(8):
        rows[f"kp_x_{k}"], rows[f"kp_y_{k}"] = px[:, k], py[:, k]
    s = inst.track_beat_series(rows, 8, FPS)
    assert s.index[0] == 500 and s.index[-1] == 699
    assert np.allclose(s["inst_freq_hz"].iloc[30:-30], 12.0, rtol=0.03)


def test_frames_the_wavelet_reaches_past_the_track_are_never_confident() -> None:
    px, py = travelling_wave(300, freq_hz=12.0)
    excess = inst.beat_series(px, py, FPS)["inst_excess"].to_numpy()
    reach = int(np.ceil(np.sqrt(2.0) * inst.OMEGA0 / (2.0 * np.pi * 12.0) * FPS))
    assert np.isnan(excess[: reach - 1]).all() and np.isnan(excess[-reach + 1:]).all()
    assert (excess[reach + 1:-reach - 1] >= inst.CONFIDENT_EXCESS).all()


def test_a_flagellum_that_never_moves_reads_nothing() -> None:
    px, py = gb.keypoints("small", 1)
    px, py = np.repeat(px, 200, axis=0), np.repeat(py, 200, axis=0)
    s = inst.beat_series(px, py, gb.FPS)
    assert s["inst_freq_hz"].isna().all() and s["inst_direction"].isna().all()
    assert s["inst_curvature_mean"].notna().all()


def test_turning_the_cell_in_the_image_changes_nothing() -> None:
    px, py = travelling_wave(300, freq_hz=9.3, wavelength_ratio=0.9, noise_px=0.3, seed=4)
    turn = np.deg2rad(37.0)
    qx = np.cos(turn) * px - np.sin(turn) * py
    qy = np.sin(turn) * px + np.cos(turn) * py
    a, b = inst.beat_series(px, py, FPS), inst.beat_series(qx, qy, FPS)
    for col in inst.SERIES_COLUMNS:
        assert np.allclose(a[col], b[col], rtol=1e-6, atol=1e-9, equal_nan=True), col


def test_magnification_changes_nothing_but_lengths() -> None:
    px, py = travelling_wave(300, freq_hz=9.3, wavelength_ratio=0.9, noise_px=0.3, seed=4)
    a, b = inst.beat_series(px, py, FPS), inst.beat_series(3.0 * px, 3.0 * py, FPS)
    for col in ("inst_freq_hz", "inst_excess", "inst_amp_mean", "inst_amp_asym", "inst_amp_para",
                "inst_phase_asym", "inst_phase_para", "inst_direction", "inst_angle_mean"):
        assert np.allclose(a[col], b[col], rtol=1e-6, atol=1e-9, equal_nan=True), col
    assert np.allclose(a["inst_phase_slope"], 3.0 * b["inst_phase_slope"], equal_nan=True)
    assert np.allclose(3.0 * a["inst_wavelength_px"], b["inst_wavelength_px"], equal_nan=True)
