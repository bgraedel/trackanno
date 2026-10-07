"""Tests for the per-frame series and its window reduction.

Closed-form tracks give known feature values; a window's features must not depend
on the series built around it. Real projects can be added with
``pytest --project-root PATH``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fixtures import make_project
from trackanno.config import load_config
from trackanno.measure.trackseries import (
    SERIES_COLUMNS, SERIES_SPEC, WINDOW_FUNCTIONALS, aggregate_window, build_series,
    feature_names,
)

TOL = 1e-9
FPS, DT = 100.0, 0.01

# Spectral estimators are out of scope for the series.
SPECTRAL = ("beat_", "wave_")


@pytest.fixture(scope="module")
def cfg(root):
    """A generated pose project."""
    return load_config(make_project(root / "pose", pose=True))


@pytest.fixture(scope="module")
def detect_cfg(root):
    """The same with no keypoints, so the pose columns must come out NaN."""
    return load_config(make_project(root / "detect", pose=False))


@pytest.fixture(scope="module")
def roots(cfg, request):
    """The fixture project, plus any real ones named with ``--project-root``."""
    given = [Path(a).resolve() for a in request.config.getoption("--project-root")]
    return [Path(cfg.paths.work_dir).parent] + given


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def synthetic_track(n_frames: int, n_kpts: int, seed: int = 0,
                    drop: tuple[int, ...] = (), f0: int = 0) -> pd.DataFrame:
    """One track that swims, turns and beats; ``drop`` names absolute frames to remove."""
    rng = np.random.default_rng(seed)
    f = f0 + np.arange(n_frames)
    t = np.arange(n_frames) / 100.0
    cx = 40.0 + 90.0 * t + 12.0 * np.sin(2 * np.pi * 0.7 * t) + rng.normal(0, 0.4, n_frames)
    cy = 60.0 + 25.0 * np.sin(2 * np.pi * 0.4 * t) + rng.normal(0, 0.4, n_frames)
    half = 11.0 + 2.0 * np.sin(2 * np.pi * 1.1 * t)
    data = {
        "frame": f.astype(np.int32),
        "track_id": np.full(n_frames, 3, dtype=np.int32),
        "x1": (cx - half).astype(np.float32), "y1": (cy - half * 0.6).astype(np.float32),
        "x2": (cx + half).astype(np.float32), "y2": (cy + half * 0.6).astype(np.float32),
        "conf": (0.72 + 0.2 * rng.random(n_frames)).astype(np.float32),
    }
    if n_kpts:
        # The body axis rotates, so the angle unwraps are exercised.
        heading = np.arctan2(np.gradient(cy), np.gradient(cx))
        s = np.linspace(0.0, 1.0, n_kpts - 1)
        phase = 2 * np.pi * 9.0 * t
        data["kp_x_0"] = (cx - 7.0 * np.cos(heading)).astype(np.float32)
        data["kp_y_0"] = (cy - 7.0 * np.sin(heading)).astype(np.float32)
        data["kp_c_0"] = np.full(n_frames, 0.91, dtype=np.float32)
        for k in range(n_kpts - 1):
            along, across = s[k] * 24.0, 4.5 * np.sin(2 * np.pi * 1.3 * s[k] - phase)
            data[f"kp_x_{k + 1}"] = (
                cx + along * np.cos(heading) - across * np.sin(heading)).astype(np.float32)
            data[f"kp_y_{k + 1}"] = (
                cy + along * np.sin(heading) + across * np.cos(heading)).astype(np.float32)
            data[f"kp_c_{k + 1}"] = np.full(n_frames, 0.88, dtype=np.float32)
    frame = pd.DataFrame(data)
    return frame[~frame["frame"].isin(drop)].reset_index(drop=True)


def _pack(cx: np.ndarray, cy: np.ndarray, half_w: float, half_h: float,
          n_kpts: int, f0: int, conf: float = 0.8, kpt_conf: float = 0.9,
          head_ahead: float = 20.0, seg: float = 6.0) -> pd.DataFrame:
    """Track rows for a centroid path with a rigid, collinear, axis-aligned pose.

    Head sits ``head_ahead`` px along +x from Base and the chain runs straight
    back in ``seg`` steps, so the minor axis is exactly zero.
    """
    n = len(cx)
    data = {
        "frame": (f0 + np.arange(n)).astype(np.int32),
        "track_id": np.full(n, 1, dtype=np.int32),
        "x1": (cx - half_w).astype(np.float32), "y1": (cy - half_h).astype(np.float32),
        "x2": (cx + half_w).astype(np.float32), "y2": (cy + half_h).astype(np.float32),
        "conf": np.full(n, conf, dtype=np.float32),
    }
    if n_kpts:
        data["kp_x_0"] = (cx + head_ahead).astype(np.float32)
        data["kp_y_0"] = cy.astype(np.float32)
        data["kp_c_0"] = np.full(n, kpt_conf, dtype=np.float32)
        for k in range(1, n_kpts):
            data[f"kp_x_{k}"] = (cx - seg * (k - 1)).astype(np.float32)
            data[f"kp_y_{k}"] = cy.astype(np.float32)
            data[f"kp_c_{k}"] = np.full(n, kpt_conf, dtype=np.float32)
    return pd.DataFrame(data)


def straight_track(n_frames: int, n_kpts: int, v: float = 10.0, half_w: float = 12.0,
                   half_h: float = 6.0, f0: int = 0) -> pd.DataFrame:
    """Constant velocity ``v`` px/frame along +x, fixed box, rigid pose."""
    i = np.arange(n_frames, dtype=float)
    return _pack(40.0 + v * i, np.full(n_frames, 60.0), half_w, half_h, n_kpts, f0)


def circular_track(n_frames: int, n_kpts: int, radius: float = 200.0, w: float = 0.05,
                   f0: int = 0) -> pd.DataFrame:
    """Constant angular rate ``w`` rad/frame; chord headings advance at ``w`` for any lag."""
    i = np.arange(n_frames, dtype=float)
    return _pack(400.0 + radius * np.cos(w * i), 400.0 + radius * np.sin(w * i),
                 12.0, 6.0, n_kpts, f0)


def spinning_tip_track(n_frames: int, n_kpts: int, w: float = 0.1, f0: int = 0,
                       seg: float = 6.0, head_ahead: float = 20.0,
                       base_angle: float = 0.0) -> pd.DataFrame:
    """The flagellum chain rotating about a fixed Base at ``w`` rad/frame."""
    ang = base_angle + w * np.arange(n_frames, dtype=float)
    cx, cy = 200.0, 200.0
    data = {
        "frame": (f0 + np.arange(n_frames)).astype(np.int32),
        "track_id": np.full(n_frames, 1, dtype=np.int32),
        "x1": np.full(n_frames, cx - 40.0, dtype=np.float32),
        "y1": np.full(n_frames, cy - 40.0, dtype=np.float32),
        "x2": np.full(n_frames, cx + 40.0, dtype=np.float32),
        "y2": np.full(n_frames, cy + 40.0, dtype=np.float32),
        "conf": np.full(n_frames, 0.8, dtype=np.float32),
        "kp_x_0": np.full(n_frames, cx + head_ahead, dtype=np.float32),
        "kp_y_0": np.full(n_frames, cy, dtype=np.float32),
        "kp_c_0": np.full(n_frames, 0.9, dtype=np.float32),
    }
    for k in range(1, n_kpts):
        r = seg * (k - 1)
        data[f"kp_x_{k}"] = (cx + r * np.cos(ang)).astype(np.float32)
        data[f"kp_y_{k}"] = (cy + r * np.sin(ang)).astype(np.float32)
        data[f"kp_c_{k}"] = np.full(n_frames, 0.9, dtype=np.float32)
    return pd.DataFrame(data)


def close(got, expected, tol: float = 1e-6) -> bool:
    return bool(np.isfinite(got)) and abs(float(got) - float(expected)) <= tol


def assert_close(got: dict, expected: dict, tol: float = 1e-6) -> None:
    bad = {k: (got[k], v) for k, v in expected.items() if not close(got[k], v, tol)}
    assert not bad, f"got vs expected: {bad}"


def diff_features(ref: dict, got: dict) -> dict[str, float]:
    """Per-column absolute difference; NaN on both sides agrees, on one side is inf."""
    out: dict[str, float] = {}
    for key, value in ref.items():
        if key.startswith(SPECTRAL):
            continue
        if key not in got:
            out[key] = np.inf
            continue
        mine = got[key]
        if np.isnan(value) and np.isnan(mine):
            out[key] = 0.0
        elif np.isnan(value) != np.isnan(mine):
            out[key] = np.inf
        else:
            out[key] = abs(float(value) - float(mine))
    return out


def accumulate(stats: dict, diffs: dict[str, float]) -> None:
    for key, value in diffs.items():
        entry = stats.setdefault(key, {"max": 0.0, "over": 0})
        entry["max"] = max(entry["max"], value)
        if value > TOL:
            entry["over"] += 1


def report(stats: dict, n_clips: int, title: str) -> list[str]:
    """Print the per-column table; return the columns that failed."""
    print(f"\n  {title}: {n_clips} windows, {len(stats)} columns")
    print(f"  {'column':30s} {'max abs diff':>13s} {'n > 1e-9':>9s}")
    failed = []
    for key in sorted(stats):
        entry = stats[key]
        note = ""
        if entry["max"] > TOL:
            failed.append(key)
            note = "   <-- MISMATCH"
        print(f"  {key:30s} {entry['max']:13.3e} {entry['over']:9d}{note}")
    return failed


def clip_windows(work: Path):
    """Yield (movie_dir, manifest, tracks, fps) for every prepared movie."""
    for d in sorted(p for p in work.iterdir() if p.is_dir()):
        manifest, tracks = d / "clips.parquet", d / "tracks.parquet"
        cache = d / "cache.json"
        if not (manifest.exists() and tracks.exists() and cache.exists()):
            continue
        man = pd.read_parquet(manifest)
        if man.empty:
            continue
        fps = float(json.loads(cache.read_text())["movie"]["fps"])
        yield d, man, pd.read_parquet(tracks), fps


def track_series(tdf: pd.DataFrame, man: pd.DataFrame, track_id: int, cfg, fps: float):
    """One series spanning the track and every clip planned from it."""
    clips = man[man["track_id"] == track_id]
    whole = tdf.sort_values("frame")
    lo = min(int(whole["frame"].min()), int(clips["f_start"].min()))
    hi = max(int(whole["frame"].max()) + 1, int(clips["f_end"].max()))
    return build_series(whole, cfg, fps, lo, hi - lo)


# --------------------------------------------------------------------------- #
# series
# --------------------------------------------------------------------------- #
GAPS = (513, 514, 515, 540)


@pytest.fixture(scope="module")
def gapped_series(cfg):
    return build_series(synthetic_track(80, cfg.n_keypoints, drop=GAPS, f0=500),
                        cfg, FPS, 500, 80)


def test_series_layout(gapped_series) -> None:
    series = gapped_series
    assert series.index[0] == 500 and series.index[-1] == 579 and len(series) == 80
    assert tuple(series.columns) == SERIES_COLUMNS
    assert all(dt == np.float64 for dt in series.dtypes)


def test_the_series_marks_gaps(gapped_series) -> None:
    series = gapped_series
    present = series["present"].to_numpy()
    assert set(np.unique(present)) == {0.0, 1.0}
    assert present.sum() == 76
    assert not present[np.array(GAPS) - 500].any()
    # Undetected frames are NaN, never zero.
    assert bool(series.loc[513:515, ["cx", "body_len", "speed"]].isna().all().all())
    assert bool(series.loc[513, "flagellum_complete"] == 0.0), "0 where absent, not NaN"
    assert bool(np.isfinite(series.loc[500:512, ["body_len", "shape_pc1",
                                                 "body_axis_ang"]]).all().all())


def test_a_gap_does_not_become_a_step(gapped_series) -> None:
    speed = gapped_series["speed"]
    assert np.isnan(speed[516]) and np.isnan(speed[541]), "no predecessor after a gap"
    assert np.isnan(speed[500]), "the first frame has no incoming step"
    assert np.isfinite(speed[512]) and np.isfinite(speed[517])


def test_a_detect_project_has_nan_pose_columns(detect_cfg) -> None:
    series = build_series(synthetic_track(60, 0), detect_cfg, FPS, 0, 60)
    pose_cols = ["kpt_conf", "shape_extent", "shape_pc1", "shape_pc2", "shape_elong",
                 "body_len", "flagellum_len", "flagellum_complete", "anchor_x", "anchor_y",
                 "body_speed", "body_axis_ang", "tip_ang"]
    assert bool(series[pose_cols].isna().all().all())
    assert bool(np.isfinite(series["cx"]).all() and np.isfinite(series["area"]).all())

    got = aggregate_window(series, 0, 60, FPS, detect_cfg)
    assert np.isnan(got["flagellum_complete_frac"]), "NaN, not 0.0"
    assert got["present_frac"] == 1.0


def test_a_detect_project_gets_exact_box_features(detect_cfg) -> None:
    n, v, half_w, half_h = 60, 10.0, 12.0, 6.0
    plain = build_series(straight_track(n, 0, v, half_w, half_h), detect_cfg, FPS, 0, n)
    got = aggregate_window(plain, 0, n, FPS, detect_cfg)
    assert_close(got, {
        "speed_mean": v * 100.0, "speed_std": 0.0, "path_length": (n - 1) * v,
        "net_displacement": (n - 1) * v, "straightness": 1.0,
        "area_mean": (2 * half_w) * (2 * half_h), "aspect_mean": half_w / half_h,
    })


# --------------------------------------------------------------------------- #
# known values
# --------------------------------------------------------------------------- #
N, V, HALF_W, HALF_H, HEAD, SEG = 60, 10.0, 12.0, 6.0, 20.0, 6.0


@pytest.fixture(scope="module")
def straight(cfg) -> dict:
    series = build_series(straight_track(N, cfg.n_keypoints, V, HALF_W, HALF_H), cfg, FPS, 0, N)
    return aggregate_window(series, 0, N, FPS, cfg)


def test_a_straight_line_has_one_constant_speed(straight) -> None:
    speed = V / DT
    assert_close(straight, {
        "speed_mean": speed, "speed_median": speed, "speed_p90": speed,
        "speed_max": speed, "speed_std": 0.0, "immotile_frac": 0.0,
    })


def test_a_straight_line_has_closed_form_path_features(straight) -> None:
    # n frames give n-1 steps; evenly spaced points have Rg = v * sqrt((n^2 - 1) / 12).
    assert_close(straight, {
        "path_length": (N - 1) * V, "net_displacement": (N - 1) * V, "straightness": 1.0,
        "radius_gyration": V * np.sqrt((N * N - 1) / 12.0),
    })


def test_a_straight_line_does_not_turn(straight) -> None:
    assert_close(straight, {
        "turn_abs_mean": 0.0, "turn_std": 0.0, "turn_abs_p90": 0.0,
        "turn_lag_frames": 1.0, "turn_n_valid": float(N - 2),
    })


def test_a_fixed_box_has_exact_area_aspect_and_confidence(straight) -> None:
    assert_close(straight, {
        "area_mean": (2 * HALF_W) * (2 * HALF_H), "area_std": 0.0,
        "aspect_mean": HALF_W / HALF_H, "aspect_std": 0.0,
        "present_frac": 1.0, "conf_mean": 0.8, "conf_std": 0.0,
    })


def test_a_rigid_pose_has_exact_lengths_and_motion(cfg, straight) -> None:
    arc = SEG * (cfg.n_keypoints - 2)
    speed = V / DT
    assert_close(straight, {
        "body_len_mean": HEAD, "body_len_std": 0.0, "flagellum_len_mean": arc,
        "flagellum_len_std": 0.0, "flagellum_len_cv": 0.0, "flagellum_complete_frac": 1.0,
        # The Head-Base midpoint travels with the box.
        "body_speed_mean": speed, "body_path_length": (N - 1) * V,
        "body_net_displacement": (N - 1) * V, "body_straightness": 1.0,
        "body_displacement_per_len": (N - 1) * V / HEAD,
        "body_rotation_speed": 0.0, "tip_angle_std": 0.0, "tip_angular_speed": 0.0,
    })


def test_a_collinear_chain_has_no_minor_axis(cfg, straight) -> None:
    span = HEAD + SEG * (cfg.n_keypoints - 2)
    assert_close(straight, {"shape_extent_mean": span, "shape_pc1_mean": span,
                            "shape_pc2_mean": 0.0})
    assert np.isnan(straight["shape_elongation"])


RADIUS, W = 200.0, 0.05


@pytest.fixture(scope="module")
def circle(cfg):
    return build_series(circular_track(N, cfg.n_keypoints, RADIUS, W), cfg, FPS, 0, N)


def test_a_circle_has_a_constant_turn_rate(cfg, circle) -> None:
    got = aggregate_window(circle, 0, N, FPS, cfg)
    assert close(got["turn_abs_mean"], W / DT, 1e-3), got["turn_abs_mean"]
    assert close(got["turn_abs_p90"], W / DT, 1e-2)
    assert close(got["turn_std"], 0.0, 1e-2)


def test_a_circle_has_closed_form_speed_and_path(cfg, circle) -> None:
    got = aggregate_window(circle, 0, N, FPS, cfg)
    chord = 2.0 * RADIUS * np.sin(W / 2.0)
    assert close(got["speed_mean"], chord / DT, 1e-3)
    assert close(got["path_length"], (N - 1) * chord, 1e-4)
    assert close(got["net_displacement"], 2.0 * RADIUS * np.sin(W * (N - 1) / 2.0), 1e-4)


def test_the_turn_rate_does_not_move_with_the_window_length(cfg, circle) -> None:
    rates = [aggregate_window(circle, 0, length, FPS, cfg)["turn_abs_mean"]
             for length in (12, 25, 40, 60)]
    assert max(abs(r - W / DT) for r in rates) < 1e-2, rates


def test_closed_forms_hold_at_every_window_length_and_offset(cfg) -> None:
    total = 240
    series = build_series(straight_track(total, cfg.n_keypoints, V), cfg, FPS, 0, total)

    n_windows, worst, worst_note = 0, 0.0, ""
    for length in (12, 25, 50, 120, 240):
        for start in range(0, total - length + 1, max(length // 3, 1)):
            got = aggregate_window(series, start, start + length, FPS, cfg)
            expected = {
                "speed_mean": V / DT, "speed_std": 0.0,
                "path_length": (length - 1) * V, "net_displacement": (length - 1) * V,
                "straightness": 1.0, "area_mean": 288.0, "aspect_mean": 2.0,
                "present_frac": 1.0, "turn_abs_mean": 0.0,
                "radius_gyration": V * np.sqrt((length * length - 1) / 12.0),
            }
            for key, want in expected.items():
                deviation = abs(float(got[key]) - want)
                if deviation > worst:
                    worst, worst_note = deviation, f"{key} at [{start}, {start + length})"
            n_windows += 1

    assert worst <= 1e-6, f"worst {worst:.3e} on {worst_note}"
    assert n_windows >= 40


# --------------------------------------------------------------------------- #
# tip rotation across gaps
# --------------------------------------------------------------------------- #
TIP_N, TIP_W = 100, 0.1


@pytest.fixture(scope="module")
def spinning(cfg):
    return spinning_tip_track(TIP_N, cfg.n_keypoints, TIP_W)


def _tip(cfg, track) -> dict:
    return aggregate_window(build_series(track, cfg, FPS, 0, TIP_N), 0, TIP_N, FPS, cfg)


def test_a_constant_tip_rotation_comes_back(cfg, spinning) -> None:
    got = _tip(cfg, spinning)
    assert close(got["tip_angular_speed"], TIP_W / DT, 1e-4)
    # Evenly spaced angles, so the spread is w * sd(0..n-1).
    assert close(got["tip_angle_std"], TIP_W * np.std(np.arange(TIP_N)), 1e-4)


def test_a_long_gap_does_not_inflate_the_tip_rotation_rate(cfg, spinning) -> None:
    """Compacting before differencing folds a 4.1 rad gap into one wrapped step."""
    got = _tip(cfg, spinning[~spinning["frame"].isin(np.arange(40, 80))])
    assert close(got["tip_angular_speed"], TIP_W / DT, 1e-3), got["tip_angular_speed"]


def test_the_tip_angle_spread_is_over_the_frames_seen(cfg, spinning) -> None:
    """Only recoverable across a gap that rotates less than pi."""
    got = _tip(cfg, spinning[~spinning["frame"].isin(np.arange(40, 60))])
    kept = np.concatenate([np.arange(40), np.arange(60, TIP_N)])
    assert close(got["tip_angle_std"], TIP_W * np.std(kept), 1e-4)


def test_tip_angle_spread_survives_a_gap_across_the_branch_cut(cfg) -> None:
    """A tip near +-pi lost while crossing the cut keeps its spread."""
    w = 0.002
    sub = spinning_tip_track(TIP_N, cfg.n_keypoints, w, base_angle=np.pi - 0.1)
    got = _tip(cfg, sub[~sub["frame"].isin(np.arange(45, 55))])
    kept = np.concatenate([np.arange(45), np.arange(55, TIP_N)])
    assert close(got["tip_angle_std"], w * np.std(kept), 1e-4), got["tip_angle_std"]


# --------------------------------------------------------------------------- #
# window independence and guards
# --------------------------------------------------------------------------- #
def test_window_is_independent_of_the_series_around_it(cfg) -> None:
    sub = synthetic_track(300, cfg.n_keypoints, seed=2, drop=(90, 91, 150))
    whole = build_series(sub, cfg, FPS, 0, 300)
    worst, worst_key = 0.0, ""
    for start, length in ((0, 50), (25, 50), (75, 40), (140, 60), (250, 50), (12, 7)):
        window = sub[(sub["frame"] >= start) & (sub["frame"] < start + length)]
        alone = build_series(window, cfg, FPS, start, length)
        a = aggregate_window(whole, start, start + length, FPS, cfg)
        b = aggregate_window(alone, start, start + length, FPS, cfg)
        for key, value in diff_features(a, b).items():
            if value > worst:
                worst, worst_key = value, key
    assert worst <= TOL, f"{worst_key} differs by {worst:.3e}"


@pytest.mark.parametrize("start,end", [(100, 100), (120, 110), (90, 110), (130, 145)],
                         ids=["empty", "reversed", "before", "past"])
def test_a_window_the_series_cannot_answer_is_refused(cfg, start, end) -> None:
    series = build_series(synthetic_track(40, cfg.n_keypoints), cfg, FPS, 100, 40)
    with pytest.raises(ValueError):
        aggregate_window(series, start, end, FPS, cfg)


# The feature names are the contract with every fitted model, so they are written out.
EXPECTED_FEATURES = frozenset({
    "area_mean", "area_std", "aspect_mean", "aspect_std",
    "speed_mean", "speed_median", "speed_p90", "speed_std", "speed_max",
    "immotile_frac", "present_frac", "conf_mean", "conf_std", "kpt_conf_mean",
    "shape_extent_mean", "shape_extent_std", "shape_pc1_mean", "shape_pc2_mean",
    "shape_elongation",
    "body_len_mean", "body_len_std",
    "flagellum_len_mean", "flagellum_len_std", "flagellum_complete_frac",
    "flagellum_len_cv",
    "body_speed_mean", "body_speed_median", "body_speed_p90",
    "path_length", "net_displacement", "straightness", "radius_gyration",
    "turn_lag_frames", "turn_abs_mean", "turn_std", "turn_abs_p90", "turn_n_valid",
    "body_path_length", "body_net_displacement", "body_straightness",
    "body_displacement_per_len", "body_rotation_speed",
    "tip_angle_std", "tip_angular_speed",
})


def test_spec_covers_every_column(cfg) -> None:
    series = build_series(synthetic_track(60, cfg.n_keypoints), cfg, FPS, 0, 60)
    got = aggregate_window(series, 0, 60, FPS, cfg)
    assert set(got) == set(EXPECTED_FEATURES), set(got) ^ set(EXPECTED_FEATURES)
    assert not any(k.startswith(SPECTRAL) for k in got)
    assert set(feature_names()) == set(got) and len(feature_names()) == len(got)
    # Most features are declared in SERIES_SPEC rather than hand-written.
    assert len(feature_names()) - len(WINDOW_FUNCTIONALS) > len(WINDOW_FUNCTIONALS)
    assert set(SERIES_SPEC) == set(series.columns)


# --------------------------------------------------------------------------- #
# real data
# --------------------------------------------------------------------------- #
def regress_project(root: Path) -> tuple[int, dict] | None:
    """Whole-track versus clip-alone features for every clip of a prepared project."""
    config = root / "config.yaml"
    if not config.exists():
        return None
    cfg = load_config(config)
    work = cfg.paths.work_dir
    if not work.exists():
        return None
    stats: dict = {}
    n = 0
    for _, man, tracks, fps in clip_windows(work):
        by_track = {t: df for t, df in tracks.groupby("track_id", sort=False)}
        cache: dict[int, pd.DataFrame] = {}
        for spec in man.itertuples(index=False):
            tdf = by_track.get(int(spec.track_id))
            if tdf is None:
                continue
            f_start, f_end = int(spec.f_start), int(spec.f_end)
            sub = tdf[(tdf["frame"] >= f_start) & (tdf["frame"] < f_end)].sort_values("frame")
            if sub.empty:
                continue
            key = int(spec.track_id)
            if key not in cache:
                cache[key] = track_series(tdf, man, spec.track_id, cfg, fps)
            whole = aggregate_window(cache[key], f_start, f_end, fps, cfg)
            alone = aggregate_window(
                build_series(sub, cfg, fps, f_start, f_end - f_start), f_start, f_end, fps, cfg)
            accumulate(stats, diff_features(whole, alone))
            n += 1
    return (n, stats) if n else None


def test_real_projects(roots: list[Path]) -> None:
    total, pooled = 0, {}
    for root in roots:
        result = regress_project(root)
        if result is None:
            print(f"  --  {root}: no prepared work/, skipped")
            continue
        n, stats = result
        failed = report(stats, n, str(root))
        assert not failed, f"{root.name}: columns depend on the surrounding track: {failed}"
        total += n
        for key, entry in stats.items():
            merged = pooled.setdefault(key, {"max": 0.0, "over": 0})
            merged["max"] = max(merged["max"], entry["max"])
            merged["over"] += entry["over"]
    if total:
        failed = report(pooled, total, "POOLED over every project")
        assert not failed, f"pooled: {failed}"


# --------------------------------------------------------------------------- #
# cost
# --------------------------------------------------------------------------- #
def _time(fn) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


# With no overlap both do the same work, so only "not slower" is asserted there.
@pytest.mark.parametrize("length,stride,floor", [(50, 50, 0.8), (50, 25, 1.0), (50, 10, 1.0)])
def test_one_series_plus_slices_beats_a_series_per_window(cfg, length, stride, floor) -> None:
    n_frames = 500
    sub = synthetic_track(n_frames, cfg.n_keypoints, seed=3, drop=(101, 102, 300))
    whole = build_series(sub, cfg, FPS, 0, n_frames)
    windows = [(s, s + length) for s in range(0, n_frames - length + 1, stride)]

    def per_window() -> None:
        for start, end in windows:
            window = sub[(sub["frame"] >= start) & (sub["frame"] < end)]
            aggregate_window(build_series(window, cfg, FPS, start, end - start),
                             start, end, FPS, cfg)

    def shared() -> None:
        series = build_series(sub, cfg, FPS, 0, n_frames)
        for start, end in windows:
            aggregate_window(series, start, end, FPS, cfg)

    def slicing_only() -> None:
        for start, end in windows:
            aggregate_window(whole, start, end, FPS, cfg)

    def best(fn, repeats: int = 5) -> float:
        fn()
        return min(_time(fn) for _ in range(repeats))

    t_each, t_shared, t_slice = best(per_window), best(shared), best(slicing_only)
    assert t_each / t_shared > floor, f"{t_each / t_shared:.2f}x"
    # Slicing is the cheap half, which is why amortising construction pays.
    assert t_slice < t_shared, f"slicing {t_slice * 1e3:.2f} ms of {t_shared * 1e3:.2f} ms"
