"""Checks for the per-frame series and its window reduction.

The point of :mod:`trackanno.measure.trackseries` is that a window's features can be
obtained by slicing a per-frame series instead of recomputing a clip, so what is
worth testing is a mixture of KNOWN VALUES and an INVARIANCE:

* the features of a track whose motion is known in closed form - a constant
  speed, a straight path, a constant turn rate, a fixed box - come out equal to
  the analytic answer, at every window length and offset;
* a window's features do not depend on how much track was built around it, which
  is the property that makes the clip boundary stop being the measurement unit.
  checked on every clip of the generated fixture project, whose tracks carry
  gaps, short tracks and clips planned past the last detection;
* the gaps stay gaps - a series built over a whole track must not bridge a
  missing frame any more than ``_on_grid`` does.

The analytic cases are deliberately the primary check. Comparing one
implementation against another only proves they agree; comparing against a
closed form proves the number means what it says.

Nothing here reads data the repository does not carry: the fixture project is
generated into a temporary directory and deleted afterwards. Real projects may
be added as an extra, opt-in check with ``pytest --project-root PATH``; each
needs a ``config.yaml`` and a prepared ``work/``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fixtures import make_project
from harness import check
from trackanno.config import load_config
from trackanno.measure.trackseries import (
    SERIES_COLUMNS, SERIES_SPEC, WINDOW_FUNCTIONALS, aggregate_window, build_series,
    feature_names,
)

TOL = 1e-9


@pytest.fixture(scope="module")
def cfg(root):
    """A generated pose project. Nothing here reads the machine's own data."""
    return load_config(make_project(root / "pose", pose=True))


@pytest.fixture(scope="module")
def detect_cfg(root):
    """The same, with no keypoints, so the pose columns must come out NaN."""
    return load_config(make_project(root / "detect", pose=False))


@pytest.fixture(scope="module")
def roots(cfg, request):
    """The fixture project, plus any real ones named with ``--project-root``."""
    given = [Path(a).resolve() for a in request.config.getoption("--project-root")]
    return [Path(cfg.paths.work_dir).parent] + given

# Spectral estimators, out of scope for the series (see the trackseries module
# docstring). `aggregate_window` does not emit them at all; the prefixes are
# named here so the closing report can say so rather than leave it implied.
SPECTRAL = ("beat_", "wave_")

# Columns measured from `conf`, which is float32 in tracks.parquet. Both sides of
# every comparison below are float64 reductions of the same float32 input, so
# they agree exactly; the entry is kept empty rather than deleted because the
# reporting machinery below wants somewhere to put a per-column tolerance the
# day a genuinely float32 column appears.
FLOAT32_COLUMNS: dict[str, float] = {}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def synthetic_track(n_frames: int, n_kpts: int, seed: int = 0,
                    drop: tuple[int, ...] = (), f0: int = 0) -> pd.DataFrame:
    """One track that swims, turns and beats, with named frames missing.

    Deliberately not a straight line: straightness, turn_* and the shape PCA are
    all degenerate on one, so a bug in any of them would still compare equal.
    ``drop`` names ABSOLUTE frames, so a track that does not start at frame 0
    can be tested - which is the case the absolute-frame index exists for.
    """
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
        # Head trails the body axis, which itself rotates, so body_axis_ang and
        # tip_ang actually wrap during the track and the unwrap is exercised.
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
    """Track rows for a given centroid path, with everything else held constant.

    Every stored number is chosen to be exact in float32 (the dtype
    tracks.parquet uses) so the analytic expectations below are not competing
    with a rounding error they did not ask for.

    The pose is rigid and axis-aligned: Head sits ``head_ahead`` px along +x from
    Base and the flagellum runs straight back from Base in equal ``seg`` steps.
    So body_len is ``head_ahead``, flagellum_len is ``seg * (n_kpts - 2)``, and
    the whole chain is COLLINEAR - which is what makes the minor axis exactly
    zero and shape_elongation undefined.
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
    """Constant angular rate ``w`` rad/frame around a circle of ``radius``.

    The heading of a chord of a circle advances at exactly the angular rate,
    whatever the chord's length, so ``turn_abs_mean`` must come out as ``w * fps``
    rad/s no matter which lag `_heading_lag` happens to pick. That is worth a
    check of its own: the lag is chosen from the data, so a turn rate that moved
    with it would be an artefact of the chooser.
    """
    i = np.arange(n_frames, dtype=float)
    return _pack(400.0 + radius * np.cos(w * i), 400.0 + radius * np.sin(w * i),
                 12.0, 6.0, n_kpts, f0)


def spinning_tip_track(n_frames: int, n_kpts: int, w: float = 0.1, f0: int = 0,
                       seg: float = 6.0, head_ahead: float = 20.0) -> pd.DataFrame:
    """The flagellum chain rotating about a fixed Base at ``w`` rad/frame.

    Box, Head and Base are all stationary; only the chain turns. ``tip_ang`` is
    Base -> most distal node, so it is exactly ``w * i`` and wraps several times
    over a hundred frames, which is what makes the unwrap load-bearing.
    """
    i = np.arange(n_frames, dtype=float)
    ang = w * i
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


def diff_features(ref: dict, got: dict) -> dict[str, float]:
    """Per-column absolute difference, with NaN==NaN treated as agreement.

    A NaN on one side only is `inf`, not a skip: which features come out NaN is
    itself a measurement here (see `_heading_lag`), so a version that computed a
    number where the original abstained has changed the feature.
    """
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
        limit = FLOAT32_COLUMNS.get(key, TOL)
        note = ""
        if entry["max"] > limit:
            failed.append(key)
            note = "   <-- MISMATCH"
        elif key in FLOAT32_COLUMNS:
            note = "   float32 in tracks.parquet"
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
    """Build one series covering the track AND every clip planned from it.

    A clip may start before the track's first detection or end after its last -
    the manifest window is planned, the detections are found - so the series has
    to span the union. `aggregate_window` refuses a window it was not built for
    rather than padding, which is the whole reason this is spelled out.
    """
    clips = man[man["track_id"] == track_id]
    whole = tdf.sort_values("frame")
    lo = min(int(whole["frame"].min()), int(clips["f_start"].min()))
    hi = max(int(whole["frame"].max()) + 1, int(clips["f_end"].max()))
    return build_series(whole, cfg, fps, lo, hi - lo)


# --------------------------------------------------------------------------- #
# series
# --------------------------------------------------------------------------- #
def test_series_shape(cfg) -> None:
    n_kpts = cfg.n_keypoints
    gaps = (513, 514, 515, 540)
    sub = synthetic_track(80, n_kpts, drop=gaps, f0=500)
    series = build_series(sub, cfg, 100.0, 500, 80)

    check("series is indexed by absolute frame number",
          series.index[0] == 500 and series.index[-1] == 579 and len(series) == 80)
    check("series carries exactly the SERIES_SPEC columns",
          tuple(series.columns) == SERIES_COLUMNS)
    check("series is float throughout",
          all(dt == np.float64 for dt in series.dtypes))

    present = series["present"].to_numpy()
    check("present is 1/0 at every frame and marks the dropped ones",
          set(np.unique(present)) == {0.0, 1.0}
          and present.sum() == 76
          and not present[np.array(gaps) - 500].any(),
          str(present[10:18]))
    check("undetected frames are NaN, never zero",
          bool(series.loc[513:515, "cx"].isna().all()
               and series.loc[513:515, "body_len"].isna().all()
               and series.loc[513:515, "speed"].isna().all()))

    # A 3-frame gap must NOT become one big step: the frame after the gap has no
    # predecessor, so its speed is undefined. This is the inflation _on_grid
    # exists to prevent, restated for the series.
    check("a gap does not become a step",
          bool(np.isnan(series.loc[516, "speed"]) and np.isnan(series.loc[541, "speed"])))
    check("the first frame has no incoming step",
          bool(np.isnan(series.loc[500, "speed"])))
    check("speed is finite on either side of the gap",
          bool(np.isfinite(series.loc[512, "speed"]) and np.isfinite(series.loc[517, "speed"])))

    if n_kpts >= 3:
        check("pose columns are populated where the track is present",
              bool(np.isfinite(series.loc[500:512, "body_len"]).all()
                   and np.isfinite(series.loc[500:512, "shape_pc1"]).all()
                   and np.isfinite(series.loc[500:512, "body_axis_ang"]).all()))
        check("flagellum_complete is 0 where the chain is absent, not NaN",
              bool(series.loc[513, "flagellum_complete"] == 0.0))


def test_detect_project(detect_cfg) -> None:
    """A project with no keypoints must produce NaN, never a confident zero."""
    sub = synthetic_track(60, 0)
    series = build_series(sub, detect_cfg, 100.0, 0, 60)
    pose_cols = ["kpt_conf", "shape_extent", "shape_pc1", "shape_pc2", "shape_elong",
                 "body_len", "flagellum_len", "flagellum_complete", "anchor_x", "anchor_y",
                 "body_speed", "body_axis_ang", "tip_ang"]
    check("detect project: every pose column is all-NaN",
          bool(series[pose_cols].isna().all().all()))
    check("detect project: the box columns still work",
          bool(np.isfinite(series["cx"]).all() and np.isfinite(series["area"]).all()))

    got = aggregate_window(series, 0, 60, 100.0, detect_cfg)
    check("detect project: flagellum_complete_frac is NaN, not 0.0",
          bool(np.isnan(got["flagellum_complete_frac"])))
    check("detect project: present_frac is still measured",
          got["present_frac"] == 1.0)

    # The box features are all a detect project has, so they had better be right.
    # A straight track at 10 px/frame over 60 frames, box 24 x 12.
    n, v, half_w, half_h = 60, 10.0, 12.0, 6.0
    plain = build_series(straight_track(n, 0, v, half_w, half_h), detect_cfg, 100.0, 0, n)
    got = aggregate_window(plain, 0, n, 100.0, detect_cfg)
    check("detect project: box features equal their analytic values",
          close(got["speed_mean"], v * 100.0)
          and close(got["speed_std"], 0.0)
          and close(got["path_length"], (n - 1) * v)
          and close(got["net_displacement"], (n - 1) * v)
          and close(got["straightness"], 1.0)
          and close(got["area_mean"], (2 * half_w) * (2 * half_h))
          and close(got["aspect_mean"], half_w / half_h),
          f"speed {got['speed_mean']}, path {got['path_length']}, "
          f"area {got['area_mean']}, aspect {got['aspect_mean']}")


# --------------------------------------------------------------------------- #
# known values
# --------------------------------------------------------------------------- #
def test_analytic_values(cfg) -> None:
    """Closed-form motion in, closed-form features out.

    Every expectation below is derived from the definition of the feature and
    the geometry of the track, not from another implementation. That is the
    point: an equivalence check tells you two functions agree, and these tell
    you what the number means.
    """
    n_kpts = cfg.n_keypoints
    fps, dt = 100.0, 0.01

    # -- a straight line at constant speed ----------------------------------- #
    n, v, half_w, half_h, head, seg = 60, 10.0, 12.0, 6.0, 20.0, 6.0
    sub = straight_track(n, n_kpts, v, half_w, half_h)
    series = build_series(sub, cfg, fps, 0, n)
    got = aggregate_window(series, 0, n, fps, cfg)

    speed = v / dt                                   # px per second
    check("straight line: every speed column is the one constant speed",
          close(got["speed_mean"], speed) and close(got["speed_median"], speed)
          and close(got["speed_p90"], speed) and close(got["speed_max"], speed)
          and close(got["speed_std"], 0.0),
          f"mean {got['speed_mean']} vs {speed}, std {got['speed_std']}")
    check("straight line: immotile_frac is 0 well above the 50 px/s gate",
          close(got["immotile_frac"], 0.0), str(got["immotile_frac"]))
    # n frames -> n-1 steps, and the endpoints span the same distance.
    check("straight line: path equals net equals (n-1) * v, so straightness is 1",
          close(got["path_length"], (n - 1) * v)
          and close(got["net_displacement"], (n - 1) * v)
          and close(got["straightness"], 1.0),
          f"path {got['path_length']}, net {got['net_displacement']}")
    # Points evenly spaced on a line: the radius of gyration is v * sd(0..n-1),
    # i.e. v * sqrt((n^2 - 1) / 12).
    check("straight line: radius of gyration matches the closed form",
          close(got["radius_gyration"], v * np.sqrt((n * n - 1) / 12.0)),
          f"{got['radius_gyration']} vs {v * np.sqrt((n * n - 1) / 12.0)}")
    check("straight line: a constant heading gives exactly zero turn",
          close(got["turn_abs_mean"], 0.0) and close(got["turn_std"], 0.0)
          and close(got["turn_abs_p90"], 0.0)
          and close(got["turn_lag_frames"], 1.0)
          and close(got["turn_n_valid"], float(n - 2)),
          f"turn_abs_mean {got['turn_abs_mean']}, lag {got['turn_lag_frames']}")
    check("straight line: the fixed box gives an exact area and aspect",
          close(got["area_mean"], (2 * half_w) * (2 * half_h))
          and close(got["area_std"], 0.0)
          and close(got["aspect_mean"], (2 * half_w) / (2 * half_h))
          and close(got["aspect_std"], 0.0),
          f"area {got['area_mean']}, aspect {got['aspect_mean']}")
    check("straight line: present_frac is 1 on a track with no gaps",
          close(got["present_frac"], 1.0) and close(got["conf_mean"], 0.8, 1e-6)
          and close(got["conf_std"], 0.0),
          f"present {got['present_frac']}, conf {got['conf_mean']}")

    if n_kpts >= 3:
        arc = seg * (n_kpts - 2)
        check("straight line: the rigid pose gives exact body and flagellum lengths",
              close(got["body_len_mean"], head) and close(got["body_len_std"], 0.0)
              and close(got["flagellum_len_mean"], arc)
              and close(got["flagellum_len_std"], 0.0)
              and close(got["flagellum_len_cv"], 0.0)
              and close(got["flagellum_complete_frac"], 1.0),
              f"body {got['body_len_mean']} vs {head}, "
              f"flagellum {got['flagellum_len_mean']} vs {arc}")
        # The Head-Base midpoint travels with the box, so the body translocation
        # is the same distance, and net / body_len is (n-1)*v / head.
        check("straight line: body translocation matches the centroid's",
              close(got["body_speed_mean"], speed)
              and close(got["body_path_length"], (n - 1) * v)
              and close(got["body_net_displacement"], (n - 1) * v)
              and close(got["body_straightness"], 1.0)
              and close(got["body_displacement_per_len"], (n - 1) * v / head),
              f"body_speed {got['body_speed_mean']}, "
              f"per_len {got['body_displacement_per_len']}")
        check("straight line: a rigid pose does not rotate or wag its tip",
              close(got["body_rotation_speed"], 0.0)
              and close(got["tip_angle_std"], 0.0)
              and close(got["tip_angular_speed"], 0.0),
              f"rot {got['body_rotation_speed']}, tip_sd {got['tip_angle_std']}")
        # Head at +20 and the most distal node at -seg*(n_kpts-2) from the box
        # centre, all on one line: the extent is the full span and the minor
        # axis is exactly zero.
        span = head + seg * (n_kpts - 2)
        check("straight line: a collinear chain has a known extent and no minor axis",
              close(got["shape_extent_mean"], span) and close(got["shape_pc1_mean"], span)
              and close(got["shape_pc2_mean"], 0.0)
              and bool(np.isnan(got["shape_elongation"])),
              f"extent {got['shape_extent_mean']} vs {span}, "
              f"pc2 {got['shape_pc2_mean']}, elong {got['shape_elongation']}")

    # -- a circle at constant angular rate ------------------------------------ #
    # Chord headings on a circle advance at exactly the angular rate, so the
    # turn rate is w/dt for ANY lag - which is why it can be asserted without
    # knowing which lag `_heading_lag` picked.
    radius, w = 200.0, 0.05
    circ = build_series(circular_track(n, n_kpts, radius, w), cfg, fps, 0, n)
    got = aggregate_window(circ, 0, n, fps, cfg)
    chord = 2.0 * radius * np.sin(w / 2.0)
    check("circle: the constant turn rate comes back in rad/s",
          close(got["turn_abs_mean"], w / dt, 1e-3)
          and close(got["turn_abs_p90"], w / dt, 1e-2)
          and close(got["turn_std"], 0.0, 1e-2),
          f"{got['turn_abs_mean']} vs {w / dt}")
    check("circle: chord speed, arc path and net displacement match the closed form",
          close(got["speed_mean"], chord / dt, 1e-3)
          and close(got["path_length"], (n - 1) * chord, 1e-4)
          and close(got["net_displacement"], 2.0 * radius * np.sin(w * (n - 1) / 2.0), 1e-4),
          f"speed {got['speed_mean']} vs {chord / dt}, path {got['path_length']}")

    # The turn rate is a RATE, so it must not move with the window length even
    # though the lag chosen inside each window can.
    rates = []
    for length in (12, 25, 40, 60):
        rates.append(aggregate_window(circ, 0, length, fps, cfg)["turn_abs_mean"])
    check("circle: the turn rate does not move with the window length",
          max(abs(r - w / dt) for r in rates) < 1e-2,
          ", ".join(f"{r:.4f}" for r in rates))


def test_analytic_values_at_every_window(cfg) -> None:
    """The same closed forms, swept over window length and offset.

    A feature that is right over the whole track and wrong over a slice of it is
    a feature that has quietly read something outside its window. Speed and box
    area are constant here, so their expected value does not depend on the
    slice at all; path length and straightness do, in a way that is still exact.
    """
    n_kpts = cfg.n_keypoints
    fps, dt, v = 100.0, 0.01, 10.0
    total = 240
    sub = straight_track(total, n_kpts, v)
    series = build_series(sub, cfg, fps, 0, total)

    n_windows, worst, worst_note = 0, 0.0, ""
    for length in (12, 25, 50, 120, 240):
        for start in range(0, total - length + 1, max(length // 3, 1)):
            got = aggregate_window(series, start, start + length, fps, cfg)
            # A window's first frame has no incoming step, so it holds
            # length - 1 measurable steps spanning (length - 1) * v px.
            expected = {
                "speed_mean": v / dt,
                "speed_std": 0.0,
                "path_length": (length - 1) * v,
                "net_displacement": (length - 1) * v,
                "straightness": 1.0,
                "area_mean": 288.0,
                "aspect_mean": 2.0,
                "present_frac": 1.0,
                "turn_abs_mean": 0.0,
                "radius_gyration": v * np.sqrt((length * length - 1) / 12.0),
            }
            for key, want in expected.items():
                deviation = abs(float(got[key]) - want)
                if deviation > worst:
                    worst, worst_note = deviation, f"{key} at [{start}, {start + length})"
            n_windows += 1

    check("analytic: every window length and offset gives the closed-form answer",
          worst <= 1e-6, f"worst {worst:.3e} on {worst_note}")
    check("analytic: the sweep actually covered several window lengths",
          n_windows >= 40, str(n_windows))


def test_tip_rotation_is_not_inflated_by_a_gap(cfg) -> None:
    """A constant tip rotation reads the same rate across a long gap.

    ``tip_angular_speed`` is a rate, so the frames the tracker lost must cost it
    samples and nothing else. Compacting the angle before differencing makes the
    gap one step, the unwrap folds that step into ``(-pi, pi]``, and the rate is
    read off a rotation that never happened.
    """
    if cfg.n_keypoints < 3:
        return
    n, w, fps, dt = 100, 0.1, 100.0, 0.01
    sub = spinning_tip_track(n, cfg.n_keypoints, w)
    got = aggregate_window(build_series(sub, cfg, fps, 0, n), 0, n, fps, cfg)
    check("gapless: a constant tip rotation comes back in rad/s",
          close(got["tip_angular_speed"], w / dt, 1e-4),
          f"{got['tip_angular_speed']} vs {w / dt}")
    # Evenly spaced angles, so the spread is w * sd(0..n-1).
    check("gapless: the unwrapped tip angle has the closed-form spread",
          close(got["tip_angle_std"], w * np.std(np.arange(n)), 1e-4),
          f"{got['tip_angle_std']} vs {w * np.std(np.arange(n))}")

    gone = np.arange(40, 80)
    gapped = sub[~sub["frame"].isin(gone)]
    got = aggregate_window(build_series(gapped, cfg, fps, 0, n), 0, n, fps, cfg)
    # 41 frames of true rotation is 4.1 rad; folded it is 4.1 - 2*pi, and the 59
    # compacted steps then average to 13.53 rad/s instead of 10.
    folded = (58 * w + abs(4.1 - 2 * np.pi)) / 59 / dt
    check("a 40-frame gap does not inflate the tip rotation rate",
          close(got["tip_angular_speed"], w / dt, 1e-3),
          f"{got['tip_angular_speed']} vs {w / dt}; a folded bridge gives {folded:.2f}")

    # The spread can only be recovered across a gap that rotates less than pi:
    # beyond that the unwrap takes the shorter way round.
    gone = np.arange(40, 60)
    gapped = sub[~sub["frame"].isin(gone)]
    got = aggregate_window(build_series(gapped, cfg, fps, 0, n), 0, n, fps, cfg)
    kept = np.concatenate([np.arange(40), np.arange(60, n)])
    check("and the angle spread is still the one over the frames that were seen",
          close(got["tip_angle_std"], w * np.std(kept), 1e-4),
          f"{got['tip_angle_std']} vs {w * np.std(kept)}")


def test_tip_angle_spread_survives_a_gap_across_the_branch_cut(cfg) -> None:
    """A tip pointing near +-pi, lost while it crosses the cut, keeps its spread.

    Interpolating the wrapped angle across the gap bridges +3.1 to -3.1 through
    zero and leaves the samples after it 2 pi away from those before.
    """
    if cfg.n_keypoints < 3:
        return
    n, w, fps = 100, 0.002, 100.0
    sub = spinning_tip_track(n, cfg.n_keypoints, w)
    # Rotate the chain so the tip angle runs from pi - 0.1 to pi + 0.098.
    base = np.pi - 0.1
    for k in range(1, cfg.n_keypoints):
        r = 6.0 * (k - 1)
        ang = base + w * np.arange(n)
        sub[f"kp_x_{k}"] = (200.0 + r * np.cos(ang)).astype(np.float32)
        sub[f"kp_y_{k}"] = (200.0 + r * np.sin(ang)).astype(np.float32)
    gapped = sub[~sub["frame"].isin(np.arange(45, 55))]
    got = aggregate_window(build_series(gapped, cfg, fps, 0, n), 0, n, fps, cfg)
    kept = np.concatenate([np.arange(45), np.arange(55, n)])
    check("a gap across +-pi does not add 2 pi to the tip angle spread",
          close(got["tip_angle_std"], w * np.std(kept), 1e-4),
          f"{got['tip_angle_std']} vs {w * np.std(kept)}")


def test_window_is_independent_of_the_series_around_it(cfg) -> None:
    """The property that makes the clip boundary stop being the measurement unit.

    A window aggregated out of a long series must equal the same window
    aggregated out of a series built for it alone. Anything that leaked across
    the window edge - an unwrap, a gap fill, a detrend, a step measured into the
    window - would show up here and nowhere else.
    """
    n_kpts = cfg.n_keypoints
    sub = synthetic_track(300, n_kpts, seed=2, drop=(90, 91, 150))
    whole = build_series(sub, cfg, 100.0, 0, 300)
    worst, worst_key = 0.0, ""
    for start, length in ((0, 50), (25, 50), (75, 40), (140, 60), (250, 50), (12, 7)):
        window = sub[(sub["frame"] >= start) & (sub["frame"] < start + length)]
        alone = build_series(window, cfg, 100.0, start, length)
        a = aggregate_window(whole, start, start + length, 100.0, cfg)
        b = aggregate_window(alone, start, start + length, 100.0, cfg)
        for key, value in diff_features(a, b).items():
            if value > worst:
                worst, worst_key = value, key
    check("a window's features do not depend on the track built around it",
          worst <= TOL, f"{worst_key} differs by {worst:.3e}")


def test_guards(cfg) -> None:
    """A window the series cannot answer for is refused, never silently padded."""
    sub = synthetic_track(40, cfg.n_keypoints)
    series = build_series(sub, cfg, 100.0, 100, 40)
    for name, args in (("empty window", (100, 100)), ("reversed window", (120, 110)),
                       ("window before the series", (90, 110)),
                       ("window past the series", (130, 145))):
        try:
            aggregate_window(series, args[0], args[1], 100.0, cfg)
            refused = False
        except ValueError:
            refused = True
        check(f"{name} is refused rather than silently padded", refused)


# The feature set, written out. A frozen literal rather than anything derived
# from the module, so that renaming or dropping a column has to be done here too
# and cannot pass unnoticed - these names are the contract with every model
# already fitted, and a column that silently changes name is a silent
# misalignment rather than an error.
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
    sub = synthetic_track(60, cfg.n_keypoints)
    series = build_series(sub, cfg, 100.0, 0, 60)
    got = aggregate_window(series, 0, 60, 100.0, cfg)
    check("aggregate_window emits exactly the declared feature set",
          set(got) == set(EXPECTED_FEATURES), str(set(got) ^ set(EXPECTED_FEATURES)))
    check("and no spectral column leaks into it",
          not any(k.startswith(SPECTRAL) for k in got),
          str([k for k in got if k.startswith(SPECTRAL)]))
    check("feature_names() is the schema aggregate_window actually produces",
          set(feature_names()) == set(got) and len(feature_names()) == len(got),
          str(set(feature_names()) ^ set(got)))
    reduced = len(feature_names()) - len(WINDOW_FUNCTIONALS)
    check("most features are declared in SERIES_SPEC rather than hand-written",
          reduced > len(WINDOW_FUNCTIONALS), f"{reduced} declared vs {len(WINDOW_FUNCTIONALS)}")
    check("every SERIES_SPEC column exists in the series",
          set(SERIES_SPEC) == set(series.columns))


# --------------------------------------------------------------------------- #
# real data
# --------------------------------------------------------------------------- #
def regress_project(root: Path) -> tuple[int, dict] | None:
    """Window-independence on every clip of a prepared project.

    For each clip, the features aggregated out of a series covering the WHOLE
    track are compared against the features aggregated out of a series built for
    that clip alone. They must be identical: a window's features are a property
    of the window.

    Run over a whole manifest rather than one hand-made track, because that is
    where the awkward cases are - gaps, short tracks, clips planned past the
    last detection, near-degenerate poses. Anything that leaked across a window
    edge (an unwrap, a gap fill, a step measured in from the frame before, a lag
    chosen from the surrounding track) shows up here and almost nowhere else.
    """
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
        check(f"{root.name}: every clip's features depend on the window alone",
              not failed, str(failed))
        total += n
        for key, entry in stats.items():
            merged = pooled.setdefault(key, {"max": 0.0, "over": 0})
            merged["max"] = max(merged["max"], entry["max"])
            merged["over"] += entry["over"]
    if total:
        failed = report(pooled, total, "POOLED over every project")
        check("pooled over every real clip: no column depends on the surrounding track",
              not failed, str(failed))


# --------------------------------------------------------------------------- #
# cost
# --------------------------------------------------------------------------- #
def test_faster_than_recomputing(cfg) -> None:
    """One series plus N slices against one series PER window.

    Per-window construction is what the module replaced: measuring a clip used
    to mean walking that clip's frames from scratch, so overlapping clips paid
    for the same frames repeatedly. Building a series per window reproduces that
    cost exactly, in code that still exists, and the comparison is the honest
    one - both sides compute the same features by the same arithmetic and differ
    only in how often the per-frame work is done.

    Measured on non-overlapping AND overlapping windows, because that is where
    the asymptotics live. With stride == length the two do the same per-frame
    work and the shared series only saves fixed overhead; every increment of
    overlap is work the per-window path repeats and this one does not, so the
    ratio grows with it.
    """
    n_frames = 500
    sub = synthetic_track(n_frames, cfg.n_keypoints, seed=3, drop=(101, 102, 300))

    def per_window(windows) -> None:
        for start, end in windows:
            window = sub[(sub["frame"] >= start) & (sub["frame"] < end)]
            series = build_series(window, cfg, 100.0, start, end - start)
            aggregate_window(series, start, end, 100.0, cfg)

    def shared(windows) -> None:
        series = build_series(sub, cfg, 100.0, 0, n_frames)
        for start, end in windows:
            aggregate_window(series, start, end, 100.0, cfg)

    def slicing_only(windows) -> None:
        for start, end in windows:
            aggregate_window(whole, start, end, 100.0, cfg)

    def best(fn, windows, repeats: int = 5) -> float:
        fn(windows)                            # warm the import-time lazy paths
        return min(_time(lambda: fn(windows)) for _ in range(repeats))

    whole = build_series(sub, cfg, 100.0, 0, n_frames)

    for label, length, stride in (("no overlap", 50, 50), ("stride 25", 50, 25),
                                  ("stride 10", 50, 10)):
        windows = [(s, s + length) for s in range(0, n_frames - length + 1, stride)]
        t_each = best(per_window, windows)
        t_shared = best(shared, windows)
        t_slice = best(slicing_only, windows)
        print(f"\n  {n_frames}-frame track, {len(windows)} windows of {length}, {label}")
        print(f"    {len(windows):3d}x build_series + aggregate {t_each * 1e3:8.2f} ms")
        print(f"    build_series + {len(windows):3d} windows     {t_shared * 1e3:8.2f} ms"
              f"   (slicing ~{t_slice * 1e3:.2f} ms of that)")
        print(f"    {t_each / t_shared:5.2f}x overall,  slicing is "
              f"{t_slice / t_shared:.0%} of the shared path")
        check(f"{label}: one series + N slices beats a series per window",
              t_each / t_shared > 1.0, f"{t_each / t_shared:.2f}x")
        # The structural claim underneath the timing: slicing is the cheap half,
        # which is why amortising the construction is what buys the speed-up. If
        # this ever inverted, the module would not be worth its complexity.
        check(f"{label}: slicing is cheaper than building the series it slices",
              t_slice < t_shared, f"slicing {t_slice * 1e3:.2f} ms of "
              f"{t_shared * 1e3:.2f} ms")


def _time(fn) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


# --------------------------------------------------------------------------- #
