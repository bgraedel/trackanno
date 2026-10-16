"""Keypoint tracks and dense ground truth from the synthetic Leishmania simulator, without rendering.

The simulator is loaded from a path (``synthetic_leishmania.py``); nothing of it is copied here.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

N_INTERIOR = 5                     # Head, Base, Flag1..Flag5, Tip: the projects' skeleton
N_KEYPOINTS = N_INTERIOR + 3


def load_simulator(path: str | Path):
    """The simulator module at ``path``."""
    path = Path(path)
    spec = importlib.util.spec_from_file_location("synthetic_leishmania", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"no simulator at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module           # dataclasses look their module up while it loads
    spec.loader.exec_module(module)
    return module


def mode_params(sl, p, mode: str) -> tuple[float, float]:
    """(frequency in Hz, wavelength in um) the simulator uses for ``mode``."""
    if mode == "tip_to_base":
        return (sl._coalesce(p.tip_to_base_frequency, p.beat_frequency),
                sl._coalesce(p.tip_to_base_wavelength, p.beat_wavelength))
    if mode == "base_to_tip":
        return (sl._coalesce(p.base_to_tip_frequency, p.beat_frequency),
                sl._coalesce(p.base_to_tip_wavelength, p.beat_wavelength))
    return np.nan, np.nan


def body_spine(sl, pw) -> tuple[np.ndarray, np.ndarray]:
    """Body centreline (posterior to anterior) in image px and its half-width there."""
    xs, ys, _phi, s = sl._body_spine(pw, 200)
    spine = sl._apply_body_pose(pw, np.column_stack([xs, ys]).astype(float))
    peak = pw.body_peak_position
    rel = np.where(s < peak, (s - peak) / (peak + 1.05), (s - peak) / (1.05 - peak))
    half = (pw.body_width / 2) * np.sqrt(np.clip(1 - rel ** 2, 0, 1))
    return spine, half


def simulate(sl, p, fps: float, n_frames: int, pixel_size_um: float, dense: int = 201,
             move: bool = True, traces: bool = False) -> tuple[pd.DataFrame, dict]:
    """A noise-free track as trackanno reads it, and the simulator's state per frame.

    Keypoints follow the simulator's own placement: Head at the posterior pole of the
    straight body axis, Base at the flagellum root, chain nodes evenly spaced in arc length.
    With ``traces`` the state also holds each frame's body spine, its half-width and the
    dense flagellum, in image px.
    """
    p = copy.deepcopy(p)
    initial = copy.deepcopy(p)
    optics = sl.OpticsParams(pixel_size_um=pixel_size_um)
    n_pts = (N_INTERIOR + 1) * 100 + 1          # exact even node spacing
    rows, theta, offsets, centres, orientations, spines, flagella = [], [], [], [], [], [], []
    for i in range(n_frames):
        t = i / fps
        p_px = sl._parasite_lengths_to_px(p, optics)
        pw = sl._apply_body_wobble(p_px, t)
        flag = np.asarray(sl.flagellum_curve(pw, t, n_points=n_pts), dtype=float)
        head = np.array([pw.center_x, pw.center_y]) - (pw.body_length / 2) * np.array(
            [np.cos(pw.body_orientation), np.sin(pw.body_orientation)])
        chain = [np.asarray(q, dtype=float) for q in sl._flag_chain_points(flag, N_INTERIOR)]
        pts = np.array([head, flag[0], *chain, flag[-1]])
        body = np.asarray(sl.body_polygon(pw), dtype=float)
        lo, hi = np.vstack([body, flag]).min(axis=0), np.vstack([body, flag]).max(axis=0)
        row = {"frame": i, "track_id": 1, "x1": lo[0], "y1": lo[1], "x2": hi[0], "y2": hi[1],
               "conf": 1.0}
        for k, (x, y) in enumerate(pts):
            row[f"kp_x_{k}"], row[f"kp_y_{k}"], row[f"kp_c_{k}"] = x, y, 1.0
        rows.append(row)
        _, th = sl.beat_tangent_angle(p_px, t, n_points=dense)
        theta.append(th)
        # Head-to-Base axis to the curve's base tangent, beyond the beat's own angle there.
        tangent0 = np.arctan2(flag[1, 1] - flag[0, 1], flag[1, 0] - flag[0, 0])
        axis = np.arctan2(flag[0, 1] - head[1], flag[0, 0] - head[0])
        th0 = np.interp(0.5 / (n_pts - 1), np.linspace(0, 1, dense), th)
        offsets.append(np.angle(np.exp(1j * (tangent0 - axis - th0))))
        if traces:
            spines.append(body_spine(sl, pw))
            flagella.append(flag)
        centres.append((pw.center_x, pw.center_y))
        orientations.append(pw.body_orientation)
        if move:
            sl.advance_parasites([p], 1.0 / fps, (10 ** 6, 10 ** 6), periodic=False, t=t,
                                 optics=optics)
    ppum = sl.pixels_per_um(optics)
    state = {
        "params": initial,
        "theta": np.asarray(theta) + np.asarray(offsets)[:, None],   # body frame, (frames, dense)
        "s": np.linspace(0.0, 1.0, dense),
        "flagellum_len_px": initial.flagellum_length * ppum,
        "body_len_px": initial.body_length * ppum,
        "centre": np.asarray(centres),
        "orientation": np.asarray(orientations),
        "ppum": ppum,
    }
    if traces:
        state["body_spine"] = spines
        state["flagellum"] = flagella
    return pd.DataFrame(rows), state


def add_noise(track: pd.DataFrame, body_px: float, flagellum_px: float, seed: int) -> pd.DataFrame:
    """Gaussian keypoint jitter: ``body_px`` on Head and Base, ``flagellum_px`` on the chain."""
    rng = np.random.default_rng(seed)
    noisy = track.copy()
    for k in range(N_KEYPOINTS):
        sigma = body_px if k < 2 else flagellum_px
        for axis in ("x", "y"):
            col = f"kp_{axis}_{k}"
            noisy[col] = track[col] + rng.normal(0.0, sigma, len(track))
    return noisy


def active_mode(sl, p, t0: float, t1: float) -> str | None:
    """The one beat mode active over [t0, t1] outside any blend, else None."""
    states = [sl._get_active_mode_blend(p, t) for t in np.linspace(t0, t1, 12)]
    if all(a == b == states[0][0] and blend == 0.0 for a, b, blend in states):
        return states[0][0]
    return None
