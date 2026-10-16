"""Deterministic beating cells shared by the wavespectra tests and the MATLAB runs that pin them.

A straight body from Head to Base along +x and a flagellum whose tangent angle is
``static(s) + amp(s) sin(2 pi f t - direction k s)``, integrated at 0.25 px.
"""

from __future__ import annotations

import numpy as np

FPS = 100.0
HEAD = np.array([100.0, 200.0])
BASE = np.array([160.0, 200.0])
LENGTH = 150.0                    # flagellum, px
NODES = 61                        # Base .. Tip, evenly spaced in arc length

CASES = {
    # A tip-to-base beat with an envelope pinned near the base.
    "tip_to_base": dict(freq=21.3, wavelength=0.7 * LENGTH, direction=-1,
                        amp=lambda u: 0.5 * np.sin(np.pi * u) ** 0.8 + 0.05,
                        static=lambda u: 0.1 * u),
    # A slow base-to-tip beat on a curled flagellum.
    "base_to_tip": dict(freq=5.7, wavelength=2.0 * LENGTH, direction=+1,
                        amp=lambda u: 0.9 * np.sqrt(u), static=lambda u: 1.2 * u ** 2),
    # A beat small enough for the smoothing to matter.
    "small": dict(freq=12.4, wavelength=1.1 * LENGTH, direction=-1,
                  amp=lambda u: 0.08 + 0 * u, static=lambda u: 0 * u),
}


def curve(case: str, n_frames: int, step: float = 0.25) -> np.ndarray:
    """Flagellum points (frames, n, 2) at ``step`` px from Base to Tip."""
    spec = CASES[case]
    n = int(round(LENGTH / step))
    u = (np.arange(n) + 0.5) / n
    t = np.arange(n_frames)[:, None] / FPS
    k = 2.0 * np.pi / spec["wavelength"]
    theta = spec["static"](u)[None, :] + spec["amp"](u)[None, :] * np.sin(
        2.0 * np.pi * spec["freq"] * t - spec["direction"] * k * u[None, :] * LENGTH)
    x = np.concatenate([np.zeros((n_frames, 1)), np.cumsum(step * np.cos(theta), axis=1)], 1)
    y = np.concatenate([np.zeros((n_frames, 1)), np.cumsum(step * np.sin(theta), axis=1)], 1)
    return np.stack([x + BASE[0], y + BASE[1]], axis=-1)


def keypoints(case: str, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Head, Base and NODES - 1 more flagellum points: px, py of shape (frames, NODES + 1)."""
    dense = curve(case, n_frames)
    idx = np.round(np.linspace(0, dense.shape[1] - 1, NODES)).astype(int)
    nodes = dense[:, idx]
    px = np.concatenate([np.full((n_frames, 1), HEAD[0]), nodes[..., 0]], axis=1)
    py = np.concatenate([np.full((n_frames, 1), HEAD[1]), nodes[..., 1]], axis=1)
    return px, py


def trace_lines(case: str, n_frames: int) -> list[str]:
    """The same cell as the MATLAB pipeline reads it: body then flagellum, about 1 px apart, x,y,width."""
    # The body stops a pixel short of the Base, where the flagellum's first point is.
    body = np.linspace(HEAD, BASE, int(round(np.linalg.norm(BASE - HEAD))) + 1)[:-1]
    dense = curve(case, n_frames)[:, ::4]
    lines = []
    for frame in dense:
        parts = [f"{x:.3f},{y:.3f},6" for x, y in body]
        parts += [f"{x:.3f},{y:.3f},2" for x, y in frame]
        lines.append(" ".join(parts))
    return lines
