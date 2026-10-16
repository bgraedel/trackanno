"""True values of the wave measurements: the same analysis on the simulator's exact flagellum."""

from __future__ import annotations

import numpy as np

from trackanno.measure import wavespectra as ws


def dense_curves(theta: np.ndarray, s: np.ndarray, length_px: float,
                 head_base_px: np.ndarray) -> dict:
    """The flagellum of each frame from its exact tangent angle (frames x s), as the pipeline frames it.

    The angle is relative to the Head-Base axis, so the Head sits ``head_base_px`` behind the Base
    on -x; the curve is then smoothed and framed exactly as a tracked one.
    """
    arc = s * length_px
    step = np.diff(arc)
    mid_c = 0.5 * (np.cos(theta[:, 1:]) + np.cos(theta[:, :-1]))
    mid_s = 0.5 * (np.sin(theta[:, 1:]) + np.sin(theta[:, :-1]))
    zero = np.zeros((theta.shape[0], 1))
    x = np.concatenate([zero, np.cumsum(mid_c * step, axis=1)], axis=1)
    y = np.concatenate([zero, np.cumsum(mid_s * step, axis=1)], axis=1)
    out = ws.body_frame_curves(x, y, -0.5 * head_base_px, np.zeros_like(head_base_px))
    out["overall"] = head_base_px + length_px
    return out


def wave_truth(theta: np.ndarray, s: np.ndarray, length_px: float, head_base_px: np.ndarray,
               fps: float) -> dict:
    """The ``wave_*`` outputs of one window of the exact, noise-free angle (frames x s)."""
    curves = dense_curves(theta, s, length_px, head_base_px)
    return ws.analyse_window(curves, 0, theta.shape[0], fps)


def physical_truth(mode: str, freq: float, wavelength_px: float, length_px: float) -> dict:
    """The simulator's own beat parameters, where a measurement estimates one."""
    direction = {"tip_to_base": -1.0, "base_to_tip": 1.0}.get(mode, np.nan)
    return {"wave_freq_hz": freq, "wave_wavelength_px": wavelength_px,
            "wave_direction": direction, "wave_speed_px_s": direction * wavelength_px * freq,
            "wave_length_px": length_px}
