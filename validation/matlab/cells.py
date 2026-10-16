"""Simulated cells as inputs for the MATLAB pipeline and for trackanno.

    uv run python validation/matlab/cells.py --simulator PATH/synthetic_leishmania.py --out OUT

Per cell and 0.75 s snippet, four MATLAB inputs (``rawcoordinates.txt``: x,y,half-width per point,
body then flagellum, px):

* ``dense_clean``: the simulator's body spine and flagellum at 1 px;
* ``dense_traced``: the same on integer pixels, as a skeleton tracer reports it;
* ``kp_clean``, ``kp_jitter``: what trackanno reads, the keypoints, as a trace: Head to Base
  straight, then trackanno's spline through the flagellum keypoints.

The keypoints trackanno reads go to ``cell_NNN_keypoints.parquet`` (clean and jittered).
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulate import N_INTERIOR, N_KEYPOINTS, load_simulator, mode_params, simulate  # noqa: E402

FLAG_HALF_WIDTH = 2               # px; the body is at least two wider, so MATLAB tells them apart
SNIPPET = 75                      # frames, trackanno's default window
# Jitter measured on a real project at 0.325 um/px; scaled to keep it the same in um.
NOISE_BODY_PX = 0.3
NOISE_FLAGELLUM_PX = 0.6
KINDS = ("dense_clean", "dense_traced", "kp_clean", "kp_jitter")


def resample(points: np.ndarray, step: float) -> np.ndarray:
    """``points`` evenly in arclength, about ``step`` apart."""
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(points, axis=0).T))])
    u = np.linspace(0.0, s[-1], max(2, int(round(s[-1] / step)) + 1))
    return np.column_stack([np.interp(u, s, points[:, 0]), np.interp(u, s, points[:, 1])])


def line(points: np.ndarray, widths: np.ndarray, integer: bool = False) -> str:
    if integer:
        points = np.round(points).astype(int)
        keep = np.concatenate([[True], np.any(np.diff(points, axis=0) != 0, axis=1)])
        points, widths = points[keep], widths[keep]
        return " ".join(f"{x},{y},{w:g}" for (x, y), w in zip(points, widths))
    return " ".join(f"{x:.3f},{y:.3f},{w:g}" for (x, y), w in zip(points, widths))


def dense_frame(spine: np.ndarray, half: np.ndarray, flag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Body spine then flagellum at 1 px, with half-widths."""
    body = resample(spine, 1.0)
    body_half = np.interp(np.linspace(0, 1, len(body)), np.linspace(0, 1, len(half)), half)
    body_half = np.maximum(np.round(body_half), FLAG_HALF_WIDTH + 2)
    fl = resample(flag, 1.0)
    return np.vstack([body, fl]), np.concatenate([body_half, np.full(len(fl), FLAG_HALF_WIDTH)])


def keypoint_frame(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Head to Base straight (stopping a pixel short), then the spline through the flagellum."""
    head, base = np.array([x[0], y[0]]), np.array([x[1], y[1]])
    body = np.linspace(head, base, max(2, int(round(np.linalg.norm(base - head))) + 1))[:-1]
    k = len(x) - 1
    fine = np.linspace(0.0, k - 1.0, 512)
    curve = np.column_stack([CubicSpline(np.arange(k), x[1:])(fine),
                             CubicSpline(np.arange(k), y[1:])(fine)])
    fl = resample(curve, 1.0)
    return (np.vstack([body, fl]),
            np.concatenate([np.full(len(body), FLAG_HALF_WIDTH + 4), np.full(len(fl), FLAG_HALF_WIDTH)]))


def jitter(track: pd.DataFrame, seed: int, pixel_size_um: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    noisy = track.copy()
    scale = 0.325 / pixel_size_um
    for k in range(N_KEYPOINTS):
        sigma = (NOISE_BODY_PX if k < 2 else NOISE_FLAGELLUM_PX) * scale
        for axis in ("x", "y"):
            noisy[f"kp_{axis}_{k}"] = track[f"kp_{axis}_{k}"] + rng.normal(0.0, sigma, len(track))
    return noisy


def one_cell(sl, seed: int, out: Path, fps: float, frames: int, pixel_size_um: float,
             starts: list[int]) -> dict:
    rng = np.random.default_rng(seed)
    p = sl.sample_random_parasite(rng, (4000, 4000))
    p.mode_switch_rate = 0.0
    p.mode_schedule = []
    p.n_flagellum_keypoints = N_INTERIOR
    clean, state = simulate(sl, p, fps, frames, pixel_size_um, traces=True)
    noisy = jitter(clean, seed, pixel_size_um)
    pd.concat([clean.assign(kind="clean"), noisy.assign(kind="jitter")]).to_parquet(
        out / f"cell_{seed:03d}_keypoints.parquet")
    kx = [f"kp_x_{k}" for k in range(N_KEYPOINTS)]
    ky = [f"kp_y_{k}" for k in range(N_KEYPOINTS)]
    for lo in starts:
        lines = {kind: [] for kind in KINDS}
        for i in range(lo, lo + SNIPPET):
            spine, half = state["body_spine"][i]
            pts, widths = dense_frame(spine, half, state["flagellum"][i])
            lines["dense_clean"].append(line(pts, widths))
            lines["dense_traced"].append(line(pts, widths, integer=True))
            for kind, track in (("kp_clean", clean), ("kp_jitter", noisy)):
                pts, widths = keypoint_frame(track[kx].to_numpy(float)[i], track[ky].to_numpy(float)[i])
                lines[kind].append(line(pts, widths))
        for kind, text in lines.items():
            folder = out / kind / f"cell_{seed:03d}_f{lo}.tif_analysis"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "rawcoordinates.txt").write_text("\n".join(text) + "\n", encoding="ascii")
    mode = state["params"].beat_mode
    _freq, wavelength_um = mode_params(sl, state["params"], mode)
    length = state["flagellum_len_px"]
    return {"seed": seed, "mode": mode,
            "segments_per_wavelength": wavelength_um * state["ppum"] / (length / (N_INTERIOR + 1))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--simulator", required=True, help="path to synthetic_leishmania.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cells", type=int, default=60)
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--fps", type=float, default=100.0)
    ap.add_argument("--pixel-size-um", type=float, default=0.325)
    ap.add_argument("--starts", type=int, nargs="+", default=[100, 250],
                    help="first frame of each snippet")
    args = ap.parse_args(argv)
    warnings.simplefilter("ignore")
    sl = load_simulator(args.simulator)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cells = [one_cell(sl, seed, out, args.fps, args.frames, args.pixel_size_um, args.starts)
             for seed in range(args.cells)]
    pd.DataFrame(cells).assign(fps=args.fps, pixel_size_um=args.pixel_size_um).to_csv(
        out / "cells.csv", index=False)
    print(f"{len(cells)} cells -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
