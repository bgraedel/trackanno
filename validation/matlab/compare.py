"""trackanno against the MATLAB pipeline, output by output, on the snippets cells.py wrote.

    uv run python validation/matlab/compare.py --cells OUT

Writes comparison.csv, same_input.png (MATLAB reads the keypoints trackanno reads),
dense_trace.png (MATLAB reads the dense skeleton) and summary.md.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulate import N_KEYPOINTS  # noqa: E402

from trackanno.measure import wavespectra as ws  # noqa: E402

SNIPPET = 75
# trackanno column -> (MATLAB output, Legendre index or None, a length to convert from um)
PAIRS = {
    "wave_freq_hz": ("domFreq", None, False),
    "wave_freq_proximal_hz": ("domFreqProximal", None, False),
    "wave_freq_central_hz": ("domFreqCentral", None, False),
    "wave_freq_distal_hz": ("domFreqDistal", None, False),
    "wave_amp_mean": ("legendreCoeffsDomFreqAmp", 0, False),
    "wave_amp_asym": ("legendreCoeffsDomFreqAmp", 1, False),
    "wave_amp_para": ("legendreCoeffsDomFreqAmp", 2, False),
    "wave_amp_integral": ("domFreqAmpSummed", None, True),
    "wave_phase_asym": ("legendreCoeffsDomFreqPhase", 1, False),
    "wave_phase_para": ("legendreCoeffsDomFreqPhase", 2, False),
    "wave_phase_rsq": ("phaseLinearityRsq", None, False),
    "wave_phase_cycles": ("numPhaseCycles", None, False),
    "wave_static_mean": ("averageFlagAngleFromBase", None, False),
    "wave_freq_dominance": ("domFreqDominance", None, False),
    "wave_fit_error": ("qualityOfFitAngle", None, False),
    "wave_xy_fit_error": ("qualityOfFitXYOptimal", None, False),
    "wave_disp_amp_max": ("amplitudeSpatialAllData", None, True),
    "wave_angular_amp": ("amplitudeAngularAllData", None, False),
    "wave_length_px": ("flagLength", None, True),
    "wave_bad_window": ("badFileFlag", None, False),
}
FLAGS = {"wave_bad_window"}
# MATLAB input -> the keypoints trackanno reads for it
INPUTS = {"kp_clean": "clean", "kp_jitter": "jitter", "dense_clean": "clean", "dense_traced": "jitter"}
FIGURES = {
    "same_input": (("kp_clean", "keypoints"), ("kp_jitter", "keypoints + jitter")),
    "dense_trace": (("dense_clean", "clean: keypoints vs exact skeleton"),
                    ("dense_traced", "realistic: jittered keypoints vs integer-pixel skeleton")),
}


def matlab_values(folder: Path, scale: float) -> dict:
    m = scipy.io.loadmat(folder / "compared.mat", squeeze_me=True)
    out = {}
    for col, (var, index, length) in PAIRS.items():
        v = np.atleast_1d(m[var]) if var in m else np.array([np.nan])
        v = float(v[index] if index is not None else v[0])
        out[col] = v / scale if length else v
    return out


def collect(cells: Path) -> pd.DataFrame:
    meta = pd.read_csv(cells / "cells.csv")
    fps, scale = float(meta.fps.iloc[0]), float(meta.pixel_size_um.iloc[0])
    kx = [f"kp_x_{k}" for k in range(N_KEYPOINTS)]
    ky = [f"kp_y_{k}" for k in range(N_KEYPOINTS)]
    rows = []
    for cell in meta.itertuples():
        tracks = pd.read_parquet(cells / f"cell_{cell.seed:03d}_keypoints.parquet")
        for folder in sorted(cells.glob(f"*/cell_{cell.seed:03d}_f*.tif_analysis")):
            if not (folder / "compared.mat").exists():
                continue
            kind, lo = folder.parent.name, int(folder.name.split("_f")[1].split(".")[0])
            track = tracks[tracks.kind == INPUTS[kind]]
            px, py = track[kx].to_numpy(float), track[ky].to_numpy(float)
            got = ws.spectral_series(px[lo:lo + SNIPPET], py[lo:lo + SNIPPET], fps,
                                     window=SNIPPET, hop=SNIPPET).iloc[0]
            want = matlab_values(folder, scale)
            for col in PAIRS:
                rows.append({"seed": cell.seed, "start": lo, "mode": cell.mode,
                             "segments_per_wavelength": cell.segments_per_wavelength,
                             "matlab_input": kind, "column": col,
                             "matlab": want[col], "trackanno": float(got[col])})
    return pd.DataFrame(rows)


def unit(col: str, d: pd.DataFrame) -> np.ndarray:
    """What 1% is taken of: the amplitude for its shape coefficients, pi for the phase curvature."""
    if col in ("wave_amp_asym", "wave_amp_para"):
        return d["amp"].abs().to_numpy(float)
    if col == "wave_phase_para":
        return np.full(len(d), np.pi)
    return d["matlab"].abs().to_numpy(float)


def stats(df: pd.DataFrame, kind: str, col: str) -> dict:
    """r and the share within 1% over resolvable beats; agreement for flags."""
    d = df[(df.matlab_input == kind) & (df.column == col) & df.resolvable]
    d = d[np.isfinite(d.matlab) & np.isfinite(d.trackanno)]
    x, y = d.matlab.to_numpy(float), d.trackanno.to_numpy(float)
    if col in FLAGS:
        return {"n": len(d), "agree": float(np.mean(x == y)) if len(d) else np.nan}
    r = np.corrcoef(x, y)[0, 1] if len(d) > 2 and np.std(x) > 0 else np.nan
    within = float(np.mean(np.abs(y - x) <= 0.01 * unit(col, d) + 1e-9)) if len(d) else np.nan
    return {"n": len(d), "r": r, "within": within}


def figure(df: pd.DataFrame, kinds, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, surface = "#0b0b0b", "#7a7974", "#e4e3df", "#fcfcfb"
    colors = ("#2a78d6", "#eb6834")
    plt.rcParams.update({"font.size": 7, "axes.edgecolor": muted, "xtick.color": muted,
                         "ytick.color": muted, "axes.linewidth": 0.6})
    cols = list(PAIRS)
    n_cols = 5
    n_rows = int(np.ceil(len(cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.3 * n_cols, 2.35 * n_rows),
                             facecolor=surface, squeeze=False)
    rng = np.random.default_rng(0)
    for ax, col in zip(axes.flat, cols):
        ax.set_facecolor(surface)
        shown = []
        # The realistic series first, so the clean one is drawn on top.
        for i in (1, 0):
            kind = kinds[i][0]
            d = df[(df.matlab_input == kind) & (df.column == col)]
            x, y = d.matlab.to_numpy(float), d.trackanno.to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            res = d.resolvable.to_numpy()
            if col in FLAGS:
                x = x + rng.uniform(-0.2, 0.2, x.size)
                y = y + rng.uniform(-0.2, 0.2, y.size)
            ax.scatter(x[ok & ~res], y[ok & ~res], s=7, facecolor="none", edgecolor=colors[i],
                       linewidth=0.5, alpha=0.6, zorder=2)
            ax.scatter(x[ok & res], y[ok & res], s=7, color=colors[i], edgecolor=surface,
                       linewidth=0.3, alpha=0.8, zorder=3)
            shown.append(np.r_[x[ok & res], y[ok & res]])
            s = stats(df, kind, col)
            note = (f"{s['agree']:.0%} agree" if col in FLAGS
                    else f"r {s['r']:.3f}, {s['within']:.0%} within 1%")
            ax.text(0.03, 0.97 - 0.08 * i, note, transform=ax.transAxes, ha="left", va="top",
                    fontsize=5.5, color=colors[i])
        if col in FLAGS:
            lo, hi = -0.4, 1.4
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
        else:
            vals = np.concatenate(shown)
            lo, hi = np.nanpercentile(vals, [1, 99]) if vals.size else (0.0, 1.0)
            pad = 0.08 * (hi - lo if hi > lo else 1.0)
            lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], color=muted, lw=0.6, zorder=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color=grid, lw=0.5, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.set_title(col, fontsize=7, color=ink, pad=3)
    for ax in axes.flat[len(cols):]:
        ax.axis("off")
    fig.supxlabel("MATLAB pipeline", fontsize=7, color=muted)
    fig.supylabel("trackanno", fontsize=7, color=muted)
    handles = [plt.Line2D([], [], ls="", marker="o", color=colors[i], markersize=4,
                          label=kinds[i][1]) for i in (0, 1)]
    handles.append(plt.Line2D([], [], ls="", marker="o", markerfacecolor="none",
                              markeredgecolor=muted, markersize=4,
                              label="static cell or wave < 2.5 keypoint segments"))
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=170, facecolor=surface)
    plt.close(fig)


def summary(df: pd.DataFrame) -> str:
    lines = []
    for name, kinds in FIGURES.items():
        n = stats(df, kinds[0][0], "wave_freq_hz")["n"]
        lines += [f"## {name} ({n} resolvable beating snippets)", "",
                  f"| column | {kinds[0][1]} | {kinds[1][1]} |", "|---|---|---|"]
        for col in PAIRS:
            cells = []
            for kind, _label in kinds:
                s = stats(df, kind, col)
                cells.append(f"{s['agree']:.0%} agree" if col in FLAGS
                             else f"r {s['r']:.3f}, {s['within']:.0%} within 1%")
            lines.append(f"| {col} | {cells[0]} | {cells[1]} |")
        lines.append("")
    lines.append("Within 1%: of MATLAB's value; of its wave_amp_mean for wave_amp_asym and "
                 "wave_amp_para; of pi for wave_phase_para.")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cells", required=True, help="folder cells.py wrote and run_matlab.py filled")
    args = ap.parse_args(argv)
    warnings.simplefilter("ignore")
    cells = Path(args.cells)
    df = collect(cells)
    df["resolvable"] = (df["mode"] != "static") & (df.segments_per_wavelength >= 2.5)
    amp = df[df.column == "wave_amp_mean"].set_index(["seed", "start", "matlab_input"]).matlab
    df["amp"] = amp.reindex(pd.MultiIndex.from_frame(df[["seed", "start", "matlab_input"]])).to_numpy()
    df.drop(columns="amp").to_csv(cells / "comparison.csv", index=False)
    for name, kinds in FIGURES.items():
        figure(df, kinds, cells / f"{name}.png")
    (cells / "summary.md").write_text(summary(df), encoding="utf-8")
    print(summary(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
