"""Validate every trackanno measurement against the synthetic Leishmania simulator.

    uv run python validation/run.py --simulator PATH/synthetic_leishmania.py --out OUT_DIR

Random cells from the simulator's own sampler are tracked as keypoints with realistic jitter,
dropouts and detector confidences. Each spectral window, each clip and each frame is compared with
its truth: the same beat analysis on the simulator's exact flagellum (waveform and beat per frame),
the simulator's beat parameters (frequency, wavelength, direction, length), its exact body centre
and axis (body motion), or the noise-free keypoints (descriptors defined on the keypoints themselves).
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from simulate import N_KEYPOINTS, active_mode, add_noise, load_simulator, mode_params, simulate  # noqa: E402
from truth import beat_truth, physical_truth, wave_truth  # noqa: E402

from trackanno.measure import instantaneous as inst  # noqa: E402
from trackanno.measure import trackseries as ts  # noqa: E402
from trackanno.measure import wavespectra as ws  # noqa: E402

FPS = 100.0
CLIP_FRAMES = 50
# Measured on a real project: robust sigma of keypoint second differences.
NOISE_BODY_PX = 0.3
NOISE_FLAGELLUM_PX = 0.6
# Detection dropouts in the noisy track: whole frames, and single flagellum keypoints.
FRAME_DROP = 0.03
KEYPOINT_DROP = 0.005

# Truth of each column. dense: the same analysis on the simulator's exact flagellum; motion: the
# same estimator on the exact body centre and axis; param: the simulator's value; clean: noise-free
# keypoints; input: what was fed in (dropouts, confidences).
WAVE_TRUTH = {
    "wave_interp_frac": "input",
    # A correlation whose size depends on how finely the flagellum is sampled; its sign is checked
    # against the simulator as wave_flow_direction.
    "wave_flow": "clean", "wave_flow_coherence": "clean",
    # A signal-to-noise statistic with no true value: shown against the true amplitude.
    "wave_excess": "behaviour", "inst_excess": "behaviour",
}
KIN_TRUTH = {
    "present_frac": "input", "conf_mean": "input", "conf_std": "input", "kpt_conf_mean": "input",
    "flagellum_complete_frac": "input", "flagellum_len_mean": "param", "flagellum_len_std": "param",
    "flagellum_len_cv": "param",
}
for _name in ts.feature_names():
    KIN_TRUTH.setdefault(_name, "motion" if _name.startswith("body_") else "clean")
# Measurements that estimate one of the simulator's beat parameters, shown against it as well.
PHYSICAL = ("wave_freq_hz", "wave_wavelength_px", "wave_direction", "wave_speed_px_s",
            "wave_length_px", "inst_freq_hz", "inst_wavelength_px", "inst_direction")
# Beat columns: their truth is the same analysis on the exact flagellum.
BEAT_PREFIXES = ("wave_", "inst_")
# Drawn on log axes (sign dropped; direction has its own panel).
LOG_AXES = {"wave_wavelength_px", "wave_wavelength_ratio", "wave_speed_px_s", "inst_wavelength_px"}
# How an error is read: agree for signs and flags, rel for scales, abs otherwise.
AGREE = {"wave_direction", "wave_flow_direction", "wave_censored", "wave_bad_window",
         "inst_direction"}
REL = {"wave_freq_hz", "wave_freq_proximal_hz", "wave_freq_central_hz", "wave_freq_distal_hz",
       "inst_freq_hz", "inst_amp_mean", "inst_wavelength_px",
       "wave_wavelength_px", "wave_wavelength_ratio", "wave_speed_px_s", "wave_amp_mean",
       "wave_amp_integral", "wave_angular_amp", "wave_disp_amp_mean", "wave_disp_amp_max",
       "wave_disp_amp_tip", "wave_length_px", "flagellum_len_mean", "body_len_mean",
       "shape_extent_mean", "shape_pc1_mean", "shape_pc2_mean", "area_mean", "box_w", "box_h"}


def pose_config(folder: Path):
    """A project config with the 8-node skeleton, for the trajectory measurements."""
    from trackanno.config import load_config
    from trackanno.project import create_project, write_config

    names = ["Head", "Base"] + [f"Flag{i}" for i in range(1, 6)] + ["Tip"]
    existing = folder / "validation_project" / "config.yaml"
    if existing.exists():
        return load_config(existing)
    path = create_project(folder / "validation_project")
    write_config(path, {"detection.task": "pose"},
                 keypoints={"nodes": names,
                            "edges": [[names[i], names[i + 1]] for i in range(len(names) - 1)]})
    return load_config(path)


def motion_track(clean: pd.DataFrame, state: dict) -> pd.DataFrame:
    """The clean track with box centre, Head and Base on the simulator's exact body centre and axis."""
    out = clean.copy()
    cx, cy = state["centre"][:, 0], state["centre"][:, 1]
    ux, uy = np.cos(state["orientation"]), np.sin(state["orientation"])
    half = state["body_len_px"] / 2.0
    out["x1"], out["x2"], out["y1"], out["y2"] = cx - 1, cx + 1, cy - 1, cy + 1
    out["kp_x_0"], out["kp_y_0"] = cx - half * ux, cy - half * uy
    out["kp_x_1"], out["kp_y_1"] = cx + half * ux, cy + half * uy
    return out


def degrade(clean: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Jitter, dropped frames, dropped flagellum keypoints and random detector confidences."""
    rng = np.random.default_rng(seed + 1)
    noisy = add_noise(clean, NOISE_BODY_PX, NOISE_FLAGELLUM_PX, seed)
    jitter = rng.normal(0.0, NOISE_BODY_PX, (len(noisy), 2))
    for i, (lo_col, hi_col) in enumerate((("x1", "x2"), ("y1", "y2"))):
        noisy[lo_col] = noisy[lo_col] + jitter[:, i]
        noisy[hi_col] = noisy[hi_col] + jitter[:, i]
    noisy["conf"] = rng.uniform(0.4, 1.0, len(noisy))
    for k in range(N_KEYPOINTS):
        noisy[f"kp_c_{k}"] = rng.uniform(0.3, 1.0, len(noisy))
    for k in range(2, N_KEYPOINTS):
        gone = rng.random(len(noisy)) < KEYPOINT_DROP
        for col in (f"kp_x_{k}", f"kp_y_{k}", f"kp_c_{k}"):
            noisy.loc[gone, col] = np.nan
    keep = rng.random(len(noisy)) >= FRAME_DROP
    keep[:2] = keep[-2:] = True
    return noisy[keep].reset_index(drop=True)


def input_truth(noisy: pd.DataFrame, lo: int, hi: int) -> dict:
    """What the degraded track holds over frames [lo, hi): presence, completeness, confidences."""
    rows = noisy[(noisy.frame >= lo) & (noisy.frame < hi)]
    chain = rows[[f"kp_x_{k}" for k in range(1, N_KEYPOINTS)]]
    complete = chain.notna().all(axis=1)
    kc = rows[[f"kp_c_{k}" for k in range(N_KEYPOINTS)]].to_numpy(float)
    return {
        "present_frac": len(rows) / (hi - lo),
        "flagellum_complete_frac": complete.sum() / (hi - lo),
        "conf_mean": rows.conf.mean(), "conf_std": rows.conf.std(ddof=0),
        "kpt_conf_mean": np.nanmean(kc),
    }


def flow_truth(state: dict, centres: np.ndarray, hop: int) -> np.ndarray:
    """``wave_flow`` of the dense, noise-free angle read at the window centres."""
    theta = state["theta"]
    s = state["s"]
    arc = np.broadcast_to(s * state["flagellum_len_px"], theta.shape)
    flow = ws.flow_series(theta, arc, np.ones(theta.shape[0], bool), ws.DEFAULT_FLOW_FRAMES)
    return ws._read_at(flow, centres, max(ws.DEFAULT_FLOW_FRAMES // 2, hop // 2))


def one_cell(sl, cfg, seed: int, frames: int, window: int, hop: int,
             pixel_size_um: float) -> tuple[list[dict], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    p = sl.sample_random_parasite(rng, (2000, 2000))
    p.n_flagellum_keypoints = 5
    if p.mode_switch_rate > 0:
        p.mode_schedule = sl.generate_mode_schedule(p, frames / FPS, rng)
    clean, state = simulate(sl, p, FPS, frames, pixel_size_um)
    noisy = degrade(clean, seed)
    series = {
        "clean": ws.track_spectral_series(clean, N_KEYPOINTS, FPS, window=window, hop=hop),
        "noisy": ws.track_spectral_series(noisy, N_KEYPOINTS, FPS, window=window, hop=hop,
                                          n_frames=frames, f_start=0),
    }
    kinematic = {
        name: ts.build_series(track, cfg, FPS, 0, frames)
        for name, track in (("clean", clean), ("noisy", noisy), ("motion", motion_track(clean, state)))
    }
    m = int(series["clean"].attrs.get("window", window))
    length = state["flagellum_len_px"]
    head_base = np.hypot(clean.kp_x_1 - clean.kp_x_0, clean.kp_y_1 - clean.kp_y_0).to_numpy(float)
    beats = {
        "clean": inst.track_beat_series(clean, N_KEYPOINTS, FPS, n_frames=frames, f_start=0),
        "noisy": inst.track_beat_series(noisy, N_KEYPOINTS, FPS, n_frames=frames, f_start=0),
        "exact": beat_truth(state["theta"], state["s"], length, head_base, FPS),
    }
    centres = series["clean"].index.to_numpy(int)
    flow = dict(zip(centres, flow_truth(state, centres, hop)))
    complete = np.zeros(frames, bool)
    chain = noisy[[f"kp_x_{k}" for k in range(N_KEYPOINTS)]].notna().all(axis=1).to_numpy()
    complete[noisy.frame.to_numpy(int)[chain]] = True
    rows = []
    for centre in centres:
        lo = int(centre) - m // 2
        hi = lo + m
        mode = active_mode(sl, state["params"], lo / FPS, (hi - 1) / FPS)
        if lo < 0 or hi > frames or mode is None:
            continue
        freq, wavelength_um = mode_params(sl, state["params"], mode)
        truth = wave_truth(state["theta"][lo:hi], state["s"], length, head_base[lo:hi], FPS)
        truth["wave_flow"] = flow[centre]
        truth["wave_flow_coherence"] = abs(flow[centre])
        truth["wave_flow_direction"] = np.sign(flow[centre]) if flow[centre] != 0 else np.nan
        truth["wave_interp_frac"] = 1.0 - complete[lo:hi].mean()
        row = {"seed": seed, "centre": int(centre), "mode": mode,
               "cycles": freq * m / FPS,
               "wavelength_segments": wavelength_um * state["ppum"] / (length / 6.0)}
        for key, value in truth.items():
            row["true_" + key] = value
        physical = physical_truth(mode, freq, wavelength_um * state["ppum"], length)
        for key, value in physical.items():
            row["phys_" + key] = value
        for key in ("freq_hz", "wavelength_px", "direction"):
            row["phys_inst_" + key] = physical["wave_" + key]
        for kind, frame in series.items():
            for column in ws.SERIES_COLUMNS:
                row[f"{kind}_{column}"] = float(frame.loc[centre, column])
        c_lo = int(centre) - CLIP_FRAMES // 2
        if 0 <= c_lo and c_lo + CLIP_FRAMES <= frames:
            for kind, frame in kinematic.items():
                values = ts.aggregate_window(frame, c_lo, c_lo + CLIP_FRAMES, FPS, cfg)
                for feature in ts.feature_names():
                    row[f"kin_{kind}_{feature}"] = float(values.get(feature, np.nan))
            for key, value in input_truth(noisy, c_lo, c_lo + CLIP_FRAMES).items():
                row["kin_input_" + key] = value
            row["kin_param_flagellum_len_mean"] = length
            row["kin_param_flagellum_len_std"] = 0.0
            row["kin_param_flagellum_len_cv"] = 0.0
            for kind, beat in beats.items():
                prefix = "true_" if kind == "exact" else f"{kind}_"
                for key, value in inst.window_values(beat, c_lo, c_lo + CLIP_FRAMES).items():
                    row[prefix + key] = value
        rows.append(row)
    # Frequency stability: the same spread statistics over the true per-window frequencies.
    cell = pd.DataFrame(rows)
    if len(cell):
        true_f = np.where(cell.true_wave_censored == 1.0, np.nan, cell.get("true_wave_freq_hz", np.nan))
        drift = ws._local_spread(np.asarray(true_f, float), max(1, m // hop))
        good = true_f[np.isfinite(true_f)]
        sd = float(np.std(good, ddof=1)) if good.size >= 3 else np.nan
        for row, d in zip(rows, drift):
            row["true_wave_freq_drift_hz"] = d
            row["true_wave_freq_track_sd_hz"] = sd
    return rows, frame_rows(sl, state, seed, frames, beats, series["noisy"])


def frame_rows(sl, state: dict, seed: int, frames: int, beats: dict,
               windows: pd.DataFrame) -> pd.DataFrame:
    """Per frame: the simulator's beat, the per-frame series and the windowed frequency."""
    out = pd.DataFrame({"seed": seed, "frame": np.arange(frames), "mode": "blend",
                        "phys_f": np.nan, "phys_d": np.nan, "phys_lam": np.nan,
                        "length_px": state["flagellum_len_px"]})
    for i in range(frames):
        a, _b, blend = sl._get_active_mode_blend(state["params"], i / FPS)
        if blend != 0.0:
            continue
        out.loc[i, "mode"] = a
        if a in ("tip_to_base", "base_to_tip"):
            f, lam_um = mode_params(sl, state["params"], a)
            out.loc[i, ["phys_f", "phys_d", "phys_lam"]] = (
                f, -1.0 if a == "tip_to_base" else 1.0, lam_um * state["ppum"])
    centres = windows.index.to_numpy(float)
    out["windowed_freq_hz"] = np.interp(out.frame, centres, windows["wave_freq_hz"].to_numpy(float),
                                        left=np.nan, right=np.nan)
    out["windowed_direction"] = np.sign(np.interp(
        out.frame, centres, windows["wave_direction"].to_numpy(float), left=np.nan, right=np.nan))
    for kind, beat in beats.items():
        for col in inst.SERIES_COLUMNS:
            out[f"{kind}_{col}"] = beat[col].to_numpy(float)
    return out


def truth_column(df: pd.DataFrame, feature: str) -> tuple[pd.Series | None, str]:
    """The truth for ``feature`` (``phys:`` for a simulator parameter) and the name of its source."""
    if feature.startswith("phys:"):
        return df.get("phys_" + feature[5:]), "simulator parameter"
    if feature.startswith(BEAT_PREFIXES):
        source = WAVE_TRUTH.get(feature, "dense")
        if source == "clean":
            return df.get("clean_" + feature), "noise-free keypoints"
        if source == "behaviour":
            return df.get("true_wave_amp_mean"), "true a0 (no true value)"
        return df.get("true_" + feature), {"dense": "exact flagellum",
                                           "input": "dropouts fed in"}[source]
    source = KIN_TRUTH[feature]
    label = {"motion": "simulator motion", "param": "simulator", "clean": "noise-free keypoints",
             "input": "what was fed in"}[source]
    return df.get(f"kin_{source}_{feature}"), label


def measured_column(df: pd.DataFrame, feature: str) -> pd.Series | None:
    feature = feature.removeprefix("phys:")
    return df.get(("noisy_" if feature.startswith(BEAT_PREFIXES) else "kin_noisy_") + feature)


def error_row(df: pd.DataFrame, feature: str) -> dict:
    want, source = truth_column(df, feature)
    got = measured_column(df, feature)
    kind = feature.removeprefix("phys:")
    if want is None or got is None:
        return {"feature": feature, "truth": source, "n": 0}
    both = want.notna() & got.notna()
    out = {"feature": feature, "truth": source, "n": int(both.sum()),
           "missing": float(got[want.notna()].isna().mean()) if want.notna().any() else np.nan}
    if WAVE_TRUTH.get(feature) == "behaviour":
        rho = pd.Series(got[both]).rank().corr(pd.Series(want[both]).rank())
        out.update(kind="rank corr", median=float(rho), p90=np.nan)
        return out
    if kind in AGREE:
        out.update(kind="agree", median=float((got[both] == want[both]).mean()), p90=np.nan)
        return out
    err = (got / want - 1) if kind in REL else (got - want)
    err = err[both].replace([np.inf, -np.inf], np.nan).dropna()
    out.update(kind="rel" if kind in REL else "abs", median=float(err.median()),
               p90=float(err.abs().quantile(0.9)))
    return out


def resolvable(df: pd.DataFrame) -> pd.Series:
    return (df["mode"] != "static") & (df.cycles >= 2.5) & (df.wavelength_segments >= 2.5)


def report(df: pd.DataFrame) -> str:
    beat = resolvable(df)
    lines = [f"{len(df)} windows from {df.seed.nunique()} random cells; {int(beat.sum())} of "
             f"{int((df['mode'] != 'static').sum())} beating windows are resolvable "
             "(>= 2.5 cycles per window and a wavelength >= 2.5 segments).", "",
             f"Measured with jitter ({NOISE_BODY_PX} px body, {NOISE_FLAGELLUM_PX} px flagellum), "
             f"{FRAME_DROP:.0%} dropped frames and {KEYPOINT_DROP:.1%} dropped flagellum keypoints. Errors: rel = measured / true - 1, abs = measured - true, agree = "
             "share equal.", ""]
    wave = [c for c in ws.SERIES_COLUMNS]
    for label, rows, cols in (
            ("Waveform, resolvable beats", beat, wave),
            ("Waveform against the simulator's parameters, resolvable beats", beat,
             [f"phys:{c}" for c in PHYSICAL]),
            (f"Beat per frame, summarised over {CLIP_FRAMES}-frame clips, resolvable beats", beat,
             list(inst.CLIP_COLUMNS)),
            ("Waveform, unresolvable beats", (df["mode"] != "static") & ~beat, wave),
            ("Waveform, static cells", df["mode"] == "static", wave),
            (f"Trajectory, body, shape and quality ({CLIP_FRAMES}-frame clips)",
             df.get("kin_noisy_speed_mean", pd.Series(dtype=float)).notna() if "kin_noisy_speed_mean" in df
             else df.index == -1, list(ts.feature_names()))):
        lines += [f"## {label}", "", "| feature | truth | error | median | p90 abs | n | missing |",
                  "|---|---|---|---|---|---|---|"]
        for col in cols:
            r = error_row(df[rows], col)
            if not r.get("n"):
                lines.append(f"| {col} | {r['truth']} | | | | 0 | |")
                continue
            p90 = "" if np.isnan(r["p90"]) else f"{r['p90']:.3f}"
            lines.append(f"| {col} | {r['truth']} | {r['kind']} | {r['median']:.3f} | {p90} | "
                         f"{r['n']} | {r['missing']:.2f} |")
        lines.append("")
    return "\n".join(lines)


def figures(df: pd.DataFrame, out: Path) -> None:
    """One scatter per column against its truth, grouped by where the truth comes from."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, surface = "#0b0b0b", "#7a7974", "#e4e3df", "#fcfcfb"
    blue, grey = "#2a78d6", "#b5b3ad"
    plt.rcParams.update({"font.size": 7, "axes.edgecolor": muted, "xtick.color": muted,
                         "ytick.color": muted, "axes.linewidth": 0.6})
    beat = resolvable(df)
    groups = {
        "waveform": list(ws.SERIES_COLUMNS),
        "physical": [f"phys:{c}" for c in PHYSICAL],
        "beat_clips": list(inst.CLIP_COLUMNS),
        "body": [c for c in ts.feature_names() if KIN_TRUTH[c] == "motion"],
        "box_shape": [c for c in ts.feature_names() if KIN_TRUTH[c] in ("clean", "param")],
        "inputs": [c for c in ts.feature_names() if KIN_TRUTH[c] == "input"],
    }
    rng = np.random.default_rng(0)
    for name, cols in groups.items():
        n_cols = 6
        n_rows = int(np.ceil(len(cols) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.25 * n_rows),
                                 facecolor=surface, squeeze=False)
        for ax in axes.flat[len(cols):]:
            ax.axis("off")
        for ax, col in zip(axes.flat, cols):
            ax.set_facecolor(surface)
            want, source = truth_column(df, col)
            got = measured_column(df, col)
            if want is None or got is None:
                ax.axis("off")
                continue
            both = want.notna() & got.notna()
            x, y = want[both].to_numpy(float), got[both].to_numpy(float)
            if col.removeprefix("phys:") in AGREE:     # flags and signs: spread them so the counts show
                x = x + rng.uniform(-0.25, 0.25, x.size)
                y = y + rng.uniform(-0.25, 0.25, y.size)
            wave = col.removeprefix("phys:").startswith(BEAT_PREFIXES)
            focus = beat[both].to_numpy() if wave else np.ones(x.size, bool)
            behaviour = WAVE_TRUTH.get(col) == "behaviour"
            if behaviour:
                y = np.log10(np.maximum(y, 1e-3))
            logged = col.removeprefix("phys:") in LOG_AXES
            if logged:                            # 2 pi / slope: a flat phase gives a huge value
                x, y = np.log10(np.maximum(np.abs(x), 1e-3)), np.log10(np.maximum(np.abs(y), 1e-3))
                ax.set_xlabel("log10 |value|", fontsize=6, color=muted)
            ax.scatter(x[~focus], y[~focus], s=2, color=grey, alpha=0.35, lw=0, zorder=2)
            ax.scatter(x[focus], y[focus], s=2, color=blue, alpha=0.35, lw=0, zorder=3)
            if x.size and behaviour:
                ax.set_ylabel("log10", fontsize=6, color=muted)
            elif x.size:
                lo, hi = np.nanpercentile(np.r_[x[focus], y[focus]] if focus.any() else np.r_[x, y], [2, 98])
                pad = 0.1 * (hi - lo if hi > lo else 1.0)
                lo, hi = lo - pad, hi + pad
                ax.plot([lo, hi], [lo, hi], color=muted, lw=0.6, zorder=1)
                ax.set_xlim(lo, hi)
                ax.set_ylim(lo, hi)
            ax.grid(color=grid, lw=0.5, zorder=0)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            ax.set_title(col.removeprefix("phys:"), fontsize=7, color=ink, pad=3)
            r = error_row(df[beat] if wave else df, col)
            if r.get("kind") == "agree":
                note = f"{r['median']:.0%} agree"
            elif r.get("kind") == "rank corr":
                note = f"rank corr {r['median']:.2f}"
            else:
                note = f"p90 {r.get('p90', np.nan):.2g}" + (" rel" if r.get("kind") == "rel" else "")
            ax.text(0.03, 0.97, f"vs {source}\n{note}", transform=ax.transAxes, ha="left", va="top",
                    fontsize=5.5, color=muted)
        fig.supxlabel("true", fontsize=7, color=muted)
        fig.supylabel("measured", fontsize=7, color=muted)
        fig.tight_layout()
        fig.savefig(out / f"scatter_{name}.png", dpi=160, facecolor=surface)
        plt.close(fig)


SETTLE_FRAMES = 10


def steady_frames(frames: pd.DataFrame) -> pd.Series:
    """Frames inside one beat type whose wave spans at least 2.5 keypoint segments."""
    return np.isfinite(frames.phys_f) & (frames.phys_lam / (frames.length_px / 6.0) >= 2.5)


def switch_response(frames: pd.DataFrame, column: str) -> pd.DataFrame:
    """Frames from the end of each switch until ``column`` stays within 10% of the new beat."""
    out = []
    for seed, g in frames.groupby("seed"):
        mode = g["mode"].to_numpy()
        f = g[column].to_numpy(float)
        new_f = g["phys_f"].to_numpy(float)
        for i in range(1, len(g)):
            if mode[i - 1] != "blend" or mode[i] not in ("tip_to_base", "base_to_tip"):
                continue
            before = np.flatnonzero(mode[:i] != "blend")
            if before.size == 0 or mode[before[-1]] == mode[i]:
                continue
            ok = np.abs(f / new_f[i] - 1) < 0.1
            settle = next((k - i for k in range(before[-1] + 1, len(f) - SETTLE_FRAMES)
                           if ok[k:k + SETTLE_FRAMES].all()), np.nan)
            out.append({"seed": seed, "frame": i, "settle": settle})
    return pd.DataFrame(out, columns=["seed", "frame", "settle"])


def frame_report(frames: pd.DataFrame) -> str:
    steady = steady_frames(frames)
    confident = frames.noisy_inst_excess >= inst.CONFIDENT_EXCESS
    n = int(steady.sum())
    lines = [f"## Beat per frame ({n} frames inside one beat type with a resolvable wave, "
             f"{frames.seed.nunique()} cells)", "",
             "Against the simulator's beat at that frame. Windowed: the window centred on the "
             "frame. Shares count frames without an estimate as wrong.", "",
             "| estimate | frames | with a value | frequency within 10% | within 5% | "
             "median rel error | direction right |", "|---|---|---|---|---|---|---|"]
    estimates = (
        ("per frame, clean", "clean_inst_freq_hz", "clean_inst_direction", steady),
        ("per frame, jitter", "noisy_inst_freq_hz", "noisy_inst_direction", steady),
        (f"per frame, jitter, inst_excess >= {inst.CONFIDENT_EXCESS:g}", "noisy_inst_freq_hz",
         "noisy_inst_direction", steady & confident),
        ("windowed, jitter", "windowed_freq_hz", "windowed_direction", steady),
    )
    for label, f_col, d_col, rows in estimates:
        e = (frames[f_col] / frames.phys_f - 1)[rows]
        right = (frames[d_col] == frames.phys_d)[rows]
        lines.append(f"| {label} | {int(rows.sum())} | {e.notna().mean():.2f} | "
                     f"{(e.abs() < 0.1).mean():.3f} | {(e.abs() < 0.05).mean():.3f} | "
                     f"{e.median():+.3f} | {right.mean():.3f} |")
    lines += ["", f"Per-frame columns with jitter against the same readout of the exact flagellum "
              f"({n} frames):", "", "| column | error | median | p90 abs |", "|---|---|---|---|"]
    for col in inst.SERIES_COLUMNS:
        if WAVE_TRUTH.get(col) == "behaviour":       # no true value
            continue
        got, want = frames["noisy_" + col][steady], frames["exact_" + col][steady]
        if col in AGREE:
            lines.append(f"| {col} | agree | {(got == want).mean():.3f} | |")
            continue
        err = ((got / want - 1) if col in REL else (got - want)).replace([np.inf, -np.inf], np.nan)
        err = err.dropna()
        kind = "rel" if col in REL else "abs"
        lines.append(f"| {col} | {kind} | {err.median():+.3f} | {err.abs().quantile(0.9):.3f} |")
    lines += ["", f"Switches between beat types: frames from the end of the simulator's blend until "
              f"the frequency stays within 10% of the new beat for {SETTLE_FRAMES} frames "
              "(negative: already during the blend).", "",
              "| estimate | switches | settled | median frames |", "|---|---|---|---|"]
    for label, col in (("per frame, jitter", "noisy_inst_freq_hz"),
                       ("windowed, jitter", "windowed_freq_hz")):
        sw = switch_response(frames, col)
        lines.append(f"| {label} | {len(sw)} | {sw.settle.notna().mean():.2f} | "
                     f"{sw.settle.median():.1f} |")
    return "\n".join(lines) + "\n"


def frame_figures(frames: pd.DataFrame, out: Path) -> None:
    """Per-frame scatter against the truth, and one switching cell over time."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, surface = "#0b0b0b", "#7a7974", "#e4e3df", "#fcfcfb"
    blue, orange = "#2a78d6", "#eb6834"
    plt.rcParams.update({"font.size": 7, "axes.edgecolor": muted, "xtick.color": muted,
                         "ytick.color": muted, "axes.linewidth": 0.6})
    rng = np.random.default_rng(0)
    steady = frames[steady_frames(frames)]
    if len(steady) > 20000:
        steady = steady.iloc[rng.choice(len(steady), 20000, replace=False)]
    panels = [(c, "exact_" + c, "exact flagellum") for c in inst.SERIES_COLUMNS
              if WAVE_TRUTH.get(c) != "behaviour"]
    panels += [("inst_freq_hz", "phys_f", "simulator parameter"),
               ("inst_direction", "phys_d", "simulator parameter"),
               ("inst_wavelength_px", "phys_lam", "simulator parameter")]
    n_cols = 5
    n_rows = int(np.ceil(len(panels) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.25 * n_rows),
                             facecolor=surface, squeeze=False)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    for ax, (col, truth, source) in zip(axes.flat, panels):
        ax.set_facecolor(surface)
        x, y = steady[truth].to_numpy(float), steady["noisy_" + col].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if col in AGREE:
            note = f"{np.mean(x == y):.0%} agree"
            x, y = x + rng.uniform(-0.25, 0.25, x.size), y + rng.uniform(-0.25, 0.25, y.size)
        else:
            note = ""
        if col in LOG_AXES:
            x, y = np.log10(np.maximum(np.abs(x), 1e-3)), np.log10(np.maximum(np.abs(y), 1e-3))
            ax.set_xlabel("log10 |value|", fontsize=6, color=muted)
        ax.scatter(x, y, s=1.5, color=blue, alpha=0.2, lw=0, zorder=2)
        if col in AGREE:
            ax.plot([-1.6, 1.6], [-1.6, 1.6], color=muted, lw=0.6, zorder=1)
            ax.set_xlim(-1.6, 1.6)
            ax.set_ylim(-1.6, 1.6)
            ax.set_xticks([-1, 1])
            ax.set_yticks([-1, 1])
        elif x.size:
            lo, hi = np.nanpercentile(np.r_[x, y], [2, 98])
            pad = 0.1 * (hi - lo if hi > lo else 1.0)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=muted, lw=0.6, zorder=1)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
        ax.grid(color=grid, lw=0.5, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.set_title(col, fontsize=7, color=ink, pad=3)
        ax.text(0.03, 0.97, f"vs {source}\n{note}".rstrip(), transform=ax.transAxes, ha="left",
                va="top", fontsize=5.5, color=muted)
    fig.supxlabel("true", fontsize=7, color=muted)
    fig.supylabel("measured with jitter, per frame", fontsize=7, color=muted)
    fig.tight_layout()
    fig.savefig(out / "scatter_beat_frames.png", dpi=160, facecolor=surface)
    plt.close(fig)

    # The cell with the most switches between beat types.
    counts = frames.groupby("seed")["mode"].apply(
        lambda m: int(((m != m.shift()) & m.isin(["tip_to_base", "base_to_tip"])).sum()))
    if counts.max() < 2:
        return
    g = frames[frames.seed == counts.idxmax()]
    t = g.frame.to_numpy() / FPS
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.0, 3.4), sharex=True, facecolor=surface,
                                      gridspec_kw={"height_ratios": [3, 1]})
    for ax in (top, bottom):
        ax.set_facecolor(surface)
        ax.grid(color=grid, lw=0.5, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    top.plot(t, g.phys_f, color=ink, lw=1.2, label="simulator", zorder=3)
    top.plot(t, g.windowed_freq_hz, color=orange, lw=1.0, label="windowed (0.75 s)", zorder=2)
    top.plot(t, g.noisy_inst_freq_hz, color=blue, lw=0.8, label="per frame", zorder=4)
    top.set_ylabel("beat frequency (Hz)", color=muted)
    top.legend(frameon=False, fontsize=6.5, loc="upper right", ncol=3)
    bottom.plot(t, g.phys_d, color=ink, lw=1.2, zorder=3)
    bottom.plot(t, g.noisy_inst_direction, color=blue, lw=0.8, zorder=4)
    bottom.set_yticks([-1, 1], ["tip to base", "base to tip"])
    bottom.set_xlabel("time (s)", color=muted)
    fig.tight_layout()
    fig.savefig(out / "beat_switch.png", dpi=180, facecolor=surface)
    plt.close(fig)


def run_cell(job: tuple) -> tuple[list[dict], pd.DataFrame]:
    """:func:`one_cell` in a fresh process: the simulator and config are loaded here."""
    simulator, out, seed, frames, window, hop, pixel_size_um = job
    warnings.simplefilter("ignore")
    return one_cell(load_simulator(simulator), pose_config(out), seed, frames, window, hop,
                    pixel_size_um)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--simulator", required=True, help="path to synthetic_leishmania.py")
    ap.add_argument("--out", required=True, help="folder for windows.parquet, report.md and figures")
    ap.add_argument("--cells", type=int, default=250)
    ap.add_argument("--first-seed", type=int, default=0, help="seed of the first cell")
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--window", type=int, default=ws.DEFAULT_WINDOW)
    ap.add_argument("--hop", type=int, default=10)
    ap.add_argument("--pixel-size-um", type=float, default=0.325)
    ap.add_argument("--workers", type=int, default=1, help="cells simulated in parallel")
    args = ap.parse_args(argv)
    warnings.simplefilter("ignore")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pose_config(out)
    jobs = [(args.simulator, out, seed, args.frames, args.window, args.hop, args.pixel_size_um)
            for seed in range(args.first_seed, args.first_seed + args.cells)]
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(args.workers) as pool:
            cells = list(pool.map(run_cell, jobs))
    else:
        cells = [run_cell(job) for job in jobs]
    df = pd.DataFrame([row for rows, _ in cells for row in rows])
    per_frame = pd.concat([frames for _, frames in cells], ignore_index=True)
    df.to_parquet(out / "windows.parquet")
    per_frame.to_parquet(out / "frames.parquet")
    (out / "report.md").write_text(report(df) + "\n" + frame_report(per_frame), encoding="utf-8")
    figures(df, out)
    frame_figures(per_frame, out)
    print(f"{len(df)} windows, {len(per_frame)} frames -> {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
