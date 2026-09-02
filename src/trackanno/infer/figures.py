"""Figures over a transitions report, decoded bouts, and dense features.

The downstream-analysis notebook binds its knobs and palette here; every
function takes plain data plus a palette and returns the matplotlib figure.
Style (fonts, grid, spines) is inherited from the caller's rcParams.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .transitions import _pyplot

__all__ = [
    "GREY", "transition_graph", "jump_matrix", "survival",
    "lifetimes_occupancy", "ethogram", "trajectories", "usage",
    "feature_violins", "beat_map", "phase_cycles",
]

# Stretches the decode could not label.
GREY = "#c8c8c8"


def _steady(markov: dict, name: str, n_classes: int) -> float:
    """Steady-state share where the chain has one, else the measured share."""
    s = (markov.get("steady_state") or {}).get(name)
    if s and s.get("p") is not None:
        return float(s["p"])
    v = (markov.get("occupancy_observed") or {}).get(name)
    return float(v) if v is not None else 1.0 / max(n_classes, 1)


def transition_graph(markov: dict, palette: dict, *, group: str = ""):
    """The fitted chain as a ring of nodes sized by their share of time."""
    plt = _pyplot()
    from matplotlib.patches import Circle, FancyArrowPatch

    classes = list(markov["classes"])
    unit = str(markov["unit"])
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    n = len(classes)
    angle = {c: np.pi / 2 - 2 * np.pi * i / n for i, c in enumerate(classes)}
    pos = {c: np.array([np.cos(a), np.sin(a)]) for c, a in angle.items()}
    radius = {c: 0.10 + 0.28 * np.sqrt(_steady(markov, c, n)) for c in classes}

    for a, row in (markov.get("jump") or {}).items():
        for b, cell in row.items():
            p = cell.get("p")
            if not p or a not in pos or b not in pos:
                continue
            u, v = pos[a], pos[b]
            d = (v - u) / max(np.linalg.norm(v - u), 1e-9)
            start, end = u + d * radius[a] * 1.05, v - d * radius[b] * 1.15
            ax.add_patch(FancyArrowPatch(
                start, end, connectionstyle="arc3,rad=0.18",
                arrowstyle="-|>", mutation_scale=13,
                lw=0.8 + 5.5 * p, color=palette[a], alpha=0.85, zorder=1,
            ))
            q = ((markov.get("rates") or {}).get(a, {}).get(b) or {}).get("q")
            mid = (start + end) / 2
            off = np.array([-d[1], d[0]]) * 0.16
            ax.annotate(
                f"{p:.2f}" + (f"\n{q:.2g}/{unit[:-1]}" if q else ""),
                mid + off, ha="center", va="center", fontsize=8,
                color=palette[a],
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                          alpha=0.8),
            )

    for c in classes:
        ax.add_patch(Circle(pos[c], radius[c], color=palette[c], alpha=0.92,
                            zorder=3))
        life = markov["lifetime"][c]
        text = c
        if life.get("mean") is not None:
            text += f"\n{life['mean']:.2g} {unit[0]}"
        ax.annotate(text, pos[c], ha="center", va="center", zorder=4,
                    fontsize=9, color="white")
    xs = [pos[c][0] for c in classes]
    ys = [pos[c][1] for c in classes]
    r = max(radius.values())
    ax.set_xlim(min(xs) - r - 0.45, max(xs) + r + 0.45)
    ax.set_ylim(min(ys) - r - 0.35, max(ys) + r + 0.35)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"{group}: mode switching\nnode = time share, label = mean lifetime; "
        "edge = P(next | ended) with rate q",
    )
    return fig


def jump_matrix(group_report: dict, *, group: str = ""):
    """Jump probabilities as a matrix, each with the count it came from."""
    plt = _pyplot()
    classes = list(group_report["classes"])
    counts = group_report["transitions"]["counts"]
    prob = group_report["transitions"]["probabilities"]
    n = len(classes)
    P = np.full((n, n), np.nan)
    C = np.zeros((n, n), int)
    for i, a in enumerate(classes):
        for j, b in enumerate(classes):
            C[i, j] = int(counts.get(a, {}).get(b, 0))
            p = prob.get(a, {}).get(b)
            if p is not None:
                P[i, j] = p
    fig, ax = plt.subplots(figsize=(1.15 * n + 2.4, 1.05 * n + 1.6))
    ax.grid(False)
    im = ax.imshow(P, cmap="Blues", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.annotate("-", (j, i), ha="center", va="center", color="#999")
            elif np.isfinite(P[i, j]):
                dark = P[i, j] > 0.55
                ax.annotate(f"{P[i, j]:.2f}\nn={C[i, j]}", (j, i), ha="center",
                            va="center", fontsize=8,
                            color="white" if dark else "#1a3a5c")
    ax.set_xticks(range(n), classes, rotation=30, ha="right")
    ax.set_yticks(range(n), classes)
    ax.set_xlabel("next bout")
    ax.set_ylabel("this bout ended")
    ax.set_title(f"{group}: jump probabilities with their counts")
    fig.colorbar(im, ax=ax, shrink=0.8, label="P(next | ended)")
    return fig


def survival(markov: dict, palette: dict, *, group: str = ""):
    """Kaplan-Meier dwell survival per class beside its exponential fit."""
    plt = _pyplot()
    classes = list(markov["classes"])
    unit = str(markov["unit"])
    n = len(classes)
    cols = min(n, 3)
    rows_n = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(3.4 * cols, 2.9 * rows_n),
                             sharex=False, squeeze=False)
    for k, name in enumerate(classes):
        ax = axes[k // cols][k % cols]
        s = markov["survival"][name]
        # One mask for all three. Filtering each list on its own Nones shifts
        # them against each other, and a KM point is (t, p) or nothing.
        raw_t, raw_p, raw_f = s["t"], s["p"], s["fit"]
        keep = [i for i, (a, b) in enumerate(zip(raw_t, raw_p))
                if a is not None and b is not None]
        t = np.asarray([raw_t[i] for i in keep], float)
        p = np.asarray([raw_p[i] for i in keep], float)
        fitted = len(raw_f) == len(raw_t) and all(raw_f[i] is not None for i in keep)
        f = np.asarray([raw_f[i] for i in keep] if fitted else [], float)
        life = markov["lifetime"][name]
        if t.size:
            ax.plot(t, p, ".", ms=5, color=palette[name], label="measured (KM)")
            if f.size == t.size:
                ax.plot(t, f, "--", lw=1.4, color="#555", label="exponential fit")
            ax.set_yscale("log")
            ax.set_ylim(max(min(p[p > 0].min() if (p > 0).any() else 0.05, 0.05),
                            1e-3), 1.1)
        note = []
        if life.get("mean") is not None:
            se = life.get("se")
            note.append(f"mean {life['mean']:.3g}"
                        + (f" ± {se:.2g}" if se else "") + f" {unit[0]}")
        if life.get("fit_max_deviation") is not None:
            note.append(f"worst gap {life['fit_max_deviation']:.2f}")
        ax.set_title(name, color=palette[name])
        ax.annotate("\n".join(note), (0.97, 0.95), xycoords="axes fraction",
                    ha="right", va="top", fontsize=8)
        ax.set_xlabel(f"dwell time ({unit})")
        ax.set_ylabel("P(T > t)")
        if k == 0:
            ax.legend(loc="lower left")
    for k in range(n, rows_n * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle(f"{group}: how long each behaviour lasts")
    return fig


def lifetimes_occupancy(markov: dict, palette: dict, *, group: str = ""):
    """Lifetimes with bootstrap errors, and the model-adequacy comparison."""
    plt = _pyplot()
    classes = list(markov["classes"])
    unit = str(markov["unit"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.4))
    x = np.arange(len(classes))
    means = [markov["lifetime"][c].get("mean") or np.nan for c in classes]
    errs = [markov["lifetime"][c].get("se") or 0.0 for c in classes]
    ax1.bar(x, means, yerr=errs, capsize=3,
            color=[palette[c] for c in classes], alpha=0.9)
    for i, c in enumerate(classes):
        ax1.annotate(f"n={markov['lifetime'][c]['n_ends']}", (i, 0), (0, 4),
                     textcoords="offset points", ha="center", fontsize=8,
                     color="white")
    ax1.set_xticks(x, classes, rotation=20, ha="right")
    ax1.set_ylabel(f"mean lifetime ({unit})")
    ax1.set_title("lifetime (censoring-corrected)")

    steady = [_steady(markov, c, len(classes))
              if markov.get("steady_state") else np.nan for c in classes]
    occ = [(markov.get("occupancy_observed") or {}).get(c) or np.nan
           for c in classes]
    w = 0.38
    ax2.bar(x - w / 2, steady, w, label="fitted chain, long run",
            color=[palette[c] for c in classes], alpha=0.95)
    ax2.bar(x + w / 2, occ, w, label="observed share of time",
            color=[palette[c] for c in classes], alpha=0.45, hatch="//")
    ax2.set_xticks(x, classes, rotation=20, ha="right")
    ax2.set_ylabel("share of time")
    ax2.set_title("steady state vs what was measured")
    ax2.legend()
    fig.suptitle(f"{group}: the fitted chain, and whether to believe it")
    return fig


def ethogram(rows: pd.DataFrame, classes: list, palette: dict, *,
             fps: float | None = None, movie_id: str = "", max_tracks: int = 40):
    """Every track of one movie as a bar over time, coloured by behaviour.

    ``rows`` are the movie's bouts (label, track_id, f_start, f_end,
    n_frames); an empty label draws grey.
    """
    plt = _pyplot()
    order = (rows.groupby("track_id")["n_frames"].sum()
             .sort_values(ascending=False).index[:max_tracks])
    scale = (1.0 / fps) if fps else 1.0
    fig, ax = plt.subplots(figsize=(8.5, 0.28 * len(order) + 1.4))
    for y, track in enumerate(order):
        for r in rows[rows["track_id"] == track].itertuples():
            colour = palette.get(r.label, GREY) if r.label else GREY
            ax.barh(y, (r.f_end - r.f_start) * scale, left=r.f_start * scale,
                    height=0.72, color=colour,
                    edgecolor="white", linewidth=0.2)
    ax.set_yticks(range(len(order)), [f"t{t}" for t in order])
    ax.invert_yaxis()
    ax.grid(False)
    ax.set_xlabel("time (s)" if fps else "frame")
    ax.set_title(f"{movie_id}: what every animal was doing, and when")
    from matplotlib.patches import Rectangle
    handles = [Rectangle((0, 0), 1, 1, color=palette[c]) for c in classes]
    handles.append(Rectangle((0, 0), 1, 1, color=GREY))
    ax.legend(handles, classes + ["unlabelled"], loc="upper left",
              bbox_to_anchor=(1.005, 1))
    return fig


def trajectories(tracks: pd.DataFrame, rows: pd.DataFrame, classes: list,
                 palette: dict, *, movie_id: str = "",
                 min_track_frames: int = 25):
    """Centroid paths of one movie's tracks, coloured by decoded behaviour.

    ``tracks`` must carry frame, track_id, cx, cy (see
    :func:`trackanno.infer.motifs.track_paths`); ``rows`` the movie's bouts.
    """
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    for track_id, tdf in tracks.groupby("track_id"):
        if len(tdf) < min_track_frames:
            continue
        tdf = tdf.sort_values("frame")
        frames = tdf["frame"].to_numpy()
        ax.plot(tdf["cx"], tdf["cy"], color=GREY, lw=1.6, alpha=0.5,
                solid_capstyle="round", zorder=1)
        for r in rows[rows["track_id"] == track_id].itertuples():
            if not r.label:
                continue
            m = (frames >= r.f_start) & (frames < r.f_end)
            if m.sum() >= 2:
                ax.plot(tdf["cx"][m], tdf["cy"][m], color=palette[r.label],
                        lw=0.6, alpha=0.95, solid_capstyle="round", zorder=2)
        first = tdf.iloc[0]
        ax.plot(first["cx"], first["cy"], "o", ms=3, color="#444", zorder=3)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.grid(False)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=palette[c], lw=3) for c in classes]
    ax.legend(handles, classes, loc="upper left", bbox_to_anchor=(1.005, 1))
    ax.set_title(f"{movie_id}: where each behaviour happened\n"
                 "dot = track start, grey = unlabelled frames")
    return fig


def usage(labels: pd.Series, group_report: dict, palette: dict, *,
          group: str = ""):
    """Bout counts, share of the record, and median dwell per class."""
    plt = _pyplot()
    classes = list(group_report["classes"])
    markov = group_report["markov"]
    unit = str(markov["unit"])
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))
    counts = [int((labels == c).sum()) for c in classes]
    time_share = [(markov.get("occupancy_observed") or {}).get(c) or 0.0
                  for c in classes]
    med = [group_report["dwell"]["per_class"][c]
           ["seconds" if unit == "seconds" else "frames"].get("median") or np.nan
           for c in classes]
    for ax, values, title, label in zip(
        axes, (counts, time_share, med),
        ("bouts", "share of the record", "median dwell (complete bouts)"),
        ("count", "fraction of time", f"{unit}"),
    ):
        ax.bar(range(len(classes)), values, color=[palette[c] for c in classes],
               alpha=0.9)
        ax.set_xticks(range(len(classes)), classes, rotation=20, ha="right")
        ax.set_title(title)
        ax.set_ylabel(label)
    fig.suptitle(f"{group}: behaviour usage")
    return fig


def feature_violins(feats: pd.DataFrame, classes: list, palette: dict,
                    features: list):
    """Distribution of chosen feature columns per ``_behaviour`` class.

    ``features`` pairs a column name with its axis label; columns absent or
    empty are skipped.
    """
    plt = _pyplot()
    wanted = [(c, t) for c, t in features
              if c in feats.columns and feats[c].notna().any()]
    if not wanted:
        print("none of the showcased feature columns are in the table")
        return None
    cols = 3
    rows_n = int(np.ceil(len(wanted) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(3.3 * cols, 2.8 * rows_n),
                             squeeze=False)
    for k, (column, title) in enumerate(wanted):
        ax = axes[k // cols][k % cols]
        data, shown = [], []
        for c in classes:
            v = feats.loc[feats["_behaviour"] == c, column].to_numpy(float)
            v = v[np.isfinite(v)]
            if v.size >= 5:
                data.append(v)
                shown.append(c)
        if not data:
            ax.axis("off")
            continue
        parts = ax.violinplot(data, showmedians=True, widths=0.8)
        for body, c in zip(parts["bodies"], shown):
            body.set_facecolor(palette[c])
            body.set_alpha(0.6)
        for piece in ("cmedians", "cbars", "cmins", "cmaxes"):
            parts[piece].set_color("#555")
            parts[piece].set_linewidth(1.0)
        ax.set_xticks(range(1, len(shown) + 1), shown, rotation=20, ha="right")
        ax.set_title(title)
    for k in range(len(wanted), rows_n * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle("what the features measure inside each predicted behaviour")
    return fig


def beat_map(feats: pd.DataFrame, classes: list, palette: dict):
    """Beat frequency against amplitude, one dot per window."""
    plt = _pyplot()
    fcol, acol = "wave_freq_hz", "wave_amp_mean"
    if fcol not in feats.columns or not feats[fcol].notna().any():
        print("no waveform columns (needs detection.task: pose); skipped")
        return None
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for c in classes:
        rows = feats[feats["_behaviour"] == c]
        x = rows[fcol].to_numpy(float)
        y = rows[acol].to_numpy(float)
        good = np.isfinite(x) & np.isfinite(y)
        if good.sum():
            ax.scatter(x[good], y[good], s=12, alpha=0.55, lw=0,
                       color=palette[c], label=f"{c} ({int(good.sum())})")
    ax.set_xlabel("beat frequency (Hz)")
    ax.set_ylabel("beat amplitude (rad)")
    ax.set_title("the beat, per window, coloured by behaviour")
    ax.legend()
    return fig


def phase_cycles(results: dict, classes: list, flag_chain: list, *,
                 n_phase: int, fold: bool = True, xlim=None, ylim=None):
    """The phase-averaged beat cycle, one panel per class, shared limits.

    ``results`` maps each class to its
    :func:`trackanno.infer.motifs.phase_cycle` result.
    """
    plt = _pyplot()
    pts = np.concatenate(
        [m[:, flag_chain].reshape(-1, 2) for m, *_ in results.values()
         if m is not None] or [np.zeros((1, 2))])
    pts = pts[np.isfinite(pts).all(axis=1)]
    if not len(pts):
        pts = np.zeros((1, 2))
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    w = max(x1 - x0, 1.0)
    # The cycle is long and flat; a panel squashed to the data strip has no
    # room for its own labels, so the height gets a floor.
    h = max(y1 - y0, 0.35 * w, 1.0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    xlim = xlim or (cx - 0.58 * w, cx + 0.58 * w)
    ylim = ylim or (cy - 0.58 * h, cy + 0.58 * h)
    box = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])

    cols = min(len(classes), 3)
    rows_n = int(np.ceil(len(classes) / cols))
    cyc = plt.colormaps["twilight"]
    fig, axes = plt.subplots(rows_n, cols,
                             figsize=(3.4 * cols, (3.4 * box + 0.7) * rows_n),
                             squeeze=False)
    for k, name in enumerate(classes):
        ax = axes[k // cols][k % cols]
        ax.set_title(name)
        ax.grid(False)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_box_aspect(box)
        mean, n_bouts, _, n_folded = results[name]
        if mean is None:
            ax.annotate(
                f"nothing oscillating\n({n_bouts} bout(s) cleared the gate)",
                (0.5, 0.5), xycoords="axes fraction", ha="center", va="center",
                fontsize=8, color="0.45")
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        for b in range(n_phase):
            p = mean[b][flag_chain]
            ax.plot(p[:, 0], p[:, 1], "-", color=cyc(b / n_phase), lw=1.6)
        ax.plot(0, 0, "o", ms=4, color="0.25", zorder=5)
        note = f"{n_bouts} bouts" + (f", folded {n_bouts - n_folded}:{n_folded}"
                                     if fold else "")
        ax.annotate(note, (0.02, 0.97), xycoords="axes fraction", va="top",
                    fontsize=8, color="0.35")
        ax.set_xlabel("along body (px)")
        ax.set_ylabel("lateral (px)")
    for k in range(len(classes), rows_n * cols):
        axes[k // cols][k % cols].axis("off")
    return fig
