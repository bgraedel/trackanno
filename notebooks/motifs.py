"""Behavioural motifs over decoded bouts.

The machinery behind the downstream-analysis notebook: onset-aligned
instances and their median motif, the keypoint-MoSeq trajectory canvas, the
phase-averaged beat cycle with its oscillation gate, medoid election, and
window labelling. Everything here is parameterised; the notebook holds the
knobs and the figure layout.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.signal import hilbert, lfilter

from trackanno.infer.figures import _pyplot

__all__ = [
    "track_paths",
    "keypoint_window", "align_at_onset", "body_frame", "class_instances",
    "fold_instances", "beat_phase", "phase_cycle", "class_posterior",
    "elect_medoid", "label_windows", "get_limits", "plot_trajectories",
]

# An axis shorter than this is jitter, not a heading.
MIN_AXIS_PX = 4.0


# --------------------------------------------------------------------------- #
# poses out of a track table
# --------------------------------------------------------------------------- #
def track_paths(cfg, movie_id: str) -> pd.DataFrame | None:
    """The movie's track table with box-centre columns, or None without one."""
    path = cfg.movie_dir(str(movie_id)) / "tracks.parquet"
    if not path.exists():
        return None
    t = pd.read_parquet(path)
    t["cx"] = (t["x1"] + t["x2"]) / 2.0
    t["cy"] = (t["y1"] + t["y2"]) / 2.0
    return t



def keypoint_window(tdf, f_start: int, n_frames: int, n_keypoints: int) -> np.ndarray:
    """(n_frames, K, 2) keypoints on the frame grid, NaN where undetected."""
    kx = [f"kp_x_{k}" for k in range(n_keypoints)]
    ky = [f"kp_y_{k}" for k in range(n_keypoints)]
    win = np.full((n_frames, n_keypoints, 2), np.nan)
    sel = tdf[(tdf["frame"] >= f_start) & (tdf["frame"] < f_start + n_frames)]
    idx = sel["frame"].to_numpy(int) - int(f_start)
    win[idx, :, 0] = sel[kx].to_numpy(float)
    win[idx, :, 1] = sel[ky].to_numpy(float)
    return win


def align_at_onset(win: np.ndarray, onset: int) -> np.ndarray | None:
    """The kpms alignment: ONE rigid transform per instance, fixed at onset.

    Anchor is the Head-Base midpoint, heading the Base-to-Head axis, both at
    the first detected frame at or after the onset. Motion during the bout is
    kept, which is the point.
    """
    ok = np.isfinite(win[:, 0]).all(axis=1) & np.isfinite(win[:, 1]).all(axis=1)
    ok[:onset] = False
    if not ok.any():
        return None
    i0 = int(np.flatnonzero(ok)[0])
    head, base = win[i0, 0], win[i0, 1]
    if np.hypot(*(head - base)) < MIN_AXIS_PX:
        return None
    h = np.arctan2(head[1] - base[1], head[0] - base[0])
    c, s = np.cos(-h), np.sin(-h)
    rot = np.array([[c, -s], [s, c]])
    return (win - (head + base) / 2.0) @ rot.T


def body_frame(win: np.ndarray) -> np.ndarray:
    """(T, K, 2) poses rotated into the INSTANTANEOUS body frame.

    Anchored at Base with the Base-to-Head axis along +x per frame, so
    locomotion is removed and posture remains. Frames without a usable axis
    come back NaN.
    """
    head, base = win[:, 0], win[:, 1]
    ok = (np.isfinite(head).all(axis=1) & np.isfinite(base).all(axis=1)
          & (np.hypot(*(head - base).T) >= MIN_AXIS_PX))
    h = np.arctan2(head[:, 1] - base[:, 1], head[:, 0] - base[:, 0])
    c, s = np.cos(-h), np.sin(-h)
    rel = win - base[:, None, :]
    out = np.stack([rel[..., 0] * c[:, None] - rel[..., 1] * s[:, None],
                    rel[..., 0] * s[:, None] + rel[..., 1] * c[:, None]], axis=-1)
    out[~ok] = np.nan
    return out


# --------------------------------------------------------------------------- #
# onset-aligned instances and their folding
# --------------------------------------------------------------------------- #
def class_instances(
    rows: pd.DataFrame, tracks_for, fps_for, *, n_keypoints: int,
    window_mode: str = "auto", pre_s: float = 0.167, post_s: float = 0.5,
    num_samples: int = 20, max_instances: int = 150, seed: int = 0,
) -> tuple[np.ndarray | None, dict]:
    """Onset-aligned instances of one class on a common ``num_samples`` grid.

    ``rows`` holds one class's bouts (movie_id, track_id, f_start, n_frames);
    ``tracks_for``/``fps_for`` map a movie_id to its track table / frame rate.
    Returns ``(instances, meta)``: meta carries where the onset sits on the
    grid and a note for a caption. Frames past a bout's own end are ALWAYS
    masked out, so the behaviour that follows a short bout cannot leak into
    the median's tail.
    """
    if not len(rows):
        return None, {"onset_frac": 0.0, "note": ""}

    if window_mode == "relative":
        pre_s, post_s = 0.0, None
        note = "0→100% of each bout"
    elif window_mode == "auto":
        seconds = [r.n_frames / fps_for(r.movie_id) for r in rows.itertuples()
                   if fps_for(r.movie_id)]
        if not seconds:
            return None, {"onset_frac": 0.0, "note": ""}
        post_s = float(np.median(seconds))
        pre_s = 0.25 * post_s
        note = f"window auto -{pre_s:.2g}s→+{post_s:.2g}s (median dwell)"
    else:
        pre_s, post_s = float(pre_s), float(post_s)
        note = f"-{pre_s:g}s→+{post_s:g}s"

    out = []
    # Shuffled, so the max_instances cap samples the whole record instead of
    # keeping only the first movies.
    for r in rows.sample(frac=1.0, random_state=seed).itertuples():
        rate = fps_for(r.movie_id)
        if not rate or r.n_frames < 3:            # kpms min_duration
            continue
        t = tracks_for(str(r.movie_id))
        if t is None:
            continue
        tdf = t[t["track_id"] == r.track_id]
        if window_mode == "relative":
            span = int(r.n_frames)
            win = keypoint_window(tdf, int(r.f_start), span, n_keypoints)
            aligned = align_at_onset(win, 0)
            total = span
        else:
            pre = int(round(pre_s * rate))
            post = max(int(round(post_s * rate)), 3)
            win = keypoint_window(tdf, int(r.f_start) - pre, pre + post,
                                  n_keypoints)
            # The bout ends where it ends; whatever the track did afterwards
            # belongs to the next bout, not this motif.
            if r.n_frames < post:
                win[pre + int(r.n_frames):] = np.nan
            aligned = align_at_onset(win, pre)
            total = pre + post
        if aligned is None:
            continue
        picks = np.linspace(0, total - 1, num_samples).round().astype(int)
        out.append(aligned[picks])
        if len(out) >= max_instances:
            break
    onset_frac = 0.0 if window_mode == "relative" else pre_s / (pre_s + post_s)
    meta = {"onset_frac": float(onset_frac), "note": note}
    return (np.stack(out) if out else None), meta


def fold_instances(instances: np.ndarray, onset_at: int, *,
                   enabled: bool = True) -> tuple[np.ndarray, str]:
    """Reflect each instance onto one canonical side before the median.

    Chirality is the anchor's mean lateral drift after onset. The note's side
    split is how to read the result: near 50:50 with a clear curve is a
    genuinely two-sided behaviour shown as its one-sided prototype; near
    50:50 with a bend that is small against the travel is a straight class
    showing its own noise floor, folded one-sided.
    """
    if not enabled:
        return instances, ""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lat = np.nanmean(instances[:, onset_at:, :2, 1], axis=(1, 2))
    ok = np.isfinite(lat)
    if not ok.any():
        return instances, ""
    flip = ok & (lat < 0.0)
    folded = instances.copy()
    folded[flip, :, :, 1] *= -1.0
    return folded, f", folded {int((ok & ~flip).sum())}:{int(flip.sum())}"


# --------------------------------------------------------------------------- #
# the beat: oscillation gate, phase, cycle
# --------------------------------------------------------------------------- #
def beat_phase(lateral: np.ndarray, rng, *, min_cycle_frames: int = 24,
               min_osc_px: float = 1.0, n_null: int = 200) -> np.ndarray | None:
    """Hilbert phase of one bout's reference-node lateral trace, or None.

    The gate is a rhythm test against RED noise, not white: tracking jitter
    and slow wobble are autocorrelated, their power piles up at low
    frequency, and a shuffle null passes them. The trace's own lag-1
    autocorrelation sets an AR(1) background; the statistic is the largest
    peak-to-background ratio over frequencies with at least 2 full cycles in
    the bout, and it must beat the 99th percentile of the same statistic on
    ``n_null`` AR(1) surrogates matched to the trace (p < 0.01), at an
    amplitude of at least ``min_osc_px``. The phase is then the Hilbert
    phase of the band around the winning frequency, so drift under a real
    beat does not bend its phase.
    """
    good = np.isfinite(lateral)
    if good.sum() < min_cycle_frames or good.mean() < 0.8:
        return None
    i = np.arange(lateral.size)
    filled = np.interp(i, i[good], lateral[good])
    filled = filled - np.polyval(np.polyfit(i, filled, 1), i)
    n = filled.size
    spec = np.abs(np.fft.rfft(filled)) ** 2
    var = float(filled.var())
    if spec.size <= 3 or var <= 0.0:
        return None
    d = filled - filled.mean()
    alpha = float(np.clip((d[1:] * d[:-1]).sum() / max((d ** 2).sum(), 1e-12),
                          0.0, 0.99))
    background = (1.0 - alpha ** 2) / (
        1.0 + alpha ** 2
        - 2.0 * alpha * np.cos(2.0 * np.pi * np.arange(spec.size) / n))
    ratio = spec[2:] / background[2:]
    k = int(np.argmax(ratio) + 2)
    if 2.0 * np.sqrt(spec[k]) / n < min_osc_px:
        return None
    e = rng.normal(0.0, np.sqrt(var * (1.0 - alpha ** 2)), (n_null, n + 50))
    surrogate = lfilter([1.0], [1.0, -alpha], e, axis=1)[:, 50:]
    sp = np.abs(np.fft.rfft(surrogate - surrogate.mean(1, keepdims=True),
                            axis=1)) ** 2
    null = (sp[:, 2:] / background[2:]).max(axis=1)
    if float(ratio.max()) <= np.percentile(null, 99):
        return None
    fx = np.fft.rfft(filled)
    keep = np.zeros(fx.size, dtype=bool)
    keep[max(2, k // 2):int(np.ceil(1.5 * k)) + 1] = True
    band = np.fft.irfft(np.where(keep, fx, 0.0), n)
    phase = np.angle(hilbert(band))
    phase[~good] = np.nan
    return np.mod(phase, 2 * np.pi)


def phase_cycle(
    rows: pd.DataFrame, tracks_for, *, n_keypoints: int, ref_node: int,
    flag_chain: list[int], n_phase: int = 12, min_cycle_frames: int = 24,
    min_osc_px: float = 1.0, n_null: int = 200, fold: bool = True,
    seed: int = 0,
) -> tuple[np.ndarray | None, int, int, int]:
    """Mean pose per phase bin, pooled over one class's bouts.

    The WAVE is side-immune by itself: mirroring a bout negates its reference
    trace, which shifts the Hilbert phase by pi, and half a period later the
    wave is its own mirror. A STATIC bend is not: a population bent to both
    sides averages its mean curvature toward straight. So under ``fold`` each
    bout is folded by its mean bend BEFORE the phase is computed.

    Returns ``(mean, n_bouts, n_frames, n_folded)``; mean is None when fewer
    than 3 bouts (or too few frames) clear the oscillation gate.
    """
    rng = np.random.default_rng(seed)
    sums = np.zeros((n_phase, n_keypoints, 2))
    counts = np.zeros((n_phase, n_keypoints, 2))
    n_bouts = n_frames = n_folded = 0
    for r in rows.itertuples():
        tracks = tracks_for(str(r.movie_id))
        if tracks is None:
            continue
        win = keypoint_window(tracks[tracks["track_id"] == r.track_id],
                              int(r.f_start), int(r.n_frames), n_keypoints)
        poses = body_frame(win)
        flipped = False
        if fold:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                bend = np.nanmean(poses[:, flag_chain, 1])
            if np.isfinite(bend) and bend < 0.0:
                poses = poses * np.array([1.0, -1.0])
                flipped = True
        phase = beat_phase(poses[:, ref_node, 1], rng,
                           min_cycle_frames=min_cycle_frames,
                           min_osc_px=min_osc_px, n_null=n_null)
        if phase is None:
            continue
        # Counted only for bouts the gate kept, so a caption's split sums to
        # the bouts actually in the average.
        n_folded += int(flipped)
        bins = np.floor(phase / (2 * np.pi) * n_phase).astype(float)
        n_bouts += 1
        for b in range(n_phase):
            member = bins == b
            if not member.any():
                continue
            block = poses[member]
            finite = np.isfinite(block)
            sums[b] += np.where(finite, block, 0.0).sum(axis=0)
            counts[b] += finite.sum(axis=0)
            n_frames += int(member.sum())
    # One noisy bout is not a beat: a phase estimate that only a couple of
    # bouts support says more about the Hilbert transform than the behaviour.
    if n_bouts < 3 or n_frames < 5 * n_phase:
        return None, n_bouts, n_frames, n_folded
    with np.errstate(invalid="ignore"):
        mean = np.where(counts > 0, sums / np.where(counts > 0, counts, 1), np.nan)
    return mean, n_bouts, n_frames, n_folded


# --------------------------------------------------------------------------- #
# the medoid: the class's most ordinary window
# --------------------------------------------------------------------------- #
def class_posterior(rows: pd.DataFrame, name: str, cfg, group: str) -> np.ndarray | None:
    """p(class | window) from the saved model, or None when it cannot be had."""
    try:
        import joblib

        from trackanno.learn.suggest import MODEL_FILE

        payload = joblib.load(cfg.paths.work_dir / MODEL_FILE)
        model = (payload.get("models") or {}).get(group)
        cols = list(payload.get("columns") or [])
        built = (int(rows["_feature_version"].iloc[0])
                 if "_feature_version" in rows.columns else None)
        if (model is None or not cols or built is None
                or int(payload.get("feature_version", -1)) != built
                or any(c not in rows.columns for c in cols)):
            return None
        proba = np.asarray(model.predict_proba(rows[cols].to_numpy(np.float32)),
                           dtype=float)
        names = [str(c) for c in model.classes_]
        return proba[:, names.index(name)] if name in names else None
    except Exception:
        return None


def elect_medoid(rows: pd.DataFrame, *, space: str = "model",
                 shown: tuple = (), pick: int = 0) -> dict | None:
    """The row nearest the class's own feature centre, by robust z-scores.

    ``space`` is "model" (every column the classifier reads except the cnn
    identity block, which fingerprints INDIVIDUALS), "model+cnn", or "shown"
    (the ``shown`` columns only). ``pick`` = 0 is the medoid, 1 the
    runner-up, and so on. None when the rows cannot support an election.
    """
    if space == "shown":
        wanted = list(shown)
    else:
        from trackanno.measure.clipfeatures import feature_columns, group_of

        dense_meta = {"f_end", "n_present", "skip_reason", "_behaviour"}
        keep_cnn = space == "model+cnn"
        wanted = [c for c in feature_columns(rows)
                  if (keep_cnn or group_of(c) != "cnn") and c not in dense_meta
                  and rows[c].dtype.kind in "fiu"]
    cols = [c for c in wanted
            if c in rows.columns and rows[c].notna().sum() >= max(5, len(rows) // 4)]
    if len(rows) < 3 or not cols:
        return None
    X = rows[cols].to_numpy(float)
    med = np.nanmedian(X, axis=0)
    mad = np.nanmedian(np.abs(X - med), axis=0)
    Z = (X - med) / np.where(mad > 1e-9, mad, 1.0)
    dist = np.nansum(Z ** 2, axis=1) / np.maximum(np.isfinite(Z).sum(axis=1), 1)
    order = np.argsort(dist, kind="stable")
    pick = min(max(int(pick), 0), len(order) - 1)
    row = dict(rows.iloc[int(order[pick])])
    row["_how"] = "medoid" if pick == 0 else f"medoid runner-up #{pick}"
    return row


def label_windows(feats: pd.DataFrame, bouts: pd.DataFrame) -> pd.DataFrame:
    """``feats`` windows labelled by the bout covering each window's centre.

    ``bouts`` carries movie_id, track_id, f_start, f_end and ``label``; rows
    whose centre no bout covers are dropped.
    """
    centre = ((feats["f_start"] + feats["f_end"]) // 2).to_numpy()
    spans = {}
    for key, rows in bouts.groupby(["movie_id", "track_id"]):
        # searchsorted below needs ascending starts, and groupby preserves the
        # table's own row order, which a bout table written per class does not.
        rows = rows.sort_values("f_start")
        spans[(str(key[0]), int(key[1]))] = (
            rows["f_start"].to_numpy(), rows["f_end"].to_numpy(),
            rows["label"].to_numpy(object),
        )
    labels = np.full(len(feats), "", object)
    movie_ids = feats["movie_id"].astype(str).to_numpy()
    track_ids = feats["track_id"].to_numpy()
    for i in range(len(feats)):
        entry = spans.get((movie_ids[i], int(track_ids[i])))
        if entry is None:
            continue
        starts, ends, names = entry
        j = int(np.searchsorted(starts, centre[i], side="right")) - 1
        if j >= 0 and centre[i] < ends[j]:
            labels[i] = names[j] or ""
    feats = feats.assign(_behaviour=labels)
    return feats[feats["_behaviour"] != ""]


# --------------------------------------------------------------------------- #
# the kpms trajectory canvas
# --------------------------------------------------------------------------- #
def get_limits(coordinates, pctl: float = 1, left: float = 0.2,
               right: float = 0.2, top: float = 0.2, bottom: float = 0.2
               ) -> np.ndarray:
    """kpms get_limits: percentile bounds padded per side, [[xmin,ymin],[xmax,ymax]]."""
    X = coordinates.reshape(-1, 2)
    xmin, ymin = np.nanpercentile(X, pctl, axis=0)
    xmax, ymax = np.nanpercentile(X, 100 - pctl, axis=0)
    width, height = xmax - xmin, ymax - ymin
    xmin -= width * left
    xmax += width * right
    ymin -= height * bottom
    ymax += height * top
    return np.array([[xmin, ymin], [xmax, ymax]])


def _scale_bar(ax, ref_width, xmin, ymin, xmax, ymax, z):
    """A round-length bar at the canvas corner, standing in for the axes."""
    raw = 0.3 * ref_width
    exp = 10.0 ** np.floor(np.log10(raw))
    nice = next(m * exp for m in (5.0, 2.0, 1.0) if m * exp <= raw)
    bx, by = xmin + 0.02 * (xmax - xmin), ymin + 0.03 * (ymax - ymin)
    ax.plot([bx, bx + nice], [by, by], "-", color="0.3", lw=1.3,
            solid_capstyle="butt", clip_on=False, zorder=z)
    ax.annotate(f"{nice:g} px", (bx + nice / 2.0, by), xytext=(0, 2),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=6.5, color="0.3", zorder=z)


def plot_trajectories(titles, Xs, *, colors, n_cols: int = 3,
                      node_size: float = 30.0, line_width: float = 2.0,
                      alpha: float = 0.2, plot_width: float = 3.6,
                      overlap: tuple = (0.15, 0.0), lims=None,
                      min_height_frac: float = 0.45):
    """keypoint_moseq.viz.plot_trajectories, ported: pose sequences on one
    shared canvas with shared tile limits, poses whitening with age.

    ``Xs`` holds one (n_poses, n_nodes, 2) sequence per tile, nodes in chain
    order; ``colors`` one colour per node. Returns the figure.
    """
    plt = _pyplot()
    Xs = np.stack(Xs)
    edges = list(zip(range(Xs.shape[2] - 1), range(1, Xs.shape[2])))
    if lims is None:
        lims = get_limits(Xs)
    # A swimming cell travels far along +x but barely in y, where kpms mice
    # fill both; give every tile a minimum height so the canvas is not a strip.
    width, height = np.diff(lims, axis=0)[0]
    if height < min_height_frac * width:
        pad = (min_height_frac * width - height) / 2.0
        lims = lims + np.array([[0.0, -pad], [0.0, pad]])

    n_cols = min(n_cols, len(Xs))
    n_rows = np.ceil(len(Xs) / n_cols)
    offsets = np.stack(
        np.meshgrid(
            np.arange(n_cols) * np.diff(lims[:, 0]) * (1 - overlap[0]),
            np.arange(n_rows) * np.diff(lims[:, 1]) * (overlap[1] - 1),
        ),
        axis=-1,
    ).reshape(-1, 2)[: len(Xs)]

    Xs = Xs + offsets[:, None, None]
    xmin, ymin = lims[0] + offsets.min(0)
    xmax, ymax = lims[1] + offsets.max(0)

    fig, ax = plt.subplots(frameon=False)
    fig.set_layout_engine("none")
    ax.fill_between([xmin, xmax], y1=[ymax, ymax], y2=[ymin, ymin],
                    facecolor="w", zorder=0, clip_on=False)

    title_xy = (lims * np.array([[0.5, 0.1], [0.5, 0.9]])).sum(0)
    for xy, text in zip(offsets + title_xy, titles):
        ax.text(*xy, text, c="k", ha="center", va="top", fontsize=9,
                zorder=Xs.shape[1] * 4 + 4)

    final_width = xmax - xmin
    final_height = title_xy[1] - ymin
    fig_width = plot_width * (n_cols - (n_cols - 1) * overlap[0])
    fig_height = final_height / final_width * fig_width
    fig.set_size_inches((fig_width, fig_height))

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.axis("off")

    for i in range(Xs.shape[1]):
        for X in Xs:
            for ii, jj in edges:
                ax.plot(*X[i, (ii, jj)].T, c="k", zorder=i * 4,
                        linewidth=line_width, clip_on=False)
            for ii, jj in edges:
                ax.plot(*X[i, (ii, jj)].T, c=colors[ii], zorder=i * 4 + 1,
                        linewidth=line_width * 0.9, clip_on=False)
            ax.scatter(*X[i].T, c=colors, zorder=i * 4 + 2, edgecolor="k",
                       linewidth=0.4, s=node_size, clip_on=False)
        # kpms fade: wash the whole canvas after every timestep but the last,
        # so older poses whiten and the final pose sits saturated on top.
        if i < Xs.shape[1] - 1:
            ax.fill_between([xmin, xmax], y1=[ymax, ymax], y2=[ymin, ymin],
                            facecolor="w", alpha=alpha, zorder=i * 4 + 3,
                            clip_on=False)

    _scale_bar(ax, float(np.diff(lims[:, 0])[0]), xmin, ymin, xmax, ymax,
               z=Xs.shape[1] * 4 + 4)
    return fig
