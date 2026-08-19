"""Clip-level features for the suggestion model.

Two halves. **Kinematic** (~25 features from ``tracks.parquet``): the annotated
categories are mostly *motion*, which no single frame shows, so speed,
straightness, turn rate and beat frequency carry the signal. **CNN**
(PCA-reduced ROI features): these encode individual identity almost perfectly
(100% re-ID in testing), so a model can score well by recognising animals rather
than behaviours. Fit the PCA on *per-instance* vectors, not on the handful of
clip vectors, and judge the payoff by grouped cross-validation in
:mod:`trackanno.suggest`.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .clips import FEATURES_FILE, MANIFEST_FILE, TRACKS_FILE
from .config import Config

__all__ = ["build_clip_features", "CLIP_FEATURES_FILE", "PCA_FILE"]

CLIP_FEATURES_FILE = "clip_features.parquet"
PCA_FILE = "feature_pca.joblib"
# Clip-level CNN aggregate BEFORE any PCA, row-aligned with the parquet, so
# cross-validation can refit PCA inside each fold. A PCA fitted on all movies has
# seen the held-out movie's pixels, and between-movie variance is a dominant
# axis, so that leak is first-order.
CLIP_CNN_RAW_FILE = "clip_cnn_raw.npy"

# Rates are per SECOND everywhere, so movies acquired at different frame rates
# stay comparable (ND2 files report their own fps; TIFF takes it from config).
IMMOTILE_PX_PER_SEC = 50.0
LOCALISATION_NOISE_PX = 1.8   # measured per-frame keypoint jitter
BEAT_NFFT = 128               # fixed grid so beat frequency is comparable across clips
BEAT_F_MIN_HZ = 1.0
# A frequency is measurable only if the window spans several cycles of it: a
# tethered cell beating at ~1.2 Hz has a period longer than a whole 50-frame clip
# at 100 fps. Report NaN rather than the lowest searchable bin.
BEAT_MIN_CYCLES = 3.0
# Do not reinstate a significance threshold: a fixed 10x excess gate passes 30% of
# pure red noise at 25 frames, 81% at 50 and ~100% at 100+, since peak-over-band-
# median scales with the search band and so with clip length. `beat_excess` and
# `beat_censored` are emitted as features instead.
#
# Above this filled share a spectral estimate describes the interpolator, not the
# cell, and is NaN. The 20% `badFileFlag` in computeGoodFrameRange.m.
MAX_INTERP_FRAC = 0.20

# Every column a feature row carries, whatever the task, clip length or tracking
# quality, assembled from the three modules that produce them so it cannot drift.
# A stable schema is what lets a model trained on one project run on another.
_BEAT_FEATURES = (
    "beat_amp_mean", "beat_amp_max", "beat_speed_mean", "beat_amp_distal",
    "beat_freq_hz", "beat_power_frac", "beat_excess", "beat_resolvable_hz",
    "beat_censored",
)


_SCHEMA_CACHE: tuple[str, ...] | None = None


def feature_schema() -> tuple[str, ...]:
    """The full column set, resolved lazily so the imports stay acyclic."""
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is None:
        from . import trackseries, wavespectra

        _SCHEMA_CACHE = tuple(dict.fromkeys(
            tuple(trackseries.feature_names())
            + _BEAT_FEATURES
            + tuple(wavespectra.SERIES_COLUMNS)
        ))
    return _SCHEMA_CACHE


# --------------------------------------------------------------------------- #
# kinematics
# --------------------------------------------------------------------------- #
def _safe(value: float, default: float = np.nan) -> float:
    """Undefined stays NaN. Never collapse "could not compute" onto 0.0: a
    straightness of 0.0 is a perfect circle and a beat frequency of 0.0 is DC, so
    a failure becomes an extreme but plausible value the classifier will fit.
    """
    return float(value) if np.isfinite(value) else default


def _on_grid(sub: pd.DataFrame, columns: list[str], n_frames: int, f_start: int) -> np.ndarray:
    """Lay a track's rows onto the full clip frame grid, NaN where undetected.

    Missing frames are absent ROWS in tracks.parquet, so np.diff would bridge a
    gap and read a 5-frame jump as one frame of motion, inflating speed by up to
    27%. On a NaN-filled grid the gap propagates to NaN and the nan-aware
    reducers drop it.
    """
    out = np.full((n_frames, len(columns)), np.nan, dtype=float)
    idx = sub["frame"].to_numpy(dtype=int) - f_start
    keep = (idx >= 0) & (idx < n_frames)
    out[idx[keep]] = sub[columns].to_numpy(dtype=float)[keep]
    return out


def _interp_nan(values: np.ndarray) -> np.ndarray:
    """Linear fill of NaNs along TIME (axis 0), for spectral estimation only.

    Interior gaps are interpolated; leading and trailing ones clamp to the
    nearest finite value, and a column with fewer than two finite values is
    zeroed. Same discipline as ``interpInTime.m``: a missing frame is FILLED,
    never removed, so ``dt`` stays uniform for the FFT.
    """
    out = np.array(values, dtype=float, copy=True)
    for c in range(out.shape[1] if out.ndim > 1 else 1):
        col = out[:, c] if out.ndim > 1 else out
        good = np.isfinite(col)
        if good.sum() < 2:
            col[:] = 0.0
            continue
        i = np.arange(col.size)
        col[~good] = np.interp(i[~good], i[good], col[good])
    return out


def _beat_estimate(lateral: np.ndarray, fps: float) -> dict:
    """Dominant oscillation frequency of body-frame lateral displacement.

    Three rules, none of which an argmax on the raw bin index obeys:

    * Search a PHYSICAL band (>= BEAT_F_MIN_HZ), never a bin index: a red
      spectrum searched from bin 2 always returns bin 2, making the feature a
      readout of track length (r = -0.999 with frame count).
    * Fix the FFT grid (zero-padded to BEAT_NFFT) so clips of different length
      are comparable.
    * REPORT the peak's excess over the red-noise baseline (``beat_excess``);
      never blank the frequency on it.
    """
    out = {"beat_freq_hz": np.nan, "beat_power_frac": np.nan, "beat_excess": np.nan}
    if lateral.ndim != 2 or lateral.shape[0] < 8 or lateral.shape[1] == 0:
        return out

    sig = _interp_nan(lateral)
    n = sig.shape[0]
    t = np.arange(n)
    # Remove per-keypoint drift so slow body reorientation does not dominate.
    for c in range(sig.shape[1]):
        sig[:, c] -= np.polyval(np.polyfit(t, sig[:, c], 1), t)

    # A flagellum that does not move has no beat, yet the argmax of an all-zero
    # spectrum returns a real-looking number. A DEGENERACY test, not a
    # significance one: a still-but-jittery cell does vary and IS reported.
    if not np.isfinite(sig).any() or float(np.nanmax(np.abs(sig))) < 1e-9:
        return out

    window = np.hanning(n)[:, None]
    # BEAT_NFFT is a FLOOR, never a ceiling. `np.fft.rfft(x, n=128)` zero-pads a
    # shorter signal but TRUNCATES a longer one, and the Hanning taper is built
    # for the full length, so a longer clip would reach the FFT as its first 128
    # samples under an asymmetric taper with a step at the cut.
    nfft = max(BEAT_NFFT, n)
    spec = np.abs(np.fft.rfft(sig * window, n=nfft, axis=0)) ** 2        # power
    power = spec.sum(axis=1)                                            # pool keypoints
    freqs = np.fft.rfftfreq(nfft, d=1.0 / max(fps, 1e-6))

    # Only search frequencies the window is actually long enough to resolve.
    duration = n / max(fps, 1e-6)
    f_floor = max(BEAT_F_MIN_HZ, BEAT_MIN_CYCLES / duration)
    out["beat_resolvable_hz"] = float(f_floor)
    band = freqs >= f_floor
    if band.sum() < 3:
        return out
    band_power, band_freqs = power[band], freqs[band]
    peak = int(np.argmax(band_power))
    baseline = float(np.median(band_power))
    total = float(band_power.sum())

    out["beat_freq_hz"] = float(band_freqs[peak])
    out["beat_power_frac"] = _safe(band_power[peak] / total if total > 0 else np.nan)
    out["beat_excess"] = _safe(band_power[peak] / baseline if baseline > 0 else np.nan)
    # CENSORED, flagged rather than blanked: an argmax on the bottom edge of the
    # band means the real oscillation is at or below what this window can
    # resolve, so the number is an upper bound. Tolerance is half a NATIVE
    # resolution element, not half a grid step - the grid is zero-padded to
    # BEAT_NFFT and so is finer than the window can resolve.
    out["beat_censored"] = float(
        band_freqs[peak] < f_floor + 0.5 * max(fps, 1e-6) / n
    )
    return out


def _path_length(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Path length and net displacement over the SAME endpoints.

    Summing only the finite one-frame steps drops a gap from the path while the
    net displacement still spans it, so ``net / path`` can exceed 1. Joining
    consecutive PRESENT positions bridges each gap with the straight line - the
    shortest path that could have been taken - so ``path >= net`` is an identity.

    The speed columns must NOT do this: they keep the per-frame grid, because a
    bridged 5-frame jump read as one frame of motion is the inflation
    ``_on_grid`` exists to prevent.
    """
    present = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
    if present.size < 2:
        return np.nan, np.nan
    px_, py_ = x[present], y[present]
    path = float(np.hypot(np.diff(px_), np.diff(py_)).sum())
    net = float(np.hypot(px_[-1] - px_[0], py_[-1] - py_[0]))
    return path, net


def _heading_lag(x: np.ndarray, y: np.ndarray, n_frames: int) -> int:
    """Smallest frame lag whose typical displacement clears the noise floor.

    Heading is an angle, defined only once the cell has moved further than the
    localisation jitter. Test that over a LAG, never over a single frame: a
    per-frame test demands ``LOCALISATION_NOISE_PX`` per frame, 180 px/s at
    100 fps against a median cell speed of 83, so it leaves turn_* undefined on
    most clips. It is also a speed threshold in disguise, which makes the NaN
    pattern itself a speed measurement (AUC 0.985 against speed_p90) that a
    NaN-native model splits on and reads as "turning".
    """
    max_lag = max(1, min(n_frames // 4, 25))
    floor = 3.0 * LOCALISATION_NOISE_PX
    for lag in range(1, max_lag + 1):
        d = np.hypot(x[lag:] - x[:-lag], y[lag:] - y[:-lag])
        d = d[np.isfinite(d)]
        if d.size and float(np.median(d)) > floor:
            return lag
    return max_lag


def _shifted_legendre(s: np.ndarray, order: int) -> np.ndarray:
    """Shifted Legendre polynomials P~n on s in [0, 1], as (order+1, len(s))."""
    x = 2.0 * s - 1.0                      # map [0,1] -> [-1,1]
    out = [np.ones_like(x), x, 0.5 * (3.0 * x ** 2 - 1.0)]
    return np.stack(out[: order + 1], axis=0)


def _body_frame_motion(px: np.ndarray, py: np.ndarray, dt: float = 1.0) -> dict:
    """Flagellum motion after removing translation and body rotation.

    Node 0 is Head and node 1 is Base, so Head->Base defines the body axis.
    Expressing the flagellum in that frame isolates beating from swimming: a cell
    held in place can beat hard, and a drifting dead cell can translate without
    beating.
    """
    out: dict[str, float] = {}
    if px.shape[0] < 2 or px.shape[1] < 3:
        return out

    ox, oy = px[:, 1], py[:, 1]                       # Base = origin
    axis = np.arctan2(py[:, 0] - oy, px[:, 0] - ox)   # Base -> Head
    valid = np.isfinite(axis) & np.isfinite(ox) & np.isfinite(oy)
    if valid.sum() < 2:
        return out

    cos, sin = np.cos(-axis), np.sin(-axis)
    rx = (px[:, 2:] - ox[:, None]) * cos[:, None] - (py[:, 2:] - oy[:, None]) * sin[:, None]
    ry = (px[:, 2:] - ox[:, None]) * sin[:, None] + (py[:, 2:] - oy[:, None]) * cos[:, None]

    # Mask, never compact. Dropping the rows where Head or Base was undetected
    # leaves the survivors non-uniformly sampled while _beat_estimate still
    # assumes 1/fps, scaling beat_freq_hz by the reciprocal of the surviving
    # fraction (8 Hz read as 14.1 at 50% loss).
    rx = np.where(valid[:, None], rx, np.nan)
    ry = np.where(valid[:, None], ry, np.nan)
    if valid.sum() < 2:
        return out

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN rows are expected
        amp = np.sqrt(np.nanmean((ry - np.nanmean(ry, axis=0)) ** 2, axis=1))  # lateral
        step = np.sqrt(np.nanmean(np.diff(rx, axis=0) ** 2 + np.diff(ry, axis=0) ** 2, axis=1))
        out["beat_amp_mean"] = _safe(np.nanmean(amp))
        out["beat_amp_max"] = _safe(np.nanmax(amp))
        out["beat_speed_mean"] = _safe(np.nanmean(step) / dt)

    # The most distal node alone: pooling all flagellar keypoints averages Flag1
    # (rms ~3 px) with Tip (~15 px), a ~5x dilution.
    out["beat_amp_distal"] = _safe(
        np.sqrt(np.nanmean((ry[:, -1] - np.nanmean(ry[:, -1])) ** 2))
    )

    # Unwrap on the filled grid, then keep only the steps whose BOTH endpoints
    # were really detected, so a gap neither breaks the unwrap nor contributes a
    # bridged rotation to the rate.
    ang = np.unwrap(_interp_nan(axis[:, None])[:, 0])
    turn = np.diff(ang)[valid[1:] & valid[:-1]]
    out["body_rotation_speed"] = _safe(np.nanmean(np.abs(turn)) / dt) if turn.size else np.nan

    # Handed to the spectral estimator: lateral displacement is linear, so it
    # needs no unwrapping. Held back when too much of the clip would have to be
    # invented - see MAX_INTERP_FRAC.
    if (1.0 - valid.mean()) <= MAX_INTERP_FRAC:
        out["_lateral"] = ry
    return out


# --------------------------------------------------------------------------- #
# per-track measurement
# --------------------------------------------------------------------------- #
def _beat_on_window(tdf: pd.DataFrame, cfg: Config, fps: float,
                    f_start: int, f_end: int) -> dict:
    """Beat amplitude and frequency on a window CENTRED on the clip.

    The clip is where the behaviour is LABELLED, not where a beat can be
    MEASURED; see :class:`trackanno.config.Spectral` for the default. Keep this
    on the same window as the waveform series, or beat_freq_hz off 25 frames sits
    beside wave_freq_hz off 75 and one row carries two answers to one question.
    """
    if not cfg.is_pose:
        return {}
    frames = tdf["frame"].to_numpy(dtype=int)
    first, last = int(frames.min()), int(frames.max()) + 1
    span = int(cfg.spectral.window)
    centre = (int(f_start) + int(f_end)) // 2
    hi = min(max(centre + span // 2, first + span), last)
    lo = max(first, hi - span)
    n = hi - lo
    if n < 8:
        return {}
    sub = tdf[(tdf["frame"] >= lo) & (tdf["frame"] < hi)]
    if sub.empty:
        return {}
    kx = [f"kp_x_{k}" for k in range(cfg.n_keypoints)]
    ky = [f"kp_y_{k}" for k in range(cfg.n_keypoints)]
    if not all(c in sub.columns for c in kx + ky):
        return {}
    px, py = _on_grid(sub, kx, n, lo), _on_grid(sub, ky, n, lo)
    motion = _body_frame_motion(px, py, 1.0 / max(fps, 1e-6))
    lateral = motion.pop("_lateral", None)
    # body_rotation_speed belongs to the clip, not the measurement window:
    # trackseries already reports it over the clip.
    out = {k: v for k, v in motion.items() if k.startswith("beat_")}
    if lateral is not None:
        out.update(_beat_estimate(lateral, fps))
    return out


def window_features(
    tdf: pd.DataFrame, cfg: Config, fps: float, windows: Sequence[tuple[int, int]]
) -> list[dict]:
    """Features for several windows of ONE track, measured once over the track.

    The single entry point for turning a track into feature rows -
    :func:`build_clip_features` for the annotation manifest,
    :mod:`trackanno.densepredict` for the dense export grid. Keep it that way:
    the spectral columns are measured over a window CENTRED on the requested
    range rather than over the range itself, and a second caller reimplementing
    that would emit columns with the same names and a different meaning.

    The track is laid out once (:mod:`trackanno.trackseries`) and the beat
    transformed once (:mod:`trackanno.wavespectra`), so overlapping windows cost
    only the slicing (500-frame track: 2.7x for 10 windows, 6.7x at stride 10).
    """
    from . import trackseries, wavespectra

    frames = tdf["frame"].to_numpy(dtype=int)
    lo = min(int(frames.min()), min(int(w[0]) for w in windows))
    hi = max(int(frames.max()) + 1, max(int(w[1]) for w in windows))
    series = trackseries.build_series(tdf, cfg, fps, lo, hi - lo)

    spectra = None
    if cfg.is_pose:
        spectra = wavespectra.track_spectral_series(
            tdf, cfg.n_keypoints, fps,
            window=cfg.spectral.window, hop=cfg.spectral.hop,
        )

    rows = []
    for f_start, f_end in windows:
        entry = trackseries.aggregate_window(series, int(f_start), int(f_end), fps, cfg)
        entry.update(_beat_on_window(tdf, cfg, fps, int(f_start), int(f_end)))
        if spectra is not None and len(spectra):
            entry.update(wavespectra.window_values(spectra, int(f_start), int(f_end)))
        # One schema whatever the task or the data: a detect project, or a track
        # too short to transform, still produces the same columns.
        for key in feature_schema():
            entry.setdefault(key, np.nan)
        rows.append(entry)
    return rows


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def build_clip_features(cfg: Config, verbose: bool = True, reporter=None) -> pd.DataFrame:
    """Compute (and cache) one feature row per clip across every prepared movie."""
    from .progress import NullReporter

    reporter = reporter or NullReporter()
    work = cfg.paths.work_dir
    movie_dirs = sorted(p for p in work.iterdir() if p.is_dir() and (p / MANIFEST_FILE).exists())
    if not movie_dirs:
        raise FileNotFoundError(f"no prepared movies under {work}; run prepare.py first")

    manifests, tracks_by_movie, feats_by_movie = [], {}, {}
    for d in movie_dirs:
        man = pd.read_parquet(d / MANIFEST_FILE)
        if man.empty:
            continue
        manifests.append(man)
        tracks_by_movie[d.name] = pd.read_parquet(d / TRACKS_FILE)
        fpath = d / FEATURES_FILE
        feats_by_movie[d.name] = np.load(fpath) if fpath.exists() else None

    manifest = pd.concat(manifests, ignore_index=True)

    # -- PCA on per-instance vectors, pooled across movies -------------------- #
    pca, dim = None, 0
    usable = {k: v for k, v in feats_by_movie.items() if v is not None}
    if usable:
        # The CNN vector's width follows the model's channel widths (1920-d for
        # an x-model, 448-d for a nano); numpy's own error on concatenating two
        # widths names neither the movie nor the model. Say it properly.
        widths = {k: int(v.shape[1]) for k, v in usable.items()}
        if len(set(widths.values())) > 1:
            by_width: dict[int, list[str]] = {}
            for movie_id, width in sorted(widths.items()):
                by_width.setdefault(width, []).append(movie_id)
            detail = "\n".join(
                f"    {width:>5}-d : {', '.join(ids)}" for width, ids in sorted(by_width.items())
            )
            raise RuntimeError(
                "these movies were prepared with different detection models, so their "
                f"CNN features cannot be pooled:\n{detail}\n"
                "The vector width follows the model's channel widths. Re-prepare every "
                "movie with the same weights (Detect & crop, with 'redo' ticked), or set "
                "features.enabled: false to use the kinematic features alone."
            )
        stacked = np.concatenate([v for v in usable.values()], axis=0).astype(np.float32)
        n_comp = int(min(cfg.suggest.pca_components, stacked.shape[0], stacked.shape[1]))
        if n_comp >= 2:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=n_comp, random_state=cfg.suggest.seed)
            pca.fit(stacked)
            dim = n_comp
            var = float(pca.explained_variance_ratio_.sum())
            message = (
                f"PCA: {stacked.shape[0]} instance vectors x {stacked.shape[1]}-d "
                f"-> {n_comp} comps ({var:.1%} variance)"
            )
            reporter.log(message)
            if verbose:
                print(f"  {message}")
        del stacked

    rows: list[dict] = []
    raw_rows: list[np.ndarray | None] = []
    reporter.stage("clip features", len(manifest))
    for movie_id, man in manifest.groupby("movie_id", sort=True):
        tracks = tracks_by_movie[movie_id]
        feats = feats_by_movie.get(movie_id)
        fps = float(json.loads((work / movie_id / "cache.json").read_text())["movie"]["fps"])
        by_track = {t: df for t, df in tracks.groupby("track_id", sort=False)}

        # Grouped by TRACK, not walked clip by clip: the series and the sliding
        # transform are built once per track and each window is a slice, so
        # overlapping windows (the dense grid, any stride < frames) are free.
        by_spec: dict[int, list] = {}
        for spec in man.itertuples(index=False):
            by_spec.setdefault(int(spec.track_id), []).append(spec)

        for track_id, specs in by_spec.items():
            reporter.raise_if_cancelled()
            tdf = by_track.get(track_id)
            if tdf is None:
                reporter.advance(len(specs))
                continue
            # Sort rather than trust parquet order: nothing asserts it is
            # frame-major, and the CNN delta below is order-dependent.
            tdf = tdf.sort_values("frame")
            windows = [(int(s.f_start), int(s.f_end)) for s in specs]
            measured = window_features(tdf, cfg, fps, windows)

            for spec, features in zip(specs, measured):
                reporter.advance()
                mask = (tdf["frame"] >= spec.f_start) & (tdf["frame"] < spec.f_end)
                sub = tdf[mask]
                if sub.empty:
                    continue

                entry = {
                    "clip_id": spec.clip_id,
                    "movie_id": movie_id,
                    "track_id": int(spec.track_id),
                    "f_start": int(spec.f_start),
                }
                entry.update(features)

                if pca is not None and feats is not None:
                    rows_idx = sub.index.to_numpy()
                    # instance_features.npy is row-aligned with tracks.parquet.
                    # Assert it rather than trust it: these vectors re-identify
                    # individuals at 100%, so a mis-pairing shows up not as lower
                    # accuracy but as a model that tests well and fails later.
                    if len(feats) != len(tracks) or rows_idx.max(initial=0) >= len(feats):
                        raise RuntimeError(
                            f"{movie_id}: instance_features.npy has {len(feats)} rows but "
                            f"tracks.parquet has {len(tracks)}; features are not row-aligned"
                        )
                    block = feats[rows_idx].astype(np.float32)
                    codes = pca.transform(block)
                    entry.update({f"cnn{i}_mean": float(v) for i, v in enumerate(codes.mean(axis=0))})
                    entry.update({f"cnn{i}_std": float(v) for i, v in enumerate(codes.std(axis=0))})

                    # Frame-to-frame change, measured ON THE FRAME GRID. np.diff
                    # over present rows alone compares distant frames as though
                    # adjacent - the gap-bridging _on_grid exists to prevent.
                    span = int(spec.f_end - spec.f_start)
                    offsets = sub["frame"].to_numpy(dtype=int) - int(spec.f_start)
                    inside = (offsets >= 0) & (offsets < span)

                    def _grid_delta(values: np.ndarray) -> np.ndarray:
                        grid = np.full((span, values.shape[1]), np.nan)
                        grid[offsets[inside]] = values[inside]
                        steps = np.abs(np.diff(grid, axis=0))
                        if not np.isfinite(steps).any():
                            return np.zeros(values.shape[1])
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            return np.nan_to_num(np.nanmean(steps, axis=0))

                    delta = _grid_delta(codes)
                    entry.update({f"cnn{i}_delta": float(v) for i, v in enumerate(delta)})

                    # Same aggregation, un-reduced, for fold-wise PCA at CV time.
                    raw_delta = _grid_delta(block).astype(np.float32)
                    raw_rows.append(
                        np.concatenate([block.mean(axis=0), block.std(axis=0), raw_delta])
                    )
                elif pca is not None:
                    raw_rows.append(None)
                rows.append(entry)

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("no clip features could be built")
    # Keep NaN: it means "undefined", and both HistGradientBoosting and CatBoost
    # handle it natively. See `_safe`.
    out = out.replace([np.inf, -np.inf], np.nan)
    # Stamped into the table, not kept beside it, so a copy, move or partial sync
    # cannot separate the version from the data it describes.
    out["_feature_version"] = FEATURE_VERSION
    out.to_parquet(work / CLIP_FEATURES_FILE, index=False)
    if pca is not None:
        import joblib

        joblib.dump(pca, work / PCA_FILE)
        if raw_rows and len(raw_rows) == len(out) and all(r is not None for r in raw_rows):
            np.save(work / CLIP_CNN_RAW_FILE, np.stack(raw_rows).astype(np.float32))
        else:
            (work / CLIP_CNN_RAW_FILE).unlink(missing_ok=True)
    else:
        (work / CLIP_CNN_RAW_FILE).unlink(missing_ok=True)

    n_kin = len([c for c in out.columns if not c.startswith("cnn")]) - 4
    n_cnn = len([c for c in out.columns if c.startswith("cnn")])
    summary = f"{len(out)} clips x {n_kin} kinematic + {n_cnn} CNN features -> {CLIP_FEATURES_FILE}"
    reporter.log(summary)
    if verbose:
        print(f"  {summary}")
    return out


# Bump whenever a feature's MEANING changes, not just when one is added or
# removed. `Suggester.load`'s column check catches a renamed or dropped column
# but not a redefined one, and a model whose inputs changed meaning under
# unchanged names loads, reports itself ready, and predicts confidently.
#
# 1  the original per-clip features
# 2  spectral columns measured on a centred window (config `spectral`); no
#    significance gating; straightness bounded by construction; lag-based turn
#    rates; rotation-invariant shape_*; circular static mode as cos/sin
FEATURE_VERSION = 2

# Carried in the table, never fed to a model: `n_windows` and `centre_offset`
# describe HOW the series was read out, not the cell, so a model given them
# learns the readout grid.
META_COLUMNS = (
    "clip_id", "movie_id", "track_id", "f_start", "_row",
    "_feature_version", "n_windows", "centre_offset",
)

# `needs` lets the GUI say WHY a group is unavailable rather than leaving it off.
FEATURE_GROUPS: dict[str, dict] = {
    "kinematic": {
        "needs": None,
        "label": "trajectory and shape",
        "about": "Speed, straightness, turning, box area and aspect. Computed "
                 "from the per-frame box alone, so always available.",
        "prefixes": (
            "speed_", "immotile_frac", "path_length", "net_displacement",
            "straightness", "radius_gyration", "turn_", "area_", "aspect_",
            "present_frac", "conf_",
        ),
    },
    "pose": {
        "needs": "pose",
        "label": "body and flagellum",
        "about": "Body length, flagellum length, beat amplitude and frequency, "
                 "and translocation anchored on the cell body rather than the "
                 "box centroid. Needs keypoints.",
        "prefixes": (
            "kpt_", "body_", "flagellum_", "beat_", "tip_", "shape_",
        ),
    },
    "waveform": {
        "needs": "pose",
        "label": "beat waveform",
        "about": "Tangent-angle shape descriptors after Geyer, Howard and "
                 "Sartori (2022): static curvature, amplitude and phase "
                 "profiles, and the wavelength. Needs keypoints.",
        "prefixes": ("wave_",),
    },
    "cnn": {
        "needs": "features",
        "label": "CNN appearance",
        "about": "PCA of per-detection embeddings from the detector's own "
                 "forward pass. Powerful, but they re-identify individuals "
                 "almost perfectly, so they reward memorising animals rather "
                 "than learning behaviour. Needs features.enabled.",
        "prefixes": ("cnn",),
    },
}


def group_of(column: str) -> str | None:
    """Which feature group a column belongs to, or None if it is metadata."""
    if column in META_COLUMNS:
        return None
    for name, spec in FEATURE_GROUPS.items():
        if any(column.startswith(p) for p in spec["prefixes"]):
            return name
    return "kinematic"          # anything unclassified is box-derived


def available_groups(cfg: Config) -> dict[str, str]:
    """{group: ""} when usable, or {group: why not} when it cannot be computed."""
    out: dict[str, str] = {}
    for name, spec in FEATURE_GROUPS.items():
        need = spec["needs"]
        if need == "pose" and not cfg.is_pose:
            out[name] = "needs detection.task: pose"
        elif need == "features" and not cfg.features.enabled:
            out[name] = "needs features.enabled: true"
        else:
            out[name] = ""
    return out


def feature_columns(frame: pd.DataFrame, groups=None) -> list[str]:
    """The model's input columns, optionally restricted to certain groups.

    Restricting is a modelling decision, not tidiness: the CNN columns
    fingerprint individuals, so a project wanting a behaviour model rather than a
    re-identification one may want them out entirely.
    """
    columns = [c for c in frame.columns if c not in META_COLUMNS]
    if groups is None:
        return columns
    wanted = set(groups)
    return [c for c in columns if group_of(c) in wanted]
