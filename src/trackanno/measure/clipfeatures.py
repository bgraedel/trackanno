"""Clip-level features for the suggestion model.

**Kinematic** features from ``tracks.parquet`` (speed, straightness, turning,
beat frequency), optional **context** over a longer centred span, and optional
**embeddings** summarised from per-detection vectors
(:mod:`trackanno.measure.embeddings`). Embeddings can re-identify individuals,
so judge them by the grouped cross-validation in :mod:`trackanno.learn.suggest`.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..annotate.clips import MANIFEST_FILE, TRACKS_FILE
from ..config import FIXED_SETTINGS, Config
from . import embeddings as emb

# Frames between spectral estimates, and the support of the wave_flow columns.
SPECTRAL_HOP = FIXED_SETTINGS["spectral.hop"]
FLOW_FRAMES = FIXED_SETTINGS["spectral.flow_frames"]

__all__ = ["build_clip_features", "cached_clip_features", "feature_cache_key",
           "grid_delta", "window_features", "CLIP_FEATURES_FILE", "CONTEXT_PREFIX"]

CLIP_FEATURES_FILE = "clip_features.parquet"
# The cache key the table on disk was built under, beside the table.
CLIP_FEATURES_KEY_FILE = "clip_features.key"
CONTEXT_PREFIX = "ctx_"

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
# No significance threshold: a fixed excess gate passes most pure red noise,
# because peak-over-band-median scales with the search band and so with clip
# length. `beat_excess` and `beat_censored` are emitted as features instead.
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
    gap and read a 5-frame jump as one frame of motion, inflating speed. On a
    NaN-filled grid the gap propagates to NaN and the nan-aware reducers drop it.
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


def grid_delta(values: np.ndarray, offsets: np.ndarray, span: int) -> np.ndarray:
    """Mean absolute step per column, measured ON THE FRAME GRID.

    ``offsets`` are in-window frame offsets for the rows of ``values``, already
    restricted to ``[0, span)``. Only adjacent-frame pairs contribute: differencing
    the present rows alone compares distant frames as though they were neighbours,
    which is the gap-bridging ``_on_grid`` exists to prevent. Zeros where no
    adjacent pair survives, since the quantity is a summed change and there was
    none to measure.

    Shared with the dense table, which must form this column identically or the
    model is scored on a feature it was not fitted on.
    """
    grid = np.full((int(span), values.shape[1]), np.nan)
    grid[np.asarray(offsets, dtype=int)] = values
    steps = np.abs(np.diff(grid, axis=0))
    if not np.isfinite(steps).any():
        return np.zeros(values.shape[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nan_to_num(np.nanmean(steps, axis=0))


def _beat_estimate(lateral: np.ndarray, fps: float) -> dict:
    """Dominant oscillation frequency of body-frame lateral displacement.

    Three rules, none of which an argmax on the raw bin index obeys:

    * Search a PHYSICAL band (>= BEAT_F_MIN_HZ), never a bin index: a red
      spectrum searched from a fixed bin always returns that bin, making the
      feature a readout of track length.
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
    # `polyfit` takes a 2-D y and fits each column independently, so this is one
    # call for every keypoint.
    slope, intercept = np.polyfit(t, sig, 1)
    sig -= t[:, None] * slope + intercept

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
    per-frame test demands ``LOCALISATION_NOISE_PX`` of motion every frame, so
    it leaves turn_* undefined on most clips, and it is a speed threshold in
    disguise - which makes the NaN pattern itself a speed measurement that a
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

    Node 0 is Head and node 1 is Base, so Base->Head defines the body axis.
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
    # fraction.
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

    # The most distal node alone: pooling all flagellar keypoints averages the
    # low-amplitude proximal nodes with the tip and dilutes the signal.
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
class _KeypointGrid:
    """One track's keypoints on its full frame grid, built once and sliced.

    :func:`_on_grid` is a scatter with no interpolation, so a slice of the whole
    track's grid equals the array a per-window call builds.
    """

    __slots__ = ("px", "py", "present", "first", "last")

    def __init__(self, tdf: pd.DataFrame, n_keypoints: int):
        frames = tdf["frame"].to_numpy(dtype=int)
        self.first, self.last = int(frames.min()), int(frames.max()) + 1
        n = self.last - self.first
        kx = [f"kp_x_{k}" for k in range(n_keypoints)]
        ky = [f"kp_y_{k}" for k in range(n_keypoints)]
        self.px = _on_grid(tdf, kx, n, self.first)
        self.py = _on_grid(tdf, ky, n, self.first)
        # Which frames have a row. Not the same as which are finite: a row of
        # all-NaN keypoints is still a row, and is measured.
        self.present = np.zeros(n, dtype=bool)
        self.present[frames - self.first] = True

    @staticmethod
    def maybe(tdf: pd.DataFrame, cfg: Config) -> "_KeypointGrid | None":
        """The grid, or None on a detect project or a table without keypoints."""
        if not cfg.is_pose:
            return None
        needed = [f"kp_{a}_{k}" for k in range(cfg.n_keypoints) for a in ("x", "y")]
        if not all(c in tdf.columns for c in needed):
            return None
        return _KeypointGrid(tdf, cfg.n_keypoints)


def _beat_on_window(grid: "_KeypointGrid | None", cfg: Config, fps: float,
                    f_start: int, f_end: int) -> dict:
    """Beat amplitude and frequency on a window CENTRED on the clip.

    The clip is where the behaviour is LABELLED, not where a beat can be
    MEASURED; see :class:`trackanno.config.Spectral` for the default. Keep this
    on the same window as the waveform series, or beat_freq_hz off 25 frames sits
    beside wave_freq_hz off 75 and one row carries two answers to one question.
    """
    if grid is None:
        return {}
    first, last = grid.first, grid.last
    span = int(cfg.spectral.window)
    centre = (int(f_start) + int(f_end)) // 2
    hi = min(max(centre + span // 2, first + span), last)
    lo = max(first, hi - span)
    n = hi - lo
    if n < 8:
        return {}
    lo_i, hi_i = lo - first, hi - first
    if not grid.present[lo_i:hi_i].any():
        return {}
    px, py = grid.px[lo_i:hi_i], grid.py[lo_i:hi_i]
    motion = _body_frame_motion(px, py, 1.0 / max(fps, 1e-6))
    lateral = motion.pop("_lateral", None)
    # body_rotation_speed belongs to the clip, not the measurement window:
    # trackseries already reports it over the clip.
    out = {k: v for k, v in motion.items() if k.startswith("beat_")}
    if lateral is not None:
        out.update(_beat_estimate(lateral, fps))
    return out


def _context_span(first: int, last: int, f_start: int, f_end: int, n: int
                  ) -> tuple[int, int]:
    """``n`` frames centred on the window, clipped to ``[first, last)``."""
    centre = (int(f_start) + int(f_end)) // 2
    hi = min(max(centre + n - n // 2, first + n), last)
    lo = max(first, hi - n)
    return lo, max(hi, lo + 1)


_CONTEXT_CACHE: tuple[str, ...] | None = None


def _context_keys() -> tuple[str, ...]:
    """The kinematic trackseries features the context family repeats."""
    global _CONTEXT_CACHE
    if _CONTEXT_CACHE is None:
        from . import trackseries

        _CONTEXT_CACHE = tuple(k for k in trackseries.feature_names()
                               if group_of(k) == "kinematic")
    return _CONTEXT_CACHE


def context_names() -> list[str]:
    """Every column of the context family."""
    return [CONTEXT_PREFIX + k for k in _context_keys()]


def window_features(
    tdf: pd.DataFrame, cfg: Config, fps: float, windows: Sequence[tuple[int, int]],
    codes: dict[str, np.ndarray] | None = None,
) -> list[dict]:
    """Features for several windows of ONE track, measured once over the track.

    The single entry point for turning a track into feature rows -
    :func:`build_clip_features` for the annotation manifest,
    :mod:`trackanno.infer.densepredict` for the dense export grid. Keep it that way:
    the spectral columns are measured over a window CENTRED on the requested
    range rather than over the range itself, and a second caller reimplementing
    that would emit columns with the same names and a different meaning.

    Overlapping windows are supported: nothing here assumes the windows are
    ordered, adjacent, or a partition of the track, so a dense stride-1 grid and
    the annotation tiling go through the same code and get the same answers.

    They are not free. The `wave_*` STFT and the :mod:`trackanno.measure.trackseries`
    layout are amortised over the track, each window being a slice, but the
    `beat_*` block is not: :func:`_beat_on_window` pays a fresh mask over the
    whole track and a fresh FFT per window, so cost is linear in the window
    count over a substantial constant. See docs/prediction.md for the timings.

    ``tdf`` must be sorted by frame. ``codes`` maps an embedding name to PCA
    codes row-aligned with ``tdf``; each window gets their summary. With
    ``suggest.context_frames`` set, each window also gets the ``ctx_`` family.
    """
    from . import trackseries, wavespectra

    frames = tdf["frame"].to_numpy(dtype=int)
    first, last = int(frames.min()), int(frames.max()) + 1
    n_context = int(cfg.suggest.context_frames)
    lo = min(int(frames.min()), min(int(w[0]) for w in windows))
    hi = max(int(frames.max()) + 1, max(int(w[1]) for w in windows))
    series = trackseries.build_series(tdf, cfg, fps, lo, hi - lo)

    spectra = None
    if cfg.is_pose:
        spectra = wavespectra.track_spectral_series(
            tdf, cfg.n_keypoints, fps,
            window=cfg.spectral.window, hop=SPECTRAL_HOP,
            flow_frames=FLOW_FRAMES,
        )

    # Both hold the per-track work; the loop below only slices them.
    grid = _KeypointGrid.maybe(tdf, cfg)
    waves = wavespectra.WindowSource(spectra) if spectra is not None else None

    rows = []
    for f_start, f_end in windows:
        entry = trackseries.aggregate_window(series, int(f_start), int(f_end), fps, cfg)
        entry.update(_beat_on_window(grid, cfg, fps, int(f_start), int(f_end)))
        if waves is not None and len(spectra):
            entry.update(waves.at(int(f_start), int(f_end)))
        # One schema whatever the task or the data: a detect project, or a track
        # too short to transform, still produces the same columns.
        for key in feature_schema():
            entry.setdefault(key, np.nan)
        if n_context > 0:
            c_lo, c_hi = _context_span(first, last, int(f_start), int(f_end), n_context)
            context = trackseries.aggregate_window(series, c_lo, c_hi, fps, cfg)
            entry.update({CONTEXT_PREFIX + k: context.get(k, np.nan)
                          for k in _context_keys()})
        if codes:
            sel = (frames >= int(f_start)) & (frames < int(f_end))
            offsets = frames[sel] - int(f_start)
            for name, values in codes.items():
                entry.update(emb.summarise(
                    name, values[sel], offsets, int(f_end) - int(f_start)))
        rows.append(entry)
    return rows


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def movie_codes(cfg: Config, movie_dir: Path, tracks: pd.DataFrame, embeddings: dict
                ) -> dict[str, np.ndarray]:
    """PCA codes row-aligned with ``tracks``, for ``{name: (spec, pca)}``."""
    return {
        name: emb.to_codes(pca, emb.detection_vectors(cfg, spec, movie_dir, tracks),
                           name, movie_dir.name)
        for name, (spec, pca) in embeddings.items()
    }


def build_clip_features(cfg: Config, verbose: bool = True, reporter=None,
                        write: bool = True) -> pd.DataFrame:
    """Compute (and cache) one feature row per clip across every prepared movie.

    Fits one PCA per declared embedding over every detection and summarises each
    clip. With ``write`` the table and the bases are saved; the table is stamped
    with the basis hashes and ``context_frames`` it was built with.
    """
    from ..progress import NullReporter

    reporter = reporter or NullReporter()
    work = cfg.paths.work_dir
    movie_dirs = _movie_dirs(work)
    if not movie_dirs:
        raise FileNotFoundError(f"no prepared movies under {work}; run `trackanno prepare` first")
    # Before any input is read, so a file changing mid-build leaves a stale key.
    key = feature_cache_key(cfg)

    manifests, tracks_by_movie = [], {}
    for d in movie_dirs:
        man = pd.read_parquet(d / MANIFEST_FILE)
        if man.empty:
            continue
        manifests.append(man)
        tracks_by_movie[d.name] = pd.read_parquet(d / TRACKS_FILE).reset_index(drop=True)

    manifest = pd.concat(manifests, ignore_index=True)

    specs = emb.active(cfg)
    bases = emb.fit_bases(cfg, {
        spec.name: {
            movie_id: (len(tracks), lambda s=spec, m=movie_id, t=tracks:
                       emb.detection_vectors(cfg, s, work / m, t))
            for movie_id, tracks in tracks_by_movie.items()
        }
        for spec in specs
    }, reporter)
    inputs = {spec.name: (spec, bases[spec.name]) for spec in specs}

    rows: list[dict] = []
    reporter.stage("clip features", len(manifest))
    for movie_id, man in manifest.groupby("movie_id", sort=True):
        tracks = tracks_by_movie[movie_id]
        # One movie's codes at a time.
        codes = movie_codes(cfg, work / movie_id, tracks, inputs)
        fps = float(json.loads((work / movie_id / "cache.json").read_text())["movie"]["fps"])
        by_track = {t: df for t, df in tracks.groupby("track_id", sort=False)}

        # Grouped by TRACK: the series and the sliding transform are built once
        # per track and each window is a slice.
        by_spec: dict[int, list] = {}
        for spec in man.itertuples(index=False):
            by_spec.setdefault(int(spec.track_id), []).append(spec)

        for track_id, specs in by_spec.items():
            reporter.raise_if_cancelled()
            tdf = by_track.get(track_id)
            if tdf is None:
                reporter.advance(len(specs))
                continue
            tdf = tdf.sort_values("frame")
            frames = tdf["frame"].to_numpy(dtype=int)
            rows_idx = tdf.index.to_numpy()
            windows = [(int(s.f_start), int(s.f_end)) for s in specs]
            measured = window_features(
                tdf, cfg, fps, windows, {n: c[rows_idx] for n, c in codes.items()})

            for spec, features in zip(specs, measured):
                reporter.advance()
                if not ((frames >= spec.f_start) & (frames < spec.f_end)).any():
                    continue
                entry = {
                    "clip_id": spec.clip_id,
                    "movie_id": movie_id,
                    "track_id": int(spec.track_id),
                    "f_start": int(spec.f_start),
                }
                entry.update(features)
                rows.append(entry)

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("no clip features could be built")
    # NaN means "undefined"; both HistGradientBoosting and CatBoost handle it.
    out = out.replace([np.inf, -np.inf], np.nan)
    out["_feature_version"] = FEATURE_VERSION
    out["_context_frames"] = int(cfg.suggest.context_frames)
    out["_embedding_basis"] = json.dumps(
        {n: emb.basis_hash(p) for n, p in bases.items()}, sort_keys=True)
    if write:
        key_path = work / CLIP_FEATURES_KEY_FILE
        key_path.unlink(missing_ok=True)
        out.to_parquet(work / CLIP_FEATURES_FILE, index=False)
        emb.save_bases(work / emb.BASES_FILE, bases)
        key_path.write_text(key, encoding="utf-8")

    n_emb = len([c for c in out.columns if c.startswith("emb_")])
    n_ctx = len([c for c in out.columns if c.startswith(CONTEXT_PREFIX)])
    n_kin = len([c for c in out.columns if c not in META_COLUMNS]) - n_emb - n_ctx
    summary = (f"{len(out)} clips x {n_kin} kinematic + {n_ctx} context + "
               f"{n_emb} embedding features -> {CLIP_FEATURES_FILE}")
    reporter.log(summary)
    if verbose:
        print(f"  {summary}")
    return out


def _movie_dirs(work: Path) -> list[Path]:
    return sorted(p for p in work.iterdir() if p.is_dir() and (p / MANIFEST_FILE).exists())


def _plain(value):
    """``value`` as JSON-ready data, for hashing config sections."""
    import dataclasses

    if dataclasses.is_dataclass(value):
        return _plain(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _stat(path: Path) -> list | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return [st.st_size, st.st_mtime_ns]


def feature_cache_key(cfg: Config) -> str:
    """Digest of everything the clip feature table depends on.

    The feature-relevant config, the schema and PCA constants, and the size and
    mtime of each movie's manifest, tracks, metadata and embedding inputs.
    Labels are not part of it.
    """
    import hashlib

    work = cfg.paths.work_dir
    specs = cfg.suggest.embeddings
    movies = {}
    for d in _movie_dirs(work):
        files = {name: _stat(d / name) for name in
                 (MANIFEST_FILE, TRACKS_FILE, "cache.json", emb.DETECTOR_FILE)}
        for spec in specs:
            if spec.source == "file":
                files[f"emb:{spec.name}"] = _stat(emb._file_path(cfg, spec, d.name))
        movies[d.name] = files
    state = {
        "feature_version": FEATURE_VERSION,
        "pca_max_rows": emb.PCA_MAX_ROWS,
        "clips": _plain(cfg.clips),
        "spectral": [_plain(cfg.spectral), SPECTRAL_HOP, FLOW_FRAMES],
        "fixed": {k: v for k, v in FIXED_SETTINGS.items() if k.startswith("clips.")},
        "pose": [bool(cfg.is_pose), int(cfg.n_keypoints)],
        "detector_features": bool(cfg.features.enabled),
        "suggest": {
            "pca_components": int(emb.PCA_COMPONENTS),
            "seed": int(emb.SEED),
            "context_frames": int(cfg.suggest.context_frames),
            "embeddings": _plain(specs),
        },
        "movies": movies,
    }
    text = json.dumps(state, sort_keys=True, default=str)
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def cached_clip_features(cfg: Config) -> pd.DataFrame | None:
    """The table on disk if it was built under the current key, else None.

    Also None when the saved embedding bases no longer match the table's basis
    stamp, since a model fitted on the table would carry the wrong bases.
    """
    work = cfg.paths.work_dir
    table, key_path = work / CLIP_FEATURES_FILE, work / CLIP_FEATURES_KEY_FILE
    if not table.exists() or not key_path.exists():
        return None
    try:
        if key_path.read_text(encoding="utf-8").strip() != feature_cache_key(cfg):
            return None
        features = pd.read_parquet(table)
        stamped = {}
        if len(features) and "_embedding_basis" in features.columns:
            stamped = json.loads(str(features["_embedding_basis"].iloc[0]))
        saved = {n: emb.basis_hash(p)
                 for n, p in emb.load_bases(work / emb.BASES_FILE).items()}
    except Exception:                                # noqa: BLE001
        return None
    if stamped != saved:
        return None
    return features


# Bump whenever a feature's MEANING changes, not just when one is added or
# removed. `Suggester.load`'s column check catches a renamed or dropped column
# but not a redefined one, and a model whose inputs changed meaning under
# unchanged names loads, reports itself ready, and predicts confidently.
#
# 1  the original per-clip features
# 2  spectral columns measured on a centred window (config `spectral`); no
#    significance gating; straightness bounded by construction; lag-based turn
#    rates; rotation-invariant shape_*; circular static mode as cos/sin
# 3  censored windows excluded from wave_freq_drift_hz / wave_freq_track_sd_hz;
#    tip_angular_speed no longer bridges gaps
# 4  wave_flow, wave_flow_coherence, wave_flow_direction: the wave direction
#    from the phase gradient over FLOW_FRAMES, needing no resolvable
#    beat. Added beside the spectral direction, which is unchanged
# 5  cnn* replaced by emb_<name>_* from one project-wide PCA per embedding;
#    optional ctx_* context family
FEATURE_VERSION = 5

# Carried in the table, never fed to a model: `n_windows` and `centre_offset`
# describe HOW the series was read out, not the cell, so a model given them
# learns the readout grid. `f_end`, `n_present` and `skip_reason` say where a
# dense window sits and what could be measured in it, and `skip_reason` is a
# string, so a caller taking "every feature column" must not be handed them.
META_COLUMNS = (
    "clip_id", "movie_id", "track_id", "f_start", "f_end",
    "_feature_version", "_context_frames", "_embedding_basis", "n_windows", "centre_offset", "n_present", "skip_reason",
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
    "context": {
        "needs": "context",
        "label": "longer context",
        "about": "The trajectory and box features again, over "
                 "suggest.context_frames centred on the window.",
        "prefixes": (CONTEXT_PREFIX,),
    },
}


def group_of(column: str) -> str | None:
    """Which feature group a column belongs to, or None if it is metadata."""
    if column in META_COLUMNS:
        return None
    match = emb.EMBEDDING_COLUMN.match(column)
    if match:
        return f"emb_{match['name']}"
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
        elif need == "context" and int(cfg.suggest.context_frames) <= 0:
            out[name] = "needs suggest.context_frames above 0"
        else:
            out[name] = ""
    for spec in cfg.suggest.embeddings:
        out[spec.group] = ("needs features.enabled: true"
                           if spec.source == "detector" and not cfg.features.enabled
                           else "")
    return out


def known_columns(cfg: Config | None = None) -> list[str]:
    """Every column a built feature table could hold, without reading one.

    Without ``cfg`` the context family is included. Embedding columns are an
    upper bound: the PCA may keep fewer than ``embeddings.PCA_COMPONENTS``.
    """
    columns = list(feature_schema())
    if cfg is None or int(cfg.suggest.context_frames) > 0:
        columns += context_names()
    if cfg is not None:
        for spec in cfg.suggest.embeddings:
            columns += emb.column_names(spec.name, emb.PCA_COMPONENTS)
    return columns


def feature_columns(frame: pd.DataFrame, groups=None) -> list[str]:
    """The model's input columns, optionally restricted to certain groups.

    Restricting is a modelling decision: embedding columns can fingerprint
    individuals, so a behaviour model may want them out.
    """
    columns = [c for c in frame.columns if c not in META_COLUMNS]
    if groups is None:
        return columns
    wanted = set(groups)
    return [c for c in columns if group_of(c) in wanted]
