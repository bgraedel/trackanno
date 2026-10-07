"""Clip-level features for the suggestion model: kinematic, context and embedding columns."""

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

# Rates are per second so movies at different frame rates stay comparable.
IMMOTILE_PX_PER_SEC = 50.0
LOCALISATION_NOISE_PX = 1.8   # measured per-frame keypoint jitter
BEAT_NFFT = 128               # fixed grid so beat frequency is comparable across clips
BEAT_F_MIN_HZ = 1.0
# The window must span this many cycles of a frequency to resolve it.
BEAT_MIN_CYCLES = 3.0
# Above this interpolated share a spectral estimate is NaN (computeGoodFrameRange.m).
MAX_INTERP_FRAC = 0.20

_BEAT_FEATURES = (
    "beat_amp_mean", "beat_amp_max", "beat_speed_mean", "beat_amp_distal",
    "beat_freq_hz", "beat_power_frac", "beat_excess", "beat_resolvable_hz",
    "beat_censored",
)


_SCHEMA_CACHE: tuple[str, ...] | None = None


def feature_schema() -> tuple[str, ...]:
    """Every column a feature row carries, whatever the task or data."""
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
    """``value`` as a float, or ``default`` (NaN) when it is not finite."""
    return float(value) if np.isfinite(value) else default


def _on_grid(sub: pd.DataFrame, columns: list[str], n_frames: int, f_start: int) -> np.ndarray:
    """Lay a track's rows onto the clip frame grid, NaN where undetected."""
    out = np.full((n_frames, len(columns)), np.nan, dtype=float)
    idx = sub["frame"].to_numpy(dtype=int) - f_start
    keep = (idx >= 0) & (idx < n_frames)
    out[idx[keep]] = sub[columns].to_numpy(dtype=float)[keep]
    return out


def _interp_nan(values: np.ndarray) -> np.ndarray:
    """Linearly fill NaNs along time for spectral estimation, as ``interpInTime.m``."""
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
    """Mean absolute step per column over adjacent frames only; zeros where no pair exists.

    ``offsets`` are the rows' frame offsets within ``[0, span)``.
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
    """Dominant frequency (Hz) of body-frame lateral displacement, with excess and censoring."""
    out = {"beat_freq_hz": np.nan, "beat_power_frac": np.nan, "beat_excess": np.nan}
    if lateral.ndim != 2 or lateral.shape[0] < 8 or lateral.shape[1] == 0:
        return out

    sig = _interp_nan(lateral)
    n = sig.shape[0]
    t = np.arange(n)
    # Remove per-keypoint linear drift.
    slope, intercept = np.polyfit(t, sig, 1)
    sig -= t[:, None] * slope + intercept

    # A motionless signal has no beat.
    if not np.isfinite(sig).any() or float(np.nanmax(np.abs(sig))) < 1e-9:
        return out

    window = np.hanning(n)[:, None]
    # BEAT_NFFT is a floor: rfft would truncate a longer signal.
    nfft = max(BEAT_NFFT, n)
    spec = np.abs(np.fft.rfft(sig * window, n=nfft, axis=0)) ** 2        # power
    power = spec.sum(axis=1)                                            # pool keypoints
    freqs = np.fft.rfftfreq(nfft, d=1.0 / max(fps, 1e-6))

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
    # Peak at the band floor: an upper bound. Tolerance is half a native resolution element.
    out["beat_censored"] = float(
        band_freqs[peak] < f_floor + 0.5 * max(fps, 1e-6) / n
    )
    return out


def _path_length(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Path length and net displacement over present positions, so ``path >= net``."""
    present = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
    if present.size < 2:
        return np.nan, np.nan
    px_, py_ = x[present], y[present]
    path = float(np.hypot(np.diff(px_), np.diff(py_)).sum())
    net = float(np.hypot(px_[-1] - px_[0], py_[-1] - py_[0]))
    return path, net


def _heading_lag(x: np.ndarray, y: np.ndarray, n_frames: int) -> int:
    """Smallest frame lag whose median displacement clears the localisation noise floor."""
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
    """Flagellum motion in the body frame (Base node 1 to Head node 0)."""
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

    # Mask rather than drop rows, keeping uniform sampling.
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

    out["beat_amp_distal"] = _safe(
        np.sqrt(np.nanmean((ry[:, -1] - np.nanmean(ry[:, -1])) ** 2))
    )

    # Unwrap on the filled grid; keep only steps between two detected frames.
    ang = np.unwrap(_interp_nan(axis[:, None])[:, 0])
    turn = np.diff(ang)[valid[1:] & valid[:-1]]
    out["body_rotation_speed"] = _safe(np.nanmean(np.abs(turn)) / dt) if turn.size else np.nan

    # For the spectral estimator, unless too much would be interpolated.
    if (1.0 - valid.mean()) <= MAX_INTERP_FRAC:
        out["_lateral"] = ry
    return out


# --------------------------------------------------------------------------- #
# per-track measurement
# --------------------------------------------------------------------------- #
class _KeypointGrid:
    """One track's keypoints on its full frame grid, built once and sliced per window."""

    __slots__ = ("px", "py", "present", "first", "last")

    def __init__(self, tdf: pd.DataFrame, n_keypoints: int):
        frames = tdf["frame"].to_numpy(dtype=int)
        self.first, self.last = int(frames.min()), int(frames.max()) + 1
        n = self.last - self.first
        kx = [f"kp_x_{k}" for k in range(n_keypoints)]
        ky = [f"kp_y_{k}" for k in range(n_keypoints)]
        self.px = _on_grid(tdf, kx, n, self.first)
        self.py = _on_grid(tdf, ky, n, self.first)
        # Frames with a row, even if its keypoints are all NaN.
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
    """Beat features on a ``spectral.window`` span centred on the clip, as the waveform series."""
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
    # body_rotation_speed comes from trackseries, over the clip.
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
    """Feature rows for several (possibly overlapping) windows of one frame-sorted track.

    ``codes`` maps an embedding name to PCA codes row-aligned with ``tdf``.
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

    grid = _KeypointGrid.maybe(tdf, cfg)
    waves = wavespectra.WindowSource(spectra) if spectra is not None else None

    rows = []
    for f_start, f_end in windows:
        entry = trackseries.aggregate_window(series, int(f_start), int(f_end), fps, cfg)
        entry.update(_beat_on_window(grid, cfg, fps, int(f_start), int(f_end)))
        if waves is not None and len(spectra):
            entry.update(waves.at(int(f_start), int(f_end)))
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
    """One feature row per clip over every prepared movie; ``write`` saves table and bases."""
    from ..progress import NullReporter

    reporter = reporter or NullReporter()
    work = cfg.paths.work_dir
    movie_dirs = _movie_dirs(work)
    if not movie_dirs:
        raise FileNotFoundError(f"no prepared movies under {work}; run `trackanno prepare` first")
    # Keyed before reading, so a file changing mid-build leaves a stale key.
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
        codes = movie_codes(cfg, work / movie_id, tracks, inputs)
        fps = float(json.loads((work / movie_id / "cache.json").read_text())["movie"]["fps"])
        by_track = {t: df for t, df in tracks.groupby("track_id", sort=False)}

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
    """Digest of the feature config and each movie's input files (not the labels)."""
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
    """The cached table if its key and embedding bases are current, else None."""
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


# Bump whenever a feature's meaning changes, not only when columns change.
FEATURE_VERSION = 5

# Carried in the table, never fed to a model.
META_COLUMNS = (
    "clip_id", "movie_id", "track_id", "f_start", "f_end",
    "_feature_version", "_context_frames", "_embedding_basis", "n_windows", "centre_offset", "n_present", "skip_reason",
)

# `needs` lets the GUI say why a group is unavailable.
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
    """Every column a feature table could hold (embedding columns are an upper bound)."""
    columns = list(feature_schema())
    if cfg is None or int(cfg.suggest.context_frames) > 0:
        columns += context_names()
    if cfg is not None:
        for spec in cfg.suggest.embeddings:
            columns += emb.column_names(spec.name, emb.PCA_COMPONENTS)
    return columns


def feature_columns(frame: pd.DataFrame, groups=None) -> list[str]:
    """The model's input columns, optionally restricted to certain groups."""
    columns = [c for c in frame.columns if c not in META_COLUMNS]
    if groups is None:
        return columns
    wanted = set(groups)
    return [c for c in columns if group_of(c) in wanted]
