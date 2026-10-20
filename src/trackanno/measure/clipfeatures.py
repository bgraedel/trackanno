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
# Below this a cell is not translocating; in px/s, so it depends on magnification (about 3 um/s
# at 0.325 um/px), and well above the 1-2.5 px/s that keypoint jitter leaves in the lagged speed.
IMMOTILE_PX_PER_SEC = 10.0
# Measured about 0.3 px on body points and 0.5-0.8 px on flagellum points.
LOCALISATION_NOISE_PX = 0.5
KINEMATIC_LAG_S = 0.1         # rates are changes over this span, not per frame
BEAT_F_MIN_HZ = 1.0
# The window must span this many cycles of a frequency to resolve it (validated: <1% error).
BEAT_MIN_CYCLES = 2.5

_SCHEMA_CACHE: tuple[str, ...] | None = None


def feature_schema() -> tuple[str, ...]:
    """Every column a feature row carries, whatever the task or data."""
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is None:
        from . import instantaneous, trackseries, wavespectra

        _SCHEMA_CACHE = tuple(dict.fromkeys(
            tuple(trackseries.feature_names())
            + tuple(wavespectra.SERIES_COLUMNS)
            + tuple(instantaneous.CLIP_COLUMNS)
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


# --------------------------------------------------------------------------- #
# per-track measurement
# --------------------------------------------------------------------------- #
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
    from . import instantaneous, trackseries, wavespectra

    frames = tdf["frame"].to_numpy(dtype=int)
    first, last = int(frames.min()), int(frames.max()) + 1
    n_context = int(cfg.suggest.context_frames)
    lo = min(int(frames.min()), min(int(w[0]) for w in windows))
    hi = max(int(frames.max()) + 1, max(int(w[1]) for w in windows))
    series = trackseries.build_series(tdf, cfg, fps, lo, hi - lo)

    spectra = beats = None
    if cfg.is_pose:
        spectra = wavespectra.track_spectral_series(
            tdf, cfg.n_keypoints, fps,
            window=cfg.spectral.window, hop=SPECTRAL_HOP,
            flow_frames=FLOW_FRAMES,
        )
        beats = instantaneous.track_beat_series(tdf, cfg.n_keypoints, fps)

    waves = wavespectra.WindowSource(spectra) if spectra is not None else None

    rows = []
    for f_start, f_end in windows:
        entry = trackseries.aggregate_window(series, int(f_start), int(f_end), fps, cfg)
        if waves is not None and len(spectra):
            entry.update(waves.at(int(f_start), int(f_end)))
        if beats is not None:
            entry.update(instantaneous.window_values(beats, int(f_start), int(f_end)))
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
FEATURE_VERSION = 10

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
        "about": "Body and flagellum length, keypoint shape, the tip's angle to the body, "
                 "and translocation and rotation of the cell body rather than the box "
                 "centroid. Needs keypoints.",
        "prefixes": (
            "kpt_", "body_", "flagellum_", "tip_", "shape_",
        ),
    },
    "waveform": {
        "needs": "pose",
        "label": "beat waveform",
        "about": "The beat as the flagella-analysis-pipeline (Fochler et al. 2025) measures it: "
                 "frequency, amplitude and phase profiles, spatial and angular amplitude, average "
                 "angle and fit quality, plus wavelength and direction. Needs keypoints.",
        "prefixes": ("wave_",),
    },
    "beat": {
        "needs": "pose",
        "label": "beat per frame",
        "about": "The beat read in every frame from a wavelet transform of the flagellum's "
                 "tangent angle, then summarised over the clip: frequency and its spread, "
                 "direction, amplitude and phase profiles, mean angle and curvature. "
                 "Needs keypoints.",
        "prefixes": ("inst_",),
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
