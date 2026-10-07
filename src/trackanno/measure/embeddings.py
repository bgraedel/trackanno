"""Per-detection embedding vectors and their clip-level summary.

An embedding source supplies one vector per detection, keyed to
``(movie_id, track_id, frame)``. ``detector`` reads ``instance_features.npy``,
row-aligned with ``tracks.parquet``; ``file`` reads a table per movie with
``track_id``, ``frame`` and numeric vector columns.

One PCA per embedding is fitted over every detection of the project, without
labels, when the clip features are built. Clip features and dense windows both
summarise the codes through :func:`summarise`, so training, cross-validation and
prediction see the same columns.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import FIXED_SETTINGS

__all__ = [
    "active", "basis_hash", "BASES_FILE", "DETECTOR_FILE", "column_names",
    "detection_vectors",
    "fit_bases", "load_bases", "save_bases", "to_codes", "summarise",
    "EMBEDDING_COLUMN",
]

BASES_FILE = "embedding_pca.joblib"
DETECTOR_FILE = "instance_features.npy"
# Rows the PCA is fitted on at most, drawn with SEED.
PCA_MAX_ROWS = 200_000
PCA_COMPONENTS = FIXED_SETTINGS["suggest.pca_components"]   # width of each embedding
SEED = FIXED_SETTINGS["suggest.seed"]
STATS = ("mean", "std", "delta")
EMBEDDING_COLUMN = re.compile(r"^emb_(?P<name>[A-Za-z0-9]+)_(?P<i>\d+)_(?P<stat>mean|std|delta)$")


def active(cfg) -> list:
    """Declared embeddings this project can compute (detector needs features.enabled)."""
    return [e for e in cfg.suggest.embeddings
            if e.source != "detector" or cfg.features.enabled]


def names_in(columns) -> list[str]:
    """Embedding names referenced by ``emb_*`` columns, in first-seen order."""
    out: list[str] = []
    for c in columns:
        m = EMBEDDING_COLUMN.match(str(c))
        if m and m["name"] not in out:
            out.append(m["name"])
    return out


def basis_hash(pca) -> str:
    """A short digest of a fitted PCA's basis and centre."""
    import hashlib

    data = (np.ascontiguousarray(pca.components_, dtype=np.float64).tobytes()
            + np.ascontiguousarray(pca.mean_, dtype=np.float64).tobytes())
    return hashlib.blake2b(data, digest_size=8).hexdigest()


def column_names(name: str, n_components: int) -> list[str]:
    """The columns one embedding contributes, ``emb_<name>_<i>_<stat>``."""
    return [f"emb_{name}_{i}_{s}" for s in STATS for i in range(int(n_components))]


def _file_path(cfg, spec, movie_id: str) -> Path:
    path = Path(str(spec.path).replace("{movie_id}", movie_id))
    return path if path.is_absolute() else cfg.path.parent / path


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def detection_vectors(cfg, spec, movie_dir: Path, tracks: pd.DataFrame) -> np.ndarray:
    """``(len(tracks), D)`` vectors row-aligned with ``tracks``, NaN where missing.

    ``tracks`` is the movie's full ``tracks.parquet`` table. A movie with no
    vectors at all is refused; single missing detections become NaN rows. The
    detector file is memory-mapped.
    """
    movie_id = movie_dir.name
    if spec.source == "detector":
        path = movie_dir / DETECTOR_FILE
        if not path.exists():
            raise RuntimeError(
                f"{movie_id}: embedding {spec.name!r} reads {DETECTOR_FILE}, which "
                "is missing. Re-prepare the movie with features.enabled, or drop "
                "the embedding."
            )
        vectors = np.load(path, mmap_mode="r")
        if len(vectors) != len(tracks):
            raise RuntimeError(
                f"{movie_id}: {DETECTOR_FILE} has {len(vectors)} rows but "
                f"tracks.parquet has {len(tracks)}; features are not row-aligned"
            )
        return vectors

    path = _file_path(cfg, spec, movie_id)
    if not path.exists():
        raise RuntimeError(
            f"{movie_id}: embedding {spec.name!r} has no file at {path}"
        )
    table = _read_table(path)
    missing = [c for c in ("track_id", "frame") if c not in table.columns]
    if missing:
        raise RuntimeError(f"{path}: needs columns track_id and frame, missing {missing}")
    if table[["track_id", "frame"]].isna().any().any():
        raise RuntimeError(f"{path}: track_id or frame is empty on some rows")
    # A saved pandas index is not part of the vector.
    vector_cols = [
        c for c in table.columns
        if c not in ("track_id", "frame", "movie_id")
        and not re.match(r"^(Unnamed|__index_level_)", str(c))
        and pd.api.types.is_numeric_dtype(table[c])
    ]
    if not vector_cols:
        raise RuntimeError(f"{path}: no numeric vector columns besides track_id and frame")
    keys = table[["track_id", "frame"]].astype(np.int64)
    if keys.duplicated().any():
        raise RuntimeError(f"{path}: more than one vector for some (track_id, frame)")
    values = table[vector_cols].to_numpy(dtype=np.float32)
    index = pd.MultiIndex.from_frame(keys)
    wanted = pd.MultiIndex.from_frame(tracks[["track_id", "frame"]].astype(np.int64))
    where = index.get_indexer(wanted)
    out = np.full((len(tracks), len(vector_cols)), np.nan, dtype=np.float32)
    hit = where >= 0
    if len(tracks) and not hit.any():
        raise RuntimeError(
            f"{path}: none of its {len(table)} rows match a detection of {movie_id} "
            "by (track_id, frame). Frames are 0-based as in tracks.parquet, and "
            "track ids change when detection is re-run."
        )
    out[hit] = values[where[hit]]
    return out


def _finite_rows(vectors: np.ndarray) -> np.ndarray:
    if np.issubdtype(vectors.dtype, np.floating):
        return np.isfinite(vectors).all(axis=1)
    return np.ones(len(vectors), dtype=bool)


def fit_bases(cfg, vectors: dict, reporter=None) -> dict:
    """``{name: PCA}``, one per embedding, fitted over every detection.

    ``vectors`` maps embedding name to ``{movie_id: source}``, where a source is
    an ``(n, D)`` array or ``(n_rows, load)`` with ``load()`` returning one. Movies
    are loaded one at a time; above PCA_MAX_ROWS each contributes a seeded
    sample in proportion to its rows.
    """
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(SEED)
    bases = {}
    for name, by_movie in vectors.items():
        sources = {m: v if isinstance(v, tuple) else (len(v), lambda v=v: v)
                   for m, v in by_movie.items()}
        total = sum(n for n, _ in sources.values())
        parts, widths = [], {}
        for movie_id in sorted(sources):
            n_rows, load = sources[movie_id]
            block = load()
            widths[movie_id] = int(block.shape[1])
            rows = np.flatnonzero(_finite_rows(block))
            if total > PCA_MAX_ROWS:
                take = int(round(PCA_MAX_ROWS * n_rows / total))
                rows = np.sort(rng.choice(rows, size=min(take, len(rows)), replace=False))
            parts.append(np.asarray(block[rows], dtype=np.float32))
            del block
        if len(set(widths.values())) > 1:
            detail = ", ".join(f"{m} {w}-d" for m, w in sorted(widths.items()))
            raise RuntimeError(
                f"embedding {name!r} has vectors of different widths across movies "
                f"({detail}). A detector embedding follows the model's channel "
                "widths, so re-prepare every movie with the same weights."
            )
        stacked = np.concatenate(parts, axis=0)
        if not len(stacked):
            raise RuntimeError(f"embedding {name!r} has no complete vector in any movie")
        n_comp = int(min(PCA_COMPONENTS, stacked.shape[0], stacked.shape[1]))
        pca = PCA(n_components=n_comp, random_state=SEED).fit(stacked)
        bases[name] = pca
        if reporter is not None:
            reporter.log(
                f"embedding {name}: PCA over {stacked.shape[0]} detections x "
                f"{stacked.shape[1]}-d -> {n_comp} components "
                f"({float(pca.explained_variance_ratio_.sum()):.1%} variance)"
            )
    return bases


def save_bases(path: Path, bases: dict) -> None:
    import joblib

    if bases:
        joblib.dump(bases, path)
    else:
        path.unlink(missing_ok=True)


def load_bases(path: Path) -> dict:
    import joblib

    return dict(joblib.load(path)) if path.exists() else {}


def to_codes(pca, vectors: np.ndarray, name: str = "", movie_id: str = "") -> np.ndarray:
    """PCA codes for every row, NaN where the vector is incomplete."""
    if vectors.shape[1] != int(pca.n_features_in_):
        raise RuntimeError(
            f"{movie_id}: embedding {name!r} is {vectors.shape[1]}-d but its saved "
            f"PCA was fitted on {int(pca.n_features_in_)}-d vectors. Re-prepare "
            "with the same source, or rebuild the clip features and retrain."
        )
    out = np.full((len(vectors), int(pca.n_components_)), np.nan, dtype=np.float64)
    ok = _finite_rows(vectors)
    if ok.any():
        out[ok] = pca.transform(vectors[ok].astype(np.float32))
    return out


def summarise(name: str, codes: np.ndarray, offsets: np.ndarray, span: int) -> dict[str, float]:
    """Mean, std and grid delta of one window's codes; NaN where no row has one."""
    from .clipfeatures import grid_delta

    k = codes.shape[1]
    ok = np.isfinite(codes).all(axis=1) if len(codes) else np.zeros(0, dtype=bool)
    if not ok.any():
        nan = np.full(k, np.nan)
        mean = std = delta = nan
    else:
        good = codes[ok]
        mean, std = good.mean(axis=0), good.std(axis=0)
        offsets = np.asarray(offsets, dtype=int)[ok]
        inside = (offsets >= 0) & (offsets < int(span))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            delta = grid_delta(good[inside], offsets[inside], int(span))
    out = {}
    for stat, values in (("mean", mean), ("std", std), ("delta", delta)):
        out.update({f"emb_{name}_{i}_{stat}": float(v) for i, v in enumerate(values)})
    return out
