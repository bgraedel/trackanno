"""Dense feature table: ``clips.frames``-long windows every ``predict.window_stride`` frames.

Features come from ``clipfeatures.window_features`` with the saved embedding bases;
windows too sparse to score keep a row with ``skip_reason``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..measure import embeddings as emb
from ..measure.clipfeatures import (
    CLIP_FEATURES_FILE, FEATURE_VERSION, movie_codes, window_features,
)
from ..annotate.clips import (
    HEADER_FILE, TRACKS_FILE, StagedWrite, border_flags,
    clip_id_for, too_much_border,
)
from ..config import Config

__all__ = [
    "DENSE_FEATURES_FILE", "DENSE_META_COLUMNS", "window_length",
    "window_stride", "window_grid", "plan_dense_windows", "count_windows",
    "build_dense_features", "dense_features_path",
]

DENSE_FEATURES_FILE = "dense_clip_features.parquet"

# Metadata columns only the dense table carries; never model inputs.
DENSE_META_COLUMNS = ("f_end", "n_present", "skip_reason")


def dense_features_path(cfg: Config) -> Path:
    return cfg.paths.work_dir / DENSE_FEATURES_FILE


# --------------------------------------------------------------------------- #
# the grid
# --------------------------------------------------------------------------- #
def window_length(cfg: Config) -> int:
    """Frames per dense window: ``clips.frames``, the length the model was fitted at."""
    return max(1, int(cfg.clips.frames))


def window_stride(cfg: Config) -> int:
    """Frames between one dense window's start and the next."""
    return cfg.predict.resolved_stride(cfg.clips)


def window_grid(
    n_frames: int, length: int, stride: int | None = None
) -> list[tuple[int, int]]:
    """Windows over ``[0, n_frames)``, ``length`` long and ``stride`` (default ``length``) apart.

    Tiling keeps a short tail; overlapping grids end on a window flush with ``n_frames``.
    """
    if n_frames <= 0 or length <= 0:
        return []
    step = length if stride is None else int(stride)
    if step <= 0:
        return []
    if n_frames <= length:
        return [(0, n_frames)]
    if step >= length:
        return [(start, min(start + length, n_frames))
                for start in range(0, n_frames, step)]
    starts = list(range(0, n_frames - length + 1, step))
    if starts[-1] != n_frames - length:
        starts.append(n_frames - length)
    return [(start, start + length) for start in starts]


def plan_dense_windows(
    tracks: pd.DataFrame, n_frames: int, length: int, stride: int | None = None
) -> dict[int, list[tuple[int, int]]]:
    """``{track_id: windows}``: every grid window between each track's first and last detection."""
    grid = window_grid(n_frames, length, stride)
    out: dict[int, list[tuple[int, int]]] = {}
    if tracks.empty or not grid:
        return out
    starts = np.array([w[0] for w in grid], dtype=np.int64)
    ends = np.array([w[1] for w in grid], dtype=np.int64)
    for track_id, tdf in tracks.groupby("track_id", sort=True):
        frames = tdf["frame"].to_numpy(dtype=np.int64)
        if frames.size == 0:
            continue
        # The contiguous run of windows intersecting [first, last].
        lo = int(np.searchsorted(ends, int(frames.min()), side="right"))
        hi = int(np.searchsorted(starts, int(frames.max()), side="right")) - 1
        chosen = grid[lo : hi + 1]
        if chosen:
            out[int(track_id)] = chosen
    return out


def _movie_frames(header_frames: int, tracks: pd.DataFrame) -> int:
    """Frames the grid spans: the header's count or past the last detection, whichever is larger."""
    if tracks.empty:
        return int(header_frames)
    return max(int(header_frames), int(tracks["frame"].max()) + 1)


def _movie_size(header: dict) -> tuple[int, int] | None:
    """``(height, width)`` out of a ``cache.json`` body, or None if it is absent."""
    movie = (header or {}).get("movie") or {}
    try:
        return int(movie["height"]), int(movie["width"])
    except (KeyError, TypeError, ValueError):
        return None


def _read_header(movie_dir: Path) -> dict:
    """``cache.json``, or ``{}`` where a project is only half prepared."""
    header = movie_dir / HEADER_FILE
    if not header.exists():
        return {}
    try:
        return json.loads(header.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _without_border_rows(tracks: pd.DataFrame, header: dict, cfg: Config
                         ) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """``tracks`` without rows near the frame edge, and the dropped frames per track."""
    if int(cfg.tracking.border_margin_px) <= 0 or _movie_size(header) is None:
        return tracks, {}
    flags = border_flags(tracks, _movie_size(header), cfg).to_numpy()
    if not flags.any():
        return tracks, {}
    edge = tracks[flags]
    return tracks[~flags], {
        int(t): g["frame"].to_numpy(dtype=np.int64)
        for t, g in edge.groupby("track_id", sort=False)
    }


def _n_border(border_frames: np.ndarray | None, f_start: int, f_end: int) -> int:
    """How many of a track's excluded frames fall in one window."""
    if border_frames is None:
        return 0
    return int(np.count_nonzero((border_frames >= f_start) & (border_frames < f_end)))


def _header_frames(header: dict) -> int:
    """What a ``cache.json`` body says the movie holds, or 0 if it cannot say."""
    try:
        return int((header or {})["movie"]["n_frames"])
    except (KeyError, TypeError, ValueError):
        return 0


# --------------------------------------------------------------------------- #
# the saved embedding bases
# --------------------------------------------------------------------------- #
def _load_bases(cfg: Config, reporter) -> dict:
    """``{name: (spec, pca)}`` from the config and the bases on disk. Never fitted."""
    bases = emb.load_bases(cfg.paths.work_dir / emb.BASES_FILE)
    out = {}
    for spec in emb.active(cfg):
        if spec.name in bases:
            out[spec.name] = (spec, bases[spec.name])
        else:
            reporter.log(
                f"embedding {spec.name!r} has no saved basis in {emb.BASES_FILE}; "
                "retrain to add its columns",
                "warning",
            )
    return out


# --------------------------------------------------------------------------- #
# the table
# --------------------------------------------------------------------------- #
def build_dense_features(
    cfg: Config, reporter=None, verbose: bool = False,
    embeddings: dict | None = None, required: list[str] | None = None,
) -> pd.DataFrame:
    """Build and save one feature row per (track, dense window) across every prepared movie.

    ``embeddings`` is the model's ``{name: (spec, PCA)}``; ``required`` columns must be produced.
    """
    from ..progress import NullReporter

    reporter = reporter or NullReporter()
    work = cfg.paths.work_dir
    if not work.exists():
        raise FileNotFoundError(f"no work dir at {work}; run `trackanno prepare` first")
    movie_dirs = sorted(
        p for p in work.iterdir() if p.is_dir() and (p / TRACKS_FILE).exists()
    )
    if not movie_dirs:
        raise FileNotFoundError(f"no prepared movies under {work}; run `trackanno prepare` first")

    length = window_length(cfg)
    stride = window_stride(cfg)
    min_present = int(cfg.predict.min_frames)
    if embeddings is None:
        embeddings = _load_bases(cfg, reporter)

    plans, total = [], 0
    for d in movie_dirs:
        tracks = pd.read_parquet(d / TRACKS_FILE).reset_index(drop=True)
        header = json.loads((d / HEADER_FILE).read_text(encoding="utf-8"))
        # Codes align with the unfiltered table; row labels survive the border filter.
        codes = movie_codes(cfg, d, tracks, embeddings)
        tracks, border_frames = _without_border_rows(tracks, header, cfg)
        windows = plan_dense_windows(
            tracks, _movie_frames(header["movie"]["n_frames"], tracks), length, stride
        )
        total += sum(len(spans) for spans in windows.values())
        plans.append((d, tracks, codes, header, windows, border_frames))

    reporter.log(
        f"planning {total} dense window(s) of {length} frame(s) at stride "
        f"{stride} over {len(plans)} movie(s)"
    )
    reporter.stage("dense clip features", total)
    rows: list[dict] = []
    # Distinct detected frames reached by any window, and by a featurised one.
    covered = usable = 0
    for d, tracks, codes, header, windows, border_frames in plans:
        movie_id = d.name
        fps = float(header["movie"]["fps"])
        by_track = {int(t): df for t, df in tracks.groupby("track_id", sort=False)}

        for track_id in sorted(windows):
            tdf = by_track[track_id].sort_values("frame")
            frames = tdf["frame"].to_numpy(dtype=np.int64)
            # The same entry point as build_clip_features, once per track.
            spans = windows[track_id]
            measured = window_features(
                tdf, cfg, fps, spans,
                {n: c[tdf.index.to_numpy()] for n, c in codes.items()})
            reached = np.zeros(frames.size, dtype=bool)
            featurised = np.zeros(frames.size, dtype=bool)
            for (f_start, f_end), features in zip(spans, measured):
                reporter.raise_if_cancelled()
                reporter.advance()
                sel = (frames >= f_start) & (frames < f_end)
                reached |= sel
                n_present = int(sel.sum())
                entry = {
                    "clip_id": clip_id_for(movie_id, track_id, f_start),
                    "movie_id": movie_id,
                    "track_id": int(track_id),
                    "f_start": int(f_start),
                    "f_end": int(f_end),
                    "n_present": n_present,
                    "skip_reason": "",
                }
                n_border = _n_border(border_frames.get(track_id), f_start, f_end)
                if too_much_border(n_border, n_present, cfg):
                    entry["skip_reason"] = (
                        f"{n_border} of {n_border + n_present} detection(s) in "
                        f"frames {f_start}-{f_end} are within "
                        f"tracking.border_margin_px "
                        f"({cfg.tracking.border_margin_px}px) of the frame edge"
                    )
                    rows.append(entry)
                    continue
                if n_present < min_present:
                    entry["skip_reason"] = (
                        f"only {n_present} detection(s) in frames {f_start}-{f_end}; "
                        f"a window needs {min_present}"
                    )
                    rows.append(entry)
                    continue

                featurised |= sel
                entry.update(features)
                rows.append(entry)
            covered += int(reached.sum())
            usable += int(featurised.sum())

    out = pd.DataFrame(rows)
    if not out.empty:
        out["_feature_version"] = FEATURE_VERSION
    if out.empty:
        raise RuntimeError(
            "no dense windows could be planned; every prepared movie has an empty "
            "tracks.parquet"
        )
    out = out.replace([np.inf, -np.inf], np.nan)
    absent = [c for c in (required or ()) if c not in out.columns]
    if absent and (out["skip_reason"].astype(str) == "").any():
        shown = ", ".join(absent[:6]) + ("..." if len(absent) > 6 else "")
        raise RuntimeError(
            f"the dense windows cannot supply {len(absent)} model input column(s) "
            f"({shown}); retrain under the current settings"
        )
    out = _align_to_reference(out, work / CLIP_FEATURES_FILE, reporter)

    work.mkdir(parents=True, exist_ok=True)
    staged = StagedWrite(work)
    try:
        out.to_parquet(staged.path(DENSE_FEATURES_FILE), index=False)
    except BaseException:
        staged.discard()
        raise
    staged.commit()

    summary = _summary(out, length, stride, min_present, covered, usable)
    reporter.log(summary)
    if verbose:
        print(f"  {summary}")
    return out


def _align_to_reference(frame: pd.DataFrame, reference: Path, reporter) -> pd.DataFrame:
    """Add every column of the training table missing from ``frame``, as NaN."""
    if not reference.exists():
        return frame
    try:
        columns = list(pd.read_parquet(reference).columns)
    except (OSError, ValueError):
        return frame
    missing = [c for c in columns if c not in frame.columns]
    if not missing:
        return frame
    shown = ", ".join(missing[:4]) + ("..." if len(missing) > 4 else "")
    reporter.log(
        f"{len(missing)} column(s) of {reference.name} have no dense counterpart "
        f"({shown}); added as NaN",
        "warning",
    )
    for column in missing:
        frame[column] = np.nan
    return frame


def _summary(
    frame: pd.DataFrame, length: int, stride: int, min_present: int,
    covered: int, usable: int,
) -> str:
    """The summary line the run logs; ``covered`` and ``usable`` count distinct frames."""
    blank = frame["skip_reason"].astype(str) == ""
    share = usable / covered if covered else float("nan")
    span = (f"{length} frame(s)" if stride >= length
            else f"{length} frame(s) at stride {stride}")
    return (
        f"{len(frame)} dense windows of {span} over "
        f"{frame['movie_id'].nunique()} movie(s); {int((~blank).sum())} of them "
        f"under {min_present} detections; {usable}/{covered} detected "
        f"frames ({share:.1%}) fall in a window that can be featurised"
    )


# --------------------------------------------------------------------------- #
# saying what it will be, before it is
# --------------------------------------------------------------------------- #
def count_windows(cfg: Config) -> tuple[int, int]:
    """``(windows, tracks)`` a dense run would score; keep it off the UI thread."""
    work = cfg.paths.work_dir
    if not work.exists():
        return 0, 0
    length, stride = window_length(cfg), window_stride(cfg)
    n_windows = n_tracks = 0
    for d in sorted(p for p in work.iterdir() if p.is_dir() and (p / TRACKS_FILE).exists()):
        wanted = ["track_id", "frame"]
        if int(cfg.tracking.border_margin_px) > 0:
            wanted += ["x1", "y1", "x2", "y2"]
        header = _read_header(d)
        tracks = pd.read_parquet(d / TRACKS_FILE, columns=wanted)
        tracks, _ = _without_border_rows(tracks, header, cfg)
        windows = plan_dense_windows(
            tracks, _movie_frames(_header_frames(header), tracks), length, stride
        )
        n_tracks += len(windows)
        n_windows += sum(len(spans) for spans in windows.values())
    return n_windows, n_tracks
