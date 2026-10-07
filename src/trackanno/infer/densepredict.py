"""Prediction windows that cover every frame of every track.

The annotation clip grid is a sampling *for annotation*, sparse on purpose. This
module builds a second, DENSE feature table - one row per (track, window) over a
grid laid across the whole movie - for the same models. Rules:

* No pixels. Features come from ``tracks.parquet``, the embedding sources and
  the frame rate in ``cache.json``, so a dense grid costs arithmetic.
* Windows are ``clips.frames`` long, the length the model was fitted at, and
  start every ``predict.window_stride`` frames. The default stride tiles; a
  shorter one overlaps, and :mod:`trackanno.infer.predict` averages the overlap
  into a per-frame posterior before cutting bouts.
* Features come through :func:`trackanno.measure.clipfeatures.window_features`, the
  same entry point that builds ``clip_features.parquet``: its spectral columns
  are measured over a window CENTRED on each clip rather than over the clip, so
  any other route is train/serve skew no shape check could catch.
* Embedding PCAs are LOADED from beside the clip features, never refitted. A
  refit gives a different basis, silently.

A window with too few detections still gets a row carrying ``skip_reason`` and
``n_present``; an absent row is indistinguishable from one the model declined
to label. The dense table has a file of its own - the clip features, manifests
and cache belong to the annotation flow and are never touched here.
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

# Columns the dense table carries that ``clip_features.parquet`` does not.
# Metadata, never model inputs (the model is fed by name from
# ``Suggester.columns``), but they are what makes an unpredicted row readable.
# `clipfeatures.META_COLUMNS` holds them too, so a caller asking that module for
# "every feature column" of a dense table is not handed them.
DENSE_META_COLUMNS = ("f_end", "n_present", "skip_reason")


def dense_features_path(cfg: Config) -> Path:
    return cfg.paths.work_dir / DENSE_FEATURES_FILE


# --------------------------------------------------------------------------- #
# the grid
# --------------------------------------------------------------------------- #
def window_length(cfg: Config) -> int:
    """How many frames one dense window spans: ``clips.frames``.

    The length the model was fitted at. Most features scale with it
    (``path_length``, every ``*_std``, ``present_frac``), so any other length
    would be train/serve skew.
    """
    return max(1, int(cfg.clips.frames))


def window_stride(cfg: Config) -> int:
    """Frames between one dense window's start and the next."""
    return cfg.predict.resolved_stride(cfg.clips)


def window_grid(
    n_frames: int, length: int, stride: int | None = None
) -> list[tuple[int, int]]:
    """Windows over ``[0, n_frames)``, ``length`` long and ``stride`` apart.

    ``stride is None`` means ``stride = length``: the exact tiling, no gap and no
    overlap, short tail included. That tail is a known wart - window length
    changes the feature VALUES, so a short window is scored by a model fitted at
    a different length - but at ``stride == length`` it is one miscalibrated row
    per track.

    Under overlap (``stride < length``) it would be one such row per stride at
    every track's end, so short windows are dropped and every window is full
    length. A window flush with ``n_frames`` is added whenever the stride does
    not land on one, or the last ``(n_frames - length) % stride`` frames would
    have no window at all. That flush window sits less than a stride from its
    neighbour, so the last few frames are in one window more than the interior's
    ``length // stride``; a decode averages however many windows reach a frame,
    so that costs nothing.

    A movie SHORTER than one window has no full window to give, in either mode.
    It gets the one short window anyway: a miscalibrated row says what it is
    (``f_end - f_start``, ``n_present``, ``present_frac``) and can be filtered
    downstream, while a missing row is indistinguishable from a model that
    declined to label.
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
    """``{track_id: windows}`` covering each track's whole detected lifetime.

    Every window between a track's first and last detection, including ones it
    is missing from entirely - "no detections here" is more use than no row.
    Windows outside that span are not emitted. Every track is covered, including
    ones shorter than ``tracking.min_track_length``: that threshold is about
    annotation units, not about whether the animal was on screen.

    The span is found by SEARCHING the grid, never by dividing a frame number by
    ``length``: that division assumes ``grid[i]`` starts at ``i * length``, true
    only while the grid tiles, and under an overlapping stride a track would
    quietly lose coverage at its tail. Two ``searchsorted`` calls per track keep
    it O(n log n) instead of walking the whole grid per track.
    """
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
        # The windows that INTERSECT [first, last]: the first whose end is past
        # the first detection, through the last that starts at or before the last
        # one. Both arrays are increasing, so the slice is contiguous.
        lo = int(np.searchsorted(ends, int(frames.min()), side="right"))
        hi = int(np.searchsorted(starts, int(frames.max()), side="right")) - 1
        chosen = grid[lo : hi + 1]
        # An empty slice can only mean detections outside [0, n_frames), and
        # window_features cannot be handed an empty window list anyway.
        if chosen:
            out[int(track_id)] = chosen
    return out


def _movie_frames(header_frames: int, tracks: pd.DataFrame) -> int:
    """How many frames the grid spans, given what the header says the movie holds.

    The header's count is what the movie HOLDS; the tracks' last frame is what
    was actually processed (``trackanno prepare --limit-frames`` stops early). Take the
    larger, so the grid covers every detection even if the two disagree.
    """
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
    """``tracks`` under ``tracking.border_margin_px``, and the frames that went.

    Applied to dense coverage and not only to clip planning: a keypoint placed
    against an animal half outside the frame is as wrong for a predicted bout as
    for a clip somebody looks at. Only those frames go; the track carries on
    where it comes back, and its windows there are windows like any other.

    The excluded frames come back per track because a window still has to see
    how much of itself was spent at the edge - the same
    ``tracking.border_max_frac`` budget the clip planner applies, so a dense
    window is skipped exactly where a clip of it would not have been cut.

    Passes everything through where the header cannot say how big the frame is.
    """
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
                f"embedding {spec.name!r} has no saved basis in {emb.BASES_FILE}, so "
                "the dense table carries no columns for it. Rebuild the clip "
                "features (retrain) to add them.",
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
    """One feature row per (track, dense window) across every prepared movie.

    Reads ``tracks.parquet``, the embedding sources and ``cache.json`` per movie
    plus the embedding bases, writes ``dense_clip_features.parquet``, and
    touches nothing in the annotation flow. ``embeddings`` is the model's own
    ``{name: (spec, PCA)}``; without it the config and the bases on disk are
    used. A ``required`` column no featurised window produced is refused.
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
        # Embedding codes are aligned to the UNFILTERED table; the row LABELS
        # survive the border filter, so lookups by label still land.
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
    # DISTINCT detected frames a window reaches, and how many of them a window
    # that could be featurised reaches. Accumulated here rather than summed out
    # of n_present afterwards, which counts a frame once per window holding it.
    covered = usable = 0
    for d, tracks, codes, header, windows, border_frames in plans:
        movie_id = d.name
        fps = float(header["movie"]["fps"])
        by_track = {int(t): df for t, df in tracks.groupby("track_id", sort=False)}

        for track_id in sorted(windows):
            tdf = by_track[track_id].sort_values("frame")
            frames = tdf["frame"].to_numpy(dtype=np.int64)
            # Through the SAME entry point build_clip_features uses, once per
            # track rather than once per window. Driving trackseries and
            # wavespectra directly would give a different feature definition
            # under identical column NAMES: the spectral columns are measured
            # over a window centred on the clip, not over the clip. `spans` must
            # be a non-empty re-iterable Sequence with every f_end past its
            # f_start - window_features walks the list twice.
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
                    # Named before the count, because the count is its
                    # consequence: those detections were excluded, so a window
                    # mostly at the edge also has few left.
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
        # Same stamp the clip table carries: this table is scored by a model
        # fitted against a particular definition of these columns, and only the
        # version can attest to that.
        out["_feature_version"] = FEATURE_VERSION
    if out.empty:
        raise RuntimeError(
            "no dense windows could be planned; every prepared movie has an empty "
            "tracks.parquet"
        )
    # NaN means "undefined", and both HistGradientBoosting and CatBoost read it
    # that way. Filling with 0.0 would disguise a failed computation as a real
    # extreme (straightness 0.0 = a perfect circle, beat_freq_hz 0.0 = DC).
    out = out.replace([np.inf, -np.inf], np.nan)
    absent = [c for c in (required or ()) if c not in out.columns]
    if absent and (out["skip_reason"].astype(str) == "").any():
        shown = ", ".join(absent[:6]) + ("..." if len(absent) > 6 else "")
        raise RuntimeError(
            f"the dense windows cannot supply {len(absent)} model input column(s) "
            f"({shown}); they would be scored as NaN. Retrain under the current "
            "settings."
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
    """Give the dense table every column the training table had, NaN if absent.

    ``window_features`` fills its whole schema in, so this normally changes
    nothing. It matters where the dense grid produced no row that could compute a
    family at all - every window under ``predict.min_frames``, or a missing saved
    embedding basis - because then the column is absent rather than NaN.
    """
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
        f"({shown}); they are added as NaN so the model still gets its full input "
        "space. If they are emb_* columns, the saved embedding basis is missing.",
        "warning",
    )
    for column in missing:
        frame[column] = np.nan
    return frame


def _summary(
    frame: pd.DataFrame, length: int, stride: int, min_present: int,
    covered: int, usable: int,
) -> str:
    """The line the run logs.

    ``covered`` and ``usable`` are DISTINCT detected frames, counted while the
    rows were built. Summing ``n_present`` over the rows would count a frame once
    per window holding it, so under an overlapping stride both
    sides of the ratio would inflate and the line would claim more detected
    frames than the movie has.
    """
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
    """``(windows, tracks)`` a dense run under this config would score.

    The same grid :func:`build_dense_features` lays down, planned from
    ``tracks.parquet`` alone. One parquet read per movie, so keep it off a UI
    thread.
    """
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
