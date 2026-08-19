"""Prediction windows that cover every frame of every track.

The annotation clip grid is a sampling *for annotation*, sparse on purpose:
``clips.window_mode``, a ``clips.stride`` past ``clips.frames`` and
``clips.min_clip_frames`` all leave holes, and frames in those holes come back
unlabelled from :func:`trackanno.predict.predict_all` and unpainted in the
overlay. This module builds a second, DENSE feature table - one row per
(track, window) over a grid tiling the whole movie - for the same models.
Rules:

* No pixels. Features come from ``tracks.parquet``, ``instance_features.npy``
  and the frame rate in ``cache.json``, so a dense grid costs arithmetic.
* The windows tile exactly: stride equals length, every detected frame in
  exactly one window. Do not reproduce the annotation grid's one-frame overlap
  (``frames: 51`` with ``stride: 50``) - the renderer paints
  ``range(f_start, f_end)`` per track, so a shared frame means one prediction
  silently overwrites the other and the colour depends on row order.
* Features come through :func:`trackanno.clipfeatures.window_features`, the
  same entry point that builds ``clip_features.parquet``: its spectral columns
  are measured over a window CENTRED on each clip rather than over the clip, so
  any other route is train/serve skew no shape check could catch.
* The CNN PCA is LOADED from beside the model, never refitted. A refit gives a
  different orthogonal basis - no error, no warning, and a model whose input
  space quietly stops meaning anything.

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

from .clipfeatures import (
    CLIP_FEATURES_FILE, FEATURE_VERSION, PCA_FILE, window_features,
)
from .clips import FEATURES_FILE, HEADER_FILE, TRACKS_FILE, StagedWrite, clip_id_for
from .config import Config

__all__ = [
    "DENSE_FEATURES_FILE", "DENSE_META_COLUMNS", "window_length", "window_grid",
    "plan_dense_windows", "build_dense_features", "dense_features_path", "coverage",
]

DENSE_FEATURES_FILE = "dense_clip_features.parquet"

# Columns the dense table carries that ``clip_features.parquet`` does not.
# Metadata, never model inputs (the model is fed by name from
# ``Suggester.columns``), but they are what makes an unpredicted row readable.
DENSE_META_COLUMNS = ("f_end", "n_present", "skip_reason")


def dense_features_path(cfg: Config) -> Path:
    return cfg.paths.work_dir / DENSE_FEATURES_FILE


# --------------------------------------------------------------------------- #
# the grid
# --------------------------------------------------------------------------- #
def window_length(cfg: Config) -> int:
    """How many frames one dense window spans.

    ``auto`` resolves to ``min(clips.frames, clips.stride)``, the longest window
    that is both a TILING (stride equals length here, so anything longer than
    ``clips.stride`` lands between the annotation grid's own starts and stops
    reproducing a coincident clip's ``clip_id``) and THE LENGTH THE MODEL WAS
    FITTED ON (window length changes the feature values themselves - path
    length, beat resolvability, present fraction). With the usual
    ``stride == frames`` the ``min`` is a no-op.
    """
    value = cfg.predict.window_frames
    if isinstance(value, str):
        return max(1, min(int(cfg.clips.frames), int(cfg.clips.stride)))
    return max(1, int(value))


def window_grid(n_frames: int, length: int) -> list[tuple[int, int]]:
    """Windows tiling ``[0, n_frames)``: no gap, no overlap, tail included.

    The final window is short whenever the movie does not divide evenly and is
    still emitted. ``clips._windows`` drops that tail because it needs
    ``min_clip_frames`` to be watchable; a frame to be coloured does not.
    """
    if n_frames <= 0 or length <= 0:
        return []
    return [(start, min(start + length, n_frames))
            for start in range(0, n_frames, length)]


def plan_dense_windows(
    tracks: pd.DataFrame, n_frames: int, length: int
) -> dict[int, list[tuple[int, int]]]:
    """``{track_id: windows}`` covering each track's whole detected lifetime.

    Every window between a track's first and last detection, including ones it
    is missing from entirely - "no detections here" is more use than no row.
    Windows outside that span are not emitted. Every track is covered, including
    ones shorter than ``tracking.min_track_length``: that threshold is about
    annotation units, not about whether the animal was on screen.
    """
    grid = window_grid(n_frames, length)
    out: dict[int, list[tuple[int, int]]] = {}
    if tracks.empty or not grid:
        return out
    for track_id, tdf in tracks.groupby("track_id", sort=True):
        frames = tdf["frame"].to_numpy(dtype=np.int64)
        if frames.size == 0:
            continue
        lo = int(frames.min()) // length
        hi = int(frames.max()) // length
        out[int(track_id)] = grid[lo : hi + 1]
    return out


# --------------------------------------------------------------------------- #
# the saved PCA
# --------------------------------------------------------------------------- #
def _load_pca(cfg: Config, reporter):
    """The PCA that reduced the CNN features the model was fitted on, or None.

    LOADED, NEVER FITTED. ``build_clip_features`` fits one because it is building
    the space the model will be trained in; here that space already exists, and a
    refit returns a different basis - sign flips at minimum, a different rotation
    whenever the dense set of detections differs from the clipped one. Nothing
    raises, the column names still line up, and every ``cnn*`` value the model
    sees is nonsense.
    """
    path = cfg.paths.work_dir / PCA_FILE
    if not path.exists():
        reporter.log(
            f"no {PCA_FILE} in the work dir, so the dense table will carry no CNN "
            "columns. If the model was fitted with the 'cnn' feature group this "
            "run cannot proceed - rebuild the clip features (retrain) first.",
            "warning",
        )
        return None
    import joblib

    return joblib.load(path)


def _cnn_block(pca, block: np.ndarray) -> dict[str, float]:
    """The clip-level CNN aggregate, exactly as ``build_clip_features`` forms it."""
    codes = pca.transform(block)
    out = {f"cnn{i}_mean": float(v) for i, v in enumerate(codes.mean(axis=0))}
    out.update({f"cnn{i}_std": float(v) for i, v in enumerate(codes.std(axis=0))})
    delta = (
        np.abs(np.diff(codes, axis=0)).mean(axis=0)
        if codes.shape[0] > 1 else np.zeros(codes.shape[1])
    )
    out.update({f"cnn{i}_delta": float(v) for i, v in enumerate(delta)})
    return out


# --------------------------------------------------------------------------- #
# the table
# --------------------------------------------------------------------------- #
def build_dense_features(
    cfg: Config, reporter=None, verbose: bool = False
) -> pd.DataFrame:
    """One feature row per (track, dense window) across every prepared movie.

    Reads ``tracks.parquet``, ``instance_features.npy`` and ``cache.json`` per
    movie plus the saved PCA, writes ``dense_clip_features.parquet``, and
    touches nothing in the annotation flow.
    """
    from .progress import NullReporter

    reporter = reporter or NullReporter()
    work = cfg.paths.work_dir
    if not work.exists():
        raise FileNotFoundError(f"no work dir at {work}; run prepare.py first")
    movie_dirs = sorted(
        p for p in work.iterdir() if p.is_dir() and (p / TRACKS_FILE).exists()
    )
    if not movie_dirs:
        raise FileNotFoundError(f"no prepared movies under {work}; run prepare.py first")

    length = window_length(cfg)
    min_present = int(cfg.predict.min_frames)
    pca = _load_pca(cfg, reporter)

    plans, total = [], 0
    for d in movie_dirs:
        tracks = pd.read_parquet(d / TRACKS_FILE)
        header = json.loads((d / HEADER_FILE).read_text(encoding="utf-8"))
        # The header's frame count is what the movie holds; the tracks' last
        # frame is what was actually processed (prepare.py --limit-frames stops
        # early). Take the larger, so the grid covers every detection even if the
        # two disagree.
        n_frames = int(header["movie"]["n_frames"])
        if not tracks.empty:
            n_frames = max(n_frames, int(tracks["frame"].max()) + 1)
        windows = plan_dense_windows(tracks, n_frames, length)
        total += sum(len(w) for w in windows.values())
        plans.append((d, tracks, header, windows))

    reporter.stage("dense clip features", total)
    rows: list[dict] = []
    for d, tracks, header, windows in plans:
        movie_id = d.name
        fps = float(header["movie"]["fps"])
        feats = np.load(d / FEATURES_FILE) if (d / FEATURES_FILE).exists() else None
        if feats is not None and len(feats) != len(tracks):
            # instance_features.npy is written row-aligned with tracks.parquet.
            # Assert it rather than trust it: these vectors re-identify
            # individuals almost perfectly, so a mis-pairing shows up not as
            # reduced accuracy but as a model that tests well and fails later.
            raise RuntimeError(
                f"{movie_id}: instance_features.npy has {len(feats)} rows but "
                f"tracks.parquet has {len(tracks)}; features are not row-aligned"
            )
        if pca is not None and feats is not None and feats.shape[1] != pca.n_features_in_:
            # A different detector checkpoint has different channel widths, so
            # its per-detection vector is a different length; transform() would
            # raise a numpy error naming neither the movie nor the model.
            raise RuntimeError(
                f"{movie_id}: instance_features.npy is {feats.shape[1]}-d but "
                f"{PCA_FILE} was fitted on {int(pca.n_features_in_)}-d vectors. "
                "This movie was prepared with a different detection model - "
                "re-prepare it with the same weights, then retrain."
            )
        by_track = {int(t): df for t, df in tracks.groupby("track_id", sort=False)}

        for track_id in sorted(windows):
            tdf = by_track[track_id].sort_values("frame")
            frames = tdf["frame"].to_numpy(dtype=np.int64)
            # Through the SAME entry point build_clip_features uses, once per
            # track rather than once per window. Driving trackseries and
            # wavespectra directly would give the export a different feature
            # definition under identical column NAMES: the spectral columns are
            # measured over a window centred on the clip, not over the clip.
            measured = window_features(tdf, cfg, fps, windows[track_id])
            for (f_start, f_end), features in zip(windows[track_id], measured):
                reporter.raise_if_cancelled()
                reporter.advance()
                sel = (frames >= f_start) & (frames < f_end)
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
                if n_present < min_present:
                    entry["skip_reason"] = (
                        f"only {n_present} detection(s) in frames {f_start}-{f_end}; "
                        f"predict.min_frames is {min_present}"
                    )
                    rows.append(entry)
                    continue

                sub = tdf.iloc[np.flatnonzero(sel)]
                entry.update(features)
                if pca is not None and feats is not None:
                    entry.update(
                        _cnn_block(pca, feats[sub.index.to_numpy()].astype(np.float32))
                    )
                rows.append(entry)

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
    out = _align_to_reference(out, work / CLIP_FEATURES_FILE, reporter)

    work.mkdir(parents=True, exist_ok=True)
    staged = StagedWrite(work)
    try:
        out.to_parquet(staged.path(DENSE_FEATURES_FILE), index=False)
    except BaseException:
        staged.discard()
        raise
    staged.commit()

    summary = _summary(out, length, min_present)
    reporter.log(summary)
    if verbose:
        print(f"  {summary}")
    return out


def _align_to_reference(frame: pd.DataFrame, reference: Path, reporter) -> pd.DataFrame:
    """Give the dense table every column the training table had, NaN if absent.

    ``window_features`` fills its whole schema in, so this normally changes
    nothing. It matters where the dense grid produced no row that could compute a
    family at all - every window under ``min_frames``, or a missing saved PCA -
    because then the column is *absent*, not NaN, and
    ``features[suggester.columns]`` raises a KeyError naming neither file.
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
        "space. If they are cnn* columns, the saved PCA is missing.",
        "warning",
    )
    for column in missing:
        frame[column] = np.nan
    return frame


def _summary(frame: pd.DataFrame, length: int, min_present: int) -> str:
    blank = frame["skip_reason"].astype(str) == ""
    covered = int(frame["n_present"].sum())
    usable = int(frame.loc[blank, "n_present"].sum())
    share = usable / covered if covered else float("nan")
    return (
        f"{len(frame)} dense windows of {length} frame(s) over "
        f"{frame['movie_id'].nunique()} movie(s); {int((~blank).sum())} of them "
        f"under predict.min_frames ({min_present}); {usable}/{covered} detected "
        f"frames ({share:.1%}) fall in a window that can be featurised"
    )


# --------------------------------------------------------------------------- #
# checking it
# --------------------------------------------------------------------------- #
def coverage(cfg: Config, frame: pd.DataFrame) -> pd.DataFrame:
    """Per movie: detected frames, how many a window covers, and any overlap.

    The grid's claim is "exactly once", worth checking rather than asserting: a
    frame covered twice is a prediction the renderer overwrites with whichever
    row came last, and a frame covered zero times is the hole this module exists
    to close.
    """
    out = []
    for movie_id, group in frame.groupby("movie_id", sort=True):
        tracks = pd.read_parquet(cfg.movie_dir(str(movie_id)) / TRACKS_FILE)
        spans = {
            int(t): list(zip(g["f_start"].astype(int), g["f_end"].astype(int)))
            for t, g in group.groupby("track_id", sort=False)
        }
        detected = missed = doubled = 0
        for track_id, tdf in tracks.groupby("track_id", sort=False):
            frames = tdf["frame"].to_numpy(dtype=np.int64)
            detected += frames.size
            hits = np.zeros(frames.size, dtype=np.int64)
            for f_start, f_end in spans.get(int(track_id), []):
                hits += ((frames >= f_start) & (frames < f_end)).astype(np.int64)
            missed += int((hits == 0).sum())
            doubled += int((hits > 1).sum())
        out.append({
            "movie_id": str(movie_id),
            "detected_frames": detected,
            "covered_once": detected - missed - doubled,
            "uncovered": missed,
            "covered_twice_or_more": doubled,
            "coverage": (detected - missed) / detected if detected else float("nan"),
        })
    return pd.DataFrame(out)
