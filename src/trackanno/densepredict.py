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
* The windows TILE by default: ``predict.window_stride`` is the window length,
  every detected frame lands in exactly one window, and one label per frame
  falls out of ``predictions.parquet`` unaided. A SHORTER stride is permitted,
  and only with a ``predict.decode``: overlapping windows pin a transition to
  within one stride instead of one window, but then several windows speak for
  the same frame, and the renderer paints ``range(f_start, f_end)`` per track -
  so without a decode one prediction silently overwrites the other and the
  colour depends on row order. The decode is what collapses the overlap into a
  per-frame posterior and disjoint bouts before anything paints them, which is
  why :func:`trackanno.config.load_config` refuses the pairing without it.
  Never reproduce the annotation grid's one-frame overlap (``frames: 51`` with
  ``stride: 50``) though: that is an off-by-one, not a decode.
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
    "DENSE_FEATURES_FILE", "DENSE_META_COLUMNS", "window_length", "window_stride",
    "window_grid", "plan_dense_windows", "build_dense_features",
    "dense_features_path", "coverage",
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
    that both LANDS ON THE ANNOTATION GRID (anything longer than ``clips.stride``
    falls between that grid's own starts and stops reproducing a coincident
    clip's ``clip_id``) and IS THE LENGTH THE MODEL WAS FITTED ON. The length is
    no longer also the stride - that is :func:`window_stride` - but it is still
    what the features are measured over, and most of them scale with it:
    ``path_length``, ``net_displacement``, ``radius_gyration``, ``straightness``,
    ``turn_n_valid``, every ``*_std``, ``speed_max``/``speed_p90`` and
    ``present_frac``. Score a window of another length and the model reads those
    values on a scale it was never fitted at. The spectral half (``beat_*``,
    ``wave_*``) is the exception: since ``FEATURE_VERSION`` 2 it is measured over
    ``cfg.spectral.window`` centred on the window, so it is length-invariant.
    With the usual ``stride == frames`` the ``min`` is a no-op.
    """
    value = cfg.predict.window_frames
    if isinstance(value, str):
        return max(1, min(int(cfg.clips.frames), int(cfg.clips.stride)))
    return max(1, int(value))


def window_stride(cfg: Config) -> int:
    """Frames between one dense window's start and the next.

    ``auto`` is :func:`window_length`, the tiling grid this module was born with.
    Anything shorter overlaps the windows, which is only reachable together with
    a ``predict.decode``; ``load_config`` refuses a stride longer than the window
    (it would leave holes the dense pass exists to close) and refuses an
    overlapping one without a decode, so by the time a Config gets here the value
    only has to be spelled out, never judged.
    """
    value = cfg.predict.window_stride
    if isinstance(value, str):
        return window_length(cfg)
    return max(1, int(value))


def window_grid(
    n_frames: int, length: int, stride: int | None = None
) -> list[tuple[int, int]]:
    """Windows over ``[0, n_frames)``, ``length`` long and ``stride`` apart.

    ``stride is None`` means ``stride = length``: the exact tiling - no gap, no
    overlap - and the ONLY behaviour this function had until strides existed, so
    it is reproduced window for window, short tail included. That tail is
    emitted even though it is measured over fewer frames than the model was
    fitted on (``clips._windows`` drops it because a clip has to be watchable; a
    frame to be coloured does not), and it is a known wart: window length changes
    the feature VALUES - ``path_length``, ``net_displacement``,
    ``radius_gyration``, ``straightness``, ``turn_n_valid``, every ``*_std``,
    ``speed_max``/``speed_p90`` and ``present_frac`` all scale with the span - so
    a short window is scored by a model fitted at a different length. At
    ``stride == length`` that is one miscalibrated row per track, older than this
    argument and documented here rather than silently changed.

    Under overlap (``stride < length``) it would be one such row per stride at
    every track's end, feeding the per-frame posterior exactly where tracks
    finish, so short windows are dropped: every window is full length. Nothing is
    lost. A window flush with ``n_frames`` is added whenever the stride does not
    land on one, so every frame is still inside a full window - without it the
    last ``(n_frames - length) % stride`` frames of the movie would have no
    window at all, which is the hole this module exists to close. That flush
    window sits less than a stride from its neighbour, so the movie's last few
    frames are in one window more than the interior's ``length // stride``. A
    decode averages however many windows reach a frame, so that costs nothing;
    a short window scored at the wrong length would not have been averaged, it
    would have been wrong.

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
    ``length``. That division assumes ``grid[i]`` starts at ``i * length``, true
    only while the grid tiles: under an overlapping stride the grid is longer,
    both bounds name windows far short of the ones they meant to, and a track
    quietly loses coverage at its tail - no exception, just rows that are not
    there. Two ``searchsorted`` calls per track keep it O(n log n) instead of
    walking the whole grid per track.
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
        # the first detection, through the last that starts at or before the
        # last one. Both arrays are increasing, so this is the same contiguous
        # slice the integer division used to name whenever the grid tiles.
        lo = int(np.searchsorted(ends, int(frames.min()), side="right"))
        hi = int(np.searchsorted(starts, int(frames.max()), side="right")) - 1
        chosen = grid[lo : hi + 1]
        # An empty slice can only mean detections outside [0, n_frames), and
        # window_features cannot be handed an empty window list anyway.
        if chosen:
            out[int(track_id)] = chosen
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
    stride = window_stride(cfg)
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
        windows = plan_dense_windows(tracks, n_frames, length, stride)
        total += sum(len(w) for w in windows.values())
        plans.append((d, tracks, header, windows))

    # A track costs ~37 ms to lay out plus ~4 ms per window, so halving the
    # stride roughly doubles the run. Say the count and the stride BEFORE
    # spending them: a run five times slower than the last one is otherwise a
    # mystery with nothing in the log to pin it on.
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
    for d, tracks, header, windows in plans:
        movie_id = d.name
        fps = float(header["movie"]["fps"])
        feats = np.load(d / FEATURES_FILE) if (d / FEATURES_FILE).exists() else None
        if pca is not None and feats is None:
            # A saved PCA means the training features CONTAIN a cnn* block, so a
            # movie that cannot supply one is not a degraded input, it is a
            # different feature space wearing the same column names. Every cnn*
            # column comes out NaN, and because the classifier is NaN-native
            # nothing downstream objects: it predicts at unchanged confidence on
            # 55% fewer inputs and can drop a whole class from the output. That
            # is what this refusal is for. The two checks below already refuse a
            # file that is MISALIGNED or the WRONG WIDTH; absent is the third
            # case, and it was the silent one.
            raise RuntimeError(
                f"{movie_id}: {FEATURES_FILE} is missing, but {PCA_FILE} exists, "
                "so the model was fitted with detector-embedding columns this "
                "movie cannot supply. They would be written as NaN and predicted "
                "on without complaint. Re-prepare this movie with the same "
                "detection weights as the others, or retrain without the "
                "embedding features."
            )
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
            # A non-empty re-iterable Sequence, every f_end past its f_start:
            # window_features is exact on overlapping windows (bit-identical to
            # one window at a time, order-invariant) but it walks the list twice.
            measured = window_features(tdf, cfg, fps, windows[track_id])
            reached = np.zeros(frames.size, dtype=bool)
            featurised = np.zeros(frames.size, dtype=bool)
            for (f_start, f_end), features in zip(windows[track_id], measured):
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
                if n_present < min_present:
                    entry["skip_reason"] = (
                        f"only {n_present} detection(s) in frames {f_start}-{f_end}; "
                        f"predict.min_frames is {min_present}"
                    )
                    rows.append(entry)
                    continue

                featurised |= sel
                sub = tdf.iloc[np.flatnonzero(sel)]
                entry.update(features)
                if pca is not None and feats is not None:
                    entry.update(
                        _cnn_block(pca, feats[sub.index.to_numpy()].astype(np.float32))
                    )
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


def _summary(
    frame: pd.DataFrame, length: int, stride: int, min_present: int,
    covered: int, usable: int,
) -> str:
    """The line the run logs. ``covered`` and ``usable`` are DISTINCT detected
    frames, counted while the rows were built.

    They used to be ``n_present`` summed over the rows, which is the same number
    only while the grid tiles. Under an overlapping stride a frame is counted
    once per window holding it, so both sides of the ratio inflate by about
    ``length / stride`` and the line claims more detected frames than the movie
    has. The honest quantity under overlap is how many detected frames a
    scorable window REACHES, whatever the number of windows that reach them.
    """
    blank = frame["skip_reason"].astype(str) == ""
    share = usable / covered if covered else float("nan")
    span = (f"{length} frame(s)" if stride >= length
            else f"{length} frame(s) at stride {stride}")
    return (
        f"{len(frame)} dense windows of {span} over "
        f"{frame['movie_id'].nunique()} movie(s); {int((~blank).sum())} of them "
        f"under predict.min_frames ({min_present}); {usable}/{covered} detected "
        f"frames ({share:.1%}) fall in a window that can be featurised"
    )


# --------------------------------------------------------------------------- #
# checking it
# --------------------------------------------------------------------------- #
def coverage(cfg: Config, frame: pd.DataFrame) -> pd.DataFrame:
    """Per movie: how many detected frames a dense window reaches, and how many
    windows reach each of them.

    The invariant is now MODE-DEPENDENT, so this reports the quantity rather than
    asserting the old rule. ``uncovered`` is a bug either way: a detected frame
    no window speaks for comes back unlabelled and unpainted, which is the hole
    this module exists to close. Windows per frame is not a bug: it is exactly 1
    while the grid tiles, and rises to about ``length / stride`` under an
    overlapping stride, where several windows speaking for one frame is the whole
    point and ``predict.decode`` is what turns them back into one label. Above 1
    with ``decode: none`` it IS a bug - the renderer paints
    ``range(f_start, f_end)`` per track, so the colour would follow row order -
    and that is the pairing ``load_config`` refuses.

    Kept called, by ``tests/test_dense_grid.py``, on purpose: it was dead code
    while the grid changed under it, and a grid that quietly stops covering
    everything shows up as a missing row rather than as an error.
    """
    out = []
    for movie_id, group in frame.groupby("movie_id", sort=True):
        tracks = pd.read_parquet(cfg.movie_dir(str(movie_id)) / TRACKS_FILE)
        spans = {
            int(t): list(zip(g["f_start"].astype(int), g["f_end"].astype(int)))
            for t, g in group.groupby("track_id", sort=False)
        }
        detected = missed = once = total_hits = most = 0
        for track_id, tdf in tracks.groupby("track_id", sort=False):
            frames = tdf["frame"].to_numpy(dtype=np.int64)
            detected += frames.size
            hits = np.zeros(frames.size, dtype=np.int64)
            for f_start, f_end in spans.get(int(track_id), []):
                hits += ((frames >= f_start) & (frames < f_end)).astype(np.int64)
            missed += int((hits == 0).sum())
            once += int((hits == 1).sum())
            total_hits += int(hits.sum())
            most = max(most, int(hits.max()) if hits.size else 0)
        out.append({
            "movie_id": str(movie_id),
            "detected_frames": detected,
            "uncovered": missed,
            "covered_once": once,
            "coverage": (detected - missed) / detected if detected else float("nan"),
            # Over every detected frame, uncovered ones included: a mean below 1
            # is another way of seeing the hole, and the max is what a decode has
            # to reconcile at the busiest frame.
            "windows_per_frame_mean": total_hits / detected if detected else float("nan"),
            "windows_per_frame_max": most,
        })
    return pd.DataFrame(out)
