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
  falls out of ``predictions.parquet`` unaided. A SHORTER stride pins a
  transition to within one stride instead of one window, but then several
  windows speak for the same frame and the renderer paints
  ``range(f_start, f_end)`` per track, so it is permitted only with a
  ``predict.decode``, which collapses the overlap into disjoint bouts first.
* ``predict.window_scales`` plays the same game with the window's LENGTH rather
  than its offset: the grid is built once per listed length and the rows go into
  one table. It needs the same decode plus one condition overlap does not - the
  model must have been fitted on columns that do not scale with the window
  length. See :func:`scale_grid` for the weighting.
* Features come through :func:`trackanno.clipfeatures.window_features`, the
  same entry point that builds ``clip_features.parquet``: its spectral columns
  are measured over a window CENTRED on each clip rather than over the clip, so
  any other route is train/serve skew no shape check could catch.
* The CNN PCA is LOADED from beside the model, never refitted. A refit gives a
  different orthogonal basis, silently.

A window with too few detections still gets a row carrying ``skip_reason`` and
``n_present``; an absent row is indistinguishable from one the model declined
to label. The dense table has a file of its own - the clip features, manifests
and cache belong to the annotation flow and are never touched here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .clipfeatures import (
    CLIP_FEATURES_FILE, FEATURE_VERSION, PCA_FILE, window_features,
)
from .clips import FEATURES_FILE, HEADER_FILE, TRACKS_FILE, StagedWrite, clip_id_for
from .config import Config

__all__ = [
    "DENSE_FEATURES_FILE", "DENSE_META_COLUMNS", "window_length", "window_stride",
    "window_scales", "scale_stride", "scale_grid",
    "window_grid", "plan_dense_windows", "plan_scaled_windows",
    "build_dense_features", "dense_features_path", "coverage",
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
    that both LANDS ON THE ANNOTATION GRID (so a coincident window reproduces the
    annotation clip's ``clip_id``) and IS THE LENGTH THE MODEL WAS FITTED ON -
    two goals that are one number only while ``clips.stride >= clips.frames``.
    Below that :func:`trackanno.config.load_config` refuses ``auto`` for a dense
    project rather than resolving it.

    The length is what the features are measured over, and most of them scale
    with it: ``path_length``, ``net_displacement``, ``radius_gyration``,
    ``straightness``, ``turn_n_valid``, every ``*_std``,
    ``speed_max``/``speed_p90`` and ``present_frac``. The spectral half
    (``beat_*``, ``wave_*``) is the exception, measured over
    ``cfg.spectral.window`` centred on the window and so length-invariant.
    """
    value = cfg.predict.window_frames
    if isinstance(value, str):
        return max(1, min(int(cfg.clips.frames), int(cfg.clips.stride)))
    return max(1, int(value))


def window_stride(cfg: Config) -> int:
    """Frames between one dense window's start and the next.

    ``auto`` is :func:`window_length`, an exact tiling. ``load_config`` refuses a
    stride longer than the window (it would leave holes the dense pass exists to
    close) and an overlapping one without a ``predict.decode``, so by the time a
    Config gets here the value only has to be spelled out, never judged.
    """
    value = cfg.predict.window_stride
    if isinstance(value, str):
        return window_length(cfg)
    return max(1, int(value))


def window_scales(cfg: Config) -> list[int]:
    """Every window LENGTH this run scores at, the base one first.

    One entry unless ``predict.window_scales`` says otherwise, which is what
    keeps every caller here free of a "is multi-scale on" branch.
    """
    return cfg.predict.resolved_scales(cfg.clips)


def scale_stride(length: int, base_length: int, base_stride: int) -> int:
    """The stride to walk a ``length``-frame grid with, given the base grid.

    Proportional to the length, and that proportionality IS the weighting
    decision - see :func:`scale_grid`. ``base_stride >= base_length`` (the
    default exact tiling) returns ``length`` itself rather than the same number
    through a division, so the ordinary case cannot pick up a rounding error.
    Elsewhere the ratio need not be whole, and that rounding is the one place the
    equal weighting is approximate rather than exact.
    """
    base_length, base_stride, length = int(base_length), int(base_stride), int(length)
    if base_stride >= base_length:
        return max(1, length)
    return max(1, int(round(base_stride * length / max(base_length, 1))))


def scale_grid(cfg: Config) -> list[tuple[int, int]]:
    """``(length, stride)`` per scale, the base grid first.

    THE WEIGHTING DECISION. Each scale gets EQUAL WEIGHT in the per-frame
    posterior, and nothing new computes that:
    :func:`trackanno.bouts.frame_posterior` takes the plain mean over the windows
    covering a frame, so a scale's weight is however many of its windows reach
    that frame, and a stride proportional to the length makes that count
    ``length / stride`` - the same number at every scale. One stride for every
    scale instead would weight a scale by its LENGTH, letting the longest window
    outvote the rest, which is exactly the answer adding a short scale was meant
    to question. The cost of equal weight is that a frame deep inside a long
    uniform bout gains no extra confidence from the long window agreeing.

    Two limits on the word "equal":

    * It is exact in a track's INTERIOR. Near either end :func:`window_grid`
      clips the grid, drops short windows and adds a flush window, and
      :func:`plan_dense_windows` cuts the plan to the track's lifetime, so the
      covering counts differ per scale there. ``n_covering_mean`` on each bout
      reports how much evidence a bout rests on.
    * It is exact only where ``base_stride * length / base_length`` comes out
      whole. At the default tiling stride it is exact at every scale; elsewhere
      :func:`scale_stride` rounds.

    A movie shorter than every scale is degenerate: each scale returns the same
    single clipped window, so the mean of identical rows is that row.
    """
    base_length = window_length(cfg)
    base_stride = window_stride(cfg)
    return [(length, scale_stride(length, base_length, base_stride))
            for length in window_scales(cfg)]


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


def plan_scaled_windows(
    tracks: pd.DataFrame, n_frames: int, grid: Sequence[tuple[int, int]]
) -> dict[int, list[tuple[int, int, int]]]:
    """``{track_id: [(f_start, f_end, scale)]}`` over several ``(length, stride)`` grids.

    Every grid is planned by :func:`plan_dense_windows`, so a single grid gives
    exactly the windows and the order it gives on its own, with the scale
    appended. The scale is NOT recoverable from ``f_end - f_start``: a window
    clipped against a short movie spans fewer frames than the scale it belongs
    to, and those are the rows whose provenance is worth knowing.

    Grids are laid out in the order given - base first, by
    :meth:`trackanno.config.Predict.resolved_scales` - and the windows of one
    scale are contiguous within a track, so the dense table reads as one block
    per scale rather than interleaved.
    """
    out: dict[int, list[tuple[int, int, int]]] = {}
    for length, stride in grid:
        for track_id, spans in plan_dense_windows(
            tracks, n_frames, int(length), int(stride)
        ).items():
            out.setdefault(int(track_id), []).extend(
                (int(f_start), int(f_end), int(length)) for f_start, f_end in spans
            )
    return out


def _dense_clip_id(
    movie_id: str, track_id: int, f_start: int, length: int, base_length: int
) -> str:
    """The dense row's id, with the scale appended for every scale but the base.

    ``clip_id_for`` is built from the START ALONE, so two scales beginning at the
    same frame would be handed the same id - duplicate keys in a table other code
    joins on, and no error anywhere. Hence the suffix.

    The BASE scale keeps the bare id because the dense grid is laid out so its
    windows collide with the annotation grid's ids deliberately (see
    :func:`trackanno.predict._bout_id`), and that collision is what
    ``human_labelled`` and ``include_labelled`` are built on. Suffixing only the
    added scales leaves a single-scale table byte for byte what it was.
    """
    clip_id = clip_id_for(movie_id, track_id, f_start)
    if int(length) == int(base_length):
        return clip_id
    return f"{clip_id}__w{int(length):04d}"


# --------------------------------------------------------------------------- #
# the saved PCA
# --------------------------------------------------------------------------- #
def _load_pca(cfg: Config, reporter):
    """The PCA that reduced the CNN features the model was fitted on, or None.

    LOADED, NEVER FITTED. ``build_clip_features`` fits one because it is building
    the space the model will be trained in; here that space already exists, and a
    refit would return a different basis under the same column names, with
    nothing raising.
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
    grid = scale_grid(cfg)
    min_present = int(cfg.predict.min_frames)
    pca = _load_pca(cfg, reporter)

    plans, total = [], 0
    per_scale = {scale: 0 for scale, _ in grid}
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
        windows = plan_scaled_windows(tracks, n_frames, grid)
        for spans in windows.values():
            total += len(spans)
            for _, _, scale in spans:
                per_scale[scale] += 1
        plans.append((d, tracks, header, windows))

    # Halving the stride roughly doubles the run, so say the window count and the
    # stride BEFORE spending them: a run several times slower than the last one
    # is otherwise a mystery with nothing in the log to pin it on.
    if len(grid) == 1:
        reporter.log(
            f"planning {total} dense window(s) of {length} frame(s) at stride "
            f"{stride} over {len(plans)} movie(s)"
        )
    else:
        # Same reason, one scale further out. The window multiple is an UPPER
        # bound on the cost rather than the cost: the per-track setup inside
        # `window_features` - the trackseries layout and the sliding spectral
        # transform - is paid once for every scale of a track, because they all
        # go through it in one call.
        parts = "; ".join(
            f"{per_scale[scale]} of {scale} frame(s) at stride {step}"
            for scale, step in grid
        )
        base = max(per_scale[grid[0][0]], 1)
        reporter.log(
            f"planning {total} dense window(s) over {len(plans)} movie(s) at "
            f"{len(grid)} scales - {parts} - which is {total / base:.1f}x the "
            f"{length}-frame grid alone. Every scale carries the same weight in "
            "the per-frame posterior, which is what the stride scaling above buys"
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
            # movie that cannot supply one is a different feature space wearing
            # the same column names. Every cnn* column would come out NaN, and
            # the classifier is NaN-native, so it would predict at unchanged
            # confidence on far fewer inputs. The two checks below cover the
            # MISALIGNED and WRONG WIDTH cases; absent is the third.
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
            # Asserted rather than trusted: these vectors re-identify individuals
            # almost perfectly, so a mis-pairing shows up not as reduced accuracy
            # but as a model that tests well and fails later.
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
            # wavespectra directly would give a different feature definition
            # under identical column NAMES: the spectral columns are measured
            # over a window centred on the clip, not over the clip. `spans` must
            # be a non-empty re-iterable Sequence with every f_end past its
            # f_start - window_features walks the list twice. Every scale of this
            # track goes through ONE call, so the per-track work is done once and
            # only the per-window terms are paid again.
            spans = [(f_start, f_end) for f_start, f_end, _ in windows[track_id]]
            measured = window_features(tdf, cfg, fps, spans)
            # Row positions into `feats`, which is aligned with tracks.parquet.
            # Only where the cnn block is actually built.
            rows_in_feats = (
                tdf.index.to_numpy()
                if pca is not None and feats is not None else None
            )
            reached = np.zeros(frames.size, dtype=bool)
            featurised = np.zeros(frames.size, dtype=bool)
            for (f_start, f_end, scale), features in zip(windows[track_id], measured):
                reporter.raise_if_cancelled()
                reporter.advance()
                sel = (frames >= f_start) & (frames < f_end)
                reached |= sel
                n_present = int(sel.sum())
                entry = {
                    "clip_id": _dense_clip_id(
                        movie_id, track_id, f_start, scale, length
                    ),
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
                entry.update(features)
                if rows_in_feats is not None:
                    entry.update(
                        _cnn_block(pca, feats[rows_in_feats[sel]].astype(np.float32))
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

    summary = _summary(out, grid, min_present, covered, usable)
    reporter.log(summary)
    if verbose:
        print(f"  {summary}")
    return out


def _align_to_reference(frame: pd.DataFrame, reference: Path, reporter) -> pd.DataFrame:
    """Give the dense table every column the training table had, NaN if absent.

    ``window_features`` fills its whole schema in, so this normally changes
    nothing. It matters where the dense grid produced no row that could compute a
    family at all - every window under ``min_frames``, or a missing saved PCA -
    because then the column is absent rather than NaN.
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
    frame: pd.DataFrame, grid: Sequence[tuple[int, int]], min_present: int,
    covered: int, usable: int,
) -> str:
    """The line the run logs.

    ``covered`` and ``usable`` are DISTINCT detected frames, counted while the
    rows were built. Summing ``n_present`` over the rows would count a frame once
    per window holding it, so under an overlapping stride or several scales both
    sides of the ratio would inflate and the line would claim more detected
    frames than the movie has.
    """
    blank = frame["skip_reason"].astype(str) == ""
    share = usable / covered if covered else float("nan")
    # One join over every scale, base grid included: a single-scale grid gives
    # "50 frame(s)", or "50 frame(s) at stride 25" when it overlaps.
    span = " + ".join(
        f"{scale} frame(s)" if step >= scale
        else f"{scale} frame(s) at stride {step}"
        for scale, step in grid
    )
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

    Reports the quantities rather than asserting a rule, because the expected
    windows-per-frame is mode-dependent: exactly 1 while the grid tiles, about
    ``length / stride`` under an overlapping stride, and about that times the
    number of ``predict.window_scales`` with several scales - where
    ``predict.decode`` is what turns them back into one label. Above 1 with
    ``decode: none`` it IS a bug, and that is the pairing ``load_config``
    refuses. ``uncovered`` is a bug either way: a detected frame no window speaks
    for comes back unlabelled and unpainted.

    Called by ``tests/test_dense_grid.py``, because a grid that quietly stops
    covering everything shows up as a missing row rather than as an error.
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
