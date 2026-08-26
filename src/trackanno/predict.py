"""Label every clip with the trained model, into a file of its own.

    python -m trackanno.predict --config config.yaml

The model scores every clip in one pass, so the whole dataset can be exported,
plotted or compared against the hand labels. Three rules, all about provenance:

* **Predictions never touch the annotation CSV.** They go to
  ``predictions.parquet`` and nowhere else, or the next model trains on its own
  output and the ground truth is gone. Every row carries ``human_labelled`` so a
  consumer merging the two can tell them apart.
* **A group that the report gated off is not predicted**, and is named as
  skipped. The saved report decides, not the presence of a fitted model.
* **The file is replaced atomically**, since everything downstream reads it as
  authoritative.

``coverage`` decides which clips get a row: ``manifest`` takes only the clips
cut for annotation, leaving every frame between two windows unlabelled and
unpainted, while ``dense`` (the default) lays its own grid over every frame of
every track and adds ``n_present`` and ``skip_reason``, so a window the model
could not be given is still a row that says so. :mod:`trackanno.densepredict`
explains why that needs no pixels and how the two grids stay compatible.

``predict.decode`` decides what a row IS. With the default ``none`` a row is a
WINDOW, exactly as above, and this module has not changed. With a decode the
dense windows are averaged onto the frames they cover, that per-frame posterior
is cut into runs, and a row becomes a BOUT: a variable-length stretch between
two inferred behavioural boundaries. Bouts are what an overlapping
``predict.window_stride`` needs - several windows then speak for one frame, and
``load_config`` refuses that pairing without a decode - but a decode at the
tiling stride is legal and does something too, joining adjacent windows that
agreed and enforcing ``predict.min_bout_frames``.

``predict.window_scales`` widens the same posterior in the other direction: the
dense grid is built at several window LENGTHS and their rows all reach the same
average, so a behaviour shorter than the base window is visible to the shorter
scales and a long one is not chopped up by the longer. It needs the decode for
the same reason overlap does, and a model whose features do not scale with the
window length as well - see :meth:`trackanno.suggest.Suggester.scale_problem`.

Bouts go into ``predictions.parquet`` beside the windows rather than into a file
of their own, because they carry the same ``f_start``/``f_end``/``track_id``/
``<group>_pred``/``<group>_conf`` the renderer already reads and are disjoint by
construction, so ``range(f_start, f_end)`` per track keeps meaning what it meant.
What they do NOT carry is ``clip_id`` - see :func:`_bout_id` for why an absent
column is the safe answer there. :mod:`trackanno.bouts` holds the arithmetic.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .bouts import decode_min_bout, decode_viterbi, frame_posterior, runs_to_bouts
from .clipfeatures import CLIP_FEATURES_FILE
from .clips import MANIFEST_FILE, StagedWrite
from .config import Config, load_config
from .suggest import MODEL_FILE, REPORT_FILE, Suggester

__all__ = [
    "PREDICTIONS_FILE", "POSTERIORS_FILE", "Verdict", "verdict", "prediction_blocker",
    "predictions_path", "posteriors_path", "predict_all", "load_predictions",
    "staleness", "summarise", "main",
]

PREDICTIONS_FILE = "predictions.parquet"
# The per-frame posterior a decode was computed from, written only when
# `predict.save_posteriors` asks for it. Its own file rather than more columns
# on the predictions: it has one row per (track, frame) against one per bout, so
# folding it in would multiply the size of the file everything downstream reads.
POSTERIORS_FILE = "frame_posteriors.parquet"


def predictions_path(cfg: Config) -> Path:
    return cfg.paths.work_dir / PREDICTIONS_FILE


def posteriors_path(cfg: Config) -> Path:
    return cfg.paths.work_dir / POSTERIORS_FILE


# --------------------------------------------------------------------------- #
# what the model is allowed to label
# --------------------------------------------------------------------------- #
@dataclass
class Verdict:
    """Which groups may be predicted, which may not, and why not."""

    usable: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    problem: str = ""            # why NOTHING can be predicted, or ""


def verdict(cfg: Config) -> Verdict:
    """Read the saved report and decide what a bulk labelling run may do.

    Two file probes and a small JSON read, cheap enough for the Model page to
    ask on every repaint. The gate is ``suggest_report.json``, not the model
    payload; when the two disagree the conservative answer is the one that
    refuses.
    """
    work = cfg.paths.work_dir
    if not (work / MODEL_FILE).exists():
        return Verdict(problem="no model has been trained yet - train one on the Model page")
    if not (work / CLIP_FEATURES_FILE).exists():
        return Verdict(
            problem=f"{CLIP_FEATURES_FILE} is missing - retrain to rebuild it"
        )

    report_path = work / REPORT_FILE
    if not report_path.exists():
        return Verdict(
            problem=(
                f"there is no {REPORT_FILE} beside the model, so nothing says any "
                "category group ever cleared the reliability bar. Retrain on the "
                "Model page rather than labelling a whole project on trust."
            )
        )
    try:
        reports = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return Verdict(problem=f"could not read {report_path}: {exc}")

    out = Verdict()
    for group, rep in reports.items():
        if rep.get("reliable"):
            out.usable.append(group)
        else:
            out.skipped[group] = str(
                rep.get("reason") or "the report does not say it is reliable"
            )
    if not out.usable:
        detail = "; ".join(f"{g}: {why}" for g, why in out.skipped.items())
        out.problem = (
            "no category group is reliable enough to label with"
            + (f" ({detail})" if detail else "")
            + ". Annotate more clips and retrain."
        )
    return out


def prediction_blocker(cfg: Config) -> str:
    """Why a bulk labelling run must not start, or "" if it may."""
    return verdict(cfg).problem


# --------------------------------------------------------------------------- #
# what the rows are made of
# --------------------------------------------------------------------------- #
def _human_labels(cfg: Config, groups: list[str]) -> dict[str, set[str]]:
    """{group: clip_ids the annotator has already decided}, read-only.

    Feeds the ``human_labelled`` column and ``include_labelled``. The CSV is
    never written from here.
    """
    out: dict[str, set[str]] = {group: set() for group in groups}
    if not cfg.paths.csv.exists():
        return out
    try:
        frame = pd.read_csv(cfg.paths.csv)
    except (OSError, ValueError, pd.errors.ParserError):
        # A broken CSV must not stop the model labelling clips; it only means
        # nothing can be claimed to be hand labelled.
        return out
    if frame.empty or "clip_id" not in frame.columns:
        return out
    ids = frame["clip_id"].astype(str)
    for group in groups:
        if group not in frame.columns:
            continue
        # An empty cell reads back as NaN, whose str() is "nan" - truthy, and
        # taken as a real label it would hide clips nobody has looked at.
        values = frame[group].astype(str).str.strip()
        keep = values.ne("") & values.str.lower().ne("nan")
        out[group] = set(ids[keep])
    return out


def _human_spans(cfg: Config, groups: list[str]) -> dict[tuple[str, int], list]:
    """{(movie_id, track_id): [(f_start, f_end)]} a person has already judged.

    What ``human_labelled`` is built from once the rows stop being clips. The
    window path answers "did a person decide this" with an ``isin`` over
    ``clip_id``, which works only because a dense window's id is deliberately the
    same string as a coincident annotation clip's. A bout starts wherever a
    behaviour started and has no clip id at all, so the question has to be asked
    in FRAMES instead, and the answer is a fraction rather than a flag - hence
    ``human_overlap_frac``.

    The spans of every predicted group, UNIONED. A per-group version would be the
    sharper statement about leakage, since only a group's own labels trained that
    group's model, but ``human_labelled`` has meant "did a person decide any of
    this" since it was introduced and one column meaning two things in two modes
    is worse than a superset. A union over-reports contamination, which is the
    safe direction for a provenance flag.

    Read-only, like :func:`_human_labels`, and just as tolerant of a broken CSV:
    a file that will not parse means nothing can be CLAIMED to be hand labelled,
    not that the run stops.
    """
    out: dict[tuple[str, int], list] = {}
    if not cfg.paths.csv.exists():
        return out
    try:
        frame = pd.read_csv(cfg.paths.csv)
    except (OSError, ValueError, pd.errors.ParserError):
        return out
    needed = ("movie_id", "track_id", "f_start", "f_end")
    if frame.empty or any(c not in frame.columns for c in needed):
        return out

    decided = np.zeros(len(frame), dtype=bool)
    for group in groups:
        if group not in frame.columns:
            continue
        # Same emptiness rule as `_human_labels`: an empty cell reads back as
        # NaN, whose str() is "nan", and taken as a label it would claim a clip
        # nobody has looked at.
        values = frame[group].astype(str).str.strip()
        decided |= (values.ne("") & values.str.lower().ne("nan")).to_numpy()
    if not decided.any():
        return out

    rows = frame.loc[decided]
    movies = rows["movie_id"].astype(str).to_numpy()
    tracks = pd.to_numeric(rows["track_id"], errors="coerce").to_numpy()
    starts = pd.to_numeric(rows["f_start"], errors="coerce").to_numpy()
    ends = pd.to_numeric(rows["f_end"], errors="coerce").to_numpy()
    for movie_id, track_id, f_start, f_end in zip(movies, tracks, starts, ends):
        # A row whose geometry will not parse is dropped rather than defaulted:
        # a guessed span would put a hand label on frames nobody annotated.
        if not (np.isfinite(track_id) and np.isfinite(f_start) and np.isfinite(f_end)):
            continue
        out.setdefault((str(movie_id), int(track_id)), []).append(
            (int(f_start), int(f_end))
        )
    return out


def _labelled_mask(spans, f_lo: int, f_hi: int) -> np.ndarray:
    """Which frames of ``[f_lo, f_hi)`` fall inside a hand-labelled clip.

    A mask rather than an interval intersection, because annotation clips may
    OVERLAP each other - ``clips.stride`` below ``clips.frames`` cuts them that
    way on purpose - and summing per-clip overlaps would count a doubly-annotated
    frame twice and report a fraction above 1.
    """
    mask = np.zeros(max(int(f_hi) - int(f_lo), 0), dtype=bool)
    for start, end in spans:
        lo = max(int(start), int(f_lo)) - int(f_lo)
        hi = min(int(end), int(f_hi)) - int(f_lo)
        if hi > lo:
            mask[lo:hi] = True
    return mask


def _bout_id(movie_id: str, track_id: int, f_start: int, f_end: int) -> str:
    """The identity of one decoded bout, and deliberately NOT a ``clip_id``.

    ``clip_id_for`` is ``{movie}__t{track:04d}__f{f_start:06d}`` - derived from
    the START ALONE - and the dense grid is laid out so its windows collide with
    the annotation grid's ids ON PURPOSE: ``window_frames: auto`` picks
    ``min(clips.frames, clips.stride)`` precisely so the starts coincide, and
    that coincidence is the whole mechanism behind ``human_labelled`` and
    ``include_labelled``, which are set-membership tests over clip ids.

    A bout starts wherever a behaviour started. One that happens to begin on an
    annotated clip's ``f_start`` would therefore inherit that clip's entire
    identity while spanning a different number of frames, and a downstream merge
    of ``predictions.parquet`` with ``annotations.csv`` on ``clip_id`` would
    attribute a 300-frame bout's prediction to a 51-frame hand label. Nothing in
    this codebase checks length agreement on such a collision.

    So a bout carries this instead, embedding BOTH ends so it cannot collide with
    a clip id by construction, and a bout row carries no ``clip_id`` column at
    all. The absent column turns that merge into a KeyError naming ``clip_id``;
    a present-but-wrong one would be silent mis-attribution. Loud wins.

    Not unique on its own across GROUPS: each group decodes the same track into
    its own bouts and two of them may agree on a span. The row key is
    (``group``, ``bout_id``) - this names a stretch of a track, not a row.
    """
    return (f"{movie_id}__t{int(track_id):04d}"
            f"__f{int(f_start):06d}-{int(f_end):06d}")


def _clip_ends(cfg: Config, features: pd.DataFrame, reporter) -> np.ndarray:
    """Absolute end frame (exclusive) for each clip, from the clip manifests.

    ``clip_features.parquet`` records only ``f_start``, and clips at the end of a
    movie are shorter than ``clips.frames``, so assuming a fixed length would
    paint over frames the clip never covered.
    """
    starts = features["f_start"].to_numpy(dtype=np.int64)
    ends = np.full(len(features), -1, dtype=np.int64)
    lookup: dict[str, int] = {}
    for movie_id in sorted(features["movie_id"].astype(str).unique()):
        path = cfg.movie_dir(movie_id) / MANIFEST_FILE
        if not path.exists():
            continue
        manifest = pd.read_parquet(path, columns=["clip_id", "f_end"])
        lookup.update(
            zip(manifest["clip_id"].astype(str), manifest["f_end"].astype(np.int64))
        )
    for i, clip_id in enumerate(features["clip_id"].astype(str)):
        ends[i] = lookup.get(clip_id, -1)

    unmatched = ends < 0
    if unmatched.any():
        # The feature table and the clip cache came from different runs. Fall
        # back to the configured window rather than dropping the clip, and say
        # so: the fallback is the one number here that can be wrong.
        reporter.log(
            f"{int(unmatched.sum())} clip(s) are not in any clips.parquet manifest; "
            f"their frame range falls back to f_start + clips.frames "
            f"({cfg.clips.frames}). Rebuild the clip features if that matters.",
            "warning",
        )
        ends[unmatched] = starts[unmatched] + int(cfg.clips.frames)
    return ends


def _rows_to_score(cfg: Config, suggester, coverage: str, reporter):
    """The feature table to score, its frame ranges, and why a row cannot be.

    Returns ``(features, f_end, skip_reason, extra_columns)``; ``skip_reason`` is
    "" for every row the model can be given a vector for, and otherwise says
    what stopped it.
    """
    if coverage == "dense":
        from .densepredict import DENSE_FEATURES_FILE, build_dense_features

        features = build_dense_features(cfg, reporter=reporter)
        # The dense table is built without ever consulting the model, so this is
        # where the two meet. A missing column is not a degraded prediction but a
        # KeyError naming a column and neither file - most often the SAVED PCA
        # went missing and every cnn* column with it.
        missing = [c for c in suggester.columns if c not in features.columns]
        if missing:
            shown = ", ".join(missing[:6]) + ("..." if len(missing) > 6 else "")
            raise RuntimeError(
                f"{DENSE_FEATURES_FILE} is missing {len(missing)} of the "
                f"{len(suggester.columns)} columns the model was fitted on "
                f"({shown}). Rebuild the clip features (retrain) so the saved "
                "PCA and the model agree, then label again."
            )
        # A column the dense pass cannot compute arrives as NaN rather than as
        # an error - `_align_to_reference` fills it from the training table's
        # schema, so the model keeps its full input space. Right failure mode,
        # but invisible: an input undefined for EVERY window is one the model is
        # now guessing without.
        blank = [
            c for c in suggester.columns
            if not np.isfinite(features[c].to_numpy(dtype=np.float64)).any()
        ]
        if blank:
            shown = ", ".join(blank[:6]) + ("..." if len(blank) > 6 else "")
            reporter.log(
                f"{len(blank)} of the model's {len(suggester.columns)} input "
                f"columns are undefined for every dense window ({shown}); the "
                "model will see NaN for them. Retrain if they matter.",
                "warning",
            )
        reason = features["skip_reason"].astype(str).to_numpy()
        extra = {
            "n_present": features["n_present"].to_numpy(dtype=np.int64),
            "skip_reason": reason,
        }
        return features, features["f_end"].to_numpy(dtype=np.int64), reason, extra

    features = suggester.features
    if features is None or features.empty:
        raise RuntimeError(f"{CLIP_FEATURES_FILE} has no rows to label")
    # Every row of clip_features.parquet is a clip that was successfully
    # featurised, so on this path there is nothing that could be skipped.
    return (features, _clip_ends(cfg, features, reporter),
            np.full(len(features), "", dtype=object), {})


def _write_atomic(frame: pd.DataFrame, work: Path, name: str = PREDICTIONS_FILE) -> Path:
    """Write ``name`` beside its real name and rename it in.

    Everything downstream treats ``predictions.parquet`` as authoritative, so a
    run that dies mid-write must leave the previous predictions intact. The
    posterior file goes through the same route for the same reason: a truncated
    parquet under the real name is worse than no file, because only one of the
    two announces itself.
    """
    work.mkdir(parents=True, exist_ok=True)
    staged = StagedWrite(work)
    try:
        frame.to_parquet(staged.path(name), index=False)
    except BaseException:
        staged.discard()
        raise
    staged.commit()
    return work / name


# --------------------------------------------------------------------------- #
# bouts
# --------------------------------------------------------------------------- #
def _decode(cfg: Config, posterior: np.ndarray) -> np.ndarray:
    """Per-frame class indices from a per-frame posterior, per ``predict.decode``.

    ``none`` cannot be reached from a loaded config - it is what selects the
    window path in the first place - so arriving here with it means a Config was
    edited in place after loading, and the honest answer is to say so rather than
    to pick a decode nobody asked for.
    """
    decode = str(cfg.predict.decode)
    if decode == "min_bout":
        return decode_min_bout(posterior, int(cfg.predict.min_bout_frames))
    if decode == "viterbi":
        return decode_viterbi(posterior, cfg.predict.resolved_self_transition())
    raise RuntimeError(
        f"predict.decode is {decode!r}, so there is nothing to turn a per-frame "
        "posterior into bouts with. This run reached the decoding path anyway, "
        "which load_config's own checks make unreachable - the Config was "
        "changed after it was loaded."
    )


def _bout_predictions(
    cfg: Config, features: pd.DataFrame, ends: np.ndarray, skip_reason: np.ndarray,
    suggester, groups: list[str], reporter,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """One row per (group, decoded bout), and the posterior those bouts came from.

    The windows are read OFF THE FEATURE TABLE rather than re-planned with
    :func:`trackanno.densepredict.plan_dense_windows`. The table's rows ARE that
    plan - ``build_dense_features`` writes one per (track, window) - and
    :func:`trackanno.bouts.frame_posterior` matches windows to probability rows
    BY POSITION, so a second copy of the plan is a second thing that can disagree
    with the rows actually scored. Re-planning would also mean re-reading every
    movie's ``tracks.parquet`` and header to learn a grid this function is
    already holding.

    That is also why ``predict.window_scales`` needs nothing here. Windows of
    several LENGTHS are just more rows with different spans, and
    ``frame_posterior`` was written to take ``(f_start, f_end)`` pairs of
    arbitrary extent that may overlap, repeat and hang off either end - equal
    spans were never an assumption it made. The multi-scale posterior is
    therefore the same arithmetic over a wider set of windows, and the only
    thing that had to be decided was the WEIGHT each scale carries, which
    :func:`trackanno.densepredict.scale_grid` settles by making the covering
    counts equal before the mean is taken.

    A window that could not be featurised is LEFT OUT of the array handed to
    ``frame_posterior``, never passed as NaN: that function refuses a non-finite
    probability because its running sum would spread one NaN over every later
    frame. The frames those windows would have spoken for simply have fewer
    covering windows, or none - and a frame with none comes back as an all-NaN
    posterior row, decodes to ``-1``, and leaves the span as a bout with an empty
    label and a ``skip_reason``, which is the same convention a window under
    ``predict.min_frames`` already follows.

    The span decoded for a track is the union of ITS OWN WINDOWS, not the range
    of its detections. The two differ at both ends, where a window legitimately
    reaches past the first and last detection, and those frames carry real
    evidence: cutting them would throw away the model's opinion of exactly the
    frames the tiling path would have painted.

    ``n_covering`` does not depend on the group - the same windows are scorable
    for all of them - so it is taken from the first group's posterior and
    reported once.
    """
    movie_ids = features["movie_id"].astype(str).to_numpy()
    track_ids = features["track_id"].to_numpy(dtype=np.int64)
    starts = features["f_start"].to_numpy(dtype=np.int64)
    stops = np.asarray(ends, dtype=np.int64)
    scorable = np.asarray(skip_reason).astype(str) == ""
    scored = np.flatnonzero(scorable)
    # THE SCORED ROWS ONLY, ONE COLUMN AT A TIME. `features[columns].to_numpy(
    # dtype=np.float32)` builds two whole copies of the model's input space
    # before a single row is predicted - pandas materialises the column subset at
    # the table's own float64, then numpy casts that to float32 - and then
    # `matrix[scored]` copies a third time for the model. The dense table is one
    # row per window per track per SCALE, so on a multi-scale run over a real
    # project those copies are the largest things in the process, and the rows a
    # skip_reason withheld are never handed to a model at all. Same values in the
    # same order, so the predictions are unchanged; the per-column transient is
    # one column of the table.
    matrix = np.empty((scored.size, len(suggester.columns)), dtype=np.float32)
    for position, column in enumerate(suggester.columns):
        matrix[:, position] = features[column].to_numpy(dtype=np.float32)[scored]

    # Row of `features` -> its row in the compact posteriors below, -1 for a
    # window that was never scored. Every position the decode slices with comes
    # from `scorable`, so it never looks one of those up.
    rank = np.full(len(features), -1, dtype=np.int64)
    rank[scored] = np.arange(scored.size)

    proba_by_group: dict[str, np.ndarray] = {}
    classes_by_group: dict[str, list[str]] = {}
    reporter.stage("scoring dense windows", len(groups))
    for group in groups:
        reporter.raise_if_cancelled()
        model = suggester.models[group]
        classes = [str(c) for c in np.asarray(model.classes_)]
        # (n_scored, n_classes), not (n_rows, n_classes): these are held for
        # every group at once until the decode below has walked every track, and
        # the rows a skip_reason withheld would be a NaN block per group that
        # nothing ever reads - the slices taken below are all `scorable` rows.
        # `rank` is what keeps the compact form addressable by row position.
        scores = np.full((scored.size, len(classes)), np.nan)
        if scored.size:
            proba = np.asarray(model.predict_proba(matrix), dtype=np.float64)
            if proba.ndim != 2 or len(proba) != scored.size:
                raise RuntimeError(
                    f"{group}: the model returned {proba.shape} probabilities for "
                    f"{scored.size} windows; the model and the feature table disagree"
                )
            if proba.shape[1] != len(classes):
                raise RuntimeError(
                    f"{group}: the model returned {proba.shape[1]} probabilities per "
                    f"window but names {len(classes)} classes; the posterior columns "
                    "would be labelled with the wrong behaviour"
                )
            if not np.isfinite(proba).all():
                # Caught here rather than in `frame_posterior`, which would be
                # right to refuse it but would report a position within one
                # track's windows and blame the caller for passing an unscored
                # window. This is the other case: the window WAS scored and the
                # model's own answer is not a probability.
                bad = int(np.flatnonzero(~np.isfinite(proba).all(axis=1))[0])
                raise RuntimeError(
                    f"{group}: the model returned a non-finite probability for the "
                    f"window at row {int(scored[bad])} of the dense feature table. "
                    "A decode averages probabilities across windows, so one of these "
                    "would spread over every frame the window touches"
                )
            scores[:] = proba
        proba_by_group[group] = scores
        classes_by_group[group] = classes
        reporter.log(
            f"scoring {scored.size} of {len(features)} dense windows with {group}"
        )
        reporter.advance()

    # The models have been asked everything they are going to be asked, and the
    # decode below allocates per track. Dropped here rather than left to fall out
    # of scope at the end of the function, which is after the expensive part.
    del matrix

    labelled = _human_spans(cfg, groups)
    by_track = pd.DataFrame(
        {"movie_id": movie_ids, "track_id": track_ids}
    ).groupby(["movie_id", "track_id"], sort=True).indices
    min_frames = int(cfg.predict.min_frames)

    rows: list[dict] = []
    blocks: list[pd.DataFrame] = []
    reporter.stage("decoding bouts", len(by_track))
    for (movie_id, track_id), positions in by_track.items():
        reporter.raise_if_cancelled()
        positions = np.sort(np.asarray(positions, dtype=np.int64))
        f_lo = int(starts[positions].min())
        f_hi = int(stops[positions].max())
        usable = positions[scorable[positions]]
        spans = np.column_stack(
            [starts[usable], stops[usable]]
        ) if usable.size else np.zeros((0, 2), dtype=np.int64)
        human = _labelled_mask(
            labelled.get((str(movie_id), int(track_id)), ()), f_lo, f_hi
        )

        n_covering = None
        block: dict[str, np.ndarray] = {}
        for group in groups:
            posterior, covering = frame_posterior(
                spans, proba_by_group[group][rank[usable]], f_lo, f_hi
            )
            n_covering = covering if n_covering is None else n_covering
            labels = _decode(cfg, posterior)
            for bout in runs_to_bouts(labels, posterior, classes_by_group[group], f_lo):
                lo, hi = bout["f_start"] - f_lo, bout["f_end"] - f_lo
                row = {
                    "bout_id": _bout_id(
                        movie_id, track_id, bout["f_start"], bout["f_end"]
                    ),
                    "movie_id": str(movie_id),
                    "track_id": int(track_id),
                    "f_start": int(bout["f_start"]),
                    "f_end": int(bout["f_end"]),
                    "n_frames": int(bout["n_frames"]),
                    "group": group,
                    # How much evidence the row rests on, the way `n_present`
                    # says it for a window. Without it a bout at the very start
                    # of a track, where only one window reaches, reads exactly
                    # like one in the interior that five windows agreed on.
                    "n_covering_mean": float(covering[lo:hi].mean()),
                    "skip_reason": "" if bout["label"] else (
                        f"no window over frames {bout['f_start']}-{bout['f_end']} "
                        f"could be featurised, so those frames have no posterior "
                        f"to decode; predict.min_frames is {min_frames}"
                    ),
                    "human_overlap_frac": float(human[lo:hi].mean()),
                }
                # Every predicted group gets its pair of columns on every row,
                # so the file has one schema rather than one per group; only the
                # group that owns the bout fills them in. An empty prediction is
                # already what the renderer skips, so another group's rows are
                # invisible to it instead of being a second opinion on the frame.
                for other in groups:
                    row[f"{other}_pred"] = bout["label"] if other == group else ""
                    row[f"{other}_conf"] = bout["conf"] if other == group else np.nan
                rows.append(row)
            if cfg.predict.save_posteriors:
                for column, name in enumerate(classes_by_group[group]):
                    block[f"{group}__p__{name}"] = posterior[:, column]
        if cfg.predict.save_posteriors:
            n = f_hi - f_lo
            blocks.append(pd.DataFrame({
                "movie_id": np.full(n, str(movie_id), dtype=object),
                "track_id": np.full(n, int(track_id), dtype=np.int64),
                "frame": np.arange(f_lo, f_hi, dtype=np.int64),
                "n_covering": np.asarray(n_covering, dtype=np.int64),
                **block,
            }))
        reporter.advance()

    if not rows:
        raise RuntimeError(
            "the decode produced no bouts at all, which means the dense feature "
            "table had no windows to decode; rebuild it and label again"
        )
    out = pd.DataFrame(rows)
    for group in groups:
        out[f"{group}_pred"] = out[f"{group}_pred"].astype(str)
        out[f"{group}_conf"] = pd.to_numeric(out[f"{group}_conf"], errors="coerce")
    # Same one-flag-for-all-groups rule the window path uses; the fraction beside
    # it is what says how much of the bout a person actually decided.
    out["human_labelled"] = out["human_overlap_frac"] > 0.0
    posteriors = pd.concat(blocks, ignore_index=True) if blocks else None
    return out, posteriors


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
def _check_trained_window(suggester, length: int, coverage: str, reporter) -> None:
    """Refuse a model fitted at another window length; only warn if it never said.

    Window length changes what the features are WORTH, not what they are called,
    so a model fitted on 50-frame rows and handed 25-frame ones passes every
    check ``Suggester.load`` makes - same feature version, same column names -
    and then predicts confidently on a scale it has never seen. That is the one
    train/serve skew in this pipeline nothing else guards.

    The two answers are deliberately different. A MISMATCH is a known-wrong
    number and worth refusing over: every row of the output would be scored on
    the wrong scale and nothing in the file would show it. An UNRECORDED length
    is a model saved before the stamp existed, and refusing those would mean
    upgrading trackanno silently invalidates every model anyone already has - so
    it is a warning, and the run continues unchecked rather than not at all.
    ``window_problem`` returns a sentence in both cases so this caller can tell
    them apart by ``trained_window`` alone.
    """
    problem = suggester.window_problem(length)
    if not problem:
        return
    if suggester.trained_window is None:
        reporter.log(problem, "warning")
        return
    # `window_problem` says "retrain, or restore the window length it was trained
    # at" without being able to say WHICH key that is - it is a method on the
    # model and does not know which coverage asked. Naming the key here is the
    # difference between a diagnosis and an instruction, and the two coverages
    # need different instructions: on the dense path the length is a setting of
    # its own and can simply be written down, while on the manifest path it IS
    # the length the clip features were cut at, so there is nothing to write down
    # that would not also have to re-cut them.
    fix = (
        f"The {length} frames are predict.window_frames, this run's dense grid. "
        f"Writing predict.window_frames: {suggester.trained_window} is the edit "
        "that resolves it - no retraining, and not one existing label changes, "
        "because this decides what the predictor measures and not what anyone "
        "annotated."
        if coverage == "dense" else
        f"The {length} frames are clips.frames, the length "
        f"{CLIP_FEATURES_FILE} - the table manifest coverage scores - was cut at. "
        f"It read {suggester.trained_window} when this model was fitted, so "
        "either put clips.frames back and rebuild the clip features, or retrain "
        "at the length it reads now."
    )
    raise RuntimeError(f"{problem} {fix}")


def _check_scales(suggester, scales: list[int]) -> None:
    """Refuse a model that cannot be scored at several window lengths at once.

    ``_check_trained_window`` above has already established that the BASE scale
    is the one the model was fitted at. Every other scale is by definition one
    it was not, so the question is no longer "do these two numbers match" but
    "does this model read features that survive the difference" - and the answer
    is in the column names in its own payload.

    Unlike the unrecorded-window case this is never a warning. There is no
    legacy model to be kind to: a project only reaches here by having typed
    ``predict.window_scales`` into its config, and the mistake it makes is one
    the output cannot show. Every scale's rows would be scored at unchanged
    confidence and then averaged into the same per-frame posterior as the
    correct ones, so the damage is spread evenly over every bout in the file.

    ``load_config`` refuses the same combination against
    ``suggest.feature_scope``. It is not the same check: that one is about the
    model this project will fit next, and this one is about the model on disk.
    Retraining is exactly what moves a project from failing this to passing it,
    and a config edit alone must not.
    """
    problem = suggester.scale_problem(scales)
    if not problem:
        return
    raise RuntimeError(
        f"{problem} The scales are predict.window_scales, resolved against "
        f"predict.window_frames: {', '.join(str(s) for s in scales)} frames."
    )


def _window_predictions(
    cfg: Config, features: pd.DataFrame, ends: np.ndarray, skip_reason: np.ndarray,
    extra: dict, suggester, groups: list[str], include_labelled: bool, noun: str,
    reporter,
) -> pd.DataFrame:
    """One row per scored window, exactly as this module has always written them.

    Untouched by the decode work on purpose: every project on disk was labelled
    by this path, and ``predict.decode: none`` has to keep reproducing it row for
    row, column for column and value for value. Anything that would improve it
    belongs on the bout path until that guarantee is deliberately given up.
    """
    labelled = _human_labels(cfg, groups)
    # Through numpy rather than by assigning Series: the feature table's index
    # is whatever the parquet read produced, and index-aligned assignment would
    # pair a clip with another clip's prediction without raising.
    out = pd.DataFrame({
        "clip_id": features["clip_id"].astype(str).to_numpy(),
        "movie_id": features["movie_id"].astype(str).to_numpy(),
        "track_id": features["track_id"].to_numpy(dtype=np.int64),
        "f_start": features["f_start"].to_numpy(dtype=np.int64),
        "f_end": ends,
    })
    for name, values in extra.items():
        out[name] = values

    # The models are used directly rather than through Suggester.suggest, which
    # drops audit clips and low-confidence rows to stop the ANNOTATOR being
    # anchored. Nothing here is shown during annotation, and a hole in the
    # exported movie is a worse lie than a prediction with its confidence beside
    # it.
    matrix = features[suggester.columns].to_numpy(dtype=np.float32)
    # A skipped row is NaN in every column, which both backends read as
    # "undefined" and will happily turn into a class. Withheld from the model
    # rather than predicted and then blanked.
    scorable = skip_reason.astype(str) == ""
    reporter.stage(f"labelling {noun}", len(groups))
    for group in groups:
        reporter.raise_if_cancelled()

        # No `suggest.min_confidence` filter here, on purpose: it exists to stop
        # a near-uniform posterior anchoring the annotator, and a consumer that
        # wants a threshold can apply its own to the confidence column.
        keep = scorable.copy()
        if not include_labelled and labelled[group]:
            keep &= ~out["clip_id"].isin(labelled[group]).to_numpy()
        rows = np.flatnonzero(keep)
        reporter.log(f"labelling {len(rows)} of {len(out)} {noun} with {group}")

        pred = np.full(len(out), "", dtype=object)
        conf = np.full(len(out), np.nan, dtype=np.float64)
        if rows.size:
            model = suggester.models[group]
            proba = np.asarray(model.predict_proba(matrix[rows]), dtype=np.float64)
            if proba.ndim != 2 or len(proba) != rows.size:
                raise RuntimeError(
                    f"{group}: the model returned {proba.shape} probabilities for "
                    f"{rows.size} clips; the model and the feature table disagree"
                )
            chosen = proba.argmax(axis=1)
            classes = np.asarray(model.classes_).astype(str)
            pred[rows] = classes[chosen]
            conf[rows] = proba[np.arange(rows.size), chosen]
        out[f"{group}_pred"] = pred.astype(str)
        out[f"{group}_conf"] = conf
        reporter.advance()

    # One flag rather than one per group: a consumer merging this with the CSV
    # needs "did a person decide this clip", and any hand label makes it yes.
    any_labelled = set().union(*labelled.values()) if labelled else set()
    out["human_labelled"] = out["clip_id"].isin(any_labelled)
    return out


def predict_all(
    cfg: Config,
    reporter=None,
    include_labelled: bool = False,
    coverage: str | None = None,
) -> pd.DataFrame:
    """Score the whole project with the trained model, into ``predictions.parquet``.

    ``coverage`` defaults to ``predict.coverage`` and decides WHICH frames are
    scored; ``predict.decode`` decides what one row of the answer is. With the
    default ``decode: none`` it is one row per clip or per dense window, whether
    or not the model had anything to say about it, so the file joins on
    ``clip_id`` with no reasoning about missing clips. With a decode the windows
    are averaged onto frames and re-cut, and a row is a BOUT carrying ``bout_id``
    and no ``clip_id`` at all - see the module docstring and :func:`_bout_id`.

    ``include_labelled`` decides what happens on clips the annotator has already
    judged. The default leaves them blank: those clips were training data, so the
    model's opinion of them is not evidence. Turn it on to measure agreement,
    knowing the number is optimistic.

    UNDER A DECODE THAT FLAG DOES NOTHING, and says so in the log. Withholding a
    hand-labelled window there punches a hole in the per-frame posterior, and the
    decode does not merely lose those frames - a gap breaks the run structure
    either side of it, so the label of frames nobody annotated moves too. Under
    an overlapping stride it would not even work: the neighbouring windows, whose
    features come from almost the same frames, paint the annotated frames anyway.
    So every window is scored and the provenance moves onto the row, as
    ``human_overlap_frac`` - the share of the bout's frames a person did decide.
    A flag that quietly half-works is worse than one that openly does not.
    """
    from .progress import NullReporter

    reporter = reporter or NullReporter()
    coverage = str(coverage or cfg.predict.coverage)
    if coverage not in ("dense", "manifest"):
        raise ValueError(
            f"unknown coverage {coverage!r}; expected 'dense' or 'manifest'"
        )
    decision = verdict(cfg)
    if decision.problem:
        raise RuntimeError(decision.problem)

    suggester = Suggester(cfg)
    if not suggester.ready:
        raise RuntimeError(f"the model could not be loaded: {suggester.problem}")

    groups, skipped = [], dict(decision.skipped)
    for group in decision.usable:
        if group in suggester.models:
            groups.append(group)
        else:
            # Reliable per the report but absent from the payload: the two files
            # came from different runs.
            skipped[group] = (
                f"{REPORT_FILE} calls it reliable but {MODEL_FILE} holds no "
                "classifier for it; retrain"
            )
    if not groups:
        raise RuntimeError(
            "nothing can be predicted: "
            + "; ".join(f"{g} ({why})" for g, why in skipped.items())
        )
    for group, why in skipped.items():
        reporter.log(f"group {group!r} not predicted: {why}", "warning")

    # The length the rows about to be scored were MEASURED over: the dense grid's
    # own window on the dense path, and `clips.frames` on the manifest path,
    # where the rows are clip_features.parquet's and that table is built over the
    # annotation manifest. Checked before the features are built, because on the
    # dense path building them is the expensive part of the run.
    length = (cfg.predict.resolved_window(cfg.clips) if coverage == "dense"
              else int(cfg.clips.frames))
    _check_trained_window(suggester, length, coverage, reporter)

    # The other window-length skew, and it only exists on the dense path: the
    # manifest path scores clip_features.parquet, which is one table built at
    # one length, so there is nothing to score at several. Checked here for the
    # same reason as the base length - the feature build is the expensive part
    # of a dense run, and a refusal after it is a refusal that cost ten minutes.
    scales = cfg.predict.resolved_scales(cfg.clips) if coverage == "dense" else [length]
    _check_scales(suggester, scales)
    if len(scales) > 1:
        reporter.log(
            f"scoring at {len(scales)} window lengths "
            f"({', '.join(str(s) for s in scales)} frames), each carrying the same "
            "weight in the per-frame posterior. A transition shorter than the "
            f"{length}-frame grid is only visible to the shorter scales"
        )
    elif coverage != "dense" and len(cfg.predict.window_scales) > 1:
        # Reachable only through `--coverage manifest`, which overrides a config
        # `load_config` would otherwise have refused. Said rather than ignored:
        # the run produces a perfectly ordinary file, and nothing in it shows
        # that the scales the config asks for were not used.
        reporter.log(
            f"predict.window_scales lists {len(cfg.predict.window_scales)} window "
            "lengths but this run was asked for manifest coverage, which scores "
            f"{CLIP_FEATURES_FILE} - one table, built over the annotation clip "
            f"grid at {int(cfg.clips.frames)} frames. Nothing is scored at any "
            "other length.",
            "warning",
        )

    stride = cfg.predict.resolved_stride(cfg.clips)
    decoding = coverage == "dense" and str(cfg.predict.decode) != "none"
    if coverage != "dense" and str(cfg.predict.decode) != "none":
        reporter.log(
            f"predict.decode is {cfg.predict.decode!r} but this run was asked for "
            "manifest coverage, which scores only the clips cut for annotation and "
            "so has no posterior for the frames between them. Nothing is decoded: "
            "the rows below are clips, not bouts.",
            "warning",
        )
    if decoding and include_labelled:
        reporter.log(
            "--include-labelled has no effect under predict.decode "
            f"{cfg.predict.decode!r}: every window is scored either way. "
            "Withholding a hand-labelled window would leave its frames with no "
            "posterior, and a gap re-cuts the bouts on both sides of it"
            + (f"; at predict.window_stride {stride} against a "
               f"{length}-frame window the neighbouring windows would paint those "
               "frames from nearly the same features anyway"
               if stride < length else "")
            + ". Read human_overlap_frac on each bout instead.",
            "warning",
        )

    reporter.raise_if_cancelled()
    features, ends, skip_reason, extra = _rows_to_score(cfg, suggester, coverage, reporter)
    reporter.raise_if_cancelled()

    noun = "bouts" if decoding else ("dense windows" if coverage == "dense" else "clips")
    posteriors = None
    if decoding:
        out, posteriors = _bout_predictions(
            cfg, features, ends, skip_reason, suggester, groups, reporter
        )
    else:
        out = _window_predictions(
            cfg, features, ends, skip_reason, extra, suggester, groups,
            include_labelled, noun, reporter,
        )

    target = _write_atomic(out, cfg.paths.work_dir)
    reporter.log(f"{len(out)} {noun} -> {target}")
    if posteriors is not None:
        where = _write_atomic(posteriors, cfg.paths.work_dir, POSTERIORS_FILE)
        reporter.log(f"{len(posteriors)} (track, frame) posterior row(s) -> {where}")
    for line in summarise(out).splitlines():
        reporter.log(line)
    return out


# --------------------------------------------------------------------------- #
# reading it back
# --------------------------------------------------------------------------- #
def load_predictions(cfg: Config, group_name: str | None = None) -> pd.DataFrame:
    """Read ``predictions.parquet``, or say what to do about its absence.

    Raises rather than returning an empty frame: on screen "nothing was drawn"
    is indistinguishable from "the model predicted nothing".
    """
    path = predictions_path(cfg)
    if not path.exists():
        raise FileNotFoundError(
            f"no predictions at {path}. Run 'Label all clips with the model' on "
            "the Model page first (or `python -m trackanno.predict`)."
        )
    frame = pd.read_parquet(path)
    if frame.empty:
        raise RuntimeError(f"{path} has no rows; re-run the labelling pass")
    if group_name is not None and f"{group_name}_pred" not in frame.columns:
        available = sorted(
            c[: -len("_pred")] for c in frame.columns if c.endswith("_pred")
        )
        raise ValueError(
            f"the predictions carry no column for {group_name!r} - it was not "
            f"reliable enough to label with when they were written. Predicted "
            f"groups: {available or 'none'}"
        )
    return frame


def staleness(cfg: Config) -> str:
    """Why the predictions no longer match what made them, or "" if they do.

    Only mtimes, so it is cheap enough to ask on every render. Never blocks: a
    stale file is reported, not refused.
    """
    path = predictions_path(cfg)
    if not path.exists():
        return f"there are no predictions at {path}"
    work = cfg.paths.work_dir
    when = path.stat().st_mtime
    notes = []
    for name, what in ((MODEL_FILE, "the model was retrained"),
                       (CLIP_FEATURES_FILE, "the clip features were rebuilt")):
        other = work / name
        if other.exists() and other.stat().st_mtime > when:
            notes.append(f"{what} after these predictions were written")
    return "; ".join(notes)


def _distinct_frames(frame: pd.DataFrame) -> int:
    """Frames at least one row speaks for, counted ONCE.

    ``(f_end - f_start).sum()`` was the old answer and is a track-frame count
    only while the rows are disjoint. They are not, in either of the two ways
    this file is now written: an overlapping ``predict.window_stride`` puts every
    frame in about ``window_frames / window_stride`` windows, and a bout table
    carries one row per (group, bout), so a project with three category groups
    would claim three times the frames its movies contain. Both inflate a number
    a reader has no way to sanity-check - it is only ever compared against
    itself, run to run.

    Intervals are merged per (movie_id, track_id), so overlapping annotation
    clips (``clips.stride`` below ``clips.frames``) and the several groups'
    identical bout spans all collapse to the frames they actually cover. A frame
    of two different tracks is two track-frames, which is what the name says.
    """
    if not {"movie_id", "track_id", "f_start", "f_end"}.issubset(frame.columns):
        return 0
    starts = pd.to_numeric(frame["f_start"], errors="coerce").to_numpy(dtype=np.int64)
    ends = pd.to_numeric(frame["f_end"], errors="coerce").to_numpy(dtype=np.int64)
    keys = pd.DataFrame({
        "movie_id": frame["movie_id"].astype(str).to_numpy(),
        "track_id": pd.to_numeric(frame["track_id"], errors="coerce").to_numpy(),
    })
    total = 0
    for positions in keys.groupby(["movie_id", "track_id"], sort=False).indices.values():
        order = np.asarray(positions)[np.argsort(starts[positions], kind="stable")]
        lo, hi = starts[order[0]], ends[order[0]]
        for i in order[1:]:
            if starts[i] > hi:
                total += max(int(hi - lo), 0)
                lo, hi = starts[i], ends[i]
            else:
                hi = max(hi, ends[i])
        total += max(int(hi - lo), 0)
    return total


def summarise(frame: pd.DataFrame) -> str:
    """Human-readable tally of a predictions frame, for the log and the GUI.

    What one row IS is read off the FILE, never off the config: ``predict.decode``
    can be changed after a run, and a summary that consulted it would describe
    the run somebody is about to do instead of the one that wrote these rows.
    ``bout_id`` is what a decoded row carries in place of ``clip_id``, and
    ``skip_reason`` is what the dense grid adds to a window - the same two probes
    :func:`trackanno.render.prediction_unit` makes, and for the same reason.
    """
    groups = [c[: -len("_pred")] for c in frame.columns if c.endswith("_pred")]
    n_human = int(frame["human_labelled"].sum()) if "human_labelled" in frame else 0
    bouts = "bout_id" in frame.columns
    noun = "bouts" if bouts else ("dense windows" if "skip_reason" in frame.columns
                                  else "clips")
    lines = [
        f"{len(frame)} {noun}"
        # Spelled out because a bout table has one row per (group, bout): with
        # three groups "600 bouts" is 200 stretches of behaviour asked three
        # questions, and the sentence has to say so or the number reads as
        # three times the animal's behaviour.
        + (f" over {len(groups)} category group(s)" if bouts else "")
        + f", {n_human} of them "
        + ("overlapping a clip a person labelled" if bouts
           else "already labelled by hand")
    ]
    if "skip_reason" in frame.columns:
        # These rows are present and blank, which looks exactly like a clip the
        # model chose not to label unless the count is in front of you.
        n_skipped = int((frame["skip_reason"].astype(str) != "").sum())
        n_frames = _distinct_frames(frame)
        lines.append(
            f"  covering {n_frames} distinct track-frames; {n_skipped} "
            + (f"{noun[:-1]}(s) span frames no window could be featurised over"
               if bouts else
               f"{noun[:-1]}(s) had too few detections to featurise")
            + " and carry a skip_reason instead"
        )
    for group in groups:
        pred = frame[f"{group}_pred"].astype(str)
        conf = pd.to_numeric(frame[f"{group}_conf"], errors="coerce")
        given = pred.ne("") & pred.str.lower().ne("nan")
        counts = pred[given].value_counts()
        mean_conf = float(conf[given].mean()) if given.any() else float("nan")
        tally = ", ".join(f"{name} {int(n)}" for name, n in counts.items()) or "nothing"
        # Named, not bare "predicted": under a decode this counts the group's own
        # bouts and skips the other groups' rows, which carry an empty prediction
        # in this column, so "N predicted" out of a row count several times larger
        # would look like a coverage failure.
        lines.append(
            f"  {group}: {int(given.sum())} {noun} predicted ({tally}); "
            f"mean confidence {mean_conf:.3f}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="predict",
        description="Label every clip with the trained model (never the annotation CSV).",
    )
    p.add_argument("--config", default=None, help="path to config.yaml")
    p.add_argument(
        "--include-labelled", action="store_true",
        help="also predict clips you have already labelled (agreement checking; "
             "those clips were training data, so the number flatters the model). "
             "IGNORED under predict.decode: a decode needs a posterior for every "
             "frame, so every window is scored and each bout reports "
             "human_overlap_frac instead",
    )
    p.add_argument(
        "--coverage", choices=("dense", "manifest"), default=None,
        help="which clips to score; defaults to predict.coverage in the config. "
             "dense = a window grid over every frame of every track (no pixels "
             "are read, nothing is re-detected); manifest = only the clips that "
             "were cut for annotation, leaving everything between them unlabelled",
    )
    args = p.parse_args(argv)
    cfg = load_config(args.config)

    from .progress import ConsoleReporter

    try:
        frame = predict_all(
            cfg, reporter=ConsoleReporter(), include_labelled=args.include_labelled,
            coverage=args.coverage,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"\n{summarise(frame)}\nWrote {predictions_path(cfg)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
