"""Label every frame of every track with the trained model, as decoded bouts.

    trackanno predict --config config.yaml

Windows of ``clips.frames`` are laid over every track every
``predict.window_stride`` frames (see :mod:`trackanno.infer.densepredict`),
their class probabilities are averaged onto the frames they cover, and that
per-frame posterior is cut into bouts no shorter than
``predict.min_bout_frames``. :mod:`trackanno.infer.bouts` holds the arithmetic.

* **Predictions never touch the annotation CSV.** They go to
  ``predictions.parquet`` only, and every bout carries ``human_overlap_frac``
  so a consumer can tell hand-labelled frames from predicted ones.
* **A group the report gated off is not predicted**, and is named as skipped.
* **The file is replaced atomically.**
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .bouts import decode_min_bout, frame_posterior, runs_to_bouts
from ..measure.clipfeatures import CLIP_FEATURES_FILE
from ..annotate.clips import StagedWrite
from ..config import Config, load_config
from ..learn.suggest import MODEL_FILE, REPORT_FILE, Suggester

__all__ = [
    "PREDICTIONS_FILE", "Verdict", "verdict", "prediction_blocker",
    "predictions_path", "predict_all", "load_predictions", "staleness",
    "summarise", "main",
]

PREDICTIONS_FILE = "predictions.parquet"


def predictions_path(cfg: Config) -> Path:
    return cfg.paths.work_dir / PREDICTIONS_FILE


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
    payload; when the two disagree the refusing answer wins.
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
def _human_spans(cfg: Config, groups: list[str]) -> dict[tuple[str, int], list]:
    """{(movie_id, track_id): [(f_start, f_end)]} a person has already judged.

    A bout has no clip id, so "did a person decide this" is asked in FRAMES and
    answered as ``human_overlap_frac``. The spans of every predicted group are
    unioned, which over-reports contamination: the safe direction for a
    provenance flag. Read-only; a CSV that will not parse means nothing can be
    claimed to be hand labelled, not that the run stops.
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
        # An empty cell reads back as NaN, whose str() is "nan".
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
    OVERLAP each other (``clips.stride`` below ``clips.frames``) and summing
    per-clip overlaps would report a fraction above 1.
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

    Both ends are embedded so it cannot collide with a clip id (which is built
    from the start alone), and a bout row carries no ``clip_id`` column, so a
    merge on it fails loudly. Not unique across groups: the row key is
    (``group``, ``bout_id``).
    """
    return (f"{movie_id}__t{int(track_id):04d}"
            f"__f{int(f_start):06d}-{int(f_end):06d}")


def _rows_to_score(cfg: Config, suggester, reporter):
    """The dense feature table, its window ends, and why a row cannot be scored.

    ``skip_reason`` is "" for every row the model can be given a vector for.
    """
    from .densepredict import DENSE_FEATURES_FILE, build_dense_features

    # Embeddings are summarised in the model's own basis, never the one on disk.
    features = build_dense_features(cfg, reporter=reporter,
                                    embeddings=suggester.embedding_inputs(),
                                    required=suggester.columns)
    # The dense table is built without consulting the model, so this is where
    # the two meet; most often an embedding basis went missing with its columns.
    missing = [c for c in suggester.columns if c not in features.columns]
    if missing:
        shown = ", ".join(missing[:6]) + ("..." if len(missing) > 6 else "")
        raise RuntimeError(
            f"{DENSE_FEATURES_FILE} is missing {len(missing)} of the "
            f"{len(suggester.columns)} columns the model was fitted on "
            f"({shown}). Rebuild the clip features (retrain) so the saved "
            "embedding bases and the model agree, then label again."
        )
    # An input undefined for EVERY window is one the model is guessing without.
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
    return features, features["f_end"].to_numpy(dtype=np.int64), reason


def _write_atomic(frame: pd.DataFrame, work: Path, name: str = PREDICTIONS_FILE) -> Path:
    """Write ``name`` beside its real name and rename it in.

    Everything downstream treats ``predictions.parquet`` as authoritative, so a
    run that dies mid-write must leave the previous predictions intact.
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
def _bout_predictions(
    cfg: Config, features: pd.DataFrame, ends: np.ndarray, skip_reason: np.ndarray,
    suggester, groups: list[str], reporter,
) -> pd.DataFrame:
    """One row per (group, decoded bout).

    The windows are read OFF THE FEATURE TABLE rather than re-planned with
    :func:`trackanno.infer.densepredict.plan_dense_windows`: the table's rows ARE that
    plan, and :func:`trackanno.infer.bouts.frame_posterior` matches windows to
    probability rows BY POSITION, so a second copy could disagree with the rows
    actually scored.

    A window that could not be featurised is LEFT OUT of the array handed to
    ``frame_posterior``, never passed as NaN, because that function's running sum
    would spread one NaN over every later frame. A frame no window reaches comes
    back as an all-NaN posterior row, decodes to ``-1``, and leaves the span as a
    bout with an empty label and a ``skip_reason``.

    The span decoded for a track is the union of ITS OWN WINDOWS, not the range
    of its detections: a window legitimately reaches past the first and last
    detection, and those frames carry real evidence.
    """
    movie_ids = features["movie_id"].astype(str).to_numpy()
    track_ids = features["track_id"].to_numpy(dtype=np.int64)
    starts = features["f_start"].to_numpy(dtype=np.int64)
    stops = np.asarray(ends, dtype=np.int64)
    scorable = np.asarray(skip_reason).astype(str) == ""
    scored = np.flatnonzero(scorable)
    # THE SCORED ROWS ONLY, ONE COLUMN AT A TIME. `features[columns].to_numpy(
    # dtype=np.float32)` would build two whole copies of the model's input space
    # (pandas materialises the subset at the table's float64, then numpy casts)
    # and `matrix[scored]` a third. Same values in the same order; the transient
    # is one column of the table.
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
        # the rows a skip_reason withheld would be a NaN block nothing reads.
        # `rank` keeps the compact form addressable by row position.
        scores = np.empty((0, len(classes)))
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
                # Caught here rather than in `frame_posterior`, which would
                # report a position within one track's windows and blame the
                # caller for passing an unscored window. This is the other case:
                # the window WAS scored and the model's answer is not a
                # probability.
                bad = int(np.flatnonzero(~np.isfinite(proba).all(axis=1))[0])
                raise RuntimeError(
                    f"{group}: the model returned a non-finite probability for the "
                    f"window at row {int(scored[bad])} of the dense feature table. "
                    "A decode averages probabilities across windows, so one of these "
                    "would spread over every frame the window touches"
                )
            scores = proba
        proba_by_group[group] = scores
        classes_by_group[group] = classes
        reporter.log(
            f"scoring {scored.size} of {len(features)} dense windows with {group}"
        )
        reporter.advance()

    # The models have been asked everything they are going to be asked, and the
    # decode below allocates per track.
    del matrix

    labelled = _human_spans(cfg, groups)
    by_track = pd.DataFrame(
        {"movie_id": movie_ids, "track_id": track_ids}
    ).groupby(["movie_id", "track_id"], sort=True).indices
    min_frames = int(cfg.predict.min_frames)
    min_bout = int(cfg.predict.min_bout_frames)

    rows: list[dict] = []
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

        for group in groups:
            posterior, covering = frame_posterior(
                spans, proba_by_group[group][rank[usable]], f_lo, f_hi
            )
            labels = decode_min_bout(posterior, min_bout)
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
                    # says it for a window: a bout at the start of a track, where
                    # only one window reaches, would otherwise read like one
                    # several windows agreed on.
                    "n_covering_mean": float(covering[lo:hi].mean()),
                    "skip_reason": "" if bout["label"] else (
                        f"no window over frames {bout['f_start']}-{bout['f_end']} "
                        f"could be featurised, so those frames have no posterior "
                        f"to decode; predict.min_frames is {min_frames}"
                    ),
                    "human_overlap_frac": float(human[lo:hi].mean()),
                }
                # Every predicted group gets its pair of columns on every row, so
                # the file has one schema rather than one per group; only the
                # group that owns the bout fills them in. The renderer already
                # skips an empty prediction, so another group's rows are
                # invisible to it.
                for other in groups:
                    row[f"{other}_pred"] = bout["label"] if other == group else ""
                    row[f"{other}_conf"] = bout["conf"] if other == group else np.nan
                rows.append(row)
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
    # One flag for all groups; the fraction beside it says how much of the bout a
    # person actually decided.
    out["human_labelled"] = out["human_overlap_frac"] > 0.0
    return out


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
def predict_all(cfg: Config, reporter=None) -> pd.DataFrame:
    """Score the whole project with the trained model, into ``predictions.parquet``.

    One row per (group, decoded bout), carrying ``bout_id`` and no ``clip_id``;
    see :func:`_bout_id`. Every window is scored, hand-labelled or not, and each
    bout reports ``human_overlap_frac``.
    """
    from ..progress import NullReporter

    reporter = reporter or NullReporter()
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

    # Checked before the features are built, which is the expensive part.
    length = int(cfg.clips.frames)
    suggester.refuse_window_mismatch(
        length,
        f"The dense window is clips.frames ({length}); retrain at this length, "
        "or put clips.frames back.",
        reporter,
    )
    suggester.refuse_context_mismatch(cfg.suggest.context_frames)

    reporter.raise_if_cancelled()
    features, ends, skip_reason = _rows_to_score(cfg, suggester, reporter)
    reporter.raise_if_cancelled()
    out = _bout_predictions(
        cfg, features, ends, skip_reason, suggester, groups, reporter
    )

    target = _write_atomic(out, cfg.paths.work_dir)
    reporter.log(f"{len(out)} bouts -> {target}")
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
            "the Model page first (or `trackanno predict`)."
        )
    frame = pd.read_parquet(path)
    if frame.empty:
        raise RuntimeError(f"{path} has no rows; re-run the labelling pass")
    if "bout_id" not in frame.columns:
        raise RuntimeError(
            f"{path} holds windows from an older version, not decoded bouts; "
            "re-run the labelling pass"
        )
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

    A bout table carries one row per (group, bout), so the groups' identical
    spans are merged per (movie_id, track_id) before counting.
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
    """Human-readable tally of a predictions frame, for the log and the GUI."""
    groups = [c[: -len("_pred")] for c in frame.columns if c.endswith("_pred")]
    n_human = int(frame["human_labelled"].sum()) if "human_labelled" in frame else 0
    lines = [
        # One row per (group, bout): with three groups "600 bouts" is 200
        # stretches of behaviour asked three questions.
        f"{len(frame)} bouts over {len(groups)} category group(s), {n_human} of "
        "them overlapping a clip a person labelled"
    ]
    if "skip_reason" in frame.columns:
        n_skipped = int((frame["skip_reason"].astype(str) != "").sum())
        lines.append(
            f"  covering {_distinct_frames(frame)} distinct track-frames; "
            f"{n_skipped} bout(s) span frames no window could be featurised over "
            "and carry a skip_reason instead"
        )
    for group in groups:
        pred = frame[f"{group}_pred"].astype(str)
        conf = pd.to_numeric(frame[f"{group}_conf"], errors="coerce")
        given = pred.ne("") & pred.str.lower().ne("nan")
        counts = pred[given].value_counts()
        mean_conf = float(conf[given].mean()) if given.any() else float("nan")
        tally = ", ".join(f"{name} {int(n)}" for name, n in counts.items()) or "nothing"
        lines.append(
            f"  {group}: {int(given.sum())} bouts predicted ({tally}); "
            f"mean confidence {mean_conf:.3f}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="trackanno predict",
        description="Label every frame with the trained model, as decoded bouts "
                    "(never the annotation CSV).",
    )
    p.add_argument("--config", default=None, help="path to config.yaml")
    args = p.parse_args(argv)
    cfg = load_config(args.config)

    from ..progress import ConsoleReporter

    try:
        frame = predict_all(cfg, reporter=ConsoleReporter())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"\n{summarise(frame)}\nWrote {predictions_path(cfg)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
