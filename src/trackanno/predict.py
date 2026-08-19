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
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .clipfeatures import CLIP_FEATURES_FILE
from .clips import MANIFEST_FILE, StagedWrite
from .config import Config, load_config
from .suggest import MODEL_FILE, REPORT_FILE, Suggester

__all__ = [
    "PREDICTIONS_FILE", "Verdict", "verdict", "prediction_blocker", "predictions_path",
    "predict_all", "load_predictions", "staleness", "summarise", "main",
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
# the run
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


def _write_atomic(frame: pd.DataFrame, work: Path) -> Path:
    """Write ``predictions.parquet`` beside its real name and rename it in.

    Everything downstream treats this file as authoritative, so a run that dies
    mid-write must leave the previous predictions intact.
    """
    work.mkdir(parents=True, exist_ok=True)
    staged = StagedWrite(work)
    try:
        frame.to_parquet(staged.path(PREDICTIONS_FILE), index=False)
    except BaseException:
        staged.discard()
        raise
    staged.commit()
    return work / PREDICTIONS_FILE


def predict_all(
    cfg: Config,
    reporter=None,
    include_labelled: bool = False,
    coverage: str | None = None,
) -> pd.DataFrame:
    """Score every clip with the trained model and write ``predictions.parquet``.

    One row per clip, whether or not the model had anything to say about it, so
    the file joins on ``clip_id`` with no reasoning about missing clips.
    ``coverage`` defaults to ``predict.coverage``; see the module docstring.

    ``include_labelled`` decides what happens on clips the annotator has already
    judged. The default leaves them blank: those clips were training data, so the
    model's opinion of them is not evidence. Turn it on to measure agreement,
    knowing the number is optimistic.
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

    reporter.raise_if_cancelled()
    features, ends, skip_reason, extra = _rows_to_score(cfg, suggester, coverage, reporter)
    reporter.raise_if_cancelled()

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
    reporter.stage("labelling clips", len(groups))
    for group in groups:
        reporter.raise_if_cancelled()

        # No `suggest.min_confidence` filter here, on purpose: it exists to stop
        # a near-uniform posterior anchoring the annotator, and a consumer that
        # wants a threshold can apply its own to the confidence column.
        keep = scorable.copy()
        if not include_labelled and labelled[group]:
            keep &= ~out["clip_id"].isin(labelled[group]).to_numpy()
        rows = np.flatnonzero(keep)
        reporter.log(f"labelling {len(rows)} of {len(out)} clips with {group}")

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

    target = _write_atomic(out, cfg.paths.work_dir)
    reporter.log(f"{len(out)} clips -> {target}")
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


def summarise(frame: pd.DataFrame) -> str:
    """Human-readable tally of a predictions frame, for the log and the GUI."""
    groups = [c[: -len("_pred")] for c in frame.columns if c.endswith("_pred")]
    n_human = int(frame["human_labelled"].sum()) if "human_labelled" in frame else 0
    lines = [f"{len(frame)} clips, {n_human} of them already labelled by hand"]
    if "skip_reason" in frame.columns:
        # These rows are present and blank, which looks exactly like a clip the
        # model chose not to label unless the count is in front of you.
        n_skipped = int((frame["skip_reason"].astype(str) != "").sum())
        n_frames = int(pd.to_numeric(frame["f_end"] - frame["f_start"]).sum())
        lines.append(
            f"  covering {n_frames} track-frames; {n_skipped} window(s) had too "
            "few detections to featurise and carry a skip_reason instead"
        )
    for group in groups:
        pred = frame[f"{group}_pred"].astype(str)
        conf = pd.to_numeric(frame[f"{group}_conf"], errors="coerce")
        given = pred.ne("") & pred.str.lower().ne("nan")
        counts = pred[given].value_counts()
        mean_conf = float(conf[given].mean()) if given.any() else float("nan")
        tally = ", ".join(f"{name} {int(n)}" for name, n in counts.items()) or "nothing"
        lines.append(
            f"  {group}: {int(given.sum())} predicted ({tally}); "
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
             "those clips were training data, so the number flatters the model)",
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
