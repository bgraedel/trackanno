"""Annotation windows whose recorded label is not what the model predicted.

A confusion matrix says a model confuses two classes; this says which clips to
open. Three rules:

* The unit is the annotation window, one row of ``annotations.csv``. No frame
  arithmetic and no decode.
* The model's half is the OUT-OF-FOLD prediction, from a model fitted without
  that clip's movie. The saved model was fitted on these labels and reproduces
  them, so its own opinion of one is not evidence about it.
* A window no fold scored is not a disagreement, and :func:`summarise` counts
  those separately. Most confident first.

Clips in the audit holdout have no fold prediction and so never appear, which
keeps the model's opinion of the last unbiased sample away from a reviewer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

__all__ = ["Disagreement", "differs", "find", "load_cv_predictions", "summarise"]

_NOT_A_LABEL = {"", "nan", "none", "<na>"}

IDENTITY = ("clip_id", "movie_id", "track_id", "f_start", "f_end")


def _groups(annotations: pd.DataFrame, suggestions: Mapping, given) -> list[str]:
    """Which columns are category groups: named by the caller, or else taken
    from the suggestions. Not from the CSV's columns, most of which are
    ``notes``, ``annotator`` and the rest."""
    named = [g for g in (given or []) if g in annotations.columns]
    if named:
        return named
    return sorted({
        group for answer in suggestions.values() for group in (answer or {})
        if group in annotations.columns
    })


def load_cv_predictions(cfg) -> tuple[dict, str]:
    """``({clip_id: {group: (value, confidence)}}, problem)`` from the last fit.

    Out-of-fold. ``problem`` is a sentence to show instead when there are none.
    """
    from ..learn.suggest import CV_PREDICTIONS_FILE

    path = cfg.paths.work_dir / CV_PREDICTIONS_FILE
    if not path.exists():
        return {}, (
            "this project has no saved cross-validated predictions. Train the "
            "model again on the Model page: a model fitted before this build "
            "did not keep them, and the saved model's own opinion of a label it "
            "was fitted on is not evidence about that label"
        )
    try:
        table = pd.read_parquet(path)
    except (OSError, ValueError) as exc:
        return {}, f"{CV_PREDICTIONS_FILE} could not be read ({exc})"
    if table.empty:
        return {}, (
            "the last fit produced no cross-validated predictions, so no group "
            "cleared the checks needed to score one"
        )
    out: dict[str, dict[str, tuple[str, float]]] = {}
    for clip_id, group, predicted, confidence in zip(
        table["clip_id"].astype(str), table["group"].astype(str),
        table["predicted"].astype(str), table["confidence"],
    ):
        answer = _clean(predicted)
        if not answer:
            continue
        try:
            sure = float(confidence)
        except (TypeError, ValueError):
            sure = None
        out.setdefault(clip_id, {})[group] = (
            answer, None if sure is None or sure != sure else sure
        )
    return out, ""


@dataclass(frozen=True)
class Disagreement:
    """One labelled window the model would have labelled otherwise."""

    clip_id: str
    movie_id: str
    track_id: int
    f_start: int
    f_end: int
    group: str
    label: str          # what the person recorded
    suggestion: str     # what the model suggests instead
    confidence: float | None   # how sure it was, or None if it cannot say

    @property
    def frame(self) -> int:
        """The frame to look at: the middle of the window."""
        return int((self.f_start + self.f_end) // 2)


def _clean(value) -> str:
    text = str(value if value is not None else "").strip()
    return "" if text.lower() in _NOT_A_LABEL else text


def differs(labels: Mapping, suggestion: Mapping, groups=None) -> list[str]:
    """``["group: predicted"]`` for every group the two disagree on.

    ``labels`` is one annotation row, ``suggestion`` the model's answer for it.
    The one definition of the comparison, used by both screens.
    """
    out = []
    for group, answer in suggestion.items():
        if groups is not None and group not in groups:
            continue
        said = _clean(labels.get(group, ""))
        guess = _clean(answer[0] if isinstance(answer, (tuple, list)) else answer)
        if said and guess and said != guess:
            out.append(f"{group}: {guess}")
    return out


def find(annotations: pd.DataFrame, suggestions: Mapping, groups=None
         ) -> list[Disagreement]:
    """Every labelled window whose label is not what the model predicted.

    ``suggestions`` is ``{clip_id: {group: (value, confidence)}}``, the shape
    :func:`load_cv_predictions` returns. Most confident first.
    """
    if annotations.empty or not suggestions:
        return []
    if any(c not in annotations.columns for c in IDENTITY):
        return []

    columns = {c: annotations[c].to_numpy() for c in IDENTITY}
    named = _groups(annotations, suggestions, groups)
    said = {g: annotations[g].to_numpy() for g in named}

    out: list[Disagreement] = []
    for i in range(len(annotations)):
        clip_id = str(columns["clip_id"][i])
        answers = suggestions.get(clip_id)
        if not answers:
            continue
        for group, answer in answers.items():
            if group not in said:
                continue
            label = _clean(said[group][i])
            guess = _clean(answer[0])
            if not label or not guess or label == guess:
                continue
            try:
                out.append(Disagreement(
                    clip_id=clip_id,
                    movie_id=str(columns["movie_id"][i]),
                    track_id=int(columns["track_id"][i]),
                    f_start=int(columns["f_start"][i]),
                    f_end=int(columns["f_end"][i]),
                    group=group, label=label, suggestion=guess,
                    confidence=None if answer[1] is None else float(answer[1]),
                ))
            except (TypeError, ValueError):
                continue        # a row whose identity will not parse
    # Unknown last: an estimator with no posterior said nothing about how sure
    # it was, which is not the same as saying it was not sure at all.
    out.sort(key=lambda d: (d.confidence is None, -(d.confidence or 0.0),
                            d.movie_id, d.track_id, d.f_start))
    return out


def summarise(annotations: pd.DataFrame, suggestions: Mapping,
              found: list[Disagreement], groups=None) -> dict:
    """How many labelled windows were compared, and how many were not.

    A count of disagreements with no denominator beside it reads as a model
    that agrees.
    """
    named = _groups(annotations, suggestions, groups)
    labelled = compared = 0
    if not annotations.empty and "clip_id" in annotations.columns:
        clip_ids = annotations["clip_id"].astype(str)
        for group in named:
            has_label = ~(
                annotations[group].astype(str).str.strip().str.lower()
                .isin(_NOT_A_LABEL)
            )
            answered = clip_ids.isin({
                clip_id for clip_id, groups in suggestions.items()
                if group in (groups or {})
            })
            labelled += int(has_label.sum())
            compared += int((has_label & answered).sum())

    per_group: dict[str, int] = {}
    pairs: dict[tuple[str, str], int] = {}
    for d in found:
        per_group[d.group] = per_group.get(d.group, 0) + 1
        pairs[(d.label, d.suggestion)] = pairs.get((d.label, d.suggestion), 0) + 1
    worst = max(pairs.items(), key=lambda kv: kv[1], default=None)
    return {
        "n_labelled": labelled,
        "n_compared": compared,
        "n_unanswered": labelled - compared,
        "n_disagree": len(found),
        "per_group": per_group,
        "most_common": (
            {"label": worst[0][0], "suggestion": worst[0][1], "n": worst[1]}
            if worst else None
        ),
    }
