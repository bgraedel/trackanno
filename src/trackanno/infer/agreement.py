"""Annotation windows whose label differs from the model's out-of-fold prediction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

__all__ = ["Disagreement", "differs", "find", "load_cv_predictions"]

_NOT_A_LABEL = {"", "nan", "none", "<na>"}

IDENTITY = ("clip_id", "movie_id", "track_id", "f_start", "f_end")


def _groups(annotations: pd.DataFrame, suggestions: Mapping, given) -> list[str]:
    """The category group columns: those given, else those the suggestions name."""
    named = [g for g in (given or []) if g in annotations.columns]
    if named:
        return named
    return sorted({
        group for answer in suggestions.values() for group in (answer or {})
        if group in annotations.columns
    })


def load_cv_predictions(cfg) -> tuple[dict, str]:
    """``({clip_id: {group: (value, confidence)}}, problem)`` from the last fit's CV."""
    from ..learn.suggest import CV_PREDICTIONS_FILE

    path = cfg.paths.work_dir / CV_PREDICTIONS_FILE
    if not path.exists():
        return {}, ("no saved cross-validated predictions; Train the model again "
                    "on the Model page")
    try:
        table = pd.read_parquet(path)
    except (OSError, ValueError) as exc:
        return {}, f"{CV_PREDICTIONS_FILE} could not be read ({exc})"
    if table.empty:
        return {}, "the last fit produced no cross-validated predictions"
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
    """``["group: predicted"]`` for each group where an annotation row and the model differ."""
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
    """Every labelled window whose label differs from ``suggestions``, most confident first."""
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
    # Unknown confidence sorts last.
    out.sort(key=lambda d: (d.confidence is None, -(d.confidence or 0.0),
                            d.movie_id, d.track_id, d.f_start))
    return out

