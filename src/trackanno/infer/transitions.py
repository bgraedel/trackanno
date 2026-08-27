"""Transition rates and dwell times over decoded bouts.

    python -m trackanno.infer.transitions --config config.yaml

Reads ``predictions.parquet`` when it holds BOUTS, writes ``transitions.json``
beside it, and prints the same numbers. Four rules decide what is counted:

* A transition is a change between two CONSECUTIVE BOUTS of one track, never a
  frame-to-frame change. Frame counts are dominated by self-transitions the
  decode put there, so they measure ``predict.min_bout_frames``.
* A pair is formed only inside one (movie, track) and only across touching
  bouts. A gap, a track end and a movie end all break the pair rather than
  joining across it.
* A bout that runs to a track end or into a gap is right-censored: its true
  length is at least what was seen. A bout that starts at a track start or out
  of a gap has an unseen start and is dropped.
* Every rate is emitted beside the count it came from.

The states are decoded predictions, so the matrix carries the classifier's
errors: two classes it confuses appear as transitions between them.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .predict import RowKind, load_predictions, predictions_path, row_kind
from ..annotate.clips import HEADER_FILE, StagedWrite
from ..config import Config, load_config
from ..learn.suggest import REPORT_FILE

__all__ = [
    "TRANSITIONS_FILE", "transitions_path", "movie_fps", "kaplan_meier",
    "analyse", "compute", "write_report", "summarise", "main",
]

TRANSITIONS_FILE = "transitions.json"

# Columns a bout table must carry for any of this to mean anything.
BOUT_COLUMNS = (
    "bout_id", "movie_id", "track_id", "f_start", "f_end", "group",
    "human_overlap_frac", "n_covering_mean",
)

# What a run the decode could not label comes back as.
GAP = ""

_NOT_A_LABEL = {"", "nan", "none", "<na>"}


def transitions_path(cfg: Config) -> Path:
    return cfg.paths.work_dir / TRANSITIONS_FILE


# --------------------------------------------------------------------------- #
# what a row has to be
# --------------------------------------------------------------------------- #
def _refusal(kind: RowKind) -> str:
    """Why a non-bout predictions file cannot be turned into transitions."""
    return (
        f"these predictions are {kind.noun}, not bouts, and a transition is a "
        "change between two consecutive BOUTS. A decode is what cuts bouts out "
        "of the windows: set predict.coverage: dense and predict.decode: "
        "min_bout with predict.min_bout_frames: N (or predict.decode: viterbi "
        "with predict.self_transition), then run the labelling pass again."
    )


def _require_bouts(frame: pd.DataFrame) -> None:
    kind = row_kind(frame)
    if kind is not RowKind.BOUT:
        raise ValueError(_refusal(kind))
    missing = [c for c in BOUT_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(
            f"the bout table is missing {', '.join(missing)}; it was not written "
            "by this version of the decode, so relabel before measuring it"
        )


def _state(value) -> str:
    """A bout's class name, or ``""`` for a run the decode could not label."""
    text = str(value).strip()
    return GAP if text.lower() in _NOT_A_LABEL else text


# --------------------------------------------------------------------------- #
# frame rate
# --------------------------------------------------------------------------- #
def movie_fps(cfg: Config) -> dict[str, float]:
    """Acquisition frame rate per prepared movie, from each movie's cache.json."""
    out: dict[str, float] = {}
    work = cfg.paths.work_dir
    if not work.exists():
        return out
    for movie_dir in sorted(p for p in work.iterdir() if p.is_dir()):
        header = movie_dir / HEADER_FILE
        if not header.exists():
            continue
        try:
            rate = float(json.loads(header.read_text(encoding="utf-8"))["movie"]["fps"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if np.isfinite(rate) and rate > 0.0:
            out[movie_dir.name] = rate
    return out


def _fps_table(fps, movies) -> dict[str, float]:
    """``fps`` as {movie_id: rate}, whether it arrives as a number or a mapping."""
    if fps is None:
        return {}
    if isinstance(fps, (int, float)):
        rate = float(fps)
        return {m: rate for m in movies} if np.isfinite(rate) and rate > 0.0 else {}
    return {
        str(k): float(v) for k, v in dict(fps).items()
        if np.isfinite(float(v)) and float(v) > 0.0
    }


# --------------------------------------------------------------------------- #
# survival
# --------------------------------------------------------------------------- #
def kaplan_meier(durations, events) -> dict:
    """Survival estimate over right-censored durations.

    ``events[i]`` is True when bout ``i`` was seen to end, False when it only
    ran at least that long. Returns the median, the mean restricted to the
    longest observation (the plain sample mean when nothing is censored), and
    the horizon that mean is restricted to.
    """
    d = np.asarray(durations, dtype=float)
    e = np.asarray(events, dtype=bool)
    finite = np.isfinite(d)
    d, e = d[finite], e[finite]
    empty = {
        "n": 0, "n_events": 0, "n_censored": 0, "median": None,
        "restricted_mean": None, "restricted_to": None, "reaches_zero": False,
    }
    if d.size == 0:
        return empty

    curve: list[tuple[float, float]] = []
    surviving = 1.0
    for t in np.unique(d[e]):
        at_risk = int((d >= t).sum())
        died = int(((d == t) & e).sum())
        surviving *= 1.0 - died / at_risk
        curve.append((float(t), surviving))

    horizon = float(d.max())
    area, last_t, last_s = 0.0, 0.0, 1.0
    for t, s in curve:
        area += last_s * (t - last_t)
        last_t, last_s = t, s
    area += last_s * (horizon - last_t)

    median = next((t for t, s in curve if s <= 0.5), None)
    return {
        "n": int(d.size),
        "n_events": int(e.sum()),
        "n_censored": int((~e).sum()),
        "median": median,
        "restricted_mean": float(area),
        "restricted_to": horizon,
        "reaches_zero": bool(curve and curve[-1][1] <= 0.0),
    }


# --------------------------------------------------------------------------- #
# numbers into JSON
# --------------------------------------------------------------------------- #
def _f(value) -> float | None:
    """A float, or None where JSON would otherwise carry NaN."""
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _nanmean(values) -> float | None:
    """Mean over the finite entries, or None when there are none."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return _f(v.mean()) if v.size else None


def _stats(values) -> dict:
    """Descriptive summary of an array, all-None when it is empty."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return {"n": 0, "mean": None, "median": None, "p25": None, "p75": None,
                "min": None, "max": None}
    return {
        "n": int(v.size),
        "mean": _f(v.mean()),
        "median": _f(np.median(v)),
        "p25": _f(np.percentile(v, 25)),
        "p75": _f(np.percentile(v, 75)),
        "min": _f(v.min()),
        "max": _f(v.max()),
    }


# --------------------------------------------------------------------------- #
# the analysis
# --------------------------------------------------------------------------- #
def _floor(decode: str, min_bout_frames: int) -> tuple[int, str]:
    """The shortest bout that can exist, and the setting that says so."""
    n = int(min_bout_frames)
    if decode == "min_bout" and n > 1:
        return n, (
            f"predict.decode: min_bout with predict.min_bout_frames: {n}. No bout "
            f"shorter than {n} frame(s) can exist, so the left tail of every "
            "dwell distribution below is set by that number and not by the animal"
        )
    if decode == "viterbi":
        return 1, (
            f"predict.decode: viterbi. predict.min_bout_frames: {n} sets the "
            "self-transition prior, a preference for bouts about that long rather "
            "than a hard floor; one frame is still the shortest possible bout"
        )
    return 1, (
        f"predict.decode: {decode}. No hard floor; one frame is the shortest "
        "possible bout"
    )


def _confusion_note(report: dict | None, group: str) -> dict | None:
    """The model's own worst class confusion, as a pointer into suggest_report.json."""
    entry = (report or {}).get(group)
    if not isinstance(entry, dict):
        return None
    classes = [str(c) for c in entry.get("classes") or []]
    matrix = entry.get("confusion") or []
    worst = None
    if classes and len(matrix) == len(classes):
        for i, row in enumerate(matrix):
            total = float(sum(row))
            if total <= 0.0 or len(row) != len(classes):
                continue
            for j, count in enumerate(row):
                if i == j:
                    continue
                frac = float(count) / total
                if count and (worst is None or frac > worst["frac"]):
                    worst = {"from": classes[i], "to": classes[j], "frac": frac,
                             "n": int(count), "of": int(total)}
    return {
        "classes": classes,
        "kappa": _f(entry.get("kappa", float("nan"))),
        "accuracy": _f(entry.get("accuracy", float("nan"))),
        "reliable": bool(entry.get("reliable")),
        "most_confused": worst,
    }


def _analyse_group(
    sub: pd.DataFrame, group: str, rates: dict[str, float],
    exclude_human_overlap: bool, floor_frames: int, floor_source: str,
    model: dict | None,
) -> dict:
    """Transitions and dwell times for one category group's bouts."""
    sub = sub.sort_values(["movie_id", "track_id", "f_start"], kind="stable")
    state = np.array([_state(v) for v in sub[f"{group}_pred"]], dtype=object)
    movie = sub["movie_id"].astype(str).to_numpy()
    track = pd.to_numeric(sub["track_id"], errors="coerce").to_numpy(dtype=np.int64)
    f_start = pd.to_numeric(sub["f_start"], errors="coerce").to_numpy(dtype=np.int64)
    f_end = pd.to_numeric(sub["f_end"], errors="coerce").to_numpy(dtype=np.int64)
    overlap = pd.to_numeric(
        sub["human_overlap_frac"], errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)
    covering = pd.to_numeric(sub["n_covering_mean"], errors="coerce").to_numpy(dtype=float)
    frames = (f_end - f_start).astype(float)
    seconds = np.array(
        [frames[i] / rates[movie[i]] if movie[i] in rates else np.nan
         for i in range(len(sub))], dtype=float
    )
    is_gap = np.array([s == GAP for s in state], dtype=bool)
    dropped = (overlap > 0.0) if exclude_human_overlap else np.zeros(len(sub), dtype=bool)
    usable = ~is_gap & ~dropped

    # Track runs, in the sorted order above. A pair is never formed across one
    # of these boundaries, which is what keeps a track end and a movie end out
    # of the counts.
    keys = list(zip(movie.tolist(), track.tolist()))
    bounds: list[tuple[int, int]] = []
    lo = 0
    for i in range(1, len(keys) + 1):
        if i == len(keys) or keys[i] != keys[lo]:
            bounds.append((lo, i))
            lo = i

    counts: dict[str, dict[str, int]] = {}
    skipped = {"gap": 0, "not_adjacent": 0, "human_overlap": 0}
    n_pairs = 0
    censor_left = np.zeros(len(sub), dtype=bool)
    censor_right = np.zeros(len(sub), dtype=bool)

    for lo, hi in bounds:
        for i in range(lo, hi):
            if not usable[i]:
                continue
            before, after = i - 1, i + 1
            censor_left[i] = (
                i == lo or f_end[before] != f_start[i] or is_gap[before]
            )
            censor_right[i] = (
                after == hi or f_end[i] != f_start[after] or is_gap[after]
            )
        for i in range(lo, hi - 1):
            j = i + 1
            n_pairs += 1
            if f_end[i] != f_start[j]:
                skipped["not_adjacent"] += 1
            elif is_gap[i] or is_gap[j]:
                skipped["gap"] += 1
            elif dropped[i] or dropped[j]:
                skipped["human_overlap"] += 1
            else:
                counts.setdefault(state[i], {})
                counts[state[i]][state[j]] = counts[state[i]].get(state[j], 0) + 1

    observed = sorted({s for s in state[usable]})
    classes = sorted(set(observed) | set(counts) | {t for row in counts.values() for t in row})
    row_totals = {c: int(sum(counts.get(c, {}).values())) for c in classes}
    probabilities = {
        c: {t: _f(n / row_totals[c]) for t, n in sorted(counts.get(c, {}).items())}
        for c in classes if row_totals[c]
    }
    n_counted = int(sum(row_totals.values()))

    dwell: dict[str, dict] = {}
    for name in classes:
        member = usable & np.array([s == name for s in state], dtype=bool)
        keep = member & ~censor_left
        complete = keep & ~censor_right
        km_frames = kaplan_meier(frames[keep], (~censor_right)[keep])
        with_fps = keep & np.isfinite(seconds)
        km_seconds = kaplan_meier(seconds[with_fps], (~censor_right)[with_fps])
        dwell[name] = {
            "n_bouts": int(member.sum()),
            "n_complete": int(complete.sum()),
            "n_right_censored": int((keep & censor_right).sum()),
            "n_left_open_dropped": int((member & censor_left).sum()),
            "n_without_fps": int((keep & ~np.isfinite(seconds)).sum()),
            # Complete bouts only: every one of these was seen to start and to end.
            "frames": _stats(frames[complete]),
            "seconds": _stats(seconds[complete]),
            # Complete and right-censored together, the censored ones as lower bounds.
            "km_frames": km_frames,
            "km_seconds": km_seconds,
            "observed_min_frames": _f(frames[member].min()) if member.any() else None,
            "mean_n_covering": _nanmean(covering[member]),
        }

    exit_rates: dict[str, dict] = {}
    for name in classes:
        mean_seconds = dwell[name]["km_seconds"]["restricted_mean"]
        if not row_totals[name] or not mean_seconds:
            continue
        exit_rates[name] = {
            "exit_per_second": _f(1.0 / mean_seconds),
            "mean_dwell_seconds": _f(mean_seconds),
            "to": {
                t: _f(p / mean_seconds) for t, p in probabilities.get(name, {}).items()
                if p is not None
            },
        }

    return {
        "classes": classes,
        "n_bouts": int(len(sub)),
        "n_bouts_gap": int(is_gap.sum()),
        "n_bouts_human_overlap": int((overlap > 0.0).sum()),
        "n_bouts_used": int(usable.sum()),
        "n_tracks": len(bounds),
        "n_movies": int(len(set(movie.tolist()))),
        "mean_n_covering": _nanmean(covering[usable]),
        "transitions": {
            "counts": {c: dict(sorted(counts.get(c, {}).items())) for c in classes},
            "row_totals": row_totals,
            "probabilities": probabilities,
            "n_pairs_considered": n_pairs,
            "n_counted": n_counted,
            "skipped": skipped,
            "n_track_ends": len(bounds),
            "n_movie_ends": int(len(set(movie.tolist()))),
            "note": (
                "the jump chain of the bout sequence: the diagonal is empty by "
                "construction, because a decode emits maximal runs and two "
                "consecutive bouts never share a label. A probability is "
                "P(next bout is j | this bout is i and it ended), not a "
                "per-frame transition probability"
            ),
        },
        "dwell": {
            "floor_frames": floor_frames,
            "floor_source": floor_source,
            "per_class": dwell,
            "note": (
                "frames and seconds cover COMPLETE bouts only. km_* adds the "
                "right-censored ones as lower bounds via Kaplan-Meier; its "
                "restricted_mean is the mean truncated at restricted_to, and "
                "equals the sample mean when nothing is censored"
            ),
        },
        "rates": {
            "per_second": exit_rates,
            "basis": (
                "q(i->j) = P(next=j | in i) / mean dwell in i, the mean taken "
                "from km_seconds. This is the generator of a Markov jump process "
                "and assumes the dwell in each state is exponential"
            ),
            "caveat": (
                f"predict.min_bout_frames removes every bout shorter than "
                f"{floor_frames} frame(s), which lifts the mean dwell, so these "
                "rates are LOWER BOUNDS on the true ones"
                if floor_frames > 1 else
                "no minimum bout length is imposed, so the mean dwell is not "
                "inflated by a decode setting"
            ),
        },
        "model": model,
    }


def analyse(
    frame: pd.DataFrame,
    *,
    fps=None,
    decode: str = "none",
    min_bout_frames: int = 0,
    exclude_human_overlap: bool = False,
    source: str | None = None,
    model_report: dict | None = None,
    model_report_path: str | None = None,
) -> dict:
    """Transition matrix and dwell distributions for a bout table.

    ``frame`` must hold bouts; a windows or clips table is refused. ``fps`` is a
    rate or a {movie_id: rate} mapping and is what puts seconds beside frames.
    ``decode`` and ``min_bout_frames`` are the settings the bouts were cut with,
    and are reported as the floor on the dwell distribution.
    """
    _require_bouts(frame)
    movies = sorted(frame["movie_id"].astype(str).unique())
    rates = _fps_table(fps, movies)
    floor_frames, floor_source = _floor(str(decode), int(min_bout_frames))
    uniform = sorted({rates[m] for m in movies if m in rates})

    groups = sorted(frame["group"].astype(str).unique())
    missing = [g for g in groups if f"{g}_pred" not in frame.columns]
    if missing:
        raise ValueError(
            f"the bout table names group(s) {', '.join(missing)} but carries no "
            f"{missing[0]}_pred column, so nothing says what those bouts were "
            "labelled; relabel before measuring"
        )

    return {
        "schema_version": 1,
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "predictions": source,
            "row_kind": RowKind.BOUT.value,
            "n_rows": int(len(frame)),
            "decode": str(decode),
            "min_bout_frames": int(min_bout_frames),
            "suggest_report": model_report_path,
        },
        "options": {"exclude_human_overlap": bool(exclude_human_overlap)},
        "fps": {
            "per_movie": {m: _f(rates[m]) for m in movies if m in rates},
            "uniform": _f(uniform[0]) if len(uniform) == 1 else None,
            "missing": [m for m in movies if m not in rates],
        },
        "caveats": [
            "these are DECODED PREDICTIONS, not ground truth. Two classes the "
            "classifier confuses produce transitions between them that the "
            "animal never made"
            + (f"; compare against the confusion matrix in {model_report_path}"
               if model_report_path else
               f"; there is no {REPORT_FILE} beside these predictions to compare "
               "against, so nothing here says how often the model is wrong"),
            floor_source,
            "a transition is counted only between two touching bouts of one "
            "track. A gap, a track end and a movie end break the pair instead "
            "of joining across it",
            "bouts overlapping hand-labelled frames are training data, not "
            "independent evidence about the model"
            + (" and are excluded here" if exclude_human_overlap else
               "; pass --exclude-human-overlap to drop them"),
        ],
        "groups": {
            group: _analyse_group(
                frame.loc[frame["group"].astype(str) == group], group, rates,
                exclude_human_overlap, floor_frames, floor_source,
                _confusion_note(model_report, group),
            )
            for group in groups
        },
    }


# --------------------------------------------------------------------------- #
# reading and writing
# --------------------------------------------------------------------------- #
def _model_report(cfg: Config) -> tuple[dict | None, str | None]:
    path = cfg.paths.work_dir / REPORT_FILE
    if not path.exists():
        return None, None
    try:
        return json.loads(path.read_text(encoding="utf-8")), str(path)
    except (OSError, ValueError):
        return None, str(path)


def compute(cfg: Config, *, exclude_human_overlap: bool = False) -> dict:
    """Measure the project's predictions, without writing anything."""
    frame = load_predictions(cfg)
    report, report_path = _model_report(cfg)
    return analyse(
        frame,
        fps=movie_fps(cfg),
        decode=str(cfg.predict.decode),
        min_bout_frames=int(cfg.predict.min_bout_frames),
        exclude_human_overlap=exclude_human_overlap,
        source=str(predictions_path(cfg)),
        model_report=report,
        model_report_path=report_path,
    )


def write_report(cfg: Config, report: dict, name: str = TRANSITIONS_FILE) -> Path:
    """Write the report beside the predictions, renaming it in at the end."""
    work = cfg.paths.work_dir
    work.mkdir(parents=True, exist_ok=True)
    staged = StagedWrite(work)
    try:
        staged.path(name).write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )
    except BaseException:
        staged.discard()
        raise
    staged.commit()
    return work / name


# --------------------------------------------------------------------------- #
# the printed form
# --------------------------------------------------------------------------- #
def _n(value, spec: str = ".3g") -> str:
    return "-" if value is None else format(float(value), spec)


def _pair(frames_value, seconds_value) -> str:
    """A duration as frames and, where the frame rate is known, seconds."""
    if frames_value is None:
        return "-"
    text = f"{float(frames_value):.4g} f"
    return text if seconds_value is None else f"{text}  {float(seconds_value):.3g} s"


def summarise(report: dict) -> str:
    """The whole report as text: what follows what, how long, and on what evidence."""
    src = report["source"]
    fps = report["fps"]
    lines = ["transitions and dwell times over decoded bouts"]
    lines.append(
        f"  source: {src.get('predictions') or 'an in-memory bout table'} "
        f"({src['n_rows']} bout row(s), predict.decode: {src['decode']}, "
        f"predict.min_bout_frames: {src['min_bout_frames']})"
    )
    if fps["uniform"]:
        lines.append(
            f"  {fps['uniform']:g} fps across {len(fps['per_movie'])} movie(s)"
        )
    elif fps["per_movie"]:
        rates = ", ".join(f"{m} {r:g}" for m, r in sorted(fps["per_movie"].items()))
        lines.append(f"  frame rate differs between movies: {rates}")
    if fps["missing"]:
        lines.append(
            f"  no frame rate for {', '.join(fps['missing'])}; those bouts are "
            "measured in frames only"
        )
    for caveat in report["caveats"]:
        lines.append(f"  ! {caveat}")

    for group, g in report["groups"].items():
        lines.append("")
        lines.append(
            f"{group}: {len(g['classes'])} class(es), {g['n_bouts_used']} usable "
            f"bout(s) of {g['n_bouts']} over {g['n_tracks']} track(s) in "
            f"{g['n_movies']} movie(s)"
        )
        lines.append(
            f"  {g['n_bouts_gap']} bout(s) the decode could not label, "
            f"{g['n_bouts_human_overlap']} overlapping hand-labelled frames, "
            f"mean {_n(g['mean_n_covering'])} window(s) covering each bout"
        )
        model = g.get("model")
        if model:
            worst = model.get("most_confused")
            note = (
                f", most often mistakes {worst['from']} for {worst['to']} "
                f"({worst['n']} of {worst['of']} = {100 * worst['frac']:.0f}%)"
                if worst else ""
            )
            lines.append(
                f"  model: kappa {_n(model['kappa'], '+.3f')}, accuracy "
                f"{_n(model['accuracy'])}{note}"
            )

        t = g["transitions"]
        lines.append("")
        lines.append("  what follows what   P(next | this bout ended), with counts")
        for name in g["classes"]:
            total = t["row_totals"].get(name, 0)
            row = t["probabilities"].get(name, {})
            body = "   ".join(
                f"{to} {p:.2f} ({t['counts'][name][to]})"
                for to, p in sorted(row.items(), key=lambda kv: -(kv[1] or 0.0))
            ) or "nothing follows it in this data"
            lines.append(f"    {name:<14} n={total:<5} -> {body}")
        skipped = t["skipped"]
        lines.append(
            f"    {t['n_counted']} pair(s) counted of {t['n_pairs_considered']} "
            f"consecutive; skipped {skipped['gap']} across a gap, "
            f"{skipped['not_adjacent']} not touching, "
            f"{skipped['human_overlap']} at a hand-labelled bout"
        )
        lines.append(
            f"    {t['n_track_ends']} track-final bout(s) across "
            f"{t['n_movie_ends']} movie(s) have no successor by construction"
        )
        lines.append(f"    {t['note']}")

        d = g["dwell"]
        lines.append("")
        lines.append(
            f"  how long each state lasts   shortest bout the decode can emit: "
            f"{d['floor_frames']} frame(s)"
        )
        lines.append(
            f"    {'class':<14} {'compl':>6} {'cens':>5} {'drop':>5} "
            f"{'median (complete)':>22} {'mean (Kaplan-Meier)':>24}"
        )
        for name in g["classes"]:
            row = d["per_class"][name]
            lines.append(
                f"    {name:<14} {row['n_complete']:>6} "
                f"{row['n_right_censored']:>5} {row['n_left_open_dropped']:>5} "
                f"{_pair(row['frames']['median'], row['seconds']['median']):>22} "
                f"{_pair(row['km_frames']['restricted_mean'], row['km_seconds']['restricted_mean']):>24}"
            )
        lines.append(
            "    compl = a bout with an observed bout on both sides. cens = ran "
            "to a track end or into a gap, so its true length is at least what "
            "was seen; folded in by Kaplan-Meier. drop = started at a track "
            "start or out of a gap, so its own start was never seen."
        )
        lines.append(f"    {d['floor_source']}")

        rates = g["rates"]["per_second"]
        lines.append("")
        lines.append("  exit rates per second   q(i->j) = P(next=j) / mean dwell in i")
        if not rates:
            lines.append(
                "    none: no class has both a counted transition and a dwell "
                "estimate in seconds"
            )
        for name in g["classes"]:
            row = rates.get(name)
            if not row:
                continue
            body = "   ".join(
                f"{to} {q:.3g}" for to, q in
                sorted(row["to"].items(), key=lambda kv: -(kv[1] or 0.0))
            )
            lines.append(
                f"    {name:<14} exits {row['exit_per_second']:.3g}   {body}"
            )
        lines.append(f"    {g['rates']['caveat']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="transitions",
        description="Transition rates and dwell times over decoded bouts.",
    )
    p.add_argument("--config", default=None, help="path to config.yaml")
    p.add_argument(
        "--exclude-human-overlap", action="store_true",
        help="drop bouts overlapping hand-labelled frames; they were training "
             "data, so they are not independent evidence about the model. The "
             "counts either way are in the report",
    )
    p.add_argument(
        "--no-write", action="store_true",
        help=f"print the summary without writing {TRANSITIONS_FILE}",
    )
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    try:
        report = compute(cfg, exclude_human_overlap=args.exclude_human_overlap)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(summarise(report))
    if not args.no_write:
        print(f"\nWrote {write_report(cfg, report)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
