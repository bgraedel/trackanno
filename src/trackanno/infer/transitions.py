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

On top of those counts it fits the continuous-time Markov chain of Boggon et al.
2026 (Nat Commun, doi:10.1038/s41467-026-75076-8) Fig. 1h-i: a mean lifetime per
state, the measured survival curve beside that fit, the jump matrix, the rates
between states, and the share of time each state holds. Error bars resample
whole tracks.

The states are decoded predictions, so the matrix carries the classifier's
errors: two classes it confuses appear as transitions between them.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
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
    "exponential_fit", "analyse", "compute", "write_report", "summarise",
    "plot", "plot_survival", "main",
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
        "curve_t": [], "curve_p": [],
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
    knots = _thin([(0.0, 1.0)] + curve)
    return {
        "n": int(d.size),
        "n_events": int(e.sum()),
        "n_censored": int((~e).sum()),
        "median": median,
        "restricted_mean": float(area),
        "restricted_to": horizon,
        "reaches_zero": bool(curve and curve[-1][1] <= 0.0),
        # The curve itself, so P(t > t) can be drawn and not only summarised.
        "curve_t": [float(t) for t, _ in knots],
        "curve_p": [float(s) for _, s in knots],
    }


def _thin(curve: list[tuple[float, float]], most: int = 240
          ) -> list[tuple[float, float]]:
    """At most ``most`` points of a step curve, keeping the first and the last."""
    if len(curve) <= most:
        return curve
    take = np.unique(np.linspace(0, len(curve) - 1, most).round().astype(int))
    return [curve[i] for i in take]


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
# the fitted Markov chain
# --------------------------------------------------------------------------- #
def exponential_fit(durations, events) -> dict:
    """Constant-hazard fit to right-censored durations.

    ``events[i]`` is True when bout ``i`` was seen to end. The maximum
    likelihood rate is the number of ends over the total time at risk, so a
    censored bout contributes its time but not an end. ``mean`` is None when
    nothing was seen to end.
    """
    d = np.asarray(durations, dtype=float)
    e = np.asarray(events, dtype=bool)
    finite = np.isfinite(d)
    d, e = d[finite], e[finite]
    ends, at_risk = int(e.sum()), float(d.sum())
    if ends == 0 or at_risk <= 0.0:
        return {"n": int(d.size), "n_ends": ends, "mean": None, "rate": None}
    mean = at_risk / ends
    return {"n": int(d.size), "n_ends": ends, "mean": _f(mean),
            "rate": _f(1.0 / mean)}


def _stationary(rho: np.ndarray) -> np.ndarray | None:
    """A jump chain's stationary distribution, or None where it has no unique
    one: a class with no observed successor, or classes that split into groups
    never reaching each other."""
    n = rho.shape[0]
    if n == 0 or not np.all(np.isfinite(rho)):
        return None
    if not np.allclose(rho.sum(axis=1), 1.0, atol=1e-9):
        return None
    a = rho.T - np.eye(n)
    if np.linalg.matrix_rank(a, tol=1e-9) != n - 1:
        return None
    nu, *_ = np.linalg.lstsq(
        np.vstack([a, np.ones(n)]), np.append(np.zeros(n), 1.0), rcond=None
    )
    if not np.all(np.isfinite(nu)) or nu.min() < -1e-8:
        return None
    nu = np.clip(nu, 0.0, None)
    total = float(nu.sum())
    return nu / total if total > 0.0 else None


def _chain(counts: np.ndarray, at_risk: np.ndarray, ends: np.ndarray) -> dict:
    """The chain implied by summed tallies.

    ``counts[i, j]`` is how often j followed i, ``at_risk[i]`` the time spent in
    i over the bouts whose own start was seen, ``ends[i]`` how many of those
    were seen to finish. Any entry whose counts are empty comes back NaN, and
    the steady state comes back None.
    """
    row = counts.sum(axis=1)
    rho = np.where(row[:, None] > 0, counts / np.where(row > 0, row, 1.0)[:, None],
                   np.nan)
    tau = np.where(ends > 0, at_risk / np.where(ends > 0, ends, 1.0), np.nan)
    q = rho / tau[:, None]
    nu = _stationary(rho)
    pi = None
    if nu is not None and np.all(np.isfinite(tau)):
        weight = nu * tau
        total = float(weight.sum())
        # Time-weighted: a state entered rarely but held a long time still
        # takes up most of the record.
        pi = weight / total if total > 0.0 else None
    return {"rho": rho, "tau": tau, "q": q, "pi": pi}


def _tally(bounds, state, index, usable, is_gap, dropped, f_start, f_end,
           duration, censor_left, censor_right) -> tuple[np.ndarray, ...]:
    """One transition matrix and one time-at-risk vector per track, the unit a
    bootstrap resamples. Bouts whose own start was never seen are left out of
    the time, as they are out of the Kaplan-Meier estimate."""
    k, t = len(index), len(bounds)
    counts = np.zeros((t, k, k), dtype=float)
    at_risk = np.zeros((t, k), dtype=float)
    ends = np.zeros((t, k), dtype=float)
    for n, (lo, hi) in enumerate(bounds):
        for i in range(lo, hi):
            if not usable[i] or censor_left[i] or not np.isfinite(duration[i]):
                continue
            a = index[state[i]]
            at_risk[n, a] += float(duration[i])
            ends[n, a] += 0.0 if censor_right[i] else 1.0
        for i in range(lo, hi - 1):
            j = i + 1
            if (f_end[i] == f_start[j] and not (is_gap[i] or is_gap[j])
                    and not (dropped[i] or dropped[j])):
                counts[n, index[state[i]], index[state[j]]] += 1.0
    return counts, at_risk, ends


def _spread(draws: list, shape: tuple) -> np.ndarray | None:
    """Bootstrap standard error. A cell fewer than two resamples could estimate
    comes back NaN rather than zero."""
    if len(draws) < 2:
        return None
    stack = np.stack(draws)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        se = np.nanstd(stack, axis=0, ddof=1)
    se = np.where(np.isfinite(stack).sum(axis=0) >= 2, se, np.nan)
    return se if se.shape == shape else None


def _bootstrap(tally, n_boot: int, seed: int) -> tuple[dict, int]:
    """Standard errors for the chain, resampling whole tracks with replacement.

    The track and not the bout, the bouts of one track being dependent.
    """
    counts, at_risk, ends = tally
    n_tracks = counts.shape[0]
    empty = {"rho": None, "tau": None, "q": None, "pi": None}
    if n_tracks < 2 or n_boot < 2:
        return empty, 0
    rng = np.random.default_rng(seed)
    draws: dict = {"rho": [], "tau": [], "q": [], "pi": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n_tracks, n_tracks)
        fit = _chain(counts[idx].sum(0), at_risk[idx].sum(0), ends[idx].sum(0))
        for key, value in fit.items():
            if value is not None:
                draws[key].append(value)
    k = counts.shape[1]
    return {
        "rho": _spread(draws["rho"], (k, k)),
        "tau": _spread(draws["tau"], (k,)),
        "q": _spread(draws["q"], (k, k)),
        "pi": _spread(draws["pi"], (k,)),
    }, len(draws["pi"])


def _cells(values, se, classes, key: str) -> dict:
    """A matrix as {from: {to: {key: value, se: value}}}, empty cells dropped."""
    out: dict = {}
    for i, a in enumerate(classes):
        row = {}
        for j, b in enumerate(classes):
            v = values[i, j]
            if not np.isfinite(v) or v == 0.0:
                continue
            row[b] = {key: _f(v), "se": None if se is None else _f(se[i, j])}
        if row:
            out[a] = row
    return out


def _markov(tally, classes, unit: str, dwell: dict, occupancy: dict,
            *, n_boot: int, seed: int) -> dict:
    """The continuous-time Markov chain these bouts imply: a lifetime per state,
    the share of the record each holds, and the rates between them. The measured
    survival curve is carried beside the fit."""
    fit = _chain(*(t.sum(0) for t in tally))
    se, n_stationary = _bootstrap(tally, n_boot, seed)
    tau, pi = fit["tau"], fit["pi"]
    at_risk, ends = tally[1].sum(0), tally[2].sum(0)
    km_key = "km_seconds" if unit == "seconds" else "km_frames"

    lifetime, survival = {}, {}
    for i, name in enumerate(classes):
        km = dwell.get(name, {}).get(km_key) or {}
        mean = _f(tau[i]) if np.isfinite(tau[i]) else None
        t = np.asarray(km.get("curve_t") or [], dtype=float)
        p = np.asarray(km.get("curve_p") or [], dtype=float)
        fitted = np.exp(-t / mean) if mean and t.size else np.array([])
        lifetime[name] = {
            "mean": mean,
            "se": None if se["tau"] is None else _f(se["tau"][i]),
            "rate": _f(1.0 / mean) if mean else None,
            "n_ends": int(ends[i]),
            "at_risk": _f(at_risk[i]),
            "km_restricted_mean": _f(km.get("restricted_mean")),
            # How far the fitted line sits from the measured curve, at worst.
            "fit_max_deviation": (
                _f(np.abs(p - fitted).max()) if fitted.size else None
            ),
        }
        survival[name] = {
            "t": [_f(x) for x in t],
            "p": [_f(x) for x in p],
            "fit": [_f(x) for x in fitted],
        }

    steady = None
    if pi is not None:
        steady = {
            name: {"p": _f(pi[i]),
                   "se": None if se["pi"] is None else _f(se["pi"][i])}
            for i, name in enumerate(classes)
        }

    return {
        "unit": unit,
        "classes": list(classes),
        "lifetime": lifetime,
        "survival": survival,
        "jump": _cells(fit["rho"], se["rho"], classes, "p"),
        "rates": _cells(fit["q"], se["q"], classes, "q"),
        "exit_rate": {
            name: _f(1.0 / tau[i]) for i, name in enumerate(classes)
            if np.isfinite(tau[i]) and tau[i] > 0.0
        },
        "steady_state": steady,
        "steady_state_refusal": None if steady is not None else (
            "no unique steady state: a class here has no observed successor, or "
            "the classes split into groups that never reach each other, and "
            "each group then has a steady state of its own"
        ),
        "occupancy_observed": occupancy,
        "bootstrap": {
            "resamples": int(n_boot), "unit": "track", "seed": int(seed),
            "n_tracks": int(tally[0].shape[0]),
            "n_with_steady_state": int(n_stationary),
        },
        "note": (
            "the chain of Boggon et al. 2026 Fig. 1h-i "
            "(doi:10.1038/s41467-026-75076-8). lifetime.mean is the maximum "
            "likelihood mean of an exponential fitted to the bout durations, a "
            "censored bout contributing its time but not an end; survival "
            "carries the measured Kaplan-Meier curve P(T > t) beside that fit, "
            "and fit_max_deviation is the widest gap between them. jump is "
            "P(next = j | in i); rates is q(i->j) = jump / lifetime; "
            "steady_state is the share of TIME each state holds in the long "
            "run, p(i) proportional to nu(i) * lifetime(i) with nu the jump "
            "chain's own stationary law. occupancy_observed is that same share "
            "measured straight off the record, and the two disagreeing is the "
            "sign the chain does not describe these bouts. Every se is the "
            "spread over bootstrap resamples of whole TRACKS"
        ),
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
    model: dict | None, n_bootstrap: int, bootstrap_seed: int,
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
    members: dict[str, np.ndarray] = {}
    for name in classes:
        member = usable & np.array([s == name for s in state], dtype=bool)
        members[name] = member
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

    # Seconds throughout, unless a movie here has no frame rate at all; then
    # frames, which are only comparable across movies acquired at one rate.
    unit = ("seconds" if usable.any() and bool(np.isfinite(seconds[usable]).all())
            else "frames")
    duration = seconds if unit == "seconds" else frames
    held = float(np.nansum(duration[usable]))
    occupancy = {
        name: (_f(float(np.nansum(duration[members[name]])) / held)
               if held > 0.0 else None)
        for name in classes
    }
    markov = _markov(
        _tally(bounds, state, {c: i for i, c in enumerate(classes)}, usable,
               is_gap, dropped, f_start, f_end, duration, censor_left,
               censor_right),
        classes, unit, dwell, occupancy,
        n_boot=int(n_bootstrap), seed=int(bootstrap_seed),
    )

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
        "markov": dict(markov, caveat=(
            f"predict.min_bout_frames removes every bout shorter than "
            f"{floor_frames} frame(s), which lifts every lifetime, so these "
            "rates are LOWER BOUNDS on the true ones and the steady state is "
            "pulled towards whichever state the floor lengthens most"
            if floor_frames > 1 else
            "no minimum bout length is imposed, so no lifetime here is "
            "inflated by a decode setting"
        )),
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
    n_bootstrap: int = 200,
    bootstrap_seed: int = 0,
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
        "schema_version": 2,
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
                _confusion_note(model_report, group), n_bootstrap, bootstrap_seed,
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


def _pm(value, se, spec: str = ".3g") -> str:
    """A number with its bootstrap error, or on its own where there is none."""
    if value is None:
        return "-"
    text = format(float(value), spec)
    return text if se is None else f"{text} +- {format(float(se), '.2g')}"


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

        lines.extend(_markov_lines(g))
    return "\n".join(lines)


def _markov_lines(g: dict) -> list[str]:
    """The fitted chain: lifetimes, steady state, and the rates between states."""
    m, unit = g["markov"], g["markov"]["unit"]
    boot = m["bootstrap"]
    short = "s" if unit == "seconds" else "f"
    lines = ["", f"  the Markov chain fitted to those bouts   lifetimes in {unit}"]

    lines.append(f"    {'class':<14} {'lifetime':>22} {'ends':>6} "
                 f"{'Kaplan-Meier':>14} {'worst gap':>10}")
    for name in m["classes"]:
        row = m["lifetime"][name]
        lines.append(
            f"    {name:<14} {_pm(row['mean'], row['se']) + ' ' + short:>22} "
            f"{row['n_ends']:>6} "
            f"{_n(row['km_restricted_mean']) + ' ' + short:>14} "
            f"{_n(row['fit_max_deviation'], '.2f'):>10}"
        )
    lines.append(
        "    lifetime is the mean of an exponential fitted to the durations, "
        "censored bouts contributing time but no end. Kaplan-Meier is the same "
        "mean measured without assuming a shape, and worst gap is the widest "
        "distance between the fitted curve and the measured one: above ~0.1 the "
        "state does not leave at a constant rate and one number cannot stand "
        "for its lifetime."
    )

    lines.append("")
    lines.append("  how much of the record each state holds")
    if m["steady_state"] is None:
        lines.append(f"    no steady state: {m['steady_state_refusal']}")
    else:
        lines.append(f"    {'class':<14} {'steady state':>18} {'measured':>10}")
        for name in m["classes"]:
            row = m["steady_state"][name]
            lines.append(
                f"    {name:<14} {_pm(row['p'], row['se'], '.4g'):>18} "
                f"{_n(m['occupancy_observed'].get(name), '.4g'):>10}"
            )
        lines.append(
            "    steady state is what the fitted chain predicts in the long "
            "run; measured is the share of time these bouts actually spend "
            "there. The two disagreeing means the chain does not describe them."
        )

    lines.append("")
    lines.append(f"  what follows what, and how fast   rates per {unit[:-1]}")
    for name in m["classes"]:
        jump = m["jump"].get(name)
        if not jump:
            continue
        rates = m["rates"].get(name, {})
        body = "   ".join(
            f"{to} {_pm(cell['p'], cell['se'], '.2f')} "
            f"(q {_n((rates.get(to) or {}).get('q'))})"
            for to, cell in sorted(jump.items(), key=lambda kv: -(kv[1]["p"] or 0.0))
        )
        lines.append(
            f"    {name:<14} exits {_n(m['exit_rate'].get(name))}   {body}"
        )
    lines.append(
        f"    every +- is the spread over {boot['resamples']} bootstrap "
        f"resamples of the {boot['n_tracks']} track(s), the track being the "
        "unit because the bouts within one are not independent"
    )
    lines.append(f"    {m['caveat']}")
    return lines


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #

def _pyplot():
    """matplotlib, imported here because it is a dependency of the drawing only.

    Agg is forced only while pyplot is not yet imported, so a headless CLI run
    gets a writable backend and a notebook or GUI that already chose an
    interactive one keeps it.
    """
    try:
        import matplotlib
        if "matplotlib.pyplot" not in sys.modules:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "drawing needs matplotlib, which this project does not depend on. "
            "Install it, or read the numbers in " + TRANSITIONS_FILE + "."
        ) from exc
    return plt


def plot_survival(report: dict, group: str, out_path, *,
                  drop: tuple[str, ...] = ()) -> "object":
    """Draw P(T > t) per class with its exponential fit. Returns the figure.

    The measured Kaplan-Meier curve as points, the fitted constant-hazard
    survival as a line over it. On the log axis an exponential is a straight
    line, so a class whose points bend away from its line does not leave at a
    constant rate and has no single lifetime.
    """
    plt = _pyplot()
    m = report["groups"][group]["markov"]
    classes = [c for c in m["classes"]
               if c not in drop and m["survival"].get(c, {}).get("t")]
    if not classes:
        raise ValueError("no class here has a survival curve to draw")

    unit = "s" if m["unit"] == "seconds" else "frames"
    colours = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for i, name in enumerate(classes):
        curve, life = m["survival"][name], m["lifetime"][name]
        colour = colours(i % 10)
        ax.plot(curve["t"], curve["p"], "o", ms=4.2, color=colour, alpha=0.85,
                label="{}   {} {} {}".format(
                    name, _pm(life["mean"], life["se"]), unit,
                    "" if life["fit_max_deviation"] is None
                    else "(gap {:.2f})".format(life["fit_max_deviation"])))
        if any(v is not None for v in curve["fit"]):
            ax.plot(curve["t"], curve["fit"], "-", lw=1.4, color="black",
                    alpha=0.8, zorder=3)

    ax.set_yscale("log")
    ax.set_xlabel("t ({})".format(unit))
    ax.set_ylabel("P(T > t)")
    ax.set_title("{}: how long each state lasts, over {} decoded bouts".format(
        group, report["groups"][group]["n_bouts_used"]), fontsize=12, pad=12)
    ax.legend(title="mean lifetime", fontsize=8.4, title_fontsize=8.6,
              frameon=False, loc="upper right")
    ax.grid(True, which="both", lw=0.4, alpha=0.25)
    fig.text(0.5, -0.06,
             "Points are Kaplan-Meier, so a bout running to a track end counts "
             "as a lower bound rather than an end. The black line is the "
             "exponential fitted to the same bouts; where the points leave it, "
             "one lifetime does not describe the state.",
             ha="center", fontsize=7.8, color="#666", wrap=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor="white")
    return fig


def plot(report: dict, group: str, out_path, *, min_prob: float = 0.0,
         drop: tuple[str, ...] = ()) -> "object":
    """Draw one group's fitted chain as a state diagram. Returns the figure.

    Node area is the steady-state share of time, edge width is
    P(next | this bout ended), and each edge carries the rate behind it.
    """
    plt = _pyplot()
    from matplotlib.patches import Circle, FancyArrowPatch
    import numpy as np

    g = report["groups"][group]
    classes = [c for c in g["classes"] if c not in drop]
    if not classes:
        raise ValueError("no classes left to draw after dropping " + repr(list(drop)))
    probs, counts = g["transitions"]["probabilities"], g["transitions"]["counts"]
    per, m = g["dwell"]["per_class"], g["markov"]
    steady, unit = m["steady_state"], ("s" if m["unit"] == "seconds" else "f")

    def n_bouts(c):
        d = per.get(c, {})
        return int(d.get("n_complete", 0)) + int(d.get("n_right_censored", 0))

    def share(c):
        """The steady state where the chain has one, else the measured share."""
        if steady is not None and steady.get(c, {}).get("p") is not None:
            return float(steady[c]["p"])
        return float(m["occupancy_observed"].get(c) or 0.0)

    weight = {c: share(c) for c in classes}
    total = sum(weight.values()) or 1.0

    theta = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, len(classes) + 1)[:-1]
    pos = {c: (float(np.cos(t)), float(np.sin(t))) for c, t in zip(classes, theta)}

    fig, ax = plt.subplots(figsize=(9.6, 8.4))
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-1.8, 1.8)
    ax.set_aspect("equal")
    ax.axis("off")

    edges = [(a, b, float(probs.get(a, {}).get(b, 0.0)),
              int(counts.get(a, {}).get(b, 0)))
             for a in classes for b in classes
             if a != b and float(probs.get(a, {}).get(b, 0.0)) > min_prob]
    for a, b, pr, n in sorted(edges, key=lambda e: e[2]):
        thin = n < 10
        ax.add_patch(FancyArrowPatch(
            pos[a], pos[b], connectionstyle="arc3,rad=0.17", arrowstyle="-|>",
            mutation_scale=13 + 20 * pr, linewidth=0.6 + 6.0 * pr,
            shrinkA=34, shrinkB=38, alpha=0.30 + 0.60 * pr, zorder=1,
            color="#C9A227" if thin else "#B0413E",
            linestyle="--" if thin else "-"))
        mx, my = (pos[a][0] + pos[b][0]) / 2, (pos[a][1] + pos[b][1]) / 2
        dx, dy = pos[b][0] - pos[a][0], pos[b][1] - pos[a][1]
        q = (m["rates"].get(a, {}).get(b) or {}).get("q")
        ax.text(mx - dy * 0.115, my + dx * 0.115,
                "{:.2f}\n({})".format(pr, n) if q is None else
                "{:.2f}  ({})\n{:.3g} /{}".format(pr, n, q, unit),
                ha="center", va="center",
                fontsize=8.4, color="#7a6416" if thin else "#7a2c2a",
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none",
                          alpha=0.86), zorder=3)

    radius = {c: 0.135 + 0.135 * (weight[c] / total) ** 0.5 for c in classes}
    for c in classes:
        r = radius[c]
        ax.add_patch(Circle(pos[c], r, facecolor="#4C6EA8", edgecolor="white",
                            linewidth=2.0, zorder=4, alpha=0.93))
        ax.text(pos[c][0], pos[c][1], c.replace("-to-", "-to-\n"), ha="center",
                va="center", fontsize=9.0, color="white", zorder=5,
                fontweight="bold")
        life = m["lifetime"].get(c, {})
        label = "{} bouts".format(n_bouts(c))
        if life.get("mean") is not None:
            label += "\nlasts {} {}".format(_pm(life["mean"], life["se"]), unit)
        if steady is not None and steady.get(c, {}).get("p") is not None:
            label += "\nholds {} of the record".format(
                _pm(steady[c]["p"], steady[c]["se"], ".3g"))
        # Radially outside the circle, so a big node never sits on its caption.
        out = 1.0 + radius[c] + 0.12
        ax.text(pos[c][0] * out, pos[c][1] * out, label, ha="center",
                va="bottom" if pos[c][1] > 0.0 else "top",
                fontsize=8.2, color="#444", zorder=5)

    src, mdl = report.get("source", {}), (g.get("model") or {})
    ax.set_title("{}: the chain fitted to {} decoded bouts".format(
        group, g["n_bouts_used"]), fontsize=12.5, pad=16)
    sub = "{} bouts from a {} decode  |  {} pairs counted, {} skipped across a gap".format(
        src.get("n_rows", "?"), src.get("decode", "?"),
        g["transitions"]["n_counted"], g["transitions"]["skipped"]["gap"])
    if mdl.get("kappa") is not None:
        sub += "  |  model kappa {:+.3f}, accuracy {:.3f}".format(
            mdl["kappa"], mdl.get("accuracy", float("nan")))
    ax.text(0.5, 0.962, sub, transform=fig.transFigure, ha="center",
            fontsize=8.6, color="#555")
    ax.text(0.5, 0.021,
            "Edge label is P(next | this bout ended) with the count behind it, "
            "over the rate q = that probability / lifetime. Dashed amber edges "
            "rest on fewer than 10 observations.\nNode area is the share of "
            "time the state holds in the long run"
            + ("" if steady is not None else
               ", measured off the record because the chain has no steady state")
            + ". These are DECODED PREDICTIONS: classes the model confuses make "
            "transitions the animal never made.",
            transform=fig.transFigure, ha="center", fontsize=8.0, color="#666")
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor="white")
    return fig

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
    p.add_argument("--plot", default=None, metavar="PATH",
                   help="also draw the state diagram to this image file")
    p.add_argument("--plot-survival", default=None, metavar="PATH",
                   help="also draw P(T > t) and its exponential fit to this "
                        "image file")
    p.add_argument("--plot-group", default=None,
                   help="which category group to draw (default: the first)")
    p.add_argument("--plot-drop", default="", metavar="A,B",
                   help="classes to leave out of the drawing")
    p.add_argument("--plot-min-prob", type=float, default=0.0,
                   help="hide transitions below this probability")
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
    if args.plot or args.plot_survival:
        groups = report["groups"]
        group = args.plot_group or next(iter(groups), "")
        if group not in groups:
            print(
                f"error: nothing to draw for {group!r}; this report has "
                + (", ".join(groups) if groups else "no category groups at all"),
                file=sys.stderr,
            )
            return 2
        drop = tuple(x.strip() for x in args.plot_drop.split(",") if x.strip())
        if args.plot:
            plot(report, group, args.plot, min_prob=args.plot_min_prob, drop=drop)
            print(f"diagram written to {args.plot}")
        if args.plot_survival:
            plot_survival(report, group, args.plot_survival, drop=drop)
            print(f"survival curves written to {args.plot_survival}")

    if not args.no_write:
        print(f"\nWrote {write_report(cfg, report)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
