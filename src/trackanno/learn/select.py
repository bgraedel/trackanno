"""Rank clips for annotation by the model's margin, from raw ``predict_proba`` rows."""

from __future__ import annotations

import numpy as np

__all__ = ["RULE", "margin", "rank_by", "select_batch"]

# The rule name recorded in each record's ``rule`` field.
RULE = "margin"

# Allowed deviation of a row sum from 1.
_SUM_TOL = 1e-6


def _as_proba(proba, clip_ids=None) -> np.ndarray:
    """``proba`` as a validated probability matrix; raises naming the bad row."""
    proba = np.asarray(proba, dtype=float)
    if proba.ndim != 2:
        raise ValueError(
            f"proba must be (n_clips, n_classes), got shape {proba.shape}"
        )

    def where(i: int) -> str:
        if clip_ids is None or i >= len(clip_ids):
            return f"row {i}"
        return f"row {i} (clip {clip_ids[i]!r})"

    if proba.size and not np.isfinite(proba).all():
        bad = int(np.flatnonzero(~np.isfinite(proba).all(axis=1))[0])
        raise ValueError(f"proba {where(bad)} is not finite: {proba[bad]}")
    if proba.size and proba.min() < 0.0:
        bad = int(np.unravel_index(int(np.argmin(proba)), proba.shape)[0])
        raise ValueError(
            f"proba {where(bad)} has a negative entry: {proba[bad]}; pass "
            "predict_proba output, not decision scores"
        )
    if proba.size:
        sums = proba.sum(axis=1)
        off = np.flatnonzero(np.abs(sums - 1.0) > _SUM_TOL)
        if off.size:
            bad = int(off[0])
            raise ValueError(
                f"proba {where(bad)} sums to {sums[bad]:.6g}, not 1 "
                f"(tolerance {_SUM_TOL:g})"
            )
    if proba.shape[1] < 2:
        raise ValueError(
            f"proba has {proba.shape[1]} class column(s); margin needs at least two"
        )
    return proba


def margin(proba) -> np.ndarray:
    """``1 - (p1 - p2)``: how contested the top two classes are, in [0, 1]."""
    proba = _as_proba(proba)
    top2 = np.partition(proba, -2, axis=1)[:, -2:]
    return 1.0 - (top2[:, 1] - top2[:, 0])


def rank_by(scores, clip_ids) -> np.ndarray:
    """Row indices, best score first, ties broken by ``clip_id``."""
    scores = np.asarray(scores, dtype=float)
    ids = np.asarray(clip_ids, dtype=str)
    if scores.shape != ids.shape:
        raise ValueError(f"{scores.size} score(s) against {ids.size} clip id(s)")
    # lexsort takes the primary key last: score descending, then id ascending.
    return np.lexsort((ids, -scores))


def select_batch(proba, clip_ids, classes, n: int, *, candidates=None) -> list[dict]:
    """Up to ``n`` candidate clips, most contested first, one record dict each.

    ``candidates`` is an optional boolean mask of selectable rows; ``clip_ids`` must be unique.
    """
    ids = np.asarray(clip_ids, dtype=str)
    names = [str(c) for c in classes]
    proba = _as_proba(proba, ids)
    n = int(n)

    if ids.shape != (proba.shape[0],):
        raise ValueError(
            f"{ids.size} clip id(s) for {proba.shape[0]} probability row(s)"
        )
    if len(names) != proba.shape[1]:
        raise ValueError(
            f"{len(names)} class name(s) for {proba.shape[1]} probability column(s)"
        )
    unique, counts = np.unique(ids, return_counts=True)
    if unique.size != ids.size:
        repeated = [str(u) for u in unique[counts > 1]][:3]
        raise ValueError(f"clip ids must be unique, e.g. {repeated} repeat")
    if n <= 0:
        raise ValueError(f"batch size must be at least 1, got {n}")

    if candidates is None:
        pool = np.arange(proba.shape[0])
    else:
        mask = np.asarray(candidates, dtype=bool)
        if mask.shape != (proba.shape[0],):
            raise ValueError(
                f"candidates must be a boolean mask of {proba.shape[0]} row(s), "
                f"got shape {mask.shape}"
            )
        pool = np.flatnonzero(mask)
    if not pool.size:
        return []

    scores = margin(proba[pool])
    order = rank_by(scores, ids[pool])[:n]
    predicted = proba.argmax(axis=1)
    out: list[dict] = []
    for rank, position in enumerate(order):
        row = int(pool[position])
        out.append({
            "clip_id": str(ids[row]),
            "index": row,
            "rank": rank,
            "rule": RULE,
            "score": float(scores[position]),
            "predicted_class": names[int(predicted[row])],
            "predicted_proba": float(proba[row, int(predicted[row])]),
            "pool_size": int(pool.size),
        })
    return out
