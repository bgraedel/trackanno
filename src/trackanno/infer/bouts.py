"""Per-frame class posteriors from overlapping windows, decoded into bouts of a minimum length.

Unscored frames stay NaN and become bouts with an empty label.
"""

from __future__ import annotations

import heapq

import numpy as np

__all__ = [
    "frame_posterior", "decode_min_bout", "runs_to_bouts",
]

# --------------------------------------------------------------------------- #
# the posterior
# --------------------------------------------------------------------------- #
def frame_posterior(
    windows, proba, f_lo: int, f_hi: int
) -> tuple[np.ndarray, np.ndarray]:
    """``(posterior, n_covering)`` over ``[f_lo, f_hi)``, the mean ``proba`` of the covering windows.

    Windows are absolute half-open pairs, clipped to the span; uncovered frames are NaN.
    """
    proba = np.asarray(proba, dtype=float)
    if proba.ndim != 2:
        raise ValueError(f"proba must be (n_windows, n_classes), got shape {proba.shape}")
    spans = np.asarray(windows, dtype=np.int64)
    if spans.size == 0:
        spans = spans.reshape(0, 2)
    if spans.ndim != 2 or spans.shape[1] != 2:
        raise ValueError(
            f"windows must be (f_start, f_end) pairs, got shape {spans.shape}"
        )
    if spans.shape[0] != proba.shape[0]:
        raise ValueError(
            f"{spans.shape[0]} window(s) but {proba.shape[0]} probability row(s)"
        )
    if proba.size and not np.isfinite(proba).all():
        bad = int(np.flatnonzero(~np.isfinite(proba).all(axis=1))[0])
        raise ValueError(
            f"proba row {bad} is not finite; leave unscored windows out instead"
        )
    f_lo, f_hi = int(f_lo), int(f_hi)
    if f_hi <= f_lo:
        raise ValueError(f"empty frame span [{f_lo}, {f_hi})")

    n_frames, n_classes = f_hi - f_lo, proba.shape[1]
    lo = np.clip(spans[:, 0], f_lo, f_hi) - f_lo
    hi = np.clip(spans[:, 1], f_lo, f_hi) - f_lo
    # Drops reversed and empty windows too.
    keep = hi > lo
    lo, hi, rows = lo[keep], hi[keep], proba[keep]

    edges = np.zeros((n_frames + 1, n_classes))
    counts = np.zeros(n_frames + 1, dtype=np.int64)
    np.add.at(edges, lo, rows)
    np.add.at(edges, hi, -rows)
    np.add.at(counts, lo, 1)
    np.add.at(counts, hi, -1)

    n_covering = np.cumsum(counts[:-1])
    total = np.cumsum(edges[:-1], axis=0)
    covered = n_covering > 0
    posterior = np.full((n_frames, n_classes), np.nan)
    posterior[covered] = total[covered] / n_covering[covered, None]
    return posterior, n_covering


# --------------------------------------------------------------------------- #
# runs
# --------------------------------------------------------------------------- #
def _argmax_labels(posterior: np.ndarray) -> np.ndarray:
    """Per-frame argmax, ``-1`` where any entry of the row is non-finite."""
    defined = np.isfinite(posterior).all(axis=1)
    labels = np.full(posterior.shape[0], -1, dtype=np.int64)
    if defined.any():
        labels[defined] = np.argmax(posterior[defined], axis=1)
    return labels


def _runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """``(start, end, label)`` for each maximal equal-label span, half-open."""
    n = int(labels.size)
    if n == 0:
        return []
    cut = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate([[0], cut])
    ends = np.concatenate([cut, [n]])
    return [(int(s), int(e), int(labels[s])) for s, e in zip(starts, ends)]


# --------------------------------------------------------------------------- #
# decoding
# --------------------------------------------------------------------------- #
def decode_min_bout(posterior: np.ndarray, min_bout_frames: int) -> np.ndarray:
    """Argmax, then absorb runs shorter than ``min_bout_frames`` into a neighbour, shortest first.

    The neighbour class with the higher mean posterior over the run wins; ``-1`` runs are inert.
    """
    posterior = np.asarray(posterior, dtype=float)
    if posterior.ndim != 2:
        raise ValueError(f"posterior must be (n_frames, n_classes), got {posterior.shape}")
    labels = _argmax_labels(posterior)
    floor = int(min_bout_frames)
    if floor <= 1 or labels.size == 0:
        return labels

    spans = _runs(labels)
    n = len(spans)
    start = np.array([s for s, _, _ in spans], dtype=np.int64)
    end = np.array([e for _, e, _ in spans], dtype=np.int64)
    lab = np.array([v for _, _, v in spans], dtype=np.int64)
    # Doubly-linked list over the runs plus a shortest-first heap.
    prev = np.arange(-1, n - 1)
    nxt = np.arange(1, n + 1)
    nxt[n - 1] = -1
    alive = np.ones(n, dtype=bool)

    def neighbours(i: int) -> list[int]:
        out = []
        if prev[i] >= 0 and lab[prev[i]] >= 0:
            out.append(int(prev[i]))
        if nxt[i] >= 0 and lab[nxt[i]] >= 0:
            out.append(int(nxt[i]))
        return out

    def absorbing(i: int) -> int:
        best, best_key = int(lab[i]), None
        for offset, j in enumerate(neighbours(i)):
            key = (-float(posterior[start[i]:end[i], lab[j]].mean()),
                   -(int(end[j]) - int(start[j])), offset)
            if best_key is None or key < best_key:
                best, best_key = int(lab[j]), key
        return best

    heap = [(int(end[i] - start[i]), int(start[i]), i)
            for i in range(n) if lab[i] >= 0 and end[i] - start[i] < floor]
    heapq.heapify(heap)
    while heap:
        length, at, i = heapq.heappop(heap)
        # Skip stale heap entries.
        if not alive[i] or start[i] != at or end[i] - start[i] != length:
            continue
        if lab[i] < 0 or end[i] - start[i] >= floor or not neighbours(i):
            continue
        lab[i] = absorbing(i)
        for j in (int(prev[i]), int(nxt[i])):
            if j >= 0 and alive[j] and lab[j] == lab[i]:
                a, b = (j, i) if start[j] < start[i] else (i, j)
                end[a] = end[b]
                alive[b] = False
                nxt[a] = nxt[b]
                if nxt[b] >= 0:
                    prev[nxt[b]] = a
                i = a
        if end[i] - start[i] < floor and neighbours(i):
            heapq.heappush(heap, (int(end[i] - start[i]), int(start[i]), i))

    out = np.full(labels.size, -1, dtype=np.int64)
    for i in range(n):
        if alive[i]:
            out[start[i]:end[i]] = lab[i]
    return out


# --------------------------------------------------------------------------- #
# bouts
# --------------------------------------------------------------------------- #
def runs_to_bouts(
    labels: np.ndarray, posterior: np.ndarray, classes, f_lo: int
) -> list[dict]:
    """Run-length encode labels starting at frame ``f_lo`` into gapless bout dicts.

    Keys: ``f_start``, ``f_end``, ``n_frames``, ``label`` ("" for a gap), ``class_index``, ``conf``.
    """
    labels = np.asarray(labels, dtype=np.int64)
    posterior = np.asarray(posterior, dtype=float)
    names = [str(c) for c in classes]
    if posterior.ndim != 2:
        raise ValueError(f"posterior must be (n_frames, n_classes), got {posterior.shape}")
    if labels.size != posterior.shape[0]:
        raise ValueError(
            f"{labels.size} label(s) against {posterior.shape[0]} posterior row(s)"
        )
    if len(names) != posterior.shape[1]:
        raise ValueError(
            f"{len(names)} class name(s) for {posterior.shape[1]} posterior column(s)"
        )
    if labels.size and int(labels.max(initial=-1)) >= len(names):
        raise ValueError(
            f"label {int(labels.max())} has no name in {names}"
        )

    out: list[dict] = []
    for start, end, label in _runs(labels):
        entry = {
            "f_start": int(f_lo) + start,
            "f_end": int(f_lo) + end,
            "n_frames": end - start,
            "label": "" if label < 0 else names[label],
            "class_index": int(label),
            "conf": np.nan if label < 0 else float(posterior[start:end, label].mean()),
        }
        out.append(entry)
    return out
