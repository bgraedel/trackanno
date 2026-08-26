"""Per-frame class posteriors from overlapping windows, decoded into bouts.

Dense prediction scores WINDOWS on a tiling grid, so a transition can only be
reported at a multiple of the window length: the boundary error is a property of
the grid, not of the model. Scoring the same track on an OVERLAPPING grid and
averaging each window's class probabilities onto the frames it covers removes
that quantisation and moves the reported boundary to within about one stride of
the real one. The rest of the module turns that per-frame posterior into runs.

Averaging localises, it does not sharpen. Overlapping windows are not
independent evidence: neighbouring rows' ``beat_*`` and ``wave_*`` features come
out of measurement windows sharing most of their frames, and near a track's ends
adjacent rows collapse onto an identical measurement window. A window straddling
a transition is still blurred by it, and a narrow bout is still a single
measurement.

Two decoders, deliberately both. :func:`decode_min_bout` imposes a hard floor on
bout length: a constraint rather than a model, but one whose floor is a number a
biologist can argue about. :func:`decode_viterbi` is the probabilistic version,
a flat-prior HMM whose self-transition encodes the same expected bout length
(:func:`self_transition_for`), trading a short bout for a long one on the
evidence rather than on the length alone.

NaN means "undefined" throughout and is never filled with a value. A frame no
window could score is a different fact from a frame scored as uniform, and a
label of ``""`` is a different fact from a class name; both survive to the bout
table rather than being tidied away.
"""

from __future__ import annotations

import heapq

import numpy as np

__all__ = [
    "frame_posterior", "decode_min_bout", "decode_viterbi", "self_transition_for",
    "runs_to_bouts",
]

# Emission floor for the Viterbi decode. A class given probability exactly 0.0
# would contribute log(0) = -inf, and two paths both at -inf cannot be compared.
# The floor forbids the class in practice - log(tiny) is about -708, against a
# per-step transition cost of order 4 - while keeping every number finite.
_EMISSION_FLOOR = np.finfo(float).tiny


# --------------------------------------------------------------------------- #
# the posterior
# --------------------------------------------------------------------------- #
def frame_posterior(
    windows, proba, f_lo: int, f_hi: int
) -> tuple[np.ndarray, np.ndarray]:
    """Mean class posterior per FRAME over every window covering that frame.

    ``windows`` are ``(f_start, f_end)`` half-open pairs in ABSOLUTE frame
    numbers; they may overlap, repeat, be unsorted, and hang off either end of
    the span, where the part outside is clipped rather than refused. ``proba`` is
    ``(len(windows), n_classes)`` in the model's class order, row ``i`` being
    window ``i``'s posterior. ``f_lo, f_hi`` is the half-open span to produce.

    Returns ``(posterior, n_covering)`` of shapes ``(f_hi - f_lo, n_classes)``
    and ``(f_hi - f_lo,)``. ``posterior[f]`` is the plain mean of ``proba[i]``
    over every window ``i`` covering frame ``f``, and is all-NaN wherever
    ``n_covering[f] == 0``: an uncovered frame is the absence of a measurement,
    where 0.0 would say every class is equally unlikely.

    Weighting is UNIFORM over the covering windows, so a window centred on the
    frame and one that only just laps it count the same. The mean is not more
    precise than its windows, which share most of their frames.

    ``proba`` must be finite. The implementation scatter-adds onto the window
    EDGES and takes a cumulative sum, so a NaN would be carried forward by the
    running sum and poison every later frame rather than staying where it was.
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
            f"{spans.shape[0]} window(s) but {proba.shape[0]} probability row(s); "
            "they are matched by POSITION, so a mismatch means the rows belong to "
            "different windows"
        )
    if proba.size and not np.isfinite(proba).all():
        bad = int(np.flatnonzero(~np.isfinite(proba).all(axis=1))[0])
        raise ValueError(
            f"proba row {bad} is not finite; a window with no prediction must be "
            "left out of `windows` rather than passed as NaN, which the running "
            "sum here would spread over every later frame"
        )
    f_lo, f_hi = int(f_lo), int(f_hi)
    if f_hi <= f_lo:
        raise ValueError(f"empty frame span [{f_lo}, {f_hi})")

    n_frames, n_classes = f_hi - f_lo, proba.shape[1]
    lo = np.clip(spans[:, 0], f_lo, f_hi) - f_lo
    hi = np.clip(spans[:, 1], f_lo, f_hi) - f_lo
    # Also drops reversed and empty windows, which contribute no frame and so
    # cannot be an error here.
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
    """Per-frame argmax, ``-1`` wherever the row is not fully defined.

    A row with even one non-finite entry is undefined rather than "the best of
    the rest": the missing class might have been the winner. Exact ties go to the
    lowest class index, arbitrary but fixed, so a decode does not depend on the
    summation order that produced the posterior.
    """
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


def _merge(runs: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """Join neighbouring runs that now carry the same label."""
    out = [runs[0]]
    for span in runs[1:]:
        start, end, label = span
        prev_start, _, prev_label = out[-1]
        if label == prev_label:
            out[-1] = (prev_start, end, label)
        else:
            out.append(span)
    return out


# --------------------------------------------------------------------------- #
# decoding
# --------------------------------------------------------------------------- #
def decode_min_bout(posterior: np.ndarray, min_bout_frames: int) -> np.ndarray:
    """Argmax, then absorb every run shorter than ``min_bout_frames``.

    Returns ``(n_frames,)`` class indices, ``-1`` where the posterior row is
    undefined. ``min_bout_frames <= 1`` is a no-op and returns the plain argmax.

    A short run is absorbed into whichever NEIGHBOURING run's class has the higher
    mean posterior across the absorbed run's own frames, so the evidence that
    decides is the evidence over the frames being relabelled. Shortest run first,
    ties by earliest; after each absorption the runs are re-joined and the sweep
    repeats, terminating because every absorption merges at least two runs. On an
    equal mean posterior the LONGER neighbour wins, and then the EARLIER one;
    both need exact floating-point equality.

    Undefined frames are inert. A ``-1`` run is never absorbed, never absorbs,
    and never merges the runs either side of it, since two bouts of the same
    class separated by frames nobody scored are two bouts. A short run with one
    defined neighbour goes to that side; one with no defined neighbour at all is
    left as it is.

    This is a CONSTRAINT, not a model: the result is the argmax made to obey a
    minimum bout length, not the most probable label sequence under any
    distribution. :func:`decode_viterbi` is the version that is.
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
    # A doubly-linked list over the runs plus a shortest-first heap. Rescanning
    # every run to find the next target, and rebuilding the list to merge it,
    # makes this quadratic in the run count, and the run count is what an
    # overlapping stride and several window scales both increase.
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
        # Lazily deleted: an entry whose run has since been merged or grown is
        # stale, and cheaper to skip here than to find and remove on merging.
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



def _neighbours(runs: list[tuple[int, int, int]], i: int) -> list[tuple[int, int, int]]:
    """The defined runs either side of ``runs[i]``, left first."""
    out = []
    if i > 0 and runs[i - 1][2] >= 0:
        out.append(runs[i - 1])
    if i + 1 < len(runs) and runs[i + 1][2] >= 0:
        out.append(runs[i + 1])
    return out



def decode_viterbi(posterior: np.ndarray, self_transition: float) -> np.ndarray:
    """Most probable label sequence under a flat-prior, uniform-switch HMM.

    Emissions are ``log(posterior)``; transitions are ``self_transition`` on the
    diagonal and ``(1 - self_transition) / (n_classes - 1)`` off it, so one number
    sets how sticky the decode is - see :func:`self_transition_for` for reading it
    as an expected bout length. The initial state is uniform. Returns
    ``(n_frames,)`` class indices, ``-1`` where the posterior row is undefined.

    An HMM wants an emission LIKELIHOOD ``p(features | class)`` and a classifier
    gives a POSTERIOR ``p(class | features)``; the two differ by the class prior,
    so using the posterior directly is exact only if that prior is flat.
    ``auto_class_weights="Balanced"`` in :mod:`trackanno.learn.suggest` pushes the
    fitted model towards a balanced prior without guaranteeing one, and the
    mismatch is a constant per-class bias in the emissions, showing up as
    slightly displaced boundaries. A division by the empirical class frequencies
    is the fix if it matters.

    Undefined frames BREAK THE CHAIN. Each contiguous run of defined frames is
    decoded on its own and the gaps come back as ``-1``, since carrying a path
    across frames nobody scored would let a state persist on no evidence.

    ``n_classes == 1`` returns the trivial answer rather than dividing by
    ``n_classes - 1``. ``self_transition`` outside the open interval ``(0, 1)``
    is refused: 0 forbids staying and 1 forbids leaving, and both make the log
    transitions infinite.
    """
    posterior = np.asarray(posterior, dtype=float)
    if posterior.ndim != 2:
        raise ValueError(f"posterior must be (n_frames, n_classes), got {posterior.shape}")
    labels = _argmax_labels(posterior)
    n_classes = posterior.shape[1]
    if n_classes <= 1 or labels.size == 0:
        return labels

    stay = float(self_transition)
    if not 0.0 < stay < 1.0:
        raise ValueError(
            f"self_transition must be in (0, 1), got {stay}; 0 forbids a bout from "
            "continuing and 1 forbids it from ever ending"
        )
    transition = np.full((n_classes, n_classes), np.log((1.0 - stay) / (n_classes - 1)))
    np.fill_diagonal(transition, np.log(stay))

    out = np.full(labels.size, -1, dtype=np.int64)
    for start, end, label in _runs(labels >= 0):
        if not label:
            continue
        emission = np.log(np.maximum(posterior[start:end], _EMISSION_FLOOR))
        out[start:end] = _viterbi(emission, transition)
    return out


def _viterbi(emission: np.ndarray, transition: np.ndarray) -> np.ndarray:
    """Textbook Viterbi in log space over one gap-free segment.

    Emissions need not be normalised: scaling a whole FRAME's row multiplies
    every path through it by the same constant, leaving the argmax unchanged.
    Per-CLASS scaling is the prior question the docstring above is about.
    """
    n_frames, n_classes = emission.shape
    score = emission[0].copy()
    back = np.zeros((n_frames, n_classes), dtype=np.int64)
    for t in range(1, n_frames):
        candidates = score[:, None] + transition          # (from, to)
        best = np.argmax(candidates, axis=0)
        back[t] = best
        score = candidates[best, np.arange(n_classes)] + emission[t]

    path = np.empty(n_frames, dtype=np.int64)
    path[-1] = int(np.argmax(score))
    for t in range(n_frames - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def self_transition_for(min_bout_frames: int) -> float:
    """The HMM self-transition whose expected bout length is ``min_bout_frames``.

    Under a constant self-transition ``a`` the holding time is geometric with
    expectation ``1 / (1 - a)``, so inverting gives ``a = 1 - 1/min_bout_frames``.
    The same number can therefore be handed to :func:`decode_min_bout` as a hard
    floor or to :func:`decode_viterbi` as a prior, one being a bound and the
    other a mean. :meth:`trackanno.config.Predict.resolved_self_transition` is
    what calls this for ``predict.self_transition: auto``.

    ``min_bout_frames < 2`` is refused: at 1 the identity returns ``a = 0``, a
    chain forbidden to stay in any state, and below that a negative probability.
    """
    frames = int(min_bout_frames)
    if frames < 2:
        raise ValueError(
            f"min_bout_frames must be at least 2, got {frames}; 1 - 1/{frames} is "
            "not a usable self-transition"
        )
    return 1.0 - 1.0 / frames


# --------------------------------------------------------------------------- #
# bouts
# --------------------------------------------------------------------------- #
def runs_to_bouts(
    labels: np.ndarray, posterior: np.ndarray, classes, f_lo: int
) -> list[dict]:
    """Run-length encode decoded labels into bout records.

    ``labels`` and ``posterior`` are frame-aligned and start at absolute frame
    ``f_lo``; ``classes`` names the posterior's columns, in the model's own class
    order (``model.classes_``). One dict per run:

    ``f_start``, ``f_end``
        Absolute, half-open.
    ``n_frames``
        ``f_end - f_start``.
    ``label``
        The class NAME, or ``""`` for a run of undefined frames.
    ``class_index``
        The column of ``posterior`` the name came from, ``-1`` for a gap. Carried
        so a caller can index the posterior back without re-looking-up the name.
    ``conf``
        Mean posterior of the chosen class over the run's frames, NaN for a gap.
        A mean rather than the peak, which would report the one frame that
        agreed.

    The bouts are sorted, disjoint and GAPLESS across
    ``[f_lo, f_lo + len(labels))``: an unscored stretch is a row with an empty
    label, never a hole. That mirrors the ``skip_reason`` convention in
    :mod:`trackanno.infer.densepredict`, and it is a contract downstream code may rely
    on.

    ``classes`` must have one name per posterior column, a shorter list being
    what produces a plausible bout table naming the wrong behaviour.
    """
    labels = np.asarray(labels, dtype=np.int64)
    posterior = np.asarray(posterior, dtype=float)
    names = [str(c) for c in classes]
    if posterior.ndim != 2:
        raise ValueError(f"posterior must be (n_frames, n_classes), got {posterior.shape}")
    if labels.size != posterior.shape[0]:
        raise ValueError(
            f"{labels.size} label(s) against {posterior.shape[0]} posterior row(s); "
            "they must be frame-aligned"
        )
    if len(names) != posterior.shape[1]:
        raise ValueError(
            f"{len(names)} class name(s) for {posterior.shape[1]} posterior "
            "column(s); the names and the columns come from the same model and a "
            "mismatch means they do not"
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
            # NaN, not 0.0: nobody scored these frames, which is not the same as
            # scoring them badly.
            "conf": np.nan if label < 0 else float(posterior[start:end, label].mean()),
        }
        out.append(entry)
    return out
