"""Per-frame class posteriors from overlapping windows, decoded into bouts.

Dense prediction scores WINDOWS, and those windows tile - stride equals length in
:func:`trackanno.densepredict.window_grid` - so a behavioural transition can only
ever be reported at a multiple of the window length. A reversal at frame 137 with
50-frame windows is placed at 100 or at 150 and no arithmetic downstream recovers
the 37: the boundary error is a property of the GRID, not of the model. Scoring
the same track on an OVERLAPPING grid and averaging each window's class
probabilities onto the frames it covers moves the reported boundary to within
about one stride of the real one. That is what this module is for; the rest of it
turns that per-frame posterior into runs.

Averaging LOCALISES here, it does not sharpen. Overlapping windows are not
independent evidence. At stride 10 with the default 75-frame spectral window
(:data:`trackanno.wavespectra.DEFAULT_WINDOW`) two neighbouring rows' ``beat_*``
and ``wave_*`` features come out of measurement windows overlapping about 93% of
their frames, and near a track's ends - where a window can barely be centred at
all - adjacent rows collapse onto an IDENTICAL measurement window and are
literally equal. So the posterior is a smoothed readout of one highly correlated
estimator, not the mean of N independent ones, and its variance falls far short
of the 1/N that averaging N independent estimates would buy. A narrow bout is
still a single measurement; read it as one.

Two decoders, deliberately both:

* :func:`decode_min_bout` imposes a HARD floor on bout length. It is a
  constraint, not a model - the answer is not the most probable label sequence
  under anything - but it is the one that says exactly what it does, and the
  floor is the number a biologist can argue about.
* :func:`decode_viterbi` is the probabilistic version: a flat-prior HMM whose
  self-transition encodes the same expected bout length
  (:func:`self_transition_for`). It trades a short bout for a long one on the
  EVIDENCE rather than on the length alone.

NaN means "undefined" throughout, as everywhere else in this codebase, and is
never filled with a value. A frame no window could score is a different fact from
a frame scored as uniform, and a label of "" is a different fact from a class
name; both survive to the bout table rather than being tidied away.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "frame_posterior", "decode_min_bout", "decode_viterbi", "self_transition_for",
    "runs_to_bouts",
]

# Emission floor for the Viterbi decode. A class a window gave probability
# exactly 0.0 would otherwise contribute log(0) = -inf, and once two competing
# paths are both -inf their comparison is meaningless (and -inf + inf, reachable
# through an all-zero row, is NaN). The floor forbids the class in practice -
# log(tiny) is about -708, against a per-step transition cost of order 4 - while
# keeping every number finite.
_EMISSION_FLOOR = np.finfo(float).tiny


# --------------------------------------------------------------------------- #
# the posterior
# --------------------------------------------------------------------------- #
def frame_posterior(
    windows, proba, f_lo: int, f_hi: int
) -> tuple[np.ndarray, np.ndarray]:
    """Mean class posterior per FRAME over every window covering that frame.

    Parameters
    ----------
    windows
        ``(f_start, f_end)`` half-open pairs in ABSOLUTE frame numbers. They may
        overlap, may repeat, need not be sorted, and may hang off either end of
        the requested span - the part outside is clipped rather than refused,
        because a window centred near a track's end legitimately reaches past it.
    proba
        ``(len(windows), n_classes)``; row ``i`` is window ``i``'s class
        posterior, in the model's own class order. Must be finite: see below.
    f_lo, f_hi
        The half-open frame span to produce.

    Returns
    -------
    ``(posterior, n_covering)`` of shapes ``(f_hi - f_lo, n_classes)`` and
    ``(f_hi - f_lo,)``. ``posterior[f]`` is the plain mean of ``proba[i]`` over
    every window ``i`` covering frame ``f``, and is all-NaN wherever
    ``n_covering[f] == 0``.

    Notes
    -----
    Weighting is UNIFORM over the covering windows. A window whose centre sits on
    the frame and one that only just laps it count the same, which is the wrong
    thing to believe about a window whose features are dominated by its middle. A
    distance-from-centre taper (Hann, or plain triangular) is a defensible
    refinement and is deliberately NOT taken yet: it has a free shape parameter
    nothing here can fit, and the uniform mean is the version whose failure mode
    is obvious. Do not add one without a boundary-localisation number showing it
    helps.

    The mean is NOT more precise than its windows. Neighbouring windows share
    most of their frames - see the module docstring - so this reduces variance
    far less than averaging that many independent estimates would.

    An uncovered frame is NaN, never 0.0. Zero would say "every class is equally
    unlikely", which is a measurement; no window covering the frame is the
    absence of one, and only NaN says that.

    Implementation: one scatter-add onto the window EDGES and a cumulative sum,
    so the cost is O(n_windows + n_frames * n_classes) rather than O(sum of
    window lengths) - at stride 10 with 50-frame windows that is five times less
    work, and the gap widens with the overlap. The price is that the running sum
    carries values forward, which is why a non-finite ``proba`` is refused rather
    than propagated: one NaN would silently poison every frame after it.
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

    A row with even one non-finite entry is undefined, not "the best of the rest":
    the missing class might have been the winner, and answering anyway would turn
    a hole into a confident label. Exact ties go to the lowest class index, which
    is arbitrary but fixed - a decode must not depend on the summation order that
    produced the posterior.
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
    mean posterior across the absorbed run's own frames - so the evidence that
    decides is the evidence over the frames being relabelled, not over the
    neighbour. Shortest run first, ties by earliest; after each absorption the
    runs are re-joined and the sweep repeats until nothing is short enough to
    move, which terminates because every absorption merges at least two runs.

    Ties, spelled out because an undocumented tiebreak is a silent choice:

    * equal mean posterior -> the LONGER neighbour, on the grounds that the
      better-established bout is the safer thing to extend;
    * still equal -> the EARLIER (left) neighbour.

    Both need exact floating-point equality, so in practice they fire when the
    two neighbours are the same class - where the choice cannot matter - or on
    hand-built input.

    Undefined frames are inert. A ``-1`` run is never absorbed, never absorbs,
    and never merges the runs either side of it: two bouts of the same class
    separated by frames nobody scored are two bouts, and joining them would
    invent coverage. A short run whose only defined neighbour is on one side goes
    to that side; a short run with no defined neighbour at all - a whole island
    between two gaps - is left as it is, since the alternative is to delete a
    measurement for being lonely.

    This is a CONSTRAINT, not a model: the result is the argmax made to obey a
    minimum bout length, and it is not the most probable label sequence under any
    distribution. :func:`decode_viterbi` is the version that is.
    """
    posterior = np.asarray(posterior, dtype=float)
    if posterior.ndim != 2:
        raise ValueError(f"posterior must be (n_frames, n_classes), got {posterior.shape}")
    labels = _argmax_labels(posterior)
    floor = int(min_bout_frames)
    if floor <= 1 or labels.size == 0:
        return labels

    runs = _runs(labels)
    for _ in range(len(runs)):
        target = _shortest_absorbable(runs, floor)
        if target is None:
            break
        start, end = runs[target][0], runs[target][1]
        runs[target] = (start, end, _absorbing_class(runs, target, posterior))
        runs = _merge(runs)

    out = np.full(labels.size, -1, dtype=np.int64)
    for start, end, label in runs:
        out[start:end] = label
    return out


def _shortest_absorbable(runs: list[tuple[int, int, int]], floor: int) -> int | None:
    """Index of the shortest too-short run that has somewhere to go, or None.

    Shortest first so the least-supported claim is the one that gives way; ties
    to the earliest, so the answer does not depend on list order.
    """
    best, best_length = None, floor
    for i, (start, end, label) in enumerate(runs):
        if label < 0:
            continue
        length = end - start
        if length >= best_length:
            continue
        if not _neighbours(runs, i):
            continue
        best, best_length = i, length
    return best


def _neighbours(runs: list[tuple[int, int, int]], i: int) -> list[tuple[int, int, int]]:
    """The defined runs either side of ``runs[i]``, left first."""
    out = []
    if i > 0 and runs[i - 1][2] >= 0:
        out.append(runs[i - 1])
    if i + 1 < len(runs) and runs[i + 1][2] >= 0:
        out.append(runs[i + 1])
    return out


def _absorbing_class(
    runs: list[tuple[int, int, int]], i: int, posterior: np.ndarray
) -> int:
    """Which neighbour's class ``runs[i]`` becomes. See :func:`decode_min_bout`.

    Falls back to the run's OWN label, so a run with no defined neighbour is a
    no-op rather than a fabricated gap.
    """
    start, end, own = runs[i]
    best, best_key = own, None
    for offset, (n_start, n_end, n_label) in enumerate(_neighbours(runs, i)):
        # Negated so a plain "greater is better" ordering covers all three rules:
        # higher mean posterior, then longer neighbour, then earlier one.
        key = (-float(posterior[start:end, n_label].mean()), -(n_end - n_start), offset)
        if best_key is None or key < best_key:
            best, best_key = n_label, key
    return best


def decode_viterbi(posterior: np.ndarray, self_transition: float) -> np.ndarray:
    """Most probable label sequence under a flat-prior, uniform-switch HMM.

    Emissions are ``log(posterior)``; transitions are ``self_transition`` on the
    diagonal and ``(1 - self_transition) / (n_classes - 1)`` off it, so one number
    sets how sticky the decode is - see :func:`self_transition_for` for reading it
    as an expected bout length. The initial state is uniform. Returns
    ``(n_frames,)`` class indices, ``-1`` where the posterior row is undefined.

    THE APPROXIMATION, stated rather than buried: an HMM wants an emission
    LIKELIHOOD ``p(features | class)``, and what a classifier gives is a
    POSTERIOR ``p(class | features)``. The two differ by the class prior, so using
    the posterior directly is exact only if that prior is flat. Here it is
    approximately so and not exactly: ``auto_class_weights="Balanced"`` in
    :mod:`trackanno.suggest` reweights the training loss by inverse class
    frequency, which pushes the fitted model towards a balanced prior, but the
    match is a property of a particular fit rather than an identity - the weights
    apply to the loss, not to the output, and nothing constrains a boosted tree's
    calibration. The effect of the mismatch is a constant per-class bias in the
    emissions, which shows up as slightly displaced boundaries rather than as
    nonsense. This is a known approximation, not an oversight; a division by the
    empirical class frequencies is the fix if it ever matters.

    Undefined frames BREAK THE CHAIN. Each contiguous run of defined frames is
    decoded on its own and the gaps come back as ``-1``, because carrying a path
    across frames nobody scored would let a state persist on no evidence and
    would make the label after a gap depend on the one before it.

    ``n_classes == 1`` returns the trivial answer (that class, wherever the frame
    is defined) rather than dividing by ``n_classes - 1``. ``self_transition``
    outside the open interval ``(0, 1)`` is refused: 0 forbids staying and 1
    forbids leaving, and both make the log transitions infinite.
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

    Emissions need not be normalised: scaling a whole FRAME's row multiplies every
    path through it by the same constant, so the argmax is unchanged. Per-CLASS
    scaling is exactly the prior question the docstring above is about.
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

    Under a constant self-transition ``a`` the holding time is geometric, so a
    bout lasts ``k`` frames with probability ``a**(k - 1) * (1 - a)`` and its
    expectation is ``1 / (1 - a)``. Inverting that identity gives
    ``a = 1 - 1 / min_bout_frames``, which is what this returns - so the same
    number can be handed to :func:`decode_min_bout` as a hard floor or to
    :func:`decode_viterbi` as a prior, and the two mean the same thing on average
    even though one is a bound and the other a mean.

    ``min_bout_frames < 2`` is refused. At 1 the identity returns ``a = 0``, a
    chain forbidden to stay in any state - an actively wrong model rather than a
    weak one - and below that it returns a negative probability.
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
        A mean, not the peak: a bout is as confident as its weakest frames, and
        the maximum would report the one frame that agreed.

    The bouts are sorted, disjoint and GAPLESS across
    ``[f_lo, f_lo + len(labels))``: an unscored stretch is a row with an empty
    label, never a hole. That mirrors the ``skip_reason`` convention in
    :mod:`trackanno.densepredict` - a window the model declined to score still
    gets a row, because an absent row is indistinguishable from one nobody
    thought about - and it is a contract downstream code may rely on.

    ``classes`` must have one name per posterior column. A shorter list would
    index the wrong name for a real class, which is the failure that produces a
    plausible bout table naming the wrong behaviour.
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
            # scoring them badly. A defined label over an undefined row would
            # propagate NaN here too, and should.
            "conf": np.nan if label < 0 else float(posterior[start:end, label].mean()),
        }
        out.append(entry)
    return out
