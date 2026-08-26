"""Checks for :mod:`trackanno.infer.bouts`, the per-frame posterior and its decoders.

Nothing here runs a model. Every posterior below is built from a KNOWN
ground-truth label sequence by the definition of what a window measures - a
window's class probabilities are the class fractions of the frames it covers -
so the expected answer is available in closed form and a disagreement is an
error in the module rather than a disagreement between two implementations.

Four things are worth testing separately:

* the averaging is the arithmetic it claims to be: the mean over covering
  windows, on a hand-computed overlapping set, invariant to the order the windows
  arrive in, clipped at the ends, and NaN - never 0.0 - where no window reached;
* the BOUNDARY MOVES, which is the entire reason the module exists. With a
  transition planted at frame 137 and 50-frame windows, a tiling grid can only
  answer 100 or 150; an overlapping one must land within about a stride of 137,
  and closer than the tiling grid does;
* the decoders do what their docstrings say - a short spurious run is absorbed
  and a long one is not, flicker the per-frame argmax cannot resist is suppressed,
  and the absorption goes to the class the absorbed frames' own posterior
  prefers;
* undefined frames stay undefined. A gap is never bridged, never absorbed, never
  carried across by a Viterbi path, and comes out of ``runs_to_bouts`` as a row
  with an empty label rather than as a hole in the frame coverage.
"""

from __future__ import annotations

import numpy as np
import pytest
from harness import check
from trackanno.infer.bouts import (
    decode_min_bout, decode_viterbi, frame_posterior, runs_to_bouts,
    self_transition_for,
)

TOL = 1e-12


# --------------------------------------------------------------------------- #
# generators
# --------------------------------------------------------------------------- #
def tiled(n_frames: int, length: int) -> list[tuple[int, int]]:
    """The grid dense prediction uses today: stride equals length, no overlap."""
    return [(s, min(s + length, n_frames)) for s in range(0, n_frames, length)]


def overlapping(n_frames: int, length: int, stride: int) -> list[tuple[int, int]]:
    """Whole windows every ``stride`` frames; the tail that would hang off is dropped."""
    return [(s, s + length) for s in range(0, n_frames - length + 1, stride)]


def window_fractions(truth: np.ndarray, windows, n_classes: int) -> np.ndarray:
    """What a PERFECT model would report for each window: its class fractions.

    A window measures the frames inside it and nothing else, so a model that made
    no mistake at all still reports a mixture wherever the window straddles a
    transition. Building the evidence this way means any boundary error the tests
    below find belongs to the GRID rather than to a model's noise, which is the
    quantity the module claims to improve.
    """
    rows = []
    for f_start, f_end in windows:
        counts = np.bincount(truth[f_start:f_end], minlength=n_classes).astype(float)
        rows.append(counts / counts.sum())
    return np.asarray(rows)


def two_bouts(n_frames: int, cut: int) -> np.ndarray:
    """Class 0 before ``cut``, class 1 from it on."""
    return (np.arange(n_frames) >= cut).astype(np.int64)


def boundary(labels: np.ndarray, label: int = 1) -> int:
    """First frame carrying ``label``, or -1 if it never appears."""
    hit = np.flatnonzero(labels == label)
    return int(hit[0]) if hit.size else -1


def flat(rows: list[list[float]]) -> np.ndarray:
    return np.asarray(rows, dtype=float)


def spans(labels: np.ndarray, posterior: np.ndarray, n_classes: int = 2) -> list[tuple]:
    """The decoded bouts as ``(f_start, f_end, label)``, from frame 0.

    Read through :func:`runs_to_bouts` rather than off the label array, so the
    claims below are made about the thing callers will actually be handed.
    """
    names = [chr(ord("a") + i) for i in range(n_classes)]
    return [(b["f_start"], b["f_end"], b["label"])
            for b in runs_to_bouts(labels, posterior, names, 0)]


# --------------------------------------------------------------------------- #
# the posterior
# --------------------------------------------------------------------------- #
def test_posterior_is_the_mean_over_covering_windows() -> None:
    """The closed form, on a set small enough to average by hand.

    Windows [0,4), [2,6) and [5,9) over the span [0,10): frames 0-1 see the first
    alone, 2-3 see the first two, 4 sees the second alone, 5 sees the second and
    third, 6-8 see the third alone, and frame 9 sees nothing at all.
    """
    windows = [(0, 4), (2, 6), (5, 9)]
    proba = flat([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    posterior, n_covering = frame_posterior(windows, proba, 0, 10)

    expected = flat([
        [1.00, 0.00],      # 0: w0
        [1.00, 0.00],      # 1: w0
        [0.50, 0.50],      # 2: w0, w1
        [0.50, 0.50],      # 3: w0, w1
        [0.00, 1.00],      # 4: w1
        [0.25, 0.75],      # 5: w1, w2
        [0.50, 0.50],      # 6: w2
        [0.50, 0.50],      # 7: w2
        [0.50, 0.50],      # 8: w2
        [np.nan, np.nan],  # 9: nothing
    ])
    check("the posterior has one row per frame of the requested span",
          posterior.shape == (10, 2) and n_covering.shape == (10,),
          f"{posterior.shape}, {n_covering.shape}")
    check("every covered frame equals the hand-computed mean of its windows",
          np.allclose(posterior[:9], expected[:9], atol=TOL),
          str(posterior[:9] - expected[:9]))
    check("n_covering counts the windows over each frame",
          list(n_covering) == [1, 1, 2, 2, 1, 2, 1, 1, 1, 0], str(n_covering))
    check("a mean of probability vectors is still a probability vector",
          np.allclose(posterior[:9].sum(axis=1), 1.0, atol=TOL),
          str(posterior[:9].sum(axis=1)))


def test_an_uncovered_frame_is_nan_and_not_zero() -> None:
    """Nobody looked is not the same measurement as everything is equally likely.

    A row of zeros would be read downstream as a posterior - argmax picks class 0
    from it without complaint - so the distinction has to survive in the values,
    not only in the count beside them.
    """
    posterior, n_covering = frame_posterior([(0, 3)], flat([[0.8, 0.2]]), 0, 6)
    check("the uncovered frames are NaN in every class",
          bool(np.isnan(posterior[3:]).all()), str(posterior[3:]))
    check("and are not zero anywhere",
          not bool((posterior[3:] == 0.0).any()), str(posterior[3:]))
    check("n_covering is exactly 0 there and 1 where the window reached",
          list(n_covering) == [1, 1, 1, 0, 0, 0], str(n_covering))
    check("an uncovered frame decodes to -1 rather than to a class",
          list(decode_min_bout(posterior, 1)) == [0, 0, 0, -1, -1, -1],
          str(decode_min_bout(posterior, 1)))


def test_windows_are_clipped_to_the_requested_span() -> None:
    """A window may hang off either end; only its in-range part counts.

    Windows centred near a track's end legitimately reach past it, so refusing
    them would throw away exactly the evidence the ends are short of. The frames
    outside must contribute nothing at all - not a shortened mean, not a partial
    weight.
    """
    windows = [(-5, 3), (7, 20)]
    proba = flat([[0.9, 0.1], [0.2, 0.8]])
    posterior, n_covering = frame_posterior(windows, proba, 0, 10)

    check("the part of a window inside the span contributes in full",
          np.allclose(posterior[0:3], proba[0], atol=TOL)
          and np.allclose(posterior[7:10], proba[1], atol=TOL),
          f"{posterior[0:3]} / {posterior[7:10]}")
    check("the frames between the two windows stay uncovered",
          bool(np.isnan(posterior[3:7]).all()) and not n_covering[3:7].any(),
          str(n_covering))
    check("a window entirely outside the span contributes nothing",
          np.allclose(frame_posterior([(0, 3), (50, 60)], proba, 0, 4)[0][:3],
                      proba[0], atol=TOL))


def test_window_order_does_not_matter() -> None:
    """A mean has no order, so neither may this.

    The windows arrive in whatever order the dense table was written in, which is
    an implementation detail of a groupby; if it reached the answer the answer
    would move for no reason anybody could see.
    """
    rng = np.random.default_rng(4)
    truth = two_bouts(240, 91)
    windows = overlapping(240, 50, 7)
    proba = window_fractions(truth, windows, 2)
    proba = np.clip(proba + rng.normal(0, 0.05, proba.shape), 1e-6, None)
    proba /= proba.sum(axis=1, keepdims=True)

    reference, count_ref = frame_posterior(windows, proba, 0, 240)
    order = rng.permutation(len(windows))
    shuffled, count_shuffled = frame_posterior(
        [windows[i] for i in order], proba[order], 0, 240)

    check("shuffling the windows leaves the posterior unchanged",
          np.allclose(reference, shuffled, atol=TOL, equal_nan=True),
          str(np.nanmax(np.abs(reference - shuffled))))
    check("and leaves the coverage count exactly equal",
          bool((count_ref == count_shuffled).all()))
    check("and leaves the decoded labels identical",
          bool((decode_min_bout(reference, 25) == decode_min_bout(shuffled, 25)).all()))


# --------------------------------------------------------------------------- #
# the point of the module
# --------------------------------------------------------------------------- #
def test_overlap_localises_the_boundary() -> None:
    """A transition at frame 137, measured on a tiling grid and on a dense one.

    With 50-frame windows the tiling grid CANNOT answer 137: every window is
    either wholly before or wholly after, and the one straddling the transition
    (frames 100-149, 74% class 0) is a single vote for a single class over all
    fifty of its frames. The best it can do is 100 or 150.

    The dense grid scores the same frames every 10 frames instead, and the mean
    over the five windows covering a frame moves with the frame - so the crossing
    lands within one stride of the truth. Both posteriors here are the ones a
    PERFECT model would produce, so the whole difference is the grid.
    """
    n_frames, length, cut = 300, 50, 137
    truth = two_bouts(n_frames, cut)

    tiles = tiled(n_frames, length)
    tile_post, tile_count = frame_posterior(
        tiles, window_fractions(truth, tiles, 2), 0, n_frames)
    tile_edge = boundary(decode_min_bout(tile_post, 25))

    stride = 10
    dense = overlapping(n_frames, length, stride)
    dense_post, dense_count = frame_posterior(
        dense, window_fractions(truth, dense, 2), 0, n_frames)
    dense_edge = boundary(decode_min_bout(dense_post, 25))
    dense_viterbi = boundary(decode_viterbi(dense_post, self_transition_for(25)))

    check("both grids cover every frame of the track",
          bool((tile_count > 0).all() and (dense_count > 0).all()),
          f"{int((tile_count == 0).sum())} / {int((dense_count == 0).sum())} uncovered")
    check("the tiling grid can only report a multiple of the window length",
          tile_edge % length == 0 and tile_edge in (100, 150), str(tile_edge))
    check("the overlapping grid places the boundary within one stride of the truth",
          abs(dense_edge - cut) <= stride, f"{dense_edge} vs {cut}")
    check("which is closer than the tiling grid manages",
          abs(dense_edge - cut) < abs(tile_edge - cut),
          f"dense {dense_edge}, tiled {tile_edge}, truth {cut}")
    check("the Viterbi decode agrees with the constrained one on where it is",
          abs(dense_viterbi - cut) <= stride and dense_viterbi == dense_edge,
          f"viterbi {dense_viterbi}, min_bout {dense_edge}")
    # The resolution is the stride and nothing finer: between one window start
    # and the next the set of covering windows does not change, so neither can
    # the mean. Worth stating, so nobody reads a 3-frame agreement as 3-frame
    # accuracy.
    block = dense_post[cut - cut % stride: cut - cut % stride + stride]
    check("the posterior is constant between one window start and the next",
          np.allclose(block, block[0], atol=TOL), str(block))
    check("and does change at the next one, so the resolution is the stride",
          not np.allclose(block[0], dense_post[cut - cut % stride - 1], atol=TOL),
          str(dense_post[cut - cut % stride - 1]))
    check("both decodes see exactly two bouts and no flicker",
          len(spans(decode_min_bout(dense_post, 25), dense_post)) == 2)


# --------------------------------------------------------------------------- #
# decode_min_bout
# --------------------------------------------------------------------------- #
def spurious(n_frames: int, spans, n_classes: int = 2, base: float = 0.8) -> np.ndarray:
    """Class 0 everywhere, except ``(start, end, class)`` spans that favour another."""
    rest = (1.0 - base) / (n_classes - 1)
    posterior = np.full((n_frames, n_classes), rest)
    posterior[:, 0] = base
    for start, end, klass in spans:
        posterior[start:end] = rest
        posterior[start:end, klass] = base
    return posterior


def test_min_bout_absorbs_the_short_run_and_keeps_the_long_one() -> None:
    """A three-frame excursion is noise; a twenty-frame bout is a bout.

    Both are equally confident per frame here - the posterior says 0.8 in each -
    so nothing but LENGTH separates them, which is exactly the statement
    ``min_bout_frames`` makes.
    """
    posterior = spurious(80, [(40, 43, 1), (60, 80, 1)])
    plain = decode_min_bout(posterior, 1)
    decoded = decode_min_bout(posterior, 10)

    check("the plain argmax reports the spurious run",
          bool((plain[40:43] == 1).all()) and boundary(plain) == 40, str(boundary(plain)))
    check("a floor of 10 frames absorbs the 3-frame run into its neighbours",
          bool((decoded[0:60] == 0).all()), str(np.flatnonzero(decoded[0:60] != 0)))
    check("and leaves the 20-frame bout exactly where it was",
          bool((decoded[60:80] == 1).all()) and boundary(decoded) == 60,
          str(boundary(decoded)))
    check("so the decode is two bouts rather than four",
          len(spans(decoded, posterior)) == 2, str(spans(decoded, posterior)))
    check("a floor of 1 frame is a no-op",
          bool((decode_min_bout(posterior, 1) == plain).all()))
    check("a run exactly min_bout_frames long survives",
          bool((decode_min_bout(spurious(80, [(40, 50, 1)]), 10)[40:50] == 1).all()))


def test_absorption_follows_the_absorbed_frames_own_posterior() -> None:
    """Which neighbour a short run joins is decided by evidence, not by proximity.

    Three classes, with a two-frame run of class 2 wedged between a long class 0
    and a long class 1. Its own frames rank the two neighbours 0.34 to 0.30, so it
    must go to whichever of them the 0.34 belongs to - and the mirror image must
    go the other way. Anything that preferred the left neighbour, or the longer
    one, or the nearer boundary would pass one of these two and fail the other.
    """
    def wedge(left_score: float, right_score: float) -> np.ndarray:
        posterior = spurious(60, [(30, 60, 1)], n_classes=3)
        posterior[28:30] = [left_score, right_score, 0.36]
        return posterior

    to_right = decode_min_bout(wedge(0.30, 0.34), 5)
    to_left = decode_min_bout(wedge(0.34, 0.30), 5)

    check("the wedge really is its own class before the floor is applied",
          bool((decode_min_bout(wedge(0.30, 0.34), 1)[28:30] == 2).all()))
    check("a short run goes to the neighbour class its own frames rank higher",
          bool((to_right[28:30] == 1).all()), str(to_right[26:32]))
    check("and to the other one when the ranking is reversed",
          bool((to_left[28:30] == 0).all()), str(to_left[26:32]))
    check("either way class 2 has vanished and the bouts stay adjacent",
          not (to_right == 2).any() and not (to_left == 2).any()
          and len(spans(to_right, wedge(0.30, 0.34), 3)) == 2)


def test_min_bout_does_not_reach_across_a_gap() -> None:
    """Frames nobody scored are not evidence for joining what sits either side.

    Two class-0 bouts separated by three unscored frames are two bouts: merging
    them would claim coverage of frames no window reached. And a short run whose
    only defined neighbour is on one side must go to that side rather than being
    absorbed into the gap, which is not a class at all.
    """
    posterior = spurious(45, [(10, 12, 1)])
    posterior[20:23] = np.nan          # a gap between two class-0 stretches
    posterior[12:15] = np.nan          # and one on the short run's right
    decoded = decode_min_bout(posterior, 8)

    check("both gaps stay undefined",
          bool((decoded[12:15] == -1).all() and (decoded[20:23] == -1).all()),
          str(decoded[10:24]))
    check("the runs either side of a gap are not merged across it",
          spans(decoded, posterior)
          == [(0, 12, "a"), (12, 15, ""), (15, 20, "a"), (20, 23, ""), (23, 45, "a")],
          str(spans(decoded, posterior)))
    check("a short run beside a gap is absorbed by its one defined neighbour",
          bool((decoded[10:12] == 0).all()), str(decoded[8:14]))

    # An island shorter than the floor, with a gap on both sides, has nowhere to
    # go. Deleting it would throw away the only measurement of those frames.
    island = spurious(30, [(12, 15, 1)])
    island[10:12] = np.nan
    island[15:17] = np.nan
    check("a short run walled in by gaps is kept rather than deleted",
          bool((decode_min_bout(island, 8)[12:15] == 1).all()),
          str(decode_min_bout(island, 8)[8:20]))


def test_min_bout_leaves_nothing_short_behind() -> None:
    """The sweep runs to a FIXED POINT, checked on posteriors that decode to confetti.

    One pass is not enough: absorbing a short run can merge two others into a run
    that is still too short, and can leave the neighbour it joined below the floor
    as well. Random Dirichlet rows with holes punched in them produce hundreds of
    one- and two-frame runs, so anything that stopped early would show up as a
    survivor. The only bouts allowed to stay under the floor are islands with no
    defined neighbour to be absorbed by.
    """
    rng = np.random.default_rng(11)
    floor, offenders, before, islands = 15, [], 0, 0
    for trial in range(20):
        posterior = rng.dirichlet(np.ones(3), size=400)
        for start in rng.integers(0, 390, 8):
            posterior[start:start + int(rng.integers(1, 6))] = np.nan
        before += sum(b["n_frames"] < floor for b in runs_to_bouts(
            decode_min_bout(posterior, 1), posterior, ["a", "b", "c"], 0))
        bouts = runs_to_bouts(decode_min_bout(posterior, floor), posterior,
                              ["a", "b", "c"], 0)
        for i, bout in enumerate(bouts):
            if bout["label"] == "" or bout["n_frames"] >= floor:
                continue
            neighbours = [bouts[j]["label"] for j in (i - 1, i + 1) if 0 <= j < len(bouts)]
            if any(name != "" for name in neighbours):
                offenders.append((trial, bout))
            else:
                islands += 1

    check("the raw argmax on this input really is confetti", before > 200, str(before))
    check("after the sweep no bout under the floor has a neighbour to join",
          not offenders, f"{len(offenders)} left, e.g. {offenders[:2]}")
    print(f"  --  {before} short runs before the sweep, {islands} island(s) left after")


# --------------------------------------------------------------------------- #
# decode_viterbi
# --------------------------------------------------------------------------- #
def noisy_two_bouts(
    n_frames: int, cut: int, flicker: tuple[int, ...], seed: int = 7
) -> tuple[np.ndarray, np.ndarray]:
    """A two-bout posterior with per-frame noise and named single-frame lies.

    The flicker frames favour the WRONG class by 0.58 to 0.42 - enough for a
    per-frame argmax, nowhere near the cost of leaving a bout and coming back.
    """
    rng = np.random.default_rng(seed)
    truth = two_bouts(n_frames, cut)
    right = 0.72 + rng.uniform(-0.06, 0.06, n_frames)
    posterior = np.empty((n_frames, 2))
    posterior[np.arange(n_frames), truth] = right
    posterior[np.arange(n_frames), 1 - truth] = 1.0 - right
    for f in flicker:
        posterior[f, truth[f]] = 0.42
        posterior[f, 1 - truth[f]] = 0.58
    return truth, posterior


def test_viterbi_recovers_the_bouts_and_suppresses_flicker() -> None:
    """The transition cost is what a single frame's evidence has to beat.

    At ``self_transition_for(40)`` leaving a bout and returning costs
    ``2 * log(0.975 / 0.025)`` = 7.3 nats, against the 0.32 a single flickering
    frame offers. The per-frame argmax has no such term and takes every one.
    """
    truth, posterior = noisy_two_bouts(200, 120, (17, 55, 88, 140, 175))
    plain = decode_min_bout(posterior, 1)
    decoded = decode_viterbi(posterior, self_transition_for(40))

    check("the per-frame argmax follows every planted flicker",
          bool((plain[[17, 55, 88, 140, 175]] != truth[[17, 55, 88, 140, 175]]).all())
          and len(spans(plain, posterior)) >= 7,
          str(len(spans(plain, posterior))))
    check("Viterbi recovers the planted labels exactly",
          bool((decoded == truth).all()), str(np.flatnonzero(decoded != truth)))
    check("so it reports two bouts, cut where the truth was cut",
          spans(decoded, posterior) == [(0, 120, "a"), (120, 200, "b")],
          str(spans(decoded, posterior)))
    # Stickiness has to be a knob, not a constant: a chain barely stickier than
    # uniform must give the flicker back, or the parameter is doing nothing.
    check("a weak self-transition stops suppressing the flicker",
          len(spans(decode_viterbi(posterior, 0.51), posterior)) >= 7,
          str(len(spans(decode_viterbi(posterior, 0.51), posterior))))


def test_viterbi_decodes_the_segments_around_a_gap_independently() -> None:
    """A gap breaks the chain, so what is after it cannot depend on what is before.

    Checked both ways round: the whole-array decode must equal the two segments
    decoded on their own, and flipping the evidence BEFORE the gap must leave
    every label after it untouched. A path allowed to run through the gap would
    fail the second even when it passes the first.
    """
    first, second = 60, 80
    posterior = np.empty((first + 10 + second, 2))
    _, before = noisy_two_bouts(first, first, (), seed=3)          # all class 0
    _, after = noisy_two_bouts(second, 0, (), seed=5)              # all class 1
    posterior[:first] = before
    posterior[first:first + 10] = np.nan
    posterior[first + 10:] = after
    stay = self_transition_for(30)

    decoded = decode_viterbi(posterior, stay)
    check("the gap decodes to -1 rather than to a carried-over state",
          bool((decoded[first:first + 10] == -1).all()), str(decoded[first - 2:first + 12]))
    check("each segment equals the same frames decoded alone",
          bool((decoded[:first] == decode_viterbi(before, stay)).all()
               and (decoded[first + 10:] == decode_viterbi(after, stay)).all()),
          str(decoded))

    flipped = posterior.copy()
    flipped[:first] = after[:first]                                # class 1 before the gap
    check("flipping the evidence before the gap changes nothing after it",
          bool((decode_viterbi(flipped, stay)[first + 10:] == decoded[first + 10:]).all()),
          str(decode_viterbi(flipped, stay)[first + 10:first + 20]))
    check("and does change the labels before it, so the test is not vacuous",
          bool((decode_viterbi(flipped, stay)[:first] == 1).all()))
    check("the bout table spans the gap as a row, not as a hole",
          [name for _, _, name in spans(decoded, posterior)] == ["a", "", "b"],
          str(spans(decoded, posterior)))


# --------------------------------------------------------------------------- #
# the identity behind the prior
# --------------------------------------------------------------------------- #
def test_self_transition_matches_the_geometric_holding_time() -> None:
    """``1 / (1 - a)`` is the mean of the geometric holding time, summed to check.

    The identity is the whole justification for handing the same number to both
    decoders, so it is worth evaluating rather than asserting: the mean bout
    length of a chain with self-transition ``a`` is ``sum_k k a**(k-1) (1 - a)``.
    """
    check("self_transition_for(50) is 0.98", self_transition_for(50) == 0.98,
          repr(self_transition_for(50)))
    for frames in (2, 10, 25, 50, 200):
        a = self_transition_for(frames)
        k = np.arange(1, 200_000)
        expected = float((k * a ** (k - 1) * (1.0 - a)).sum())
        check(f"a bout at self_transition_for({frames}) lasts {frames} frames",
              abs(expected - frames) < 1e-6 * frames
              and abs(1.0 / (1.0 - a) - frames) < 1e-9,
              f"{expected} vs {frames}")
    check("a floor below 2 frames is refused rather than returning a=0",
          all(_refused(self_transition_for, n) for n in (1, 0, -5)))


def _refused(fn, *args) -> bool:
    try:
        fn(*args)
    except ValueError:
        return True
    return False


# --------------------------------------------------------------------------- #
# bouts
# --------------------------------------------------------------------------- #
def test_bouts_tile_the_span_exactly() -> None:
    """Sorted, disjoint, gapless - the contract other code will be written against.

    Checked on a decode that has everything awkward in it: an uncovered head, an
    interior gap, and a track that does not start at frame 0.
    """
    f_lo, n_frames, cut = 500, 120, 70
    truth = two_bouts(n_frames, cut)
    # No window before frame 10 or after 115 (a track detected late and lost
    # early), and none starting in [20, 40) - a stretch every window over it was
    # under `predict.min_frames` and so has no row to average.
    windows = [(f_lo + s, f_lo + s + 20)
               for s in range(10, n_frames - 20, 5) if not 20 <= s < 40]
    posterior, n_covering = frame_posterior(
        windows, window_fractions(truth, [(a - f_lo, b - f_lo) for a, b in windows], 2),
        f_lo, f_lo + n_frames)
    labels = decode_min_bout(posterior, 8)
    bouts = runs_to_bouts(labels, posterior, ["swim", "tumble"], f_lo)

    check("the decode really has an unscored head and an interior gap",
          bool((labels[:10] == -1).all()) and bool((n_covering == 0).sum() > 12),
          f"{int((n_covering == 0).sum())} uncovered frames")
    check("the bouts are sorted and disjoint",
          all(b["f_start"] < b["f_end"] for b in bouts)
          and all(a["f_end"] == b["f_start"] for a, b in zip(bouts, bouts[1:])),
          str([(b["f_start"], b["f_end"]) for b in bouts]))
    check("they are gapless and cover exactly the input span",
          bouts[0]["f_start"] == f_lo and bouts[-1]["f_end"] == f_lo + n_frames
          and sum(b["n_frames"] for b in bouts) == n_frames,
          f"{bouts[0]['f_start']}..{bouts[-1]['f_end']}")
    check("n_frames agrees with the span on every bout",
          all(b["n_frames"] == b["f_end"] - b["f_start"] for b in bouts))
    check("frames are absolute, so f_lo is added rather than assumed to be 0",
          all(b["f_start"] >= f_lo for b in bouts))

    unscored = [b for b in bouts if b["label"] == ""]
    scored = [b for b in bouts if b["label"] != ""]
    check("an unscored stretch is a row with an empty label and NaN confidence",
          bool(unscored) and all(np.isnan(b["conf"]) and b["class_index"] == -1
                                 for b in unscored),
          str(unscored))
    check("every scored bout is named from `classes` and carries its own mean posterior",
          all(b["label"] in ("swim", "tumble") for b in scored)
          and all(np.isclose(
              b["conf"],
              posterior[b["f_start"] - f_lo: b["f_end"] - f_lo, b["class_index"]].mean(),
              atol=TOL) for b in scored),
          str([(b["label"], round(b["conf"], 4)) for b in scored]))
    check("and the decoded transition is within one stride of the planted one",
          any(abs(b["f_start"] - (f_lo + cut)) <= 5 for b in scored[1:]),
          str([b["f_start"] - f_lo for b in scored]))


# --------------------------------------------------------------------------- #
# guards
# --------------------------------------------------------------------------- #
def test_a_misuse_is_refused_rather_than_answered() -> None:
    """Every guard, and what each one prevents being silently produced."""
    proba = flat([[0.6, 0.4], [0.1, 0.9]])
    windows = [(0, 5), (5, 10)]

    check("a probability row per window is required",
          _refused(frame_posterior, windows, proba[:1], 0, 10))
    check("a window that is not an (f_start, f_end) pair is refused",
          _refused(frame_posterior, [(0, 5, 9)], proba[:1], 0, 10))
    check("no windows at all is an all-NaN posterior, not an error",
          bool(np.isnan(frame_posterior([], np.zeros((0, 2)), 0, 4)[0]).all())
          and not frame_posterior([], np.zeros((0, 2)), 0, 4)[1].any())
    check("a non-finite probability is refused, not spread over the later frames",
          _refused(frame_posterior, windows, flat([[np.nan, 1.0], [0.1, 0.9]]), 0, 10))
    check("an empty frame span is refused",
          _refused(frame_posterior, windows, proba, 5, 5))
    check("a self-transition of 0 or 1 is refused",
          _refused(decode_viterbi, np.full((4, 2), 0.5), 0.0)
          and _refused(decode_viterbi, np.full((4, 2), 0.5), 1.0)
          and _refused(decode_viterbi, np.full((4, 2), 0.5), 1.5))
    check("a class list that does not match the posterior is refused",
          _refused(runs_to_bouts, np.zeros(4, dtype=int), np.full((4, 2), 0.5), ["a"], 0)
          and _refused(runs_to_bouts, np.zeros(3, dtype=int), np.full((4, 2), 0.5),
                       ["a", "b"], 0))

    # A single-class project is degenerate, not broken: the off-diagonal
    # transition would divide by n_classes - 1, so it is answered trivially.
    single = np.full((6, 1), 1.0)
    single[2:4] = np.nan
    check("a single-class posterior decodes trivially rather than dividing by zero",
          list(decode_viterbi(single, 0.9)) == [0, 0, -1, -1, 0, 0],
          str(decode_viterbi(single, 0.9)))
    check("a probability of exactly zero does not become an infinite emission",
          list(decode_viterbi(flat([[1.0, 0.0]] * 3 + [[0.0, 1.0]] * 3), 0.6))
          == [0, 0, 0, 1, 1, 1])
    check("an empty span of frames decodes to an empty answer",
          decode_min_bout(np.zeros((0, 2)), 5).size == 0
          and decode_viterbi(np.zeros((0, 2)), 0.9).size == 0
          and runs_to_bouts(np.zeros(0, dtype=int), np.zeros((0, 2)), ["a", "b"], 0) == [])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
