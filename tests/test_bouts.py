"""The per-frame posterior and the min-bout decode, against closed-form truth."""

from __future__ import annotations

import numpy as np
import pytest
from trackanno.infer.bouts import decode_min_bout, frame_posterior, runs_to_bouts

TOL = 1e-12


# --------------------------------------------------------------------------- #
# generators
# --------------------------------------------------------------------------- #
def tiled(n_frames: int, length: int) -> list[tuple[int, int]]:
    """Stride equals length, no overlap."""
    return [(s, min(s + length, n_frames)) for s in range(0, n_frames, length)]


def overlapping(n_frames: int, length: int, stride: int) -> list[tuple[int, int]]:
    """Whole windows every ``stride`` frames; the tail that would hang off is dropped."""
    return [(s, s + length) for s in range(0, n_frames - length + 1, stride)]


def window_fractions(truth: np.ndarray, windows, n_classes: int) -> np.ndarray:
    """What a perfect model reports for each window: its class fractions."""
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
    """The decoded bouts as ``(f_start, f_end, label)``, read through runs_to_bouts."""
    names = [chr(ord("a") + i) for i in range(n_classes)]
    return [(b["f_start"], b["f_end"], b["label"])
            for b in runs_to_bouts(labels, posterior, names, 0)]


def spurious(n_frames: int, spans, n_classes: int = 2, base: float = 0.8) -> np.ndarray:
    """Class 0 everywhere, except ``(start, end, class)`` spans that favour another."""
    rest = (1.0 - base) / (n_classes - 1)
    posterior = np.full((n_frames, n_classes), rest)
    posterior[:, 0] = base
    for start, end, klass in spans:
        posterior[start:end] = rest
        posterior[start:end, klass] = base
    return posterior


# --------------------------------------------------------------------------- #
# the posterior
# --------------------------------------------------------------------------- #
def test_posterior_is_the_mean_over_covering_windows() -> None:
    """Windows [0,4), [2,6), [5,9) over [0,10), averaged by hand."""
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
    assert posterior.shape == (10, 2) and n_covering.shape == (10,)
    assert np.allclose(posterior[:9], expected[:9], atol=TOL), posterior[:9] - expected[:9]
    assert list(n_covering) == [1, 1, 2, 2, 1, 2, 1, 1, 1, 0], n_covering
    assert np.allclose(posterior[:9].sum(axis=1), 1.0, atol=TOL), (
        "a mean of probability vectors is still a probability vector")


def test_an_uncovered_frame_is_nan_and_not_zero() -> None:
    """A zero row would decode to class 0; an uncovered frame must decode to -1."""
    posterior, n_covering = frame_posterior([(0, 3)], flat([[0.8, 0.2]]), 0, 6)
    assert np.isnan(posterior[3:]).all(), posterior[3:]
    assert list(n_covering) == [1, 1, 1, 0, 0, 0], n_covering
    assert list(decode_min_bout(posterior, 1)) == [0, 0, 0, -1, -1, -1]


def test_windows_are_clipped_to_the_requested_span() -> None:
    """Only the in-range part of a window counts, and it counts in full."""
    windows = [(-5, 3), (7, 20)]
    proba = flat([[0.9, 0.1], [0.2, 0.8]])
    posterior, n_covering = frame_posterior(windows, proba, 0, 10)

    assert np.allclose(posterior[0:3], proba[0], atol=TOL), posterior[0:3]
    assert np.allclose(posterior[7:10], proba[1], atol=TOL), posterior[7:10]
    assert np.isnan(posterior[3:7]).all() and not n_covering[3:7].any(), n_covering
    assert np.allclose(frame_posterior([(0, 3), (50, 60)], proba, 0, 4)[0][:3],
                       proba[0], atol=TOL), "a window entirely outside the span contributes"


def test_window_order_does_not_matter() -> None:
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

    assert np.allclose(reference, shuffled, atol=TOL, equal_nan=True), (
        np.nanmax(np.abs(reference - shuffled)))
    assert (count_ref == count_shuffled).all()
    assert (decode_min_bout(reference, 25) == decode_min_bout(shuffled, 25)).all()


# --------------------------------------------------------------------------- #
# overlap localises the boundary
# --------------------------------------------------------------------------- #
N_FRAMES, LENGTH, CUT, STRIDE = 300, 50, 137, 10


@pytest.fixture(scope="module")
def planted():
    """Perfect-model posteriors for a cut at 137, on the tiling and a stride-10 grid."""
    truth = two_bouts(N_FRAMES, CUT)
    tiles = tiled(N_FRAMES, LENGTH)
    dense = overlapping(N_FRAMES, LENGTH, STRIDE)
    return (frame_posterior(tiles, window_fractions(truth, tiles, 2), 0, N_FRAMES),
            frame_posterior(dense, window_fractions(truth, dense, 2), 0, N_FRAMES))


def test_overlap_localises_the_boundary(planted) -> None:
    """The tiling grid can only answer 100 or 150; the dense one lands within a stride."""
    (tile_post, tile_count), (dense_post, dense_count) = planted
    tile_edge = boundary(decode_min_bout(tile_post, 25))
    dense_edge = boundary(decode_min_bout(dense_post, 25))

    assert (tile_count > 0).all() and (dense_count > 0).all(), (
        f"{int((tile_count == 0).sum())} / {int((dense_count == 0).sum())} uncovered")
    assert tile_edge in (100, 150), tile_edge
    assert abs(dense_edge - CUT) <= STRIDE, f"{dense_edge} vs {CUT}"
    assert abs(dense_edge - CUT) < abs(tile_edge - CUT), (
        f"dense {dense_edge}, tiled {tile_edge}, truth {CUT}")
    assert len(spans(decode_min_bout(dense_post, 25), dense_post)) == 2, (
        "the decode sees two bouts and no flicker")


def test_the_dense_resolution_is_the_stride(planted) -> None:
    """The posterior is constant between window starts and changes at the next one."""
    _, (dense_post, _) = planted
    lo = CUT - CUT % STRIDE
    block = dense_post[lo: lo + STRIDE]
    assert np.allclose(block, block[0], atol=TOL), block
    assert not np.allclose(block[0], dense_post[lo - 1], atol=TOL), dense_post[lo - 1]


# --------------------------------------------------------------------------- #
# decode_min_bout
# --------------------------------------------------------------------------- #
def test_min_bout_absorbs_the_short_run_and_keeps_the_long_one() -> None:
    """Equally confident runs of 3 and 20 frames: only length separates them."""
    posterior = spurious(80, [(40, 43, 1), (60, 80, 1)])
    plain = decode_min_bout(posterior, 1)
    decoded = decode_min_bout(posterior, 10)

    assert (plain[40:43] == 1).all() and boundary(plain) == 40, boundary(plain)
    assert (decoded[0:60] == 0).all(), np.flatnonzero(decoded[0:60] != 0)
    assert (decoded[60:80] == 1).all() and boundary(decoded) == 60, boundary(decoded)
    assert len(spans(decoded, posterior)) == 2, spans(decoded, posterior)


def test_a_floor_of_one_frame_is_the_plain_argmax() -> None:
    posterior = spurious(80, [(40, 43, 1), (60, 80, 1)])
    assert (decode_min_bout(posterior, 1) == posterior.argmax(axis=1)).all()


def test_a_run_exactly_the_floor_long_survives() -> None:
    assert (decode_min_bout(spurious(80, [(40, 50, 1)]), 10)[40:50] == 1).all()


def test_absorption_follows_the_absorbed_frames_own_posterior() -> None:
    """A short class-2 wedge joins the neighbour its own frames rank higher, either side."""
    def wedge(left_score: float, right_score: float) -> np.ndarray:
        posterior = spurious(60, [(30, 60, 1)], n_classes=3)
        posterior[28:30] = [left_score, right_score, 0.36]
        return posterior

    to_right = decode_min_bout(wedge(0.30, 0.34), 5)
    to_left = decode_min_bout(wedge(0.34, 0.30), 5)

    assert (decode_min_bout(wedge(0.30, 0.34), 1)[28:30] == 2).all(), (
        "the wedge is its own class before the floor")
    assert (to_right[28:30] == 1).all(), to_right[26:32]
    assert (to_left[28:30] == 0).all(), to_left[26:32]
    assert not (to_right == 2).any() and not (to_left == 2).any()
    assert len(spans(to_right, wedge(0.30, 0.34), 3)) == 2


def test_min_bout_does_not_reach_across_a_gap() -> None:
    """Gaps stay undefined and unmerged; a short run beside one joins its defined side."""
    posterior = spurious(45, [(10, 12, 1)])
    posterior[20:23] = np.nan          # a gap between two class-0 stretches
    posterior[12:15] = np.nan          # and one on the short run's right
    decoded = decode_min_bout(posterior, 8)

    assert (decoded[12:15] == -1).all() and (decoded[20:23] == -1).all(), decoded[10:24]
    assert (spans(decoded, posterior)
            == [(0, 12, "a"), (12, 15, ""), (15, 20, "a"), (20, 23, ""), (23, 45, "a")]), (
        spans(decoded, posterior))
    assert (decoded[10:12] == 0).all(), decoded[8:14]


def test_a_short_run_walled_in_by_gaps_is_kept() -> None:
    island = spurious(30, [(12, 15, 1)])
    island[10:12] = np.nan
    island[15:17] = np.nan
    assert (decode_min_bout(island, 8)[12:15] == 1).all(), decode_min_bout(island, 8)[8:20]


def test_min_bout_leaves_nothing_short_behind() -> None:
    """The sweep runs to a fixed point: only gap-walled islands stay under the floor."""
    rng = np.random.default_rng(11)
    floor, offenders, before = 15, [], 0
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

    assert before > 200, f"the raw argmax is not confetti: {before}"
    assert not offenders, f"{len(offenders)} short bouts left, e.g. {offenders[:2]}"


# --------------------------------------------------------------------------- #
# bouts
# --------------------------------------------------------------------------- #
F_LO, N_SPAN, SPAN_CUT = 500, 120, 70


@pytest.fixture(scope="module")
def awkward():
    """A decode with an unscored head, an interior gap, and a track starting at 500."""
    truth = two_bouts(N_SPAN, SPAN_CUT)
    # No window before 10 or after 115, and none starting in [20, 40).
    windows = [(F_LO + s, F_LO + s + 20)
               for s in range(10, N_SPAN - 20, 5) if not 20 <= s < 40]
    posterior, n_covering = frame_posterior(
        windows, window_fractions(truth, [(a - F_LO, b - F_LO) for a, b in windows], 2),
        F_LO, F_LO + N_SPAN)
    labels = decode_min_bout(posterior, 8)
    bouts = runs_to_bouts(labels, posterior, ["swim", "tumble"], F_LO)
    return posterior, n_covering, labels, bouts


def test_bouts_tile_the_span_exactly(awkward) -> None:
    """Sorted, disjoint, gapless, in absolute frames."""
    _, n_covering, labels, bouts = awkward
    assert (labels[:10] == -1).all() and (n_covering == 0).sum() > 12, (
        "the case lacks an unscored head and interior gap")
    assert all(b["f_start"] < b["f_end"] for b in bouts)
    assert all(a["f_end"] == b["f_start"] for a, b in zip(bouts, bouts[1:])), (
        [(b["f_start"], b["f_end"]) for b in bouts])
    assert bouts[0]["f_start"] == F_LO and bouts[-1]["f_end"] == F_LO + N_SPAN
    assert sum(b["n_frames"] for b in bouts) == N_SPAN
    assert all(b["n_frames"] == b["f_end"] - b["f_start"] for b in bouts)


def test_an_unscored_stretch_is_a_row_with_an_empty_label(awkward) -> None:
    _, _, _, bouts = awkward
    unscored = [b for b in bouts if b["label"] == ""]
    assert unscored
    assert all(np.isnan(b["conf"]) and b["class_index"] == -1 for b in unscored), unscored


def test_a_scored_bout_carries_its_class_and_mean_posterior(awkward) -> None:
    posterior, _, _, bouts = awkward
    scored = [b for b in bouts if b["label"] != ""]
    assert all(b["label"] in ("swim", "tumble") for b in scored)
    for b in scored:
        mean = posterior[b["f_start"] - F_LO: b["f_end"] - F_LO, b["class_index"]].mean()
        assert np.isclose(b["conf"], mean, atol=TOL), (b["label"], b["conf"], mean)
    assert any(abs(b["f_start"] - (F_LO + SPAN_CUT)) <= 5 for b in scored[1:]), (
        [b["f_start"] - F_LO for b in scored])


# --------------------------------------------------------------------------- #
# guards
# --------------------------------------------------------------------------- #
PROBA = flat([[0.6, 0.4], [0.1, 0.9]])
WINDOWS = [(0, 5), (5, 10)]


@pytest.mark.parametrize("call", [
    pytest.param(lambda: frame_posterior(WINDOWS, PROBA[:1], 0, 10), id="row-per-window"),
    pytest.param(lambda: frame_posterior([(0, 5, 9)], PROBA[:1], 0, 10), id="not-a-pair"),
    pytest.param(lambda: frame_posterior(WINDOWS, flat([[np.nan, 1.0], [0.1, 0.9]]), 0, 10),
                 id="non-finite"),
    pytest.param(lambda: frame_posterior(WINDOWS, PROBA, 5, 5), id="empty-span"),
    pytest.param(lambda: runs_to_bouts(np.zeros(4, dtype=int), np.full((4, 2), 0.5), ["a"], 0),
                 id="too-few-classes"),
    pytest.param(lambda: runs_to_bouts(np.zeros(3, dtype=int), np.full((4, 2), 0.5),
                                       ["a", "b"], 0), id="length-mismatch"),
])
def test_a_misuse_is_refused(call) -> None:
    with pytest.raises(ValueError):
        call()


def test_no_windows_is_an_all_nan_posterior() -> None:
    posterior, n_covering = frame_posterior([], np.zeros((0, 2)), 0, 4)
    assert np.isnan(posterior).all() and not n_covering.any()


def test_an_empty_span_decodes_to_an_empty_answer() -> None:
    assert decode_min_bout(np.zeros((0, 2)), 5).size == 0
    assert runs_to_bouts(np.zeros(0, dtype=int), np.zeros((0, 2)), ["a", "b"], 0) == []


def test_a_viterbi_config_warns_that_it_was_removed() -> None:
    import warnings

    from trackanno.config import Predict

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parsed = Predict.parse({"decode": "viterbi", "self_transition": 0.9,
                                "min_bout_frames": 5})
    text = " ".join(str(w.message) for w in caught)
    assert "Viterbi" in text and "min_bout_frames" in text, text
    assert "unknown key" not in text, text
    assert parsed.min_bout_frames == 5
