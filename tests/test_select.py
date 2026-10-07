"""Margin selection in :mod:`trackanno.learn.select`, on hand-written probabilities."""

from __future__ import annotations

import json

import numpy as np
import pytest
from trackanno.learn.select import RULE, margin, rank_by, select_batch

TOL = 1e-12


def clip_id(track: int, frame: int, movie: str = "movie") -> str:
    """The id format the rest of the pipeline uses (``clips.clip_id_for``)."""
    return f"{movie}__t{int(track):04d}__f{int(frame):06d}"


def flat(rows) -> np.ndarray:
    return np.asarray(rows, dtype=float)


def batch_ids(records) -> list[str]:
    return [r["clip_id"] for r in records]


def pool(n: int = 40, seed: int = 0):
    """``n`` three-class rows with ten exact ties, and their clip ids."""
    rng = np.random.default_rng(seed)
    proba = rng.dirichlet([1.0, 1.0, 1.0], size=n)
    proba[20:30] = proba[:10]
    ids = np.array([clip_id(t, 0) for t in range(n)])
    return proba, ids


def test_margin_is_the_top_two_gap() -> None:
    proba = flat([[0.5, 0.4, 0.1], [0.5, 0.45, 0.05],
                  [1.0, 0.0, 0.0], [0.4, 0.4, 0.2]])
    scores = margin(proba)
    assert np.allclose(scores, [0.9, 0.95, 0.0, 1.0], atol=TOL), (
        f"the score is one minus the gap between the top two: {scores}")


def test_the_most_contested_clips_come_first() -> None:
    proba = flat([[0.9, 0.1], [0.55, 0.45], [0.7, 0.3], [0.5, 0.5]])
    ids = np.array([clip_id(t, 0) for t in range(4)])
    batch = select_batch(proba, ids, ["a", "b"], 3)
    assert batch_ids(batch) == [ids[3], ids[1], ids[2]], (
        f"the batch is ordered by margin: {batch_ids(batch)}")


def test_the_batch_does_not_depend_on_the_row_order() -> None:
    proba, ids = pool()
    reference = select_batch(proba, ids, list("abc"), 10)
    order = np.random.default_rng(99).permutation(len(ids))
    shuffled = select_batch(proba[order], ids[order], list("abc"), 10)
    assert int((margin(proba)[:10] == margin(proba)[20:30]).sum()) == 10, (
        "the pool really does contain exact ties to break")
    assert batch_ids(reference) == batch_ids(shuffled), (
        "shuffling the rows selects the same clips in the same order")
    assert all(ids[order][r["index"]] == r["clip_id"] for r in shuffled), (
        "the recorded index points at the clip in the shuffled array")


def test_ties_are_broken_by_clip_id() -> None:
    scores = np.array([0.5, 0.5, 0.9])
    ids = np.array(["b", "a", "c"])
    assert list(rank_by(scores, ids)) == [2, 1, 0], "ties fall back to clip id"


def test_every_selection_records_how_it_was_chosen() -> None:
    proba, ids = pool()
    candidates = np.ones(len(ids), dtype=bool)
    candidates[::3] = False
    batch = select_batch(proba, ids, list("abc"), 10, candidates=candidates)
    fields = {"clip_id", "index", "rank", "rule", "score",
              "predicted_class", "predicted_proba", "pool_size"}
    assert all(set(r) == fields for r in batch), (
        f"every record carries the same fields: {set(batch[0])}")
    assert [r["rank"] for r in batch] == list(range(len(batch))), "ranks are 0..n-1"
    assert all(r["rule"] == RULE for r in batch), "the rule is margin"
    assert (all(np.isclose(r["score"], margin(proba)[r["index"]], atol=TOL)
              for r in batch)), (
        "the score is the clip's margin")
    assert (all(r["predicted_class"] == "abc"[int(proba[r["index"]].argmax())]
              and np.isclose(r["predicted_proba"], proba[r["index"]].max())
              for r in batch)), (
        "the model's own answer is recorded beside it")
    assert not (set(batch_ids(batch)) & set(ids[~candidates])), (
        "a clip excluded by the mask is never selected")
    assert all(r["pool_size"] == int(candidates.sum()) for r in batch), (
        "the pool size counts candidates only")
    assert (all(isinstance(v, (str, int, float)) and not isinstance(v, np.generic)
              for r in batch for v in r.values())), (
        "every value is a plain scalar")
    assert json.loads(json.dumps(batch)) == batch, "the batch round-trips through JSON"


def test_a_small_pool_gives_a_short_batch() -> None:
    proba = flat([[0.6, 0.4], [0.1, 0.9], [0.5, 0.5]])
    ids = np.array([clip_id(t, 0) for t in range(3)])
    assert len(select_batch(proba, ids, ["a", "b"], 9)) == 3, (
        "a batch larger than the pool is cut to the pool")
    assert (select_batch(proba, ids, ["a", "b"], 2,
                       candidates=np.zeros(3, dtype=bool)) == []), (
        "an empty candidate mask gives an empty batch")


MISUSE_PROBA = [[0.6, 0.4], [0.1, 0.9], [0.5, 0.5]]
MISUSE_IDS = [clip_id(t, 0) for t in range(3)]


@pytest.mark.parametrize("proba, ids, names, n, kwargs, match", [
    ([[0.6, 0.4], [0.3, 0.4]], MISUSE_IDS[:2], ["a", "b"], 1, {}, "row 1"),
    ([[1.4, -0.4]], MISUSE_IDS[:1], ["a", "b"], 1, {}, "negative"),
    ([[0.6, 0.4], [np.nan, np.nan]], MISUSE_IDS[:2], ["a", "b"], 1, {}, "finite"),
    ([[1.0], [1.0]], MISUSE_IDS[:2], ["a"], 1, {}, ""),
    (MISUSE_PROBA, MISUSE_IDS, ["a", "b"], 0, {}, ""),
    (MISUSE_PROBA, [clip_id(0, 0)] * 2 + [clip_id(1, 0)], ["a", "b"], 1, {}, "unique"),
    (MISUSE_PROBA, MISUSE_IDS[:2], ["a", "b"], 1, {}, ""),
    (MISUSE_PROBA, MISUSE_IDS, ["a", "b", "c"], 1, {}, ""),
    (MISUSE_PROBA, MISUSE_IDS, ["a", "b"], 1, {"candidates": np.ones(2, dtype=bool)}, ""),
], ids=["unnormalised", "negative", "non-finite", "single-class", "zero-batch",
        "duplicate-ids", "ids-per-row", "names-per-column", "mask-length"])
def test_a_misuse_is_refused_rather_than_answered(proba, ids, names, n, kwargs, match) -> None:
    with pytest.raises(ValueError, match=match) as caught:
        select_batch(flat(proba), np.array(ids), names, n, **kwargs)
    if match == "row 1":
        assert clip_id(1, 0) in str(caught.value), "the unnormalised row names its clip"


def test_margin_refuses_something_that_is_not_a_matrix() -> None:
    with pytest.raises(ValueError):
        margin(np.zeros(3))

