"""Checks for :mod:`trackanno.learn.select`, which decides what a human annotates next.

Every probability matrix is written out by hand, so the expected margin scores
are known in closed form (``1 - (p1 - p2)``).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from harness import check
from trackanno.learn.select import RULE, margin, rank_by, select_batch

TOL = 1e-12


def clip_id(track: int, frame: int, movie: str = "movie") -> str:
    """The id format the rest of the pipeline uses (``clips.clip_id_for``)."""
    return f"{movie}__t{int(track):04d}__f{int(frame):06d}"


def flat(rows) -> np.ndarray:
    return np.asarray(rows, dtype=float)


def batch_ids(records) -> list[str]:
    return [r["clip_id"] for r in records]


def refusal(fn, *args, **kwargs) -> str:
    """The message a refusal carries, or "" if it did not refuse."""
    try:
        fn(*args, **kwargs)
    except ValueError as exc:
        return str(exc)
    return ""


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
    check("the score is one minus the gap between the top two",
          np.allclose(scores, [0.9, 0.95, 0.0, 1.0], atol=TOL), str(scores))


def test_the_most_contested_clips_come_first() -> None:
    proba = flat([[0.9, 0.1], [0.55, 0.45], [0.7, 0.3], [0.5, 0.5]])
    ids = np.array([clip_id(t, 0) for t in range(4)])
    batch = select_batch(proba, ids, ["a", "b"], 3)
    check("the batch is ordered by margin",
          batch_ids(batch) == [ids[3], ids[1], ids[2]], str(batch_ids(batch)))


def test_the_batch_does_not_depend_on_the_row_order() -> None:
    proba, ids = pool()
    reference = select_batch(proba, ids, list("abc"), 10)
    order = np.random.default_rng(99).permutation(len(ids))
    shuffled = select_batch(proba[order], ids[order], list("abc"), 10)
    check("the pool really does contain exact ties to break",
          int((margin(proba)[:10] == margin(proba)[20:30]).sum()) == 10)
    check("shuffling the rows selects the same clips in the same order",
          batch_ids(reference) == batch_ids(shuffled))
    check("the recorded index points at the clip in the shuffled array",
          all(ids[order][r["index"]] == r["clip_id"] for r in shuffled))


def test_ties_are_broken_by_clip_id() -> None:
    scores = np.array([0.5, 0.5, 0.9])
    ids = np.array(["b", "a", "c"])
    check("ties fall back to clip id", list(rank_by(scores, ids)) == [2, 1, 0])


def test_every_selection_records_how_it_was_chosen() -> None:
    proba, ids = pool()
    candidates = np.ones(len(ids), dtype=bool)
    candidates[::3] = False
    batch = select_batch(proba, ids, list("abc"), 10, candidates=candidates)
    fields = {"clip_id", "index", "rank", "rule", "score",
              "predicted_class", "predicted_proba", "pool_size"}
    check("every record carries the same fields",
          all(set(r) == fields for r in batch), str(set(batch[0])))
    check("ranks are 0..n-1", [r["rank"] for r in batch] == list(range(len(batch))))
    check("the rule is margin", all(r["rule"] == RULE for r in batch))
    check("the score is the clip's margin",
          all(np.isclose(r["score"], margin(proba)[r["index"]], atol=TOL)
              for r in batch))
    check("the model's own answer is recorded beside it",
          all(r["predicted_class"] == "abc"[int(proba[r["index"]].argmax())]
              and np.isclose(r["predicted_proba"], proba[r["index"]].max())
              for r in batch))
    check("a clip excluded by the mask is never selected",
          not (set(batch_ids(batch)) & set(ids[~candidates])))
    check("the pool size counts candidates only",
          all(r["pool_size"] == int(candidates.sum()) for r in batch))
    check("every value is a plain scalar",
          all(isinstance(v, (str, int, float)) and not isinstance(v, np.generic)
              for r in batch for v in r.values()))
    check("and the batch round-trips through JSON",
          json.loads(json.dumps(batch)) == batch)


def test_a_small_pool_gives_a_short_batch() -> None:
    proba = flat([[0.6, 0.4], [0.1, 0.9], [0.5, 0.5]])
    ids = np.array([clip_id(t, 0) for t in range(3)])
    check("a batch larger than the pool is cut to the pool",
          len(select_batch(proba, ids, ["a", "b"], 9)) == 3)
    check("an empty candidate mask gives an empty batch",
          select_batch(proba, ids, ["a", "b"], 2,
                       candidates=np.zeros(3, dtype=bool)) == [])


def test_a_misuse_is_refused_rather_than_answered() -> None:
    proba = flat([[0.6, 0.4], [0.1, 0.9], [0.5, 0.5]])
    ids = np.array([clip_id(t, 0) for t in range(3)])
    names = ["a", "b"]

    unnormalised = refusal(select_batch, flat([[0.6, 0.4], [0.3, 0.4]]), ids[:2],
                           names, 1)
    check("a row that does not sum to 1 is refused, naming the clip",
          "row 1" in unnormalised and clip_id(1, 0) in unnormalised, unnormalised)
    check("a row of decision scores is refused",
          "negative" in refusal(select_batch, flat([[1.4, -0.4]]), ids[:1], names, 1))
    check("a non-finite row is refused",
          "finite" in refusal(select_batch, flat([[0.6, 0.4], [np.nan, np.nan]]),
                              ids[:2], names, 1))
    check("a single-class matrix is refused",
          refusal(select_batch, flat([[1.0], [1.0]]), ids[:2], ["a"], 1))
    check("a batch size of zero is refused",
          refusal(select_batch, proba, ids, names, 0))
    check("duplicate clip ids are refused",
          "unique" in refusal(select_batch, proba,
                              np.array([clip_id(0, 0)] * 2 + [clip_id(1, 0)]),
                              names, 1))
    check("a clip id per row is required",
          refusal(select_batch, proba, ids[:2], names, 1))
    check("a class name per column is required",
          refusal(select_batch, proba, ids, ["a", "b", "c"], 1))
    check("a mask of the wrong length is refused",
          refusal(select_batch, proba, ids, names, 1,
                  candidates=np.ones(2, dtype=bool)))
    check("margin on something that is not a matrix is refused",
          refusal(margin, np.zeros(3)))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
