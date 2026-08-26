"""Checks for :mod:`trackanno.learn.select`, which decides what a human annotates next.

Nothing here runs a model. Every probability matrix below is written out by hand
so that the row each rule must choose is known BEFORE the rule is called - the
scores are available in closed form from the definitions
(``1 - max p``, ``1 - (p1 - p2)``, ``-sum p log p / log K``), so a disagreement is
an error in the module rather than a disagreement between two implementations.

Six things are worth testing separately:

* the three uncertainty rules are three different functions, not three names for
  one. The matrix in the first test is built so that each of them ranks a
  DIFFERENT row first, which is the only claim that establishes it;
* a score orders what A-SOiD's 0.5 threshold can only flag;
* the answer depends on the pool's contents and nothing else - not on the order
  the rows arrived in, not on the run - including the seeded random stratum;
* the diversity rule really suppresses a near-duplicate, and says so by
  returning a short batch rather than by quietly padding it;
* a starved class gets targeted, and the two targeting rules pick different
  clips on data built to separate them;
* the audit stratum stays a UNIFORM sample: its draws ignore the scores
  entirely, cannot be suppressed by the diversity rule, and are labelled with
  their own reason so the sample can be recovered from the annotation file
  months later.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from harness import check
from trackanno.learn.select import (
    AUDIT_RULE, TARGET_RULES, UNCERTAINTY_RULES, entropy, least_confidence, margin,
    rank_by, select_batch, starved_classes, target_confident, target_runner_up,
)

TOL = 1e-12


# --------------------------------------------------------------------------- #
# generators
# --------------------------------------------------------------------------- #
def clip_id(track: int, frame: int, movie: str = "movie") -> str:
    """The id format the rest of the pipeline uses (``clips.clip_id_for``)."""
    return f"{movie}__t{int(track):04d}__f{int(frame):06d}"


def flat(rows) -> np.ndarray:
    return np.asarray(rows, dtype=float)


def hand_entropy(row) -> float:
    """``-sum p log p / log K``, written out from the definition."""
    return -sum(p * np.log(p) for p in row if p > 0) / np.log(len(row))


# Five classes, three rows, each of which one rule must rank first. See the
# module docstring of trackanno.learn.select for the argument; the numbers are checked
# below rather than asserted.
#
#   A  lowest maximum (0.30) but a clear leader          -> least_confidence
#   B  top two exactly tied, tail dead                   -> margin
#   C  flattest tail, and the WIDEST top-two gap of all  -> entropy
#   D  an easy clip, so the pool is not all borderline
DISAGREE = flat([
    [0.30, 0.25, 0.25, 0.20, 0.00],
    [0.35, 0.35, 0.30, 0.00, 0.00],
    [0.34, 0.165, 0.165, 0.165, 0.165],
    [0.90, 0.04, 0.03, 0.02, 0.01],
])
DISAGREE_IDS = np.array([clip_id(t, 0) for t in range(len(DISAGREE))])
FIVE = [f"c{i}" for i in range(5)]


def batch_ids(records) -> list[str]:
    return [r["clip_id"] for r in records]


def refusal(fn, *args, **kwargs) -> str:
    """The message a refusal carries, or "" if it did not refuse."""
    try:
        fn(*args, **kwargs)
    except ValueError as exc:
        return str(exc)
    return ""


# --------------------------------------------------------------------------- #
# the rules are three different functions
# --------------------------------------------------------------------------- #
def test_each_uncertainty_rule_ranks_a_different_row_first() -> None:
    """The claim that proves they are not three names for one function.

    Every score here is the closed form of the rule's own definition, so the
    expected values are arithmetic rather than a second implementation:
    ``1 - max p`` is 0.70 / 0.65 / 0.66 / 0.10, the top-two gaps are
    0.05 / 0.00 / 0.175 / 0.86, and the normalised entropies come out of
    ``-sum p log p / log 5``.
    """
    lc, mg, en = least_confidence(DISAGREE), margin(DISAGREE), entropy(DISAGREE)

    check("least_confidence is 1 - the top probability",
          np.allclose(lc, [0.70, 0.65, 0.66, 0.10], atol=TOL), str(lc))
    check("margin is 1 - the gap between the top two",
          np.allclose(mg, [0.95, 1.00, 0.825, 0.14], atol=TOL), str(mg))
    check("entropy is the Shannon entropy over log(n_classes)",
          np.allclose(en, [hand_entropy(r) for r in DISAGREE], atol=TOL), str(en))
    check("and so lands in [0, 1] with the uniform row at the ceiling",
          bool((en >= 0).all() and (en <= 1).all())
          and abs(float(entropy(flat([[0.2] * 5]))[0]) - 1.0) < TOL,
          str(en))

    check("least_confidence ranks first the row with the weakest top answer",
          int(np.argmax(lc)) == 0, f"{np.argmax(lc)}, scores {lc}")
    check("margin ranks first the row whose top two are tied",
          int(np.argmax(mg)) == 1, f"{np.argmax(mg)}, scores {mg}")
    check("entropy ranks first the flattest row",
          int(np.argmax(en)) == 2, f"{np.argmax(en)}, scores {en}")
    check("so the three rules disagree on the same matrix, first choice included",
          len({int(np.argmax(lc)), int(np.argmax(mg)), int(np.argmax(en))}) == 3)
    # Entropy's pick has the WIDEST top-two gap of the three borderline rows and
    # the highest maximum of the three: it is not a near-miss for the other two,
    # it is the row they both rank last of the three.
    check("and entropy's pick is the one margin ranks lowest of the three",
          int(np.argmin(mg[:3])) == 2 and int(np.argmin(lc[:3])) == 1,
          f"margin {mg[:3]}, least_confidence {lc[:3]}")

    for name, expected in (("least_confidence", 0), ("margin", 1), ("entropy", 2)):
        picked = select_batch(DISAGREE, DISAGREE_IDS, FIVE, 1, rule=name)
        check(f"a batch of one under {name} is that same row",
              batch_ids(picked) == [DISAGREE_IDS[expected]]
              and picked[0]["rule"] == name,
              str(picked))

    one_hot = flat([[1.0, 0.0, 0.0, 0.0, 0.0]])
    check("a one-hot row scores exactly 0 everywhere rather than NaN",
          float(entropy(one_hot)[0]) == 0.0 and float(margin(one_hot)[0]) == 0.0
          and float(least_confidence(one_hot)[0]) == 0.0)
    check("every rule name in UNCERTAINTY_RULES resolves to the function it names",
          UNCERTAINTY_RULES["least_confidence"] is least_confidence
          and UNCERTAINTY_RULES["margin"] is margin
          and UNCERTAINTY_RULES["entropy"] is entropy)


def test_a_threshold_flags_what_it_cannot_order() -> None:
    """A-SOiD's rule is a flag; this module's is a score. Six classes.

    "No class above 0.5" marks both rows below, but they are not equally unsure:
    the first is 0.45 against a uniform posterior of 1/6 = 0.167, which is 2.7
    times uniform and a perfectly usable prediction, while the second has three
    classes within 0.01 of each other. A flag cannot say which the annotator
    should see first, and with six classes it is flagging the first one at all
    that is the more obvious mistake.
    """
    rows = flat([
        [0.45, 0.11, 0.11, 0.11, 0.11, 0.11],
        [0.26, 0.25, 0.25, 0.12, 0.06, 0.06],
    ])
    lc, mg = least_confidence(rows), margin(rows)

    check("A-SOiD's threshold flags both rows",
          bool((rows.max(axis=1) < 0.5).all()), str(rows.max(axis=1)))
    check("the flagged 0.45 row is 2.7 times the uniform posterior of 1/6",
          abs(0.45 / (1 / 6) - 2.7) < 0.005, str(0.45 / (1 / 6)))
    check("least_confidence separates them, and by a wide margin",
          np.allclose(lc, [0.55, 0.74], atol=TOL) and lc[1] - lc[0] > 0.15, str(lc))
    check("margin separates them further still, 0.66 against 0.99",
          np.allclose(mg, [0.66, 0.99], atol=TOL), str(mg))
    check("so the batch shows the genuinely contested row first",
          batch_ids(select_batch(rows, np.array([clip_id(0, 0), clip_id(1, 0)]),
                                 [f"c{i}" for i in range(6)], 2))
          == [clip_id(1, 0), clip_id(0, 0)])
    # The class-count dependence the threshold ignores, stated as arithmetic: a
    # uniform row scores 1 - 1/K, so the same 0.55 is nearly the ceiling at K=2
    # and middling at K=6.
    check("least_confidence's own ceiling moves with the class count",
          abs(float(least_confidence(flat([[0.5, 0.5]]))[0]) - 0.5) < TOL
          and abs(float(least_confidence(flat([[1 / 6] * 6]))[0]) - 5 / 6) < TOL)
    check("while margin is 1.0 for a tie at either class count",
          float(margin(flat([[0.5, 0.5]]))[0]) == 1.0
          and abs(float(margin(flat([[1 / 6] * 6]))[0]) - 1.0) < TOL)


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #
def realistic_pool(seed: int = 3):
    """60 clips over 6 tracks, with deliberate exact ties and a mask of each kind.

    Ten rows are exact copies of another row, because identical posteriors are
    routine on easy clips and a tie broken by POSITION is exactly the bug the
    ``clip_id`` tiebreak exists to prevent - a pool without ties in it could not
    catch one.
    """
    rng = np.random.default_rng(seed)
    tracks = np.repeat(np.arange(6), 10)
    frames = np.tile(np.arange(10) * 50, 6)
    ids = np.array([clip_id(t, f) for t, f in zip(tracks, frames)])
    proba = rng.dirichlet(np.ones(4), size=60)
    proba[:10] = proba[20:30]                       # ten exact ties
    candidates = np.ones(60, dtype=bool)
    candidates[[3, 17, 41]] = False                 # already annotated
    audit = np.zeros(60, dtype=bool)
    audit[[5, 15, 25, 35, 45, 55]] = True           # the holdout
    return proba, ids, np.array([f"t{t}" for t in tracks]), frames, candidates, audit


def kwargs_for(proba, ids, tracks, frames, candidates, audit) -> dict:
    return dict(
        rule="margin", candidates=candidates, audit=audit, audit_share=0.2, seed=7,
        class_counts={"a": 40, "b": 3, "c": 30, "d": 25}, target_quota=6,
        track_ids=tracks, frames=frames, min_frame_gap=200,
    )


def test_the_batch_does_not_depend_on_the_row_order() -> None:
    """Shuffling the pool must not move the answer, ties and random draws included.

    The rows arrive in whatever order a groupby produced, which is not a fact
    about the data. With ten exact ties in the pool, a ranking that fell back on
    row position would reorder here; and the random stratum would too, since a
    draw taken straight off the array order is a draw from an arrangement rather
    than from a set.
    """
    proba, ids, tracks, frames, candidates, audit = realistic_pool()
    kwargs = kwargs_for(proba, ids, tracks, frames, candidates, audit)
    reference = select_batch(proba, ids, list("abcd"), 10, **kwargs)

    rng = np.random.default_rng(99)
    order = rng.permutation(len(ids))
    shuffled = select_batch(
        proba[order], ids[order], list("abcd"), 10,
        **kwargs_for(proba, ids, tracks[order], frames[order],
                     candidates[order], audit[order]),
    )

    check("the pool really does contain exact ties to break",
          int((margin(proba)[:10] == margin(proba)[20:30]).sum()) == 10)
    check("and the batch really does exercise every path - targeting, diversity "
          "and a random draw - so the invariance below is not invariance over "
          "one code path",
          {r["rule"] for r in reference} >= {"margin", "target_runner_up", AUDIT_RULE}
          and batch_ids(reference) != batch_ids(select_batch(
              proba, ids, list("abcd"), 10, **{**kwargs, "min_frame_gap": 0})),
          str([r["rule"] for r in reference]))
    check("shuffling the rows selects the same SET of clips",
          set(batch_ids(reference)) == set(batch_ids(shuffled)),
          f"{sorted(set(batch_ids(reference)) ^ set(batch_ids(shuffled)))}")
    check("and in the same ORDER, so the rank recorded on each is the same too",
          batch_ids(reference) == batch_ids(shuffled),
          f"{batch_ids(reference)}\n{batch_ids(shuffled)}")
    check("with the same rule and score attached to each clip",
          [(r["rule"], r["target_class"]) for r in reference]
          == [(r["rule"], r["target_class"]) for r in shuffled]
          and np.allclose([r["score"] for r in reference],
                          [r["score"] for r in shuffled], equal_nan=True))
    check("and the recorded index points at the same clip in the shuffled array",
          all(ids[order][r["index"]] == r["clip_id"] for r in shuffled))
    check("including the random audit draws, which come from a sorted pool",
          [r["clip_id"] for r in reference if r["rule"] == AUDIT_RULE]
          == [r["clip_id"] for r in shuffled if r["rule"] == AUDIT_RULE],
          str([r["clip_id"] for r in shuffled if r["rule"] == AUDIT_RULE]))


def test_the_same_input_twice_gives_the_same_batch() -> None:
    """Reproducibility, checked through the serialised form the CSV will hold.

    Compared as JSON rather than as dicts because a NaN score - what a random
    draw carries, having no score - is not equal to itself, and the thing that
    actually has to be reproducible is the text that lands in the file.
    """
    proba, ids, tracks, frames, candidates, audit = realistic_pool()
    kwargs = kwargs_for(proba, ids, tracks, frames, candidates, audit)
    first = select_batch(proba, ids, list("abcd"), 10, **kwargs)
    second = select_batch(proba, ids, list("abcd"), 10, **kwargs)

    check("two calls on one input serialise identically",
          json.dumps(first) == json.dumps(second), json.dumps(first))
    check("a different seed moves the random draws but not the ranked picks",
          [r["clip_id"] for r in first if r["rule"] != AUDIT_RULE]
          == [r["clip_id"] for r in select_batch(
              proba, ids, list("abcd"), 10, **{**kwargs, "seed": 8})
              if r["rule"] != AUDIT_RULE])
    check("rank_by is stable on exact ties, ordering them by clip id",
          list(rank_by(np.zeros(3), ["c", "a", "b"])) == [1, 2, 0],
          str(rank_by(np.zeros(3), ["c", "a", "b"])))


# --------------------------------------------------------------------------- #
# redundancy
# --------------------------------------------------------------------------- #
def near_duplicate_pool():
    """Two near-identical borderline clips 10 frames apart on ONE track.

    Rows 0 and 1 are the two most contested clips in the pool and they are
    adjacent windows of the same animal - the case pure uncertainty sampling
    returns twice and charges the human twice for. Rows 2 and 3 are on other
    tracks and are the next most contested.
    """
    proba = flat([
        [0.50, 0.50, 0.00],      # track 1, frame 100  - margin 1.00
        [0.49, 0.49, 0.02],      # track 1, frame 110  - margin 1.00, loses on id
        [0.45, 0.40, 0.15],      # track 2, frame 300  - margin 0.95
        [0.60, 0.35, 0.05],      # track 3, frame 700  - margin 0.75
        [0.95, 0.03, 0.02],      # track 4, frame 900  - easy
    ])
    ids = np.array([clip_id(1, 100), clip_id(1, 110), clip_id(2, 300),
                    clip_id(3, 700), clip_id(4, 900)])
    tracks = np.array(["t1", "t1", "t2", "t3", "t4"])
    frames = np.array([100, 110, 300, 700, 900])
    return proba, ids, tracks, frames


def test_the_diversity_rule_suppresses_a_near_duplicate() -> None:
    """Two adjacent windows of one track are one piece of information.

    Both score the maximum 1.00 on margin, so without the rule they are the whole
    of a two-clip batch. With a 50-frame separation required they are 10 frames
    apart, so the second is skipped and the slot goes to the next track down -
    which is the cost, stated: that clip on track 2 is less informative than the
    one that was skipped.
    """
    proba, ids, tracks, frames = near_duplicate_pool()
    common = dict(track_ids=tracks, frames=frames)
    without = select_batch(proba, ids, list("abc"), 2, min_frame_gap=0, **common)
    with_gap = select_batch(proba, ids, list("abc"), 2, min_frame_gap=50, **common)

    check("the two near-duplicates are the top two clips by score",
          np.allclose(margin(proba)[:2], 1.0, atol=TOL), str(margin(proba)))
    check("without the rule the batch is both of them",
          batch_ids(without) == [ids[0], ids[1]], str(batch_ids(without)))
    check("with a 50-frame gap required only one of the pair survives",
          [i for i in batch_ids(with_gap) if i in (ids[0], ids[1])] == [ids[0]],
          str(batch_ids(with_gap)))
    check("and the survivor is the higher-ranked of the two",
          with_gap[0]["clip_id"] == ids[0])
    check("the freed slot goes to a different track, not to nothing",
          with_gap[1]["clip_id"] == ids[2] and len(with_gap) == 2,
          str(batch_ids(with_gap)))
    check("which is the documented cost: the batch takes a lower-scoring clip",
          with_gap[1]["score"] < without[1]["score"],
          f"{with_gap[1]['score']} vs {without[1]['score']}")
    check("a pair exactly min_frame_gap apart is allowed, the bound being closed",
          len(select_batch(proba, ids, list("abc"), 2, min_frame_gap=10,
                           **common)) == 2)
    check("and one frame more is not",
          batch_ids(select_batch(proba, ids, list("abc"), 2, min_frame_gap=11,
                                 **common)) == [ids[0], ids[2]])
    check("clips on other tracks are never suppressed however close in frame",
          len(select_batch(
              flat([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.9, 0.1, 0.0]]),
              np.array([clip_id(1, 100), clip_id(2, 100), clip_id(3, 100)]),
              list("abc"), 2, track_ids=np.array(["t1", "t2", "t3"]),
              frames=np.array([100, 100, 100]), min_frame_gap=500)) == 2)


def test_a_batch_diversity_cannot_fill_comes_back_short() -> None:
    """A short batch is a result; too few candidates is a refusal. Both, side by side.

    Five clips all on one track inside one 500-frame gap can yield exactly one
    non-duplicate, so a request for three returns one rather than padding it with
    two clips that teach nothing. Asking for six clips from a pool of five is a
    different thing entirely - the caller built the mask and can count it - and
    that refuses.
    """
    frames = np.arange(5) * 20
    proba = flat([[0.5, 0.5], [0.49, 0.51], [0.48, 0.52], [0.47, 0.53], [0.46, 0.54]])
    ids = np.array([clip_id(1, f) for f in frames])
    tracks = np.array(["t1"] * 5)
    short = select_batch(proba, ids, ["a", "b"], 3, track_ids=tracks, frames=frames,
                         min_frame_gap=500)

    check("the batch is short rather than padded with a near-duplicate",
          len(short) == 1 and short[0]["clip_id"] == ids[0], str(batch_ids(short)))
    check("the same pool without the rule fills the batch",
          len(select_batch(proba, ids, ["a", "b"], 3, track_ids=tracks,
                           frames=frames, min_frame_gap=0)) == 3)
    check("while a batch larger than the candidate pool is refused, not shortened",
          "6" in refusal(select_batch, proba, ids, ["a", "b"], 6)
          and "5" in refusal(select_batch, proba, ids, ["a", "b"], 6),
          refusal(select_batch, proba, ids, ["a", "b"], 6))


# --------------------------------------------------------------------------- #
# starved classes
# --------------------------------------------------------------------------- #
# Three classes with "flick" (column 1) starved. Row 0 is deep inside the class
# the model already recognises; rows 3 and 4 are on its boundary - the model is
# considering flick and deciding against it.
STARVED = flat([
    [0.05, 0.90, 0.05],      # the model is sure this is flick
    [0.55, 0.40, 0.05],      # flick is a close second
    [0.70, 0.05, 0.25],      # flick is third; not a boundary clip at all
    [0.34, 0.33, 0.33],      # flick tied for second
    [0.45, 0.45, 0.10],      # flick tied with the leader: the boundary itself
])
STARVED_IDS = np.array([clip_id(t, 0) for t in range(len(STARVED))])
GROUPS = ["glide", "flick", "spin"]


def test_the_two_targeting_rules_choose_different_clips() -> None:
    """Fill the class fast, or move its boundary. Both, on one matrix.

    ``target_confident`` takes the clip the model is already sure about, which
    adds a label in the middle of the region the model ALREADY calls flick.
    ``target_runner_up`` takes the clip where flick is tied with the leader,
    which is the boundary exactly. They cannot both be right, and the point of
    implementing both is that which one is right depends on whether the class
    needs examples or needs an edge.
    """
    confident = target_confident(STARVED, 1)
    runner_up = target_runner_up(STARVED, 1)

    check("target_confident is the starved class's own probability",
          np.allclose(confident, STARVED[:, 1], atol=TOL), str(confident))
    check("and so ranks first the clip the model is surest about",
          int(np.argmax(confident)) == 0, str(confident))
    check("target_runner_up scores 0 for a clip the starved class WINS",
          runner_up[0] == 0.0, str(runner_up))
    check("and 0 for a clip where it is only third",
          runner_up[2] == 0.0, str(runner_up))
    check("and 1 - the gap to the leader where it is second",
          np.allclose(runner_up[[1, 3, 4]], [0.85, 0.99, 1.00], atol=TOL),
          str(runner_up))
    check("so it ranks first the clip on the boundary, while the clip confident "
          "ranks first is one it refuses outright",
          int(np.argmax(runner_up)) == 4
          and runner_up[int(np.argmax(confident))] == 0.0,
          f"{runner_up} / {confident}")

    counts = {"glide": 40, "flick": 1, "spin": 30}
    by_confidence = select_batch(STARVED, STARVED_IDS, GROUPS, 1, class_counts=counts,
                                 target_quota=6, target_rule="target_confident")
    by_boundary = select_batch(STARVED, STARVED_IDS, GROUPS, 1, class_counts=counts,
                               target_quota=6, target_rule="target_runner_up")
    check("a batch under each rule picks a different clip for the same starved class",
          batch_ids(by_confidence) == [STARVED_IDS[0]]
          and batch_ids(by_boundary) == [STARVED_IDS[4]],
          f"{batch_ids(by_confidence)} / {batch_ids(by_boundary)}")
    check("both record the class the slot was spent on and the rule that spent it",
          by_confidence[0]["target_class"] == "flick"
          and by_boundary[0]["target_class"] == "flick"
          and {by_confidence[0]["rule"], by_boundary[0]["rule"]} == set(TARGET_RULES))


def test_a_starved_class_is_targeted_until_it_is_not() -> None:
    """Slots go to the scarcest class each time, and stop at the quota.

    Two equally starved classes must not let one of them take every slot: the
    projection increments as the slots are handed out, so the next slot re-asks
    and they alternate. When one is further behind than the other it is levelled
    up first, which is the same rule and not a different one. Once nothing is
    starved the uncertainty rule takes over, and the record says which is which.
    """
    check("starved_classes reports the scarcest first, ties by name",
          starved_classes({"a": 1, "b": 1, "c": 9}, ["c", "b", "a"], 4) == ["a", "b"],
          str(starved_classes({"a": 1, "b": 1, "c": 9}, ["c", "b", "a"], 4)))
    check("a class absent from the counts is treated as having none, not skipped",
          starved_classes({"a": 9}, ["a", "b"], 4) == ["b"])

    counts = {"glide": 40, "flick": 1, "spin": 1}
    batch = select_batch(STARVED, STARVED_IDS, GROUPS, 5, class_counts=counts,
                         target_quota=3, target_rule="target_confident")
    spent = [r["target_class"] for r in batch]
    check("two equally starved classes alternate rather than one taking every slot",
          spent[:4] == ["flick", "spin", "flick", "spin"], str(spent))
    check("and once both reach the quota the uncertainty rule takes the rest",
          spent[4] == "" and batch[4]["rule"] == "margin", str(batch[4]))

    behind = [r["target_class"] for r in select_batch(
        STARVED, STARVED_IDS, GROUPS, 5, target_quota=4,
        class_counts={"glide": 40, "flick": 1, "spin": 3},
        target_rule="target_confident")]
    check("a class three labels further behind is levelled up before the other "
          "gets a slot",
          behind == ["flick", "flick", "flick", "spin", ""], str(behind))
    check("no targeting at all without a quota, whatever the counts say",
          {r["target_class"] for r in select_batch(
              STARVED, STARVED_IDS, GROUPS, 3, class_counts=counts)} == {""})
    check("and none when every class already clears it",
          {r["rule"] for r in select_batch(
              STARVED, STARVED_IDS, GROUPS, 3, target_quota=4,
              class_counts={"glide": 40, "flick": 40, "spin": 40})} == {"margin"})

    # A starved class with no boundary clip at all: the slot cannot be spent as
    # planned, and the record must not claim it was.
    no_boundary = flat([[0.90, 0.02, 0.08], [0.80, 0.05, 0.15], [0.50, 0.00, 0.50]])
    ids = np.array([clip_id(t, 0) for t in range(3)])
    fell_back = select_batch(no_boundary, ids, GROUPS, 1,
                             class_counts={"glide": 9, "flick": 0, "spin": 9},
                             target_quota=4, target_rule="target_runner_up")
    check("the pool really has no clip with the starved class as runner-up",
          bool((target_runner_up(no_boundary, 1) == 0).all()),
          str(target_runner_up(no_boundary, 1)))
    check("so the slot falls back to the uncertainty rule and SAYS so",
          fell_back[0]["rule"] == "margin" and fell_back[0]["target_class"] == "",
          str(fell_back))


# --------------------------------------------------------------------------- #
# the audit stratum
# --------------------------------------------------------------------------- #
def audit_pool(n_audit: int = 8, n_ranked: int = 20):
    """A pool whose audit clips are also its most uncertain ones.

    Built that way on purpose: if ranking ever reached the audit clips, it would
    take them first, so any leak shows up immediately rather than in the tail.
    """
    proba = np.concatenate([
        np.tile([0.5, 0.5], (n_audit, 1)),                    # audit: maximally unsure
        np.linspace(0.6, 0.95, n_ranked)[:, None] * [1, 0] +
        (1 - np.linspace(0.6, 0.95, n_ranked))[:, None] * [0, 1],
    ])
    ids = np.array([clip_id(0, i * 10) for i in range(n_audit)]
                   + [clip_id(1 + i, 0) for i in range(n_ranked)])
    audit = np.zeros(len(ids), dtype=bool)
    audit[:n_audit] = True
    return proba, ids, audit


def test_the_audit_draw_is_uniform_and_labelled_as_such() -> None:
    """The stratum that keeps `suggest.audit_accuracy` an unbiased estimate.

    Audit clips are annotatable like any others, so ranking the whole pool would
    fill the labelled audit subset with hard clips and quietly turn "accuracy on
    the holdout" into "accuracy on the hard part of the holdout". Here the audit
    clips ARE the most uncertain clips in the pool, so a leak is unmissable: the
    ranked picks must contain none of them, and the reserved slots must contain
    only clips a coin chose.
    """
    proba, ids, audit = audit_pool()
    batch = select_batch(proba, ids, ["a", "b"], 10, audit=audit, seed=1)
    drawn = [r for r in batch if r["rule"] == AUDIT_RULE]
    ranked = [r for r in batch if r["rule"] != AUDIT_RULE]

    check("the audit clips really are the most uncertain in the pool",
          bool((margin(proba[audit]) == 1.0).all()
               and (margin(proba[~audit]) < 1.0).all()))
    check("no ranked pick is an audit clip, however uncertain it is",
          not any(r["clip_id"] in set(ids[audit]) for r in ranked),
          str([r["clip_id"] for r in ranked]))
    check("the reserved slots are the documented int(share * n + 0.5)",
          len(drawn) == 1 and len(batch) == 10, str(len(drawn)))
    check("a draw carries its own rule, so the sample can be recovered later",
          all(r["rule"] == AUDIT_RULE and r["target_class"] == "" for r in drawn))
    check("and no score, because a uniform draw did not earn one",
          all(np.isnan(r["score"]) for r in drawn), str(drawn))
    check("its pool_size is the AUDIT pool, which is half of its inclusion probability",
          drawn[0]["pool_size"] == int(audit.sum())
          and ranked[0]["pool_size"] == int((~audit).sum()),
          f"{drawn[0]['pool_size']} / {ranked[0]['pool_size']}")

    # Uniformity, checked the sharp way: the draw must not depend on the SCORES
    # at all. Same ids, entirely different posteriors, same clips drawn.
    rekeyed = proba.copy()
    rekeyed[:8] = np.linspace(0.5, 0.99, 8)[:, None] * [1, 0] + \
        (1 - np.linspace(0.5, 0.99, 8))[:, None] * [0, 1]
    moved = select_batch(rekeyed, ids, ["a", "b"], 10, audit=audit, seed=1)
    check("the drawn clip does not change when the audit posteriors do",
          [r["clip_id"] for r in moved if r["rule"] == AUDIT_RULE]
          == [r["clip_id"] for r in drawn],
          str([r["clip_id"] for r in moved if r["rule"] == AUDIT_RULE]))

    seen = {}
    for seed in range(200):
        one = select_batch(proba, ids, ["a", "b"], 10, audit=audit, seed=seed)
        for record in one:
            if record["rule"] == AUDIT_RULE:
                seen[record["clip_id"]] = seen.get(record["clip_id"], 0) + 1
    check("over 200 seeds every audit clip is drawn, none excluded by construction",
          len(seen) == int(audit.sum()), str(sorted(seen)))
    check("and no clip takes more than twice its share of the draws",
          max(seen.values()) < 2 * 200 / int(audit.sum()),
          f"{sorted(seen.values())} against an expected {200 / int(audit.sum()):.0f}")

    spread = select_batch(proba, ids, ["a", "b"], 20, audit=audit, audit_share=0.25,
                          seed=2)
    check("draws are spread through the batch rather than parked at its end",
          [r["rank"] for r in spread if r["rule"] == AUDIT_RULE] == [2, 6, 10, 14, 18],
          str([r["rank"] for r in spread if r["rule"] == AUDIT_RULE]))
    check("audit_share=0 reserves nothing and leaves the holdout unlabelled",
          not any(r["rule"] == AUDIT_RULE for r in select_batch(
              proba, ids, ["a", "b"], 10, audit=audit, audit_share=0.0)))


def test_the_diversity_rule_cannot_suppress_an_audit_draw() -> None:
    """A draw that can be vetoed is not a uniform draw.

    One audit clip, 10 frames from the most contested ranked clip on the same
    track. The diversity rule would suppress a RANKED clip that close; it must
    not touch the draw, and must not let the draw veto the ranked pick either -
    or which informative clip the human sees would be decided by the seed.
    """
    proba = flat([[0.5, 0.5], [0.5, 0.5], [0.52, 0.48], [0.9, 0.1], [0.85, 0.15]])
    ids = np.array([clip_id(1, 100), clip_id(1, 110), clip_id(1, 120),
                    clip_id(2, 0), clip_id(3, 0)])
    tracks = np.array(["t1", "t1", "t1", "t2", "t3"])
    frames = np.array([100, 110, 120, 0, 0])
    audit = np.array([True, False, False, False, False])
    batch = select_batch(proba, ids, ["a", "b"], 3, audit=audit, audit_share=0.34,
                         track_ids=tracks, frames=frames, min_frame_gap=50, seed=0)

    check("the drawn audit clip is in the batch despite sitting 10 frames away",
          ids[0] in batch_ids(batch), str(batch_ids(batch)))
    check("and the ranked clip 10 frames from it is there too, not vetoed by it",
          ids[1] in batch_ids(batch), str(batch_ids(batch)))
    check("while the ranked pair 10 frames apart is still suppressed, so the "
          "test is not vacuous",
          ids[2] not in batch_ids(batch), str(batch_ids(batch)))
    check("the batch is full and the draw sits where the slot plan put it",
          len(batch) == 3 and batch[1]["rule"] == AUDIT_RULE, str(batch))


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #
def test_every_selection_records_how_it_was_chosen() -> None:
    """The record is the only thing that survives to explain a biased label set.

    Six months later the annotation CSV is all there is: the model that ranked
    the pool has been retrained, the posteriors are gone, and the question is
    whether an accuracy computed over these labels means anything. It does only
    if the rows can be partitioned into the ones a coin chose and the ones the
    model asked for.
    """
    proba, ids, tracks, frames, candidates, audit = realistic_pool()
    batch = select_batch(proba, ids, list("abcd"), 10,
                         **kwargs_for(proba, ids, tracks, frames, candidates, audit))
    fields = {"clip_id", "index", "rank", "rule", "score", "target_class",
              "predicted_class", "predicted_proba", "pool_size"}

    check("every record carries the same fields",
          all(set(r) == fields for r in batch), str(set(batch[0])))
    check("ranks are 0..n-1 in the order the batch is returned",
          [r["rank"] for r in batch] == list(range(len(batch))))
    check("no clip appears twice",
          len({r["clip_id"] for r in batch}) == len(batch))
    check("index points back at the row the clip came from",
          all(ids[r["index"]] == r["clip_id"] for r in batch))
    check("the recorded rule is one anybody can look up",
          all(r["rule"] in set(UNCERTAINTY_RULES) | set(TARGET_RULES) | {AUDIT_RULE}
              for r in batch), str({r["rule"] for r in batch}))
    check("the recorded score is the score that rule gives that clip",
          all(np.isclose(r["score"],
                         TARGET_RULES[r["rule"]](proba, list("abcd").index(
                             r["target_class"]))[r["index"]]
                         if r["rule"] in TARGET_RULES
                         else UNCERTAINTY_RULES[r["rule"]](proba)[r["index"]],
                         atol=TOL)
              for r in batch if r["rule"] != AUDIT_RULE))
    check("the model's own answer is recorded beside it",
          all(r["predicted_class"] == list("abcd")[int(proba[r["index"]].argmax())]
              and np.isclose(r["predicted_proba"], proba[r["index"]].max(), atol=TOL)
              for r in batch))
    check("a candidate excluded by the mask is never selected",
          not ({r["clip_id"] for r in batch} & set(ids[~candidates])))
    check("every value is a plain scalar, so a record survives a CSV cell",
          all(isinstance(v, (str, int, float)) and not isinstance(v, np.generic)
              for r in batch for v in r.values()))
    check("and the batch round-trips through JSON unchanged",
          json.loads(json.dumps(batch))[0]["clip_id"] == batch[0]["clip_id"]
          and len(json.loads(json.dumps(batch))) == len(batch))
    check("the labelled set can be partitioned into random and selected",
          0 < len([r for r in batch if r["rule"] == AUDIT_RULE]) < len(batch),
          str([r["rule"] for r in batch]))


# --------------------------------------------------------------------------- #
# guards
# --------------------------------------------------------------------------- #
def test_a_misuse_is_refused_rather_than_answered() -> None:
    """Every refusal, the value it names, and what it prevents being produced."""
    proba = flat([[0.6, 0.4], [0.1, 0.9], [0.5, 0.5]])
    ids = np.array([clip_id(t, 0) for t in range(3)])
    names = ["a", "b"]

    unnormalised = refusal(select_batch, flat([[0.6, 0.4], [0.3, 0.4]]), ids[:2],
                           names, 1)
    check("a row that does not sum to 1 is refused, naming the row and the sum",
          "row 1" in unnormalised and "0.7" in unnormalised, unnormalised)
    check("and the message names the clip, not just the row number",
          clip_id(1, 0) in unnormalised, unnormalised)
    check("a row of decision scores is refused rather than ranked",
          "negative" in refusal(select_batch, flat([[1.4, -0.4]]), ids[:1], names, 1))

    non_finite = refusal(select_batch, flat([[0.6, 0.4], [np.nan, np.nan]]),
                         ids[:2], names, 1)
    check("a non-finite row is refused, naming it",
          "row 1" in non_finite and "finite" in non_finite, non_finite)
    check("a single-class matrix is refused: there is nothing to be unsure about",
          "1 class column" in refusal(select_batch, flat([[1.0], [1.0]]), ids[:2],
                                      ["a"], 1))

    unknown = refusal(select_batch, proba, ids, names, 1,
                      class_counts={"a": 4, "wriggle": 2}, target_quota=9)
    check("a class count naming a class the matrix has no column for is refused",
          "'wriggle'" in unknown and "['a', 'b']" in unknown, unknown)
    check("and refused even when no slot would have been spent on it",
          "'wriggle'" in refusal(select_batch, proba, ids, names, 1,
                                 class_counts={"wriggle": 2}))

    too_big = refusal(select_batch, proba, ids, names, 9)
    check("a batch larger than the pool is refused, naming both numbers",
          "9" in too_big and "3" in too_big, too_big)
    check("an empty pool is refused rather than answered with an empty batch",
          "empty" in refusal(select_batch, np.zeros((0, 2)), np.array([]), names, 1))
    check("a mask that excludes everything is refused, and says which way",
          "excludes all 3" in refusal(select_batch, proba, ids, names, 1,
                                      candidates=np.zeros(3, dtype=bool)))
    check("a batch size of zero or less is refused",
          "0" in refusal(select_batch, proba, ids, names, 0)
          and refusal(select_batch, proba, ids, names, -2))

    starved_audit = refusal(select_batch, proba, ids, names, 3,
                            audit=np.array([True, False, False]), audit_share=1.0)
    check("an audit pool too small for its reserved slots is refused, naming both",
          "3" in starved_audit and "only 1" in starved_audit, starved_audit)
    check("rather than being topped up from the ranked pool",
          "ranked pool" in starved_audit, starved_audit)
    check("an audit mask covering every candidate is refused too",
          "audit clip" in refusal(select_batch, proba, ids, names, 2,
                                  audit=np.ones(3, dtype=bool), audit_share=0.0))
    check("a share outside [0, 1] is refused",
          "1.5" in refusal(select_batch, proba, ids, names, 1,
                           audit=np.array([True, False, False]), audit_share=1.5))

    gap = refusal(select_batch, proba, ids, names, 1, min_frame_gap=50)
    check("a frame gap without the track ids to apply it to is refused",
          "track_ids" in gap and "min_frame_gap=50" in gap, gap)
    check("and a negative gap is refused rather than disabling the rule",
          "-5" in refusal(select_batch, proba, ids, names, 1, min_frame_gap=-5,
                          track_ids=np.array(["t"] * 3), frames=np.zeros(3)))

    check("duplicate clip ids are refused: they break the tiebreak and the batch",
          "unique" in refusal(select_batch, proba,
                              np.array([clip_id(0, 0), clip_id(0, 0), clip_id(1, 0)]),
                              names, 1))
    check("a clip id per row is required",
          refusal(select_batch, proba, ids[:2], names, 1))
    check("a class name per column is required",
          refusal(select_batch, proba, ids, ["a", "b", "c"], 1))
    check("an unknown rule is refused, listing the ones there are",
          "margin" in refusal(select_batch, proba, ids, names, 1, rule="uncertain")
          and "target_runner_up" in refusal(select_batch, proba, ids, names, 1,
                                            target_rule="boundary"))
    check("a mask of the wrong length is refused",
          refusal(select_batch, proba, ids, names, 1,
                  candidates=np.ones(2, dtype=bool))
          and refusal(select_batch, proba, ids, names, 1, audit=np.ones(9, dtype=bool)))
    check("track ids and frames of the wrong length are refused",
          refusal(select_batch, proba, ids, names, 1, min_frame_gap=10,
                  track_ids=np.array(["t"] * 2), frames=np.zeros(3)))
    check("a rule called on something that is not a matrix is refused",
          refusal(margin, np.zeros(3)) and refusal(entropy, np.zeros((2, 2, 2))))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
