"""Which clips a human should annotate next, and the recorded reason for each.

Clips currently reach the annotator in manifest order, which is the order the
window grid happened to produce - effectively arbitrary with respect to what the
model does or does not know. Most clips are easy, the model already agrees with
the answer, and the label teaches it nothing; the annotator's hour is the scarce
resource here and manifest order spends it uniformly.

A-SOiD's answer is to show the clips the model is unsure about, its rule being
"no class scores above 0.5". That rule is a THRESHOLDED SPECIAL CASE of
least-confidence sampling, and thresholding is where it goes wrong twice. It
degrades with the class count - with six classes the uniform posterior is
1/6 = 0.167, so a top score of 0.45 is 2.7 times uniform and is not an unsure
prediction, yet the rule flags it - and being a flag rather than a score it
cannot ORDER the clips it flags, so which of them the human sees first is again
arbitrary. This module scores instead of flagging, and adds the corrections the
flag does not make: redundancy, determinism, provenance, a candidate mask, and a
reserved random stratum that keeps the audit holdout honest.

Where the probabilities must come from
--------------------------------------
Raw ``predict_proba`` output over the FULL candidate pool, one row per clip.
Specifically NOT :meth:`trackanno.suggest.Suggester.suggest` output, for two
reasons that both make it the wrong source: it returns ``{}`` for an audit clip,
so the audit stratum below would have nothing to draw from, and it drops any
prediction under ``cfg.suggest.min_confidence`` - which is the low-confidence end
of the pool, the exact clips uncertainty sampling exists to find. (That filter is
inert at its 0.0 default, so the bug it would cause is one that appears the day
somebody raises the setting.)

The scoring rules, and what each actually prioritises
-----------------------------------------------------
All five return one score per row, HIGHER MEANS MORE WORTH ANNOTATING, so one
ranking path serves all of them.

* :func:`least_confidence` looks at the TOP probability and nothing else. It
  answers "how weak is the model's best answer", and is blind to whether the
  remaining mass sits on one rival or is smeared over five. Its ceiling depends
  on the class count - a uniform row scores ``1 - 1/n_classes`` - so the same
  number means different things in a 2-class and a 6-class project.
* :func:`margin` looks at the TOP TWO and nothing else: ``1 - (p1 - p2)``. That
  gap is the quantity a label actually resolves - it is how much evidence the
  model's answer has over its nearest rival - and its scale does not move with
  the class count.
* :func:`entropy` uses the whole vector, normalised by ``log(n_classes)`` so it
  lands in [0, 1] at any class count. It prefers mass spread over MANY classes,
  which means its ranking is largely decided by the TAIL.

They rank differently, and a caller choosing between them deserves the grounds.
Three rows over five classes, each of which one rule ranks first:

    A  [0.30, 0.25, 0.25, 0.20, 0.00]   weakest top answer, but a clear leader
    B  [0.35, 0.35, 0.30, 0.00, 0.00]   two tied leaders, dead tail
    C  [0.34, 0.165, 0.165, 0.165, 0.165]   flattest overall, widest top-two gap

Least-confidence takes A (its 0.30 is the lowest maximum), margin takes B (its
top two are tied, so the gap it scores is 0), entropy takes C (its tail is
uniform, and entropy is maximised by flatness). None of the three is wrong; they
are answering different questions. ``margin`` is the default here on the grounds
that the annotation resolves a contest between two candidate labels, so the
top-two gap is the quantity the human's answer actually settles - and because
entropy's ranking rests on the relative sizes of tail probabilities that no
boosted tree is calibrated to get right.

Class-targeted selection, and the trade-off between its two rules
-----------------------------------------------------------------
Uncertainty sampling has nothing to say about a class with four labels: whether
those four are enough is a fact about the label COUNTS, not about any clip's
posterior. Both rules take a starved class and both are implemented, because
they buy different things:

* :func:`target_confident` picks clips the model predicts as the starved class
  with HIGH confidence. It fills the class fastest - most of those clips really
  are that class - but every one of them sits in the middle of the region the
  model ALREADY calls that class, so the labels only ever confirm the model's
  existing notion of it. The class gets bigger; its BOUNDARY never moves.
* :func:`target_runner_up` picks clips where the starved class is the model's
  SECOND choice, ranked by how close it came. Those clips are rarer, and a good
  share of them will turn out to be the other class - so it fills the count more
  slowly and wastes more of the human's time - but each one lands exactly where
  the boundary between the starved class and its rival currently is, which is
  the only place a label can move it.

Neither is the right default for every project, so neither is hard-coded:
``target_rule`` selects, and the choice is recorded per clip. Both carry a
consequence outside the training set, spelled out in their docstrings: they
change the label MARGINALS, which is what the reliability gate is computed
against.

The corrections that are the reason this module exists
-------------------------------------------------------
**Redundancy.** Pure uncertainty sampling returns near-duplicates. The most
uncertain clip and the second most uncertain are frequently adjacent windows of
one track, which is a run of the same animal doing the same thing a few frames
later: they teach the model one thing and charge the human twice. The batch
enforces a minimum separation - see ``min_frame_gap`` in :func:`select_batch` -
and what it costs is stated there.

**Determinism.** Identical probability rows are routine on easy clips, so ties
are common, and a tie broken by ROW ORDER makes the batch depend on the order
the feature table came out of a groupby. Every ranking here is broken by
``clip_id``, which is stable and unique, and the random stratum draws from a
pool sorted by ``clip_id`` under a seeded generator - so the batch is a function
of the pool's CONTENTS and its seed, never of its arrangement.

**Provenance.** Every selection carries the rule that chose it, the score that
earned it, and its rank. This is not decoration. Once selection stops being
random the labelled set stops being a random sample of the data, and any
accuracy computed on it afterwards is an accuracy on a set assembled to be hard -
lower than the truth, by an amount nobody can estimate without knowing how it was
assembled. The records are plain scalars so they can be persisted next to the
label, and ``rule == "audit_random"`` is what partitions the labelled set into
"chosen at random" and "chosen because the model asked" long after the fact.

**A candidate mask.** Already-annotated clips must be excludable. This module
does not decide that policy: it takes a boolean mask and refuses to proceed if
the mask leaves fewer candidates than the caller asked for.

**A reserved random stratum, which protects the only unbiased number here.**
Audit clips (:func:`trackanno.suggest.is_audit_clip`) are held out of training
and out of the suggestion display, but they are NOT held out of the annotation
queue - the queue filters on existence, movie and done-ness and nothing else - so
a human labels them like any other clip, and ``audit_accuracy`` is computed over
whichever of them happen to have been labelled. Rank the whole pool by
uncertainty and the labelled audit subset stops being a random sample of the
audit set: it fills up with hard clips, and the one unbiased accuracy estimate in
the tool drifts DOWNWARD as the selector gets better at its job. Nothing
downstream would notice.

So the batch is stratified. Audit clips are excluded from the ranked pool
entirely, and a reserved share of the slots is drawn UNIFORMLY AT RANDOM from
the audit pool (``audit`` and ``audit_share`` in :func:`select_batch`). The
labelled audit subset therefore stays a random sample of the audit set whatever
the ranking does, and the price is that those slots do not go to the most
informative clip available. The random draws carry their own ``rule`` and are
exempt from the diversity rule, because a draw that can be suppressed is no
longer uniform.

Numpy only, no config, no file I/O: plain arrays in, plain dicts out.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "least_confidence", "margin", "entropy", "target_confident", "target_runner_up",
    "UNCERTAINTY_RULES", "TARGET_RULES", "AUDIT_RULE", "DEFAULT_AUDIT_SHARE",
    "rank_by", "starved_classes", "select_batch",
]

# How far a row's probabilities may drift from summing to 1 before it is refused.
# Loose enough for float32 predict_proba output accumulated over a few hundred
# trees, tight enough that a row missing a class, doubled, or built from decision
# scores that were never softmaxed cannot pass.
_SUM_TOL = 1e-6

# A target rule scores 0 for a row that does not qualify at all - the starved
# class is not the runner-up, or the model gives it no mass. Those rows are not
# candidates for that rule: filling the slot with one would hand the annotator a
# clip picked for no reason while claiming it was picked to feed a class.
_TARGET_FLOOR = 0.0

# The recorded reason for a clip drawn at random from the audit pool. A distinct
# string, and not a rule anyone can ask for: it is the label that lets someone
# later split the annotations into the ones chosen because the model asked and
# the ones chosen by a coin, and compute an honest accuracy over the second
# group alone.
AUDIT_RULE = "audit_random"

# Share of a batch reserved for that draw. Matches the 0.10 default of
# `suggest.audit_fraction`, so the labelled set keeps the same audit proportion
# as the clip population does and the audit count grows in step with the rest
# rather than stalling or swamping the batch.
DEFAULT_AUDIT_SHARE = 0.10


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def _as_proba(proba, clip_ids=None) -> np.ndarray:
    """The shared guard: a real probability matrix or a refusal naming the row.

    Everything downstream reads these numbers as a posterior - subtracts them,
    takes logs of them, compares them across rows - and every one of those
    operations returns something plausible for input that is not a posterior at
    all. Decision-function scores that were never softmaxed, a row that lost a
    class to a filter, a row doubled by a bad merge: all of them rank, and the
    batch that comes out looks exactly like a good one.

    ``clip_ids`` is optional and only improves the message. A row number is not
    something the caller can act on when the rows came out of a join.
    """
    proba = np.asarray(proba, dtype=float)
    if proba.ndim != 2:
        raise ValueError(
            f"proba must be (n_clips, n_classes), got shape {proba.shape}; one row "
            "per candidate clip, in the model's own class order"
        )

    def where(i: int) -> str:
        if clip_ids is None or i >= len(clip_ids):
            return f"row {i}"
        return f"row {i} (clip {clip_ids[i]!r})"

    if proba.size and not np.isfinite(proba).all():
        bad = int(np.flatnonzero(~np.isfinite(proba).all(axis=1))[0])
        raise ValueError(
            f"proba {where(bad)} is not finite: {proba[bad]}. A clip the model "
            "could not score has no posterior to rank, so leave it out of the "
            "candidate mask rather than passing NaN, which would sort silently"
        )
    if proba.size and proba.min() < 0.0:
        bad = int(np.unravel_index(int(np.argmin(proba)), proba.shape)[0])
        raise ValueError(
            f"proba {where(bad)} has a negative entry: {proba[bad]}. These must be "
            "probabilities, not decision-function scores - pass predict_proba "
            "output, or softmax the scores first"
        )
    if proba.size:
        sums = proba.sum(axis=1)
        off = np.flatnonzero(np.abs(sums - 1.0) > _SUM_TOL)
        if off.size:
            bad = int(off[0])
            raise ValueError(
                f"proba {where(bad)} sums to {sums[bad]:.6g}, not 1 "
                f"(tolerance {_SUM_TOL:g}); {off.size} of {proba.shape[0]} row(s) "
                "are off. A row that is not a distribution still ranks, so this is "
                "refused rather than renormalised - renormalising would hide "
                "whichever bug dropped or duplicated a class"
            )
    if proba.shape[1] < 2:
        raise ValueError(
            f"proba has {proba.shape[1]} class column(s); every rule here compares "
            "classes against each other, and a model with one class has nothing to "
            "be unsure about. Train a group with at least two classes"
        )
    return proba


def _class_index(classes, name) -> int:
    """Column of ``name``, or a refusal listing the columns there are."""
    names = [str(c) for c in classes]
    if str(name) not in names:
        raise ValueError(
            f"class {str(name)!r} has no column in the probability matrix, whose "
            f"classes are {names}. The counts and the columns come from different "
            "places - the annotation CSV and the model - so a renamed or mistyped "
            "class would otherwise target nothing at all"
        )
    return names.index(str(name))


# --------------------------------------------------------------------------- #
# uncertainty rules
# --------------------------------------------------------------------------- #
def least_confidence(proba) -> np.ndarray:
    """``1 - max(p)``: how weak the model's best answer is.

    Reads only the top probability. Two rows with the same maximum score the same
    however their remaining mass is arranged, so this cannot tell a genuine
    two-way contest from mass smeared thinly over classes nobody is considering.

    The scale moves with the class count: a uniform row over ``n_classes`` scores
    ``1 - 1/n_classes``, so 0.55 is nearly the ceiling in a 2-class project and
    is middling in a 6-class one. That dependence is exactly what makes A-SOiD's
    fixed 0.5 threshold on the same quantity fragile - and this returns a score,
    so the clips a threshold would lump together are still ordered.
    """
    proba = _as_proba(proba)
    return 1.0 - proba.max(axis=1)


def margin(proba) -> np.ndarray:
    """``1 - (p1 - p2)``: how contested the top two classes are.

    The gap between first and second is what a label resolves - it is the
    evidence the model's answer holds over its nearest rival - so this ranks
    clips by how little it would take to change the answer. Blind to the tail by
    construction: ``[0.5, 0.4, 0.1]`` and ``[0.5, 0.4, 0.05, 0.05]`` score the
    same, which is right if the decision in doubt is between the first two.

    Bounded in [0, 1] at every class count, unlike :func:`least_confidence`, and
    exactly 1 for a tie. This is the default rule in :func:`select_batch`.
    """
    proba = _as_proba(proba)
    # Partition rather than a full sort: the top two are all that is read, and
    # this stays linear in the class count.
    top2 = np.partition(proba, -2, axis=1)[:, -2:]
    return 1.0 - (top2[:, 1] - top2[:, 0])


def entropy(proba) -> np.ndarray:
    """Shannon entropy over the row, divided by ``log(n_classes)``.

    Uses every probability, so unlike the other two it distinguishes mass on one
    rival from mass spread over four - which also means its ranking is decided
    largely by the TAIL, and tail probabilities are the part of a boosted tree's
    output least likely to be calibrated. Maximised by the uniform row.

    Normalised by ``log(n_classes)`` so the answer is in [0, 1] whatever the class
    count and a threshold set on one project means the same on the next.
    ``0 * log 0`` is taken as 0, its limit, so a one-hot row scores exactly 0
    rather than NaN - a class the model rules out entirely is a legitimate
    posterior, not a defect.
    """
    proba = _as_proba(proba)
    positive = proba > 0.0
    terms = np.where(positive, proba * np.log(np.where(positive, proba, 1.0)), 0.0)
    return -terms.sum(axis=1) / np.log(proba.shape[1])


# --------------------------------------------------------------------------- #
# class-targeted rules
# --------------------------------------------------------------------------- #
def target_confident(proba, class_index: int) -> np.ndarray:
    """``p[class]``: clips the model most confidently calls the starved class.

    Fills a class fast, because most of these clips really are that class. What
    it cannot do is move the class BOUNDARY: a clip the model already scores 0.9
    for the class sits in the middle of the region the model already assigns to
    it, so the label agrees with the model and the decision surface does not
    move. Use it when a class is below ``min_per_class`` and the model needs
    examples at all; :func:`target_runner_up` is the other half of the trade-off.

    Both targeting rules deliberately skew the label marginals, and that reaches
    further than the training set. ``suggest.train_group`` gates the model on
    Cohen's kappa and on the majority-class baseline, both of which are computed
    against the marginals of the LABELLED set: over-sampling a rare class raises
    the chance-agreement floor kappa is corrected against, so kappa falls on a
    model that has not changed, and ``reliable`` can flip to False - which turns
    the suggester off for the annotator. Targeting is worth doing; it is not free,
    and it is not free in a place anyone would look.
    """
    proba = _as_proba(proba)
    return proba[:, int(class_index)].copy()


def target_runner_up(proba, class_index: int) -> np.ndarray:
    """``1 - (p_top - p[class])`` where the class is the model's SECOND choice, else 0.

    These are the clips sitting on the boundary between the starved class and
    whatever currently outranks it: the model is considering the class and
    deciding against it, and a human's answer either way moves the surface. Each
    label is worth more than a :func:`target_confident` one and a good share of
    them will come back as the OTHER class, so the count grows more slowly - that
    is the trade-off, and it is why both rules exist.

    Rows where the class is not the runner-up score exactly 0 and are not
    eligible for this rule in :func:`select_batch`. That includes every row the
    class WINS: a clip the model calls the starved class is not on its boundary,
    it is inside the class, and it is what :func:`target_confident` is for. Were
    winners admitted they would all score the maximum 1.0 - their gap to the top
    is zero by definition - and this rule would collapse into that one.

    Ties are counted generously at the other end: a class tied with the top
    without winning the argmax scores the maximum, because a tie is the boundary
    exactly.

    The same warning as :func:`target_confident` applies, and applies harder here
    because this rule deliberately picks clips the model will often be wrong
    about: skewing the label marginals moves Cohen's kappa and the majority
    baseline in ``suggest.train_group``, and those decide whether the suggester
    speaks at all.
    """
    proba = _as_proba(proba)
    k = int(class_index)
    column = proba[:, k]
    top = proba.max(axis=1)
    # Counting only classes STRICTLY above, so a class tied with the leader is
    # rank 0 and still qualifies; the argmax test is what keeps the actual winner
    # out, ties there going to the lowest column index as everywhere else.
    rank = (proba > column[:, None]).sum(axis=1)
    qualifies = (proba.argmax(axis=1) != k) & (rank <= 1)
    return np.where(qualifies, 1.0 - (top - column), 0.0)


UNCERTAINTY_RULES = {
    "least_confidence": least_confidence,
    "margin": margin,
    "entropy": entropy,
}

TARGET_RULES = {
    "target_confident": target_confident,
    "target_runner_up": target_runner_up,
}


# --------------------------------------------------------------------------- #
# ranking
# --------------------------------------------------------------------------- #
def rank_by(scores, clip_ids) -> np.ndarray:
    """Row indices, best score first, ties broken by ``clip_id``.

    The tiebreak is the whole point. Easy clips produce identical probability
    rows in numbers, so identical scores are routine, and breaking those ties by
    position would make the batch depend on the order the pool arrived in - a
    property of a groupby upstream, not of the data. ``clip_id`` is unique and
    stable across runs, so the same pool always yields the same order however it
    is arranged.
    """
    scores = np.asarray(scores, dtype=float)
    ids = np.asarray(clip_ids, dtype=str)
    if scores.shape != ids.shape:
        raise ValueError(
            f"{scores.size} score(s) against {ids.size} clip id(s); they are "
            "matched by position"
        )
    # lexsort takes the PRIMARY key last: score descending, then id ascending.
    return np.lexsort((ids, -scores))


def starved_classes(class_counts, classes, quota: int) -> list[str]:
    """Classes with fewer than ``quota`` labels, scarcest first, ties by name.

    A class ABSENT from ``class_counts`` counts as zero rather than being skipped.
    Skipping it would hide the most starved class of all behind a missing key,
    and a class with no labels is precisely the one a batch should be spent on.
    """
    names = [str(c) for c in classes]
    counts = {str(k): int(v) for k, v in dict(class_counts).items()}
    for name in counts:
        _class_index(names, name)
    under = [(counts.get(name, 0), name)
             for name in names if counts.get(name, 0) < int(quota)]
    return [name for _, name in sorted(under)]


def _plan_slots(
    n: int, rule: str, target_rule: str, class_counts, quota: int, names: list[str]
) -> list[tuple[str, str]]:
    """``(rule, target_class)`` for each of the ``n`` ranked slots, in order.

    Starved classes are served first and round-robin: the slot goes to whichever
    class is scarcest, that class's count is incremented, and the next slot
    re-asks - so two classes on 3 and 5 labels against a quota of 6 get the
    pattern 3, 3, 4, 5 rather than one class taking every slot. Once every class
    projects to reach the quota the remaining slots go to ``rule``.

    The projection counts a REQUEST, not a label. A targeted clip is not
    guaranteed to come back as that class - under ``target_runner_up`` a good
    share of them will not, which is the point of it - so this over-counts, and
    the correction arrives with the real counts on the next batch. Assuming the
    labels instead would either under- or over-fill; assuming nothing would spend
    the whole batch on one class.
    """
    if not class_counts or int(quota) <= 0:
        return [(rule, "")] * n
    projected = {name: 0 for name in names}
    projected.update({str(k): int(v) for k, v in dict(class_counts).items()})
    plan: list[tuple[str, str]] = []
    for _ in range(n):
        under = starved_classes(projected, names, quota)
        if not under:
            plan.append((rule, ""))
            continue
        plan.append((target_rule, under[0]))
        projected[under[0]] += 1
    return plan


def _audit_slots(n: int, reserved: int) -> set[int]:
    """Which positions in a batch of ``n`` belong to the random stratum.

    Spread evenly rather than parked at the end, because a batch is not always
    finished: an annotator who stops after six of twenty clips should still have
    labelled the audit draws in proportion, and draws at the end would simply
    never happen. Evenly means the midpoint of each equal share of the batch.
    """
    slots: list[int] = []
    for j in range(reserved):
        position = min(max(int((j + 0.5) * n / reserved), 0), n - 1)
        while position in slots:
            position = (position + 1) % n
        slots.append(position)
    return set(slots)


# --------------------------------------------------------------------------- #
# the batch
# --------------------------------------------------------------------------- #
def select_batch(
    proba,
    clip_ids,
    classes,
    n: int,
    *,
    rule: str = "margin",
    candidates=None,
    audit=None,
    audit_share: float = DEFAULT_AUDIT_SHARE,
    seed: int = 0,
    class_counts=None,
    target_quota: int = 0,
    target_rule: str = "target_runner_up",
    track_ids=None,
    frames=None,
    min_frame_gap: int = 0,
) -> list[dict]:
    """The ordered batch a human should annotate next, with a reason per clip.

    Parameters
    ----------
    proba
        ``(n_clips, n_classes)`` posteriors, in the model's own class order, from
        ``predict_proba`` over the whole pool - see the module docstring on why
        ``Suggester.suggest`` is the wrong source. Refused if any row is not a
        distribution.
    clip_ids
        One id per row. Must be unique - they are the tiebreak that makes the
        answer independent of row order, and a duplicate would both break that
        and let one clip be handed out twice in a batch.
    classes
        One name per column, in the model's order (``model.classes_``).
    n
        How many clips to select. The returned batch may be SHORTER (see below)
        but is never longer and never repeats a clip.
    rule
        One of :data:`UNCERTAINTY_RULES`. Default ``"margin"``; the module
        docstring gives the grounds and what the alternatives prioritise.
    candidates
        Optional boolean mask, ``True`` where a clip may be selected. This is how
        already-annotated clips stay out of the pool. The POLICY is the caller's;
        the refusal when the mask leaves fewer candidates than the caller asked
        for is not.
    audit
        Optional boolean mask marking the audit holdout
        (:func:`trackanno.suggest.is_audit_clip`). Those clips are removed from
        the ranked pool and form the pool the random stratum draws from. Passing
        ``None`` asserts the project HAS no holdout: it does not disable a
        holdout that exists, it hides it, and the ranked pool then eats audit
        clips by difficulty - which is the failure the stratum exists to prevent.
    audit_share
        Share of the batch reserved for uniform random draws from the audit pool,
        default :data:`DEFAULT_AUDIT_SHARE`. The slot count is
        ``int(audit_share * n + 0.5)``, so a batch of 10 at the default reserves
        1 and a batch of 4 reserves none - the share is over clips, not over
        batches, and small batches simply draw less often.
    seed
        Seeds the random stratum. The same seed, pool and batch size give the
        same draw, on any machine and whatever order the rows arrive in.
    class_counts
        ``{class name: labels so far}``. Only spends slots with ``target_quota``,
        but a name with no column in ``proba`` is refused whether it does or not.
    target_quota
        Labels a class needs before it stops being starved. Ranked slots are
        spent on starved classes first, scarcest first, round-robin. ``0`` (the
        default) disables targeting and every ranked slot comes from ``rule``.
    target_rule
        One of :data:`TARGET_RULES`, defaulting to ``"target_runner_up"`` - the
        boundary-moving one. See the module docstring for the trade-off against
        ``"target_confident"``, and both rules' docstrings for what targeting
        does to the reliability gate.
    track_ids, frames
        Per-row track identity and clip start frame (``f_start``). Required when
        ``min_frame_gap`` is set, and unused otherwise.
    min_frame_gap
        No two RANKED clips in one batch may come from the same track less than
        this many frames apart. Pairs exactly ``min_frame_gap`` apart are
        allowed, so setting it to the clip length admits adjacent-but-non-
        overlapping windows and anything larger excludes them too.

    Returns
    -------
    A list of dicts, in the order to annotate, each with:

    ``clip_id``, ``index``
        The clip, and its row in the ``proba`` that was passed in.
    ``rank``
        0-based position in the batch.
    ``rule``
        Which rule chose this clip - a key of :data:`UNCERTAINTY_RULES` or
        :data:`TARGET_RULES`, or :data:`AUDIT_RULE` for a random draw. Always the
        rule that ACTUALLY chose it, which is not always the one the plan asked
        for (see below).
    ``score``
        Its score under that rule, and NaN for a random draw - which has no
        score, and would be misread as a score of zero if one were invented.
    ``target_class``
        The starved class the slot was spent on, ``""`` otherwise.
    ``predicted_class``, ``predicted_proba``
        The model's own answer for the clip at selection time. Recorded because
        the useful question afterwards is not only "was this clip hard" but "was
        the model wrong about it", and the posterior that produced the batch is
        gone by then.
    ``pool_size``
        Candidates this clip was chosen from: the ranked pool for a ranked pick,
        the AUDIT pool for a random draw. Per row rather than per batch because
        the batch structure is lost once these are written beside individual
        labels - and for a random draw it is half of the inclusion probability,
        without which the sample cannot be weighted afterwards.

    Every value is a plain ``str``/``int``/``float``, so a record serialises into
    one CSV cell or JSON object unchanged.

    The random stratum
    ------------------
    Audit clips are excluded from the ranked pool and ``int(audit_share * n +
    0.5)`` slots are drawn uniformly at random from the audit pool instead. That
    keeps the LABELLED audit subset a random sample of the audit set, which is
    what makes ``suggest``'s ``audit_accuracy`` an unbiased estimate; rank those
    clips by uncertainty and the estimate silently becomes "accuracy on hard
    audit clips", drifting down as the selector improves.

    Those draws are exempt from the diversity rule and cannot be suppressed by
    it - a draw that can be vetoed is not a uniform draw any more - and they
    neither block nor are blocked by ranked picks, so which informative clip the
    human sees is never decided by the seed. If the audit pool cannot fill its
    reserved slots this refuses, rather than topping up from the ranked pool:
    quietly converting audit slots into ranked ones is precisely the failure
    being guarded against.

    Diversity, and what it costs
    ----------------------------
    Uncertainty is smooth along a track: the most uncertain clip and the next are
    frequently adjacent windows of the same animal, and labelling both charges
    the human twice for one piece of information. Ranked candidates are therefore
    taken greedily in score order and one is SKIPPED if a ranked clip already in
    the batch is from the same track less than ``min_frame_gap`` frames away.

    The cost is real and is paid on purpose: a genuinely informative clip is
    sometimes skipped. If an animal changes behaviour twice inside the gap, the
    batch sees one of the two transitions and the other waits for the next round.
    That is the deliberate trade - the skipped clip is still in the pool, while a
    duplicated one is spent human time that cannot be recovered.

    What it does NOT catch is redundancy between different tracks: two cells in
    one field of view doing the same thing at the same time are just as
    correlated, and this rule cannot see it. Catching that needs a distance in
    feature space (a greedy max-min, or a k-means over the clip features), which
    is deliberately not done here: this module is handed posteriors, not
    features, and a posterior is a poor proxy for similarity - two clips with
    identical class probabilities can be entirely different behaviours, since
    those few numbers are what the model kept, not what the clip contains. The
    frame rule is the one relation this codebase can assert with certainty, since
    two windows of one track within a clip length are literally the same footage.
    Doing better means passing the features in, and should be a deliberate change
    with a number behind it.

    A SHORT batch is a result, not an error. When diversity forbids the rest,
    fewer than ``n`` records come back and the batch is never padded with a
    near-duplicate; ``len()`` is the signal, and it is deterministic. That is
    deliberately different from too few candidates, which raises: the pool size is
    something the caller knows before it calls, because it built the mask, so
    asking for more than it holds is a bug in the caller, while what the
    diversity rule will permit depends on the ranking and cannot be known in
    advance.

    A slot planned for a starved class whose rule finds NO eligible clip - no
    remaining candidate has that class as its runner-up, say - falls back to
    ``rule`` for that slot rather than picking an arbitrary clip in the starved
    class's name. The record then names the rule that really chose it, so the
    provenance stays true even when the plan could not be honoured.
    """
    ids = np.asarray(clip_ids, dtype=str)
    names = [str(c) for c in classes]
    proba = _as_proba(proba, ids)
    n = int(n)

    if ids.shape != (proba.shape[0],):
        raise ValueError(
            f"{ids.size} clip id(s) for {proba.shape[0]} probability row(s); they "
            "are matched by position, so a mismatch means they describe different "
            "clips"
        )
    if len(names) != proba.shape[1]:
        raise ValueError(
            f"{len(names)} class name(s) for {proba.shape[1]} probability "
            "column(s); both come from the same model, and a mismatch means the "
            "batch would report the wrong behaviour for every clip in it"
        )
    unique, counts = np.unique(ids, return_counts=True)
    if unique.size != ids.size:
        repeated = [str(u) for u in unique[counts > 1]][:3]
        raise ValueError(
            f"clip ids must be unique, and {int((counts > 1).sum())} are not "
            f"(e.g. {repeated}); ids are the tiebreak that makes this ranking "
            "independent of row order, and a duplicate could also be handed to "
            "the annotator twice in one batch"
        )
    if n <= 0:
        raise ValueError(f"batch size must be at least 1, got {n}")
    if rule not in UNCERTAINTY_RULES:
        raise ValueError(
            f"unknown rule {rule!r}; choose from {sorted(UNCERTAINTY_RULES)}"
        )
    if target_rule not in TARGET_RULES:
        raise ValueError(
            f"unknown target_rule {target_rule!r}; choose from {sorted(TARGET_RULES)}"
        )
    if class_counts:
        for name in dict(class_counts):
            _class_index(names, name)

    mask = _candidate_mask(candidates, proba.shape[0])
    held = _audit_mask(audit, proba.shape[0]) & mask
    ranked_pool = np.flatnonzero(mask & ~held)
    audit_pool = np.flatnonzero(held)
    n_reserved = _reserved_slots(audit_share, n, audit is not None, audit_pool.size)
    _check_pool(ranked_pool.size, n - n_reserved, int(mask.sum()), audit_pool.size)

    gap = int(min_frame_gap)
    tracks, starts = _diversity_arrays(track_ids, frames, proba.shape[0], gap)
    drawn = _draw_audit(audit_pool, ids, n_reserved, seed)
    reserved_at = _audit_slots(n, n_reserved)
    plan = _plan_slots(n - n_reserved, rule, target_rule, class_counts,
                       int(target_quota), names)
    predicted = proba.argmax(axis=1)

    ranked: dict[tuple[str, str], np.ndarray] = {}
    scored: dict[tuple[str, str], np.ndarray] = {}

    def order_for(rule_name: str, target: str) -> tuple[np.ndarray, np.ndarray]:
        """Pool positions best-first under one rule, computed once and reused."""
        key = (rule_name, target)
        if key not in ranked:
            if rule_name in TARGET_RULES:
                scores = TARGET_RULES[rule_name](proba[ranked_pool],
                                                 _class_index(names, target))
            else:
                scores = UNCERTAINTY_RULES[rule_name](proba[ranked_pool])
            scored[key] = scores
            ranked[key] = rank_by(scores, ids[ranked_pool])
        return ranked[key], scored[key]

    taken: set[int] = set()
    placed: list[int] = []                           # ranked rows chosen so far
    chosen: list[tuple[int, str, float, str, int]] = []
    exhausted = False
    plan_at, draw_at = 0, 0
    for slot in range(n):
        if slot in reserved_at:
            chosen.append((int(drawn[draw_at]), AUDIT_RULE, float("nan"), "",
                           int(audit_pool.size)))
            draw_at += 1
            continue
        rule_name, target = plan[plan_at]
        plan_at += 1
        if exhausted:
            continue
        pick = None
        if target:
            order, scores = order_for(rule_name, target)
            pick = _first_admissible(order, scores, taken, placed, tracks, starts,
                                     ranked_pool, gap, floor=_TARGET_FLOOR)
            if pick is not None:
                row = int(ranked_pool[pick])
                chosen.append((row, rule_name, float(scores[pick]), target,
                               int(ranked_pool.size)))
                taken.add(pick)
                placed.append(row)
                continue
        order, scores = order_for(rule, "")
        pick = _first_admissible(order, scores, taken, placed, tracks, starts,
                                 ranked_pool, gap, floor=None)
        if pick is None:
            # Nothing left that diversity permits. Uncertainty rules make every
            # remaining candidate eligible, so no later ranked slot could be
            # filled either - this is the short batch the docstring promises. The
            # reserved draws still stand: they are a measurement, not a ranking.
            exhausted = True
            continue
        row = int(ranked_pool[pick])
        chosen.append((row, rule, float(scores[pick]), "", int(ranked_pool.size)))
        taken.add(pick)
        placed.append(row)

    out: list[dict] = []
    for rank, (row, rule_name, score, target, pool_size) in enumerate(chosen):
        out.append({
            "clip_id": str(ids[row]),
            "index": int(row),
            "rank": rank,
            "rule": rule_name,
            "score": float(score),
            "target_class": target,
            "predicted_class": names[int(predicted[row])],
            "predicted_proba": float(proba[row, int(predicted[row])]),
            "pool_size": int(pool_size),
        })
    return out


def _candidate_mask(candidates, n_rows: int) -> np.ndarray:
    """The selectable rows, or a refusal."""
    if n_rows == 0:
        raise ValueError(
            "the candidate pool is empty (0 probability rows); there is nothing to "
            "select from, which is a state to report rather than a batch to build"
        )
    if candidates is None:
        return np.ones(n_rows, dtype=bool)
    mask = np.asarray(candidates, dtype=bool)
    if mask.shape != (n_rows,):
        raise ValueError(
            f"candidates must be a boolean mask of {n_rows} row(s), got shape "
            f"{mask.shape}; it is matched to the probability rows by position"
        )
    if not mask.any():
        raise ValueError(
            f"the candidate mask excludes all {n_rows} clip(s); every clip is "
            "already annotated, or the mask was inverted - which of those it is "
            "decides what to do, so this is refused rather than answered with an "
            "empty batch"
        )
    return mask


def _audit_mask(audit, n_rows: int) -> np.ndarray:
    if audit is None:
        return np.zeros(n_rows, dtype=bool)
    mask = np.asarray(audit, dtype=bool)
    if mask.shape != (n_rows,):
        raise ValueError(
            f"audit must be a boolean mask of {n_rows} row(s), got shape "
            f"{mask.shape}; it is matched to the probability rows by position"
        )
    return mask


def _reserved_slots(share: float, n: int, has_audit: bool, available: int) -> int:
    """How many slots the random stratum takes, or a refusal naming both numbers."""
    share = float(share)
    if not 0.0 <= share <= 1.0:
        raise ValueError(
            f"audit_share must be in [0, 1], got {share}; it is a share of the "
            "batch, and 1.0 (every slot a random draw) is already the degenerate "
            "end of it"
        )
    if not has_audit:
        return 0
    reserved = int(share * n + 0.5)
    if reserved > available:
        raise ValueError(
            f"audit_share={share:g} reserves {reserved} of {n} slot(s) for random "
            f"draws but the audit pool holds only {available} unannotated clip(s); "
            "ask for a smaller batch, lower audit_share, or accept that the audit "
            "set is exhausted. Topping the slots up from the ranked pool instead "
            "would quietly turn the unbiased sample into a selected one, which is "
            "the failure this stratum exists to prevent"
        )
    return reserved


def _check_pool(available: int, wanted: int, n_candidates: int, n_audit: int) -> None:
    """The ranked pool must cover the ranked slots, or say which way it is short."""
    if wanted <= 0:
        return
    if available == 0:
        raise ValueError(
            f"every one of the {n_candidates} candidate(s) is an audit clip, so "
            f"the ranked pool is empty and the {wanted} ranked slot(s) cannot be "
            "filled; the audit fraction that produced this holdout is too large "
            "for the project, or the masks were swapped"
        )
    if available < wanted:
        raise ValueError(
            f"{wanted} ranked slot(s) were requested but only {available} "
            f"candidate(s) are available to rank ({n_candidates} candidate(s), of "
            f"which {n_audit} are held out for audit); ask for at most "
            f"{available}, or widen the candidate mask. Returning a short batch "
            "here would hide that the project has run out of clips to annotate"
        )


def _draw_audit(
    audit_pool: np.ndarray, ids: np.ndarray, reserved: int, seed
) -> np.ndarray:
    """``reserved`` rows drawn uniformly without replacement from the audit pool.

    The pool is sorted by ``clip_id`` before the draw so the result depends on
    the pool's CONTENTS and the seed alone. Drawing straight from the array order
    would make the sample depend on the order the rows arrived in, which is the
    one thing a random draw and a deterministic module must both be free of.
    """
    if reserved == 0:
        return np.zeros(0, dtype=np.int64)
    ordered = audit_pool[np.argsort(ids[audit_pool], kind="stable")]
    rng = np.random.default_rng(int(seed))
    return ordered[rng.choice(ordered.size, size=reserved, replace=False)]


def _diversity_arrays(track_ids, frames, n_rows: int, gap: int):
    """Track identity and start frame per row, validated only if the rule is on."""
    if gap < 0:
        raise ValueError(
            f"min_frame_gap must be at least 0, got {gap}; 0 disables the "
            "diversity rule, and a negative separation is not a thing to require"
        )
    if gap == 0:
        return None, None
    if track_ids is None or frames is None:
        raise ValueError(
            f"min_frame_gap={gap} needs `track_ids` and `frames` to say which "
            "clips are near-duplicates of each other, and got "
            f"track_ids={'None' if track_ids is None else 'given'}, "
            f"frames={'None' if frames is None else 'given'}; pass both, or set "
            "min_frame_gap=0 to turn the rule off deliberately"
        )
    tracks = np.asarray(track_ids, dtype=str)
    starts = np.asarray(frames)
    if tracks.shape != (n_rows,) or starts.shape != (n_rows,):
        raise ValueError(
            f"track_ids {tracks.shape} and frames {starts.shape} must both have "
            f"one entry per probability row ({n_rows}); they are matched by "
            "position"
        )
    return tracks, starts.astype(np.int64)


def _first_admissible(order, scores, taken, placed, tracks, starts, pool, gap, floor):
    """Best-ranked pool position that is untaken, above ``floor``, and far enough.

    ``floor`` is ``None`` for the uncertainty rules, where every candidate is
    eligible however confident the model is: a batch bigger than the number of
    genuinely uncertain clips should still be filled, and the recorded score says
    plainly that the last few were easy.

    ``placed`` holds the RANKED rows already in the batch - not the random draws,
    which are outside this relation in both directions.
    """
    for position in order:
        position = int(position)
        if position in taken:
            continue
        if floor is not None and scores[position] <= floor:
            # Sorted, so nothing further down qualifies either.
            return None
        if tracks is not None:
            row = int(pool[position])
            near = any(tracks[row] == tracks[other]
                       and abs(int(starts[row]) - int(starts[other])) < gap
                       for other in placed)
            if near:
                continue
        return position
    return None
