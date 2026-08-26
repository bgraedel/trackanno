"""Which clips a human should annotate next, and the recorded reason for each.

Manifest order spends the annotator's time uniformly, mostly on clips the model
already gets right. This module ranks a candidate pool instead and records, per
clip, which rule chose it, what it scored and where it ranked.

Probabilities must be raw ``predict_proba`` output over the FULL candidate pool,
one row per clip, and not
:meth:`trackanno.suggest.Suggester.suggest` output: that returns ``{}`` for an
audit clip and drops anything under ``cfg.suggest.min_confidence``, which is the
low-confidence end uncertainty sampling exists to find.

Five rules, all returning one score per row with higher meaning more worth
annotating. Three read a posterior: :func:`least_confidence` (the top
probability alone, on a scale that moves with the class count), :func:`margin`
(the top-two gap, the default) and :func:`entropy` (the whole vector, normalised
into [0, 1]). Two take a starved class instead: :func:`target_confident` fills
the class fastest, :func:`target_runner_up` picks clips on its current boundary.
Both targeting rules skew the label marginals, which is what the reliability
gate in ``suggest.train_group`` is computed against.

:func:`select_batch` assembles the batch: a candidate mask keeps annotated clips
out, ``min_frame_gap`` keeps near-duplicate windows of one track out, every tie
is broken by ``clip_id`` so the answer does not depend on row order, and a
reserved share of the slots is drawn uniformly at random from the audit holdout
(:func:`trackanno.suggest.is_audit_clip`). That stratum keeps the labelled audit
subset a random sample of the audit set, which is what makes ``audit_accuracy``
an unbiased estimate; a uniform draw carries no score, and is exempt from the
diversity rule because a draw that can be suppressed is no longer uniform.

Numpy only, no config, no file I/O: plain arrays in, plain dicts out.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "least_confidence", "margin", "entropy", "target_confident", "target_runner_up",
    "UNCERTAINTY_RULES", "TARGET_RULES", "AUDIT_RULE", "DEFAULT_AUDIT_SHARE",
    "rank_by", "starved_classes", "select_batch",
]

# How far a row's probabilities may drift from summing to 1 before it is refused:
# loose enough for float32 predict_proba output, tight enough that a row missing
# a class or built from unsoftmaxed decision scores cannot pass.
_SUM_TOL = 1e-6

# A target rule scores 0 for a row that does not qualify at all, and those rows
# are not candidates for it.
_TARGET_FLOOR = 0.0

# The recorded reason for a clip drawn at random from the audit pool, and not a
# rule a caller can ask for. It is what later splits the annotations into the
# ones a coin chose and the ones the model asked for.
AUDIT_RULE = "audit_random"

# Share of a batch reserved for that draw. Matches the default of
# `suggest.audit_fraction`, so the labelled set keeps the same audit proportion
# as the clip population.
DEFAULT_AUDIT_SHARE = 0.10


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def _as_proba(proba, clip_ids=None) -> np.ndarray:
    """The shared guard: a real probability matrix or a refusal naming the row.

    Input that is not a posterior still ranks, and the batch it produces looks
    like a good one, so it is refused here rather than renormalised. ``clip_ids``
    is optional and only names the offending row in the message.
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

    Reads only the top probability, so it cannot tell a two-way contest from mass
    smeared over classes nobody is considering. The scale moves with the class
    count - a uniform row scores ``1 - 1/n_classes`` - so the same number means
    different things in a 2-class and a 6-class project.
    """
    proba = _as_proba(proba)
    return 1.0 - proba.max(axis=1)


def margin(proba) -> np.ndarray:
    """``1 - (p1 - p2)``: how contested the top two classes are.

    Blind to the tail by construction: ``[0.5, 0.4, 0.1]`` and
    ``[0.5, 0.4, 0.05, 0.05]`` score the same. Bounded in [0, 1] at every class
    count, unlike :func:`least_confidence`, and exactly 1 for a tie. The default
    rule in :func:`select_batch`.
    """
    proba = _as_proba(proba)
    # Partition rather than a full sort: the top two are all that is read, and
    # this stays linear in the class count.
    top2 = np.partition(proba, -2, axis=1)[:, -2:]
    return 1.0 - (top2[:, 1] - top2[:, 0])


def entropy(proba) -> np.ndarray:
    """Shannon entropy over the row, divided by ``log(n_classes)``.

    Uses every probability, so its ranking is decided largely by the tail.
    Maximised by the uniform row, and normalised so the answer is in [0, 1]
    whatever the class count. ``0 * log 0`` is taken as its limit 0, so a one-hot
    row scores exactly 0 rather than NaN.
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

    Fills a class fast, but cannot move its boundary: these clips sit inside the
    region the model already assigns to the class, so the labels agree with it.

    Targeting skews the label marginals, and ``suggest.train_group`` gates the
    model on Cohen's kappa and the majority-class baseline, both computed against
    those marginals - so over-sampling a rare class can flip ``reliable`` to
    False on a model that has not changed.
    """
    proba = _as_proba(proba)
    return proba[:, int(class_index)].copy()


def target_runner_up(proba, class_index: int) -> np.ndarray:
    """``1 - (p_top - p[class])`` where the class is the model's SECOND choice, else 0.

    These clips sit on the boundary between the starved class and whatever
    outranks it, so a label either way moves the decision surface, but a good
    share come back as the other class and the count grows more slowly than under
    :func:`target_confident`.

    Rows where the class is not the runner-up score exactly 0 and are not
    eligible for this rule in :func:`select_batch`. That includes every row the
    class WINS - those are inside the class, not on its boundary, and admitting
    them would collapse this rule into :func:`target_confident`. A class tied
    with the top without winning the argmax scores the maximum, a tie being the
    boundary exactly.

    The marginals warning on :func:`target_confident` applies here too.
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

    Identical scores are routine on easy clips. Breaking those ties by position
    would make the batch depend on the order the pool arrived in; ``clip_id`` is
    unique and stable, so the same pool always yields the same order.
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

    A class absent from ``class_counts`` counts as zero rather than being
    skipped, since a class with no labels is the most starved one there is.
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

    Starved classes are served first and round-robin: each slot goes to whichever
    class is scarcest and increments its projected count, so two classes on 3 and
    5 labels against a quota of 6 get 3, 3, 4, 5 rather than one class taking
    every slot. Once every class projects to reach the quota the rest go to
    ``rule``. The projection counts a REQUEST rather than a label, so it
    over-counts, and the real counts correct it on the next batch.
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

    The midpoint of each equal share of the batch, rather than the end, so an
    annotator who stops part way through has still labelled the draws in
    proportion.
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

    ``proba`` is ``(n_clips, n_classes)`` from ``predict_proba`` over the whole
    pool, in the model's class order; ``clip_ids`` names one row each and must be
    unique; ``classes`` names one column each; ``n`` is how many clips to select.
    ``rule`` is a key of :data:`UNCERTAINTY_RULES`. ``candidates`` is an optional
    boolean mask of rows that may be selected, which is how already-annotated
    clips stay out. ``audit`` is an optional boolean mask of the audit holdout
    (:func:`trackanno.suggest.is_audit_clip`); ``audit_share`` of the slots,
    rounded to ``int(audit_share * n + 0.5)``, are then drawn uniformly at random
    from it under ``seed`` while the rest of the audit clips stay out of the
    ranked pool. Passing ``audit=None`` asserts the project has no holdout rather
    than disabling one that exists. ``class_counts`` and ``target_quota`` spend
    ranked slots on classes under the quota first, scarcest first and
    round-robin, using ``target_rule`` from :data:`TARGET_RULES`.
    ``min_frame_gap`` forbids two RANKED clips of one track less than that many
    frames apart, needing ``track_ids`` and ``frames`` (``f_start``) to say which
    those are; pairs exactly ``min_frame_gap`` apart are allowed, so setting it
    to the clip length admits adjacent non-overlapping windows.

    Returns a list of dicts in annotation order, each holding ``clip_id``,
    ``index`` (its row in ``proba``), ``rank``, ``rule``, ``score``,
    ``target_class``, ``predicted_class``, ``predicted_proba`` and ``pool_size``
    (the ranked pool for a ranked pick, the audit pool for a random draw). Every
    value is a plain ``str``/``int``/``float``, so a record serialises unchanged.
    ``rule`` is always the rule that actually chose the clip: a slot planned for
    a starved class whose rule finds no eligible clip falls back to ``rule``
    rather than picking an arbitrary clip in that class's name. A random draw
    carries no score, so its ``score`` is NaN rather than an invented 0.0.

    Refuses a ``proba`` that is not a distribution, duplicate ids, a mismatch
    between rows, ids and class names, an unknown rule, a class name with no
    column, a mask that leaves fewer candidates than the ranked slots need, and
    an audit pool too small to fill its reserved slots - topping those up from
    the ranked pool would turn the unbiased sample into a selected one.

    A SHORT batch is a result rather than an error: when the diversity rule
    forbids the rest, fewer than ``n`` records come back and the batch is never
    padded with a near-duplicate. What that rule cannot see is redundancy between
    different tracks, which would need a distance in feature space; this module
    is handed posteriors, not features.
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
            # filled either. The reserved draws still stand.
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

    Sorted by ``clip_id`` before the draw, so the sample depends on the pool's
    contents and the seed alone rather than on the order the rows arrived in.
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
    eligible however confident the model is. ``placed`` holds the RANKED rows
    already in the batch; random draws are outside this relation both ways.
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
