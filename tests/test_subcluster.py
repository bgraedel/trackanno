"""Checks for :mod:`trackanno.subcluster`, the sub-behaviour search inside one class.

Every matrix below is built by planting a structure and then asking whether the
module finds THAT, so the expected answer is known before the clusterer runs and
a disagreement is an error rather than two implementations differing. Nothing
here compares :mod:`trackanno.subcluster` against another clustering.

The partition is judged by adjusted Rand index against the planted assignment,
never by cluster id. Cluster ids are arbitrary - HDBSCAN numbers them in the
order the condensed tree hands them over - so an id-by-id comparison would fail
on a relabelling that is the same answer, and would pass on nothing extra.

Six things are worth separating:

* the module recovers a structure that is there, and the uniform rows scattered
  around it come back as noise rather than being tidied into the nearest blob;
* it does NOT invent one that is not there. A single Gaussian at the module's
  default produces no clusters at all, which is the claim that protects an
  annotator's afternoon and the easiest one to leave untested;
* stability separates the two, at matched dimension, row count and settings -
  and the same suite states where it does not, because a reproducible split of a
  cloud with no groups in it is still reproducible;
* the scaling decision is the one documented in :func:`trackanno.subcluster._scaled`:
  a single column carrying a large unit destroys the answer without it, the
  answer is unchanged by per-column units with it, and the price it charges on
  nuisance columns is measured rather than hidden;
* the representatives span the cluster instead of crowding its centre, which is
  the difference between an annotator who can see the group is two things and
  one who cannot;
* every refusal fires on its own trigger and names the offending number.
"""

from __future__ import annotations

import numpy as np
import pytest
from harness import check
from sklearn.metrics import adjusted_rand_score
from trackanno.subcluster import (
    DEFAULT_MIN_CLUSTER_SIZE, NOISE, STABLE, subcluster,
)

# The margin the stability claim is stated with, measured below rather than
# chosen: at matched settings the planted clusters sit at 0.98 and the best
# cluster any unstructured draw produces sits at 0.59.
MARGIN = 0.35


# --------------------------------------------------------------------------- #
# generators
# --------------------------------------------------------------------------- #
def planted(
    n_per: int = 120, n_columns: int = 6, sep: float = 6.0,
    n_uniform: int = 40, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Three Gaussian blobs plus uniform rows, and the assignment that made them.

    The blob centres are spread over every column, which is what a sub-behaviour
    differing in beat frequency looks like in this basis: the frequency, the
    wavelength, the wave speed and the amplitude coefficients are algebraically
    linked, so a change to one moves all of them. The case where a split hides in
    two columns out of twenty-five is a different question and has its own test.

    The uniform rows are the planted answer to "resembles nothing else": they are
    drawn across the whole occupied box, so they sit between the blobs rather
    than beyond them, and a clusterer that assigns every row to its nearest blob
    has to swallow them.

    Returns ``(features, truth)`` with ``truth`` ``-1`` on the uniform rows, and
    the rows SHUFFLED - a partition that depended on the order the rows arrived
    in would not be a property of the data.
    """
    rng = np.random.default_rng(seed)
    centres = np.array([
        sep * np.cos(2.0 * np.pi * k / 3.0 + np.arange(n_columns) * 0.7)
        for k in range(3)
    ])
    rows = [rng.normal(c, 1.0, size=(n_per, n_columns)) for c in centres]
    truth = [np.full(n_per, k) for k in range(3)]
    rows.append(rng.uniform(-sep - 3.0, sep + 3.0, size=(n_uniform, n_columns)))
    truth.append(np.full(n_uniform, NOISE))
    X, y = np.vstack(rows), np.concatenate(truth)
    order = rng.permutation(y.size)
    return X[order], y[order]


def unstructured(n_rows: int = 400, n_columns: int = 6, seed: int = 0) -> np.ndarray:
    """One Gaussian: no groups, no gaps, nothing to find."""
    return np.random.default_rng(seed).normal(0.0, 1.0, size=(n_rows, n_columns))


def named(n: int, prefix: str = "clip") -> list[str]:
    """Ids of the shape the project uses, so the answer is read the way a caller reads it."""
    return [f"{prefix}{i:04d}" for i in range(n)]


def ari(truth: np.ndarray, labels: np.ndarray, planted_only: bool = True) -> float:
    """Adjusted Rand index, by default over the rows that were planted in a blob.

    The uniform rows are excluded from the default because they are not a class:
    ARI would score them as a fourth group that has to be recovered together,
    which is the opposite of what they mean. Whether they came back as noise is a
    separate check, made separately.
    """
    keep = truth >= 0 if planted_only else np.ones(truth.size, dtype=bool)
    return float(adjusted_rand_score(truth[keep], labels[keep]))


def refused(fn, *args, **kwargs) -> str:
    """The refusal message, or "" if the call was answered."""
    try:
        fn(*args, **kwargs)
    except ValueError as exc:
        return str(exc)
    return ""


# --------------------------------------------------------------------------- #
# it finds what is there
# --------------------------------------------------------------------------- #
def test_a_planted_structure_is_recovered_as_a_partition() -> None:
    """Three blobs and a scatter of uniform rows, judged on the partition.

    The strong statement is the adjusted Rand index over the planted rows: 1.0 is
    the same partition up to relabelling, and nothing weaker would distinguish
    recovering the blobs from splitting one of them in half.

    The uniform rows carry the second statement. They are the reason the noise
    label has to survive: a clusterer that assigned every row to its nearest blob
    would still score 1.0 above, and would have told the annotator that forty
    clips resembling nothing in particular belong to one of three behaviours.
    """
    X, truth = planted()
    ids = named(truth.size)
    found = subcluster(X, ids, min_cluster_size=DEFAULT_MIN_CLUSTER_SIZE, seed=0)

    check("the three planted blobs come back as three clusters",
          len(found.clusters) == 3, str([c.size for c in found.clusters]))
    check("the recovered partition is the planted one, up to relabelling",
          ari(truth, found.labels) == 1.0, str(ari(truth, found.labels)))
    check("not one planted row was thrown out as noise",
          int(((truth >= 0) & (found.labels == NOISE)).sum()) == 0,
          str(int(((truth >= 0) & (found.labels == NOISE)).sum())))
    check("and most of the uniform rows are noise rather than joined to a blob",
          int(((truth < 0) & (found.labels == NOISE)).sum()) >= 30,
          f"{int(((truth < 0) & (found.labels == NOISE)).sum())} of 40")
    check("so the ARI over every row, noise included, is still near 1",
          ari(truth, found.labels, planted_only=False) > 0.95,
          str(ari(truth, found.labels, planted_only=False)))

    sizes = sum(c.size for c in found.clusters)
    check("the clusters and the noise account for every row exactly once",
          sizes + found.n_noise == truth.size
          and sorted(sum((c.member_ids for c in found.clusters), found.noise_ids))
          == sorted(ids),
          f"{sizes} + {found.n_noise} vs {truth.size}")
    check("member_ids agree with the label array they were read off",
          all(set(c.member_ids) == {ids[i] for i in np.flatnonzero(found.labels == c.label)}
              for c in found.clusters))
    check("the settings that produced the answer travel with it",
          found.settings["min_cluster_size"] == DEFAULT_MIN_CLUSTER_SIZE
          and found.settings["seed"] == 0 and found.settings["scaling"] == "robust",
          str(found.settings))


def test_every_planted_cluster_is_stable() -> None:
    """A real group survives being shown a different 80% of the rows.

    The number is the mean Jaccard overlap with the best-matching cluster of a
    rerun, so 1.0 means the same rows landed together in every trial. Well
    separated blobs should be near it; anything much below would mean the score
    is measuring the resampling rather than the data.
    """
    X, truth = planted()
    found = subcluster(X, named(truth.size), seed=0)
    scores = [c.stability for c in found.clusters]

    check("every planted cluster scores above 0.95",
          min(scores) > 0.95, str(np.round(scores, 4)))
    check("and every cluster clears the STABLE bar",
          all(c.stable for c in found.clusters) and found.stable_clusters == found.clusters,
          f"{np.round(scores, 4)} against {STABLE}")
    check("each score is averaged over every resample, not over a lucky few",
          all(c.n_trials == 25 for c in found.clusters),
          str([c.n_trials for c in found.clusters]))
    print(f"  --  planted stability {np.round(scores, 4)}")


# --------------------------------------------------------------------------- #
# it does not invent what is not there
# --------------------------------------------------------------------------- #
def test_unstructured_data_never_becomes_a_confident_discovery() -> None:
    """One Gaussian, at the default: nothing an annotator would ever be sent to.

    This is the claim that protects the afternoon, and it takes both gates, which
    is why it is asserted over thirty-six draws rather than one. A single
    Gaussian has no groups and no gaps, so a discovery tool that invents a class
    even one draw in six is worse than none - the corrections train the next
    model.

    Almost every draw is stopped by the first gate: ``min_cluster_size`` at its
    default will not carve groups that large out of a cloud with no gap in it, so
    every row comes back as noise and the module refuses. The draws that do get
    past it are stopped by the second: what they found dissolves completely under
    resampling. Neither number is chosen - both are printed below.
    """
    refusals, answered, stable, wrong = [], [], [], []
    for n_columns in (3, 6, 12):
        for seed in range(12):
            X = unstructured(n_columns=n_columns, seed=700 + seed)
            message = refused(subcluster, X, named(400), seed=0)
            if message:
                refusals.append(message)
                if "came back as noise" not in message:
                    wrong.append(message)
                continue
            found = subcluster(X, named(400), seed=0)
            scores = [c.stability for c in found.clusters]
            answered.append((n_columns, seed, len(scores), round(max(scores), 3)))
            stable += [c for c in found.clusters if c.stable]

    check("nearly every draw is refused outright, with no cluster to report",
          len(refusals) >= 30, f"{len(refusals)} of 36 refused, the rest {answered}")
    check("and every one of those refusals is the all-noise one",
          not wrong, str(wrong[:1]))
    check("and not one cluster found in an unstructured cloud is ever stable",
          not stable, str(stable[:1]))
    check("what does get past the first gate dissolves entirely under resampling",
          all(top < 0.25 for *_, top in answered), str(answered))

    message = refused(subcluster, unstructured(seed=700), named(400), n_resamples=0)
    check("and the refusal names the row count and the setting that produced it",
          "400" in message and f"min_cluster_size={DEFAULT_MIN_CLUSTER_SIZE}" in message,
          message)

    # Not vacuous: HDBSCAN is perfectly willing to carve this same cloud up when
    # it is allowed to call five rows a group. The default is doing the work.
    carved = [len(subcluster(unstructured(seed=700 + s), named(400),
                             min_cluster_size=5, n_resamples=0).clusters)
              for s in range(6)]
    check("the same clouds do produce clusters at min_cluster_size=5",
          all(k >= 2 for k in carved), str(carved))
    print(f"  --  {len(refusals)} of 36 draws refused; the rest scored {answered}; "
          f"min_cluster_size 5 carves the same clouds into {carved}")


def test_stability_separates_planted_from_unstructured() -> None:
    """The margin, at matched width, row count, resamples and min_cluster_size.

    The comparison is only meaningful like for like - a thinner resample or a
    larger ``min_cluster_size`` moves every score - so both sides here are 400
    rows of six columns clustered at 5 with 25 resamples, and the only difference
    between them is whether a structure was planted.

    ``min_cluster_size=5`` rather than the default because the default finds
    nothing in the unstructured cloud at all, and a score cannot be compared with
    an absence.
    """
    X, truth = planted()
    good = subcluster(X, named(truth.size), min_cluster_size=5, seed=0)
    worst = min(c.stability for c in good.clusters)

    best, tally = 0.0, []
    for seed in range(10):
        found = subcluster(unstructured(seed=600 + seed), named(400),
                           min_cluster_size=5, seed=0)
        top = max(c.stability for c in found.clusters)
        tally.append(round(top, 3))
        best = max(best, top)

    check("the planted structure is still recovered exactly at this setting",
          ari(truth, good.labels) == 1.0, str(ari(truth, good.labels)))
    check("no cluster found in an unstructured cloud clears the STABLE bar",
          best < STABLE, f"best {best:.4f} over {tally}")
    check(f"and the worst planted cluster beats the best unstructured one by {MARGIN}",
          worst - best > MARGIN, f"{worst:.4f} - {best:.4f} = {worst - best:.4f}")
    print(f"  --  planted worst {worst:.4f}, unstructured best {best:.4f} over {tally}")


def test_stability_does_not_certify_a_split_of_a_smooth_cloud() -> None:
    """The limitation, asserted rather than left for someone to discover.

    Resampling varies which rows are present; it does not vary the clusterer. So
    a cut that a smooth unimodal cloud provokes by its shape alone is drawn in
    the same place trial after trial and scores highly - the members really do
    keep landing together, and they are still not a group.

    Two dimensions at ``min_cluster_size=5``, both well below anything this
    module defaults to, are enough to produce one. The point of the check is that
    this suite states the number instead of only stating the good one: a caller
    who reads :attr:`Cluster.stability` as "the group is real" is reading it
    wrong, and :func:`trackanno.subcluster.subcluster` says so where it is
    documented.
    """
    found = subcluster(unstructured(n_rows=300, n_columns=2, seed=406), named(300),
                       min_cluster_size=5, seed=0)
    scores = [c.stability for c in found.clusters]

    check("a plain 2-column Gaussian is cut into groups at this setting",
          len(found.clusters) >= 2, str(len(found.clusters)))
    check("and those groups clear the STABLE bar, some of them well past it",
          max(scores) > 0.9, f"{np.round(scores, 4)} against {STABLE}")
    check("so stability alone would have sent an annotator to label a cloud",
          len(found.stable_clusters) >= 2, str(np.round(scores, 4)))
    check("which is why the module's own default refuses this same cloud outright",
          "came back as noise" in refused(
              subcluster, unstructured(n_rows=300, n_columns=2, seed=406), named(300)),
          refused(subcluster, unstructured(n_rows=300, n_columns=2, seed=406), named(300)))
    print(f"  --  unstructured cloud, deliberately mis-set: {np.round(scores, 4)}")


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #
def test_the_same_seed_gives_the_same_answer() -> None:
    """A discovery that moves between runs is not a discovery.

    Two statements, and the second is the one with content. The partition must
    not depend on the seed at all - nothing in the clustering draws a random
    number, so a seed-dependent partition would mean an accidental source of
    randomness had crept in. The stability score does depend on the seed, since
    the seed IS the resampling, but only in its last decimals: a score that moved
    by more than the margin it is compared against would be a measurement of the
    draw rather than of the data.
    """
    X, truth = planted()
    ids = named(truth.size)
    base = subcluster(X, ids, seed=0)
    again = subcluster(X, ids, seed=0)

    check("the same seed reproduces the labels exactly",
          bool((base.labels == again.labels).all()))
    check("and reproduces every stability score bit for bit",
          [c.stability for c in base.clusters] == [c.stability for c in again.clusters],
          str([c.stability for c in again.clusters]))
    check("and reproduces the medoid and the whole panel",
          [c.representative_ids for c in base.clusters]
          == [c.representative_ids for c in again.clusters])

    moved = []
    for seed in (1, 2, 3, 7, 99):
        other = subcluster(X, ids, seed=seed)
        if not (other.labels == base.labels).all():
            moved.append(seed)
        for a, b in zip(base.clusters, other.clusters):
            moved.append((seed, abs(a.stability - b.stability))
                         if abs(a.stability - b.stability) > 0.05 else None)
    check("a different seed leaves the recovered partition untouched",
          not [m for m in moved if m is not None],
          str([m for m in moved if m is not None]))

    spread = np.ptp([[c.stability for c in subcluster(X, ids, seed=s).clusters]
                     for s in (0, 1, 2, 3, 7, 99)], axis=0)
    check("and moves each stability score by under 0.01",
          bool((spread < 0.01).all()), str(np.round(spread, 4)))
    print(f"  --  seed-to-seed spread of the stability score {np.round(spread, 4)}")


def test_row_order_moves_only_the_noise_boundary() -> None:
    """Rows arrive in whatever order a table was written in. What may move, and what may not.

    Cluster IDS are allowed to move - they are handed out in the order the
    condensed tree yields them - so every claim here is about the partition, read
    through the ids that came back rather than off the label array.

    The grouping must not move at all: the rows that both runs place in a cluster
    have to be partitioned identically, or the answer is a property of a sort
    order. A row on the boundary between a cluster and noise is a different
    matter and does move - HDBSCAN is deterministic without being exactly
    permutation-invariant - so the honest claim is that it is a handful of rows
    and no more. The number is printed rather than assumed; the module says the
    same thing where the clusterer is called.
    """
    X, truth = planted()
    ids = named(truth.size)
    base = subcluster(X, ids, n_resamples=0)
    row = {name: i for i, name in enumerate(ids)}

    crossed, sizes = [], []
    for trial in range(5):
        order = np.random.default_rng(trial).permutation(truth.size)
        shuffled = subcluster(X[order], [ids[i] for i in order], n_resamples=0)
        back = np.empty_like(base.labels)
        back[[row[name] for name in shuffled.ids]] = shuffled.labels
        clustered = (back >= 0) & (base.labels >= 0)
        check("the rows both runs cluster are partitioned identically",
              adjusted_rand_score(base.labels[clustered], back[clustered]) == 1.0,
              str(adjusted_rand_score(base.labels[clustered], back[clustered])))
        crossed.append(int(((back >= 0) != (base.labels >= 0)).sum()))
        sizes.append(sorted(c.size for c in shuffled.clusters))

    check("the same number of clusters comes back every time",
          all(len(s) == len(base.clusters) for s in sizes), str(sizes))
    check("and at most a handful of rows cross between a cluster and noise",
          max(crossed) <= 5, f"{crossed} of {truth.size} rows")
    print(f"  --  {crossed} of {truth.size} rows changed side under five permutations")


# --------------------------------------------------------------------------- #
# the scaling decision
# --------------------------------------------------------------------------- #
def test_scaling_survives_a_column_carrying_a_large_unit() -> None:
    """One nuisance column in pixels is enough to erase the answer without scaling.

    The construction is the real one: the same three blobs, except that the last
    column carries none of the structure and is multiplied by 400, standing in
    for ``wave_wavelength_px`` beside ``wave_static_cos``. A Euclidean distance
    over the raw columns is then that one column and nothing else.

    Both directions are asserted, because only the pair is evidence: with the
    scaling the answer must be identical whatever units the columns are in, and
    without it the same matrix must fall apart. One alone would pass on a module
    that ignored the column.
    """
    X, truth = planted()
    ids = named(truth.size)
    X[:, 5] = np.random.default_rng(1).normal(0.0, 1.0, truth.size)   # carries nothing
    inflated = X * np.array([1.0, 1.0, 1.0, 1.0, 1.0, 400.0])

    scaled = subcluster(inflated, ids, n_resamples=0)
    plain = subcluster(X, ids, n_resamples=0)
    unscaled = subcluster(inflated, ids, scaling="none", n_resamples=0)

    check("robust scaling recovers the planted partition through the large unit",
          ari(truth, scaled.labels) > 0.95, str(ari(truth, scaled.labels)))
    check("and gives the identical answer with and without that unit, "
          "so the scaling is what makes the units irrelevant",
          bool((scaled.labels == plain.labels).all()),
          str(adjusted_rand_score(scaled.labels, plain.labels)))
    check("without scaling the one large column destroys the partition entirely, "
          "leaving an agreement no better than chance",
          abs(ari(truth, unscaled.labels)) < 0.01, str(ari(truth, unscaled.labels)))
    check("the divisor actually used is reported, and points at the column it was",
          np.argmax(scaled.column_scale) == 5, str(np.round(scaled.column_scale, 3)))
    check("and it absorbs the unit exactly, an interquartile range being "
          "homogeneous: 400x the data is 400x the divisor",
          abs(scaled.column_scale[5] / plain.column_scale[5] - 400.0) < 1e-9
          and np.allclose(scaled.column_scale[:5], plain.column_scale[:5]),
          str(scaled.column_scale[5] / plain.column_scale[5]))
    check("scaling='none' says so in the result and divides by nothing",
          unscaled.scaling == "none" and bool((unscaled.column_scale == 1.0).all()))
    print(f"  --  ARI {ari(truth, scaled.labels):.3f} scaled, "
          f"{ari(truth, unscaled.labels):.3f} unscaled")


def test_per_column_scaling_charges_for_nuisance_columns() -> None:
    """The documented price of the scaling choice, measured rather than asserted away.

    Per-column scaling divides a column by its own spread, and a column that is
    bimodal BECAUSE of the split being hunted has a spread that spans the gap. So
    the informative column is shrunk onto the same footing as a column carrying
    only noise, and four noise columns then outweigh the two that know the
    answer.

    The mitigation is not another scaling. It is the caller passing the columns
    the split is expected to live in, which is exactly what
    :func:`trackanno.subcluster.subcluster` tells them to do when it refuses an
    all-noise result - so it is worth knowing that it works.
    """
    rng = np.random.default_rng(5)
    X, truth = planted(sep=8.0)
    X[:, 2:] = rng.normal(0.0, 1.0, size=(truth.size, 4))     # four nuisance columns
    ids = named(truth.size)

    wide = subcluster(X, ids, n_resamples=0)
    narrow = subcluster(X[:, :2], ids, n_resamples=0)

    check("the split is lost among four columns that do not carry it",
          ari(truth, wide.labels) < 0.2, str(ari(truth, wide.labels)))
    check("badly enough that two groups are reported where three were planted",
          len(wide.clusters) == 2, str([c.size for c in wide.clusters]))
    check("and is recovered exactly from the two columns that do",
          ari(truth, narrow.labels) == 1.0 and len(narrow.clusters) == 3,
          str(ari(truth, narrow.labels)))
    print(f"  --  ARI {ari(truth, wide.labels):.3f} on six columns, "
          f"{ari(truth, narrow.labels):.3f} on the two informative ones")


# --------------------------------------------------------------------------- #
# representatives
# --------------------------------------------------------------------------- #
def elongated(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A long thin cluster beside a compact one, in commensurate columns.

    The point of the shape: the annotator's question about the long cluster is
    "is this one behaviour or two?", and the only evidence that could answer it
    is how far the group reaches. A panel drawn from the middle cannot show that
    however many clips it holds.

    Columns are already on a common scale here and the test asks for
    ``scaling="none"``, so distances in the returned space are distances in the
    array below and the claims can be read off column 0 directly.
    """
    rng = np.random.default_rng(seed)
    bar = np.hstack([rng.normal(0.0, 4.0, size=(200, 1)),
                     rng.normal(0.0, 0.2, size=(200, 3))])
    blob = rng.normal([0.0, 30.0, 0.0, 0.0], 0.5, size=(150, 4))
    return (np.vstack([bar, blob]),
            np.concatenate([np.zeros(200, dtype=int), np.ones(150, dtype=int)]))


def test_representatives_span_the_cluster_rather_than_crowd_its_centre() -> None:
    """The panel covers the group's extent; the nearest neighbours of the medoid do not.

    Measured on the long thin cluster, along the axis it is long in. The eight
    members nearest the medoid are what a naive panel would show, so they are
    computed here and compared: if the two spans were similar the max-min
    traversal would be doing nothing.
    """
    X, truth = elongated()
    ids = named(truth.size, "elong")
    found = subcluster(X, ids, min_cluster_size=15, scaling="none",
                       n_resamples=0, n_representatives=8)
    row = {name: i for i, name in enumerate(ids)}
    bar = max(found.clusters, key=lambda c: np.ptp(X[[row[m] for m in c.member_ids], 0]))

    member = np.array([row[m] for m in bar.member_ids])
    panel = np.array([row[m] for m in bar.representative_ids])
    medoid = row[bar.medoid_id]
    nearest = member[np.argsort(np.linalg.norm(X[member] - X[medoid], axis=1),
                                kind="stable")[:8]]
    extent = float(np.ptp(X[member, 0]))

    check("the long cluster is found in one piece rather than in fragments",
          bar.size > 100 and len(found.clusters) == 2,
          f"{[c.size for c in found.clusters]}, noise {found.n_noise}")
    check("the panel reaches both ends of the cluster",
          np.ptp(X[panel, 0]) >= 0.99 * extent,
          f"{np.ptp(X[panel, 0]):.3f} of {extent:.3f}")
    check("where the eight nearest the medoid cover under a tenth of it",
          np.ptp(X[nearest, 0]) < 0.1 * extent,
          f"{np.ptp(X[nearest, 0]):.3f} of {extent:.3f}")
    check("so the panel is not the nearest neighbours under another name",
          len(set(bar.representative_ids) & {ids[i] for i in nearest}) <= 1,
          str(sorted(set(bar.representative_ids) & {ids[i] for i in nearest})))
    print(f"  --  panel spans {np.ptp(X[panel, 0]) / extent:.3f} of the cluster, "
          f"the nearest eight span {np.ptp(X[nearest, 0]) / extent:.3f}")


def test_the_medoid_is_a_member_and_the_most_central_one() -> None:
    """The defining property, checked against every alternative member.

    A medoid rather than a mean because the answer has to be a clip somebody can
    watch, and the claim that makes it a medoid is that no other member has a
    smaller total distance to the rest. That is checked directly here, not
    against a second search for one.
    """
    X, truth = elongated()
    ids = named(truth.size, "elong")
    found = subcluster(X, ids, min_cluster_size=15, scaling="none",
                       n_resamples=0, n_representatives=6)
    row = {name: i for i, name in enumerate(ids)}

    for cluster in found.clusters:
        member = np.array([row[m] for m in cluster.member_ids])
        totals = np.linalg.norm(X[member][:, None, :] - X[member][None, :, :],
                                axis=2).sum(axis=1)
        chosen = int(np.flatnonzero(member == row[cluster.medoid_id])[0])
        check(f"cluster {cluster.label}: no member is more central than the medoid",
              totals[chosen] <= totals.min() + 1e-9,
              f"{totals[chosen]} against {totals.min()}")
        check(f"cluster {cluster.label}: the medoid leads the panel",
              cluster.representative_ids[0] == cluster.medoid_id,
              str(cluster.representative_ids[:2]))
        check(f"cluster {cluster.label}: the panel is distinct members of it",
              len(set(cluster.representative_ids)) == 6
              and set(cluster.representative_ids) <= set(cluster.member_ids))

    check("a smaller panel is a prefix of a larger one, so asking for fewer "
          "shows the same clips",
          all(a.representative_ids[:3] == b.representative_ids
              for a, b in zip(found.clusters,
                              subcluster(X, ids, min_cluster_size=15, scaling="none",
                                         n_resamples=0, n_representatives=3).clusters)))


# --------------------------------------------------------------------------- #
# the unmeasured score
# --------------------------------------------------------------------------- #
def test_an_unmeasured_stability_is_nan_and_never_counts_as_stable() -> None:
    """"Nobody looked" is not "the cluster is weak", and neither is it "fine".

    ``n_resamples=0`` is the fast path a caller reaches for while exploring, and
    the danger is that its output looks like the measured one. NaN says the
    measurement is absent - as it does everywhere else in this codebase - and
    :attr:`Cluster.stable` is False on it, so the fast path cannot leak a cluster
    past a stability gate.
    """
    X, truth = planted()
    quick = subcluster(X, named(truth.size), n_resamples=0)

    check("skipping the resampling leaves every stability NaN",
          all(np.isnan(c.stability) for c in quick.clusters),
          str([c.stability for c in quick.clusters]))
    check("with a trial count of zero beside it, so the NaN is explained",
          all(c.n_trials == 0 for c in quick.clusters))
    check("and none of them is stable, since NaN never clears a bar",
          not any(c.stable for c in quick.clusters) and quick.stable_clusters == [])
    check("the partition itself is the same one the measured call returns",
          bool((quick.labels == subcluster(X, named(truth.size), seed=0).labels).all()),
          "the resampling must not feed back into the clustering")


# --------------------------------------------------------------------------- #
# guards
# --------------------------------------------------------------------------- #
def test_a_misuse_is_refused_and_the_message_names_the_number() -> None:
    """Every refusal, its trigger, and the value it has to put in the message.

    A refusal that does not name the offending number is a refusal the caller has
    to reproduce under a debugger, so each of these asserts on the message and
    not only on the exception.
    """
    X, truth = planted()
    ids = named(truth.size)

    holed = X.copy()
    holed[17, 3] = np.nan
    message = refused(subcluster, holed, ids)
    check("a non-finite value is refused rather than handed to the clusterer",
          "row 17" in message and "column 3" in message and ids[17] in message, message)
    check("and the message says why imputing it would be worse than stopping",
          "median" in message and "manufacture" in message, message)
    check("an infinity is refused on the same path",
          "non-finite" in refused(subcluster, np.where(np.arange(X.size).reshape(X.shape) == 4,
                                                       np.inf, X), ids))

    check("an id list that does not match the rows is refused, naming both counts",
          all(part in refused(subcluster, X, ids[:-1])
              for part in ("399", "400", "POSITION")),
          refused(subcluster, X, ids[:-1]))
    duplicate = list(ids)
    duplicate[9] = duplicate[8]
    check("a repeated id is refused, naming the id",
          repr(ids[8]) in refused(subcluster, X, duplicate),
          refused(subcluster, X, duplicate))

    check("a matrix that is not two-dimensional is refused",
          "shape" in refused(subcluster, X[:, 0], ids))
    check("a matrix with no columns is refused",
          "no columns" in refused(subcluster, np.zeros((400, 0)), ids))

    check("too few rows to hold two groups is refused, naming both numbers",
          all(part in refused(subcluster, X[:20], ids[:20], min_cluster_size=15)
              for part in ("20", "15")),
          refused(subcluster, X[:20], ids[:20], min_cluster_size=15))
    check("and the message says what to lower it to",
          "lower min_cluster_size" in refused(subcluster, X[:20], ids[:20],
                                              min_cluster_size=15))
    thin = refused(subcluster, X[:34], ids[:34], min_cluster_size=15, subsample_frac=0.5)
    check("a resample too thin to hold two groups is refused separately, "
          "naming the row count the fraction leaves",
          "stability trial" in thin and "17 rows" in thin, thin)
    check("and that guard stops firing once the resampling is switched off, "
          "so it is about the stability score and not about the clustering",
          "stability trial" not in refused(subcluster, X[:34], ids[:34],
                                           min_cluster_size=15, subsample_frac=0.5,
                                           n_resamples=0))

    check("an all-noise result is refused, naming the rows and the setting",
          all(part in refused(subcluster, unstructured(seed=700), named(400))
              for part in ("400", "min_cluster_size=15", "min_samples=15")),
          refused(subcluster, unstructured(seed=700), named(400)))

    check("a min_cluster_size below 2 is refused",
          "at least 2" in refused(subcluster, X, ids, min_cluster_size=1))
    check("a subsample fraction outside (0, 1] is refused",
          all(refused(subcluster, X, ids, subsample_frac=f) for f in (0.0, -0.5, 1.5)))
    check("a negative resample count is refused",
          "negative" in refused(subcluster, X, ids, n_resamples=-1))
    check("an empty panel is refused",
          "at least 1" in refused(subcluster, X, ids, n_representatives=0))
    check("an unknown scaling is refused rather than silently ignored",
          "scaling must be one of" in refused(subcluster, X, ids, scaling="whiten"),
          refused(subcluster, X, ids, scaling="whiten"))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
