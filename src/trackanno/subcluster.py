"""Is one label actually several behaviours, and which clips would show that?

Takes the feature rows of the clips carrying a single label, asks whether they
fall into groups, and names the clips a human should watch to decide what those
groups are. The clusterer is :class:`sklearn.cluster.HDBSCAN`, run in the linear
feature basis the caller passes rather than in an embedding, so a distance
between two rows is a distance between two measurements.

Noise is an answer, not a failure. HDBSCAN's ``-1`` says "this clip resembles
nothing else here", and noise rows keep their ids in
:attr:`SubClusters.noise_ids` rather than being tidied into a nearest cluster.

Two things stand between a caller and a partition of unstructured data, and they
catch different failures. :data:`DEFAULT_MIN_CLUSTER_SIZE`, not the stability
score, is what keeps a gapless cloud from being carved up in the first place.
:attr:`Cluster.stability` then asks whether a group that WAS found survives
being shown a different 80% of the rows, which catches a group held together by
a handful of clips.

Everything here takes plain arrays. No :class:`~trackanno.config.Config`, no
file paths, no table: the ids are whatever the caller uses to name a row and
come back untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics import pairwise_distances_chunked

__all__ = [
    "Cluster", "SubClusters", "subcluster",
    "NOISE", "STABLE", "DEFAULT_MIN_CLUSTER_SIZE", "SCALINGS",
]

# HDBSCAN's own label for a row in no cluster. Named rather than spelled -1 at
# every use, because -1 also means "undefined" in `trackanno.bouts` and the two
# are different statements about a row.
NOISE = -1

# The bar a cluster has to clear before a human is sent to annotate it: Hennig's
# rule of thumb for bootstrap cluster stability (Hennig 2007, "Cluster-wise
# assessment of cluster stability"). A convention, not a measurement of this
# data.
STABLE = 0.75

# The smallest group that counts as a discovery. A sub-cluster is only useful if
# it can become a LABEL, and the suggester refuses a class with fewer than
# `suggest.min_per_class` examples while cross-validating in folds GROUPED by
# (movie, track), so the examples have to come from several animals.
#
# It is also the module's first defence against inventing structure, and a
# stronger one than the stability score: excess-of-mass selection will not carve
# groups this large out of a cloud with no gap in it. Lower it knowingly.
DEFAULT_MIN_CLUSTER_SIZE = 15

SCALINGS = ("robust", "none")


# --------------------------------------------------------------------------- #
# what comes back
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Cluster:
    """One discovered group, and the evidence that it is a group."""

    label: int
    size: int
    # Mean maximum Jaccard overlap under resampling; NaN when nobody measured it
    # (`n_resamples=0`). NaN is not a low score: see `Cluster.stable`.
    stability: float
    # Resamples that could say anything about this cluster at all. A score
    # averaged over two trials is not the same claim as one averaged over
    # twenty-five, and the denominator has to travel with the number.
    n_trials: int
    member_ids: list
    medoid_id: object
    # The medoid first, then a max-min spread over the rest. See `subcluster`.
    representative_ids: list

    @property
    def stable(self) -> bool:
        """Whether this cluster clears :data:`STABLE`.

        NaN is False: an unmeasured cluster is not a stable one.
        """
        return bool(self.stability >= STABLE)


@dataclass(frozen=True)
class SubClusters:
    """The answer for one class: a partition of its rows, with per-group evidence."""

    # (n_rows,) int, aligned with `ids`. NOISE for a row in no cluster.
    labels: np.ndarray
    ids: list
    clusters: list[Cluster]
    noise_ids: list
    scaling: str
    # The per-column divisor actually used. A column whose scale is far off its
    # neighbours' is the column that decided the answer.
    column_scale: np.ndarray
    # Every number that produced this. Two runs are not comparable unless these
    # match.
    settings: dict = field(default_factory=dict)

    @property
    def n_noise(self) -> int:
        return int((self.labels == NOISE).sum())

    @property
    def noise_frac(self) -> float:
        return self.n_noise / len(self.labels) if len(self.labels) else float("nan")

    @property
    def stable_clusters(self) -> list[Cluster]:
        """The clusters worth an annotator's time, in the order they were found."""
        return [c for c in self.clusters if c.stable]


# --------------------------------------------------------------------------- #
# scaling
# --------------------------------------------------------------------------- #
def _robust_scale(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column median and interquartile range, with the degenerate cases named.

    Returns ``(centre, scale)``, both ``(n_features,)``. The IQR rather than the
    standard deviation because these columns have long tails by construction, and
    one extreme row would otherwise squash every other row into a point.

    Two degenerate cases, both explicit because a divisor of zero would turn
    every distance into NaN:

    * IQR zero but the column varies, as in a near-binary column where one value
      holds more than half the rows: the full range is used instead.
    * a constant column: scale 1.0, exact rather than a fallback, since every
      scaled value is then 0 whatever divisor is chosen.
    """
    centre = np.median(X, axis=0)
    q1, q3 = np.percentile(X, [25, 75], axis=0)
    scale = q3 - q1
    span = X.max(axis=0) - X.min(axis=0)
    scale = np.where(scale > 0, scale, span)
    return centre, np.where(scale > 0, scale, 1.0)


def _scaled(X: np.ndarray, scaling: str) -> tuple[np.ndarray, np.ndarray]:
    """Put the columns on a common footing, or deliberately decline to.

    Returns ``(Z, column_scale)``, ``Z`` always a fresh array. ``"robust"``, the
    default, divides each column by its own interquartile range
    (:func:`_robust_scale`); these columns span pixels, ``[-1, 1]`` and
    ``{-1, +1}``, so an unscaled Euclidean distance would be the widest column
    and nothing else. ``"none"`` is for a caller whose matrix is already
    commensurate, and has to be asked for. Centring changes no Euclidean
    distance and is here only to keep the subtraction well conditioned.

    The price: a column that is bimodal BECAUSE of the split being hunted has an
    interquartile range spanning the gap, so scaling shrinks the column carrying
    the signal onto the same footing as one carrying only noise. The mitigation
    is the caller's - pass the columns the split is expected to live in rather
    than every column there is.
    """
    if scaling == "none":
        return np.array(X, dtype=float), np.ones(X.shape[1])
    centre, scale = _robust_scale(X)
    return (X - centre) / scale, scale


# --------------------------------------------------------------------------- #
# the clustering
# --------------------------------------------------------------------------- #
def _fit(Z: np.ndarray, min_cluster_size: int, min_samples: int) -> np.ndarray:
    """One HDBSCAN run. Deterministic: nothing in it draws a random number.

    Deterministic but not quite permutation-invariant. Reordering the rows leaves
    the groups identical while a row on the boundary between a cluster and noise
    can cross it, so membership of a boundary clip is not a fact to build on.

    ``allow_single_cluster`` is left off: "all of it, one group" is the null
    answer to the question this module asks, and with the flag off it arrives as
    an all-noise result that :func:`subcluster` refuses by name.

    ``copy=False`` is safe because every matrix reaching here was allocated by
    :func:`_scaled`, so the caller's array is never overwritten.
    """
    return HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples),
        copy=False,
    ).fit(Z).labels_


def _max_jaccard(reference: np.ndarray, found: np.ndarray, n_reference: int) -> np.ndarray:
    """Best Jaccard overlap of each reference cluster with any cluster in ``found``.

    Both arrays label the SAME rows - the rows a resample drew - with ``NOISE``
    for a row in no cluster. Returns ``(n_reference,)``.

    Noise is never a candidate to match against, so a reference cluster whose
    members all came back as noise scores 0: the group dissolved.
    """
    n_found = int(found.max()) + 1 if found.size and found.max() >= 0 else 0
    reference_size = np.bincount(
        reference[reference >= 0], minlength=n_reference).astype(float)
    if n_found == 0:
        return np.zeros(n_reference)

    both = (reference >= 0) & (found >= 0)
    shared = np.zeros((n_reference, n_found))
    np.add.at(shared, (reference[both], found[both]), 1.0)
    found_size = np.bincount(found[found >= 0], minlength=n_found).astype(float)
    union = reference_size[:, None] + found_size[None, :] - shared
    return np.where(union > 0, shared / np.where(union > 0, union, 1.0), 0.0).max(axis=1)


def _stability(
    X: np.ndarray, labels: np.ndarray, n_clusters: int, *,
    min_cluster_size: int, min_samples: int, scaling: str,
    seed: int, n_resamples: int, subsample_frac: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resampling stability per cluster: ``(score, n_trials)``, both ``(n_clusters,)``.

    Each trial draws ``subsample_frac`` of the rows without replacement, reruns
    the whole procedure on them including the scaling, and takes each reference
    cluster's Jaccard overlap with the best-matching cluster of the rerun; the
    score is the mean over trials. This is Hennig's bootstrap cluster stability
    with subsampling in place of the bootstrap, since a bootstrap's duplicated
    rows are coincident points a density-based clusterer reads as structure.

    A trial that drew fewer than two of a cluster's members says nothing about it
    and is skipped rather than scored as a failure, which is why the trial count
    comes back beside the score.

    The drawn indices keep their original order, so a shuffle of the rows cannot
    move the score. Each trial's generator is seeded from ``(seed, trial)``, so
    raising ``n_resamples`` extends the answer rather than replacing it.
    """
    n_rows = X.shape[0]
    drawn_size = max(2, int(round(subsample_frac * n_rows)))
    total = np.zeros(n_clusters)
    trials = np.zeros(n_clusters, dtype=np.int64)

    for trial in range(n_resamples):
        rng = np.random.default_rng([seed, trial])
        drawn = np.sort(rng.choice(n_rows, size=drawn_size, replace=False))
        reference = labels[drawn]
        found = _fit(_scaled(X[drawn], scaling)[0], min_cluster_size, min_samples)
        counted = np.bincount(
            reference[reference >= 0], minlength=n_clusters) >= 2
        total[counted] += _max_jaccard(reference, found, n_clusters)[counted]
        trials += counted

    with np.errstate(invalid="ignore"):
        score = np.where(trials > 0, total / np.where(trials > 0, trials, 1), np.nan)
    return score, trials


# --------------------------------------------------------------------------- #
# representatives
# --------------------------------------------------------------------------- #
def _medoid(Z: np.ndarray) -> int:
    """Row minimising the total distance to every other row. Ties to the lowest index.

    The medoid rather than the mean, because the answer has to be a CLIP the
    annotator can watch. Chunked, so the ``m x m`` distance matrix is never held.
    """
    if Z.shape[0] == 1:
        return 0
    totals = np.concatenate(list(
        pairwise_distances_chunked(Z, reduce_func=lambda block, start: block.sum(axis=1))
    ))
    return int(np.argmin(totals))


def _spread(Z: np.ndarray, start: int, k: int) -> list[int]:
    """``start``, then repeatedly whichever row is furthest from everything chosen.

    A max-min traversal rather than the ``k`` rows nearest the medoid, so the
    panel spans the cluster's extent: that extent is the evidence that could tell
    an annotator the group is really two things wearing one label.

    Stops early rather than returning a row at distance 0 from one already
    chosen. Ties go to the lowest index.
    """
    chosen = [int(start)]
    if Z.shape[0] == 1:
        return chosen
    nearest = np.linalg.norm(Z - Z[start], axis=1)
    while len(chosen) < min(int(k), Z.shape[0]):
        pick = int(np.argmax(nearest))
        if nearest[pick] <= 0.0:
            break
        chosen.append(pick)
        nearest = np.minimum(nearest, np.linalg.norm(Z - Z[pick], axis=1))
    return chosen


# --------------------------------------------------------------------------- #
# the entry point
# --------------------------------------------------------------------------- #
def subcluster(
    features, ids, *,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    min_samples: int | None = None,
    scaling: str = "robust",
    seed: int = 0,
    n_resamples: int = 25,
    subsample_frac: float = 0.8,
    n_representatives: int = 8,
) -> SubClusters:
    """Split one class's feature rows into candidate sub-behaviours, with evidence.

    ``features`` is ``(n_rows, n_features)`` float and must be all finite; ``ids``
    names one row each, matched by POSITION, and comes back untouched in
    ``member_ids``, ``medoid_id`` and ``representative_ids``. ``min_cluster_size``
    is the smallest group that counts as a discovery
    (:data:`DEFAULT_MIN_CLUSTER_SIZE`), ``min_samples`` how conservative the
    density estimate is (``None`` means ``min_cluster_size``; raising it moves
    more rows into noise), ``scaling`` ``"robust"`` or ``"none"``
    (:func:`_scaled`), ``seed`` the module's only source of randomness,
    ``n_resamples`` the trials behind the stability score (``0`` skips the
    measurement and every ``stability`` is NaN, "nobody looked" rather than
    "unstable"), ``subsample_frac`` the share of rows a trial draws and
    ``n_representatives`` the rows named per cluster, medoid included. Returns a
    :class:`SubClusters` whose ``labels`` is aligned with ``ids`` and carries
    :data:`NOISE` for a row in no cluster.

    The stability score (:func:`_stability`) is evidence that a grouping is a
    property of the data rather than of which rows happened to be labelled. It is
    not evidence that a group exists at all: resampling varies which rows are
    present, not the clusterer, so a split a smooth cloud provokes by its shape
    alone recurs trial after trial and scores highly. ``min_cluster_size``, not
    this score, is what keeps a gapless cloud from being carved up. Nor is it
    evidence that a cluster is a BEHAVIOUR, since clips from one movie or animal
    cluster tightly on the recording and are perfectly stable; nor a p-value; nor
    a statement about clips absent from ``features``. Scores are comparable only
    at the same ``subsample_frac`` and ``min_cluster_size``.

    Refused: a non-finite value anywhere, because imputing would put every clip
    nobody could measure at the column median and manufacture a dense, perfectly
    stable group out of the rows we know least about, so the caller must drop the
    row or the column and decide which; ``ids`` not matching the rows or
    repeating, which leaves it ambiguous which clip to show; too few rows to hold
    two groups of ``min_cluster_size``, in the data or in a RESAMPLE, where every
    trial would be forced to answer "one cluster"; and every row classed as
    noise, which is a real answer but not one returnable as a list of clusters.
    """
    X = np.asarray(features, dtype=float)
    ids = list(ids)
    if X.ndim != 2:
        raise ValueError(f"features must be (n_rows, n_features), got shape {X.shape}")
    n_rows, n_features = X.shape
    if n_features == 0:
        raise ValueError(
            f"features has {n_rows} row(s) and no columns; there is no space to "
            "measure a distance in"
        )
    if len(ids) != n_rows:
        raise ValueError(
            f"{len(ids)} id(s) for {n_rows} feature row(s); they are matched by "
            "POSITION, so a mismatch means the ids belong to different clips"
        )
    if len(set(ids)) != len(ids):
        seen, repeated = set(), None
        for name in ids:
            if name in seen:
                repeated = name
                break
            seen.add(name)
        raise ValueError(
            f"id {repeated!r} appears more than once among the {n_rows} row(s); "
            "clusters are reported as ids, so a duplicate leaves the caller unable "
            "to say which clip to show"
        )
    if not np.isfinite(X).all():
        bad = np.argwhere(~np.isfinite(X))
        row, column = int(bad[0][0]), int(bad[0][1])
        raise ValueError(
            f"features holds {len(bad)} non-finite value(s), the first at row {row} "
            f"(id {ids[row]!r}), column {column}; NaN means 'undefined' here and "
            "HDBSCAN cannot read it, so drop that row or that column and say which - "
            "filling it with the column median would place every clip nobody could "
            "measure at the centre of the cloud and manufacture a dense cluster out "
            "of the rows we know least about"
        )
    if scaling not in SCALINGS:
        raise ValueError(f"scaling must be one of {SCALINGS}, got {scaling!r}")
    floor = int(min_cluster_size)
    if floor < 2:
        raise ValueError(
            f"min_cluster_size must be at least 2, got {floor}; a group of one is "
            "the noise label, which HDBSCAN already reports"
        )
    if not 0.0 < subsample_frac <= 1.0:
        raise ValueError(
            f"subsample_frac must be in (0, 1], got {subsample_frac}"
        )
    if int(n_resamples) < 0:
        raise ValueError(f"n_resamples must not be negative, got {n_resamples}")
    if int(n_representatives) < 1:
        raise ValueError(
            f"n_representatives must be at least 1, got {n_representatives}; the "
            "medoid alone is the smallest useful panel"
        )
    if n_rows < 2 * floor:
        raise ValueError(
            f"{n_rows} row(s) cannot hold two groups of min_cluster_size={floor}, so "
            f"there is no sub-structure that could be found; label more clips of this "
            f"class, or lower min_cluster_size below {n_rows // 2 + 1}"
        )
    trials = int(n_resamples)
    drawn_size = max(2, int(round(subsample_frac * n_rows)))
    if trials and drawn_size < 2 * floor:
        raise ValueError(
            f"a {subsample_frac:g} resample of {n_rows} rows is {drawn_size} rows, "
            f"which cannot hold two groups of min_cluster_size={floor}: every "
            f"stability trial would be forced to answer 'one cluster' and the score "
            f"would measure the arithmetic rather than the data. Needs at least "
            f"{int(np.ceil(2 * floor / subsample_frac))} rows, a larger "
            f"subsample_frac, or a smaller min_cluster_size"
        )

    neighbours = floor if min_samples is None else int(min_samples)
    Z, column_scale = _scaled(X, scaling)
    labels = _fit(Z, floor, neighbours)
    n_clusters = int(labels.max()) + 1 if labels.max() >= 0 else 0
    if n_clusters == 0:
        raise ValueError(
            f"all {n_rows} row(s) came back as noise at min_cluster_size={floor}, "
            f"min_samples={neighbours}, scaling={scaling!r}: HDBSCAN found no group "
            "at all, so there is nothing to send an annotator to look at. Either this "
            "class is one behaviour in this feature basis, or the smallest group "
            "present is under "
            f"{floor} rows - lower min_cluster_size, or narrow the feature columns to "
            "the ones the split is expected to live in"
        )

    score, n_trials = (
        _stability(X, labels, n_clusters, min_cluster_size=floor,
                   min_samples=neighbours, scaling=scaling, seed=int(seed),
                   n_resamples=trials, subsample_frac=float(subsample_frac))
        if trials else (np.full(n_clusters, np.nan), np.zeros(n_clusters, dtype=np.int64))
    )

    clusters = []
    for label in range(n_clusters):
        member = np.flatnonzero(labels == label)
        block = Z[member]
        medoid = _medoid(block)
        panel = _spread(block, medoid, int(n_representatives))
        clusters.append(Cluster(
            label=label,
            size=int(member.size),
            stability=float(score[label]),
            n_trials=int(n_trials[label]),
            member_ids=[ids[i] for i in member],
            medoid_id=ids[member[medoid]],
            representative_ids=[ids[member[i]] for i in panel],
        ))

    return SubClusters(
        labels=labels,
        ids=ids,
        clusters=clusters,
        noise_ids=[ids[i] for i in np.flatnonzero(labels == NOISE)],
        scaling=scaling,
        column_scale=column_scale,
        settings={
            "min_cluster_size": floor,
            "min_samples": neighbours,
            "scaling": scaling,
            "seed": int(seed),
            "n_resamples": trials,
            "subsample_frac": float(subsample_frac),
            "n_representatives": int(n_representatives),
        },
    )
