"""Is one label actually several behaviours, and which clips would show that?

One class swallows most of an annotation project - almost everything is
"swimming" - and a label the schema never split apart is not evidence that it is
one behaviour. This module takes the feature rows of the clips already carrying
a single label and asks whether they fall into groups, then names the clips a
human should watch in order to decide what those groups are.

The idea is A-SOiD's; the pipeline deliberately is not. Two departures:

* **The clustering runs in the linear feature basis, never in an embedding.**
  Handing a density-based clusterer the coordinates of a UMAP or t-SNE
  projection makes it cluster densities the PROJECTION produced: both optimise a
  neighbourhood-preservation objective that is free to move points closer or
  further apart than the data has them, so a blob in the picture is not
  necessarily a blob in the measurement. The ``wave_*`` block is a least-squares
  projection of the amplitude, phase and static profiles onto shifted Legendre
  polynomials (:func:`trackanno.wavespectra._legendre_batch`) and the ``cnn*``
  block a PCA of the detector embeddings - both LINEAR maps of measured
  quantities, so a distance between two rows is a distance between two
  measurements, and it is already fitted. An embedding is a good way to DRAW an
  answer; it is not a thing to compute one from. If a caller wants a scatter
  plot, project the rows AFTER :func:`subcluster` has run and colour it by
  ``labels`` - never the other way round.
* **The clusterer is** :class:`sklearn.cluster.HDBSCAN`, already a dependency.
  ``umap-learn`` and the standalone ``hdbscan`` package add nothing this needs.

Noise is an answer, not a failure. HDBSCAN's ``-1`` says "this clip resembles
nothing else here", which for a clip pulled out of a dominant class is a
substantive finding and is preserved throughout - noise rows keep their ids in
:attr:`SubClusters.noise_ids` rather than being tidied into a nearest cluster.

HDBSCAN returns *something* on almost anything, and an annotator sent to label a
partition of noise has lost the afternoon and poisoned the next model with a
class that does not exist. Two things stand between the caller and that, and
they catch different failures - neither is sufficient alone:

* :data:`DEFAULT_MIN_CLUSTER_SIZE` keeps a gapless cloud from being carved up in
  the first place. Set high enough it is what stops unstructured data producing
  clusters at all, and the module refuses by name when nothing survives it.
* :attr:`Cluster.stability` asks whether a group that WAS found survives being
  shown a different 80% of the rows. That catches a group held together by a
  handful of clips; it does not catch a reproducible split of a cloud that has
  no groups in it, and :func:`subcluster` says so at length. Read that section
  before gating on the number.

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

# The bar a cluster has to clear before a human is sent to annotate it. This is
# Hennig's rule of thumb for bootstrap cluster stability (Hennig 2007,
# "Cluster-wise assessment of cluster stability"): at or below 0.5 the cluster is
# dissolved, 0.6 to 0.75 indicates a pattern without strong support, above 0.75
# is a valid stable cluster and above 0.85 a highly stable one. It is a
# convention, not a measurement of this data, and a project with expensive
# annotation should raise it rather than lower it - see `tests/test_subcluster.py`
# for where planted and unstructured data actually land.
STABLE = 0.75

# The smallest group that counts as a discovery. A discovered sub-cluster is only
# useful if it can become a LABEL, and the suggester refuses a class with fewer
# than `suggest.min_per_class` (5) examples while cross-validating in folds
# GROUPED by (movie, track) - so those examples have to come from several animals
# before a fold split can even score the new class. Three times the classifier's
# floor is the smallest number that plausibly survives that. That part is a
# judgement, not a measurement, and it is the first thing to change on a project
# whose clips are scarce.
#
# It is also, and this was NOT expected, the module's first defence against
# inventing structure - a stronger one than the stability score. A 400-row
# Gaussian with no structure in it at all comes back entirely as noise at this
# setting on 35 of the 36 draws `tests/test_subcluster.py` makes across 3, 6 and
# 12 columns, because excess-of-mass selection will not carve a cloud with no gap
# in it into groups this large; the one draw that got past scored 0.0 stability,
# i.e. dissolved completely. Drop to 5 and every one of those clouds yields
# clusters. Lower it knowingly.
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

        NaN is False. An unmeasured cluster is not a stable one, and the whole
        point of the score is that a cluster does not get the benefit of the
        doubt.
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
    # The per-column divisor actually used, so the scaling is inspectable rather
    # than a thing that happened. A column whose scale is far off its neighbours'
    # is the column that decided the answer.
    column_scale: np.ndarray
    # Every number that produced this. A discovery is a claim about a procedure
    # as much as about the data, and two runs are not comparable unless these
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

    Returns ``(centre, scale)``, both ``(n_features,)``.

    The IQR rather than the standard deviation because these columns have long
    tails by construction: :mod:`trackanno.wavespectra` records
    ``wave_wavelength_ratio`` running to 53.5 with a p90 of 7.6, since the
    wavelength blows up whenever a phase profile is not linear. One such row sets
    the standard deviation for the whole column and squashes every other row into
    a point, which deletes the column from the distance - the IQR does not move.

    Two degenerate cases, both explicit because a divisor of zero would turn
    every distance into NaN and the clusterer would refuse the matrix outright:

    * IQR zero but the column varies - a near-binary column such as
      ``wave_direction`` where one value holds more than half the rows. There is
      no bulk spread left to protect, so the full range is used. An outlier CAN
      set the scale of such a column; a column whose middle half is a single
      value has nothing better to offer.
    * a constant column - scale 1.0, which is exact rather than a fallback: every
      scaled value is then 0 and the column contributes 0 to every distance,
      whatever divisor is chosen.
    """
    centre = np.median(X, axis=0)
    q1, q3 = np.percentile(X, [25, 75], axis=0)
    scale = q3 - q1
    span = X.max(axis=0) - X.min(axis=0)
    scale = np.where(scale > 0, scale, span)
    return centre, np.where(scale > 0, scale, 1.0)


def _scaled(X: np.ndarray, scaling: str) -> tuple[np.ndarray, np.ndarray]:
    """Put the columns on a common footing, or deliberately decline to.

    Returns ``(Z, column_scale)``, ``Z`` always a fresh array.

    THE DECISION, since it changes the answer and there is no neutral option.
    ``"robust"`` divides each column by its own interquartile range
    (:func:`_robust_scale`). It is the default, chosen against two alternatives:

    * **Leaving the columns alone.** ``wave_wavelength_px`` is hundreds of
      pixels, ``wave_static_cos`` lives in ``[-1, 1]`` and ``wave_direction`` in
      ``{-1, +1}``. A Euclidean distance over that is the wavelength column and
      nothing else, so HDBSCAN would be clustering one feature while appearing to
      use twenty-five. Offered as ``"none"`` anyway, for a caller whose matrix is
      already commensurate - PCA scores, say - but it has to be asked for.
    * **Whitening**, i.e. decorrelating and giving every DIRECTION unit variance.
      That is the one this deliberately does not do. The ``wave_*`` columns are
      strongly related by construction - the amplitude Legendre coefficients move
      with the amplitude mean, the wave speed is the wavelength times the
      frequency - so the covariance's smallest eigen-directions carry almost
      nothing but measurement noise, and dividing by the square root of a
      near-zero eigenvalue promotes that noise to the same footing as the beat
      frequency. For a DENSITY-based clusterer that is not a cosmetic error: an
      inflated noise direction dilutes every neighbourhood along it and dissolves
      real clusters into the background. Per-column scaling equalises the columns
      while leaving the correlations - and so the fact that some directions
      genuinely carry more variance than others - exactly where they were.

    Centring is not part of the decision. A Euclidean distance is unchanged by
    it; it is here only to keep the subtraction well conditioned, since a column
    sitting at 1e6 +- 1 spends about six of float64's sixteen digits before the
    difference is taken.

    THE PRICE, because per-column scaling is not free either. A column that is
    bimodal BECAUSE of the split being hunted has an interquartile range that
    spans the gap, so dividing by it shrinks exactly the column carrying the
    signal down onto the same footing as a column carrying only noise. Three
    blobs planted in two of six columns, with the other four pure noise, are
    recovered at an adjusted Rand index of 1.00 when only those two columns are
    passed and under 0.05 when all six are - measured in
    `tests/test_subcluster.py`, which asserts both. Nor is the loss confined to
    the score: at all six columns HDBSCAN reports two groups where three were
    planted. The mitigation is the caller's and it is not a scaling
    choice: hand this function the columns the split is expected to live in
    rather than every column there is. Whitening does not help here either; it
    equalises directions rather than columns, so it dilutes the signal in the
    same way while additionally amplifying the noise directions.
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

    It is deterministic but not quite permutation-invariant, which is worth
    knowing before a caller re-sorts a table and reads the difference as a
    change in the data. Reordering the rows leaves the GROUPS identical - the
    rows both runs put in a cluster are partitioned exactly the same way - while
    a row sitting on the boundary between a cluster and noise can cross it: one
    row in four hundred, on the planted data in `tests/test_subcluster.py`, which
    asserts both halves of that. So membership of a boundary clip is not a fact
    to build on - and a cluster whose SHAPE depends on such rows is what
    :func:`_stability` is built to catch, since resampling removes them a fifth
    at a time.

    ``allow_single_cluster`` is left at its default of off. The question this
    module asks is whether a class is SEVERAL things, and "all of it, one group"
    is the null answer rather than a discovery; with the flag off that answer
    arrives as an all-noise result, which :func:`subcluster` refuses by name
    instead of handing an annotator the class they already had.

    ``copy=False`` is safe because every matrix that reaches here was allocated
    by :func:`_scaled` - the estimator is allowed to overwrite its input, and the
    caller's array is never the thing being overwritten.
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

    Noise is never a candidate to match against. A reference cluster whose
    members all came back as noise scores 0, which is the correct reading: the
    group dissolved. Scoring it against the noise "cluster" would reward a
    clustering for consistently failing to find anything.
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

    Each trial draws ``subsample_frac`` of the rows without replacement, runs the
    WHOLE procedure again on them - scaling included, since the scale is
    estimated from the data and a score that reused the full-data scale would
    understate how much the answer depends on which rows were drawn - and asks,
    for each cluster of the reference clustering, how well its surviving members
    are recovered as a group. The measure is the Jaccard overlap with the
    best-matching cluster of the rerun, and the score is the mean of that over
    the trials. This is Hennig's bootstrap cluster stability with subsampling in
    place of the bootstrap; subsampling because HDBSCAN is density-based and a
    bootstrap's duplicated rows are stacks of coincident points, which is a
    density spike the estimator would read as structure.

    A trial where fewer than two of a cluster's members were drawn says nothing
    about that cluster and is skipped rather than scored as a failure, which is
    why the trial count comes back beside the score.

    The drawn indices are kept in their original order. Tie-breaking inside a
    clusterer can depend on row order, and a stability score that moved when the
    rows were shuffled would not be a property of the data.

    Determinism: the only randomness is the draw, and each trial's generator is
    seeded from ``(seed, trial)`` so the whole sequence is fixed by ``seed``
    alone - independently of how many trials were run before it, so raising
    ``n_resamples`` extends the answer rather than replacing it.
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
    annotator can watch; the mean of a group of waveforms is not a waveform
    anybody recorded.

    Chunked, so the ``m x m`` distance matrix is never held: one dominant class
    is easily thousands of clips and the square of that is the memory failure
    this exists to avoid.
    """
    if Z.shape[0] == 1:
        return 0
    totals = np.concatenate(list(
        pairwise_distances_chunked(Z, reduce_func=lambda block, start: block.sum(axis=1))
    ))
    return int(np.argmin(totals))


def _spread(Z: np.ndarray, start: int, k: int) -> list[int]:
    """``start``, then repeatedly whichever row is furthest from everything chosen.

    NOT the ``k`` rows nearest the medoid. Those are ``k`` views of the same
    clip: they show the annotator the most typical member several times over and
    hide the cluster's extent, which is the only evidence that could tell them
    the group is really two things wearing one label. The max-min traversal
    starts at the centre and then covers the group - the extremes come out first,
    and each later pick fills the largest remaining hole - so the panel spans
    what the cluster contains.

    Stops early rather than returning a row at distance 0 from one already
    chosen: a second identical panel tells the annotator nothing. Ties go to the
    lowest index, so the panel is fixed by the row order and not by an ordering
    inside the search.
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

    Parameters
    ----------
    features
        ``(n_rows, n_features)`` float, all finite. See the refusals below for
        why NaN is not accepted here even though it is the codebase's ordinary
        way of saying "undefined".
    ids
        One id per row, matched by POSITION - typically ``clip_id``. They are
        opaque: they come back in ``member_ids``, ``medoid_id`` and
        ``representative_ids`` exactly as handed in.
    min_cluster_size
        The smallest group that counts as a discovery. See
        :data:`DEFAULT_MIN_CLUSTER_SIZE`.
    min_samples
        How conservative the density estimate is; ``None`` means
        ``min_cluster_size``, which is scikit-learn's default and the
        conservative end. Raising it moves more rows into noise.
    scaling
        ``"robust"`` or ``"none"``; the choice is documented in :func:`_scaled`
        and recorded in the result.
    seed
        Fixes the resampling. A discovery that changes between runs is not a
        discovery, so this is the only source of randomness in the module and
        every other step is deterministic.
    n_resamples
        Trials behind the stability score. ``0`` skips the measurement entirely
        and every ``stability`` comes back NaN - which is "nobody looked", never
        "unstable", and :attr:`Cluster.stable` is False for it either way.
    subsample_frac
        Share of the rows each trial draws.
    n_representatives
        Rows to name per cluster, medoid included.

    Returns
    -------
    :class:`SubClusters`. ``labels`` is aligned with ``ids`` and carries
    :data:`NOISE` for a row in no cluster.

    THE STABILITY SCORE, which is the part worth reading twice. It is the mean,
    over resamples, of the Jaccard overlap between a cluster's surviving members
    and the best-matching cluster of a rerun on those rows alone
    (:func:`_stability`). What it IS evidence of: that the grouping is a property
    of the data rather than of which rows happened to be labelled - a cluster
    that scores well would still have been found with a different 80% of the
    clips. It also penalises both failure modes rather than one, because a
    cluster that keeps being swallowed by a larger group scores its own share of
    that group and not 1: the members did stay together, but not as a group of
    their own.

    What it is NOT evidence of, none of which it can see:

    * that a group exists at all. This is the one that matters and it is easy to
      read the wrong way round. Resampling varies WHICH ROWS are present; it does
      not vary the clusterer, so a split that a smooth unimodal cloud provokes by
      its shape alone comes back trial after trial and scores highly. Measured:
      a plain two-column Gaussian clustered at ``min_cluster_size=5`` is cut into
      three, and those pieces score 0.78, 0.91 and 0.95 - all above
      :data:`STABLE` - in `tests/test_subcluster.py`, which asserts it so nobody
      has to find out later. Stability rules out "this group is an accident
      of which clips happened to be labelled". It does not rule out "this group
      is an artefact of what a density clusterer does to a cloud with no gap in
      it", and nothing here does: what keeps that out is
      :data:`DEFAULT_MIN_CLUSTER_SIZE`, which refuses to carve groups that large
      out of a gapless cloud in the first place.
    * that the cluster is a BEHAVIOUR. Clips from one movie, one animal, or one
      recording session share illumination, focus and body size, cluster tightly
      on that, and are perfectly stable. This module is handed a bare matrix and
      cannot check; the caller can, and should look at the movie and track
      composition of a cluster before sending anyone to annotate it. The
      re-identification warning on the ``cnn*`` features in
      :mod:`trackanno.clipfeatures` is the same trap in a different place.
    * significance. There is no null distribution here and the score is not a
      p-value; :data:`STABLE` is a convention.
    * anything about clips not in ``features``. It resamples the rows present, so
      it measures reproducibility over this labelled set and not generalisation
      to the class as a whole.
    * anything comparable across settings. Fewer rows per trial means every
      cluster is thinner and ``min_cluster_size`` bites harder, so scores only
      mean the same thing at the same ``subsample_frac`` and
      ``min_cluster_size``.

    Refusals. Each of these is a case where an answer could be produced and would
    mislead:

    * a non-finite value anywhere. NaN means "undefined" throughout this codebase
      and HDBSCAN has no way to read it. The caller must drop the row or the
      column and decide which, because imputing - the obvious silent fix - would
      put every clip nobody could measure at the column median, i.e. would
      manufacture a dense group out of exactly the rows we know least about, and
      that group would then be stable.
    * ``ids`` not matching the rows, or repeating. Representatives are reported
      as ids, so a duplicate makes the answer ambiguous about which clip to show.
    * too few rows for the question. Below two groups of ``min_cluster_size``
      there is nothing to find, and below two of them in a RESAMPLE every
      stability trial is forced to answer "one cluster" and the score measures
      the arithmetic rather than the data.
    * every row classed as noise. That is HDBSCAN saying it found no group at
      this setting; it is a real answer, but it is not one that can be returned
      as a list of clusters without the caller having to guess what an empty list
      meant, so it is raised with the numbers that produced it.
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
