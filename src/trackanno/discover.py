"""Is one of my classes really several behaviours, and which clips would show that?

    uv run python -m trackanno.discover --list
    uv run python -m trackanno.discover --class swimming
    uv run python -m trackanno.discover --class swimming --columns wave_freq_hz,wave_amp_*

:mod:`trackanno.subcluster` answers that question about a bare matrix. This
module is what a person runs: it chooses the rows (the clips of ONE class, from
``clip_features.parquet`` joined to the annotation CSV), chooses the columns,
runs the clustering, makes the check :mod:`trackanno.subcluster` structurally
cannot make for itself, and writes the answer where a later step can read it.

The columns are the decision, so it is made explicitly and printed. Per-column
scaling puts a column carrying the split on the same footing as one carrying
noise, so passing every column in the table is close to the worst available
choice. The default is the ``wave_*`` block
(:data:`trackanno.wavespectra.SERIES_COLUMNS`), in which a sub-behaviour is a
difference in BEATING rather than in framing, taken whole because the beat
quantities are algebraically linked. ``--columns`` widens or narrows it, and
narrowing is the mitigation when a run comes back all noise. Four of the block's
columns describe the measurement rather than the cell (``wave_interp_frac``,
``wave_window_frames``, ``wave_censored``, ``wave_resolvable_hz``) and are left
in, since a cluster forming on them says the class holds a population of badly
tracked clips.

The batch-effect check is the reason this module is worth having. A cluster of
clips from one movie, animal or session shares illumination, focus, cell density
and body size; it is tight, reproducible under resampling, and
:attr:`trackanno.subcluster.Cluster.stability` reports it as excellent. This
module has ``movie_id`` and ``track_id`` beside every row, so it computes each
cluster's composition, prints it whether or not it looks bad, and says in words
what a lopsided one means.

Every parameter is a command-line flag rather than a config key, a discovery run
being something repeated interactively with the settings changed. The config is
read only for the work directory, the annotation CSV and the seed
(:attr:`trackanno.config.Suggest.seed`).

Noise is an answer, as it is in :mod:`trackanno.subcluster`: the clips that
resemble nothing else in their class are reported and written out rather than
tidied into the nearest group.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .clipfeatures import CLIP_FEATURES_FILE, FEATURE_GROUPS, feature_columns
from .config import Config, load_config
from .store import IDENTITY_COLUMNS, TRAILING_COLUMNS
from .subcluster import (
    DEFAULT_MIN_CLUSTER_SIZE, SCALINGS, STABLE, SubClusters, subcluster,
)
from .suggest import is_audit_clip

__all__ = [
    "Composition", "DiscoveredCluster", "Discovery",
    "discover", "resolve_columns", "class_table",
    "format_report", "report_path", "to_payload", "write_report", "load_report",
    "BATCH_FRACTION", "DEFAULT_COLUMNS", "SCHEMA", "main",
]

# The default `--columns` spec: the name of the feature group whose prefix is
# `wave_`. Spelled as the group rather than as a prefix so the CLI, the GUI's
# Configure page and `suggest.feature_groups` all use one vocabulary for the same
# 25 columns.
DEFAULT_COLUMNS = "waveform"

# Above this share of one movie (or one track), a cluster is called out as a
# possible batch effect. A convention rather than a measurement, and close to 1
# so the check catches the cluster that is ONE recording rather than grumbling at
# every cluster with a favourite movie. `--batch-frac` lowers it, and the
# composition is printed in full either way.
BATCH_FRACTION = 0.9

# Stamped into the written report. The version has to travel INSIDE the artefact,
# since a name or a folder can be renamed away from it.
SCHEMA = "trackanno.discover/1"

# What a written report is called when `--out` does not say. One file per
# (group, class), so a second run does not delete the first run's answer.
REPORT_TEMPLATE = "subclusters_{group}_{klass}.json"


# --------------------------------------------------------------------------- #
# what comes back
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Composition:
    """Where one cluster's clips came from, and whether that is suspicious.

    ``kind`` is ``"movie"`` or ``"track"``; ``counts`` is every key with its
    member count, largest first. ``class_frac`` is the same key's share of the
    WHOLE clustered class, and it is what makes the flag mean anything: a cluster
    that is 95% movie A is unremarkable in a class that is itself 95% movie A,
    and damning in one spread evenly over six.
    """

    kind: str
    counts: dict
    top: str
    top_frac: float
    class_frac: float
    flagged: bool

    @property
    def n(self) -> int:
        """How many distinct movies (or tracks) the cluster draws on."""
        return len(self.counts)


@dataclass(frozen=True)
class DiscoveredCluster:
    """One candidate sub-behaviour: the group, its evidence, and its provenance."""

    label: int
    size: int
    stability: float
    n_trials: int
    stable: bool
    member_ids: list
    medoid_id: object
    # The medoid first, then a max-min spread over the rest, each as a record
    # carrying the movie and track it came from - so the panel can be read
    # without joining anything back.
    representatives: list
    movies: Composition
    tracks: Composition

    @property
    def flagged(self) -> bool:
        """Whether either composition says this may be a recording, not a behaviour."""
        return self.movies.flagged or self.tracks.flagged


@dataclass(frozen=True)
class Discovery:
    """Everything one run found, and everything needed to judge it."""

    group: str
    class_name: str
    # The spec as typed, and the columns it resolved to. Both, because
    # "waveform" is what a person repeats and the column list is what the answer
    # actually depends on.
    column_spec: str
    columns: list
    clusters: list
    result: SubClusters
    class_movies: dict
    class_tracks: dict
    # Row and column bookkeeping: what was labelled, what survived the join, what
    # the NaN drop cost and which column cost it. A discovery over half a class
    # is a different claim from one over all of it.
    census: dict
    settings: dict
    # Non-empty when the class is so concentrated in one movie that the
    # composition check cannot separate anything. See :func:`_class_caveat`.
    caveat: str = ""

    @property
    def flagged(self) -> list:
        return [c for c in self.clusters if c.flagged]

    @property
    def stable_clusters(self) -> list:
        return [c for c in self.clusters if c.stable]


# --------------------------------------------------------------------------- #
# columns
# --------------------------------------------------------------------------- #
def resolve_columns(features: pd.DataFrame, spec: str) -> list[str]:
    """Turn a ``--columns`` spec into the explicit, ordered list it means.

    A spec is a comma-separated list of tokens, each one of:

    * ``all`` - every feature column in the table.
    * a feature-group name from :data:`trackanno.clipfeatures.FEATURE_GROUPS`
      (``kinematic``, ``pose``, ``waveform``, ``cnn``), the same vocabulary
      ``suggest.feature_groups`` uses.
    * a prefix glob such as ``wave_amp_*``.
    * an exact column name.

    The result keeps the table's own column order and drops duplicates, so
    ``waveform,wave_freq_hz`` does not weight one column twice. A token matching
    nothing is refused rather than skipped, since skipping it would let a typo
    quietly change the feature space the answer was computed in.
    """
    available = feature_columns(features)
    if not available:
        raise ValueError(
            "the feature table has no feature columns at all, only metadata "
            f"({', '.join(map(str, features.columns[:6]))}...); it was not built by "
            "build_clip_features, or it was built with every feature group off"
        )
    tokens = [t.strip() for t in str(spec).split(",") if t.strip()]
    if not tokens:
        raise ValueError(
            f"--columns {spec!r} names no columns; pass a feature group "
            f"({', '.join(FEATURE_GROUPS)}), 'all', a prefix like 'wave_*', or "
            "explicit column names"
        )

    chosen: list[str] = []
    for token in tokens:
        if token == "all":
            picked = list(available)
        elif token in FEATURE_GROUPS:
            picked = feature_columns(features, [token])
            if not picked:
                raise ValueError(
                    f"--columns names the feature group {token!r}, but this table has "
                    f"no {token} columns. {FEATURE_GROUPS[token]['about']} Rebuild the "
                    "feature table with that group available, or choose another group: "
                    + ", ".join(
                        g for g in FEATURE_GROUPS if feature_columns(features, [g])
                    )
                )
        elif token.endswith("*"):
            prefix = token[:-1]
            picked = [c for c in available if c.startswith(prefix)]
            if not picked:
                raise ValueError(
                    f"--columns has a glob {token!r} that matches none of the "
                    f"{len(available)} feature columns in this table"
                )
        else:
            if token not in available:
                near = difflib.get_close_matches(token, available, n=4, cutoff=0.6)
                raise ValueError(
                    f"--columns names a column {token!r} that this feature table does "
                    "not have"
                    + (f"; did you mean {', '.join(near)}?" if near else "")
                    + f". It has {len(available)} feature columns; pass a group name "
                    f"({', '.join(FEATURE_GROUPS)}) or 'all' to take a whole block"
                )
            picked = [token]
        chosen.extend(picked)

    wanted = set(chosen)
    return [c for c in available if c in wanted]


# --------------------------------------------------------------------------- #
# the rows
# --------------------------------------------------------------------------- #
def _labelled(labels: pd.DataFrame, group: str) -> pd.DataFrame:
    """The non-empty values of one category column, as strings.

    The same normalisation :func:`trackanno.suggest.train_group` makes: an
    unlabelled clip is an empty cell or a missing one depending on how the CSV
    was written, and a class called ``"nan"`` or ``""`` would otherwise be
    counted as a behaviour.
    """
    if group not in labels.columns:
        category = [
            c for c in labels.columns
            if c not in IDENTITY_COLUMNS and c not in TRAILING_COLUMNS
        ]
        raise ValueError(
            f"the annotation CSV has no column {group!r}, so no clip carries a "
            f"label in that group. Its category columns are "
            f"{category or ['(none)']} - pass one of those with --group, or check "
            "that this is the right project's CSV"
        )
    # movie_id comes along when the CSV has it, because it is what
    # :func:`class_table` compares against the feature table's own answer; a
    # projection down to the two columns would silently disable that check.
    keep = ["clip_id", group] + (["movie_id"] if "movie_id" in labels.columns else [])
    values = labels[keep].dropna(subset=[group]).copy()
    values[group] = values[group].astype(str).str.strip()
    return values[values[group].str.len() > 0]


def _duplicate(ids) -> str | None:
    """The first id that appears twice, or None. Order-preserving, so it names the same one twice."""
    seen = set()
    for name in ids:
        if name in seen:
            return str(name)
        seen.add(name)
    return None


def class_table(
    features: pd.DataFrame, labels: pd.DataFrame, group: str,
    class_name: str | None = None, *, exclude_movies=(),
) -> tuple[pd.DataFrame, str, dict]:
    """The feature rows of one class, joined to its labels, with the join audited.

    Returns ``(rows, class_name, census)``. ``rows`` carries the feature table's
    own columns plus the label column; ``census`` records what the join cost.

    ``class_name`` of ``None`` selects the MAJORITY class, the one most likely to
    be hiding structure. Ties go to the alphabetically first name, so the default
    does not depend on row order.

    Every refusal here is a case where clustering would produce a confident
    answer about the wrong rows:

    * a feature table with no ``movie_id`` or ``track_id``, which is what the
      batch-effect check reads.
    * a ``clip_id`` appearing twice in either table: the answer names clips, and
      a duplicated feature row is also a coincident pair of points, a density
      spike the clusterer reads as structure.
    * no labelled clip of the class having a feature row at all. ``clip_id``
      embeds the track id, so re-running detection renumbers the tracks and every
      earlier label stops resolving; that is a disagreement between the two
      tables, not an empty class.
    * the two tables naming a different movie for the same clip, so the
      composition below would be computed from the wrong provenance.
    """
    missing = [c for c in ("clip_id", "movie_id", "track_id") if c not in features.columns]
    if missing:
        raise ValueError(
            f"the feature table has no {missing} column(s). movie_id and track_id are "
            "what the batch-effect check reads, and that check is the whole reason to "
            "run this rather than call subcluster directly - a cluster of one movie is "
            "perfectly stable and is not a behaviour. Rebuild the table with "
            "build_clip_features, which writes them"
        )
    if "clip_id" not in labels.columns:
        raise ValueError(
            "the annotation CSV has no clip_id column, so its labels cannot be "
            "matched to feature rows at all"
        )

    wanted = _labelled(labels, group)
    counts = wanted[group].value_counts().to_dict()
    if not counts:
        raise ValueError(
            f"no clip in the annotation CSV carries a label in group {group!r}; there "
            "is no class to look inside"
        )

    # Computed ONCE and reused below, so the tie rule cannot drift between the
    # class chosen and the census claiming the majority was chosen.
    majority = max(sorted(counts), key=lambda name: counts[name])
    if class_name is None:
        class_name = majority
    class_name = str(class_name)
    if class_name not in counts:
        listed = ", ".join(f"{k} ({v})" for k, v in sorted(counts.items()))
        raise ValueError(
            f"no clip is labelled {group}={class_name!r}. The classes present are: "
            f"{listed}"
        )

    mine = wanted[wanted[group] == class_name]
    repeated = _duplicate(mine["clip_id"].astype(str))
    if repeated is not None:
        raise ValueError(
            f"clip {repeated!r} appears more than once in the annotation CSV under "
            f"{group}={class_name!r}. One clip is one row - a duplicate means the file "
            "was edited or concatenated by hand, and the clustering would weight that "
            "clip twice"
        )
    repeated = _duplicate(features["clip_id"].astype(str))
    if repeated is not None:
        raise ValueError(
            f"clip {repeated!r} has more than one row in the feature table. The answer "
            "names clips, so a duplicate leaves nobody able to say which one to watch, "
            "and two identical rows are a density spike the clusterer would read as "
            "structure. Rebuild the feature table"
        )

    merged = features.merge(mine[["clip_id", group]], on="clip_id", how="inner")
    n_orphaned = len(mine) - len(merged)
    if merged.empty:
        raise ValueError(
            f"none of the {len(mine)} clip(s) labelled {group}={class_name!r} has a row "
            f"in the feature table, so the two disagree completely. clip_id embeds the "
            "track id, so re-running detection renumbers the tracks and every label "
            "written beforehand stops resolving - run the label repair on the Model "
            "page, or rebuild the feature table over the current clips"
        )

    # The CSV records the movie the clip came from as well, so the two can be
    # compared instead of one being trusted. They disagree only if the tables come
    # from different runs, and that is exactly when the composition below would be
    # computed from the wrong provenance.
    if "movie_id" in mine.columns:
        both = merged.merge(
            mine[["clip_id", "movie_id"]].rename(columns={"movie_id": "_csv_movie"}),
            on="clip_id", how="left",
        )
        told = both["_csv_movie"].astype(str)
        found = both["movie_id"].astype(str)
        clash = both[(told != found) & (told != "nan")]
        if len(clash):
            first = clash.iloc[0]
            raise ValueError(
                f"the feature table and the annotation CSV disagree about where clip "
                f"{first['clip_id']!r} came from: the table says movie "
                f"{first['movie_id']!r} and the CSV says {first['_csv_movie']!r} "
                f"({len(clash)} clip(s) disagree). They were built from different runs, "
                "and the batch-effect check below would be computed from the wrong "
                "provenance. Rebuild the feature table"
            )

    excluded = [str(m) for m in exclude_movies]
    n_excluded = 0
    if excluded:
        present = set(merged["movie_id"].astype(str))
        unknown = [m for m in excluded if m not in present]
        if unknown:
            raise ValueError(
                f"--exclude-movie names {unknown}, which no clip of "
                f"{group}={class_name!r} comes from. This class is in "
                f"{', '.join(sorted(present))}"
            )
        keep = ~merged["movie_id"].astype(str).isin(excluded)
        n_excluded = int((~keep).sum())
        merged = merged[keep]

    census = {
        "n_labelled": int(len(mine)),
        "n_orphaned": int(n_orphaned),
        "n_excluded_movies": n_excluded,
        "excluded_movies": excluded,
        "class_counts": {str(k): int(v) for k, v in sorted(counts.items())},
        "chose_majority": bool(class_name == majority),
    }
    return merged.reset_index(drop=True), class_name, census


def _finite_rows(
    rows: pd.DataFrame, columns: list[str], group: str, class_name: str, spec: str
) -> tuple[pd.DataFrame, list[str], dict]:
    """Drop what cannot be clustered, and record exactly what that cost.

    :func:`trackanno.subcluster.subcluster` refuses a non-finite value, so
    something has to be dropped and the choice is the caller's. Two steps: a
    column that is NaN for EVERY row of this class goes first, costing nothing;
    then any row still holding a NaN goes, and the columns responsible are
    counted and reported.

    Rows rather than columns at the second step, because a partly-NaN column is
    still a measurement of the clips that have it while a row with a hole has no
    position in the space. The dropped clips are NOT a random sample - a
    ``wave_*`` column is undefined exactly when the window could not be
    transformed, i.e. on short, sparse or heavily interpolated tracks - so a
    discovery is a statement about the measurable part of the class. The report
    names the column doing most of the damage, since dropping it with
    ``--columns`` is often better than dropping a third of the clips.
    """
    block = rows[columns]
    all_nan = [c for c in columns if not np.isfinite(block[c].to_numpy(dtype=float)).any()]
    kept = [c for c in columns if c not in all_nan]
    if not kept:
        raise ValueError(
            f"every one of the {len(columns)} column(s) selected by --columns {spec!r} "
            f"is undefined for all {len(rows)} clip(s) of {group}={class_name!r}, so "
            "there is nothing to measure a distance in. The waveform and pose blocks "
            "need detection.task: pose and a window long enough to transform; the "
            "kinematic block is always computed. Choose another block with --columns, "
            "or rebuild the feature table"
        )

    finite = np.isfinite(rows[kept].to_numpy(dtype=float))
    good = finite.all(axis=1)
    # Attributed per column, so the report can name the one column that is
    # costing the clips rather than only the total.
    by_column = {
        c: int((~finite[:, i]).sum()) for i, c in enumerate(kept) if (~finite[:, i]).any()
    }
    constant = [
        c for c in kept
        if good.any() and float(np.ptp(rows.loc[good, c].to_numpy(dtype=float))) == 0.0
    ]
    census = {
        "n_all_nan_columns": len(all_nan),
        "all_nan_columns": all_nan,
        "n_dropped_rows": int((~good).sum()),
        "dropped_rows_by_column": dict(sorted(by_column.items(), key=lambda kv: -kv[1])),
        "constant_columns": constant,
    }
    return rows[good].reset_index(drop=True), kept, census


# --------------------------------------------------------------------------- #
# the batch-effect check
# --------------------------------------------------------------------------- #
def _counts(values) -> dict:
    """Tally, largest first, ties broken by key so a report does not reshuffle."""
    tally: dict = {}
    for value in values:
        tally[str(value)] = tally.get(str(value), 0) + 1
    return dict(sorted(tally.items(), key=lambda kv: (-kv[1], kv[0])))


def _track_keys(rows: pd.DataFrame) -> np.ndarray:
    """``movie::track`` per row - the identity of one animal.

    Track ids are numbered per MOVIE, so a bare ``track_id`` would read two
    movies' track 3 as one animal. The same key
    :meth:`trackanno.suggest.Suggester._track_keys` builds.
    """
    return np.array([
        f"{m}::{t}" for m, t in zip(rows["movie_id"].astype(str), rows["track_id"].astype(str))
    ], dtype=object)


def _compose(kind: str, member_keys, class_counts: dict, n_class: int,
             threshold: float) -> Composition:
    """One cluster's composition over movies or tracks, with the flag rule applied.

    A cluster is flagged when its most common movie (or track) holds at least
    ``threshold`` of it AND holds less than ``threshold`` of the class as a
    whole. The second half stops the check crying wolf on a class that exists in
    one movie only, where every cluster is 100% that movie and the comparison has
    no contrast; :func:`_class_caveat` reports that case once, at the top.
    """
    counts = _counts(member_keys)
    size = sum(counts.values())
    top = next(iter(counts))
    top_frac = counts[top] / size if size else float("nan")
    class_frac = class_counts.get(top, 0) / n_class if n_class else float("nan")
    return Composition(
        kind=kind, counts=counts, top=top, top_frac=float(top_frac),
        class_frac=float(class_frac),
        flagged=bool(top_frac >= threshold and class_frac < threshold),
    )


def _class_caveat(class_movies: dict, n_class: int, threshold: float) -> str:
    """The sentence for a class so concentrated that no composition can inform.

    Empty when the class is spread widely enough for the check to mean something.
    """
    if not class_movies or not n_class:
        return ""
    top = next(iter(class_movies))
    frac = class_movies[top] / n_class
    if frac < threshold:
        return ""
    return (
        f"the batch-effect check cannot separate anything here: {frac:.0%} of the "
        f"clustered clips of this class come from movie {top!r} already, so every "
        "group found is a group of that movie and the compositions below carry no "
        "information about whether a cluster is a behaviour or a recording. Label "
        "this class in another movie before acting on anything found here - it is "
        "the same constraint suggest.py reports when a class cannot be held out."
    )


def _batch_note(cluster: DiscoveredCluster) -> list[str]:
    """What a lopsided composition means, in words, and what to do about it.

    Written out per cluster rather than left to a symbol, because every other
    number on the screen says a batch-effect cluster is real.
    """
    out: list[str] = []
    movies, tracks = cluster.movies, cluster.tracks
    if movies.flagged:
        out += [
            f"POSSIBLE BATCH EFFECT: {movies.top_frac:.0%} of this cluster is movie "
            f"{movies.top!r}, which is {movies.class_frac:.0%} of the class.",
            "Clips from one movie share illumination, focus, cell density and body "
            "size, so they group tightly on the RECORDING; that group is perfectly "
            "reproducible under resampling and its stability score cannot tell you "
            "otherwise.",
            "Watch the clips below before annotating anything: if they look like the "
            "same preparation rather than the same motion, this is not a "
            "sub-behaviour. Re-running with "
            f"--exclude-movie {movies.top} answers it directly - a real "
            "sub-behaviour survives losing one movie.",
        ]
    if tracks.flagged:
        out += [
            f"POSSIBLE BATCH EFFECT: {tracks.top_frac:.0%} of this cluster is the "
            f"single track {tracks.top!r}, i.e. ONE animal, which is "
            f"{tracks.class_frac:.0%} of the class.",
            "Successive windows of one track are not independent measurements - with "
            "clips.stride below clips.frames they share most of their frames, and "
            "their spectral features are measured over windows that overlap further "
            "still - so this is one cell measured repeatedly rather than a group of "
            "cells behaving alike.",
        ]
    return out


# --------------------------------------------------------------------------- #
# the entry point
# --------------------------------------------------------------------------- #
def discover(
    features: pd.DataFrame, labels: pd.DataFrame, group: str,
    class_name: str | None = None, *,
    columns: str = DEFAULT_COLUMNS,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    min_samples: int | None = None,
    scaling: str = "robust",
    seed: int = 0,
    n_resamples: int = 25,
    subsample_frac: float = 0.8,
    n_representatives: int = 8,
    batch_fraction: float = BATCH_FRACTION,
    exclude_movies=(),
    audit_of=None,
    exclude_audit: bool = False,
) -> Discovery:
    """Look inside one class for sub-behaviours, and check what each group is made of.

    ``features`` is ``clip_features.parquet`` as a frame (``clip_id``,
    ``movie_id``, ``track_id`` and the feature columns) and ``labels`` the
    annotation CSV as a frame (``clip_id`` and one column per category group).
    ``group`` names the category column holding the class and ``class_name`` the
    class, ``None`` taking the majority one (see :func:`class_table`).
    ``columns`` is a ``--columns`` spec (see :func:`resolve_columns`);
    ``batch_fraction`` is the bar the composition check applies (see
    :func:`_compose`); ``exclude_movies`` leaves movies out entirely, the
    follow-up to a flagged cluster. ``audit_of`` is an optional
    ``clip_id -> bool`` predicate marking the audit holdout, so the panel can
    mark a representative ``(audit)`` rather than quietly sending an annotator to
    a clip whose label was meant to be independent of any automated procedure;
    ``exclude_audit`` leaves those clips out of the search instead, and is off by
    default because dropping 10% of a class would shrink the search and skew the
    composition the batch check reads. Everything else is passed through to
    :func:`trackanno.subcluster.subcluster` and documented there.

    Returns a :class:`Discovery`. Refusals are :class:`ValueError` naming the
    number that triggered them. The two that are this function's own rather than
    :func:`class_table`'s are a class too small to hold two groups, and an
    all-noise result - the latter being HDBSCAN saying this class is one thing in
    these columns, where the fix is usually to narrow the columns.
    """
    rows, class_name, census = class_table(
        features, labels, group, class_name, exclude_movies=exclude_movies
    )
    if audit_of is not None:
        audit = np.array([bool(audit_of(str(c))) for c in rows["clip_id"]], dtype=bool)
    elif exclude_audit:
        raise ValueError(
            "audit clips were asked to be excluded, but nothing here knows which they "
            "are: the holdout is defined by suggest.audit_fraction and "
            "suggest.audit_seed, so pass audit_of (the CLI passes it from the config)"
        )
    else:
        audit = np.zeros(len(rows), dtype=bool)
    census["n_audit"] = int(audit.sum())
    census["excluded_audit"] = bool(exclude_audit)
    if exclude_audit:
        rows, audit = rows[~audit], audit[~audit]
    rows = rows.reset_index(drop=True).assign(_audit=audit)

    chosen = resolve_columns(features, columns)
    rows, kept, drops = _finite_rows(rows, chosen, group, class_name, columns)
    census.update(drops)
    census["n_clustered"] = int(len(rows))

    floor = int(min_cluster_size)
    if len(rows) < 2 * floor:
        blame = ""
        if drops["n_dropped_rows"] and drops["dropped_rows_by_column"]:
            worst, n_worst = next(iter(drops["dropped_rows_by_column"].items()))
            blame = (
                f" {drops['n_dropped_rows']} clip(s) were dropped for an undefined "
                f"value, most often in {worst} ({n_worst}); dropping that column with "
                "--columns may be cheaper than dropping those clips."
            )
        raise ValueError(
            f"{len(rows)} clip(s) of {group}={class_name!r} can be clustered "
            f"({census['n_labelled']} labelled, {census['n_orphaned']} with no feature "
            f"row, {drops['n_dropped_rows']} with an undefined value in the chosen "
            f"columns), which cannot hold two groups of --min-cluster-size {floor}: "
            "there is no sub-structure that could be found. Label more clips of this "
            f"class, or take --min-cluster-size below {len(rows) // 2 + 1} - and read "
            "what it is for first, because it is what stops a cloud with no groups in "
            f"it being carved into some.{blame}"
        )

    ids = [str(c) for c in rows["clip_id"]]
    try:
        result = subcluster(
            rows[kept].to_numpy(dtype=float), ids,
            min_cluster_size=floor, min_samples=min_samples, scaling=scaling,
            seed=int(seed), n_resamples=int(n_resamples),
            subsample_frac=float(subsample_frac),
            n_representatives=int(n_representatives),
        )
    except ValueError as exc:
        if "came back as noise" not in str(exc):
            raise
        raise ValueError(
            f"{exc}. Here that was {len(kept)} column(s) from --columns "
            f"{columns!r} over {len(rows)} clip(s) of {group}={class_name!r}. What to "
            "try, in order: NARROW --columns to the columns a split would live in - "
            "per-column scaling puts an informative column on the same footing as a "
            "noise column, and three groups planted in two of six columns are "
            "recovered at ARI 1.00 from those two and under 0.05 from all six "
            "(tests/test_subcluster.py) - then lower --min-cluster-size, knowing that "
            "it is the setting that keeps an unstructured cloud from being carved up "
            "at all. If neither changes the answer, the honest reading is that this "
            "class is one behaviour in this feature basis."
        ) from exc

    movies = rows["movie_id"].astype(str).to_numpy()
    tracks = _track_keys(rows)
    by_id = {clip: i for i, clip in enumerate(ids)}
    class_movies, class_tracks = _counts(movies), _counts(tracks)
    n_class = len(ids)

    clusters = []
    for cluster in result.clusters:
        member = np.array([by_id[m] for m in cluster.member_ids])
        panel = [by_id[m] for m in cluster.representative_ids]
        clusters.append(DiscoveredCluster(
            label=cluster.label,
            size=cluster.size,
            stability=cluster.stability,
            n_trials=cluster.n_trials,
            stable=cluster.stable,
            member_ids=list(cluster.member_ids),
            medoid_id=cluster.medoid_id,
            representatives=[{
                "clip_id": ids[i],
                "movie_id": str(rows["movie_id"].iloc[i]),
                "track_id": str(rows["track_id"].iloc[i]),
                "f_start": (int(rows["f_start"].iloc[i])
                            if "f_start" in rows.columns else None),
                "medoid": ids[i] == str(cluster.medoid_id),
                "audit": bool(rows["_audit"].iloc[i]),
            } for i in panel],
            movies=_compose("movie", movies[member], class_movies, n_class, batch_fraction),
            tracks=_compose("track", tracks[member], class_tracks, n_class, batch_fraction),
        ))

    return Discovery(
        group=group, class_name=class_name,
        column_spec=str(columns), columns=list(kept),
        clusters=clusters, result=result,
        class_movies=class_movies, class_tracks=class_tracks,
        census=census,
        settings={**result.settings, "batch_fraction": float(batch_fraction),
                  "columns": str(columns), "exclude_movies": list(census["excluded_movies"])},
        caveat=_class_caveat(class_movies, n_class, batch_fraction),
    )


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def _share(counts: dict, size: int, limit: int = 4) -> str:
    """``m1 58 (91%), m2 4 (6%), ...`` - the head of a composition, in one line."""
    parts = [f"{k} {v} ({v / size:.0%})" for k, v in list(counts.items())[:limit]]
    if len(counts) > limit:
        parts.append(f"+{len(counts) - limit} more")
    return ", ".join(parts)


def _stability_text(cluster: DiscoveredCluster) -> str:
    """The score with its denominator, or 'not measured' - never a bare number.

    NaN is "nobody looked", not "unstable", and a mean over two trials is not the
    claim a mean over twenty-five is, so the trial count travels with the score.
    """
    if not np.isfinite(cluster.stability):
        return "stability not measured (--resamples 0)"
    return (f"stability {cluster.stability:.2f} over {cluster.n_trials} resample(s)"
            + ("  STABLE" if cluster.stable else f"  below {STABLE}"))


def format_report(found: Discovery) -> str:
    """The human-readable answer: what was found, and what each group is made of.

    Composition is printed for EVERY cluster, not only the flagged ones, or the
    absence of a warning would be indistinguishable from nobody having looked.
    """
    census, out = found.census, []
    out.append(f"=== {found.group} = {found.class_name!r} ===")
    if census.get("chose_majority") and census.get("class_counts"):
        out.append(
            f"  the majority of the {len(census['class_counts'])} class(es) in this "
            "group, and so the one most likely to be hiding something"
        )
    out.append(
        f"  {census['n_labelled']} labelled clip(s); {census['n_orphaned']} have no "
        f"feature row; {census['n_dropped_rows']} dropped for an undefined value; "
        f"{census['n_clustered']} clustered"
    )
    if census["n_dropped_rows"]:
        worst = list(census["dropped_rows_by_column"].items())[:3]
        out.append(
            "    the dropped clips are not a random sample - a column is undefined "
            "where it could not be measured: "
            + ", ".join(f"{c} ({n})" for c, n in worst)
        )
    out.append(
        f"  columns: {len(found.columns)} from --columns {found.column_spec!r}"
        + (f"; {census['n_all_nan_columns']} dropped as undefined for every clip"
           if census["n_all_nan_columns"] else "")
        + (f"; {len(census['constant_columns'])} constant and so contributing nothing "
           f"({', '.join(census['constant_columns'][:3])})"
           if census["constant_columns"] else "")
    )
    settings = found.settings
    out.append(
        f"  settings: min_cluster_size={settings['min_cluster_size']} "
        f"min_samples={settings['min_samples']} scaling={settings['scaling']} "
        f"seed={settings['seed']} resamples={settings['n_resamples']} "
        f"subsample={settings['subsample_frac']}"
    )
    if census["excluded_movies"]:
        out.append(
            f"  excluded: {', '.join(census['excluded_movies'])} "
            f"({census['n_excluded_movies']} clip(s) left out)"
        )
    out.append(
        f"  the clustered clips span {len(found.class_movies)} movie(s) and "
        f"{len(found.class_tracks)} track(s): {_share(found.class_movies, census['n_clustered'])}"
    )
    if found.caveat:
        out.append("")
        out += _flagged_lines([found.caveat], indent="  ")
    out.append("")
    out.append(
        f"  {len(found.clusters)} cluster(s); {found.result.n_noise} clip(s) "
        f"({found.result.noise_frac:.0%}) resemble nothing else in this class, which "
        "is a finding rather than a failure"
    )

    for cluster in found.clusters:
        out.append("")
        out.append(f"  -- cluster {cluster.label}: {cluster.size} clip(s), "
                   f"{_stability_text(cluster)}")
        out.append(f"     movies  {cluster.movies.n} movie(s): "
                   f"{_share(cluster.movies.counts, cluster.size)}")
        out.append(f"     tracks  {cluster.tracks.n} track(s); the largest is "
                   f"{cluster.tracks.top} with {cluster.tracks.counts[cluster.tracks.top]} "
                   f"({cluster.tracks.top_frac:.0%})")
        out += _flagged_lines(_batch_note(cluster), indent="     ")
        out.append(f"     medoid  {cluster.medoid_id}")
        out.append("     watch   " + ", ".join(
            r["clip_id"] + (" (audit)" if r["audit"] else "")
            for r in cluster.representatives
        ))

    # The summary states the flags whether or not the class-level caveat fired.
    # The caveat kills the MOVIE comparison, but a cluster that is one animal is
    # one animal however few movies the class spans.
    flagged = found.flagged
    out.append("")
    if flagged:
        out += _flagged_lines([
            f"{len(flagged)} of {len(found.clusters)} cluster(s) may be a batch effect "
            f"rather than a behaviour: {', '.join(str(c.label) for c in flagged)}. That "
            "is the classic way this technique embarrasses people, and no stability "
            "score can catch it - read the notes above before annotating anything."
        ], indent="  ")
    if found.caveat:
        out.append(
            "  the movie-level comparison is uninformative on this project - see the "
            "note at the top"
        )
    elif not flagged:
        out.append(
            f"  no cluster is more than {found.settings['batch_fraction']:.0%} one "
            "movie or one track, so none of them is obviously a recording rather than "
            "a behaviour. That is a check on the crudest failure, not a guarantee."
        )
    return "\n".join(out)


def _flagged_lines(paragraphs: list[str], indent: str) -> list[str]:
    """``** `` on the first line of each paragraph, aligned continuations after it.

    The marker on every line would turn the paragraph into a wall.
    """
    out: list[str] = []
    for paragraph in paragraphs:
        for i, line in enumerate(_wrap(paragraph, width=72 - len(indent))):
            out.append(indent + ("** " if i == 0 else "   ") + line)
    return out


def _wrap(text: str, width: int = 74) -> list[str]:
    """Wrap to a terminal width without importing textwrap for one call."""
    words, lines, current = text.split(), [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


# --------------------------------------------------------------------------- #
# the durable answer
# --------------------------------------------------------------------------- #
def _slug(text: str) -> str:
    """A class name reduced to something safe to put in a filename.

    Class names are free text from ``config.yaml``, so a slash would write the
    report into a directory nobody meant and ``..`` outside the work directory.
    """
    out = re.sub(r"[^A-Za-z0-9_-]+", "-", str(text)).strip("-")
    if not out:
        raise ValueError(
            f"{text!r} has no characters that can go in a filename, so the report has "
            "nowhere to be written; pass --out with an explicit path"
        )
    return out


def report_path(cfg: Config, group: str, class_name: str) -> Path:
    """Where this run's answer is written, beside the other work-dir artefacts."""
    return cfg.paths.work_dir / REPORT_TEMPLATE.format(
        group=_slug(group), klass=_slug(class_name)
    )


def _plain(value):
    """Python scalars ``json`` will take, with non-finite floats becoming ``null``.

    The bare ``NaN`` token ``json.dumps`` would otherwise emit is not valid JSON
    and does not read back the same everywhere; ``null`` survives the round trip
    and keeps the distinction that an unmeasured stability is absent, not 0.0.
    ``allow_nan=False`` at the dump makes anything this misses fail loudly.
    """
    if isinstance(value, (str, bool, type(None))):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(v) for v in value]
    return str(value)


def to_payload(found: Discovery) -> dict:
    """The written form: everything a later step needs, and nothing it must guess.

    ``member_ids`` is the load-bearing field, since the step that directs
    annotation at a discovered cluster works from it, so it is written in full
    rather than summarised; so are ``noise_ids``. ``settings`` and ``columns``
    travel with the answer because two reports are not comparable unless they
    match, and ``composition`` travels with ``flagged`` written per cluster so a
    reader that only walks the clusters cannot lose the flag.
    """
    return _plain({
        "schema": SCHEMA,
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "group": found.group,
        "class": found.class_name,
        "column_spec": found.column_spec,
        "columns": found.columns,
        "settings": found.settings,
        "census": found.census,
        "class_composition": {"movie": found.class_movies, "track": found.class_tracks},
        "caveat": found.caveat,
        "n_clusters": len(found.clusters),
        "n_noise": found.result.n_noise,
        "noise_frac": found.result.noise_frac,
        "noise_ids": found.result.noise_ids,
        "clusters": [{
            "label": c.label,
            "size": c.size,
            "stability": c.stability,
            "n_trials": c.n_trials,
            "stable": c.stable,
            "flagged": c.flagged,
            "batch_note": _batch_note(c),
            "composition": {
                kind: {
                    "counts": comp.counts, "top": comp.top,
                    "top_frac": comp.top_frac, "class_frac": comp.class_frac,
                    "flagged": comp.flagged,
                }
                for kind, comp in (("movie", c.movies), ("track", c.tracks))
            },
            "medoid_id": c.medoid_id,
            "representatives": c.representatives,
            "member_ids": c.member_ids,
        } for c in found.clusters],
    })


def write_report(found: Discovery, path: Path) -> Path:
    """Write the answer as JSON. Returns the path, for the caller to print."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_payload(found), indent=2, allow_nan=False), encoding="utf-8"
    )
    return path


def load_report(path) -> dict:
    """Read a written report back, refusing one this build does not understand.

    Checked here rather than further in, where a missing key would blame the data
    rather than the version. The refusal names both versions.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no discovery report at {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "schema" not in payload:
        raise ValueError(
            f"{path} is not a discovery report: it carries no schema stamp. Reports "
            f"written by this module start with {SCHEMA!r}"
        )
    if payload["schema"] != SCHEMA:
        raise ValueError(
            f"{path} was written with schema {payload['schema']!r} and this build "
            f"reads {SCHEMA!r}. Re-run the discovery rather than reading it anyway - "
            "the fields may have the same names and different meanings"
        )
    return payload


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _load_tables(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The two tables, or a refusal naming the command that builds the missing one.

    The feature table is READ, never rebuilt: rebuilding is minutes of work over
    every movie and belongs to the training path.
    """
    path = cfg.paths.work_dir / CLIP_FEATURES_FILE
    if not path.exists():
        raise ValueError(
            f"no feature table at {path}. Build one with `uv run suggest.py "
            "--features-only`, or from the Model page; this command reads it rather "
            "than rebuilding it, which would be minutes of work over every movie"
        )
    if not cfg.paths.csv.exists():
        raise ValueError(
            f"no annotations at {cfg.paths.csv}; there are no labels to look inside"
        )
    labels = pd.read_csv(cfg.paths.csv)
    if labels.empty:
        raise ValueError(f"{cfg.paths.csv} has no rows; there are no labels to look inside")
    return pd.read_parquet(path), labels


def _list_classes(cfg: Config, features: pd.DataFrame, labels: pd.DataFrame) -> None:
    """What could be looked inside, and how many clips each candidate has."""
    have = set(features["clip_id"].astype(str))
    for cat in cfg.categories:
        if cat.is_multi:
            # A multi-select label is a SET, so "the clips of one class" is not a
            # partition of the group and the majority-class default has no meaning.
            print(f"{cat.name}: multi-select, not supported")
            continue
        if cat.name not in labels.columns:
            print(f"{cat.name}: no column in the CSV yet")
            continue
        rows = _labelled(labels, cat.name)
        counts = rows[cat.name].value_counts().to_dict()
        if not counts:
            print(f"{cat.name}: no labels yet")
            continue
        # The same tie rule `class_table` applies, so the line marked here is the
        # class a bare `--class`-less run would actually take.
        default = max(sorted(counts), key=lambda name: counts[name])
        print(f"{cat.name}:")
        for name, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            usable = int(rows[rows[cat.name] == name]["clip_id"].astype(str).isin(have).sum())
            mark = " <- majority, the default" if name == default else ""
            print(f"   {name:<20} {n:>5} labelled, {usable:>5} with a feature row{mark}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="discover",
        description="Look inside one label for sub-behaviours, and check whether "
                    "each group found is a behaviour or a recording.",
    )
    p.add_argument("--config", default=None)
    p.add_argument("--list", action="store_true",
                   help="list the classes that could be looked inside, and stop")
    p.add_argument("--group", default=None,
                   help="category group holding the class (default: the first "
                        "single-choice group in config order)")
    p.add_argument("--class", dest="class_name", default=None,
                   help="class to look inside (default: the majority class, the one "
                        "most likely to be hiding structure)")
    p.add_argument("--columns", default=DEFAULT_COLUMNS,
                   help="feature columns to cluster in: a group name "
                        f"({', '.join(FEATURE_GROUPS)}), 'all', a prefix like "
                        "'wave_amp_*', explicit names, or any comma-separated mix "
                        f"(default: {DEFAULT_COLUMNS!r}, the 25 wave_* columns)")
    p.add_argument("--min-cluster-size", type=int, default=DEFAULT_MIN_CLUSTER_SIZE,
                   help="smallest group that counts as a discovery; this, not the "
                        "stability score, is what stops an unstructured class being "
                        f"carved up (default: {DEFAULT_MIN_CLUSTER_SIZE})")
    p.add_argument("--min-samples", type=int, default=None,
                   help="how conservative the density estimate is (default: "
                        "--min-cluster-size)")
    p.add_argument("--scaling", choices=SCALINGS, default="robust")
    p.add_argument("--resamples", type=int, default=25,
                   help="stability trials; 0 skips the measurement (default: 25)")
    p.add_argument("--subsample-frac", type=float, default=0.8)
    p.add_argument("--representatives", type=int, default=8,
                   help="clips named per cluster, medoid included (default: 8)")
    p.add_argument("--seed", type=int, default=None,
                   help="fixes the resampling (default: suggest.seed from the config)")
    p.add_argument("--batch-frac", type=float, default=BATCH_FRACTION,
                   help="share of one movie or track above which a cluster is called "
                        f"a possible batch effect (default: {BATCH_FRACTION})")
    p.add_argument("--exclude-movie", action="append", default=[], metavar="MOVIE_ID",
                   help="leave a movie out entirely; the follow-up to a flagged "
                        "cluster, repeatable")
    p.add_argument("--exclude-audit", action="store_true",
                   help="leave the suggester's audit clips out of the search")
    p.add_argument("--out", default=None,
                   help="where to write the report (default: subclusters_<group>_"
                        "<class>.json in the work directory)")
    args = p.parse_args(argv)
    cfg = load_config(args.config)

    try:
        features, labels = _load_tables(cfg)
        if args.list:
            _list_classes(cfg, features, labels)
            return 0

        group = args.group
        if group is None:
            # The first single-choice group, which is the one the suggester ranks
            # on, so a project with several groups gets the same default here.
            single = [c.name for c in cfg.categories if not c.is_multi]
            if not single:
                raise ValueError(
                    "this project has only multi-select category groups. A "
                    "multi-select label is a SET, so 'the clips of one class' is not "
                    "a partition and the question this asks is not well posed for one"
                )
            group = single[0]

        found = discover(
            features, labels, group, args.class_name,
            columns=args.columns,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            scaling=args.scaling,
            seed=cfg.suggest.seed if args.seed is None else args.seed,
            n_resamples=args.resamples,
            subsample_frac=args.subsample_frac,
            n_representatives=args.representatives,
            batch_fraction=args.batch_frac,
            exclude_movies=args.exclude_movie,
            audit_of=(lambda clip: is_audit_clip(clip, cfg)),
            exclude_audit=args.exclude_audit,
        )
    except ValueError as exc:
        print(f"discover: {exc}", file=sys.stderr)
        return 2

    print(format_report(found))
    out = Path(args.out) if args.out else report_path(cfg, found.group, found.class_name)
    print(f"\nwritten to {write_report(found, out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
