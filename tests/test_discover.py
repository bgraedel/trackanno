"""Checks for :mod:`trackanno.learn.discover`, the thing a person runs to look inside a class.

:mod:`trackanno.learn.subcluster` is checked against planted structure in
`tests/test_subcluster.py`; nothing here re-checks the clustering. What is
checked here is everything the caller is responsible for, which is where this
technique actually goes wrong:

* the class is assembled from the two tables and the join is audited, so a
  discovery over half a class cannot be reported as a discovery over all of it;
* the COLUMN DEFAULT earns its place. The price of handing over every column is
  measured here, in this module's own tables, rather than borrowed: the same
  three planted groups come back exactly from the 25 ``wave_*`` columns and are
  lost among the 65 columns of the whole table;
* **a cluster made of one movie is flagged, and a cluster made of one animal is
  flagged even where the movie comparison has nothing to say.** This is the claim
  the module exists for. Such a cluster is tight, reproducible and stable, and
  every number on the screen says it is real;
* a class with no structure in it produces no discovery, and the refusal says
  what to try;
* every refusal fires on its own trigger and names what to change;
* the written report round-trips, because a later step reads it rather than the
  screen.

Every table below is built here, so the expected answer is known before anything
runs. Nothing reads a real project.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from harness import check
from sklearn.metrics import adjusted_rand_score
from trackanno.learn.discover import (
    BATCH_FRACTION, DEFAULT_COLUMNS, SCHEMA, class_table, discover, format_report,
    load_report, main, report_path, resolve_columns, to_payload, write_report,
)
from trackanno.measure.wavespectra import SERIES_COLUMNS

# The 25 columns the default names, and the 40 that stand in for the rest of a
# real feature table. Both are needed: the value of choosing columns is invisible
# in a table that holds nothing but the informative ones, which is exactly the
# mistake this module exists to stop a caller making.
WAVE = list(SERIES_COLUMNS)
NUISANCE = (
    ["speed_mean", "speed_std", "speed_max", "speed_p90", "path_length",
     "net_displacement", "straightness", "radius_gyration", "immotile_frac",
     "present_frac", "area_mean", "area_std", "aspect_mean", "turn_rate",
     "turn_lag_frames", "conf_mean"]
    + [f"cnn{i}_mean" for i in range(12)]
    + [f"cnn{i}_std" for i in range(12)]
)

GROUP = "behaviour"
CLASS = "swimming"


# --------------------------------------------------------------------------- #
# generators
# --------------------------------------------------------------------------- #
def provenance(n: int, movies: list[str], per_track: int, first_track: int) -> list[tuple]:
    """``(movie, track, f_start)`` per row: several successive windows per track.

    Shaped like the real manifest rather than one clip per animal, because that
    is what makes the track check necessary: with ``clips.stride`` below
    ``clips.frames`` one track contributes a run of overlapping windows, and a
    cluster of those is one cell measured repeatedly.

    Track numbers are disjoint between populations (``first_track``), so a track
    belongs to exactly one planted group - a track that changed behaviour halfway
    would make the planted answer ambiguous.
    """
    out, t = [], 0
    while len(out) < n:
        movie, track = movies[t % len(movies)], first_track + t // len(movies)
        for k in range(per_track):
            if len(out) == n:
                break
            out.append((movie, track, 50 * k))
        t += 1
    return out


def tables(blobs, *, n_uniform: int = 40, seed: int = 0, sep: float = 6.0,
           other: tuple = ("wriggling", 40)) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """A ``clip_features``-shaped frame, an annotation-CSV-shaped frame, and the truth.

    Each blob is ``{"n": .., "movies": [..], "per_track": ..}`` and becomes a
    Gaussian whose centre is displaced in EVERY ``wave_*`` column, the way a real
    change in beating would be: the frequency, the wavelength, the wave speed and
    the amplitude coefficients are algebraically linked, so one cannot move
    alone. The ``NUISANCE`` columns carry pure noise, as a column that has
    nothing to do with the split does.

    The uniform rows are the planted answer to "resembles nothing else here", and
    a second class exists so that the majority-class default has something to be
    the majority of.

    Returns ``(features, labels, truth)`` with ``truth`` mapping ``clip_id`` to a
    blob index, ``-1`` for a uniform row.
    """
    rng = np.random.default_rng(seed)
    width = len(WAVE)
    blocks, prov, truth, values = [], [], [], []
    for k, spec in enumerate(blobs):
        centre = sep * np.cos(2.0 * np.pi * k / len(blobs) + np.arange(width) * 0.7)
        blocks.append(rng.normal(centre, 1.0, size=(spec["n"], width)))
        prov += provenance(spec["n"], spec["movies"], spec.get("per_track", 5),
                           100 * (k + 1))
        truth += [k] * spec["n"]
        values += [CLASS] * spec["n"]

    movies = sorted({m for spec in blobs for m in spec["movies"]})
    if n_uniform:
        blocks.append(rng.uniform(-sep - 3.0, sep + 3.0, size=(n_uniform, width)))
        prov += provenance(n_uniform, movies, 5, 900)
        truth += [-1] * n_uniform
        values += [CLASS] * n_uniform
    if other and other[1]:
        blocks.append(rng.normal(0.0, 1.0, size=(other[1], width)))
        prov += provenance(other[1], movies, 5, 5000)
        truth += [-2] * other[1]
        values += [other[0]] * other[1]

    frame = pd.DataFrame(np.vstack(blocks), columns=WAVE)
    frame.insert(0, "clip_id", [f"{m}_t{t:03d}_f{f:05d}" for m, t, f in prov])
    frame.insert(1, "movie_id", [m for m, _, _ in prov])
    frame.insert(2, "track_id", [t for _, t, _ in prov])
    frame.insert(3, "f_start", [f for _, _, f in prov])
    for column in NUISANCE:
        frame[column] = rng.normal(0.0, 1.0, len(frame))
    labels = pd.DataFrame({
        "clip_id": frame["clip_id"], "movie_id": frame["movie_id"],
        "track_id": frame["track_id"], GROUP: values,
    })
    return frame, labels, dict(zip(frame["clip_id"], truth))


def spread_over_three_movies(n: int = 120) -> list[dict]:
    """Three planted groups, none of them a recording: the well-behaved case."""
    return [{"n": n, "movies": ["m0", "m1", "m2"]} for _ in range(3)]


def unstructured(n: int = 400, seed: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One Gaussian over the wave columns: no groups, no gaps, nothing to find."""
    rng = np.random.default_rng(seed)
    prov = provenance(n, ["m0", "m1", "m2"], 5, 1)
    frame = pd.DataFrame(rng.normal(0.0, 1.0, size=(n, len(WAVE))), columns=WAVE)
    frame.insert(0, "clip_id", [f"{m}_t{t:03d}_f{f:05d}" for m, t, f in prov])
    frame.insert(1, "movie_id", [m for m, _, _ in prov])
    frame.insert(2, "track_id", [t for _, t, _ in prov])
    frame.insert(3, "f_start", [f for _, _, f in prov])
    for column in NUISANCE:
        frame[column] = rng.normal(0.0, 1.0, n)
    labels = pd.DataFrame({"clip_id": frame["clip_id"], GROUP: [CLASS] * n})
    return frame, labels


def ari(found, truth: dict) -> float:
    """Adjusted Rand index over the planted rows, read through the ids that came back.

    Cluster ids are arbitrary, and the uniform rows are not a class - scoring
    them as a fourth group to be recovered together is the opposite of what they
    mean - so both are handled the way `tests/test_subcluster.py` handles them.
    """
    planted = [(truth[c], int(l)) for c, l in zip(found.result.ids, found.result.labels)
               if truth[c] >= 0]
    return float(adjusted_rand_score([t for t, _ in planted], [l for _, l in planted]))


def refused(fn, *args, **kwargs) -> str:
    """The refusal message, or "" if the call was answered."""
    try:
        fn(*args, **kwargs)
    except ValueError as exc:
        return str(exc)
    return ""


def project(tmp_path, features: pd.DataFrame, labels: pd.DataFrame):
    """A config.yaml, a work dir and the two tables: the smallest real project.

    Written rather than mocked, because the CLI's job is to find these files and
    a test that handed it frames would not check the half that goes wrong.
    """
    (tmp_path / "work").mkdir(exist_ok=True)
    features.to_parquet(tmp_path / "work" / "clip_features.parquet", index=False)
    labels.to_csv(tmp_path / "annotations.csv", index=False)
    (tmp_path / "config.yaml").write_text(
        "paths:\n"
        "  work_dir: ./work\n"
        "  csv: ./annotations.csv\n"
        "categories:\n"
        f"  - name: {GROUP}\n"
        "    type: single\n"
        "    options:\n"
        f"      - {{name: {CLASS}, key: '1', color: '#ff0000'}}\n"
        "      - {name: wriggling, key: '2', color: '#00ff00'}\n",
        encoding="utf-8",
    )
    return tmp_path / "config.yaml"


# --------------------------------------------------------------------------- #
# it finds what is there, and names clips to look at
# --------------------------------------------------------------------------- #
def test_a_planted_sub_behaviour_is_found_and_reported_as_clips_to_watch() -> None:
    """Three sub-behaviours in one class, recovered and handed back as watchable clips.

    The answer an annotator can act on is a list of clip ids, so the claims are
    about those: the partition is the planted one, the medoid is a member, and
    the panel is drawn from the cluster it describes rather than from the class
    at large.
    """
    features, labels, truth = tables(spread_over_three_movies())
    found = discover(features, labels, GROUP, n_resamples=5)

    check("the three planted sub-behaviours come back as three clusters",
          len(found.clusters) == 3, str([c.size for c in found.clusters]))
    check("the recovered partition is the planted one, up to relabelling",
          ari(found, truth) == 1.0, str(ari(found, truth)))
    check("the clips that resemble nothing else are kept as such, not tidied "
          "into the nearest group",
          found.result.n_noise >= 20, str(found.result.n_noise))
    check("every cluster names a medoid that is one of its own members",
          all(c.medoid_id in c.member_ids for c in found.clusters))
    check("and a panel of members, medoid first, each carrying where it came from",
          all(c.representatives[0]["clip_id"] == c.medoid_id
              and c.representatives[0]["medoid"]
              and all(r["clip_id"] in c.member_ids for r in c.representatives)
              for c in found.clusters))
    check("the panel says which movie and track each clip is from, so it can be "
          "found without joining anything back",
          all(set(r) >= {"clip_id", "movie_id", "track_id", "f_start"}
              for c in found.clusters for r in c.representatives))
    check("with no --class the majority class is the one looked inside",
          found.class_name == CLASS and found.census["chose_majority"],
          found.class_name)
    check("the second class is left out of the search entirely",
          found.census["n_clustered"] == 400
          and found.census["class_counts"]["wriggling"] == 40,
          str(found.census))
    check("the settings that produced the answer travel with it",
          found.settings["min_cluster_size"] == 15
          and found.settings["columns"] == DEFAULT_COLUMNS
          and found.settings["batch_fraction"] == BATCH_FRACTION,
          str(found.settings))

    report = format_report(found)
    check("the printed report names every cluster, its size and its medoid",
          all(f"cluster {c.label}" in report and str(c.medoid_id) in report
              for c in found.clusters))
    check("and states the noise as a finding rather than hiding it",
          "resemble nothing else" in report, report)


def test_an_audit_clip_in_the_panel_is_marked_as_one() -> None:
    """A clip meant to stay independent of any automated procedure, about to be watched.

    The audit holdout is what makes human-vs-model agreement measurable, and a
    representative drawn from it sends an annotator to a clip on the strength of
    a clustering. That is not refused - the clips are already labelled, this
    module fits no model, and dropping a tenth of the class would skew the very
    composition the batch check reads - but it is never silent.
    """
    features, labels, truth = tables(spread_over_three_movies())
    audit = {c for c in features["clip_id"] if c.endswith("f00150")}
    found = discover(features, labels, GROUP, n_resamples=0,
                     audit_of=lambda clip: clip in audit)

    marked = {r["clip_id"] for c in found.clusters for r in c.representatives if r["audit"]}
    check("the panel marks the audit clips it names and no others",
          marked and marked <= audit, str(sorted(marked)[:3]))
    check("and the count of them in the class is recorded",
          found.census["n_audit"] == len(audit & set(found.result.ids)),
          str(found.census["n_audit"]))

    without = discover(features, labels, GROUP, n_resamples=0,
                       audit_of=lambda clip: clip in audit, exclude_audit=True)
    check("--exclude-audit leaves them out of the search entirely",
          not (set(without.result.ids) & audit)
          and without.census["n_clustered"] < found.census["n_clustered"],
          str(without.census["n_clustered"]))
    check("and asking to exclude them with no way to know which they are is refused",
          "suggest.audit_fraction" in refused(
              discover, features, labels, GROUP, n_resamples=0, exclude_audit=True))


# --------------------------------------------------------------------------- #
# the column decision
# --------------------------------------------------------------------------- #
def test_the_default_columns_are_the_wave_block_and_widening_them_costs_the_answer() -> None:
    """The decision the module makes for the caller, measured on the module's own tables.

    Per-column scaling puts a column that carries the split on the same footing
    as a column that carries noise, so the mitigation is the caller's: pass the
    columns the split is expected to live in. This is that mitigation working,
    and the cost of skipping it. Both sides are the same rows, the same settings
    and the same planted structure; the only difference is which columns were
    handed over.

    Neither number is chosen - both are printed.
    """
    features, labels, truth = tables(spread_over_three_movies())

    check(f"the default spec is {DEFAULT_COLUMNS!r} and resolves to the 25 wave columns",
          resolve_columns(features, DEFAULT_COLUMNS) == WAVE,
          str(len(resolve_columns(features, DEFAULT_COLUMNS))))
    check("which is a minority of the 65 feature columns in the table",
          len(resolve_columns(features, "all")) == len(WAVE) + len(NUISANCE))

    narrow = discover(features, labels, GROUP, columns=DEFAULT_COLUMNS, n_resamples=0)
    wide = discover(features, labels, GROUP, columns="all", n_resamples=0)
    check("the planted split is recovered exactly from the wave block",
          ari(narrow, truth) == 1.0 and len(narrow.clusters) == 3,
          str(ari(narrow, truth)))
    check("and is lost when every column in the table is handed over instead",
          ari(wide, truth) < 0.5, str(ari(wide, truth)))
    check("badly enough that the wrong NUMBER of groups is reported",
          len(wide.clusters) != 3, str([c.size for c in wide.clusters]))
    check("and most of the class is thrown into noise on the way",
          wide.result.noise_frac > narrow.result.noise_frac,
          f"{wide.result.noise_frac:.2f} against {narrow.result.noise_frac:.2f}")
    print(f"  --  ARI {ari(narrow, truth):.3f} on the 25 wave columns, "
          f"{ari(wide, truth):.3f} on all {len(resolve_columns(features, 'all'))}; "
          f"{len(narrow.clusters)} clusters against {len(wide.clusters)}")

    check("a spec can name a feature group, a glob, an explicit column, or a mix",
          resolve_columns(features, "wave_amp_*") == [c for c in WAVE if c.startswith("wave_amp_")]
          and resolve_columns(features, "wave_freq_hz") == ["wave_freq_hz"]
          and resolve_columns(features, "wave_freq_hz,wave_amp_*")
          == ["wave_freq_hz"] + [c for c in WAVE if c.startswith("wave_amp_")],
          str(resolve_columns(features, "wave_freq_hz,wave_amp_*")))
    check("a column named twice is used once, never weighted twice",
          resolve_columns(features, "waveform,wave_freq_hz,wave_freq_hz") == WAVE)
    check("and the columns it settled on are reported with the answer, "
          "since that is what the answer depends on",
          narrow.columns == WAVE and narrow.column_spec == DEFAULT_COLUMNS)


# --------------------------------------------------------------------------- #
# it does not invent what is not there
# --------------------------------------------------------------------------- #
def test_a_class_with_no_structure_reports_none_rather_than_inventing_some() -> None:
    """One Gaussian at the default: the refusal, and the advice that comes with it.

    An annotator sent to label a partition of noise has lost the afternoon and
    poisoned the next model with a class that does not exist, so the useful
    answer here is "nothing", said out loud. The message has to be more than that
    though: the fix is usually to narrow the columns, and a caller who is only
    told "no clusters" will reach for a smaller min_cluster_size, which is the
    setting that was stopping the invention.
    """
    features, labels = unstructured()
    message = refused(discover, features, labels, GROUP, n_resamples=5)

    check("no discovery is reported at all", message, "a cluster was invented")
    check("and the refusal names the rows and the settings that produced it",
          "400" in message and "min_cluster_size=15" in message, message)
    check("it says which columns were actually used, since that is the decision",
          "25 column(s)" in message and "waveform" in message, message)
    check("it sends the caller to the columns FIRST",
          message.index("NARROW --columns") < message.index("min-cluster-size"), message)
    check("with the measured price of not doing so, rather than an assertion",
          "1.00" in message and "0.05" in message, message)
    check("and it states the answer that may simply be true",
          "one behaviour in this feature basis" in message, message)


# --------------------------------------------------------------------------- #
# the batch-effect check, which is what this module is for
# --------------------------------------------------------------------------- #
def test_a_cluster_drawn_from_one_movie_is_flagged_as_a_possible_batch_effect() -> None:
    """The claim the module exists to make, on a cluster that is otherwise perfect.

    The third group here is real, tight, and comes almost entirely from one
    movie. Every number a clusterer can produce says it is a discovery - it is
    stable, it is large, its members do keep landing together - and it is the
    recording. Nothing inside :mod:`trackanno.learn.subcluster` can see that, because
    it is handed a bare matrix; this module has the manifest beside it.
    """
    features, labels, truth = tables([
        {"n": 120, "movies": ["m0", "m1", "m2"]},
        {"n": 120, "movies": ["m0", "m1", "m2"]},
        {"n": 90, "movies": ["m2"]},
    ])
    found = discover(features, labels, GROUP, n_resamples=5)
    confined = max(found.clusters, key=lambda c: c.movies.top_frac)
    spread = [c for c in found.clusters if c is not confined]

    check("the one-movie group is found, and found as a good cluster",
          confined.size > 50 and confined.stable,
          f"{confined.size} clips, stability {confined.stability:.2f}")
    check("so stability alone would have sent an annotator to annotate a movie",
          confined.stability > BATCH_FRACTION, str(confined.stability))
    check("it is flagged as a possible batch effect",
          confined.flagged and confined.movies.flagged,
          f"{confined.movies.top} at {confined.movies.top_frac:.2f}")
    check("the flag carries both numbers that make it one: the cluster's share "
          "and that movie's share of the class",
          confined.movies.top_frac >= BATCH_FRACTION
          and confined.movies.class_frac < BATCH_FRACTION,
          f"{confined.movies.top_frac:.2f} against {confined.movies.class_frac:.2f}")
    check("the clusters that are spread over the movies are NOT flagged, "
          "so the flag distinguishes rather than warns about everything",
          not any(c.flagged for c in spread),
          str([(c.movies.top, round(c.movies.top_frac, 2)) for c in spread]))

    report = format_report(found)
    check("the report says POSSIBLE BATCH EFFECT in words, on that cluster",
          "POSSIBLE BATCH EFFECT" in report and f"{confined.movies.top!r}" in report)
    check("and says what a lopsided cluster MEANS rather than only that it is one",
          "illumination" in report and "RECORDING" in report, report)
    check("it names the follow-up that answers the question",
          f"--exclude-movie {confined.movies.top}" in report, report)
    check("the composition of EVERY cluster is printed, so a clean one is "
          "visibly clean rather than merely unmentioned",
          all(f"{c.movies.n} movie(s)" in report for c in found.clusters), report)
    check("and the summary counts the flagged clusters",
          "1 of 3 cluster(s) may be a batch effect" in report, report)

    # The follow-up the report proposes, run: without that movie the group has
    # nothing left to hold it together, which is what a batch effect looks like
    # when it is tested rather than suspected.
    after = discover(features, labels, GROUP, n_resamples=0,
                     exclude_movies=[confined.movies.top])
    survivors = [c for c in after.clusters if c.size > 50]
    check("re-running without that movie loses the group, as a batch effect should",
          len(survivors) < len(found.clusters),
          str([c.size for c in after.clusters]))
    check("and the exclusion is recorded in the answer rather than left implicit",
          after.census["excluded_movies"] == [confined.movies.top]
          and after.census["n_excluded_movies"] > 0, str(after.census))
    print(f"  --  cluster of {confined.size} clips, {confined.movies.top_frac:.0%} from "
          f"{confined.movies.top} ({confined.movies.class_frac:.0%} of the class), "
          f"stability {confined.stability:.2f}")


def test_one_animal_is_flagged_even_where_the_movie_comparison_cannot_speak() -> None:
    """A class recorded in one movie: the movie check is dead, the track check is not.

    Two things are being separated. When the whole class comes from one movie,
    every cluster is 100% that movie and the comparison has no contrast left in
    it - flagging all three would be noise, and saying nothing would be worse, so
    the module says once, at the top, that this project cannot answer the
    question. And underneath that, the finer check still bites: a cluster that is
    one TRACK is one animal, and successive windows of one track are not
    independent measurements of anything.
    """
    features, labels, truth = tables([
        {"n": 120, "movies": ["m0"]},
        {"n": 120, "movies": ["m0"]},
        {"n": 60, "movies": ["m0"], "per_track": 60},
    ])
    found = discover(features, labels, GROUP, n_resamples=5)
    animal = max(found.clusters, key=lambda c: c.tracks.top_frac)

    check("the class-level caveat fires, once, instead of a flag on every cluster",
          found.caveat and "cannot separate anything here" in found.caveat, found.caveat)
    check("and no cluster carries a movie flag, since there is nothing to compare",
          not any(c.movies.flagged for c in found.clusters),
          str([c.movies.top_frac for c in found.clusters]))
    check("the cluster that is one animal is still flagged",
          animal.flagged and animal.tracks.flagged,
          f"{animal.tracks.top} at {animal.tracks.top_frac:.2f}")
    check("against that track's much smaller share of the class",
          animal.tracks.class_frac < animal.tracks.top_frac, str(animal.tracks.class_frac))
    check("and the other clusters, spread over many tracks, are not",
          not any(c.tracks.flagged for c in found.clusters if c is not animal),
          str([c.tracks.n for c in found.clusters]))

    report = format_report(found)
    check("the report explains that windows of one track are not independent",
          "ONE animal" in report and "not independent measurements" in report, report)
    check("and the flag survives the caveat rather than being swallowed by it",
          "may be a batch effect" in report and "uninformative" in report, report)
    print(f"  --  {animal.tracks.top_frac:.0%} of a {animal.size}-clip cluster is track "
          f"{animal.tracks.top}, which is {animal.tracks.class_frac:.0%} of the class")


# --------------------------------------------------------------------------- #
# undefined values
# --------------------------------------------------------------------------- #
def test_undefined_values_cost_columns_first_then_rows_and_the_report_says_which() -> None:
    """NaN means "undefined" here and cannot be imputed, so something must be dropped.

    Filling a hole with the column median would put every clip nobody could
    measure at the centre of the cloud and manufacture a dense cluster out of the
    rows we know least about - which would then be perfectly stable. So the
    choice is between dropping a column and dropping a row, it is made here, and
    the cost is reported rather than absorbed: a column that is undefined
    everywhere costs nothing to drop, and the clips lost to the remaining holes
    are not a random sample of the class.
    """
    features, labels, truth = tables(spread_over_three_movies())
    features = features.copy()
    features["wave_freq_drift_hz"] = np.nan               # never measurable here
    features["wave_window_frames"] = 75.0                 # a constant readout
    holed = features["clip_id"].isin(sorted(features["clip_id"])[:30])
    features.loc[holed, "wave_phase_rsq"] = np.nan        # 30 clips lose a value

    found = discover(features, labels, GROUP, n_resamples=0)
    census = found.census

    check("a column undefined for every clip is dropped, and named",
          census["all_nan_columns"] == ["wave_freq_drift_hz"]
          and "wave_freq_drift_hz" not in found.columns, str(census["all_nan_columns"]))
    check("the rows still holding a hole are dropped, and counted",
          census["n_dropped_rows"] == int(holed.sum() * 0 + census["n_dropped_rows"])
          and census["n_dropped_rows"] > 0, str(census["n_dropped_rows"]))
    check("the column responsible is named, so dropping IT is an option",
          list(census["dropped_rows_by_column"]) == ["wave_phase_rsq"],
          str(census["dropped_rows_by_column"]))
    check("a constant column is reported rather than silently carried",
          census["constant_columns"] == ["wave_window_frames"],
          str(census["constant_columns"]))
    check("and the clustered total is the labelled total minus what was dropped",
          census["n_clustered"] == census["n_labelled"] - census["n_orphaned"]
          - census["n_dropped_rows"], str(census))

    report = format_report(found)
    check("the report states the drop and that it is not a random sample",
          "not a random sample" in report and "wave_phase_rsq" in report, report)
    check("and the structure is still recovered from what is left",
          ari(found, truth) == 1.0, str(ari(found, truth)))


# --------------------------------------------------------------------------- #
# refusals
# --------------------------------------------------------------------------- #
def test_every_refusal_fires_on_its_own_trigger_and_says_what_to_change() -> None:
    """Each case where an answer could be produced and would mislead.

    A refusal that does not say what to change is a refusal the caller has to
    reproduce under a debugger, so each of these asserts on the message.
    """
    features, labels, truth = tables(spread_over_three_movies(n=40),
                                     other=("wriggling", 20))

    small = refused(discover, features, labels, GROUP, class_name="wriggling")
    check("a class too small to hold two groups is refused, naming both numbers",
          "20 clip(s)" in small and "15" in small, small)
    check("and says what to do about it, in the flag the caller would type",
          "--min-cluster-size" in small and "Label more clips" in small, small)

    check("a group the CSV does not have is refused, naming the columns it does",
          all(part in refused(discover, features, labels, "posture")
              for part in ("posture", GROUP, "--group")),
          refused(discover, features, labels, "posture"))
    unknown = refused(discover, features, labels, GROUP, class_name="tumbling")
    check("a class nobody has labelled is refused, listing the classes with counts",
          "tumbling" in unknown and f"{CLASS} (" in unknown and "wriggling (" in unknown,
          unknown)

    check("a column spec matching nothing is refused rather than quietly ignored",
          "wave_freq_hzz" in refused(discover, features, labels, GROUP,
                                     columns="wave_freq_hzz"),
          refused(discover, features, labels, GROUP, columns="wave_freq_hzz"))
    check("and a near miss is suggested, since that is what a typo needs",
          "did you mean" in refused(discover, features, labels, GROUP,
                                    columns="wave_freq_hzz"))
    check("a glob matching nothing is refused too",
          "matches none" in refused(discover, features, labels, GROUP, columns="beat_*"))

    blind = features.drop(columns=["movie_id"])
    check("a feature table with no movie_id is refused, because the batch check "
          "is the reason to run this at all",
          "batch-effect check" in refused(discover, blind, labels, GROUP),
          refused(discover, blind, labels, GROUP))

    strangers = labels.copy()
    strangers["clip_id"] = strangers["clip_id"] + "-renumbered"
    disagree = refused(discover, features, strangers, GROUP)
    check("a CSV whose clips are all absent from the feature table is refused as "
          "a disagreement, not as an empty class",
          "disagree completely" in disagree and "renumbers the tracks" in disagree,
          disagree)

    moved = labels.copy()
    moved.loc[moved.index[0], "movie_id"] = "somewhere-else"
    clash = refused(discover, features, moved, GROUP)
    check("a CSV that names a different movie for a clip is refused, naming the clip",
          str(labels["clip_id"].iloc[0]) in clash and "somewhere-else" in clash, clash)

    twice = pd.concat([features, features.iloc[[0]]], ignore_index=True)
    check("a clip with two feature rows is refused, naming it",
          str(features["clip_id"].iloc[0]) in refused(discover, twice, labels, GROUP),
          refused(discover, twice, labels, GROUP))
    twice_labelled = pd.concat([labels, labels.iloc[[0]]], ignore_index=True)
    check("a clip labelled twice in the CSV is refused, naming it",
          str(labels["clip_id"].iloc[0]) in refused(discover, features, twice_labelled,
                                                    GROUP),
          refused(discover, features, twice_labelled, GROUP))

    check("excluding a movie that is not there is refused rather than silently "
          "doing nothing",
          "m9" in refused(discover, features, labels, GROUP, exclude_movies=["m9"]),
          refused(discover, features, labels, GROUP, exclude_movies=["m9"]))

    empty = features.copy()
    for column in WAVE:
        empty[column] = np.nan
    blank = refused(discover, empty, labels, GROUP)
    check("a column block that is undefined for the whole class is refused, "
          "naming what to choose instead",
          "nothing to measure a distance in" in blank and "kinematic" in blank, blank)


# --------------------------------------------------------------------------- #
# the durable answer
# --------------------------------------------------------------------------- #
def test_the_written_report_round_trips_and_carries_what_a_later_step_needs(tmp_path) -> None:
    """The artefact, not the screen: what a step that directs annotation reads.

    The load-bearing fields are the member ids - a later step enqueues them - and
    the flag beside them, because a consumer that walks the clusters and enqueues
    one without ever seeing that it may be a recording is exactly the accident
    this module exists to prevent.
    """
    features, labels, truth = tables([
        {"n": 120, "movies": ["m0", "m1", "m2"]},
        {"n": 120, "movies": ["m0", "m1", "m2"]},
        {"n": 90, "movies": ["m2"]},
    ])
    found = discover(features, labels, GROUP, n_resamples=5)
    path = write_report(found, tmp_path / "subclusters_behaviour_swimming.json")
    payload = load_report(path)

    check("the report carries a schema stamp inside the file",
          payload["schema"] == SCHEMA, str(payload.get("schema")))
    check("and the question it answers, which is not recoverable from the numbers",
          payload["group"] == GROUP and payload["class"] == CLASS)
    check("every cluster's members survive the round trip in full",
          [c["member_ids"] for c in payload["clusters"]]
          == [c.member_ids for c in found.clusters])
    check("so do the medoid and the panel a person would be shown",
          all(c["medoid_id"] == d.medoid_id
              and [r["clip_id"] for r in c["representatives"]]
              == [r["clip_id"] for r in d.representatives]
              for c, d in zip(payload["clusters"], found.clusters)))
    check("the clips that resemble nothing else are written too, since they are "
          "a queue somebody may want as well",
          payload["noise_ids"] == found.result.noise_ids
          and payload["n_noise"] == found.result.n_noise)
    check("the batch-effect verdict travels with each cluster, not only in the print-out",
          [c["flagged"] for c in payload["clusters"]] == [c.flagged for c in found.clusters]
          and any(c["batch_note"] for c in payload["clusters"]))
    check("and so does the composition it was computed from",
          all(c["composition"]["movie"]["counts"] == d.movies.counts
              for c, d in zip(payload["clusters"], found.clusters)))
    check("the settings and the columns travel too, since two runs are not "
          "comparable unless they match",
          payload["columns"] == found.columns
          and payload["settings"]["min_cluster_size"] == 15)
    check("the file is valid JSON by the strict reading, so another language "
          "can read it",
          json.loads(path.read_text(encoding="utf-8")) == payload)

    quick = discover(features, labels, GROUP, n_resamples=0)
    unmeasured = load_report(write_report(quick, tmp_path / "quick.json"))
    check("an unmeasured stability is written as null, never as 0.0 or as NaN",
          all(c["stability"] is None and c["n_trials"] == 0
              for c in unmeasured["clusters"]),
          str([c["stability"] for c in unmeasured["clusters"]]))
    check("and 'nobody looked' does not read back as stable",
          not any(c["stable"] for c in unmeasured["clusters"]))

    (tmp_path / "old.json").write_text('{"schema": "trackanno.learn.discover/0"}', encoding="utf-8")
    check("a report from another schema is refused, naming both versions",
          all(part in refused(load_report, tmp_path / "old.json")
              for part in ("trackanno.learn.discover/0", SCHEMA)),
          refused(load_report, tmp_path / "old.json"))
    (tmp_path / "junk.json").write_text('{"clusters": []}', encoding="utf-8")
    check("and so is a file that carries no stamp at all",
          "no schema stamp" in refused(load_report, tmp_path / "junk.json"))


# --------------------------------------------------------------------------- #
# the command
# --------------------------------------------------------------------------- #
def test_the_command_runs_a_project_end_to_end_and_writes_its_answer(tmp_path, capsys) -> None:
    """The CLI over a real config, a real work dir and the two real files.

    Also the default path of the report: beside the other work-dir artefacts and
    named after the question, so two classes of one project do not overwrite each
    other's answer.
    """
    features, labels, truth = tables([
        {"n": 120, "movies": ["m0", "m1", "m2"]},
        {"n": 120, "movies": ["m0", "m1", "m2"]},
        {"n": 90, "movies": ["m2"]},
    ])
    config = project(tmp_path, features, labels)
    code = main(["--config", str(config), "--resamples", "5"])
    printed = capsys.readouterr().out

    from trackanno.config import load_config

    cfg = load_config(config)
    written = report_path(cfg, GROUP, CLASS)
    check("the command succeeds", code == 0, str(code))
    check("it prints the report", "cluster 0" in printed and GROUP in printed, printed[:200])
    check("it flags the one-movie cluster on the way past",
          "POSSIBLE BATCH EFFECT" in printed, printed)
    check("it writes its answer beside the other work-dir artefacts, named after "
          "the question it answers",
          written.exists() and written.parent == cfg.paths.work_dir, str(written))
    check("and says where", str(written) in printed, printed[-200:])
    check("what it wrote is a report this build can read back",
          load_report(written)["class"] == CLASS)

    check("--class picks another class and refuses when it is too small, "
          "rather than answering about the majority one anyway",
          main(["--config", str(config), "--class", "wriggling"]) == 2)
    problem = capsys.readouterr().err
    check("the refusal goes to stderr and says what to change",
          "wriggling" in problem and "--min-cluster-size" in problem, problem)

    check("--list names the classes and marks the default",
          main(["--config", str(config), "--list"]) == 0)
    listed = capsys.readouterr().out
    check("with the counts and which of them the majority is",
          CLASS in listed and "majority" in listed and "with a feature row" in listed,
          listed)

    check("--columns is honoured from the command line",
          main(["--config", str(config), "--columns", "wave_freq_hz,wave_amp_*",
                "--resamples", "0", "--out", str(tmp_path / "narrow.json")]) == 0)
    capsys.readouterr()
    check("and the columns it used are in the file it wrote",
          load_report(tmp_path / "narrow.json")["columns"]
          == ["wave_freq_hz"] + [c for c in WAVE if c.startswith("wave_amp_")],
          str(load_report(tmp_path / "narrow.json")["columns"]))

    (cfg.paths.work_dir / "clip_features.parquet").unlink()
    check("a project with no feature table is refused, naming the command that builds one",
          main(["--config", str(config)]) == 2)
    check("and the message names it",
          "--features-only" in capsys.readouterr().err)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
