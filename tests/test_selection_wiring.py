"""Wiring :mod:`trackanno.select` into a project: the config, the pool, the batch.

``select.py`` decides HOW to rank and is tested on its own arithmetic in
``test_select.py``. Nothing there knows what a project is. These are the claims
about the join between the two, which is where the failures are quiet:

* the default config selects NOTHING. Every key here is new, and a new key that
  changes the order clips arrive in would silently invalidate the comparison
  between a project annotated before the upgrade and the same project after it;
* an incoherent combination - a selection rule with no audit holdout - is refused
  when the config loads, not warned about while it runs;
* every new key is REGISTERED. ``config._known`` ignores what it does not
  recognise, so an unregistered key is a setting that reads as applied and is
  not, and this file already carries the scar of that bug;
* the pool handed to ``select_batch`` is assembled from the model directly, holds
  the audit clips flagged rather than filtered, excludes what cannot be scored,
  and identifies a track by MOVIE AND TRACK - ``track_id`` and ``f_start`` are
  both per movie, so a bare track id makes two animals in two movies look like
  one animal twice, and the diversity rule then drops a clip that is not a
  duplicate of anything;
* a model fitted at another window length is refused before it ranks anything,
  because an ordering carries no confidence number that could look wrong
  afterwards;
* every record can actually be written into the annotation log. A provenance
  record that the store's codec refuses is not provenance - it is a crash at the
  moment a human presses a label key.

No Qt, no fitted model: the classifier is a stub whose posterior is read
straight off one feature column, so the clip each rule must choose is known
before the rule is called.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from harness import check

MOVIES = ("movieA", "movieB")
TRACKS = 5
WINDOWS = 20
FRAMES = 50                      # clips.frames in the default config
GROUP = "behaviour"              # the default project's first single-choice group
CLASSES = ("immotile", "swimming")


# --------------------------------------------------------------------------- #
# a project, a manifest, and a model that is not a fit
# --------------------------------------------------------------------------- #
class StubModel:
    """A posterior read straight off a feature column, so the ranking is known.

    ``predict_proba`` returns ``[u, 1 - u]`` for the clip's ``certainty`` column,
    which makes the margin ``1 - |2u - 1|``: maximal at ``u = 0.5`` and falling
    away symmetrically. A real fit would answer the same shape and nothing here
    would learn anything from running one.
    """

    classes_ = np.array(CLASSES)

    def predict_proba(self, X):
        u = np.asarray(X, dtype=float)[:, 0]
        return np.column_stack([u, 1.0 - u])


def hand_margin(u: float) -> float:
    """``1 - (p1 - p2)`` written out from the definition, for two classes."""
    return 1.0 - abs(2.0 * float(np.float32(u)) - 1.0)


def build_manifest() -> pd.DataFrame:
    """The clip manifest columns the selector reads, over two movies.

    Both movies number their tracks from 0 and their frames from 0, exactly as a
    real project does - that collision is the point of the fixture.
    """
    rows = []
    for movie in MOVIES:
        for track in range(TRACKS):
            for window in range(WINDOWS):
                start = window * FRAMES
                rows.append({
                    "clip_id": f"{movie}__t{track:04d}__f{start:06d}",
                    "movie_id": movie,
                    "track_id": track,
                    "f_start": start,
                    "f_end": start + FRAMES,
                })
    return pd.DataFrame(rows)


def _project(root: Path, name: str, **settings):
    """A loadable project with nothing in it but a config."""
    from trackanno.config import load_config
    from trackanno.project import create_project, write_config

    project = Path(root) / name
    config_path = create_project(project)
    updates = {
        "paths.work_dir": str(project / "work").replace("\\", "/"),
        "paths.csv": str(project / "annotations.csv").replace("\\", "/"),
    }
    updates.update(settings)
    write_config(config_path, updates)
    cfg = load_config(config_path)
    cfg.paths.work_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def _install_model(cfg, manifest, certainty, trained_window: int | None = FRAMES,
                   counts: dict | None = None, missing: set | None = None):
    """Write the two files ``Suggester.load`` needs, and nothing else.

    ``missing`` names clips deliberately left OUT of the feature table - the
    ordinary state after re-running detection, when the manifest has been rebuilt
    and the features have not.
    """
    import json

    import joblib

    from trackanno.clipfeatures import CLIP_FEATURES_FILE, FEATURE_VERSION
    from trackanno.suggest import MODEL_FILE, REPORT_FILE

    work = cfg.paths.work_dir
    ids = [c for c in manifest["clip_id"].astype(str) if c not in (missing or set())]
    pd.DataFrame({
        "clip_id": ids,
        "certainty": [float(certainty[c]) for c in ids],
        "filler": np.zeros(len(ids)),
        "_feature_version": FEATURE_VERSION,
    }).to_parquet(work / CLIP_FEATURES_FILE, index=False)

    payload = {
        "columns": ["certainty", "filler"],
        "models": {GROUP: StubModel()},
        "feature_version": FEATURE_VERSION,
    }
    if trained_window is not None:
        payload["trained_window"] = int(trained_window)
    joblib.dump(payload, work / MODEL_FILE)

    if counts is not None:
        (work / REPORT_FILE).write_text(
            json.dumps({GROUP: {"per_class_counts": counts, "macro_f1": 0.9,
                                "reliable": True}}),
            encoding="utf-8",
        )
    return work


def spread_certainty(manifest, contested: str | None = None) -> dict:
    """A distinct posterior per clip, and optionally one uniquely contested clip.

    Spread over (0.05, 0.95) in manifest order, which is NOT the ranking order -
    the ranking is by distance from 0.5. ``contested`` is pinned to exactly 0.5,
    the only value the linear spread never lands on, so its margin of 1.0 is the
    unique maximum and the clip the batch must open with is known by name.
    """
    ids = manifest["clip_id"].astype(str).tolist()
    values = dict(zip(ids, np.linspace(0.05, 0.95, len(ids))))
    if contested is not None:
        values[contested] = 0.5
    return values


def _suggester(cfg):
    from trackanno.suggest import Suggester

    suggester = Suggester(cfg)
    assert suggester.ready, suggester.problem
    return suggester


def refusal(fn, *args, **kwargs) -> str:
    """The message a refusal carries, or "" if it did not refuse."""
    try:
        fn(*args, **kwargs)
    except (ValueError, RuntimeError) as exc:
        return str(exc)
    return ""


# --------------------------------------------------------------------------- #
# the default: nothing happens
# --------------------------------------------------------------------------- #
def test_the_default_config_selects_nothing(root: Path) -> None:
    """The optionality guarantee. A project that says nothing about selection
    must annotate in exactly the order it always did, so the only thing that can
    turn the selector on is the rule - not a ready model, not a batch size, not
    the presence of an audit set."""
    from trackanno.config import load_config
    from trackanno.project import write_config

    manifest = build_manifest()
    cfg = _project(root, "default")
    _install_model(cfg, manifest, spread_certainty(manifest))
    suggester = _suggester(cfg)

    check("selection is off in a freshly created project",
          cfg.suggest.select_rule == "off", repr(cfg.suggest.select_rule))
    check("and `selection_on` says so",
          cfg.suggest.selection_on is False)
    check("a model is loaded and ready, so nothing else is what suppresses it",
          suggester.ready, suggester.problem)
    check("no batch size is asked for that produces a batch",
          all(suggester.selection_batch(manifest, set(), n) == []
              for n in (1, 2, 5, 20, 100)))
    check("the GUI's shuffle is still the order, untouched",
          cfg.ui.shuffle is True and cfg.ui.shuffle_seed == 0)

    # The same project with the rule set, and nothing else changed.
    write_config(cfg.path, {"suggest.select_rule": "margin"})
    turned_on = _suggester(load_config(cfg.path))
    check("setting the rule alone is what turns selection on",
          len(turned_on.selection_batch(manifest, set(), 20)) == 20)


def test_the_audit_share_is_derived_from_the_audit_fraction(root: Path) -> None:
    """One number in two roles: the share of clips held out, and the share of a
    batch drawn from that holdout. Restate it as a constant and the labelled
    audit subset fills at a different rate than the audit set was cut at, so the
    estimate it exists to protect quietly starts describing a differently
    composed sample."""
    from trackanno.config import load_config
    from trackanno.project import write_config

    cfg = _project(root, "share")
    check("the default is 'auto', not a number copied from somewhere",
          cfg.suggest.select_audit_share == "auto",
          repr(cfg.suggest.select_audit_share))
    check("and it resolves to the audit fraction itself",
          cfg.suggest.resolved_audit_share() == cfg.suggest.audit_fraction,
          f"{cfg.suggest.resolved_audit_share()} vs {cfg.suggest.audit_fraction}")

    write_config(cfg.path, {"suggest.audit_fraction": 0.25})
    moved = load_config(cfg.path)
    check("moving the audit fraction moves the batch's share with it",
          moved.suggest.resolved_audit_share() == 0.25,
          str(moved.suggest.resolved_audit_share()))

    write_config(cfg.path, {"suggest.select_audit_share": 0.05})
    pinned = load_config(cfg.path)
    check("a share written out explicitly is still honoured",
          pinned.suggest.resolved_audit_share() == 0.05,
          str(pinned.suggest.resolved_audit_share()))


def test_the_diversity_gap_is_derived_from_the_clip_length(root: Path) -> None:
    """'auto' is one clip length, which is the only redundancy this codebase can
    assert rather than estimate: closer than that and two windows of one track
    are partly the same footage. Windows exactly that far apart are kept, so on
    the default non-overlapping grid the rule suppresses nothing at all."""
    from trackanno.config import load_config
    from trackanno.project import write_config

    cfg = _project(root, "gap")
    check("the default is 'auto'", cfg.suggest.select_frame_gap == "auto",
          repr(cfg.suggest.select_frame_gap))
    check("which is the annotation clip length",
          cfg.suggest.resolved_frame_gap(cfg.clips) == cfg.clips.frames == 50,
          str(cfg.suggest.resolved_frame_gap(cfg.clips)))

    write_config(cfg.path, {"clips.frames": 30, "clips.stride": 30,
                            "clips.min_clip_frames": 25})
    shorter = load_config(cfg.path)
    check("shorter clips mean a shorter gap, without anyone editing the gap",
          shorter.suggest.resolved_frame_gap(shorter.clips) == 30,
          str(shorter.suggest.resolved_frame_gap(shorter.clips)))


# --------------------------------------------------------------------------- #
# the batch
# --------------------------------------------------------------------------- #
def test_a_rule_returns_a_full_ordered_batch_with_a_reason_per_clip(root: Path) -> None:
    """What the GUI is handed: `n` unannotated clips, in the order to do them,
    each carrying why it was chosen - and every one of those reasons has to
    survive the trip into annotations.csv, or the provenance is a crash rather
    than a record."""
    from trackanno.select import AUDIT_RULE, UNCERTAINTY_RULES
    from trackanno.store import decode_provenance, encode_provenance
    from trackanno.suggest import is_audit_clip

    manifest = build_manifest()
    ids = manifest["clip_id"].astype(str).tolist()
    done = set(ids[:40])
    # Unannotated and not held out for audit, or it would be out of the ranked
    # pool for a reason that has nothing to do with how contested it is.
    contested = next(c for c in ids if c not in done and not _is_audit(c))
    certainty = spread_certainty(manifest, contested=contested)
    cfg = _project(root, "batch", **{"suggest.select_rule": "margin"})
    _install_model(cfg, manifest, certainty)
    suggester = _suggester(cfg)

    batch = suggester.selection_batch(manifest, done, 20)

    check("the batch is the size that was asked for", len(batch) == 20, str(len(batch)))
    check("no clip appears twice", len({r["clip_id"] for r in batch}) == 20)
    check("nothing already annotated is handed back",
          not ({r["clip_id"] for r in batch} & done))
    check("the batch is ordered, and says so",
          [r["rank"] for r in batch] == list(range(20)))
    check("every record names the rule that chose it",
          all(r["rule"] in set(UNCERTAINTY_RULES) | {AUDIT_RULE} for r in batch),
          str({r["rule"] for r in batch}))
    check("every record names the model that scored it",
          all(r["group"] == GROUP for r in batch))
    check("every record carries the model's own answer at selection time",
          all(r["predicted_class"] in CLASSES and 0.0 <= r["predicted_proba"] <= 1.0
              for r in batch))

    encoded = [encode_provenance(r) for r in batch]
    check("every record can be written into one CSV cell",
          all(cell for cell in encoded))
    check("and reads back naming the same rule",
          [decode_provenance(cell)["rule"] for cell in encoded]
          == [r["rule"] for r in batch])
    check("a random draw reads back with no score rather than a score of zero",
          all(decode_provenance(cell)["score"] is None
              for cell, r in zip(encoded, batch) if r["rule"] == AUDIT_RULE))

    ranked = [r for r in batch if r["rule"] != AUDIT_RULE]
    check("the most contested clip in the pool opens the batch",
          ranked[0]["clip_id"] == contested,
          f"{ranked[0]['clip_id']} not {contested}")
    check("the ranked picks are in descending score order",
          all(a["score"] >= b["score"] for a, b in zip(ranked, ranked[1:])),
          str([r["score"] for r in ranked]))
    check("and every score is the margin of that clip's own posterior",
          all(abs(r["score"] - hand_margin(certainty[r["clip_id"]])) < 1e-9
              for r in ranked))

    # The claim that this really is the most informative n available, not merely
    # a plausible-looking n. Audit clips are excluded from the ranked pool by
    # design, and the diversity rule cannot bite on a non-overlapping grid.
    chosen = {r["clip_id"] for r in batch}
    rest = [hand_margin(certainty[c]) for c in ids
            if c not in chosen and c not in done and not is_audit_clip(c, cfg)]
    check("no unselected candidate was more contested than a selected one",
          min(r["score"] for r in ranked) >= max(rest),
          f"{min(r['score'] for r in ranked)} vs {max(rest)}")


def test_the_batch_does_not_depend_on_the_order_the_manifest_arrived_in(
    root: Path
) -> None:
    """The pool is a set of clips, not a table. A batch that moved when the rows
    were reordered would be a batch decided by a groupby upstream."""
    manifest = build_manifest()
    certainty = spread_certainty(manifest)
    cfg = _project(root, "stable", **{"suggest.select_rule": "margin"})
    _install_model(cfg, manifest, certainty)
    suggester = _suggester(cfg)

    forward = suggester.selection_batch(manifest, set(), 15)
    shuffled = manifest.sample(frac=1.0, random_state=7).reset_index(drop=True)
    backwards = suggester.selection_batch(shuffled, set(), 15)

    check("the same pool gives the same batch, whatever order it arrives in",
          [r["clip_id"] for r in forward] == [r["clip_id"] for r in backwards])
    check("with the same reasons and the same scores",
          [(r["rule"], r["score"]) for r in forward]
          == [(r["rule"], r["score"]) for r in backwards])


def test_audit_clips_are_drawn_at_the_configured_share(root: Path) -> None:
    """The reserved random stratum, which is the whole reason selection is
    allowed to run at all: those slots are drawn by a coin from the holdout, are
    labelled with their own reason, and are never ranked - so the labelled audit
    subset stays a random sample of the audit set however good the selector
    gets."""
    from trackanno.select import AUDIT_RULE
    from trackanno.suggest import is_audit_clip

    manifest = build_manifest()
    certainty = spread_certainty(manifest)
    cfg = _project(root, "audit", **{"suggest.select_rule": "margin"})
    _install_model(cfg, manifest, certainty)
    suggester = _suggester(cfg)

    batch = suggester.selection_batch(manifest, set(), 20)
    drawn = [r for r in batch if r["rule"] == AUDIT_RULE]
    expected = int(cfg.suggest.resolved_audit_share() * 20 + 0.5)

    check("the batch reserves the configured share of its slots",
          len(drawn) == expected == 2, f"{len(drawn)} drawn, expected {expected}")
    check("which is the share of clips the project holds out",
          abs(len(drawn) / 20 - cfg.suggest.audit_fraction) < 1e-9,
          f"{len(drawn) / 20} vs {cfg.suggest.audit_fraction}")
    check("every draw really is an audit clip",
          all(is_audit_clip(r["clip_id"], cfg) for r in drawn))
    check("no audit clip reached the batch by being ranked",
          not any(is_audit_clip(r["clip_id"], cfg) for r in batch
                  if r["rule"] != AUDIT_RULE))
    check("a draw carries no score, because it was not scored",
          all(r["score"] is None for r in drawn), str([r["score"] for r in drawn]))
    check("and it records how many clips it was drawn from, so it can be weighted",
          all(r["pool_size"] > 0 for r in drawn))
    check("the draws are spread through the batch, not parked at its end",
          min(r["rank"] for r in drawn) < 10,
          str([r["rank"] for r in drawn]))

    # An annotator who stops halfway must still have labelled audit clips in
    # proportion; that is what spreading them buys, and it is checkable.
    half = [r for r in batch[:10] if r["rule"] == AUDIT_RULE]
    check("so half a batch still contains a draw", len(half) >= 1, str(len(half)))


def test_a_clip_with_no_feature_row_is_excluded_rather_than_scored(root: Path) -> None:
    """The ordinary state after re-running detection: the manifest is rebuilt and
    the feature table is not, so a clip in the queue has no row to score. It is
    left out of the pool - an imputed row would still rank, on numbers the clip
    does not have."""
    manifest = build_manifest()
    ids = manifest["clip_id"].astype(str).tolist()
    absent = set(ids[::3])                     # a third of the project, scattered
    certainty = spread_certainty(manifest)
    cfg = _project(root, "orphans", **{"suggest.select_rule": "margin"})
    _install_model(cfg, manifest, certainty, missing=absent)
    suggester = _suggester(cfg)

    check("the featureless clips are genuinely missing from the model's table",
          not any(suggester._by_clip.get(c) is not None for c in absent))

    # Asked for more than the pool holds, so nothing can hide behind the batch
    # size: every clip that CAN be selected is selected.
    batch = suggester.selection_batch(manifest, set(), len(ids))
    chosen = {r["clip_id"] for r in batch}
    check("a clip with no feature row is never selected",
          not (chosen & absent), str(sorted(chosen & absent)[:3]))
    check("while the clips that do have one still fill the batch",
          len(batch) > 0 and chosen <= (set(ids) - absent))
    check("and a batch larger than the pool comes back short rather than refusing",
          len(batch) <= len(ids) - len(absent), str(len(batch)))


def test_a_batch_near_the_end_of_a_project_shrinks_instead_of_refusing(
    root: Path
) -> None:
    """``select_batch`` refuses a batch bigger than its pools, on the grounds
    that whoever built the mask knows its size. This is that caller doing the
    knowing: an annotator with three clips left is handed three, and one with
    none is handed the empty batch that tells the GUI to leave its order
    alone."""
    manifest = build_manifest()
    ids = manifest["clip_id"].astype(str).tolist()
    certainty = spread_certainty(manifest)
    cfg = _project(root, "endgame", **{"suggest.select_rule": "margin"})
    _install_model(cfg, manifest, certainty)
    suggester = _suggester(cfg)

    nearly = set(ids[:-3])
    tail = suggester.selection_batch(manifest, nearly, 20)
    check("a batch of 20 with 3 clips left comes back with at most 3",
          0 < len(tail) <= 3, str(len(tail)))
    check("and those are the clips that are left",
          {r["clip_id"] for r in tail} <= set(ids[-3:]))

    check("with nothing left at all the batch is empty, not an exception",
          suggester.selection_batch(manifest, set(ids), 20) == [])
    check("an empty manifest is the same answer",
          suggester.selection_batch(manifest.iloc[:0], set(), 20) == [])


# --------------------------------------------------------------------------- #
# track identity is per movie
# --------------------------------------------------------------------------- #
def _is_audit(clip_id: str, audit_seed: int = 12345, fraction: float = 0.10) -> bool:
    """`suggest.is_audit_clip` for the default seeds, without a Config to hand."""
    import hashlib

    digest = hashlib.blake2b(f"{audit_seed}:{clip_id}".encode(), digest_size=8).digest()
    return (int.from_bytes(digest, "big") % 10_000) < fraction * 10_000


def test_a_track_is_identified_by_movie_and_track_not_by_track_alone(
    root: Path
) -> None:
    """Both ``track_id`` and ``f_start`` are numbered per MOVIE, so two movies
    routinely hold a track 3 whose clips start at frame 250. Handed a bare track
    id the diversity rule reads those as one animal at one moment and suppresses
    one of them - a clip dropped for being a duplicate of a clip in a different
    movie, which is not a relationship that exists.

    The gap is widened past the clip length here so the rule can actually bite;
    the audit share is zeroed so the only thing deciding the batch is the
    ranking and the rule under test.
    """
    manifest = build_manifest()

    # Three clips: two successive windows of one track in movieA, and the
    # same-numbered track and frame in movieB. None may be an audit clip, or it
    # would be out of the ranked pool for a reason that has nothing to do with
    # this claim.
    pair = next(
        (track, window)
        for track in range(TRACKS) for window in range(WINDOWS - 1)
        if not _is_audit(f"movieA__t{track:04d}__f{window * FRAMES:06d}")
        and not _is_audit(f"movieA__t{track:04d}__f{(window + 1) * FRAMES:06d}")
        and not _is_audit(f"movieB__t{track:04d}__f{window * FRAMES:06d}")
    )
    track, window = pair
    first = f"movieA__t{track:04d}__f{window * FRAMES:06d}"
    neighbour = f"movieA__t{track:04d}__f{(window + 1) * FRAMES:06d}"
    other_movie = f"movieB__t{track:04d}__f{window * FRAMES:06d}"

    # Everything else is an easy clip, so these three are the top three by margin
    # and any absence is the diversity rule and nothing else.
    certainty = {c: 0.95 for c in manifest["clip_id"].astype(str)}
    certainty[first] = 0.50
    certainty[neighbour] = 0.51
    certainty[other_movie] = 0.52

    cfg = _project(root, "identity", **{
        "suggest.select_rule": "margin",
        "suggest.select_frame_gap": 100,      # two clip lengths: the pair is inside it
        "suggest.select_audit_share": 0.0,
    })
    _install_model(cfg, manifest, certainty)
    suggester = _suggester(cfg)

    batch = suggester.selection_batch(manifest, set(), 3)
    chosen = [r["clip_id"] for r in batch]

    check("the most contested clip is selected", first in chosen, str(chosen))
    check("its own next window, inside the gap, is suppressed as a near-duplicate",
          neighbour not in chosen, str(chosen))
    check("but the same track number in another movie is NOT a near-duplicate",
          other_movie in chosen, str(chosen))
    check("and it is ranked second, immediately behind the clip it does not duplicate",
          chosen[1] == other_movie, str(chosen))

    # The suppressed clip is skipped, not spent: it is still in the pool.
    later = suggester.selection_batch(manifest, {first, other_movie}, 3)
    check("a clip skipped for redundancy is still a candidate next time",
          neighbour in [r["clip_id"] for r in later],
          str([r["clip_id"] for r in later]))


# --------------------------------------------------------------------------- #
# the config refuses what it cannot honour
# --------------------------------------------------------------------------- #
def test_a_rule_without_an_audit_holdout_is_refused_at_load(root: Path) -> None:
    """Selection makes the labelled set a sample assembled to be hard, so every
    accuracy computed over it is biased downward by an amount nobody can
    estimate. The audit holdout is the one unbiased estimate left standing, and
    a project that turns the selector on while switching the holdout off has
    asked for a number that cannot exist. That is worth stopping, not warning
    about."""
    from trackanno.config import ConfigError, load_config
    from trackanno.project import create_project, write_config

    project = Path(root) / "no-audit"
    config_path = create_project(project)
    write_config(config_path, {"suggest.audit_fraction": 0.0})
    check("an audit fraction of 0 on its own is a legitimate project",
          load_config(config_path).suggest.audit_fraction == 0.0)

    # write_config validates before it saves, so the refusal arrives here.
    message = refusal(write_config, config_path, {"suggest.select_rule": "margin"})
    check("adding a rule to it is refused", message != "")
    check("and the message names the rule", "select_rule" in message, message)
    check("and the setting that contradicts it",
          "audit_fraction" in message, message)
    check("and says what the holdout is for",
          "unbiased" in message.lower(), message)
    check("and how to resolve it either way",
          "0.10" in message and "'off'" in message, message)
    check("the refusal is a ConfigError, so the GUI reports it as a settings problem",
          isinstance(refusal_exception(write_config, config_path,
                                       {"suggest.select_rule": "margin"}),
                     ConfigError))


def refusal_exception(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except Exception as exc:            # noqa: BLE001 - the type IS the claim
        return exc
    return None


def test_a_rule_with_the_suggester_disabled_is_refused(root: Path) -> None:
    """With `suggest.enabled: false` no model is ever loaded, so the rule would
    silently do nothing and the order would stay the shuffle. A setting that
    reads as applied and is not is the failure `_known` already guards against
    for typos; this is the same failure spelled correctly."""
    from trackanno.project import create_project, write_config

    config_path = create_project(Path(root) / "disabled")
    write_config(config_path, {"suggest.enabled": False})
    message = refusal(write_config, config_path, {"suggest.select_rule": "entropy"})
    check("a rule with the suggester switched off is refused", message != "")
    check("and the message names both keys",
          "select_rule" in message and "enabled" in message, message)


def test_the_choice_settings_refuse_a_value_they_do_not_know(root: Path) -> None:
    """A misspelt rule that fell back to a default would look applied and rank on
    something else entirely, and 'off' is exactly the word YAML turns into a
    boolean - so that one is worth naming."""
    from trackanno.project import create_project, write_config

    config_path = create_project(Path(root) / "choices")

    unknown = refusal(write_config, config_path, {"suggest.select_rule": "uncertainty"})
    check("an unknown rule is refused rather than defaulted", unknown != "")
    check("and the message lists the rules there are",
          "margin" in unknown and "entropy" in unknown, unknown)

    target = refusal(write_config, config_path,
                     {"suggest.select_target_rule": "target_random"})
    check("so is an unknown targeting rule", target != "")
    check("and the message says what the two real ones buy",
          "boundary" in target, target)

    text = config_path.read_text(encoding="utf-8").replace(
        'select_rule: "off"', "select_rule: off")
    config_path.write_text(text, encoding="utf-8")
    boolean = refusal(_load, config_path)
    check("an unquoted `off`, which YAML makes a boolean, is refused by name",
          boolean != "" and "boolean" in boolean, boolean)
    check("and the message says to quote it", '"off"' in boolean, boolean)


def _load(path):
    from trackanno.config import load_config

    return load_config(path)


def test_every_new_key_is_registered(root: Path) -> None:
    """`config._known` ignores what it does not recognise, so an unregistered key
    is a setting a user writes, saves, and never has applied. This file already
    carries that bug once. The claim is symmetric: the real names pass silently,
    a typo does not."""
    from trackanno.config import Suggest

    new_keys = {
        "select_rule": "margin",
        "select_batch_size": 12,
        "select_audit_share": 0.2,
        "select_frame_gap": 60,
        "select_target_quota": 8,
        "select_target_rule": "target_confident",
    }

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        parsed = Suggest.parse(dict(new_keys))
    check("every new key parses without being ignored",
          parsed.select_rule == "margin" and parsed.select_batch_size == 12
          and parsed.select_audit_share == 0.2 and parsed.select_frame_gap == 60
          and parsed.select_target_quota == 8
          and parsed.select_target_rule == "target_confident")

    for key, value in new_keys.items():
        typo = dict(new_keys)
        typo.pop(key)
        typo[key + "s"] = value
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(UserWarning) as caught:
                Suggest.parse(typo)
        check(f"a typo of {key!r} is reported rather than ignored",
              key + "s" in str(caught.value), str(caught.value))


def test_the_numeric_settings_refuse_a_value_that_cannot_mean_anything(
    root: Path
) -> None:
    """Each of these has a value that parses, looks set, and would make the
    selector do something other than what the number says."""
    from trackanno.config import Suggest

    check("a batch of no clips is refused",
          "at least 1" in refusal(Suggest.parse, {"select_rule": "margin",
                                                  "select_batch_size": 0}))
    check("a share outside [0, 1] is refused",
          refusal(Suggest.parse, {"select_rule": "margin",
                                  "select_audit_share": 1.5}) != "")
    check("a negative frame gap is refused",
          refusal(Suggest.parse, {"select_rule": "margin",
                                  "select_frame_gap": -1}) != "")
    check("a negative quota is refused",
          refusal(Suggest.parse, {"select_rule": "margin",
                                  "select_target_quota": -3}) != "")
    check("'auto' is the only word either of the two auto keys accepts",
          refusal(Suggest.parse, {"select_frame_gap": "clip"}) != ""
          and refusal(Suggest.parse, {"select_audit_share": "same"}) != "")


# --------------------------------------------------------------------------- #
# the window length
# --------------------------------------------------------------------------- #
def test_a_window_length_mismatch_stops_the_batch(root: Path) -> None:
    """A model fitted at one length ranking features built at another ranks on
    the wrong scale, and an ORDER carries no confidence number that could look
    wrong afterwards - the batch it produces is indistinguishable from a good
    one. Refused the same way `predict.py` refuses it, and for the same reason a
    model that never recorded its length is only warned about."""
    manifest = build_manifest()
    certainty = spread_certainty(manifest)

    cfg = _project(root, "skew", **{"suggest.select_rule": "margin"})
    _install_model(cfg, manifest, certainty, trained_window=87)
    suggester = _suggester(cfg)

    message = refusal(suggester.selection_batch, manifest, set(), 20)
    check("a batch from a model fitted at another window length is refused",
          message != "")
    check("and the message names the length it was fitted at", "87" in message, message)
    check("and the length the candidates were built at", "50" in message, message)
    check("and where that number comes from, so it can be acted on",
          "clips.frames" in message, message)

    matched = _project(root, "matched", **{"suggest.select_rule": "margin"})
    _install_model(matched, manifest, certainty, trained_window=FRAMES)
    check("the same batch at the matching length goes through",
          len(_suggester(matched).selection_batch(manifest, set(), 20)) == 20)


def test_a_model_that_never_recorded_its_window_is_warned_about_not_refused(
    root: Path
) -> None:
    """Every model saved before the length was recorded would otherwise stop
    selecting the day trackanno is upgraded, which is a worse failure than the
    one being guarded against: 'cannot be checked' is not 'is wrong'."""
    manifest = build_manifest()
    certainty = spread_certainty(manifest)
    cfg = _project(root, "legacy", **{"suggest.select_rule": "margin"})
    _install_model(cfg, manifest, certainty, trained_window=None)
    suggester = _suggester(cfg)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        batch = suggester.selection_batch(manifest, set(), 20)
    check("the batch is still produced", len(batch) == 20, str(len(batch)))
    check("and the unchecked length is warned about",
          any("does not record" in str(w.message) for w in caught),
          str([str(w.message)[:60] for w in caught]))


# --------------------------------------------------------------------------- #
# targeting a starved class
# --------------------------------------------------------------------------- #
def test_a_starved_class_is_targeted_only_when_a_quota_asks_for_it(
    root: Path
) -> None:
    """Targeting is off even with selection on, because it deliberately skews the
    label marginals and the reliability gate is computed against them. When a
    quota does ask, the slot says which class it was spent on and which rule
    spent it, so the skew is on the record rather than inferred later."""
    from trackanno.config import load_config
    from trackanno.project import write_config
    from trackanno.select import AUDIT_RULE

    manifest = build_manifest()
    certainty = spread_certainty(manifest)
    counts = {"immotile": 3, "swimming": 90}
    cfg = _project(root, "starved", **{"suggest.select_rule": "margin"})
    _install_model(cfg, manifest, certainty, counts=counts)

    untargeted = _suggester(cfg).selection_batch(manifest, set(), 20)
    check("with the default quota no slot is spent on a class",
          all(r["target_class"] == "" for r in untargeted))

    write_config(cfg.path, {"suggest.select_target_quota": 10,
                            "suggest.select_target_rule": "target_confident"})
    targeted = _suggester(load_config(cfg.path)).selection_batch(manifest, set(), 20)
    spent = [r for r in targeted if r["target_class"]]
    check("a quota above the scarcest class's count spends slots on it",
          len(spent) == 7, f"{len(spent)} slots spent")
    check("every one of them names the starved class",
          all(r["target_class"] == "immotile" for r in spent),
          str({r["target_class"] for r in spent}))
    check("and names the rule that was asked to find it",
          all(r["rule"] == "target_confident" for r in spent),
          str({r["rule"] for r in spent}))
    check("the ranked slots that are left still come from the uncertainty rule",
          {r["rule"] for r in targeted if not r["target_class"]}
          <= {"margin", AUDIT_RULE})
