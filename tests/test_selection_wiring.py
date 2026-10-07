"""Wiring :mod:`trackanno.learn.select` into a project: the config, the pool, the batch.

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
* the pool handed to ``select_batch`` is assembled from the model directly,
  never ranks audit clips, and excludes what cannot be scored;
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
                   missing: set | None = None):
    """Write the two files ``Suggester.load`` needs, and nothing else.

    ``missing`` names clips deliberately left OUT of the feature table - the
    ordinary state after re-running detection, when the manifest has been rebuilt
    and the features have not.
    """
    import joblib

    from trackanno.measure.clipfeatures import CLIP_FEATURES_FILE, FEATURE_VERSION
    from trackanno.learn.suggest import MODEL_FILE

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
    from trackanno.learn.suggest import Suggester

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
          cfg.ui.shuffle is True)

    # The same project with the rule set, and nothing else changed.
    write_config(cfg.path, {"suggest.select_rule": "margin"})
    turned_on = _suggester(load_config(cfg.path))
    check("setting the rule alone is what turns selection on",
          len(turned_on.selection_batch(manifest, set(), 20)) == 20)


# --------------------------------------------------------------------------- #
# the batch
# --------------------------------------------------------------------------- #
def test_a_rule_returns_a_full_ordered_batch_with_a_reason_per_clip(root: Path) -> None:
    """What the GUI is handed: `n` unannotated clips, in the order to do them,
    each carrying why it was chosen - and every one of those reasons has to
    survive the trip into annotations.csv, or the provenance is a crash rather
    than a record."""
    from trackanno.annotate.store import decode_provenance, encode_provenance
    from trackanno.learn.suggest import AuditHoldout

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
          all(r["rule"] == "margin" for r in batch),
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

    ranked = batch
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
    # design.
    chosen = {r["clip_id"] for r in batch}
    # Nothing is labelled in this project, so nothing records a membership and
    # the ids are the whole answer - which is what the suggester read too.
    holdout = AuditHoldout.by_hash(cfg)
    rest = [hand_margin(certainty[c]) for c in ids
            if c not in chosen and c not in done and c not in holdout]
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


def test_audit_clips_are_never_ranked(root: Path) -> None:
    """The holdout stays a random sample: the selector never hands one over."""
    from trackanno.learn.suggest import AuditHoldout

    manifest = build_manifest()
    ids = manifest["clip_id"].astype(str).tolist()
    cfg = _project(root, "audit", **{"suggest.select_rule": "margin"})
    _install_model(cfg, manifest, spread_certainty(manifest))
    suggester = _suggester(cfg)

    holdout = AuditHoldout.by_hash(cfg)
    check("the project has audit clips", any(c in holdout for c in ids))
    batch = suggester.selection_batch(manifest, set(), len(ids))
    check("no audit clip is in the batch",
          not any(r["clip_id"] in holdout for r in batch))
    check("every other clip is",
          len(batch) == sum(c not in holdout for c in ids), str(len(batch)))


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
    """An annotator with three clips left is handed at most three, and one with
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


def _is_audit(clip_id: str, audit_seed: int = 12345, fraction: float = 0.10) -> bool:
    """`AuditHoldout` for the default seeds, without a Config to hand."""
    import hashlib

    digest = hashlib.blake2b(f"{audit_seed}:{clip_id}".encode(), digest_size=8).digest()
    return (int.from_bytes(digest, "big") % 10_000) < fraction * 10_000


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
    message = refusal(write_config, config_path, {"suggest.select_rule": "margin"})
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
          "margin" in unknown and "off" in unknown, unknown)

    from trackanno.config import Suggest

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parsed = Suggest.parse({"select_rule": "entropy"})
    check("a rule that was removed loads as margin, with a warning",
          parsed.select_rule == "margin"
          and any("entropy" in str(w.message) for w in caught))

    text = config_path.read_text(encoding="utf-8").replace(
        'select_rule: "off"', "select_rule: off")
    config_path.write_text(text, encoding="utf-8")
    check("an unquoted `off`, which YAML makes a boolean, still reads as 'off'",
          _load(config_path).suggest.select_rule == "off")


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
        "retrain_every": 12,
    }

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        parsed = Suggest.parse(dict(new_keys))
    check("every new key parses without being ignored",
          parsed.select_rule == "margin" and parsed.retrain_every == 12)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Suggest.parse({"select_rule": "margin", "select_frame_gap": 60,
                       "select_target_quota": 8})
    check("a removed key still loads, with a warning",
          any("select_frame_gap" in str(w.message) for w in caught))

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

    check("a refit after no labels is refused",
          "at least 1" in refusal(Suggest.parse, {"retrain_every": 0}))


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
