"""The audit holdout has to be the same set of clips tomorrow that it is today.

The holdout is the one estimate in this tool that survives directed selection:
its clips are never suggested on and never trained on, so agreement measured
there is unbiased in a way nothing computed over the labelled set can be. That
guarantee rested on membership being a hash of ``clip_id`` - and ``clip_id``
embeds the track id, which re-running detection renumbers. Every id changes, the
holdout is redrawn from scratch, and ``reassociate`` then re-points each label at
its clip's NEW id, so a label earned on a held-out clip can arrive under an id
that hashes into the training set. The next fit trains on the holdout and
reports its accuracy over a different, partly overlapping set. Nothing raises,
nothing warns, and nothing in the program can even tell.

Membership is therefore recorded in ``annotations.csv`` the moment a clip is
labelled, and the record is what counts from then on. This suite states the four
things that has to mean:

* an upgrade moves NOBODY - the holdout an existing project has is the one it
  keeps, clip for clip;
* a labelled clip keeps its side across a re-detection that changes every id,
  including through ``reassociate``;
* an UNLABELLED clip is still free to move, which is not a leak but the reason
  the fix is this cheap - nothing has depended on it yet;
* and the numbers that come out say how much of the holdout they rest on.
"""

from __future__ import annotations

import contextlib
import csv
import io
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from harness import check
from trackanno.config import RESERVED_GROUP_NAMES, load_config
from trackanno.reassociate import apply_reassociation, plan_reassociation, write_labels
from trackanno.store import (
    AUDIT_COLUMN, IDENTITY_COLUMNS, PROVENANCE_COLUMN, TRAILING_COLUMNS,
    AnnotationStore, decode_audit, encode_audit, read_audit_records,
)
from trackanno.suggest import _print_report, audit_records, is_audit_clip, train_group

CONFIG_TEXT = """
paths:  {video_dir: ./videos, model: ./model.pt, work_dir: ./work, csv: ./annotations.csv}
tiff:   {fps: 100}
categories:
  - name: behaviour
    type: single
    required: false
    options:
      - {name: swimming, key: "1", color: "#4C9F70"}
      - {name: tumbling, key: "2", color: "#D95F02"}
ui: {annotator: tester}
"""

# Enough clips that the flip rate below is a measurement rather than an anecdote,
# and small enough that every test here is instant.
N_CLIPS = 200

# What a re-detection does to the numbering. Any offset changes every clip_id;
# this one guarantees no new id collides with an old one.
RENUMBER_BY = 500


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _cfg(root: Path, name: str, text: str = CONFIG_TEXT):
    project = Path(root) / name
    project.mkdir(parents=True, exist_ok=True)
    path = project / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return load_config(path)


def _clip_id(track: int, f_start: int, movie: str = "MOVIE") -> str:
    """``clips.py``'s own format, which is where the track id gets embedded."""
    return f"{movie}__t{track:04d}__f{f_start:06d}"


def _meta(track: int, f_start: int, movie: str = "MOVIE") -> dict:
    return {
        "clip_id": _clip_id(track, f_start, movie), "movie_id": movie,
        "source_file": f"/movies/{movie}.tif", "series": 0, "channel": 0,
        "track_id": track, "f_start": f_start, "f_end": f_start + 49,
        # A crop centre unique within its (movie, f_start) bucket, so
        # `reassociate` has exactly one candidate and never has to arbitrate.
        "crop_x0": 10 + track, "crop_y0": 20 + track, "crop_w": 96, "crop_h": 96,
        "scale": 1.0,
    }


def _population(n: int = N_CLIPS) -> list[dict]:
    """One clip per track, each at its own start frame."""
    return [_meta(track, track * 50) for track in range(1, n + 1)]


def _old_schema(cfg) -> list[str]:
    """The header a release that predates BOTH new columns wrote."""
    columns = IDENTITY_COLUMNS + [g.name for g in cfg.categories] + TRAILING_COLUMNS
    return [c for c in columns if c not in (PROVENANCE_COLUMN, AUDIT_COLUMN)]


def _row(meta: dict, behaviour: str = "swimming") -> dict:
    row = dict(meta)
    row.update({
        "behaviour": behaviour, "uncertain": "0", "notes": "", "annotator": "benny",
        "timestamp_utc": "2025-11-04T09:12:33+00:00", "dwell_s": "12.75",
        "suggested": "", "suggest_conf": "", "suggest_shown": "0",
    })
    return row


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def _read_csv(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _header(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return next(csv.reader(fh))


def _load_quietly(path, cfg) -> tuple[AnnotationStore, list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        store = AnnotationStore(path, cfg)
    return store, [str(w.message) for w in caught]


def _refusal(fn, *args, **kwargs) -> str:
    try:
        fn(*args, **kwargs)
    except (ValueError, TypeError) as exc:
        return str(exc)
    return ""


def _held_out(cfg, clip_ids, records=None) -> set[str]:
    """The holdout as the program would compute it, as a plain set of ids."""
    return {c for c in clip_ids if is_audit_clip(c, cfg, records)}


def _renumbered_manifest(metas: list[dict]) -> pd.DataFrame:
    """The clip manifest a re-detection leaves behind: same animals, same crops,
    same frames, every track id and therefore every clip_id different."""
    rows = []
    for meta in metas:
        new = dict(meta)
        new["track_id"] = meta["track_id"] + RENUMBER_BY
        new["clip_id"] = _clip_id(new["track_id"], meta["f_start"], meta["movie_id"])
        rows.append(new)
    return pd.DataFrame(rows)


def _redetect(path: Path, manifest: pd.DataFrame) -> tuple[int, dict[str, str]]:
    """Re-run detection and repair the labels, the way the Model page does.

    Returns how many rows were re-linked and the old -> new id map, so a claim
    can be made about the clips by name rather than by count alone.
    """
    exact = pd.read_csv(path, dtype=str, keep_default_na=False)
    plan = plan_reassociation(exact, manifest)
    write_labels(path, apply_reassociation(exact, plan))
    return len(plan.matched), {m.old_clip_id: m.new_clip_id for m in plan.matched}


# --------------------------------------------------------------------------- #
# the upgrade
# --------------------------------------------------------------------------- #
def test_upgrading_an_existing_project_moves_no_clip_into_or_out_of_the_holdout(
    root: Path
) -> None:
    """The migration's whole job, stated as the absence of a change.

    Recording what the hash says TODAY is what makes this an upgrade rather than
    a redraw. Any other seed for the recorded value - a fresh draw, a
    renumbering, the clip's position in the file - would move roughly a fifth of
    the clips at the default fraction, and every one of those moves is either a
    clip the model has already seen being scored as if it had not, or a clip
    withheld from it for nothing.
    """
    cfg = _cfg(root, "upgrade")
    path = cfg.paths.csv
    metas = _population()
    _write_csv(path, _old_schema(cfg), [_row(m) for m in metas])

    ids = [m["clip_id"] for m in metas]
    before_rows = _read_csv(path)
    before = _held_out(cfg, ids)
    check("the fixture predates the column", AUDIT_COLUMN not in _header(path))
    check("...and has a holdout worth preserving", 0 < len(before) < len(ids),
          f"{len(before)} of {len(ids)}")

    store, said = _load_quietly(path, cfg)
    records = read_audit_records(path)
    after = _held_out(cfg, ids, records)

    check("every row still loads", len(store) == len(metas), str(len(store)))
    check("every labelled clip now records a membership",
          len(records) == len(metas), f"{len(records)} of {len(metas)}")
    check("THE HOLDOUT IS THE SAME SET OF CLIPS, clip for clip", after == before,
          f"{len(after ^ before)} clip(s) changed side: {sorted(after ^ before)[:5]}")
    check("...which is not vacuous - it is a real, non-empty subset",
          0 < len(after) < len(ids), f"{len(after)} of {len(ids)}")
    check("...and every recorded value is exactly what the hash said",
          all(records[c] == cfg.suggest.audit_by_hash(c) for c in ids))

    after_rows = _read_csv(path)
    check("no other column was touched by the upgrade",
          all(old == {k: v for k, v in new.items() if k in old}
              for old, new in zip(before_rows, after_rows)),
          str(next((n for o, n in zip(before_rows, after_rows)
                    if o != {k: v for k, v in n.items() if k in o}), "")))
    check("...and the rows kept their order", [r["clip_id"] for r in after_rows] == ids)
    check("the upgrade announced itself once, saying it moved nobody",
          sum(1 for m in said if AUDIT_COLUMN in m and "moves no clip" in m) == 1,
          str(said))

    # A migration that warns for ever trains people to ignore the warning that
    # matters - and re-opening must also not re-derive anything.
    again, said_again = _load_quietly(path, cfg)
    check("re-opening the upgraded file says nothing about the audit column",
          not any(AUDIT_COLUMN in m for m in said_again), str(said_again))
    check("...and reports the same holdout a third time",
          _held_out(cfg, ids, read_audit_records(path)) == before)
    check("...from a store that agrees row for row",
          all(decode_audit(again.get(c)[AUDIT_COLUMN]) == (c in before) for c in ids))


def test_a_project_with_no_holdout_records_that_too_rather_than_leaving_it_open(
    root: Path
) -> None:
    """``audit_fraction: 0`` stamps every labelled clip as NOT held out, which is
    the mirror image of the same failure. Left blank, raising the fraction later
    would pull clips the model has already trained on into the "never trained
    on" holdout - contaminating the one number that is supposed to be clean, in
    the direction that flatters it."""
    text = CONFIG_TEXT + "\nsuggest: {audit_fraction: 0.0}\n"
    cfg = _cfg(root, "no-holdout", text)
    path = cfg.paths.csv
    metas = _population(20)
    _write_csv(path, _old_schema(cfg), [_row(m) for m in metas])

    _load_quietly(path, cfg)
    records = read_audit_records(path)
    check("the project really has no holdout", cfg.suggest.audit_fraction == 0.0)
    check("every labelled clip still records a membership",
          len(records) == len(metas), str(len(records)))
    check("...and every one of them says 'not held out'",
          set(records.values()) == {False}, str(set(records.values())))

    # Now the user turns the holdout on. The clips already judged stay where
    # they were; only clips nothing has been fitted on yet can join.
    later = _cfg(root, "no-holdout-then-on", CONFIG_TEXT)
    ids = [m["clip_id"] for m in metas]
    check("raising audit_fraction would have hashed some of them in",
          any(later.suggest.audit_by_hash(c) for c in ids))
    check("...but the record keeps every already-judged clip out of the holdout",
          _held_out(later, ids, records) == set(),
          str(_held_out(later, ids, records)))


# --------------------------------------------------------------------------- #
# re-detection
# --------------------------------------------------------------------------- #
def test_a_labelled_clip_keeps_its_membership_across_a_real_reassociation(
    root: Path
) -> None:
    """The failure this column exists for, driven through the code that causes it.

    ``reassociate`` rewrites ``clip_id`` in place to put a label back on its
    clip after detection renumbered the tracks. That repair is correct and
    necessary - without it the label is dropped from the fit silently - and it
    is exactly what used to move a clip between the holdout and the training
    set, because the id it writes hashes differently from the one it replaces.
    """
    cfg = _cfg(root, "redetect")
    path = cfg.paths.csv
    metas = _population()
    _write_csv(path, _old_schema(cfg), [_row(m) for m in metas])
    _load_quietly(path, cfg)                      # the upgrade freezes membership

    old_ids = [m["clip_id"] for m in metas]
    frozen = _held_out(cfg, old_ids, read_audit_records(path))

    manifest = _renumbered_manifest(metas)
    n_matched, moved_to = _redetect(path, manifest)
    new_ids = [moved_to[c] for c in old_ids]
    check("the re-detection changed every clip_id",
          not set(old_ids) & set(new_ids) and n_matched == len(metas), str(n_matched))

    # The measurement that says why this matters at all. Without the record the
    # holdout is drawn afresh from the new ids, and the two draws are
    # independent - about 2f(1-f) of the clips change side.
    by_hash_now = {moved_to[c] for c in old_ids if cfg.suggest.audit_by_hash(moved_to[c])}
    flipped = sum(1 for c in old_ids
                  if (c in frozen) != (moved_to[c] in by_hash_now))
    check("re-detection would have redrawn a large minority of the holdout",
          0.10 < flipped / len(old_ids) < 0.30,
          f"{flipped} of {len(old_ids)} = {flipped / len(old_ids):.1%} would move")

    store, said = _load_quietly(path, cfg)
    records = read_audit_records(path)
    now = {c for c in new_ids if is_audit_clip(c, cfg, records)}

    check("the repair re-linked every row and lost none",
          len(store) == len(metas), str(len(store)))
    check("EVERY labelled clip is on the side it was labelled on",
          now == {moved_to[c] for c in frozen},
          f"{len(now ^ {moved_to[c] for c in frozen})} clip(s) changed side")
    check("...which is a different set from what the new ids hash to",
          now != by_hash_now, "the renumbering happened to be a no-op; widen it")
    check("the holdout is still exactly as big as it was",
          len(now) == len(frozen), f"{len(now)} vs {len(frozen)}")
    check("the human's answer travelled with it",
          store.get(new_ids[0])["behaviour"] == "swimming")
    check("...and so did the geometry the repair rewrote",
          int(store.get(new_ids[0])["track_id"]) == metas[0]["track_id"] + RENUMBER_BY)

    # Not a warning per row - disagreeing with the hash is the case this
    # mechanism exists to handle - but the count is the only way anyone learns
    # their project was re-detected at all.
    counted = [m for m in said if "recorded when they were labelled" in m]
    check("the disagreement is surfaced exactly once, as a count", len(counted) == 1,
          str(said))
    check("...and the count is the number of clips actually disagreeing",
          str(flipped) in counted[0], counted[0])
    check("...and it does not call it an error",
          "Nothing to fix" in counted[0], counted[0])


def test_a_clip_that_was_held_out_does_not_enter_the_training_set_when_re_linked(
    root: Path
) -> None:
    """Stated where it is decided: ``train_group``'s own audit mask.

    A clip labelled while held out was never suggested on, so the human's answer
    on it is independent of the model. Training on it afterwards spends that
    independence for one more row and leaves an ``audit_accuracy`` computed
    partly over clips the model was fitted on - the circularity the holdout
    exists to prevent, arrived at by a repair that was doing the right thing.
    """
    cfg = _cfg(root, "training-set")
    path = cfg.paths.csv

    # Ten clips of one class, none of which hash into the holdout, so anything
    # held out below is held out because the CSV says so and for no other reason.
    metas = [m for m in _population(60) if not cfg.suggest.audit_by_hash(m["clip_id"])][:10]
    check("the fixture's ids all hash OUT of the holdout",
          not any(cfg.suggest.audit_by_hash(m["clip_id"]) for m in metas))

    held = metas[0]["clip_id"]
    rows = [_row(m) for m in metas]
    rows[0][AUDIT_COLUMN] = encode_audit(True)    # judged while held out, then re-linked
    _write_csv(path, IDENTITY_COLUMNS + ["behaviour"] + TRAILING_COLUMNS, rows)

    labels = pd.read_csv(path)
    features = pd.DataFrame({"clip_id": [m["clip_id"] for m in metas],
                             "movie_id": "MOVIE",
                             "speed_mean": range(len(metas))})

    _, report = train_group(features, labels, "behaviour", cfg,
                            columns=["speed_mean"], verbose=False)
    check("the held-out clip is counted as audit, against its own hash",
          report.audit_labelled == 1, str(report.audit_labelled))
    check("...and is NOT among the clips the fit would see",
          report.n_labels == len(metas) - 1, str(report.n_labels))
    check("...and is counted in the audit population as well",
          report.audit_population == 1, str(report.audit_population))

    # The same table with the column taken away is the old behaviour, and it is
    # the bug: nothing is held out, and the clip walks into the fit.
    forgotten = labels.drop(columns=[AUDIT_COLUMN])
    _, without = train_group(features, forgotten, "behaviour", cfg,
                             columns=["speed_mean"], verbose=False)
    check("without the record that same clip would be trained on",
          without.n_labels == len(metas) and without.audit_labelled == 0,
          f"n_labels={without.n_labels} audit_labelled={without.audit_labelled}")
    check("so the record is what is keeping it out, not the hash",
          held not in set(features["clip_id"]) - set(features["clip_id"])
          and not cfg.suggest.audit_by_hash(held))


def test_an_unlabelled_clip_is_still_free_to_change_sides(root: Path) -> None:
    """Asserted as the intended behaviour, because it is what makes the fix cheap.

    Freezing membership for every clip in the project would mean carrying a
    record for hundreds of thousands of clips nobody has looked at, to protect a
    guarantee none of them carry: nothing has been trained on an unlabelled clip
    and nothing has been withheld from it, so moving it costs nothing and can be
    proved to cost nothing. The line is drawn at the label, and it is drawn
    there deliberately rather than for want of somewhere better.
    """
    cfg = _cfg(root, "unlabelled")
    path = cfg.paths.csv
    metas = _population()

    # Label half; leave half untouched.
    labelled, untouched = metas[:100], metas[100:]
    _write_csv(path, _old_schema(cfg), [_row(m) for m in labelled])
    _load_quietly(path, cfg)
    records = read_audit_records(path)

    manifest = _renumbered_manifest(metas)
    _redetect(path, manifest)
    records = read_audit_records(path)
    moved = {m["clip_id"]: _clip_id(m["track_id"] + RENUMBER_BY, m["f_start"])
             for m in metas}

    check("no unlabelled clip carries a record", not any(
        moved[m["clip_id"]] in records or m["clip_id"] in records for m in untouched))
    check("...so each is answered by the hash on its CURRENT id",
          all(is_audit_clip(moved[m["clip_id"]], cfg, records)
              == cfg.suggest.audit_by_hash(moved[m["clip_id"]]) for m in untouched))

    changed = [m for m in untouched
               if cfg.suggest.audit_by_hash(m["clip_id"])
               != cfg.suggest.audit_by_hash(moved[m["clip_id"]])]
    check("and some of them really did change side, which is allowed",
          len(changed) > 0, f"{len(changed)} of {len(untouched)}")
    check("...while every LABELLED clip in the same file did not",
          all(is_audit_clip(moved[m["clip_id"]], cfg, records)
              == cfg.suggest.audit_by_hash(m["clip_id"]) for m in labelled))

    # The moment one of them is labelled, it stops moving.
    store = AnnotationStore(path, cfg)
    joining = moved[changed[0]["clip_id"]]
    row = store.record(dict(_meta(changed[0]["track_id"] + RENUMBER_BY,
                                  changed[0]["f_start"])),
                       {"behaviour": "tumbling"})
    check("labelling a clip records the membership it was labelled under",
          decode_audit(row[AUDIT_COLUMN]) == cfg.suggest.audit_by_hash(joining),
          repr(row[AUDIT_COLUMN]))
    check("...and it is on disk, not only in memory",
          read_audit_records(path).get(joining) == cfg.suggest.audit_by_hash(joining))


def test_editing_a_re_linked_label_does_not_re_derive_its_membership(
    root: Path
) -> None:
    """``record`` runs again for every edit, undo and correction. Re-deriving
    membership there would undo the whole guarantee through the back door: a
    held-out clip would move into the training set because somebody fixed a typo
    in its notes, months after the answer on it was given."""
    cfg = _cfg(root, "edit")
    path = cfg.paths.csv
    meta = _meta(RENUMBER_BY + 1, 50)
    clip_id = meta["clip_id"]
    check("this clip hashes OUT of the holdout, so any 'in' comes from the record",
          not cfg.suggest.audit_by_hash(clip_id))

    row = _row(meta)
    row[AUDIT_COLUMN] = encode_audit(True)
    _write_csv(path, IDENTITY_COLUMNS + ["behaviour"] + TRAILING_COLUMNS, [row])

    store = AnnotationStore(path, cfg)
    check("it loads as held out", decode_audit(store.get(clip_id)[AUDIT_COLUMN]) is True)

    edited = store.record(meta, {"behaviour": "tumbling"}, notes="second look")
    check("the edit landed", edited["behaviour"] == "tumbling")
    check("...and the clip is STILL held out", decode_audit(edited[AUDIT_COLUMN]) is True)
    check("...on disk too", read_audit_records(path)[clip_id] is True)
    check("...and after an undo", store.undo() == clip_id
          and read_audit_records(path)[clip_id] is True)

    # Clearing the row removes the label, and with it the thing that was being
    # protected: the clip is unlabelled again and free to move again.
    store.clear(clip_id)
    check("clearing the label clears the record with it",
          clip_id not in read_audit_records(path))
    check("...so a fresh label takes the hash's answer",
          decode_audit(
              AnnotationStore(path, cfg).record(meta, {"behaviour": "swimming"})[AUDIT_COLUMN]
          ) is False)


# --------------------------------------------------------------------------- #
# the codec and the file
# --------------------------------------------------------------------------- #
def test_not_recorded_is_not_the_same_fact_as_not_held_out() -> None:
    """The same distinction ``selected_by`` had to make, for a stricter reason:
    here the two answers do not merely mean different things, they send the clip
    to different sides. A blank read as "not held out" would feed every
    pre-upgrade holdout clip straight into the fit."""
    check("an empty cell decodes to nothing at all", decode_audit("") is None)
    check("...and so does a column that was never there", decode_audit(None) is None)
    check("...and whitespace, which a spreadsheet round-trip introduces",
          decode_audit("  ") is None)
    check("...and pandas' NaN, which is what read_csv makes of a blank",
          decode_audit(float("nan")) is None)

    check("held out reads back as held out", decode_audit(encode_audit(True)) is True)
    check("not held out reads back as not held out",
          decode_audit(encode_audit(False)) is False)
    check("not recorded reads back as not recorded",
          decode_audit(encode_audit(None)) is None)
    check("the three are distinguishable without a convention",
          len({str(decode_audit(encode_audit(v))) for v in (True, False, None)}) == 3)

    # pandas reads a 1/0 column with blanks as floats. Refusing those would make
    # `suggest.train` unable to read a file the store had just written.
    check("a float 1.0 is the same fact as '1'", decode_audit(1.0) is True)
    check("...and 0.0 as '0'", decode_audit(0.0) is False)

    # A cell nobody can read is a FOURTH fact, and both ways of guessing it are
    # silent corruption in opposite directions.
    check("an unreadable flag raises rather than picking a side",
          "not a readable" in _refusal(decode_audit, "maybe"))
    check("...naming both legal values, so it can be fixed by hand",
          all(s in _refusal(decode_audit, "maybe") for s in ("'1'", "'0'")))
    check("...including a value that looks numeric but is neither",
          "not a readable" in _refusal(decode_audit, "2"))


def test_an_unreadable_flag_stops_the_project_opening_rather_than_being_averaged(
    root: Path
) -> None:
    """Loud, not lenient. While one row will not say which side it is on, no fit
    and no audit over this file means anything - and a store that quietly
    dropped the bad row would take its label with it."""
    cfg = _cfg(root, "corrupt")
    path = cfg.paths.csv
    rows = [_row(m) for m in _population(3)]
    rows[1][AUDIT_COLUMN] = "maybe"
    _write_csv(path, IDENTITY_COLUMNS + ["behaviour"] + TRAILING_COLUMNS, rows)

    refusal = _refusal(AnnotationStore, path, cfg)
    check("the store refuses to open the file at all", refusal != "", "it opened")
    check("...naming the file", path.name in refusal, refusal)
    check("...and the column", AUDIT_COLUMN in refusal, refusal)
    check("...and saying why it cannot simply carry on",
          "trained or audited" in refusal, refusal)
    check("the CSV reader refuses it identically, so neither path guesses",
          AUDIT_COLUMN in _refusal(read_audit_records, path))


def test_the_column_survives_rewrite_and_a_torn_append(root: Path) -> None:
    """``rewrite()`` runs on every edit, undo and clear, and a kill mid-append
    leaves half a line. A membership that decayed on either path would move a
    clip between the holdout and the training set as a side effect of ordinary
    use - the silent version of exactly the bug this column closes."""
    cfg = _cfg(root, "roundtrip")
    path = cfg.paths.csv
    store = AnnotationStore(path, cfg)

    metas = _population(40)
    for meta in metas:
        store.record(meta, {"behaviour": "swimming"},
                     selected_by={"rule": "uncertainty", "score": 0.41})
    ids = [m["clip_id"] for m in metas]
    expected = {c: cfg.suggest.audit_by_hash(c) for c in ids}
    check("the fixture holds clips on both sides",
          0 < sum(expected.values()) < len(ids), str(sum(expected.values())))
    check("every row was stamped as it was written", read_audit_records(path) == expected)

    before = path.read_bytes()
    store.rewrite()
    check("a rewrite of unchanged rows is byte-for-byte identical",
          path.read_bytes() == before)
    check("...and the memberships come back identical", read_audit_records(path) == expected)

    # The tear: a row cut mid-field, which the byte-level repair must drop whole.
    with open(path, "a", encoding="utf-8", newline="") as fh:
        fh.write('MOVIE__t0099__f004950,MOVIE,/movies/MOVIE.tif,0,0,99,4950,"{""ru')
    repaired, said = _load_quietly(path, cfg)
    check("the half-written row is dropped", len(repaired) == len(metas), str(len(repaired)))
    check("...loudly", any("partially written" in m for m in said), str(said))
    check("...and the file is back to what it was", path.read_bytes() == before)
    check("...with every membership intact and none re-derived",
          read_audit_records(path) == expected)

    # Annotating after a repair must produce a readable file, and the new row
    # must be stamped like any other.
    fresh = _meta(99, 4950)
    repaired.record(fresh, {"behaviour": "tumbling"})
    check("annotating after a repair still works",
          len(AnnotationStore(path, cfg)) == len(metas) + 1)
    check("...and the new row records its membership",
          read_audit_records(path)[fresh["clip_id"]]
          == cfg.suggest.audit_by_hash(fresh["clip_id"]))


def test_the_new_column_is_last_so_an_older_build_leaves_the_file_alone(
    root: Path
) -> None:
    """The downgrade direction, tested by running the old build's RULE: ``_load``
    treats an unrecognised column as ``extra`` and appends it to the end of its
    own list. That reconstruction matches the header on disk only if the newer
    columns are last, in the order they were added - and if it matches, the old
    build does not rewrite the log and does not drop the values it cannot read.

    Two vintages, because there are now two: the build that knows ``selected_by``
    and not ``audit_holdout``, and the one that predates both.
    """
    cfg = _cfg(root, "downgrade")
    store = AnnotationStore(cfg.paths.csv, cfg)
    store.record(_meta(1, 50), {"behaviour": "swimming"},
                 selected_by={"rule": "uncertainty", "score": 0.41})
    disk = _header(cfg.paths.csv)

    check("the audit column is the last one written",
          disk[-1] == AUDIT_COLUMN, str(disk[-3:]))
    check("...and the provenance column is still the one before it",
          disk[-2] == PROVENANCE_COLUMN, str(disk[-3:]))

    last_release = [c for c in disk if c != AUDIT_COLUMN]
    check("a build that knows selected_by reconstructs the header exactly",
          last_release + [AUDIT_COLUMN] == disk, str(last_release))
    older = _old_schema(cfg)
    check("...and so does one that predates both columns",
          older + [PROVENANCE_COLUMN, AUDIT_COLUMN] == disk, str(older))

    # An older build appends a row one field SHORT. The value it could not write
    # is unrecorded, which is the truth - and the next open has to stamp it
    # rather than leave it to be re-derived from a hash that may have moved on.
    with open(cfg.paths.csv, "a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=last_release, extrasaction="ignore")
        writer.writerow({**_row(_meta(2, 100)), PROVENANCE_COLUMN: ""})

    reopened, _ = _load_quietly(cfg.paths.csv, cfg)
    short = _meta(2, 100)["clip_id"]
    check("the short row still loads", reopened.is_done(short))
    check("...and its membership was stamped on the way in",
          read_audit_records(cfg.paths.csv).get(short) == cfg.suggest.audit_by_hash(short))
    check("...leaving the file back at full width",
          _header(cfg.paths.csv) == store.columns, str(_header(cfg.paths.csv)))
    check("the older build's blank provenance was NOT invented for it",
          reopened.get(short)[PROVENANCE_COLUMN] == "")


def test_the_reserved_names_still_cover_every_built_in_column(root: Path) -> None:
    """``config.RESERVED_GROUP_NAMES`` restates the store's column lists by hand.
    A column added to the store and forgotten there is not an error - it is a
    name a user is then allowed to give a category, after which ``record``
    writes the built-in and the category into one key and the built-in is gone.
    For this column that means a project whose audit membership is overwritten
    by a behaviour label, silently."""
    check("the reserved names are exactly the built-in columns",
          set(RESERVED_GROUP_NAMES) == set(IDENTITY_COLUMNS + TRAILING_COLUMNS),
          str(set(RESERVED_GROUP_NAMES) ^ set(IDENTITY_COLUMNS + TRAILING_COLUMNS)))
    check("the audit column is among them", AUDIT_COLUMN in RESERVED_GROUP_NAMES)

    collides = CONFIG_TEXT.replace("name: behaviour", f"name: {AUDIT_COLUMN}")
    check("a category named after it is refused at config load",
          AUDIT_COLUMN in _refusal(_cfg, root, "collide", collides),
          _refusal(_cfg, root, "collide", collides) or "it was ACCEPTED")


# --------------------------------------------------------------------------- #
# what the numbers say
# --------------------------------------------------------------------------- #
def _printed(reports: dict) -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        _print_report(reports)
    return buffer.getvalue()


def test_the_audit_count_is_never_reported_without_its_population(root: Path) -> None:
    """"10 audit clips, all correct" and "10 of 200 audit clips have been
    labelled" are the same ten clips and completely different claims - the first
    reads as a result, the second as a result resting on 5% of the holdout. The
    count was reported alone, so the two were indistinguishable in the one place
    the tool asks people to trust a number."""
    base = {
        "group": "behaviour", "n_labels": 180, "n_orphaned": 0, "orphaned_by_class": {},
        "n_classes": 2, "classes": ["swimming", "tumbling"],
        "per_class_counts": {"swimming": 120, "tumbling": 60}, "dropped_classes": {},
        "macro_f1": 0.71, "accuracy": 0.78, "baseline_accuracy": 0.66,
        "kappa": 0.44, "kappa_lower": 0.21, "n_groups": 4,
        "scope": "leave-movie-out over 4 movies (4-fold)",
        "cv_pca": "per-fold CNN PCA (saved model uses the global one)",
        "audit_pca": "global CNN PCA, fitted over every prepared movie INCLUDING "
                     "these audit clips' own detections - unsupervised leakage",
        "audit_n": 10, "audit_population": 200, "audit_labelled": 10,
        "audit_accuracy": 0.80, "audit_kappa": 0.51,
        "confusion": [], "top_features": [], "reliable": True, "reason": "ok",
    }
    text = _printed({"behaviour": base})
    check("the accuracy is still reported", "0.800" in text, text)
    check("...next to how many audit clips were scored", "10 scored" in text, text)
    check("...and how much of the holdout that is",
          "10 of 200 audit clips labelled" in text, text)
    check("the unsupervised leakage is admitted where the number is printed",
          "leakage" in text and "caveat" in text, text)

    # A holdout that exists but has barely been touched must not read as silence.
    thin = dict(base, audit_n=0, audit_labelled=2,
                audit_accuracy=float("nan"), audit_kappa=float("nan"))
    text = _printed({"behaviour": thin})
    check("a holdout too thin to score says so rather than vanishing",
          "2 of 200 audit clips labelled" in text, text)
    check("...and says what is missing, not just that it is missing",
          "no unbiased estimate" in text, text)

    none_at_all = dict(base, audit_n=0, audit_population=0, audit_labelled=0)
    check("a project with no holdout prints no audit line at all",
          "audit holdout" not in _printed({"behaviour": none_at_all}))


def test_the_report_counts_the_whole_holdout_not_only_the_labelled_part(
    root: Path
) -> None:
    """The population comes off the feature table - the clips that could be
    scored - so it answers "how much of this holdout is left to spend", which is
    the question a thin audit number actually raises."""
    cfg = _cfg(root, "population")
    path = cfg.paths.csv
    metas = _population(120)
    audit = [m for m in metas if cfg.suggest.audit_by_hash(m["clip_id"])]
    check("the fixture has a holdout to count", len(audit) >= 5, str(len(audit)))

    # Two of the audit clips are labelled; the rest of the holdout is untouched.
    labelled = audit[:2] + [m for m in metas if m not in audit][:8]
    _write_csv(path, _old_schema(cfg), [_row(m) for m in labelled])
    _load_quietly(path, cfg)

    labels = pd.read_csv(path)
    features = pd.DataFrame({"clip_id": [m["clip_id"] for m in metas],
                             "movie_id": "MOVIE",
                             "speed_mean": range(len(metas))})
    _, report = train_group(features, labels, "behaviour", cfg,
                            columns=["speed_mean"], verbose=False)

    check("the population is the whole holdout, not the labelled part of it",
          report.audit_population == len(audit),
          f"{report.audit_population} vs {len(audit)}")
    check("...and the labelled part is reported separately",
          report.audit_labelled == 2, str(report.audit_labelled))
    check("...and they are different numbers here, so the distinction is live",
          report.audit_population > report.audit_labelled)
    check("nothing was scored, so audit_n claims nothing",
          report.audit_n == 0, str(report.audit_n))
    check("the unfitted report still admits how its audit score would be built",
          isinstance(report.audit_pca, str) and report.audit_pca != "")

    text = _printed({"behaviour": asdict(report)})
    check("and the printed report puts the two together",
          f"2 of {len(audit)} audit clips labelled" in text, text)


def _fitted_project(root: Path, name: str, columns: list[str], fold_pca: bool = False):
    """A project small enough to fit in a second and complete enough to reach
    the audit score: two movies, two classes present in both, and half the clips
    held out so the holdout is big enough to be scored.

    `fold_pca` supplies the raw CNN aggregate, which is what makes `train_group`
    refit the basis inside each fold. Without it the CV falls back to the global
    basis and `cv_pca` says so - a caller asserting the CV/audit CONTRAST has to
    ask for it, or it is comparing two descriptions of the same global fit.
    """
    text = CONFIG_TEXT + "\nsuggest: {model: hist, audit_fraction: 0.5, " \
                         "min_per_class: 2, iterations: 20}\n"
    cfg = _cfg(root, name, text)
    metas, behaviours = [], []
    for movie in ("MOVIE_A", "MOVIE_B"):
        for track in range(1, 31):
            metas.append(_meta(track, track * 50, movie))
            behaviours.append("swimming" if track % 2 else "tumbling")

    _write_csv(cfg.paths.csv, _old_schema(cfg),
               [_row(m, b) for m, b in zip(metas, behaviours)])
    _load_quietly(cfg.paths.csv, cfg)

    features = pd.DataFrame({
        "clip_id": [m["clip_id"] for m in metas],
        "movie_id": [m["movie_id"] for m in metas],
        "track_id": [m["track_id"] for m in metas],
        "f_start": [m["f_start"] for m in metas],
        # Separable, so the fit has something to find and the audit score is a
        # number rather than noise. Nothing here claims it is a good number.
        "speed_mean": [1.0 if b == "swimming" else 9.0 for b in behaviours],
        "cnn_00": [0.5 if b == "swimming" else 8.5 for b in behaviours],
    })
    labels = pd.read_csv(cfg.paths.csv)
    # The positional tag train_group needs before it will use a fold basis at
    # all; harmless when no aggregate is supplied.
    features["_row"] = np.arange(len(features))
    raw_cnn = None
    if fold_pca:
        # Three columns because the fold PCA needs n_components >= 2 to run,
        # separable so the fold fit has something to find.
        base = np.array([0.5 if b == "swimming" else 8.5 for b in behaviours])
        raw_cnn = np.column_stack([base, base * 0.5 + 1.0, base * -0.25])
    _, report = train_group(features, labels, "behaviour", cfg,
                            columns=columns, verbose=False, raw_cnn=raw_cnn)
    return cfg, features, labels, report


def test_the_audit_pca_caveat_is_recorded_on_a_report_that_actually_scored(
    root: Path
) -> None:
    """The audit score is computed on ``columns``, whose ``cnn*`` block comes
    from a PCA fitted in ``clipfeatures`` over every prepared movie - these
    clips' own detections included. The CV refits that basis per fold; the audit
    path does not. Unsupervised only, and probably small, but the size of it is
    not measured anywhere, and it inflates the one number this tool asks people
    to trust. ``cv_pca`` documented the cross-validated side and nothing
    documented this one, so a reader had to find it in the source."""
    _, _, _, report = _fitted_project(root, "pca-caveat", ["speed_mean", "cnn_00"],
                                      fold_pca=True)

    check("the fit really reached the audit score", report.audit_n >= 3,
          f"audit_n={report.audit_n} reason={report.reason}")
    check("the caveat names the fit it is a caveat about",
          "global CNN PCA" in report.audit_pca, report.audit_pca)
    check("...says whose data went into it",
          "INCLUDING these audit clips" in report.audit_pca, report.audit_pca)
    check("...says what kind of leakage it is, so it is not overstated",
          "unsupervised leakage" in report.audit_pca
          and "labels never enter" in report.audit_pca, report.audit_pca)
    check("...and contrasts it with the CV path, which does refit",
          "per audit clip" in report.audit_pca, report.audit_pca)
    check("the CV side is still documented separately, and differently",
          report.cv_pca != report.audit_pca and "fold" in report.cv_pca)

    _, _, _, plain = _fitted_project(root, "pca-none", ["speed_mean"])
    check("a CNN-free model carries no such caveat, because there is none",
          plain.audit_pca == "no CNN columns", plain.audit_pca)
    check("...and it still scored an audit set", plain.audit_n >= 3, str(plain.audit_n))

    # The report is written to suggest_report.json, so a field that is not a
    # plain scalar would break the write rather than the read.
    import json

    payload = json.loads(json.dumps(asdict(report)))
    check("the new fields survive the JSON report",
          payload["audit_pca"] == report.audit_pca
          and payload["audit_population"] == report.audit_population
          and payload["audit_labelled"] == report.audit_labelled)


def test_a_scored_audit_report_states_how_much_of_the_holdout_it_rests_on(
    root: Path
) -> None:
    """The end of the complaint, on a report that ran the whole way through: an
    accuracy, the clips it was computed over, and the size of the holdout those
    came from - so "0.80" can be read as what it is rather than as a verdict."""
    cfg, features, _, report = _fitted_project(root, "scored", ["speed_mean"])

    expected = sum(cfg.suggest.audit_by_hash(c) for c in features["clip_id"])
    check("the population counts the project's whole holdout",
          report.audit_population == expected,
          f"{report.audit_population} vs {expected}")
    check("...every one of which is labelled here, so the two agree",
          report.audit_labelled == report.audit_population,
          f"{report.audit_labelled} vs {report.audit_population}")
    check("...and the scored count cannot exceed what was labelled",
          report.audit_n <= report.audit_labelled, str(report.audit_n))
    check("the audit accuracy is a real number", report.audit_accuracy == report.audit_accuracy,
          str(report.audit_accuracy))
    check("the training set excludes every one of those clips",
          report.n_labels + report.audit_labelled == len(features),
          f"{report.n_labels} + {report.audit_labelled} vs {len(features)}")

    text = _printed({"behaviour": asdict(report)})
    check("and the printed line carries the accuracy and the denominator together",
          f"{report.audit_labelled} of {report.audit_population} audit clips labelled" in text
          and f"{report.audit_n} scored" in text, text)


def test_train_reads_the_membership_it_is_handed_rather_than_deriving_it(
    root: Path
) -> None:
    """``audit_records`` and ``store.read_audit_records`` are two readings of one
    file - one from a DataFrame, one from the CSV. They must agree exactly: a
    disagreement about a single cell would put a clip in the training set for
    the fit and in the holdout for the GUI, and the two would each be internally
    consistent."""
    cfg = _cfg(root, "readers")
    path = cfg.paths.csv
    metas = _population(50)
    _write_csv(path, _old_schema(cfg), [_row(m) for m in metas])
    _load_quietly(path, cfg)

    from_file = read_audit_records(path)
    from_frame = audit_records(pd.read_csv(path))
    check("the two readings agree cell for cell", from_file == from_frame,
          str(set(from_file.items()) ^ set(from_frame.items())))
    check("...and both found the whole file", len(from_file) == len(metas))

    exact = audit_records(pd.read_csv(path, dtype=str, keep_default_na=False))
    check("...as does the dtype=str reading the repair path uses", exact == from_file)

    check("a table without the column yields no records rather than guessing",
          audit_records(pd.read_csv(path).drop(columns=[AUDIT_COLUMN])) == {})
    check("...and so does no table at all", audit_records(None) == {})
    check("a clip that is not in the mapping falls back to the hash",
          is_audit_clip("MOVIE__t9999__f999999", cfg, from_file)
          == cfg.suggest.audit_by_hash("MOVIE__t9999__f999999"))
    check("...and the two-argument form is still the hash, unchanged",
          all(is_audit_clip(m["clip_id"], cfg)
              == cfg.suggest.audit_by_hash(m["clip_id"]) for m in metas))
