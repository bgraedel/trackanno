"""Audit holdout membership is recorded at labelling time and survives re-detection."""

from __future__ import annotations

import contextlib
import csv
import io
import warnings
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest

from trackanno.config import RESERVED_GROUP_NAMES, load_config
from trackanno.annotate.reassociate import apply_reassociation, plan_reassociation, write_labels
from trackanno.annotate.store import (
    AUDIT_COLUMN, IDENTITY_COLUMNS, PROVENANCE_COLUMN, TRAILING_COLUMNS,
    AnnotationStore, decode_audit, encode_audit, read_audit_records,
)
from trackanno.learn.suggest import AuditHoldout, _print_report, train_group


@pytest.fixture(autouse=True)
def _short_fits(monkeypatch):
    """Short fits: the tests need a model, not a good one."""
    monkeypatch.setattr("trackanno.learn.suggest._iterations", lambda cfg: 20)


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

N_CLIPS = 200

# Offset a re-detection adds to every track id; large enough that no new id collides.
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
    """The clip_id format of ``clips.py``, which embeds the track id."""
    return f"{movie}__t{track:04d}__f{f_start:06d}"


def _meta(track: int, f_start: int, movie: str = "MOVIE") -> dict:
    return {
        "clip_id": _clip_id(track, f_start, movie), "movie_id": movie,
        "source_file": f"/movies/{movie}.tif", "series": 0, "channel": 0,
        "track_id": track, "f_start": f_start, "f_end": f_start + 49,
        # A unique crop centre per (movie, f_start), so reassociation has one candidate.
        "crop_x0": 10 + track, "crop_y0": 20 + track, "crop_w": 96, "crop_h": 96,
        "scale": 1.0,
    }


def _population(n: int = N_CLIPS) -> list[dict]:
    """One clip per track, each at its own start frame."""
    return [_meta(track, track * 50) for track in range(1, n + 1)]


def _old_schema(cfg) -> list[str]:
    """A header without the provenance and audit columns."""
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


def _held_out(holdout, clip_ids) -> set[str]:
    """The ids of ``clip_ids`` that ``holdout`` holds out."""
    return {c for c in clip_ids if c in holdout}


def _renumbered_manifest(metas: list[dict]) -> pd.DataFrame:
    """The manifest after a re-detection: same clips, every track id shifted."""
    rows = []
    for meta in metas:
        new = dict(meta)
        new["track_id"] = meta["track_id"] + RENUMBER_BY
        new["clip_id"] = _clip_id(new["track_id"], meta["f_start"], meta["movie_id"])
        rows.append(new)
    return pd.DataFrame(rows)


def _redetect(path: Path, manifest: pd.DataFrame) -> tuple[int, dict[str, str]]:
    """Reassociate the CSV to ``manifest``; return the match count and old -> new ids."""
    exact = pd.read_csv(path, dtype=str, keep_default_na=False)
    plan = plan_reassociation(exact, manifest)
    write_labels(path, apply_reassociation(exact, plan))
    return len(plan.matched), {m.old_clip_id: m.new_clip_id for m in plan.matched}


def _upgraded(root: Path, name: str, n: int = N_CLIPS):
    """A project whose old-schema CSV was opened once, freezing membership."""
    cfg = _cfg(root, name)
    metas = _population(n)
    _write_csv(cfg.paths.csv, _old_schema(cfg), [_row(m) for m in metas])
    _load_quietly(cfg.paths.csv, cfg)
    return cfg, metas


# --------------------------------------------------------------------------- #
# the upgrade
# --------------------------------------------------------------------------- #
def test_upgrading_an_existing_project_moves_no_clip_into_or_out_of_the_holdout(
    root: Path
) -> None:
    """The first open records what the hash says today, so nothing moves."""
    cfg = _cfg(root, "upgrade")
    path = cfg.paths.csv
    metas = _population()
    _write_csv(path, _old_schema(cfg), [_row(m) for m in metas])

    ids = [m["clip_id"] for m in metas]
    before_rows = _read_csv(path)
    before = _held_out(AuditHoldout.by_hash(cfg), ids)
    assert AUDIT_COLUMN not in _header(path)
    assert 0 < len(before) < len(ids), f"{len(before)} of {len(ids)}"

    store, said = _load_quietly(path, cfg)
    records = read_audit_records(path)
    after = _held_out(AuditHoldout.from_csv(cfg, path), ids)

    assert len(store) == len(metas)
    assert len(records) == len(metas), "every labelled clip records a membership"
    assert after == before, f"clips changed side: {sorted(after ^ before)[:5]}"
    assert all(records[c] == cfg.suggest.audit_by_hash(c) for c in ids)

    after_rows = _read_csv(path)
    assert all(old == {k: v for k, v in new.items() if k in old}
               for old, new in zip(before_rows, after_rows)), (
        "the upgrade touched no other column")
    assert [r["clip_id"] for r in after_rows] == ids, "the rows kept their order"
    assert sum(1 for m in said if AUDIT_COLUMN in m and "moves no clip" in m) == 1, said

    again, said_again = _load_quietly(path, cfg)
    assert not any(AUDIT_COLUMN in m for m in said_again), (
        f"re-opening says nothing about the audit column: {said_again}")
    assert _held_out(AuditHoldout.from_csv(cfg, path), ids) == before
    assert all(decode_audit(again.get(c)[AUDIT_COLUMN]) == (c in before) for c in ids)


def test_a_project_with_no_holdout_records_that_too_rather_than_leaving_it_open(
    root: Path
) -> None:
    """``audit_fraction: 0`` stamps every clip as not held out, so raising it later moves none."""
    text = CONFIG_TEXT + "\nsuggest: {audit_fraction: 0.0}\n"
    cfg = _cfg(root, "no-holdout", text)
    path = cfg.paths.csv
    metas = _population(20)
    _write_csv(path, _old_schema(cfg), [_row(m) for m in metas])

    _load_quietly(path, cfg)
    records = read_audit_records(path)
    assert cfg.suggest.audit_fraction == 0.0
    assert len(records) == len(metas), "every labelled clip records a membership"
    assert set(records.values()) == {False}

    later = _cfg(root, "no-holdout-then-on", CONFIG_TEXT)
    ids = [m["clip_id"] for m in metas]
    assert any(later.suggest.audit_by_hash(c) for c in ids), (
        "raising audit_fraction would have hashed some of them in")
    holdout = AuditHoldout.from_csv(later, path)
    assert _held_out(holdout, ids) == set(), "already-judged clips stay out"


# --------------------------------------------------------------------------- #
# re-detection
# --------------------------------------------------------------------------- #
def test_a_labelled_clip_keeps_its_membership_across_a_real_reassociation(
    root: Path
) -> None:
    """Reassociation rewrites clip_id, but the recorded membership travels with the row."""
    cfg, metas = _upgraded(root, "redetect")
    path = cfg.paths.csv

    old_ids = [m["clip_id"] for m in metas]
    frozen = _held_out(AuditHoldout.from_csv(cfg, path), old_ids)

    n_matched, moved_to = _redetect(path, _renumbered_manifest(metas))
    new_ids = [moved_to[c] for c in old_ids]
    assert not set(old_ids) & set(new_ids) and n_matched == len(metas), n_matched

    # Without the record roughly 2f(1-f) of clips would change side.
    by_hash_now = {moved_to[c] for c in old_ids if cfg.suggest.audit_by_hash(moved_to[c])}
    flipped = sum(1 for c in old_ids if (c in frozen) != (moved_to[c] in by_hash_now))
    assert 0.10 < flipped / len(old_ids) < 0.30, f"{flipped} of {len(old_ids)} would move"

    store, said = _load_quietly(path, cfg)
    now = _held_out(AuditHoldout.from_csv(cfg, path), new_ids)

    assert len(store) == len(metas), "the repair re-linked every row"
    assert now == {moved_to[c] for c in frozen}, "every labelled clip kept its side"
    assert now != by_hash_now, "the renumbering was a no-op; widen it"
    assert len(now) == len(frozen)
    assert store.get(new_ids[0])["behaviour"] == "swimming"
    assert int(store.get(new_ids[0])["track_id"]) == metas[0]["track_id"] + RENUMBER_BY

    counted = [m for m in said if "recorded when they were labelled" in m]
    assert len(counted) == 1, f"the disagreement is reported once, as a count: {said}"
    assert str(flipped) in counted[0], counted[0]
    assert "Nothing to fix" in counted[0], counted[0]


def test_a_clip_that_was_held_out_does_not_enter_the_training_set_when_re_linked(
    root: Path
) -> None:
    """``train_group`` honours the recorded flag even where the hash says otherwise."""
    cfg = _cfg(root, "training-set")
    path = cfg.paths.csv

    # Ten clips that hash out, so anything held out is held out by the record.
    metas = [m for m in _population(60) if not cfg.suggest.audit_by_hash(m["clip_id"])][:10]
    assert not any(cfg.suggest.audit_by_hash(m["clip_id"]) for m in metas)

    held = metas[0]["clip_id"]
    rows = [_row(m) for m in metas]
    rows[0][AUDIT_COLUMN] = encode_audit(True)
    _write_csv(path, IDENTITY_COLUMNS + ["behaviour"] + TRAILING_COLUMNS, rows)

    labels = pd.read_csv(path)
    features = pd.DataFrame({"clip_id": [m["clip_id"] for m in metas],
                             "movie_id": "MOVIE",
                             "speed_mean": range(len(metas))})

    _, report = train_group(features, labels, "behaviour", cfg,
                            columns=["speed_mean"], verbose=False)
    assert report.audit_labelled == 1, report.audit_labelled
    assert report.n_labels == len(metas) - 1, "the held-out clip is not trained on"
    assert report.audit_population == 1, report.audit_population

    forgotten = labels.drop(columns=[AUDIT_COLUMN])
    _, without = train_group(features, forgotten, "behaviour", cfg,
                             columns=["speed_mean"], verbose=False)
    assert without.n_labels == len(metas) and without.audit_labelled == 0, (
        "without the record the clip would be trained on")
    assert not cfg.suggest.audit_by_hash(held), "the record, not the hash, keeps it out"


def test_an_unlabelled_clip_is_still_free_to_change_sides(root: Path) -> None:
    """Only labelled clips are frozen; unlabelled ones follow the hash on their current id."""
    cfg = _cfg(root, "unlabelled")
    path = cfg.paths.csv
    metas = _population()

    labelled, untouched = metas[:100], metas[100:]
    _write_csv(path, _old_schema(cfg), [_row(m) for m in labelled])
    _load_quietly(path, cfg)

    _redetect(path, _renumbered_manifest(metas))
    records = read_audit_records(path)
    holdout = AuditHoldout.from_csv(cfg, path)
    moved = {m["clip_id"]: _clip_id(m["track_id"] + RENUMBER_BY, m["f_start"])
             for m in metas}

    assert not any(moved[m["clip_id"]] in records or m["clip_id"] in records
                   for m in untouched), "no unlabelled clip carries a record"
    assert all((moved[m["clip_id"]] in holdout)
               == cfg.suggest.audit_by_hash(moved[m["clip_id"]]) for m in untouched)

    changed = [m for m in untouched
               if cfg.suggest.audit_by_hash(m["clip_id"])
               != cfg.suggest.audit_by_hash(moved[m["clip_id"]])]
    assert changed, "some unlabelled clips changed side"
    assert all((moved[m["clip_id"]] in holdout) == cfg.suggest.audit_by_hash(m["clip_id"])
               for m in labelled), "no labelled clip changed side"

    # Labelling one freezes it.
    store = AnnotationStore(path, cfg)
    joining = moved[changed[0]["clip_id"]]
    row = store.record(dict(_meta(changed[0]["track_id"] + RENUMBER_BY,
                                  changed[0]["f_start"])),
                       {"behaviour": "tumbling"})
    assert decode_audit(row[AUDIT_COLUMN]) == cfg.suggest.audit_by_hash(joining)
    assert read_audit_records(path).get(joining) == cfg.suggest.audit_by_hash(joining)


def test_editing_a_re_linked_label_does_not_re_derive_its_membership(
    root: Path
) -> None:
    """Edits and undo keep the recorded flag; clearing the label clears it."""
    cfg = _cfg(root, "edit")
    path = cfg.paths.csv
    meta = _meta(RENUMBER_BY + 1, 50)
    clip_id = meta["clip_id"]
    assert not cfg.suggest.audit_by_hash(clip_id), "the clip hashes out of the holdout"

    row = _row(meta)
    row[AUDIT_COLUMN] = encode_audit(True)
    _write_csv(path, IDENTITY_COLUMNS + ["behaviour"] + TRAILING_COLUMNS, [row])

    store = AnnotationStore(path, cfg)
    assert decode_audit(store.get(clip_id)[AUDIT_COLUMN]) is True

    edited = store.record(meta, {"behaviour": "tumbling"}, notes="second look")
    assert edited["behaviour"] == "tumbling"
    assert decode_audit(edited[AUDIT_COLUMN]) is True, "still held out after an edit"
    assert read_audit_records(path)[clip_id] is True
    assert store.undo() == clip_id
    assert read_audit_records(path)[clip_id] is True, "still held out after an undo"

    store.clear(clip_id)
    assert clip_id not in read_audit_records(path), "clearing the label clears the record"
    fresh = AnnotationStore(path, cfg).record(meta, {"behaviour": "swimming"})
    assert decode_audit(fresh[AUDIT_COLUMN]) is False, "a fresh label takes the hash"


# --------------------------------------------------------------------------- #
# the codec and the file
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("blank", ["", None, "  ", float("nan")])
def test_a_blank_audit_cell_decodes_to_not_recorded(blank) -> None:
    assert decode_audit(blank) is None


def test_the_audit_codec_round_trips_three_distinct_values() -> None:
    for value in (True, False, None):
        assert decode_audit(encode_audit(value)) is value
    assert len({str(decode_audit(encode_audit(v))) for v in (True, False, None)}) == 3
    # pandas reads a 1/0 column with blanks as floats.
    assert decode_audit(1.0) is True
    assert decode_audit(0.0) is False


@pytest.mark.parametrize("bad", ["maybe", "2"])
def test_an_unreadable_audit_flag_raises_naming_the_legal_values(bad) -> None:
    with pytest.raises((ValueError, TypeError), match="not a readable") as caught:
        decode_audit(bad)
    assert "'1'" in str(caught.value) and "'0'" in str(caught.value)


def test_every_clip_of_a_track_lands_on_the_same_side(root: Path) -> None:
    """The holdout is drawn per track, so no track straddles the split."""
    cfg = _cfg(root, "per_track", CONFIG_TEXT.replace(
        "ui: {annotator: tester}", "ui: {annotator: tester}\nsuggest: {audit_fraction: 0.5}"))
    held = cfg.suggest.audit_by_hash
    sides = {}
    for movie in ("MOVIE", "OTHER__2024"):
        for track in range(1, 41):
            sides[(movie, track)] = {held(_clip_id(track, f, movie)) for f in range(0, 5000, 50)}
    assert all(len(s) == 1 for s in sides.values()), "every track's clips share one side"
    assert {next(iter(s)) for s in sides.values()} == {True, False}
    assert any(sides[("MOVIE", t)] != sides[("OTHER__2024", t)] for t in range(1, 41)), (
        "the movie is part of the key")


def test_an_unreadable_flag_stops_the_project_opening_rather_than_being_averaged(
    root: Path
) -> None:
    cfg = _cfg(root, "corrupt")
    path = cfg.paths.csv
    rows = [_row(m) for m in _population(3)]
    rows[1][AUDIT_COLUMN] = "maybe"
    _write_csv(path, IDENTITY_COLUMNS + ["behaviour"] + TRAILING_COLUMNS, rows)

    with pytest.raises((ValueError, TypeError)) as caught:
        AnnotationStore(path, cfg)
    refusal = str(caught.value)
    assert path.name in refusal, refusal
    assert AUDIT_COLUMN in refusal, refusal
    assert "use '1', '0' or empty" in refusal, refusal
    with pytest.raises((ValueError, TypeError), match=AUDIT_COLUMN):
        read_audit_records(path)


@pytest.fixture
def stamped(root: Path):
    """A store with 40 recorded clips on both sides of the holdout."""
    cfg = _cfg(root, "roundtrip")
    path = cfg.paths.csv
    store = AnnotationStore(path, cfg)
    metas = _population(40)
    for meta in metas:
        store.record(meta, {"behaviour": "swimming"},
                     selected_by={"rule": "margin", "score": 0.41})
    expected = {m["clip_id"]: cfg.suggest.audit_by_hash(m["clip_id"]) for m in metas}
    assert 0 < sum(expected.values()) < len(metas)
    return cfg, store, metas, expected


def test_every_recorded_row_is_stamped_and_survives_a_rewrite(stamped) -> None:
    cfg, store, _, expected = stamped
    path = cfg.paths.csv
    assert read_audit_records(path) == expected

    before = path.read_bytes()
    store.rewrite()
    assert path.read_bytes() == before, "a rewrite of unchanged rows is byte-identical"
    assert read_audit_records(path) == expected


def test_a_torn_append_is_dropped_and_the_memberships_kept(stamped) -> None:
    cfg, _, metas, expected = stamped
    path = cfg.paths.csv
    before = path.read_bytes()
    with open(path, "a", encoding="utf-8", newline="") as fh:
        fh.write('MOVIE__t0099__f004950,MOVIE,/movies/MOVIE.tif,0,0,99,4950,"{""ru')
    repaired, said = _load_quietly(path, cfg)
    assert len(repaired) == len(metas), "the half-written row is dropped"
    assert any("partially written" in m for m in said), said
    assert path.read_bytes() == before
    assert read_audit_records(path) == expected

    fresh = _meta(99, 4950)
    repaired.record(fresh, {"behaviour": "tumbling"})
    assert len(AnnotationStore(path, cfg)) == len(metas) + 1
    assert (read_audit_records(path)[fresh["clip_id"]]
            == cfg.suggest.audit_by_hash(fresh["clip_id"]))


def test_the_new_column_is_last_so_an_older_build_leaves_the_file_alone(
    root: Path
) -> None:
    """New columns are appended last, so older builds reconstruct the header exactly."""
    cfg = _cfg(root, "downgrade")
    store = AnnotationStore(cfg.paths.csv, cfg)
    store.record(_meta(1, 50), {"behaviour": "swimming"},
                 selected_by={"rule": "margin", "score": 0.41})
    disk = _header(cfg.paths.csv)

    assert disk[-1] == AUDIT_COLUMN, disk[-3:]
    assert disk[-2] == PROVENANCE_COLUMN, disk[-3:]

    last_release = [c for c in disk if c != AUDIT_COLUMN]
    assert last_release + [AUDIT_COLUMN] == disk
    assert _old_schema(cfg) + [PROVENANCE_COLUMN, AUDIT_COLUMN] == disk

    # An older build appends a row one field short; the next open stamps it.
    with open(cfg.paths.csv, "a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=last_release, extrasaction="ignore")
        writer.writerow({**_row(_meta(2, 100)), PROVENANCE_COLUMN: ""})

    reopened, _ = _load_quietly(cfg.paths.csv, cfg)
    short = _meta(2, 100)["clip_id"]
    assert reopened.is_done(short), "the short row still loads"
    assert read_audit_records(cfg.paths.csv).get(short) == cfg.suggest.audit_by_hash(short)
    assert _header(cfg.paths.csv) == store.columns, "the file is back at full width"
    assert reopened.get(short)[PROVENANCE_COLUMN] == "", "no provenance is invented"


def test_the_reserved_names_still_cover_every_built_in_column(root: Path) -> None:
    builtin = set(IDENTITY_COLUMNS + TRAILING_COLUMNS)
    assert set(RESERVED_GROUP_NAMES) == builtin, set(RESERVED_GROUP_NAMES) ^ builtin
    assert AUDIT_COLUMN in RESERVED_GROUP_NAMES

    collides = CONFIG_TEXT.replace("name: behaviour", f"name: {AUDIT_COLUMN}")
    with pytest.raises((ValueError, TypeError), match=AUDIT_COLUMN):
        _cfg(root, "collide", collides)


# --------------------------------------------------------------------------- #
# what the numbers say
# --------------------------------------------------------------------------- #
def _printed(reports: dict) -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        _print_report(reports)
    return buffer.getvalue()


_REPORT = {
    "group": "behaviour", "n_labels": 180, "n_orphaned": 0, "orphaned_by_class": {},
    "n_classes": 2, "classes": ["swimming", "tumbling"],
    "per_class_counts": {"swimming": 120, "tumbling": 60}, "dropped_classes": {},
    "macro_f1": 0.71, "accuracy": 0.78, "baseline_accuracy": 0.66,
    "kappa": 0.44, "kappa_lower": 0.21, "n_groups": 4,
    "scope": "leave-movie-out over 4 movies (4-fold)",
    "audit_n": 10, "audit_population": 200, "audit_labelled": 10,
    "audit_accuracy": 0.80, "audit_kappa": 0.51,
    "confusion": [], "top_features": [], "reliable": True, "reason": "ok",
}


def test_the_audit_accuracy_is_printed_with_its_population() -> None:
    text = _printed({"behaviour": _REPORT})
    assert "0.800" in text, text
    assert "10 scored" in text, text
    assert "10 of 200 audit clips labelled" in text, text


def test_a_holdout_too_thin_to_score_says_so() -> None:
    thin = dict(_REPORT, audit_n=0, audit_labelled=2,
                audit_accuracy=float("nan"), audit_kappa=float("nan"))
    text = _printed({"behaviour": thin})
    assert "2 of 200 audit clips labelled" in text, text
    assert "no unbiased estimate" in text, text


def test_a_project_with_no_holdout_says_so() -> None:
    none_at_all = dict(_REPORT, audit_n=0, audit_population=0, audit_labelled=0)
    text = _printed({"behaviour": none_at_all})
    assert "audit holdout: none" in text, text
    assert "fitted on" in text, text
    assert "kappa n/a" not in text and "audit" not in text.split("kappa")[-1], (
        f"no audit kappa is claimed: {text}")


def test_a_report_older_than_the_population_count_still_prints_its_audit_line() -> None:
    """A report without ``audit_population`` prints its audit line and names the missing count."""
    old = {k: v for k, v in _REPORT.items()
           if k not in ("audit_population", "audit_labelled")}
    text = _printed({"behaviour": old})
    assert "audit holdout" in text, text
    assert "0.800" in text, text
    assert "+0.510" in text, text
    assert "never suggested nor trained on" in text, text
    assert "predates the audit population count" in text, text
    assert "retrain" in text, text
    assert "of 0 audit clips" not in text and "0 audit clips labelled" not in text, (
        f"no denominator is invented: {text}")

    none_at_all = dict(old, audit_n=0, audit_population=0, audit_labelled=0,
                       audit_accuracy=float("nan"), audit_kappa=float("nan"))
    assert "predates" not in _printed({"behaviour": none_at_all}), (
        "no holdout does not read as an out-of-date report")
    assert "audit holdout: none" not in text, "an out-of-date report does not read as no holdout"


def test_the_report_counts_the_whole_holdout_not_only_the_labelled_part(
    root: Path
) -> None:
    """The audit population comes off the feature table, not the labels."""
    cfg = _cfg(root, "population")
    path = cfg.paths.csv
    metas = _population(120)
    audit = [m for m in metas if cfg.suggest.audit_by_hash(m["clip_id"])]
    assert len(audit) >= 5, len(audit)

    labelled = audit[:2] + [m for m in metas if m not in audit][:8]
    _write_csv(path, _old_schema(cfg), [_row(m) for m in labelled])
    _load_quietly(path, cfg)

    labels = pd.read_csv(path)
    features = pd.DataFrame({"clip_id": [m["clip_id"] for m in metas],
                             "movie_id": "MOVIE",
                             "speed_mean": range(len(metas))})
    _, report = train_group(features, labels, "behaviour", cfg,
                            columns=["speed_mean"], verbose=False)

    assert report.audit_population == len(audit), (report.audit_population, len(audit))
    assert report.audit_labelled == 2, report.audit_labelled
    assert report.audit_n == 0, "nothing was scored"

    text = _printed({"behaviour": asdict(report)})
    assert f"2 of {len(audit)} audit clips labelled" in text, text


def _fitted_project(root: Path, name: str, columns: list[str]):
    """Two movies, two separable classes, half the clips held out; returns the fit."""
    text = CONFIG_TEXT + "\nsuggest: {model: hist, audit_fraction: 0.5, " \
                         "min_per_class: 2}\n"
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
        "speed_mean": [1.0 if b == "swimming" else 9.0 for b in behaviours],
    })
    labels = pd.read_csv(cfg.paths.csv)
    _, report = train_group(features, labels, "behaviour", cfg,
                            columns=columns, verbose=False)
    return cfg, features, labels, report


def test_a_scored_audit_report_states_how_much_of_the_holdout_it_rests_on(
    root: Path
) -> None:
    cfg, features, _, report = _fitted_project(root, "scored", ["speed_mean"])

    expected = sum(cfg.suggest.audit_by_hash(c) for c in features["clip_id"])
    assert report.audit_population == expected, (report.audit_population, expected)
    assert report.audit_labelled == report.audit_population
    assert report.audit_n <= report.audit_labelled
    assert report.audit_accuracy == report.audit_accuracy, "the audit accuracy is not NaN"
    assert report.n_labels + report.audit_labelled == len(features), (
        "the training set excludes every audit clip")

    text = _printed({"behaviour": asdict(report)})
    assert f"{report.audit_labelled} of {report.audit_population} audit clips labelled" in text
    assert f"{report.audit_n} scored" in text, text


@pytest.fixture
def readers(root: Path):
    """An upgraded 50-clip project and its two holdout readings."""
    cfg, metas = _upgraded(root, "readers", 50)
    path = cfg.paths.csv
    return (cfg, metas, AuditHoldout.from_csv(cfg, path),
            AuditHoldout.from_labels(cfg, pd.read_csv(path)))


def test_the_csv_and_labels_frame_readings_agree(readers) -> None:
    cfg, metas, from_file, from_frame = readers
    path = cfg.paths.csv
    assert from_file.recorded == from_frame.recorded, (
        set(from_file.recorded.items()) ^ set(from_frame.recorded.items()))
    assert from_file.n_recorded == len(metas)
    assert read_audit_records(path) == from_file.recorded
    exact = AuditHoldout.from_labels(cfg, pd.read_csv(path, dtype=str, keep_default_na=False))
    assert exact.recorded == from_file.recorded, "the dtype=str reading agrees too"


def test_a_missing_audit_column_yields_no_records(readers) -> None:
    cfg, _, _, _ = readers
    stripped = pd.read_csv(cfg.paths.csv).drop(columns=[AUDIT_COLUMN])
    assert AuditHoldout.from_labels(cfg, stripped).recorded == {}
    assert AuditHoldout.from_labels(cfg, None).recorded == {}


def test_a_clip_without_a_record_falls_back_to_the_hash(readers) -> None:
    cfg, metas, from_file, _ = readers
    assert (("MOVIE__t9999__f999999" in from_file)
            == cfg.suggest.audit_by_hash("MOVIE__t9999__f999999"))
    assert all((m["clip_id"] in AuditHoldout.by_hash(cfg))
               == cfg.suggest.audit_by_hash(m["clip_id"]) for m in metas)
    assert all((m["clip_id"] in from_file) == from_file.recorded[m["clip_id"]]
               for m in metas), "a recorded clip is answered from the record"


# --------------------------------------------------------------------------- #
# one resolved holdout
# --------------------------------------------------------------------------- #
class _HashConsulted(Exception):
    """Raised by the trap below when the hash is asked about a recorded clip."""


def test_a_re_detected_project_holds_out_exactly_what_its_csv_records(
    root: Path
) -> None:
    """After every clip_id changes, both readings hold out the recorded clips."""
    cfg, metas = _upgraded(root, "resolved-redetect")
    path = cfg.paths.csv

    old_ids = [m["clip_id"] for m in metas]
    frozen = _held_out(AuditHoldout.from_csv(cfg, path), old_ids)

    n_matched, moved_to = _redetect(path, _renumbered_manifest(metas))
    new_ids = [moved_to[c] for c in old_ids]
    assert n_matched == len(metas) and not set(old_ids) & set(new_ids), n_matched

    from_file = AuditHoldout.from_csv(cfg, path)
    from_frame = AuditHoldout.from_labels(cfg, pd.read_csv(path))
    expected = {moved_to[c] for c in frozen}

    assert _held_out(from_file, new_ids) == expected
    assert _held_out(from_frame, new_ids) == expected
    assert 0 < len(expected) < len(new_ids)
    assert from_file.count(new_ids) == from_frame.count(new_ids) == len(expected)

    by_hash = _held_out(AuditHoldout.by_hash(cfg), new_ids)
    assert by_hash != expected, "the renumbering was a no-op; widen RENUMBER_BY"
    assert from_file.n_moved() == len(by_hash ^ expected)
    assert from_file.n_recorded == len(metas)


def test_no_path_can_reach_the_hash_for_a_clip_that_has_a_record(root: Path) -> None:
    """With the hash trapped for recorded clips, every membership path still answers."""
    from trackanno.learn.suggest import Suggester

    cfg, metas = _upgraded(root, "no-hash", 40)
    path = cfg.paths.csv
    ids = [m["clip_id"] for m in metas]

    recorded = read_audit_records(path)
    assert len(recorded) == len(metas)

    honest = cfg.suggest.audit_by_hash

    def trap(clip_id: str) -> bool:
        if str(clip_id) in recorded:
            raise _HashConsulted(f"the hash was asked about {clip_id}, which is recorded")
        return honest(clip_id)

    cfg.suggest.audit_by_hash = trap
    with pytest.raises(_HashConsulted):
        AuditHoldout.by_hash(cfg).count(ids[:1])

    from_file = AuditHoldout.from_csv(cfg, path)
    from_frame = AuditHoldout.from_labels(cfg, pd.read_csv(path))
    assert all((c in from_file) == recorded[c] for c in ids)
    assert all((c in from_frame) == recorded[c] for c in ids)

    suggester = Suggester(cfg)
    assert all(suggester.is_audit_clip(c) == recorded[c] for c in ids), (
        "the Suggester answers from the record, model or no model")

    labels = pd.read_csv(path)
    features = pd.DataFrame({"clip_id": ids, "movie_id": "MOVIE",
                             "speed_mean": range(len(ids))})
    _, report = train_group(features, labels, "behaviour", cfg,
                            columns=["speed_mean"], verbose=False)
    assert report.audit_labelled == sum(recorded.values())
    assert report.audit_population == sum(recorded.values())

    fresh = _clip_id(9999, 123456)
    assert (fresh in from_file) == honest(fresh), "an unrecorded clip uses the hash"
