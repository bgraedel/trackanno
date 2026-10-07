"""The ``selected_by`` column: its codec, schema evolution, and the paths it must survive."""

from __future__ import annotations

import csv
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trackanno.config import RESERVED_GROUP_NAMES, ConfigError, load_config
from trackanno.annotate.store import (
    AUDIT_COLUMN, IDENTITY_COLUMNS, PROVENANCE_COLUMN, TRAILING_COLUMNS,
    AnnotationStore, decode_provenance, encode_provenance,
)
from fixtures import AWKWARD, STORE_CONFIG, clip_meta, old_row, old_schema, write_old_csv


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _cfg(root: Path, name: str, text: str = STORE_CONFIG):
    project = Path(root) / name
    project.mkdir(parents=True, exist_ok=True)
    path = project / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return load_config(path)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _header(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return next(csv.reader(fh))


def _load_quietly(path, cfg) -> tuple[AnnotationStore, list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        store = AnnotationStore(path, cfg)
    return store, [str(w.message) for w in caught]


def _provenance(store: AnnotationStore, i: int) -> dict:
    return decode_provenance(store.get(clip_meta(i)["clip_id"])[PROVENANCE_COLUMN])


# --------------------------------------------------------------------------- #
# the codec
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("blank", ["", None, "   "])
def test_a_blank_cell_decodes_to_not_recorded(blank) -> None:
    assert decode_provenance(blank) == {}


def test_not_recorded_is_not_the_same_fact_as_chosen_at_random() -> None:
    at_random = encode_provenance({"rule": "random"})
    assert decode_provenance(at_random) == {"rule": "random"}, at_random
    assert not decode_provenance("") and bool(decode_provenance(at_random))
    assert decode_provenance("").get("rule") is None


@pytest.mark.parametrize("decision, match", [
    ({"score": 0.4}, "rule"),
    ({"rule": "  "}, "rule"),
    ({"rule": ""}, "not recorded"),
    (["margin", 0.4], "'rule'"),
])
def test_a_decision_without_a_rule_is_refused(decision, match) -> None:
    with pytest.raises((ValueError, TypeError), match=match):
        encode_provenance(decision)


@pytest.mark.parametrize("cell, match", [
    ("{oops", "not readable"),
    ('["margin"]', "provenance is an object"),
])
def test_a_corrupt_cell_raises_rather_than_reading_as_not_recorded(cell, match) -> None:
    with pytest.raises((ValueError, TypeError), match=match):
        decode_provenance(cell)


def test_a_decision_carries_the_rule_and_the_score_that_earned_it() -> None:
    encoded = encode_provenance({"rule": "margin", "score": 0.4213})
    assert decode_provenance(encoded) == {"rule": "margin", "score": 0.4213}, encoded


def test_an_unknown_rule_with_unknown_fields_round_trips_whole() -> None:
    future = {"rule": "starved-class", "score": 0.08, "klass": "tumbling",
              "model": "catboost-2026-08-01", "pool": 4120, "shown": True,
              "runners_up": [0.07, 0.06]}
    assert decode_provenance(encode_provenance(future)) == future
    assert encode_provenance(future) == encode_provenance(
        dict(reversed(list(future.items())))), "the encoding is key-order stable"
    assert encode_provenance(encode_provenance(future)) == encode_provenance(future), (
        "encoding an encoded decision is idempotent")


def test_a_value_that_is_no_number_is_written_as_null_and_never_as_zero() -> None:
    scored = decode_provenance(encode_provenance({"rule": "margin", "score": 0.0}))
    unscored = decode_provenance(encode_provenance({"rule": "audit", "score": float("nan")}))
    assert unscored["score"] is None, unscored
    assert scored["score"] == 0.0
    cell = encode_provenance({"rule": "audit", "score": float("nan")})
    assert "null" in cell and "NaN" not in cell, cell
    inf = encode_provenance({"rule": "margin", "score": float("inf")})
    assert decode_provenance(inf)["score"] is None


def test_a_numpy_scalar_survives_the_cell_rather_than_aborting_the_save() -> None:
    encoded = encode_provenance({
        "rule": "margin", "score": np.float32(0.25), "rank": np.int64(3),
        "shown": np.bool_(True), "runners_up": np.array([0.5, np.nan]),
    })
    got = decode_provenance(encoded)
    assert got["score"] == 0.25, encoded
    assert got["rank"] == 3
    assert got["shown"] is True, got["shown"]
    assert got["runners_up"] == [0.5, None], got["runners_up"]
    assert "np." not in encoded, encoded


# --------------------------------------------------------------------------- #
# schema evolution
# --------------------------------------------------------------------------- #
def test_a_csv_from_before_the_column_still_opens_and_admits_it_does_not_know(root: Path) -> None:
    """Old rows load as provenance unknown, never as random."""
    cfg = _cfg(root, "old-file")
    path = cfg.paths.csv
    write_old_csv(path, cfg, [
        old_row(1, "swimming", notes="out of focus, maybe two cells"),
        old_row(2, "tumbling", uncertain="1"),
    ])
    assert PROVENANCE_COLUMN not in _header(path)

    store, said = _load_quietly(path, cfg)
    assert len(store) == 2
    assert store.get(clip_meta(1)["clip_id"])["notes"] == "out of focus, maybe two cells"
    assert int(store.get(clip_meta(2)["clip_id"])["crop_x0"]) == clip_meta(2)["crop_x0"]
    for i in (1, 2):
        assert _provenance(store, i) == {}, f"row {i} reports provenance unknown"

    assert PROVENANCE_COLUMN in _header(path), "the file was upgraded in place"
    with open(path, "r", encoding="utf-8", newline="") as fh:
        raw = list(csv.reader(fh))
    at = raw[0].index(PROVENANCE_COLUMN)
    assert all(row[at] == "" for row in raw[1:]), raw[1]
    assert any(PROVENANCE_COLUMN in m and "NOT RECORDED" in m for m in said), said

    again, said_again = _load_quietly(path, cfg)
    assert said_again == [], "re-opening the upgraded file is silent"
    assert _provenance(again, 1) == {}


def test_an_older_build_reading_a_new_csv_leaves_it_alone(root: Path) -> None:
    """Added columns are last, so an older build's extras reproduce the header exactly."""
    cfg = _cfg(root, "downgrade")
    store = AnnotationStore(cfg.paths.csv, cfg)
    store.record(clip_meta(1), {"behaviour": "swimming"},
                 selected_by={"rule": "margin", "score": 0.41})

    disk = _header(cfg.paths.csv)
    old_columns = old_schema(cfg)
    extra = [c for c in disk if c not in old_columns]
    assert extra == [PROVENANCE_COLUMN, AUDIT_COLUMN], extra
    assert old_columns + extra == disk
    assert _provenance(AnnotationStore(cfg.paths.csv, cfg), 1)["rule"] == "margin"


def test_the_reserved_names_cannot_drift_from_the_schema() -> None:
    builtin = set(IDENTITY_COLUMNS + TRAILING_COLUMNS)
    assert set(RESERVED_GROUP_NAMES) == builtin, set(RESERVED_GROUP_NAMES) ^ builtin
    assert PROVENANCE_COLUMN in RESERVED_GROUP_NAMES
    assert not set(IDENTITY_COLUMNS) & set(TRAILING_COLUMNS)


def test_a_category_named_after_a_built_in_column_is_refused_at_load(root: Path) -> None:
    collides = STORE_CONFIG.replace("name: behaviour", f"name: {PROVENANCE_COLUMN}")
    with pytest.raises(ConfigError, match=PROVENANCE_COLUMN):
        _cfg(root, "collide", collides)


def test_the_store_refuses_a_colliding_category_before_touching_the_file(root: Path) -> None:
    cfg = _cfg(root, "handmade")
    cfg.categories[0].name = PROVENANCE_COLUMN
    with pytest.raises((ValueError, TypeError), match=PROVENANCE_COLUMN):
        AnnotationStore(cfg.paths.csv, cfg)
    assert not cfg.paths.csv.exists()

    clean = _cfg(root, "clean")
    ok = AnnotationStore(clean.paths.csv, clean)
    assert len(ok.columns) == len(set(ok.columns)), ok.columns


# --------------------------------------------------------------------------- #
# the paths the value has to survive
# --------------------------------------------------------------------------- #
DECISIONS = [
    {"rule": "margin", "score": 0.4213},
    {"rule": "random"},
    {"rule": "starved-class", "score": 0.08, "klass": "tumbling"},
]


@pytest.fixture
def recorded(root: Path):
    """A store with one row per decision in ``DECISIONS``."""
    cfg = _cfg(root, "roundtrip")
    store = AnnotationStore(cfg.paths.csv, cfg)
    for i, decision in enumerate(DECISIONS, start=1):
        store.record(clip_meta(i), {"behaviour": "swimming"}, selected_by=decision)
    return cfg, store


def test_a_recorded_decision_round_trips_through_load_and_rewrite(recorded) -> None:
    cfg, _ = recorded
    path = cfg.paths.csv
    reloaded = AnnotationStore(path, cfg)
    for i, decision in enumerate(DECISIONS, start=1):
        assert _provenance(reloaded, i) == decision, f"decision {i} after reload"

    before = _digest(path)
    reloaded.rewrite()
    assert _digest(path) == before, "a rewrite of unchanged rows is byte-identical"
    third = AnnotationStore(path, cfg)
    for i, decision in enumerate(DECISIONS, start=1):
        assert _provenance(third, i) == decision, f"decision {i} after rewrite"


def test_an_edit_and_its_undo_keep_every_provenance(recorded) -> None:
    cfg, store = recorded
    path = cfg.paths.csv
    store.record(clip_meta(1), {"behaviour": "tumbling"}, selected_by=DECISIONS[0])
    assert _provenance(AnnotationStore(path, cfg), 3) == DECISIONS[2], (
        "editing one row leaves another's provenance alone")
    assert store.undo() is not None
    assert _provenance(AnnotationStore(path, cfg), 1) == DECISIONS[0]


def test_a_caller_that_does_not_say_why_leaves_the_cell_empty(root: Path) -> None:
    cfg = _cfg(root, "silent")
    row = AnnotationStore(cfg.paths.csv, cfg).record(clip_meta(9), {"behaviour": "swimming"})
    assert row[PROVENANCE_COLUMN] == "", row[PROVENANCE_COLUMN]
    assert decode_provenance(row[PROVENANCE_COLUMN]) == {}


def test_an_awkward_value_survives_the_csv_and_the_torn_append_repair(root: Path) -> None:
    """The encoded cell holds no literal newline, which the byte-level repair relies on."""
    cfg = _cfg(root, "awkward")
    path = cfg.paths.csv
    store = AnnotationStore(path, cfg)
    decision = {"rule": AWKWARD, "score": 0.5, AWKWARD: "value, with a comma"}
    store.record(clip_meta(1), {"behaviour": "swimming"},
                 notes='a note, with a comma and a "quote"', selected_by=decision)

    cell = store.get(clip_meta(1)["clip_id"])[PROVENANCE_COLUMN]
    assert "\n" not in cell and "\r" not in cell and "\t" not in cell, cell
    assert len(path.read_bytes().rstrip(b"\r\n").split(b"\n")) == 2, path.read_bytes()

    reloaded = AnnotationStore(path, cfg)
    assert len(reloaded) == 1
    row = reloaded.get(clip_meta(1)["clip_id"])
    assert decode_provenance(row[PROVENANCE_COLUMN]) == decision, row[PROVENANCE_COLUMN]
    assert row["notes"] == 'a note, with a comma and a "quote"'

    before = _digest(path)
    reloaded.rewrite()
    assert _digest(path) == before


def test_what_the_selector_actually_emits_goes_in_whole(root: Path) -> None:
    from trackanno.learn.select import select_batch

    cfg = _cfg(root, "selector")
    store = AnnotationStore(cfg.paths.csv, cfg)
    batch = select_batch(
        np.array([[0.34, 0.33, 0.33], [0.90, 0.05, 0.05],
                  [0.50, 0.45, 0.05], [0.70, 0.20, 0.10]]),
        [f"MOVIE__t{i:04d}__f000000" for i in range(1, 5)],
        ["swimming", "tumbling", "unclear"],
        2,
    )
    assert len(batch) == 2, batch

    for i, record in enumerate(batch, start=1):
        store.record(clip_meta(i), {"behaviour": "swimming"}, selected_by=record)
    reloaded = AnnotationStore(cfg.paths.csv, cfg)
    for i, record in enumerate(batch, start=1):
        got = _provenance(reloaded, i)
        assert got == record, f"selection {i}: {got} vs {record}"
        assert got["rule"]


def test_the_other_readers_of_this_file_still_read_it(root: Path) -> None:
    """pandas and ``reassociate.write_labels`` read and rewrite the cell unchanged."""
    from trackanno.annotate.reassociate import write_labels

    cfg = _cfg(root, "readers")
    path = cfg.paths.csv
    store = AnnotationStore(path, cfg)
    decision = {"rule": AWKWARD, "score": 0.5}
    store.record(clip_meta(1), {"behaviour": "swimming"}, selected_by=decision)
    store.record(clip_meta(2), {"behaviour": "tumbling"}, selected_by={"rule": "random"})

    frame = pd.read_csv(path)
    assert len(frame) == 2
    assert list(frame.columns) == store.columns
    assert decode_provenance(frame[PROVENANCE_COLUMN].iat[0]) == decision
    assert (list(pd.read_csv(path, usecols=["clip_id"])["clip_id"].astype(str))
            == [clip_meta(1)["clip_id"], clip_meta(2)["clip_id"]])

    write_labels(path, pd.read_csv(path, dtype=str, keep_default_na=False))
    repaired = AnnotationStore(path, cfg)
    assert _provenance(repaired, 1) == decision
    assert _provenance(repaired, 2) == {"rule": "random"}
    assert _header(path) == store.columns, "the store did not need to rewrite the header"


def test_a_torn_append_costs_only_the_torn_row(root: Path) -> None:
    cfg = _cfg(root, "torn-new")
    path = cfg.paths.csv
    store = AnnotationStore(path, cfg)
    store.record(clip_meta(1), {"behaviour": "swimming"},
                 selected_by={"rule": "margin", "score": 0.9})
    store.record(clip_meta(2), {"behaviour": "tumbling"}, selected_by={"rule": "random"})
    intact = _digest(path)

    with open(path, "a", encoding="utf-8", newline="") as fh:
        fh.write('MOVIE__t0003__f000000,MOVIE,/movies/MOVIE.tif,0,0,3,150,"{""rul')

    repaired, said = _load_quietly(path, cfg)
    assert len(repaired) == 2, "the half-written row is dropped"
    assert any("partially written" in m for m in said), said
    assert _digest(path) == intact
    assert _provenance(repaired, 1) == {"rule": "margin", "score": 0.9}
    assert _provenance(repaired, 2) == {"rule": "random"}


def test_a_torn_old_schema_file_is_repaired_and_upgraded(root: Path) -> None:
    cfg = _cfg(root, "torn-old")
    path = cfg.paths.csv
    write_old_csv(path, cfg, [old_row(1), old_row(2)])
    with open(path, "a", encoding="utf-8", newline="") as fh:
        fh.write("MOVIE__t0003__f000000,MOVIE,/movies/MOV")

    upgraded, said = _load_quietly(path, cfg)
    assert len(upgraded) == 2, "the torn row is dropped"
    assert upgraded.is_done(clip_meta(2)["clip_id"])
    assert all(_provenance(upgraded, i) == {} for i in (1, 2)), "unknown, not random"
    assert any("NOT RECORDED" in m for m in said), said
    assert PROVENANCE_COLUMN in _header(path)

    upgraded.record(clip_meta(4), {"behaviour": "swimming"}, selected_by={"rule": "random"})
    reopened = AnnotationStore(path, cfg)
    assert len(reopened) == 3, "annotating after a repair still works"
    assert _provenance(reopened, 4) == {"rule": "random"}
