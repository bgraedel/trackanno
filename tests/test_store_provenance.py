"""Why each clip was put in front of a human, and what must not break to say it.

The annotation CSV is about to stop being a random sample. Until now clips were
served in manifest order, so the labelled set was an unbiased draw and every
statistic over it - class balance, cross-validated F1, "the model is 89%
accurate" - was a statement about the data. An active-learning selector makes it
a statement about a sample the model assembled, and nothing in the file says so.

This suite fixes the two things that make that recoverable: the ``selected_by``
column exists and survives every path the log takes (append, torn append,
rewrite, an older build, an older file), and a row that does not know why it was
shown is never mistaken for a row that was genuinely drawn at random.
"""

from __future__ import annotations

import csv
import hashlib
import warnings
from pathlib import Path

from harness import check
from trackanno.config import RESERVED_GROUP_NAMES, ConfigError, load_config
from trackanno.annotate.store import (
    AUDIT_COLUMN, IDENTITY_COLUMNS, PROVENANCE_COLUMN, TRAILING_COLUMNS,
    AnnotationStore, decode_provenance, encode_provenance,
)

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

# The character the CSV format finds awkward (comma, quote) *and* the one that
# would break the torn-append repair (a literal newline), in one string.
AWKWARD = 'comma, "quoted" \\ backslash\nnewline\ttab'


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _cfg(root: Path, name: str, text: str = CONFIG_TEXT):
    project = Path(root) / name
    project.mkdir(parents=True, exist_ok=True)
    path = project / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return load_config(path)


def _meta(i: int) -> dict:
    return {
        "clip_id": f"MOVIE__t{i:04d}__f000000", "movie_id": "MOVIE",
        "source_file": "/movies/MOVIE.tif", "series": 0, "channel": 0,
        "track_id": i, "f_start": i * 50, "f_end": i * 50 + 49,
        "crop_x0": 10 + i, "crop_y0": 20 + i, "crop_w": 96, "crop_h": 96,
        "scale": 1.0,
    }


def _schema_before_this_change(cfg) -> list[str]:
    """Exactly the header an installed release wrote before either new column.

    Both are excluded, not only ``selected_by``: a build that predates the
    provenance column predates everything added after it too, and a fixture
    that carried ``audit_holdout`` while omitting ``selected_by`` would be a
    file no release ever wrote.
    """
    columns = IDENTITY_COLUMNS + [g.name for g in cfg.categories] + TRAILING_COLUMNS
    return [c for c in columns if c not in (PROVENANCE_COLUMN, AUDIT_COLUMN)]


def _write_old_csv(path: Path, cfg, rows: list[dict]) -> None:
    """A CSV written by that release: same dialect, one column short."""
    columns = _schema_before_this_change(cfg)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def _old_row(i: int, behaviour: str = "swimming", **extra) -> dict:
    row = dict(_meta(i))
    row.update({
        "behaviour": behaviour, "uncertain": "0", "notes": "", "annotator": "benny",
        "timestamp_utc": "2025-11-04T09:12:33+00:00", "dwell_s": "12.75",
        "suggested": "", "suggest_conf": "", "suggest_shown": "0",
    })
    row.update(extra)
    return row


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _header(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return next(csv.reader(fh))


def _load_quietly(path, cfg) -> tuple[AnnotationStore, list[str]]:
    """A store plus the warnings it raised, so a claim can be made about them."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        store = AnnotationStore(path, cfg)
    return store, [str(w.message) for w in caught]


def _refusal(fn, *args, **kwargs) -> str:
    """The message a call refuses with, or "" if it did not refuse at all."""
    try:
        fn(*args, **kwargs)
    except (ValueError, TypeError) as exc:
        return str(exc)
    return ""


# --------------------------------------------------------------------------- #
# the two facts that must not merge
# --------------------------------------------------------------------------- #
def test_not_recorded_is_not_the_same_fact_as_chosen_at_random() -> None:
    """The whole point of the column, stated as a property of the codec.

    If "" ever decoded to something a caller could confuse with a recorded
    decision, upgrading the tool would silently relabel every historical row as
    an unbiased draw - a claim nobody made and which would then be inherited by
    every accuracy figure computed afterwards.
    """
    check("an empty cell decodes to nothing at all", decode_provenance("") == {})
    check("...and so does a column that was never there", decode_provenance(None) == {})
    check("...and whitespace, which a spreadsheet round-trip can introduce",
          decode_provenance("   ") == {})

    at_random = encode_provenance({"rule": "random"})
    check("a clip drawn at random says so explicitly",
          decode_provenance(at_random) == {"rule": "random"}, at_random)
    check("the two are distinguishable by truthiness alone",
          not decode_provenance("") and bool(decode_provenance(at_random)))
    check("...and by the one key that matters",
          decode_provenance("").get("rule") is None
          and decode_provenance(at_random).get("rule") == "random")

    # "not recorded" must not be spellable as a rule, or the empty cell stops
    # being the single unambiguous way to say it.
    check("a nameless rule is refused", "rule" in _refusal(encode_provenance, {"score": 0.4}))
    check("...and so is a blank one", "rule" in _refusal(encode_provenance, {"rule": "  "}))
    check("...with an actionable message",
          "not recorded" in _refusal(encode_provenance, {"rule": ""}),
          _refusal(encode_provenance, {"rule": ""}))

    # A cell that will not parse is a THIRD fact. Reporting it as "not recorded"
    # would hide a damaged log behind a plausible answer.
    check("a corrupt cell raises rather than reporting 'not recorded'",
          "not readable" in _refusal(decode_provenance, "{oops"))
    check("...as does a value that is JSON but not a decision",
          "provenance is an object" in _refusal(decode_provenance, '["uncertainty"]'))


def test_a_decision_carries_the_rule_and_the_score_that_earned_it() -> None:
    """A rule name alone cannot be audited: "uncertainty picked it" is not a
    reason until the number it beat the others with is there too. The encoding
    is open, so a rule that needs to record more does not force a migration."""
    encoded = encode_provenance({"rule": "uncertainty", "score": 0.4213})
    check("the rule survives", decode_provenance(encoded)["rule"] == "uncertainty")
    check("the score survives exactly, unrounded",
          decode_provenance(encoded)["score"] == 0.4213, encoded)

    # A rule this module has never heard of, carrying fields it has never heard
    # of. Nothing here knows what "starved-class" is, and it must not need to.
    future = {"rule": "starved-class", "score": 0.08, "klass": "tumbling",
              "model": "catboost-2026-08-01", "pool": 4120, "shown": True,
              "runners_up": [0.07, 0.06]}
    check("an unknown rule with unknown fields round-trips whole",
          decode_provenance(encode_provenance(future)) == future)

    check("the encoding is byte-stable, so re-recording is a no-op in a diff",
          encode_provenance(future) == encode_provenance(dict(reversed(list(future.items())))))
    check("encoding an already-encoded decision is idempotent",
          encode_provenance(encode_provenance(future)) == encode_provenance(future))

    check("a non-mapping is refused with the shape it wanted",
          "'rule'" in _refusal(encode_provenance, ["uncertainty", 0.4]))


# --------------------------------------------------------------------------- #
# one codec, every column that puts JSON in a cell
# --------------------------------------------------------------------------- #
def test_a_value_that_is_no_number_is_written_as_null_and_never_as_zero() -> None:
    """The non-finite rule, stated once because it is implemented once.

    `select` scores a uniform audit draw NaN deliberately: that clip earned no
    score, and 0.0 would read as "the model was certain it was wrong". NaN is
    not JSON - `json.dumps` spells it as a bare token a strict reader rejects -
    so the cell says `null`, which reads back as None and is still not zero.
    """
    scored = decode_provenance(encode_provenance({"rule": "margin", "score": 0.0}))
    unscored = decode_provenance(
        encode_provenance({"rule": "audit", "score": float("nan")})
    )
    check("a score nothing measured round-trips as no value at all",
          unscored["score"] is None, str(unscored))
    check("...which is not the score of a clip the model was certain about",
          unscored["score"] != scored["score"] and scored["score"] == 0.0,
          f"{unscored['score']!r} vs {scored['score']!r}")
    check("the cell holds JSON's own spelling of it, not a bare NaN token",
          "null" in encode_provenance({"rule": "audit", "score": float("nan")})
          and "NaN" not in encode_provenance({"rule": "audit", "score": float("nan")}))
    check("an infinite score goes the same way, being no more JSON than NaN",
          decode_provenance(
              encode_provenance({"rule": "margin", "score": float("inf")})
          )["score"] is None)

    # The same rule, reached through the other column that uses the codec: a
    # bout nothing could score has no confidence, and a correction over it is
    # not evidence that the model was confidently wrong.
    from trackanno.annotate.corrections import decode_overruled, encode_overruled

    bouts = decode_overruled(encode_overruled([
        {"f_start": 0, "f_end": 40, "label": "swimming", "conf": 0.9},
        {"f_start": 40, "f_end": 80, "label": "", "conf": float("nan")},
    ]))["bouts"]
    check("the overruled column answers non-finite the same way, being one codec",
          bouts[1]["conf"] is None and bouts[0]["conf"] == 0.9, str(bouts))
    check("...and refuses a quoted bout that will not say which frames it covers",
          "both ends" in _refusal(encode_overruled, [{"label": "swimming"}]))


def test_a_numpy_scalar_survives_the_cell_rather_than_aborting_the_save() -> None:
    """A score straight out of `predict_proba` is a numpy.float32, which `json`
    refuses outright - and the save it would abort is a label a person has just
    typed. Widened, not rounded: `.item()` gives the float32's exact value."""
    import numpy as np

    encoded = encode_provenance({
        "rule": "margin", "score": np.float32(0.25), "rank": np.int64(3),
        "shown": np.bool_(True), "runners_up": np.array([0.5, np.nan]),
    })
    got = decode_provenance(encoded)
    check("a float32 score is written, exactly", got["score"] == 0.25, encoded)
    check("...an int64 rank as a number and not as its repr", got["rank"] == 3)
    check("...a numpy bool as a boolean, so a reader need not parse 'True'",
          got["shown"] is True, str(got["shown"]))
    check("...and an array widens elementwise, the non-finite rule with it",
          got["runners_up"] == [0.5, None], str(got["runners_up"]))
    check("nothing in the cell is a numpy repr", "np." not in encoded, encoded)


# --------------------------------------------------------------------------- #
# schema evolution
# --------------------------------------------------------------------------- #
def test_a_csv_from_before_the_column_still_opens_and_admits_it_does_not_know(root: Path) -> None:
    """Upgrading the tool must not invalidate work somebody already did, and
    must not invent a provenance for it either. Those two together are the whole
    migration: the rows load, and they load as *unknown*, not as *random*."""
    cfg = _cfg(root, "old-file")
    path = cfg.paths.csv
    _write_old_csv(path, cfg, [
        _old_row(1, "swimming", notes="out of focus, maybe two cells"),
        _old_row(2, "tumbling", uncertain="1"),
    ])
    old_header = _header(path)
    check("the fixture really is the old schema",
          PROVENANCE_COLUMN not in old_header, str(old_header))

    store, said = _load_quietly(path, cfg)
    check("every historical row still loads", len(store) == 2, str(len(store)))
    check("...with its human values untouched",
          store.get(_meta(1)["clip_id"])["notes"] == "out of focus, maybe two cells")
    check("...and its geometry untouched",
          int(store.get(_meta(2)["clip_id"])["crop_x0"]) == _meta(2)["crop_x0"])

    for i in (1, 2):
        row = store.get(_meta(i)["clip_id"])
        check(f"row {i} reports provenance unknown",
              decode_provenance(row[PROVENANCE_COLUMN]) == {},
              repr(row[PROVENANCE_COLUMN]))
        check(f"row {i} does NOT report itself as chosen at random",
              decode_provenance(row[PROVENANCE_COLUMN]).get("rule") != "random")

    check("the file itself was upgraded in place, once",
          PROVENANCE_COLUMN in _header(path), str(_header(path)))
    # By column position rather than by the line's trailing bytes: the
    # provenance cell stopped being the last field on the line when a later
    # column was appended after it, and the claim was never about where it sits.
    with open(path, "r", encoding="utf-8", newline="") as fh:
        raw = list(csv.reader(fh))
    at = raw[0].index(PROVENANCE_COLUMN)
    check("the upgrade left the cell empty rather than guessing",
          all(row[at] == "" for row in raw[1:]), str(raw[1]))
    check("and it said so out loud, in the one moment it mattered",
          any(PROVENANCE_COLUMN in m and "NOT RECORDED" in m for m in said), str(said))

    # Re-opening must be quiet: a migration that warns for ever trains people to
    # ignore the warning that matters.
    again, said_again = _load_quietly(path, cfg)
    check("re-opening the upgraded file is silent", said_again == [], str(said_again))
    check("...and still reports unknown, not random",
          decode_provenance(again.get(_meta(1)["clip_id"])[PROVENANCE_COLUMN]) == {})


def test_an_older_build_reading_a_new_csv_leaves_it_alone(root: Path) -> None:
    """The downgrade direction, which cannot be tested by running old code, so
    it is tested by running old code's *rule*: `_load` treats an unrecognised
    column as `extra` and appends it to the end of its own column list.

    That reconstruction matches the header on disk only because the columns
    added since are LAST and in the order they were added, and if it matches,
    the old build does not rewrite the file and does not drop them - it carries
    the values through untouched. The worst it can do is append its own rows
    with the cells blank, which decodes to "not recorded" and is the truth: that
    build did not know why it showed them.
    """
    cfg = _cfg(root, "downgrade")
    store = AnnotationStore(cfg.paths.csv, cfg)
    store.record(_meta(1), {"behaviour": "swimming"},
                 selected_by={"rule": "uncertainty", "score": 0.41})

    disk = _header(cfg.paths.csv)
    old_columns = _schema_before_this_change(cfg)
    extra = [c for c in disk if c not in old_columns]
    check("an older build sees exactly the columns added since, in that order",
          extra == [PROVENANCE_COLUMN, AUDIT_COLUMN], str(extra))
    check("...and its union reproduces the header exactly, so it does not rewrite",
          old_columns + extra == disk, f"{old_columns + extra} vs {disk}")
    check("...so the values it cannot interpret are still preserved",
          decode_provenance(
              AnnotationStore(cfg.paths.csv, cfg).get(_meta(1)["clip_id"])[PROVENANCE_COLUMN]
          )["rule"] == "uncertainty")


def test_the_reserved_names_cannot_drift_from_the_schema(root: Path) -> None:
    """``config.RESERVED_GROUP_NAMES`` restates the store's column lists by hand
    and nothing in the language holds the two together. A column added to the
    store and forgotten there is not an error - it is a name a user is then
    allowed to give a category, after which ``record`` writes the built-in and
    the category into one key and the built-in is gone with no exception
    anywhere. This claim is the only thing that catches the omission."""
    check("the reserved names are exactly the built-in columns",
          set(RESERVED_GROUP_NAMES) == set(IDENTITY_COLUMNS + TRAILING_COLUMNS),
          str(set(RESERVED_GROUP_NAMES) ^ set(IDENTITY_COLUMNS + TRAILING_COLUMNS)))
    check("the provenance column is among them", PROVENANCE_COLUMN in RESERVED_GROUP_NAMES)
    check("the built-in lists do not overlap each other",
          not set(IDENTITY_COLUMNS) & set(TRAILING_COLUMNS))

    collides = CONFIG_TEXT.replace("name: behaviour", f"name: {PROVENANCE_COLUMN}")
    check("a category named after a built-in column is refused at config load",
          PROVENANCE_COLUMN in _refusal(_cfg, root, "collide", collides),
          _refusal(_cfg, root, "collide", collides) or "it was ACCEPTED")
    check("...and the refusal is a ConfigError, so the GUI reports it as advice",
          isinstance(_raised(_cfg, root, "collide2", collides), ConfigError))

    # A Config assembled without going through load_config skips that guard, so
    # the store refuses too rather than corrupt the log it owns.
    cfg = _cfg(root, "handmade")
    cfg.categories[0].name = PROVENANCE_COLUMN
    check("the store refuses it too, before touching the file",
          PROVENANCE_COLUMN in _refusal(AnnotationStore, cfg.paths.csv, cfg),
          _refusal(AnnotationStore, cfg.paths.csv, cfg) or "it was ACCEPTED")
    check("...so the CSV was never created", not cfg.paths.csv.exists())

    ok = AnnotationStore(_cfg(root, "clean").paths.csv, _cfg(root, "clean"))
    check("a legitimate config yields no duplicate columns",
          len(ok.columns) == len(set(ok.columns)), str(ok.columns))


def _raised(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except Exception as exc:      # noqa: BLE001 - the type IS the claim
        return exc
    return None


# --------------------------------------------------------------------------- #
# the paths the value has to survive
# --------------------------------------------------------------------------- #
def test_a_recorded_decision_round_trips_through_load_and_rewrite(root: Path) -> None:
    """Written, read back, rewritten. ``rewrite()`` runs on every edit, undo and
    clear, so a value it mutates would decay a little on each keystroke - the
    kind of loss nobody notices until the numbers stop adding up."""
    cfg = _cfg(root, "roundtrip")
    path = cfg.paths.csv
    store = AnnotationStore(path, cfg)
    decisions = [
        {"rule": "uncertainty", "score": 0.4213},
        {"rule": "random"},
        {"rule": "starved-class", "score": 0.08, "klass": "tumbling"},
    ]
    for i, decision in enumerate(decisions, start=1):
        store.record(_meta(i), {"behaviour": "swimming"}, selected_by=decision)

    reloaded = AnnotationStore(path, cfg)
    for i, decision in enumerate(decisions, start=1):
        check(f"decision {i} survives the append and the reload",
              decode_provenance(reloaded.get(_meta(i)["clip_id"])[PROVENANCE_COLUMN]) == decision)

    before = _digest(path)
    reloaded.rewrite()
    check("a rewrite of unchanged rows is byte-for-byte identical",
          _digest(path) == before)

    third = AnnotationStore(path, cfg)
    for i, decision in enumerate(decisions, start=1):
        check(f"decision {i} survives the rewrite too",
              decode_provenance(third.get(_meta(i)["clip_id"])[PROVENANCE_COLUMN]) == decision)

    # An edit rewrites the whole file; the untouched rows must come through it.
    third.record(_meta(1), {"behaviour": "tumbling"}, selected_by=decisions[0])
    after_edit = AnnotationStore(path, cfg)
    check("editing one row does not disturb another's provenance",
          decode_provenance(after_edit.get(_meta(3)["clip_id"])[PROVENANCE_COLUMN])
          == decisions[2])
    check("...and undo puts the edited row back with its provenance",
          (third.undo() is not None)
          and decode_provenance(
              AnnotationStore(path, cfg).get(_meta(1)["clip_id"])[PROVENANCE_COLUMN]
          ) == decisions[0])

    # Saying nothing is not saying "random".
    store2 = AnnotationStore(_cfg(root, "silent").paths.csv, _cfg(root, "silent"))
    row = store2.record(_meta(9), {"behaviour": "swimming"})
    check("a caller that does not say why leaves the cell empty",
          row[PROVENANCE_COLUMN] == "", repr(row[PROVENANCE_COLUMN]))
    check("...which reads back as unknown, never as random",
          decode_provenance(row[PROVENANCE_COLUMN]) == {})


def test_an_awkward_value_survives_the_csv_and_the_torn_append_repair(root: Path) -> None:
    """Two escaping regimes stacked on one cell.

    The csv module quotes commas and doubles quotes, so those were never in
    danger. The newline is: ``_repair_torn_append`` finds the end of the last row
    by searching for the last newline BYTE and never by parsing, so a literal
    newline inside a field would let it cut an intact row in half. JSON escapes
    it, which is why this format was chosen over a bare ``rule|score`` pair.
    """
    cfg = _cfg(root, "awkward")
    path = cfg.paths.csv
    store = AnnotationStore(path, cfg)
    decision = {"rule": AWKWARD, "score": 0.5, AWKWARD: "value, with a comma"}
    store.record(_meta(1), {"behaviour": "swimming"},
                 notes='a note, with a comma and a "quote"', selected_by=decision)

    cell = store.get(_meta(1)["clip_id"])[PROVENANCE_COLUMN]
    check("the encoded cell contains no literal newline", "\n" not in cell and "\r" not in cell)
    check("...and no literal tab either", "\t" not in cell)
    check("the file is still exactly two physical lines",
          len(path.read_bytes().rstrip(b"\r\n").split(b"\n")) == 2,
          repr(path.read_bytes()))

    reloaded = AnnotationStore(path, cfg)
    check("the comma did not split the row into two", len(reloaded) == 1)
    row = reloaded.get(_meta(1)["clip_id"])
    check("the awkward decision decodes back identical",
          decode_provenance(row[PROVENANCE_COLUMN]) == decision,
          repr(row[PROVENANCE_COLUMN]))
    check("the newline inside the rule name came back as a newline",
          "\n" in decode_provenance(row[PROVENANCE_COLUMN])["rule"])
    check("the neighbouring free-text column survived its own comma",
          row["notes"] == 'a note, with a comma and a "quote"')

    before = _digest(path)
    reloaded.rewrite()
    check("and it rewrites byte-for-byte identically", _digest(path) == before)


def test_what_the_selector_actually_emits_goes_in_whole(root: Path) -> None:
    """The column is only worth having if the selector's own record fits in it.

    ``select.select_batch`` returns plain dicts; this checks one of them survives
    the trip unaltered, including the keys this module has never heard of. The
    encoding hard-codes none of that vocabulary on purpose - the rule set is
    expected to grow, and a new rule that records one more thing must not need a
    schema migration, which would rewrite every user's log.
    """
    import numpy as np

    from trackanno.learn.select import AUDIT_RULE, select_batch

    cfg = _cfg(root, "selector")
    store = AnnotationStore(cfg.paths.csv, cfg)
    batch = select_batch(
        np.array([[0.34, 0.33, 0.33], [0.90, 0.05, 0.05],
                  [0.50, 0.45, 0.05], [0.70, 0.20, 0.10]]),
        [f"MOVIE__t{i:04d}__f000000" for i in range(1, 5)],
        ["swimming", "tumbling", "unclear"],
        2,
    )
    check("the selector produced a batch to record", len(batch) == 2, str(len(batch)))

    for i, record in enumerate(batch, start=1):
        store.record(_meta(i), {"behaviour": "swimming"}, selected_by=record)
    reloaded = AnnotationStore(cfg.paths.csv, cfg)
    for i, record in enumerate(batch, start=1):
        got = decode_provenance(reloaded.get(_meta(i)["clip_id"])[PROVENANCE_COLUMN])
        check(f"selection {i} round-trips key for key", got == record,
              f"{got} vs {record}")
    check("the rule that chose it is legible without decoding the rest",
          all(decode_provenance(
              reloaded.get(_meta(i)["clip_id"])[PROVENANCE_COLUMN])["rule"]
              for i in (1, 2)))

    # The selector's own marker for a clip drawn by a coin. It must stay
    # distinguishable from a row that simply does not know, or the audit stratum
    # - the only unbiased accuracy estimate in the tool - cannot be recovered.
    at_random = encode_provenance({"rule": AUDIT_RULE, "score": 0.0, "rank": 0})
    check("a reserved random draw is recorded as such",
          decode_provenance(at_random)["rule"] == AUDIT_RULE)
    check("...and is not the same fact as an unrecorded row",
          decode_provenance(at_random) != decode_provenance(""))


def test_the_other_readers_of_this_file_still_read_it(root: Path) -> None:
    """The store is not the only thing that opens this CSV.

    ``suggest``, ``predict``, ``render`` and the hub all read it with pandas, and
    ``reassociate.write_labels`` is the only other thing that WRITES it - it
    rewrites the whole log after a clip cache rebuild. Every one of them
    addresses columns by name, so an extra column is harmless; what is not
    obviously harmless is the quoting, because the provenance cell is the first
    value in this file that contains a comma and a doubled quote on purpose. If
    pandas and the csv module disagreed about it, a repair would silently rewrite
    every row's provenance into something else.
    """
    import pandas as pd

    from trackanno.annotate.reassociate import write_labels

    cfg = _cfg(root, "readers")
    path = cfg.paths.csv
    store = AnnotationStore(path, cfg)
    decision = {"rule": AWKWARD, "score": 0.5}
    store.record(_meta(1), {"behaviour": "swimming"}, selected_by=decision)
    store.record(_meta(2), {"behaviour": "tumbling"}, selected_by={"rule": "random"})

    frame = pd.read_csv(path)
    check("pandas sees the same number of rows the store does", len(frame) == 2, str(len(frame)))
    check("...and the same columns, in order",
          list(frame.columns) == store.columns, str(list(frame.columns)))
    check("pandas parses the awkward provenance cell identically",
          decode_provenance(frame[PROVENANCE_COLUMN].iat[0]) == decision,
          repr(frame[PROVENANCE_COLUMN].iat[0]))

    check("the hub's clip_id-only read is unaffected",
          list(pd.read_csv(path, usecols=["clip_id"])["clip_id"].astype(str))
          == [_meta(1)["clip_id"], _meta(2)["clip_id"]])

    # The repair path: read the way the hub does, write the way reassociate does,
    # then load it back with the store.
    exact = pd.read_csv(path, dtype=str, keep_default_na=False)
    write_labels(path, exact)
    repaired = AnnotationStore(path, cfg)
    check("a reassociation rewrite preserves the awkward provenance exactly",
          decode_provenance(repaired.get(_meta(1)["clip_id"])[PROVENANCE_COLUMN]) == decision,
          repr(repaired.get(_meta(1)["clip_id"])[PROVENANCE_COLUMN]))
    check("...and the plain one",
          decode_provenance(repaired.get(_meta(2)["clip_id"])[PROVENANCE_COLUMN])
          == {"rule": "random"})
    check("...and did not change the column set, so the store did not rewrite it",
          _header(path) == store.columns, str(_header(path)))


def test_a_torn_append_costs_only_the_torn_row(root: Path) -> None:
    """A kill mid-append leaves half a line. The repair is byte-level - it never
    parses - so it does not care how many columns a row has; what it cares about
    is that a complete row ends in a newline and no field contains one. Both
    schemas are checked, because a torn OLD file is what an upgrade meets."""
    cfg = _cfg(root, "torn-new")
    path = cfg.paths.csv
    store = AnnotationStore(path, cfg)
    store.record(_meta(1), {"behaviour": "swimming"},
                 selected_by={"rule": "uncertainty", "score": 0.9})
    store.record(_meta(2), {"behaviour": "tumbling"}, selected_by={"rule": "random"})
    intact = _digest(path)

    with open(path, "a", encoding="utf-8", newline="") as fh:
        fh.write('MOVIE__t0003__f000000,MOVIE,/movies/MOVIE.tif,0,0,3,150,"{""rul')

    repaired, said = _load_quietly(path, cfg)
    check("the half-written row is dropped", len(repaired) == 2, str(len(repaired)))
    check("...loudly", any("partially written" in m for m in said), str(said))
    check("...and the file is back to exactly what it was", _digest(path) == intact)
    check("the surviving provenance is intact and not truncated",
          decode_provenance(repaired.get(_meta(1)["clip_id"])[PROVENANCE_COLUMN])
          == {"rule": "uncertainty", "score": 0.9})
    check("...for the second row too",
          decode_provenance(repaired.get(_meta(2)["clip_id"])[PROVENANCE_COLUMN])
          == {"rule": "random"})

    # The same tear, in a file written before the column existed. The repair is
    # schema-blind, so the row width it was torn at is irrelevant; what matters
    # is that the intact rows come through as UNKNOWN and not as random.
    cfg_old = _cfg(root, "torn-old")
    old_path = cfg_old.paths.csv
    _write_old_csv(old_path, cfg_old, [_old_row(1), _old_row(2)])
    with open(old_path, "a", encoding="utf-8", newline="") as fh:
        fh.write("MOVIE__t0003__f000000,MOVIE,/movies/MOV")

    upgraded, said_old = _load_quietly(old_path, cfg_old)
    check("a row torn under the old schema is dropped by the new repair",
          len(upgraded) == 2, str(len(upgraded)))
    check("...and the intact old rows still load", upgraded.is_done(_meta(2)["clip_id"]))
    check("...reporting unknown provenance, not random",
          all(decode_provenance(upgraded.get(_meta(i)["clip_id"])[PROVENANCE_COLUMN]) == {}
              for i in (1, 2)))
    check("...and the upgrade still announced itself",
          any("NOT RECORDED" in m for m in said_old), str(said_old))
    check("the repaired file carries the new column",
          PROVENANCE_COLUMN in _header(old_path))

    # Appending after a repair must produce a readable file, which is the failure
    # `_repair_torn_append` exists for: without it the next row is concatenated
    # onto the stump and both are lost.
    upgraded.record(_meta(4), {"behaviour": "swimming"}, selected_by={"rule": "random"})
    check("annotating after a repair still works",
          len(AnnotationStore(old_path, cfg_old)) == 3)
    check("...and the new row's provenance is readable",
          decode_provenance(
              AnnotationStore(old_path, cfg_old).get(_meta(4)["clip_id"])[PROVENANCE_COLUMN]
          ) == {"rule": "random"})
