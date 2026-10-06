"""The annotation log.

One CSV, one row per annotated clip. Design constraints:

* **Resumable** - the CSV *is* the state; a clip whose ``clip_id`` appears is done.
* **Crash-safe** - each annotation is appended and fsynced at once, so a kill
  loses at most the clip on screen.
* **Reassociatable** - every row carries the source file, track id, frame range
  and crop rectangle in original pixel coordinates.
* **Self-describing** - every row records why that clip was put in front of a
  human, and whether it was held out for audit at the moment it was labelled.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import tempfile
import threading
import warnings
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from ..config import Config

__all__ = [
    "AnnotationStore", "MULTI_SEP", "PROVENANCE_COLUMN", "AUDIT_COLUMN",
    "plain_json", "encode_cell", "decode_cell",
    "encode_provenance", "decode_provenance",
    "encode_audit", "decode_audit", "read_audit_records",
    "IDENTITY_COLUMNS", "TRAILING_COLUMNS",
]

MULTI_SEP = "|"

# Why this clip was shown. See `encode_provenance` for what goes in it.
PROVENANCE_COLUMN = "selected_by"

# Whether this clip was in the audit holdout AT THE MOMENT IT WAS LABELLED.
# See `encode_audit` for why that has to be a recorded fact rather than a
# function of the clip_id.
AUDIT_COLUMN = "audit_holdout"

IDENTITY_COLUMNS = [
    "clip_id", "movie_id", "source_file", "series", "channel",
    "track_id", "f_start", "f_end",
    "crop_x0", "crop_y0", "crop_w", "crop_h", "scale",
]
TRAILING_COLUMNS = [
    "uncertain", "notes", "annotator", "timestamp_utc", "dwell_s",
    # Provenance for the suggestion model: without these, agreement cannot be
    # told from influence, nor drift measured.
    "suggested", "suggest_conf", "suggest_shown",
    # These two are LAST, in the order they were ADDED, and that order is
    # load-bearing: an older build reconstructs its header as its own columns
    # plus any it does not recognise appended in file order (see `_load`), which
    # matches the header on disk only while the newer columns are already last.
    # Insert one anywhere else and every downgrade rewrites the whole log.
    PROVENANCE_COLUMN,
    AUDIT_COLUMN,
]

# config.RESERVED_GROUP_NAMES restates these two lists by hand. A column added
# here but not there becomes a name a user may give a category, after which
# `record` writes the built-in and the category into one key and the built-in is
# gone silently; tests/test_store_provenance.py asserts the two sets are equal.


# --------------------------------------------------------------------------- #
# one JSON object in one CSV cell
# --------------------------------------------------------------------------- #
# One codec for `selected_by`. What a column additionally requires of its own
# payload is layered on top, not built in here.
def plain_json(value):
    """A value in the form ``json`` will take, exactly, and never rounded.

    Two rules, and they have to be the same in every column or a cell written by
    one reads back differently from a cell written by another:

    * **A numpy scalar widens to a Python scalar.** A score straight out of
      ``predict_proba`` is a ``numpy.float32``, which ``json`` refuses, and the
      save it would abort is a label just typed. ``.item()`` widens rather than
      rounds, so nothing is guessed, and an array falls back to ``.tolist()``.
    * **A non-finite float becomes ``None``, written as ``null``.** NaN and
      Infinity are not JSON - ``json.dumps`` spells them as bare tokens a strict
      reader rejects - and ``null`` is the JSON spelling of "no value", which is
      what a NaN here means. ``null`` reads back as ``None``, which is
      still not ``0.0``, so "nobody measured this" stays distinguishable from
      "the model was certain".

    Anything else passes through to fail at ``dumps``. A value this cannot
    convert is better refused than written as its ``repr``, which reads back as
    a string nobody can turn into the thing it was.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, bytes, bool, int, type(None))):
        return value
    if isinstance(value, Mapping):
        return {str(k): plain_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_json(v) for v in value]
    # A numpy scalar, then a numpy array: `.item()` raises on anything holding
    # more than one element, and `.tolist()` on a scalar returns the same widened
    # Python value. Recursed over, so a float inside an array meets both rules.
    for name in ("item", "tolist"):
        widen = getattr(value, name, None)
        if not callable(widen):
            continue
        try:
            widened = widen()
        except (TypeError, ValueError):
            continue
        return plain_json(widened) if widened is not value else value
    return value


def encode_cell(payload: Mapping | None, what: str) -> str:
    """Serialise one mapping into one CSV cell. ``None`` gives ``""``.

    JSON, one line, keys sorted, rather than a fixed set of columns: a caller
    can record one more thing without a schema migration, and re-recording the
    same fact is a no-op in a diff. ASCII-only (the ``json.dumps`` default) is
    what keeps a literal newline out of the cell -
    :meth:`AnnotationStore._repair_torn_append` finds where a row ends by the
    last newline *byte* and never by parsing.

    ``""`` means *not recorded*, and every column using this codec keeps that as
    a fact distinct from any value the cell can hold.
    """
    if payload is None:
        return ""
    plain = plain_json(payload)
    try:
        return json.dumps(plain, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{what} {plain!r} cannot be written to the CSV: {exc}. Use plain "
            "strings, numbers and booleans; a non-finite number is written as "
            "null, and a value json cannot spell at all is refused here rather "
            "than written into a cell only some readers can parse"
        ) from exc


def decode_cell(value, column: str, shape: str, what: str = "") -> dict:
    """Read such a cell back. ``{}`` means *the row does not say*.

    An empty cell - an old file, or a caller that declined to say - decodes to
    ``{}``. A cell that will not decode RAISES rather than returning ``{}``,
    since reporting a damaged log as "not recorded" hides it behind a plausible
    answer; that difference is why this is a codec and not a ``json.loads`` at
    each call site.
    """
    if value is None or not str(value).strip():
        return {}
    # `what` names the thing the cell holds, so a column keeps its own wording
    # without keeping its own copy of the policy.
    noun = f" {what}" if what else ""
    try:
        out = json.loads(value)
    except ValueError as exc:
        raise ValueError(
            f"{column}={value!r} is not readable{noun}: {exc}"
        ) from exc
    if not isinstance(out, dict):
        raise ValueError(
            f"{column}={value!r} decoded to a {type(out).__name__}, but {shape}"
        )
    return out


# --------------------------------------------------------------------------- #
# selection provenance
# --------------------------------------------------------------------------- #
def encode_provenance(decision: Mapping | str | None) -> str:
    """Serialise *why* a clip was put in front of a human, for one CSV cell.

    :func:`encode_cell` does the writing; what this adds is the one thing this
    column requires of its payload, a rule that NAMES what chose the clip.

    ``None`` encodes as ``""``, meaning *not recorded*, which is a different
    fact from *chosen at random*. A blank rule name is refused below, so the
    empty cell stays the only way to say "this row does not know".
    """
    if decision is None:
        return ""
    if isinstance(decision, str):
        # Already encoded - the GUI may carry a decision through as an opaque
        # string. Re-encode rather than trust it: an unreadable value in the log
        # would only be found by whoever tried to analyse it.
        return encode_provenance(decode_provenance(decision)) if decision.strip() else ""
    if not isinstance(decision, Mapping):
        raise TypeError(
            "selection provenance must be a mapping like "
            f"{{'rule': 'uncertainty', 'score': 0.42}}, got {type(decision).__name__}"
        )
    rule = decision.get("rule")
    if not isinstance(rule, str) or not rule.strip():
        raise ValueError(
            f"selection provenance needs a non-empty 'rule' naming what chose the clip "
            f"(got {rule!r}). An empty cell already means 'not recorded', so a nameless "
            "rule would be indistinguishable from no rule at all"
        )
    return encode_cell(decision, "selection provenance")


def decode_provenance(value: str | None) -> dict:
    """Read a ``selected_by`` cell back. ``{}`` means *the row does not say*.

    An empty cell decodes to ``{}``; a clip genuinely drawn at random decodes to
    a mapping whose ``rule`` says so, so the two are never conflated.
    """
    return decode_cell(
        value, PROVENANCE_COLUMN,
        "provenance is an object like {'rule': 'uncertainty', 'score': 0.42}",
        "selection provenance",
    )


# --------------------------------------------------------------------------- #
# audit membership
# --------------------------------------------------------------------------- #
def encode_audit(member: bool | None) -> str:
    """Serialise *was this clip held out for audit when it was labelled*.

    A recorded fact rather than a hash of the ``clip_id``, because the id embeds
    the track id and re-running detection renumbers the tracks: re-hashing would
    move a labelled clip INTO the holdout after the model had already seen it,
    or OUT of it after it was excluded. Only a LABELLED clip needs the
    guarantee; an unlabelled clip's membership is nobody's dependency and is
    left to the hash.

    ``"1"``/``"0"``, matching the ``uncertain`` column. ``None`` encodes as
    ``""``, meaning *not recorded*, which is a different fact from *not held
    out*.
    """
    if member is None:
        return ""
    return "1" if member else "0"


def decode_audit(value) -> bool | None:
    """Read an ``audit_holdout`` cell back. ``None`` means *the row does not say*.

    Accepts more than the two strings written: ``suggest.train`` reads the CSV
    with ``pandas.read_csv`` and no dtype, so a column of ``1``/``0``/``""``
    arrives as floats with ``NaN`` for the blanks. Anything else RAISES rather
    than defaulting - guessing "held out" scores a trained-on clip as fresh, and
    guessing "not held out" feeds the model a clip kept from it.
    """
    if value is None or isinstance(value, bool):
        return value
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none"):
        return None
    if text in ("1", "1.0", "True", "true"):
        return True
    if text in ("0", "0.0", "False", "false"):
        return False
    raise ValueError(
        f"{AUDIT_COLUMN}={value!r} is not a readable audit-holdout flag: it must "
        "be '1' (this clip was held out when it was labelled), '0' (it was not) "
        "or empty (the row predates the column). Nothing can be trained or "
        "audited while one row will not say which side it is on."
    )


def read_audit_records(path: str | Path) -> dict[str, bool]:
    """``{clip_id: was_held_out}`` for the rows that SAY, straight off the CSV.

    Rows with an empty cell are absent from the mapping rather than present as
    ``None``: the caller's question is "is there a record", and a key mapping to
    "no" would have to be checked for by identity at every call site.

    Read with the ``csv`` module and not pandas, because this is on the import
    path of the GUI's clip view and pandas is not.
    """
    path = Path(path)
    if not path.exists():
        return {}
    out: dict[str, bool] = {}
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if AUDIT_COLUMN not in (reader.fieldnames or []):
            return {}
        for row in reader:
            clip_id = row.get("clip_id")
            if not clip_id:
                continue
            member = decode_audit(row.get(AUDIT_COLUMN))
            if member is not None:
                out[str(clip_id)] = member
    return out


class AnnotationStore:
    def __init__(self, path: str | Path, cfg: Config):
        self.path = Path(path)
        self.cfg = cfg
        self.group_columns = [g.name for g in cfg.categories]
        # `columns` is plain concatenation, so a category named after a built-in
        # would appear twice and the built-in would be silently discarded:
        # DictWriter emits the name twice, `record`'s group loop has already
        # overwritten the built-in's value, and DictReader keeps only the last
        # occurrence. `load_config` refuses these names, but a Config built any
        # other way does not go through it.
        clashes = sorted(set(self.group_columns) & set(IDENTITY_COLUMNS + TRAILING_COLUMNS))
        if clashes:
            raise ValueError(
                f"category group(s) {clashes} are named after built-in annotation CSV "
                "column(s) and would overwrite them. Rename the group(s) in config.yaml."
            )
        self.columns = IDENTITY_COLUMNS + self.group_columns + TRAILING_COLUMNS
        self.rows: dict[str, dict] = {}
        self._order: list[str] = []
        self._undo_stack: list[tuple[str, dict | None, int]] = []
        # All CSV access is serialised: the GUI retrains on a background thread,
        # and on Windows os.replace() fails outright if another handle is open.
        self._lock = threading.RLock()
        self._load()

    # -- io ----------------------------------------------------------------- #
    def _repair_torn_append(self) -> None:
        """Drop a partial final line left by a crash mid-append: otherwise the
        next append is concatenated onto it and BOTH annotations are lost."""
        try:
            size = self.path.stat().st_size
            if size == 0:
                return
            with open(self.path, "rb+") as fh:
                fh.seek(-1, os.SEEK_END)
                if fh.read(1) == b"\n":
                    return
                fh.seek(0)
                data = fh.read()
                cut = data.rfind(b"\n")
                if cut < 0:
                    return
                warnings.warn(
                    f"{self.path.name}: discarding a partially written final row "
                    "(interrupted mid-append)",
                    stacklevel=2,
                )
                fh.seek(cut + 1)
                fh.truncate()
        except OSError:
            pass

    def _load(self) -> None:
        if not self.path.exists():
            return
        self._repair_torn_append()

        with open(self.path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            disk_columns = list(reader.fieldnames or [])
            for row in reader:
                clip_id = row.get("clip_id")
                if not clip_id:
                    continue
                if clip_id not in self.rows:
                    self._order.append(clip_id)
                self.rows[clip_id] = row

        # Category groups can change between sessions. Never append against a
        # stale header (it shifts every later column), and never rewrite with
        # only the new columns (it drops historical ones): take the union.
        # dict.fromkeys, not a list comprehension: a header that repeats a name
        # would otherwise put it in `columns` twice and DictWriter would widen
        # every row on every rewrite.
        extra = list(dict.fromkeys(c for c in disk_columns if c not in self.columns))
        if extra:
            warnings.warn(
                f"{self.path.name} has column(s) not in the current config: {extra}. "
                "They are preserved but no longer written to.",
                stacklevel=2,
            )
            self.columns = self.columns + extra
        for row in self.rows.values():
            for col in self.columns:
                row.setdefault(col, "")

        # FREEZE the audit membership of every labelled clip that has not
        # recorded one, at whatever the hash says TODAY, so the holdout an
        # existing project has now is the holdout it keeps, clip for clip.
        # Here rather than at train time because this module owns the file: a
        # fit must be able to READ the membership without deciding it, or two
        # fits of the same project could disagree.
        stamped, by_record = 0, 0
        for clip_id, row in self.rows.items():
            try:
                member = decode_audit(row.get(AUDIT_COLUMN))
            except ValueError as exc:
                raise ValueError(f"{self.path.name}: {exc}") from exc
            hashed = self.cfg.suggest.audit_by_hash(clip_id)
            if member is None:
                row[AUDIT_COLUMN] = encode_audit(hashed)
                stamped += 1
            elif member != hashed:
                by_record += 1

        if stamped and AUDIT_COLUMN not in disk_columns:
            warnings.warn(
                f"{self.path.name} predates the {AUDIT_COLUMN!r} column: the audit "
                f"membership of its {stamped} labelled clip(s) has been written down "
                "as it stands today, so this upgrade moves no clip into or out of the "
                "holdout. From here it is a recorded fact and survives a re-detection "
                "renumbering the tracks.",
                stacklevel=2,
            )
        # Said once per open, and only when non-zero. Not a warning per row:
        # disagreement is the case this column exists to handle, not an error.
        if by_record:
            warnings.warn(
                f"{self.path.name}: {by_record} of {len(self.rows)} labelled clip(s) "
                "are held to the audit membership recorded when they were labelled, "
                "which is no longer what their track hashes to. Re-running detection "
                "renumbers the tracks, and stamps written before the holdout was drawn "
                "per track were drawn per clip; the recorded value keeps the holdout "
                "the model was judged under. Nothing to fix.",
                stacklevel=2,
            )

        # Said once, in the one moment it matters: the rewrite below fills the
        # new column with "" for every historical row, and "" means NOT
        # RECORDED, which a later analysis must not read as "chosen at random".
        if self.rows and disk_columns and PROVENANCE_COLUMN not in disk_columns:
            warnings.warn(
                f"{self.path.name} predates the {PROVENANCE_COLUMN!r} column: its "
                f"{len(self.rows)} existing row(s) are recorded as provenance NOT "
                "RECORDED, which is not the same as 'chosen at random'. Any statistic "
                "that treats the labelled set as a random sample must say which rows "
                "it is claiming that of.",
                stacklevel=2,
            )
        # `stamped` on its own is enough to force the rewrite: an older build
        # appending to an upgraded file writes a row one field SHORT, which
        # reads back as unrecorded while the header on disk already matches, so
        # a stamp left in memory would be recomputed on the next open.
        if stamped or (disk_columns and disk_columns != self.columns):
            self.rewrite()

    def _write_header_if_needed(self) -> None:
        if self.path.exists() and self.path.stat().st_size > 0:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8", newline="") as fh:
            csv.DictWriter(fh, fieldnames=self.columns).writeheader()
            fh.flush()
            os.fsync(fh.fileno())

    def _append(self, row: dict) -> None:
        with self._lock:
            self._write_header_if_needed()
            with open(self.path, "a", encoding="utf-8", newline="") as fh:
                csv.DictWriter(fh, fieldnames=self.columns, extrasaction="ignore").writerow(row)
                fh.flush()
                os.fsync(fh.fileno())

    def rewrite(self) -> None:
        """Atomically rewrite the whole file (used when an existing row is edited)."""
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".csv.tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
                    writer = csv.DictWriter(fh, fieldnames=self.columns, extrasaction="ignore")
                    writer.writeheader()
                    for clip_id in self._order:
                        if clip_id in self.rows:
                            writer.writerow(self.rows[clip_id])
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, self.path)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise

    def snapshot(self, destination: Path) -> Path:
        """Copy the CSV under the lock, for a reader on another thread: reading
        the live path while the UI thread is mid-``os.replace`` fails the
        replace outright on Windows."""
        with self._lock:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.path, destination)
        return destination

    # -- queries ------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.rows)

    def is_done(self, clip_id: str) -> bool:
        return clip_id in self.rows

    def get(self, clip_id: str) -> dict | None:
        return self.rows.get(clip_id)

    @property
    def done_ids(self) -> set[str]:
        return set(self.rows)

    @property
    def labelled_order(self) -> list[str]:
        """Clip ids in the order they were first labelled.

        A review reads a session back, and a session happened in an order. The
        set ``done_ids`` returns cannot say what came after what.
        """
        return list(self._order)

    @property
    def last_clip_id(self) -> str | None:
        return self._order[-1] if self._order else None

    def missing_required(self, values: dict[str, str]) -> list[str]:
        """Names of required groups that are still unset."""
        return [g.name for g in self.cfg.categories if g.required and not values.get(g.name)]

    # -- writing ------------------------------------------------------------ #
    def record(
        self,
        meta,
        values: dict[str, str],
        uncertain: bool = False,
        notes: str = "",
        dwell_s: float = 0.0,
        suggestion: tuple[str, float] | None = None,
        suggestion_shown: bool = False,
        selected_by: Mapping | str | None = None,
    ) -> dict:
        """Insert or update the annotation for one clip and persist it.

        ``selected_by`` is the selector's own account of why this clip was shown
        (see :func:`encode_provenance`). It defaults to ``None`` - *not
        recorded* - and never to "random": a caller that says nothing has not
        said the sample was unbiased.

        ``audit_holdout`` has no parameter and is not the caller's to supply.
        Provenance is a fact only the selector knows; membership anyone with the
        clip_id and the config can derive, so a parameter would only be a way to
        forget it. See :meth:`_audit_membership`.
        """
        clip_id = str(meta["clip_id"])
        previous = self.rows.get(clip_id)

        row = {
            "clip_id": clip_id,
            "movie_id": str(meta["movie_id"]),
            "source_file": str(meta.get("source_file", "")),
            "series": int(meta.get("series", 0)),
            "channel": int(meta.get("channel", 0)),
            "track_id": int(meta["track_id"]),
            "f_start": int(meta["f_start"]),
            "f_end": int(meta["f_end"]),
            "crop_x0": int(meta["crop_x0"]),
            "crop_y0": int(meta["crop_y0"]),
            "crop_w": int(meta["crop_w"]),
            "crop_h": int(meta["crop_h"]),
            "scale": float(meta["scale"]),
            "uncertain": "1" if uncertain else "0",
            "notes": notes or "",
            "annotator": self.cfg.ui.annotator,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dwell_s": f"{dwell_s:.2f}",
            # Same rule as the provenance column below: an edit with nothing to
            # say about the model keeps what the row already said, because the
            # suggestion visible now is not the one the first judgement was made
            # under.
            "suggested": (
                suggestion[0] if suggestion else (previous or {}).get("suggested", "")
            ),
            "suggest_conf": (
                f"{suggestion[1]:.3f}" if suggestion
                else (previous or {}).get("suggest_conf", "")
            ),
            "suggest_shown": (
                ("1" if suggestion_shown else "0") if suggestion
                else (previous or {}).get("suggest_shown", "0")
            ),
            # Encoded before anything is written: a decision that cannot be
            # serialised must stop the record outright, not land as a blank cell
            # that reads back as "this row does not know why it was shown".
            #
            # An EDIT that says nothing keeps what the row already said. The
            # default is "not recorded", and nobody revising a label months
            # later has a selector to ask, so writing the default over the
            # original would quietly turn every corrected row into one that
            # cannot say whether the model chose it.
            PROVENANCE_COLUMN: (
                encode_provenance(selected_by) if selected_by is not None
                else (previous or {}).get(PROVENANCE_COLUMN, "")
            ),
            AUDIT_COLUMN: encode_audit(self._audit_membership(clip_id, previous)),
        }
        for group in self.cfg.categories:
            # Runs AFTER the literal above, so a group named like a built-in
            # column would overwrite it without a word; __init__ refuses those
            # names for this reason.
            row[group.name] = values.get(group.name, "")

        # Persist before committing: on a failed write (a concurrent reader
        # blocking os.replace on Windows) roll memory back, so it cannot
        # diverge from disk.
        with self._lock:
            previous_order = list(self._order)
            self.rows[clip_id] = row
            if previous is None:
                self._order.append(clip_id)
            try:
                self._append(row) if previous is None else self.rewrite()
            except BaseException:
                if previous is None:
                    self.rows.pop(clip_id, None)
                else:
                    self.rows[clip_id] = previous
                self._order = previous_order
                raise
            position = self._order.index(clip_id)
            self._undo_stack.append((clip_id, dict(previous) if previous else None, position))
        return row

    def _audit_membership(self, clip_id: str, previous: dict | None) -> bool:
        """Whether this clip counts as held out, for the row about to be written.

        An existing row KEEPS what it recorded: after a re-detection has
        renumbered the tracks, re-deriving membership here would let correcting
        a typo in a held-out clip's notes re-hash it into the training set. The
        recorded value is what the human judged under, so it stands.

        A row written for the FIRST time has no record to keep and takes the
        hash, which is what everything else was already using for it. That is
        the moment an unlabelled clip's membership stops being free to move.
        """
        if previous is not None:
            recorded = decode_audit(previous.get(AUDIT_COLUMN))
            if recorded is not None:
                return recorded
        return self.cfg.suggest.audit_by_hash(clip_id)

    def undo(self) -> str | None:
        """Revert the most recent record(). Returns the affected clip_id."""
        with self._lock:
            if not self._undo_stack:
                return None
            clip_id, previous, position = self._undo_stack.pop()
            if previous is None:
                self.rows.pop(clip_id, None)
                if clip_id in self._order:
                    self._order.remove(clip_id)
            else:
                self.rows[clip_id] = previous
                # rewrite() iterates _order, so a row restored to `rows` alone is
                # in memory but absent from the file.
                if clip_id not in self._order:
                    self._order.insert(min(position, len(self._order)), clip_id)
            self.rewrite()
            return clip_id

    def clear(self, clip_id: str) -> None:
        with self._lock:
            if clip_id not in self.rows:
                return
            position = self._order.index(clip_id) if clip_id in self._order else len(self._order)
            self._undo_stack.append((clip_id, dict(self.rows[clip_id]), position))
            self.rows.pop(clip_id, None)
            if clip_id in self._order:
                self._order.remove(clip_id)
            self.rewrite()

    # -- helpers ------------------------------------------------------------ #
    @staticmethod
    def encode_multi(values) -> str:
        return MULTI_SEP.join(sorted(values))

    @staticmethod
    def decode_multi(value: str | None) -> set[str]:
        return {v for v in (value or "").split(MULTI_SEP) if v}
