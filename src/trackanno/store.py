"""The annotation log.

One CSV, one row per annotated clip. Design constraints:

* **Resumable** - the CSV *is* the state; a clip whose ``clip_id`` appears is done.
* **Crash-safe** - each annotation is appended and fsynced at once, so a kill
  loses at most the clip on screen.
* **Reassociatable** - every row carries the source file, track id, frame range
  and crop rectangle in original pixel coordinates.
* **Self-describing** - every row records *why* that clip was put in front of a
  human, because the answer stopped being "the next one in the manifest" the
  moment a selector started picking by model uncertainty.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
import threading
import warnings
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from .config import Config

__all__ = [
    "AnnotationStore", "MULTI_SEP", "PROVENANCE_COLUMN",
    "encode_provenance", "decode_provenance",
    "IDENTITY_COLUMNS", "TRAILING_COLUMNS",
]

MULTI_SEP = "|"

# Why this clip was shown. See `encode_provenance` for what goes in it.
PROVENANCE_COLUMN = "selected_by"

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
    # LAST, and that position is load-bearing rather than cosmetic. An older
    # build opening a file this one wrote treats an unrecognised column as
    # `extra` and appends it to the end of its own list (see `_load`); only if
    # it is already last does that reconstructed order match the header on disk,
    # so the old build leaves the file alone instead of rewriting it. Insert it
    # anywhere else and every downgrade churns the whole log.
    PROVENANCE_COLUMN,
]

# config.RESERVED_GROUP_NAMES restates these two lists by hand, and a column
# added here but not there is not an error - it is a name a user is then allowed
# to give a category, after which `record` writes the built-in and the category
# into one key and the built-in is gone, silently.
# tests/test_store_provenance.py asserts the two sets are equal, which is the
# only thing that catches the omission.
#
# The duplication is not forced: `Config` is used here only in an annotation, and
# `from __future__ import annotations` makes that lazy, so this import could move
# under TYPE_CHECKING and config.py could derive its set from these two lists.
# That inversion was left out of the change that added the provenance column
# rather than bundled into it; the equality test is what makes leaving it safe.


# --------------------------------------------------------------------------- #
# selection provenance
# --------------------------------------------------------------------------- #
def _plain(value):
    """numpy scalars to Python scalars, exactly, so ``json`` will take them.

    A score straight out of ``predict_proba`` is a ``numpy.float32``, which
    ``json`` refuses. Left to fail it would abort the save of a label a person
    had just typed - the one moment in this program where losing work is least
    acceptable. ``.item()`` widens rather than rounds, so nothing is guessed;
    anything it cannot convert is passed through to fail loudly at ``dumps``.
    """
    if isinstance(value, (str, bytes, bool, int, float, type(None))):
        return value
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            return value
    return value


def encode_provenance(decision: Mapping | str | None) -> str:
    """Serialise *why* a clip was put in front of a human, for one CSV cell.

    Clips used to be served in manifest order, which made the labelled set an
    unbiased sample and made every statistic computed over it - class balance,
    cross-validated F1 - a statement about the data. An active-learning selector
    breaks that silently: the labelled set becomes whatever the model asked for
    and nothing in the file says so, and "the model is 89% accurate" can no
    longer be read at all, because nobody can tell which rows were drawn at
    random and which the model chose.

    JSON, one line, keys sorted, rather than a fixed pair of columns: the set of
    selection rules will grow, and a rule that needs to record one more thing
    (which class was starved, which model scored it) must not force another
    schema migration - each of which rewrites everyone's CSV. Sorting the keys
    makes the encoding byte-stable, so re-recording the same decision is a no-op
    in a diff rather than a spurious change.

    ASCII-only (the ``json.dumps`` default) is what makes the value safe for
    :meth:`AnnotationStore._repair_torn_append`, which decides where a row ends
    by looking for the last newline *byte* and never by parsing: a literal
    newline inside any field would let it cut an intact row in half. JSON
    escapes those to ``\\n``, so no rule name ever can. Commas and quotes need no
    such care - the csv module quotes and doubles them and reads them back
    identically.

    ``None`` encodes as ``""``, which is the encoding of *not recorded*. That is
    a different fact from *chosen at random*, and it is deliberately not
    spellable as a rule: a blank rule name is refused below, so the empty cell
    stays the one and only way to say "this row does not know".
    """
    if decision is None:
        return ""
    if isinstance(decision, str):
        # Already encoded - the GUI may carry a decision through as an opaque
        # string. Re-encode rather than trust it: an unreadable value written
        # into the log would only be discovered by whoever tried to analyse it.
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
    plain = {str(k): _plain(v) for k, v in decision.items()}
    try:
        return json.dumps(plain, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"selection provenance {plain!r} cannot be written to the CSV: {exc}. "
            "Use plain strings, numbers and booleans; NaN and Infinity are refused "
            "because they are not JSON and do not read back the same everywhere"
        ) from exc


def decode_provenance(value: str | None) -> dict:
    """Read a ``selected_by`` cell back. ``{}`` means *the row does not say*.

    This function exists so that "not recorded" and "chosen at random" cannot be
    conflated. An empty cell - an old file, or a caller that declined to say -
    decodes to ``{}``; a clip genuinely drawn at random decodes to a mapping
    whose ``rule`` says so. Anything that defaults the empty case to "random" is
    re-introducing the bug this column was added to prevent.

    A cell that will not decode raises rather than returning ``{}``: corrupt is a
    third fact again, and reporting it as "not recorded" would hide a damaged log
    behind a plausible-looking answer.
    """
    if value is None or not str(value).strip():
        return {}
    try:
        out = json.loads(value)
    except ValueError as exc:
        raise ValueError(
            f"{PROVENANCE_COLUMN}={value!r} is not readable selection provenance: {exc}"
        ) from exc
    if not isinstance(out, dict):
        raise ValueError(
            f"{PROVENANCE_COLUMN}={value!r} decoded to a {type(out).__name__}, but "
            "provenance is an object like {'rule': 'uncertainty', 'score': 0.42}"
        )
    return out


class AnnotationStore:
    def __init__(self, path: str | Path, cfg: Config):
        self.path = Path(path)
        self.cfg = cfg
        self.group_columns = [g.name for g in cfg.categories]
        # `columns` is plain concatenation, so a category named after a built-in
        # appears in it TWICE and nothing downstream complains: DictWriter emits
        # the header name twice, `record` has already overwritten the built-in's
        # value with the category's (the group loop runs after the row literal),
        # and DictReader keeps only the last occurrence on the way back in. The
        # net effect is a built-in column silently discarded - which for
        # PROVENANCE_COLUMN is exactly the failure it was added to prevent.
        # `load_config` refuses these names, but a Config built any other way
        # does not go through it, so refuse here too rather than corrupt the log.
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

        # Say it once, in the one moment it matters: the rewrite below fills the
        # new column with "" for every historical row, and "" means NOT RECORDED.
        # Left unsaid, a later analysis would read those blanks as "chosen at
        # random" - the exact conflation this column exists to make impossible.
        if self.rows and disk_columns and PROVENANCE_COLUMN not in disk_columns:
            warnings.warn(
                f"{self.path.name} predates the {PROVENANCE_COLUMN!r} column: its "
                f"{len(self.rows)} existing row(s) are recorded as provenance NOT "
                "RECORDED, which is not the same as 'chosen at random'. Any statistic "
                "that treats the labelled set as a random sample must say which rows "
                "it is claiming that of.",
                stacklevel=2,
            )
        if disk_columns and disk_columns != self.columns:
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
        (see :func:`encode_provenance`). It defaults to ``None`` - *not recorded*
        - and deliberately not to "random": a caller that says nothing has not
        told us the sample was unbiased, it has told us nothing, and writing the
        stronger claim on its behalf is the guess this column exists to refuse.
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
            "suggested": suggestion[0] if suggestion else "",
            "suggest_conf": f"{suggestion[1]:.3f}" if suggestion else "",
            "suggest_shown": "1" if suggestion_shown else "0",
            # Encoded here, before anything is written: a decision that cannot be
            # serialised must stop the record outright, not land as a blank cell
            # that reads back as "this row does not know why it was shown".
            PROVENANCE_COLUMN: encode_provenance(selected_by),
        }
        for group in self.cfg.categories:
            # This runs AFTER the literal above, so a group named like a built-in
            # column would overwrite it without a word. __init__ refuses those
            # names outright; this loop is the reason it has to.
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
