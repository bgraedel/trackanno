"""The annotation CSV: one fsynced row per labelled clip, with geometry, provenance and audit flag."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import socket
import tempfile
import threading
import time
import warnings
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from ..config import Config

__all__ = [
    "AnnotationStore", "MULTI_SEP", "PROVENANCE_COLUMN", "AUDIT_COLUMN",
    "plain_json", "encode_cell", "decode_cell",
    "encode_provenance", "decode_provenance",
    "encode_audit", "decode_audit", "read_audit_records",
    "IDENTITY_COLUMNS", "TRAILING_COLUMNS",
    "CsvLockedError", "csv_lock",
]

MULTI_SEP = "|"

# Why this clip was shown (see `encode_provenance`).
PROVENANCE_COLUMN = "selected_by"

# Whether the clip was in the audit holdout when it was labelled (see `encode_audit`).
AUDIT_COLUMN = "audit_holdout"

IDENTITY_COLUMNS = [
    "clip_id", "movie_id", "source_file", "series", "channel",
    "track_id", "f_start", "f_end",
    "crop_x0", "crop_y0", "crop_w", "crop_h", "scale",
]
TRAILING_COLUMNS = [
    "uncertain", "notes", "annotator", "timestamp_utc", "dwell_s",
    # What the suggester showed, to tell agreement from influence.
    "suggested", "suggest_conf", "suggest_shown",
    # Keep newer columns last so older builds read the header unchanged.
    PROVENANCE_COLUMN,
    AUDIT_COLUMN,
]

# config.RESERVED_GROUP_NAMES must list these columns too (a test checks).


# --------------------------------------------------------------------------- #
# one JSON object in one CSV cell
# --------------------------------------------------------------------------- #
def plain_json(value):
    """``value`` made JSON-safe: numpy scalars and arrays widened, non-finite floats to ``None``."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, bytes, bool, int, type(None))):
        return value
    if isinstance(value, Mapping):
        return {str(k): plain_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_json(v) for v in value]
    # Numpy scalar via .item(), else array via .tolist(), recursing.
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
    """A mapping as one-line, key-sorted JSON for a CSV cell; ``None`` gives ``""`` (not recorded)."""
    if payload is None:
        return ""
    plain = plain_json(payload)
    try:
        return json.dumps(plain, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{what} {plain!r} cannot be written to the CSV ({exc}); use plain "
            "strings, numbers and booleans"
        ) from exc


def decode_cell(value, column: str, shape: str, what: str = "") -> dict:
    """Decode a JSON cell; empty gives ``{}`` (not recorded), a damaged cell raises."""
    if value is None or not str(value).strip():
        return {}
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
    """Encode why a clip was shown; needs a non-empty ``rule``; ``None`` gives ``""`` (not recorded)."""
    if decision is None:
        return ""
    if isinstance(decision, str):
        # Already encoded: decode and re-encode to validate it.
        return encode_provenance(decode_provenance(decision)) if decision.strip() else ""
    if not isinstance(decision, Mapping):
        raise TypeError(
            "selection provenance must be a mapping like "
            f"{{'rule': 'uncertainty', 'score': 0.42}}, got {type(decision).__name__}"
        )
    rule = decision.get("rule")
    if not isinstance(rule, str) or not rule.strip():
        raise ValueError(
            f"selection provenance needs a non-empty 'rule' (got {rule!r}); an empty "
            "cell means 'not recorded'"
        )
    return encode_cell(decision, "selection provenance")


def decode_provenance(value: str | None) -> dict:
    """Decode a ``selected_by`` cell; ``{}`` means not recorded."""
    return decode_cell(
        value, PROVENANCE_COLUMN,
        "provenance is an object like {'rule': 'uncertainty', 'score': 0.42}",
        "selection provenance",
    )


# --------------------------------------------------------------------------- #
# audit membership
# --------------------------------------------------------------------------- #
def encode_audit(member: bool | None) -> str:
    """Encode audit membership at labelling time as ``"1"``/``"0"``; ``None`` gives ``""``.

    Recorded rather than re-hashed, since re-detection renumbers tracks."""
    if member is None:
        return ""
    return "1" if member else "0"


def decode_audit(value) -> bool | None:
    """Decode an ``audit_holdout`` cell, pandas floats included; ``None`` if empty, raises if bad."""
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
        f"{AUDIT_COLUMN}={value!r} is not a readable audit-holdout flag; use '1', '0' or empty"
    )


def read_audit_records(path: str | Path) -> dict[str, bool]:
    """``{clip_id: was_held_out}`` for rows with a recorded flag, read without pandas."""
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


# --------------------------------------------------------------------------- #
# cross-process lock
# --------------------------------------------------------------------------- #
LOCK_STALE_S = 600.0
LOCK_WAIT_S = 15.0


class CsvLockedError(OSError):
    """Another process is writing the annotation CSV."""


def _lock_path(path: Path) -> Path:
    return path.with_name(path.name + ".lock")


def _pid_alive(pid: int) -> bool:
    """Whether a process with this pid exists on this machine."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.GetLastError() == 5  # access denied: it exists
        try:
            code = ctypes.c_ulong()
            kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _lock_holder(lock: Path) -> dict:
    try:
        return json.loads(lock.read_text(encoding="utf-8") or "{}")
    except (OSError, ValueError):
        return {}


def _lock_is_stale(lock: Path) -> bool:
    try:
        age = time.time() - lock.stat().st_mtime
    except OSError:
        return False
    if age > LOCK_STALE_S:
        return True
    holder = _lock_holder(lock)
    pid = holder.get("pid")
    return (holder.get("host") == socket.gethostname() and isinstance(pid, int)
            and not _pid_alive(pid))


def _try_lock(path: Path) -> bool:
    """Create the lock file once, breaking a stale one. True if acquired."""
    lock = _lock_path(path)
    for _ in range(2):
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if not _lock_is_stale(lock):
                return False
            lock.unlink(missing_ok=True)
            continue
        except OSError as exc:
            # Windows denies access to a lock file pending deletion.
            try:
                os.stat(lock)
            except FileNotFoundError:
                raise OSError(
                    exc.errno, f"cannot create {lock} ({exc.strerror}); "
                    "the annotation folder must be writable"
                ) from exc
            except OSError:
                pass
            return False
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"pid": os.getpid(), "host": socket.gethostname(),
                       "time": datetime.now(timezone.utc).isoformat(timespec="seconds")}, fh)
        return True
    return False


def _unlock(path: Path) -> None:
    _lock_path(path).unlink(missing_ok=True)


@contextmanager
def csv_lock(path: str | Path, wait: float | None = None):
    """Hold ``<csv>.lock`` for the duration; raise `CsvLockedError` after ``wait`` s."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + (LOCK_WAIT_S if wait is None else wait)
    while not _try_lock(path):
        if time.monotonic() >= deadline:
            holder = _lock_holder(_lock_path(path))
            who = f"pid {holder.get('pid')} on {holder.get('host')}" if holder else "another process"
            raise CsvLockedError(
                f"{path.name} is being written by {who}; try again, or delete "
                f"{_lock_path(path).name} if that process is gone"
            )
        time.sleep(0.05)
    try:
        yield
    finally:
        _unlock(path)


def _norm(row: dict | None) -> dict | None:
    """A row as comparable text, ignoring empty cells."""
    if row is None:
        return None
    return {k: str(v) for k, v in row.items() if k is not None and v not in ("", None)}


class AnnotationStore:
    def __init__(self, path: str | Path, cfg: Config):
        self.path = Path(path)
        self.cfg = cfg
        self.group_columns = [g.name for g in cfg.categories]
        # load_config refuses these names too, but a Config may be built directly.
        clashes = sorted(set(self.group_columns) & set(IDENTITY_COLUMNS + TRAILING_COLUMNS))
        if clashes:
            raise ValueError(
                f"category group(s) {clashes} are named after built-in CSV columns; "
                "rename them in config.yaml"
            )
        self.columns = IDENTITY_COLUMNS + self.group_columns + TRAILING_COLUMNS
        self.rows: dict[str, dict] = {}
        self._order: list[str] = []
        self._undo_stack: list[tuple[str, dict | None, int]] = []
        # Serialises CSV access across threads (os.replace fails on open handles on Windows).
        self._lock = threading.RLock()
        # Disk state at our last read or write, for detecting other writers.
        self._base: dict[str, dict] = {}
        self._stat: tuple[int, int] | None = None
        self._held = False
        self._needs_rewrite = False
        self._load()

    # -- io ----------------------------------------------------------------- #
    def _repair_torn_append(self) -> None:
        """Drop a partial final line left by a crash mid-append."""
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

    def _disk_stat(self) -> tuple[int, int] | None:
        try:
            st = self.path.stat()
        except FileNotFoundError:
            return None
        return (st.st_size, st.st_mtime_ns)

    def _read_disk(self) -> tuple[list[str], dict[str, dict], list[str]]:
        """Header, rows by clip_id and first-seen order, straight off the file."""
        rows: dict[str, dict] = {}
        order: list[str] = []
        with open(self.path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            columns = list(reader.fieldnames or [])
            for row in reader:
                clip_id = row.get("clip_id")
                if not clip_id:
                    continue
                if clip_id not in rows:
                    order.append(clip_id)
                rows[clip_id] = row
        return columns, rows, order

    def _widen(self, disk_columns: list[str]) -> None:
        # Keep columns from the file that the config no longer has (deduplicated).
        extra = list(dict.fromkeys(c for c in disk_columns if c not in self.columns))
        if extra:
            warnings.warn(
                f"{self.path.name} has column(s) not in the config, kept as they are: {extra}",
                stacklevel=3,
            )
            self.columns = self.columns + extra

    def _stamp(self, rows: dict[str, dict]) -> tuple[int, int]:
        """Fill missing columns and stamp audit membership; returns ``(stamped, held_by_record)``."""
        stamped, by_record = 0, 0
        for clip_id, row in rows.items():
            for col in self.columns:
                row.setdefault(col, "")
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
        return stamped, by_record

    def _load(self) -> None:
        if not self.path.exists():
            return
        self._repair_torn_append()
        self._stat = self._disk_stat()
        disk_columns, self.rows, self._order = self._read_disk()
        self._widen(disk_columns)

        # Stamp unrecorded audit membership from today's hash, so it stays fixed.
        stamped, by_record = self._stamp(self.rows)
        self._base = {cid: _norm(row) for cid, row in self.rows.items()}

        if stamped and AUDIT_COLUMN not in disk_columns:
            warnings.warn(
                f"{self.path.name}: added the {AUDIT_COLUMN!r} column for {stamped} "
                "labelled clip(s) from the current hash; this moves no clip",
                stacklevel=2,
            )
        if by_record:
            warnings.warn(
                f"{self.path.name}: {by_record} of {len(self.rows)} labelled clip(s) keep "
                "the audit membership recorded when they were labelled, not their "
                "current hash. Nothing to fix.",
                stacklevel=2,
            )
        if self.rows and disk_columns and PROVENANCE_COLUMN not in disk_columns:
            warnings.warn(
                f"{self.path.name}: added the {PROVENANCE_COLUMN!r} column; its "
                f"{len(self.rows)} existing row(s) read as NOT RECORDED, not as chosen "
                "at random",
                stacklevel=2,
            )
        # Any stamp forces a rewrite, so it is on disk before the next open.
        self._needs_rewrite = bool(stamped or (disk_columns and disk_columns != self.columns))
        # Deferred to the next write while another process holds the lock.
        try:
            locked = self._needs_rewrite and _try_lock(self.path)
        except OSError:
            locked = False  # read-only: open anyway, writes will say why
        if locked:
            self._held = True
            try:
                self._sync()
                self.rewrite()
            finally:
                self._held = False
                _unlock(self.path)

    @contextmanager
    def _writing(self):
        """Hold the thread and file locks, with memory synced to disk."""
        with self._lock:
            if self._held:
                yield
                return
            with csv_lock(self.path):
                self._held = True
                try:
                    self._sync()
                    yield
                finally:
                    self._held = False

    def _sync(self) -> None:
        """Merge rows another process wrote: rows changed here win, others take the disk's."""
        stat = self._disk_stat()
        if stat == self._stat:
            return
        if stat is None:
            self._needs_rewrite = True
            return
        self._repair_torn_append()
        disk_columns, theirs, their_order = self._read_disk()
        self._widen(disk_columns)
        stamped, _ = self._stamp(theirs)
        merged: dict[str, dict] = {}
        for clip_id in dict.fromkeys([*self._order, *their_order, *self._base]):
            ours = self.rows.get(clip_id)
            changed_here = _norm(ours) != self._base.get(clip_id)
            pick = ours if changed_here else theirs.get(clip_id)
            if pick is not None:
                merged[clip_id] = pick
        seen = set(self._order)
        self._order = [c for c in self._order if c in merged] + [
            c for c in their_order if c in merged and c not in seen
        ]
        self.rows = merged
        self._stamp(self.rows)
        self._base = {cid: _norm(row) for cid, row in theirs.items()}
        self._stat = self._disk_stat()
        self._needs_rewrite = bool(stamped or disk_columns != self.columns)

    def _write_header_if_needed(self) -> None:
        if self.path.exists() and self.path.stat().st_size > 0:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8", newline="") as fh:
            csv.DictWriter(fh, fieldnames=self.columns).writeheader()
            fh.flush()
            os.fsync(fh.fileno())

    def _append(self, row: dict) -> None:
        with self._writing():
            if self._needs_rewrite:
                self.rewrite()
                return
            self._write_header_if_needed()
            with open(self.path, "a", encoding="utf-8", newline="") as fh:
                csv.DictWriter(fh, fieldnames=self.columns, extrasaction="ignore").writerow(row)
                fh.flush()
                os.fsync(fh.fileno())
            self._base[str(row["clip_id"])] = _norm(row)
            self._stat = self._disk_stat()

    def rewrite(self) -> None:
        """Atomically rewrite the whole file."""
        with self._writing():
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
            self._base = {cid: _norm(self.rows[cid]) for cid in self._order if cid in self.rows}
            self._stat = self._disk_stat()
            self._needs_rewrite = False

    def snapshot(self, destination: Path) -> Path:
        """Copy the CSV under the lock, for a reader on another thread."""
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
        """Clip ids in the order they were first labelled."""
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
        """Insert or update one clip's annotation and persist it.

        ``selected_by`` says why the clip was shown; ``None`` means not recorded."""
        clip_id = str(meta["clip_id"])
        with self._writing():
            return self._record(clip_id, meta, values, uncertain, notes, dwell_s,
                                suggestion, suggestion_shown, selected_by)

    def _record(self, clip_id, meta, values, uncertain, notes, dwell_s,
                suggestion, suggestion_shown, selected_by) -> dict:
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
            # An edit without a suggestion keeps the recorded one.
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
            # Encoded before writing, so a bad decision fails the record; an edit
            # without one keeps the recorded provenance.
            PROVENANCE_COLUMN: (
                encode_provenance(selected_by) if selected_by is not None
                else (previous or {}).get(PROVENANCE_COLUMN, "")
            ),
            AUDIT_COLUMN: encode_audit(self._audit_membership(clip_id, previous)),
        }
        for group in self.cfg.categories:
            row[group.name] = values.get(group.name, "")

        # Roll memory back if the write fails, so it matches disk.
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
        """Audit membership for the row being written: the recorded value, else the hash."""
        if previous is not None:
            recorded = decode_audit(previous.get(AUDIT_COLUMN))
            if recorded is not None:
                return recorded
        return self.cfg.suggest.audit_by_hash(clip_id)

    def undo(self) -> str | None:
        """Revert the most recent record(). Returns the affected clip_id."""
        with self._writing():
            if not self._undo_stack:
                return None
            clip_id, previous, position = self._undo_stack.pop()
            if previous is None:
                self.rows.pop(clip_id, None)
                if clip_id in self._order:
                    self._order.remove(clip_id)
            else:
                self.rows[clip_id] = previous
                # rewrite() writes only rows listed in _order.
                if clip_id not in self._order:
                    self._order.insert(min(position, len(self._order)), clip_id)
            self.rewrite()
            return clip_id

    def clear(self, clip_id: str) -> None:
        with self._writing():
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
