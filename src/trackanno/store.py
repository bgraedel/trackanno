"""The annotation log.

One CSV, one row per annotated clip. Design constraints:

* **Resumable** - the CSV *is* the state; a clip whose ``clip_id`` appears is done.
* **Crash-safe** - each annotation is appended and fsynced at once, so a kill
  loses at most the clip on screen.
* **Reassociatable** - every row carries the source file, track id, frame range
  and crop rectangle in original pixel coordinates.
"""

from __future__ import annotations

import csv
import os
import shutil
import tempfile
import threading
import warnings
from datetime import datetime, timezone
from pathlib import Path

from .config import Config

__all__ = ["AnnotationStore", "MULTI_SEP"]

MULTI_SEP = "|"

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
]


class AnnotationStore:
    def __init__(self, path: str | Path, cfg: Config):
        self.path = Path(path)
        self.cfg = cfg
        self.group_columns = [g.name for g in cfg.categories]
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
        extra = [c for c in disk_columns if c not in self.columns]
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
    ) -> dict:
        """Insert or update the annotation for one clip and persist it."""
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
        }
        for group in self.cfg.categories:
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
