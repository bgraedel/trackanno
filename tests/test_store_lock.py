"""Two writers on one annotation CSV."""

from __future__ import annotations

import csv
import json
import os
import socket
import subprocess
import sys
import time
import warnings
from pathlib import Path

import pytest

from trackanno.annotate import store as store_mod
from trackanno.annotate.store import (
    AUDIT_COLUMN, AnnotationStore, CsvLockedError, csv_lock,
)
from trackanno.config import load_config
from test_store_provenance import CONFIG_TEXT, _meta, _old_row, _write_old_csv


@pytest.fixture
def cfg(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG_TEXT, encoding="utf-8")
    return load_config(path)


def _rows(path: Path) -> dict[str, dict]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return {r["clip_id"]: r for r in csv.DictReader(fh)}


def _id(i: int) -> str:
    return _meta(i)["clip_id"]


def _lock_file(path: Path, pid: int) -> Path:
    lock = path.with_name(path.name + ".lock")
    lock.write_text(json.dumps({"pid": pid, "host": socket.gethostname()}), encoding="utf-8")
    return lock


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_rewrite_keeps_rows_appended_by_another_store(cfg) -> None:
    a = AnnotationStore(cfg.paths.csv, cfg)
    a.record(_meta(1), {"behaviour": "swimming"})
    b = AnnotationStore(cfg.paths.csv, cfg)
    a.record(_meta(2), {"behaviour": "swimming"})
    b.record(_meta(3), {"behaviour": "tumbling"})
    # An edit rewrites the whole file from memory: rows 2 and 3 must survive.
    a.record(_meta(1), {"behaviour": "tumbling"}, notes="edit by a")
    b.record(_meta(2), {"behaviour": "tumbling"}, notes="edit by b")
    rows = _rows(cfg.paths.csv)
    assert set(rows) == {_id(1), _id(2), _id(3)}
    assert rows[_id(1)]["notes"] == "edit by a"
    assert rows[_id(2)]["notes"] == "edit by b"
    assert rows[_id(3)]["behaviour"] == "tumbling"
    assert len(a) == len(b) == 3
    assert not cfg.paths.csv.with_name("annotations.csv.lock").exists()


def test_a_clear_by_one_store_is_not_undone_by_the_other(cfg) -> None:
    a = AnnotationStore(cfg.paths.csv, cfg)
    for i in (1, 2):
        a.record(_meta(i), {"behaviour": "swimming"})
    b = AnnotationStore(cfg.paths.csv, cfg)
    a.clear(_id(1))
    b.record(_meta(2), {"behaviour": "tumbling"})
    assert set(_rows(cfg.paths.csv)) == {_id(2)}


def test_a_live_lock_refuses_and_a_dead_one_is_broken(cfg, monkeypatch) -> None:
    monkeypatch.setattr(store_mod, "LOCK_WAIT_S", 0.2)
    a = AnnotationStore(cfg.paths.csv, cfg)
    lock = _lock_file(cfg.paths.csv, os.getpid())
    with pytest.raises(CsvLockedError, match="being written"):
        a.record(_meta(1), {"behaviour": "swimming"})
    assert len(a) == 0
    _lock_file(cfg.paths.csv, _dead_pid())
    a.record(_meta(1), {"behaviour": "swimming"})
    assert set(_rows(cfg.paths.csv)) == {_id(1)}
    assert not lock.exists()


def test_open_defers_the_audit_stamp_rewrite_while_locked(cfg, monkeypatch) -> None:
    monkeypatch.setattr(store_mod, "LOCK_WAIT_S", 0.2)
    path = cfg.paths.csv
    _write_old_csv(path, cfg, [_old_row(1)])
    before = path.read_bytes()
    lock = _lock_file(path, os.getpid())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = AnnotationStore(path, cfg)
    assert path.read_bytes() == before
    lock.unlink()
    a.record(_meta(2), {"behaviour": "tumbling"})
    rows = _rows(path)
    assert set(rows) == {_id(1), _id(2)}
    assert rows[_id(1)][AUDIT_COLUMN] in ("0", "1")


def test_an_unwritable_folder_fails_at_once(cfg, monkeypatch) -> None:
    a = AnnotationStore(cfg.paths.csv, cfg)
    real_open = os.open

    def refuse(path, *args, **kwargs):
        if str(path).endswith(".lock"):
            raise PermissionError(13, "Permission denied")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(store_mod.os, "open", refuse)
    started = time.monotonic()
    with pytest.raises(OSError, match="must be writable"):
        a.record(_meta(1), {"behaviour": "swimming"})
    assert time.monotonic() - started < 1.0
    assert len(a) == 0


def test_csv_lock_excludes_a_store(cfg, monkeypatch) -> None:
    monkeypatch.setattr(store_mod, "LOCK_WAIT_S", 0.2)
    a = AnnotationStore(cfg.paths.csv, cfg)
    with csv_lock(cfg.paths.csv):
        with pytest.raises(CsvLockedError):
            a.record(_meta(1), {"behaviour": "swimming"})
    a.record(_meta(1), {"behaviour": "swimming"})
    assert len(_rows(cfg.paths.csv)) == 1
