"""Project status and the recent list must not hang on a dead share."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from trackanno import project
from trackanno.annotate.store import AnnotationStore
from trackanno.config import load_config
from test_store_provenance import AWKWARD, CONFIG_TEXT, _meta

HANG_S = 5.0


@pytest.fixture(autouse=True)
def _fresh_probes(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACKANNO_STATE_DIR", str(tmp_path / "state"))
    project.forget_path_probes()
    yield
    project.forget_path_probes()


def _hang_on(monkeypatch, marker: str) -> None:
    """Make every probe of a path containing ``marker`` block like a dead share."""
    real = project._probe

    def slow(path, kind):
        if marker in str(path):
            time.sleep(HANG_S)
        return real(path, kind)

    monkeypatch.setattr(project, "_probe", slow)


def test_a_dead_share_in_the_recent_list_does_not_stall(tmp_path, monkeypatch) -> None:
    local = tmp_path / "local" / "config.yaml"
    dead = tmp_path / "deadshare" / "config.yaml"
    for path in (local, dead):
        path.parent.mkdir(parents=True)
        path.write_text("", encoding="utf-8")
    state = tmp_path / "state" / "trackanno"
    state.mkdir(parents=True)
    (state / project.RECENT_FILE).write_text(json.dumps([str(dead), str(local)]))
    _hang_on(monkeypatch, "deadshare")

    started = time.monotonic()
    recent = project.recent_projects()
    assert time.monotonic() - started < 2.0
    assert recent == [local]
    # Remembering another project keeps the unreachable entry for later.
    project.remember_project(local)
    stored = json.loads((state / project.RECENT_FILE).read_text())
    assert str(dead) in stored


def test_status_does_not_stall_on_a_hanging_listing(tmp_path, monkeypatch) -> None:
    (tmp_path / "config.yaml").write_text(CONFIG_TEXT, encoding="utf-8")
    (tmp_path / "videos").mkdir()
    cfg = load_config(tmp_path / "config.yaml")
    monkeypatch.setattr(project, "_LIST_TIMEOUT", 0.3)
    monkeypatch.setattr(project, "_count_videos", lambda *a: time.sleep(HANG_S))
    started = time.monotonic()
    status = project.project_status(cfg)
    assert time.monotonic() - started < 2.0
    assert status.video_dir_ok and status.n_videos is None
    assert status.videos_text.startswith("?")


def test_annotated_counts_rows_not_lines(tmp_path) -> None:
    (tmp_path / "config.yaml").write_text(CONFIG_TEXT, encoding="utf-8")
    cfg = load_config(tmp_path / "config.yaml")
    store = AnnotationStore(cfg.paths.csv, cfg)
    for i in (1, 2, 3):
        store.record(_meta(i), {"behaviour": "swimming"}, notes=AWKWARD)
    assert project.project_status(cfg).n_annotated == 3
