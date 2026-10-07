"""Project status and the recent list must not hang on a dead share."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from trackanno import project
from trackanno.annotate.store import AnnotationStore
from trackanno.config import load_config
from fixtures import AWKWARD, STORE_CONFIG, clip_meta

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
    (tmp_path / "config.yaml").write_text(STORE_CONFIG, encoding="utf-8")
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
    (tmp_path / "config.yaml").write_text(STORE_CONFIG, encoding="utf-8")
    cfg = load_config(tmp_path / "config.yaml")
    store = AnnotationStore(cfg.paths.csv, cfg)
    for i in (1, 2, 3):
        store.record(clip_meta(i), {"behaviour": "swimming"}, notes=AWKWARD)
    assert project.project_status(cfg).n_annotated == 3


class _Unreachable(type(Path())):
    """Every probe raises, the way a dropped share does (WinError 64)."""

    def _gone(self, *args):
        raise OSError(64, "The specified network name is no longer available")

    is_dir = is_file = exists = glob = _gone


@pytest.mark.parametrize("kind", ["dir", "file", None])
def test_an_unreachable_path_reads_as_absent(kind) -> None:
    path = _Unreachable("//nowhere.invalid/share")
    assert not (project._exists(path, kind) if kind else project._exists(path))


def test_a_dropped_share_still_reports_the_local_cache(tmp_path) -> None:
    from trackanno.project import create_project, write_config

    path = create_project(tmp_path / "offline")
    movie = tmp_path / "offline" / "work" / "movie_a"
    movie.mkdir(parents=True)
    (movie / "prepare_done.json").write_text(json.dumps({"n_clips": 12}), encoding="utf-8")
    write_config(path, {"paths.video_dir": "//nowhere.invalid/share/videos",
                        "paths.model": "//nowhere.invalid/share/best.pt"})
    cfg = load_config(path)
    cfg.paths.video_dir = _Unreachable("//nowhere.invalid/share/videos")
    cfg.paths.model = _Unreachable("//nowhere.invalid/share/best.pt")

    status = project.project_status(cfg)
    assert not status.video_dir_ok and not status.model_ok
    assert status.n_clips == 12, status.n_clips


def test_a_probe_that_never_answers_is_bounded_and_remembered(monkeypatch) -> None:
    calls = []

    def glacial(path, kind):
        calls.append(str(path))
        time.sleep(HANG_S)
        return True

    monkeypatch.setattr(project, "_probe", glacial)
    share = Path("//nowhere.invalid/share/videos")
    started = time.perf_counter()
    assert project._exists(share, "dir") is False, "reports absent rather than hanging"
    assert time.perf_counter() - started < 2.0

    started = time.perf_counter()
    project._exists(share, "dir")
    assert time.perf_counter() - started < 0.2, "the verdict is remembered"
    assert len(calls) == 1, calls


def test_local_paths_stay_fast_and_correct(tmp_path) -> None:
    started = time.perf_counter()
    for _ in range(100):
        project._exists(Path(__file__), "file")
    assert time.perf_counter() - started < 0.5
    assert project._exists(Path(__file__), "file")
    assert not project._exists(tmp_path / "nope", "file")
