"""Projects: a folder, a config, and the state of the work done in it.

A *project* is a directory holding

    config.yaml        every setting, with its comments intact
    work/              the clip cache and the trained suggester
    annotations.csv    the labels

This module creates one from a template, remembers the ones you have opened,
and summarises how far each has got without opening a single movie file.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from .yamledit import replace_section, set_values

__all__ = [
    "DEFAULT_CONFIG", "ProjectStatus", "create_project", "project_status",
    "recent_projects", "remember_project", "forget_project", "write_config",
    "candidate_config", "config_path_for",
]

CONFIG_NAME = "config.yaml"
RECENT_FILE = "recent.json"
MAX_RECENT = 12


DEFAULT_CONFIG_FILE = "default_config.yaml"


def _default_config() -> str:
    """The commented config.yaml a new project starts from.

    A data file beside the code rather than a string inside it, so it can be
    edited, diffed and syntax-checked as the yaml it is.
    """
    path = resources.files(__package__).joinpath(DEFAULT_CONFIG_FILE)
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, FileNotFoundError) as exc:
        raise RuntimeError(
            f"{DEFAULT_CONFIG_FILE} is missing from the installed package, so a "
            "new project has no settings to start from. Reinstall trackanno."
        ) from exc


DEFAULT_CONFIG = _default_config()


# --------------------------------------------------------------------------- #
# creating and locating
# --------------------------------------------------------------------------- #
def config_path_for(target: str | Path) -> Path:
    """Accept either a project folder or the config file itself."""
    path = Path(target).expanduser()
    return path / CONFIG_NAME if path.is_dir() else path


def create_project(
    directory: str | Path,
    video_dir: str | Path | None = None,
    model: str | Path | None = None,
    work_dir: str | Path | None = None,
    csv: str | Path | None = None,
    categories: list[dict] | None = None,
    overwrite: bool = False,
    template: str | None = None,
) -> Path:
    """Write a fresh, fully commented ``config.yaml``. Returns its path.

    ``template`` names a set of saved settings to start from
    (:mod:`trackanno.templates`); without one, the packaged default is used. The
    paths below are applied on top either way, so a template never carries
    another project's movie folder into this one.
    """
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / CONFIG_NAME
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists")

    from .templates import template_text

    text = template_text(template)
    updates: dict[str, object] = {}
    if video_dir:
        updates["paths.video_dir"] = _relative_to(Path(video_dir), directory)
    if model:
        updates["paths.model"] = _relative_to(Path(model), directory)
    updates["paths.work_dir"] = _relative_to(Path(work_dir), directory) if work_dir else "./work"
    updates["paths.csv"] = (
        _relative_to(Path(csv), directory) if csv else "./annotations.csv"
    )
    text = set_values(text, updates)
    if categories:
        text = replace_section(
            text, "categories", categories,
            "# --- Annotation categories ---------------------------------------------------\n"
            "# Each group becomes a block of buttons; `key` is its keyboard shortcut.",
        )

    path.write_text(text, encoding="utf-8")
    (directory / "work").mkdir(exist_ok=True)
    remember_project(path)
    return path


def _relative_to(path: Path, root: Path) -> str:
    """Keep a project portable when its pieces live inside it."""
    path = path.expanduser()
    try:
        resolved = path.resolve()
        if resolved.is_relative_to(root):
            return "./" + resolved.relative_to(root).as_posix()
    except (OSError, ValueError):
        pass
    return str(path).replace("\\", "/")


def candidate_config(path: str | Path, updates: dict, categories: list[dict] | None = None,
                     keypoints: dict | None = None) -> str:
    """The text an edit *would* produce, without writing anything."""
    text = Path(path).read_text(encoding="utf-8")
    if updates:
        text = set_values(text, updates)
    if categories is not None:
        text = replace_section(
            text, "categories", categories,
            "# --- Annotation categories ---------------------------------------------------\n"
            "# Each group becomes a block of buttons; `key` is its keyboard shortcut and\n"
            "# `color` is used for the buttons AND the rendered overlay movie.",
        )
    if keypoints is not None:
        text = replace_section(
            text, "keypoints", keypoints,
            "# --- Keypoints (order MUST match the trained model) --------------------------",
        )
    return text


def write_config(path: str | Path, updates: dict, categories: list[dict] | None = None,
                 keypoints: dict | None = None, validate: bool = True) -> Path:
    """Apply edits to an existing config.yaml, comments and all.

    With ``validate`` the candidate is parsed by the real loader *before* it
    replaces anything, so an edit that would not load is refused outright rather
    than written and rolled back. That is what lets the GUI autosave.
    """
    path = Path(path)
    text = candidate_config(path, updates, categories, keypoints)

    # Same folder, so relative paths inside resolve exactly as they will after
    # the rename - validating somewhere else would check a different config.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    try:
        if validate:
            from .config import load_config

            load_config(tmp)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(path)
    return path


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #
@dataclass
class ProjectStatus:
    config_path: Path
    n_videos: int = 0
    n_prepared: int = 0
    # Movies whose detections are here but whose clips are not: the state a
    # project is in when detection ran on another machine.
    n_detected: int = 0
    n_clips: int = 0
    n_annotated: int = 0
    model_ok: bool = False
    video_dir_ok: bool = False
    suggester: str = ""

    @property
    def can_prepare(self) -> bool:
        return self.video_dir_ok and self.model_ok and self.n_videos > 0

    @property
    def can_annotate(self) -> bool:
        return self.n_clips > 0


def _probe(path, kind: str) -> bool:
    """One raw filesystem question, with "unreachable" folded into "no".

    ``Path.is_dir()`` swallows the ordinary not-found errors but not the ones a
    dropped network share raises, so a bare is_dir() can crash.
    """
    try:
        if kind == "dir":
            return path.is_dir()
        if kind == "file":
            return path.is_file()
        return path.exists()
    except OSError:
        return False


# A path that answered once, and when. On a disconnected share a single is_dir()
# can block for minutes before Windows gives up, and the window calls
# project_status on every refresh, so the verdict is cached and _PROBE_TIMEOUT
# keeps even the first probe off the UI thread.
_PROBE_CACHE: dict = {}
_PROBE_RUNNING: set = set()
_PROBE_LOCK = __import__("threading").Lock()
_PROBE_TTL = 30.0            # seconds a "not found" verdict is trusted for
_PROBE_TIMEOUT = 0.6         # seconds the caller is willing to wait at all
# 0.6s because a local path answers in microseconds and a dead share never
# answers at all; longer only helps a merely-slow share, which the cache picks
# up on the call after next.


def forget_path_probes() -> None:
    """Drop the cache, so a reconnected share is noticed immediately."""
    with _PROBE_LOCK:
        _PROBE_CACHE.clear()


def _exists(path, kind: str = "any", timeout: float = _PROBE_TIMEOUT) -> bool:
    """Does this path exist? Never blocks the caller for more than ``timeout``.

    A local path answers in microseconds; a share that has gone away answers in
    minutes, so it is probed on a throwaway thread and reported absent until the
    real answer lands in the cache.
    """
    import threading
    import time

    key = (str(path), kind)
    now = time.monotonic()
    with _PROBE_LOCK:
        cached = _PROBE_CACHE.get(key)
        # A positive verdict is kept until the TTL as well; re-probing a live
        # share on every repaint is what makes the window feel slow.
        if cached is not None and now - cached[1] < _PROBE_TTL:
            return cached[0]
        in_flight = key in _PROBE_RUNNING
        if not in_flight:
            _PROBE_RUNNING.add(key)

    if in_flight:
        # Another caller is already waiting on this exact question. Starting a
        # second thread would just queue a second multi-minute SMB timeout.
        return False

    result: list = []

    def work():
        answer = _probe(path, kind)
        with _PROBE_LOCK:
            _PROBE_CACHE[key] = (answer, time.monotonic())
            _PROBE_RUNNING.discard(key)
        result.append(answer)

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    worker.join(timeout)
    if result:
        return result[0]

    # Still blocked after the timeout: write a provisional "absent" so the next
    # call answers instantly. The live thread overwrites it with the truth, and
    # is a daemon so it cannot hold the app open.
    with _PROBE_LOCK:
        _PROBE_CACHE.setdefault(key, (False, time.monotonic()))
    return False


def project_status(cfg) -> ProjectStatus:
    """Summarise a project without opening a movie or a clip cache.

    Directory listings and small JSON reads only: this runs every time the
    workflow window is shown, where one ND2 open on a share is a visible stall.
    """
    status = ProjectStatus(config_path=cfg.path)
    video_dir = cfg.paths.video_dir
    # A path on a disconnected share does not answer "no", it RAISES: Windows
    # reports WinError 64 out of is_dir(), glob() and is_file() alike. The clip
    # cache is local, so annotating needs no share and that must not propagate.
    status.video_dir_ok = _exists(video_dir, "dir")
    if status.video_dir_ok:
        seen = set()
        try:
            for pattern in cfg.paths.patterns:
                seen.update(p for p in video_dir.glob(pattern) if p.is_file())
        except OSError:
            status.video_dir_ok = False
        status.n_videos = len(seen)
    from .annotate.clips import DETECT_FILE, DONE_FILE
    from .video.models import local_model_path

    status.model_ok = _exists(local_model_path(cfg), "file")

    work = cfg.paths.work_dir
    if _exists(work, "dir"):
        try:
            movie_dirs = sorted(work.iterdir())
        except OSError:
            movie_dirs = []
        for movie_dir in movie_dirs:
            if not _exists(movie_dir, "dir"):
                continue
            # Detected but not yet cut: the movie has tracks, from this machine or
            # a GPU box, and still needs its clips. It is deliberately NOT counted
            # as prepared - there is nothing to annotate yet.
            if _exists(movie_dir / DETECT_FILE):
                status.n_detected += 1
            done = movie_dir / DONE_FILE
            if not _exists(done):
                continue
            status.n_prepared += 1
            try:
                status.n_clips += int(json.loads(done.read_text(encoding="utf-8"))["n_clips"])
            except (OSError, KeyError, ValueError):
                pass

        report = work / "suggest_report.json"
        if _exists(report):
            try:
                reports = json.loads(report.read_text(encoding="utf-8"))
                live = [g for g, r in reports.items() if r.get("reliable")]
                status.suggester = (
                    "suggesting: " + ", ".join(live) if live
                    else "trained, staying silent"
                )
            except (OSError, ValueError):
                status.suggester = "unreadable report"

    csv = cfg.paths.csv
    if _exists(csv):
        try:
            with open(csv, "r", encoding="utf-8") as fh:
                status.n_annotated = max(0, sum(1 for _ in fh) - 1)
        except OSError:
            pass
    return status


# --------------------------------------------------------------------------- #
# recent list
# --------------------------------------------------------------------------- #
def _state_dir() -> Path:
    base = (
        os.environ.get("TRACKANNO_STATE_DIR")
        or os.environ.get("APPDATA")
        or os.environ.get("XDG_CONFIG_HOME")
        or str(Path.home() / ".config")
    )
    return Path(base) / "trackanno"


def recent_projects() -> list[Path]:
    """Most recently opened first. Paths that have vanished are dropped."""
    path = _state_dir() / RECENT_FILE
    if not path.exists():
        return []
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out, seen = [], set()
    for entry in entries:
        candidate = Path(str(entry))
        if candidate.exists() and candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
    return out[:MAX_RECENT]


def remember_project(config_path: str | Path) -> None:
    path = Path(config_path).resolve()
    entries = [str(path)] + [str(p) for p in recent_projects() if p != path]
    _write_recent(entries[:MAX_RECENT])


def forget_project(config_path: str | Path) -> None:
    path = Path(config_path).resolve()
    _write_recent([str(p) for p in recent_projects() if p != path])


def _write_recent(entries: list[str]) -> None:
    try:
        directory = _state_dir()
        directory.mkdir(parents=True, exist_ok=True)
        (directory / RECENT_FILE).write_text(json.dumps(entries, indent=2), encoding="utf-8")
    except OSError:
        pass  # a missing recent list is never worth an error dialog
