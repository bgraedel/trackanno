"""What every hub page shares: the Page base and its background jobs."""

from __future__ import annotations

import threading
import time
import traceback
from typing import Callable

from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import (
    QLabel, QWidget,
)

from ...progress import CallbackReporter, Cancelled
from ..theme import (
    Heading,
)


PAGES = ["Project", "Configure", "Detect & crop", "Annotate", "Model",
         "Review", "Export"]

PAGE_HINTS = [
    "Pick the folder to work in",
    "Point it at your movies and choose your labels",
    "Find the animals and cut them into clips",
    "Watch each clip and press one key",
    "Learn from your labels and suggest the next one",
    "Check where the model put each boundary",
    "Put the labels back onto the original movies",
]


def _heading(text: str, subtitle: str = "", help: str = "") -> QWidget:
    """A page title, a one-line subtitle, and a `?` button for longer help."""
    return Heading(text, subtitle, help=help)


def _muted(label: QLabel) -> QLabel:
    label.setProperty("role", "muted")
    return label


class _Mailbox(QObject):
    """Runs callables posted from worker threads on the UI thread."""

    post = Signal(object)

    def __init__(self):
        super().__init__()
        self.post.connect(self._run)

    def _run(self, call) -> None:
        call()


_MAILBOX: _Mailbox | None = None


def _mailbox() -> _Mailbox:
    """The one mailbox, created on the UI thread and never deleted."""
    global _MAILBOX
    if _MAILBOX is None:
        _MAILBOX = _Mailbox()
    return _MAILBOX


class _Job:
    def __init__(self, name: str, project, dropped: Callable | None = None):
        self.name, self.project = name, project
        self.dropped = dropped
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None


class Jobs:
    """Background jobs tagged with their project; results land on the UI thread.

    Stale or superseded results are dropped; ``dropped`` runs on cancel or project change.
    """

    def __init__(self, project: Callable[[], object]):
        self._project = project
        self._live: dict[str, _Job] = {}
        self._closed = False

    def run(self, name: str, work: Callable, done: Callable | None = None,
            failed: Callable | None = None, dropped: Callable | None = None) -> _Job | None:
        """Start ``work(reporter)`` on a daemon thread; supersedes ``name``."""
        if self._closed:
            return None
        previous = self._live.pop(name, None)
        if previous is not None:
            previous.stop.set()
        job = _Job(name, self._project(), dropped)
        reporter = CallbackReporter(stop=job.stop)
        mailbox = _mailbox()

        def body():
            try:
                outcome = (True, work(reporter))
            except Cancelled:
                outcome = None
            except BaseException as exc:  # noqa: BLE001 - reported, never lost
                traceback.print_exc()
                outcome = (False, f"{type(exc).__name__}: {exc}")
            try:
                mailbox.post.emit(lambda: self._land(job, outcome, done, failed))
            except RuntimeError:          # the application is gone
                pass

        job.thread = threading.Thread(target=body, name=f"trackanno-{name}",
                                      daemon=True)
        self._live[name] = job
        job.thread.start()
        return job

    def _land(self, job: _Job, outcome, done, failed) -> None:
        if self._live.get(job.name) is not job:
            return                        # superseded
        del self._live[job.name]
        if self._closed:
            return
        if outcome is None or job.project != self._project():
            self._drop(job)
            return
        ok, value = outcome
        callback = done if ok else failed
        if callback is None:
            return
        try:
            callback(value)
        except RuntimeError as exc:       # a widget deleted under the job
            print(f"[trackanno] {job.name} result dropped: {exc}")

    def _drop(self, job: _Job) -> None:
        if job.dropped is None or self._closed:
            return
        try:
            job.dropped()
        except RuntimeError as exc:       # a widget deleted under the job
            print(f"[trackanno] {job.name} drop not reported: {exc}")

    def running(self, name: str) -> bool:
        return name in self._live

    def cancel(self, name: str | None = None) -> None:
        """Ask one job, or every job, to stop. Its result is dropped."""
        for key in [name] if name is not None else list(self._live):
            job = self._live.pop(key, None)
            if job is not None:
                job.stop.set()
                self._drop(job)

    def shutdown(self, timeout: float = 5.0) -> None:
        """Cancel everything and wait up to ``timeout`` seconds in total."""
        self._closed = True
        jobs = list(self._live.values())
        self.cancel()
        deadline = time.monotonic() + timeout
        for job in jobs:
            if job.thread is not None:
                job.thread.join(max(0.0, deadline - time.monotonic()))


class Page(QWidget):
    """Common contract: a page is told when the project changes."""

    status_message = Signal(str, int)

    def __init__(self, hub, parent=None):
        super().__init__(parent)
        self.hub = hub
        self._jobs = Jobs(lambda: getattr(self.hub, "project_gen", 0))
        # Set when the config changed while this page was not on screen.
        self.rebuild_pending = False

    @property
    def cfg(self):
        return self.hub.cfg

    def run_bg(self, name: str, work: Callable, done: Callable | None = None,
               failed: Callable | None = None, dropped: Callable | None = None):
        """Run ``work(reporter)`` off the UI thread; see :class:`Jobs`."""
        return self._jobs.run(name, work, done, failed, dropped)

    def bg_running(self, name: str) -> bool:
        return self._jobs.running(name)

    def on_project(self) -> None:
        pass

    def blocker(self) -> str:
        """Why this page cannot be used yet, or "" if it can."""
        return "" if self.cfg is not None else "no project open"

    def refresh_gate(self) -> None:
        """Re-check the blocker cheaply after each annotation (no file reads, no rebuild)."""

    def shutdown(self) -> None:
        pass
