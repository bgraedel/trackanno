"""Where a long job sends its progress, and how it is told to stop.

A :class:`Reporter` is the seam that keeps ``print`` and ``tqdm`` out of the slow
jobs (``prepare``, ``suggest``) and lets the GUI drive them: the CLI passes
:class:`ConsoleReporter`, the GUI a :class:`CallbackReporter` whose callbacks
marshal onto the Qt thread. Cancellation lives here because it must be checked
wherever progress is emitted: a job that only checks between movies cannot be
stopped inside one.
"""

from __future__ import annotations

import sys
import threading
from typing import Callable, Iterable, Iterator

__all__ = ["Cancelled", "Reporter", "ConsoleReporter", "CallbackReporter", "NullReporter"]


class Cancelled(RuntimeError):
    """Raised inside a job when the caller has asked it to stop."""


class Reporter:
    """No-op base. Every hook is optional to override."""

    def log(self, message: str, level: str = "info") -> None:
        pass

    def overall(self, done: int, total: int, label: str = "") -> None:
        """Progress across the whole run - movies finished, say."""

    def stage(self, label: str, total: int | None = None) -> None:
        """Start a new sub-job with a known length."""

    def advance(self, n: int = 1) -> None:
        pass

    def cancelled(self) -> bool:
        return False

    def raise_if_cancelled(self) -> None:
        if self.cancelled():
            raise Cancelled("cancelled by the user")

    def track(self, iterable: Iterable, label: str | None = None,
              total: int | None = None) -> Iterator:
        """Iterate, advancing the current stage and honouring cancellation. The
        check lives in the same loop as the progress tick, so anything long
        enough to need a progress bar is also interruptible."""
        if label is not None:
            if total is None:
                try:
                    total = len(iterable)  # type: ignore[arg-type]
                except TypeError:
                    total = None
            self.stage(label, total)
        for item in iterable:
            self.raise_if_cancelled()
            yield item
            self.advance()


class NullReporter(Reporter):
    """Explicit "discard everything", for tests and library use."""


class ConsoleReporter(Reporter):
    """tqdm + stdout: what the CLI has always done."""

    def __init__(self, quiet: bool = False):
        self.quiet = quiet
        self._bar = None

    def log(self, message: str, level: str = "info") -> None:
        if self.quiet and level == "info":
            return
        stream = sys.stderr if level == "error" else sys.stdout
        prefix = "  ! " if level in ("warning", "error") else "  "
        print(f"{prefix}{message}", file=stream)

    def stage(self, label: str, total: int | None = None) -> None:
        self._close()
        if self.quiet:
            return
        try:
            from tqdm import tqdm

            self._bar = tqdm(total=total, desc=f"  {label}", unit="frame", leave=False)
        except ImportError:
            self._bar = None

    def advance(self, n: int = 1) -> None:
        if self._bar is not None:
            self._bar.update(n)

    def _close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def __del__(self):
        try:
            self._close()
        except Exception:
            pass


class CallbackReporter(Reporter):
    """Forwards to plain callables, and polls a threading.Event to cancel. Used
    by the GUI worker thread; the callbacks emit Qt signals, the only thread-safe
    way to touch widgets from off the UI thread."""

    def __init__(
        self,
        on_log: Callable[[str, str], None] | None = None,
        on_stage: Callable[[str, int | None], None] | None = None,
        on_progress: Callable[[int, int | None], None] | None = None,
        on_overall: Callable[[int, int, str], None] | None = None,
        stop: threading.Event | None = None,
        throttle: int = 1,
    ):
        self._on_log = on_log
        self._on_stage = on_stage
        self._on_progress = on_progress
        self._on_overall = on_overall
        self._stop = stop or threading.Event()
        # A signal per frame at 100 fps floods the event loop and stops the UI
        # repainting.
        self._throttle = max(1, throttle)
        self._done = 0
        self._total: int | None = None

    def log(self, message: str, level: str = "info") -> None:
        if self._on_log:
            self._on_log(message, level)

    def overall(self, done: int, total: int, label: str = "") -> None:
        if self._on_overall:
            self._on_overall(done, total, label)

    def stage(self, label: str, total: int | None = None) -> None:
        self._done, self._total = 0, total
        if self._on_stage:
            self._on_stage(label, total)
        if self._on_progress:
            self._on_progress(0, total)

    def advance(self, n: int = 1) -> None:
        self._done += n
        if self._on_progress and (
            self._done % self._throttle == 0 or self._done == self._total
        ):
            self._on_progress(self._done, self._total)

    def cancel(self) -> None:
        self._stop.set()

    def cancelled(self) -> bool:
        return self._stop.is_set()
