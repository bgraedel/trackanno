"""Long jobs on a worker thread, with a panel to watch and stop them.

Detection takes minutes to hours, so it cannot run on the UI thread.
:class:`TaskRunner` is a ``QThread`` that hands the job a
:class:`~trackanno.progress.CallbackReporter` whose callbacks emit signals;
never touch a widget from the worker thread, since a queued signal is the only
safe route. :class:`TaskPanel` renders those signals - stage, progress, an
overall bar across movies, a log - for every long-running page in the hub.
"""

from __future__ import annotations

import threading
import traceback
from typing import Callable

from qtpy.QtCore import QThread, Signal
from qtpy.QtGui import QFont, QTextCursor
from qtpy.QtWidgets import (
    QHBoxLayout, QLabel, QPlainTextEdit, QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

from ..progress import CallbackReporter, Cancelled
from .theme import tokens

__all__ = ["TaskRunner", "TaskPanel"]


class TaskRunner(QThread):
    """Runs ``work(reporter)`` off the UI thread and reports back as signals."""

    log_line = Signal(str, str)          # message, level
    stage_started = Signal(str, object)  # label, total or None
    progressed = Signal(int, object)     # done, total or None
    overall = Signal(int, int, str)      # done, total, label
    succeeded = Signal(object)
    failed = Signal(str)
    was_cancelled = Signal()

    def __init__(self, work: Callable, parent=None, throttle: int = 5):
        super().__init__(parent)
        self._work = work
        self._stop = threading.Event()
        self._throttle = throttle

    def cancel(self) -> None:
        self._stop.set()

    def run(self) -> None:  # noqa: D102 - QThread entry point
        reporter = CallbackReporter(
            on_log=lambda message, level: self.log_line.emit(message, level),
            on_stage=lambda label, total: self.stage_started.emit(label, total),
            on_progress=lambda done, total: self.progressed.emit(done, total),
            on_overall=lambda done, total, label: self.overall.emit(done, total, label),
            stop=self._stop,
            throttle=self._throttle,
        )
        try:
            result = self._work(reporter)
        except Cancelled:
            self.was_cancelled.emit()
        except BaseException as exc:  # noqa: BLE001 - the thread must not die silently
            self.log_line.emit(traceback.format_exc(), "error")
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.succeeded.emit(result)


class TaskPanel(QWidget):
    """Run / cancel button, two progress bars and a log."""

    started = Signal()
    finished = Signal(bool)  # succeeded?

    def __init__(self, run_label: str = "Run", parent=None):
        super().__init__(parent)
        self._runner: TaskRunner | None = None
        self._run_label = run_label
        self._build_work: Callable | None = None

        self.run_button = QPushButton(run_label)
        self.run_button.setProperty("accent", "true")
        self.run_button.clicked.connect(self._on_run_clicked)
        self.run_button.setMinimumWidth(150)

        self.stage_label = QLabel("idle")
        self.stage_label.setProperty("role", "muted")

        self.overall_bar = QProgressBar()
        self.overall_bar.setFormat("%v / %m")
        self.overall_bar.setVisible(False)

        self.bar = QProgressBar()
        self.bar.setTextVisible(True)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(4000)
        font = QFont("Consolas")
        font.setStyleHint(QFont.Monospace)
        font.setPointSize(9)
        self.log.setFont(font)
        self.log.setPlaceholderText("output appears here")

        controls = QHBoxLayout()
        controls.addWidget(self.run_button)
        controls.addWidget(self.stage_label, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(controls)
        layout.addWidget(self.overall_bar)
        layout.addWidget(self.bar)
        layout.addWidget(self.log, 1)

    # -- api ---------------------------------------------------------------- #
    @property
    def busy(self) -> bool:
        return self._runner is not None and self._runner.isRunning()

    def set_work(self, build_work: Callable | None) -> None:
        """``build_work()`` returns the ``work(reporter)`` callable, or None to
        refuse the run (after the page has explained why)."""
        self._build_work = build_work

    def append(self, message: str, level: str = "info") -> None:
        palette = tokens()
        colour = {"error": palette["danger"], "warning": palette["warning"],
                  "good": palette["success"]}.get(level)
        if colour:
            self.log.appendHtml(
                f'<span style="color:{colour};">{_escape(message)}</span>'
            )
        else:
            self.log.appendPlainText(message)
        self.log.moveCursor(QTextCursor.End)

    def start(self, work: Callable, throttle: int = 5) -> None:
        if self.busy:
            return
        self.log.clear()
        self.bar.setRange(0, 0)          # indeterminate until the first stage
        self.bar.setFormat("starting")
        self.stage_label.setText("starting")
        self.run_button.setText("Cancel")

        runner = TaskRunner(work, self, throttle=throttle)
        runner.log_line.connect(self.append)
        runner.stage_started.connect(self._on_stage)
        runner.progressed.connect(self._on_progress)
        runner.overall.connect(self._on_overall)
        runner.succeeded.connect(lambda _r: self._on_done(True, "finished"))
        runner.failed.connect(lambda message: self._on_done(False, message))
        runner.was_cancelled.connect(lambda: self._on_done(False, "cancelled"))
        self._runner = runner
        self.started.emit()
        runner.start()

    def cancel(self) -> None:
        if self._runner is not None:
            self._runner.cancel()
            self.stage_label.setText("stopping after the current frame...")
            self.run_button.setEnabled(False)

    # -- internals ---------------------------------------------------------- #
    def _on_run_clicked(self) -> None:
        if self.busy:
            self.cancel()
            return
        work = self._build_work() if self._build_work else None
        if work is not None:
            self.start(work)

    def _on_stage(self, label: str, total) -> None:
        self.stage_label.setText(label)
        if total:
            self.bar.setRange(0, int(total))
            self.bar.setValue(0)
            self.bar.setFormat(f"{label}  %v / %m")
        else:
            self.bar.setRange(0, 0)
            self.bar.setFormat(label)

    def _on_progress(self, done: int, total) -> None:
        if total:
            self.bar.setValue(int(done))

    def _on_overall(self, done: int, total: int, label: str) -> None:
        self.overall_bar.setVisible(True)
        self.overall_bar.setRange(0, max(1, int(total)))
        self.overall_bar.setValue(int(done))
        self.overall_bar.setFormat(f"movie %v / %m   {label}")

    def _on_done(self, ok: bool, message: str) -> None:
        self.bar.setRange(0, 1)
        self.bar.setValue(1 if ok else 0)
        self.bar.setFormat(message)
        self.stage_label.setText(message)
        self.run_button.setText(self._run_label)
        self.run_button.setEnabled(True)
        self.append(message, "good" if ok else "error")
        self._runner = None
        self.finished.emit(ok)

    def shutdown(self, timeout: int = 5000) -> bool:
        """Ask a running job to stop and wait. Returns False if it is still going.

        A job's prologue is not always interruptible - importing torch has no
        cancellation point - and destroying a live QThread aborts during
        teardown. If it will not stop, cut it loose: reparent to the application
        and let it delete itself when it finishes."""
        runner = self._runner
        if runner is None:
            return True
        runner.cancel()
        if runner.wait(timeout):
            return True
        from qtpy.QtWidgets import QApplication

        runner.setParent(None)
        app = QApplication.instance()
        if app is not None:
            runner.setParent(app)
        runner.finished.connect(runner.deleteLater)
        self._runner = None
        return False


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        .replace("\n", "<br>")
    )
