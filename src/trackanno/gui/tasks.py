"""Long jobs on a worker thread, and the panel that shows and cancels them.

The worker reports through signals only; it never touches a widget.
"""

from __future__ import annotations

import threading
import traceback
from typing import Callable

from qtpy.QtCore import QThread, Signal
from qtpy.QtGui import QColor, QFont, QTextCharFormat, QTextCursor
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

    def set_run_label(self, text: str) -> None:
        """Rename the run button, including the label restored after a run."""
        self._run_label = text
        if not self.busy:
            self.run_button.setText(text)

    def set_work(self, build_work: Callable | None) -> None:
        """``build_work()`` returns ``work(reporter)``, or None to refuse the run."""
        self._build_work = build_work

    def append(self, message: str, level: str = "info") -> None:
        """Append one message in its level's colour (error, warning, good or plain)."""
        palette = tokens()
        colour = {"error": palette["danger"], "warning": palette["warning"],
                  "good": palette["success"]}.get(level)
        document = self.log.document()
        cursor = QTextCursor(document)
        cursor.movePosition(QTextCursor.End)
        if not document.isEmpty():
            cursor.insertBlock()
        # Set, not merge, so colour does not carry over to the next line.
        fmt = QTextCharFormat()
        if colour:
            fmt.setForeground(QColor(colour))
        cursor.setCharFormat(fmt)
        cursor.insertText(message)
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
        # Cancelling is a warning, not an error.
        runner.was_cancelled.connect(
            lambda: self._on_done(False, "cancelled", "warning"))
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

    def _on_done(self, ok: bool, message: str, level: str | None = None) -> None:
        self.bar.setRange(0, 1)
        self.bar.setValue(1 if ok else 0)
        self.bar.setFormat(message)
        self.stage_label.setText(message)
        self.run_button.setText(self._run_label)
        self.run_button.setEnabled(True)
        self.append(message, level or ("good" if ok else "error"))
        self._runner = None
        self.finished.emit(ok)

    def shutdown(self, timeout: int = 5000) -> bool:
        """Stop a running job and wait; False if it is still running.

        A job that will not stop is reparented to the application to finish on its own.
        """
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
