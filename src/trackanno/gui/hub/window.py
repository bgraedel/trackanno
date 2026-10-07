"""The hub window, its entry point and its exception hook."""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem,
    QMainWindow, QMessageBox, QPushButton, QStackedWidget, QVBoxLayout, QWidget,
)

from ...config import load_config
from ...project import (
    CONFIG_NAME, config_path_for, project_status, recent_projects,
    remember_project,
)
from ..theme import (
    app_icon, apply_theme, tokens,
)
from .annotate import AnnotatePage
from .base import PAGE_HINTS, PAGES, Jobs, Page
from .configure import ConfigPage
from .export import ExportPage
from .model import ModelPage
from .prepare import PreparePage
from .project import ProjectPage
from .review import ReviewPage


class Hub(QMainWindow):
    def __init__(self, config_path: str | Path | None = None):
        super().__init__()
        self.setWindowTitle("trackanno")
        self.setWindowIcon(app_icon())
        self.resize(1280, 860)
        self.cfg = None
        # Last config.yaml opened, even if it failed to load.
        self.config_path: Path | None = None
        self.status = None
        # True while a job rewrites the clip cache; keeps Annotate from re-mapping it.
        self.cache_job_running = False
        # Bumped whenever a different project (or none) is opened; tags jobs.
        self.project_gen = 0
        self._open_path: Path | None = None
        self._jobs = Jobs(lambda: self.project_gen)

        self.sidebar = QListWidget()
        self.sidebar.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.sidebar.setFocusPolicy(Qt.NoFocus)
        for name in PAGES:
            self.sidebar.addItem(QListWidgetItem(name))
        self.sidebar.setStyleSheet(
            f"QListWidget{{background:{tokens()['surface']};padding:8px;"
            f"border:1px solid {tokens()['border']};border-radius:8px;}}"
            "QListWidget::item{padding:10px 8px;margin:1px 0;}"
        )

        self.pages: list[Page] = [
            ProjectPage(self), ConfigPage(self), PreparePage(self),
            AnnotatePage(self), ModelPage(self),
            ReviewPage(self), ExportPage(self),
        ]
        self.stack = QStackedWidget()
        for page in self.pages:
            page.status_message.connect(
                lambda text, timeout=0: self.statusBar().showMessage(text, timeout)
            )
            self.stack.addWidget(page)

        # The "Next:" line under every page.
        self.hint = QLabel("")
        self.hint.setWordWrap(True)
        self.hint.setTextFormat(Qt.RichText)
        self.hint_button = QPushButton("")
        self.hint_button.setProperty("accent", "true")
        self.hint_button.setVisible(False)
        self.hint_button.clicked.connect(lambda: self.go_to(self._hint_page))
        self._hint_page = 0

        hint_bar = self._hint_bar = QWidget()
        hint_bar.setObjectName("hintBar")
        hint_bar.setStyleSheet(
            f"#hintBar{{background:{tokens()['surface']};"
            f"border:1px solid {tokens()['border']};border-radius:8px;}}"
        )
        hint_layout = QHBoxLayout(hint_bar)
        hint_layout.setContentsMargins(10, 6, 10, 6)
        hint_layout.addWidget(self.hint, 1)
        hint_layout.addWidget(self.hint_button)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.stack, 1)
        right_layout.addWidget(hint_bar)

        self.nav = QWidget()
        self.nav.setFixedWidth(200)
        nav_layout = QVBoxLayout(self.nav)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(8)
        nav_layout.addWidget(self.sidebar, 1)

        body = QWidget()
        layout = QHBoxLayout(body)
        layout.setContentsMargins(10, 10, 10, 6)
        layout.setSpacing(10)
        layout.addWidget(self.nav)
        layout.addWidget(right, 1)
        self.setCentralWidget(body)

        self.sidebar.currentRowChanged.connect(self._on_page_changed)
        self.sidebar.setCurrentRow(0)
        self.statusBar().showMessage("no project open")

        opened = config_path or (recent_projects()[0] if recent_projects() else None)
        if not (opened and self.open_project(opened, quiet=True)):
            # Nothing opened: paint the empty state.
            self.refresh_status(quiet=True)

    # -- project ------------------------------------------------------------ #
    def open_project(self, target: str | Path, quiet: bool = False) -> bool:
        path = config_path_for(target)
        if not path.exists():
            if not quiet:
                QMessageBox.warning(self, "Not found", f"No {CONFIG_NAME} at {path}")
            return False
        self.config_path = path
        try:
            self.cfg = load_config(path)
        except Exception as exc:  # a broken config must not take the window down
            self.cfg = None
            self._switch_to(None)
            # Tell every page, so no panel keeps writing to the old project.
            self.status = None
            self.setWindowTitle("trackanno")
            self.refresh_status(quiet=True)
            if not quiet:
                QMessageBox.warning(self, "Could not open", f"{path}\n\n{exc}")
            self.statusBar().showMessage(f"could not open {path}: {exc}")
            return False
        self._switch_to(self.cfg.path)
        remember_project(path)
        self.setWindowTitle(f"trackanno - {path.parent.name}[*]")
        self.refresh_status()
        return True

    def _switch_to(self, path: Path | None) -> None:
        """Start a new project generation when ``path`` is not the open one."""
        if path == self._open_path:
            return
        self._open_path = path
        self.project_gen += 1
        self._jobs.cancel()
        for page in self.pages:
            page._jobs.cancel()

    def page(self, kind):
        """The page of a given class."""
        return next((p for p in self.pages if isinstance(p, kind)), None)

    def reload_project(self) -> None:
        """Re-read the project and refresh every page now."""
        if self.cfg is not None:
            self.open_project(self.cfg.path, quiet=True)

    def config_saved(self) -> None:
        """Re-read the saved config; pages rebuild when next shown."""
        try:
            self.cfg = load_config(self.cfg.path)
        except Exception:
            self.reload_project()             # reports the failure
            return
        self.status = project_status(self.cfg)
        if not self.status.work_known:
            self._count_work(self.status)
        for page in self.pages:
            page.rebuild_pending = True
        current = self.stack.currentWidget()
        if isinstance(current, Page):
            self._rebuild(current)
        for page in self.pages:
            page.refresh_gate()
        self._paint_sidebar()
        self._paint_hint()

    def _rebuild(self, page: Page) -> None:
        """Run a page's on_project, reporting rather than raising its errors."""
        page.rebuild_pending = False
        try:
            page.on_project()
        except Exception as exc:
            print(f"[trackanno] {type(page).__name__} failed to refresh: {exc}")
            self.statusBar().showMessage(f"{type(page).__name__}: {exc}", 10000)

    def release_caches(self) -> None:
        """Release every clip-cache memory map (needed before a rewrite on Windows)."""
        for page in self.pages:
            if isinstance(page, AnnotatePage):
                page._teardown()
                page._built_from = None

    def refresh_status(self, quiet: bool = False) -> None:
        if self.cfg is not None:
            self.status = project_status(self.cfg)
            if not self.status.work_known:
                self._count_work(self.status)
        self._apply_status(quiet)

    def _count_work(self, status) -> None:
        """Finish a work-dir scan that timed out, off the UI thread."""
        from ...project import scan_work

        cfg = self.cfg

        def done(scanned):
            if self.status is not status:
                return
            for name, value in scanned.items():
                setattr(status, name, value)
            status.work_known = True
            self._apply_status(quiet=True)

        self._jobs.run("work-scan", lambda reporter: scan_work(cfg), done)

    def _apply_status(self, quiet: bool = False) -> None:
        """Tell every page about the current cfg and status."""
        for page in self.pages:
            self._rebuild(page)
        self._paint_sidebar()
        self._paint_hint()
        if self.cfg is not None and not quiet:
            self.statusBar().showMessage(
                f"{self.cfg.path}   |   {self.status.n_clips} clips, "
                f"{self.status.n_annotated} annotated"
            )
        # Rebuild the annotation panel if it is on screen.
        current = self.pages[self.sidebar.currentRow()] if self.sidebar.currentRow() >= 0 else None
        if isinstance(current, AnnotatePage):
            current.ensure_loaded()

    def _paint_hint(self) -> None:
        advice, page, label = self.next_step()
        self.hint.setText(f"<b>Next:</b> {advice}")
        self._hint_page = page
        self.hint_button.setText(label)
        self.hint_button.setVisible(bool(label) and self.sidebar.currentRow() != page)

    def set_cache_job(self, running: bool) -> None:
        """Flag that something is rewriting the clip cache."""
        self.cache_job_running = bool(running)
        if running:
            self.release_caches()
        for page in self.pages:
            page.refresh_gate()
        self._paint_sidebar()
        if not running:
            row = self.sidebar.currentRow()
            current = self.pages[row] if 0 <= row < len(self.pages) else None
            if isinstance(current, AnnotatePage):
                current.ensure_loaded()

    def note_progress(self, n_annotated: int) -> None:
        """Update the label count and page gates without rebuilding any page."""
        if self.status is not None:
            self.status.n_annotated = n_annotated
        for page in self.pages:
            page.refresh_gate()
        self._paint_sidebar()
        self._paint_hint()

    def next_step(self) -> tuple[str, int, str]:
        """(advice, page index, button label) for what this project needs next."""
        if self.cfg is None:
            return ("Open or create a project to begin.", 0, "")
        status = self.status
        if not status.work_known:
            return ("The work folder is slow to answer; still counting.", 0, "")
        # Missing paths only block the next detection run.
        if status.n_clips == 0 and (not status.video_dir_ok or not status.model_ok):
            missing = []
            if not status.video_dir_ok:
                missing.append(f"video folder <code>{self.cfg.paths.video_dir}</code>")
            if not status.model_ok:
                missing.append(f"model <code>{self.cfg.paths.model}</code>")
            return (
                "Not found: " + " and ".join(missing) + ". Set it on the Configure page.",
                1, "Go to Configure",
            )
        if status.n_clips == 0:
            return (
                f"{status.videos_text} movie(s) found, none prepared. Run detection.",
                2, "Go to Detect & crop",
            )
        if status.n_annotated < status.n_clips:
            note = (
                "" if status.video_dir_ok and status.model_ok
                else "<br><i>(video folder or model missing: no more movies can be prepared)</i>"
            )
            return (
                f"{status.n_annotated} of {status.n_clips} clips labelled. "
                + note,
                3, "Go to Annotate",
            )
        target = PAGES.index("Model")
        return (
            "Every clip is labelled. Train the model or export overlay movies.",
            target, f"Go to {PAGES[target]}",
        )

    # -- navigation --------------------------------------------------------- #
    def go_to(self, index: int) -> None:
        self.sidebar.setCurrentRow(index)

    def _on_page_changed(self, index: int) -> None:
        if index < 0:
            return
        previous = self.stack.currentWidget()
        if isinstance(previous, ConfigPage) and not previous.confirm_leave():
            self.sidebar.blockSignals(True)
            self.sidebar.setCurrentRow(self.pages.index(previous))
            self.sidebar.blockSignals(False)
            return
        page = self.pages[index]
        self.stack.setCurrentWidget(page)
        if page.rebuild_pending:
            self._rebuild(page)
        page.refresh_gate()
        if isinstance(page, AnnotatePage):
            page.ensure_loaded()
            if page.panel is not None:
                page.panel.setFocus()
        elif isinstance(page, ExportPage) and page._list_stale:
            page.reload_movies()
        elif isinstance(page, ReviewPage):
            page.ensure_loaded()
        elif isinstance(page, ModelPage):
            page.on_shown()
        self._paint_hint()
        blocker = page.blocker()
        if blocker:
            self.statusBar().showMessage(blocker)

    def _step_states(self) -> list[tuple[str, str]]:
        """(glyph, note) per sidebar step."""
        if self.cfg is None:
            return [(">", "start here")] + [("", "needs a project")] * (len(PAGES) - 1)
        s = self.status
        paths_ok = s.video_dir_ok and s.model_ok
        review = self.page(ReviewPage)
        predicted = review is not None and review._has_predictions
        return [
            ("*", ""),
            ("*" if paths_ok else "!", "" if paths_ok else "something is missing"),
            ("?", "still counting") if not s.work_known else
            ("*" if s.n_clips else ">", f"{s.n_prepared} done" if s.n_clips else "not run yet"),
            ("*" if s.n_clips and s.n_annotated >= s.n_clips else
             (">" if s.n_clips else ""),
             f"{s.n_annotated} / {s.n_clips}" if s.n_clips else ""),
            ("*" if s.suggester else "", s.suggester.split(":")[0] if s.suggester else ""),
            ("*" if predicted else "", "" if predicted else "nothing predicted yet"),
            ("*" if s.n_annotated else "", ""),
        ]

    def _paint_sidebar(self) -> None:
        states = self._step_states()
        for i, name in enumerate(PAGES):
            item = self.sidebar.item(i)
            glyph, note = states[i]
            item.setText(f" {i + 1}. {name}" + (f"   {glyph}" if glyph else ""))
            item.setToolTip(
                f"{PAGE_HINTS[i]}" + (f"\n{note}" if note else "")
            )
            blocked = self.cfg is None and i != 0
            item.setFlags(
                (item.flags() & ~Qt.ItemIsEnabled) if blocked
                else (item.flags() | Qt.ItemIsEnabled)
            )

    # -- close -------------------------------------------------------------- #
    def closeEvent(self, event):  # noqa: N802 (Qt naming)
        for page in self.pages:
            if isinstance(page, ConfigPage) and not page.confirm_leave():
                event.ignore()
                return
        for page in self.pages:
            page.shutdown()
        # One shared 5 s budget for all jobs.
        deadline = time.monotonic() + 5.0
        for jobs in [self._jobs] + [page._jobs for page in self.pages]:
            jobs.shutdown(max(0.0, deadline - time.monotonic()))
        super().closeEvent(event)


def install_excepthook(window) -> None:
    """Log uncaught exceptions to the terminal and status bar instead of aborting."""
    def hook(kind, value, tb):
        if issubclass(kind, KeyboardInterrupt):
            sys.__excepthook__(kind, value, tb)
            return
        traceback.print_exception(kind, value, tb, file=sys.stderr)
        try:
            window.statusBar().showMessage(
                f"error: {kind.__name__}: {value} (details in the terminal)", 15000)
        except RuntimeError:  # the window is already gone
            pass

    sys.excepthook = hook


def _preferred_theme(project) -> str:
    """Read ui.theme from the project about to be opened, if there is one."""
    candidates = [project] if project else []
    candidates += recent_projects()[:1]
    for candidate in candidates:
        try:
            return load_config(config_path_for(candidate)).ui.theme
        except Exception:
            continue
    return "dark"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="trackanno", description="The whole annotation pipeline in one window."
    )
    parser.add_argument("project", nargs="?", default=None,
                        help="a project folder or its config.yaml (default: the last one)")
    args = parser.parse_args(argv)

    app = QApplication.instance() or QApplication(sys.argv[:1])
    # Theme before any widget exists.
    apply_theme(app, _preferred_theme(args.project))
    window = Hub(args.project)
    install_excepthook(window)
    window.show()
    return app.exec_() if hasattr(app, "exec_") else app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
