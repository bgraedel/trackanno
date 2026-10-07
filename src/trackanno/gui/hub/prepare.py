"""Step 3: detect, track and cut clips."""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem,
    QMessageBox, QPushButton, QSpinBox, QSplitter,
    QVBoxLayout, QWidget,
)

from ...project import (
    _exists, forget_path_probes,
)
from ..theme import (
    set_role,
)
from ..tasks import TaskPanel
from .base import Page, _heading


_PREPARE_STATES = {
    "prepared": "[prepared]",
    "detected": "[detected]",
    "stale": "[ stale  ]",
    "new": "[  new   ]",
}


class PreparePage(Page):
    def __init__(self, hub, parent=None):
        super().__init__(hub, parent)
        self._scanned_for: Path | None = None
        # Per-movie state; _filling stops the list refill re-entering the toggle handler.
        self._states: dict[str, str] = {}
        self._stale: dict[str, list[str]] = {}
        self._filling = False

        self.movies = QListWidget()
        self.movies.setSelectionMode(QListWidget.NoSelection)
        self.movies.setAlternatingRowColors(True)
        self.movies.itemChanged.connect(self._on_item_toggled)

        scan_btn = QPushButton("Scan video folder")
        scan_btn.clicked.connect(self.scan)
        self.select_all = QPushButton("select all")
        self.select_all.clicked.connect(lambda: self._set_all(True))
        self.select_none = QPushButton("select none")
        self.select_none.clicked.connect(lambda: self._set_all(False))
        self.scan_state = QLabel("not scanned yet")
        self.scan_state.setProperty("role", "muted")
        self.scan_state.setWordWrap(True)

        self.fix_button = QPushButton("Fix this on the Configure page")
        self.fix_button.setProperty("accent", "true")
        self.fix_button.setVisible(False)
        self.fix_button.clicked.connect(lambda: self.hub.go_to(1))

        movie_bar = QHBoxLayout()
        movie_bar.addWidget(scan_btn)
        movie_bar.addWidget(self.select_all)
        movie_bar.addWidget(self.select_none)
        movie_bar.addWidget(self.fix_button)
        movie_bar.addStretch(1)

        self.force = QCheckBox("redo movies that are already prepared")
        self.limit_frames = QSpinBox()
        self.limit_frames.setRange(0, 1_000_000)
        self.limit_frames.setSpecialValueText("all frames")
        self.limit_frames.setToolTip(
            "Stop after this many frames per movie, for a quick test run."
        )
        options = QFormLayout()
        options.addRow(self.force)
        options.addRow("frame limit", self.limit_frames)
        options_box = QGroupBox("Run options")
        options_box.setLayout(options)

        self.tasks = TaskPanel("Run detection")
        self.tasks.set_work(self._build_work)
        self.tasks.started.connect(lambda: self.hub.set_cache_job(True))
        self.tasks.finished.connect(self._on_run_finished)

        movies_box = QGroupBox("Movies")
        movies_layout = QVBoxLayout(movies_box)
        movies_layout.addLayout(movie_bar)
        movies_layout.addWidget(self.scan_state)
        movies_layout.addWidget(self.movies)

        splitter = QSplitter(Qt.Vertical)
        top = QWidget()
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.addWidget(movies_box, 2)
        top_layout.addWidget(options_box, 1)
        splitter.addWidget(top)
        splitter.addWidget(self.tasks)
        splitter.setSizes([300, 380])

        layout = QVBoxLayout(self)
        layout.addWidget(_heading(
            "Detect & crop",
            "Find the animals and cut them into clips.",
            help="""
<p>Detects the animals in every frame, tracks them, and cuts one clip per track
window.</p>

<p>The <b>slow, GPU-bound step</b>. It is <b>resumable</b>: finished movies are
skipped on the next run.</p>

<p>Only detection needs a GPU and the model. A movie marked <b>[detected]</b>
already has its tracks (from here or a copied <code>work/</code> folder) and only
needs its clips cut.</p>

<p>Cancel stops at the next frame.</p>
""",
        ))
        layout.addWidget(splitter, 1)

    def blocker(self) -> str:
        if self.cfg is None or self.hub.status is None:
            return "Open or create a project first."
        status = self.hub.status
        if not status.video_dir_ok:
            return (
                f"Movie folder not found ({self.cfg.paths.video_dir}). "
                "Set \"video folder\" on the Configure page."
            )
        # Cutting clips needs no model, so detections from elsewhere suffice.
        if not status.model_ok and not status.n_detected:
            return (
                f"Detection model not found ({self.cfg.paths.model}). "
                "Set \"detection model\" on the Configure page."
            )
        if status.n_videos == 0:
            return (
                f"No files matching {cfg_patterns(self.cfg)} in {self.cfg.paths.video_dir}. "
                "Change the folder or \"file patterns\" on the Configure page."
            )
        return ""

    def on_project(self) -> None:
        blocker = self.blocker()
        self.tasks.run_button.setEnabled(self.tasks.busy or not blocker)
        self.fix_button.setVisible(bool(blocker) and self.cfg is not None)
        if blocker or self.cfg is None:
            self.movies.clear()
            self.scan_state.setText(blocker)
            set_role(self.scan_state, "warning")
            self._scanned_for = None
            return
        set_role(self.scan_state, "muted")
        if self._scanned_for != self.cfg.path:
            self.movies.clear()
            self.scan_state.setText("not scanned yet: press Scan to list the movies")
            self._scanned_for = None
            return
        # Same project: keep the list, rescanning a share is slow.
        self._mark_prepared()

    # -- scanning ----------------------------------------------------------- #
    def _on_run_finished(self, ok: bool) -> None:
        self.hub.set_cache_job(False)
        # Drop cached "missing" verdicts so new markers show at once.
        forget_path_probes()
        self.hub.reload_project()

    def scan(self) -> None:
        blocker = self.blocker()
        if blocker:
            QMessageBox.information(self, "Cannot scan", blocker)
            return
        if self.bg_running("scan"):
            return
        cfg = self.cfg
        self.scan_state.setText("scanning... (ND2 files on a share can take a moment)")

        def work(reporter):
            from ...video.movies import discover_movies

            out = []
            for movie in discover_movies(cfg):
                reporter.raise_if_cancelled()
                try:
                    out.append(movie.describe())
                finally:
                    movie.close()
            return out

        self.run_bg("scan", work, self._on_scanned,
                    lambda message: self.scan_state.setText(f"scan failed: {message}"))

    def _on_scanned(self, movies) -> None:
        if self.cfg is None:
            return
        self.movies.clear()
        for movie in movies or []:
            item = QListWidgetItem()
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setData(Qt.UserRole, movie["movie_id"])
            item.setData(Qt.UserRole + 1, movie)
            self.movies.addItem(item)
        self._scanned_for = self.cfg.path if self.cfg else None
        self._mark_prepared(select_new=True)

    def _mark_prepared(self, select_new: bool = False) -> None:
        states = self._states = self._movie_states()
        n_todo = 0
        self._filling = True
        try:
            for i in range(self.movies.count()):
                item = self.movies.item(i)
                movie = item.data(Qt.UserRole + 1)
                if not movie:
                    continue
                state = states.get(movie["movie_id"], "new")
                item.setText(
                    f"{_PREPARE_STATES[state]}  {movie['movie_id']}"
                    f"      {movie['n_frames']} frames  {movie['width']}x{movie['height']}"
                    f"  @ {movie['fps']:g} fps"
                )
                item.setToolTip("\n".join(self._stale.get(movie["movie_id"], [])))
                # Tick new and detected movies; stale ones only rebuild under redo.
                if select_new:
                    item.setCheckState(
                        Qt.Unchecked if state in ("prepared", "stale") else Qt.Checked
                    )
                if state not in ("prepared", "stale"):
                    n_todo += 1
        finally:
            self._filling = False
        self._update_run_label(states)
        if self.movies.count():
            n_detected = sum(1 for s in states.values() if s == "detected")
            n_stale = sum(1 for s in states.values() if s == "stale")
            detail = f", {n_detected} detected elsewhere" if n_detected else ""
            if n_stale:
                detail += (f", {n_stale} stale (settings changed; tick 'redo movies "
                           "that are already prepared' to rebuild)")
            self.scan_state.setText(
                f"{self.movies.count()} movie(s), {n_todo} not yet prepared{detail}"
            )

    def _on_item_toggled(self, _item) -> None:
        """Update the run button label after a tick."""
        if self._filling:
            return
        self._update_run_label(self._states)

    def _only_cutting(self, states: dict | None = None) -> bool:
        """Whether every ticked movie only needs clips cut (drives label and stage)."""
        states = self._states if states is None else states
        pending = [
            self.movies.item(i).data(Qt.UserRole)
            for i in range(self.movies.count())
            if self.movies.item(i).checkState() == Qt.Checked
        ]

        def cut_only(movie_id) -> bool:
            state = states.get(movie_id, "new")
            if state == "stale":
                # Only clip settings changed: re-cut, do not detect.
                return all(r.startswith("clip") for r in self._stale.get(movie_id, [""]))
            return state == "detected"

        return bool(pending) and all(cut_only(m) for m in pending)

    def _update_run_label(self, states: dict) -> None:
        """Label the run button "Cut clips" or "Run detection"."""
        self.tasks.set_run_label(
            "Cut clips" if self._only_cutting(states) else "Run detection"
        )

    def _movie_states(self) -> dict[str, str]:
        """movie_id -> "prepared" | "detected" (tracks, no clips) | "stale" | "new"."""
        from ...annotate.clips import DETECT_FILE, DONE_FILE
        from ...detect.prepare import stale_reasons

        self._stale = {}

        if self.cfg is None:
            return {}
        work = self.cfg.paths.work_dir
        # Timeout-guarded: a share can block per directory.
        if not _exists(work, "dir"):
            return {}
        try:
            entries = sorted(work.iterdir())
        except OSError:
            return {}
        states: dict[str, str] = {}
        for d in entries:
            if not _exists(d, "dir"):
                continue
            if _exists(d / DONE_FILE):
                states[d.name] = "prepared"
            elif _exists(d / DETECT_FILE):
                states[d.name] = "detected"
            else:
                continue
            reasons = stale_reasons(self.cfg, d.name, check_model=False)
            if reasons:
                states[d.name] = "stale"
                self._stale[d.name] = reasons
        return states

    def _set_all(self, checked: bool) -> None:
        for i in range(self.movies.count()):
            self.movies.item(i).setCheckState(Qt.Checked if checked else Qt.Unchecked)

    def _selected(self) -> list[str] | None:
        picked = [
            self.movies.item(i).data(Qt.UserRole)
            for i in range(self.movies.count())
            if self.movies.item(i).checkState() == Qt.Checked
        ]
        return picked or None

    # -- running ------------------------------------------------------------ #
    def _build_work(self):
        blocker = self.blocker()
        if blocker:
            QMessageBox.information(self, "Cannot run", blocker)
            return None
        if self.movies.count() and not self._selected():
            QMessageBox.information(
                self, "Nothing selected", "Tick at least one movie, or press Scan."
            )
            return None

        # Windows will not overwrite a file this window has memory-mapped.
        self.hub.release_caches()

        from copy import copy

        from ...detect.prepare import prepare_all

        cfg = copy(self.cfg)
        movie_ids = self._selected()
        force = self.force.isChecked()
        limit = self.limit_frames.value() or None
        # A cut-only run gets the clip stage so it can never start detection.
        stage = "clips" if self._only_cutting() else "all"

        def work(reporter):
            return prepare_all(
                cfg, movie_ids, force, limit, reporter=reporter, stage=stage
            )

        return work

    def shutdown(self) -> None:
        self.tasks.shutdown()


def cfg_patterns(cfg) -> str:
    return ", ".join(cfg.paths.patterns)
