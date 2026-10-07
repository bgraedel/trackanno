"""Step 4: the annotation panel inside the hub."""

from __future__ import annotations


from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QLabel, QStackedWidget, QVBoxLayout,
)

from ...annotate.clips import ClipStore
from ...annotate.store import AnnotationStore
from ..app import AnnotationWidget
from .base import Page


class AnnotatePage(Page):
    def __init__(self, hub, parent=None):
        super().__init__(hub, parent)
        self.panel: AnnotationWidget | None = None
        self._store = None
        self._annotations = None
        self._built_from = None

        self.placeholder = QLabel("")
        self.placeholder.setWordWrap(True)
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setProperty("role", "muted")

        self.stack = QStackedWidget()
        self.stack.addWidget(self.placeholder)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.stack)

    def blocker(self) -> str:
        if self.cfg is None or self.hub.status is None:
            return "no project open"
        if self.hub.cache_job_running:
            return (
                "Detection is running and is rewriting the clips. This page "
                "reopens when it finishes."
            )
        if not self.hub.status.work_known:
            return "the work folder is slow to answer; still counting the clips"
        if self.hub.status.n_clips == 0:
            return "no clips yet - run Detect & crop first"
        return ""

    def _fingerprint(self):
        """What the annotation panel is built from; it is rebuilt only when this
        moves, because a rebuild discards the clip on screen."""
        if self.cfg is None:
            return None
        ui = self.cfg.ui
        return (
            str(self.cfg.paths.work_dir), str(self.cfg.paths.csv),
            self.hub.status.n_clips if self.hub.status else 0,
            # Colours and keys included: the buttons are built from them.
            tuple((g.name, g.type, g.required,
                   tuple((o.name, o.key, o.color) for o in g.options))
                  for g in self.cfg.categories),
            tuple(sorted((k, tuple(v)) for k, v in self.cfg.keys.items())),
            tuple(self.cfg.keypoints.nodes), tuple(self.cfg.keypoints.edges),
            # Everything the panel or store reads at write time, ui.annotator too.
            ui.shuffle, ui.annotator, ui.auto_advance, ui.fps,
            self.cfg.clips.frames, self.cfg.detection.task,
            self.cfg.suggest.enabled,
        )

    def on_project(self) -> None:
        # Display-only settings first, BEFORE the fingerprint check: they are not
        # in the fingerprint (no rebuild, no lost clip), so push them in by hand.
        if self.panel is not None and self.cfg is not None:
            self.panel.apply_view_settings(self.cfg)

        fingerprint = self._fingerprint()
        if self.panel is not None and fingerprint == self._built_from:
            return  # nothing this panel depends on changed
        self._teardown()
        self._built_from = None
        blocker = self.blocker()
        if blocker:
            self.placeholder.setText(blocker)
            self.stack.setCurrentWidget(self.placeholder)
            return
        self.placeholder.setText("loading the clip cache...")
        self.stack.setCurrentWidget(self.placeholder)

    def refresh_gate(self) -> None:
        # A cache job started while this page is open: let go of the panel now.
        if self.hub.cache_job_running and self.panel is not None:
            self._teardown()
            self._built_from = None
            self.placeholder.setText(self.blocker())
            self.stack.setCurrentWidget(self.placeholder)

    def ensure_loaded(self) -> None:
        """Built on first visit, not on project open: constructing it reads the
        whole manifest and starts the prefetch thread."""
        if self.panel is not None or self.blocker():
            if self.blocker():
                self.placeholder.setText(self.blocker())
                self.stack.setCurrentWidget(self.placeholder)
            return
        try:
            self._store = ClipStore(self.cfg.paths.work_dir)
            self._annotations = AnnotationStore(self.cfg.paths.csv, self.cfg)
            self.panel = AnnotationWidget(self.cfg, self._store, self._annotations)
        except Exception as exc:
            self.placeholder.setText(f"could not open the clip cache:\n{exc}")
            self.stack.setCurrentWidget(self.placeholder)
            return
        self._built_from = self._fingerprint()
        self.panel.status_message.connect(self.status_message)
        # NOT refresh_status: that calls on_project on every page and tears this
        # panel down, once per clip. The panel counts the CURRENT FILTER, so the
        # project-wide total that gates later steps comes off the store.
        self.panel.counts_changed.connect(
            lambda done, total: self.hub.note_progress(len(self._annotations))
        )
        self.stack.addWidget(self.panel)
        self.stack.setCurrentWidget(self.panel)
        self.panel.setFocus()

    def _teardown(self) -> None:
        if self.panel is not None:
            self.panel.shutdown()
            self.stack.removeWidget(self.panel)
            self.panel.deleteLater()
            self.panel = None
        if self._store is not None:
            self._store.close()
        self._store = None
        self._annotations = None

    def shutdown(self) -> None:
        self._teardown()
