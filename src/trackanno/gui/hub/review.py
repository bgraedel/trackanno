"""Step 6: check predicted bouts against the footage."""

from __future__ import annotations


from qtpy.QtWidgets import (
    QLabel, QVBoxLayout,
)

from ..review import REVIEW_HELP, BoutReviewPanel
from .base import Page, _heading, _muted


class ReviewPage(Page):
    """Hosts the review panel, built on first visit and released when hidden."""

    def __init__(self, hub, parent=None):
        super().__init__(hub, parent)
        self.panel: BoutReviewPanel | None = None
        self._has_predictions = False
        self._project: str | None = None
        # Predictions changed while the panel was off screen.
        self._stale = False

        self.note = _muted(QLabel(""))
        self.note.setWordWrap(True)

        self.host = QVBoxLayout()
        self.host.setContentsMargins(0, 0, 0, 0)

        layout = QVBoxLayout(self)
        layout.addWidget(_heading(
            "Review",
            "Check where the model put each boundary, against the footage.",
            # Shown before the panel exists, too.
            help=REVIEW_HELP,
        ))
        layout.addWidget(self.note)
        layout.addLayout(self.host, 1)

    def blocker(self) -> str:
        if self.cfg is None:
            return "no project open"
        if not self._has_predictions:
            return ("nothing predicted yet: run 'Label all clips with the model' "
                    "on the Model page")
        return ""

    def on_project(self) -> None:
        from ...infer.predict import predictions_path

        self._has_predictions = (
            self.cfg is not None and predictions_path(self.cfg).exists()
        )
        project = str(self.cfg.path) if self.cfg is not None else None
        if project != self._project:
            # A different project: drop the old panel.
            self._project = project
            self._discard()
        else:
            self._stale = True
        if self.isVisible():
            self.ensure_loaded()
        self._paint()

    def ensure_loaded(self) -> None:
        """Build or refresh the panel. Called on arrival, never at startup."""
        self._paint()
        if self.blocker():
            self._discard()
            return
        if self.panel is None:
            self.panel = BoutReviewPanel(self.cfg, self)
            self.host.addWidget(self.panel, 1)
            self._stale = False
        elif self._stale:
            self.panel.set_config(self.cfg)
            self._stale = False

    def _paint(self) -> None:
        blocker = self.blocker()
        self.note.setText(blocker)
        self.note.setVisible(bool(blocker))

    def _discard(self) -> None:
        if self.panel is None:
            return
        self.panel.release()
        self.host.removeWidget(self.panel)
        self.panel.setParent(None)
        self.panel.deleteLater()
        self.panel = None
        self._stale = False

    def shutdown(self) -> None:
        self._discard()
