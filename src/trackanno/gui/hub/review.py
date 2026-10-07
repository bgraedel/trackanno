"""Step 6: check predicted bouts against the footage."""

from __future__ import annotations


from qtpy.QtWidgets import (
    QLabel, QVBoxLayout,
)

from ..review import REVIEW_HELP, BoutReviewPanel
from .base import Page, _heading, _muted


class ReviewPage(Page):
    """Where a predicted boundary is checked.

    The panel reads the predictions, so it is built on the first visit rather than
    at startup, and there is one of it per window. Leaving the page hides it,
    which is what makes it let go of the movie and its decode thread.
    """

    def __init__(self, hub, parent=None):
        super().__init__(hub, parent)
        self.panel: BoutReviewPanel | None = None
        self._has_predictions = False
        self._project: str | None = None
        # Set when the file under the panel has changed and the panel has not
        # been told yet, because it is not on screen to be told.
        self._stale = False

        self.note = _muted(QLabel(""))
        self.note.setWordWrap(True)

        self.host = QVBoxLayout()
        self.host.setContentsMargins(0, 0, 0, 0)

        layout = QVBoxLayout(self)
        layout.addWidget(_heading(
            "Review",
            "Check where the model put each boundary, against the footage.",
            # The panel carries the same `?`, but the panel is not built until
            # there are predictions - and the page is worth explaining before
            # then, to somebody wondering what it would show.
            help=REVIEW_HELP,
        ))
        layout.addWidget(self.note)
        layout.addLayout(self.host, 1)

    def blocker(self) -> str:
        if self.cfg is None:
            return "no project open"
        if not self._has_predictions:
            return ("nothing has been predicted yet - run Label every clip on "
                    "the Model page")
        return ""

    def on_project(self) -> None:
        from ...infer.predict import predictions_path

        self._has_predictions = (
            self.cfg is not None and predictions_path(self.cfg).exists()
        )
        project = str(self.cfg.path) if self.cfg is not None else None
        if project != self._project:
            # A different project: the panel holds the old one's tables.
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
            # Re-reads the predictions, and lets go of the previous movie first.
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
