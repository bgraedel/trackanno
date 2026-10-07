"""The Review panel: movie and track selectors, bout lanes and footage."""

from __future__ import annotations

import threading

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QScrollArea, QSizePolicy, QSplitter,
    QVBoxLayout, QWidget,
)

from ...config import Config
from ..theme import help_button, tokens
from .decode import review_crop
from .footage import TrackFootage
from .lanes import BoutTimeline
from .reader import PredictionReader, TrackPredictions

__all__ = ["REVIEW_HELP", "BoutReviewPanel"]

REVIEW_HELP = """
<h3>Lanes</h3>
<p>Each bar is one predicted <b>bout</b>; its height is the mean confidence.</p>
<p>Hatched stretches could not be scored; hover for the reason.</p>

<h3>Video</h3>
<p>Click or drag on the lanes to move the video there.</p>
"""


def _combo(min_chars: int) -> QComboBox:
    combo = QComboBox()
    combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(min_chars)
    combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    return combo


class BoutReviewPanel(QWidget):
    """Read-only view of one track's predicted bouts beside its footage."""

    _loaded = Signal(int, object)         # load generation, (reader or reason, crop)

    def __init__(self, cfg: Config | None = None, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.reader: PredictionReader | None = None
        self._loading = False             # combos being refilled
        self._load_gen = 0
        self._pending: tuple[str, int] | None = None
        self._loaded.connect(self._on_loaded)
        self._track: TrackPredictions | None = None

        self.movie = _combo(10)
        self.movie.currentIndexChanged.connect(self._on_movie)
        self.track = _combo(8)
        self.track.currentIndexChanged.connect(self._on_track)
        self.headline = QLabel("")
        self.headline.setWordWrap(True)
        self.headline.setTextFormat(Qt.RichText)
        self.help_button = help_button(self, "Reviewing predictions", REVIEW_HELP)

        top = QHBoxLayout()
        top.addWidget(QLabel("movie"))
        top.addWidget(self.movie, 1)
        top.addSpacing(12)
        top.addWidget(QLabel("track"))
        top.addWidget(self.track, 1)
        top.addStretch(1)
        top.addWidget(self.help_button)

        self.timeline = BoutTimeline(cfg)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.timeline)
        scroll.setMinimumWidth(320)

        self.footage = TrackFootage(cfg)
        self.footage.setMinimumWidth(240)
        self.footage.setMaximumWidth(520)
        self.timeline.positionPicked.connect(self.footage.show_frame)
        self.footage.positionChanged.connect(self.timeline.set_position)
        self.split = QSplitter(Qt.Horizontal)
        self.split.addWidget(self.footage)
        self.split.addWidget(scroll)
        self.split.setStretchFactor(0, 0)
        self.split.setStretchFactor(1, 1)
        self.split.setSizes([400, 760])

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.headline)
        layout.addWidget(self.split, 1)

        if cfg is not None:
            self.reload()

    # -- lifecycle ----------------------------------------------------------- #
    def set_config(self, cfg: Config | None) -> None:
        self.cfg = cfg
        self.timeline.cfg = cfg
        self.footage.set_config(cfg)
        self.reload()

    def release(self) -> None:
        """Let go of the movie. Never blocks; safe to call repeatedly."""
        self.footage.release()

    def hideEvent(self, event):  # noqa: N802 (Qt naming)
        self.release()
        super().hideEvent(event)

    def showEvent(self, event):  # noqa: N802 (Qt naming)
        super().showEvent(event)
        track = self._track
        if track is None or track.empty:
            return
        was = self.timeline.position
        self.footage.set_track(track.movie_id, track.track_id, track.f_lo, track.f_hi)
        if was is not None:
            self.footage.show_frame(was)

    def closeEvent(self, event):  # noqa: N802 (Qt naming)
        self.release()
        super().closeEvent(event)

    # -- selection ----------------------------------------------------------- #
    def reload(self) -> None:
        """Re-read the predictions off the UI thread, then rebuild the selectors."""
        self._load_gen += 1
        self.reader = None
        self._loading = True
        try:
            self.movie.clear()
            self.track.clear()
        finally:
            self._loading = False
        if self.cfg is None:
            self._say("no project open")
            return
        self._say("loading predictions...")
        threading.Thread(target=self._load, args=(self._load_gen, self.cfg),
                         name="trackanno-review-load", daemon=True).start()

    def _load(self, gen: int, cfg: Config) -> None:
        """Worker: read predictions, first movie's tracks and crop, then warm the rest."""
        crop = None
        try:
            reader = PredictionReader(cfg)
            if not reader.problem and reader.movies:
                reader.tracks(reader.movies[0])
                crop = review_crop(cfg)
        except Exception as exc:
            reader = " ".join(str(exc).split()) or type(exc).__name__
        try:
            self._loaded.emit(gen, (reader, crop))
        except RuntimeError:              # the panel is gone
            return
        if isinstance(reader, PredictionReader) and not reader.problem:
            for movie_id in reader.movies[1:]:
                if gen != self._load_gen:
                    return
                reader.tracks(movie_id)

    def _on_loaded(self, gen: int, loaded) -> None:
        if gen != self._load_gen:
            return
        reader, crop = loaded
        if crop is not None:
            self.footage.set_crop(crop)
        if not isinstance(reader, PredictionReader):
            self._say(f"the predictions could not be read ({reader})")
            return
        self.reader = reader
        if reader.problem:
            self._say(reader.problem)
            return
        self._loading = True
        try:
            self.movie.addItems(reader.movies)
        finally:
            self._loading = False
        pending, self._pending = self._pending, None
        if pending is not None and self.select(*pending):
            return
        self._fill_tracks()

    @property
    def loaded(self) -> bool:
        """Whether the predictions have been read (successfully or not)."""
        return self.reader is not None

    def select(self, movie_id: str, track_id: int) -> bool:
        """Show one (movie, track); before loading finishes, the choice is queued."""
        if self.reader is None:
            self._pending = (str(movie_id), int(track_id))
            return True
        if self.reader.problem:
            return False
        self._loading = True
        try:
            self.movie.setCurrentText(str(movie_id))
            self._fill_track_combo()
            index = self.track.findData(int(track_id))
            if index >= 0:
                self.track.setCurrentIndex(index)
        finally:
            self._loading = False
        self._paint()
        return index >= 0

    def _on_movie(self, _index: int) -> None:
        if not self._loading:
            self._fill_tracks()

    def _on_track(self, _index: int) -> None:
        if not self._loading:
            self._paint()

    def _fill_track_combo(self) -> None:
        self.track.clear()
        for track_id, count in self.reader.tracks(self.movie.currentText()):
            label = f"track {track_id}" if count else f"track {track_id}  (none)"
            self.track.addItem(label, track_id)

    def _fill_tracks(self) -> None:
        if self.reader is None or self.reader.problem:
            return
        self._loading = True
        try:
            self._fill_track_combo()
        finally:
            self._loading = False
        self._paint()

    # -- painting ------------------------------------------------------------ #
    def _paint(self) -> None:
        if self.reader is None or self.reader.problem:
            return
        movie_id = self.movie.currentText()
        track_id = self.track.currentData()
        if not movie_id or track_id is None:
            self._say("the predictions name no movie to review")
            return
        try:
            track = self.reader.track(movie_id, int(track_id))
        except Exception as exc:          # one unreadable track, not the page
            self._say(f"track {track_id} of {movie_id} could not be read "
                      f"({' '.join(str(exc).split())})")
            return

        self._track = track
        self.timeline.set_track(track)
        if track.empty:
            self.footage.clear(f"track {track.track_id} of {track.movie_id} has no "
                               "predicted rows")
        else:
            self.footage.set_track(movie_id, int(track_id), track.f_lo, track.f_hi)
        head = (
            f"<b>predicted bouts</b> over {len(self.reader.groups)} group(s); "
            f"<b>{track.n_segments}</b> row(s) across frames {track.f_lo}-{track.f_hi}."
            if not track.empty else "<b>predicted bouts</b>; this track has none."
        )
        if self.reader.stale:
            head += (f"<br><span style='color:{tokens()['warning']}'>Predictions "
                     f"may be out of date: {self.reader.stale}.</span>")
        self.headline.setText(head)

    def _say(self, problem: str) -> None:
        """Show a message in place of the lanes and footage."""
        self.headline.setText("")
        self._track = None
        self.timeline.set_track(TrackPredictions("", 0), note=problem)
        self.footage.clear(problem)
