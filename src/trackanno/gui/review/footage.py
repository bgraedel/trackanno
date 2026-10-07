"""The footage pane: decoded chunks of one track, shown through ``SimpleViewer``."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ...config import Config
from ...view.overlay import build_clip_view
from ..viewer import SimpleViewer, follow_combo
from ..viewprefs import load_view
from .decode import (
    REVIEW_CHUNK, ChunkDecoder, MovieSource, ReviewCrop, TrackCutter, review_crop,
)

__all__ = ["CHUNKS_KEPT", "REVIEW_FOLLOW", "ChunkCache", "TrackFootage"]

CHUNKS_KEPT = 6
# Bouts are long, so review always re-centres per frame.
REVIEW_FOLLOW = "locked"


class ChunkCache:
    """An LRU of decoded chunks keyed by first frame."""

    def __init__(self, capacity: int = CHUNKS_KEPT):
        self.capacity = max(1, int(capacity))
        self._items: OrderedDict[int, tuple] = OrderedDict()

    def get(self, first: int):
        payload = self._items.get(int(first))
        if payload is not None:
            self._items.move_to_end(int(first))
        return payload

    def put(self, first: int, payload) -> None:
        self._items[int(first)] = payload
        self._items.move_to_end(int(first))
        while len(self._items) > self.capacity:
            self._items.popitem(last=False)

    def clear(self) -> None:
        self._items.clear()

    def keys(self) -> list[int]:
        return list(self._items)

    def __iter__(self):
        return iter(list(self._items))

    def __contains__(self, first) -> bool:
        return int(first) in self._items

    def __len__(self) -> int:
        return len(self._items)


class TrackFootage(QWidget):
    """Footage of the track under review, decoded off the UI thread.

    A generation token drops late replies after a track switch.
    """

    positionChanged = Signal(int)         # movie frame now on screen
    _crop_measured = Signal(int, object)  # config generation, ReviewCrop or None

    def __init__(self, cfg: Config | None = None, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        # View toggles shared with the annotation window.
        self.prefs = load_view()
        self.viewer = SimpleViewer(fps=self._fps(), loop=True)
        self.viewer.frame_changed.connect(self._on_viewer_frame)
        self.viewer.canvas.setMinimumSize(240, 240)
        self.viewer.set_scrubber(False)    # the lanes are the frame axis
        self.viewer.setVisible(False)
        self._set_edges()
        self.source = QLabel("")
        self.source.setWordWrap(True)
        self.source.setProperty("role", "muted")

        self.follow_combo = follow_combo(REVIEW_FOLLOW)
        self.follow_combo.currentIndexChanged.connect(lambda _: self._refresh_view())
        self.marker_box = QCheckBox("marker")
        self.marker_box.setToolTip("Outline the animal this track follows.")
        self.marker_box.setChecked(bool(self.prefs["show_overlay"]))
        self.marker_box.setFocusPolicy(Qt.NoFocus)
        self.marker_box.toggled.connect(lambda _: self._refresh_view())

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("view"))
        row.addWidget(self.follow_combo)
        row.addWidget(self.marker_box)
        row.addStretch(1)
        self.controls = QWidget()
        self.controls.setLayout(row)
        self.controls.setVisible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.viewer, 1)
        layout.addWidget(self.controls)
        layout.addWidget(self.source)

        # Coalesces rapid set_track calls so only the last track is opened.
        self._kick = QTimer(self)
        self._kick.setSingleShot(True)
        self._kick.setInterval(0)
        self._kick.timeout.connect(self._on_kick)

        self._crop: ReviewCrop | None = None
        self._config_gen = 0
        self._measuring = False
        self._crop_measured.connect(self._on_crop_measured)
        self._cache = ChunkCache()
        self._decoder: ChunkDecoder | None = None
        self._gen = 0
        self._movie_id, self._track_id = "", 0
        self._span = (0, 0)
        self._problem = ""            # why this track cannot be opened, or ""
        self._wanted: int | None = None   # frame waited for
        self._resume = False          # play on when the wanted frame lands
        self._shown: int | None = None    # first frame of the chunk on screen
        self._length = 0              # frames in that chunk
        self._offset = 0
        self._position = -1
        self._boxes = None
        self._shape = (0, 0)
        self._seeking = False
        self._skeleton_problem = ""

    # -- the project --------------------------------------------------------- #
    def _fps(self) -> int:
        return int(getattr(getattr(self.cfg, "ui", None), "fps", 30))

    def set_config(self, cfg: Config | None) -> None:
        self.release()
        self.cfg = cfg
        self._crop = None
        self._config_gen += 1
        self._measuring = False
        self.viewer.set_fps(self._fps())
        self._set_edges()

    def set_crop(self, crop: ReviewCrop) -> None:
        """Adopt a crop measured off the UI thread for the current config."""
        if self._crop is None:
            self._crop = crop
            if self._wanted is not None and self._decoder is None:
                self._kick.start()

    @property
    def crop(self) -> ReviewCrop | None:
        """The session's crop, or None until it has been measured."""
        return self._crop

    def _measure_crop(self) -> None:
        """Measure the session crop on a worker thread."""
        if self._measuring or self.cfg is None:
            return
        self._measuring = True
        import threading

        cfg, gen = self.cfg, self._config_gen

        def work():
            try:
                crop = review_crop(cfg)
            except Exception as exc:               # noqa: BLE001
                print(f"[trackanno] review crop failed: {exc}")
                crop = None
            try:
                self._crop_measured.emit(gen, crop)
            except RuntimeError:                   # the widget is gone
                pass

        threading.Thread(target=work, name="trackanno-review-crop", daemon=True).start()

    def _on_crop_measured(self, gen: int, crop) -> None:
        if gen != self._config_gen:
            return
        self._measuring = False
        if crop is None:
            if self._crop is None and self._wanted is not None:
                self._no_video("the footage crop could not be measured")
            return
        self.set_crop(crop)

    def release(self) -> None:
        """Stop the decoder (without waiting) and drop every frame held."""
        self._kick.stop()
        self._stop_decoder()
        self._drop_frames()

    def _drop_frames(self) -> None:
        self.viewer.pause()
        self.viewer.clear()
        self._cache.clear()
        self._problem = ""
        self._shown = None
        self._length = 0
        self._boxes = None

    def _stop_decoder(self) -> None:
        self._gen += 1
        self._wanted = None
        self._resume = False
        decoder, self._decoder = self._decoder, None
        if decoder is not None:
            decoder.stop()

    # -- one track ----------------------------------------------------------- #
    def set_track(self, movie_id: str, track_id: int, f_lo: int, f_hi: int) -> None:
        """Show ``f_lo`` of this track; decoding starts on the next event-loop turn."""
        self.viewer.pause()
        key = (str(movie_id), int(track_id))
        if key[0] != self._movie_id or self._decoder is None:
            self.release()
        elif key[1] != self._track_id:
            self._gen += 1
            self._decoder.retarget(self._gen)
            self._wanted = None
            self._drop_frames()
        self._movie_id, self._track_id = key
        self._span = (int(f_lo), max(int(f_hi), int(f_lo) + 1))
        self._position = -1
        self._resume = False
        self._wanted = int(f_lo)
        if not self._length:
            self._set_caption(f"reading {self._movie_id}...")
        self._kick.start()

    def _on_kick(self) -> None:
        if self._wanted is not None:
            self._show(self._wanted)

    @property
    def position(self) -> int:
        """The movie frame on screen, or -1."""
        return self._position

    @property
    def loading(self) -> bool:
        """Whether a frame is being waited for."""
        return self._wanted is not None and not self._problem

    @property
    def showing(self) -> str:
        """``"movie"`` when frames are on screen, else ``""``."""
        return "movie" if self._length else ""

    def frame_image(self) -> np.ndarray | None:
        """The (S, S) array on screen."""
        array = self.viewer._array
        if array is None or not len(array):
            return None
        return np.asarray(array[min(self.viewer._frame, array.shape[0] - 1)])

    def _set_caption(self, text: str) -> None:
        if text != self.source.text():
            self.source.setText(text)

    # -- moving -------------------------------------------------------------- #
    def _clamp(self, frame: int) -> int:
        lo, hi = self._span
        return int(min(max(int(frame), lo), hi - 1))

    def _chunk_bounds(self, frame: int) -> tuple[int, int]:
        """The grid cell holding ``frame``, cut down to the span."""
        lo, hi = self._span
        cell = (int(frame) // REVIEW_CHUNK) * REVIEW_CHUNK
        return max(cell, lo), min(cell + REVIEW_CHUNK, hi)

    def show_frame(self, frame: int) -> None:
        """Put ``frame`` of this track on screen (clamped to the span)."""
        self._resume = False
        if self._kick.isActive():
            self._wanted = self._clamp(frame)
            return
        self._show(frame)

    def _show(self, frame: int) -> bool:
        """Display ``frame`` now if held, else ask for it. True if displayed."""
        if self.cfg is None or not self._movie_id:
            self._no_video("no track is selected")
            return False
        frame = self._clamp(frame)
        self._wanted = frame
        if self._problem:
            self._no_video(self._problem)
            return False
        if self._crop is None:
            # Waits for set_crop, from the page or from _measure_crop.
            self._measure_crop()
            return False
        first, last = self._chunk_bounds(frame)
        payload = self._cache.get(first)
        if payload is not None:
            self._wanted = None
            self._display(first, payload, frame)
            self._read_ahead(last)
            return True
        self._ensure_decoder().want(self._gen, self._track_id, first, last)
        if not self._length:
            self._set_caption(f"reading frames {first}-{last} of {self._movie_id}...")
        return False

    def _ensure_decoder(self) -> ChunkDecoder:
        if self._decoder is None:
            movie_id, source = self._movie_id, MovieSource(self.cfg, self._movie_id)
            self._decoder = ChunkDecoder(
                lambda track: self.make_cutter(movie_id, track, source),
                closer=source.close)
            self._decoder.ready.connect(self._on_ready)
            self._decoder.failed.connect(self._on_failed)
        return self._decoder

    def make_cutter(self, movie_id: str, track_id: int, source=None) -> TrackCutter:
        """A cutter for one track on the decoder's shared movie (worker thread)."""
        return TrackCutter(self.cfg, movie_id, track_id, self._crop, REVIEW_CHUNK,
                           source=source)

    def _read_ahead(self, after: int) -> None:
        """Queue the chunk starting at ``after``, if any and not held."""
        if after >= self._span[1]:
            return
        first, last = self._chunk_bounds(after)
        if first not in self._cache:
            self._ensure_decoder().read_ahead(self._gen, self._track_id, first, last)

    def _display(self, first: int, payload, frame: int) -> None:
        array, marks = payload
        offset = int(min(max(frame - first, 0), len(array) - 1))
        if self._shown != first or self.viewer._array is not array:
            self.viewer.set_clip(array, self._view(marks, array), frame=offset)
            self._mark()
            self.viewer.setVisible(True)
            self.controls.setVisible(True)
            self._shown = int(first)
            self._length = int(len(array))
        self._seek(offset)
        self._set_caption(self._note(frame))

    def _seek(self, offset: int) -> None:
        offset = int(min(max(offset, 0), max(0, self._length - 1)))
        self._offset = offset
        self._position = int(self._shown) + offset
        self._seeking = True
        try:
            self.viewer.seek(offset)
        finally:
            self._seeking = False
        self.positionChanged.emit(self._position)

    def _on_viewer_frame(self, offset: int) -> None:
        """Playback walks the marker; the chunk's end hands over to the next."""
        if self._seeking or not self._length or self._shown is None:
            return
        offset, was = int(offset), self._offset
        self._offset = offset
        self._position = self._shown + offset
        self.positionChanged.emit(self._position)
        if not self.viewer.is_playing:
            return
        arrived = offset >= self._length - 1
        looped = offset == 0 and was >= self._length - 1 and self._length > 1
        if arrived or looped:
            # Queued to avoid re-entering the viewer's signal.
            QTimer.singleShot(0, self._roll_on)

    def _roll_on(self) -> None:
        """Carry playback into the next chunk, or back to the start of the track."""
        if not self.viewer.is_playing or self._shown is None:
            return
        nxt = self._shown + self._length
        end_of_chunk = self._chunk_bounds(self._shown)[1]
        if nxt >= self._span[1] or nxt < end_of_chunk:
            nxt = self._span[0]           # end of the track, or of the movie
        self.viewer.pause()
        if self._show(nxt):
            self.viewer.play()
        else:
            self._resume = True

    # -- replies ------------------------------------------------------------- #
    def _on_ready(self, gen: int, first: int, payload) -> None:
        if gen != self._gen:
            return                        # a previous track's decoder
        self._cache.put(first, payload)
        if self._wanted is None or self._chunk_bounds(self._wanted)[0] != int(first):
            return
        frame, self._wanted = self._wanted, None
        self._display(int(first), payload, frame)
        self._read_ahead(self._chunk_bounds(frame)[1])
        if self._resume:
            self._resume = False
            self.viewer.play()

    def _on_failed(self, gen: int, first: int, message: str, fatal: bool) -> None:
        if gen != self._gen:
            return
        if fatal:
            # Remembered until the track changes, so scrubbing does not retry.
            self._wanted = None
            self._problem = f"{self._movie_id} could not be read ({message})"
            self._no_video(self._problem)
            return
        if self._wanted is None or self._chunk_bounds(self._wanted)[0] != int(first):
            return
        self._wanted = None
        self._resume = False
        self.viewer.pause()
        reason = f"frames from {first} could not be read ({message})"
        if self._length:
            self._set_caption(f"{reason}.")
        else:
            self._no_video(reason)

    # -- the view ------------------------------------------------------------ #
    def _view(self, boxes, clip: np.ndarray):
        """The view for a newly shown array, remembered for the controls."""
        self._boxes = boxes
        self._shape = (int(clip.shape[0]), int(clip.shape[1]))
        return self._build_view()

    def _build_view(self):
        return build_clip_view(
            self._boxes, self._shape[0], self._shape[1],
            mode=self.follow,
            zoom=float(self.prefs["view_zoom"]),
            with_context=bool(self.prefs["show_context_tracks"]),
            n_kpts=(int(getattr(self.cfg, "n_keypoints", 0))
                    if self.prefs["show_skeleton"] else 0),
        )

    @property
    def follow(self) -> str:
        return str(self.follow_combo.currentData() or REVIEW_FOLLOW)

    def _set_edges(self) -> None:
        self.viewer.set_edges(
            self.cfg.keypoints.edge_idx if getattr(self.cfg, "is_pose", False) else []
        )

    def _mark(self) -> None:
        self.viewer.set_overlay(
            bool(self.marker_box.isChecked()),
            bool(self.prefs["show_context_tracks"]),
            bool(self.prefs["show_skeleton"]),
        )

    def _refresh_view(self) -> None:
        """Re-centre and re-mark the array on screen without decoding."""
        if self._boxes is None or not self._length:
            return
        self.viewer.set_view(self._build_view())
        self._mark()

    # -- captions ------------------------------------------------------------ #
    def _note(self, wanted: int) -> str:
        crop = self.crop
        note = f"frame {self._position} of track {self._track_id}"
        if crop is not None:
            note += f", {crop.side} px crop"
        note += "."
        if self._position != wanted:
            note += f" Frame {wanted} is past the end of the movie."
        cutter = getattr(self._decoder, "cutter", None)
        gap = getattr(cutter, "skeleton_problem", "")
        if gap and bool(self.prefs["show_skeleton"]):
            note += f" {gap}."
        return note

    def clear(self, reason: str = "") -> None:
        """No track under review."""
        self.release()
        self._position = -1
        self._span = (0, 0)
        self._movie_id, self._track_id = "", 0
        self._no_video(reason or "no track is selected")

    def _no_video(self, reason: str) -> None:
        """Hide the picture and say why in one line."""
        self.viewer.pause()
        self.viewer.clear()
        self.viewer.setVisible(False)
        self.controls.setVisible(False)
        self._shown = None
        self._length = 0
        self._boxes = None
        text = f"No video: {reason.strip().rstrip('.')}."
        if self._span[1] > self._span[0]:
            text += " The timeline is unaffected."
        self._set_caption(text)
