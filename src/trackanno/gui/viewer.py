"""Clip viewers behind one small interface.

``SimpleViewer`` is a self-contained player: canvas, QTimer and slider. The
overlay is never rasterised, so the cached clip stays grayscale and the marker
is vector graphics one hairline wide at any zoom. Centring and zoom are
display-time transforms of a :class:`~trackanno.view.overlay.ClipView`, unclamped, so
a clip cropped at the movie border still shows the animal dead centre.
"""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from qtpy.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from qtpy.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QSizePolicy, QSlider, QVBoxLayout, QWidget,
)

from ..view.overlay import CONTEXT_RGB, TARGET_RGB, ClipView
from .theme import tokens

__all__ = ["ClipViewer", "SimpleViewer", "ClipCanvas"]

def _background() -> QColor:
    """Deliberately darker than the window: the crop should read as film, not
    as another panel."""
    return QColor(tokens()["canvas"])


def _to_qimage(frame: np.ndarray) -> QImage:
    frame = np.ascontiguousarray(frame)
    if not frame.flags.writeable:
        # QImage wants a writable buffer; clip-cache views are read-only memmaps.
        frame = frame.copy()
    if frame.ndim == 2:
        h, w = frame.shape
        return QImage(frame.data, w, h, w, QImage.Format_Grayscale8).copy()
    h, w, _ = frame.shape
    return QImage(frame.data, w, h, w * 3, QImage.Format_RGB888).copy()


class ClipViewer(QWidget):
    """Interface implemented by both backends."""

    frame_changed = Signal(int)

    def set_clip(self, array: np.ndarray, view: ClipView | None = None) -> None:  # pragma: no cover
        raise NotImplementedError

    def set_overlay(self, show: bool, context: bool = False, skeleton: bool = False) -> None:
        raise NotImplementedError  # pragma: no cover - interface

    def set_fps(self, fps: int) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def play(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def pause(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def toggle_play(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def is_playing(self) -> bool:  # pragma: no cover - interface
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# canvas
# --------------------------------------------------------------------------- #
class ClipCanvas(QWidget):
    """Draws one frame: the pixels through a view transform, the marker on top."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAutoFillBackground(False)
        # Clicking the video is how keyboard control comes back after a note.
        self.setFocusPolicy(Qt.ClickFocus)

        self._array: np.ndarray | None = None
        self._view: ClipView | None = None
        self._pixmaps: list[QPixmap | None] = []
        self._frame = 0
        self._edges: list[tuple[int, int]] = []
        self.show_overlay = True
        self.show_context = False
        self.show_skeleton = False

    # -- content ------------------------------------------------------------ #
    def set_clip(self, array: np.ndarray, view: ClipView | None) -> None:
        self._array = array
        self._view = view
        self._pixmaps = [None] * int(array.shape[0])
        self._frame = 0
        self.update()

    def set_frame(self, t: int) -> None:
        self._frame = int(t)
        self.update()

    def set_edges(self, edges) -> None:
        self._edges = list(edges or [])

    # -- geometry ----------------------------------------------------------- #
    def _mapping(self):
        """(origin_x, origin_y, k, source_x, source_y) for the current frame.

        ``k`` is display px per cache px. The source square may hang off the
        cached crop; the caller clips and the outside stays background."""
        if self._array is None:
            return None
        size = float(self._array.shape[1])
        if self._view is not None:
            sx, sy, side, _ = self._view.source_rect(self._frame)
        else:
            sx, sy, side = 0.0, 0.0, size
        extent = min(self.width(), self.height()) - 4
        if extent <= 0 or side <= 0:
            return None
        k = extent / side
        ox = (self.width() - extent) / 2.0
        oy = (self.height() - extent) / 2.0
        return ox, oy, k, sx, sy

    def _pixmap(self, t: int) -> QPixmap | None:
        if self._array is None or not 0 <= t < len(self._pixmaps):
            return None
        pix = self._pixmaps[t]
        if pix is None:
            # Unscaled, so the cache survives zoom, resize and re-centre; the
            # painter applies the scale at blit time.
            pix = QPixmap.fromImage(_to_qimage(self._array[t]))
            self._pixmaps[t] = pix
        return pix

    # -- painting ----------------------------------------------------------- #
    def paintEvent(self, event):  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        painter.fillRect(self.rect(), _background())
        mapping = self._mapping()
        if mapping is None:
            painter.end()
            return
        ox, oy, k, sx, sy = mapping
        t = min(self._frame, self._array.shape[0] - 1)
        pix = self._pixmap(t)
        if pix is None:
            painter.end()
            return

        extent = min(self.width(), self.height()) - 4
        painter.setClipRect(QRectF(ox, oy, extent, extent))
        # Nearest neighbour on purpose: these are 100-250 px crops, and
        # smoothing invents detail an annotator would read as real.
        painter.setRenderHint(QPainter.SmoothPixmapTransform, False)
        painter.drawPixmap(
            QRectF(ox - sx * k, oy - sy * k, pix.width() * k, pix.height() * k),
            pix,
            QRectF(0, 0, pix.width(), pix.height()),
        )

        if self.show_overlay and self._view is not None:
            painter.setRenderHint(QPainter.Antialiasing, True)
            self._paint_overlay(painter, t, ox, oy, k, sx, sy)
        painter.end()

    def _paint_overlay(self, painter, t, ox, oy, k, sx, sy) -> None:
        view = self._view

        def to_screen(x, y):
            return QPointF(ox + (float(x) - sx) * k, oy + (float(y) - sy) * k)

        if self.show_context:
            pen = QPen(QColor(*CONTEXT_RGB, 150), 1.0)
            painter.setPen(pen)
            for b in view.context_for(t):
                a, c = to_screen(b[0], b[1]), to_screen(b[2], b[3])
                painter.drawRect(QRectF(a, c))

        box, dim = view.box(t), False
        if box is None:
            box, dim = view.held_box(t), True
        if box is None:
            return
        a, c = to_screen(box[0], box[1]), to_screen(box[2], box[3])
        self._paint_reticle(painter, QRectF(a, c).normalized(), dim)

        if self.show_skeleton and view.kpts is not None:
            self._paint_skeleton(painter, view.kpts[t], to_screen, dim)

    def _paint_reticle(self, painter, rect: QRectF, dim: bool) -> None:
        """Four corner brackets plus a faint box, in screen space: the pen stays
        1.6 px at any zoom, so the marker never grows into the animal."""
        arm = max(5.0, min(rect.width(), rect.height()) * 0.26)
        x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
        corners = [
            ((x1, y1), (arm, arm)), ((x2, y1), (-arm, arm)),
            ((x1, y2), (arm, -arm)), ((x2, y2), (-arm, -arm)),
        ]
        alpha = 110 if dim else 255
        colour = QColor(*TARGET_RGB, alpha)

        def strokes(pen):
            painter.setPen(pen)
            for (cx, cy), (dx, dy) in corners:
                painter.drawLine(QPointF(cx, cy), QPointF(cx + dx, cy))
                painter.drawLine(QPointF(cx, cy), QPointF(cx, cy + dy))

        # Dark halo first: the crop is bright where the animal is and near-black
        # around it, so a single-colour marker vanishes on one or the other.
        halo = QPen(QColor(0, 0, 0, 140), 4.0)
        halo.setCapStyle(Qt.RoundCap)
        strokes(halo)

        pen = QPen(colour, 1.6)
        pen.setCapStyle(Qt.RoundCap)
        if dim:
            pen.setStyle(Qt.DashLine)
        strokes(pen)

        painter.setPen(QPen(QColor(*TARGET_RGB, 38 if dim else 60), 1.0))
        painter.drawRect(rect)

    def _paint_skeleton(self, painter, xy, to_screen, dim: bool) -> None:
        if xy is None:
            return
        alpha = 110 if dim else 220
        pts = [
            None if not np.isfinite(p).all() else to_screen(p[0], p[1])
            for p in np.asarray(xy)
        ]
        painter.setPen(QPen(QColor(*TARGET_RGB, alpha), 1.2))
        for ia, ib in self._edges:
            if ia < len(pts) and ib < len(pts) and pts[ia] and pts[ib]:
                painter.drawLine(pts[ia], pts[ib])
        painter.setBrush(QColor(*TARGET_RGB, alpha))
        painter.setPen(Qt.NoPen)
        for p in pts:
            if p is not None:
                painter.drawEllipse(p, 1.8, 1.8)
        painter.setBrush(Qt.NoBrush)


# --------------------------------------------------------------------------- #
# built-in player
# --------------------------------------------------------------------------- #
class SimpleViewer(ClipViewer):
    def __init__(self, fps: int = 30, loop: bool = True, parent=None):
        super().__init__(parent)
        self._array: np.ndarray | None = None
        self._frame = 0
        self._loop = loop

        self.canvas = ClipCanvas()

        self._play_btn = QPushButton("Pause")
        self._play_btn.setFixedWidth(70)
        self._play_btn.clicked.connect(self.toggle_play)
        self._play_btn.setFocusPolicy(Qt.NoFocus)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setFocusPolicy(Qt.NoFocus)
        self._slider.valueChanged.connect(self._on_slider)

        self._counter = QLabel("0 / 0")
        self._counter.setFixedWidth(70)
        self._counter.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.addWidget(self._play_btn)
        controls.addWidget(self._slider, 1)
        controls.addWidget(self._counter)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas, 1)
        layout.addLayout(controls)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)
        self.set_fps(fps)

    # -- interface ---------------------------------------------------------- #
    def set_clip(self, array: np.ndarray, view: ClipView | None = None) -> None:
        self._array = array
        self.canvas.set_clip(array, view)
        n = int(array.shape[0])
        self._slider.blockSignals(True)
        self._slider.setRange(0, max(0, n - 1))
        self._slider.setValue(0)
        self._slider.blockSignals(False)
        self._frame = 0
        self.canvas.set_frame(0)
        self._counter.setText(f"1 / {n}")

    def clear(self) -> None:
        """Drop the clip so nothing keeps the cache's memory map alive."""
        self._array = None
        self.canvas.set_clip(np.zeros((0, 1, 1), np.uint8), None)

    def set_view(self, view: ClipView | None) -> None:
        """Re-centre or re-zoom without touching the pixels or the pixmap cache."""
        self.canvas._view = view
        self.canvas.update()

    def set_overlay(self, show: bool, context: bool = False, skeleton: bool = False) -> None:
        self.canvas.show_overlay = show
        self.canvas.show_context = context
        self.canvas.show_skeleton = skeleton
        self.canvas.update()

    def set_edges(self, edges) -> None:
        self.canvas.set_edges(edges)

    def set_fps(self, fps: int) -> None:
        self._timer.setInterval(max(10, int(1000 / max(1, fps))))

    def play(self) -> None:
        self._timer.start()
        self._play_btn.setText("Pause")

    def pause(self) -> None:
        self._timer.stop()
        self._play_btn.setText("Play")

    def toggle_play(self) -> None:
        self.pause() if self._timer.isActive() else self.play()

    @property
    def is_playing(self) -> bool:
        return self._timer.isActive()

    def seek(self, frame: int) -> None:
        """Jump to a frame. Drives the slider, not ``_frame``, which is what
        keeps the counter and ``frame_changed`` consistent."""
        if self._array is None:
            return
        self._slider.setValue(max(0, min(int(frame), self._array.shape[0] - 1)))

    # -- internals ---------------------------------------------------------- #
    def _advance(self) -> None:
        if self._array is None:
            return
        n = self._array.shape[0]
        nxt = self._frame + 1
        if nxt >= n:
            if not self._loop:
                self.pause()
                return
            nxt = 0
        self._slider.setValue(nxt)

    def _on_slider(self, value: int) -> None:
        self._frame = int(value)
        self.canvas.set_frame(self._frame)
        if self._array is not None:
            self._counter.setText(f"{self._frame + 1} / {self._array.shape[0]}")
        self.frame_changed.emit(self._frame)
