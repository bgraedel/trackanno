"""The custom-painted bout lanes of one track."""

from __future__ import annotations

from bisect import bisect_right

import numpy as np
from qtpy.QtCore import QEvent, QPointF, QRect, QRectF, QSize, Qt, Signal
from qtpy.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPolygonF
from qtpy.QtWidgets import QSizePolicy, QToolTip, QWidget

from ...annotate.store import MULTI_SEP
from ...config import Config
from ..theme import tokens
from .reader import Segment, TrackPredictions

__all__ = ["BoutTimeline", "axis_ticks"]


class BoutTimeline(QWidget):
    """One lane per predicted group on a shared frame axis.

    Painting and hit-testing only visit the segments in view, so cost does not
    grow with track length.
    """

    LANE = 52
    BAR = 34
    TOP = 10
    AXIS = 26
    RIGHT = 12

    positionPicked = Signal(int)          # frame clicked or dragged to

    def __init__(self, cfg: Config | None = None, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.track = TrackPredictions("", 0)
        self.note = ""
        self.position: int | None = None
        self._starts: dict[str, list[int]] = {}
        self._scrub = False
        self._geom: tuple[float, float] = (0.0, 0.0)   # (gutter, px per frame)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumWidth(420)
        self.setMouseTracking(True)

    def set_track(self, track: TrackPredictions, note: str = "") -> None:
        self.track = track
        self.note = note
        self.position = None
        self._scrub = False
        self._starts = {g: [s.f_start for s in segs] for g, segs in track.lanes.items()}
        gutter = self._gutter()
        self._geom = (gutter, self._scale(gutter))
        self.updateGeometry()
        self.update()

    def set_position(self, frame: int | None) -> None:
        """Move the playhead marker; repaints only the strips it leaves and enters."""
        frame = None if frame is None else int(frame)
        if frame == self.position:
            return
        old, self.position = self.position, frame
        for f in (old, frame):
            strip = self._strip(f)
            if strip is None:
                self.update()
                return
            self.update(strip)

    def _strip(self, frame: int | None) -> QRect | None:
        if frame is None:
            return QRect()
        gutter, scale = self._geom
        if scale <= 0:
            return None
        x = int(self._x(frame, gutter, scale))
        return QRect(x - 6, 0, 13, self.height())

    def sizeHint(self) -> QSize:
        lanes = max(1, len(self.track.lanes))
        return QSize(760, self.TOP + lanes * self.LANE + self.AXIS + 8)

    def minimumSizeHint(self) -> QSize:
        return QSize(self.minimumWidth(), self.sizeHint().height())

    def resizeEvent(self, event):  # noqa: N802 (Qt naming)
        gutter = self._gutter()
        self._geom = (gutter, self._scale(gutter))
        super().resizeEvent(event)

    # -- geometry ------------------------------------------------------------ #
    def _gutter(self, metrics=None) -> float:
        metrics = metrics if metrics is not None else self.fontMetrics()
        names = list(self.track.lanes)
        return min(
            max(70.0, max(metrics.horizontalAdvance(n) for n in names or [""]) + 16.0),
            max(80.0, self.width() * 0.28),
        )

    def _scale(self, gutter: float) -> float:
        span = max(1, self.track.f_hi - self.track.f_lo)
        return max(0.0, self.width() - gutter - self.RIGHT) / span

    def _x(self, frame: float, gutter: float, scale: float) -> float:
        return gutter + (frame - self.track.f_lo) * scale

    def _frame_float(self, x: float) -> float:
        gutter, scale = self._geom
        if scale <= 0:
            return float(self.track.f_lo)
        return self.track.f_lo + (x - gutter) / scale

    def _frame_at(self, x: float) -> int:
        """The frame under a pixel, clamped to ``[f_lo, f_hi - 1]``."""
        last = max(self.track.f_lo, self.track.f_hi - 1)
        return int(min(max(int(np.floor(self._frame_float(x) + 1e-6)), self.track.f_lo), last))

    def _visible(self, group: str, f_a: float, f_b: float):
        """The segments of ``group`` overlapping frames ``[f_a, f_b]``."""
        segments = self.track.lanes.get(group, [])
        starts = self._starts.get(group, [])
        i = max(0, bisect_right(starts, f_a) - 1)
        while i < len(segments) and segments[i].f_start <= f_b:
            if segments[i].f_end >= f_a:
                yield segments[i]
            i += 1

    # -- painting ------------------------------------------------------------ #
    def paintEvent(self, event):  # noqa: N802 (Qt naming)
        t = tokens()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)

        if self.note or self.track.empty:
            painter.setPen(QColor(t["muted"]))
            painter.drawText(
                self.rect().adjusted(16, 0, -16, 0),
                int(Qt.AlignCenter | Qt.TextWordWrap),
                self.note or self._nothing_here(),
            )
            painter.end()
            return

        gutter = self._gutter(painter.fontMetrics())
        scale = self._scale(gutter)
        self._geom = (gutter, scale)
        area = event.rect()
        # One frame of slack each side for minimum-width bars and labels.
        f_a = self._frame_float(area.left()) - 1.0 / max(scale, 1e-9)
        f_b = self._frame_float(area.right()) + 1.0 / max(scale, 1e-9)
        y = float(self.TOP)
        for group in self.track.lanes:
            self._paint_lane(painter, t, group, y, gutter, scale, f_a, f_b)
            y += self.LANE
        self._paint_axis(painter, t, y, gutter, scale)
        self._paint_position(painter, t, y, gutter, scale)
        painter.end()

    def _paint_position(self, painter, t, bottom, gutter, scale) -> None:
        pos = self.position
        if pos is None or not self.track.f_lo <= pos < self.track.f_hi:
            return
        x = self._x(pos, gutter, scale)
        painter.setPen(QPen(QColor(t["accent"]), 1))
        painter.drawLine(QPointF(x, self.TOP - 6), QPointF(x, bottom))
        head = QPolygonF([QPointF(x - 4, self.TOP - 8), QPointF(x + 4, self.TOP - 8),
                          QPointF(x, self.TOP - 2)])
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(t["accent"]))
        painter.drawPolygon(head)

    def _nothing_here(self) -> str:
        return (
            f"No predicted bouts for track {self.track.track_id} of "
            f"{self.track.movie_id}.\n\n"
            "That is normal, not an error: a track with no featurisable window "
            "gets no row."
        )

    def _bar(self, y: float, gutter: float) -> QRectF:
        top = y + (self.LANE - self.BAR) / 2
        return QRectF(gutter, top, max(0.0, self.width() - gutter - self.RIGHT), self.BAR)

    def _paint_lane(self, painter, t, group, y, gutter, scale, f_a, f_b) -> None:
        bar = self._bar(y, gutter)
        painter.setPen(QColor(t["muted"]))
        painter.drawText(
            QRectF(2, bar.top(), gutter - 8, self.BAR),
            int(Qt.AlignVCenter | Qt.AlignRight),
            painter.fontMetrics().elidedText(group, Qt.ElideRight, int(gutter - 10)),
        )
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(t["raised"]))
        painter.drawRoundedRect(bar, 4, 4)
        for segment in self._visible(group, f_a, f_b):
            self._paint_segment(painter, t, group, segment, bar, gutter, scale)

    def _colour(self, group: str, label: str, t) -> QColor:
        """The class colour from the config, faint grey for an unknown class."""
        spec = self.cfg.group(group) if self.cfg is not None else None
        option = spec.option(label.split(MULTI_SEP)[0]) if spec is not None else None
        colour = QColor(option.color) if option is not None else QColor(t["faint"])
        return colour if colour.isValid() else QColor(t["faint"])

    def _rect(self, segment: Segment, bar: QRectF, gutter: float, scale: float) -> QRectF:
        x0 = self._x(segment.f_start, gutter, scale)
        x1 = self._x(segment.f_end, gutter, scale)
        return QRectF(x0, bar.top(), max(1.0, x1 - x0), self.BAR)

    def _paint_segment(self, painter, t, group, segment, bar, gutter, scale) -> None:
        rect = self._rect(segment, bar, gutter, scale)
        if not segment.label:
            hatch = QColor(t["faint"])
            hatch.setAlpha(70)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(hatch, Qt.FDiagPattern))
            painter.drawRect(rect)
            painter.setPen(QPen(QColor(t["border"]), 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(rect)
            return

        colour = self._colour(group, segment.label, t)
        conf = segment.conf if np.isfinite(segment.conf) else 0.0
        height = max(1.0, conf * self.BAR)
        painter.setPen(Qt.NoPen)
        painter.setBrush(colour)
        painter.drawRect(QRectF(rect.left(), rect.bottom() - height, rect.width(), height))
        painter.setPen(QPen(colour.darker(140), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect)

        text = (f"{segment.label}  {segment.conf:.2f}" if np.isfinite(segment.conf)
                else segment.label)
        if rect.width() > painter.fontMetrics().horizontalAdvance(text) + 8:
            painter.setPen(QColor(t["text"]))
            painter.drawText(rect, int(Qt.AlignCenter), text)

    def _paint_axis(self, painter, t, y, gutter, scale) -> None:
        painter.setPen(QPen(QColor(t["border"]), 1))
        painter.drawLine(QPointF(gutter, y + 2), QPointF(self.width() - self.RIGHT, y + 2))
        painter.setPen(QColor(t["faint"]))
        font = QFont(self.font())
        font.setPointSizeF(max(7.0, font.pointSizeF() - 1))
        painter.setFont(font)
        for frame in axis_ticks(self.track.f_lo, self.track.f_hi):
            x = self._x(frame, gutter, scale)
            painter.drawLine(QPointF(x, y + 2), QPointF(x, y + 6))
            painter.drawText(QRectF(x - 30, y + 7, 60, 16),
                             int(Qt.AlignHCenter | Qt.AlignTop), str(frame))
        painter.drawText(QRectF(2, y + 7, gutter - 8, 16),
                         int(Qt.AlignRight | Qt.AlignTop), "frame")

    # -- tooltips ------------------------------------------------------------ #
    def event(self, e):
        if e.type() == QEvent.ToolTip:
            hit = self._at(e.pos())
            if hit is None:
                QToolTip.hideText()
                e.ignore()
                return True
            QToolTip.showText(e.globalPos(), self._describe(*hit), self)
            return True
        return super().event(e)

    def _at(self, point):
        """``(group, segment)`` under a widget point, or None."""
        if self.note or self.track.empty:
            return None
        gutter, scale = self._geom
        if scale <= 0:
            return None
        lane = int((point.y() - self.TOP) // self.LANE)
        groups = list(self.track.lanes)
        if not 0 <= lane < len(groups):
            return None
        group = groups[lane]
        bar = self._bar(self.TOP + lane * self.LANE, gutter)
        frame = self._frame_float(point.x())
        slack = 1.0 / scale
        for segment in self._visible(group, frame - slack, frame + slack):
            if self._rect(segment, bar, gutter, scale).contains(QPointF(point)):
                return group, segment
        return None

    def _describe(self, group: str, segment: Segment) -> str:
        lines = [
            f"{group}: {segment.label or '(no prediction)'}",
            f"frames {segment.f_start}-{segment.f_end} ({segment.n_frames} frames)",
        ]
        if np.isfinite(segment.conf):
            lines.append(f"confidence {segment.conf:.3f}: the MEAN posterior over "
                         "the bout, not its peak")
        if segment.skip_reason:
            lines.append(segment.skip_reason)
        return "\n".join(lines)

    # -- scrubbing ----------------------------------------------------------- #
    def mousePressEvent(self, event):  # noqa: N802 (Qt naming)
        if event.button() != Qt.LeftButton or self.note or self.track.empty:
            super().mousePressEvent(event)
            return
        self._scrub = True
        self._pick(event.pos().x())

    def mouseMoveEvent(self, event):  # noqa: N802 (Qt naming)
        if self._scrub:
            self._pick(event.pos().x())
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802 (Qt naming)
        if self._scrub:
            self._scrub = False
            return
        super().mouseReleaseEvent(event)

    def _pick(self, x: float) -> None:
        frame = self._frame_at(x)
        self.set_position(frame)
        self.positionPicked.emit(frame)


def axis_ticks(f_lo: int, f_hi: int, target: int = 6) -> list[int]:
    """Round frame numbers across the span: 1, 2 or 5 times a power of ten."""
    span = max(1, f_hi - f_lo)
    raw = span / max(1, target)
    power = max(1, 10 ** max(0, int(np.floor(np.log10(max(raw, 1.0))))))
    step = power
    for multiple in (1, 2, 5, 10):
        step = multiple * power
        if span / step <= target:
            break
    first = int(np.ceil(f_lo / step) * step)
    return list(range(first, f_hi + 1, int(step)))
