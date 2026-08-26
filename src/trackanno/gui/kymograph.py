"""The curvature kymograph, drawn beside the clip it belongs to.

Time runs left to right and arclength top to bottom, base at the top and tip at
the bottom - the flagellar convention, under which a travelling wave reads as
diagonal stripes sloping down-right. Curvature is signed and its zero is
meaningful, so the colour scale is diverging: a neutral grey midpoint and the
accent and danger tokens as poles. Frames the tracker missed stay bare surface,
since painting a gap as the midpoint colour would show a straight flagellum
where there was no measurement.
"""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import QPointF, QRectF, QSize, Qt, Signal
from qtpy.QtGui import QColor, QFont, QImage, QPainter, QPen
from qtpy.QtWidgets import QSizePolicy, QWidget

from ..measure.kymograph import Kymograph, canonical_sign
from .theme import current, tokens

__all__ = ["KymographView", "diverging_ramp"]

# The neutral midpoint, per theme. Not a token: it has to sit clear of the
# surface in lightness so "zero curvature" and "no data" cannot be confused.
MIDPOINT = {"dark": (0x45, 0x4C, 0x59), "light": (0xD3, 0xD8, 0xE0)}

MARGIN_LEFT = 46
MARGIN_RIGHT = 58
MARGIN_TOP = 20
MARGIN_BOTTOM = 26


def _linear(c: float) -> float:
    c /= 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _srgb(c: float) -> float:
    c = 0.0 if c < 0.0 else (1.0 if c > 1.0 else c)
    v = 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1 / 2.4)) - 0.055
    return v * 255.0


def diverging_ramp(steps: int = 256, theme: str | None = None) -> np.ndarray:
    """(steps, 3) uint8 from the negative pole through neutral to the positive.

    Mixed through linear light, not straight sRGB, where the interpolation dips
    in luminance and leaves a dark band either side of zero."""
    t = tokens()
    theme = theme or current()
    negative = QColor(t["accent"])
    positive = QColor(t["danger"])
    mid = MIDPOINT.get(theme, MIDPOINT["dark"])

    lo = np.array([_linear(negative.red()), _linear(negative.green()), _linear(negative.blue())])
    mi = np.array([_linear(mid[0]), _linear(mid[1]), _linear(mid[2])])
    hi = np.array([_linear(positive.red()), _linear(positive.green()), _linear(positive.blue())])

    half = steps // 2
    lower = np.linspace(0.0, 1.0, half)[:, None] * (mi - lo) + lo
    upper = np.linspace(0.0, 1.0, steps - half)[:, None] * (hi - mi) + mi
    linear = np.vstack([lower, upper])
    return np.clip([[_srgb(v) for v in row] for row in linear], 0, 255).astype(np.uint8)


class KymographView(QWidget):
    """Paints one clip's curvature field, with a cursor on the current frame."""

    frame_picked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.kymo: Kymograph | None = None
        self.frame = 0
        self.normalised = True
        self.interpolate = True
        self.field_kind = "curvature"
        self.sign_mode = "canonical"
        self.verdict = None
        self._verdict_for = None
        self._image: QImage | None = None
        self._image_key = None
        self._limit = 1.0
        self._hover: tuple[int, int] | None = None
        self.setMinimumHeight(150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setToolTip(
            "Curvature of the flagellum along its length (top = base, bottom = "
            "tip) against time.\nA travelling wave appears as diagonal stripes; "
            "blank columns are frames the tracker missed."
        )

    # -- what is actually drawn ---------------------------------------------- #
    def field(self):
        """The (frames, arclength) array to paint, after field and sign choices.
        Both are applied here rather than baked into the Kymograph, so switching
        either one repaints without refitting the splines."""
        if self.kymo is None:
            return None
        if self.field_kind == "tangent" and self.kymo.psi is not None:
            data = np.asarray(self.kymo.psi, dtype=np.float32)
        else:
            data = self.kymo.normalised() if self.normalised else self.kymo.kappa

        # The sign is judged on the curvature even when the tangent angle is on
        # screen: the static mode deciding it is a property of the cell, not of
        # the selected view. Cached against the clip, since field() runs on every
        # repaint and mouse move.
        if self.verdict is None or self._verdict_for is not self.kymo:
            self.verdict = canonical_sign(self.kymo.kappa)
            self._verdict_for = self.kymo
        if self.sign_mode == "absolute":
            return np.abs(data)
        if self.sign_mode == "canonical" and self.verdict.confident:
            return data * self.verdict.sign
        return data

    def set_field_kind(self, kind: str) -> None:
        if kind != self.field_kind:
            self.field_kind = kind
            self._image = None
            self.update()

    def set_sign_mode(self, mode: str) -> None:
        if mode != self.sign_mode:
            self.sign_mode = mode
            self._image = None
            self.update()

    # -- data ---------------------------------------------------------------- #
    def set_kymograph(self, kymo: Kymograph | None) -> None:
        self.kymo = kymo
        self.verdict = None
        self._verdict_for = None
        self._image = None
        self._image_key = None
        self.update()

    def set_frame(self, t: int) -> None:
        t = int(t)
        if t != self.frame:
            self.frame = t
            self.update()

    def set_normalised(self, on: bool) -> None:
        if bool(on) != self.normalised:
            self.normalised = bool(on)
            self._image = None
            self.update()

    def set_interpolate(self, on: bool) -> None:
        """Filtered display, or the raw sample grid."""
        if bool(on) != self.interpolate:
            self.interpolate = bool(on)
            self.update()

    def clear(self) -> None:
        self.set_kymograph(None)

    def restyle(self) -> None:
        self._image = None
        self.update()

    # -- geometry ------------------------------------------------------------ #
    def _plot_rect(self) -> QRectF:
        return QRectF(
            MARGIN_LEFT, MARGIN_TOP,
            max(10.0, self.width() - MARGIN_LEFT - MARGIN_RIGHT),
            max(10.0, self.height() - MARGIN_TOP - MARGIN_BOTTOM),
        )

    def sizeHint(self) -> QSize:
        return QSize(420, 190)

    # -- the field as an image ----------------------------------------------- #
    def _build_image(self) -> QImage | None:
        if self.kymo is None or not self.kymo.usable:
            return None
        field = self.field()
        if field is None or not np.isfinite(field).any():
            return None
        self._limit = float(np.percentile(np.abs(field[np.isfinite(field)]), 98.0)) or 1.0
        if self._limit <= 1e-9:
            self._limit = 1.0

        ramp = diverging_ramp(256)
        # Transposed to (rows = arclength, cols = time); stored frame-major.
        grid = field.T
        finite = np.isfinite(grid)
        scaled = np.clip(np.nan_to_num(grid) / self._limit, -1.0, 1.0)
        index = ((scaled + 1.0) * 0.5 * (len(ramp) - 1)).astype(np.int32)
        rgb = ramp[index]

        height, width = grid.shape
        rgba = np.empty((height, width, 4), dtype=np.uint8)
        rgba[..., 0] = rgb[..., 2]                      # QImage wants BGRA
        rgba[..., 1] = rgb[..., 1]
        rgba[..., 2] = rgb[..., 0]
        rgba[..., 3] = np.where(finite, 255, 0)         # gaps stay transparent

        # The buffer must outlive the QImage: QImage does not copy it.
        self._buffer = np.ascontiguousarray(rgba)
        return QImage(self._buffer.data, width, height, 4 * width,
                      QImage.Format_ARGB32).copy()

    # -- painting ------------------------------------------------------------ #
    def paintEvent(self, event):  # noqa: N802 (Qt naming)
        t = tokens()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        plot = self._plot_rect()

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(t["surface"]))
        painter.drawRect(self.rect())

        if self.kymo is None or not self.kymo.usable:
            painter.setPen(QColor(t["faint"]))
            painter.drawText(
                self.rect(), Qt.AlignCenter,
                "no flagellum tracked in this clip"
                if self.kymo is not None else "no keypoints",
            )
            return

        key = (self.width(), self.height(), current(), self.normalised,
               self.field_kind, self.sign_mode, id(self.kymo))
        if self._image is None or self._image_key != key:
            self._image = self._build_image()
            self._image_key = key
        if self._image is None:
            return

        # The plot ground, so a transparent gap reads as a hole.
        painter.setBrush(QColor(t["raised"]))
        painter.drawRect(plot)
        # Filtered: both axes sample a continuous quantity, so smoothing invents
        # nothing. Switchable, for a noisy track.
        painter.setRenderHint(QPainter.SmoothPixmapTransform, self.interpolate)
        painter.drawImage(plot, self._image)

        painter.setPen(QPen(QColor(t["border"]), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(plot)

        self._draw_axes(painter, plot, t)
        self._draw_cursor(painter, plot, t)
        self._draw_scale(painter, plot, t)
        painter.end()

    def _draw_axes(self, painter, plot, t) -> None:
        small = QFont(self.font())
        small.setPointSizeF(max(7.5, small.pointSizeF() - 2))
        painter.setFont(small)
        painter.setPen(QColor(t["muted"]))

        painter.save()
        painter.translate(12, plot.center().y())
        painter.rotate(-90)
        painter.drawText(QRectF(-60, -12, 120, 14), int(Qt.AlignCenter), "arclength")
        painter.restore()

        painter.setPen(QColor(t["faint"]))
        painter.drawText(QRectF(20, plot.top() - 2, 24, 12),
                         int(Qt.AlignRight | Qt.AlignVCenter), "base")
        painter.drawText(QRectF(20, plot.bottom() - 10, 24, 12),
                         int(Qt.AlignRight | Qt.AlignVCenter), "tip")

        n = self.kymo.n_frames
        painter.setPen(QColor(t["muted"]))
        painter.drawText(
            QRectF(plot.left(), plot.bottom() + 4, plot.width(), 14),
            int(Qt.AlignCenter), f"time  (frame 1 to {n})",
        )

    def _draw_cursor(self, painter, plot, t) -> None:
        n = self.kymo.n_frames
        if n <= 0:
            return
        frame = min(max(self.frame, 0), n - 1)
        x = plot.left() + (frame + 0.5) / n * plot.width()
        painter.setPen(QPen(QColor(t["text"]), 1))
        painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
        # A tick outside the plot too, for where the line sits over a pale field.
        painter.setPen(QPen(QColor(t["text"]), 2))
        painter.drawLine(QPointF(x, plot.bottom()), QPointF(x, plot.bottom() + 3))

    def _draw_scale(self, painter, plot, t) -> None:
        """The colour key. A continuous encoding needs one to be readable."""
        bar = QRectF(plot.right() + 12, plot.top(), 10, plot.height())
        ramp = diverging_ramp(128)
        step = bar.height() / len(ramp)
        painter.setPen(Qt.NoPen)
        for i, (r, g, b) in enumerate(ramp[::-1]):     # positive at the top
            painter.setBrush(QColor(int(r), int(g), int(b)))
            painter.drawRect(QRectF(bar.left(), bar.top() + i * step,
                                    bar.width(), step + 0.6))
        painter.setPen(QPen(QColor(t["border"]), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(bar)

        small = QFont(self.font())
        small.setPointSizeF(max(7.0, small.pointSizeF() - 2.5))
        painter.setFont(small)
        painter.setPen(QColor(t["muted"]))
        unit = "" if self.normalised else " /px"
        for value, y, align in (
            (f"+{self._limit:.2g}", bar.top() - 1, Qt.AlignTop),
            ("0", bar.center().y() - 6, Qt.AlignVCenter),
            (f"-{self._limit:.2g}", bar.bottom() - 11, Qt.AlignBottom),
        ):
            painter.drawText(QRectF(bar.right() + 3, y, MARGIN_RIGHT - 26, 12),
                             int(Qt.AlignLeft | Qt.AlignVCenter), value)
        if self.field_kind == "tangent":
            label = "psi"
        else:
            label = "kL" if self.normalised else f"k{unit}"
        painter.drawText(
            QRectF(bar.right() + 3, bar.center().y() + 6, MARGIN_RIGHT - 26, 12),
            int(Qt.AlignLeft | Qt.AlignVCenter), label,
        )
        # Say when the handedness could not be decided, so the colours do not
        # imply a convention the data never established.
        if self.sign_mode == "canonical" and self.verdict is not None:
            if not self.verdict.confident:
                painter.setPen(QColor(t["warning"]))
                painter.drawText(
                    QRectF(plot.left(), plot.top() - MARGIN_TOP + 2, plot.width(), 14),
                    int(Qt.AlignRight | Qt.AlignVCenter), "sign not determined",
                )
            elif self.verdict.flipped:
                painter.setPen(QColor(t["faint"]))
                painter.drawText(
                    QRectF(plot.left(), plot.top() - MARGIN_TOP + 2, plot.width(), 14),
                    int(Qt.AlignRight | Qt.AlignVCenter), "sign flipped to match",
                )

    # -- interaction --------------------------------------------------------- #
    def mouseMoveEvent(self, event):  # noqa: N802 (Qt naming)
        if self.kymo is None or not self.kymo.usable:
            return
        plot = self._plot_rect()
        pos = event.position() if hasattr(event, "position") else event.posF()
        if not plot.contains(pos):
            return
        n, s = self.kymo.n_frames, self.kymo.n_samples
        frame = min(n - 1, max(0, int((pos.x() - plot.left()) / plot.width() * n)))
        node = min(s - 1, max(0, int((pos.y() - plot.top()) / plot.height() * s)))
        field = self.field()
        value = field[frame, node]
        arc = node / max(1, s - 1)
        text = "not tracked" if not np.isfinite(value) else (
            f"{value:+.3g}" + ("" if self.normalised else " /px")
        )
        self.setToolTip(f"frame {frame + 1}   s = {arc:.2f}   {text}")

    def mousePressEvent(self, event):  # noqa: N802 (Qt naming)
        """Click to scrub: the chart is a timeline as much as a picture."""
        if self.kymo is None or not self.kymo.n_frames:
            return
        plot = self._plot_rect()
        pos = event.position() if hasattr(event, "position") else event.posF()
        if not plot.contains(pos):
            return
        n = self.kymo.n_frames
        frame = min(n - 1, max(0, int((pos.x() - plot.left()) / plot.width() * n)))
        self.set_frame(frame)
        self.frame_picked.emit(frame)
