"""Display contrast for the clip player: limits and gamma applied when a frame is painted.

``auto: clip`` sets the limits from percentiles of the whole clip, ``auto: frame`` from each
frame, and ``fixed`` keeps the limits set on the histogram for every clip. Clips are uint8.
"""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import QPointF, Qt, Signal
from qtpy.QtGui import QColor, QPainter, QPainterPath, QPen
from qtpy.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QSlider, QWidget

from .theme import tokens
from .viewer import choice_combo

__all__ = ["CONTRAST_MODES", "ContrastBar", "counts_of", "limits_of", "lut"]

_MODES = {
    "clip": ("auto: clip", "Limits from percentiles of the whole clip."),
    "frame": ("auto: frame", "Limits from percentiles of each frame."),
    "fixed": ("fixed", "The limits set on the histogram, kept for every clip."),
}
CONTRAST_MODES = tuple(_MODES)
AUTO_PERCENTILES = (0.5, 99.5)
GAMMA_MAX = 5.0           # the slider spans 1 / GAMMA_MAX to GAMMA_MAX
_GAMMA_STEPS = 100        # slider steps per side of gamma 1
_SAMPLE = 1 << 18         # pixels; larger arrays are subsampled for statistics


# --------------------------------------------------------------------------- #
# mapping
# --------------------------------------------------------------------------- #
def counts_of(pixels) -> np.ndarray:
    """256-bin histogram of uint8 ``pixels``, without the zero padding outside the movie frame."""
    a = np.asarray(pixels)
    if a.size > _SAMPLE:
        step = int(np.ceil(np.sqrt(a.size / _SAMPLE)))
        a = a[..., ::step, ::step]
    counts = np.bincount(a.ravel(), minlength=256)
    if counts[1:].any():
        counts[0] = 0
    return counts


def _apart(lo: float, hi: float) -> tuple[float, float]:
    """``lo`` and ``hi`` at least one level apart, within 0..255."""
    if hi > lo:
        return lo, hi
    return (lo, lo + 1.0) if lo < 255 else (254.0, 255.0)


def limits_of(counts: np.ndarray, percentiles=AUTO_PERCENTILES) -> tuple[float, float]:
    """Percentile limits from a 256-bin histogram; min and max when those coincide."""
    cum = np.cumsum(counts)
    if cum[-1] == 0:
        return 0.0, 255.0
    lo, hi = (float(np.searchsorted(cum, cum[-1] * p / 100.0)) for p in percentiles)
    if hi <= lo:
        present = np.flatnonzero(counts)
        lo, hi = float(present[0]), float(present[-1])
    return _apart(lo, hi)


def lut(lo: float, hi: float, gamma: float = 1.0) -> np.ndarray:
    """uint8 lookup table: ``lo`` to 0, ``hi`` to 255, then raised to ``gamma``."""
    x = (np.arange(256, dtype=np.float32) - np.float32(lo)) / np.float32(max(hi - lo, 1e-6))
    np.clip(x, 0.0, 1.0, out=x)
    if gamma != 1.0:
        np.power(x, np.float32(gamma), out=x)
    return (x * 255.0 + 0.5).astype(np.uint8)


def _gamma_of(step: int) -> float:
    return float(GAMMA_MAX ** (step / _GAMMA_STEPS))


def _step_of(gamma: float) -> int:
    return int(round(np.log(gamma) / np.log(GAMMA_MAX) * _GAMMA_STEPS))


# --------------------------------------------------------------------------- #
# histogram
# --------------------------------------------------------------------------- #
class _Histogram(QWidget):
    """The clip's histogram over 0..255 with the display window on top.

    Dragging a limit moves it, dragging between the limits moves both.
    """

    moved = Signal(float, float)     # lo, hi
    MARGIN = 6
    GRAB = 6

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(30)
        self.setMinimumWidth(200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFocusPolicy(Qt.NoFocus)
        self.setCursor(Qt.SizeHorCursor)
        self._heights = np.zeros(0)
        self._lo, self._hi, self._gamma = 0.0, 255.0, 1.0
        self._drag: str | None = None
        self._grab = (0.0, 0.0, 0.0)

    def set_counts(self, counts: np.ndarray) -> None:
        logged = np.log1p(np.asarray(counts, dtype=float))
        peak = float(logged.max()) if logged.size else 0.0
        self._heights = logged / peak if peak > 0 else logged
        self.update()

    def set_window(self, lo: float, hi: float, gamma: float) -> None:
        self._lo, self._hi, self._gamma = float(lo), float(hi), float(gamma)
        self.update()

    # -- coordinates -------------------------------------------------------- #
    def _width(self) -> float:
        return max(1.0, self.width() - 2 * self.MARGIN)

    def _x(self, value: float) -> float:
        return self.MARGIN + value / 255.0 * self._width()

    def _value(self, x: float) -> float:
        return (x - self.MARGIN) / self._width() * 255.0

    # -- painting ----------------------------------------------------------- #
    def paintEvent(self, event):  # noqa: N802 (Qt naming)
        t = tokens()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(t["canvas"]))
        top, bottom = 3.0, self.height() - 3.0
        left, right = float(self.MARGIN), float(self.width() - self.MARGIN)

        n = self._heights.size
        if n:
            path = QPainterPath(QPointF(left, bottom))
            step = (right - left) / n
            for i, h in enumerate(self._heights):
                y = bottom - h * (bottom - top)
                path.lineTo(QPointF(left + i * step, y))
                path.lineTo(QPointF(left + (i + 1) * step, y))
            path.lineTo(QPointF(right, bottom))
            path.closeSubpath()
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(t["faint"]))
            painter.drawPath(path)
            painter.setBrush(Qt.NoBrush)

        xl, xh = self._x(self._lo), self._x(self._hi)
        curve = QPainterPath(QPointF(left, bottom))
        curve.lineTo(QPointF(xl, bottom))
        for i in range(1, 33):
            f = i / 32
            curve.lineTo(QPointF(xl + f * (xh - xl), bottom - f ** self._gamma * (bottom - top)))
        curve.lineTo(QPointF(right, top))
        painter.setPen(QPen(QColor(t["accent"]), 1.5))
        painter.drawPath(curve)

        painter.setPen(QPen(QColor(t["accent_hi"]), 1.0))
        for x in (xl, xh):
            painter.drawLine(QPointF(x, top - 2), QPointF(x, bottom + 2))
        painter.end()

    # -- dragging ----------------------------------------------------------- #
    @staticmethod
    def _event_x(event) -> float:
        return float(event.position().x()) if hasattr(event, "position") else float(event.x())

    def mousePressEvent(self, event):  # noqa: N802 (Qt naming)
        if event.button() != Qt.LeftButton:
            return
        x = self._event_x(event)
        xl, xh = self._x(self._lo), self._x(self._hi)
        if min(abs(x - xl), abs(x - xh)) <= self.GRAB:
            self._drag = "lo" if abs(x - xl) <= abs(x - xh) else "hi"
        elif xl < x < xh:
            self._drag = "both"
        else:
            self._drag = "lo" if x < xl else "hi"
        self._grab = (self._value(x), self._lo, self._hi)
        if self._drag != "both":
            self._move_to(x)

    def mouseMoveEvent(self, event):  # noqa: N802 (Qt naming)
        if self._drag is not None:
            self._move_to(self._event_x(event))

    def mouseReleaseEvent(self, event):  # noqa: N802 (Qt naming)
        self._drag = None

    def _move_to(self, x: float) -> None:
        value = min(max(self._value(x), 0.0), 255.0)
        lo, hi = self._lo, self._hi
        if self._drag == "lo":
            lo = min(value, hi - 1.0)
        elif self._drag == "hi":
            hi = max(value, lo + 1.0)
        else:
            start, lo0, hi0 = self._grab
            shift = min(max(value - start, -lo0), 255.0 - hi0)
            lo, hi = lo0 + shift, hi0 + shift
        self.moved.emit(lo, hi)


# --------------------------------------------------------------------------- #
# the bar
# --------------------------------------------------------------------------- #
class ContrastBar(QWidget):
    """Histogram, auto mode and gamma for the clip on screen; ``mapper()`` paints frames."""

    changed = Signal()

    def __init__(self, mode: str = "clip", gamma: float = 1.0, parent=None):
        super().__init__(parent)
        self.mode = mode if mode in CONTRAST_MODES else "clip"
        self.gamma = float(min(max(gamma, 1.0 / GAMMA_MAX), GAMMA_MAX))
        self.lo, self.hi = 0.0, 255.0
        self._array: np.ndarray | None = None
        self._clip_limits = (0.0, 255.0)
        self._frame_limits: dict[int, tuple[float, float]] = {}
        self._frame = 0
        # Raw intensity of 0 and 255, for the readout.
        self._units: tuple[float, float] | None = None
        # Whether a clip has set the limits; fixed mode keeps them from then on.
        self._seeded = False

        self.histogram = _Histogram()
        self.histogram.setToolTip(
            "Contrast: drag a limit, or between the limits to move both. "
            "Dragging fixes the limits.")
        self.histogram.moved.connect(self._on_dragged)

        self.readout = QLabel("")
        self.readout.setProperty("role", "muted")
        self.readout.setMinimumWidth(80)
        self.readout.setAlignment(Qt.AlignCenter)

        self.mode_combo = choice_combo(_MODES, self.mode)
        self.mode_combo.currentIndexChanged.connect(self._on_mode)

        self.gamma_label = QLabel("")
        self.gamma_label.setMinimumWidth(64)
        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setRange(-_GAMMA_STEPS, _GAMMA_STEPS)
        self.gamma_slider.setValue(_step_of(self.gamma))
        self.gamma_slider.setFixedWidth(90)
        self.gamma_slider.setFocusPolicy(Qt.NoFocus)
        self.gamma_slider.setToolTip("Gamma: below 1 brightens dim pixels, above 1 darkens them.")
        self.gamma_slider.valueChanged.connect(self._on_gamma)

        self.reset_btn = QPushButton("reset")
        self.reset_btn.setFocusPolicy(Qt.NoFocus)
        self.reset_btn.setToolTip("Back to auto: clip and gamma 1.")
        self.reset_btn.clicked.connect(self.reset)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 2, 0, 0)
        row.addWidget(self.histogram, 1)
        row.addWidget(self.readout)
        row.addWidget(self.mode_combo)
        row.addSpacing(8)
        row.addWidget(self.gamma_label)
        row.addWidget(self.gamma_slider)
        row.addWidget(self.reset_btn)
        self._sync()

    # -- content ------------------------------------------------------------ #
    def set_clip(self, array: np.ndarray, intensity: dict | None = None) -> None:
        """Take the histogram and, in an auto mode, the limits from the uint8 ``array``.

        ``intensity`` (``{"lo", "hi"}``, raw values of 0 and 255) sets the readout units.
        """
        self._array = array
        self._units = (float(intensity["lo"]), float(intensity["hi"])) if intensity else None
        self._frame = 0
        self._frame_limits = {}
        counts = counts_of(array)
        self._clip_limits = limits_of(counts)
        if self.mode == "frame":
            self.lo, self.hi = self._limits_at(0)
        elif self.mode == "clip" or not self._seeded:
            self.lo, self.hi = self._clip_limits
        self._seeded = True
        self.histogram.set_counts(counts)
        self._sync()
        self.changed.emit()

    def clear(self) -> None:
        """Drop the clip so nothing keeps the cache's memory map alive."""
        self._array = None
        self._frame_limits = {}

    def set_frame(self, t: int) -> None:
        """Follow the frame on screen; in ``auto: frame`` the window shows its limits."""
        self._frame = int(t)
        if self.mode == "frame" and self._array is not None:
            self.lo, self.hi = self._limits_at(self._frame)
            self._sync()

    def mapper(self):
        """``(frame, t) -> uint8`` for the current settings."""
        gamma = self.gamma
        if self.mode == "frame":
            return lambda frame, t: lut(*self._limits_at(t), gamma)[frame]
        table = lut(self.lo, self.hi, gamma)
        return lambda frame, t: table[frame]

    def reset(self) -> None:
        """Back to ``auto: clip`` and gamma 1."""
        self.gamma_slider.setValue(0)
        self.mode_combo.setCurrentIndex(CONTRAST_MODES.index("clip"))

    # -- internals ---------------------------------------------------------- #
    def _limits_at(self, t: int) -> tuple[float, float]:
        if t not in self._frame_limits:
            self._frame_limits[t] = limits_of(counts_of(self._array[t]))
        return self._frame_limits[t]

    def _fmt(self, value: float) -> str:
        if self._units is not None:
            lo, hi = self._units
            value = lo + value * (hi - lo) / 255.0
        return f"{value:.0f}" if abs(value) >= 10 else f"{value:.3g}"

    def _sync(self) -> None:
        self.histogram.set_window(self.lo, self.hi, self.gamma)
        self.readout.setText(f"{self._fmt(self.lo)} - {self._fmt(self.hi)}")
        self.gamma_label.setText(f"gamma {self.gamma:.2f}")

    def _on_dragged(self, lo: float, hi: float) -> None:
        self.lo, self.hi = _apart(float(round(lo)), float(round(hi)))
        if self.mode != "fixed":
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentIndex(CONTRAST_MODES.index("fixed"))
            self.mode_combo.blockSignals(False)
            self.mode = "fixed"
        self._sync()
        self.changed.emit()

    def _on_mode(self, _index: int) -> None:
        self.mode = self.mode_combo.currentData()
        if self.mode == "clip":
            self.lo, self.hi = self._clip_limits
        elif self.mode == "frame" and self._array is not None:
            self.lo, self.hi = self._limits_at(self._frame)
        self._sync()
        self.changed.emit()

    def _on_gamma(self, step: int) -> None:
        self.gamma = _gamma_of(int(step))
        self._sync()
        self.changed.emit()
