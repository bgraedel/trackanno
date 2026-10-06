"""One track's predicted bouts along the frame axis, beside the footage.

A row of ``predictions.parquet`` is a decoded bout: a variable-length run
between two inferred behavioural boundaries. The panel is read-only.

Every frame of the footage is cut from the movie, at one crop for the whole
session and centred on the track's own boxes, never from the clip cache.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from qtpy.QtCore import (
    QEvent, QObject, QPointF, QRectF, QSize, Qt, QTimer, Signal,
)
from qtpy.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPolygonF
from qtpy.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QScrollArea, QSizePolicy,
    QSplitter, QToolTip, QVBoxLayout, QWidget,
)

from ..config import Config
from ..view.overlay import build_clip_view
from ..annotate.store import MULTI_SEP
from .theme import help_button, tokens
from .viewer import SimpleViewer, follow_combo

__all__ = [
    "Segment", "TrackPredictions", "PredictionReader",
    "ReviewCrop", "review_crop", "MovieFrames", "TrackFootage",
    "BoutTimeline", "BoutReviewPanel",
]

# What the review window opens on, rather than ui.follow. That setting is chosen
# for one 25-frame clip, where a moving average removes drift and keeps the beat.
# A bout runs for hundreds of frames, and over that long the average lags and the
# animal walks out of the middle of the picture; `locked` re-centres every frame.
REVIEW_FOLLOW = "locked"


@dataclass(frozen=True)
class Segment:
    """One decoded bout.

    ``label`` is ``""`` for a stretch the model has no opinion about, drawn as a
    hole rather than dropped. ``conf`` is NaN where the file does not record it.
    """

    f_start: int
    f_end: int
    label: str
    conf: float
    skip_reason: str

    @property
    def n_frames(self) -> int:
        return self.f_end - self.f_start


@dataclass
class TrackPredictions:
    """Everything one (movie, track) has to say, ready to paint."""

    movie_id: str
    track_id: int
    f_lo: int = 0
    f_hi: int = 0
    lanes: dict = field(default_factory=dict)        # group -> [Segment]

    @property
    def empty(self) -> bool:
        return not any(self.lanes.values())

    @property
    def n_segments(self) -> int:
        return sum(len(v) for v in self.lanes.values())


# --------------------------------------------------------------------------- #
# reading the files
# --------------------------------------------------------------------------- #
def _labelled(values: pd.Series) -> pd.Series:
    """Which cells hold a real class name.

    An absent prediction reads back out of parquet as ``""`` or as NaN whose
    ``str()`` is ``"nan"``; neither counts.
    """
    text = values.astype(str).str.strip()
    return text.ne("") & text.str.lower().ne("nan")


class PredictionReader:
    """``predictions.parquet``, read once per open and then queried per track.

    ``problem`` is set instead of raising, because a traceback out of a Qt slot
    takes the window down.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.problem = ""
        self.frame = pd.DataFrame()
        self.groups: list[str] = []
        self.movies: list[str] = []
        self.stale = ""
        self._tracks: dict[str, list[tuple[int, int]]] = {}

        from ..infer.predict import load_predictions, staleness

        try:
            self.frame = load_predictions(cfg)
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            self.problem = str(exc)
            return

        if "bout_id" not in self.frame.columns:
            self.problem = (
                "these predictions are not decoded bouts; label every clip with "
                "the model again"
            )
            return
        self.stale = staleness(cfg)
        self.groups = sorted(
            c[: -len("_pred")] for c in self.frame.columns if c.endswith("_pred")
        )
        if not self.groups:
            self.problem = (
                "the predictions carry no <group>_pred column at all, so there is "
                "nothing in them to review; re-run 'Label all clips with the model'"
            )
            return
        self.movies = sorted(self.frame["movie_id"].astype(str).unique())

    # -- what can be selected ------------------------------------------------ #
    def tracks(self, movie_id: str) -> list[tuple[int, int]]:
        """``[(track_id, n_rows)]`` for one movie, predicted tracks and silent ones.

        A track the run produced no row for is a normal outcome, so the list is
        the union with ``tracks.parquet``: those tracks appear with a count of 0
        and draw an empty timeline that says so.
        """
        movie_id = str(movie_id)
        cached = self._tracks.get(movie_id)
        if cached is not None:
            return cached

        counts: dict[int, int] = {}
        rows = self.frame[self.frame["movie_id"].astype(str) == movie_id]
        for track_id, n in rows["track_id"].value_counts().items():
            counts[int(track_id)] = int(n)
        for track_id in self._detected_tracks(movie_id):
            counts.setdefault(track_id, 0)
        out = [(t, counts[t]) for t in sorted(counts)]
        self._tracks[movie_id] = out
        return out

    def _detected_tracks(self, movie_id: str) -> list[int]:
        """Track ids from ``tracks.parquet``, or [] if it cannot be read.

        Tolerant on purpose: this only ever ADDS silent tracks to a list the
        predictions already filled.
        """
        from ..annotate.clips import TRACKS_FILE

        path = self.cfg.movie_dir(movie_id) / TRACKS_FILE
        if not path.exists():
            return []
        try:
            table = pd.read_parquet(path, columns=["track_id"])
        except (OSError, ValueError, KeyError):
            return []
        ids = pd.to_numeric(table["track_id"], errors="coerce").dropna()
        return sorted({int(t) for t in ids})

    # -- one track ----------------------------------------------------------- #
    def track(self, movie_id: str, track_id: int) -> TrackPredictions:
        out = TrackPredictions(movie_id=str(movie_id), track_id=int(track_id))
        rows = self.frame[
            (self.frame["movie_id"].astype(str) == str(movie_id))
            & (pd.to_numeric(self.frame["track_id"], errors="coerce") == int(track_id))
        ]
        if rows.empty:
            return out

        starts = pd.to_numeric(rows["f_start"], errors="coerce")
        ends = pd.to_numeric(rows["f_end"], errors="coerce")
        out.f_lo = int(starts.min())
        out.f_hi = int(ends.max())
        for group in self.groups:
            out.lanes[group] = self._segments(rows, group)
        return out

    def _segments(self, rows: pd.DataFrame, group: str) -> list[Segment]:
        """One group's bouts out of one track's rows, in frame order."""
        rows = rows[rows["group"].astype(str) == group]
        if rows.empty:
            return []

        blank = pd.Series(np.nan, index=rows.index)
        pred = rows[f"{group}_pred"].astype(str).str.strip()
        known = _labelled(pred).to_numpy()
        conf = pd.to_numeric(
            rows[f"{group}_conf"] if f"{group}_conf" in rows.columns else blank,
            errors="coerce",
        ).to_numpy(dtype=float)
        starts = pd.to_numeric(rows["f_start"], errors="coerce").to_numpy(dtype=float)
        ends = pd.to_numeric(rows["f_end"], errors="coerce").to_numpy(dtype=float)
        reason = (rows["skip_reason"].astype(str) if "skip_reason" in rows.columns
                  else pd.Series("", index=rows.index)).to_numpy()
        names = pred.to_numpy()

        out = []
        for i in range(len(rows)):
            # Dropped, never defaulted: a guessed span would put a coloured bar
            # over frames nothing scored.
            if not (np.isfinite(starts[i]) and np.isfinite(ends[i])):
                continue
            text = str(reason[i])
            out.append(Segment(
                f_start=int(starts[i]), f_end=int(ends[i]),
                label=str(names[i]) if known[i] else "",
                conf=float(conf[i]) if known[i] else float("nan"),
                skip_reason="" if text.lower() == "nan" else text,
            ))
        out.sort(key=lambda s: (s.f_start, s.f_end))
        return out


# --------------------------------------------------------------------------- #
# what the picture cannot say for itself
# --------------------------------------------------------------------------- #
REVIEW_HELP = """
<h3>What the lanes are</h3>
<p>Each bar is one decoded <b>bout</b>: a run of frames between two inferred
behavioural boundaries. Bar height is the bout's mean posterior for its own
class, so a marginal bout is a <i>short</i> bar, never a pale one.</p>

<p>A hatched stretch is one no window could score; hover it for the reason.</p>

<h3>The video</h3>
<p>Click or drag on the lanes to move the video there. Every frame is cut from
the movie, never from the clip cache, at one crop for the whole session. It has
no scrubber of its own: the lanes are the frame axis, and they span the whole
track.</p>
"""


# --------------------------------------------------------------------------- #
# painting
# --------------------------------------------------------------------------- #
class BoutTimeline(QWidget):
    """One track's lanes on a shared frame axis, one lane per predicted group."""

    LANE = 52
    BAR = 34
    TOP = 10
    AXIS = 26
    RIGHT = 12

    # The frame the reviewer pointed at, so the video can go there. Emitted on
    # press and on every drag.
    positionPicked = Signal(int)

    def __init__(self, cfg: Config | None = None, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.track = TrackPredictions("", 0)
        self.note = ""
        # The frame the video is showing, or None before it shows one.
        self.position: int | None = None
        self._scrub = False
        self._hits: list[tuple[QRectF, str, Segment]] = []
        # Set on every paint so the mouse handlers measure with the same ruler
        # the picture was drawn with.
        self._geom: tuple[float, float] = (0.0, 0.0)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumWidth(420)
        self.setMouseTracking(True)

    def set_track(self, track: TrackPredictions, note: str = "") -> None:
        self.track = track
        self.note = note
        self.position = None
        self._scrub = False
        self.updateGeometry()
        self.update()

    def set_position(self, frame: int | None) -> None:
        """Where the video is. Drawn, never emitted: this is the widget being
        told, and telling it back would be a loop."""
        frame = None if frame is None else int(frame)
        if frame == self.position:
            return
        self.position = frame
        self.update()

    def sizeHint(self) -> QSize:
        lanes = max(1, len(self.track.lanes))
        return QSize(760, self.TOP + lanes * self.LANE + self.AXIS + 8)

    def minimumSizeHint(self) -> QSize:
        """As narrow as ``minimumWidth`` allows, and no shorter than its lanes,
        which are drawn at fixed pixel sizes."""
        return QSize(self.minimumWidth(), self.sizeHint().height())

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

    def _frame_at(self, x: float) -> int:
        """The frame under a pixel, clamped to the track."""
        gutter, scale = self._geom
        if scale <= 0:
            return self.track.f_lo
        frame = self.track.f_lo + (x - gutter) / scale
        return int(min(max(round(frame), self.track.f_lo), self.track.f_hi))

    # -- painting ------------------------------------------------------------ #
    def paintEvent(self, event):  # noqa: N802 (Qt naming)
        t = tokens()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        self._hits = []

        if self.note or self.track.empty:
            painter.setPen(QColor(t["muted"]))
            painter.drawText(
                self.rect().adjusted(16, 0, -16, 0),
                int(Qt.AlignCenter | Qt.TextWordWrap),
                self.note or self._nothing_here(),
            )
            painter.end()
            return

        metrics = painter.fontMetrics()
        gutter = self._gutter(metrics)
        scale = self._scale(gutter)
        self._geom = (gutter, scale)
        y = float(self.TOP)
        for group, segments in self.track.lanes.items():
            self._paint_lane(painter, t, group, segments, y, gutter, scale)
            y += self.LANE
        self._paint_axis(painter, t, y, gutter, scale)
        self._paint_position(painter, t, y, gutter, scale)
        painter.end()

    def _paint_position(self, painter, t, bottom, gutter, scale) -> None:
        """The frame the video is on, across every lane at once, over the bars."""
        if self.position is None or not self.track.f_hi > self.track.f_lo:
            return
        if not self.track.f_lo <= self.position <= self.track.f_hi:
            return
        x = self._x(self.position, gutter, scale)
        painter.setPen(QPen(QColor(t["accent"]), 1))
        painter.drawLine(QPointF(x, self.TOP - 6), QPointF(x, bottom))
        head = QPolygonF([QPointF(x - 4, self.TOP - 8), QPointF(x + 4, self.TOP - 8),
                          QPointF(x, self.TOP - 2)])
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(t["accent"]))
        painter.drawPolygon(head)

    def _nothing_here(self) -> str:
        """The empty state, spelled out: a track with no predicted rows is a
        normal outcome, and a blank widget would read as a broken one."""
        return (
            f"No predicted bouts for track {self.track.track_id} of "
            f"{self.track.movie_id}.\n\n"
            "That is normal, not an error: a track with no featurisable window "
            "over it gets no row at all."
        )

    def _paint_lane(self, painter, t, group, segments, y, gutter, scale) -> None:
        bar_top = y + (self.LANE - self.BAR) / 2
        bar = QRectF(gutter, bar_top, max(0.0, self.width() - gutter - self.RIGHT),
                     self.BAR)

        painter.setPen(QColor(t["muted"]))
        painter.drawText(
            QRectF(2, bar_top, gutter - 8, self.BAR),
            int(Qt.AlignVCenter | Qt.AlignRight),
            painter.fontMetrics().elidedText(group, Qt.ElideRight, int(gutter - 10)),
        )

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(t["raised"]))
        painter.drawRoundedRect(bar, 4, 4)

        for segment in segments:
            self._paint_segment(painter, t, group, segment, bar, gutter, scale)

    def _colour(self, group: str, label: str, t) -> QColor:
        """The class's own colour from the config, matching the exported movie.

        Falls back to the theme's faint grey for a value the config no longer
        declares, as :func:`trackanno.view.render.build_frame_marks` does, so a
        renamed option looks orphaned here too.
        """
        spec = self.cfg.group(group) if self.cfg is not None else None
        option = spec.option(label.split(MULTI_SEP)[0]) if spec is not None else None
        colour = QColor(option.color) if option is not None else QColor(t["faint"])
        return colour if colour.isValid() else QColor(t["faint"])

    def _paint_segment(self, painter, t, group, segment, bar, gutter, scale) -> None:
        x0 = self._x(segment.f_start, gutter, scale)
        x1 = self._x(segment.f_end, gutter, scale)
        rect = QRectF(x0, bar.top(), max(1.0, x1 - x0), self.BAR)
        self._hits.append((rect, group, segment))

        if not segment.label:
            # A hole, drawn as one: no fill, a hatch, and the skip_reason in the
            # tooltip. The model has no opinion over these frames.
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
        fill = QRectF(rect.left(), rect.bottom() - max(1.0, conf * self.BAR),
                      rect.width(), max(1.0, conf * self.BAR))
        painter.setPen(Qt.NoPen)
        painter.setBrush(colour)
        painter.drawRect(fill)

        painter.setPen(QPen(colour.darker(140), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect)

        text = f"{segment.label}  {segment.conf:.2f}" if np.isfinite(segment.conf) \
            else segment.label
        metrics = painter.fontMetrics()
        if rect.width() > metrics.horizontalAdvance(text) + 8:
            painter.setPen(QColor(t["text"]))
            painter.drawText(rect, int(Qt.AlignCenter), text)

    def _paint_axis(self, painter, t, y, gutter, scale) -> None:
        painter.setPen(QPen(QColor(t["border"]), 1))
        painter.drawLine(QPointF(gutter, y + 2),
                         QPointF(self.width() - self.RIGHT, y + 2))
        painter.setPen(QColor(t["faint"]))
        font = QFont(self.font())
        font.setPointSizeF(max(7.0, font.pointSizeF() - 1))
        painter.setFont(font)
        for frame in _axis_ticks(self.track.f_lo, self.track.f_hi):
            x = self._x(frame, gutter, scale)
            painter.drawLine(QPointF(x, y + 2), QPointF(x, y + 6))
            painter.drawText(QRectF(x - 30, y + 7, 60, 16),
                             int(Qt.AlignHCenter | Qt.AlignTop), str(frame))
        painter.drawText(QRectF(2, y + 7, gutter - 8, 16),
                         int(Qt.AlignRight | Qt.AlignTop), "frame")

    # -- the detail that would clutter the picture ---------------------------- #
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
        for rect, group, segment in self._hits:
            if rect.contains(QPointF(point)):
                return group, segment
        return None

    # -- scrubbing with the mouse --------------------------------------------- #
    def mousePressEvent(self, event):  # noqa: N802 (Qt naming)
        """Take the video to the frame under the mouse."""
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
        self.update()

    def _describe(self, group: str, segment: Segment) -> str:
        lines = [
            f"{group}: {segment.label or '(no prediction)'}",
            f"frames {segment.f_start}-{segment.f_end} ({segment.n_frames} frames)",
        ]
        if np.isfinite(segment.conf):
            lines.append(
                f"confidence {segment.conf:.3f} - the MEAN posterior for this "
                "class over this bout's own frames, not its peak"
            )
        if segment.skip_reason:
            lines.append(segment.skip_reason)
        return "\n".join(lines)


def _axis_ticks(f_lo: int, f_hi: int, target: int = 6) -> list[int]:
    """Round frame numbers across the span - 1, 2 or 5 times a power of ten."""
    span = max(1, f_hi - f_lo)
    raw = span / max(1, target)
    # Clamped at 1: a fractional step is finer than a frame and turns range()
    # into a ValueError.
    power = max(1, 10 ** max(0, int(np.floor(np.log10(max(raw, 1.0))))))
    step = power
    for multiple in (1, 2, 5, 10):
        step = multiple * power
        if span / step <= target:
            break
    first = int(np.ceil(f_lo / step) * step)
    return list(range(first, f_hi + 1, int(step)))


# --------------------------------------------------------------------------- #
# the footage the timeline is a picture of
# --------------------------------------------------------------------------- #
# Frames per decode, and how many decoded chunks are kept around the playhead.
# A chunk is the unit of both the worker's work and the viewer's array: short
# enough to arrive quickly, long enough to play through.
REVIEW_CHUNK = 50
CHUNKS_KEPT = 5
# Frames between two deliveries of a chunk still being cut.
PARTIAL_EVERY = 5


@dataclass(frozen=True)
class ReviewCrop:
    """The one crop every review frame is cut at, for the whole session.

    ``side`` is movie pixels taken, ``size`` is the square they are delivered in.
    One of each per PROJECT, not per movie and not per track: ``clips.crop_size:
    auto`` sizes the cache per movie, so a window that took its geometry from
    the cache changed the size of the picture whenever the reviewer changed
    movie, and a size that changes on its own is read as something in the
    recording.
    """

    side: int
    size: int

    @property
    def scale(self) -> float:
        """Delivered pixels per movie pixel."""
        return self.size / float(self.side)



def review_crop(cfg: Config) -> ReviewCrop:
    """The session's crop, measured once over every clips.parquet in the project.

    The 99th percentile of the widths the cache was cut at rather than the
    largest of them, and rounded the way
    :func:`trackanno.annotate.clips.choose_crop_size` rounds: one track whose
    detections merged into a double-width box would otherwise shrink every
    animal in the project to fit it. Delivered at that side 1:1 unless it is
    wider than ``clips.max_size``, which is the project's own bound on how big a
    crop it is willing to hold.
    """
    from ..annotate.clips import MANIFEST_FILE

    widths: list[np.ndarray] = []
    work = getattr(getattr(cfg, "paths", None), "work_dir", None)
    if work is not None and work.exists():
        for movie_dir in sorted(p for p in work.iterdir() if p.is_dir()):
            path = movie_dir / MANIFEST_FILE
            if not path.exists():
                continue
            try:
                column = pd.read_parquet(path, columns=["crop_w"])["crop_w"]
            except (OSError, ValueError, KeyError):
                continue
            values = pd.to_numeric(column, errors="coerce").dropna().to_numpy(float)
            if values.size:
                widths.append(values)

    ceiling = max(32, int(getattr(cfg.clips, "max_size", 256) or 256))
    if widths:
        pooled = np.concatenate(widths)
        side = int(np.ceil(float(np.percentile(pooled, 99)) / 16.0) * 16)
    else:
        declared = getattr(cfg.clips, "crop_size", "auto")
        side = int(declared) if isinstance(declared, int) else ceiling
    side = max(16, side)
    return ReviewCrop(side=side, size=min(side, ceiling))


class MovieFrames(QObject):
    """Every review frame, cut from the movie on a worker thread.

    One crop side for the session and one path per track, so the cut is a pure
    function of the frame: two reads that share a frame cut it identically, and
    nothing changes size or centring at the edge of a read.

    The clip cache is not read here at all. It is cut per movie and centred per
    annotation window, so a picture that took the frames a window covers from
    the cache and the rest from the movie stepped in size and in centring
    wherever a window ran out, which is exactly the boundary worth watching.

    Decoding runs off the UI thread and arrives as a signal: one seek into a
    TIFF on a share is long enough to freeze the window.
    """

    # request id, first frame, ((T, S, S) uint8, the marks on them). Both
    # together, because a second request overwrites whatever the worker keeps
    # and the pixels and the marker on them must be the same chunk.
    ready = Signal(int, int, object)
    # The same, for the frames cut so far. After the first frame and then
    # every `PARTIAL_EVERY`, so a reader on a share sees a frame after one
    # read rather than after fifty. Read-aheads arrive whole.
    partial = Signal(int, int, object)
    failed = Signal(int, str)

    AHEAD = 0                      # the request id a read-ahead arrives under

    def __init__(self, cfg: Config, movie_id: str, track_id: int,
                 crop: ReviewCrop, chunk: int = REVIEW_CHUNK, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.movie_id = str(movie_id)
        self.track_id = int(track_id)
        self.crop = crop
        self.chunk = max(1, int(chunk))
        # Why a pose project has no skeleton, or "". Set on the decode thread and
        # only ever read afterwards.
        self.skeleton_problem = ""
        self._request: tuple[int, int] | None = None
        self._ahead: list[int] = []
        self._wake = threading.Event()
        self._lock = threading.Lock()
        self._stop = False
        self._thread: threading.Thread | None = None
        self._movie = None
        self._normalizer = None
        self._boxes: np.ndarray | None = None     # (n_frames, 4), held-filled
        self._present: np.ndarray | None = None   # (n_frames,) really detected
        self._kpts: np.ndarray | None = None      # (n_frames, K, 2) movie px
        self._context: tuple | None = None        # the other tracks, by frame
        self._path: np.ndarray | None = None      # (n_frames, 2) crop centres

    # -- the request -------------------------------------------------------- #
    def want(self, request_id: int, first: int, last: int | None = None) -> None:
        """Ask for frames ``first`` to ``last`` now, ``first + chunk`` by default.
        Latest wins: a scrub produces a request per mouse move and only the
        last is worth decoding."""
        with self._lock:
            self._request = (int(request_id), int(first),
                             None if last is None else int(last))
        self._start()

    def read_ahead(self, chunks) -> None:
        """Queue the chunks the playhead is about to reach, behind the urgent
        one: each a first frame, or ``(first, last)``. The same worker,
        because one movie handle is not two threads' to seek."""
        ahead = []
        for chunk in chunks:
            try:
                first, last = chunk
            except TypeError:
                first, last = chunk, None
            ahead.append((int(first), None if last is None else int(last)))
        with self._lock:
            self._ahead = ahead
        self._start()

    def _start(self) -> None:
        if self._thread is None and not self._stop:
            self._thread = threading.Thread(
                target=self._run, name="trackanno-review-movie", daemon=True
            )
            self._thread.start()
        self._wake.set()

    def stop(self, timeout: float = 5.0) -> bool:
        """Stop and WAIT, then close the movie. The thread holds a file handle
        the caller may be about to overwrite."""
        self._stop = True
        self._wake.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout)
            if thread.is_alive():
                return False
        self._close()
        return True

    # -- the worker --------------------------------------------------------- #
    def _run(self) -> None:
        while not self._stop:
            with self._lock:
                request, self._request = self._request, None
                if request is None and self._ahead:
                    request = (self.AHEAD, *self._ahead.pop(0))
            if request is None:
                self._wake.wait(timeout=0.25)
                self._wake.clear()
                continue
            if self._stop:
                break
            request_id, first, last = request
            try:
                emit = None
                if request_id != self.AHEAD:
                    def emit(payload, rid=request_id, f=first):
                        self.partial.emit(rid, f, payload)
                self.ready.emit(request_id, first, self._cut(first, last, emit))
            except Exception as exc:   # a missing movie is a sentence, not a crash
                # A read-ahead that fails is silent: it is routinely asked for
                # frames past the end of the movie, and nobody is looking at it.
                if request_id != self.AHEAD:
                    self.failed.emit(request_id, str(exc))
        self._close()

    def _close(self) -> None:
        movie, self._movie = self._movie, None
        if movie is not None:
            try:
                movie.close()
            except Exception:
                pass

    # -- what the track is -------------------------------------------------- #
    def _open(self):
        from ..annotate.clips import NORM_FILE
        from ..video.movies import Normalizer, open_movie

        if self._movie is not None:
            return self._movie
        movie = open_movie(self.cfg, self.movie_id)
        try:
            path = self.cfg.movie_dir(self.movie_id) / NORM_FILE
            # The mapping prepare wrote, or measured the way it measured it. A
            # different one would put the review at a different grey from the
            # window the label was applied in.
            self._normalizer = (
                Normalizer.load(path) if path.exists()
                else Normalizer.from_movie(movie, self.cfg.normalize)
            )
            self._load_tracks(int(len(movie)))
        except BaseException:
            movie.close()
            raise
        self._movie = movie
        return movie

    def _load_tracks(self, n: int) -> None:
        """The track's boxes, its keypoints, its neighbours and its crop path.

        On the decode thread, because it is a parquet read per track and the UI
        thread is the one place it must not happen. The keypoints are read here
        too: without them a pose project's frames carry a reticle and no
        skeleton, which is a different picture from the one the label was
        applied in and says nothing about the shape the label was about.
        """
        from ..annotate.clips import TRACKS_FILE, _hold_fill

        n_kpts = int(getattr(self.cfg, "n_keypoints", 0) or 0)
        kcols = [f"kp_{a}_{k}" for k in range(n_kpts) for a in ("x", "y")]
        needed = ["frame", "track_id", "x1", "y1", "x2", "y2"]
        path = self.cfg.movie_dir(self.movie_id) / TRACKS_FILE
        try:
            table = pd.read_parquet(path, columns=needed + ["conf"] + kcols)
        except (OSError, ValueError, KeyError):
            table = pd.read_parquet(path)
        missing = [c for c in needed if c not in table.columns]
        if missing:
            raise ValueError(
                f"{TRACKS_FILE} has no {', '.join(missing)} column, so no crop "
                "can be centred on the track"
            )

        frames = pd.to_numeric(table["frame"], errors="coerce").to_numpy(np.int64)
        ids = pd.to_numeric(table["track_id"], errors="coerce").to_numpy(np.int64)
        xyxy = table[["x1", "y1", "x2", "y2"]].to_numpy(float)
        inside = (frames >= 0) & (frames < n)
        mine = inside & (ids == self.track_id)
        if not mine.any():
            raise ValueError(
                f"track {self.track_id} was never detected in the {n} frames of "
                f"{self.movie_id}, so there is no crop to centre on it"
            )

        boxes = np.full((n, 4), np.nan)
        boxes[frames[mine]] = xyxy[mine]
        present = np.zeros(n, bool)
        present[frames[mine]] = True
        self._boxes = _hold_fill(boxes, present)
        self._present = present

        self._kpts = None
        if n_kpts and all(c in table.columns for c in kcols):
            kpts = np.full((n, n_kpts, 2), np.nan, np.float32)
            kpts[frames[mine]] = (
                table[kcols].to_numpy(np.float32).reshape(-1, n_kpts, 2)[mine]
            )
            self._kpts = kpts
        self.skeleton_problem = (
            "" if self._kpts is not None or not n_kpts
            else f"{TRACKS_FILE} carries no keypoints, so there is no skeleton"
        )

        # The neighbours the cache's own box table carries, under the same
        # setting, so turning context tracks on marks them in both windows.
        self._context = None
        if bool(getattr(self.cfg.clips, "context_tracks", False)):
            other = inside & (ids != self.track_id)
            conf = (table["conf"].to_numpy(float) if "conf" in table.columns
                    else np.ones(len(table)))
            order = np.argsort(frames[other], kind="stable")
            self._context = (frames[other][order], ids[other][order],
                             xyxy[other][order], conf[other][order])

        self._path = self._crop_path()

    def _crop_path(self) -> np.ndarray:
        """Where the crop is centred, for every frame of the track.

        A moving average over the WHOLE track, then pushed the least distance
        that holds the animal's box. Smoothing per read would move the centre at
        the edge of every read; smoothing alone loses the animal over the
        hundreds of frames a bout runs for. Nothing here looks at which read a
        frame belongs to, so a frame is cut the same way in all of them.
        """
        from ..annotate.clips import _moving_average

        boxes = self._boxes
        centres = np.stack([(boxes[:, 0] + boxes[:, 2]) * 0.5,
                            (boxes[:, 1] + boxes[:, 3]) * 0.5], axis=1)
        window = int(getattr(self.cfg.clips, "smooth_window", 25) or 1)
        path = _moving_average(centres, window)
        half = self.crop.side / 2.0
        out = np.empty_like(path)
        for axis, (far, near) in enumerate(((2, 0), (3, 1))):
            lo, hi = boxes[:, far] - half, boxes[:, near] + half
            # lo > hi means the box is wider than the crop; centre on it and
            # lose the least of it rather than pinning it against an edge.
            out[:, axis] = np.where(lo > hi, centres[:, axis],
                                    np.clip(path[:, axis], lo, hi))
        return out

    # -- the pixels --------------------------------------------------------- #
    def _cut(self, first: int, last: int | None = None, emit=None,
             ) -> tuple[np.ndarray, pd.DataFrame]:
        """Frames ``first`` to ``last``, ``first + chunk`` by default, cropped.

        ``emit`` is handed the frames cut so far, after the first and then
        every ``PARTIAL_EVERY``, and never the finished chunk.
        """
        import cv2

        from ..annotate.clips import _crop

        movie = self._open()
        wanted = self.chunk if last is None else int(last) - int(first)
        n = min(max(1, wanted), max(0, int(len(movie)) - int(first)))
        if n <= 0:
            raise ValueError(f"frame {first} is past the end of {self.movie_id}")
        side, size = self.crop.side, self.crop.size
        origins = np.round(self._path[first: first + n] - side / 2.0).astype(np.int64)
        out = np.zeros((n, size, size), np.uint8)
        for t in range(n):
            if self._stop:
                raise RuntimeError("cancelled")
            frame = int(first + t)
            u8 = self._normalizer.to_u8(movie.frame(frame), frame)
            crop = _crop(u8, int(origins[t, 0]), int(origins[t, 1]), side, side,
                         movie.height, movie.width)
            if crop.shape[0] != size:
                interp = cv2.INTER_AREA if crop.shape[0] > size else cv2.INTER_LINEAR
                crop = cv2.resize(crop, (size, size), interpolation=interp)
            out[t] = crop
            if emit is not None and t + 1 < n and (t == 0 or (t + 1) % PARTIAL_EVERY == 0):
                emit((out[:t + 1].copy(), self._marks(first, t + 1, origins[:t + 1])))
        return out, self._marks(first, n, origins)

    def _marks(self, first: int, n: int, origins: np.ndarray) -> pd.DataFrame:
        """The box table the cache would have carried for these frames.

        The columns and the coordinates of
        :func:`trackanno.annotate.clips.build_box_table`, so the review window
        marks the animal with the geometry the annotation window marks it with.
        A frame with no detection gets NO row, as it gets none there: the viewer
        then dims the marker rather than asserting a box nobody measured.
        """
        scale = self.crop.scale
        span = slice(first, first + n)
        present = self._present[span]
        t = np.flatnonzero(present).astype(np.int32)
        # tile, not repeat: a box is (x1, y1, x2, y2) and the origin has to
        # come off it as (x0, y0, x0, y0).
        shift = np.tile(origins[present], 2)
        marks = (self._boxes[span][present] - shift) * scale
        block = {
            "t": t,
            "track_id": np.full(t.size, self.track_id, np.int32),
            "is_target": np.ones(t.size, bool),
            "x1": marks[:, 0], "y1": marks[:, 1],
            "x2": marks[:, 2], "y2": marks[:, 3],
            "conf": np.ones(t.size, np.float32),
        }
        if self._kpts is not None:
            kpts = (self._kpts[span][present] - origins[present][:, None, :]) * scale
            for k in range(kpts.shape[1]):
                block[f"kp_x_{k}"] = kpts[:, k, 0]
                block[f"kp_y_{k}"] = kpts[:, k, 1]
        target = pd.DataFrame(block)
        if self._context is None:
            return target

        frames, ids, xyxy, conf = self._context
        lo, hi = np.searchsorted(frames, [first, first + n])
        if hi <= lo:
            return target
        rel = (frames[lo:hi] - first).astype(np.int64)
        near = (xyxy[lo:hi] - np.tile(origins[rel], 2)) * scale
        edge = float(self.crop.size)
        seen = ((near[:, 2] > 0) & (near[:, 0] < edge)
                & (near[:, 3] > 0) & (near[:, 1] < edge))
        if not seen.any():
            return target
        other = pd.DataFrame({
            "t": rel[seen].astype(np.int32),
            "track_id": ids[lo:hi][seen].astype(np.int32),
            "is_target": np.zeros(int(seen.sum()), bool),
            "x1": near[seen, 0], "y1": near[seen, 1],
            "x2": near[seen, 2], "y2": near[seen, 3],
            "conf": conf[lo:hi][seen].astype(np.float32),
        })
        return pd.concat([target, other], ignore_index=True)


class TrackFootage(QWidget):
    """The video beside the timeline: the frames of the track under review.

    Every frame is cut from the movie, at one crop for the whole session and
    centred on this track's own boxes. Nothing is served from the clip cache:
    the cache is cut per movie and centred per annotation window, so a picture
    stitched from both changed size and centring wherever a window ran out.

    Two things about it differ from the annotation window on purpose - the crop,
    which is one size for the project rather than one per movie, and the view,
    which re-centres every frame rather than following ``ui.follow``.

    With no movie the picture goes away rather than going black: a dark
    rectangle reads as a decoder that failed, and the question worth answering
    here is why there is nothing to see.
    """

    positionChanged = Signal(int)         # the movie frame now on screen

    def __init__(self, cfg: Config | None = None, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        # Looping, as the annotation window does: the end of a decoded chunk is
        # not the end of the track, and the wrap is what says so in time to
        # carry playback into the chunk that holds the next frame.
        self.viewer = SimpleViewer(fps=int(getattr(getattr(cfg, "ui", None), "fps", 30)),
                                   loop=True)
        self.viewer.frame_changed.connect(self._on_viewer_frame)
        self._set_edges()
        # Smaller than the annotation window asks for: there the crop IS the
        # window, here it shares one with the lanes and the captions under them,
        # and a 320 px floor takes the room out of the video rather than giving
        # it any.
        self.viewer.canvas.setMinimumSize(240, 240)
        # The lanes beside this ARE the frame axis, and they span the whole
        # track. The viewer's own slider spans one decode chunk, which is a
        # number with no meaning outside this widget.
        self.viewer.set_scrubber(False)
        # Hidden until there are frames: an empty canvas is a black rectangle,
        # and this widget's whole job on a project with no pixels is to not be
        # one.
        self.viewer.setVisible(False)
        self.source = QLabel("")
        self.source.setWordWrap(True)
        self.source.setProperty("role", "muted")

        # The annotation window's own two controls over the same rendering. Both
        # are display-time transforms of one decoded array, so either can change
        # mid-review without reading a frame again.
        self.follow_combo = follow_combo(REVIEW_FOLLOW)
        self.follow_combo.currentIndexChanged.connect(lambda _: self._refresh_view())
        self.marker_box = QCheckBox("marker")
        self.marker_box.setToolTip("outline the animal this track is about")
        self.marker_box.setChecked(bool(getattr(getattr(cfg, "ui", None),
                                                "show_overlay", True)))
        self.marker_box.setFocusPolicy(Qt.NoFocus)
        self.marker_box.toggled.connect(lambda _: self._refresh_view())

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("view"))
        row.addWidget(self.follow_combo)
        row.addWidget(self.marker_box)
        row.addStretch(1)
        # Kept together so they can go away with the picture: controls over a
        # video that is not there are three widgets that do nothing.
        self.controls = QWidget()
        self.controls.setLayout(row)
        self.controls.setVisible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.viewer, 1)
        layout.addWidget(self.controls)
        layout.addWidget(self.source)

        self._boxes = None                         # the marks of the array shown
        self._shape = (0, 0)                       # its (n_frames, side)
        self._crop: ReviewCrop | None = None       # one for the whole session
        self._movie: MovieFrames | None = None
        self._problem = ""                         # why there is no video, or ""
        self._chunks: dict[int, tuple] = {}        # first frame -> (pixels, marks)
        # The chunk being cut for the frame asked for: (first, last, request).
        self._inflight: tuple[int, int, int] | None = None
        self._movie_id, self._track_id = "", 0
        self._span = (0, 0)
        self._loaded = -1                          # first frame of the array shown
        self._first = 0                            # movie frame of offset 0
        self._length = 0                           # offsets of it that are real
        self._offset = 0                           # offset the viewer is on
        self._position = -1
        # Bumped before every request, so it is never MovieFrames.AHEAD and a
        # read-ahead can never be mistaken for the frame being waited for.
        self._request = 0
        self._pending = (0, 0)                     # request id, frame wanted
        self._awaiting = -1                        # frame a decode is out for
        self._resume_at = -1                       # play on when this frame lands
        self._seeking = False

    # -- the project --------------------------------------------------------- #
    def set_config(self, cfg: Config | None) -> None:
        self.release()
        self.cfg = cfg
        self._crop = None
        self.viewer.set_fps(int(getattr(getattr(cfg, "ui", None), "fps", 30)))
        self._set_edges()

    @property
    def crop(self) -> ReviewCrop | None:
        """The session's one crop, measured on the first track and kept.

        Kept rather than re-measured per track, so nothing this widget does can
        make the picture change size while a reviewer is looking at it.
        """
        if self._crop is None and self.cfg is not None:
            self._crop = review_crop(self.cfg)
        return self._crop

    def release(self) -> None:
        """Drop the movie, the decode thread and every frame this widget holds.

        Not optional and not left to the garbage collector: the Model page keeps
        one review dialog for the session, and a decode thread left running over
        a movie is a file handle held for as long as that window exists.
        """
        self.viewer.pause()
        self.viewer.clear()
        self._release_movie()
        self._chunks.clear()
        self._loaded = -1
        self._length = 0
        self._boxes = None

    def _release_movie(self) -> None:
        movie, self._movie = self._movie, None
        # A cancelled decode still emits, and by then the answer is about a
        # frame nobody is looking at any more.
        self._request += 1
        self._resume_at = self._awaiting = -1
        self._inflight = None
        if movie is not None:
            movie.ready.disconnect()
            movie.partial.disconnect()
            movie.failed.disconnect()
            if not movie.stop():
                print("[trackanno] review decode thread did not stop in time")
        self._problem = ""

    # -- one track ----------------------------------------------------------- #
    def set_track(self, movie_id: str, track_id: int, f_lo: int, f_hi: int) -> None:
        self.viewer.pause()
        key = (str(movie_id), int(track_id))
        if key == (self._movie_id, self._track_id) and self._movie is not None:
            # The same animal: its decoder, the frames already cut and any
            # chunk still being cut all stay. `show_frame` asks afresh.
            self._resume_at = self._awaiting = -1
        else:
            self._release_movie()
            # The frames of the track being left, which are not this one's.
            self._chunks.clear()
        self._movie_id, self._track_id = key
        self._span = (int(f_lo), int(f_hi))
        self._loaded = -1
        self._position = -1
        self._length = 0
        self.show_frame(f_lo)

    @property
    def position(self) -> int:
        """The movie frame on screen, or -1 when there is nothing to show."""
        return self._position

    @property
    def loading(self) -> bool:
        """Whether a chunk is still being cut for the frame asked for."""
        return self._inflight is not None

    @property
    def showing(self) -> str:
        """``"movie"`` when frames are on screen, ``""`` when there are none."""
        return "movie" if self._length else ""

    def frame_image(self) -> np.ndarray | None:
        """The (S, S) array currently drawn, so what is on screen can be checked
        against the movie it is supposed to have come from."""
        array = self.viewer._array
        if array is None or not len(array):
            return None
        return np.asarray(array[min(self.viewer._frame, array.shape[0] - 1)])

    def _set_caption(self, text: str) -> None:
        if text != self.source.text():
            self.source.setText(text)

    # -- moving ------------------------------------------------------------- #
    def _chunk_bounds(self, frame: int) -> tuple[int, int]:
        """The frames the chunk holding ``frame`` covers.

        The grid cell, cut down to the span: a 25-frame window is then read
        once and not padded to fifty, and on a share each frame is a network
        read. A frame the clamped cell does not hold gets the plain cell.
        """
        lo, hi = self._span
        cell = (int(frame) // REVIEW_CHUNK) * REVIEW_CHUNK
        first, last = max(cell, lo), min(cell + REVIEW_CHUNK, hi)
        if not first <= int(frame) < last:
            return cell, cell + REVIEW_CHUNK
        return first, last

    def _held(self, frame: int):
        """``(first, payload)`` of the held chunk holding ``frame``, or None."""
        for first, payload in self._chunks.items():
            if first <= int(frame) < first + len(payload[0]):
                return first, payload
        return None

    def show_frame(self, frame: int) -> None:
        """Put ``frame`` of this track on screen, from the movie."""
        frame = int(frame)
        self._resume_at = self._awaiting = -1
        if self.cfg is None or not self._movie_id:
            self._no_video("no track is selected")
            return
        if self._problem:
            self._no_video(self._problem)
            return
        held = self._held(frame)
        if held is not None:
            first, payload = held
            # Nothing is waited for now: a piece of a chunk asked for earlier
            # must not put that earlier frame back up when it lands.
            self._pending = (-1, frame)
            self._show_chunk(first, payload, frame)
            self._read_ahead(frame)
            return
        cutting = self._inflight
        if cutting is not None and cutting[0] <= frame < cutting[1]:
            # Being cut now: wait for the piece that holds it, under the
            # request already out, rather than ask for the chunk again.
            self._pending = (cutting[2], frame)
            self._awaiting = frame
            return
        first, last = self._chunk_bounds(frame)
        self._request += 1
        self._pending = (self._request, frame)
        self._awaiting = frame
        self._inflight = (first, last, self._request)
        self._frames().want(self._request, first, last)
        self._read_ahead(frame)
        if not self._length:
            self._set_caption(
                f"reading frames {first}-{last} of {self._movie_id}..."
            )

    def _frames(self) -> MovieFrames:
        """The decoder for this track, started on first use."""
        if self._movie is None:
            self._movie = MovieFrames(self.cfg, self._movie_id, self._track_id,
                                      self.crop, REVIEW_CHUNK)
            self._movie.ready.connect(self._on_chunk)
            self._movie.partial.connect(self._on_partial)
            self._movie.failed.connect(self._on_failed)
        return self._movie

    def _read_ahead(self, frame: int) -> None:
        """Decode the chunks the playhead is about to reach, on the same worker.

        A scrub asks for a frame per mouse move and playback walks straight off
        the end of the chunk it is in, so without this every chunk edge is a
        stall the length of a decode.
        """
        movie = self._movie
        if movie is None or not int(getattr(self.cfg.ui, "prefetch", 0) or 0):
            return
        lo, hi = self._span
        first, last = self._chunk_bounds(frame)
        after = self._chunk_bounds(last) if last < hi else None
        wanted = [
            after,
            self._chunk_bounds(first - 1) if first > lo else None,
            self._chunk_bounds(after[1]) if after is not None and after[1] < hi else None,
        ]
        queue: list[tuple[int, int]] = []
        for chunk in wanted:
            if (chunk is None or chunk[0] >= chunk[1] or chunk in queue
                    or self._held(chunk[0]) is not None
                    or (self._inflight is not None and chunk[0] == self._inflight[0])):
                continue
            queue.append(chunk)
        movie.read_ahead(queue)

    def _keep(self, first: int, payload) -> None:
        """Hold the decoded chunks nearest the playhead and let the rest go."""
        self._chunks[int(first)] = payload
        while len(self._chunks) > CHUNKS_KEPT:
            self._chunks.pop(max(self._chunks, key=lambda f: abs(f - self._first)))

    def _show_chunk(self, first: int, payload, frame: int) -> None:
        array, marks = payload
        if self._loaded != int(first) or int(len(array)) != self._length:
            # The frame goes in with the pixels: seeking afterwards paints
            # frame 0 of the new chunk first. A chunk that grew goes in again.
            offset = int(min(max(frame - int(first), 0), max(0, len(array) - 1)))
            self.viewer.set_clip(array, self._view(marks, array), frame=offset)
            self._mark()
            self.viewer.setVisible(True)
            self.controls.setVisible(True)
            self._loaded = int(first)
            self._first = int(first)
            self._length = int(len(array))
        self._seek(frame - self._first, frame)
        self._set_caption(self._note())

    def _view(self, boxes, clip: np.ndarray):
        """The geometry for a newly decoded array, remembered so the controls can
        rebuild it without decoding again."""
        self._boxes = boxes
        self._shape = (int(clip.shape[0]), int(clip.shape[1]))
        return self._build_view()

    def _build_view(self):
        """Everything the annotation window passes, so the same animal is marked
        the same way in both windows - a boundary judged against a rendering
        nobody labelled in is judged against something else.

        Only the follow mode differs, and deliberately: see ``REVIEW_FOLLOW``.
        """
        ui = getattr(self.cfg, "ui", None)
        return build_clip_view(
            self._boxes, self._shape[0], self._shape[1],
            mode=self.follow,
            zoom=float(getattr(ui, "view_zoom", 1.0)),
            with_context=bool(getattr(ui, "show_context_tracks", False)),
            n_kpts=(int(getattr(self.cfg, "n_keypoints", 0))
                    if getattr(ui, "show_skeleton", False) else 0),
        )

    @property
    def follow(self) -> str:
        """What the view follows, off the control that offers the choice."""
        return str(self.follow_combo.currentData() or REVIEW_FOLLOW)

    def _set_edges(self) -> None:
        """The skeleton's connectivity, as the annotation window sets it. Without
        it a pose project's keypoints draw as loose dots."""
        self.viewer.set_edges(
            self.cfg.keypoints.edge_idx if getattr(self.cfg, "is_pose", False) else []
        )

    def _mark(self) -> None:
        """Turn the marker on as the config asks and the checkbox allows."""
        ui = getattr(self.cfg, "ui", None)
        self.viewer.set_overlay(
            bool(self.marker_box.isChecked()),
            bool(getattr(ui, "show_context_tracks", False)),
            bool(getattr(ui, "show_skeleton", False)),
        )

    def _refresh_view(self) -> None:
        """Re-centre and re-mark the array already on screen. Decodes nothing:
        the follow mode and the marker are display-time decisions."""
        if self._boxes is None or not self._length:
            return
        self.viewer.set_view(self._build_view())
        self._mark()

    def _seek(self, offset: int, frame: int) -> None:
        offset = int(min(max(offset, 0), max(0, self._length - 1)))
        self._position = int(frame)
        self._offset = offset
        # The viewer's own signal would bounce straight back as a position, and
        # the frame it names is the one just asked for.
        self._seeking = True
        try:
            self.viewer.seek(offset)
        finally:
            self._seeking = False
        self.positionChanged.emit(int(frame))

    def _on_viewer_frame(self, offset: int) -> None:
        """Playing the film walks the marker along the timeline."""
        if self._seeking or not self._length:
            return
        offset, was = int(offset), self._offset
        self._offset = offset
        self._position = self._first + offset
        self.positionChanged.emit(self._position)
        if not self.viewer.is_playing:
            return
        # Two ways to reach the end of a chunk: play into it, or be sitting on
        # its last frame when play is pressed, which the viewer answers by
        # looping back to the start. Both mean the same thing here.
        arrived = offset >= self._length - 1
        looped = offset == 0 and was >= self._length - 1 and self._length > 1
        if arrived or looped:
            # Queued rather than immediate: this runs inside the viewer's own
            # slider signal, and replacing the clip under it there re-enters.
            QTimer.singleShot(0, self._roll_on)

    def _roll_on(self) -> None:
        """Carry playback into whatever holds the next frame.

        A chunk's last frame is not the end of the track, and a boundary that
        falls near one is exactly the one worth watching across. Where the next
        chunk has still to be decoded, playback waits for it rather than running
        on over the frames it already has.
        """
        if not self.viewer.is_playing:
            return                       # paused between the schedule and here
        nxt = self._first + self._length
        if nxt >= self._span[1]:
            # The end of the track. The viewer would loop the chunk it holds,
            # which restarts part-way through.
            self._restart_at(self._span[0])
            return
        was = self._loaded
        self.viewer.pause()
        self.show_frame(nxt)
        if self._awaiting == nxt:
            self._resume_at = nxt        # the movie is being read for it
        elif self._loaded != was and self._position == nxt and self._length:
            self.viewer.play()

    def _restart_at(self, frame: int) -> None:
        """Move playback to ``frame`` and carry on, waiting for a decode if one
        is needed."""
        self.viewer.pause()
        self.show_frame(int(frame))
        if self._awaiting == int(frame):
            self._resume_at = int(frame)
        elif self._length:
            self.viewer.play()

    # -- what the decoder sends back ----------------------------------------- #
    def _on_partial(self, request_id: int, first: int, payload) -> None:
        """The frames cut so far of a chunk still being cut."""
        if self.sender() is not self._movie:
            return
        self._keep(first, payload)
        self._deliver(request_id, first, payload)

    def _on_chunk(self, request_id: int, first: int, payload) -> None:
        if self.sender() is not self._movie:
            # A read the track before this one had out. Its frames are of
            # another animal and must not reach the chunks kept here.
            return
        self._keep(first, payload)
        if self._inflight is not None and self._inflight[0] == int(first):
            self._inflight = None
        self._deliver(request_id, first, payload)

    def _deliver(self, request_id: int, first: int, payload) -> None:
        """Put the pending frame up if these frames hold it.

        A piece that does not yet reach the frame asked for leaves the picture
        alone, unless there is none yet, when its last frame goes up meanwhile.
        A piece that does, while the chunk is already playing, goes in at the
        frame playback has reached rather than back at the one first asked for.
        """
        if int(request_id) != self._pending[0]:
            return                        # a read-ahead, or a scrub moved on
        first, array = int(first), payload[0]
        covered = first + int(len(array))
        wanted = self._pending[1]
        if wanted >= covered:
            if self._length:
                return
            frame = covered - 1
        else:
            frame = wanted
            self._awaiting = -1
            if self._loaded == first and frame < self._position < covered:
                frame = self._position
        self._show_chunk(first, payload, frame)
        if self._resume_at == frame:
            # Playback stopped at the edge of what was cut, waiting for these.
            self._resume_at = -1
            self.viewer.play()

    def _on_failed(self, request_id: int, message: str) -> None:
        """Asked once and then remembered: a movie that has moved will not be
        back by the next mouse move, and re-opening it per scrub would hang the
        window on a dead share."""
        if self.sender() is not self._movie or int(request_id) != self._pending[0]:
            return
        self._awaiting = -1
        self._release_movie()
        self._problem = (
            f"{self._movie_id} could not be read "
            f"({' '.join(str(message).split())})"
        )
        self._no_video(self._problem)

    # -- what there is to say ------------------------------------------------ #
    def _note(self) -> str:
        """One line under the picture: which frame this is."""
        crop = self.crop
        note = f"frame {self._position} of track {self._track_id}"
        if crop is not None:
            note += f", {crop.side} px crop"
        note += "."
        gap = getattr(self._movie, "skeleton_problem", "")
        if gap and bool(getattr(self.cfg.ui, "show_skeleton", False)):
            note += f" {gap}."
        return note

    def clear(self, reason: str = "") -> None:
        """No track under review, and the reason if there is one."""
        self.viewer.pause()
        self._release_movie()
        self._chunks.clear()
        self._position = -1
        self._span = (0, 0)
        self._movie_id, self._track_id = "", 0
        self._no_video(reason or "no track is selected")

    def _no_video(self, reason: str) -> None:
        """No picture at all: hide it and say why.

        A hidden viewer rather than an empty one, because a black square is read
        as a decoder that failed and this is usually a project whose movies are
        not where the config says.
        """
        self.viewer.pause()
        self.viewer.clear()
        self.viewer.setVisible(False)
        self.controls.setVisible(False)
        self._loaded = -1
        self._length = 0
        self._boxes = None
        text = f"No video. {reason.strip().rstrip('.')}."
        if self._span[1] > self._span[0]:
            text += (
                " The timeline still shows what was predicted; only the frames "
                "behind it are missing."
            )
        self._set_caption(text)


# --------------------------------------------------------------------------- #
# the panel and its window
# --------------------------------------------------------------------------- #
class BoutReviewPanel(QWidget):
    """Movie, track, the bout timeline and the footage it is a picture of.

    Read-only. Every failure the files can present is a sentence in this panel
    rather than an exception.
    """

    def __init__(self, cfg: Config | None = None, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.reader: PredictionReader | None = None
        # Raised while the combos are being refilled: Qt emits
        # currentIndexChanged from addItems, once per item added.
        self._loading = False
        self._track: TrackPredictions | None = None

        # Elided rather than laid out at full width: a movie id is long, and
        # unelided the two of them set a floor under the whole page.
        self.movie = QComboBox()
        self.movie.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.movie.setMinimumContentsLength(10)
        self.movie.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.movie.currentIndexChanged.connect(self._on_movie)
        self.track = QComboBox()
        self.track.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.track.setMinimumContentsLength(8)
        self.track.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
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

        # The picture and the pixels it is a picture of, side by side and tied
        # both ways: scrubbing the timeline moves the video, playing the video
        # moves the marker.
        self.footage = TrackFootage(cfg)
        # Bounded on both sides. The film is a square of reference and the frame
        # axis is the thing being read, so every pixel the window gains belongs
        # to the timeline - and without the cap the splitter took the lot on a
        # shrink and left the lanes behind a horizontal scrollbar.
        self.footage.setMinimumWidth(240)
        self.footage.setMaximumWidth(520)
        scroll.setMinimumWidth(320)
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

    # -- loading -------------------------------------------------------------- #
    def set_config(self, cfg: Config | None) -> None:
        self.cfg = cfg
        self.timeline.cfg = cfg
        # Before reload(): the movie of the project being left has to be let
        # go of, along with the thread that was decoding it.
        self.footage.set_config(cfg)
        self.reload()

    def release(self) -> None:
        """Let go of the movie and its decode thread. Idempotent, and called
        from every way this panel can stop being looked at."""
        self.footage.release()

    def hideEvent(self, event):  # noqa: N802 (Qt naming)
        self.release()
        super().hideEvent(event)

    def showEvent(self, event):  # noqa: N802 (Qt naming)
        """Put the footage back. The movie was let go of when this panel was
        last hidden, so re-showing it must not leave a sentence where the video
        was."""
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

    def reload(self) -> None:
        """Re-read the predictions and rebuild the selectors."""
        self._loading = True
        try:
            self.movie.clear()
            self.track.clear()
            if self.cfg is None:
                self.reader = None
                self._say("", "no project open")
                return
            self.reader = PredictionReader(self.cfg)
            if self.reader.problem:
                self._say("", self.reader.problem)
                return
            self.movie.addItems(self.reader.movies)
        finally:
            self._loading = False
        self._fill_tracks()

    def _on_movie(self, _index: int) -> None:
        if not getattr(self, "_loading", False):
            self._fill_tracks()

    def _on_track(self, _index: int) -> None:
        if not getattr(self, "_loading", False):
            self._paint()

    def _fill_tracks(self) -> None:
        if self.reader is None or self.reader.problem:
            return
        movie_id = self.movie.currentText()
        self._loading = True
        try:
            self.track.clear()
            for track_id, count in self.reader.tracks(movie_id):
                label = f"track {track_id}" if count else f"track {track_id}  (none)"
                self.track.addItem(label, track_id)
        finally:
            self._loading = False
        self._paint()

    # -- painting ------------------------------------------------------------- #
    def _paint(self) -> None:
        if self.reader is None or self.reader.problem:
            return
        movie_id = self.movie.currentText()
        track_id = self.track.currentData()
        if not movie_id or track_id is None:
            self._say("", "this project's predictions name no movie to review")
            return

        track = self.reader.track(movie_id, int(track_id))
        self._track = track
        self.timeline.set_track(track)
        if track.empty:
            self.footage.clear(
                f"track {track.track_id} of {track.movie_id} has no predicted "
                "rows, so there is no span to watch"
            )
        else:
            self.footage.set_track(movie_id, int(track_id), track.f_lo, track.f_hi)
        t = tokens()
        head = (
            f"<b>predicted bouts</b> over {len(self.reader.groups)} category "
            f"group(s); this track has <b>{track.n_segments}</b> row(s) across "
            f"frames {track.f_lo}-{track.f_hi}."
            if not track.empty else
            "<b>predicted bouts</b>; this track has none."
        )
        if self.reader.stale:
            head += (
                f"<br><span style='color:{t['warning']}'>These predictions may be "
                f"out of date - {self.reader.stale}.</span>"
            )
        self.headline.setText(head)

    def _say(self, head: str, problem: str) -> None:
        self.headline.setText(head)
        self._track = None
        self.timeline.set_track(TrackPredictions("", 0), note=problem)
        self.footage.clear(problem)
