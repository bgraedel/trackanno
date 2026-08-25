"""One track's predictions along the frame axis, drawn rather than tabulated.

``predictions.parquet`` is the only place the model's answer to "what was this
animal doing, and when did it change" lives, and under ``predict.decode`` a row
stopped being a window off a grid and became a BOUT - a variable-length run
between two inferred behavioural boundaries. A bout is unreviewable as a table
row: what it claims is where it starts, where it stops, and how sure the model
was over exactly those frames, and none of those three read off a column of
integers. So they are drawn, on the axis the claim is about.

Read-only, deliberately. Nothing here edits, splits, merges or writes a bout
back. A corrected bout has no defined meaning yet for a training set built from
fixed-length windows - see the note at the foot of this docstring - and a widget
that let one be dragged would be inventing that meaning by accident.

Four things the picture is not allowed to imply:

* **A boundary is not a frame.** Overlapping windows remove the QUANTISATION of
  a transition to a multiple of the window length; they do not remove the BLUR
  of a window that genuinely straddles the transition and whose features are a
  blend of both behaviours. The measured error against a planted transition is
  ``stride/4`` on average and ``stride/2`` at worst (README, "Boundary error
  against a planted transition"), so every boundary is drawn with a band that
  wide rather than as a crisp line. Drawing a hairline would claim a precision
  nothing in the method has.
* **A marginal bout must not look like a confident one.** ``conf`` is the MEAN
  posterior of the bout's own class over its own frames, so it is comparable
  between bouts and is drawn as a fill height, not as a colour wash that the
  eye reads as "some blue". Where ``frame_posteriors.parquet`` is present the
  per-frame posterior is drawn over the same scale, because a mean of 0.8 that
  is flat and a mean of 0.8 that is 0.99 either side of a 0.4 trough are
  different claims and the number cannot tell them apart.
* **A bout overlapping hand-labelled frames is not independent evidence.**
  Those frames trained the model. They are hatched, and the fraction is in the
  tooltip, so agreement there is never read as a score.
* **A windows file is not a bout file.** An older ``predictions.parquet`` holds
  fixed windows off a grid the predictor laid down itself; its edges are grid
  multiples and say nothing about the animal. It is drawn - it is still worth
  seeing - but captioned as what it is, and the boundary bands are omitted,
  since there is no inferred boundary there to be uncertain about.

Which of the two a file is comes off the FILE, never off ``predict.decode``:
the config can be edited between a run and this window opening, and a widget
that consulted it would caption the run somebody is about to do rather than the
one whose rows are on screen. ``bout_id`` is the probe, exactly as in
:func:`trackanno.render.prediction_unit` and :func:`trackanno.predict.summarise`.

Not built, and why: editing. Correcting a bout means writing a label onto a
half-open frame span, and every training set in this codebase is assembled from
fixed-length clips keyed by ``clip_id`` - which a bout deliberately does not
have (:func:`trackanno.predict._bout_id` explains why). "Which clips does a
corrected 300-frame bout produce, and at what length" is a data-model question,
not a widget one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from qtpy.QtCore import QEvent, QPointF, QRectF, QSize, Qt
from qtpy.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPolygonF
from qtpy.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QScrollArea,
    QSizePolicy, QToolTip, QVBoxLayout, QWidget,
)

from ..config import Config
from ..store import MULTI_SEP
from .theme import tokens

__all__ = [
    "Segment", "TrackPredictions", "Resolution", "PredictionReader",
    "caption_lines", "BoutTimeline", "BoutReviewPanel", "BoutReviewDialog",
]

# The README's measured boundary error, as divisors of the window stride. Taken
# from the table there ("Boundary error against a planted transition": 12.24/24
# at stride 50, 6.24/12 at 25, 2.23/4 at 10, 1.20/2 at 5) rather than restated
# as frames, because the stride is a per-project setting and the ratio is what
# generalises. Not a guess and not a rule of thumb: change the table, change
# these.
MEAN_ERROR_DIVISOR = 4.0
WORST_ERROR_DIVISOR = 2.0


# --------------------------------------------------------------------------- #
# what a row is
# --------------------------------------------------------------------------- #
def _is_bouts(frame: pd.DataFrame) -> bool:
    """Whether these rows are decoded bouts, asked of the FILE.

    The same probe :func:`trackanno.render.prediction_unit` makes, and made here
    for the same reason: ``predict.decode`` describes the next run, not the one
    that wrote these rows, so a config read would caption a picture of bouts as
    windows the moment somebody turned the decode off. The two live apart only
    because that function answers with a noun for counting and this one with a
    fact for branching; both are ``bout_id``, and neither may become a config
    lookup.
    """
    return "bout_id" in frame.columns


@dataclass(frozen=True)
class Segment:
    """One drawn span: a decoded bout, or a window off a grid.

    ``label`` is ``""`` for a stretch the model has no opinion about - a bout
    over frames no window could be featurised, or a skipped window. Those are
    drawn as holes rather than dropped, because an absent span is
    indistinguishable from one nobody thought about, which is the same
    convention the bout table itself follows.

    ``conf``, ``human_frac`` and ``n_covering`` are NaN where the file does not
    record them, never 0.0: a windows file has no ``n_covering_mean`` at all,
    and "no windows agreed" is a different statement from "the column is not
    there".
    """

    f_start: int
    f_end: int
    label: str
    conf: float
    human_frac: float
    n_covering: float
    skip_reason: str

    @property
    def n_frames(self) -> int:
        return self.f_end - self.f_start


@dataclass
class TrackPredictions:
    """Everything one (movie, track) has to say, ready to paint."""

    movie_id: str
    track_id: int
    bouts: bool                 # decoded bouts, as opposed to grid windows
    unit: str                   # the noun prediction_unit() counts these in
    f_lo: int = 0
    f_hi: int = 0
    lanes: dict = field(default_factory=dict)        # group -> [Segment]
    # group -> (n_frames,) posterior of the DECODED class per frame, or {} when
    # frame_posteriors.parquet is absent. NaN where no window covered the frame.
    trace: dict = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return not any(self.lanes.values())

    @property
    def n_segments(self) -> int:
        return sum(len(v) for v in self.lanes.values())


# --------------------------------------------------------------------------- #
# how sharp a boundary can be
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Resolution:
    """The window stride these bouts were cut on, and how far to trust an edge.

    ``stride`` is 0 when it could not be MEASURED, and then this reports the
    limit as a ratio and draws no band. The alternative - reading
    ``predict.window_stride`` out of the config - is refused for the same reason
    ``predict.decode`` is: the config describes the next run. Here it would be
    worse than a wrong caption, because the number sets the width of a band a
    reviewer reads as "the boundary is somewhere in here".
    """

    stride: int = 0
    provenance: str = ""        # where the stride came from, or why there is none

    @property
    def known(self) -> bool:
        return self.stride > 0

    @property
    def mean_error(self) -> float:
        return self.stride / MEAN_ERROR_DIVISOR

    @property
    def worst_error(self) -> float:
        return self.stride / WORST_ERROR_DIVISOR

    def sentence(self) -> str:
        if not self.known:
            return (
                "A boundary is resolved to the window grid, not to the frame: "
                "about a quarter of the window stride out on average and half of "
                f"it at worst. {self.provenance}, so that cannot be said in "
                "frames here and no band is drawn - the edges below are sharper "
                "than the method is."
            )
        return (
            f"Boundaries are resolved to the {self.stride}-frame window stride, "
            f"not to the frame: {self.mean_error:.1f} frames out on average and "
            f"{self.worst_error:.0f} at worst, measured against a planted "
            f"transition. The band at each edge is that worst case. {self.provenance}."
        )


def measure_stride(cfg: Config) -> Resolution:
    """The stride of the window grid on disk, measured, never read off the config.

    ``dense_clip_features.parquet`` IS the plan the decode ran over - one row per
    (track, window) - so the gap between one window start and the next is the
    stride that produced the bouts, as a fact rather than as a setting. Modal
    rather than minimal: a track whose windows do not divide evenly leaves one
    short step at the end, and the minimum would report that.

    A dense table written AFTER the predictions is still measured but says so.
    It is the only grid there is, and refusing to give a number would leave the
    band off exactly the file most likely to have moved; an mtime is the same
    cheap probe :func:`trackanno.predict.staleness` already trusts.
    """
    from ..densepredict import dense_features_path
    from ..predict import predictions_path

    path = dense_features_path(cfg)
    if not path.exists():
        return Resolution(0, f"{path.name} is not in this project")
    try:
        frame = pd.read_parquet(path, columns=["movie_id", "track_id", "f_start"])
    except (OSError, ValueError, KeyError) as exc:
        return Resolution(0, f"{path.name} could not be read ({exc})")
    if frame.empty:
        return Resolution(0, f"{path.name} has no windows in it")

    steps: list[int] = []
    starts = pd.to_numeric(frame["f_start"], errors="coerce").to_numpy(dtype=float)
    keys = frame[["movie_id", "track_id"]].astype({"movie_id": str})
    for positions in keys.groupby(["movie_id", "track_id"], sort=False).indices.values():
        run = np.sort(starts[np.asarray(positions)])
        gaps = np.diff(run)
        steps.extend(int(g) for g in gaps if np.isfinite(g) and g > 0)
    if not steps:
        return Resolution(0, f"every track in {path.name} has a single window")

    stride = int(pd.Series(steps).mode().iat[0])
    predictions = predictions_path(cfg)
    rebuilt = (
        predictions.exists()
        and path.stat().st_mtime > predictions.stat().st_mtime
    )
    where = f"Measured from the window grid in {path.name}"
    if rebuilt:
        where += (
            ", which was rebuilt after these predictions were written - so it may "
            "not be the grid they came off"
        )
    return Resolution(stride, where)


# --------------------------------------------------------------------------- #
# reading the files
# --------------------------------------------------------------------------- #
def _labelled(values: pd.Series) -> pd.Series:
    """Which cells hold a real class name.

    The codebase's emptiness rule, repeated because the alternative is worse: an
    absent prediction reads back out of parquet as ``""`` or as NaN whose
    ``str()`` is ``"nan"``, and a bout captioned "nan" is a bout claiming the
    model said something.
    """
    text = values.astype(str).str.strip()
    return text.ne("") & text.str.lower().ne("nan")


class PredictionReader:
    """``predictions.parquet`` and, if it is there, the posterior beside it.

    Built once per open of the review window and then queried per track. Both
    files are read whole: the predictions because every panel control is a
    question about the set of them (which movies, which tracks, which groups),
    and the posterior because a per-track read would need a parquet predicate
    pushdown this codebase does not otherwise depend on. The posterior is one
    row per (movie, track, frame), so it is the larger of the two by far and is
    read LAZILY - a project that never saved one, or a reviewer who never opens
    the window, pays nothing.

    ``problem`` is set instead of raising. This is opened from a button on a
    page that has already done its own gating, and a traceback out of a Qt slot
    takes the window down; a sentence saying which file is missing and how to
    make it is the same information without the crash.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.problem = ""
        self.frame = pd.DataFrame()
        self.bouts = False
        self.unit = ""
        self.groups: list[str] = []
        self.movies: list[str] = []
        self.stale = ""
        self._posterior: pd.DataFrame | None = None
        self._posterior_read = False
        self._tracks: dict[str, list[tuple[int, int]]] = {}
        self.resolution = Resolution()

        from ..predict import load_predictions, staleness
        from ..render import prediction_unit

        try:
            self.frame = load_predictions(cfg)
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            self.problem = str(exc)
            return

        self.bouts = _is_bouts(self.frame)
        self.unit = prediction_unit(self.frame)
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
        # Only where an inferred boundary exists to be uncertain about. A window
        # edge is a grid multiple the predictor chose, not a claim about the
        # animal, and a band around it would dress a decision up as a
        # measurement.
        if self.bouts:
            self.resolution = measure_stride(cfg)

    # -- what can be selected ------------------------------------------------ #
    def tracks(self, movie_id: str) -> list[tuple[int, int]]:
        """``[(track_id, n_rows)]`` for one movie, predicted tracks and silent ones.

        The union with ``tracks.parquet`` is the point. A track the run produced
        no row for is a NORMAL outcome - too few detections to featurise a single
        window, every window skipped - and a combo box built only from the
        predictions makes that state unreachable, so the one question a reviewer
        would ask about it ("why is this track not here?") cannot be asked. Those
        tracks are listed with a count of 0 and draw an empty timeline that says
        so.
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
        predictions already filled, so a missing or unreadable table costs a
        completeness the panel never promised, and raising here would break the
        review of the tracks that did predict.
        """
        from ..clips import TRACKS_FILE

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
        out = TrackPredictions(
            movie_id=str(movie_id), track_id=int(track_id),
            bouts=self.bouts, unit=self.unit,
        )
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
        out.trace = self._traces(out)
        return out

    def _segments(self, rows: pd.DataFrame, group: str) -> list[Segment]:
        """One group's spans out of one track's rows, in frame order.

        A bout table carries one row per (group, bout), so a group owns only the
        rows whose ``group`` column names it - the others carry an empty
        prediction in its ``_pred`` column and would otherwise be drawn as holes
        in this lane that are really another lane's bouts. A windows table has
        no ``group`` column and every row speaks for every group at once, which
        is the whole difference between the two schemas.
        """
        if self.bouts:
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
        # A windows file records a flag, a bout file a fraction. Neither is
        # invented from the other: 1.0 from a flag is exactly what the flag says
        # about a window, and a file carrying neither reports NaN rather than
        # 0.0, which would claim the run checked and found nothing.
        if "human_overlap_frac" in rows.columns:
            human = pd.to_numeric(rows["human_overlap_frac"], errors="coerce")
        elif "human_labelled" in rows.columns:
            human = rows["human_labelled"].astype(bool).astype(float)
        else:
            human = blank
        covering = pd.to_numeric(
            rows["n_covering_mean"] if "n_covering_mean" in rows.columns else blank,
            errors="coerce",
        ).to_numpy(dtype=float)
        reason = (rows["skip_reason"].astype(str) if "skip_reason" in rows.columns
                  else pd.Series("", index=rows.index)).to_numpy()
        human = human.to_numpy(dtype=float)
        names = pred.to_numpy()

        out = []
        for i in range(len(rows)):
            # A row whose geometry will not parse is dropped, never defaulted: a
            # guessed span would put a coloured bar over frames nothing scored.
            if not (np.isfinite(starts[i]) and np.isfinite(ends[i])):
                continue
            text = str(reason[i])
            out.append(Segment(
                f_start=int(starts[i]), f_end=int(ends[i]),
                label=str(names[i]) if known[i] else "",
                conf=float(conf[i]) if known[i] else float("nan"),
                human_frac=float(human[i]),
                n_covering=float(covering[i]),
                skip_reason="" if text.lower() == "nan" else text,
            ))
        out.sort(key=lambda s: (s.f_start, s.f_end))
        return out

    # -- the posterior the bouts were cut from -------------------------------- #
    def posterior(self) -> pd.DataFrame | None:
        """``frame_posteriors.parquet``, or None when the run did not save one.

        ``predict.save_posteriors`` is off by default, so None is the common
        case and everything above it must work without it. It is read once, on
        the first track a reviewer looks at.
        """
        if self._posterior_read:
            return self._posterior
        self._posterior_read = True
        from ..predict import posteriors_path

        path = posteriors_path(self.cfg)
        if not path.exists():
            return None
        try:
            self._posterior = pd.read_parquet(path)
        except (OSError, ValueError) as exc:
            # Not fatal and not silent: the timeline is worth drawing without a
            # trace, and a reviewer who asked for posteriors should be told the
            # file is unreadable rather than shown a picture that quietly omits
            # them.
            self.problem = f"{path.name} could not be read ({exc})"
            self._posterior = None
        return self._posterior

    def _traces(self, track: TrackPredictions) -> dict:
        """Per-frame posterior of the DECODED class, per group.

        The point of drawing it beside the bars: ``conf`` is a MEAN over the
        bout's frames, and a flat 0.8 and a 0.99/0.4/0.99 that averages to 0.8
        are different claims about the same bout that the mean cannot separate.
        Only the decoded class's column is followed, because that is the number
        ``conf`` is the mean of; the rival classes are a different question and
        would need their own lane to answer honestly.
        """
        table = self.posterior()
        if table is None or table.empty or not track.lanes:
            return {}
        rows = table[
            (table["movie_id"].astype(str) == track.movie_id)
            & (pd.to_numeric(table["track_id"], errors="coerce") == track.track_id)
        ]
        if rows.empty:
            return {}
        frames = pd.to_numeric(rows["frame"], errors="coerce").to_numpy(dtype=np.int64)
        order = np.argsort(frames, kind="stable")
        frames = frames[order]

        out: dict[str, np.ndarray] = {}
        n = max(track.f_hi - track.f_lo, 0)
        columns: dict[str, np.ndarray] = {}
        for group, segments in track.lanes.items():
            trace = np.full(n, np.nan)
            for segment in segments:
                column = f"{group}__p__{segment.label}"
                if not segment.label or column not in rows.columns:
                    continue
                if column not in columns:
                    columns[column] = pd.to_numeric(
                        rows[column], errors="coerce"
                    ).to_numpy(dtype=float)[order]
                index = frames - track.f_lo
                inside = (
                    (frames >= segment.f_start) & (frames < segment.f_end)
                    # The posterior spans the union of the track's windows, which
                    # is the same span the bouts came off - but a file written by
                    # a different run need not agree, and writing outside the
                    # array would be a crash rather than a wrong picture.
                    & (index >= 0) & (index < n)
                )
                trace[index[inside]] = columns[column][inside]
            if np.isfinite(trace).any():
                out[group] = trace
        return out


# --------------------------------------------------------------------------- #
# what the picture cannot say for itself
# --------------------------------------------------------------------------- #
def caption_lines(
    track: TrackPredictions, resolution: Resolution, min_confidence: float = 0.0,
    has_trace: bool = False,
) -> list[str]:
    """The sentences under the timeline. Separate from the painting so the
    claims can be tested without a screenshot, and so a claim that stops being
    true fails a test rather than sitting on screen."""
    if track.empty:
        return []
    lines: list[str] = []
    if track.bouts:
        lines.append(
            "Each bar is one decoded bout - a run of frames between two inferred "
            "behavioural boundaries. Bar height is the bout's mean posterior for "
            "its own class, so a marginal bout is a short bar, not a pale one."
        )
        lines.append(resolution.sentence())
    else:
        lines.append(
            f"This file holds {track.unit}: fixed-length windows off the grid the "
            "predictor laid down, NOT decoded bouts. Every edge below is a "
            "multiple of the window length and says nothing about when the "
            "animal changed what it was doing. Set predict.decode in the config "
            "and label again to cut bouts."
        )

    hand = [s for lane in track.lanes.values() for s in lane
            if np.isfinite(s.human_frac) and s.human_frac > 0]
    if hand:
        lines.append(
            f"{len(hand)} span(s) are hatched: they overlap frames you labelled "
            "by hand, which trained this model, so agreement there is not "
            "evidence about it. The fraction is the union over every predicted "
            "group's labels, so it over-reports for any one of them."
        )

    thin = [s for lane in track.lanes.values() for s in lane
            if np.isfinite(s.n_covering) and s.n_covering < 2.0]
    if thin:
        lines.append(
            f"{len(thin)} span(s) rest on fewer than two windows per frame. Near "
            "a track's ends only one window reaches, so the posterior there is a "
            "single window's opinion rather than an average of several."
        )

    if min_confidence > 0 and track.bouts:
        lines.append(
            f"The dashed line is suggest.min_confidence ({min_confidence:g}), the "
            "bar this project already applies before showing a suggestion to an "
            "annotator."
        )
    if track.bouts and not has_trace:
        lines.append(
            "predict.save_posteriors was off for this run, so the per-frame "
            "posterior each bar is the mean of is not on disk and only the mean "
            "is drawn."
        )
    return lines


# --------------------------------------------------------------------------- #
# painting
# --------------------------------------------------------------------------- #
class BoutTimeline(QWidget):
    """One track's lanes on a shared frame axis, one lane per predicted group.

    QPainter rather than a plotting library, matching
    :mod:`trackanno.gui.distribution`: no new dependency, it follows the theme
    without a second palette, and it stays sharp at any DPI.
    """

    LANE = 52
    BAR = 34
    GAP = 6
    TOP = 10
    AXIS = 26
    RIGHT = 12

    def __init__(self, cfg: Config | None = None, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.track = TrackPredictions("", 0, False, "")
        self.resolution = Resolution()
        self.note = ""
        self._hits: list[tuple[QRectF, str, Segment]] = []
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumWidth(420)
        self.setMouseTracking(True)

    def set_track(self, track: TrackPredictions, resolution: Resolution,
                  note: str = "") -> None:
        self.track = track
        self.resolution = resolution
        self.note = note
        self.updateGeometry()
        self.update()

    def sizeHint(self) -> QSize:
        lanes = max(1, len(self.track.lanes))
        return QSize(760, self.TOP + lanes * self.LANE + self.AXIS + 8)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    # -- geometry ------------------------------------------------------------ #
    def _gutter(self, metrics) -> float:
        names = list(self.track.lanes) or [""]
        return min(
            max(70.0, max(metrics.horizontalAdvance(n) for n in names) + 16.0),
            max(80.0, self.width() * 0.28),
        )

    def _scale(self, gutter: float) -> float:
        span = max(1, self.track.f_hi - self.track.f_lo)
        return max(0.0, self.width() - gutter - self.RIGHT) / span

    def _x(self, frame: float, gutter: float, scale: float) -> float:
        return gutter + (frame - self.track.f_lo) * scale

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
        y = float(self.TOP)
        for group, segments in self.track.lanes.items():
            self._paint_lane(painter, t, group, segments, y, gutter, scale)
            y += self.LANE
        self._paint_axis(painter, t, y, gutter, scale)
        painter.end()

    def _nothing_here(self) -> str:
        """The empty state, spelled out. A track with no predicted rows is a
        normal outcome - every window over it skipped for too few detections -
        and a blank widget would read as a broken one."""
        return (
            f"No {self.track.unit or 'predicted rows'} for track "
            f"{self.track.track_id} of {self.track.movie_id}.\n\n"
            "That is a normal outcome, not an error: a track the model could not "
            "be given a single featurisable window over gets no row at all."
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

        # The boundary bands first, under the bars: they are a statement about
        # the edges between them and must not obscure the fills.
        if self.track.bouts and self.resolution.known and len(segments) > 1:
            half = max(1.0, self.resolution.worst_error * scale)
            band = QColor(t["border_hi"])
            band.setAlpha(90)
            painter.setBrush(band)
            for segment in segments[1:]:
                x = self._x(segment.f_start, gutter, scale)
                painter.drawRect(QRectF(x - half, bar.top(), 2 * half, self.BAR))

        for segment in segments:
            self._paint_segment(painter, t, group, segment, bar, gutter, scale)

        floor = float(getattr(getattr(self.cfg, "suggest", None), "min_confidence", 0.0) or 0.0)
        if self.track.bouts and 0.0 < floor < 1.0:
            level = bar.bottom() - floor * self.BAR
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(t["border_hi"]), 1, Qt.DashLine))
            painter.drawLine(QPointF(bar.left(), level), QPointF(bar.right(), level))

        trace = self.track.trace.get(group)
        if trace is not None:
            self._paint_trace(painter, t, trace, bar, gutter, scale)

    def _colour(self, group: str, label: str, t) -> QColor:
        """The class's own colour from the config, matching the exported movie.

        Falls back to the theme's faint grey for a value the config no longer
        declares, exactly as :func:`trackanno.render.build_frame_marks` does, so
        a renamed option looks orphaned here too instead of borrowing another
        class's colour.
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
            # tooltip. Filling it would put a colour where the model has no
            # opinion at all.
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

        if np.isfinite(segment.human_frac) and segment.human_frac > 0:
            # Hatched over the whole span rather than over a share of it: the
            # file records WHAT FRACTION of the bout a person judged, never
            # which frames, and hatching part of the bar would invent that.
            mark = QColor(t["warning"])
            mark.setAlpha(int(90 + 120 * min(1.0, segment.human_frac)))
            painter.setBrush(QBrush(mark, Qt.BDiagPattern))
            painter.drawRect(rect)

        painter.setPen(QPen(colour.darker(140), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect)

        text = f"{segment.label}  {segment.conf:.2f}" if np.isfinite(segment.conf) \
            else segment.label
        metrics = painter.fontMetrics()
        if rect.width() > metrics.horizontalAdvance(text) + 8:
            painter.setPen(QColor(t["text"]))
            painter.drawText(rect, int(Qt.AlignCenter), text)

    def _paint_trace(self, painter, t, trace, bar, gutter, scale) -> None:
        """The per-frame posterior of the decoded class, on the bars' own scale.

        Broken at every undefined frame rather than bridged: a straight line
        across frames no window covered would draw a posterior nobody measured.
        """
        pen = QPen(QColor(t["text"]), 1.2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        polygon = QPolygonF()
        for i, value in enumerate(trace):
            if not np.isfinite(value):
                if polygon.count() > 1:
                    painter.drawPolyline(polygon)
                polygon = QPolygonF()
                continue
            polygon.append(QPointF(
                self._x(self.track.f_lo + i + 0.5, gutter, scale),
                bar.bottom() - float(value) * self.BAR,
            ))
        if polygon.count() > 1:
            painter.drawPolyline(polygon)

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

    def _describe(self, group: str, segment: Segment) -> str:
        noun = "bout" if self.track.bouts else "window"
        lines = [
            f"{group}: {segment.label or '(no prediction)'}",
            f"frames {segment.f_start}-{segment.f_end} ({segment.n_frames} frames)",
        ]
        if np.isfinite(segment.conf):
            lines.append(
                f"confidence {segment.conf:.3f} - the MEAN posterior for this "
                f"class over this {noun}'s own frames, not its peak"
            )
        if np.isfinite(segment.human_frac):
            if segment.human_frac > 0:
                lines.append(
                    f"{100 * segment.human_frac:.0f}% of these frames were labelled "
                    "by hand and trained this model, so agreement here is not a "
                    "measurement of it (the fraction unions every predicted group)"
                )
            else:
                lines.append("no hand-labelled frames overlap this span")
        if np.isfinite(segment.n_covering):
            lines.append(
                f"{segment.n_covering:.1f} windows per frame on average went into "
                "the posterior here"
            )
        if self.track.bouts and self.resolution.known:
            lines.append(
                f"both edges are resolved to the {self.resolution.stride}-frame "
                f"stride: {self.resolution.mean_error:.1f} frames out on average, "
                f"{self.resolution.worst_error:.0f} at worst"
            )
        if segment.skip_reason:
            lines.append(segment.skip_reason)
        return "\n".join(lines)


def _axis_ticks(f_lo: int, f_hi: int, target: int = 6) -> list[int]:
    """Round frame numbers across the span - 1, 2 or 5 times a power of ten.

    Ticks a reader can do arithmetic with: an axis labelled 0, 43, 86 is an axis
    nobody reads a boundary off.
    """
    span = max(1, f_hi - f_lo)
    raw = span / max(1, target)
    # Clamped at 1: below a whole frame there is nothing finer to tick, and a
    # fractional step turns range() into a ValueError rather than a nicer axis.
    power = max(1, 10 ** max(0, int(np.floor(np.log10(max(raw, 1.0))))))
    step = power
    for multiple in (1, 2, 5, 10):
        step = multiple * power
        if span / step <= target:
            break
    first = int(np.ceil(f_lo / step) * step)
    return list(range(first, f_hi + 1, int(step)))


# --------------------------------------------------------------------------- #
# the panel and its window
# --------------------------------------------------------------------------- #
class BoutReviewPanel(QWidget):
    """Movie, track, timeline, captions. Opens against whatever is on disk.

    Every failure the files can present is a sentence in this panel rather than
    an exception: no predictions at all, a file of the wrong kind, a track with
    nothing in it. The one thing it must never do is draw a picture that looks
    fine over a file it could not read.
    """

    def __init__(self, cfg: Config | None = None, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.reader: PredictionReader | None = None
        # Raised while the combos are being refilled. Qt emits
        # currentIndexChanged from addItems, so without it every load repaints
        # once per movie added and the last one wins by accident.
        self._loading = False

        self.movie = QComboBox()
        self.movie.setMinimumWidth(160)
        self.movie.currentIndexChanged.connect(self._on_movie)
        self.track = QComboBox()
        self.track.setMinimumWidth(160)
        self.track.currentIndexChanged.connect(self._on_track)
        self.headline = QLabel("")
        self.headline.setWordWrap(True)
        self.headline.setTextFormat(Qt.RichText)

        top = QHBoxLayout()
        top.addWidget(QLabel("movie"))
        top.addWidget(self.movie)
        top.addSpacing(12)
        top.addWidget(QLabel("track"))
        top.addWidget(self.track)
        top.addStretch(1)

        self.timeline = BoutTimeline(cfg)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.timeline)

        self.caption = QLabel("")
        self.caption.setWordWrap(True)
        self.caption.setTextFormat(Qt.RichText)
        self.caption.setProperty("role", "muted")

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.headline)
        layout.addWidget(scroll, 1)
        layout.addWidget(self.caption)

        if cfg is not None:
            self.reload()

    # -- loading -------------------------------------------------------------- #
    def set_config(self, cfg: Config | None) -> None:
        self.cfg = cfg
        self.timeline.cfg = cfg
        self.reload()

    def reload(self) -> None:
        """Re-read the files and rebuild the selectors. Cheap enough to call on
        every open; the posterior is not touched until a track is picked."""
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
        self.timeline.set_track(track, self.reader.resolution)
        t = tokens()
        head = (
            f"<b>{self.reader.unit}</b> over {len(self.reader.groups)} category "
            f"group(s); this track has <b>{track.n_segments}</b> row(s) across "
            f"frames {track.f_lo}-{track.f_hi}."
            if not track.empty else
            f"<b>{self.reader.unit}</b>; this track has none."
        )
        if not track.bouts:
            # Said before anything is drawn, not only in the caption below: a
            # window grid drawn on a frame axis looks exactly like a bout table
            # with suspiciously regular boundaries.
            head += (
                f"<br><span style='color:{t['warning']}'>These are fixed windows "
                "off a grid, not decoded bouts.</span>"
            )
        if self.reader.stale:
            head += (
                f"<br><span style='color:{t['warning']}'>These predictions may be "
                f"out of date - {self.reader.stale}.</span>"
            )
        self.headline.setText(head)

        floor = float(getattr(self.cfg.suggest, "min_confidence", 0.0) or 0.0)
        lines = caption_lines(
            track, self.reader.resolution, floor, has_trace=bool(track.trace)
        )
        self.caption.setText("<br>".join(lines))

    def _say(self, head: str, problem: str) -> None:
        self.headline.setText(head)
        self.caption.setText("")
        self.timeline.set_track(
            TrackPredictions("", 0, False, ""), Resolution(), note=problem
        )


class BoutReviewDialog(QDialog):
    """The panel in a window of its own.

    A window rather than a fifth thing on the Model page: a timeline wants the
    width of the screen, and the page it is reached from is already carrying a
    task log and a cross-validation report.
    """

    def __init__(self, cfg: Config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Review predictions")
        self.resize(1040, 560)
        self.panel = BoutReviewPanel(cfg, self)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(self.panel, 1)
        layout.addWidget(buttons)

    def reload(self) -> None:
        self.panel.reload()
