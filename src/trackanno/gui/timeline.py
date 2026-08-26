"""One track's predictions along the frame axis, drawn rather than tabulated.

A row of ``predictions.parquet`` is either a fixed window off a grid or, under
``predict.decode``, a bout: a variable-length run between two inferred
behavioural boundaries. Which of the two a file holds comes from
:func:`trackanno.infer.predict.row_kind`, which reads the file.

The footage sits beside the drawing, because whether a boundary is in the right
place is a question about the animal at that frame and no timeline can answer
it. Every frame is cut from the movie, at one crop for the whole session and
centred on the track's own boxes: the clip cache is cut per movie and centred
per annotation window, so a picture taken partly from it stepped in size and in
centring wherever a window ran out. What that costs against the window the label
was applied in is said under the picture.

Boundaries are drawn as bands the width of the measured error against a planted
transition, a bout's mean posterior as a fill height, and a span overlapping
hand-labelled frames is hatched. Corrections are recorded in
``bout_corrections.csv`` (:mod:`trackanno.annotate.corrections`) and never in
``predictions.parquet`` or ``annotations.csv``; nothing here re-decodes,
retrains or writes a prediction back. The correction lane deliberately carries
no bar height, because what is now asserted about a stretch of frames is not a
posterior over it.
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
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel,
    QMessageBox, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QSplitter,
    QToolTip, QVBoxLayout, QWidget,
)

from ..config import Config
from ..annotate.corrections import (
    CorrectionStore, Partition, Span, apply_corrections, build_corrections,
    corrections_path, derivation_report, derive_window_labels, manifest_windows,
)
from ..view.overlay import build_clip_view
from ..annotate.store import MULTI_SEP
from .theme import tokens
from .viewer import SimpleViewer, follow_combo

__all__ = [
    "Segment", "TrackPredictions", "Resolution", "PredictionReader",
    "caption_lines", "ReviewCrop", "review_crop", "MovieFrames", "TrackFootage",
    "BoutTimeline", "BoutReviewPanel", "BoutReviewDialog",
]

# The README's measured boundary error, as divisors of the window stride.
# Divisors rather than frames, because the stride is a per-project setting.
MEAN_ERROR_DIVISOR = 4.0
WORST_ERROR_DIVISOR = 2.0

# What the review window opens on, rather than ui.follow. That setting is chosen
# for one 25-frame clip, where a moving average removes drift and keeps the beat.
# A bout runs for hundreds of frames, and over that long the average lags and the
# animal walks out of the middle of the picture; `locked` re-centres every frame.
REVIEW_FOLLOW = "locked"


@dataclass(frozen=True)
class Segment:
    """One drawn span: a decoded bout, or a window off a grid.

    ``label`` is ``""`` for a stretch the model has no opinion about, drawn as a
    hole rather than dropped. ``conf``, ``human_frac`` and ``n_covering`` are
    NaN where the file does not record them, never 0.0.
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

    # group -> the RECORD: the model's bouts with this reviewer's corrections
    # laid over them. Kept apart from `lanes` so both claims stay visible.
    partitions: dict = field(default_factory=dict)   # group -> Partition
    # Corrections that no longer land on any predicted frame, because the run
    # that wrote these rows covered a different span. Reported, never dropped.
    orphans: list = field(default_factory=list)
    # Corrections only PART of which lands on a predicted frame. Drawn short and
    # saved long, and listed separately because nothing else on screen says so.
    partly_shown: list = field(default_factory=list)
    # Why this track cannot be corrected, or "". Set rather than raised: a
    # windows file or a silent track is a normal state.
    edit_problem: str = ""

    @property
    def empty(self) -> bool:
        return not any(self.lanes.values())

    @property
    def n_segments(self) -> int:
        return sum(len(v) for v in self.lanes.values())

    @property
    def n_corrected(self) -> int:
        return sum(len(p.corrected_spans()) for p in self.partitions.values())


# --------------------------------------------------------------------------- #
# how sharp a boundary can be
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Resolution:
    """The window stride these bouts were cut on, and how far to trust an edge.

    ``stride`` is 0 when it could not be measured, and then the limit is
    reported as a ratio and no band is drawn. ``scales`` is ``(window length,
    that length's stride)`` for every grid found in the dense table, narrowest
    first; ``stride`` is the COARSEST of them (see :func:`measure_stride`).
    """

    stride: int = 0
    provenance: str = ""        # where the stride came from, or why there is none
    scales: tuple[tuple[int, int], ...] = ()

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
        out = (
            f"Boundaries are resolved to the {self.stride}-frame window stride, "
            f"not to the frame: {self.mean_error:.1f} frames out on average and "
            f"{self.worst_error:.0f} at worst, measured against a planted "
            f"transition. The band at each edge is that worst case. {self.provenance}."
        )
        if len(self.scales) > 1:
            listed = ", ".join(f"{length} frames at stride {step}"
                               for length, step in self.scales)
            out += (
                f" This run scored at {len(self.scales)} window lengths "
                f"({listed}), and the band is the COARSEST grid's: a boundary is "
                "no better placed than the widest window that voted on it, and a "
                "band drawn at the finest grid would claim the short windows had "
                "the long ones' agreement."
            )
        return out


def measure_stride(cfg: Config) -> Resolution:
    """The window grids on disk, measured, never read off the config.

    ``dense_clip_features.parquet`` is the plan the decode ran over, so the gap
    between one window start and the next is the stride that produced the bouts.
    Modal rather than minimal: a track whose windows do not divide evenly leaves
    one short step at the end.

    Measured per window length. Under ``predict.window_scales`` the table holds
    several grids whose starts interleave, and one mode over the union would
    report the finest grid's. The stride returned is the COARSEST grid's,
    because the decode averages every window covering a frame and a transition
    is no better placed than the bluntest window that voted on it. A length
    appearing once per track has no gap to measure and is skipped.

    A dense table written after the predictions is still measured, but the
    provenance says so.
    """
    from ..infer.densepredict import dense_features_path
    from ..infer.predict import predictions_path

    path = dense_features_path(cfg)
    if not path.exists():
        return Resolution(0, f"{path.name} is not in this project")
    try:
        frame = pd.read_parquet(
            path, columns=["movie_id", "track_id", "f_start", "f_end"]
        )
    except (OSError, ValueError, KeyError) as exc:
        return Resolution(0, f"{path.name} could not be read ({exc})")
    if frame.empty:
        return Resolution(0, f"{path.name} has no windows in it")

    starts = pd.to_numeric(frame["f_start"], errors="coerce").to_numpy(dtype=float)
    ends = pd.to_numeric(frame["f_end"], errors="coerce").to_numpy(dtype=float)
    lengths = ends - starts
    keys = frame[["movie_id", "track_id"]].astype({"movie_id": str})
    grids: dict[int, int] = {}
    for length in np.unique(lengths[np.isfinite(lengths) & (lengths > 0)]):
        # Positions, not index labels: `starts` is a plain array off the column
        # and a parquet read need not hand back a 0..n-1 index to align with it.
        rows = np.flatnonzero(lengths == length)
        steps: list[int] = []
        for positions in keys.iloc[rows].groupby(
            ["movie_id", "track_id"], sort=False
        ).indices.values():
            run = np.sort(starts[rows[np.asarray(positions)]])
            steps.extend(int(g) for g in np.diff(run) if np.isfinite(g) and g > 0)
        if steps:
            grids[int(length)] = int(pd.Series(steps).mode().iat[0])
    if not grids:
        return Resolution(0, f"every track in {path.name} has a single window")

    scales = tuple(sorted(grids.items()))
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
    return Resolution(max(grids.values()), where, scales)


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
    """``predictions.parquet`` and, if it is there, the posterior beside it.

    Built once per open of the review window and then queried per track. Both
    files are read whole, but the posterior is one row per (movie, track, frame)
    and is read lazily, on the first track a reviewer looks at.

    ``problem`` is set instead of raising, because a traceback out of a Qt slot
    takes the window down.
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
        self.corrections: CorrectionStore | None = None

        from ..infer.predict import load_predictions, row_kind, staleness
        from ..view.render import prediction_unit

        # Before the predictions, and fatal when it fails: a corrections file
        # that will not parse would leave the timeline drawing the model's bouts
        # over frames a person has already overruled.
        try:
            self.corrections = CorrectionStore(
                corrections_path(cfg), getattr(cfg.ui, "annotator", "")
            )
        except (OSError, ValueError) as exc:
            self.problem = str(exc)
            return

        try:
            self.frame = load_predictions(cfg)
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            self.problem = str(exc)
            return

        self.bouts = row_kind(self.frame).bouts
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
        # Only where an inferred boundary exists to be uncertain about: a window
        # edge is a grid multiple the predictor chose, not a measurement.
        if self.bouts:
            self.resolution = measure_stride(cfg)

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
        self._record(out)
        return out

    def _record(self, track: TrackPredictions) -> None:
        """The model's bouts with this project's corrections laid over them.

        Every way this can fail is a sentence on ``edit_problem`` rather than an
        exception: the file holds windows and not bouts, the track has no rows
        at all, or one group's rows overlap or leave a hole. A partition is
        refused rather than patched, since patching would invent frames.
        """
        if track.empty:
            track.edit_problem = (
                "this track has no predicted rows, so there is nothing to correct"
            )
            return
        if not track.bouts:
            track.edit_problem = (
                "these rows are fixed windows off a grid, not decoded bouts. A "
                "window's edges are multiples the predictor chose, so moving one "
                "corrects nothing about the animal - and a window relabelled by "
                "hand is a clip label, which belongs in annotations.csv. Set "
                "predict.decode and label again to get bouts."
            )
            return
        store = self.corrections
        for group, segments in track.lanes.items():
            if not segments:
                continue
            try:
                model = Partition(
                    [Span(s.f_start, s.f_end, s.label, False) for s in segments],
                    segments[0].f_start, segments[-1].f_end,
                )
            except ValueError as exc:
                track.edit_problem = (
                    f"the {group} rows for this track are not a usable decode: {exc}"
                )
                return
            saved = store.for_track(track.movie_id, track.track_id, group) \
                if store is not None else []
            try:
                partition, orphans, clipped = apply_corrections(model, saved)
            except ValueError as exc:
                track.edit_problem = f"a saved correction cannot be applied: {exc}"
                return
            track.partitions[group] = partition
            track.orphans.extend(orphans)
            track.partly_shown.extend(clipped)

    def _segments(self, rows: pd.DataFrame, group: str) -> list[Segment]:
        """One group's spans out of one track's rows, in frame order.

        A bout table carries one row per (group, bout), so a group owns only the
        rows whose ``group`` column names it. A windows table has no ``group``
        column and every row speaks for every group at once.
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
        # A windows file records a flag, a bout file a fraction, and a file with
        # neither reports NaN rather than 0.0.
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
            # Dropped, never defaulted: a guessed span would put a coloured bar
            # over frames nothing scored.
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
        case. Read once, on the first track a reviewer looks at.
        """
        if self._posterior_read:
            return self._posterior
        self._posterior_read = True
        from ..infer.predict import posteriors_path

        path = posteriors_path(self.cfg)
        if not path.exists():
            return None
        try:
            self._posterior = pd.read_parquet(path)
        except (OSError, ValueError) as exc:
            # Not fatal and not silent: the timeline is worth drawing without a
            # trace, but the reviewer is told the file is unreadable.
            self.problem = f"{path.name} could not be read ({exc})"
            self._posterior = None
        return self._posterior

    def _traces(self, track: TrackPredictions) -> dict:
        """Per-frame posterior of the DECODED class, per group.

        ``conf`` is a mean over the bout's frames and cannot separate a flat
        value from one that swings either side of a trough. Only the decoded
        class's column is followed, since that is the number ``conf`` is the
        mean of.
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
        index = frames - track.f_lo
        # A posterior written by a different run need not span the same frames
        # as the bouts. Neither this nor `index` depends on the segment.
        in_span = (index >= 0) & (index < n)
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
                inside = (
                    (frames >= segment.f_start) & (frames < segment.f_end) & in_span
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
    claims can be tested without a screenshot."""
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

    if track.partitions:
        corrected = track.n_corrected
        lines.append(
            "The lower lane of each pair is THE RECORD: the model's bouts with "
            "your corrections laid over them. A corrected span is filled solid "
            "and outlined; a span still the model's is faded and dashed. The "
            "record lane carries no bar height, because what is now asserted "
            "about a stretch of frames is not a posterior over it."
        )
        lines.append(
            f"{corrected} span(s) on this track are yours." if corrected else
            "Nothing on this track has been corrected, so the record is the "
            "model's answer unchanged."
        )
        if corrected:
            lines.append(
                "A boundary you placed is drawn as a hairline and a boundary the "
                "decode placed keeps its band: yours is exact to the frame you "
                "chose, and only the decode's is uncertain to the window stride. "
                "That is a statement about where the number came from, not about "
                "whether you were right."
            )
    if track.orphans:
        lines.append(
            f"{len(track.orphans)} saved correction(s) cover frames these "
            "predictions have no row for - the run that wrote them decoded a "
            "different span. They are kept in bout_corrections.csv and are not "
            "drawn; nothing has been deleted."
        )
    if track.partly_shown:
        listed = ", ".join(f"{c.f_start}-{c.f_end}" for c in track.partly_shown[:4])
        lines.append(
            f"{len(track.partly_shown)} saved correction(s) reach past what these "
            f"predictions cover ({listed}) and are drawn cut short at frame "
            f"{track.f_lo} or {track.f_hi}. The file keeps their real ends: "
            "clipping is how they are DRAWN, never how they are saved, so a "
            "re-run over a shorter span costs a picture and not a judgement."
        )
    if track.edit_problem and not track.empty:
        lines.append(f"Not correctable here: {track.edit_problem}")

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
    """One track's lanes on a shared frame axis, one lane per predicted group."""

    LANE = 52
    BAR = 34
    RECORD_LANE = 38
    RECORD_BAR = 22
    GAP = 6
    TOP = 10
    AXIS = 26
    RIGHT = 12
    # How near a boundary a press has to land to grab it rather than select the
    # span. In PIXELS, not frames: the reviewer is aiming with a mouse.
    GRAB_PX = 5.0

    # (group, f_start, f_end) of the record span the reviewer clicked, so the
    # panel's frame boxes can be filled from the picture rather than typed.
    spanClicked = Signal(str, int, int)
    # (group, boundary was at, boundary dragged to). The widget does not apply
    # it; the partition is what enforces the invariant.
    boundaryMoved = Signal(str, int, int)
    # The frame the reviewer pointed at, so the video can go there. Emitted on
    # press and on every drag, because scrubbing is how a boundary is checked.
    positionPicked = Signal(int)

    def __init__(self, cfg: Config | None = None, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.track = TrackPredictions("", 0, False, "")
        self.resolution = Resolution()
        self.note = ""
        self.selected: tuple[str, int, int] | None = None
        # The frame the video is showing, or None before it shows one.
        self.position: int | None = None
        self._scrub = False
        self._hits: list[tuple[QRectF, str, object, str]] = []
        # Set on every paint so the mouse handlers measure with the same ruler
        # the picture was drawn with.
        self._geom: tuple[float, float] = (0.0, 0.0)
        self._drag: tuple[str, int, int] | None = None
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumWidth(420)
        self.setMouseTracking(True)

    def set_track(self, track: TrackPredictions, resolution: Resolution,
                  note: str = "") -> None:
        self.track = track
        self.resolution = resolution
        self.note = note
        self.selected = None
        self.position = None
        self._drag = None
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
        records = len(self.track.partitions)
        return QSize(760, self.TOP + lanes * self.LANE
                     + records * self.RECORD_LANE + self.AXIS + 8)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    # -- geometry ------------------------------------------------------------ #
    def _lane_plan(self) -> list[tuple[str, str, float]]:
        """``[(kind, group, y)]`` top to bottom, the model lane then its record.

        One list for the painter and the hit tests, so a lane cannot be drawn at
        one height and clicked at another.
        """
        out: list[tuple[str, str, float]] = []
        y = float(self.TOP)
        for group in self.track.lanes:
            out.append(("model", group, y))
            y += self.LANE
            if group in self.track.partitions:
                out.append(("record", group, y))
                y += self.RECORD_LANE
        return out

    def _gutter(self, metrics=None) -> float:
        metrics = metrics if metrics is not None else self.fontMetrics()
        names = list(self.track.lanes) + (["record"] if self.track.partitions else [])
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
        """The frame under a pixel, clamped to the track. Rounded, not floored,
        so a boundary is dragged to the same frame from either side of it."""
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
        bottom = float(self.TOP)
        for kind, group, y in self._lane_plan():
            if kind == "model":
                self._paint_lane(painter, t, group, self.track.lanes[group], y,
                                 gutter, scale)
                bottom = y + self.LANE
            else:
                self._paint_record(painter, t, group,
                                   self.track.partitions[group], y, gutter, scale)
                bottom = y + self.RECORD_LANE
        self._paint_axis(painter, t, bottom, gutter, scale)
        self._paint_position(painter, t, bottom, gutter, scale)
        painter.end()

    def _paint_position(self, painter, t, bottom, gutter, scale) -> None:
        """The frame the video is on, across every lane at once.

        Last, so it is over the bars: it is the one mark here that says where
        you are rather than what was predicted.
        """
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

        # Under the bars: a band is a statement about the edges between them and
        # must not obscure the fills.
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
        declares, as :func:`trackanno.view.render.build_frame_marks` does, so a
        renamed option looks orphaned here too.
        """
        spec = self.cfg.group(group) if self.cfg is not None else None
        option = spec.option(label.split(MULTI_SEP)[0]) if spec is not None else None
        colour = QColor(option.color) if option is not None else QColor(t["faint"])
        return colour if colour.isValid() else QColor(t["faint"])

    def _paint_record(self, painter, t, group, partition, y, gutter, scale) -> None:
        """The record lane: what is now asserted about every frame of this track.

        Deliberately without a bar height: the record has no confidence number,
        and a correction cut out of a bout does not inherit the bout's mean.
        """
        bar_top = y + (self.RECORD_LANE - self.RECORD_BAR) / 2
        bar = QRectF(gutter, bar_top, max(0.0, self.width() - gutter - self.RIGHT),
                     self.RECORD_BAR)

        painter.setPen(QColor(t["muted"]))
        painter.drawText(QRectF(2, bar_top, gutter - 8, self.RECORD_BAR),
                         int(Qt.AlignVCenter | Qt.AlignRight), "record")
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(t["raised"]))
        painter.drawRoundedRect(bar, 3, 3)

        # A boundary the DECODE placed keeps the stride band; one a person placed
        # does not, because only an inferred boundary is uncertain to the stride.
        model_cuts = {s.f_start for s in self.track.lanes.get(group, [])[1:]}
        if self.resolution.known:
            half = max(1.0, self.resolution.worst_error * scale)
            band = QColor(t["border_hi"])
            band.setAlpha(90)
            painter.setBrush(band)
            painter.setPen(Qt.NoPen)
            for cut in partition.boundaries():
                if cut in model_cuts:
                    x = self._x(cut, gutter, scale)
                    painter.drawRect(QRectF(x - half, bar.top(), 2 * half,
                                            self.RECORD_BAR))

        metrics = painter.fontMetrics()
        for span in partition.spans:
            x0 = self._x(span.f_start, gutter, scale)
            x1 = self._x(span.f_end, gutter, scale)
            rect = QRectF(x0, bar.top(), max(1.0, x1 - x0), self.RECORD_BAR)
            self._hits.append((rect, group, span, "record"))
            if not span.label:
                hatch = QColor(t["faint"])
                hatch.setAlpha(70)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(hatch, Qt.FDiagPattern))
                painter.drawRect(rect)
                continue
            colour = self._colour(group, span.label, t)
            if not span.corrected:
                # Faded and dashed: this span is still the model's.
                colour.setAlpha(90)
            painter.setPen(Qt.NoPen)
            painter.setBrush(colour)
            painter.drawRect(rect)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(t["text"]), 2) if span.corrected
                           else QPen(QColor(t["border"]), 1, Qt.DashLine))
            painter.drawRect(rect)
            text = span.label if span.corrected else f"({span.label})"
            if rect.width() > metrics.horizontalAdvance(text) + 8:
                painter.setPen(QColor(t["text"] if span.corrected else t["muted"]))
                painter.drawText(rect, int(Qt.AlignCenter), text)

        # A boundary a person placed, drawn as the hairline it is entitled to be.
        painter.setPen(QPen(QColor(t["text"]), 1))
        for cut in partition.boundaries():
            if cut not in model_cuts:
                x = self._x(cut, gutter, scale)
                painter.drawLine(QPointF(x, bar.top() - 3),
                                 QPointF(x, bar.bottom() + 3))

        if self.selected is not None and self.selected[0] == group:
            _, f_start, f_end = self.selected
            x0 = self._x(f_start, gutter, scale)
            x1 = self._x(f_end, gutter, scale)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(t["accent"]), 2))
            painter.drawRect(QRectF(x0 - 1, bar.top() - 3, max(2.0, x1 - x0) + 2,
                                    self.RECORD_BAR + 6))

        if self._drag is not None and self._drag[0] == group:
            x = self._x(self._drag[2], gutter, scale)
            painter.setPen(QPen(QColor(t["accent"]), 2, Qt.DashLine))
            painter.drawLine(QPointF(x, bar.top() - 6), QPointF(x, bar.bottom() + 6))

    def _paint_segment(self, painter, t, group, segment, bar, gutter, scale) -> None:
        x0 = self._x(segment.f_start, gutter, scale)
        x1 = self._x(segment.f_end, gutter, scale)
        rect = QRectF(x0, bar.top(), max(1.0, x1 - x0), self.BAR)
        self._hits.append((rect, group, segment, "model"))

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

        if np.isfinite(segment.human_frac) and segment.human_frac > 0:
            # Hatched over the whole span: the file records what fraction of the
            # bout a person judged, never which frames.
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

        Broken at every undefined frame rather than bridged, so no line is drawn
        across frames no window covered.
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
        for rect, group, item, kind in self._hits:
            if rect.contains(QPointF(point)):
                return group, item, kind
        return None

    # -- editing with the mouse ---------------------------------------------- #
    def mousePressEvent(self, event):  # noqa: N802 (Qt naming)
        """Grab a boundary if the press is near one, otherwise select a span and
        take the video to the frame under the mouse.

        A boundary is grabbed on the record lane only: the model lane is a
        picture of what the run produced and is never edited. Everything else is
        a scrub, including a click on a span, because "which frames are these?"
        is answered by watching them.
        """
        if event.button() != Qt.LeftButton or self.note or self.track.empty:
            super().mousePressEvent(event)
            return
        hit = self._at(event.pos())
        if hit is not None and hit[2] == "record":
            group, span, _ = hit
            partition = self.track.partitions.get(group)
            if partition is not None:
                gutter, scale = self._geom
                near = [c for c in partition.boundaries()
                        if abs(self._x(c, gutter, scale) - event.pos().x())
                        <= self.GRAB_PX]
                if near:
                    cut = min(near, key=lambda c: abs(self._x(c, gutter, scale)
                                                      - event.pos().x()))
                    self._drag = (group, cut, cut)
                    self.update()
                    return
                self.selected = (group, span.f_start, span.f_end)
                self.spanClicked.emit(group, span.f_start, span.f_end)
        self._scrub = True
        self._pick(event.pos().x())

    def mouseMoveEvent(self, event):  # noqa: N802 (Qt naming)
        if self._drag is not None:
            group, cut, _ = self._drag
            self._drag = (
                group, cut,
                self._clamp_boundary(group, cut, self._frame_at(event.pos().x())),
            )
            self.update()
            return
        if self._scrub:
            self._pick(event.pos().x())
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802 (Qt naming)
        if self._scrub:
            self._scrub = False
            return
        if self._drag is None:
            super().mouseReleaseEvent(event)
            return
        group, cut, moved = self._drag
        self._drag = None
        self.update()
        if moved != cut:
            self.boundaryMoved.emit(group, cut, moved)

    def _pick(self, x: float) -> None:
        frame = self._frame_at(x)
        self.set_position(frame)
        self.positionPicked.emit(frame)
        self.update()

    def _clamp_boundary(self, group: str, cut: int, frame: int) -> int:
        """Where a dragged boundary is allowed to end up.

        Inside the two spans it separates and no further: a drag past a
        neighbour's far edge would delete that span, which is what the merge
        buttons are for.
        """
        partition = self.track.partitions.get(group)
        if partition is None:
            return cut
        index = partition.index_at(cut)
        if index <= 0:
            return cut
        left, right = partition.spans[index - 1], partition.spans[index]
        return int(min(max(frame, left.f_start + 1), right.f_end - 1))

    def _describe(self, group: str, item, kind: str = "model") -> str:
        if kind == "record":
            return self._describe_span(group, item)
        segment = item
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

    def _describe_span(self, group: str, span) -> str:
        """The record's own account of a stretch of frames.

        It says who is claiming it and never how confident anyone is: a
        corrected span has no posterior, and a model span cut down by a
        correction no longer has the one its bout was measured with.
        """
        lines = [
            f"{group}: {span.label or '(no prediction)'}",
            f"frames {span.f_start}-{span.f_end} ({span.n_frames} frames)",
        ]
        if span.corrected:
            lines.append(
                "asserted by a person. This is not a measurement and carries no "
                "confidence; the model's own answer for these frames is the lane "
                "above, unchanged."
            )
        elif span.label:
            lines.append(
                "the model's, uncorrected. Where it has been cut short by a "
                "correction beside it, the confidence in the lane above is still "
                "the mean over the WHOLE bout it came from, not over this part."
            )
        else:
            lines.append(
                "no window over these frames could be featurised, so nothing has "
                "scored them. A person may still say what they are."
            )
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
    source: str                  # where the number came from, in a phrase

    @property
    def scale(self) -> float:
        """Delivered pixels per movie pixel."""
        return self.size / float(self.side)

    def sentence(self) -> str:
        shown = "" if self.size == self.side else f" shown at {self.size} px"
        return f"{self.side} px crop{shown}, the same on every track and movie here"


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
    movies = 0
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
                movies += 1

    ceiling = max(32, int(getattr(cfg.clips, "max_size", 256) or 256))
    if widths:
        pooled = np.concatenate(widths)
        side = int(np.ceil(float(np.percentile(pooled, 99)) / 16.0) * 16)
        source = (f"sized to hold 99% of the {pooled.size} windows this "
                  f"project's cache was cut over, across {movies} movie(s)")
    else:
        declared = getattr(cfg.clips, "crop_size", "auto")
        named = "clips.crop_size" if isinstance(declared, int) else "clips.max_size"
        side = int(declared) if isinstance(declared, int) else ceiling
        source = f"{named}, there being no clip manifest to measure"
    side = max(16, side)
    return ReviewCrop(side=side, size=min(side, ceiling), source=source)


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
    def want(self, request_id: int, first: int) -> None:
        """Ask for the chunk at ``first`` now. Latest wins: a scrub produces a
        request per mouse move and only the last is worth decoding."""
        with self._lock:
            self._request = (int(request_id), int(first))
        self._start()

    def read_ahead(self, firsts) -> None:
        """Queue the chunks the playhead is about to reach, behind the urgent
        one. The same worker, because one movie handle is not two threads' to
        seek."""
        with self._lock:
            self._ahead = [int(f) for f in firsts]
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
                    request = (self.AHEAD, self._ahead.pop(0))
            if request is None:
                self._wake.wait(timeout=0.25)
                self._wake.clear()
                continue
            if self._stop:
                break
            request_id, first = request
            try:
                self.ready.emit(request_id, first, self._cut(first))
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
    def _cut(self, first: int) -> tuple[np.ndarray, pd.DataFrame]:
        import cv2

        from ..annotate.clips import _crop

        movie = self._open()
        n = min(self.chunk, max(0, int(len(movie)) - int(first)))
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
    which re-centres every frame rather than following ``ui.follow``. Both are
    said under the picture, because a difference discovered mid-review is read
    as a difference in the recording.

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
        if movie is not None:
            movie.ready.disconnect()
            movie.failed.disconnect()
            if not movie.stop():
                print("[trackanno] review decode thread did not stop in time")
        self._problem = ""

    # -- one track ----------------------------------------------------------- #
    def set_track(self, movie_id: str, track_id: int, f_lo: int, f_hi: int) -> None:
        self.viewer.pause()
        self._release_movie()
        # The frames of the track being left, which are not this one's.
        self._chunks.clear()
        self._movie_id, self._track_id = str(movie_id), int(track_id)
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

    # -- moving ------------------------------------------------------------- #
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
        first = (frame // REVIEW_CHUNK) * REVIEW_CHUNK
        held = self._chunks.get(first)
        if held is not None:
            self._show_chunk(first, held, frame)
            self._read_ahead(first)
            return
        self._request += 1
        self._pending = (self._request, frame)
        self._awaiting = frame
        self._frames().want(self._request, first)
        self._read_ahead(first)
        if not self._length:
            self.source.setText(
                f"reading frames {first}-{first + REVIEW_CHUNK} of "
                f"{self._movie_id}..."
            )

    def _frames(self) -> MovieFrames:
        """The decoder for this track, started on first use."""
        if self._movie is None:
            self._movie = MovieFrames(self.cfg, self._movie_id, self._track_id,
                                      self.crop, REVIEW_CHUNK)
            self._movie.ready.connect(self._on_chunk)
            self._movie.failed.connect(self._on_failed)
        return self._movie

    def _read_ahead(self, first: int) -> None:
        """Decode the chunks the playhead is about to reach, on the same worker.

        A scrub asks for a frame per mouse move and playback walks straight off
        the end of the chunk it is in, so without this every chunk edge is a
        stall the length of a decode.
        """
        movie = self._movie
        if movie is None or not int(getattr(self.cfg.ui, "prefetch", 0) or 0):
            return
        end = max(self._span[1], self._span[0] + 1)
        movie.read_ahead([
            f for f in (first + REVIEW_CHUNK, first - REVIEW_CHUNK,
                        first + 2 * REVIEW_CHUNK)
            if 0 <= f < end and f not in self._chunks
        ])

    def _keep(self, first: int, payload) -> None:
        """Hold the decoded chunks nearest the playhead and let the rest go."""
        self._chunks[int(first)] = payload
        while len(self._chunks) > CHUNKS_KEPT:
            self._chunks.pop(max(self._chunks, key=lambda f: abs(f - self._first)))

    def _show_chunk(self, first: int, payload, frame: int) -> None:
        array, marks = payload
        if self._loaded != int(first):
            self.viewer.set_clip(array, self._view(marks, array))
            self._mark()
            self.viewer.setVisible(True)
            self.controls.setVisible(True)
            self._loaded = int(first)
            self._first = int(first)
            self._length = int(len(array))
        self._seek(frame - self._first, frame)
        self.source.setText(self._note())
        self.source.setToolTip(self._difference())

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
            smooth_window=int(getattr(ui, "smooth_window", 15)),
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
            return                       # the end of the track; the chunk loops
        was = self._loaded
        self.viewer.pause()
        self.show_frame(nxt)
        if self._awaiting == nxt:
            self._resume_at = nxt        # the movie is being read for it
        elif self._loaded != was and self._position == nxt and self._length:
            self.viewer.play()

    # -- what the decoder sends back ----------------------------------------- #
    def _on_chunk(self, request_id: int, first: int, payload) -> None:
        if self.sender() is not self._movie:
            # A read the track before this one had out. Its frames are of
            # another animal and must not reach the chunks kept here.
            return
        self._keep(first, payload)
        if int(request_id) != self._request:
            return                        # a read-ahead, or a scrub moved on
        self._awaiting = -1
        array, _ = payload
        frame = int(min(max(self._pending[1], first), first + len(array) - 1))
        self._show_chunk(first, payload, frame)
        if self._resume_at == frame:
            # Playback stopped at the edge of a chunk waiting for these.
            self._resume_at = -1
            self.viewer.play()

    def _on_failed(self, request_id: int, message: str) -> None:
        """Asked once and then remembered: a movie that has moved will not be
        back by the next mouse move, and re-opening it per scrub would hang the
        window on a dead share."""
        if self.sender() is not self._movie or int(request_id) != self._request:
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
        """One line under the picture: which frame this is, and the two ways it
        is deliberately not the frame the annotation window would show."""
        crop = self.crop
        note = (
            f"frame {self._position} of track {self._track_id}, cut from "
            f"{self._movie_id}."
        )
        if crop is not None:
            note += f" {crop.sentence()}, centred on the track every frame."
        gap = getattr(self._movie, "skeleton_problem", "")
        if gap and bool(getattr(self.cfg.ui, "show_skeleton", False)):
            note += f" {gap}."
        return note

    def _difference(self) -> str:
        """The long form, on hover: what this window does that the annotation
        window does not, and why."""
        crop = self.crop
        return (
            "Every frame is cut from the movie, never from the clip cache: the "
            "cache is cut per movie and centred per annotation window, so a "
            "picture taken from both changes size and centring wherever a "
            "window runs out.\n\n"
            f"The crop is {crop.source if crop else 'not measured yet'}. It "
            "follows a moving average of the track over the whole track, so a "
            "frame is cut the same way in every read, and the view re-centres "
            "on the animal every frame.\n\n"
            "The normalisation and the marker are the annotation window's. The "
            "crop and the view are not, and those are the only two differences."
        )

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
        self.source.setText(text)


# --------------------------------------------------------------------------- #
# the panel and its window
# --------------------------------------------------------------------------- #
class BoutReviewPanel(QWidget):
    """Movie, track, timeline, captions, and the controls that correct a bout.

    Every failure the files can present is a sentence in this panel rather than
    an exception. Every editing gesture goes through :meth:`assert_class`, which
    is :meth:`trackanno.annotate.corrections.Partition.assert_class` plus a save: on a
    gapless run-length encoding, setting a class, merging either way and
    dragging a boundary are four gestures over one operation.

    A save happens on every accepted edit rather than on a Save button, so the
    file is the state, as ``annotations.csv`` is. Undo is a stack of whole
    partitions, so an undo is a save too.
    """

    def __init__(self, cfg: Config | None = None, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.reader: PredictionReader | None = None
        # Raised while the combos are being refilled: Qt emits
        # currentIndexChanged from addItems, once per item added.
        self._loading = False
        self._track: TrackPredictions | None = None
        self._undo: list[tuple[str, Partition]] = []

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
        self.timeline.spanClicked.connect(self._on_span_clicked)
        self.timeline.boundaryMoved.connect(self._on_boundary_moved)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.timeline)

        # The picture and the pixels it is a picture of, side by side and tied
        # both ways: scrubbing the timeline moves the video, playing the video
        # moves the marker. Whether a boundary is in the right place is not a
        # question a drawing can answer on its own.
        self.footage = TrackFootage(cfg)
        self.timeline.positionPicked.connect(self.footage.show_frame)
        self.footage.positionChanged.connect(self.timeline.set_position)
        self.split = QSplitter(Qt.Horizontal)
        self.split.addWidget(self.footage)
        self.split.addWidget(scroll)
        self.split.setStretchFactor(0, 0)
        self.split.setStretchFactor(1, 1)
        self.split.setSizes([400, 760])

        self.caption = QLabel("")
        self.caption.setWordWrap(True)
        self.caption.setTextFormat(Qt.RichText)
        self.caption.setProperty("role", "muted")

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.headline)
        layout.addWidget(self.split, 1)
        layout.addLayout(self._build_edit_row())
        layout.addWidget(self.status)
        layout.addWidget(self.caption)

        if cfg is not None:
            self.reload()

    def _build_edit_row(self) -> QHBoxLayout:
        """The correction controls. Always present and never gated on a config
        key: whether a project can be corrected is a property of the file on
        disk."""
        self.group = QComboBox()
        self.group.setMinimumWidth(120)
        self.group.currentIndexChanged.connect(self._on_group)
        self.from_frame = QSpinBox()
        self.to_frame = QSpinBox()
        for box in (self.from_frame, self.to_frame):
            box.setRange(0, 10_000_000)
            box.setMinimumWidth(90)
        self.klass = QComboBox()
        self.klass.setMinimumWidth(120)

        self.apply_button = QPushButton("Set class")
        self.apply_button.setToolTip(
            "Assert that these frames are this class. Every other gesture here - "
            "moving a boundary, splitting a bout, merging two - is this same "
            "assertion over a different range."
        )
        self.apply_button.clicked.connect(self._on_apply)
        self.merge_left = QPushButton("Merge into previous")
        self.merge_left.clicked.connect(lambda: self._on_merge(-1))
        self.merge_right = QPushButton("Merge into next")
        self.merge_right.clicked.connect(lambda: self._on_merge(1))
        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self._on_undo)
        self.revert_button = QPushButton("Revert track")
        self.revert_button.setToolTip(
            "Delete every correction on this track and group, restoring the "
            "model's own bouts."
        )
        self.revert_button.clicked.connect(self._on_revert)
        self.derive_button = QPushButton("Derived labels...")
        self.derive_button.setToolTip(
            "Show which annotation windows these corrections would label, and "
            "which straddle a boundary and cannot be labelled at all. Writes "
            "nothing."
        )
        self.derive_button.clicked.connect(self._on_derive)

        self.status = QLabel("")
        self.status.setWordWrap(True)

        row = QHBoxLayout()
        row.addWidget(QLabel("frames"))
        row.addWidget(self.from_frame)
        row.addWidget(QLabel("to"))
        row.addWidget(self.to_frame)
        row.addWidget(QLabel("are"))
        row.addWidget(self.klass)
        row.addWidget(QLabel("in"))
        row.addWidget(self.group)
        row.addWidget(self.apply_button)
        row.addWidget(self.merge_left)
        row.addWidget(self.merge_right)
        row.addWidget(self.undo_button)
        row.addWidget(self.revert_button)
        row.addWidget(self.derive_button)
        row.addStretch(1)
        return row

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
        """Put the footage back. The movie was let go of when this window was
        last hidden, and the Model page reuses one dialog for the life of the
        session, so re-showing it must not leave a sentence where the video
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
        self._track = track
        self._undo.clear()
        self.timeline.set_track(track, self.reader.resolution)
        if track.empty:
            self.footage.clear(
                f"track {track.track_id} of {track.movie_id} has no predicted "
                "rows, so there is no span to watch"
            )
        else:
            self.footage.set_track(movie_id, int(track_id), track.f_lo, track.f_hi)
        self._fill_groups()
        t = tokens()
        head = (
            f"<b>{self.reader.unit}</b> over {len(self.reader.groups)} category "
            f"group(s); this track has <b>{track.n_segments}</b> row(s) across "
            f"frames {track.f_lo}-{track.f_hi}."
            if not track.empty else
            f"<b>{self.reader.unit}</b>; this track has none."
        )
        if not track.bouts:
            # Said before anything is drawn: a window grid on a frame axis looks
            # like a bout table with suspiciously regular boundaries.
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

        self._caption()

    def _caption(self) -> None:
        track, floor = self._track, 0.0
        if track is None or self.reader is None:
            return
        if self.cfg is not None:
            floor = float(getattr(self.cfg.suggest, "min_confidence", 0.0) or 0.0)
        lines = caption_lines(
            track, self.reader.resolution, floor, has_trace=bool(track.trace)
        )
        self.caption.setText("<br>".join(lines))

    def _say(self, head: str, problem: str) -> None:
        self.headline.setText(head)
        self.caption.setText("")
        self._track = None
        self.timeline.set_track(
            TrackPredictions("", 0, False, ""), Resolution(), note=problem
        )
        self.footage.clear(problem)
        self._refresh_edit_gate()

    # -- correcting ----------------------------------------------------------- #
    def current_group(self) -> str:
        return str(self.group.currentText())

    def _fill_groups(self) -> None:
        """The groups this track can be corrected in, and their classes.

        Rebuilt per track rather than per project, because which groups have a
        partition depends on the rows this track actually has.
        """
        track = self._track
        self._loading = True
        try:
            self.group.clear()
            for group in (track.partitions if track is not None else {}):
                self.group.addItem(group)
        finally:
            self._loading = False
        self._on_group(0)

    def _on_group(self, _index: int = 0) -> None:
        if getattr(self, "_loading", False):
            return
        self._fill_classes()
        self._refresh_edit_gate()

    def _fill_classes(self) -> None:
        """Only classes the config declares, and only for a single-choice group.

        A class the config does not name could not be coloured, trained on or
        exported. A multi group's class is a combination of options rather than
        one of them, which this control cannot assert.
        """
        self.klass.clear()
        spec = self.cfg.group(self.current_group()) if self.cfg is not None else None
        if spec is None or spec.is_multi:
            return
        self.klass.addItems([o.name for o in spec.options])

    def _group_problem(self, group: str | None = None) -> str:
        """Why one group cannot be corrected, or "".

        Takes the group rather than always reading the combo, so an edit
        arriving from the timeline is judged against its own lane.
        """
        track = self._track
        if track is None or self.reader is None:
            return "no track is open"
        if track.edit_problem:
            return track.edit_problem
        group = group or self.current_group()
        if group not in track.partitions:
            return "this track has no decoded bouts in that group"
        spec = self.cfg.group(group) if self.cfg is not None else None
        if spec is None:
            return (
                f"the config no longer declares a category group called {group!r}, "
                "so there is no set of classes to correct these bouts to"
            )
        if spec.is_multi:
            return (
                f"{group} is a multi-select group: one of its predictions is a "
                "COMBINATION of options, not one of them, and this control can "
                "only assert a single class. Correcting a multi-select group is "
                "not built rather than half built."
            )
        return ""

    def _refresh_edit_gate(self) -> None:
        problem = self._group_problem()
        for button in (self.apply_button, self.merge_left, self.merge_right,
                       self.revert_button, self.derive_button):
            button.setEnabled(not problem)
        for box in (self.from_frame, self.to_frame, self.klass):
            box.setEnabled(not problem)
        # The group combo stays live whatever the problem is: the refusals are
        # per group, so disabling it would trap the reviewer in one lane.
        self.group.setEnabled(self.group.count() > 1)
        self.undo_button.setEnabled(bool(self._undo) and not problem)
        if problem:
            self.status.setText(problem)

    def _on_span_clicked(self, group: str, f_start: int, f_end: int) -> None:
        if self.group.currentText() != group:
            self.group.setCurrentText(group)
        self.from_frame.setValue(int(f_start))
        self.to_frame.setValue(int(f_end))
        partition = (self._track.partitions.get(group)
                     if self._track is not None else None)
        span = partition.at(f_start) if partition is not None else None
        if span is not None and span.label:
            index = self.klass.findText(span.label)
            if index >= 0:
                self.klass.setCurrentIndex(index)
        self.status.setText("")

    def _on_boundary_moved(self, group: str, was: int, now: int) -> None:
        """A dragged boundary, as the assertion it is.

        The DIRECTION decides which class is asserted: dragging right extends
        the span on the left over the frames swept, dragging left the one on the
        right.
        """
        track = self._track
        partition = track.partitions.get(group) if track is not None else None
        if partition is None:
            return
        index = partition.index_at(was)
        if index <= 0:
            return
        left, right = partition.spans[index - 1], partition.spans[index]
        source = left if now > was else right
        f_start, f_end = (was, now) if now > was else (now, was)
        if not source.label:
            self.status.setText(
                f"frames {source.f_start}-{source.f_end} have no class: no window "
                "over them could be featurised. Dragging a boundary into them "
                "would spread 'nothing scored this' over frames something did - "
                "set a class over the range instead."
            )
            return
        self.status.setText(self.assert_class(f_start, f_end, source.label, group))

    def _on_apply(self) -> None:
        self.status.setText(self.assert_class(
            self.from_frame.value(), self.to_frame.value(), self.klass.currentText()
        ))

    def _on_merge(self, direction: int) -> None:
        """Merge the selected span into a neighbour: assert the neighbour's class
        over it. There is no separate merge in the data model, only this."""
        track = self._track
        group = self.current_group()
        partition = track.partitions.get(group) if track is not None else None
        if partition is None:
            return
        index = partition.index_at(self.from_frame.value())
        if index < 0:
            self.status.setText(
                f"frame {self.from_frame.value()} is not on this track "
                f"({partition.f_lo}-{partition.f_hi}); click a span to select it"
            )
            return
        neighbour = index + direction
        if not 0 <= neighbour < len(partition):
            self.status.setText(
                "there is no bout on that side to merge into - this one is at the "
                "end of the track"
            )
            return
        span, other = partition.spans[index], partition.spans[neighbour]
        if not other.label:
            self.status.setText(
                f"frames {other.f_start}-{other.f_end} have no class: nothing could "
                "score them, and merging into them would assert that nothing can"
            )
            return
        self.status.setText(
            self.assert_class(span.f_start, span.f_end, other.label, group)
        )

    def assert_class(self, f_start: int, f_end: int, label: str,
                     group: str | None = None) -> str:
        """Record that a person says these frames are this class. "" on success.

        A refusal comes back as a sentence rather than an exception, since a
        drag that ran backwards or a range off the end of the track is a normal
        outcome. The FILE IS WRITTEN FIRST and the partition on screen advances
        only if the write succeeded.
        """
        group = group or self.current_group()
        problem = self._group_problem(group)
        if problem:
            return problem
        track = self._track
        partition = track.partitions.get(group)
        if partition is None:
            return "this track has no decoded bouts in that group"
        spec = self.cfg.group(group) if self.cfg is not None else None
        if spec is not None and spec.option(str(label)) is None:
            # Refused here and not on the way in: the file must still LOAD a
            # class the config has since dropped, because that row records what
            # a person judged. Only a NEW judgement is refused.
            return (
                f"{group} declares no class called {str(label)!r}, so a correction "
                "to it could not be coloured, trained on or exported. Add it to "
                "config.yaml first, or pick one of: "
                + ", ".join(o.name for o in spec.options)
            )
        try:
            edited = partition.assert_class(f_start, f_end, label)
        except ValueError as exc:
            return str(exc)
        return self._commit(group, partition, edited,
                            f"frames {f_start}-{f_end} are {label}")

    def _commit(self, group: str, before: Partition, after: Partition,
                what: str, record_undo: bool = True) -> str:
        track, reader = self._track, self.reader
        if track is None or reader is None or reader.corrections is None:
            return "no project open"
        store = reader.corrections
        try:
            store.replace_track(track.movie_id, track.track_id, group, build_corrections(
                after, track.movie_id, track.track_id, group,
                getattr(self.cfg.ui, "annotator", "") if self.cfg else "",
                model_bouts=self._model_bouts(group),
                previous=store.for_track(track.movie_id, track.track_id, group),
            ))
        except (OSError, ValueError) as exc:
            return f"the correction could not be saved, so it was not made: {exc}"
        if record_undo:
            self._undo.append((group, before))
        track.partitions[group] = after
        self.timeline.selected = None
        self.timeline.update()
        self.timeline.updateGeometry()
        self._caption()
        self._refresh_edit_gate()
        return f"saved: {what}"

    def _model_bouts(self, group: str) -> list[dict]:
        """What the model said over this track, for the record of what a
        correction overruled. Read off the drawn segments, so it is the same
        rows the reviewer was looking at."""
        track = self._track
        if track is None:
            return []
        return [
            {"f_start": s.f_start, "f_end": s.f_end, "label": s.label, "conf": s.conf}
            for s in track.lanes.get(group, [])
        ]

    def _on_undo(self) -> None:
        if not self._undo:
            self.status.setText("nothing to undo on this track")
            return
        group, previous = self._undo.pop()
        current = self._track.partitions.get(group) if self._track else None
        if current is None:
            return
        # An undo is not itself undoable, so the button walks back through the
        # edits instead of ping-ponging between two versions.
        message = self._commit(group, current, previous, "undone", record_undo=False)
        self.status.setText(message)

    def _on_revert(self) -> None:
        """Drop every correction on this track and group. Confirmed first,
        because it is the one action here that destroys a human judgement rather
        than recording one."""
        track, group = self._track, self.current_group()
        if track is None or self.reader is None or self.reader.corrections is None:
            return
        saved = self.reader.corrections.for_track(track.movie_id, track.track_id, group)
        if not saved:
            self.status.setText("this track has no corrections in that group")
            return
        key = (track.movie_id, track.track_id, group)
        # The rows that are NOT on screen are counted before they are destroyed,
        # so the reviewer knows they are in the number they are confirming.
        unseen = len([c for c in track.orphans + track.partly_shown if c.key == key])
        answer = QMessageBox.question(
            self, "Delete corrections",
            f"Delete all {len(saved)} correction(s) on track {track.track_id} of "
            f"{track.movie_id} ({group})?\n\nThe model's own bouts come back. "
            "Nothing else is changed and no prediction is rewritten."
            + (f"\n\n{unseen} of them cover frames these predictions have no row "
               "for, so they are not on the timeline - they are deleted too."
               if unseen else ""),
        )
        if answer != QMessageBox.Yes:
            return
        segments = track.lanes.get(group, [])
        model = Partition(
            [Span(s.f_start, s.f_end, s.label, False) for s in segments],
            segments[0].f_start, segments[-1].f_end,
        )
        try:
            self.reader.corrections.replace_track(
                track.movie_id, track.track_id, group, []
            )
        except (OSError, ValueError) as exc:
            self.status.setText(f"the corrections could not be deleted: {exc}")
            return
        self._undo.clear()
        track.partitions[group] = model
        track.orphans = [c for c in track.orphans if c.key != key]
        track.partly_shown = [c for c in track.partly_shown if c.key != key]
        self.timeline.selected = None
        self.timeline.update()
        self._caption()
        self._refresh_edit_gate()
        self.status.setText(f"{len(saved)} correction(s) deleted")

    # -- what the corrections would mean for the training set ------------------ #
    def _on_derive(self) -> None:
        QMessageBox.information(self, "Derived window labels", self.derivation_text())

    def derivation_text(self) -> str:
        """The derivation report for this movie, as prose. Writes nothing.

        A string rather than a dialog full of labels, so the numbers in it can
        be checked by a test.
        """
        track, reader = self._track, self.reader
        if track is None or reader is None or reader.corrections is None:
            return "no project open"
        windows = manifest_windows(self.cfg, track.movie_id)
        if not windows:
            return (
                f"{track.movie_id} has no clip manifest, so there are no annotation "
                "windows to derive anything for. The corrections are still recorded."
            )
        derivation = derive_window_labels(
            reader.corrections.for_movie(track.movie_id), windows
        )
        return derivation_report(derivation)


class BoutReviewDialog(QDialog):
    """The panel in a window of its own.

    Close and no Save: every accepted edit is already on disk by the time the
    button could be pressed. Closing also lets go of the movie, through the
    panel's own hide handler, so leaving this window open holds no file open.
    """

    def __init__(self, cfg: Config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Review and correct predictions")
        # Room for a square of film beside the lanes: the two are read against
        # each other, and a video that has to be scrolled to is not.
        self.resize(1280, 780)
        self.panel = BoutReviewPanel(cfg, self)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(self.panel, 1)
        layout.addWidget(buttons)

    def reload(self) -> None:
        self.panel.reload()
