"""One track's predictions along the frame axis, drawn rather than tabulated.

``predictions.parquet`` is the only place the model's answer to "what was this
animal doing, and when did it change" lives, and under ``predict.decode`` a row
stopped being a window off a grid and became a BOUT - a variable-length run
between two inferred behavioural boundaries. A bout is unreviewable as a table
row: what it claims is where it starts, where it stops, and how sure the model
was over exactly those frames, and none of those three read off a column of
integers. So they are drawn, on the axis the claim is about.

Editable, and the edits go somewhere else. A reviewer who can see that a
boundary is wrong and cannot say so is being asked to trust a picture they have
already disbelieved. What they say is recorded in ``bout_corrections.csv``
(:mod:`trackanno.corrections`) and never in ``predictions.parquet`` or
``annotations.csv``: a correction is a record ABOUT a run, not a rewritten run,
and it is an assertion about a variable-length span, which the clip table has no
row shape for. Nothing here re-decodes, retrains or writes a prediction back.

Five things the picture is not allowed to imply:

* **A boundary is not a frame.** Overlapping windows remove the QUANTISATION of
  a transition to a multiple of the window length; they do not remove the BLUR
  of a window that genuinely straddles the transition and whose features are a
  blend of both behaviours. The measured error against a planted transition is
  ``stride/4`` on average and ``stride/2`` at worst (README, "Boundary error
  against a planted transition"), so every boundary is drawn with a band that
  wide rather than as a crisp line. Drawing a hairline would claim a precision
  nothing in the method has. Where ``predict.window_scales`` put several grids
  in the file the band is the COARSEST one's, since the decode averages every
  window covering a frame and the bluntest of them sets how well a transition
  can be placed (:func:`measure_stride`).
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
  since there is no inferred boundary there to be uncertain about. It cannot be
  corrected either: a corrected window is a clip label, and clip labels have a
  file of their own.
* **A correction is not evidence about the model.** It is drawn in a lane of its
  own, below the model's, rather than replacing the bar it overrules - so the
  two claims stay side by side on the same frame axis and a reviewer can never
  read their own judgement back as the model agreeing with them. The model lane
  keeps the numbers the run measured; the record lane carries no confidence at
  all, because "what the record now says these frames are" is not a posterior.

Which of the two a file is comes off the FILE, never off ``predict.decode``:
the config can be edited between a run and this window opening, and a widget
that consulted it would caption the run somebody is about to do rather than the
one whose rows are on screen. ``bout_id`` is the probe, exactly as in
:func:`trackanno.render.prediction_unit` and :func:`trackanno.predict.summarise`.

"Which clips does a corrected 300-frame bout produce, and at what length" was
the question that kept editing out of the first version of this file. It has an
answer now and it is NOT taken here: :func:`trackanno.corrections.derive_window_labels`
answers it, on demand, as a separate step, and this window only ever shows what
that step WOULD produce. Saving a correction writes one file and one file only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from qtpy.QtCore import QEvent, QPointF, QRectF, QSize, Qt, Signal
from qtpy.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPolygonF
from qtpy.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QScrollArea, QSizePolicy, QSpinBox, QToolTip, QVBoxLayout,
    QWidget,
)

from ..config import Config
from ..corrections import (
    CorrectionStore, Partition, Span, apply_corrections, build_corrections,
    corrections_path, derivation_report, derive_window_labels, manifest_windows,
)
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

    # group -> the RECORD: the model's bouts with this reviewer's corrections
    # laid over them. Kept apart from `lanes` rather than replacing it, because
    # the two answer different questions - what the model said, and what is now
    # asserted - and a reviewer who cannot see both at once will eventually read
    # one as the other.
    partitions: dict = field(default_factory=dict)   # group -> Partition
    # Corrections that no longer land on any predicted frame, because the run
    # that wrote these rows covered a different span. Reported, never dropped
    # and never deleted: a machine running again must not erase a human's
    # judgement, and hiding it would show the model's answer where theirs was.
    orphans: list = field(default_factory=list)
    # Corrections only PART of which lands on a predicted frame. Drawn short and
    # saved long, so the same rule holds for them: what a shorter re-run costs is
    # a picture. Listed separately because the reviewer is looking at a span with
    # the wrong ends, and nothing else on screen would say so.
    partly_shown: list = field(default_factory=list)
    # Why this track cannot be corrected, or "". Set rather than raised: this is
    # a normal state (a windows file, a silent track) and a dialog that threw
    # would take the review of everything else with it.
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

    ``stride`` is 0 when it could not be MEASURED, and then this reports the
    limit as a ratio and draws no band. The alternative - reading
    ``predict.window_stride`` out of the config - is refused for the same reason
    ``predict.decode`` is: the config describes the next run. Here it would be
    worse than a wrong caption, because the number sets the width of a band a
    reviewer reads as "the boundary is somewhere in here".

    ``scales`` is ``(window length, that length's stride)`` for every grid found
    in the dense table, narrowest first. Under ``predict.window_scales`` there
    are several, and ``stride`` is then the COARSEST of them - see
    :func:`measure_stride` for why the band must be the widest grid's and not the
    narrowest or the base one's.
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

    ``dense_clip_features.parquet`` IS the plan the decode ran over - one row per
    (track, window) - so the gap between one window start and the next is the
    stride that produced the bouts, as a fact rather than as a setting. Modal
    rather than minimal: a track whose windows do not divide evenly leaves one
    short step at the end, and the minimum would report that.

    PER WINDOW LENGTH, and that is the whole difference from taking one mode over
    the file. Under ``predict.window_scales`` the table holds several grids at
    once - lengths 25, 50 and 100 at strides 25, 50 and 100 - and their starts
    INTERLEAVE (0, 0, 0, 25, 50, 50, 75, 100...). The modal gap over the union is
    then the finest grid's, and the band would be drawn at half its true width on
    exactly the files most likely to need it. Understating a band is worse than
    drawing none: it is a claim, in pixels, that a boundary is somewhere it may
    well not be. The window's own length is in the file (``f_end - f_start``), so
    the grids can be separated and each one measured on its own rows.

    THE BAND IS THE COARSEST GRID'S. The decode averages every window covering a
    frame, whatever its length, so the transition it reports is no better placed
    than the bluntest evidence that went into it - a 100-frame window is
    quantised to 100 frames however many 25-frame ones agree with it. The base
    grid (the length the model was fitted at) is recoverable too, from the
    ``__w####`` suffix ``densepredict._dense_clip_id`` puts on every non-base
    row, and is deliberately NOT used: it is the length the FEATURES mean
    something at, which is a statement about training and not about where a
    boundary is.

    A length that appears once per track is not a grid and is skipped: it has no
    gap to measure, and it is what a window clipped against the end of a movie,
    or a movie shorter than one window, looks like. Only a length with a
    measurable gap is reported as a scale - inventing a stride for the rest (from
    the length, say) would put a band on the screen that no row on disk supports.

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
        self.corrections: CorrectionStore | None = None

        from ..predict import load_predictions, staleness
        from ..render import prediction_unit

        # Before the predictions, and fatal when it fails. A corrections file
        # that will not parse must stop the window rather than be skipped: the
        # timeline would then draw the model's bouts over frames a person has
        # already overruled, and it would look exactly like a track nobody has
        # corrected yet.
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
        self._record(out)
        return out

    def _record(self, track: TrackPredictions) -> None:
        """The model's bouts with this project's corrections laid over them.

        Every way this can fail is a sentence on ``edit_problem`` rather than an
        exception. The three that happen: the file holds windows off a grid and
        not bouts, so there is no bout to correct and a corrected window would be
        a clip label; the track has no rows at all; or the rows for one group
        overlap or leave a hole, which ``runs_to_bouts`` cannot produce but a
        hand-built or half-written file can - and a partition is refused rather
        than patched, because patching would invent the frames the file is
        missing.
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
    """One track's lanes on a shared frame axis, one lane per predicted group.

    QPainter rather than a plotting library, matching
    :mod:`trackanno.gui.distribution`: no new dependency, it follows the theme
    without a second palette, and it stays sharp at any DPI.
    """

    LANE = 52
    BAR = 34
    RECORD_LANE = 38
    RECORD_BAR = 22
    GAP = 6
    TOP = 10
    AXIS = 26
    RIGHT = 12
    # How near a boundary a press has to land to grab it rather than select the
    # span. In PIXELS, not frames: the reviewer is aiming with a mouse, and at
    # 300 frames across 900 px a frame-based tolerance would be unhittable on a
    # long track and would swallow whole bouts on a short one.
    GRAB_PX = 5.0

    # (group, f_start, f_end) of the record span the reviewer clicked, so the
    # panel's frame boxes can be filled from the picture rather than typed.
    spanClicked = Signal(str, int, int)
    # (group, boundary was at, boundary dragged to). The widget does not apply
    # it: the partition is the thing that enforces the invariant, and a widget
    # that edited its own copy would be a second place the rules live.
    boundaryMoved = Signal(str, int, int)

    def __init__(self, cfg: Config | None = None, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.track = TrackPredictions("", 0, False, "")
        self.resolution = Resolution()
        self.note = ""
        self.selected: tuple[str, int, int] | None = None
        self._hits: list[tuple[QRectF, str, object, str]] = []
        # Set on every paint so the mouse handlers measure with the same ruler
        # the picture was drawn with; a second computation could disagree by a
        # pixel and put a drag on the wrong side of a boundary.
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
        self._drag = None
        self.updateGeometry()
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

        One list, used by the painter and by the hit tests, so a lane cannot be
        drawn at one height and clicked at another.
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
        """The frame under a pixel, clamped to the track. Rounded, not floored:
        a boundary drawn at frame 100 is dragged to 100 from either side of the
        line, which floor() would make an off-by-one that depends on approach."""
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

    def _paint_record(self, painter, t, group, partition, y, gutter, scale) -> None:
        """The record lane: what is now asserted about every frame of this track.

        Deliberately without a bar height. The model lane draws confidence as a
        height because ``conf`` is a measurement over exactly those frames; the
        record has no such number, and a correction cut out of a bout does not
        inherit the bout's mean - that mean was taken over frames the correction
        no longer covers. Drawing a height here would put a posterior on a
        sentence a person said.
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
        # does not. The band says "the transition is somewhere in here", which is
        # true of an inferred boundary and false of a chosen one - and drawing it
        # on both would tell a reviewer their own frame number is uncertain to
        # half a window stride.
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
                # Faded and dashed, so a span still the model's cannot be
                # mistaken for one somebody stood behind.
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
        for rect, group, item, kind in self._hits:
            if rect.contains(QPointF(point)):
                return group, item, kind
        return None

    # -- editing with the mouse ---------------------------------------------- #
    def mousePressEvent(self, event):  # noqa: N802 (Qt naming)
        """Grab a boundary if the press is near one, otherwise select a span.

        Only in the record lane. The model lane is a picture of what the run
        produced and dragging it would suggest the run could be edited, which is
        the one thing a correction file exists NOT to do.
        """
        hit = self._at(event.pos())
        if event.button() != Qt.LeftButton or hit is None or hit[2] != "record":
            super().mousePressEvent(event)
            return
        group, span, _ = hit
        partition = self.track.partitions.get(group)
        if partition is None:
            return
        gutter, scale = self._geom
        near = [c for c in partition.boundaries()
                if abs(self._x(c, gutter, scale) - event.pos().x()) <= self.GRAB_PX]
        if near:
            cut = min(near, key=lambda c: abs(self._x(c, gutter, scale)
                                              - event.pos().x()))
            self._drag = (group, cut, cut)
            self.update()
            return
        self.selected = (group, span.f_start, span.f_end)
        self.spanClicked.emit(group, span.f_start, span.f_end)
        self.update()

    def mouseMoveEvent(self, event):  # noqa: N802 (Qt naming)
        if self._drag is None:
            super().mouseMoveEvent(event)
            return
        group, cut, _ = self._drag
        self._drag = (group, cut, self._clamp_boundary(group, cut,
                                                       self._frame_at(event.pos().x())))
        self.update()

    def mouseReleaseEvent(self, event):  # noqa: N802 (Qt naming)
        if self._drag is None:
            super().mouseReleaseEvent(event)
            return
        group, cut, moved = self._drag
        self._drag = None
        self.update()
        if moved != cut:
            self.boundaryMoved.emit(group, cut, moved)

    def _clamp_boundary(self, group: str, cut: int, frame: int) -> int:
        """Where a dragged boundary is allowed to end up.

        Inside the two spans it separates and no further. A drag that ran past a
        neighbour's far edge would DELETE that span - a real thing to want, and
        not something a slipped mouse should do silently. Merging is a button,
        where it says what it is.
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

        It says who is claiming it and never how confident anyone is. A
        corrected span has no posterior, and a model span cut down by a
        correction no longer has the one its bout was measured with - so quoting
        either number here would be quoting a measurement over different frames.
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
    """Movie, track, timeline, captions, and the controls that correct a bout.

    Every failure the files can present is a sentence in this panel rather than
    an exception: no predictions at all, a file of the wrong kind, a track with
    nothing in it, an edit that would leave a hole. The one thing it must never
    do is draw a picture that looks fine over a file it could not read.

    Every editing gesture goes through :meth:`assert_class`, which is
    :meth:`trackanno.corrections.Partition.assert_class` plus a save. The buttons
    are four names for it - set a class, merge left, merge right, drag a boundary
    - because on a gapless run-length encoding those are four gestures over one
    operation, and giving each its own code path would give each its own way to
    break the invariant.

    A save happens on every accepted edit, not on a Save button. The file is the
    state, exactly as ``annotations.csv`` is: there is then no window in which a
    reviewer's work exists only in a widget, and no way to close a dialog and
    lose an hour. Undo is a stack of whole partitions, so an undo is a save too.
    """

    def __init__(self, cfg: Config | None = None, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.reader: PredictionReader | None = None
        # Raised while the combos are being refilled. Qt emits
        # currentIndexChanged from addItems, so without it every load repaints
        # once per movie added and the last one wins by accident.
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

        self.caption = QLabel("")
        self.caption.setWordWrap(True)
        self.caption.setTextFormat(Qt.RichText)
        self.caption.setProperty("role", "muted")

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.headline)
        layout.addWidget(scroll, 1)
        layout.addLayout(self._build_edit_row())
        layout.addWidget(self.status)
        layout.addWidget(self.caption)

        if cfg is not None:
            self.reload()

    def _build_edit_row(self) -> QHBoxLayout:
        """The correction controls, which are always present and never gated on a
        config key: whether a project can be corrected is a property of the file
        on disk, and a setting that could turn this off would only ever be found
        by the person who did not need it."""
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
        self._track = track
        self._undo.clear()
        self.timeline.set_track(track, self.reader.resolution)
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
        self._refresh_edit_gate()

    # -- correcting ----------------------------------------------------------- #
    def current_group(self) -> str:
        return str(self.group.currentText())

    def _fill_groups(self) -> None:
        """The groups this track can be corrected in, and their classes.

        Rebuilt per track rather than per project: which groups have a partition
        depends on the rows this track actually has, and a combo offering a group
        with nothing in it would make the refusal below unreachable.
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

        A class the config does not name cannot be coloured, cannot be trained on
        and cannot be exported, so asserting one would file a judgement nothing
        else in the project can read. A MULTI group's class is a combination of
        options rather than one of them, and a control that offered the options
        one at a time would let a reviewer assert something narrower than they
        meant without saying so - so it refuses instead.
        """
        self.klass.clear()
        spec = self.cfg.group(self.current_group()) if self.cfg is not None else None
        if spec is None or spec.is_multi:
            return
        self.klass.addItems([o.name for o in spec.options])

    def _group_problem(self, group: str | None = None) -> str:
        """Why one group cannot be corrected, or "".

        Takes the group rather than always reading the combo, because most of
        these refusals are about one group and an edit arriving from the timeline
        names its own: checking the combo instead would refuse an edit to a
        perfectly correctable lane because a different lane is a multi group.
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
        # The group combo stays live whatever the problem is: most of the
        # refusals here are about ONE group, and disabling the control that
        # changes group would trap the reviewer in the one that cannot be
        # corrected.
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

        Which class gets asserted is decided by the DIRECTION: dragging right
        extends the span on the left over the frames swept, dragging left extends
        the one on the right. The frames a drag sweeps are exactly the frames
        whose class the reviewer just changed, and nothing else on the track
        moves - which is why a drag needs no separate operation.
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

        The refusal comes back as a sentence rather than an exception because it
        is a normal outcome of a mouse in a hurry - a drag that ran backwards, a
        range off the end of the track - and a traceback out of a Qt slot takes
        the window down with the reviewer's other edits still unsaved.

        The FILE IS WRITTEN FIRST and the partition on screen is advanced only
        if the write succeeded. The other order leaves a reviewer editing a state
        the disk does not hold, and they find out when they reopen the window.
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
            # Refused here and not on the way in, because the file must still
            # LOAD a class the config has since dropped - that row records what a
            # person judged, and a rename must not delete it. What it must not do
            # is let a new judgement be filed under a name nothing else in the
            # project can colour, train on or export.
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
        correction overruled. Read off the drawn segments rather than re-read
        from the file, so it is the same rows the reviewer was looking at."""
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
        # An undo is not itself undoable: pushing the state it left would make
        # the button ping-pong between two versions for ever instead of walking
        # back through the edits.
        message = self._commit(group, current, previous, "undone", record_undo=False)
        self.status.setText(message)

    def _on_revert(self) -> None:
        """Drop every correction on this track and group.

        Asked for explicitly and confirmed, because it is the one action here
        that destroys a human judgement rather than recording one.
        """
        track, group = self._track, self.current_group()
        if track is None or self.reader is None or self.reader.corrections is None:
            return
        saved = self.reader.corrections.for_track(track.movie_id, track.track_id, group)
        if not saved:
            self.status.setText("this track has no corrections in that group")
            return
        key = (track.movie_id, track.track_id, group)
        # The rows that are NOT on screen are named before they are destroyed.
        # Everything else here refuses to delete a judgement a machine hid; this
        # one deletes it on purpose, so the reviewer has to be told it is in the
        # count they are confirming.
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

        Kept as a method returning a string rather than as a dialog full of
        labels so that the numbers in it can be checked by a test, which is the
        only way a claim about what a correction implies stays true.
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

    A window rather than a fifth thing on the Model page: a timeline wants the
    width of the screen, and the page it is reached from is already carrying a
    task log and a cross-validation report.

    It has a Close and no Save. Every accepted edit is already on disk by the
    time the button could be pressed, so a Save would either be a no-op that
    implies the others were not saved, or a second write path - and a dialog with
    a Save button is a dialog somebody can close and lose an hour in.
    """

    def __init__(self, cfg: Config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Review and correct predictions")
        self.resize(1040, 560)
        self.panel = BoutReviewPanel(cfg, self)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(self.panel, 1)
        layout.addWidget(buttons)

    def reload(self) -> None:
        self.panel.reload()
