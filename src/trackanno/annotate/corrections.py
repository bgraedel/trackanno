"""What a person says a bout really was - in a file of its own, not in the clip log.

**A corrected bout is not a clip label.** ``annotations.csv`` holds one row per
FIXED-LENGTH window, keyed by ``clip_id``, and every consumer of it - the
feature join, ``suggest.train``, the audit holdout, ``reassociate`` - assumes
that shape. A human saying "frames 812 to 1043 of track 3 are swimming" has
placed two BOUNDARIES, and no clip grid can express a boundary that does not
fall on a grid multiple. So a correction is its own record, in its own file,
with its own schema, and turning corrections into window labels is a separate
and explicit step (:func:`derive_window_labels`) that this module never takes on
anyone's behalf.

Three things this file is NOT:

* **Not a replacement for ``predictions.parquet``.** A correction is a record
  ABOUT a run, so "what did the model say, and what did a person say instead"
  stays answerable. Nothing here re-decodes or retrains.
* **Not an edit log.** A row is an ASSERTION - these frames are this class - not
  an operation like "moved boundary 3 left by 40"; a log of operations needs a
  replayer, which is a second implementation of the edit rules.
* **Not the whole track.** Only the spans a person actually touched are written,
  so model output and human judgement stay distinguishable.

Where it lives: beside ``annotations.csv``, not under ``work/``, which is
derived and is deleted and rebuilt whenever detection is re-run.

CSV, like the annotation log: a small, human-authored file a person may want to
read, diff or grep without pandas. Schema evolution follows the rule
:mod:`trackanno.annotate.store` sets out - a column added later goes at the END, in the
order it was added - and it is IMPLEMENTED here: a column this build does not
know is carried on the row it was read from (:attr:`Correction.extra`) and
written back where the file had it, because a save rewrites rows the reviewer
was not editing.
"""

from __future__ import annotations

import csv
import os
import tempfile
import threading
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .store import decode_cell, encode_cell

__all__ = [
    "CORRECTIONS_FILE", "CORRECTION_COLUMNS", "corrections_path", "correction_id",
    "encode_overruled", "decode_overruled",
    "Span", "Partition", "Correction", "CorrectionStore",
    "apply_corrections", "build_corrections",
    "DerivedLabel", "Straddler", "Derivation", "derive_window_labels",
    "derivation_report", "manifest_windows",
]

CORRECTIONS_FILE = "bout_corrections.csv"

# The column order the file is written in. `correction_id` leads because it is
# the identity a derived label points back at, exactly as `clip_id` leads the
# annotation log. ANY COLUMN ADDED LATER GOES AT THE END, in the order it was
# added: an older build reconstructs its header as `known` plus the columns it
# does not recognise in file order (see `AnnotationStore._load`), which matches
# the header on disk only while the new columns are already last - and only then
# does it leave the file alone rather than rewriting every row.
CORRECTION_COLUMNS = [
    "correction_id",
    "movie_id", "track_id", "group", "f_start", "f_end", "label",
    "annotator", "timestamp_utc",
    "overruled",
]


def corrections_path(cfg) -> Path:
    """Beside the annotation log, deliberately, and not in ``work/``.

    Derived from ``paths.csv`` rather than from a config key of its own, so a
    project that moved its annotation log takes its corrections with it and an
    old project needs no config migration to find them.
    """
    return Path(cfg.paths.csv).with_name(CORRECTIONS_FILE)


def correction_id(movie_id, track_id: int, group: str, f_start: int, f_end: int) -> str:
    """The identity of one asserted span.

    Both ends are in it, like :func:`trackanno.infer.predict._bout_id` and unlike
    ``clip_id``, so it cannot collide with a clip id by construction; the group
    is in it because two groups may assert the same frames and mean different
    things. Never parsed back apart - a group name with an underscore in it
    would make that ambiguous, and nothing needs to.
    """
    return (f"{movie_id}__t{int(track_id):04d}__{group}"
            f"__f{int(f_start):06d}-{int(f_end):06d}")


# --------------------------------------------------------------------------- #
# what a correction overruled
# --------------------------------------------------------------------------- #
def encode_overruled(bouts: Iterable[Mapping] | None) -> str:
    """Serialise what the model said over these frames, at the moment of correction.

    Without this, a correction that CONFIRMS the model and one that OVERRULES it
    are the same row, and only the second is evidence the model is wrong. The
    model's own answer cannot be recovered later: the next run overwrites
    ``predictions.parquet``, and a decode is not reproducible once the training
    set has moved on.

    :func:`trackanno.annotate.store.encode_cell` does the writing, the same codec the
    annotation log's ``selected_by`` cell uses, so a new key does not force a
    schema migration and a non-finite confidence is written as ``null`` here for
    the same reason it is there: an unscored bout genuinely has no confidence,
    and ``null`` stays distinguishable from a confidence of zero.

    What this column requires of its own payload is the two ends: a quoted bout
    that does not say which frames it covers records nothing about what was
    overruled.
    """
    if not bouts:
        return ""
    out = []
    for bout in bouts:
        try:
            f_start, f_end = int(bout["f_start"]), int(bout["f_end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "a bout being overruled must name both ends as frame numbers, "
                f"got {bout!r}"
            ) from exc
        # Coerced here rather than in the codec: `conf` is one number, and a
        # value that is not one at all is no confidence rather than a refusal.
        try:
            conf = float(bout.get("conf"))
        except (TypeError, ValueError):
            conf = None
        out.append({
            "f_start": f_start,
            "f_end": f_end,
            "label": str(bout.get("label", "")),
            "conf": conf,
        })
    return encode_cell({"bouts": out}, "a record of what the model said")


def decode_overruled(value) -> dict:
    """Read an ``overruled`` cell back. ``{}`` means *the row does not say*.

    Raises on a cell that will not parse rather than returning ``{}``: an
    unreadable record of what was overruled would read as "this correction
    agreed with nothing in particular", which is a claim the row is not making.
    That policy is :func:`trackanno.annotate.store.decode_cell`'s and is shared rather
    than restated, so the two columns cannot come to disagree about it.
    """
    return decode_cell(
        value, "overruled",
        "it is an object like {'bouts': [{'f_start': 0, 'f_end': 100, ...}]}",
    )


# --------------------------------------------------------------------------- #
# the editable state
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Span:
    """One stretch of one track under one group, and who says so.

    ``corrected`` is the field the whole design turns on: a span a person
    asserted and a span the model produced are drawn, persisted and derived from
    differently, and once the two are interchangeable a reviewer reads their own
    corrections back as evidence about the model.

    ``label`` is ``""`` only ever on a MODEL span - a stretch no window could be
    featurised over. A person cannot assert that nobody scored something, so
    :meth:`Partition.assert_class` refuses an empty label.
    """

    f_start: int
    f_end: int
    label: str
    corrected: bool = False

    @property
    def n_frames(self) -> int:
        return self.f_end - self.f_start


class Partition:
    """One track's run-length encoding for one group: disjoint, gapless, ordered.

    The invariant is enforced HERE and not in the widget, so an interaction bug
    fails at the edit that caused it rather than in a file a week later: a hole
    in a run-length encoding is indistinguishable from a stretch nobody thought
    about.

    Edits return a NEW partition rather than mutating this one, so a refused edit
    cannot leave a half-applied state behind and undo is a stack of objects
    rather than a stack of inverse operations.

    THE ONE OPERATION. "Move a boundary", "change a bout's class", "split a bout"
    and "merge two bouts" are four gestures, not four operations: on a disjoint,
    gapless run-length encoding over a fixed frame span, the state IS (where the
    cuts are, what each cell is called), and every one of those gestures is
    :meth:`assert_class` - name a class over a half-open frame range:

    * move the boundary between A and B right   -> assert A over [old, new)
    * move it left                              -> assert B over [new, old)
    * change a bout's class                     -> assert the new class over it
    * split a bout                              -> assert a class over part of it
    * merge a bout into its neighbour           -> assert the neighbour's class
      over it, after which the two cells have the same name and no cut between
      them survives canonicalisation
    """

    def __init__(self, spans: Iterable[Span], f_lo: int, f_hi: int):
        self.f_lo = int(f_lo)
        self.f_hi = int(f_hi)
        self.spans: list[Span] = [
            Span(int(s.f_start), int(s.f_end), str(s.label), bool(s.corrected))
            for s in spans
        ]
        self._check()

    # -- invariant ------------------------------------------------------------ #
    def _check(self) -> None:
        if self.f_hi < self.f_lo:
            raise ValueError(
                f"a track cannot span frames {self.f_lo}-{self.f_hi}: it ends "
                "before it starts"
            )
        if not self.spans:
            if self.f_hi != self.f_lo:
                raise ValueError(
                    f"frames {self.f_lo}-{self.f_hi} are covered by no span at all; "
                    "an uncovered stretch is a hole, and a hole reads exactly like "
                    "a stretch nobody thought about"
                )
            return
        cursor = self.f_lo
        for span in self.spans:
            if span.f_end <= span.f_start:
                raise ValueError(
                    f"the span at frame {span.f_start} covers no frames "
                    f"({span.f_start}-{span.f_end}); an empty span is a row that "
                    "claims nothing"
                )
            if span.f_start < cursor:
                raise ValueError(
                    f"frames {span.f_start}-{cursor} are claimed twice, last by "
                    f"{span.label!r}; two classes over one frame is not a decode, "
                    "it is a contradiction"
                )
            if span.f_start > cursor:
                raise ValueError(
                    f"frames {cursor}-{span.f_start} are covered by no span; a hole "
                    "is indistinguishable from a stretch nobody thought about, so "
                    "an unscored run is a span with an empty label instead"
                )
            cursor = span.f_end
        if cursor != self.f_hi:
            raise ValueError(
                f"the spans stop at frame {cursor} but the track runs to {self.f_hi}; "
                "the tail is a hole"
            )

    # -- reading -------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.spans)

    def at(self, frame: int) -> Span | None:
        """The span covering ``frame``, or None outside the track."""
        for span in self.spans:
            if span.f_start <= frame < span.f_end:
                return span
        return None

    def index_at(self, frame: int) -> int:
        """Position of the span covering ``frame``, or -1."""
        for i, span in enumerate(self.spans):
            if span.f_start <= frame < span.f_end:
                return i
        return -1

    def boundaries(self) -> list[int]:
        """The interior cuts. The track's own two ends are not boundaries - they
        are where the evidence stops, not where a behaviour changed - so a
        reviewer cannot drag one."""
        return [s.f_start for s in self.spans[1:]]

    def corrected_spans(self) -> list[Span]:
        return [s for s in self.spans if s.corrected]

    @property
    def n_corrected_frames(self) -> int:
        return sum(s.n_frames for s in self.spans if s.corrected)

    # -- the one edit --------------------------------------------------------- #
    def assert_class(self, f_start: int, f_end: int, label: str) -> Partition:
        """A person says frames ``[f_start, f_end)`` are ``label``. See the class
        docstring for why this is the only editing operation there is.

        It refuses an empty label ("nobody could score these frames" is the
        featuriser's statement, not a person's), a backwards or empty range, and
        a range outside the track's decoded span, whose frames have no model row
        to correct.

        The new span is marked ``corrected``; whatever it partly covers keeps its
        own marking on the frames left over. Adjacent spans are joined only when
        they agree on BOTH label and origin, or a human's assertion would extend
        silently over frames nobody looked at.
        """
        label = str(label).strip()
        if not label:
            raise ValueError(
                "a correction needs a class name. An empty label means 'no window "
                "over these frames could be featurised', which is a fact about the "
                "run and not something a person can assert"
            )
        f_start, f_end = int(f_start), int(f_end)
        if f_end <= f_start:
            raise ValueError(
                f"frames {f_start}-{f_end} are not a span: a correction covers "
                "[start, end) and must contain at least one frame"
            )
        if f_start < self.f_lo or f_end > self.f_hi:
            raise ValueError(
                f"frames {f_start}-{f_end} fall outside this track's predicted span "
                f"({self.f_lo}-{self.f_hi}); there is no row there to correct"
            )

        out: list[Span] = []
        for span in self.spans:
            if span.f_start < f_start:
                out.append(Span(span.f_start, min(span.f_end, f_start), span.label,
                                span.corrected))
            if span.f_end > f_end:
                out.append(Span(max(span.f_start, f_end), span.f_end, span.label,
                                span.corrected))
        out.append(Span(f_start, f_end, label, True))
        out.sort(key=lambda s: s.f_start)
        return Partition(_canonical(out), self.f_lo, self.f_hi)


def _canonical(spans: list[Span]) -> list[Span]:
    """Join neighbours that agree on label AND origin.

    Both, not just the label: a corrected span joined to a model span of the same
    class would put a person's name on frames they never looked at. Two adjacent
    spans with the same class and different origins are a legitimate state and
    are drawn as two.
    """
    out: list[Span] = []
    for span in spans:
        if out and out[-1].f_end == span.f_start and out[-1].label == span.label \
                and out[-1].corrected == span.corrected:
            out[-1] = Span(out[-1].f_start, span.f_end, span.label, span.corrected)
        else:
            out.append(span)
    return out


# --------------------------------------------------------------------------- #
# one persisted assertion
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Correction:
    """One row of the file: a person's assertion about a half-open frame range.

    ``extra`` is every cell of that row this build has no column for, kept as
    text under the name the file gave it. It is the downgrade path: a save
    rewrites the whole file, so a build that dropped what it did not recognise
    would delete a newer build's column from rows nobody was editing.
    """

    correction_id: str
    movie_id: str
    track_id: int
    group: str
    f_start: int
    f_end: int
    label: str
    annotator: str = ""
    timestamp_utc: str = ""
    overruled: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return self.f_end - self.f_start

    @property
    def key(self) -> tuple[str, int, str]:
        return (str(self.movie_id), int(self.track_id), str(self.group))

    def to_row(self) -> dict:
        row = {
            "correction_id": self.correction_id,
            "movie_id": str(self.movie_id),
            "track_id": int(self.track_id),
            "group": str(self.group),
            "f_start": int(self.f_start),
            "f_end": int(self.f_end),
            "label": str(self.label),
            "annotator": str(self.annotator),
            "timestamp_utc": str(self.timestamp_utc),
            "overruled": encode_overruled(self.overruled.get("bouts")
                                          if self.overruled else None),
        }
        # After the known columns, never over them: a file whose header repeated
        # a name would otherwise let an unparsed cell overwrite the value this
        # build validated on the way in.
        for column, value in self.extra.items():
            row.setdefault(str(column), value)
        return row

    @classmethod
    def from_row(cls, row: Mapping, where: str) -> Correction:
        """One row off disk, or a refusal naming the row.

        Every field is validated, because a correction that cannot be read
        cannot be skipped either: the frames it covers would fall back to model
        output, and the reviewer would see the model's answer where their own
        should be.
        """
        def integer(column: str) -> int:
            value = row.get(column, "")
            try:
                return int(str(value).strip())
            except (TypeError, ValueError):
                raise ValueError(
                    f"{where}: {column}={value!r} is not a frame number"
                ) from None

        f_start, f_end = integer("f_start"), integer("f_end")
        if f_end <= f_start:
            raise ValueError(
                f"{where}: frames {f_start}-{f_end} cover nothing; a correction "
                "claims [start, end) and must contain at least one frame"
            )
        label = str(row.get("label", "")).strip()
        if not label:
            raise ValueError(
                f"{where}: the correction over frames {f_start}-{f_end} has no class "
                "name. An empty label is how the prediction table says 'nothing "
                "could be scored here', which is not something a person asserts"
            )
        movie_id = str(row.get("movie_id", "")).strip()
        group = str(row.get("group", "")).strip()
        if not movie_id or not group:
            raise ValueError(
                f"{where}: a correction must name both a movie and a category group "
                f"(got movie_id={movie_id!r}, group={group!r})"
            )
        track_id = integer("track_id")
        # Kept as TEXT and never parsed: a value this build cannot interpret is
        # handed back exactly as it came. A ``None`` key is csv.DictReader's
        # bucket for cells past the end of the header, so it is a broken row
        # rather than a column and is dropped.
        extra = {
            str(column): "" if value is None else str(value)
            for column, value in row.items()
            if column is not None and column not in CORRECTION_COLUMNS
        }
        return cls(
            correction_id=str(row.get("correction_id", "")).strip()
            or correction_id(movie_id, track_id, group, f_start, f_end),
            movie_id=movie_id, track_id=track_id, group=group,
            f_start=f_start, f_end=f_end, label=label,
            # `or ""` because csv.DictReader fills a row shorter than the header
            # with None, and str(None) would record the annotator of a truncated
            # row as a person called "None".
            annotator=str(row.get("annotator", "") or ""),
            timestamp_utc=str(row.get("timestamp_utc", "") or ""),
            overruled=decode_overruled(row.get("overruled")),
            extra=extra,
        )


class CorrectionStore:
    """The corrections file, read whole and rewritten whole.

    No append path and no torn-append repair, unlike
    :class:`trackanno.annotate.store.AnnotationStore`, because nothing here appends: a
    save replaces every row for one (movie, track, group) at once. One atomic
    ``os.replace`` per save gives the same guarantee an append and fsync gives
    there - a kill loses at most the edit in progress - without a second write
    path that can tear.

    Rows are written sorted by (movie, track, group, f_start), so the file is
    stable under re-saving and a diff shows the correction that changed rather
    than the order they were made in.

    ``columns`` is the header this store will write: the ones this build knows,
    then any it read off the file and does not, in the order the file had them.
    Same reconstruction as :class:`trackanno.annotate.store.AnnotationStore`.
    """

    def __init__(self, path: str | Path, annotator: str = ""):
        self.path = Path(path)
        self.annotator = str(annotator or "")
        self.rows: list[Correction] = []
        self.columns: list[str] = list(CORRECTION_COLUMNS)
        self._lock = threading.RLock()
        self._load()

    # -- io ------------------------------------------------------------------- #
    def _load(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            disk_columns = list(reader.fieldnames or [])
            for number, row in enumerate(reader, start=2):
                if not any(str(v).strip() for v in row.values() if v is not None):
                    continue
                self.rows.append(
                    Correction.from_row(row, f"{self.path.name} line {number}")
                )
        # dict.fromkeys and not a set: the ORDER is the whole point, since only a
        # header rebuilt in the file's own order comes out identical to the one
        # on disk and leaves the file unchurned.
        extra = list(dict.fromkeys(
            c for c in disk_columns if c and c not in self.columns
        ))
        if extra:
            warnings.warn(
                f"{self.path.name} has column(s) this build does not know: "
                f"{extra}. They are kept on every row and written back where "
                "they are, so a newer build's work is not lost by saving here.",
                stacklevel=2,
            )
            self.columns = self.columns + extra
        self._refuse_overlaps()

    def _refuse_overlaps(self) -> None:
        """Two assertions over one frame is a record nobody can read.

        The GUI cannot produce this - corrections come out of a partition that
        is disjoint by construction - so it means the file was edited by hand or
        merged from two sources. Taking the last would pick a winner nobody
        chose.
        """
        by_key: dict[tuple[str, int, str], list[Correction]] = {}
        for row in self.rows:
            by_key.setdefault(row.key, []).append(row)
        for key, group in by_key.items():
            group = sorted(group, key=lambda c: (c.f_start, c.f_end))
            for left, right in zip(group, group[1:]):
                if right.f_start < left.f_end:
                    movie_id, track_id, name = key
                    raise ValueError(
                        f"{self.path.name}: two corrections claim frame "
                        f"{right.f_start} of track {track_id} in {movie_id} "
                        f"({name}): {left.f_start}-{left.f_end} says {left.label!r} "
                        f"and {right.f_start}-{right.f_end} says {right.label!r}. "
                        "Nothing can say what was judged while both stand; delete "
                        "one of the two rows."
                    )

    def _header(self, rows: Iterable[Correction]) -> list[str]:
        """The columns to write: this build's, then everything the rows carry.

        Taken from the ROWS as well as from the header that was read, so a
        correction carrying a column this store never saw is not dropped:
        ``extrasaction="ignore"`` would do that silently, so the writer is given
        a header wide enough instead.
        """
        out = list(self.columns)
        for row in rows:
            for column in row.extra:
                if str(column) not in out:
                    out.append(str(column))
        return out

    def _rewrite(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            rows = self.sorted_rows()
            fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".csv.tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
                    # restval, so a row that predates a column the file has is
                    # written with that cell empty rather than refused.
                    writer = csv.DictWriter(fh, fieldnames=self._header(rows),
                                            restval="")
                    writer.writeheader()
                    for row in rows:
                        writer.writerow(row.to_row())
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, self.path)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise

    def sorted_rows(self) -> list[Correction]:
        return sorted(self.rows,
                      key=lambda c: (c.movie_id, c.track_id, c.group, c.f_start))

    # -- queries -------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.rows)

    def for_track(self, movie_id, track_id: int, group: str) -> list[Correction]:
        key = (str(movie_id), int(track_id), str(group))
        return sorted((c for c in self.rows if c.key == key), key=lambda c: c.f_start)

    def for_movie(self, movie_id) -> list[Correction]:
        return [c for c in self.rows if c.movie_id == str(movie_id)]

    def tracks(self) -> set[tuple[str, int]]:
        return {(c.movie_id, c.track_id) for c in self.rows}

    # -- writing -------------------------------------------------------------- #
    def replace_track(self, movie_id, track_id: int, group: str,
                      corrections: Iterable[Correction]) -> None:
        """Make the file say exactly ``corrections`` about this (track, group).

        A whole-key replacement rather than a merge: the reviewer was looking at
        this track's entire partition, so the set of assertions they now hold
        about it is the set that stands. Merging would leave an assertion they
        had just deleted in the file, to reappear on the next reload.

        THE ARGUMENT ONLY HOLDS FOR SPANS THE REVIEWER COULD SEE, and this
        method cannot tell which those were: a correction over frames the
        current predictions have no row for is absent from the partition, so a
        caller passing only what is on screen would delete it.
        :func:`build_corrections` puts those rows back, and is the only
        supported way to build the argument to this method.
        """
        key = (str(movie_id), int(track_id), str(group))
        kept = [c for c in self.rows if c.key != key]
        fresh = sorted(corrections, key=lambda c: c.f_start)
        for correction in fresh:
            if correction.key != key:
                raise ValueError(
                    f"a correction for {correction.key} was handed to a save of "
                    f"{key}; that would file a judgement about one track under "
                    "another"
                )
        with self._lock:
            previous = list(self.rows)
            self.rows = kept + fresh
            try:
                self._refuse_overlaps()
                self._rewrite()
            except BaseException:
                # Memory must never be ahead of the file: the reviewer would go on
                # editing a partition the disk does not hold and lose it silently.
                self.rows = previous
                raise


def _bouts_over(bouts: Iterable[Mapping], f_start: int, f_end: int) -> list[dict]:
    """What the model said over ``[f_start, f_end)``, clipped to it.

    Clipped rather than whole, so a row's ``overruled`` record is a statement
    about exactly the frames the row claims. A bout quoted past the correction's
    own ends would say the person overruled frames they never mentioned.
    """
    return [
        {"f_start": max(int(b["f_start"]), f_start),
         "f_end": min(int(b["f_end"]), f_end),
         "label": str(b.get("label", "")),
         "conf": b.get("conf")}
        for b in bouts
        if int(b["f_end"]) > f_start and int(b["f_start"]) < f_end
    ]


def _coalesce(pieces: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sorted, with touching or overlapping pieces joined into one."""
    out: list[tuple[int, int]] = []
    for f_start, f_end in sorted(pieces):
        if out and f_start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], f_end))
        else:
            out.append((f_start, f_end))
    return out


def _without(
    span: tuple[int, int], taken: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """The parts of ``span`` no piece of ``taken`` already covers."""
    cursor, out = span[0], []
    for f_start, f_end in _coalesce([t for t in taken
                                     if t[1] > span[0] and t[0] < span[1]]):
        if f_start > cursor:
            out.append((cursor, min(f_start, span[1])))
        cursor = max(cursor, f_end)
        if cursor >= span[1]:
            return out
    if cursor < span[1]:
        out.append((cursor, span[1]))
    return out


def _still_stands(
    stored: Correction, asserted: list[tuple[int, int]], f_lo: int, f_hi: int
) -> list[tuple[int, int]]:
    """Which frames of a stored correction the reviewer has left alone.

    Two kinds, joined into one list because they make the same claim - "nothing
    has replaced this":

    * frames OUTSIDE ``[f_lo, f_hi)``, the span the current predictions cover.
      The reviewer was never shown them, which is why a correction reaching past
      the predictions is persisted at its FULL length though drawn short.
    * frames inside it that the partition still asserts under the same class,
      whether as this correction's own span or as part of a longer one it was
      merged into for drawing.

    The two are coalesced, so a correction whose visible part is untouched comes
    back as the single range it was stored at and is written back byte for byte.
    """
    pieces: list[tuple[int, int]] = []
    if stored.f_start < f_lo:
        pieces.append((stored.f_start, min(stored.f_end, f_lo)))
    if stored.f_end > f_hi:
        pieces.append((max(stored.f_start, f_hi), stored.f_end))
    for f_start, f_end in asserted:
        left, right = max(f_start, stored.f_start), min(f_end, stored.f_end)
        if right > left:
            pieces.append((left, right))
    return _coalesce(pieces)


def build_corrections(
    partition: Partition, movie_id, track_id: int, group: str, annotator: str,
    model_bouts: Iterable[Mapping] = (), previous: Iterable[Correction] = (),
) -> list[Correction]:
    """The whole set of rows this (track, group) should now hold.

    THE WHOLE SET, not the spans on screen. :meth:`CorrectionStore.replace_track`
    replaces every row under this key, so anything missing from what comes back
    here is deleted - including the two spans easiest to leave out, a correction
    over frames the current predictions have no row for and the earlier half of
    two same-class corrections the partition merged for drawing. Both survive by
    starting from ``previous``, what the file already holds.

    **A stored row survives, unchanged, for as long as nothing has replaced it.**
    Its frames outside the predicted span cannot have been replaced, since
    nobody was shown them (:func:`_still_stands`), and its frames inside are
    replaced only where the reviewer asserted another class over them. So:

    * an edit somewhere else on the track leaves every other row byte for byte
      as it was, including its timestamp, its annotator and what it overruled;
    * a correction the predictions no longer reach keeps ITS OWN ends. Clipping
      is for drawing only, so a re-run that decoded a shorter span cannot edit a
      human judgement;
    * two same-class corrections asserted on different days stay TWO ROWS even
      though they are drawn as one span, because one row cannot hold two
      timestamps or two annotators.

    A stored row is narrowed only where the reviewer overwrote part of it, and
    then it keeps its annotator, its timestamp and the part of its ``overruled``
    record still covering the frames it claims. Its ``correction_id`` is
    recomputed, because both ends are in that id.

    Whatever the surviving rows do not account for is new: a fresh row, stamped
    now, carrying ``model_bouts`` - what ``predictions.parquet`` says over these
    frames right now, as mappings with ``f_start``, ``f_end``, ``label`` and
    ``conf`` - so a correction that CONFIRMS the model stays distinguishable
    from one that OVERRULES it once the next run has overwritten the evidence.
    """
    asserted: dict[str, list[tuple[int, int]]] = {}
    for span in partition.corrected_spans():
        asserted.setdefault(span.label, []).append((span.f_start, span.f_end))

    out: list[Correction] = []
    claimed: list[tuple[int, int]] = []
    for stored in sorted(previous, key=lambda c: (c.f_start, c.f_end)):
        for f_start, f_end in _still_stands(
            stored, asserted.get(stored.label, []), partition.f_lo, partition.f_hi
        ):
            claimed.append((f_start, f_end))
            if (f_start, f_end) == (stored.f_start, stored.f_end):
                out.append(stored)
                continue
            kept = _bouts_over(stored.overruled.get("bouts", ()), f_start, f_end)
            out.append(Correction(
                correction_id=correction_id(movie_id, track_id, group,
                                            f_start, f_end),
                movie_id=stored.movie_id, track_id=stored.track_id,
                group=stored.group, f_start=f_start, f_end=f_end,
                label=stored.label, annotator=stored.annotator,
                timestamp_utc=stored.timestamp_utc,
                overruled={"bouts": kept} if kept else {},
                extra=dict(stored.extra),
            ))

    bouts = [dict(b) for b in model_bouts]
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for label, spans in asserted.items():
        for span in spans:
            for f_start, f_end in _without(span, claimed):
                covered = _bouts_over(bouts, f_start, f_end)
                out.append(Correction(
                    correction_id=correction_id(movie_id, track_id, group,
                                                f_start, f_end),
                    movie_id=str(movie_id), track_id=int(track_id),
                    group=str(group), f_start=f_start, f_end=f_end, label=label,
                    annotator=str(annotator), timestamp_utc=stamp,
                    overruled={"bouts": covered} if covered else {},
                ))
    return sorted(out, key=lambda c: (c.f_start, c.f_end))


def apply_corrections(
    model: Partition, corrections: Iterable[Correction]
) -> tuple[Partition, list[Correction], list[Correction]]:
    """Lay a track's corrections over the model's bouts, for DRAWING.

    Returns the effective partition, the corrections that could not be laid over
    it at all, and the ones that could only be laid over part of themselves. A
    re-run can shorten the span a track was decoded over, leaving an assertion
    about frames the current file has no row for; those stay in the file, are
    reported, and are not drawn.

    A correction that hangs off the end is CLIPPED to the frames the predictions
    do cover, because those frames really were judged; only a correction with no
    overlap at all is an orphan. **The clipping is undone before anything is
    written** - :func:`build_corrections` persists such a row at its own ends -
    so what a shorter re-run costs is a picture, never a judgement. The clipped
    list is returned so the picture can say it is showing less than the file
    holds.
    """
    out = model
    orphans: list[Correction] = []
    clipped: list[Correction] = []
    for correction in sorted(corrections, key=lambda c: c.f_start):
        f_start = max(int(correction.f_start), model.f_lo)
        f_end = min(int(correction.f_end), model.f_hi)
        if f_end <= f_start:
            orphans.append(correction)
            continue
        if (f_start, f_end) != (int(correction.f_start), int(correction.f_end)):
            clipped.append(correction)
        out = out.assert_class(f_start, f_end, correction.label)
    return out, orphans, clipped


# --------------------------------------------------------------------------- #
# turning corrections into window labels - the separate, explicit step
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DerivedLabel:
    """A fixed window that falls ENTIRELY inside one corrected span."""

    clip_id: str
    movie_id: str
    track_id: int
    f_start: int
    f_end: int
    group: str
    label: str
    correction_id: str


@dataclass(frozen=True)
class Straddler:
    """A fixed window a correction touches but does not contain.

    Not an error and not a derived label: a third outcome, with the reason
    spelled out, because the reason is the thing a reviewer needs in order to
    decide what to do about it.
    """

    clip_id: str
    movie_id: str
    track_id: int
    f_start: int
    f_end: int
    group: str
    reason: str


@dataclass
class Derivation:
    """What a set of corrections implies about the fixed-window training set."""

    derived: list[DerivedLabel] = field(default_factory=list)
    straddling: list[Straddler] = field(default_factory=list)
    n_windows: int = 0            # windows considered
    n_untouched: int = 0          # windows no correction overlaps at all

    @property
    def n_derived(self) -> int:
        return len(self.derived)

    @property
    def n_straddling(self) -> int:
        return len(self.straddling)


def derive_window_labels(
    corrections: Iterable[Correction], windows: Iterable[Mapping]
) -> Derivation:
    """Which fixed windows a set of corrections labels, and which it cannot.

    THE STRADDLER RULE, stated once and enforced in one place: **a window derives
    a class if and only if it lies entirely inside a single corrected span of
    that class. A window that overlaps a corrected span without being contained
    in one derives nothing, and is returned as a straddler with the reason.**

    Containment and not majority, midpoint or nearest, because a window crossing
    a behavioural transition has features that are a BLEND of the two behaviours
    (:mod:`trackanno.infer.bouts`). Labelling it with either class teaches the next
    model that a blend looks like a pure class, exactly at the transitions it is
    already worst at.

    Straddlers are RETURNED rather than dropped, because a straddler is where a
    boundary is, and a caller cannot otherwise tell "your corrections implied
    nothing here" from "something I decided not to mention". Windows no
    correction touches are counted, not listed.

    Nothing is derived from MODEL output, only from corrections; deriving a
    label from a bout the model produced would feed the model its own answer.

    ``windows`` are mappings with ``clip_id``, ``movie_id``, ``track_id``,
    ``f_start`` and ``f_end`` - the annotation manifest's own columns, because
    those are the windows the training set is actually built from and a grid
    invented here would derive labels for clips that do not exist.
    """
    by_track: dict[tuple[str, int, str], list[Correction]] = {}
    for correction in corrections:
        by_track.setdefault(correction.key, []).append(correction)
    for spans in by_track.values():
        spans.sort(key=lambda c: c.f_start)
    groups = sorted({key[2] for key in by_track})

    out = Derivation()
    for window in windows:
        out.n_windows += 1
        movie_id = str(window["movie_id"])
        track_id = int(window["track_id"])
        w0, w1 = int(window["f_start"]), int(window["f_end"])
        clip_id = str(window["clip_id"])
        touched = False
        for group in groups:
            spans = by_track.get((movie_id, track_id, group), ())
            overlapping = [c for c in spans if c.f_end > w0 and c.f_start < w1]
            if not overlapping:
                continue
            touched = True
            inside = [c for c in overlapping if c.f_start <= w0 and w1 <= c.f_end]
            if len(inside) == 1:
                correction = inside[0]
                out.derived.append(DerivedLabel(
                    clip_id=clip_id, movie_id=movie_id, track_id=track_id,
                    f_start=w0, f_end=w1, group=group, label=correction.label,
                    correction_id=correction.correction_id,
                ))
                continue
            out.straddling.append(Straddler(
                clip_id=clip_id, movie_id=movie_id, track_id=track_id,
                f_start=w0, f_end=w1, group=group,
                reason=_straddle_reason(overlapping, w0, w1),
            ))
        if not touched:
            out.n_untouched += 1
    return out


def _straddle_reason(overlapping: list[Correction], w0: int, w1: int) -> str:
    """Why one window could not inherit a class, in frames.

    Named rather than counted, so a reviewer is told where to look.
    """
    if len(overlapping) > 1:
        cuts = [c.f_start for c in overlapping[1:]]
        labels = " then ".join(repr(c.label) for c in overlapping)
        return (
            f"frames {w0}-{w1} cross a corrected boundary at "
            f"{', '.join(str(c) for c in cuts)} - the window is {labels}, and its "
            "features are a blend of both"
        )
    correction = overlapping[0]
    before = max(0, correction.f_start - w0)
    after = max(0, w1 - correction.f_end)
    parts = []
    if before:
        parts.append(f"{before} frame(s) before it")
    if after:
        parts.append(f"{after} frame(s) after it")
    return (
        f"frames {w0}-{w1} reach outside the correction over "
        f"{correction.f_start}-{correction.f_end} ({correction.label!r}) by "
        f"{' and '.join(parts)}, which nothing has judged"
    )


def derivation_report(derivation: Derivation, examples: int = 6) -> str:
    """The derivation as prose, so the numbers can be checked without a screenshot.

    The last two sentences are part of the REPORT and not a comment in the
    source, because its reader is the person who would otherwise assume these
    labels had been written somewhere. Nothing here writes a derived label into
    ``annotations.csv``; anything that ever does must say so in this report.
    """
    lines = [
        f"{derivation.n_windows} annotation window(s) were checked against your "
        "corrections.",
        f"{derivation.n_derived} fall entirely inside a corrected span and would "
        "take its class.",
        f"{derivation.n_straddling} straddle a correction and take nothing: a "
        "window that crosses a boundary has features blended from both "
        "behaviours, and labelling it with either class teaches the next model "
        "that a blend is a pure class - exactly at the transitions it is already "
        "worst at.",
        f"{derivation.n_untouched} are outside every correction, so nothing has "
        "been asserted about them.",
    ]
    for straddler in derivation.straddling[:examples]:
        lines.append(f"  - {straddler.clip_id} ({straddler.group}): {straddler.reason}")
    if derivation.n_straddling > examples:
        lines.append(f"  - ...and {derivation.n_straddling - examples} more")
    lines.append("Nothing has been written to annotations.csv.")
    lines.append(
        "A derived label is not a hand label: it is what your correction IMPLIES "
        "about a window a person never saw. The moment the two are "
        "indistinguishable in that file, nobody can say again what a person "
        "actually judged."
    )
    return "\n".join(lines)


def manifest_windows(cfg, movie_id) -> list[dict]:
    """The annotation grid for one movie, off its clip manifest.

    The manifest and not a grid computed here: ``clip_id`` is the join key the
    annotation CSV, the feature tables and the manifest all share, and a window
    invented from ``clips.frames``/``clips.stride`` would agree with it only
    until somebody edited those settings.
    """
    import pandas as pd

    from .clips import MANIFEST_FILE

    path = cfg.movie_dir(str(movie_id)) / MANIFEST_FILE
    if not path.exists():
        return []
    frame = pd.read_parquet(
        path, columns=["clip_id", "movie_id", "track_id", "f_start", "f_end"]
    )
    return [
        {"clip_id": str(r.clip_id), "movie_id": str(r.movie_id),
         "track_id": int(r.track_id), "f_start": int(r.f_start),
         "f_end": int(r.f_end)}
        for r in frame.itertuples(index=False)
    ]
