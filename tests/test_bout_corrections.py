"""Correcting a bout, and everything that must not happen when one is corrected.

A correction is a human judgement about a variable-length span, and there was no
place to put one until now: the annotation CSV holds fixed windows keyed by
``clip_id``, and a corrected bout written there would be a row of a second kind
in a table whose every consumer assumes the first. So the checks here are about
the boundary between the two files rather than about whether a widget accepts a
click:

* the edited state stays a DISJOINT, GAPLESS run-length encoding after every
  edit, and an edit that would break that is refused rather than absorbed - the
  invariant is enforced in the data model, so a slipped mouse fails at the edit
  and not in a file a week later;
* what a person asserted and what the model produced stay separable across a
  reload, in the picture and in the file, for ever;
* a fixed window derives a class only if it lies ENTIRELY inside a corrected
  span, and a straddler is reported rather than guessed at or dropped;
* ``annotations.csv`` and ``predictions.parquet`` are byte-identical before and
  after a correction. A correction is a record about a run, not a rewritten run,
  and a derived label is not a hand label.

No ``ClipStore`` is opened: the correction path reads parquet and CSV, and a
suite that mapped the shared fixture's clip cache would leave it mapped for
whichever suite re-runs detection next.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest
from harness import check
from test_bout_review import bout_row

GROUP = "behaviour"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tiles(partition, f_lo: int, f_hi: int) -> bool:
    """Whether the spans cover ``[f_lo, f_hi)`` exactly once each and in order.

    Recomputed here rather than trusted from ``Partition._check``: the invariant
    is the claim under test, and a test that asked the code under test whether it
    held would pass for as long as the bug was in the checker.
    """
    spans = partition.spans
    if not spans:
        return f_lo == f_hi
    if spans[0].f_start != f_lo or spans[-1].f_end != f_hi:
        return False
    return all(a.f_end == b.f_start for a, b in zip(spans, spans[1:])) and \
        all(s.f_end > s.f_start for s in spans)


def model_partition():
    """Three bouts over 120 frames: A, B, A. Boundaries at 40 and 80."""
    from trackanno.annotate.corrections import Partition, Span

    return Partition([
        Span(0, 40, "swimming", False),
        Span(40, 80, "immotile", False),
        Span(80, 120, "swimming", False),
    ], 0, 120)


def select(panel, movie_id, track_id) -> None:
    panel.movie.setCurrentText(str(movie_id))
    for i in range(panel.track.count()):
        if panel.track.itemData(i) == int(track_id):
            panel.track.setCurrentIndex(i)
            return
    raise AssertionError(f"track {track_id} is not offered for {movie_id}")


BOUTS = [
    (0, 40, "swimming", 0.90),
    (40, 80, "immotile", 0.45),
    (80, 100, "", float("nan")),      # nothing could score these frames
    (100, 120, "swimming", 0.80),
]


@pytest.fixture(scope="module")
def project(root):
    """A real project - manifest, clip grid, two hand labels - carrying a bout
    table over one of its own tracks, so the derivation runs against the windows
    the training set is actually built from rather than an invented grid."""
    from test_predict import build_case

    from trackanno.infer.predict import PREDICTIONS_FILE

    cfg, manifest = build_case(root, "corrections", labels=2)
    movie = str(manifest["movie_id"].iat[0])
    track_id = int(manifest["track_id"].iat[0])
    rows = [
        bout_row(f_start, f_end, label, conf, movie=movie, track=track_id,
                 covering=0.0 if not label else 4.0,
                 reason="" if label else
                 f"no window over frames {f_start}-{f_end} could be featurised")
        for f_start, f_end, label, conf in BOUTS
    ]
    pd.DataFrame(rows).to_parquet(cfg.paths.work_dir / PREDICTIONS_FILE, index=False)
    return cfg, movie, track_id


@pytest.fixture
def panel(app, project):
    """A fresh panel per test, opened on the track under test and closed after.

    Per test rather than per module because a panel holds the corrections file in
    memory: two panels over one project would each save their own idea of a
    track and the second save would erase the first, which is exactly the state
    the Model page's single reused dialog exists to prevent.
    """
    from trackanno.gui.timeline import BoutReviewPanel

    cfg, movie, track_id = project
    widget = BoutReviewPanel(cfg)
    select(widget, movie, track_id)
    yield widget
    widget.close()
    app.processEvents()


@pytest.fixture(autouse=True)
def clean_corrections(project):
    """No correction survives into the next test. The file is the state, so a
    leftover assertion would make a later test pass for the wrong reason."""
    from trackanno.annotate.corrections import corrections_path

    cfg, _, _ = project
    path = corrections_path(cfg)
    path.unlink(missing_ok=True)
    yield
    path.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# the one operation
# --------------------------------------------------------------------------- #
def test_move_split_merge_and_relabel_are_one_operation() -> None:
    """Four gestures, one operation, and that is a finding rather than a shortcut.

    On a disjoint, gapless run-length encoding the state IS (where the cuts are,
    what each cell is called). Naming a class over a frame range moves every cut
    it crosses and renames every cell it covers, so moving a boundary, splitting
    a bout, merging two and relabelling one are the same edit with different
    arguments. Four implementations would be four chances to break the invariant
    instead of one.
    """
    moved = model_partition().assert_class(40, 60, "swimming")
    check("dragging a boundary right is asserting the LEFT class over the frames "
          "the drag swept",
          [(s.f_start, s.f_end, s.label, s.corrected) for s in moved.spans] == [
              (0, 40, "swimming", False), (40, 60, "swimming", True),
              (60, 80, "immotile", False), (80, 120, "swimming", False)],
          str(moved.spans))
    check("...and the encoding still tiles the track", tiles(moved, 0, 120))

    left = model_partition().assert_class(20, 40, "immotile")
    check("dragging it left is the same edit with the RIGHT class",
          [(s.f_start, s.f_end, s.label) for s in left.spans] == [
              (0, 20, "swimming"), (20, 40, "immotile"), (40, 80, "immotile"),
              (80, 120, "swimming")],
          str(left.spans))
    check("...and a corrected span is NOT joined to a model span of the same "
          "class, which would put a person's name on frames they never looked at",
          len(left.spans) == 4 and left.spans[1].corrected
          and not left.spans[2].corrected)

    split = model_partition().assert_class(10, 20, "immotile")
    check("splitting a bout is asserting a different class over part of it",
          [(s.f_start, s.f_end, s.label) for s in split.spans] == [
              (0, 10, "swimming"), (10, 20, "immotile"), (20, 40, "swimming"),
              (40, 80, "immotile"), (80, 120, "swimming")],
          str(split.spans))
    check("...and it still tiles", tiles(split, 0, 120))

    merged = model_partition().assert_class(40, 80, "swimming")
    check("merging a bout into its neighbour is asserting the neighbour's class "
          "over it - every frame of the track is now one class",
          {s.label for s in merged.spans} == {"swimming"}, str(merged.spans))
    check("...and it still tiles", tiles(merged, 0, 120))

    joined = model_partition().assert_class(0, 40, "immotile") \
        .assert_class(40, 80, "immotile")
    check("two ADJACENT corrections of the same class do become one span - same "
          "class, same person, so there is no cut left between them",
          [(s.f_start, s.f_end, s.label, s.corrected) for s in joined.spans] == [
              (0, 80, "immotile", True), (80, 120, "swimming", False)],
          str(joined.spans))


def test_an_edit_that_would_not_tile_the_track_is_refused() -> None:
    """Refused loudly at the data model, never absorbed by the widget.

    A hole in a run-length encoding is indistinguishable from a stretch nobody
    thought about, and an overlap is two classes over one frame. Both are states
    this file can hold and nothing downstream can read, so the only place to stop
    them is before they exist.
    """
    from trackanno.annotate.corrections import Partition, Span

    base = model_partition()
    for name, f_start, f_end, label in [
        ("an empty range claims nothing", 50, 50, "swimming"),
        ("a backwards range is not a span", 60, 40, "swimming"),
        ("a range off the front of the track has no row to correct", -10, 20,
         "swimming"),
        ("a range off the end of the track has no row to correct", 100, 200,
         "swimming"),
        ("a person cannot assert that nobody could score a frame", 0, 20, ""),
    ]:
        try:
            base.assert_class(f_start, f_end, label)
            check(name, False, "the edit was accepted")
        except ValueError as exc:
            check(name, True)
            check(f"...and the refusal says which frames ({name})",
                  str(f_start) in str(exc) or "class name" in str(exc), str(exc))

    try:
        Partition([Span(0, 40, "a"), Span(60, 120, "b")], 0, 120)
        check("a partition with a hole in it is refused", False, "it was built")
    except ValueError as exc:
        check("a partition with a hole in it is refused", "no span" in str(exc),
              str(exc))
    try:
        Partition([Span(0, 60, "a"), Span(40, 120, "b")], 0, 120)
        check("a partition that claims a frame twice is refused", False,
              "it was built")
    except ValueError as exc:
        check("a partition that claims a frame twice is refused",
              "twice" in str(exc), str(exc))


def test_a_long_sequence_of_edits_never_leaves_a_hole() -> None:
    """The invariant is checked after EVERY edit, not after the last one.

    A hole that opens and closes again is a hole a save could have caught, and
    the corrections file is written on every accepted edit.
    """
    rng = np.random.default_rng(20250825)
    partition = model_partition()
    applied = 0
    for _ in range(300):
        f_start = int(rng.integers(0, 120))
        f_end = int(rng.integers(0, 121))
        label = str(rng.choice(["swimming", "immotile", "unclear"]))
        try:
            partition = partition.assert_class(f_start, f_end, label)
        except ValueError:
            continue
        applied += 1
        if not tiles(partition, 0, 120):
            check("every accepted edit leaves the track tiled exactly once",
                  False, f"after asserting {label} over {f_start}-{f_end}: "
                         f"{partition.spans}")
            return
    check("every accepted edit leaves the track tiled exactly once", True)
    check("...over a run of edits that actually happened", applied > 100,
          f"{applied} of 300 random ranges were accepted")


# --------------------------------------------------------------------------- #
# the file
# --------------------------------------------------------------------------- #
def test_the_file_states_its_own_schema(project, panel) -> None:
    """The header on disk, checked, because the column ORDER is load-bearing.

    A column added anywhere but the end makes an older build reconstruct a
    different header from the one it reads, and every downgrade then rewrites
    somebody's whole corrections file. ``store.py`` learned that the hard way and
    this file follows the same rule.
    """
    from trackanno.annotate.corrections import CORRECTION_COLUMNS, corrections_path

    cfg, movie, track_id = project
    check("the edit is accepted", "saved" in panel.assert_class(40, 60, "swimming"))

    path = corrections_path(cfg)
    check("corrections live beside the annotation log, not under work/, because "
          "work/ is deleted and rebuilt and a human judgement is not",
          path.parent == cfg.paths.csv.parent and path.name == "bout_corrections.csv",
          str(path))
    header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
    check("the header on disk is the declared schema, in order",
          header == CORRECTION_COLUMNS, str(header))
    check("...and correction_id leads it, as clip_id leads the annotation log",
          header[0] == "correction_id")

    rows = pd.read_csv(path)
    check("one row per asserted span, and only the spans a person touched",
          len(rows) == 1, str(len(rows)))
    row = rows.iloc[0]
    check("the row names the movie, the track, the group and both ends",
          (str(row["movie_id"]), int(row["track_id"]), str(row["group"]),
           int(row["f_start"]), int(row["f_end"]), str(row["label"]))
          == (movie, track_id, GROUP, 40, 60, "swimming"), str(row.to_dict()))
    check("it is stamped with who said it", str(row["annotator"]) == "tester",
          str(row["annotator"]))
    check("...and when", "T" in str(row["timestamp_utc"]), str(row["timestamp_utc"]))


def test_the_row_records_what_the_model_said_at_the_time(project, panel) -> None:
    """A correction that confirms the model and one that overrules it are the
    same row without this, and only the second is evidence the model is wrong.

    It cannot be recovered later either: the next prediction run overwrites
    predictions.parquet, and a decode is not reproducible once the training set
    has moved on.
    """
    from trackanno.annotate.corrections import CorrectionStore, corrections_path

    cfg, movie, track_id = project
    panel.assert_class(30, 60, "unclear")

    store = CorrectionStore(corrections_path(cfg))
    correction = store.for_track(movie, track_id, GROUP)[0]
    bouts = correction.overruled.get("bouts", [])
    check("the model's own answer over those frames is recorded",
          [(b["f_start"], b["f_end"], b["label"]) for b in bouts]
          == [(30, 40, "swimming"), (40, 60, "immotile")], str(bouts))
    check("...clipped to the frames the correction covers, not the whole bout",
          bouts[0]["f_start"] == 30 and bouts[-1]["f_end"] == 60, str(bouts))
    check("and its confidence, so 'overruled a confident bout' stays askable",
          abs(float(bouts[0]["conf"]) - 0.90) < 1e-9, str(bouts[0]))


def test_two_corrections_over_one_frame_are_refused_out_loud(project) -> None:
    """A hand-edited or merged file can say two things about one frame. Applying
    them in some order and taking the last picks a winner nobody chose, and the
    reviewer is then shown a class no row on screen justifies."""
    from trackanno.annotate.corrections import (
        CORRECTION_COLUMNS, CorrectionStore, corrections_path,
    )
    from trackanno.gui.timeline import PredictionReader

    cfg, movie, track_id = project
    path = corrections_path(cfg)
    lines = [",".join(CORRECTION_COLUMNS)]
    for f_start, f_end, label in [(10, 50, "swimming"), (40, 70, "immotile")]:
        lines.append(",".join([
            f"id{f_start}", movie, str(track_id), GROUP, str(f_start), str(f_end),
            label, "tester", "2026-01-01T00:00:00+00:00", "",
        ]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    try:
        CorrectionStore(path)
        check("two corrections over one frame are refused", False, "it loaded")
    except ValueError as exc:
        check("two corrections over one frame are refused", "claim frame 40" in str(exc),
              str(exc))
        check("...and the refusal names both, so it can be fixed by hand",
              "'swimming'" in str(exc) and "'immotile'" in str(exc), str(exc))

    reader = PredictionReader(cfg)
    check("the review window reports it instead of drawing the model's bouts "
          "over frames somebody has already overruled",
          "claim frame 40" in reader.problem, reader.problem)


def test_a_correction_the_predictions_no_longer_reach_is_kept(project) -> None:
    """A machine running again must not erase a human's judgement.

    A re-run can decode a shorter span. The correction that now hangs off the end
    is clipped for drawing, reported, and left in the file: deleting it would
    destroy a judgement because a prediction ran, and dropping it silently would
    show the model's answer where the person's used to be.
    """
    from trackanno.annotate.corrections import Correction, apply_corrections

    outside = Correction("x", "m", 1, GROUP, 200, 260, "immotile")
    overhang = Correction("y", "m", 1, GROUP, 100, 200, "immotile")
    partition, orphans, clipped = apply_corrections(
        model_partition(), [overhang, outside]
    )

    check("a correction with no overlap at all is an orphan",
          [c.correction_id for c in orphans] == ["x"], str(orphans))
    check("...and one that merely overhangs is clipped to the frames that exist",
          [(s.f_start, s.f_end, s.corrected) for s in partition.spans][-1]
          == (100, 120, True), str(partition.spans))
    check("the track still tiles after an orphan was skipped",
          tiles(partition, 0, 120))
    check("a correction drawn shorter than it was asserted is REPORTED as such, "
          "so the picture can say it is showing less than the file holds",
          [(c.correction_id, c.f_start, c.f_end) for c in clipped]
          == [("y", 100, 200)], str(clipped))
    check("...and it is reported apart from the orphans, because the two need "
          "different sentences: one is not drawn at all, the other is drawn wrong",
          not set(c.correction_id for c in clipped)
          & set(c.correction_id for c in orphans))


def test_a_save_keeps_the_corrections_the_reviewer_was_never_shown(
    app, project
) -> None:
    """One edit must not delete the judgements that are off screen.

    ``replace_track`` drops every row for a (movie, track, group) and writes back
    what it is handed, and what a partition can hand back is only what could be
    drawn. A correction over frames a later, shorter run no longer reaches is
    drawn nowhere; a correction that overhangs is drawn short. If a save built
    its rows from the picture alone, moving one boundary would delete the first
    and truncate the second - a human judgement edited by a machine that ran
    again, with nothing on screen to say it happened.
    """
    from trackanno.annotate.corrections import (
        Correction, CorrectionStore, correction_id, corrections_path,
    )
    from trackanno.gui.timeline import BoutReviewPanel

    cfg, movie, track_id = project
    monday = "2026-01-05T09:00:00+00:00"

    def row(f_start, f_end):
        return Correction(
            correction_id(movie, track_id, GROUP, f_start, f_end), movie, track_id,
            GROUP, f_start, f_end, "immotile", "monday", monday,
            {"bouts": [{"f_start": f_start, "f_end": f_end, "label": "swimming",
                        "conf": 0.9}]},
        )

    # 100-160 overhangs the 0-120 the predictions cover; 200-300 is past them
    # entirely. Both were judged when the decode was longer.
    store = CorrectionStore(corrections_path(cfg))
    store.replace_track(movie, track_id, GROUP, [row(100, 160), row(200, 300)])

    panel = BoutReviewPanel(cfg)
    select(panel, movie, track_id)
    app.processEvents()
    check("the correction past the predictions is not drawn, and is reported",
          [(c.f_start, c.f_end) for c in panel.timeline.track.orphans]
          == [(200, 300)], str(panel.timeline.track.orphans))
    check("...and the one that overhangs is drawn short, and says so",
          [(c.f_start, c.f_end) for c in panel.timeline.track.partly_shown]
          == [(100, 160)], str(panel.timeline.track.partly_shown))

    check("an unrelated edit is accepted", "saved" in panel.assert_class(
        0, 20, "immotile"))

    saved = CorrectionStore(corrections_path(cfg)).for_track(movie, track_id, GROUP)
    check("the judgement the reviewer could not see survives the save",
          (200, 300) in [(c.f_start, c.f_end) for c in saved], str(saved))
    check("...with its own timestamp and annotator, not today's and not this "
          "reviewer's - the file is the only evidence of when it was made",
          [(c.annotator, c.timestamp_utc) for c in saved
           if c.f_start == 200] == [("monday", monday)], str(saved))
    check("the overhanging correction keeps ITS OWN ends: clipping is how it is "
          "drawn, never how it is saved, so the frames past 120 are not lost",
          (100, 160) in [(c.f_start, c.f_end) for c in saved], str(saved))
    check("...and it too keeps its provenance rather than being re-made today",
          [(c.annotator, c.timestamp_utc) for c in saved
           if c.f_start == 100] == [("monday", monday)], str(saved))
    check("...and what it recorded the model as having said, which the next "
          "prediction run has already overwritten",
          [c.overruled.get("bouts") for c in saved if c.f_start == 100]
          == [[{"f_start": 100, "f_end": 160, "label": "swimming", "conf": 0.9}]],
          str([c.overruled for c in saved if c.f_start == 100]))
    check("and the new edit is in the file beside them",
          [(c.f_start, c.f_end, c.annotator) for c in saved if c.f_start == 0]
          == [(0, 20, "tester")], str(saved))
    check("three rows, so nothing was merged into anything either",
          len(saved) == 3, str(len(saved)))

    before = digest(corrections_path(cfg))
    check("re-asserting what the file already says is accepted",
          "saved" in panel.assert_class(0, 20, "immotile"))
    check("...and rewrites the file byte for byte, so a reviewer clicking twice "
          "cannot re-stamp a judgement with today's date",
          digest(corrections_path(cfg)) == before)

    # The one case where an overhanging row DOES change: the reviewer overwrites
    # the part of it they can see. Only that part is theirs to change.
    check("re-judging the visible part of an overhanging correction is accepted",
          "saved" in panel.assert_class(100, 120, "swimming"))
    saved = CorrectionStore(corrections_path(cfg)).for_track(movie, track_id, GROUP)
    check("the frames past the predictions keep the earlier judgement, because "
          "the reviewer was never shown them and cannot have changed their mind",
          [(c.f_start, c.f_end, c.annotator, c.timestamp_utc) for c in saved
           if c.f_start == 120] == [(120, 160, "monday", monday)], str(saved))
    check("...including what it recorded the model as saying, clipped to the "
          "frames it still claims rather than left quoting frames it does not",
          [c.overruled.get("bouts") for c in saved if c.f_start == 120]
          == [[{"f_start": 120, "f_end": 160, "label": "swimming", "conf": 0.9}]],
          str([c.overruled for c in saved if c.f_start == 120]))
    check("and the frames they DID re-judge are a new row, in their name",
          [(c.f_start, c.f_end, c.label, c.annotator) for c in saved
           if c.f_start == 100] == [(100, 120, "swimming", "tester")], str(saved))
    panel.close()
    app.processEvents()


def test_two_corrections_drawn_as_one_span_stay_two_rows(app, project) -> None:
    """Monday's judgement and Tuesday's are drawn as one bar and filed as two.

    Two adjacent corrections of the same class have no cut between them to draw,
    so the picture merges them - that is a fact about the picture. The file must
    not: one row cannot hold two timestamps, two annotators, or two records of
    what the model said at the time, and a merged row would date Monday's work to
    Tuesday and credit it to whoever was on shift then. Six months later the
    timestamps are the only evidence of when a judgement was actually made.
    """
    from trackanno.annotate.corrections import (
        Correction, CorrectionStore, correction_id, corrections_path,
    )
    from trackanno.gui.timeline import BoutReviewPanel

    cfg, movie, track_id = project
    days = {0: ("monday", "2026-01-05T09:00:00+00:00"),
            40: ("tuesday", "2026-01-06T14:30:00+00:00")}
    store = CorrectionStore(corrections_path(cfg))
    store.replace_track(movie, track_id, GROUP, [
        Correction(correction_id(movie, track_id, GROUP, f_start, f_start + 40),
                   movie, track_id, GROUP, f_start, f_start + 40, "immotile",
                   *days[f_start])
        for f_start in (0, 40)
    ])

    panel = BoutReviewPanel(cfg)
    select(panel, movie, track_id)
    app.processEvents()
    spans = panel.timeline.track.partitions[GROUP].corrected_spans()
    check("the two are drawn as ONE span - there is no cut between them to draw",
          [(s.f_start, s.f_end, s.label) for s in spans] == [(0, 80, "immotile")],
          str(spans))

    check("an unrelated edit elsewhere is accepted",
          "saved" in panel.assert_class(100, 120, "immotile"))
    saved = CorrectionStore(corrections_path(cfg)).for_track(movie, track_id, GROUP)
    check("...and the file still holds both assertions, separately",
          [(c.f_start, c.f_end, c.annotator, c.timestamp_utc) for c in saved][:2]
          == [(0, 40) + days[0], (40, 80) + days[40]], str(saved))
    check("neither was re-stamped with today's date by the merge",
          all("2026-01-0" in c.timestamp_utc for c in saved if c.f_start < 80),
          str([c.timestamp_utc for c in saved]))
    panel.close()
    app.processEvents()


def test_a_column_this_build_does_not_know_survives_a_save(app, project) -> None:
    """The file is shared by every track, so a strip is a loss everywhere.

    ``store.py``'s rule is that a column added later goes at the END and an older
    build reconstructs the header it read, leaves the file alone and keeps what
    it cannot interpret. This file claims the same rule in its module docstring;
    the check is that it actually holds, because the failure is silent: somebody
    opens an older build, moves one boundary, and a newer build's column is gone
    from every row of everybody's corrections.
    """
    from trackanno.annotate.corrections import (
        CORRECTION_COLUMNS, CorrectionStore, corrections_path,
    )
    from trackanno.gui.timeline import BoutReviewPanel

    cfg, movie, track_id = project
    path = corrections_path(cfg)
    header = CORRECTION_COLUMNS + ["confidence_in_self"]
    # One row for the track under test and one for a track nobody is editing:
    # the blast radius of a rewrite is the whole file, not the edited key.
    path.write_text("\n".join([
        ",".join(header),
        f"keep-me,{movie},{track_id},{GROUP},40,60,immotile,monday,"
        "2026-01-05T09:00:00+00:00,,quite sure",
        f"other-track,{movie},77,{GROUP},0,10,swimming,monday,"
        "2026-01-05T09:00:00+00:00,,not at all",
    ]) + "\n", encoding="utf-8")

    panel = BoutReviewPanel(cfg)
    select(panel, movie, track_id)
    app.processEvents()
    check("a file with an unknown column still opens", panel.reader.problem == "",
          panel.reader.problem)
    check("the store rebuilds the header it read, in the file's own order",
          panel.reader.corrections.columns == header,
          str(panel.reader.corrections.columns))

    check("an edit on one track is accepted", "saved" in panel.assert_class(
        80, 100, "immotile"))
    lines = path.read_text(encoding="utf-8").splitlines()
    check("the header on disk is unchanged, so the file is not churned",
          lines[0].split(",") == header, lines[0])
    rows = pd.read_csv(path).fillna("")
    check("the row for the track nobody was editing keeps its unknown cell",
          list(rows[rows["correction_id"] == "other-track"]["confidence_in_self"])
          == ["not at all"], str(rows.to_dict("records")))
    check("...and so does the one on the edited track that the edit did not touch",
          list(rows[rows["f_start"] == 40]["confidence_in_self"]) == ["quite sure"],
          str(rows.to_dict("records")))
    check("a row this build wrote leaves the cell empty rather than inventing one",
          list(rows[rows["f_start"] == 80]["confidence_in_self"]) == [""],
          str(rows.to_dict("records")))
    panel.close()
    app.processEvents()


# --------------------------------------------------------------------------- #
# what a correction must never touch
# --------------------------------------------------------------------------- #
def test_correcting_a_bout_writes_nothing_but_the_corrections_file(
    project, panel
) -> None:
    """The whole reason a correction has a file of its own.

    A corrected bout written into annotations.csv would be a variable-length span
    in a table of fixed windows, and once a derived label and a hand label are
    indistinguishable there nobody can ever again say what a person judged. A
    correction written back into predictions.parquet would destroy the model's
    own answer, which is the thing the correction is a record ABOUT.
    """
    from trackanno.infer.predict import predictions_path

    cfg, _, _ = project
    before = {p: digest(p) for p in (cfg.paths.csv, predictions_path(cfg))}

    check("the correction is made", "saved" in panel.assert_class(40, 60, "swimming"))
    check("...and another", "saved" in panel.assert_class(80, 100, "immotile"))

    check("annotations.csv is byte-identical",
          digest(cfg.paths.csv) == before[cfg.paths.csv])
    check("predictions.parquet is byte-identical",
          digest(predictions_path(cfg)) == before[predictions_path(cfg)])


def test_a_windows_file_cannot_be_corrected(app, root) -> None:
    """A window's edges are multiples the predictor chose, so moving one corrects
    nothing about the animal - and a window relabelled by hand is a clip label,
    which already has a file. Refused with the reason, not silently inert."""
    from test_predict import build_case

    from trackanno.gui.timeline import BoutReviewPanel, PredictionReader
    from trackanno.infer.predict import predict_all

    cfg, _ = build_case(root, "corrections-windows", labels=2)
    predict_all(cfg)

    reader = PredictionReader(cfg)
    movie = reader.movies[0]
    track_id = next(t for t, n in reader.tracks(movie) if n)
    track = reader.track(movie, track_id)
    check("a windows file offers no partition to edit", track.partitions == {},
          str(track.partitions))
    check("...and says why, in terms of what a window is",
          "not decoded bouts" in track.edit_problem
          and "annotations.csv" in track.edit_problem, track.edit_problem)

    widget = BoutReviewPanel(cfg)
    select(widget, movie, track_id)
    app.processEvents()
    check("the correction controls are dead rather than misleading",
          not widget.apply_button.isEnabled())
    check("and the panel says so where a reviewer will look",
          "not decoded bouts" in widget.status.text(), widget.status.text())
    widget.close()
    app.processEvents()


# --------------------------------------------------------------------------- #
# staying separable
# --------------------------------------------------------------------------- #
def test_a_correction_round_trips_and_stays_a_correction(project, panel, app) -> None:
    """Reopened, a correction is still recognisably a person's and the model's
    bout is still recognisably the model's.

    If the two ever merge, a reviewer reads their own judgement back as the model
    agreeing with them - and there is no way to notice, because the picture is
    the only place the difference was ever shown.
    """
    from trackanno.gui.timeline import BoutReviewPanel, PredictionReader

    cfg, movie, track_id = project
    check("the edit is saved", "saved" in panel.assert_class(40, 60, "unclear"))

    reader = PredictionReader(cfg)
    track = reader.track(movie, track_id)
    partition = track.partitions[GROUP]
    check("the correction comes back from the file",
          [(s.f_start, s.f_end, s.label, s.corrected) for s in partition.spans] == [
              (0, 40, "swimming", False), (40, 60, "unclear", True),
              (60, 80, "immotile", False), (80, 100, "", False),
              (100, 120, "swimming", False)],
          str(partition.spans))
    check("...and the track still tiles after a reload", tiles(partition, 0, 120))
    check("exactly one span on this track is a person's",
          [(s.f_start, s.f_end) for s in partition.corrected_spans()] == [(40, 60)],
          str(partition.corrected_spans()))

    lane = track.lanes[GROUP]
    check("the model's own bouts are untouched in the lane above",
          [(s.f_start, s.f_end, s.label) for s in lane]
          == [(f_start, f_end, label) for f_start, f_end, label, _ in BOUTS],
          str([(s.f_start, s.f_end, s.label) for s in lane]))
    check("...including the confidence the run measured, which the record lane "
          "deliberately does not carry",
          abs(lane[1].conf - 0.45) < 1e-9, str(lane[1].conf))

    fresh = BoutReviewPanel(cfg)
    select(fresh, movie, track_id)
    app.processEvents()
    corrected = fresh.timeline.track.partitions[GROUP].corrected_spans()
    check("the dialog reopens on the reviewer's earlier work rather than on the "
          "model's answer",
          [(s.f_start, s.f_end, s.label) for s in corrected] == [(40, 60, "unclear")],
          str(corrected))
    tip = fresh.timeline._describe_span(GROUP, corrected[0])
    check("and the span says out loud that it is nobody's measurement",
          "asserted by a person" in tip and "carries no confidence" in tip, tip)
    fresh.close()
    app.processEvents()


def test_undo_and_revert_go_through_the_file_too(project, panel, monkeypatch) -> None:
    """Because the file is the state. An undo that only moved a widget back would
    leave the disk holding an edit the reviewer has already taken back."""
    from qtpy.QtWidgets import QMessageBox

    from trackanno.annotate.corrections import CorrectionStore, corrections_path

    cfg, movie, track_id = project
    panel.assert_class(40, 60, "unclear")
    panel.assert_class(0, 20, "immotile")
    check("two corrections are on the track",
          len(CorrectionStore(corrections_path(cfg)).for_track(
              movie, track_id, GROUP)) == 2)

    panel._on_undo()
    saved = CorrectionStore(corrections_path(cfg)).for_track(movie, track_id, GROUP)
    check("an undo is a save: the file loses the edit, not just the widget",
          [(c.f_start, c.f_end) for c in saved] == [(40, 60)], str(saved))
    check("...and the picture agrees with the file",
          [(s.f_start, s.f_end) for s in
           panel.timeline.track.partitions[GROUP].corrected_spans()] == [(40, 60)])

    asked = []
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: asked.append(a) or QMessageBox.No)
    panel._on_revert()
    check("reverting asks first - it is the one action here that destroys a human "
          "judgement rather than recording one",
          len(asked) == 1 and len(CorrectionStore(corrections_path(cfg)).for_track(
              movie, track_id, GROUP)) == 1)

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    panel._on_revert()
    check("and when it is confirmed the file keeps no row for that track",
          CorrectionStore(corrections_path(cfg)).for_track(
              movie, track_id, GROUP) == [])
    check("...and the model's own bouts come back on screen unchanged",
          [(s.f_start, s.f_end, s.label, s.corrected) for s in
           panel.timeline.track.partitions[GROUP].spans]
          == [(f_start, f_end, label, False) for f_start, f_end, label, _ in BOUTS],
          str(panel.timeline.track.partitions[GROUP].spans))


def test_merging_is_the_same_assertion_under_another_name(project, panel) -> None:
    """The merge buttons exist because the gesture has a name, not because the
    data model has an operation for it: each one asserts a neighbour's class over
    the selected span, and the cut between them stops existing."""
    panel.from_frame.setValue(40)
    panel._on_merge(-1)
    spans = panel.timeline.track.partitions[GROUP].spans
    check("merging a bout into the one before it takes that one's class",
          [(s.f_start, s.f_end, s.label) for s in spans][:2] == [
              (0, 40, "swimming"), (40, 80, "swimming")], str(spans))
    check("...and the encoding still tiles",
          tiles(panel.timeline.track.partitions[GROUP], 0, 120))

    panel.from_frame.setValue(0)
    panel._on_merge(-1)
    check("there is nothing to merge into at the start of a track, and it says so",
          "no bout on that side" in panel.status.text(), panel.status.text())

    panel.from_frame.setValue(100)
    panel._on_merge(-1)
    check("and merging into a stretch nothing could score is refused: it would "
          "assert that nothing can score frames something did",
          "nothing could score them" in panel.status.text(), panel.status.text())


def test_a_multi_select_group_refuses_rather_than_half_works(app, root) -> None:
    """One of a multi group's predictions is a COMBINATION of options, not one of
    them. A control that offered the options one at a time would let a reviewer
    assert something narrower than they meant without saying so, so it says no."""
    from test_predict import build_case

    from trackanno.gui.timeline import BoutReviewPanel
    from trackanno.infer.predict import PREDICTIONS_FILE

    cfg, manifest = build_case(root, "corrections-multi")
    movie, track_id = str(manifest["movie_id"].iat[0]), int(manifest["track_id"].iat[0])
    groups = (GROUP, "quality")
    rows = [
        bout_row(0, 60, "swimming", 0.9, movie=movie, track=track_id,
                 group=GROUP, groups=groups),
        bout_row(60, 120, "immotile", 0.7, movie=movie, track=track_id,
                 group=GROUP, groups=groups),
        bout_row(0, 120, "blurry|occluded", 0.6, movie=movie, track=track_id,
                 group="quality", groups=groups),
    ]
    pd.DataFrame(rows).to_parquet(cfg.paths.work_dir / PREDICTIONS_FILE, index=False)

    widget = BoutReviewPanel(cfg)
    select(widget, movie, track_id)
    app.processEvents()
    widget.group.setCurrentText("quality")
    app.processEvents()
    check("the multi group is refused with the reason",
          "multi-select" in widget.status.text()
          and "not supported" in widget.status.text(),
          widget.status.text())
    check("...and it offers no classes to pick from, rather than the halves of a "
          "combination", widget.klass.count() == 0)
    check("the refusal is per group, and the single-choice group beside it still "
          "works", "saved" in widget.assert_class(0, 30, "unclear", GROUP))
    check("...so the control that changes group stays reachable",
          widget.group.isEnabled())
    widget.close()
    app.processEvents()


def test_a_class_the_config_dropped_still_loads(project, panel) -> None:
    """Writing is validated and reading is not, deliberately.

    A row records what a person judged. Renaming an option in config.yaml must
    not delete that record - it would be a human judgement destroyed by a text
    edit - so an unknown class comes back and is drawn orphaned, exactly as the
    renderer treats a colour it can no longer find.
    """
    from trackanno.annotate.corrections import (
        Correction, CorrectionStore, correction_id, corrections_path,
    )
    from trackanno.gui.timeline import PredictionReader

    cfg, movie, track_id = project
    store = CorrectionStore(corrections_path(cfg))
    store.replace_track(movie, track_id, GROUP, [Correction(
        correction_id(movie, track_id, GROUP, 40, 60), movie, track_id, GROUP,
        40, 60, "gliding", "someone-else",
    )])

    partition = PredictionReader(cfg).track(movie, track_id).partitions[GROUP]
    check("the correction survives a config that no longer names its class",
          [(s.f_start, s.f_end, s.label, s.corrected) for s in partition.spans][1]
          == (40, 60, "gliding", True), str(partition.spans))
    check("...and the track still tiles around it", tiles(partition, 0, 120))
    check("but a NEW judgement cannot be filed under it",
          "declares no class" in panel.assert_class(80, 100, "gliding"))


def test_a_class_the_config_does_not_declare_is_refused(project, panel) -> None:
    """A judgement filed under a name nothing else in the project can read is a
    judgement nobody can act on: it cannot be coloured, trained on or exported."""
    message = panel.assert_class(40, 60, "backflip")
    check("an undeclared class is refused", "declares no class" in message, message)
    check("...and the refusal lists the ones that exist, so it is actionable",
          "swimming" in message and "immotile" in message, message)
    check("nothing was written", not panel.timeline.track.partitions[GROUP]
          .corrected_spans())


def test_a_dragged_boundary_asserts_the_class_it_was_dragged_away_from(
    project, panel
) -> None:
    """The direction of the drag decides which class the swept frames become, and
    a drag into frames nothing could score is refused.

    Extending an empty-labelled span would spread "no window could be featurised
    here" over frames something was, which is a claim about the run that a person
    is not in a position to make.
    """
    partition = panel.timeline.track.partitions[GROUP]
    check("the track starts as the model left it", not partition.corrected_spans())

    panel._on_boundary_moved(GROUP, 40, 55)
    spans = panel.timeline.track.partitions[GROUP].spans
    check("dragging right extends the span on the left",
          [(s.f_start, s.f_end, s.label, s.corrected) for s in spans][:3] == [
              (0, 40, "swimming", False), (40, 55, "swimming", True),
              (55, 80, "immotile", False)],
          str(spans))

    panel._on_boundary_moved(GROUP, 55, 45)
    spans = panel.timeline.track.partitions[GROUP].spans
    check("dragging left extends the span on the right",
          [(s.f_start, s.f_end, s.label) for s in spans][:4] == [
              (0, 40, "swimming"), (40, 45, "swimming"), (45, 55, "immotile"),
              (55, 80, "immotile")],
          str(spans))
    check("...and dragging back does not hand the frames back to the model: the "
          "reviewer swept 45-55 twice and judged them both times, so both spans "
          "stay theirs and the model's own claim stays in the lane above",
          [(s.f_start, s.f_end) for s in
           panel.timeline.track.partitions[GROUP].corrected_spans()]
          == [(40, 45), (45, 55)],
          str(panel.timeline.track.partitions[GROUP].corrected_spans()))
    check("...and the track tiles after both",
          tiles(panel.timeline.track.partitions[GROUP], 0, 120))

    # Boundary 100 separates the unscored stretch 80-100 from the bout after it,
    # so dragging it RIGHT is a request to extend "nothing scored this".
    panel._on_boundary_moved(GROUP, 100, 110)
    check("a drag that would extend a stretch nothing could score is refused",
          "no window over them could be featurised" in panel.status.text(),
          panel.status.text())
    check("...and nothing moved",
          panel.timeline.track.partitions[GROUP].at(95).label == "",
          str(panel.timeline.track.partitions[GROUP].spans))

    check("a boundary cannot be dragged past the far edge of either neighbour, "
          "which would delete a bout by accident - merging is a button, where it "
          "says what it is",
          panel.timeline._clamp_boundary(GROUP, 45, 5) == 41
          and panel.timeline._clamp_boundary(GROUP, 45, 500) == 54,
          f"{panel.timeline._clamp_boundary(GROUP, 45, 5)} / "
          f"{panel.timeline._clamp_boundary(GROUP, 45, 500)}")


def test_the_picture_keeps_the_two_claims_apart(app, project, panel) -> None:
    """A correction is drawn in a lane of its own, below the model's, and the
    caption says which is which. A reviewer who cannot tell them apart will
    eventually trust a correction they made themselves as if it were evidence."""
    from trackanno.gui.timeline import caption_lines

    panel.resize(1000, 460)
    panel.show()
    app.processEvents()
    panel.assert_class(40, 60, "unclear")
    panel.timeline.resize(940, 300)
    app.processEvents()

    plan = panel.timeline._lane_plan()
    check("every group gets a model lane and a record lane, in that order",
          [kind for kind, _, _ in plan] == ["model", "record"], str(plan))
    check("the record lane is hit-testable across the whole track, so any span "
          "can be clicked",
          len([h for h in panel.timeline._hits if h[3] == "record"])
          == len(panel.timeline.track.partitions[GROUP]),
          str(len(panel.timeline._hits)))

    lines = " ".join(caption_lines(panel.timeline.track, panel.reader.resolution))
    check("the caption names the lower lane and what a fill means",
          "THE RECORD" in lines and "filled solid" in lines, lines)
    check("...counts what is the reviewer's own", "1 span(s) on this track are yours"
          in lines, lines)
    check("...and says a boundary a person placed is not uncertain to the stride",
          "exact to the frame you chose" in lines, lines)
    check("the record lane carries no confidence at all, because what is asserted "
          "about a stretch of frames is not a posterior over it",
          "not a posterior" in lines, lines)


# --------------------------------------------------------------------------- #
# what a correction implies about the training set
# --------------------------------------------------------------------------- #
def windows(*spans, movie="m", track=1):
    return [
        {"clip_id": f"{movie}__t{track:04d}__f{f_start:06d}", "movie_id": movie,
         "track_id": track, "f_start": f_start, "f_end": f_end}
        for f_start, f_end in spans
    ]


def corrections(*spans, movie="m", track=1, group=GROUP):
    from trackanno.annotate.corrections import Correction, correction_id

    return [
        Correction(correction_id(movie, track, group, f_start, f_end), movie, track,
                   group, f_start, f_end, label)
        for f_start, f_end, label in spans
    ]


def test_a_window_inside_a_correction_derives_and_a_straddler_does_not() -> None:
    """THE STRADDLER RULE, asserted rather than left to be incidental.

    A window derives a class if and only if it lies entirely inside a single
    corrected span. A window that crosses a transition has features blended from
    both behaviours - that is what the frames contain, not an artefact - so
    labelling it with either class teaches the next model that a blend is a pure
    class, exactly at the transitions it is already worst at. A majority or
    midpoint rule would label the most misleading windows most confidently.
    """
    from trackanno.annotate.corrections import derive_window_labels

    grid = windows((0, 50), (50, 100), (100, 150))
    result = derive_window_labels(corrections((0, 100, "swimming")), grid)
    check("a window entirely inside a corrected span takes its class",
          [(d.f_start, d.f_end, d.label) for d in result.derived]
          == [(0, 50, "swimming"), (50, 100, "swimming")], str(result.derived))
    check("...and points back at the correction it came from, so a derived label "
          "is never an orphan claim",
          all(d.correction_id.endswith("f000000-000100") for d in result.derived),
          str([d.correction_id for d in result.derived]))
    check("a window no correction touches is neither derived nor a straddler - "
          "nothing was asserted about it",
          result.n_untouched == 1 and result.n_straddling == 0,
          f"{result.n_untouched} untouched, {result.n_straddling} straddling")

    partial = derive_window_labels(corrections((20, 70, "swimming")), grid)
    check("a window that only PARTLY overlaps a correction derives nothing",
          partial.n_derived == 0, str(partial.derived))
    check("...and both halves of the overlap are reported as straddlers",
          [(s.f_start, s.f_end) for s in partial.straddling] == [(0, 50), (50, 100)],
          str(partial.straddling))
    check("the reason names the frames that nothing has judged, so a reviewer can "
          "act on it",
          "20 frame(s) before it" in partial.straddling[0].reason
          and "30 frame(s) after it" in partial.straddling[1].reason,
          str([s.reason for s in partial.straddling]))

    crossing = derive_window_labels(
        corrections((0, 30, "swimming"), (30, 100, "immotile")), grid
    )
    check("a window crossing a corrected boundary derives nothing either",
          crossing.n_derived == 1 and crossing.n_straddling == 1,
          f"{crossing.n_derived} derived, {crossing.n_straddling} straddling")
    check("...and the reason names the boundary it crosses",
          "cross a corrected boundary at 30" in crossing.straddling[0].reason,
          crossing.straddling[0].reason)
    check("the window fully inside the second correction still derives, so one "
          "ambiguous window does not poison its neighbours",
          crossing.derived[0].label == "immotile", str(crossing.derived))


def test_nothing_is_derived_from_model_output(project, panel) -> None:
    """Only corrections derive. A label derived from a bout the model produced
    would feed the model its own answer, which is the rule predictions never
    touch the annotation CSV exists to enforce."""
    from trackanno.annotate.corrections import derive_window_labels, manifest_windows

    cfg, movie, track_id = project
    grid = manifest_windows(cfg, movie)
    check("the movie has a real clip grid to derive against", len(grid) > 0)

    empty = derive_window_labels([], grid)
    check("with no corrections nothing is derived, though the model has bouts "
          "over every frame of the track",
          empty.n_derived == 0 and empty.n_straddling == 0
          and empty.n_untouched == len(grid),
          f"{empty.n_derived}/{empty.n_straddling}/{empty.n_untouched}")

    panel.assert_class(0, 50, "unclear")
    report = panel.derivation_text()
    check("the report counts what the correction would imply",
          "1 fall entirely inside a corrected span" in report, report)
    check("...and says why the straddlers get nothing", "blended from both" in report,
          report)
    check("and it says, in the report itself, that nothing was written",
          "Nothing has been written to annotations.csv." in report, report)
    check("...and that a derived label is not a hand label",
          "not a hand label" in report, report)


def test_the_derivation_is_per_group(project) -> None:
    """A window can be unambiguous in one category group and straddle a boundary
    in another; collapsing the two would let one group's ambiguity delete the
    other group's answer."""
    from trackanno.annotate.corrections import derive_window_labels

    grid = windows((0, 50), (50, 100))
    result = derive_window_labels(
        corrections((0, 100, "swimming"))
        + corrections((20, 70, "blurry"), group="quality"),
        grid,
    )
    check("the unambiguous group derives both windows",
          sorted(d.group for d in result.derived) == [GROUP, GROUP],
          str([(d.group, d.f_start) for d in result.derived]))
    check("...while the ambiguous one reports both as straddlers",
          sorted(s.group for s in result.straddling) == ["quality", "quality"],
          str([(s.group, s.f_start) for s in result.straddling]))
    check("and a window counted in both is still one window",
          result.n_windows == 2 and result.n_untouched == 0,
          f"{result.n_windows} windows, {result.n_untouched} untouched")
