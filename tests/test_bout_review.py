"""Looking at what a decode produced, without being lied to by the picture.

A bout is a claim about WHEN an animal changed what it was doing, and a timeline
is the only shape that claim can be checked in. The checks here:

* the file decides what a row IS - ``bout_id``, never the config;
* a track with no rows, a file of the older kind and a project with no
  predictions at all are three normal states, each with a sentence.

Nothing here needs a display, a GPU or the repository's own data. No ``ClipStore``
is opened either: the review path reads parquet, and a suite that mapped the
shared fixture's clip cache would leave it mapped for whichever suite re-runs
detection next.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from harness import check

GROUP = "behaviour"
CLASSES = ("swimming", "immotile")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def painted(widget) -> np.ndarray:
    """The widget's own pixels, so "it drew something" is measured where it
    matters rather than inferred from a method having been called."""
    from qtpy.QtGui import QPixmap

    pixmap = QPixmap(widget.size())
    widget.render(pixmap)
    image = pixmap.toImage()
    raw = image.constBits().asstring(image.sizeInBytes())
    return np.frombuffer(raw, np.uint8).reshape(
        image.height(), image.bytesPerLine() // 4, 4
    )[..., :3]


def ink(frame: np.ndarray) -> int:
    """Pixels that are not the commonest colour in the image - i.e. not the
    background it was cleared to."""
    flat = frame.reshape(-1, 3)
    colours, counts = np.unique(flat, axis=0, return_counts=True)
    return int(len(flat) - counts.max())


def bout_row(f_start, f_end, label, conf, reason="",
             movie="m", track=1, group=GROUP, groups=(GROUP,)) -> dict:
    """One row of the bout schema :func:`trackanno.infer.predict._bout_predictions`
    writes, including the columns of every OTHER group left empty - that shape
    is the thing the reader has to get right, so a hand-built row reproduces it
    rather than a convenient subset."""
    row = {
        "bout_id": f"{movie}__t{track:04d}__f{f_start:06d}-{f_end:06d}",
        "movie_id": movie, "track_id": track,
        "f_start": f_start, "f_end": f_end, "n_frames": f_end - f_start,
        "group": group, "skip_reason": reason,
    }
    for other in groups:
        row[f"{other}_pred"] = label if other == group else ""
        row[f"{other}_conf"] = conf if other == group else np.nan
    return row


# --------------------------------------------------------------------------- #
# the schema, on a file built by hand
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def synthetic(root):
    """A project carrying a bout table nothing fitted: every column at a value
    worth reading, including the ones a happy path never produces."""
    from test_predict import build_case

    cfg, _ = build_case(root, "review-synthetic")
    write_bouts(cfg, [
        bout_row(0, 120, "swimming", 0.91),
        bout_row(120, 160, "immotile", 0.42),          # marginal
        bout_row(160, 180, "", np.nan,
                 reason="no window over frames 160-180 could be featurised"),
        bout_row(180, 300, "swimming", 0.77),
    ])
    return cfg


def write_bouts(cfg, rows) -> None:
    """Put a hand-built bout table where the review reads predictions."""
    from trackanno.infer.predict import PREDICTIONS_FILE

    pd.DataFrame(rows).to_parquet(
        cfg.paths.work_dir / PREDICTIONS_FILE, index=False)


def first_movie_bouts(cfg) -> list[dict]:
    """A few bouts on the first prepared movie's first track."""
    from trackanno.annotate.clips import TRACKS_FILE

    movie = next(p for p in sorted(cfg.paths.work_dir.iterdir())
                 if p.is_dir() and (p / TRACKS_FILE).exists())
    track = int(pd.read_parquet(movie / TRACKS_FILE)["track_id"].min())
    return [bout_row(0, 40, "swimming", 0.9, movie=movie.name, track=track),
            bout_row(40, 100, "immotile", 0.6, movie=movie.name, track=track)]


def test_the_reader_maps_every_column_a_bout_carries(synthetic) -> None:
    """One row per (group, bout), and each of its columns means something
    different: getting one of them wrong produces a plausible picture."""
    from trackanno.gui.timeline import PredictionReader

    reader = PredictionReader(synthetic)
    check("the file opens", reader.problem == "", reader.problem)
    check("the predicted group is found from its _pred column",
          reader.groups == [GROUP], str(reader.groups))

    track = reader.track("m", 1)
    check("the track's span is the union of its rows",
          (track.f_lo, track.f_hi) == (0, 300), f"{track.f_lo}-{track.f_hi}")
    lane = track.lanes[GROUP]
    check("every row of the group became a segment", len(lane) == 4, str(len(lane)))
    check("the segments are sorted, disjoint and gapless",
          all(a.f_end == b.f_start for a, b in zip(lane, lane[1:]))
          and lane[0].f_start == track.f_lo and lane[-1].f_end == track.f_hi)

    check("a confident bout keeps its class and its mean posterior",
          lane[0].label == "swimming" and abs(lane[0].conf - 0.91) < 1e-9)
    check("a marginal bout is not rounded up to a confident one",
          abs(lane[1].conf - 0.42) < 1e-9, str(lane[1].conf))

    gap = lane[2]
    check("a bout over frames nothing could score carries no class",
          gap.label == "", gap.label)
    check("...and its confidence is NaN, never 0.0 - nobody measured it",
          not np.isfinite(gap.conf), str(gap.conf))
    check("...and it keeps the skip_reason that says why",
          "could be featurised" in gap.skip_reason, gap.skip_reason)


def test_a_track_with_no_predictions_is_a_normal_state(app, synthetic) -> None:
    """Not an error, not an empty widget: a sentence saying it is normal.

    A track every window over which was skipped for too few detections gets no
    row at all, and a blank rectangle there reads as a broken widget - so the
    one question a reviewer would ask ("is this the tool or the data?") could
    not be answered from the screen.
    """
    from trackanno.gui.timeline import BoutReviewPanel, PredictionReader

    reader = PredictionReader(synthetic)
    track = reader.track("m", 4242)
    check("the reader answers with an empty track rather than raising", track.empty)

    panel = BoutReviewPanel(synthetic)
    panel.resize(900, 320)
    panel.show()
    app.processEvents()
    panel.timeline.set_track(track)
    panel.timeline.resize(900, 200)
    app.processEvents()
    check("the empty timeline paints without raising", ink(painted(panel.timeline)) > 0)
    check("and the words on it say the outcome is normal",
          "That is normal, not an error" in panel.timeline._nothing_here(),
          panel.timeline._nothing_here())
    panel.close()
    app.processEvents()


# --------------------------------------------------------------------------- #
# a real decode
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def decoded(root):
    """The planted 300-frame track, decoded at stride 10."""
    from test_predict import planted_case

    from trackanno.infer.predict import predict_all

    cfg, movie = planted_case(root, "review-decoded", 10)
    predict_all(cfg)
    return cfg, movie


def test_a_real_decode_reaches_the_timeline_intact(decoded) -> None:
    """The decoded bouts tile the track and keep the planted transition."""
    from trackanno.gui.timeline import PredictionReader

    cfg, movie = decoded
    reader = PredictionReader(cfg)
    check("the decoded file opens", reader.problem == "", reader.problem)

    track = reader.track(movie, 1)
    lane = track.lanes[GROUP]
    check("the bouts tile the track's whole span",
          lane[0].f_start == track.f_lo and lane[-1].f_end == track.f_hi
          and all(a.f_end == b.f_start for a, b in zip(lane, lane[1:])),
          str([(s.f_start, s.f_end, s.label) for s in lane]))
    check("the planted transition shows up as more than one bout",
          len({s.label for s in lane if s.label}) == 2,
          str([s.label for s in lane]))



def test_the_timeline_draws_a_decoded_track(app, decoded) -> None:
    """The panel opens on the real file and puts pixels on the screen."""
    from trackanno.gui.timeline import BoutReviewPanel

    cfg, movie = decoded
    panel = BoutReviewPanel(cfg)
    panel.resize(1000, 400)
    panel.show()
    app.processEvents()

    check("the movie selector is filled from the file",
          [panel.movie.itemText(i) for i in range(panel.movie.count())] == [movie],
          str([panel.movie.itemText(i) for i in range(panel.movie.count())]))
    check("the track selector offers the decoded track", panel.track.count() >= 1)
    check("the headline counts the rows and names what they are",
          "predicted bouts" in panel.headline.text(), panel.headline.text())

    panel.timeline.resize(940, 200)
    app.processEvents()
    check("the timeline actually paints", ink(painted(panel.timeline)) > 500,
          str(ink(painted(panel.timeline))))
    hits = panel.timeline._hits
    check("every drawn span is hit-testable, so the detail has somewhere "
          "to live",
          len(hits) == panel.timeline.track.n_segments,
          f"{len(hits)} rects for {panel.timeline.track.n_segments} segments")

    group, segment = hits[0][1], hits[0][2]
    tip = panel.timeline._describe(group, segment)
    check("the tooltip says confidence is a mean and not a peak",
          "MEAN posterior" in tip and "not its peak" in tip, tip)
    panel.close()
    app.processEvents()


def test_a_track_the_predictions_never_mention_is_still_selectable(decoded) -> None:
    """Built from the union of the predictions and ``tracks.parquet``.

    A combo filled only from the predictions makes the empty state unreachable,
    and then the reviewer cannot ask the one question a silent track raises.
    """
    from trackanno.annotate.clips import TRACKS_FILE
    from trackanno.gui.timeline import PredictionReader

    cfg, movie = decoded
    path = cfg.movie_dir(movie) / TRACKS_FILE
    tracks = pd.read_parquet(path)
    if not (tracks["track_id"] == 7).any():
        extra = tracks.head(3).copy()
        extra["track_id"] = 7
        pd.concat([tracks, extra], ignore_index=True).to_parquet(path, index=False)

    listed = PredictionReader(cfg).tracks(movie)
    check("the detected-but-unpredicted track is offered",
          any(t == 7 for t, _ in listed), str(listed))
    check("...and is offered with a count of zero, not hidden",
          dict(listed)[7] == 0, str(listed))
    check("the predicted track keeps its real count", dict(listed)[1] > 0, str(listed))
    tracks.to_parquet(path, index=False)


# --------------------------------------------------------------------------- #
# the older kind of file
# --------------------------------------------------------------------------- #
def test_a_windows_file_is_refused_with_a_sentence(app, root) -> None:
    """Only decoded bouts are reviewed; a file of fixed windows says so."""
    from test_predict import build_case

    from trackanno.gui.timeline import BoutReviewPanel, PredictionReader

    cfg, _ = build_case(root, "review-windows")
    rows = [{k: v for k, v in bout_row(0, 50, "swimming", 0.9).items()
             if k not in ("bout_id", "group")}]
    write_bouts(cfg, rows)

    reader = PredictionReader(cfg)
    check("a file with no bout_id is not read as bouts",
          "not decoded bouts" in reader.problem, reader.problem)

    panel = BoutReviewPanel(cfg)
    panel.resize(1000, 400)
    panel.show()
    app.processEvents()
    check("the panel shows the reason where the timeline would be",
          "not decoded bouts" in panel.timeline.note, panel.timeline.note)
    panel.close()
    app.processEvents()


def test_a_project_with_no_predictions_says_so_instead_of_raising(app, root) -> None:
    """Opened from a button, so a traceback out of the slot takes the window
    down; the sentence carries the same information without the crash."""
    from test_predict import build_case

    from trackanno.gui.timeline import BoutReviewPanel, PredictionReader

    cfg, _ = build_case(root, "review-none")
    reader = PredictionReader(cfg)
    check("the missing file is reported, not raised",
          "predictions" in reader.problem, reader.problem)
    check("...and the message says how to make it",
          "Label all clips with the model" in reader.problem, reader.problem)

    panel = BoutReviewPanel(cfg)
    panel.resize(800, 300)
    panel.show()
    app.processEvents()
    check("the panel shows the reason where the timeline would be",
          "Label all clips with the model" in panel.timeline.note,
          panel.timeline.note)
    check("and paints it rather than staying blank", ink(painted(panel.timeline)) > 0)
    panel.close()
    app.processEvents()


def test_the_timeline_takes_the_width_the_window_gives_it(app, root) -> None:
    """The frame axis is the thing being read, so it gets the space.

    Three separate ways this stopped happening, each of which put the lanes
    behind a horizontal scrollbar on a normal window: the timeline's
    minimumSizeHint returned its PREFERRED width and so overrode its own
    minimum; a control row was too wide for the panel to go under; and the
    splitter gave a shrinking window's remaining pixels to the film rather than
    to the lanes.
    """
    from test_predict import build_case

    from trackanno.gui.timeline import BoutReviewPanel

    cfg, _ = build_case(root, "review-resize", labels=2)
    write_bouts(cfg, first_movie_bouts(cfg))
    panel = BoutReviewPanel(cfg)
    panel.show()
    app.processEvents()
    scroll = panel.timeline.parentWidget().parentWidget()
    try:
        floor = panel.minimumSizeHint().width()
        check("the panel fits inside the window the hub opens at",
              floor <= 1180, f"needs {floor}px, the hub opens 1280 wide")
        check("the timeline may be narrower than it would like to be",
              panel.timeline.minimumSizeHint().width()
              <= panel.timeline.sizeHint().width(),
              f"{panel.timeline.minimumSizeHint().width()} vs "
              f"{panel.timeline.sizeHint().width()}")

        widths = []
        # Grown, shrunk and grown again: a splitter reapportions on every
        # resize, and the bug only showed on the way back up.
        for width in (1000, 1400, 2000, 1000, 2400):
            panel.resize(width, 700)
            app.processEvents()
            widths.append((panel.width(), panel.timeline.width()))
            check(f"no horizontal scrollbar at {width}px",
                  scroll.horizontalScrollBar().maximum() == 0,
                  str(widths))
        check("a wider window is a wider timeline, every time",
              all(b[1] > a[1] for a, b in zip(widths, widths[1:]) if b[0] > a[0]),
              str(widths))
        check("and the film does not take the growth",
              panel.footage.width() <= panel.footage.maximumWidth(),
              f"{panel.footage.width()} > {panel.footage.maximumWidth()}")
    finally:
        panel.close()
        app.processEvents()


# --------------------------------------------------------------------------- #
# the hook
# --------------------------------------------------------------------------- #
def test_reviewing_is_a_page_and_is_gated_on_the_file(app, root) -> None:
    """Review is a step of the workflow, not a window that floats over one.

    The page is gated on the FILE, not on the model: a run whose model has since
    been gated off still wrote rows somebody has to be able to look at, and
    gating the review on the same verdict that gates the run would hide exactly
    the predictions most worth arguing with.

    The panel is built on arrival rather than at startup, and it is torn down
    with the page.
    """
    from test_predict import build_case

    from trackanno.gui.hub import PAGES, Hub, ModelPage, ReviewPage

    cfg, _ = build_case(root, "review-page", labels=2)
    hub = Hub(str(cfg.path))
    hub.show()
    app.processEvents()
    model, review = hub.page(ModelPage), hub.page(ReviewPage)
    try:
        check("Review is a step in its own right, after Model",
              PAGES.index("Review") > PAGES.index("Model")
              and len(PAGES) == hub.sidebar.count()
              and len(PAGES) == len(hub.pages), str(PAGES))
        check("with no predictions the page says so rather than opening empty",
              "nothing has been predicted yet" in review.blocker())
        check("...and nothing was built to say it with", review.panel is None)

        write_bouts(cfg, first_movie_bouts(cfg))
        hub.reload_project()
        app.processEvents()
        check("the file alone unblocks the page", review.blocker() == "")
        check("but the panel is still not built until somebody goes there",
              review.panel is None)

        hub.go_to(PAGES.index("Review"))
        app.processEvents()
        check("going there shows the page in the window",
              hub.sidebar.currentRow() == PAGES.index("Review")
              and hub.stack.currentWidget() is review)
        check("arriving builds the panel", review.panel is not None
              and review.panel.isVisible())
        check("...on the predictions this project holds",
              review.panel.reader is not None
              and review.panel.reader.problem == "",
              getattr(review.panel.reader, "problem", "no reader"))
        check("no free-floating review window was created",
              not [w for w in app.topLevelWidgets()
                   if "Review" in w.windowTitle() and w is not hub])

        panel = review.panel
        hub.go_to(PAGES.index("Model"))
        app.processEvents()
        check("leaving the page lets go of the movie",
              panel.footage._movie is None)
        hub.go_to(PAGES.index("Review"))
        app.processEvents()
        check("coming back reuses the one panel rather than stacking them",
              review.panel is panel)
    finally:
        # shutdown() before close(): ModelPage keeps background threads
        # (keypoint reader, model probe) that close() does not
        # cancel, and a live QThread at interpreter exit kills the process after
        # the tests have already passed.
        model.shutdown()
        review.shutdown()
        hub.close()
        app.processEvents()
    check("closing the page takes the panel with it", review.panel is None)
