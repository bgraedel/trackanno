"""Looking at what a decode produced, without being lied to by the picture.

A bout is a claim about WHEN an animal changed what it was doing, and a timeline
is the only shape that claim can be checked in. That makes the drawing itself
load-bearing in a way a bar chart of counts is not: an edge drawn as a hairline
asserts a frame-accurate boundary, a fixed grid window drawn on a frame axis
looks exactly like a bout with suspiciously round numbers, and a bout sitting on
frames a person already labelled looks like the model agreeing with them.

So the checks here are about what the widget refuses to imply, not about whether
it renders. Four of them matter:

* the file decides what a row IS - ``bout_id``, never ``predict.decode``, which
  describes the run somebody is about to do;
* a boundary carries the stride's error band, and the stride is MEASURED off the
  window grid on disk rather than read out of the config;
* the bar height is the mean of the per-frame posterior, and the two are checked
  against each other on a real decode;
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


def bout_row(f_start, f_end, label, conf, human=0.0, covering=5.0, reason="",
             movie="m", track=1, group=GROUP, groups=(GROUP,)) -> dict:
    """One row of the bout schema :func:`trackanno.predict._bout_predictions`
    writes, including the columns of every OTHER group left empty - that shape
    is the thing the reader has to get right, so a hand-built row reproduces it
    rather than a convenient subset."""
    row = {
        "bout_id": f"{movie}__t{track:04d}__f{f_start:06d}-{f_end:06d}",
        "movie_id": movie, "track_id": track,
        "f_start": f_start, "f_end": f_end, "n_frames": f_end - f_start,
        "group": group, "n_covering_mean": covering, "skip_reason": reason,
        "human_overlap_frac": human, "human_labelled": human > 0.0,
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

    from trackanno.predict import PREDICTIONS_FILE

    cfg, _ = build_case(root, "review-synthetic")
    rows = [
        bout_row(0, 120, "swimming", 0.91, human=0.0),
        bout_row(120, 160, "immotile", 0.42, human=0.0),          # marginal
        bout_row(160, 180, "", np.nan, covering=0.0,
                 reason="no window over frames 160-180 could be featurised"),
        bout_row(180, 300, "swimming", 0.77, human=0.35, covering=1.4),
    ]
    pd.DataFrame(rows).to_parquet(
        cfg.paths.work_dir / PREDICTIONS_FILE, index=False)
    return cfg


def test_the_reader_maps_every_column_a_bout_carries(synthetic) -> None:
    """One row per (group, bout), and each of its columns means something
    different: getting one of them wrong produces a plausible picture."""
    from trackanno.gui.timeline import PredictionReader

    reader = PredictionReader(synthetic)
    check("the file opens", reader.problem == "", reader.problem)
    check("a bout_id column makes these bouts", reader.bouts)
    check("...and the noun comes from render.prediction_unit, not from here",
          reader.unit == "predicted bouts", reader.unit)
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

    check("human_overlap_frac survives as a fraction, not as a flag",
          abs(lane[3].human_frac - 0.35) < 1e-9, str(lane[3].human_frac))
    check("and so does how many windows spoke for the frames",
          abs(lane[3].n_covering - 1.4) < 1e-9, str(lane[3].n_covering))


def test_the_caption_says_what_the_picture_cannot(synthetic) -> None:
    """Each sentence is a fact about THESE rows. A caption that appeared
    whatever the data said would be decoration."""
    from trackanno.gui.timeline import PredictionReader, caption_lines

    reader = PredictionReader(synthetic)
    track = reader.track("m", 1)
    lines = caption_lines(track, reader.resolution, min_confidence=0.55)
    joined = " ".join(lines)

    check("the caption names what a bar is",
          "decoded bout" in joined and "mean posterior" in joined, joined)
    check("it warns that hand-labelled frames are not evidence about the model",
          "labelled by hand" in joined and "not" in joined, joined)
    check("...and counts the spans that overlap them", "1 span(s) are hatched" in joined,
          joined)
    check("it says which spans rest on almost no windows",
          "fewer than two windows per frame" in joined, joined)
    check("it names the confidence bar the project already applies",
          "suggest.min_confidence (0.55)" in joined, joined)
    check("...and says the per-frame posterior is not on disk for this run",
          "save_posteriors was off" in joined, joined)

    empty = reader.track("m", 99)
    check("a track with nothing in it gets no caption at all rather than a "
          "sentence about bouts it does not have",
          caption_lines(empty, reader.resolution) == [])


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
    check("...which still knows what kind of file it came from",
          track.bouts and track.unit == "predicted bouts")

    panel = BoutReviewPanel(synthetic)
    panel.resize(900, 320)
    panel.show()
    app.processEvents()
    panel.timeline.set_track(track, reader.resolution)
    panel.timeline.resize(900, 200)
    app.processEvents()
    check("the empty timeline paints without raising", ink(painted(panel.timeline)) > 0)
    check("and the words on it say the outcome is normal",
          "normal outcome" in panel.timeline._nothing_here(),
          panel.timeline._nothing_here())
    panel.close()
    app.processEvents()


# --------------------------------------------------------------------------- #
# a real decode
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def decoded(root):
    """The planted 300-frame track, decoded at stride 10, posterior kept.

    The same case :mod:`test_predict` measures boundary error on, so the numbers
    the timeline reports can be checked against the ones the decode produced
    rather than against themselves.
    """
    from test_predict import planted_case

    from trackanno.predict import predict_all

    cfg, movie = planted_case(root, "review-decoded", 10,
                              **{"predict.save_posteriors": True})
    predict_all(cfg)
    return cfg, movie


def test_a_real_decode_reaches_the_timeline_intact(decoded) -> None:
    """The bar height and the line over it are the same number twice.

    ``conf`` is the MEAN of the per-frame posterior of the bout's own class, and
    the whole reason the trace is drawn is that a mean cannot say whether it came
    from a flat 0.8 or from 0.99/0.4/0.99. If the two are ever computed from
    different columns the picture stops being an argument about the same bout.
    """
    from trackanno.gui.timeline import PredictionReader

    cfg, movie = decoded
    reader = PredictionReader(cfg)
    check("the decoded file opens", reader.problem == "", reader.problem)
    check("and it is read as bouts", reader.bouts and reader.unit == "predicted bouts")

    track = reader.track(movie, 1)
    lane = track.lanes[GROUP]
    check("the bouts tile the track's whole span",
          lane[0].f_start == track.f_lo and lane[-1].f_end == track.f_hi
          and all(a.f_end == b.f_start for a, b in zip(lane, lane[1:])),
          str([(s.f_start, s.f_end, s.label) for s in lane]))
    check("the planted transition shows up as more than one bout",
          len({s.label for s in lane if s.label}) == 2,
          str([s.label for s in lane]))

    check("the run saved a posterior and the reader found it",
          GROUP in track.trace, str(list(track.trace)))
    trace = track.trace[GROUP]
    worst = 0.0
    for segment in lane:
        if not segment.label:
            continue
        block = trace[segment.f_start - track.f_lo: segment.f_end - track.f_lo]
        worst = max(worst, abs(float(np.nanmean(block)) - segment.conf))
    check("every bar's height is the mean of the trace drawn over it",
          worst < 1e-9, f"worst disagreement {worst:.3g}")


def test_the_boundary_band_is_measured_off_the_grid_not_read_off_the_config(
    decoded,
) -> None:
    """A crisp edge would claim a precision the method does not have.

    The README's measured error against a planted transition is ``stride/4`` on
    average and ``stride/2`` at worst, and the stride comes from the window grid
    on disk. Reading ``predict.window_stride`` instead would size the band from
    a setting anybody can edit between the run and this window opening - and
    unlike a wrong caption, a wrong band is read as "the boundary is somewhere
    in here".
    """
    from trackanno.config import load_config
    from trackanno.gui.timeline import PredictionReader, measure_stride
    from trackanno.project import write_config

    cfg, _ = decoded
    resolution = measure_stride(cfg)
    check("the stride is measured from the window grid", resolution.known)
    check("...and it is the stride the run actually used",
          resolution.stride == 10, str(resolution.stride))
    check("the worst-case band is half a stride",
          resolution.worst_error == 5.0, str(resolution.worst_error))
    check("and the average error a quarter of one",
          resolution.mean_error == 2.5, str(resolution.mean_error))
    check("the sentence quotes both, and where the stride came from",
          "10-frame window stride" in resolution.sentence()
          and "2.5" in resolution.sentence()
          and "dense_clip_features.parquet" in resolution.sentence(),
          resolution.sentence())

    # The config now says something else. The band must not move: it describes
    # the run that wrote these rows, not the run this config would produce.
    write_config(cfg.path, {"predict.window_stride": 25})
    edited = load_config(cfg.path)
    check("editing predict.window_stride does not move the band",
          measure_stride(edited).stride == 10, str(measure_stride(edited).stride))
    check("...and the reader agrees",
          PredictionReader(edited).resolution.stride == 10)
    write_config(cfg.path, {"predict.window_stride": 10})


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
    check("nothing warns that these are windows",
          "fixed windows" not in panel.headline.text(), panel.headline.text())

    panel.timeline.resize(940, 200)
    app.processEvents()
    check("the timeline actually paints", ink(painted(panel.timeline)) > 500,
          str(ink(painted(panel.timeline))))
    check("every drawn span is hit-testable, so the detail has somewhere to live",
          len(panel.timeline._hits) == panel.timeline.track.n_segments,
          f"{len(panel.timeline._hits)} rects for "
          f"{panel.timeline.track.n_segments} segments")

    group, segment = panel.timeline._hits[0][1], panel.timeline._hits[0][2]
    tip = panel.timeline._describe(group, segment)
    check("the tooltip says confidence is a mean and not a peak",
          "MEAN posterior" in tip and "not its peak" in tip, tip)
    check("...and repeats the resolution limit where the edge is",
          "10-frame" in tip and "at worst" in tip, tip)
    panel.close()
    app.processEvents()


def test_a_track_the_predictions_never_mention_is_still_selectable(decoded) -> None:
    """Built from the union of the predictions and ``tracks.parquet``.

    A combo filled only from the predictions makes the empty state unreachable,
    and then the reviewer cannot ask the one question a silent track raises.
    """
    from trackanno.clips import TRACKS_FILE
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
def test_an_old_style_windows_file_is_named_as_such(app, root) -> None:
    """Fixed windows off a grid are not bouts and must not be drawn as if.

    Their edges are multiples of the window length the predictor chose - a fact
    about the grid, not about the animal - so they are drawn (still worth
    seeing) but captioned, and no boundary band is placed around them: there is
    no inferred boundary there to be uncertain about.
    """
    from test_predict import build_case

    from trackanno.gui.timeline import BoutReviewPanel, PredictionReader, caption_lines
    from trackanno.predict import predict_all

    cfg, manifest = build_case(root, "review-windows", labels=2)
    predict_all(cfg)

    reader = PredictionReader(cfg)
    check("a file with no bout_id is not read as bouts", not reader.bouts)
    check("...and is counted in the noun render.prediction_unit gives it",
          reader.unit == "predicted windows", reader.unit)
    check("no boundary band is offered for a grid",
          not reader.resolution.known, str(reader.resolution.stride))

    movie = reader.movies[0]
    track_id = next(t for t, n in reader.tracks(movie) if n)
    track = reader.track(movie, track_id)
    lines = " ".join(caption_lines(track, reader.resolution))
    check("the caption says these are windows, not bouts",
          "NOT decoded bouts" in lines, lines)
    check("...and says how to get bouts instead", "predict.decode" in lines, lines)
    check("a window's hand-labelled flag reads through as a fraction",
          all(f in (0.0, 1.0) for f in
              [s.human_frac for lane in track.lanes.values() for s in lane]),
          str([s.human_frac for lane in track.lanes.values() for s in lane]))

    panel = BoutReviewPanel(cfg)
    panel.resize(1000, 400)
    panel.show()
    app.processEvents()
    check("the panel warns before anything is read off the picture",
          "fixed windows" in panel.headline.text(), panel.headline.text())
    panel.timeline.resize(940, 200)
    app.processEvents()
    check("and it still draws them - a windows file is worth looking at",
          ink(painted(panel.timeline)) > 500, str(ink(painted(panel.timeline))))
    panel.close()
    app.processEvents()
    check("the case really was a manifest run", len(manifest) > 0)


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


# --------------------------------------------------------------------------- #
# the hook
# --------------------------------------------------------------------------- #
def test_the_model_page_opens_the_review_window(app, root) -> None:
    """The button is gated on the FILE, not on the model.

    A run whose model has since been gated off still wrote rows somebody has to
    be able to look at - and gating the review on the same verdict that gates
    the run would hide exactly the predictions most worth arguing with.
    """
    from test_predict import build_case

    from trackanno.gui.hub import Hub, ModelPage
    from trackanno.predict import predict_all

    cfg, _ = build_case(root, "review-page", labels=2)
    hub = Hub(str(cfg.path))
    hub.show()
    app.processEvents()
    page = hub.page(ModelPage)
    try:
        check("with no predictions the review button is dead",
              not page.review_button.isEnabled())

        predict_all(cfg)
        page.on_project()
        app.processEvents()
        check("the file alone enables it", page.review_button.isEnabled())

        page.review_button.click()
        app.processEvents()
        check("a review window opens", page._review is not None
              and page._review.isVisible())
        check("...on the predictions this project holds",
              page._review.panel.reader is not None
              and page._review.panel.reader.problem == "",
              getattr(page._review.panel.reader, "problem", "no reader"))

        page.review_button.click()
        app.processEvents()
        check("pressing it again reuses the one window rather than stacking them",
              len([w for w in app.topLevelWidgets()
                   if w.__class__.__name__ == "BoutReviewDialog"]) == 1)
    finally:
        hub.close()
        app.processEvents()
    check("closing the page takes the review window with it", page._review is None)
