"""Watching the animal at the frame the boundary is drawn at.

A bout is a claim about WHEN behaviour changed, and the review window used to
make that claim checkable only against itself: a picture of the spans, and no
way to see the frames under them. So the checks here are about the join between
the drawing and the footage, not about whether a video plays:

* the pixels are the CACHE's, not a second decode of the movie - the same
  memmap, crop and viewer the annotator labelled in, because a boundary judged
  against a differently rendered frame is judged against something nobody
  labelled;
* a bout is not a cached clip and never lines up with one, so the cache is
  addressed frame by frame and playback carries from the clip that ends into the
  one that holds the next frame;
* a frame the cache never covered says so and is cut from the movie instead -
  at the same crop, which is checked here against the cached frame beside it;
* a project with no cache and no movie loses the video and keeps the timeline,
  with a sentence, and never a black rectangle;
* the window lets go of the cache when it closes, because a mapped clips.dat on
  Windows is the next detection run's failure.

The fixture's PIXELS are copied into each case rather than opened where they
live: the generated project is shared by every suite, and a clips.dat left
mapped fails whichever suite re-runs detection next.
"""

from __future__ import annotations

import json
import shutil
import time

import numpy as np
import pandas as pd
import pytest
from harness import check
from test_bout_review import bout_row, ink, painted

GROUP = "behaviour"
TRACK = 1
BOUTS = [
    (0, 40, "swimming", 0.90),
    (40, 90, "immotile", 0.45),
    (90, 120, "swimming", 0.80),
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def filmed_project(root, name: str):
    """A case project carrying the fixture's clip cache as well as its tables.

    ``build_case`` copies the tables only, because predicting reads features.
    This window reads pixels.
    """
    from fixtures import make_project
    from test_predict import FIXTURES, build_case

    from trackanno.clips import BOXES_FILE, CACHE_FILE, ORIGINS_FILE
    from trackanno.config import load_config
    from trackanno.predict import PREDICTIONS_FILE

    cfg, manifest = build_case(root, name)
    fixture = load_config(make_project(FIXTURES))
    for movie_dir in sorted(p for p in fixture.paths.work_dir.iterdir() if p.is_dir()):
        out = cfg.paths.work_dir / movie_dir.name
        for pixels in (CACHE_FILE, BOXES_FILE, ORIGINS_FILE):
            if (movie_dir / pixels).exists():
                shutil.copy(movie_dir / pixels, out / pixels)

    movie = str(manifest["movie_id"].iat[0])
    pd.DataFrame([
        bout_row(f_start, f_end, label, conf, movie=movie, track=TRACK)
        for f_start, f_end, label, conf in BOUTS
    ]).to_parquet(cfg.paths.work_dir / PREDICTIONS_FILE, index=False)
    return cfg, movie


def select(panel, movie_id, track_id) -> None:
    panel.movie.setCurrentText(str(movie_id))
    for i in range(panel.track.count()):
        if panel.track.itemData(i) == int(track_id):
            panel.track.setCurrentIndex(i)
            return
    raise AssertionError(f"track {track_id} is not offered for {movie_id}")


def bases(array):
    """An array and everything it is a view on, down to the buffer's owner."""
    while array is not None:
        yield array
        array = getattr(array, "base", None)


def lane_y(timeline, kind: str, group: str = GROUP) -> float:
    """The middle of one drawn lane, off the plan the painter used."""
    for drawn, name, y in timeline._lane_plan():
        if drawn == kind and name == group:
            return y + (timeline.RECORD_LANE if kind == "record" else timeline.LANE) / 2
    raise AssertionError(f"no {kind} lane for {group}")


def click(timeline, frame: int, y: float | None = None) -> None:
    """Press the timeline where ``frame`` is drawn, through the widget's own
    geometry rather than by emitting its signal."""
    from qtpy.QtCore import QPointF, Qt
    from qtpy.QtGui import QMouseEvent

    gutter, scale = timeline._geom
    at = QPointF(timeline._x(frame, gutter, scale),
                 float(timeline.height() - 4) if y is None else float(y))
    timeline.mousePressEvent(QMouseEvent(
        QMouseEvent.Type.MouseButtonPress, at, Qt.LeftButton, Qt.LeftButton,
        Qt.NoModifier))
    timeline.mouseReleaseEvent(QMouseEvent(
        QMouseEvent.Type.MouseButtonRelease, at, Qt.LeftButton, Qt.NoButton,
        Qt.NoModifier))


def cached_frame(cfg, movie_id: str, track_id: int, frame: int) -> np.ndarray:
    """The cache's own pixels for one frame, read without the code under test.

    The store is opened and closed here rather than borrowed from the widget,
    so "what is on screen" is compared against the file and not against the
    same object that put it there.
    """
    from trackanno.clips import ClipStore

    store = ClipStore(cfg.paths.work_dir)
    try:
        rows = store.manifest
        mine = rows[
            (rows["movie_id"].astype(str) == str(movie_id))
            & (rows["track_id"].astype(int) == int(track_id))
            & (rows["f_start"] <= frame) & (frame < rows["f_end"])
        ]
        if mine.empty:
            return None
        row = mine.iloc[0]
        return np.array(store.clip(int(mine.index[0]))[frame - int(row["f_start"])])
    finally:
        store.close()


@pytest.fixture(scope="module")
def filmed(root):
    return filmed_project(root, "footage")


@pytest.fixture
def panel(app, filmed):
    """A panel per test, on the filmed track, released afterwards."""
    from trackanno.gui.timeline import BoutReviewPanel

    cfg, movie = filmed
    widget = BoutReviewPanel(cfg)
    widget.resize(1200, 620)
    widget.show()
    app.processEvents()
    select(widget, movie, TRACK)
    widget.timeline.resize(820, 260)
    painted(widget.timeline)          # a paint is what fixes the frame axis
    yield widget
    widget.close()
    app.processEvents()


# --------------------------------------------------------------------------- #
# what the cache can serve
# --------------------------------------------------------------------------- #
class Manifest:
    """Just enough ClipStore for the lookup: it reads one table."""

    def __init__(self, rows):
        self.manifest = pd.DataFrame(rows)


def test_a_bout_is_served_from_the_clips_that_overlap_it() -> None:
    """The cache is cut on the annotation grid and a bout is not.

    A bout runs between two inferred boundaries and can start and end anywhere,
    so there is no cached clip to hand it. Serving it frame by frame off
    whichever clip covers each frame is what makes the cache usable at all here;
    the alternative is decoding the movie, which is slower, needs the recording
    to still be where it was, and renders differently from the window an
    annotator labelled in.
    """
    from trackanno.gui.timeline import TrackClips

    store = Manifest([
        {"movie_id": "m", "track_id": 1, "f_start": 0, "f_end": 50},
        {"movie_id": "m", "track_id": 1, "f_start": 25, "f_end": 75},
        {"movie_id": "m", "track_id": 1, "f_start": 100, "f_end": 150},
        {"movie_id": "m", "track_id": 2, "f_start": 0, "f_end": 50},
    ])
    clips = TrackClips(store, "m", 1)
    check("only this track's clips are offered", len(clips) == 3, str(len(clips)))

    check("a frame inside one clip resolves to it, and to its offset in it",
          clips.at(10) == (0, 10), str(clips.at(10)))
    check("a bout's frames come from different clips as it runs",
          clips.at(60) == (1, 35), str(clips.at(60)))
    check("where two clips cover a frame, the one centred nearest wins - so the "
          "crop changes midway between windows, not on every frame",
          clips.at(30) == (0, 30) and clips.at(40) == (1, 15),
          f"{clips.at(30)} {clips.at(40)}")

    check("a frame between clips is a hole, reported rather than clamped to the "
          "nearest clip's pixels", clips.at(80) is None, str(clips.at(80)))
    check("...and so is one past the last clip", clips.at(200) is None)
    check("the ranges it can serve are merged, overlap and all",
          clips.runs() == [(0, 75), (100, 150)], str(clips.runs()))
    check("the clips are ordered by frame, which is the line the prefetcher "
          "reads ahead along", clips.order(1) == ([0, 1, 2], 1),
          str(clips.order(1)))


# --------------------------------------------------------------------------- #
# the join
# --------------------------------------------------------------------------- #
def test_the_video_shows_the_frame_the_timeline_marker_is_on(app, panel, filmed) -> None:
    """Scrubbing moves the film, and the film moves the marker.

    Both directions, because either one alone is a different tool: a timeline
    that drives a video is a player, and a video that drives a timeline is a
    progress bar. Reviewing a boundary needs to go both ways - point at the edge
    to see it, then play across it to decide.
    """
    from trackanno.gui.viewer import SimpleViewer

    cfg, movie = filmed
    footage, timeline = panel.footage, panel.timeline

    check("the frames come from the clip cache", footage.showing == "cache",
          footage.source.text())
    check("they are shown through the annotation window's own player, so the "
          "overlay and the scaling cannot drift apart from it",
          isinstance(footage.viewer, SimpleViewer), type(footage.viewer).__name__)
    check("...and the array behind it is a view on the cache's memory map, "
          "which is to say nothing was decoded to get here",
          any(isinstance(a, np.memmap) for a in bases(footage.viewer._array)))
    check("...and the window opens on the track's first predicted frame",
          footage.position == 0, str(footage.position))
    check("the note says where the pixels came from",
          "clip cache" in footage.source.text(), footage.source.text())

    click(timeline, 72)
    app.processEvents()
    check("clicking the timeline takes the video to that frame",
          footage.position == 72, str(footage.position))
    check("...and the marker is on it too", timeline.position == 72,
          str(timeline.position))
    check("the pixels on screen are that frame of the cache, not a re-decode "
          "of it",
          np.array_equal(footage.frame_image(), cached_frame(cfg, movie, TRACK, 72)))

    click(timeline, 12)
    app.processEvents()
    check("scrubbing back into an earlier clip works the same way",
          footage.position == 12
          and np.array_equal(footage.frame_image(),
                             cached_frame(cfg, movie, TRACK, 12)),
          str(footage.position))

    footage.viewer.play()
    for _ in range(3):
        footage.viewer._advance()
    app.processEvents()
    check("playing the video walks the marker along the timeline",
          timeline.position == 15 and footage.position == 15,
          f"{timeline.position} {footage.position}")
    footage.viewer.pause()


def test_playback_carries_from_one_cached_clip_into_the_next(app, panel) -> None:
    """A clip's last frame is not the end of the track.

    The cache is cut into fixed windows and a behavioural boundary lands
    wherever it lands, including a frame or two from a window edge - which is
    exactly the boundary worth watching across. Playback that stopped dead at
    every window edge would be unable to show the one thing this window is for.
    """
    footage = panel.footage
    footage.show_frame(48)
    first = footage._loaded
    check("frame 48 comes from the clip that ends at 50",
          footage.showing == "cache" and footage.position == 48,
          f"{footage.showing} {footage.position}")

    footage.viewer.play()
    footage.viewer._advance()             # onto the clip's last frame
    app.processEvents()                   # the hand-over is queued, not re-entrant
    check("playing past a clip's last frame carries on into the next frame of "
          "the track", footage.position == 50, str(footage.position))
    check("...out of a different clip", footage._loaded != first,
          f"{first} -> {footage._loaded}")
    check("...and it is still playing", footage.viewer.is_playing)
    check("the marker went with it", panel.timeline.position == 50,
          str(panel.timeline.position))
    footage.viewer.pause()


def test_clicking_a_bout_takes_you_to_it_and_still_selects_it(app, panel) -> None:
    """One press, both jobs, and neither may cost the other.

    The correction controls are filled by clicking a span on the record lane.
    That gesture now also moves the video, so the risk is a scrub that quietly
    stops the click from selecting - the frame boxes would keep whatever they
    had and the next 'Set class' would assert over the wrong range.
    """
    footage, timeline = panel.footage, panel.timeline
    # On the record lane, inside the middle bout, and not near either boundary:
    # a press within GRAB_PX of one of those is a drag, which is a different
    # gesture with its own test.
    click(timeline, 60, lane_y(timeline, "record"))
    app.processEvents()

    check("the span under the mouse is selected",
          timeline.selected == (GROUP, 40, 90), str(timeline.selected))
    check("...and its frames are in the correction boxes",
          (panel.from_frame.value(), panel.to_frame.value()) == (40, 90),
          f"{panel.from_frame.value()}-{panel.to_frame.value()}")
    check("...and the video went to the frame that was pointed at",
          footage.position == 60, str(footage.position))


def test_a_correction_still_round_trips_with_the_video_wired_in(app, panel, filmed) -> None:
    """The video is a way of looking, never a way of writing.

    Every edit still goes through the corrections file and comes back out of it
    unchanged, and nothing about a frame being on screen gets into what was
    asserted about it.
    """
    from trackanno.corrections import corrections_path
    from trackanno.gui.timeline import BoutReviewPanel

    cfg, movie = filmed
    corrections_path(cfg).unlink(missing_ok=True)
    try:
        click(panel.timeline, 60, lane_y(panel.timeline, "record"))
        app.processEvents()
        message = panel.assert_class(40, 70, "swimming", GROUP)
        check("the edit was accepted", message.startswith("saved"), message)

        fresh = BoutReviewPanel(cfg)
        try:
            select(fresh, movie, TRACK)
            spans = fresh._track.partitions[GROUP].corrected_spans()
            check("it is on disk, and comes back as a correction",
                  [(s.f_start, s.f_end, s.label) for s in spans]
                  == [(40, 70, "swimming")], str(spans))
            check("...and the model's own bouts are still beside it, untouched",
                  [(s.f_start, s.f_end) for s in fresh._track.lanes[GROUP]]
                  == [(a, b) for a, b, _, _ in BOUTS],
                  str(fresh._track.lanes[GROUP]))
            check("the reloaded panel has its own video on the same track",
                  fresh.footage.showing == "cache", fresh.footage.source.text())
        finally:
            fresh.close()
            app.processEvents()
    finally:
        corrections_path(cfg).unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# when there are no pixels
# --------------------------------------------------------------------------- #
def test_a_project_with_no_clip_cache_keeps_the_timeline_and_says_why(app, root) -> None:
    """The state a project sits in between predicting and preparing.

    Losing the video must cost the video only. A black rectangle would read as a
    decoder that failed, and a traceback out of the slot takes the window with
    it - so the picture goes away and a sentence takes its place.
    """
    from test_predict import build_case

    from trackanno.gui.timeline import BoutReviewPanel
    from trackanno.predict import PREDICTIONS_FILE

    cfg, manifest = build_case(root, "footage-uncached")
    movie = str(manifest["movie_id"].iat[0])
    pd.DataFrame([
        bout_row(a, b, label, conf, movie=movie, track=TRACK)
        for a, b, label, conf in BOUTS
    ]).to_parquet(cfg.paths.work_dir / PREDICTIONS_FILE, index=False)

    widget = BoutReviewPanel(cfg)
    widget.resize(1100, 600)
    widget.show()
    app.processEvents()
    select(widget, movie, TRACK)
    widget.timeline.resize(820, 260)
    app.processEvents()
    try:
        check("the timeline still draws the predictions",
              ink(painted(widget.timeline)) > 500,
              str(ink(painted(widget.timeline))))
        note = widget.footage.source.text()
        check("nothing is on screen where the video would be",
              not widget.footage.viewer.isVisible(), note)
        check("...and it says there is no video", note.startswith("No video"), note)
        check("...and why: this project has no cache to read", "cache" in note, note)
        check("...and that the timeline is unaffected", "timeline still" in note, note)
        check("no video did not cost the corrections either",
              widget._group_problem() == "", widget._group_problem())
    finally:
        widget.close()
        app.processEvents()


def test_a_frame_no_clip_covers_names_the_gap_and_where_it_looked(app, panel) -> None:
    """The cache stops where the annotation grid stopped, not where the bouts do.

    The last window of a movie is dropped when too few of its frames hold a
    detection, and a dense decode predicts over frames no window was ever cut
    for. Those frames are the fallback's whole reason for existing, and when the
    fallback cannot run either, both halves of that are said.
    """
    footage = panel.footage
    check("the cache covers the frames it was cut over and no more",
          footage._clips.runs() == [(0, 100)], str(footage._clips.runs()))

    footage.show_frame(110)
    # The movie is opened off the UI thread, so the answer arrives as a signal
    # rather than as a return value; spin the loop until it does.
    for _ in range(600):
        app.processEvents()
        if "could not be read" in footage.source.text():
            break
        time.sleep(0.005)
    note = footage.source.text()
    check("a frame outside every cached clip is named as such",
          "outside every cached clip" in note, note)
    check("...with the range the cache does hold", "0-100" in note, note)
    check("...and the movie it fell back to is named as having failed",
          "could not be read" in note, note)
    check("nothing black was left on screen instead",
          not footage.viewer.isVisible(), note)

    footage.show_frame(60)
    check("scrubbing back into the cached range brings the video back",
          footage.showing == "cache" and footage.viewer.isVisible(),
          footage.source.text())


def test_a_frame_cut_from_the_movie_is_the_frame_the_cache_would_have_held(
    app, root,
) -> None:
    """The fallback is not a second renderer, and this is how that is known.

    A crop chosen by another rule, or normalised by another mapping, would put
    the animal at a different size and a different grey from the cached frames
    on either side of it - and the reviewer would read that difference as
    something in the recording. So the fallback takes the crop off the manifest
    and the intensity mapping off norm.json, and on a frame the cache also holds
    the two agree pixel for pixel.
    """
    import tifffile
    from test_pipeline import SyntheticMovie

    from trackanno.clips import NORM_FILE
    from trackanno.gui.timeline import MovieFrames
    from trackanno.movies import Normalizer

    cfg, movie_id = filmed_project(root, "footage-fallback")
    source = SyntheticMovie()
    cfg.paths.video_dir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(cfg.paths.video_dir / f"{movie_id}.tif"),
                     np.stack([source.frame(i) for i in range(source.n_frames)]))
    # What prepare writes beside the cache. The fixture's cache was cut with
    # this mapping, so the fallback must find it rather than measure its own.
    Normalizer(0.0, 255.0).save(cfg.movie_dir(movie_id) / NORM_FILE)

    header = json.loads((cfg.movie_dir(movie_id) / "cache.json").read_text())
    manifest = pd.read_parquet(cfg.movie_dir(movie_id) / "clips.parquet")
    mine = manifest[manifest["track_id"] == TRACK]
    side = int(mine["crop_w"].max())
    check("the cache cut this track at a crop wider than it stores",
          side > int(header["size"]), f"{side} vs {header['size']}")

    frames = MovieFrames(cfg, movie_id, TRACK, int(header["size"]),
                         int(cfg.clips.frames), side)
    try:
        cut, marks = frames._cut(50)
        check("the movie gives back a chunk the size of a cached clip",
              cut.shape == (cfg.clips.frames, header["size"], header["size"]),
              str(cut.shape))
        cached = cached_frame(cfg, movie_id, TRACK, 60)
        check("and frame 60 cut from the movie is frame 60 of the cache, "
              "pixel for pixel", np.array_equal(cut[10], cached),
              f"worst difference "
              f"{int(np.abs(cut[10].astype(int) - cached.astype(int)).max())}")
        check("the target's box comes with it, so the fallback still says which "
              "animal is the track",
              len(marks) == len(cut) and bool(marks["is_target"].all()))
    finally:
        frames.stop()


# --------------------------------------------------------------------------- #
# letting go
# --------------------------------------------------------------------------- #
def test_hiding_the_window_lets_go_and_showing_it_gets_the_video_back(
    app, filmed,
) -> None:
    """The Model page keeps one review dialog for the whole session.

    Holding the cache mapped for that long would block every detection run in
    between, so the cache is dropped whenever the window goes away - which means
    re-showing it has to put the footage back. A window that came back with a
    sentence where the video used to be would look exactly like a project that
    had lost its cache.
    """
    from trackanno.gui.timeline import BoutReviewDialog

    cfg, movie = filmed
    dialog = BoutReviewDialog(cfg)
    dialog.show()
    app.processEvents()
    panel = dialog.panel
    try:
        select(panel, movie, TRACK)
        panel.timeline.resize(820, 260)
        painted(panel.timeline)
        click(panel.timeline, 72)
        app.processEvents()
        check("the dialog is showing the cache", panel.footage.showing == "cache"
              and panel.footage.position == 72, panel.footage.source.text())

        dialog.hide()
        app.processEvents()
        check("hiding it unmaps the cache, so detection can run meanwhile",
              panel.footage._store is None)

        dialog.show()
        app.processEvents()
        check("showing it again brings the video back",
              panel.footage.showing == "cache", panel.footage.source.text())
        check("...to the frame it was left on", panel.footage.position == 72,
              str(panel.footage.position))
    finally:
        dialog.close()
        app.processEvents()


def test_closing_the_window_unmaps_the_clip_cache(app, root) -> None:
    """A mapped clips.dat is the next detection run's failure.

    Windows will not let a memory-mapped file be overwritten, so a review window
    left open - or one whose store was released only when the collector got
    round to it - turns 'run detection again' into an OSError about an invalid
    argument, days later and nowhere near this code.
    """
    from trackanno.clips import CACHE_FILE
    from trackanno.gui.timeline import BoutReviewPanel

    cfg, movie = filmed_project(root, "footage-release")
    widget = BoutReviewPanel(cfg)
    widget.show()
    app.processEvents()
    select(widget, movie, TRACK)
    check("the window really did map the cache",
          widget.footage._store is not None and widget.footage.showing == "cache",
          widget.footage.source.text())

    widget.close()
    app.processEvents()
    check("closing it drops the store", widget.footage._store is None)

    path = cfg.movie_dir(movie) / CACHE_FILE
    header = json.loads((cfg.movie_dir(movie) / "cache.json").read_text())
    shape = (header["n_clips"], header["frames"], header["size"], header["size"])
    remade = np.memmap(path, dtype=np.uint8, mode="w+", shape=shape)
    remade.flush()
    del remade
    check("...and the cache can be rewritten, which is what detection does",
          path.stat().st_size == int(np.prod(shape)), str(path.stat().st_size))
