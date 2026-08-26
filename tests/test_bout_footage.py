"""Watching the animal at the frame the boundary is drawn at.

A bout is a claim about WHEN behaviour changed, and the review window used to
make that claim checkable only against itself: a picture of the spans, and no
way to see the frames under them. The frames used to come from the clip cache
wherever it reached and from the movie everywhere else, which put a step in the
picture at every window edge - the cache is sized per movie and centred per
annotation window, and a bout is neither. So every review frame is now cut from
the movie, and the checks here are about that cut:

* ONE crop for the whole session, so nothing changes size when the reviewer
  changes track or movie - the complaint that started this;
* centred on the track's own boxes every frame, so the animal neither walks out
  of the picture over the hundreds of frames a bout runs for nor steps sideways
  where one read of the movie ends and the next begins;
* the pixels are the movie's own, through the intensity mapping ``prepare``
  wrote, so the review is not a second renderer;
* the marker survives the cut: the reticle, the neighbours and - on a pose
  project - the skeleton, all read off ``tracks.parquet`` on the decode thread;
* nothing is decoded on the UI thread, and the chunks either side of the
  playhead are read ahead so a scrub is not a series of stalls;
* a project whose movies are not where the config says loses the video and keeps
  the timeline, with a sentence, and never a black rectangle.

Every project here carries its own copy of the movie and its own work directory:
the generated fixture is shared by every suite, and a suite that wrote into it
would break whichever suite ran next.
"""

from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path

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
SECOND = "synthetic2"


# --------------------------------------------------------------------------- #
# projects with a movie in them
# --------------------------------------------------------------------------- #
def _film(cfg, movie_id: str) -> None:
    """Write the synthetic movie and the intensity curve prepare writes beside
    it, so the review cuts real pixels through the mapping the cache was cut
    with."""
    import tifffile
    from test_pipeline import SyntheticMovie

    from trackanno.annotate.clips import NORM_FILE
    from trackanno.video.movies import Normalizer

    source = SyntheticMovie()
    cfg.paths.video_dir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        str(cfg.paths.video_dir / f"{movie_id}.tif"),
        np.stack([source.frame(i) for i in range(source.n_frames)]),
    )
    Normalizer(0.0, 255.0).save(cfg.movie_dir(movie_id) / NORM_FILE)


def _predict(cfg, movies, tracks=(1, 2)) -> None:
    from trackanno.infer.predict import PREDICTIONS_FILE

    pd.DataFrame([
        bout_row(f_start, f_end, label, conf, movie=movie, track=track)
        for movie in movies for track in tracks
        for f_start, f_end, label, conf in BOUTS
    ]).to_parquet(cfg.paths.work_dir / PREDICTIONS_FILE, index=False)


def _own_dirs(cfg):
    """Point the project at a video directory of its own.

    ``build_case`` leaves ``paths.video_dir`` on the shared fixture, and the
    movie written here must not land there.
    """
    from trackanno.config import load_config
    from trackanno.project import write_config

    write_config(cfg.path, {
        "paths.video_dir": str(cfg.path.parent / "videos").replace("\\", "/"),
    })
    return load_config(cfg.path)


def filmed_project(root, name: str, second_movie: bool = False):
    """A case project with the movie itself, and no cached pixels at all.

    ``build_case`` copies the tables only. The review reads the movie, so the
    clip cache's ``clips.dat`` is deliberately absent from every project here:
    if any frame still came out of it, none of these tests would pass.
    """
    from test_predict import build_case

    cfg, manifest = build_case(root, name)
    cfg = _own_dirs(cfg)
    movie = str(manifest["movie_id"].iat[0])
    _film(cfg, movie)
    movies = [movie]

    if second_movie:
        # A second movie whose cache was cut at a wider crop, which is what
        # `clips.crop_size: auto` does per movie and what used to change the
        # size of the picture when the reviewer changed movie.
        work = cfg.paths.work_dir
        shutil.copytree(work / movie, work / SECOND)
        rows = pd.read_parquet(work / SECOND / "clips.parquet")
        rows["movie_id"] = SECOND
        rows["crop_w"] = rows["crop_w"].astype(int) + 64
        rows["crop_h"] = rows["crop_w"]
        rows.to_parquet(work / SECOND / "clips.parquet", index=False)
        _film(cfg, SECOND)
        movies.append(SECOND)

    _predict(cfg, movies)
    return cfg, movie


def filmed_pose_project(root, name: str):
    """The same, on a project whose detections carry keypoints."""
    from fixtures import make_project

    from trackanno.annotate.clips import CACHE_FILE
    from trackanno.config import load_config
    from trackanno.project import write_config

    fixture = Path(make_project(root / "pose-source", pose=True))
    project = Path(root) / name
    shutil.copytree(fixture.parent, project)
    config_path = project / "config.yaml"
    write_config(config_path, {
        "paths.work_dir": str(project / "work").replace("\\", "/"),
        "paths.csv": str(project / "annotations.csv").replace("\\", "/"),
        "paths.video_dir": str(project / "videos").replace("\\", "/"),
    })
    cfg = load_config(config_path)
    movie = sorted(p.name for p in cfg.paths.work_dir.iterdir() if p.is_dir())[0]
    # Nothing here may read cached pixels, so there are none to read.
    (cfg.movie_dir(movie) / CACHE_FILE).unlink(missing_ok=True)
    _film(cfg, movie)
    _predict(cfg, [movie])
    return cfg, movie


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def select(panel, movie_id, track_id) -> None:
    panel.movie.setCurrentText(str(movie_id))
    for i in range(panel.track.count()):
        if panel.track.itemData(i) == int(track_id):
            panel.track.setCurrentIndex(i)
            return
    raise AssertionError(f"track {track_id} is not offered for {movie_id}")


def settle(app, footage, frame: int | None = None, tries: int = 3000) -> str:
    """Spin the loop until the decode thread has answered, and say what it said.

    Every frame arrives off the UI thread, so nothing in this suite can look at
    the picture without waiting for it first.
    """
    for _ in range(tries):
        app.processEvents()
        text = footage.source.text()
        settled = not text.startswith("reading") and (
            text.startswith("No video") or frame is None or footage.position == frame
        )
        if settled:
            return text
        time.sleep(0.002)
    return footage.source.text()


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


def decoder(cfg, movie_id: str, track_id: int = TRACK):
    """A :class:`MovieFrames` on the session's own crop, ready to cut."""
    from trackanno.gui.timeline import MovieFrames, review_crop

    return MovieFrames(cfg, movie_id, track_id, review_crop(cfg))


def target_boxes(marks: pd.DataFrame) -> pd.DataFrame:
    return marks[marks["is_target"]].sort_values("t")


def centres(marks: pd.DataFrame) -> dict[int, tuple[float, float]]:
    rows = target_boxes(marks)
    return {
        int(r.t): ((r.x1 + r.x2) / 2.0, (r.y1 + r.y2) / 2.0)
        for r in rows.itertuples(index=False)
    }


@pytest.fixture(scope="module")
def filmed(root):
    return filmed_project(root, "footage", second_movie=True)


@pytest.fixture(scope="module")
def posed_film(root):
    return filmed_pose_project(root, "footage-pose")


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
    settle(app, widget.footage)
    yield widget
    widget.close()
    app.processEvents()


# --------------------------------------------------------------------------- #
# one crop for the session
# --------------------------------------------------------------------------- #
def test_the_session_has_one_crop_and_it_comes_from_the_whole_project(
    tmp_path,
) -> None:
    """Where the number comes from, and what it may not depend on.

    ``clips.crop_size: auto`` sizes the cache per movie, so the widths on disk
    genuinely differ between one movie and the next. Taking the crop from
    whichever movie is on screen is what made the picture change size when the
    reviewer changed movie; taking it once from the whole project is what stops
    it.
    """
    from trackanno.config import load_config
    from trackanno.gui.timeline import review_crop
    from trackanno.project import create_project, write_config

    project = create_project(tmp_path / "sized")
    write_config(project, {"clips.crop_size": 96, "clips.max_size": 256})
    cfg = load_config(project)

    def manifest(movie_id: str, widths) -> None:
        out = cfg.movie_dir(movie_id)
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "movie_id": movie_id, "track_id": 1,
            "f_start": np.arange(len(widths)) * 50,
            "crop_w": list(widths), "crop_h": list(widths),
        }).to_parquet(out / "clips.parquet", index=False)

    bare = review_crop(cfg)
    check("a project with no clip manifest still gets a crop, rather than none",
          bare.side > 0 and bare.size > 0, str(bare))
    check("...and says which setting it took it from",
          "clips.crop_size" in bare.source, bare.source)

    manifest("movie_a", [176] * 100)
    manifest("movie_b", [192] * 100)
    crop = review_crop(cfg)
    check("with manifests it is measured off them, not off the settings",
          "cache was cut over" in crop.source, crop.source)
    check("...over every movie in the project, not the one on screen",
          "2 movie(s)" in crop.source, crop.source)
    check("...and is wide enough for the widest of them",
          crop.side >= 192, str(crop.side))
    check("...rounded as the cache's own size is rounded",
          crop.side % 16 == 0, str(crop.side))

    manifest("movie_c", [1200])
    outlier = review_crop(cfg)
    check("one absurd window - two animals tracked as one - does not shrink "
          "every other animal in the project to fit it",
          outlier.side == crop.side, f"{crop.side} -> {outlier.side}")

    manifest("movie_d", [1200] * 120)
    wide = review_crop(cfg)
    check("a project that really is cut that wide gets the wide crop",
          wide.side > crop.side, f"{crop.side} -> {wide.side}")
    check("...but it is never delivered above clips.max_size, which is the "
          "project's own bound on how big a crop it will hold",
          wide.size == cfg.clips.max_size, str(wide.size))
    check("...and below that bound nothing is resampled at all",
          crop.size == crop.side and crop.scale == 1.0, str(crop))
    check("the sentence under the picture gives the size in movie pixels",
          crop.sentence().startswith(f"{crop.side} px crop"), crop.sentence())
    check("...and says it does not change", "every track and movie" in crop.sentence(),
          crop.sentence())


def test_the_picture_does_not_change_size_across_tracks_or_movies(
    app, panel, filmed,
) -> None:
    """The reported bug, in the two ways it showed.

    A crop taken per movie changes size when the movie changes; a crop taken
    from a track's own cached windows changes size when the track changes. The
    reviewer reads either as the animal changing size, which is a claim about
    the recording that nothing here is entitled to make.
    """
    cfg, movie = filmed
    footage = panel.footage
    check("the first track is on screen", footage.showing == "movie",
          footage.source.text())
    first = footage.frame_image().shape
    crop = footage.crop
    check("it is delivered at the session's crop",
          first == (crop.size, crop.size), f"{first} vs {crop.size}")

    select(panel, movie, 2)
    settle(app, footage)
    check("another track of the same movie is the same size",
          footage.frame_image().shape == first,
          f"{first} -> {footage.frame_image().shape}")

    widths = {
        name: int(pd.read_parquet(cfg.paths.work_dir / name / "clips.parquet")
                  ["crop_w"].max())
        for name in (movie, SECOND)
    }
    check("the two movies really were cached at different crops, which is what "
          "used to make this jump", widths[movie] != widths[SECOND], str(widths))

    select(panel, SECOND, TRACK)
    settle(app, footage)
    check("the other movie is on screen", footage.showing == "movie",
          footage.source.text())
    check("...at the same size as the first", footage.frame_image().shape == first,
          f"{first} -> {footage.frame_image().shape}")
    check("...off the same crop object, measured once for the session",
          footage.crop is crop, str(footage.crop))
    check("and the line under the picture says what that crop is",
          f"{crop.side} px crop" in footage.source.text(), footage.source.text())


# --------------------------------------------------------------------------- #
# where the crop is centred
# --------------------------------------------------------------------------- #
def test_the_animal_stays_in_the_picture_across_a_long_span(app, filmed) -> None:
    """A bout runs for hundreds of frames and a cached window is fifty.

    The animal drifts steadily across this movie, so a crop fixed over a span
    that long loses it and a crop re-chosen per read of the movie steps sideways
    at every read boundary. Both are checked here over a span longer than a
    cached window, frame by frame.
    """
    from trackanno.gui.timeline import REVIEW_CHUNK
    from trackanno.view.overlay import build_clip_view

    cfg, movie = filmed
    frames = decoder(cfg, movie)
    try:
        reads = {first: frames._cut(first) for first in (0, 50, 100)}
        span = sum(len(pixels) for pixels, _ in reads.values())
        check("the span checked is longer than one cached window",
              span > cfg.clips.frames, f"{span} vs {cfg.clips.frames}")
        check("...and it took more than one read of the movie to cover it",
              len(reads) > 1 and REVIEW_CHUNK < span, str(len(reads)))

        size = frames.crop.size
        worst_edge, worst_centre = 0.0, 0.0
        for first, (pixels, marks) in reads.items():
            view = build_clip_view(marks, len(pixels), size, mode="locked")
            for t, (cx, cy) in centres(marks).items():
                worst_edge = max(worst_edge, -min(view.target[t].min(), 0.0),
                                 float(view.target[t].max()) - size)
                worst_centre = max(
                    worst_centre,
                    float(np.hypot(view.centre[t][0] - cx, view.centre[t][1] - cy)))
        check("the animal's box is inside the crop on every frame of the span, "
              "so it never walks out of the picture", worst_edge <= 0.0,
              f"{worst_edge:.1f} px outside")
        check("...and the locked view puts it dead centre on every one of them",
              worst_centre < 1e-6, f"{worst_centre:.3f} px off centre")

        # The frames either side of a read boundary are consecutive frames of
        # the same animal and have to look like it.
        steps = []
        for first, nxt in ((0, 50), (50, 100)):
            here, there = centres(reads[first][1]), centres(reads[nxt][1])
            steps.append(float(np.hypot(
                there[0][0] - here[nxt - first - 1][0],
                there[0][1] - here[nxt - first - 1][1])))
        check("the picture does not step where one read of the movie ends and "
              "the next begins", max(steps) < 3.0, str([round(s, 2) for s in steps]))

        # The crop follows a moving average, not the box itself, so the animal
        # still moves inside it. A crop locked on the box every frame would hold
        # it perfectly still, and "swimming" and "immotile" would look alike.
        within = np.array(list(centres(reads[0][1]).values()))
        check("the animal still moves within the crop, so the reviewer who asks "
              "for a fixed view can see it swim",
              float(np.ptp(within, axis=0).max()) > 1.0,
              str(np.ptp(within, axis=0).round(1)))
    finally:
        frames.stop()


def test_a_frame_is_cut_the_same_way_whichever_read_it_arrives_in(app, filmed) -> None:
    """The cut is a pure function of the frame, and that is the whole point.

    The old fallback centred each read on its own stretch, so the same movie
    frame came out differently depending on where the reader happened to have
    started - and scrubbing back and forth over one boundary showed two
    different pictures of it.
    """
    cfg, movie = filmed
    frames = decoder(cfg, movie)
    try:
        aligned, _ = frames._cut(50)
        offset, _ = frames._cut(45)
        check("frame 60 is the same picture in both reads",
              np.array_equal(aligned[10], offset[15]),
              f"worst difference "
              f"{int(np.abs(aligned[10].astype(int) - offset[15].astype(int)).max())}")
        check("...and so is every frame the two reads share",
              np.array_equal(aligned[:40], offset[5:45]))
    finally:
        frames.stop()


def test_a_review_frame_is_the_movie_and_not_a_second_rendering(app, filmed) -> None:
    """What replaced the old claim that a movie cut equalled a cached frame.

    That claim cannot hold any more and should not: the cache's crop is sized
    per movie and centred on a labelled window, and one crop for the session
    centred on the track cannot be both movies' windows at once. So the cut is
    checked against the thing it is a cut OF - the movie, through the intensity
    mapping ``prepare`` wrote beside it - which says the same thing the old
    claim said, that this is not a second renderer.
    """
    from trackanno.annotate.clips import NORM_FILE, _crop
    from trackanno.video.movies import Normalizer, open_movie

    cfg, movie_id = filmed
    frames = decoder(cfg, movie_id)
    try:
        pixels, _ = frames._cut(50)
        side = frames.crop.side
        origin = np.round(frames._path[60] - side / 2.0).astype(int)

        movie = open_movie(cfg, movie_id)
        try:
            mapping = Normalizer.load(cfg.movie_dir(movie_id) / NORM_FILE)
            expected = _crop(mapping.to_u8(movie.frame(60), 60),
                             int(origin[0]), int(origin[1]), side, side,
                             movie.height, movie.width)
        finally:
            movie.close()
        check("a review frame is the movie's own pixels at the crop it says it "
              "took", np.array_equal(pixels[10], expected),
              f"worst difference "
              f"{int(np.abs(pixels[10].astype(int) - expected.astype(int)).max())}")
        check("...through the intensity mapping prepare wrote, so it is the "
              "same grey as the window the label was applied in",
              int(pixels[10].max()) > 200 and int(pixels[10].min()) < 100,
              f"{pixels[10].min()}-{pixels[10].max()}")
        check("nothing is resampled while the crop fits the project's bound",
              frames.crop.scale == 1.0, str(frames.crop))
    finally:
        frames.stop()


# --------------------------------------------------------------------------- #
# what is marked on it
# --------------------------------------------------------------------------- #
def test_a_pose_project_gets_its_skeleton_on_a_frame_cut_from_the_movie(
    app, posed_film,
) -> None:
    """A reticle and no skeleton is a picture of a different thing.

    The cached box table carries the keypoints; a frame cut from the movie has
    to read them out of ``tracks.parquet`` itself, and the code that used not to
    handed the viewer a table of boxes. ``_kpt_columns`` then returned None and
    the skeleton simply did not appear - on a pose project, where the shape of
    the animal is what the label is about.
    """
    from trackanno.gui.timeline import TrackFootage

    cfg, movie = posed_film
    cfg.ui.show_skeleton = True
    check("this project really is a pose project with keypoints to draw",
          cfg.is_pose and cfg.n_keypoints > 0, str(cfg.n_keypoints))

    frames = decoder(cfg, movie)
    try:
        pixels, marks = frames._cut(50)
        columns = [f"kp_x_{k}" for k in range(cfg.n_keypoints)]
        check("the marks that come back with the pixels carry every keypoint",
              all(c in marks.columns for c in columns),
              str([c for c in columns if c not in marks.columns]))
        check("...read off tracks.parquet on the decode thread, with the pixels, "
              "rather than looked up while the frame is painted",
              frames._kpts is not None and frames._kpts.shape[1] == cfg.n_keypoints,
              "none at all" if frames._kpts is None else str(frames._kpts.shape))

        footage = TrackFootage(cfg)
        try:
            view = footage._view(marks, pixels)
            check("the view built from them has the skeleton",
                  view.kpts is not None and view.kpts.shape[1] == cfg.n_keypoints,
                  "none at all" if view.kpts is None else str(view.kpts.shape))
            check("...on every frame of the read, not merely on the first",
                  bool(np.isfinite(view.kpts).all()),
                  str(int(np.isfinite(view.kpts).all(axis=(1, 2)).sum())))
            check("...in the crop's own coordinates, so it lands on the animal",
                  float(np.nanmin(view.kpts)) > -frames.crop.size
                  and float(np.nanmax(view.kpts)) < 2 * frames.crop.size,
                  f"{float(np.nanmin(view.kpts)):.0f}..{float(np.nanmax(view.kpts)):.0f}")
            footage._mark()
            check("...and the canvas is told to draw it, not merely handed it",
                  footage.viewer.canvas.show_skeleton)
            check("...with the edges that say which keypoint joins which",
                  footage.viewer.canvas._edges == cfg.keypoints.edge_idx,
                  str(footage.viewer.canvas._edges))

            cfg.ui.show_skeleton = False
            check("with the skeleton off nothing is computed for a marker "
                  "nobody draws", footage._view(marks, pixels).kpts is None)
        finally:
            footage.release()
    finally:
        frames.stop()


def test_the_movie_cut_marks_the_animal_the_annotation_window_marks(
    app, filmed,
) -> None:
    """The reticle, the neighbours, and the frames nobody detected on.

    The box table a cut hands back has the columns and the coordinates
    ``build_box_table`` gives the cache, because the two windows have to mark
    the same animal the same way. A frame the detector missed gets NO row in
    either, which is what makes the viewer dim the marker there rather than
    assert a box nobody measured.
    """
    from trackanno.view.overlay import build_clip_view

    cfg, movie = filmed
    check("the project asks for neighbours to be marked, as the cache's own "
          "box table does", cfg.clips.context_tracks)

    frames = decoder(cfg, movie)
    try:
        pixels, marks = frames._cut(0)
        size = frames.crop.size
        target = target_boxes(marks)
        check("the target is marked on every frame of the read",
              len(target) == len(pixels), f"{len(target)} of {len(pixels)}")
        check("...and is named as the target, so the reticle knows which animal "
              "the bout is about", bool(target["is_target"].all()))
        check("the marks carry the columns the cache's box table carries",
              {"t", "track_id", "is_target", "x1", "y1", "x2", "y2", "conf"}
              <= set(marks.columns), str(sorted(marks.columns)))
        check("the boxes are in crop pixels, inside the picture",
              float(target[["x1", "y1"]].to_numpy().min()) >= 0.0
              and float(target[["x2", "y2"]].to_numpy().max()) <= size,
              f"{float(target[['x1', 'y1']].to_numpy().min()):.1f}.."
              f"{float(target[['x2', 'y2']].to_numpy().max()):.1f}")

        other = marks[~marks["is_target"]]
        check("a neighbour that falls inside the crop comes with it, as it does "
              "in the annotation window",
              len(other) == 0 or int(other["track_id"].iloc[0]) != TRACK,
              str(len(other)))

        view = build_clip_view(marks, len(pixels), size, mode="locked",
                               with_context=True)
        check("the view marks every frame of the read",
              bool(view.detected.all()), str(int(view.detected.sum())))
    finally:
        frames.stop()


def test_a_frame_nobody_detected_on_gets_no_box_rather_than_a_guessed_one(
    app, root,
) -> None:
    """The detector misses frames, and the cut has to say so.

    A held box asserted as a detection is a marker that sits still while the
    animal moves, which reads as the animal having stopped - the very thing a
    reviewer is here to judge.
    """
    cfg, movie = filmed_project(root, "footage-gaps")
    path = cfg.movie_dir(movie) / "tracks.parquet"
    table = pd.read_parquet(path)
    missing = set(range(55, 61))
    kept = table[~((table["track_id"] == TRACK) & (table["frame"].isin(missing)))]
    kept.to_parquet(path, index=False)

    frames = decoder(cfg, movie)
    try:
        pixels, marks = frames._cut(50)
        seen = set(target_boxes(marks)["t"].astype(int))
        check("the frames the detector missed carry no box",
              seen.isdisjoint({f - 50 for f in missing}), str(sorted(seen)))
        check("...and every other frame of the read still does",
              seen == set(range(len(pixels))) - {f - 50 for f in missing},
              str(sorted(seen)))
        check("the pixels are all there regardless: a missed detection is not a "
              "missing frame", len(pixels) == 50, str(len(pixels)))
        check("...and the crop is still centred on the animal over the gap, "
              "held from the nearest frame that saw it",
              bool(np.isfinite(frames._path[list(missing)]).all()))
    finally:
        frames.stop()


# --------------------------------------------------------------------------- #
# off the UI thread, and ahead of the playhead
# --------------------------------------------------------------------------- #
def test_nothing_is_decoded_on_the_ui_thread(app, filmed, monkeypatch) -> None:
    """One seek into a TIFF on a share is long enough to freeze the window.

    So asking for a frame has to return before the frame exists, and the decode
    has to happen somewhere else. Both halves are checked: where the work ran,
    and that the caller was not waiting for it.
    """
    from trackanno.gui.timeline import MovieFrames, TrackFootage

    cfg, movie = filmed
    threads: list[str] = []
    original = MovieFrames._cut

    def spy(self, first):
        threads.append(threading.current_thread().name)
        return original(self, first)

    monkeypatch.setattr(MovieFrames, "_cut", spy)

    footage = TrackFootage(cfg)
    try:
        footage.set_track(movie, TRACK, 0, 120)
        check("asking for a frame returns before it has been read",
              footage.showing == "" and footage._awaiting == 0,
              f"{footage.showing!r} {footage._awaiting}")
        check("...and says so rather than showing a stale or black picture",
              footage.source.text().startswith("reading"), footage.source.text())
        settle(app, footage, 0)
        check("the frame arrives", footage.showing == "movie" and footage.position == 0,
              footage.source.text())
        check("...and every decode ran off the UI thread",
              threads and all(name != "MainThread" for name in threads),
              str(sorted(set(threads))))
        check("...on the review's own decode thread",
              all(name.startswith("trackanno-review") for name in threads),
              str(sorted(set(threads))))
    finally:
        footage.release()
        app.processEvents()


def test_the_chunks_around_the_playhead_are_read_ahead(app, filmed) -> None:
    """A scrub is a request per mouse move and playback walks off the end.

    Without a read-ahead every chunk edge is a stall the length of a decode, and
    scrubbing one bout is a series of them. With one, the frame the reviewer
    asks for next is usually already in hand and goes up on the same call.
    """
    from trackanno.gui.timeline import REVIEW_CHUNK, TrackFootage

    cfg, movie = filmed
    check("the project asks for a read-ahead at all", cfg.ui.prefetch > 0,
          str(cfg.ui.prefetch))

    footage = TrackFootage(cfg)
    try:
        footage.set_track(movie, TRACK, 0, 120)
        settle(app, footage, 0)
        for _ in range(2000):
            app.processEvents()
            if REVIEW_CHUNK in footage._chunks:
                break
            time.sleep(0.002)
        check("the chunk after the one on screen is read without being asked "
              "for", REVIEW_CHUNK in footage._chunks, str(sorted(footage._chunks)))

        footage.show_frame(REVIEW_CHUNK + 5)
        check("...so scrubbing into it puts the frame up on the same call, with "
              "no decode to wait for",
              footage.position == REVIEW_CHUNK + 5 and footage._awaiting == -1,
              f"{footage.position} {footage._awaiting}")
        check("...from the chunk that was read ahead",
              footage._loaded == REVIEW_CHUNK, str(footage._loaded))

        held = len(footage._chunks)
        for frame in range(0, 120, 7):
            footage.show_frame(frame)
            app.processEvents()
        check("a scrub over the whole track keeps a bounded number of chunks, "
              "rather than the movie",
              len(footage._chunks) <= max(held, 5) and len(footage._chunks) <= 5,
              str(sorted(footage._chunks)))
    finally:
        footage.release()
        app.processEvents()


# --------------------------------------------------------------------------- #
# the join with the timeline
# --------------------------------------------------------------------------- #
def test_the_video_shows_the_frame_the_timeline_marker_is_on(app, panel, filmed) -> None:
    """Scrubbing moves the film, and the film moves the marker.

    Both directions, because either one alone is a different tool: a timeline
    that drives a video is a player, and a video that drives a timeline is a
    progress bar. Reviewing a boundary needs to go both ways - point at the edge
    to see it, then play across it to decide.
    """
    from trackanno.gui.viewer import SimpleViewer

    footage, timeline = panel.footage, panel.timeline
    check("the frames come from the movie", footage.showing == "movie",
          footage.source.text())
    check("they are shown through the annotation window's own player, so the "
          "overlay and the scaling cannot drift apart from it",
          isinstance(footage.viewer, SimpleViewer), type(footage.viewer).__name__)
    check("...and the window opens on the track's first predicted frame",
          footage.position == 0, str(footage.position))
    check("the note says which frame of which track is on screen",
          "frame 0 of track 1" in footage.source.text(), footage.source.text())

    click(timeline, 72)
    settle(app, footage, 72)
    check("clicking the timeline takes the video to that frame",
          footage.position == 72, str(footage.position))
    check("...and the marker is on it too", timeline.position == 72,
          str(timeline.position))

    click(timeline, 12)
    settle(app, footage, 12)
    check("scrubbing back into an earlier chunk works the same way",
          footage.position == 12, str(footage.position))

    footage.viewer.play()
    for _ in range(3):
        footage.viewer._advance()
    app.processEvents()
    check("playing the video walks the marker along the timeline",
          timeline.position == 15 and footage.position == 15,
          f"{timeline.position} {footage.position}")
    footage.viewer.pause()


def test_playback_carries_from_one_chunk_into_the_next(app, panel) -> None:
    """A chunk's last frame is not the end of the track.

    The movie is read in chunks and a behavioural boundary lands wherever it
    lands, including a frame or two from a chunk edge - which is exactly the
    boundary worth watching across. Playback that stopped dead at every edge
    would be unable to show the one thing this window is for.
    """
    from trackanno.gui.timeline import REVIEW_CHUNK

    footage = panel.footage
    last = REVIEW_CHUNK - 1
    footage.show_frame(last - 1)
    settle(app, footage, last - 1)
    first = footage._loaded
    check("the frame before a chunk edge comes from the chunk that ends there",
          footage.showing == "movie" and footage.position == last - 1,
          f"{footage.showing} {footage.position}")

    footage.viewer.play()
    footage.viewer._advance()             # onto the chunk's last frame
    for _ in range(2000):
        app.processEvents()               # the hand-over is queued, not re-entrant
        if footage.position == REVIEW_CHUNK:
            break
        time.sleep(0.002)
    check("playing past a chunk's last frame carries on into the next frame of "
          "the track", footage.position == REVIEW_CHUNK, str(footage.position))
    check("...out of a different chunk", footage._loaded != first,
          f"{first} -> {footage._loaded}")
    check("...and it is still playing", footage.viewer.is_playing)
    check("the marker went with it", panel.timeline.position == REVIEW_CHUNK,
          str(panel.timeline.position))
    footage.viewer.pause()


def test_clicking_a_bout_takes_you_to_it_and_still_selects_it(app, panel) -> None:
    """One press, both jobs, and neither may cost the other.

    The correction controls are filled by clicking a span on the record lane.
    That gesture also moves the video, so the risk is a scrub that quietly stops
    the click from selecting - the frame boxes would keep whatever they had and
    the next 'Set class' would assert over the wrong range.
    """
    footage, timeline = panel.footage, panel.timeline
    # On the record lane, inside the middle bout, and not near either boundary:
    # a press within GRAB_PX of one of those is a drag, which is a different
    # gesture with its own test.
    click(timeline, 60, lane_y(timeline, "record"))
    settle(app, footage, 60)

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
    from trackanno.annotate.corrections import corrections_path
    from trackanno.gui.timeline import BoutReviewPanel

    cfg, movie = filmed
    corrections_path(cfg).unlink(missing_ok=True)
    try:
        click(panel.timeline, 60, lane_y(panel.timeline, "record"))
        settle(app, panel.footage, 60)
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
            settle(app, fresh.footage)
            check("the reloaded panel has its own video on the same track",
                  fresh.footage.showing == "movie", fresh.footage.source.text())
        finally:
            fresh.close()
            app.processEvents()
    finally:
        corrections_path(cfg).unlink(missing_ok=True)


def test_changing_the_view_mid_review_re_renders_rather_than_re_decodes(
    app, panel,
) -> None:
    """Both controls are display-time transforms, and have to stay that way.

    Every mode reads the same decoded array through a different transform, so
    switching one costs a repaint. A control that dropped the frames and went
    back to the movie would stall on a decode mid-review, which is the one thing
    the read-ahead is there to prevent.
    """
    footage = panel.footage
    before = footage.viewer.canvas._view
    pixels, pixmaps = footage.frame_image(), footage.viewer.canvas._pixmaps
    check("the controls are on screen with the picture",
          footage.controls.isVisible())

    footage.follow_combo.setCurrentText("centred")
    check("choosing another view rebuilds the geometry",
          footage.viewer.canvas._view is not before and footage.follow == "centred")
    check("...off the frames already on screen: not one pixel was read again",
          np.array_equal(footage.frame_image(), pixels)
          and footage.viewer.canvas._pixmaps is pixmaps)

    footage.marker_box.setChecked(False)
    check("turning the marker off stops it being drawn",
          not footage.viewer.canvas.show_overlay)
    footage.marker_box.setChecked(True)
    check("...and back on brings it back", footage.viewer.canvas.show_overlay)


def test_the_review_window_opens_on_a_view_that_re_centres_every_frame(
    app, filmed,
) -> None:
    """``ui.follow`` is chosen for one 25-frame clip, and a bout is not one.

    A moving average removes drift and keeps the beat, which is what an
    annotator watching a single window wants. Scrubbed across the hundreds of
    frames a bout runs for, that average lags the animal and it walks out of the
    middle of the picture - so the reviewer chases the cell instead of reading
    the boundary.
    """
    from trackanno.gui.timeline import REVIEW_FOLLOW, TrackFootage
    from trackanno.view.overlay import build_clip_view

    cfg, movie = filmed
    was, cfg.ui.follow = cfg.ui.follow, "smooth"   # what the annotation window is on
    frames = decoder(cfg, movie)
    footage = TrackFootage(cfg)
    try:
        check("the review window opens on the view that re-centres every frame, "
              "rather than inheriting the annotation window's average",
              footage.follow == REVIEW_FOLLOW == "locked",
              f"{footage.follow}, with ui.follow {cfg.ui.follow!r}")
        pixels, marks = frames._cut(0)
        view = footage._view(marks, pixels)
        path = np.array([centres(marks)[t] for t in sorted(centres(marks))])
        check("...so the animal is in the middle of the picture on every frame "
              "of the span", np.allclose(view.centre, path),
              str(float(np.abs(view.centre - path).max())))

        lagged = build_clip_view(marks, len(pixels), int(pixels.shape[1]),
                                 mode=cfg.ui.follow,
                                 smooth_window=cfg.ui.smooth_window)
        check("...which is the whole difference: the project's own setting "
              "leaves it off centre",
              np.abs(lagged.centre - path).max() > 1.0,
              str(float(np.abs(lagged.centre - path).max())))

        footage.follow_combo.setCurrentText("centred")
        check("and the reviewer can still ask for either of the other two, "
              "which is what the selector is for",
              footage.follow == "centred"
              and float(np.ptp(footage._build_view().centre[:, 0])) == 0.0)
    finally:
        footage.release()
        frames.stop()
        cfg.ui.follow = was


# --------------------------------------------------------------------------- #
# when there are no pixels
# --------------------------------------------------------------------------- #
def test_a_project_whose_movies_are_missing_keeps_the_timeline_and_says_why(
    app, root,
) -> None:
    """The state a project sits in when the recordings are on a share that is
    not mounted.

    Losing the video must cost the video only. A black rectangle would read as a
    decoder that failed, and a traceback out of the slot takes the window with
    it - so the picture goes away and a sentence takes its place.
    """
    from test_predict import build_case

    from trackanno.gui.timeline import BoutReviewPanel

    cfg, manifest = build_case(root, "footage-unfilmed")
    cfg = _own_dirs(cfg)                  # a video directory of its own, and empty
    movie = str(manifest["movie_id"].iat[0])
    _predict(cfg, [movie])

    widget = BoutReviewPanel(cfg)
    widget.resize(1100, 600)
    widget.show()
    app.processEvents()
    select(widget, movie, TRACK)
    widget.timeline.resize(820, 260)
    note = settle(app, widget.footage)
    try:
        check("the timeline still draws the predictions",
              ink(painted(widget.timeline)) > 500,
              str(ink(painted(widget.timeline))))
        check("nothing is on screen where the video would be",
              not widget.footage.viewer.isVisible(), note)
        check("...and no controls over a video that is not there",
              not widget.footage.controls.isVisible())
        check("...and it says there is no video", note.startswith("No video"), note)
        check("...and names the movie it could not read", movie in note, note)
        check("...and that the timeline is unaffected", "timeline still" in note, note)
        check("no video did not cost the corrections either",
              widget._group_problem() == "", widget._group_problem())

        widget.footage.show_frame(60)
        check("scrubbing does not ask the missing movie again, which on a dead "
              "share is a hang per mouse move",
              widget.footage._movie is None and not widget.footage.viewer.isVisible(),
              widget.footage.source.text())
    finally:
        widget.close()
        app.processEvents()


def test_a_track_the_detector_never_saw_says_so_rather_than_showing_nothing(
    app, filmed,
) -> None:
    """A predicted track with no detections cannot be centred on.

    That is a broken pair of files rather than a normal state, so it is worth a
    sentence naming both halves - and it must not be an exception out of a Qt
    slot, which takes the window with it.
    """
    from trackanno.gui.timeline import TrackFootage

    cfg, movie = filmed
    footage = TrackFootage(cfg)
    try:
        footage.set_track(movie, 987, 0, 120)
        note = settle(app, footage)
        check("it says there is no video", note.startswith("No video"), note)
        check("...and why: the track was never detected in this movie",
              "987" in note and "never detected" in note, note)
        check("...and the picture is gone rather than black",
              not footage.viewer.isVisible(), note)
    finally:
        footage.release()
        app.processEvents()


# --------------------------------------------------------------------------- #
# letting go
# --------------------------------------------------------------------------- #
def test_hiding_the_window_lets_go_and_showing_it_gets_the_video_back(
    app, filmed,
) -> None:
    """The Model page keeps one review dialog for the whole session.

    Holding a movie open and a decode thread running for that long is a file
    handle and a thread nobody is looking at, so both are dropped whenever the
    window goes away - which means re-showing it has to put the footage back. A
    window that came back with a sentence where the video was would look exactly
    like a project that had lost its recordings.
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
        settle(app, panel.footage, 72)
        check("the dialog is showing the movie", panel.footage.showing == "movie"
              and panel.footage.position == 72, panel.footage.source.text())

        dialog.hide()
        app.processEvents()
        check("hiding it lets go of the movie and stops the decode thread",
              panel.footage._movie is None and not panel.footage._chunks,
              str(sorted(panel.footage._chunks)))

        dialog.show()
        settle(app, panel.footage, 72)
        check("showing it again brings the video back",
              panel.footage.showing == "movie", panel.footage.source.text())
        check("...to the frame it was left on", panel.footage.position == 72,
              str(panel.footage.position))
    finally:
        dialog.close()
        app.processEvents()


def test_closing_the_window_stops_the_decode_thread(app, root) -> None:
    """A daemon thread left reading a movie outlives the window that wanted it.

    It holds a file handle, it competes for the disk with whatever the user does
    next, and at interpreter exit it is a crash with no failing test.
    """
    from trackanno.gui.timeline import BoutReviewPanel

    cfg, movie = filmed_project(root, "footage-release")
    widget = BoutReviewPanel(cfg)
    widget.show()
    app.processEvents()
    select(widget, movie, TRACK)
    settle(app, widget.footage)
    check("the window really did open the movie",
          widget.footage._movie is not None and widget.footage.showing == "movie",
          widget.footage.source.text())
    thread = widget.footage._movie._thread
    check("...on a thread of its own", thread is not None and thread.is_alive())

    widget.close()
    app.processEvents()
    check("closing it drops the decoder", widget.footage._movie is None)
    for _ in range(500):
        if not thread.is_alive():
            break
        time.sleep(0.002)
    check("...and the thread it was running on has stopped", not thread.is_alive())
    check("...and the frames it was holding are gone",
          not widget.footage._chunks and widget.footage.frame_image() is None)
