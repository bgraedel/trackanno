"""Review footage: frames cut from the movie at one session crop, centred on the track.

Every project here has its own movie copy and work directory, never the shared fixture's.
"""

from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fixtures import bout_row, build_case, ink, loaded, painted, SyntheticMovie

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
    """Write the synthetic movie and the intensity mapping prepare writes beside it."""
    import tifffile
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
    """Point the project at a video directory of its own."""
    from trackanno.config import load_config
    from trackanno.project import write_config

    write_config(cfg.path, {
        "paths.video_dir": str(cfg.path.parent / "videos").replace("\\", "/"),
    })
    return load_config(cfg.path)


def filmed_project(root, name: str, second_movie: bool = False):
    """A case project with the movie and no clip pixels, so every frame must come from the movie."""

    cfg, manifest = build_case(root, name)
    cfg = _own_dirs(cfg)
    movie = str(manifest["movie_id"].iat[0])
    _film(cfg, movie)
    movies = [movie]

    if second_movie:
        # A second movie whose cache was cut at a wider crop.
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
    (cfg.movie_dir(movie) / CACHE_FILE).unlink(missing_ok=True)
    _film(cfg, movie)
    _predict(cfg, [movie])
    return cfg, movie


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def select(panel, movie_id, track_id) -> None:
    if not panel.select(movie_id, track_id):
        raise AssertionError(f"track {track_id} is not offered for {movie_id}")


def settle(app, footage, frame: int | None = None, tries: int = 3000) -> str:
    """Spin the loop until the decode thread has answered; return the source line."""
    for _ in range(tries):
        app.processEvents()
        text = footage.source.text()
        settled = not text.startswith("reading") and (
            text.startswith("No video")
            or (not getattr(footage, "loading", False)
                and (frame is None or footage.position == frame))
        )
        if settled:
            return text
        time.sleep(0.002)
    return footage.source.text()


def _wait(app, condition, tries: int = 3000) -> bool:
    for _ in range(tries):
        app.processEvents()
        if condition():
            return True
        time.sleep(0.002)
    return False


def click(timeline, frame: int, y: float | None = None) -> None:
    """Press and release the timeline where ``frame`` is drawn."""
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
    """A :class:`TrackCutter` on the session's own crop."""
    from trackanno.gui.review import TrackCutter, review_crop

    return TrackCutter(cfg, movie_id, track_id, review_crop(cfg))


def target_boxes(marks: pd.DataFrame) -> pd.DataFrame:
    return marks[marks["is_target"]].sort_values("t")


def centres(marks: pd.DataFrame) -> dict[int, tuple[float, float]]:
    rows = target_boxes(marks)
    return {
        int(r.t): ((r.x1 + r.x2) / 2.0, (r.y1 + r.y2) / 2.0)
        for r in rows.itertuples(index=False)
    }


def chunk() -> int:
    from trackanno.gui.review import REVIEW_CHUNK

    return int(REVIEW_CHUNK)


@pytest.fixture(scope="module")
def filmed(root):
    return filmed_project(root, "footage", second_movie=True)


@pytest.fixture(scope="module")
def posed_film(root):
    return filmed_pose_project(root, "footage-pose")


@pytest.fixture
def panel(app, filmed):
    """A panel per test, on the filmed track."""
    from trackanno.gui.review import BoutReviewPanel

    cfg, movie = filmed
    widget = BoutReviewPanel(cfg)
    loaded(app, widget)
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


@pytest.fixture
def footage(app, filmed):
    """A bare TrackFootage on the filmed project."""
    from trackanno.gui.review import TrackFootage

    cfg, _ = filmed
    widget = TrackFootage(cfg)
    yield widget
    widget.release()
    app.processEvents()


# --------------------------------------------------------------------------- #
# one crop for the session
# --------------------------------------------------------------------------- #
def test_the_session_has_one_crop_and_it_comes_from_the_whole_project(tmp_path) -> None:
    """The crop is measured over every movie's manifest, robust to outliers."""
    from trackanno.annotate.clips import MAX_SIZE
    from trackanno.config import load_config
    from trackanno.gui.review import review_crop
    from trackanno.project import create_project, write_config

    project = create_project(tmp_path / "sized")
    write_config(project, {"clips.crop_size": 96})
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
    assert bare.side > 0 and bare.size > 0, f"no manifest still gets a crop: {bare}"

    manifest("movie_a", [176] * 100)
    manifest("movie_b", [192] * 100)
    crop = review_crop(cfg)
    assert crop.side >= 192, "wide enough for the widest movie"
    assert crop.side % 16 == 0, "rounded as the cache's own size is"
    assert crop.size == crop.side and crop.scale == 1.0, (
        f"below the cache bound nothing is resampled: {crop}")

    manifest("movie_c", [1200])
    assert review_crop(cfg).side == crop.side, "one absurd window does not widen the crop"

    manifest("movie_d", [1200] * 120)
    wide = review_crop(cfg)
    assert wide.side > crop.side, "a project really cut that wide gets the wide crop"
    assert wide.size == MAX_SIZE, "but it is never delivered above the cache bound"


def test_the_picture_does_not_change_size_across_tracks_or_movies(
    app, panel, filmed,
) -> None:
    cfg, movie = filmed
    footage = panel.footage
    assert footage.showing == "movie", footage.source.text()
    first = footage.frame_image().shape
    crop = footage.crop
    assert first == (crop.size, crop.size), "delivered at the session's crop"

    select(panel, movie, 2)
    settle(app, footage)
    assert footage.frame_image().shape == first, "another track is the same size"

    widths = {
        name: int(pd.read_parquet(cfg.paths.work_dir / name / "clips.parquet")
                  ["crop_w"].max())
        for name in (movie, SECOND)
    }
    assert widths[movie] != widths[SECOND], f"the movies differ in cached crop: {widths}"

    select(panel, SECOND, TRACK)
    settle(app, footage)
    assert footage.showing == "movie", footage.source.text()
    assert footage.frame_image().shape == first, "another movie is the same size"
    assert footage.crop is crop, "the crop is measured once for the session"
    assert f"{crop.side} px crop" in footage.source.text(), footage.source.text()


# --------------------------------------------------------------------------- #
# where the crop is centred
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def long_span(filmed):
    """Three consecutive reads of the filmed track, from frames 0, 50 and 100."""
    cfg, movie = filmed
    frames = decoder(cfg, movie)
    try:
        reads = {first: frames.cut(first) for first in (0, 50, 100)}
        size = frames.crop.size
    finally:
        frames.close()
    return cfg, reads, size


def test_a_long_span_takes_several_reads(long_span) -> None:
    cfg, reads, _ = long_span
    span = sum(len(pixels) for pixels, _ in reads.values())
    assert span > cfg.clips.frames, "longer than one cached window"
    assert len(reads) > 1 and chunk() < span, "more than one read of the movie"


def test_the_animal_stays_centred_in_the_picture_across_a_long_span(long_span) -> None:
    from trackanno.view.overlay import build_clip_view

    _, reads, size = long_span
    worst_edge, worst_centre = 0.0, 0.0
    for pixels, marks in reads.values():
        view = build_clip_view(marks, len(pixels), size, mode="locked")
        for t, (cx, cy) in centres(marks).items():
            worst_edge = max(worst_edge, -min(view.target[t].min(), 0.0),
                             float(view.target[t].max()) - size)
            worst_centre = max(
                worst_centre,
                float(np.hypot(view.centre[t][0] - cx, view.centre[t][1] - cy)))
    assert worst_edge <= 0.0, f"the box leaves the crop by {worst_edge:.1f} px"
    assert worst_centre < 1e-6, f"the locked view is {worst_centre:.3f} px off centre"


def test_the_picture_does_not_step_at_a_read_boundary(long_span) -> None:
    _, reads, _ = long_span
    steps = []
    for first, nxt in ((0, 50), (50, 100)):
        here, there = centres(reads[first][1]), centres(reads[nxt][1])
        steps.append(float(np.hypot(
            there[0][0] - here[nxt - first - 1][0],
            there[0][1] - here[nxt - first - 1][1])))
    assert max(steps) < 3.0, [round(s, 2) for s in steps]


def test_the_animal_still_moves_within_the_crop(long_span) -> None:
    """The crop follows a moving average, so swimming stays visible."""
    _, reads, _ = long_span
    within = np.array(list(centres(reads[0][1]).values()))
    assert float(np.ptp(within, axis=0).max()) > 1.0, np.ptp(within, axis=0).round(1)


def test_a_frame_is_cut_the_same_way_whichever_read_it_arrives_in(filmed) -> None:
    cfg, movie = filmed
    frames = decoder(cfg, movie)
    try:
        aligned, _ = frames.cut(50)
        offset, _ = frames.cut(45)
        assert np.array_equal(aligned[10], offset[15]), "frame 60 is the same in both reads"
        assert np.array_equal(aligned[:40], offset[5:45]), "so is every shared frame"
    finally:
        frames.close()


def test_a_review_frame_is_the_movie_and_not_a_second_rendering(filmed) -> None:
    """The cut equals the movie frame through prepare's intensity mapping."""
    from trackanno.annotate.clips import NORM_FILE, _crop
    from trackanno.video.movies import Normalizer, open_movie

    cfg, movie_id = filmed
    frames = decoder(cfg, movie_id)
    try:
        pixels, _ = frames.cut(50)
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
        assert np.array_equal(pixels[10], expected), (
            f"worst difference {int(np.abs(pixels[10].astype(int) - expected.astype(int)).max())}")
        assert int(pixels[10].max()) > 200 and int(pixels[10].min()) < 100, (
            f"the intensity mapping was applied: {pixels[10].min()}-{pixels[10].max()}")
        assert frames.crop.scale == 1.0, "nothing is resampled within the crop bound"
    finally:
        frames.close()


# --------------------------------------------------------------------------- #
# what is marked on it
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def pose_cut(posed_film):
    """One read of the pose film: (cfg, pixels, marks, kpts shape, crop size)."""
    cfg, movie = posed_film
    frames = decoder(cfg, movie)
    try:
        pixels, marks = frames.cut(50)
        kpts = None if frames._kpts is None else frames._kpts.shape
        size = frames.crop.size
    finally:
        frames.close()
    return cfg, pixels, marks, kpts, size


def test_a_pose_cut_carries_every_keypoint(pose_cut) -> None:
    """Keypoints are read off tracks.parquet on the decode thread, with the pixels."""
    cfg, _, marks, kpts, _ = pose_cut
    assert cfg.is_pose and cfg.n_keypoints > 0
    missing = [c for c in (f"kp_x_{k}" for k in range(cfg.n_keypoints))
               if c not in marks.columns]
    assert not missing, missing
    assert kpts is not None and kpts[1] == cfg.n_keypoints, kpts


@pytest.fixture
def pose_footage(app, posed_film):
    from trackanno.gui.review import TrackFootage

    cfg, _ = posed_film
    footage = TrackFootage(cfg)
    footage.prefs["show_skeleton"] = True
    yield footage
    footage.release()


def test_a_pose_view_has_the_skeleton_in_crop_coordinates(pose_cut, pose_footage) -> None:
    cfg, pixels, marks, _, size = pose_cut
    view = pose_footage._view(marks, pixels)
    assert view.kpts is not None and view.kpts.shape[1] == cfg.n_keypoints
    assert np.isfinite(view.kpts).all(), "on every frame of the read"
    lo, hi = float(np.nanmin(view.kpts)), float(np.nanmax(view.kpts))
    assert -size < lo and hi < 2 * size, f"{lo:.0f}..{hi:.0f}"


def test_the_canvas_is_told_to_draw_the_skeleton(pose_cut, pose_footage) -> None:
    cfg, pixels, marks, _, _ = pose_cut
    pose_footage._view(marks, pixels)
    pose_footage._mark()
    assert pose_footage.viewer.canvas.show_skeleton
    assert pose_footage.viewer.canvas._edges == cfg.keypoints.edge_idx


def test_with_the_skeleton_off_no_keypoints_are_computed(pose_cut, pose_footage) -> None:
    _, pixels, marks, _, _ = pose_cut
    pose_footage.prefs["show_skeleton"] = False
    assert pose_footage._view(marks, pixels).kpts is None


def test_the_movie_cut_marks_the_animal_the_annotation_window_marks(filmed) -> None:
    """The cut's box table has the columns and coordinates of the cache's box table."""
    from trackanno.annotate.clips import CONTEXT_TRACKS
    from trackanno.view.overlay import build_clip_view

    cfg, movie = filmed
    assert CONTEXT_TRACKS, "neighbours are recorded"

    frames = decoder(cfg, movie)
    try:
        pixels, marks = frames.cut(0)
        size = frames.crop.size
    finally:
        frames.close()
    target = target_boxes(marks)
    assert len(target) == len(pixels), "the target is marked on every frame"
    assert target["is_target"].all()
    assert ({"t", "track_id", "is_target", "x1", "y1", "x2", "y2", "conf"}
            <= set(marks.columns)), sorted(marks.columns)
    lo = float(target[["x1", "y1"]].to_numpy().min())
    hi = float(target[["x2", "y2"]].to_numpy().max())
    assert lo >= 0.0 and hi <= size, f"boxes in crop pixels: {lo:.1f}..{hi:.1f}"

    other = marks[~marks["is_target"]]
    assert len(other) == 0 or int(other["track_id"].iloc[0]) != TRACK, (
        "neighbours inside the crop come with it")

    view = build_clip_view(marks, len(pixels), size, mode="locked", with_context=True)
    assert view.detected.all(), f"{int(view.detected.sum())} frames marked"


def test_a_frame_nobody_detected_on_gets_no_box_rather_than_a_guessed_one(root) -> None:
    cfg, movie = filmed_project(root, "footage-gaps")
    path = cfg.movie_dir(movie) / "tracks.parquet"
    table = pd.read_parquet(path)
    missing = set(range(55, 61))
    kept = table[~((table["track_id"] == TRACK) & (table["frame"].isin(missing)))]
    kept.to_parquet(path, index=False)

    frames = decoder(cfg, movie)
    try:
        pixels, marks = frames.cut(50)
        seen = set(target_boxes(marks)["t"].astype(int))
        gap = {f - 50 for f in missing}
        assert seen.isdisjoint(gap), f"missed frames carry no box: {sorted(seen)}"
        assert seen == set(range(len(pixels))) - gap, "every other frame still does"
        assert len(pixels) == 50, "a missed detection is not a missing frame"
        assert np.isfinite(frames._path[list(missing)]).all(), (
            "the crop is held on the animal over the gap")
    finally:
        frames.close()


# --------------------------------------------------------------------------- #
# off the UI thread, and ahead of the playhead
# --------------------------------------------------------------------------- #
def test_nothing_is_decoded_on_the_ui_thread(app, filmed, footage, monkeypatch) -> None:
    from trackanno.gui.review import TrackCutter

    _, movie = filmed
    threads: list[str] = []
    original = TrackCutter.cut

    def spy(self, first, *rest, **named):
        threads.append(threading.current_thread().name)
        return original(self, first, *rest, **named)

    monkeypatch.setattr(TrackCutter, "cut", spy)

    footage.set_track(movie, TRACK, 0, 120)
    assert footage.showing == "" and footage._wanted == 0, (
        "asking for a frame returns before it has been read")
    assert footage.source.text().startswith("reading"), footage.source.text()
    settle(app, footage, 0)
    assert footage.showing == "movie" and footage.position == 0, footage.source.text()
    assert threads and all(name.startswith("trackanno-review") for name in threads), (
        f"every decode ran on the review's decode thread: {sorted(set(threads))}")


def test_the_crop_is_never_measured_on_the_ui_thread(app, filmed, footage, monkeypatch) -> None:
    """Measuring reads every clips.parquet, so footage without a crop waits for one."""
    import trackanno.gui.review.footage as footage_module
    from trackanno.gui.review import review_crop

    cfg, movie = filmed
    threads: list[str] = []

    def spy(config):
        threads.append(threading.current_thread().name)
        return review_crop(config)

    monkeypatch.setattr(footage_module, "review_crop", spy)
    footage.set_track(movie, TRACK, 0, 120)
    assert footage.crop is None, "there is no crop before it is measured"
    settle(app, footage, 0)
    assert footage.showing == "movie" and footage.position == 0, footage.source.text()
    assert len(threads) == 1 and threads[0] != "MainThread", threads

    footage.set_config(cfg)
    footage.set_crop(review_crop(cfg))
    footage.set_track(movie, TRACK, 0, 120)
    settle(app, footage, 0)
    assert footage.showing == "movie" and len(threads) == 1, (
        f"a crop handed over by the page is used as it is: {threads}")


def test_the_chunks_around_the_playhead_are_read_ahead(app, filmed, footage) -> None:
    from trackanno.gui.review import CHUNKS_KEPT, REVIEW_CHUNK

    _, movie = filmed
    footage.set_track(movie, TRACK, 0, 120)
    settle(app, footage, 0)
    assert _wait(app, lambda: REVIEW_CHUNK in footage._cache, 2000), (
        f"the next chunk is read without being asked for: {sorted(footage._cache)}")

    footage.show_frame(REVIEW_CHUNK + 5)
    assert footage.position == REVIEW_CHUNK + 5 and footage._wanted is None, (
        "scrubbing into it shows the frame on the same call")
    assert footage._shown == REVIEW_CHUNK, "from the chunk that was read ahead"

    held = len(footage._cache)
    for frame in range(0, 120, 7):
        footage.show_frame(frame)
        app.processEvents()
    assert len(footage._cache) <= max(held, CHUNKS_KEPT), (
        f"a scrub over the track keeps a bounded number of chunks: {sorted(footage._cache)}")


def test_a_short_span_is_cut_to_its_own_frames(app, filmed, footage) -> None:
    """A short span is not padded to the grid cell; read-ahead stops at its end."""
    _, movie = filmed
    footage.set_track(movie, TRACK, 37, 62)
    settle(app, footage, 37)
    _wait(app, lambda: not footage.loading and 50 in footage._cache)
    held = {first: len(footage._cache.get(first)[0]) for first in footage._cache}
    assert min(held) == 37, f"the chunk starts at the span: {held}"
    assert held.get(37) == chunk() - 37, f"runs to the grid edge: {held}"
    assert held.get(50) == 62 - 50 and max(f + n for f, n in held.items()) <= 62, held


def test_a_chunk_arrives_once_and_whole(app, filmed) -> None:
    from trackanno.gui.review import ChunkDecoder, TrackCutter, review_crop

    cfg, movie = filmed
    crop = review_crop(cfg)
    chunks = ChunkDecoder(lambda t: TrackCutter(cfg, movie, t, crop))
    whole = []
    chunks.ready.connect(lambda gen, first, payload: whole.append((gen, first, len(payload[0]))))
    try:
        chunks.want(7, TRACK, 0, chunk())
        _wait(app, lambda: bool(whole))
        for _ in range(20):
            app.processEvents()
        assert whole == [(7, 0, chunk())], "the whole chunk arrives once, with its generation"
    finally:
        chunks.stop()


# --------------------------------------------------------------------------- #
# the join with the timeline
# --------------------------------------------------------------------------- #
def test_the_panel_opens_on_the_tracks_first_frame_in_the_shared_player(panel) -> None:
    from trackanno.gui.viewer import SimpleViewer

    footage = panel.footage
    assert footage.showing == "movie", footage.source.text()
    assert isinstance(footage.viewer, SimpleViewer), "the annotation window's own player"
    assert footage.position == 0
    assert "frame 0 of track 1" in footage.source.text(), footage.source.text()


def test_the_video_and_the_timeline_marker_follow_each_other(app, panel) -> None:
    footage, timeline = panel.footage, panel.timeline
    click(timeline, 72)
    settle(app, footage, 72)
    assert footage.position == 72, "clicking the timeline moves the video"
    assert timeline.position == 72, "and the marker"

    click(timeline, 12)
    settle(app, footage, 12)
    assert footage.position == 12, "scrubbing back into an earlier chunk works too"

    footage.viewer.play()
    for _ in range(3):
        footage.viewer._advance()
    app.processEvents()
    assert timeline.position == 15 and footage.position == 15, (
        "playing the video walks the marker along")
    footage.viewer.pause()


def test_playback_carries_from_one_chunk_into_the_next(app, panel) -> None:
    footage = panel.footage
    edge = chunk()
    footage.show_frame(edge - 2)
    settle(app, footage, edge - 2)
    first = footage._shown
    assert footage.showing == "movie" and footage.position == edge - 2

    footage.viewer.play()
    footage.viewer._advance()             # onto the chunk's last frame
    # The hand-over is queued, not re-entrant.
    _wait(app, lambda: footage.position == edge, 2000)
    assert footage.position == edge, "playback carries into the next chunk"
    assert footage._shown != first, f"out of a different chunk: {first} -> {footage._shown}"
    assert footage.viewer.is_playing
    assert panel.timeline.position == edge, "the marker went with it"
    footage.viewer.pause()


def test_the_end_of_a_track_loops_to_the_start_of_the_track(app, panel) -> None:
    """Not to the start of the chunk it happened to end in."""
    footage = panel.footage
    f_lo, f_hi = footage._span
    assert f_hi - f_lo > chunk(), f"the track spans more than one chunk: {f_lo}-{f_hi}"

    footage.show_frame(f_hi - 2)
    settle(app, footage, f_hi - 2)
    last_chunk = footage._shown
    assert last_chunk > f_lo, "the playhead is in the final chunk"

    footage.viewer.play()
    for _ in range(3000):
        footage.viewer._advance()
        app.processEvents()
        if footage.position <= f_lo + 2:
            break
        time.sleep(0.001)
    assert f_lo <= footage.position <= f_lo + 2, footage.position
    assert footage.position < last_chunk, "not the first frame of the last chunk"
    assert footage.viewer.is_playing
    footage.viewer.pause()


def test_the_player_has_no_scrubber_of_its_own(app, panel) -> None:
    """The timeline spanning the track is the only frame axis."""
    footage = panel.footage
    settle(app, footage)
    assert footage.showing == "movie"
    assert not footage.viewer._slider.isVisibleTo(footage), "no chunk slider"
    assert not footage.viewer._counter.isVisibleTo(footage), "no chunk-relative counter"
    assert footage.viewer._play_btn.isVisibleTo(footage), "play is still there"
    assert panel.timeline.track.f_hi - panel.timeline.track.f_lo > chunk()


def test_changing_the_view_mid_review_re_renders_rather_than_re_decodes(panel) -> None:
    footage = panel.footage
    before = footage.viewer.canvas._view
    pixels, pixmaps = footage.frame_image(), footage.viewer.canvas._pixmaps
    assert footage.controls.isVisible()

    footage.follow_combo.setCurrentText("centred")
    assert footage.viewer.canvas._view is not before and footage.follow == "centred", (
        "choosing another view rebuilds the geometry")
    assert np.array_equal(footage.frame_image(), pixels), "not one pixel was read again"
    assert footage.viewer.canvas._pixmaps is pixmaps


def test_the_marker_toggle_switches_the_overlay(panel) -> None:
    footage = panel.footage
    footage.marker_box.setChecked(False)
    assert not footage.viewer.canvas.show_overlay
    footage.marker_box.setChecked(True)
    assert footage.viewer.canvas.show_overlay


def test_the_review_window_opens_on_a_view_that_re_centres_every_frame(filmed, footage) -> None:
    """The project's moving-average follow lags the animal over a long bout."""
    from trackanno.gui.review import REVIEW_FOLLOW
    from trackanno.view.overlay import build_clip_view

    cfg, movie = filmed
    assert footage.follow == REVIEW_FOLLOW == "locked", footage.follow
    frames = decoder(cfg, movie)
    try:
        pixels, marks = frames.cut(0)
    finally:
        frames.close()
    view = footage._view(marks, pixels)
    path = np.array([centres(marks)[t] for t in sorted(centres(marks))])
    assert np.allclose(view.centre, path), "the animal is centred on every frame"

    lagged = build_clip_view(marks, len(pixels), int(pixels.shape[1]), mode="centred")
    assert np.abs(lagged.centre - path).max() > 1.0, "the moving average leaves it off centre"

    footage.follow_combo.setCurrentText("centred")
    assert footage.follow == "centred"
    assert float(np.ptp(footage._build_view().centre[:, 0])) == 0.0


def test_clicking_the_right_end_of_the_timeline_shows_the_last_frame(app, panel) -> None:
    footage, timeline = panel.footage, panel.timeline
    f_lo, f_hi = timeline.track.f_lo, timeline.track.f_hi
    assert timeline._frame_at(timeline.width() + 50) == f_hi - 1

    click(timeline, f_hi)
    settle(app, footage, f_hi - 1)
    assert footage.position == f_hi - 1, footage.source.text()
    assert footage.showing == "movie" and not footage.source.text().startswith("No video")
    assert timeline.position == f_hi - 1

    footage.show_frame(f_hi + 500)            # past the end is clamped, not an error
    settle(app, footage, f_hi - 1)
    assert footage.position == f_hi - 1 and footage.showing == "movie"
    click(timeline, f_lo + 12)
    settle(app, footage, f_lo + 12)
    assert footage.position == f_lo + 12, footage.source.text()


# --------------------------------------------------------------------------- #
# when there are no pixels
# --------------------------------------------------------------------------- #
def test_a_project_whose_movies_are_missing_keeps_the_timeline_and_says_why(app, root) -> None:
    from trackanno.gui.review import BoutReviewPanel

    cfg, manifest = build_case(root, "footage-unfilmed")
    cfg = _own_dirs(cfg)                  # an empty video directory of its own
    movie = str(manifest["movie_id"].iat[0])
    _predict(cfg, [movie])

    widget = BoutReviewPanel(cfg)
    loaded(app, widget)
    widget.resize(1100, 600)
    widget.show()
    app.processEvents()
    select(widget, movie, TRACK)
    widget.timeline.resize(820, 260)
    note = settle(app, widget.footage)
    try:
        assert ink(painted(widget.timeline)) > 500, "the timeline still draws"
        assert not widget.footage.viewer.isVisible(), "no black rectangle"
        assert not widget.footage.controls.isVisible()
        assert note.startswith("No video"), note
        assert movie in note, note
        assert "timeline is unaffected" in note, note

        widget.footage.show_frame(60)
        assert widget.footage._problem and not widget.footage.viewer.isVisible(), (
            "scrubbing does not ask the missing movie again")
    finally:
        widget.close()
        app.processEvents()


def test_a_track_the_detector_never_saw_says_so_rather_than_showing_nothing(
    app, filmed, footage,
) -> None:
    _, movie = filmed
    footage.set_track(movie, 987, 0, 120)
    note = settle(app, footage)
    assert note.startswith("No video"), note
    assert "987" in note and "never detected" in note, note
    assert not footage.viewer.isVisible(), "the picture is gone rather than black"


def test_an_unreadable_track_does_not_block_the_next_one(app, filmed, footage) -> None:
    _, movie = filmed
    footage.set_track(movie, 987, 0, 120)
    note = settle(app, footage)
    assert note.startswith("No video") and "987" in note, note
    footage.set_track(movie, TRACK, 0, 120)
    settle(app, footage, 0)
    assert footage.showing == "movie" and footage.position == 0, footage.source.text()


# --------------------------------------------------------------------------- #
# letting go
# --------------------------------------------------------------------------- #
def test_hiding_the_panel_lets_go_and_showing_it_gets_the_video_back(app, filmed) -> None:
    from trackanno.gui.review import BoutReviewPanel

    cfg, movie = filmed
    panel = BoutReviewPanel(cfg)
    loaded(app, panel)
    panel.show()
    app.processEvents()
    try:
        select(panel, movie, TRACK)
        panel.timeline.resize(820, 260)
        painted(panel.timeline)
        click(panel.timeline, 72)
        settle(app, panel.footage, 72)
        assert panel.footage.showing == "movie" and panel.footage.position == 72

        panel.hide()
        app.processEvents()
        assert panel.footage._decoder is None and not panel.footage._cache, (
            "hiding lets go of the movie and the decode thread")

        panel.show()
        settle(app, panel.footage, 72)
        assert panel.footage.showing == "movie", panel.footage.source.text()
        assert panel.footage.position == 72, "back on the frame it was left on"
    finally:
        panel.close()
        app.processEvents()


def test_closing_the_window_stops_the_decode_thread(app, root) -> None:
    from trackanno.gui.review import BoutReviewPanel

    cfg, movie = filmed_project(root, "footage-release")
    widget = BoutReviewPanel(cfg)
    loaded(app, widget)
    widget.show()
    app.processEvents()
    select(widget, movie, TRACK)
    settle(app, widget.footage)
    assert widget.footage._decoder is not None and widget.footage.showing == "movie"
    thread = widget.footage._decoder.thread
    assert thread is not None and thread.is_alive()

    widget.close()
    app.processEvents()
    assert widget.footage._decoder is None, "closing drops the decoder"
    thread.join(1.0)
    assert not thread.is_alive(), "the decode thread has stopped"
    assert not widget.footage._cache and widget.footage.frame_image() is None


# --------------------------------------------------------------------------- #
# the chunk cache and track switches
# --------------------------------------------------------------------------- #
class FakeCutter:
    """A cutter that fills every frame with ``value``, optionally held at a gate."""

    def __init__(self, value: int, gate: threading.Event | None = None):
        self.value = value
        self.gate = gate
        self.started = threading.Event()
        self.closed_on: str | None = None
        self.skeleton_problem = ""

    def open(self) -> None:
        pass

    def cut(self, first, last=None, cancelled=None):
        from trackanno.gui.review.decode import Cancelled

        self.started.set()
        if self.gate is not None:
            self.gate.wait(10)
        if cancelled is not None and cancelled():
            raise Cancelled()
        n = int(last) - int(first)
        marks = pd.DataFrame({c: [] for c in
                              ("t", "track_id", "is_target", "x1", "y1", "x2", "y2", "conf")})
        return np.full((n, 8, 8), self.value, np.uint8), marks

    def close(self) -> None:
        self.closed_on = threading.current_thread().name


def test_the_chunk_cache_evicts_the_least_recently_used() -> None:
    from trackanno.gui.review import ChunkCache

    cache = ChunkCache(3)
    for first in (0, 50, 100):
        cache.put(first, (first,))
    assert cache.get(0) == (0,)
    cache.put(150, (150,))
    assert 50 not in cache and len(cache) == 3
    assert cache.keys() == [100, 0, 150]
    assert cache.get(50) is None


def test_switching_track_does_not_wait_for_a_slow_decode(app, filmed) -> None:
    from trackanno.gui.review import TrackFootage

    cfg, movie = filmed
    gate = threading.Event()
    slow = FakeCutter(1, gate)
    footage = TrackFootage(cfg)
    footage.make_cutter = lambda m, t, source=None: slow if t == 1 else FakeCutter(2)
    try:
        footage.set_track(movie, 1, 0, 120)
        assert _wait(app, slow.started.is_set)
        decoder = footage._decoder

        began = time.perf_counter()
        footage.set_track(movie, 2, 0, 120)       # same movie: retargeted
        footage.release()                         # stopped
        footage.set_track(movie, 2, 0, 120)
        elapsed = time.perf_counter() - began
        assert elapsed < 0.1, f"{elapsed:.3f}s"
        assert decoder.thread.is_alive() and not decoder.done.is_set()

        gate.set()
        settle(app, footage, 0)
        assert footage.position == 0 and int(footage.frame_image().max()) == 2
    finally:
        gate.set()
    assert decoder.done.wait(5)
    assert slow.closed_on is not None and slow.closed_on != "MainThread"
    footage.release()
    app.processEvents()


def test_a_burst_of_track_switches_opens_only_the_last_track(app, filmed, footage) -> None:
    _, movie = filmed
    opened = []
    footage.make_cutter = lambda m, t, source=None: opened.append(t) or FakeCutter(t)
    footage.set_track(movie, 1, 0, 120)
    footage.set_track(movie, 2, 0, 120)
    footage.show_frame(70)
    settle(app, footage, 70)
    assert opened == [2], opened
    assert int(footage.frame_image().max()) == 2


def test_tracks_of_one_movie_share_one_open_movie(app, filmed, footage, monkeypatch) -> None:
    from trackanno.gui.review import MovieSource

    _, movie = filmed
    opens = []
    original = MovieSource.open

    def counting(self):
        if self.movie is None:
            opens.append(self.movie_id)
        original(self)

    monkeypatch.setattr(MovieSource, "open", counting)
    footage.set_track(movie, 1, 0, 120)
    settle(app, footage, 0)
    decoder = footage._decoder
    footage.set_track(movie, 2, 0, 120)
    settle(app, footage, 0)
    assert footage.showing == "movie", footage.source.text()
    assert footage._decoder is decoder and opens == [movie], opens

    footage.set_track(SECOND, 1, 0, 120)
    settle(app, footage, 0)
    assert footage._decoder is not decoder and opens == [movie, SECOND], opens
    assert decoder.done.wait(5), "the first movie was closed"


def test_a_previous_tracks_late_chunk_is_dropped(app, filmed, footage) -> None:
    _, movie = filmed
    gate = threading.Event()
    stale = FakeCutter(1, gate)
    footage.make_cutter = lambda m, t, source=None: stale if t == 1 else FakeCutter(2)
    try:
        footage.set_track(movie, 1, 0, 120)
        assert _wait(app, stale.started.is_set)
        old_gen = footage._gen
        footage.set_track(movie, 2, 0, 120)
        gate.set()
        settle(app, footage, 0)

        # Force a reply for the previous track through; it must be dropped.
        footage._decoder.ready.emit(old_gen, 0, FakeCutter(1).cut(0, 50))
        for _ in range(50):
            app.processEvents()
        assert all(int(footage._cache.get(f)[0].max()) == 2 for f in footage._cache)
        assert int(footage.frame_image().max()) == 2
    finally:
        gate.set()
