"""The bout review: reading the decoded bout table and drawing it as a timeline.

No ``ClipStore`` is opened here, so the shared fixture's cache is never left mapped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fixtures import bout_row, build_case, ink, loaded, painted, planted_case

GROUP = "behaviour"
CLASSES = ("swimming", "immotile")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


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


def show_panel(app, cfg, width=1000, height=400):
    from trackanno.gui.review import BoutReviewPanel

    panel = BoutReviewPanel(cfg)
    loaded(app, panel)
    panel.resize(width, height)
    panel.show()
    app.processEvents()
    return panel


# --------------------------------------------------------------------------- #
# the schema, on a file built by hand
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def synthetic(root):
    """A project with a hand-built bout table, including a marginal and a skipped bout."""

    cfg, _ = build_case(root, "review-synthetic")
    write_bouts(cfg, [
        bout_row(0, 120, "swimming", 0.91),
        bout_row(120, 160, "immotile", 0.42),
        bout_row(160, 180, "", np.nan,
                 reason="no window over frames 160-180 could be featurised"),
        bout_row(180, 300, "swimming", 0.77),
    ])
    return cfg


@pytest.fixture(scope="module")
def synthetic_lane(synthetic):
    from trackanno.gui.review import PredictionReader

    reader = PredictionReader(synthetic)
    assert reader.problem == "", reader.problem
    return reader.track("m", 1)


def test_the_reader_finds_the_group_from_its_pred_column(synthetic) -> None:
    from trackanno.gui.review import PredictionReader

    assert PredictionReader(synthetic).groups == [GROUP]


def test_the_segments_tile_the_track(synthetic_lane) -> None:
    track = synthetic_lane
    assert (track.f_lo, track.f_hi) == (0, 300), "the span is the union of its rows"
    lane = track.lanes[GROUP]
    assert len(lane) == 4, "every row of the group became a segment"
    assert all(a.f_end == b.f_start for a, b in zip(lane, lane[1:]))
    assert lane[0].f_start == track.f_lo and lane[-1].f_end == track.f_hi


def test_each_segment_keeps_its_class_and_mean_confidence(synthetic_lane) -> None:
    lane = synthetic_lane.lanes[GROUP]
    assert lane[0].label == "swimming" and abs(lane[0].conf - 0.91) < 1e-9
    assert abs(lane[1].conf - 0.42) < 1e-9, "a marginal bout is not rounded up"


def test_a_skipped_bout_has_no_class_and_no_confidence(synthetic_lane) -> None:
    gap = synthetic_lane.lanes[GROUP][2]
    assert gap.label == ""
    assert not np.isfinite(gap.conf), f"NaN, never 0.0: {gap.conf}"
    assert "could be featurised" in gap.skip_reason, gap.skip_reason


def test_a_track_with_no_predictions_is_an_empty_track(synthetic) -> None:
    from trackanno.gui.review import PredictionReader

    assert PredictionReader(synthetic).track("m", 4242).empty


def test_an_empty_track_paints_a_sentence_saying_it_is_normal(app, synthetic) -> None:
    from trackanno.gui.review import PredictionReader

    track = PredictionReader(synthetic).track("m", 4242)
    panel = show_panel(app, synthetic, 900, 320)
    panel.timeline.set_track(track)
    panel.timeline.resize(900, 200)
    app.processEvents()
    assert ink(painted(panel.timeline)) > 0
    assert "No predicted bouts" in panel.timeline._nothing_here()
    panel.close()
    app.processEvents()


# --------------------------------------------------------------------------- #
# a real decode
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def decoded(root):
    """The planted 300-frame track, decoded at stride 10."""
    from trackanno.infer.predict import predict_all

    cfg, movie = planted_case(root, "review-decoded", 10)
    predict_all(cfg)
    return cfg, movie


def test_a_real_decode_reaches_the_timeline_intact(decoded) -> None:
    from trackanno.gui.review import PredictionReader

    cfg, movie = decoded
    reader = PredictionReader(cfg)
    assert reader.problem == "", reader.problem

    track = reader.track(movie, 1)
    lane = track.lanes[GROUP]
    spans = [(s.f_start, s.f_end, s.label) for s in lane]
    assert lane[0].f_start == track.f_lo and lane[-1].f_end == track.f_hi, spans
    assert all(a.f_end == b.f_start for a, b in zip(lane, lane[1:])), spans
    assert len({s.label for s in lane if s.label}) == 2, (
        f"the planted transition shows as more than one bout: {spans}")


def test_the_selectors_are_filled_from_the_file(app, decoded) -> None:
    cfg, movie = decoded
    panel = show_panel(app, cfg)
    assert [panel.movie.itemText(i) for i in range(panel.movie.count())] == [movie]
    assert panel.track.count() >= 1
    assert "predicted bouts" in panel.headline.text(), panel.headline.text()
    panel.close()
    app.processEvents()


def test_the_timeline_draws_every_bout_hit_testably(app, decoded) -> None:
    from qtpy.QtCore import QPointF

    cfg, _ = decoded
    panel = show_panel(app, cfg)
    panel.timeline.resize(940, 200)
    app.processEvents()
    assert ink(painted(panel.timeline)) > 500

    timeline = panel.timeline
    gutter, scale = timeline._geom
    hits = []
    for lane, (group, segments) in enumerate(timeline.track.lanes.items()):
        y = timeline.TOP + lane * timeline.LANE + timeline.LANE / 2
        for segment in segments:
            x = timeline._x((segment.f_start + segment.f_end) / 2, gutter, scale)
            hits.append(timeline._at(QPointF(x, y)))
    assert len(hits) == timeline.track.n_segments and all(hits), (
        f"{sum(map(bool, hits))} hits for {timeline.track.n_segments} segments")

    tip = timeline._describe(*hits[0])
    assert "mean posterior" in tip, tip
    panel.close()
    app.processEvents()


def test_a_track_the_predictions_never_mention_is_still_selectable(decoded) -> None:
    """Tracks come from the union of the predictions and ``tracks.parquet``."""
    from trackanno.annotate.clips import TRACKS_FILE
    from trackanno.gui.review import PredictionReader

    cfg, movie = decoded
    path = cfg.movie_dir(movie) / TRACKS_FILE
    tracks = pd.read_parquet(path)
    try:
        if not (tracks["track_id"] == 7).any():
            extra = tracks.head(3).copy()
            extra["track_id"] = 7
            pd.concat([tracks, extra], ignore_index=True).to_parquet(path, index=False)

        listed = dict(PredictionReader(cfg).tracks(movie))
        assert listed.get(7) == 0, f"offered with a count of zero, not hidden: {listed}"
        assert listed[1] > 0, f"the predicted track keeps its real count: {listed}"
    finally:
        tracks.to_parquet(path, index=False)


# --------------------------------------------------------------------------- #
# files that are not decoded bouts
# --------------------------------------------------------------------------- #
def test_a_windows_file_is_refused_with_a_sentence(app, root) -> None:
    from trackanno.gui.review import PredictionReader

    cfg, _ = build_case(root, "review-windows")
    rows = [{k: v for k, v in bout_row(0, 50, "swimming", 0.9).items()
             if k not in ("bout_id", "group")}]
    write_bouts(cfg, rows)

    assert "not decoded bouts" in PredictionReader(cfg).problem

    panel = show_panel(app, cfg)
    assert "not decoded bouts" in panel.timeline.note, panel.timeline.note
    panel.close()
    app.processEvents()


@pytest.fixture(scope="module")
def unpredicted(root):

    cfg, _ = build_case(root, "review-none")
    return cfg


def test_a_missing_predictions_file_is_reported_with_how_to_make_it(unpredicted) -> None:
    from trackanno.gui.review import PredictionReader

    problem = PredictionReader(unpredicted).problem
    assert "predictions" in problem, problem
    assert "Label all clips with the model" in problem, problem


def test_a_panel_with_no_predictions_paints_the_reason(app, unpredicted) -> None:
    panel = show_panel(app, unpredicted, 800, 300)
    assert "Label all clips with the model" in panel.timeline.note, panel.timeline.note
    assert ink(painted(panel.timeline)) > 0
    panel.close()
    app.processEvents()


def test_the_timeline_takes_the_width_the_window_gives_it(app, root) -> None:
    """No horizontal scrollbar at any window width; growth goes to the lanes."""
    from trackanno.gui.review import BoutReviewPanel

    cfg, _ = build_case(root, "review-resize", labels=2)
    write_bouts(cfg, first_movie_bouts(cfg))
    panel = BoutReviewPanel(cfg)
    loaded(app, panel)
    panel.show()
    app.processEvents()
    scroll = panel.timeline.parentWidget().parentWidget()
    try:
        floor = panel.minimumSizeHint().width()
        assert floor <= 1180, f"fits in the hub's 1280 px window: needs {floor}px"
        assert (panel.timeline.minimumSizeHint().width()
                <= panel.timeline.sizeHint().width())

        widths = []
        # Grown, shrunk and grown again: the splitter reapportions on every resize.
        for width in (1000, 1400, 2000, 1000, 2400):
            panel.resize(width, 700)
            app.processEvents()
            widths.append((panel.width(), panel.timeline.width()))
            assert scroll.horizontalScrollBar().maximum() == 0, f"at {width}px: {widths}"
        assert all(b[1] > a[1] for a, b in zip(widths, widths[1:]) if b[0] > a[0]), (
            f"a wider window is a wider timeline: {widths}")
        assert panel.footage.width() <= panel.footage.maximumWidth(), (
            "the film does not take the growth")
    finally:
        panel.close()
        app.processEvents()


# --------------------------------------------------------------------------- #
# the page
# --------------------------------------------------------------------------- #
def test_reviewing_is_a_page_and_is_gated_on_the_file(app, root) -> None:
    """The page is gated on the predictions file and builds its panel on arrival."""
    from trackanno.gui.hub import PAGES, Hub, ModelPage, ReviewPage

    cfg, _ = build_case(root, "review-page", labels=2)
    hub = Hub(str(cfg.path))
    hub.show()
    app.processEvents()
    model, review = hub.page(ModelPage), hub.page(ReviewPage)
    try:
        assert PAGES.index("Review") > PAGES.index("Model"), PAGES
        assert len(PAGES) == hub.sidebar.count() == len(hub.pages)
        assert "nothing predicted yet" in review.blocker()
        assert review.panel is None

        write_bouts(cfg, first_movie_bouts(cfg))
        hub.reload_project()
        app.processEvents()
        assert review.blocker() == "", "the file alone unblocks the page"
        assert review.panel is None, "the panel is not built until visited"

        hub.go_to(PAGES.index("Review"))
        app.processEvents()
        loaded(app, review.panel)
        assert hub.sidebar.currentRow() == PAGES.index("Review")
        assert hub.stack.currentWidget() is review
        assert review.panel is not None and review.panel.isVisible(), (
            "arriving builds the panel")
        assert review.panel.reader is not None and review.panel.reader.problem == ""
        assert not [w for w in app.topLevelWidgets()
                    if "Review" in w.windowTitle() and w is not hub], (
            "no free-floating review window")

        panel = review.panel
        hub.go_to(PAGES.index("Model"))
        app.processEvents()
        assert panel.footage._decoder is None, "leaving the page lets go of the movie"
        hub.go_to(PAGES.index("Review"))
        app.processEvents()
        assert review.panel is panel, "coming back reuses the one panel"
    finally:
        # Pages own background threads that close() does not cancel.
        model.shutdown()
        review.shutdown()
        hub.close()
        app.processEvents()
    assert review.panel is None, "closing the page takes the panel with it"
