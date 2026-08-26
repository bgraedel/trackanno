"""What the renderer refuses to paint, and what it is happy to paint over.

``build_frame_marks`` collapses rows spanning frames into one mark per (frame,
track), so anything that lands on a frame twice has to have a rule. There are
four different rules here and they are easy to mistake for one another:

* two BOUTS on one frame is a broken upstream invariant and raises - a decode
  cuts a track's frames into disjoint runs, so no setting can ask for it;
* two prediction WINDOWS on one frame is a geometry, not a fault:
  ``predict.coverage: manifest`` scores the annotation clips themselves, and
  ``clips.stride`` below ``clips.frames`` makes consecutive clips of a track
  share frames on purpose;
* a hand label on a frame a prediction claims is what ``--source both`` IS, and
  must keep working;
* two ANNOTATION rows on one frame is the same overlapping clip grid seen from
  the labels side, and must keep working too.

Only the first is an error, and the point of this suite is that the other three
never become one. The guard used to fire on all prediction rows alike, which
made every movie of a manifest-coverage project over an overlapping clip grid
fail to render - the file permitted and refused identical geometry in two
adjacent loops.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from harness import check

GROUP = "behaviour"
MOVIE = "MOVIE_A"


@pytest.fixture(scope="module")
def cfg(root: Path):
    """A config and nothing else: none of these claims touch a pixel or a cache."""
    from trackanno.config import load_config
    from trackanno.project import create_project

    return load_config(create_project(Path(root) / "render"))


def _prediction(track_id: int, f_start: int, f_end: int, value: str, conf: float = 0.9):
    """A row off a GRID: a dense window, or under manifest coverage a clip."""
    return {"movie_id": MOVIE, "track_id": track_id, "f_start": f_start,
            "f_end": f_end, f"{GROUP}_pred": value, f"{GROUP}_conf": conf}


def _bout(track_id: int, f_start: int, f_end: int, value: str, conf: float = 0.9):
    """A row a DECODE produced. ``bout_id`` is what says so, and it is what the
    renderer probes: the file has to answer this, because ``predict.decode`` can
    be edited after the run that wrote it."""
    return dict(_prediction(track_id, f_start, f_end, value, conf),
                bout_id=f"{MOVIE}__t{track_id:04d}__f{f_start:06d}-{f_end:06d}")


def _label(track_id: int, f_start: int, f_end: int, value: str):
    return {"clip_id": f"{MOVIE}__t{track_id:04d}__f{f_start:06d}", "movie_id": MOVIE,
            "track_id": track_id, "f_start": f_start, "f_end": f_end, GROUP: value}


def test_two_decoded_bouts_on_one_frame_stop_the_render(cfg) -> None:
    """Painting one of the two would hide the fact that they exist.

    Whichever row pandas hands over last wins a plain dict assignment, so the
    exported colour of an overlapped frame comes from row order in the parquet
    rather than from the model - and the movie that comes out looks entirely
    normal. The only visible symptom is the one this guard makes.

    A BOUT is where that reasoning holds: a decode cuts each track's frames into
    runs between inferred boundaries, so two of them meeting is not a setting
    anyone can choose, it is the decode or the file being wrong.
    """
    from trackanno.render import build_frame_marks

    predictions = pd.DataFrame([
        _bout(3, 800, 850, "immotile"),
        _bout(3, 812, 862, "swimming"),
    ])
    try:
        build_frame_marks(pd.DataFrame(), predictions, cfg, GROUP, "predictions")
        check("overlapping bouts are refused", False, "it painted them")
    except ValueError as exc:
        message = str(exc)
        check("overlapping bouts are refused", True)
        check("...naming the frame they met on", "frame 812" in message, message)
        check("...naming the track", "track 3" in message, message)
        check("...and BOTH competing values, not just the loser",
              "'immotile'" in message and "'swimming'" in message, message)
        check("...with the two rows' spans, so a duplicate row reads differently "
              "from an overlapping one",
              "[800, 850)" in message and "[812, 862)" in message, message)
        check("...and calls them bouts, since that is the only row this can be",
              "bout" in message.lower(), message)

    # And through the 'both' path as well, where the labels pass would otherwise
    # paint over the evidence before anyone saw it.
    try:
        build_frame_marks(
            pd.DataFrame([_label(3, 800, 862, "swimming")]),
            predictions, cfg, GROUP, "both",
        )
        check("...on the 'both' path too", False, "it painted them")
    except ValueError:
        check("...on the 'both' path too", True)


def test_manifest_predictions_that_share_frames_render_rather_than_raising(cfg) -> None:
    """The same overlap the labels pass calls a configuration, one loop earlier.

    ``predict.coverage: manifest`` writes one prediction row per ANNOTATION clip,
    so a project whose ``clips.stride`` is below ``clips.frames`` has consecutive
    clips of a track sharing frames - by construction, not by accident. Guarding
    that raised inside ``render_movie``, which ``render_all`` catches per movie
    and logs as FAILED, so every movie of such a project produced no file at all.

    Nothing in ``predictions.parquet`` can say a window came off the annotation
    grid rather than the dense one - a dense window's ``clip_id`` is deliberately
    formed to collide with an annotation clip's - so the rule for windows is the
    labels pass's rule: the later row wins.
    """
    from trackanno.render import build_frame_marks

    # clips.frames 50, clips.stride 25: the second clip of the track starts
    # halfway through the first, exactly as the clip cutter laid them out.
    predictions = pd.DataFrame([
        _prediction(3, 0, 50, "swimming"),
        _prediction(3, 25, 75, "immotile"),
        _prediction(3, 50, 100, "swimming"),
    ])
    marks = build_frame_marks(pd.DataFrame(), predictions, cfg, GROUP, "predictions")

    check("overlapping window rows paint instead of stopping the render",
          len(marks) == 100, str(len(marks)))
    check("...the frames only the first clip covers keep its answer",
          marks[10][3].text.startswith("~swimming"), marks[10][3].text)
    check("...the shared frames go to the later row, as the labels pass does",
          marks[30][3].text.startswith("~immotile"), marks[30][3].text)
    check("...and the last clip's own frames are its own",
          marks[80][3].text.startswith("~swimming"), marks[80][3].text)
    check("they are still drawn dashed, so an overlap resolved this way can "
          "never read as a hand label", marks[30][3].predicted)

    # The guard is a property of the ROW KIND, not of the geometry: the very same
    # spans carrying a bout_id are refused.
    as_bouts = pd.DataFrame([_bout(3, 0, 50, "swimming"), _bout(3, 25, 75, "immotile")])
    try:
        build_frame_marks(pd.DataFrame(), as_bouts, cfg, GROUP, "predictions")
        check("...while the identical spans as bouts still raise", False, "it painted them")
    except ValueError:
        check("...while the identical spans as bouts still raise", True)


def test_disjoint_bouts_paint_without_complaint(cfg) -> None:
    """The guard defends disjointness, not equal length.

    Decoded bouts are variable-length runs between inferred boundaries - the
    whole point of the feature - so a suite that only ever fed the renderer a
    tiling grid would let a guard that secretly required one through.
    """
    from trackanno.render import build_frame_marks

    predictions = pd.DataFrame([
        _prediction(3, 0, 137, "swimming"),
        _prediction(3, 137, 300, "immotile"),
        _prediction(3, 300, 412, "swimming"),
        _prediction(4, 50, 412, "unclear"),        # another track, same frames
    ])
    marks = build_frame_marks(pd.DataFrame(), predictions, cfg, GROUP, "predictions")

    check("bouts that only touch at their boundary paint fine", len(marks) == 412)
    check("the frame before the boundary is the earlier bout",
          marks[136][3].text.startswith("~swimming"), marks[136][3].text)
    check("the boundary frame itself is the later bout",
          marks[137][3].text.startswith("~immotile"), marks[137][3].text)
    check("a half-open f_end paints nothing at f_end", 412 not in marks)
    check("two tracks sharing a frame are not a collision",
          marks[200][4].text.startswith("~unclear"), marks[200][4].text)


def test_a_hand_label_over_a_prediction_is_not_a_collision(cfg) -> None:
    """It is the entire point of ``--source both``.

    The labels pass writes into the same dict the predictions pass filled, so it
    is one line away from being caught by the same guard. It must not be: the
    hand label is meant to take the box and leave the model's answer in the
    caption.
    """
    from trackanno.render import build_frame_marks

    labels = pd.DataFrame([_label(3, 100, 150, "swimming")])
    predictions = pd.DataFrame([_prediction(3, 100, 150, "immotile", 0.73)])

    marks = build_frame_marks(labels, predictions, cfg, GROUP, "both")
    mark = marks[120][3]
    check("a label on a predicted frame paints rather than raising", not mark.predicted)
    check("...taking the box for itself", "swimming" in mark.text, mark.text)
    check("...and keeping the model's answer visible",
          "~immotile" in mark.text, mark.text)


def test_overlapping_annotations_are_a_configuration_not_a_fault(cfg) -> None:
    """``clips.stride`` below ``clips.frames`` cuts overlapping clips on purpose.

    Somebody annotating a grid like that has two legitimate rows covering one
    frame, and the last one wins - arbitrary, but a pre-existing and deliberate
    arbitrariness about hand labels, not the invariant this module guards.
    """
    from trackanno.render import build_frame_marks

    labels = pd.DataFrame([
        _label(3, 100, 150, "swimming"),
        _label(3, 125, 175, "immotile"),
    ])
    marks = build_frame_marks(labels, None, cfg, GROUP, "labels")
    check("overlapping hand labels still render", len(marks) == 75)
    check("...with the later row winning the shared frames",
          marks[130][3].text == "immotile", marks[130][3].text)


def test_a_prediction_row_is_counted_as_what_it_is(cfg) -> None:
    """Calling a prediction row a clip was true of neither kind of row.

    A dense row is a window off the predictor's own grid; a decoded row is a
    bout. The count is read off the FILE and not off ``predict.decode``, so
    editing the config after a run cannot make the Export page describe a run
    that has not happened yet.
    """
    from trackanno.render import prediction_unit

    windows = pd.DataFrame([_prediction(3, 0, 50, "swimming")])
    check("a row off a grid is a window", prediction_unit(windows) == "predicted windows",
          prediction_unit(windows))

    bouts = windows.assign(bout_id=[f"{MOVIE}__t0003__f000000-000050"])
    check("a row carrying a bout_id is a bout", prediction_unit(bouts) == "predicted bouts",
          prediction_unit(bouts))
    check("neither is called a clip",
          "clip" not in prediction_unit(windows) and "clip" not in prediction_unit(bouts))
