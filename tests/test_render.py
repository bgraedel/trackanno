"""What the renderer refuses to paint, and what it paints over.

Two bouts on one frame is a broken decode and raises; a hand label over a
prediction (``--source both``) and overlapping hand labels are allowed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


GROUP = "behaviour"
MOVIE = "MOVIE_A"


@pytest.fixture(scope="module")
def cfg(root: Path):
    """A config and nothing else: none of these claims touch a pixel or a cache."""
    from trackanno.config import load_config
    from trackanno.project import create_project

    return load_config(create_project(Path(root) / "render"))


def _bout(track_id: int, f_start: int, f_end: int, value: str, conf: float = 0.9):
    """A row the decode produced."""
    return {"movie_id": MOVIE, "track_id": track_id, "f_start": f_start,
            "f_end": f_end, f"{GROUP}_pred": value, f"{GROUP}_conf": conf,
            "bout_id": f"{MOVIE}__t{track_id:04d}__f{f_start:06d}-{f_end:06d}"}


def _label(track_id: int, f_start: int, f_end: int, value: str):
    return {"clip_id": f"{MOVIE}__t{track_id:04d}__f{f_start:06d}", "movie_id": MOVIE,
            "track_id": track_id, "f_start": f_start, "f_end": f_end, GROUP: value}


def test_two_decoded_bouts_on_one_frame_stop_the_render(cfg) -> None:
    """Painting one of the two would hide that they exist."""
    from trackanno.view.render import build_frame_marks

    predictions = pd.DataFrame([
        _bout(3, 800, 850, "immotile"),
        _bout(3, 812, 862, "swimming"),
    ])
    with pytest.raises(ValueError) as caught:
        build_frame_marks(pd.DataFrame(), predictions, cfg, GROUP, "predictions")
    message = str(caught.value)
    assert "frame 812" in message, message
    assert "track 3" in message, message
    assert "'immotile'" in message and "'swimming'" in message, message
    assert "[800, 850)" in message and "[812, 862)" in message, message
    assert "bout" in message.lower(), message

    with pytest.raises(ValueError):
        build_frame_marks(pd.DataFrame([_label(3, 800, 862, "swimming")]),
                          predictions, cfg, GROUP, "both")


def test_disjoint_bouts_paint_without_complaint(cfg) -> None:
    """The guard defends disjointness, not equal length."""
    from trackanno.view.render import build_frame_marks

    predictions = pd.DataFrame([
        _bout(3, 0, 137, "swimming"),
        _bout(3, 137, 300, "immotile"),
        _bout(3, 300, 412, "swimming"),
        _bout(4, 50, 412, "unclear"),        # another track, same frames
    ])
    marks = build_frame_marks(pd.DataFrame(), predictions, cfg, GROUP, "predictions")

    assert len(marks) == 412, "bouts that only touch at their boundary paint fine"
    assert marks[136][3].text.startswith("~swimming"), (
        f"the frame before the boundary is the earlier bout: {marks[136][3].text}")
    assert marks[137][3].text.startswith("~immotile"), (
        f"the boundary frame itself is the later bout: {marks[137][3].text}")
    assert 412 not in marks, "a half-open f_end paints nothing at f_end"
    assert marks[200][4].text.startswith("~unclear"), (
        f"two tracks sharing a frame are not a collision: {marks[200][4].text}")


def test_a_hand_label_over_a_prediction_is_not_a_collision(cfg) -> None:
    """The label takes the box and the model's answer stays in the caption."""
    from trackanno.view.render import build_frame_marks

    labels = pd.DataFrame([_label(3, 100, 150, "swimming")])
    predictions = pd.DataFrame([_bout(3, 100, 150, "immotile", 0.73)])

    marks = build_frame_marks(labels, predictions, cfg, GROUP, "both")
    mark = marks[120][3]
    assert not mark.predicted, "a label on a predicted frame paints rather than raising"
    assert "swimming" in mark.text, f"taking the box for itself: {mark.text}"
    assert "~immotile" in mark.text, f"keeping the model's answer visible: {mark.text}"


def test_overlapping_annotations_are_a_configuration_not_a_fault(cfg) -> None:
    """``clips.stride`` below ``clips.frames`` overlaps clips on purpose; the later row wins."""
    from trackanno.view.render import build_frame_marks

    labels = pd.DataFrame([
        _label(3, 100, 150, "swimming"),
        _label(3, 125, 175, "immotile"),
    ])
    marks = build_frame_marks(labels, None, cfg, GROUP, "labels")
    assert len(marks) == 75, "overlapping hand labels still render"
    assert marks[130][3].text == "immotile", (
        f"with the later row winning the shared frames: {marks[130][3].text}")


@pytest.fixture
def hidden(tmp_path: Path, monkeypatch):
    """A config with ``render.hide_unlabelled`` on and one labelled movie."""
    from trackanno.annotate.clips import TRACKS_FILE
    from trackanno.config import load_config
    from trackanno.project import create_project
    from trackanno.view import render

    cfg = load_config(create_project(tmp_path / "render_hide"))
    cfg.render.hide_unlabelled = True
    cfg.paths.csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([_label(3, 0, 10, "swimming")]).to_csv(cfg.paths.csv, index=False)
    cfg.movie_dir(MOVIE).mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"track_id": [3]}).to_parquet(cfg.movie_dir(MOVIE) / TRACKS_FILE)

    class _Movie:
        def close(self) -> None:
            pass

    monkeypatch.setattr(render, "open_movie", lambda *a, **k: _Movie())
    return cfg


def test_hide_unlabelled_argument_overrides_the_config(hidden, monkeypatch) -> None:
    """None follows ``render.hide_unlabelled``; an explicit value wins."""
    from trackanno.view import render

    seen: list = []
    monkeypatch.setattr(render, "render_movie",
                        lambda *a, **k: seen.append(k["hide_unlabelled"]))
    for given in (None, False, True):
        render.render_all(hidden, [MOVIE], GROUP, hide_unlabelled=given)
    assert seen == [True, False, True]


def test_the_cli_passes_none_without_a_flag(hidden, monkeypatch) -> None:
    from trackanno.view import render

    args: list = []
    monkeypatch.setattr(render, "load_config", lambda _path: hidden)
    monkeypatch.setattr(render, "render_all",
                        lambda *a, **k: args.append(k["hide_unlabelled"]))
    for argv in ([], ["--no-hide-unlabelled"], ["--hide-unlabelled"]):
        render.main(argv)
    assert args == [None, False, True]


def test_a_malformed_csv_is_reported_not_raised(root: Path) -> None:
    """A stray comma in the CSV must not stop the project opening."""
    import warnings

    from trackanno.config import load_config
    from trackanno.project import create_project
    from trackanno.view.render import annotated_movies

    cfg = load_config(create_project(Path(root) / "badcsv"))
    cfg.paths.csv.write_text(
        "clip_id,movie_id,behaviour\na,m,swim\nb,m,swim,EXTRA,COLUMNS\n",
        encoding="utf-8")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = annotated_movies(cfg)
    assert isinstance(result, dict)
    assert any("could not read" in str(m.message) for m in caught), (
        [str(m.message) for m in caught])
