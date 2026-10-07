"""The plan line: how many windows a labelling run would score."""

from __future__ import annotations

from pathlib import Path


from trackanno.config import load_config
from trackanno.gui.hub import predict_plan_line
from trackanno.infer.densepredict import count_windows
from trackanno.project import write_config
from fixtures import build_case, pump

LENGTH = 50          # the fixture's clips.frames
OVERLAP = 10         # a stride that divides it


def test_the_window_count_is_the_row_count_a_dense_run_writes(root: Path) -> None:
    from trackanno.infer.densepredict import build_dense_features

    cfg, _ = build_case(root, "plan-count")
    n_windows, n_tracks = count_windows(cfg)
    dense = build_dense_features(cfg)

    assert n_windows == len(dense), f"planned {n_windows}, ran {len(dense)}"
    assert n_tracks == len(dense.groupby(["movie_id", "track_id"]))

    line = predict_plan_line(cfg)
    assert line == (f"{n_windows:,} windows of {LENGTH} frames every {LENGTH} "
                    f"over {n_tracks:,} tracks"), line


def test_an_overlapping_stride_counts_more_windows(root: Path) -> None:
    cfg, _ = build_case(root, "plan-overlap")
    tiled, _ = count_windows(cfg)
    write_config(cfg.path, {"predict.window_stride": OVERLAP})
    cfg = load_config(cfg.path)
    overlapped, _ = count_windows(cfg)
    assert overlapped > tiled, f"{overlapped} overlapping against {tiled} tiled"
    assert f"every {OVERLAP} over" in predict_plan_line(cfg), predict_plan_line(cfg)


def test_the_model_page_states_the_plan_and_keeps_it_current(app, root: Path) -> None:
    """The plan is computed on a worker and re-planned when the config is saved."""
    from trackanno.gui.hub import Hub, ModelPage

    cfg, _ = build_case(root, "plan-page", labels=2)
    hub = Hub(str(cfg.path))
    hub.show()
    page = hub.page(ModelPage)
    try:
        assert pump(app, lambda: "windows" in page.predict_note.text()), (
            page.predict_note.text())
        assert f"{count_windows(cfg)[0]:,} windows" in page.predict_note.text(), (
            page.predict_note.text())

        write_config(cfg.path, {"predict.window_stride": OVERLAP})
        hub.reload_project()
        assert pump(app, lambda: f"every {OVERLAP} over" in page.predict_note.text()), (
            f"a saved edit did not re-plan: {page.predict_note.text()}")
        moved = page.predict_note.text()
        assert moved.count("\n") <= 1, f"the note is more than two lines: {moved!r}"
    finally:
        page.shutdown()
        hub.close()
        app.processEvents()


def test_the_gate_and_the_plan_are_both_on_the_note(app, root: Path) -> None:
    """A gated-off model shows the gate line first, then the plan."""
    from trackanno.gui.hub import Hub, ModelPage

    cfg, _ = build_case(root, "plan-gated", reliable=False, labels=2)
    hub = Hub(str(cfg.path))
    hub.show()
    page = hub.page(ModelPage)
    try:
        assert pump(app, lambda: "windows" in page.predict_note.text()), (
            page.predict_note.text())
        lines = page.predict_note.text().splitlines()
        assert len(lines) == 2, repr(page.predict_note.text())
        assert "reliable enough" in lines[0], lines[0]
        assert "windows of" in lines[1], lines[1]
        assert not page.predict_button.isEnabled()
    finally:
        page.shutdown()
        hub.close()
        app.processEvents()
