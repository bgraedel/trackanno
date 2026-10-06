"""The line that says how many windows a labelling run would score.

The count has to be the run's count: :func:`~trackanno.infer.densepredict.count_windows`
is checked against the rows a real
:func:`~trackanno.infer.densepredict.build_dense_features` writes. The GUI half
is the same line through :class:`~trackanno.gui.hub.ModelPage`, planned on a
worker and re-planned when the config is saved.
"""

from __future__ import annotations

from pathlib import Path

from harness import check
from test_predict import build_case, pump

from trackanno.config import load_config
from trackanno.gui.hub import predict_plan_line
from trackanno.infer.densepredict import count_windows
from trackanno.project import write_config

LENGTH = 50          # the fixture's clips.frames
OVERLAP = 10         # a stride that divides it


def test_the_window_count_is_the_row_count_a_dense_run_writes(root: Path) -> None:
    from trackanno.infer.densepredict import build_dense_features

    cfg, _ = build_case(root, "plan-count")
    n_windows, n_tracks = count_windows(cfg)
    dense = build_dense_features(cfg)

    check("the count is the number of rows the run writes",
          n_windows == len(dense), f"planned {n_windows}, ran {len(dense)}")
    check("...over the same tracks",
          n_tracks == len(dense.groupby(["movie_id", "track_id"])))

    line = predict_plan_line(cfg)
    check("the line quotes the count, the length, the stride and the tracks",
          line == f"{n_windows:,} windows of {LENGTH} frames every {LENGTH} "
                  f"over {n_tracks:,} tracks", line)


def test_an_overlapping_stride_counts_more_windows(root: Path) -> None:
    cfg, _ = build_case(root, "plan-overlap")
    tiled, _ = count_windows(cfg)
    write_config(cfg.path, {"predict.window_stride": OVERLAP})
    cfg = load_config(cfg.path)
    overlapped, _ = count_windows(cfg)
    check("a fifth of the stride is several times the windows",
          overlapped > tiled, f"{overlapped} against {tiled} tiled")
    check("the line states the stride",
          f"every {OVERLAP} over" in predict_plan_line(cfg), predict_plan_line(cfg))


def test_the_model_page_states_the_plan_and_keeps_it_current(app, root: Path) -> None:
    """Planning reads tracks.parquet per movie, so it runs on a worker, and it
    moves when the config is saved."""
    from trackanno.gui.hub import Hub, ModelPage

    cfg, _ = build_case(root, "plan-page", labels=2)
    hub = Hub(str(cfg.path))
    hub.show()
    page = hub.page(ModelPage)
    try:
        check("the plan lands once the event loop turns",
              pump(app, lambda: "windows" in page.predict_note.text()),
              page.predict_note.text())
        check("...with this project's count",
              f"{count_windows(cfg)[0]:,} windows" in page.predict_note.text(),
              page.predict_note.text())

        write_config(cfg.path, {"predict.window_stride": OVERLAP})
        hub.reload_project()
        check("a saved edit re-plans rather than leaving the old line up",
              pump(app, lambda: f"every {OVERLAP} over" in page.predict_note.text()),
              page.predict_note.text())
        moved = page.predict_note.text()
        check("the note stays a status line of at most two lines",
              moved.count("\n") <= 1, repr(moved))
    finally:
        page.shutdown()
        hub.close()
        app.processEvents()


def test_the_gate_and_the_plan_are_both_on_the_note(app, root: Path) -> None:
    """A model the report gated off must not cost the reader the plan. The gate
    line comes first, because it is the one that says the button is dead."""
    from trackanno.gui.hub import Hub, ModelPage

    cfg, _ = build_case(root, "plan-gated", reliable=False, labels=2)
    hub = Hub(str(cfg.path))
    hub.show()
    page = hub.page(ModelPage)
    try:
        check("the plan lands",
              pump(app, lambda: "windows" in page.predict_note.text()),
              page.predict_note.text())
        lines = page.predict_note.text().splitlines()
        check("the note is two lines", len(lines) == 2, repr(page.predict_note.text()))
        check("...the gate first", "reliable enough" in lines[0], lines[0])
        check("...the plan second", "windows of" in lines[1], lines[1])
        check("the button is still dead", not page.predict_button.isEnabled())
    finally:
        page.shutdown()
        hub.close()
        app.processEvents()
