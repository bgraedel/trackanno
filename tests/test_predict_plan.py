"""The line that says what a labelling run would do, before it does it.

Five keys constrain each other - ``predict.coverage``, ``window_frames``,
``window_stride``, ``window_scales`` and ``decode`` - and the only feedback used
to be a refusal after guessing wrong, or a run twenty times slower than the last
one with nothing in the window to pin it on. The Model page states the plan
instead, and the whole value of that line is that its numbers are the run's
numbers. So the claims here are about arithmetic, not about phrasing:

* the window count the line quotes is the row count a real
  :func:`~trackanno.infer.densepredict.build_dense_features` produces, not an
  estimate of it;
* the windows-per-frame it quotes is counted the long way here - every detected
  frame against every window of the grid - so a change to the planner cannot
  move the claim and the check together;
* a config that will not load gets the refusal, never a plan that cannot run.

The GUI half is the same line again through
:class:`~trackanno.gui.hub.ModelPage`: on a worker, so the note is a placeholder
until the event loop turns, and re-planned when the config is saved.

The projects come from ``test_predict.build_case``, the fixture that already
carries a clip manifest, clip features and a stubbed model, so a dense run here
is a real one. Nothing below states a wall time, because nothing below has
measured one.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from harness import check
from test_predict import build_case, pump

from trackanno.config import load_config
from trackanno.gui.hub import predict_plan_line
from trackanno.infer.densepredict import (
    plan_dense_run, plan_windows, scale_grid, window_grid,
)
from trackanno.project import write_config

LENGTH = 50          # what the fixture's clips.frames resolves 'auto' to
OVERLAP = 10         # a stride that divides it, so length/stride is exact


def hits_over_detections(cfg) -> tuple[int, int, int]:
    """(detected frames, total window hits, the busiest frame's hit count).

    Counted the long way, off ``tracks.parquet`` and
    :func:`~trackanno.infer.densepredict.window_grid` alone: every detected frame
    against every window of the movie's grid. The planner clips the grid to each
    track's lifetime, but a window holding a detected frame necessarily overlaps
    that lifetime, so the two are counting the same hits.
    """
    grid_spec = scale_grid(cfg)
    detected = total = most = 0
    for movie_dir in sorted(p for p in cfg.paths.work_dir.iterdir() if p.is_dir()):
        table = movie_dir / "tracks.parquet"
        if not table.exists():
            continue
        tracks = pd.read_parquet(table, columns=["track_id", "frame"])
        if tracks.empty:
            continue
        header = json.loads((movie_dir / "cache.json").read_text(encoding="utf-8"))
        n_frames = max(int(header["movie"]["n_frames"]),
                       int(tracks["frame"].max()) + 1)
        grid = [w for scale, step in grid_spec
                for w in window_grid(n_frames, scale, step)]
        for _track_id, tdf in tracks.groupby("track_id", sort=True):
            frames = tdf["frame"].to_numpy(dtype=np.int64)
            hits = np.zeros(frames.size, dtype=np.int64)
            for f_start, f_end in grid:
                hits += ((frames >= f_start) & (frames < f_end)).astype(np.int64)
            detected += frames.size
            total += int(hits.sum())
            most = max(most, int(hits.max()))
    return detected, total, most


def spans(frame: pd.DataFrame) -> list:
    return [(str(m), int(t), int(a), int(b)) for m, t, a, b in zip(
        frame["movie_id"], frame["track_id"], frame["f_start"], frame["f_end"])]


# --------------------------------------------------------------------------- #
# the count is the run's count
# --------------------------------------------------------------------------- #
def test_the_window_count_is_the_row_count_a_dense_run_writes(root: Path) -> None:
    """The whole point of stating the plan is that it IS the plan. A count that
    was only about right would be worse than no count, because it is exactly the
    number somebody would use to decide whether to leave a run going overnight."""
    from trackanno.infer.densepredict import build_dense_features

    cfg, _ = build_case(root, "plan-count", coverage="dense")
    plan = plan_windows(cfg)
    dense = build_dense_features(cfg)

    check("the plan has a row per window the run writes",
          len(plan) == len(dense), f"planned {len(plan)}, ran {len(dense)}")
    check("...over the same movies, tracks and spans, in the same order",
          spans(plan) == spans(dense), str(spans(plan)[:3]))
    check("the plan cost no features to make",
          "speed_mean" not in plan.columns, str(list(plan.columns)))

    summary = plan_dense_run(cfg)
    check("the summary counts the same windows",
          summary.n_windows == len(dense), f"{summary.n_windows} vs {len(dense)}")
    check("...over the same movies",
          summary.n_movies == dense["movie_id"].nunique())

    line = predict_plan_line(cfg.path)
    check("the line on the Model page quotes that number",
          f"{len(dense):,}" in line, line)
    check("...states the window length", f"{LENGTH}-frame windows" in line, line)
    check("...says the default grid tiles", "tiled" in line, line)
    check("...and that one window then speaks for each frame",
          "One window per frame" in line, line)
    check("nothing is claimed about how long it will take",
          not any(w in line.lower() for w in ("minute", "second", "hour")), line)


def test_manifest_coverage_says_it_is_not_covering_every_frame(root: Path) -> None:
    """The other coverage, and the one the shipped config starts on. It scores
    the clips cut for a human's attention, so the frames between two clips come
    back unlabelled - which is the thing 'coverage' is choosing between."""
    cfg, manifest = build_case(root, "plan-manifest", coverage="manifest")
    line = predict_plan_line(cfg.path)
    check("the line counts the clips it will score",
          f"{len(manifest):,} annotation clips" in line, line)
    check("...and says what is left out", "stay unlabelled" in line, line)
    check("...naming the setting that would close the holes",
          "predict.coverage: dense" in line, line)
    check("no window grid is described, because there is none",
          "windows per frame" not in line, line)


# --------------------------------------------------------------------------- #
# how many windows speak for a frame
# --------------------------------------------------------------------------- #
def test_an_overlapping_stride_reports_the_windows_per_frame(root: Path) -> None:
    """A shorter stride is the setting that costs the hours, and it costs them
    because several windows come to speak for every frame. That multiple is what
    cannot be worked out from the config, so it is the number the line has to get
    right - checked here against a brute-force sweep of the grid rather than
    against the planner that produced it."""
    cfg, _ = build_case(root, "plan-overlap", coverage="dense")
    tiled = plan_dense_run(cfg)

    write_config(cfg.path, {
        "predict.window_frames": LENGTH,
        "predict.window_stride": OVERLAP,
        "predict.decode": "min_bout",
        "predict.min_bout_frames": 25,
    })
    cfg = load_config(cfg.path)
    plan = plan_dense_run(cfg)
    detected, total, most = hits_over_detections(cfg)

    check("the plan and a brute-force sweep agree on the detected frames",
          plan.detected_frames == detected, f"{plan.detected_frames} vs {detected}")
    check("...and on how many windows reach them",
          abs(plan.windows_per_frame - total / detected) < 1e-9,
          f"{plan.windows_per_frame} vs {total / detected}")
    check("...and on the busiest frame",
          plan.windows_per_frame_max == most,
          f"{plan.windows_per_frame_max} vs {most}")
    check("no detected frame is left without a window",
          plan.uncovered == 0, str(plan.uncovered))
    check("the overlap is real: more than one window speaks for a frame",
          plan.windows_per_frame > 1.0, str(plan.windows_per_frame))
    check("...and no more than the grid can put there",
          plan.windows_per_frame <= LENGTH // OVERLAP + 1,
          f"{plan.windows_per_frame} against length/stride {LENGTH // OVERLAP}")
    check("a fifth of the stride is several times the windows",
          plan.n_windows > tiled.n_windows,
          f"{plan.n_windows} against {tiled.n_windows} tiled")

    line = predict_plan_line(cfg.path)
    check("the line states the stride the grid will walk",
          f"every {OVERLAP} frames" in line, line)
    check("...the windows per frame that buys",
          f"{plan.windows_per_frame:.1f} windows per frame" in line, line)
    check("...how many windows that is in this project",
          f"{plan.n_windows:,} in this project" in line, line)
    check("...and that the overlap is collapsed by a decode",
          "bouts of at least 25 frames" in line, line)


def test_several_scales_name_their_own_grids(root: Path) -> None:
    """Each scale walks its own grid at its own stride, proportional to its
    length, so one stride cannot speak for all of them. Naming a single number
    here would misreport every scale but the base."""
    cfg, _ = build_case(root, "plan-scales", coverage="dense")
    write_config(cfg.path, {
        "predict.window_frames": LENGTH,
        "predict.window_scales": [LENGTH, 2 * LENGTH],
        "predict.decode": "min_bout",
        "predict.min_bout_frames": 25,
        "suggest.feature_scope": "length_invariant",
    })
    cfg = load_config(cfg.path)
    plan = plan_dense_run(cfg)
    line = predict_plan_line(cfg.path)

    check("both window lengths are named",
          f"{LENGTH} and {2 * LENGTH}-frame windows" in line, line)
    check("...and both tile, so the line says so once",
          "each tiled" in line, line)
    check("two scales put two windows over each frame",
          abs(plan.windows_per_frame - 2.0) < 0.5, str(plan.windows_per_frame))
    check("...which the line quotes",
          f"{plan.windows_per_frame:.1f} windows per frame" in line, line)


# --------------------------------------------------------------------------- #
# a plan that cannot run is not a plan
# --------------------------------------------------------------------------- #
def test_a_config_that_will_not_load_shows_the_refusal_not_a_plan(root: Path) -> None:
    """``load_config`` refuses an overlapping stride without a decode, and writes
    the reason for it. A note answering with the previous values instead would be
    describing a run that cannot start - and this is the one state where the
    window has no Config to plan from at all."""
    cfg, _ = build_case(root, "plan-refused", coverage="dense")
    good = predict_plan_line(cfg.path)
    check("the plan is there while the config loads",
          "windows" in good and "will not load" not in good, good)

    # Hand-edited, because write_config validates and would refuse to write it.
    text = cfg.path.read_text(encoding="utf-8")
    edited = text.replace("window_stride: auto", f"window_stride: {OVERLAP}")
    check("the fixture had a stride to break", edited != text)
    cfg.path.write_text(edited, encoding="utf-8")

    line = predict_plan_line(cfg.path)
    check("a config that will not load says so", "will not load" in line, line)
    check("...and carries the refusal's own words",
          "predict.window_stride" in line and "predict.decode" in line, line)
    check("...instead of a plan", "windows per frame" not in line, line)
    check("...and stays a status line, not the whole argument",
          len(line) < 260 and "\n" not in line, f"{len(line)} chars: {line}")

    broken = cfg.path.parent / "broken.yaml"
    broken.write_text("paths:\n  : : not yaml\n", encoding="utf-8")
    unparseable = predict_plan_line(broken)
    check("an unparseable file is the same news to whoever must fix it",
          "will not load" in unparseable, unparseable)


# --------------------------------------------------------------------------- #
# the same line, through the window
# --------------------------------------------------------------------------- #
def test_the_model_page_states_the_plan_and_keeps_it_current(app, root: Path) -> None:
    """Planning reads tracks.parquet per movie, so it runs on a worker: the note
    holds a placeholder until the event loop turns. It has to move when the
    config is saved, which is the reload ConfigPage.save performs, or the line
    would quietly describe the settings from before the edit."""
    from trackanno.gui.hub import Hub, ModelPage

    cfg, _ = build_case(root, "plan-page", coverage="dense", labels=2)
    hub = Hub(str(cfg.path))
    hub.show()
    page = hub.page(ModelPage)
    try:
        check("nothing is planned on the UI thread",
              "working out" in page.predict_note.text(), page.predict_note.text())
        check("the plan lands once the event loop turns",
              pump(app, lambda: "windows" in page.predict_note.text()),
              page.predict_note.text())
        tiled = page.predict_note.text()
        check("...saying the windows tile", "tiled" in tiled, tiled)
        check("...and how many there are in this project",
              f"{plan_dense_run(cfg).n_windows:,} in this project" in tiled, tiled)

        # What ConfigPage.save does: write the file, then reload the project.
        write_config(cfg.path, {
            "predict.window_frames": LENGTH,
            "predict.window_stride": OVERLAP,
            "predict.decode": "min_bout",
            "predict.min_bout_frames": 25,
        })
        hub.reload_project()
        check("a saved edit re-plans rather than leaving the old line up",
              pump(app, lambda: f"every {OVERLAP} frames" in page.predict_note.text()),
              page.predict_note.text())
        moved = page.predict_note.text()
        check("...and the new count is the new grid's",
              f"{plan_dense_run(load_config(cfg.path)).n_windows:,} in this project"
              in moved, moved)
        check("the note stays a status line of at most two lines",
              moved.count("\n") <= 1, repr(moved))
    finally:
        page.shutdown()
        hub.close()
        app.processEvents()


def test_the_gate_and_the_plan_are_both_on_the_note(app, root: Path) -> None:
    """Two different questions - may this run, and what would it do - so a model
    the report gated off must not cost the reader the plan. The gate line comes
    first, because it is the one that says the button is dead."""
    from trackanno.gui.hub import Hub, ModelPage

    cfg, _ = build_case(root, "plan-gated", coverage="dense", reliable=False, labels=2)
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
        check("...the plan second", "tiled" in lines[1], lines[1])
        check("the button is still dead", not page.predict_button.isEnabled())
    finally:
        page.shutdown()
        hub.close()
        app.processEvents()
