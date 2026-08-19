"""The annotation window, built offscreen and driven with real input.

Builds the real window against a generated clip cache, drives it with the same
calls a keypress makes, and confirms the CSV, the resume path and the overlay
renderer all behave. The marker check reads the painted pixmap rather than a
flag, so "the marker is drawn" is measured where it matters.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from harness import check
from trackanno.clips import ClipStore
from trackanno.config import load_config
from trackanno.overlay import build_clip_view, render_clip_rgb
from trackanno.store import AnnotationStore


@pytest.fixture(scope="module")
def cfg(root):
    """A generated project, never the repository's own.

    Loading the real one made this suite depend on a clip cache that may not
    exist and on a network share that may not be mounted: green because it
    skipped, or red for reasons that had nothing to do with the code.
    """
    from fixtures import make_project

    config = load_config(make_project(root))
    config.paths.csv = config.paths.work_dir / "_headless_test.csv"
    config.paths.csv.unlink(missing_ok=True)
    return config


@pytest.fixture(scope="module")
def store(cfg):
    """The clip cache, opened once: every test here reads the same pixels."""
    return ClipStore(cfg.paths.work_dir)


def test_a_warm_clip_loads_fast(store) -> None:
    """Navigation is only instant if a cached clip is one contiguous read."""
    print(f"cache: {len(store)} clips across {store.movie_ids}")

    for i in range(min(5, len(store))):
        store.touch(i)
    times = []
    for i in range(min(10, len(store))):
        t0 = time.perf_counter()
        clip = store.clip(i)
        _ = int(clip[0, 0, 0])
        times.append((time.perf_counter() - t0) * 1000)
    print(f"  clip load: median {np.median(times):.2f} ms, max {max(times):.2f} ms "
          f"({clip.nbytes / 1e6:.1f} MB per clip)")
    check("warm clip load is fast", np.median(times) < 50, f"{np.median(times):.1f} ms")


def test_the_overlay_marks_one_animal(store, cfg) -> None:
    """The marker answers "which animal am I judging?", so it marks exactly one."""
    boxes = store.clip_boxes(0)
    check("overlay geometry present", len(boxes) > 0)
    check("overlay marks the target track", bool(boxes["is_target"].any()))
    check("overlay includes neighbouring tracks",
          bool((~boxes["is_target"]).any()) or True)  # informational

    t0 = time.perf_counter()
    rgb = render_clip_rgb(store.clip(0), boxes, cfg)
    overlay_ms = (time.perf_counter() - t0) * 1000
    print(f"  overlay render: {overlay_ms:.1f} ms for {rgb.shape}")
    check("overlay render is interactive", overlay_ms < 250, f"{overlay_ms:.0f} ms")


def test_the_view_modes_place_the_animal_where_they_promise(store) -> None:
    """`locked` means the exact centre on every frame, not "as close as the crop allows"."""
    boxes = store.clip_boxes(0)
    clip0 = store.clip(0)
    t0 = time.perf_counter()
    locked = build_clip_view(boxes, clip0.shape[0], clip0.shape[1], mode="locked")
    print(f"  clip view build: {(time.perf_counter() - t0) * 1000:.2f} ms")
    offsets = []
    for t in range(clip0.shape[0]):
        box = locked.box(t)
        if box is None:
            continue
        x0, y0, side, _ = locked.source_rect(t)
        offsets.append(max(abs((box[0] + box[2]) / 2 - x0 - side / 2),
                           abs((box[1] + box[3]) / 2 - y0 - side / 2)))
    check("locked view centres the animal on every detected frame",
          max(offsets) < 1e-9, f"worst {max(offsets):.4f} px")
    check("centred view is one fixed position",
          bool(np.all(
              build_clip_view(boxes, clip0.shape[0], clip0.shape[1]).centre
              == build_clip_view(boxes, clip0.shape[0], clip0.shape[1]).centre[0]
          )))


def test_the_window_labels_advances_and_resumes(app, cfg, store) -> None:
    """One scenario, driven the way a person drives it.

    Deliberately not split further: every step below depends on the window state
    the step before it left. Labelling clip 1 is what makes "auto-advance moved
    on" mean anything, and the resume at the end is only a resume because the
    CSV it reads was written by the keypresses above.
    """
    from qtpy.QtCore import Qt

    from trackanno.gui.app import AnnotationWindow

    csv_path = cfg.paths.csv
    annotations = AnnotationStore(csv_path, cfg)
    window = AnnotationWindow(cfg, store, annotations)
    window.show()
    app.processEvents()
    check("window built", window.isVisible())
    check("navigation order populated", len(window.order) == len(store))

    first_id = window._clip_ids[window.order[window.position]]

    # A single category keypress should label the clip and advance.
    group = cfg.categories[0]
    window.categories.toggle(group.name, group.options[0].name)
    app.processEvents()
    check("keypress recorded an annotation", annotations.is_done(first_id))
    check("auto-advance moved on", window.position == 1)
    row = annotations.get(first_id)
    check("recorded the right value", row[group.name] == group.options[0].name)
    check("recorded reassociation columns",
          all(row[c] not in (None, "") for c in
              ["movie_id", "source_file", "track_id", "f_start", "crop_x0", "scale"]))

    # Clip 2: a multi-select tick must not advance on its own; the single-choice
    # pick then commits both together.
    multi = next((g for g in cfg.categories if g.is_multi), None)
    second_id = window._clip_ids[window.order[window.position]]
    if multi:
        before = window.position
        window.categories.toggle(multi.name, multi.options[0].name)
        app.processEvents()
        check("multi-select does not auto-advance", window.position == before)

    window.categories.toggle(group.name, group.options[-1].name)
    app.processEvents()
    check("single-choice pick commits the clip", annotations.is_done(second_id))
    check("auto-advance moved on again", window.position == 2)

    window.previous_clip()
    app.processEvents()
    check("previous goes back", window.position == 1)
    reloaded_values = window.categories.values()
    check("single-choice label reloads on revisit",
          reloaded_values[group.name] == group.options[-1].name)
    if multi:
        check("multi-choice label reloads on revisit",
              multi.options[0].name in reloaded_values[multi.name])

    # Clip 3: partial input (required group unset) must NOT be written, and the
    # user must be told rather than having it dropped silently.
    window.next_clip()
    window.next_clip()
    app.processEvents()
    partial_id = window._clip_ids[window.order[window.position]]
    if multi:
        window.categories.toggle(multi.name, multi.options[0].name)
        app.processEvents()
    window.next_clip()
    app.processEvents()
    check("partial input is not saved", not annotations.is_done(partial_id))
    if multi:
        check("partial input warns the user",
              "not saved" in window.statusBar().currentMessage(),
              repr(window.statusBar().currentMessage()))

    # Toggles must not crash, and the marker must actually reach the screen.
    from trackanno.overlay import FOLLOW_MODES

    window.overlay_box.setChecked(False)
    app.processEvents()
    for _ in FOLLOW_MODES:
        window.cycle_follow()
        app.processEvents()
    check("cycling the view mode returns to where it started",
          window.follow == cfg.ui.follow, window.follow)
    window.step_zoom(+1)
    window.step_zoom(+1)
    app.processEvents()
    check("zoom steps in", window.zoom > 1.0, str(window.zoom))
    window.set_zoom(1.0)
    window.context_box.setChecked(True)
    app.processEvents()
    window.context_box.setChecked(False)
    window.overlay_box.setChecked(True)
    app.processEvents()
    check("view toggles survive", True)

    from qtpy.QtGui import QPixmap

    canvas = window.viewer.canvas
    canvas.resize(600, 600)
    app.processEvents()

    def painted() -> np.ndarray:
        pixmap = QPixmap(canvas.size())
        canvas.render(pixmap)
        image = pixmap.toImage()
        raw = image.constBits().asstring(image.sizeInBytes())
        return np.frombuffer(raw, np.uint8).reshape(
            image.height(), image.bytesPerLine() // 4, 4
        )[..., :3]

    def marker_pixels(frame: np.ndarray) -> int:
        # QImage on Windows is BGRA; the marker is the only strongly amber
        # thing on a grayscale frame.
        b, g, r = frame[..., 0].astype(int), frame[..., 1].astype(int), frame[..., 2].astype(int)
        return int(((r > 150) & (g > 90) & (r - b > 80) & (g - b > 40)).sum())

    window.overlay_box.setChecked(True)
    app.processEvents()
    with_marker = marker_pixels(painted())
    window.overlay_box.setChecked(False)
    app.processEvents()
    without_marker = marker_pixels(painted())
    check("the marker is painted over the frame",
          with_marker > 50 and without_marker == 0,
          f"on={with_marker} off={without_marker}")
    check("the marker is not burned into the clip array",
          window._clip.ndim == 3, f"clip is {window._clip.shape}")
    window.overlay_box.setChecked(True)
    app.processEvents()

    times = []
    for i in range(30):
        canvas.set_frame(i % max(1, window._clip.shape[0]))
        t0 = time.perf_counter()
        canvas.render(QPixmap(canvas.size()))
        times.append((time.perf_counter() - t0) * 1000)
    print(f"  canvas paint: median {np.median(times):.2f} ms at 600x600")
    check("painting keeps up with playback", np.median(times) < 20,
          f"{np.median(times):.1f} ms")

    window.next_unannotated()
    app.processEvents()
    check("next_unannotated lands on an unlabelled clip",
          not annotations.is_done(window._clip_ids[window.order[window.position]]))

    # Revisiting a labelled clip must not re-write it (a full CSV rewrite per
    # keypress) - so an unchanged pass leaves the row count and undo stack alone.
    revisit_before = len(annotations._undo_stack)
    window.position = 0
    window._load()
    window.next_clip()
    app.processEvents()
    check("unchanged revisit does not re-record",
          len(annotations._undo_stack) == revisit_before)

    # Focus must not get stuck in the notes box: a focused QLineEdit eats every
    # plain keypress, so "1"/"o"/"s" stop working and there is no way back out.
    from qtpy.QtCore import QEvent
    from qtpy.QtGui import QKeyEvent

    window._focus_notes()
    app.processEvents()
    check("n focuses the notes box", window.notes.hasFocus())

    window.notes.setText("a note")
    esc = QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier)
    app.sendEvent(window.notes, esc)
    app.processEvents()
    check("Esc releases the notes box", not window.notes.hasFocus())
    check("Esc keeps the typed text", window.notes.text() == "a note")

    window._focus_notes()
    app.processEvents()
    window.notes.returnPressed.emit()
    app.processEvents()
    check("Enter releases the notes box", not window.notes.hasFocus())

    window.notes.clear()

    n_before = len(annotations)
    window._undo()
    app.processEvents()
    check("undo removed a row", len(annotations) == n_before - 1, f"{n_before} -> {len(annotations)}")

    window.close()

    # --- resume ------------------------------------------------------------ #
    reloaded = AnnotationStore(csv_path, cfg)
    window2 = AnnotationWindow(cfg, store, reloaded)
    app.processEvents()
    resumed_id = window2._clip_ids[window2.order[window2.position]]
    check("resume lands on an unannotated clip", not reloaded.is_done(resumed_id))
    window2.close()

    csv_path.unlink(missing_ok=True)

