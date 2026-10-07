"""The annotation screen, built offscreen and driven with the calls a keypress makes."""

from __future__ import annotations

import time

import numpy as np
import pytest

from trackanno.annotate.clips import ClipStore
from trackanno.annotate.store import AnnotationStore
from trackanno.config import load_config
from trackanno.view.overlay import build_clip_view, render_clip_rgb


@pytest.fixture(scope="module")
def cfg(root):
    from fixtures import make_project

    return load_config(make_project(root))


@pytest.fixture(scope="module")
def store(cfg):
    """The clip cache, opened once and closed on teardown so later suites can rewrite it."""
    store = ClipStore(cfg.paths.work_dir)
    yield store
    store.close()


@pytest.fixture
def window(app, cfg, store, tmp_path, monkeypatch):
    """A shown annotation window writing to a fresh CSV; ``window.messages`` collects status."""
    from trackanno.gui.app import AnnotationWidget

    monkeypatch.setattr(cfg.paths, "csv", tmp_path / "annotations.csv")
    window = AnnotationWidget(cfg, store, AnnotationStore(cfg.paths.csv, cfg))
    window.messages = []
    window.status_message.connect(lambda text, _timeout: window.messages.append(text))
    window.show()
    app.processEvents()
    yield window
    window.shutdown()
    window.close()


def _current_id(window) -> str:
    return window._clip_ids[window.order[window.position]]


# --------------------------------------------------------------------------- #
# the clip cache and the overlay
# --------------------------------------------------------------------------- #
def test_a_warm_clip_loads_fast(store) -> None:
    """Navigation is only instant if a cached clip is one contiguous read."""
    for i in range(min(5, len(store))):
        store.touch(i)
    times = []
    for i in range(min(10, len(store))):
        t0 = time.perf_counter()
        clip = store.clip(i)
        _ = int(clip[0, 0, 0])
        times.append((time.perf_counter() - t0) * 1000)
    assert np.median(times) < 50, f"warm clip load: {np.median(times):.1f} ms"


def test_the_overlay_marks_the_target_quickly(store, cfg) -> None:
    boxes = store.clip_boxes(0)
    assert len(boxes) > 0, "overlay geometry present"
    assert bool(boxes["is_target"].any()), "the target track is marked"
    t0 = time.perf_counter()
    render_clip_rgb(store.clip(0), boxes, cfg)
    overlay_ms = (time.perf_counter() - t0) * 1000
    assert overlay_ms < 250, f"overlay render: {overlay_ms:.0f} ms"


def test_the_view_modes_place_the_animal_where_they_promise(store) -> None:
    """`locked` is the exact centre on every frame; `centred` is one fixed position."""
    boxes = store.clip_boxes(0)
    clip0 = store.clip(0)
    locked = build_clip_view(boxes, clip0.shape[0], clip0.shape[1], mode="locked")
    offsets = []
    for t in range(clip0.shape[0]):
        box = locked.box(t)
        if box is None:
            continue
        x0, y0, side, _ = locked.source_rect(t)
        offsets.append(max(abs((box[0] + box[2]) / 2 - x0 - side / 2),
                           abs((box[1] + box[3]) / 2 - y0 - side / 2)))
    assert max(offsets) < 1e-9, f"locked view is off centre by {max(offsets):.4f} px"
    centred = build_clip_view(boxes, clip0.shape[0], clip0.shape[1]).centre
    assert bool(np.all(centred == centred[0]))


# --------------------------------------------------------------------------- #
# labelling
# --------------------------------------------------------------------------- #
def test_a_keypress_labels_the_clip_and_advances(app, cfg, store, window) -> None:
    group = cfg.categories[0]
    first_id = _current_id(window)
    assert len(window.order) == len(store), "navigation order populated"
    window.categories.toggle(group.name, group.options[0].name)
    app.processEvents()
    assert window.annotations.is_done(first_id)
    assert window.position == 1, "auto-advance moved on"
    row = window.annotations.get(first_id)
    assert row[group.name] == group.options[0].name
    assert all(row[c] not in (None, "") for c in
               ["movie_id", "source_file", "track_id", "f_start", "crop_x0", "scale"]), (
        "the reassociation columns are recorded")


def test_a_multi_tick_waits_for_the_single_pick(app, cfg, window) -> None:
    group = cfg.categories[0]
    multi = next(g for g in cfg.categories if g.is_multi)
    clip_id = _current_id(window)
    window.categories.toggle(multi.name, multi.options[0].name)
    app.processEvents()
    assert window.position == 0, "a multi-select tick does not advance"
    window.categories.toggle(group.name, group.options[-1].name)
    app.processEvents()
    assert window.annotations.is_done(clip_id) and window.position == 1

    window.previous_clip()
    app.processEvents()
    assert window.position == 0
    values = window.categories.values()
    assert values[group.name] == group.options[-1].name, "the single choice reloads"
    assert multi.options[0].name in values[multi.name], "the multi choice reloads"


def test_partial_input_is_not_saved_and_says_so(app, cfg, window) -> None:
    multi = next(g for g in cfg.categories if g.is_multi)
    clip_id = _current_id(window)
    window.categories.toggle(multi.name, multi.options[0].name)
    app.processEvents()
    window.next_clip()
    app.processEvents()
    assert not window.annotations.is_done(clip_id)
    assert any("not saved" in m for m in window.messages[-3:]), window.messages[-3:]


def test_next_unannotated_skips_labelled_clips(app, cfg, window) -> None:
    group = cfg.categories[0]
    for pos in (0, 1):
        window.position = pos
        window._load()
        window.categories.toggle(group.name, group.options[0].name)
        window._commit()
        app.processEvents()
    assert all(window.annotations.is_done(window._clip_ids[window.order[p]]) for p in (0, 1))
    window.position = 0
    window._load()
    window.next_unannotated()
    app.processEvents()
    assert window.position == 2
    assert not window.annotations.is_done(_current_id(window))


def test_an_unchanged_revisit_does_not_rewrite(app, cfg, window) -> None:
    group = cfg.categories[0]
    window.categories.toggle(group.name, group.options[0].name)
    app.processEvents()
    before = len(window.annotations._undo_stack)
    window.position = 0
    window._load()
    window.next_clip()
    app.processEvents()
    assert len(window.annotations._undo_stack) == before


def test_undo_removes_the_last_row(app, cfg, window) -> None:
    group = cfg.categories[0]
    window.categories.toggle(group.name, group.options[0].name)
    app.processEvents()
    n_before = len(window.annotations)
    window._undo()
    app.processEvents()
    assert len(window.annotations) == n_before - 1


def test_a_reopened_window_resumes_on_an_unlabelled_clip(app, cfg, store, window) -> None:
    from trackanno.gui.app import AnnotationWidget

    group = cfg.categories[0]
    for _ in range(2):
        window.categories.toggle(group.name, group.options[0].name)
        app.processEvents()
    reloaded = AnnotationStore(cfg.paths.csv, cfg)
    again = AnnotationWidget(cfg, store, reloaded)
    try:
        app.processEvents()
        assert not reloaded.is_done(_current_id(again))
    finally:
        again.shutdown()
        again.close()


# --------------------------------------------------------------------------- #
# the view
# --------------------------------------------------------------------------- #
def test_view_toggles_cycle_and_zoom(app, window) -> None:
    from trackanno.view.overlay import FOLLOW_MODES

    started = window.follow
    for _ in FOLLOW_MODES:
        window.cycle_follow()
        app.processEvents()
    assert window.follow == started, "cycling the view mode returns to the start"
    window.step_zoom(+1)
    window.step_zoom(+1)
    app.processEvents()
    assert window.zoom > 1.0, window.zoom
    window.set_zoom(1.0)
    window.context_box.setChecked(True)
    app.processEvents()
    window.context_box.setChecked(False)
    app.processEvents()


def _painted(canvas) -> np.ndarray:
    from qtpy.QtGui import QPixmap

    pixmap = QPixmap(canvas.size())
    canvas.render(pixmap)
    image = pixmap.toImage()
    raw = image.constBits().asstring(image.sizeInBytes())
    return np.frombuffer(raw, np.uint8).reshape(
        image.height(), image.bytesPerLine() // 4, 4)[..., :3]


def _marker_pixels(frame: np.ndarray) -> int:
    """Strongly amber pixels (BGRA order): the marker on a grayscale frame."""
    b, g, r = (frame[..., i].astype(int) for i in range(3))
    return int(((r > 150) & (g > 90) & (r - b > 80) & (g - b > 40)).sum())


def test_the_marker_is_painted_over_the_frame(app, window) -> None:
    canvas = window.viewer.canvas
    canvas.resize(600, 600)
    window.overlay_box.setChecked(True)
    app.processEvents()
    with_marker = _marker_pixels(_painted(canvas))
    window.overlay_box.setChecked(False)
    app.processEvents()
    without_marker = _marker_pixels(_painted(canvas))
    assert with_marker > 50 and without_marker == 0, f"on={with_marker} off={without_marker}"
    assert window._clip.ndim == 3, f"the marker is not burned into the clip: {window._clip.shape}"


def test_painting_keeps_up_with_playback(app, window) -> None:
    from qtpy.QtGui import QPixmap

    canvas = window.viewer.canvas
    canvas.resize(600, 600)
    window.overlay_box.setChecked(True)
    app.processEvents()
    times = []
    for i in range(30):
        canvas.set_frame(i % max(1, window._clip.shape[0]))
        t0 = time.perf_counter()
        canvas.render(QPixmap(canvas.size()))
        times.append((time.perf_counter() - t0) * 1000)
    assert np.median(times) < 20, f"canvas paint: {np.median(times):.1f} ms"


def test_the_notes_box_gives_focus_back(app, window) -> None:
    """A focused notes box eats every plain keypress, so Esc and Enter must release it."""
    from qtpy.QtCore import QEvent, Qt
    from qtpy.QtGui import QKeyEvent

    window._focus_notes()
    app.processEvents()
    assert window.notes.hasFocus(), "n focuses the notes box"
    window.notes.setText("a note")
    app.sendEvent(window.notes, QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier))
    app.processEvents()
    assert not window.notes.hasFocus(), "Esc releases it"
    assert window.notes.text() == "a note", "Esc keeps the typed text"

    window._focus_notes()
    app.processEvents()
    window.notes.returnPressed.emit()
    app.processEvents()
    assert not window.notes.hasFocus(), "Enter releases it"


# --------------------------------------------------------------------------- #
# failures and the suggester
# --------------------------------------------------------------------------- #
class _AlwaysAudit:
    ready = True

    def is_audit_clip(self, clip_id):
        return True

    def suggest(self, clip_id):
        return {}

    def status_line(self):
        return "model loaded"


def test_a_held_out_clip_explains_its_missing_suggestion(app, window, monkeypatch) -> None:
    import trackanno.gui.app as app_module

    window.viewer.pause()
    window.suggester = _AlwaysAudit()
    window._current_suggestion = {}
    assert window._why_no_suggestion("c0") == "", "silent by default"

    monkeypatch.setattr(app_module, "AUDIT_VISIBLE", True)
    note = window._why_no_suggestion("c0")
    assert "held out for audit" in note, note
    window._current_suggestion = {"behaviour": ("swimming", 0.9)}
    assert window._why_no_suggestion("c0") == "", "no note when a suggestion is shown"

    window.categories.set_suggestion({}, "held out for audit")
    app.processEvents()
    assert "held out" in window.categories.suggestion.text()
    window.categories.set_suggestion({}, "")
    caption = window.categories.suggestion
    assert caption.isHidden() or caption.text() == "", "the caption clears without one"


class _LockedCsv:
    """An annotation store whose every write fails like a CSV open in Excel."""

    path = None

    def _locked(self, *a, **k):
        raise PermissionError(13, "The process cannot access the file")

    record = clear = undo = _locked

    def get(self, _clip_id):
        return None

    def missing_required(self, _values):
        return []

    @property
    def done_ids(self):
        return set()

    def __len__(self):
        return 0


def test_a_locked_csv_is_reported_not_fatal(app, cfg, window) -> None:
    window.viewer.pause()
    window.annotations = _LockedCsv()
    window.annotations.path = cfg.paths.csv
    group = cfg.categories[0]
    window.categories.toggle(group.name, group.options[0].name)
    app.processEvents()
    said = window.messages + [window._pending_message or ""]
    assert any("could not write" in m for m in said), said
    assert any("Excel" in m for m in said), "the hint names the likely cause"

    window._undo()
    app.processEvents()
    assert any("could not write" in m for m in window.messages), "undo survives it too"


def test_the_prefetcher_joins_before_the_store_closes(cfg) -> None:
    from trackanno.gui.prefetch import Prefetcher

    store = ClipStore(cfg.paths.work_dir)
    try:
        order = list(range(len(store)))
        for _ in range(30):
            prefetcher = Prefetcher(store, depth=8)
            prefetcher.around(order, 0)
            assert prefetcher.stop() is True, "stop() reports that it joined"
    finally:
        store.close()


def test_a_slot_exception_is_logged_not_fatal(app, monkeypatch, capsys) -> None:
    """The installed hook prints the traceback and shows it in the status bar."""
    import sys

    from qtpy.QtWidgets import QMainWindow

    from trackanno.gui.hub import install_excepthook

    window = QMainWindow()
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    install_excepthook(window)
    try:
        raise ValueError("boom")
    except ValueError:
        sys.excepthook(*sys.exc_info())
    assert "ValueError: boom" in capsys.readouterr().err
    assert "boom" in window.statusBar().currentMessage()
    window.deleteLater()
