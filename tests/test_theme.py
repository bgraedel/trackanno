"""The theme: the window icon and compact buttons that still draw their label."""

from __future__ import annotations

import numpy as np
import pytest


def _ink(widget) -> int:
    """Pixels that differ clearly from the widget's own fill."""
    from qtpy.QtGui import QPixmap

    pixmap = QPixmap(widget.size())
    widget.render(pixmap)
    image = pixmap.toImage()
    raw = image.constBits().asstring(image.sizeInBytes())
    arr = np.frombuffer(raw, np.uint8).reshape(
        image.height(), image.bytesPerLine() // 4, 4)[..., :3].astype(int)
    base = arr[1, 1]
    return int((np.abs(arr - base).max(axis=-1) > 40).sum())


def test_the_application_has_an_icon_at_every_size(app) -> None:
    from trackanno.gui.theme import app_icon, apply_theme

    apply_theme(app, "dark")
    sizes = sorted(s.width() for s in app.windowIcon().availableSizes())
    assert not app.windowIcon().isNull()
    assert sizes == [16, 32, 64, 256], "drawn at each size rather than scaled"
    small = app_icon().pixmap(16, 16).toImage()
    assert any(small.pixelColor(x, y).alpha() > 0 for x in range(16) for y in range(16)), (
        "the icon is not a blank square")


@pytest.mark.parametrize("theme", ["dark", "light"])
@pytest.mark.parametrize("label, size", [("?", 28), ("x", 26)])
def test_a_compact_button_draws_its_label(app, theme, label, size) -> None:
    """The themed padding is wider than a 26px button, so the glyph must still show."""
    from trackanno.gui.theme import apply_theme, compact_button

    apply_theme(app, theme)
    try:
        button = compact_button(label, size)
        button.show()
        app.processEvents()
        assert _ink(button) > 0, f"{label!r} drew nothing in the {theme} theme"
        button.close()
    finally:
        apply_theme(app, "dark")


def test_the_annotation_screen_buttons_keep_their_size(app, root, tmp_path) -> None:
    from fixtures import make_project

    from trackanno.annotate.clips import ClipStore
    from trackanno.annotate.store import AnnotationStore
    from trackanno.config import load_config
    from trackanno.gui.app import AnnotationWidget
    from trackanno.gui.theme import compact_button

    cfg = load_config(make_project(root))
    cfg.paths.csv = tmp_path / "buttons.csv"
    store = ClipStore(cfg.paths.work_dir)
    panel = AnnotationWidget(cfg, store, AnnotationStore(cfg.paths.csv, cfg))
    try:
        panel.viewer.pause()
        app.processEvents()
        buttons = [b for b in panel.findChildren(type(compact_button()))
                   if b.property("compact") and b.text()]
        assert buttons, "the annotation screen has a compact button"
        for button in buttons:
            assert button.width() >= 26, f"{button.text()!r} is {button.width()}px wide"
            assert _ink(button) > 0, f"{button.text()!r} draws nothing"
        row = panel.layout().itemAt(0).layout()
        assert row.minimumSize().width() <= panel.width(), (
            f"the top row needs {row.minimumSize().width()}px, has {panel.width()}px")
    finally:
        panel.shutdown()
        store.close()
