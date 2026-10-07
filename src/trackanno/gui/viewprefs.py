"""Per-user view toggles of the annotation window, kept in QSettings."""

from __future__ import annotations

from typing import Any

from ..config import VIEW_SETTINGS

__all__ = ["load_view", "save_view"]

VIEW_FILE = "view.ini"


def _store():
    from qtpy.QtCore import QSettings

    from ..project import _state_dir

    return QSettings(str(_state_dir() / VIEW_FILE), QSettings.IniFormat)


def _typed(raw, default):
    if isinstance(default, bool):
        return raw if isinstance(raw, bool) else str(raw).strip().lower() == "true"
    if isinstance(default, float):
        return float(raw)
    return str(raw)


def load_view() -> dict[str, Any]:
    """``{show_overlay, show_skeleton, show_context_tracks, follow, view_zoom}``."""
    store = _store()
    out = {}
    for dotted, default in VIEW_SETTINGS.items():
        key = dotted.split(".", 1)[1]
        try:
            out[key] = _typed(store.value(f"view/{key}", default), default)
        except (TypeError, ValueError):
            out[key] = default
    return out


def save_view(**values) -> None:
    """Remember the given toggles for this user."""
    store = _store()
    for key, value in values.items():
        store.setValue(f"view/{key}", value)
    store.sync()
