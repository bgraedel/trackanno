"""Side-panel widgets: category buttons and source metadata."""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QButtonGroup, QGridLayout, QGroupBox, QLabel, QPushButton, QVBoxLayout,
    QWidget,
)

from ..config import CategoryGroup, Config
from .theme import RADIUS, tokens
from ..annotate.store import AnnotationStore

__all__ = ["CategoryPanel", "MetadataPanel"]


def _button_style(color: str, checked: bool) -> str:
    """A category button: its own colour, the theme's shape. Checked fills with
    the colour, unchecked keeps a tinted left edge."""
    t = tokens()
    if checked:
        return (
            f"QPushButton{{background:{color};color:{t['on_accent']};"
            f"border:1px solid {color};border-radius:{RADIUS}px;"
            f"padding:8px 10px;text-align:left;font-weight:650;}}"
        )
    return (
        f"QPushButton{{background:{t['raised']};color:{t['text']};"
        f"border:1px solid {t['border_hi']};border-left:3px solid {color};"
        f"border-radius:{RADIUS}px;padding:8px 10px;text-align:left;}}"
        f"QPushButton:hover{{background:{t['raised_hi']};border-color:{color};"
        f"border-left:3px solid {color};}}"
    )


class CategoryPanel(QWidget):
    """One block of buttons per category group, driven by config.yaml."""

    changed = Signal()

    def __init__(self, cfg: Config, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self._buttons: dict[tuple[str, str], QPushButton] = {}
        self._groups: dict[str, QButtonGroup] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        for group in cfg.categories:
            layout.addWidget(self._build_group(group))

        # Suggestions are DISPLAYED, never pre-selected: the annotator presses
        # the real key, so labels stay independent of the model trained on them.
        self.suggestion = QLabel("")
        self.suggestion.setWordWrap(True)
        self.suggestion.setProperty("role", "muted")
        self.suggestion.setStyleSheet("font-style:italic;")
        self.suggestion.setVisible(False)
        layout.addWidget(self.suggestion)
        layout.addStretch(1)
        self._suggested: dict[str, str] = {}

    def set_suggestion(self, suggestions: dict[str, tuple[str, float]],
                       note: str = "") -> None:
        """Outline the suggested option and show a caption. Selects nothing."""
        for (group_name, option_name) in list(self._suggested.items()):
            self._set_button(group_name, option_name,
                             bool(self._buttons[(group_name, option_name)].property("selected")))
        self._suggested = {}

        if not suggestions:
            # A note instead of silence, when the caller has a reason to give.
            self.suggestion.setText(note or "")
            self.suggestion.setVisible(bool(note))
            return

        lines = []
        for group_name, (option_name, conf) in suggestions.items():
            key = (group_name, option_name)
            if key not in self._buttons:
                continue
            self._suggested[group_name] = option_name
            btn = self._buttons[key]
            option = self.cfg.group(group_name).option(option_name)
            if not btn.property("selected"):
                btn.setStyleSheet(
                    f"QPushButton{{background:{tokens()['raised']};color:{tokens()['text']};"
                    f"border:1px dashed {option.color};border-left:3px solid {option.color};"
                    f"border-radius:{RADIUS}px;padding:8px 10px;text-align:left;}}"
                )
            lines.append(f"suggests <b>{option_name}</b> ({conf:.2f}) - press {option.key}")
        self.suggestion.setText("<br>".join(lines))
        self.suggestion.setVisible(bool(lines))

    def _build_group(self, group: CategoryGroup) -> QGroupBox:
        title = group.name + ("  *" if group.required else "")
        box = QGroupBox(title)
        box.setToolTip(
            f"{'one choice' if not group.is_multi else 'any number of choices'}"
            f"{' - required' if group.required else ''}"
        )
        inner = QVBoxLayout(box)
        inner.setSpacing(4)

        button_group = QButtonGroup(box)
        button_group.setExclusive(False)  # exclusivity is enforced in _on_clicked
        self._groups[group.name] = button_group

        for option in group.options:
            btn = QPushButton(f" {option.key.upper()}    {option.name}")
            btn.setCheckable(True)
            btn.setFocusPolicy(Qt.NoFocus)  # keep keyboard focus on the window
            btn.setStyleSheet(_button_style(option.color, False))
            btn.clicked.connect(
                lambda _checked, g=group, o=option.name: self.toggle(g.name, o)
            )
            self._buttons[(group.name, option.name)] = btn
            button_group.addButton(btn)
            inner.addWidget(btn)
        return box

    # -- state -------------------------------------------------------------- #
    def toggle(self, group_name: str, option_name: str) -> None:
        group = self.cfg.group(group_name)
        if group is None:
            return
        btn = self._buttons[(group_name, option_name)]
        new_state = not btn.property("selected")
        if not group.is_multi:
            for option in group.options:
                self._set_button(group_name, option.name, False)
        self._set_button(group_name, option_name, new_state)
        self.changed.emit()

    def _set_button(self, group_name: str, option_name: str, on: bool) -> None:
        btn = self._buttons[(group_name, option_name)]
        option = self.cfg.group(group_name).option(option_name)
        btn.setProperty("selected", on)
        btn.setChecked(on)
        btn.setStyleSheet(_button_style(option.color, on))

    def values(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for group in self.cfg.categories:
            picked = [
                o.name for o in group.options
                if self._buttons[(group.name, o.name)].property("selected")
            ]
            out[group.name] = (
                AnnotationStore.encode_multi(picked) if group.is_multi
                else (picked[0] if picked else "")
            )
        return out

    def set_values(self, values: dict[str, str] | None) -> None:
        for group in self.cfg.categories:
            raw = (values or {}).get(group.name, "")
            selected = AnnotationStore.decode_multi(raw) if group.is_multi else ({raw} if raw else set())
            for option in group.options:
                self._set_button(group.name, option.name, option.name in selected)

    def clear(self) -> None:
        self.set_values(None)
        self.changed.emit()

    def option_for_key(self, key: str) -> tuple[str, str] | None:
        for group in self.cfg.categories:
            for option in group.options:
                if option.key.lower() == key.lower():
                    return group.name, option.name
        return None


class MetadataPanel(QWidget):
    """Where the pixels on screen came from."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._values: dict[str, QLabel] = {}
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setVerticalSpacing(2)
        grid.setColumnStretch(1, 1)
        for row, field in enumerate(["movie", "track", "frames", "crop", "detected"]):
            name = QLabel(field)
            name.setProperty("role", "muted")
            value = QLabel("-")
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            value.setWordWrap(True)
            grid.addWidget(name, row, 0, Qt.AlignTop)
            grid.addWidget(value, row, 1)
            self._values[field] = value

    def update_from(self, meta, origins=None) -> None:
        self._values["movie"].setText(str(meta["movie_id"]))
        self._values["track"].setText(f"track {int(meta['track_id'])}")
        self._values["frames"].setText(f"{int(meta['f_start'])} - {int(meta['f_end']) - 1}")
        scale = float(meta["scale"])
        crop = (
            f"({int(meta['crop_x0'])}, {int(meta['crop_y0'])})  {int(meta['crop_w'])} px"
            + ("" if abs(scale - 1.0) < 1e-6 else f"  x{scale:.2f}")
        )
        if origins is not None and len(origins) > 1:
            drift = int(np.abs(origins[-1] - origins[0]).max())
            if drift:
                crop += f"  (crop moves {drift} px)"
        self._values["crop"].setText(crop)
        self._values["detected"].setText(
            f"{int(meta['n_present'])}/{int(meta['f_end']) - int(meta['f_start'])} frames"
            f"   conf {float(meta['mean_conf']):.2f}"
        )
