"""config.yaml as a form; help, choices and ranges come from the config schema.

Only keys changed since loading are written, via :func:`trackanno.project.write_config`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QAbstractItemView, QCheckBox, QColorDialog, QComboBox,
    QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGridLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPushButton, QScrollArea, QSpinBox,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..config import help_for, meta_for
from ..project import _exists
from .theme import compact_button

__all__ = ["ConfigForm", "CategoryEditor", "KeypointEditor", "GROUPS", "FIELDS",
           "BASIC", "NEEDS_REPREPARE", "HAS_STATUS"]


def _options(dotted: str, extra: dict) -> list[str]:
    """A combo's items: the schema's choices, else the suggestions given."""
    choices = meta_for(dotted).get("choices")
    return [str(c) for c in choices] if choices else list(extra.get("options", []))


# --------------------------------------------------------------------------- #
# the fields, grouped by what they are for
# --------------------------------------------------------------------------- #
# (title, one sentence, [(dotted key, kind, label, widget extras), ...])
GROUPS: list[tuple[str, str, list[tuple[str, str, str, dict]]]] = [
    ("Project", "Where the movies are, and where this project writes.", [
        ("paths.video_dir", "dir", "video folder", {}),
        ("paths.model", "file", "detection model",
         {"filter": "PyTorch weights (*.pt);;All files (*)"}),
        ("paths.work_dir", "dir", "clip cache", {}),
        ("paths.csv", "file", "annotation csv",
         {"filter": "CSV (*.csv);;All files (*)", "save": True}),
    ]),
    ("Detection", "How animals are found in each frame.", [
        ("detection.task", "choice", "task", {}),
        ("detection.device", "choice", "compute device",
         {"options": ["auto", "cpu", "0", "mps"]}),
        ("detection.conf", "float", "confidence", {"step": 0.05}),
        ("detection.batch", "int", "batch", {"max": 64}),
        ("detection.imgsz", "int", "inference size", {"max": 4096, "step": 32}),
    ]),
    ("Tracking", "Linking detections across frames.", [
        ("tracking.backend", "choice", "tracker", {}),
        ("tracking.min_track_length", "int", "min track length", {"max": 100000}),
        ("tracking.border_margin_px", "int", "border margin (px)",
         {"max": 2000, "step": 5}),
        ("tracking.border_max_frac", "float", "border tolerance", {"step": 0.05}),
    ]),
    ("Clips", "The clips cut from each track, and how they are cropped.", [
        ("clips.frames", "int", "clip length", {"max": 10000}),
        ("clips.stride", "int", "stride", {"max": 10000}),
        ("clips.window_mode", "choice", "how much to cut", {}),
        ("clips.follow", "choice", "crop follows", {}),
        ("clips.pad", "float", "crop padding", {"max": 8.0, "step": 0.1}),
        ("clips.crop_size", "autonum", "cache crop size", {"max": 1024, "step": 16}),
        ("normalize.mode", "choice", "intensity mapping", {}),
    ]),
    ("Annotating", "Who is labelling, and how the clips arrive.", [
        ("ui.annotator", "text", "annotator name", {}),
        ("ui.fps", "int", "playback fps", {"max": 120}),
        ("ui.auto_advance", "bool", "auto-advance", {}),
        ("ui.shuffle", "bool", "shuffle clip order", {}),
    ]),
    ("Suggestions", "A classifier trained on your labels, shown beside the clip.", [
        ("suggest.enabled", "bool", "suggest labels", {}),
        ("suggest.feature_groups", "groups", "features the model may use", {}),
        ("suggest.min_confidence", "float", "minimum confidence", {"step": 0.05}),
        ("suggest.audit_fraction", "float", "audit holdout", {"step": 0.05}),
        ("suggest.select_rule", "choice", "choose the next clip by", {}),
    ]),
    ("Bulk labelling", "How the model labels every frame and cuts it into bouts.", [
        ("predict.window_stride", "autonum", "window stride", {"max": 100000}),
        ("predict.min_bout_frames", "int", "shortest bout", {"max": 100000}),
    ]),
]

# The flat view: (section, key, kind, label, tooltip, extras).
FIELDS: list[tuple[str, str, str, str, str, dict]] = [
    (*dotted.split(".", 1), kind, label, help_for(dotted), extra)
    for _title, _blurb, rows in GROUPS for dotted, kind, label, extra in rows
]

# The settings to decide when setting up a project; the rest go on Advanced.
BASIC = {
    "paths.video_dir", "paths.model", "paths.work_dir", "paths.csv",
    "detection.task", "detection.device",
    "tracking.min_track_length",
    "clips.frames", "clips.stride", "clips.window_mode", "clips.follow", "clips.pad",
    "ui.fps", "ui.annotator",
}

# Fields with a note under the row.
HAS_STATUS = {"clips.stride"}


def _reprepare_keys() -> set[str]:
    from ..detect.prepare import _CLIP_KEYS, _DETECT_KEYS

    return set(_DETECT_KEYS) | set(_CLIP_KEYS) | {
        "paths.video_dir", "paths.model", "keypoints.nodes"}


# Changing these invalidates clips already cut.
NEEDS_REPREPARE = _reprepare_keys()


class ConfigForm(QScrollArea):
    """Scalar settings grouped by purpose; ``scope`` is ``basic``, ``advanced`` or ``all``."""

    changed = Signal()

    def __init__(self, scope: str = "all", parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self._widgets: dict[str, QWidget] = {}
        self._kinds: dict[str, str] = {}
        self._status: dict[str, QLabel] = {}
        self._rows: dict[str, QHBoxLayout] = {}
        # Values at load; values() returns only what differs.
        self._at_load: dict[str, Any] = {}
        # Original tooltips while mark() shows a warning instead.
        self._tips: dict[str, str] = {}

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(14)

        for title, blurb, rows in GROUPS:
            chosen = [r for r in rows if self._in_scope(scope, r[0])]
            if not chosen:
                continue
            box = QGroupBox(title)
            outer = QVBoxLayout(box)
            caption = QLabel(blurb)
            caption.setWordWrap(True)
            caption.setProperty("role", "muted")
            outer.addWidget(caption)
            form = QFormLayout()
            form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
            form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            outer.addLayout(form)
            for dotted, kind, label, extra in chosen:
                tip = f"{dotted}: {help_for(dotted)}"
                if dotted in NEEDS_REPREPARE:
                    tip += "\nChanging this needs Detect & crop re-run."
                widget = self._make_widget(dotted, kind, extra)
                widget.setToolTip(tip)
                name = QLabel(label + ("  *" if dotted in NEEDS_REPREPARE else ""))
                name.setToolTip(tip)
                form.addRow(name, self._with_status(dotted, widget))
                self._widgets[dotted] = widget
                self._kinds[dotted] = kind
            layout.addWidget(box)

        note = QLabel("Settings marked <b>*</b> need Detect &amp; crop re-run.")
        note.setWordWrap(True)
        note.setProperty("role", "muted")
        layout.addWidget(note)
        layout.addStretch(1)
        self.setWidget(body)

    @staticmethod
    def _in_scope(scope: str, dotted: str) -> bool:
        if scope == "basic":
            return dotted in BASIC
        if scope == "advanced":
            return dotted not in BASIC
        return True

    # -- widget construction ------------------------------------------------ #
    def _with_status(self, dotted: str, widget: QWidget) -> QWidget:
        """The field with room under it for a note, for HAS_STATUS keys."""
        if dotted not in HAS_STATUS:
            return widget
        holder = QWidget()
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(widget, 1)
        box = QVBoxLayout(holder)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(2)
        box.addLayout(row)
        status = QLabel("")
        status.setProperty("role", "muted")
        status.setWordWrap(True)
        status.setVisible(False)
        box.addWidget(status)
        self._rows[dotted] = row
        self._status[dotted] = status
        return holder

    def row_for(self, dotted: str) -> QHBoxLayout | None:
        """The field's own row, for a page adding a button beside it."""
        widget = self._widgets.get(dotted)
        return getattr(widget, "row", None) or self._rows.get(dotted)

    def set_status(self, dotted: str, text: str) -> None:
        """Show a note under a field."""
        widget = self._widgets.get(dotted)
        if widget is not None and hasattr(widget, "set_status"):
            widget.set_status(text)
            return
        label = self._status.get(dotted)
        if label is None:
            return
        label.setText(text)
        label.setVisible(bool(text))

    def _make_widget(self, dotted: str, kind: str, extra: dict) -> QWidget:
        meta = meta_for(dotted)
        bounds = {"min": meta.get("min"), "max": extra.get("max", meta.get("max"))}
        if meta.get("below") is not None and bounds["max"] is None:
            bounds["max"] = meta["below"] - 0.01
        bounds = {k: v for k, v in bounds.items() if v is not None}
        extra = {**extra, **bounds}
        if kind == "bool":
            widget = QCheckBox()
            widget.stateChanged.connect(self.changed)
            return widget
        if kind == "choice":
            widget = QComboBox()
            widget.addItems(_options(dotted, extra))
            widget.setEditable(not meta.get("choices"))
            widget.currentTextChanged.connect(self.changed)
            return widget
        if kind == "int":
            widget = QSpinBox()
            widget.setRange(int(extra.get("min", 0)), int(extra.get("max", 1_000_000)))
            widget.setSingleStep(int(extra.get("step", 1)))
            widget.valueChanged.connect(self.changed)
            return widget
        if kind == "float":
            widget = QDoubleSpinBox()
            widget.setRange(float(extra.get("min", 0.0)), float(extra.get("max", 1e6)))
            widget.setSingleStep(float(extra.get("step", 0.1)))
            widget.setDecimals(int(extra.get("decimals", 2)))
            widget.valueChanged.connect(self.changed)
            return widget
        if kind in ("dir", "file"):
            return _PathField(kind, extra, self.changed)
        if kind == "groups":
            return _GroupPicker(self.changed)
        if kind == "autonum":
            return _AutoField(extra, self.changed)
        widget = QLineEdit()
        widget.textChanged.connect(self.changed)
        return widget

    # -- values ------------------------------------------------------------- #
    def load(self, data: dict) -> None:
        """Fill the form from a raw config dict, showing the values the loader uses."""
        from ..config import effective_values

        values = effective_values(data)
        for dotted, widget in self._widgets.items():
            if dotted in values:
                self._set(widget, self._kinds[dotted], values[dotted])
        self._at_load = self.current()

    def _set(self, widget: QWidget, kind: str, value: Any) -> None:
        widget.blockSignals(True)
        try:
            if kind == "bool":
                widget.setChecked(bool(value))
            elif kind == "choice":
                widget.setCurrentText(str(value))
            elif kind == "int":
                widget.setValue(int(value))
            elif kind == "float":
                widget.setValue(float(value))
            elif kind in ("dir", "file", "groups", "autonum"):
                widget.set_value(value)
            else:
                # Only when it differs: setText moves the caret mid-edit.
                text = "" if value is None else str(value)
                if widget.text() != text:
                    widget.setText(text)
        except (TypeError, ValueError):
            pass
        finally:
            widget.blockSignals(False)

    def show_resolved(self, values: dict) -> None:
        """Show what each 'auto' resolves to, given all dotted values."""
        from ..config import Clips, Predict

        def number(key: str, default):
            try:
                return type(default)(values.get(key, default))
            except (TypeError, ValueError):
                return default

        frames = max(1, number("clips.frames", 50))
        stride = max(1, number("clips.stride", frames))
        clips = Clips(frames=frames, stride=stride)
        predict = Predict(window_stride=values.get("predict.window_stride", "auto"))
        self._resolve("predict.window_stride", predict.resolved_stride(clips))
        # crop_size depends on the movies, so no value is shown.
        self._resolve("clips.crop_size", None)

        if stride == frames:
            self.set_status("clips.stride",
                            "The windows tile: every frame is in exactly one clip.")
        elif stride < frames:
            self.set_status(
                "clips.stride", f"The windows overlap by {frames - stride} frames.")
        else:
            self.set_status("clips.stride",
                            f"{stride - frames} frames between one clip and the next "
                            "are never cut.")

    def _resolve(self, dotted: str, value, note: str = "") -> None:
        widget = self._widgets.get(dotted)
        if widget is not None and hasattr(widget, "set_resolved"):
            widget.set_resolved(value, note)

    # -- validation --------------------------------------------------------- #
    def warnings(self, root: Path | None = None) -> dict[str, str]:
        """Paths that point at nothing (warnings, never blocking a save)."""
        out: dict[str, str] = {}
        values = self.current()

        def resolve(text: str) -> Path:
            p = Path(str(text))
            return p if p.is_absolute() or str(p).startswith("//") else (root or Path.cwd()) / p

        for key, must in (("paths.video_dir", "folder"), ("paths.model", "file"),
                          ("paths.work_dir", None)):
            if key not in values:
                continue
            text = str(values[key] or "").strip()
            if not text:
                out[key] = "not set"
                continue
            if must is None:
                continue
            # Timeout-guarded: is_dir() on a dropped share can block.
            target = resolve(text)
            if must == "folder" and not _exists(target, "dir"):
                out[key] = "this folder does not exist"
            elif must == "file" and not _exists(target, "file"):
                out[key] = "this file does not exist"
        return out

    def problems(self) -> dict[str, str]:
        """Fields whose text is not a usable value; these block the save."""
        out: dict[str, str] = {}
        for dotted, widget in self._widgets.items():
            message = widget.problem() if hasattr(widget, "problem") else ""
            if message:
                out[dotted] = message
        return out

    def mark(self, problems: dict[str, str]) -> None:
        """Outline the offending fields and say why on hover."""
        for dotted, widget in self._widgets.items():
            target = getattr(widget, "edit", widget)
            message = problems.get(dotted)
            target.setProperty("invalid", "true" if message else None)
            target.style().unpolish(target)
            target.style().polish(target)
            if message:
                self._tips.setdefault(dotted, target.toolTip())
                target.setToolTip(message)
            elif dotted in self._tips:
                target.setToolTip(self._tips.pop(dotted))

    def values(self) -> dict[str, Any]:
        """Dotted-key updates for fields changed since :meth:`load` or :meth:`saved`."""
        return {key: value for key, value in self.current().items()
                if key not in self._at_load or value != self._at_load[key]}

    def saved(self) -> None:
        """Mark the values on screen as saved."""
        self._at_load = self.current()

    def current(self) -> dict[str, Any]:
        """Every field as it stands, touched or not."""
        out: dict[str, Any] = {}
        for dotted, widget in self._widgets.items():
            kind = self._kinds[dotted]
            if kind == "bool":
                out[dotted] = bool(widget.isChecked())
            elif kind == "choice":
                out[dotted] = widget.currentText().strip()
            elif kind == "int":
                out[dotted] = int(widget.value())
            elif kind == "float":
                out[dotted] = round(float(widget.value()), 6)
            elif kind in ("dir", "file", "groups", "autonum"):
                out[dotted] = widget.value()
            else:
                # Always text: "2024" stays a string and empty stays "".
                out[dotted] = widget.text().strip()
        return out


class _AutoField(QWidget):
    """A number or ``auto``; the tick's label shows what ``auto`` resolves to."""

    def __init__(self, extra: dict, changed_signal):
        super().__init__()
        self._changed = changed_signal
        self._float = extra.get("numeric") == "float"
        if self._float:
            self.edit = QDoubleSpinBox()
            self.edit.setRange(float(extra.get("min", 0.0)), float(extra.get("max", 1.0)))
            self.edit.setSingleStep(float(extra.get("step", 0.05)))
            self.edit.setDecimals(int(extra.get("decimals", 2)))
        else:
            self.edit = QSpinBox()
            self.edit.setRange(int(extra.get("min", 0)), int(extra.get("max", 1_000_000)))
            self.edit.setSingleStep(int(extra.get("step", 1)))
        self.edit.valueChanged.connect(self._on_edited)

        self.auto = QCheckBox("auto")
        self.auto.setChecked(True)
        self.auto.toggled.connect(self._on_toggled)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addWidget(self.auto)
        row.addWidget(self.edit, 1)
        self.edit.setEnabled(False)

    def _on_toggled(self, on: bool) -> None:
        self.edit.setEnabled(not on)
        self._changed.emit()

    def _on_edited(self, _value) -> None:
        if not self.auto.isChecked():
            self._changed.emit()

    def set_resolved(self, value, note: str = "") -> None:
        """Label the tick ``auto (50)``, or ``auto: <note>``."""
        if note:
            self.auto.setText(f"auto: {note}")
        elif value is None:
            self.auto.setText("auto")
        else:
            self.auto.setText(f"auto ({value:g})" if self._float
                              else f"auto ({int(value)})")
        if value is not None and self.auto.isChecked():
            self.edit.blockSignals(True)
            self.edit.setValue(float(value) if self._float else int(value))
            self.edit.blockSignals(False)

    def set_value(self, value) -> None:
        auto = isinstance(value, str)
        for widget in (self.auto, self.edit):
            widget.blockSignals(True)
        try:
            self.auto.setChecked(auto)
            if not auto:
                self.edit.setValue(float(value) if self._float else int(value))
        finally:
            for widget in (self.auto, self.edit):
                widget.blockSignals(False)
        self.edit.setEnabled(not auto)

    def value(self):
        if self.auto.isChecked():
            return "auto"
        return round(float(self.edit.value()), 6) if self._float else int(self.edit.value())

    def setToolTip(self, text):  # noqa: N802 (Qt naming)
        super().setToolTip(text)
        self.edit.setToolTip(text)
        self.auto.setToolTip(text)


class _GroupPicker(QWidget):
    """Feature group checkboxes; groups this project cannot compute are disabled."""

    def __init__(self, changed_signal):
        super().__init__()
        from ..measure.clipfeatures import FEATURE_GROUPS

        self._changed = changed_signal
        self._boxes: dict[str, QCheckBox] = {}
        self._available: dict[str, str] = {}
        # Groups without a box (emb_<name>), kept as they are on save.
        self._extra: list[str] = []
        self._guard = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        for name, spec in FEATURE_GROUPS.items():
            box = QCheckBox(f"{name}: {spec['label']}")
            # Checked by default, matching Suggest.feature_groups.
            box.setChecked(True)
            box.setToolTip(spec["about"])
            box.toggled.connect(self._on_toggled)
            self._boxes[name] = box
            layout.addWidget(box)

        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.note.setProperty("role", "warning")
        self.note.setVisible(False)
        layout.addWidget(self.note)

    def _on_toggled(self, _checked: bool) -> None:
        """Refuse to untick the last group (the loader rejects an empty list)."""
        if self._guard:
            return
        box = self.sender()
        if not self.value() and isinstance(box, QCheckBox):
            self._guard = True
            box.setChecked(True)
            self._guard = False
            self.note.setText(
                "At least one feature group is needed. Tick another one first."
            )
            self.note.setVisible(True)
            return
        self._refresh_note()
        self._changed.emit()

    def _refresh_note(self) -> None:
        """Warn when no ticked group can be computed for this project."""
        chosen = self.value()
        usable = [g for g in chosen if not self._available.get(g)]
        if chosen and not usable and self._available:
            self.note.setText(
                "None of the ticked groups can be computed for this project: "
                + "; ".join(f"{g} {self._available[g]}" for g in chosen)
                + ". Tick a group that can."
            )
            self.note.setVisible(True)
        else:
            self.note.setVisible(False)

    def set_available(self, availability: dict) -> None:
        from ..measure.clipfeatures import FEATURE_GROUPS

        self._available = dict(availability)
        for name, box in self._boxes.items():
            why = availability.get(name, "")
            # Ticks are kept, so switching back to 'pose' restores them.
            box.setEnabled(not why)
            base = FEATURE_GROUPS[name]["about"]
            box.setToolTip(f"Unavailable ({why}). {base}" if why else base)
            label = f"{name}: {FEATURE_GROUPS[name]['label']}"
            box.setText(f"{label}   ({why})" if why else label)
        self._refresh_note()

    def set_value(self, groups) -> None:
        wanted = set(groups or [])
        self._extra = [g for g in (groups or []) if g not in self._boxes]
        self._guard = True
        try:
            for name, box in self._boxes.items():
                box.setChecked(name in wanted)
        finally:
            self._guard = False
        self._refresh_note()

    def value(self) -> list:
        return [n for n, b in self._boxes.items() if b.isChecked()] + list(self._extra)

    def setToolTip(self, text):  # noqa: N802 (Qt naming)
        super().setToolTip(text)


class _PathField(QWidget):
    """Line edit plus a Browse button, with an optional note underneath."""

    # Emitted on Browse only, not on typing.
    picked = Signal(str)

    def __init__(self, kind: str, extra: dict, changed_signal):
        super().__init__()
        self._kind = kind
        self._extra = extra
        self.edit = QLineEdit()
        self.edit.textChanged.connect(changed_signal)
        button = QPushButton("Browse")
        button.setFixedWidth(70)
        button.clicked.connect(self._browse)
        self.status = QLabel("")
        self.status.setProperty("role", "muted")
        self.status.setWordWrap(True)
        self.status.setVisible(False)
        self.row = QHBoxLayout()
        self.row.setContentsMargins(0, 0, 0, 0)
        self.row.addWidget(self.edit, 1)
        self.row.addWidget(button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addLayout(self.row)
        layout.addWidget(self.status)

    def set_status(self, text: str) -> None:
        self.status.setText(text)
        self.status.setVisible(bool(text))

    def setToolTip(self, text):  # noqa: N802 (Qt naming)
        super().setToolTip(text)
        self.edit.setToolTip(text)

    def set_value(self, text: str) -> None:
        """Show ``text`` without emitting a change."""
        if self.edit.text() != text:
            self.edit.blockSignals(True)
            try:
                self.edit.setText(text)
            finally:
                self.edit.blockSignals(False)

    def value(self) -> str:
        return self.edit.text().strip()

    def _browse(self) -> None:
        start = self.edit.text().strip() or str(Path.home())
        if self._kind == "dir":
            picked = QFileDialog.getExistingDirectory(self, "Choose folder", start)
        elif self._extra.get("save"):
            picked, _ = QFileDialog.getSaveFileName(
                self, "Choose file", start, self._extra.get("filter", "All files (*)")
            )
        else:
            picked, _ = QFileDialog.getOpenFileName(
                self, "Choose file", start, self._extra.get("filter", "All files (*)")
            )
        if picked:
            text = str(Path(picked)).replace("\\", "/")
            self.edit.setText(text)
            self.picked.emit(text)


# --------------------------------------------------------------------------- #
# categories
# --------------------------------------------------------------------------- #
DEFAULT_COLOURS = [
    "#4C9F70", "#D95F02", "#7570B3", "#E7298A", "#66A61E", "#A6761D",
    "#1B9E77", "#E6AB02", "#666666", "#B3446C",
]


class CategoryEditor(QWidget):
    """Editor for the label groups, their options, keys and colours."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._groups: list[_GroupEditor] = []

        self._container = QVBoxLayout()
        self._container.setSpacing(8)

        add = QPushButton("+ add group")
        add.clicked.connect(lambda: self.add_group())

        hint = QLabel("<b>single</b> = one choice, auto-advances. "
                      "<b>multi</b> = checkboxes, never advances.")
        hint.setWordWrap(True)
        hint.setProperty("role", "muted")
        hint.setToolTip(
            "Keys must be unique and not navigation keys (arrows, Tab, Space, "
            "o, s, c, u, n, m, 0, + and -)."
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)
        layout.addWidget(hint)
        layout.addLayout(self._container)
        layout.addWidget(add, 0, Qt.AlignLeft)
        layout.addStretch(1)

    def add_group(self, data: dict | None = None) -> None:
        editor = _GroupEditor(data or {"name": "", "type": "single", "required": True,
                                       "options": [{"name": "", "key": "", "color": DEFAULT_COLOURS[0]}]})
        editor.changed.connect(self.changed)
        editor.removed.connect(lambda e=editor: self._remove(e))
        self._groups.append(editor)
        self._container.addWidget(editor)
        self.changed.emit()

    def _remove(self, editor) -> None:
        if editor in self._groups:
            self._groups.remove(editor)
            editor.setParent(None)
            editor.deleteLater()
            self.changed.emit()

    def load(self, groups: list[dict]) -> None:
        for editor in list(self._groups):
            self._remove(editor)
        for group in groups or []:
            self.add_group(group)

    def values(self) -> list[dict]:
        return [g.value() for g in self._groups]

    def problems(self) -> list[str]:
        """What the config loader would refuse."""
        from ..config import RESERVED_GROUP_NAMES

        issues, seen_keys, seen_names = [], {}, set()
        nav = {"right", "return", "left", "tab", "space", "o", "s", "c", "u",
               "n", "m", "backspace", "0", "+", "=", "-"}
        for group in self.values():
            name = (group.get("name") or "").strip()
            if not name:
                issues.append("a group has no name")
            elif name in seen_names:
                issues.append(f"two groups are both called {name!r}")
            elif name in RESERVED_GROUP_NAMES:
                # Groups become CSV columns.
                issues.append(
                    f"{name!r} is a column the annotation CSV already uses; "
                    "pick another name"
                )
            seen_names.add(name)
            if not group.get("options"):
                issues.append(f"group {name!r} has no options")
            for option in group.get("options", []):
                label = f"{name}.{option.get('name') or '?'}"
                if not (option.get("name") or "").strip():
                    issues.append(f"an option in {name!r} has no name")
                key = (option.get("key") or "").strip()
                if not key:
                    issues.append(f"{label} has no shortcut key")
                    continue
                if key.lower() in nav:
                    issues.append(f"{label}: {key!r} is a navigation shortcut")
                if key.lower() in seen_keys:
                    issues.append(f"{key!r} is used by both {seen_keys[key.lower()]} and {label}")
                seen_keys[key.lower()] = label
        return issues


class _GroupEditor(QGroupBox):
    changed = Signal()
    removed = Signal()

    def __init__(self, data: dict, parent=None):
        super().__init__(parent)
        self.setTitle(data.get("name") or "group")

        self.name = QLineEdit(str(data.get("name", "")))
        self.name.setPlaceholderText("group name, e.g. behaviour")
        self.name.textChanged.connect(self._on_name)

        self.type = QComboBox()
        self.type.addItems(["single", "multi"])
        self.type.setCurrentText(str(data.get("type", "single")))
        self.type.currentTextChanged.connect(self.changed)

        self.required = QCheckBox("required")
        self.required.setToolTip("Auto-advance waits until every required group is set.")
        self.required.setChecked(bool(data.get("required", False)))
        self.required.stateChanged.connect(self.changed)

        remove = QPushButton("remove group")
        remove.clicked.connect(self.removed)

        header = QHBoxLayout()
        header.addWidget(QLabel("name"))
        header.addWidget(self.name, 1)
        header.addWidget(self.type)
        header.addWidget(self.required)
        header.addStretch(1)
        header.addWidget(remove)

        self._rows: list[_OptionRow] = []
        self._grid = QGridLayout()
        self._grid.setHorizontalSpacing(6)
        for column, title in enumerate(("value", "key", "colour", "")):
            label = QLabel(title)
            label.setProperty("role", "muted")
            self._grid.addWidget(label, 0, column)
        self._grid.setColumnStretch(0, 1)

        add = QPushButton("+ add option")
        add.clicked.connect(lambda: self.add_option())

        line = QFrame()
        line.setFrameShape(QFrame.HLine)

        layout = QVBoxLayout(self)
        layout.addLayout(header)
        layout.addWidget(line)
        layout.addLayout(self._grid)
        layout.addWidget(add, 0, Qt.AlignLeft)

        for option in data.get("options", []):
            self.add_option(option)

    def _on_name(self, text: str) -> None:
        self.setTitle(text or "group")
        self.changed.emit()

    def add_option(self, data: dict | None = None) -> None:
        colour = DEFAULT_COLOURS[len(self._rows) % len(DEFAULT_COLOURS)]
        row = _OptionRow(data or {"name": "", "key": "", "color": colour})
        row.changed.connect(self.changed)
        row.removed.connect(lambda r=row: self._remove(r))
        self._rows.append(row)
        row.attach(self._grid, len(self._rows))
        self.changed.emit()

    def _remove(self, row) -> None:
        if row not in self._rows:
            return
        self._rows.remove(row)
        row.detach()
        for i, remaining in enumerate(self._rows, start=1):
            remaining.attach(self._grid, i)
        self.changed.emit()

    def value(self) -> dict:
        return {
            "name": self.name.text().strip(),
            "type": self.type.currentText(),
            "required": bool(self.required.isChecked()),
            "options": [r.value() for r in self._rows],
        }


class _OptionRow(QWidget):
    changed = Signal()
    removed = Signal()

    def __init__(self, data: dict):
        super().__init__()
        self.name = QLineEdit(str(data.get("name", "")))
        self.name.setPlaceholderText("value")
        self.name.textChanged.connect(self.changed)

        self.key = QLineEdit(str(data.get("key", "")))
        self.key.setPlaceholderText("key")
        self.key.setMaxLength(12)
        self.key.setFixedWidth(60)
        self.key.textChanged.connect(self.changed)

        self._colour = str(data.get("color", DEFAULT_COLOURS[0]))
        self.colour_btn = QPushButton()
        self.colour_btn.setFixedWidth(60)
        self.colour_btn.clicked.connect(self._pick_colour)
        self._paint()

        self.remove_btn = compact_button("x", 26, "Remove this option.", danger=True)
        self.remove_btn.clicked.connect(self.removed)

    def attach(self, grid: QGridLayout, row: int) -> None:
        for column, widget in enumerate(
            (self.name, self.key, self.colour_btn, self.remove_btn)
        ):
            grid.addWidget(widget, row, column)

    def detach(self) -> None:
        for widget in (self.name, self.key, self.colour_btn, self.remove_btn):
            widget.setParent(None)
        self.deleteLater()

    def _paint(self) -> None:
        self.colour_btn.setText(self._colour)
        self.colour_btn.setStyleSheet(
            f"background:{self._colour}; color:#111; border-radius:3px; font-size:9px;"
        )

    def _pick_colour(self) -> None:
        picked = QColorDialog.getColor(QColor(self._colour), self, "Category colour")
        if picked.isValid():
            self._colour = picked.name().upper()
            self._paint()
            self.changed.emit()

    def value(self) -> dict:
        return {
            "name": self.name.text().strip(),
            "key": self.key.text().strip(),
            "color": self._colour,
        }


# --------------------------------------------------------------------------- #
# keypoints
# --------------------------------------------------------------------------- #
def _row_height() -> int:
    """Row height that fits a themed combo box, measured."""
    probe = QComboBox()
    probe.addItem("Head")
    return max(30, probe.sizeHint().height() + 8)


class KeypointEditor(QWidget):
    """Keypoint count (read from the model), names and skeleton links.

    Order matters: 0 is the head, 1 the base, 1 onwards the flagellum chain.
    """

    changed = Signal()
    count_requested = Signal()      # "read it from the model file"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._loading = False

        self.count = QSpinBox()
        self.count.setRange(0, 200)
        self.count.setToolTip("How many keypoints the model predicts.")
        self.count.valueChanged.connect(self._on_count_changed)

        self.read_button = QPushButton("Read from model file")
        self.read_button.setProperty("accent", "true")
        self.read_button.setToolTip(
            "Read the keypoint count from the checkpoint; you choose the names."
        )
        self.read_button.clicked.connect(self.count_requested)

        self.state = QLabel("")
        self.state.setProperty("role", "muted")

        top = QHBoxLayout()
        top.addWidget(QLabel("number of keypoints"))
        top.addWidget(self.count)
        top.addWidget(self.read_button)
        top.addWidget(self.state, 1)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["#", "name"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table.verticalHeader().setDefaultSectionSize(_row_height())

        order_note = QLabel(
            "<b>Order matters, names do not:</b> 0 = head, 1 = base, "
            "1 onwards = the flagellum chain."
        )
        order_note.setWordWrap(True)
        order_note.setProperty("role", "muted")
        order_note.setToolTip("Pose features read keypoints by position, not by name.")

        # -- skeleton --
        self.edges = QTableWidget(0, 3)
        self.edges.setHorizontalHeaderLabels(["from", "to", ""])
        self.edges.verticalHeader().setVisible(False)
        self.edges.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.edges.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.edges.horizontalHeader().setSectionResizeMode(2, QHeaderView.Fixed)
        self.edges.setColumnWidth(2, 44)
        self.edges.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.edges.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.edges.verticalHeader().setDefaultSectionSize(_row_height())

        add_edge = QPushButton("+ add link")
        add_edge.clicked.connect(lambda: self._add_edge_row())
        chain = QPushButton("connect in order")
        chain.setToolTip("Link 0-1, 1-2, and so on along the chain.")
        chain.clicked.connect(self._chain)
        clear = QPushButton("clear")
        clear.clicked.connect(lambda: self._set_edges([]))

        edge_bar = QHBoxLayout()
        edge_bar.addWidget(add_edge)
        edge_bar.addWidget(chain)
        edge_bar.addWidget(clear)
        edge_bar.addStretch(1)

        skeleton_box = QGroupBox("Skeleton (optional)")
        skeleton_layout = QVBoxLayout(skeleton_box)
        hint = QLabel("Display only; it does not affect detection or features.")
        hint.setWordWrap(True)
        hint.setProperty("role", "muted")
        skeleton_layout.addWidget(hint)
        skeleton_layout.addWidget(self.edges)
        skeleton_layout.addLayout(edge_bar)

        names_box = QGroupBox("Keypoints")
        names_layout = QVBoxLayout(names_box)
        names_layout.addLayout(top)
        names_layout.addWidget(order_note)
        names_layout.addWidget(self.table, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(14)
        layout.addWidget(names_box, 1)
        layout.addWidget(skeleton_box)

    # -- count -------------------------------------------------------------- #
    def _on_count_changed(self, n: int) -> None:
        self._resize_table(int(n))
        if not self._loading:
            self.changed.emit()

    def set_count(self, n: int, source: str = "") -> None:
        """Set the number of keypoints, keeping any names already typed."""
        self._loading = True
        self.count.setValue(int(n))
        self._loading = False
        self._resize_table(int(n))
        self.state.setText(source)
        self.changed.emit()

    def _resize_table(self, n: int) -> None:
        was = self._loading
        self._loading = True
        try:
            current = [self._name_at(r) for r in range(self.table.rowCount())]
            self.table.setRowCount(n)
            for row in range(n):
                index = QTableWidgetItem(str(row))
                index.setFlags(Qt.ItemIsEnabled)   # position is not editable
                self.table.setItem(row, 0, index)
                existing = current[row] if row < len(current) else ""
                self.table.setItem(row, 1, QTableWidgetItem(existing or f"kp{row}"))
        finally:
            self._loading = was
        self._fit(self.table)
        self._refresh_edge_choices()

    @staticmethod
    def _fit(table: QTableWidget, max_rows: int = 12) -> None:
        """Size a table to its rows; the page scrolls, not the table."""
        rows = table.rowCount()
        row_height = table.rowHeight(0) if rows else table.verticalHeader().defaultSectionSize()
        header = table.horizontalHeader().height()
        visible = min(max(rows, 1), max_rows)
        table.setMinimumHeight(header + visible * row_height + 4)
        table.setMaximumHeight(header + max(rows, 1) * row_height + 4)

    def _name_at(self, row: int) -> str:
        item = self.table.item(row, 1)
        return item.text().strip() if item is not None else ""

    def _on_item_changed(self, _item) -> None:
        if self._loading:
            return
        self._refresh_edge_choices()
        self.changed.emit()

    # -- skeleton ----------------------------------------------------------- #
    def _add_edge_row(self, a: str = "", b: str = "") -> None:
        row = self.edges.rowCount()
        self.edges.insertRow(row)
        for column, value in ((0, a), (1, b)):
            box = QComboBox()
            box.addItems(self.names())
            if value:
                box.setCurrentText(value)
            box.currentTextChanged.connect(lambda _t: self._edge_edited())
            self.edges.setCellWidget(row, column, box)
        remove = compact_button("x", 26, "remove this link", danger=True)
        remove.clicked.connect(lambda: self._remove_edge(remove))
        self.edges.setCellWidget(row, 2, remove)
        self._fit(self.edges)
        if not self._loading:
            self.changed.emit()

    def _edge_edited(self) -> None:
        if not self._loading:
            self.changed.emit()

    def _remove_edge(self, button) -> None:
        for row in range(self.edges.rowCount()):
            if self.edges.cellWidget(row, 2) is button:
                self.edges.removeRow(row)
                self._fit(self.edges)
                self.changed.emit()
                return

    def _set_edges(self, edges) -> None:
        was = self._loading
        self._loading = True
        try:
            self.edges.setRowCount(0)
            for pair in edges or []:
                if len(pair) == 2:
                    self._add_edge_row(str(pair[0]), str(pair[1]))
            self._fit(self.edges)
        finally:
            self._loading = was
        if not was:
            self.changed.emit()

    def _chain(self) -> None:
        names = self.names()
        self._set_edges([[a, b] for a, b in zip(names, names[1:])])

    def _refresh_edge_choices(self) -> None:
        """Refresh the link dropdowns after a rename, keeping each by position."""
        names = self.names()
        for row in range(self.edges.rowCount()):
            for column in (0, 1):
                box = self.edges.cellWidget(row, column)
                if box is None:
                    continue
                index = box.currentIndex()
                box.blockSignals(True)
                box.clear()
                box.addItems(names)
                if names:
                    box.setCurrentIndex(min(max(index, 0), len(names) - 1))
                box.blockSignals(False)

    # -- values ------------------------------------------------------------- #
    def names(self) -> list[str]:
        return [self._name_at(row) for row in range(self.table.rowCount())]

    def load(self, nodes, edges) -> None:
        self._loading = True
        try:
            nodes = [str(n) for n in (nodes or [])]
            self.count.setValue(len(nodes))
            self.table.setRowCount(len(nodes))
            for row, name in enumerate(nodes):
                index = QTableWidgetItem(str(row))
                index.setFlags(Qt.ItemIsEnabled)
                self.table.setItem(row, 0, index)
                self.table.setItem(row, 1, QTableWidgetItem(name))
            self._set_edges([list(e) for e in (edges or [])])
            self._refresh_edge_choices()
            self._fit(self.table)
            self._fit(self.edges)
            self.state.setText("")
        finally:
            self._loading = False

    def values(self) -> dict:
        edges = []
        for row in range(self.edges.rowCount()):
            a, b = self.edges.cellWidget(row, 0), self.edges.cellWidget(row, 1)
            if a is not None and b is not None:
                edges.append([a.currentText(), b.currentText()])
        return {"nodes": self.names(), "edges": edges}

    def problems(self) -> list[str]:
        """What the config loader would refuse."""
        names = self.names()
        issues = []
        if any(not n for n in names):
            issues.append("every keypoint needs a name")
        duplicates = sorted({n for n in names if n and names.count(n) > 1})
        if duplicates:
            issues.append("duplicate keypoint names: " + ", ".join(duplicates))
        for a, b in self.values()["edges"]:
            if a not in names or b not in names:
                issues.append(
                    f"skeleton link {a or '?'} to {b or '?'} names a keypoint that does not exist"
                )
            elif a == b:
                issues.append(f"skeleton link {a} joins a keypoint to itself")
        return issues
