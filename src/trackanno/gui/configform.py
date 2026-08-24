"""config.yaml as a form.

The file stays the source of truth: saving goes through
:func:`trackanno.project.write_config`, which rewrites values in place and keeps
every comment. Each field's tooltip repeats the comment in the file, so the two
cannot drift apart. Only settings a person actually turns are listed; anything
absent from :data:`FIELDS` is left untouched rather than dropped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QAbstractItemView, QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QPushButton, QScrollArea, QSpinBox, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from ..project import _exists
from .theme import compact_button

__all__ = ["ConfigForm", "CategoryEditor", "KeypointEditor", "FIELDS", "BASIC",
           "NEEDS_REPREPARE", "HAS_STATUS"]

# The settings someone setting up a project has to decide. Everything else is a
# tuning knob with a defensible default; all 54 on one page is unusable.
BASIC = {
    "paths.video_dir", "paths.model", "paths.work_dir", "paths.csv",
    "detection.task", "detection.device",
    "tiff.fps",
    "clips.frames", "clips.window_mode", "clips.follow", "clips.pad",
    "tracking.min_track_length",
    "ui.fps", "ui.annotator", "ui.auto_advance", "ui.shuffle", "ui.theme",
}

# Fields that need a second line under the row for something the value alone
# cannot say. paths.model has its own (see :class:`_PathField`).
HAS_STATUS = {"detection.device"}

# Changing these invalidates a clip cache that has already been built.
NEEDS_REPREPARE = {
    "detection.task", "detection.conf", "detection.iou", "detection.imgsz",
    "detection.max_det", "detection.half",
    "tiling.enabled", "tiling.tile", "tiling.overlap", "tiling.merge_iou",
    "tracking.backend", "tracking.min_track_length", "tracking.greedy_max_dist_px",
    "normalize.mode", "normalize.percentiles", "normalize.smooth_window",
    "clips.frames", "clips.stride", "clips.window_mode", "clips.follow", "clips.pad",
    "clips.crop_size", "clips.max_size", "clips.min_clip_frames",
    "clips.max_clips_per_track", "clips.context_tracks", "clips.thumbnails",
    "paths.video_dir", "paths.model", "keypoints.nodes",  # via KeypointEditor
}

# (section, key, kind, label, tooltip, extra)
FIELDS: list[tuple[str, str, str, str, str, dict]] = [
    # -- paths --
    ("paths", "video_dir", "dir", "video folder",
     "Folder containing the movies. .tif/.tiff and .nd2 are both read natively.", {}),
    ("paths", "model", "file", "detection model",
     "YOLO weights used for detection (and pose, if detection.task is 'pose').",
     {"filter": "PyTorch weights (*.pt);;All files (*)"}),
    ("paths", "work_dir", "dir", "clip cache",
     "Clip cache and per-movie tables. Keep this on a LOCAL disk: the GUI does "
     "~1 MB random reads per clip and network latency is what you feel.", {}),
    ("paths", "csv", "file", "annotation csv",
     "The annotation log; one row per annotated clip. This file is the state - "
     "any clip whose id appears in it is done.",
     {"filter": "CSV (*.csv);;All files (*)", "save": True}),
    ("paths", "patterns", "strlist", "file patterns",
     "Which files in the video folder count as movies.", {}),

    # -- detection --
    ("detection", "task", "choice", "task",
     "'pose' if the model predicts keypoints, 'detect' if it only gives boxes. "
     "Choosing 'pose' needs the keypoint names below; the button fills them in "
     "from the model file.",
     {"options": ["detect", "pose"]}),
    ("detection", "device", "choice", "compute device",
     "'auto' picks the best available and falls back to the CPU if there is no "
     "GPU here, which is what makes a project portable between machines. The "
     "list is filled in from this computer.",
     {"options": ["auto", "cpu"]}),
    ("detection", "imgsz", "int", "inference size",
     "The resolution each tile is seen at. Match what the model was trained "
     "for - changing it changes what the model detects.", {"min": 64, "max": 4096, "step": 32}),
    ("detection", "conf", "float", "confidence",
     "Minimum detection confidence.", {"min": 0.0, "max": 1.0, "step": 0.05}),
    ("detection", "iou", "float", "NMS IoU",
     "Per-tile non-maximum-suppression threshold.", {"min": 0.0, "max": 1.0, "step": 0.05}),
    ("detection", "batch", "int", "batch",
     "Tiles per forward pass. Lower this first if CUDA runs out of memory.",
     {"min": 1, "max": 64}),
    ("detection", "max_det", "int", "max detections",
     "Cap on detections per tile.", {"min": 1, "max": 5000, "step": 50}),
    ("detection", "half", "bool", "fp16 inference",
     "Half-precision inference. Faster on most GPUs, unsupported on CPU.", {}),

    # -- tiling --
    ("tiling", "enabled", "scalar", "tiling",
     "'auto' tiles only frames larger than one tile; true/false force it.", {}),
    ("tiling", "tile", "int", "tile size",
     "How the frame is CUT. detection.imgsz is the resolution each tile is SEEN "
     "at; keep the ratio the model was trained for.", {"min": 64, "max": 4096, "step": 32}),
    ("tiling", "overlap", "float", "tile overlap",
     "Fraction of a tile shared with its neighbour.", {"min": 0.0, "max": 0.89, "step": 0.05}),
    ("tiling", "merge_iou", "float", "merge IoU",
     "Global NMS threshold when merging tiles back together.",
     {"min": 0.0, "max": 1.0, "step": 0.05}),

    # -- tracking --
    ("tracking", "backend", "choice", "tracker",
     "tracktrack is built for crowded scenes with frequent occlusion; greedy is "
     "a nearest-centroid fallback that is adequate at high frame rates.",
     {"options": ["bytetrack", "botsort", "tracktrack", "ocsort", "deepocsort",
                  "fasttrack", "greedy"]}),
    ("tracking", "min_track_length", "int", "min track length",
     "Drop tracks seen in fewer frames than this. Must not exceed the number of "
     "frames actually processed.", {"min": 1, "max": 100000}),
    ("tracking", "greedy_max_dist_px", "float", "greedy max distance",
     "Only used by the greedy fallback tracker.", {"min": 1.0, "max": 500.0, "step": 5.0}),

    # -- normalize --
    ("normalize", "mode", "choice", "intensity mapping",
     "per_frame: own mapping per frame (flattest). smooth: corrects slow drift "
     "only. fixed: one mapping for the movie, faithful including flicker.",
     {"options": ["per_frame", "smooth", "fixed", "none"]}),
    ("normalize", "percentiles", "floatpair", "percentiles",
     "Low and high percentile mapped to 0 and 255.", {}),
    ("normalize", "smooth_window", "int", "smooth window",
     "Samples in the rolling median (smooth mode only).", {"min": 1, "max": 999}),

    # -- clips --
    ("clips", "frames", "int", "clip length",
     "Frames per clip - the unit you actually judge.", {"min": 2, "max": 10000}),
    ("clips", "stride", "int", "stride",
     "Spacing between windows. Equal to the clip length means non-overlapping.",
     {"min": 1, "max": 10000}),
    ("clips", "window_mode", "choice", "how much to cut",
     "first: only the movie's first window. first_per_track: the first full "
     "window of each track. all: every non-overlapping window of every track.",
     {"options": ["all", "first_per_track", "first"]}),
    ("clips", "follow", "choice", "crop follows",
     "track: re-centred every frame - smallest crop, translation invisible. "
     "smooth: follows a moving average. fixed: one crop for the window.",
     {"options": ["smooth", "track", "fixed"]}),
    ("clips", "pad", "float", "crop padding",
     "Crop side = the per-frame box extent times this.", {"min": 1.0, "max": 8.0, "step": 0.1}),
    ("clips", "crop_size", "scalar", "cache crop size",
     "'auto' takes the 99th percentile of the required sizes, rounded to a "
     "multiple of 16. A number forces it.", {}),
    ("clips", "max_size", "int", "max crop size",
     "Cap for auto. Larger crops are downscaled and their scale recorded.",
     {"min": 32, "max": 1024, "step": 16}),
    ("clips", "min_clip_frames", "int", "min detected frames",
     "A window needs this many detected frames to become a clip.", {"min": 1, "max": 10000}),
    ("clips", "max_clips_per_track", "int", "max clips per track",
     "0 means unlimited.", {"min": 0, "max": 10000}),
    ("clips", "context_tracks", "bool", "record neighbours",
     "Record neighbouring tracks that fall inside the crop, so they can be "
     "outlined on demand.", {}),
    ("clips", "thumbnails", "bool", "full-frame thumbnails",
     "Needed for the context minimap. Costs about 65 kB per clip.", {}),

    # -- ui --
    ("ui", "fps", "int", "playback fps",
     "How fast the clip loops in the GUI - not the acquisition rate.",
     {"min": 1, "max": 120}),
    ("ui", "follow", "choice", "view follows",
     "centred: one view on the track's median position, all motion visible. "
     "smooth: drift removed, beat kept. locked: re-centred every frame.",
     {"options": ["centred", "smooth", "locked"]}),
    ("ui", "view_zoom", "float", "view zoom",
     "Magnification into the crop. 1.0 shows the whole crop.",
     {"min": 1.0, "max": 4.0, "step": 0.25}),
    ("ui", "show_overlay", "bool", "show the marker",
     "Outline the animal this clip is about.", {}),
    ("ui", "show_context_tracks", "bool", "outline neighbours too",
     "Off by default: the marker's job is to say which animal is being judged.", {}),
    ("ui", "show_skeleton", "bool", "show keypoints",
     "Draw the pose skeleton on top of the marker.", {}),
    ("ui", "show_kymograph", "bool", "show curvature kymograph",
     "Plot the flagellum's curvature along its length against time, under the "
     "clip. Needs keypoints, so it only appears on a pose project.", {}),
    ("ui", "kymo_interpolate", "bool", "smooth the kymograph",
     "Filter the kymograph when it is drawn. Off shows the raw sample grid - "
     "one hard-edged block per frame - which is what you want when judging "
     "whether a track is noisy.", {}),
    ("ui", "kymo_field", "choice", "kymograph shows",
     "curvature (dpsi/ds, a second derivative: sharper but noisier) or the "
     "tangent angle psi itself (first order, unwrapped along the flagellum and "
     "over time). They are the same beat seen two standard ways.",
     {"options": ["curvature", "tangent"]}),
    ("ui", "kymo_sign", "choice", "kymograph sign",
     "A 2D image cannot tell a cell from its mirror image, so the same beat can "
     "come out blue in one clip and red in the next. 'canonical' flips each clip "
     "so its static curvature is positive when that is decidable, 'raw' leaves "
     "it as measured, 'absolute' drops the sign entirely.",
     {"options": ["canonical", "raw", "absolute"]}),
    ("ui", "kymo_smooth_px", "float", "kymograph fit tolerance",
     "How far the fitted centreline may pull away from each keypoint, in "
     "pixels. Curvature is a second derivative, so too little turns tracking "
     "jitter into stripes that are not there and too much flattens a real beat.",
     {"min": 0.0, "max": 5.0, "step": 0.1}),
    ("ui", "auto_advance", "bool", "auto-advance",
     "One keypress labels the clip and jumps to the next, once every required "
     "group is filled.", {}),
    ("ui", "shuffle", "bool", "shuffle clip order",
     "Seeded random order. Going movie-by-movie invites order effects - you "
     "calibrate on the first movie and drift.", {}),
    ("ui", "resume", "choice", "on open, go to",
     "Where the session resumes.",
     {"options": ["first_unannotated", "last_position"]}),
    ("ui", "annotator", "text", "annotator name",
     "Written into every row of the CSV.", {}),
    ("ui", "theme", "choice", "appearance",
     "dark, light, or follow the desktop.",
     {"options": ["dark", "light", "system"]}),

    # -- suggest --
    ("suggest", "enabled", "bool", "suggest labels",
     "Show what a classifier trained on your labels predicts. Displayed only, "
     "never pre-selected.", {}),
    ("suggest", "retrain_every", "int", "retrain every N labels",
     "How many new labels trigger a background refit.", {"min": 1, "max": 1000}),
    ("suggest", "min_per_class", "int", "min labels per class",
     "A class with fewer labels than this is not modelled.", {"min": 1, "max": 1000}),
    ("suggest", "min_macro_f1", "float", "min macro-F1",
     "The model stays silent below this cross-validated score.",
     {"min": 0.0, "max": 1.0, "step": 0.05}),
    ("suggest", "feature_groups", "groups", "features the model may use",
     "Which families of measurement the classifier is allowed to see. Groups "
     "that this project cannot compute are disabled with the reason.", {}),
    ("suggest", "model", "choice", "classifier",
     "catboost, the default, handles the lopsided label distribution and NaN "
     "natively; hist is scikit-learn's HistGradientBoosting.",
     {"options": ["catboost", "hist"]}),
    ("suggest", "balance_classes", "bool", "balance the classes",
     "Weight the loss by inverse class frequency. Without it the model answers "
     "a lopsided label set by predicting the dominant state.", {}),
    ("suggest", "iterations", "int", "boosting iterations",
     "More iterations fit harder. 400 is a sensible default for a few hundred "
     "labelled clips.", {"min": 10, "max": 5000, "step": 50}),
    ("suggest", "depth", "int", "tree depth",
     "CatBoost trees are symmetric, so depth alone sets the width: 2**depth "
     "leaves. Past 6 it overfits a hand-labelled set fast.", {"min": 1, "max": 10}),
    ("suggest", "learning_rate", "float", "learning rate",
     "Smaller learns more slowly and generalises better, given enough "
     "iterations.", {"min": 0.005, "max": 1.0, "step": 0.01, "decimals": 3}),
    ("suggest", "l2_leaf_reg", "float", "L2 regularisation",
     "Higher values penalise large leaf values, which helps on small label "
     "sets.", {"min": 0.0, "max": 30.0, "step": 0.5}),
    ("suggest", "pca_components", "int", "CNN components",
     "The CNN embeddings are reduced to this many components before they meet "
     "the labels. Only used when the CNN group is selected.",
     {"min": 2, "max": 256, "step": 2}),
    ("suggest", "min_confidence", "float", "minimum confidence",
     "A near-uniform posterior is noise with a number attached, and it still "
     "anchors the annotator. Below this, say nothing.",
     {"min": 0.0, "max": 1.0, "step": 0.05}),
    ("suggest", "seed", "int", "random seed",
     "Fixes the cross-validation split and the model fit.", {"min": 0, "max": 10**6}),
    ("suggest", "audit_fraction", "float", "audit holdout",
     "Share of clips that never show a suggestion and are never trained on, so "
     "human-vs-model agreement stays measurable. This is why a confident model "
     "still shows nothing on roughly one clip in ten; set it to 0 to suggest on "
     "every clip and give up the measure.",
     {"min": 0.0, "max": 0.9, "step": 0.05}),
    ("suggest", "audit_visible", "bool", "say when a clip is held out",
     "Mark held-out clips on screen instead of just showing nothing. Off by "
     "default: knowing a clip is being audited can change how carefully it is "
     "judged, which is the very influence the holdout exists to exclude. Turn "
     "it on if silent clips are more confusing than that bias is costly.", {}),

    # -- render --
    ("render", "out_dir", "dir", "render output", "Where overlay movies are written.", {}),
    ("render", "hide_unlabelled", "bool", "hide unlabelled tracks",
     "Draw only tracks that carry an annotation.", {}),
    ("render", "show_track_id", "bool", "show track ids", "Label each box with its track id.", {}),
    ("render", "quality", "int", "video quality", "imageio quality, 1-10.", {"min": 1, "max": 10}),
]

SECTION_TITLES = {
    "paths": "Paths",
    "keypoints": "Keypoints",
    "detection": "Detection",
    "tiling": "Tiled inference",
    "tracking": "Tracking",
    "normalize": "Intensity",
    "clips": "Clip extraction",
    "ui": "Annotation GUI",
    "suggest": "Suggestions",
    "render": "Overlay movies",
}


def _parse_scalar(text: str) -> Any:
    """Free-text field back into the type YAML would have given it."""
    text = text.strip()
    low = text.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "none", ""):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


class ConfigForm(QScrollArea):
    """Scalar settings, grouped by section. ``scope`` selects which appear:
    ``"basic"`` is the short list to get started, ``"advanced"`` everything else,
    ``"all"`` the lot. Two instances let the Configure page put the tuning knobs
    behind a tab."""

    changed = Signal()

    def __init__(self, scope: str = "all", parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self._widgets: dict[str, QWidget] = {}
        self._kinds: dict[str, str] = {}
        self._status: dict[str, QLabel] = {}
        self._rows: dict[str, QHBoxLayout] = {}

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(14)

        by_section: dict[str, list] = {}
        for field in FIELDS:
            dotted = f"{field[0]}.{field[1]}"
            if scope == "basic" and dotted not in BASIC:
                continue
            if scope == "advanced" and dotted in BASIC:
                continue
            by_section.setdefault(field[0], []).append(field)

        for section, fields in by_section.items():
            box = QGroupBox(SECTION_TITLES.get(section, section))
            form = QFormLayout(box)
            form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
            form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            for _, key, kind, label, tip, extra in fields:
                dotted = f"{section}.{key}"
                if dotted in NEEDS_REPREPARE:
                    tip += ("\n\nChanging this means the clips have to be cut again "
                            "(Detect & crop, with 'redo' ticked).")
                widget = self._make_widget(kind, extra)
                widget.setToolTip(tip)
                name = QLabel(label + ("  *" if dotted in NEEDS_REPREPARE else ""))
                name.setToolTip(tip)
                form.addRow(name, self._with_status(dotted, widget))
                self._widgets[dotted] = widget
                self._kinds[dotted] = kind
            layout.addWidget(box)

        note = QLabel(
            "Settings marked <b>*</b> describe how the clips were cut. Changing one "
            "means re-running Detect &amp; crop with 'redo' ticked before it takes "
            "effect; everything else applies immediately."
        )
        note.setWordWrap(True)
        note.setProperty("role", "muted")
        layout.addWidget(note)
        layout.addStretch(1)
        self.setWidget(body)

    # -- widget construction ------------------------------------------------ #
    def _with_status(self, dotted: str, widget: QWidget) -> QWidget:
        """The field as it goes into the row, with room under it for a note.

        Same shape as :class:`_PathField`: the control keeps a row of its own so
        a page can put a button beside it, and a muted line underneath says what
        the value cannot.
        """
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
        """Say something under a field that its value cannot say for itself."""
        widget = self._widgets.get(dotted)
        if widget is not None and hasattr(widget, "set_status"):
            widget.set_status(text)
            return
        label = self._status.get(dotted)
        if label is None:
            return
        label.setText(text)
        label.setVisible(bool(text))

    def _make_widget(self, kind: str, extra: dict) -> QWidget:
        if kind == "bool":
            widget = QCheckBox()
            widget.stateChanged.connect(self.changed)
            return widget
        if kind == "choice":
            widget = QComboBox()
            widget.addItems(extra.get("options", []))
            widget.setEditable(True)  # a config may legitimately hold something newer
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
        widget = QLineEdit()
        widget.textChanged.connect(self.changed)
        return widget

    # -- values ------------------------------------------------------------- #
    def load(self, data: dict) -> None:
        """Fill the form from a raw config dict. Missing keys are left blank."""
        for dotted, widget in self._widgets.items():
            section, _, key = dotted.partition(".")
            block = data.get(section) or {}
            if key not in block:
                continue
            self._set(widget, self._kinds[dotted], block[key])

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
            elif kind in ("dir", "file"):
                widget.set_value(str(value))
            elif kind == "groups":
                widget.set_value(value)
            elif kind in ("strlist", "floatpair"):
                text = ", ".join(str(v) for v in (value or []))
                if widget.text() != text:
                    widget.setText(text)
            else:
                # Only touch it if it differs: setText moves the caret to the end
                # and drops the selection, which autosave would do mid-edit.
                text = "" if value is None else str(value)
                if widget.text() != text:
                    widget.setText(text)
        except (TypeError, ValueError):
            pass
        finally:
            widget.blockSignals(False)

    # -- validation --------------------------------------------------------- #
    def warnings(self, root: Path | None = None) -> dict[str, str]:
        """Fields that are syntactically fine but point at nothing. Warnings, not
        errors: an offline share must not block editing, and a half-typed path is
        a normal intermediate state."""
        out: dict[str, str] = {}
        values = self.values()

        def resolve(text: str) -> Path:
            p = Path(str(text))
            return p if p.is_absolute() or str(p).startswith("//") else (root or Path.cwd()) / p

        for key, kind, must in (
            ("paths.video_dir", "dir", "folder"),
            ("paths.model", "file", "file"),
            ("render.out_dir", "dir", None),
            ("paths.work_dir", "dir", None),
        ):
            if key not in values:
                continue
            text = str(values[key] or "").strip()
            if not text:
                out[key] = "not set"
                continue
            if must is None:
                continue
            # The shared bounded probe, never a bare is_dir(): these settings are
            # the ones that point at network shares, one is_dir() on a
            # disconnected share blocks for two minutes, and this method runs on
            # every config load and every save.
            target = resolve(text)
            if must == "folder" and not _exists(target, "dir"):
                out[key] = "this folder does not exist"
            elif must == "file" and not _exists(target, "file"):
                out[key] = "this file does not exist"

        for key in ("normalize.percentiles",):
            if key in self._widgets and key not in values:
                out[key] = "needs two numbers, e.g. 0.5, 99.5"
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
                target.setToolTip(message)

    def values(self) -> dict[str, Any]:
        """Dotted-key updates, ready for :func:`trackanno.project.write_config`."""
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
            elif kind in ("dir", "file", "groups"):
                out[dotted] = widget.value()
            elif kind == "strlist":
                out[dotted] = [p.strip() for p in widget.text().split(",") if p.strip()]
            elif kind == "floatpair":
                parts = [p.strip() for p in widget.text().split(",") if p.strip()]
                if len(parts) == 2:
                    out[dotted] = [float(parts[0]), float(parts[1])]
            else:
                out[dotted] = _parse_scalar(widget.text())
        return out


class _GroupPicker(QWidget):
    """Which feature families the model may use, with the unusable ones disabled
    and labelled with the reason: a plain checklist would let you pick keypoint
    features on a detect project and only fail at training time."""

    def __init__(self, changed_signal):
        super().__init__()
        from ..clipfeatures import FEATURE_GROUPS

        self._changed = changed_signal
        self._boxes: dict[str, QCheckBox] = {}
        self._available: dict[str, str] = {}
        self._guard = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        for name, spec in FEATURE_GROUPS.items():
            box = QCheckBox(f"{name}  -  {spec['label']}")
            # Checked by default, matching Suggest.feature_groups: starting empty
            # would silently propose "no inputs at all" on the next save.
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
        """Hold the "at least one group" rule here, where it can be explained.
        The config loader rejects an empty list and autosave fires 700 ms after
        every click, so unticking the last box would fail the whole save."""
        if self._guard:
            return
        box = self.sender()
        if not self.value() and isinstance(box, QCheckBox):
            self._guard = True
            box.setChecked(True)
            self._guard = False
            self.note.setText(
                "The model needs at least one feature group. Tick the one you "
                "want before unticking the last one."
            )
            self.note.setVisible(True)
            return
        self._refresh_note()
        self._changed.emit()

    def _refresh_note(self) -> None:
        """Warn when everything ticked is something this project cannot compute:
        the config only checks the list is non-empty, so {pose, waveform} on a
        detect project passes validation and fails at training time instead."""
        chosen = self.value()
        usable = [g for g in chosen if not self._available.get(g)]
        if chosen and not usable and self._available:
            self.note.setText(
                "None of the ticked groups can be computed for this project - "
                + "; ".join(f"{g} {self._available[g]}" for g in chosen)
                + ". Training would have no inputs at all."
            )
            self.note.setVisible(True)
        else:
            self.note.setVisible(False)

    def set_available(self, availability: dict) -> None:
        from ..clipfeatures import FEATURE_GROUPS

        self._available = dict(availability)
        for name, box in self._boxes.items():
            why = availability.get(name, "")
            # Left checked, not unchecked: the tick records what you WANT, so a
            # project temporarily on 'detect' gets its keypoint features back on
            # returning to 'pose'. Training drops unavailable groups and says so.
            box.setEnabled(not why)
            base = FEATURE_GROUPS[name]["about"]
            box.setToolTip(("UNAVAILABLE: " + why + "\n\n" + base) if why else base)
            label = f"{name}  -  {FEATURE_GROUPS[name]['label']}"
            box.setText(f"{label}   ({why})" if why else label)
        self._refresh_note()

    def set_value(self, groups) -> None:
        wanted = set(groups or [])
        self._guard = True
        try:
            for name, box in self._boxes.items():
                box.setChecked(name in wanted)
        finally:
            self._guard = False
        self._refresh_note()

    def value(self) -> list:
        return [n for n, b in self._boxes.items() if b.isChecked()]

    def setToolTip(self, text):  # noqa: N802 (Qt naming)
        super().setToolTip(text)


class _PathField(QWidget):
    """Line edit plus a Browse button that knows what it is picking."""

    # Browsing is a deliberate choice of THIS file, which is more than a text
    # change: paths.model may be shadowed by a shared model, and only the
    # picking tells the page the person meant to stop using it.
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
        # A second line under the row, for a field whose value is not the whole
        # story: paths.model reads as a file even when a downloaded model is
        # what the project actually runs.
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
        if self.edit.text() != text:
            self.edit.setText(text)

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
    """The buttons the annotator will press, and the keys that press them."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._groups: list[_GroupEditor] = []

        self._container = QVBoxLayout()
        self._container.setSpacing(8)

        add = QPushButton("+ add group")
        add.clicked.connect(lambda: self.add_group())

        hint = QLabel(
            "A <b>single</b> group is a set of mutually exclusive choices and can "
            "auto-advance; a <b>multi</b> group is checkboxes and never advances on "
            "its own. Shortcut keys must be unique and must not collide with the "
            "navigation keys (arrows, Tab, Space, o, s, c, u, n, m, 0, + and -)."
        )
        hint.setWordWrap(True)
        hint.setProperty("role", "muted")

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
        """What the config loader would refuse, said before it is written."""
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
                # Each group becomes a CSV column and would overwrite a built-in.
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
        self.required.setToolTip("auto-advance waits until every required group is set")
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

        self.remove_btn = compact_button("x", 26, "remove this option", danger=True)
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
    """A table row tall enough for a themed combo box, measured not guessed: the
    stylesheet's padding and border make an input taller than Qt's default
    30 px row."""
    probe = QComboBox()
    probe.addItem("Head")
    return max(30, probe.sizeHint().height() + 8)


class KeypointEditor(QWidget):
    """The keypoint layout: a count you do not type, and names you fill in.

    The count comes from the checkpoint; the names do not, since no ultralytics
    checkpoint carries them. The ORDER is a contract: ``clipfeatures`` addresses
    keypoints by position - 0 the head, 1 the base and body-frame origin, 1..end
    the flagellum chain. Keypoints in a different order still produce
    body-length and beat numbers, and they mean nothing.
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
            "Load the checkpoint and take the count from it. The model is the "
            "authority on how many; the names are yours to choose."
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
            "<b>Order matters, names do not.</b> The pose features read keypoints by "
            "position: <b>0</b> is the head, <b>1</b> the base (the body-frame origin), "
            "and <b>1</b> onwards the flagellum chain. A model that predicts them in a "
            "different order will still produce body-length and beat numbers, and they "
            "will be meaningless."
        )
        order_note.setWordWrap(True)
        order_note.setProperty("role", "muted")

        # -- skeleton --
        self.edges = QTableWidget(0, 3)
        self.edges.setHorizontalHeaderLabels(["from", "to", ""])
        self.edges.verticalHeader().setVisible(False)
        self.edges.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.edges.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        # Fixed, not ResizeToContents: a scrollbar pushes the last column off-edge.
        self.edges.horizontalHeader().setSectionResizeMode(2, QHeaderView.Fixed)
        self.edges.setColumnWidth(2, 44)
        self.edges.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.edges.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.edges.verticalHeader().setDefaultSectionSize(_row_height())

        add_edge = QPushButton("+ add link")
        add_edge.clicked.connect(lambda: self._add_edge_row())
        chain = QPushButton("connect in order")
        chain.setToolTip("Link 0-1, 1-2, ... which is what a flagellum chain looks like.")
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
        hint = QLabel(
            "Only drawn on screen and written to the SLEAP export. It has no effect "
            "on detection, tracking or the features."
        )
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
        """Size a table to its rows so nothing is clipped. Both tables live in a
        scroll area, so letting them grow and letting the page scroll shows every
        keypoint at once instead of hiding rows behind an inner scrollbar."""
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
        """Keep the link dropdowns in step with renamed keypoints, tracked by
        POSITION and never by name: a renamed keypoint would leave its links
        looking for text that no longer exists and silently fall back to 0."""
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
        """What load_config would refuse, said before it is written."""
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
