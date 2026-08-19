"""The whole pipeline in one window.

    uv run gui.py

Seven pages down the left, in the order the work happens: pick a project,
configure it, run detection, annotate, train the suggester, check the labels,
export overlay movies. Each page states its own prerequisites rather than
failing when you press the button, so the sidebar doubles as a progress readout.

The CLI entry points call the same ``prepare_all`` / ``train`` / ``render_all``
functions, so the two paths cannot drift apart.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qtpy.QtCore import QSize, Qt, QTimer, Signal
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout, QFrame,
    QGridLayout, QGroupBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget,
    QListWidgetItem,
    QMainWindow, QMessageBox, QPushButton, QScrollArea, QSpinBox, QSplitter,
    QStackedWidget, QTabWidget, QTextBrowser, QVBoxLayout, QWidget,
)

from ..clips import MANIFEST_FILE, ClipStore
from ..config import ConfigError, load_config
from ..project import (
    CONFIG_NAME, _exists, config_path_for, create_project, project_status, recent_projects,
    remember_project, write_config,
)
from ..store import AnnotationStore
from .app import AnnotationWidget
from .configform import CategoryEditor, ConfigForm, KeypointEditor
from .distribution import BarChart, summary_lines, tally_annotations
from .theme import (
    Heading, app_icon, apply_theme, banner_style, compact_button, current,
    set_role, theme_icon, tokens,
)
from .tasks import TaskPanel, TaskRunner

PAGES = ["Project", "Configure", "Detect & crop", "Annotate", "Model",
         "Distribution", "Export"]

PAGE_HINTS = [
    "Pick the folder to work in",
    "Point it at your movies and choose your labels",
    "Find the animals and cut them into clips",
    "Watch each clip and press one key",
    "Learn from your labels and suggest the next one",
    "See how your labels are spread before you commit to them",
    "Put the labels back onto the original movies",
]


def _heading(text: str, subtitle: str = "") -> QWidget:
    return Heading(text, subtitle)


def _muted(label: QLabel) -> QLabel:
    label.setProperty("role", "muted")
    return label


class Page(QWidget):
    """Common contract: a page is told when the project changes."""

    status_message = Signal(str, int)

    def __init__(self, hub, parent=None):
        super().__init__(parent)
        self.hub = hub

    @property
    def cfg(self):
        return self.hub.cfg

    def on_project(self) -> None:
        pass

    def blocker(self) -> str:
        """Why this page cannot be used yet, or "" if it can."""
        return "" if self.cfg is not None else "no project open"

    def restyle(self) -> None:
        """Re-apply any style this page sets in code. Called on a theme change."""

    def refresh_gate(self) -> None:
        """Re-evaluate this page's blocker cheaply. Runs after every annotation,
        so it may only flip enabled states - never re-read a CSV or rebuild."""

    def shutdown(self) -> None:
        pass


# --------------------------------------------------------------------------- #
# 1. project
# --------------------------------------------------------------------------- #
class ProjectPage(Page):
    def __init__(self, hub, parent=None):
        super().__init__(hub, parent)

        new_btn = QPushButton("New project...")
        new_btn.setProperty("accent", "true")
        new_btn.clicked.connect(self._new_project)
        open_btn = QPushButton("Open project...")
        open_btn.clicked.connect(self._open_project)
        self.template_btn = QPushButton("Save as template...")
        self.template_btn.setToolTip(
            "Keep this project's settings, minus its paths, to start the next "
            "project from."
        )
        self.template_btn.clicked.connect(self._save_template)
        self.template_btn.setEnabled(False)

        buttons = QHBoxLayout()
        buttons.addWidget(new_btn)
        buttons.addWidget(open_btn)
        buttons.addWidget(self.template_btn)
        buttons.addStretch(1)

        self.recent = QListWidget()
        self.recent.itemActivated.connect(self._open_recent)
        self.recent.setAlternatingRowColors(True)
        recent_box = QGroupBox("Recent")
        recent_layout = QVBoxLayout(recent_box)
        recent_layout.addWidget(self.recent)

        self.summary = QGridLayout()
        self.summary.setColumnStretch(1, 1)
        self._summary_values: dict[str, QLabel] = {}
        for row, field in enumerate(
            ["project", "videos", "prepared", "clips", "annotated", "suggester"]
        ):
            name = QLabel(field)
            name.setProperty("role", "muted")
            value = QLabel("-")
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            value.setWordWrap(True)
            self.summary.addWidget(name, row, 0, Qt.AlignTop)
            self.summary.addWidget(value, row, 1)
            self._summary_values[field] = value

        self.next_step = QLabel("")
        self.next_step.setWordWrap(True)
        self.next_step.setTextFormat(Qt.RichText)

        self.go_button = QPushButton("")
        self.go_button.setProperty("accent", "true")
        self.go_button.setVisible(False)
        self.go_button.clicked.connect(self._go_next)
        self._next_page = 1

        status_box = QGroupBox("This project")
        status_layout = QVBoxLayout(status_box)
        status_layout.addLayout(self.summary)
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        status_layout.addWidget(line)
        status_layout.addWidget(self.next_step)
        status_layout.addWidget(self.go_button, 0, Qt.AlignLeft)

        layout = QVBoxLayout(self)
        layout.addWidget(_heading(
            "Project",
            "A project is a folder holding config.yaml, the clip cache and the "
            "annotation CSV. Everything else on this window operates on the open one.",
        ))
        layout.addLayout(buttons)
        layout.addWidget(status_box)
        layout.addWidget(recent_box, 1)
        self.refresh_recent()

    # -- actions ------------------------------------------------------------ #
    def refresh_recent(self) -> None:
        self.recent.clear()
        for path in recent_projects():
            item = QListWidgetItem(f"{path.parent.name}      {path.parent}")
            item.setData(Qt.UserRole, str(path))
            self.recent.addItem(item)
        if self.recent.count() == 0:
            self.recent.addItem("nothing yet - create or open a project above")

    def _open_recent(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.UserRole)
        if path:
            self.hub.open_project(path)

    def _open_project(self) -> None:
        picked, _ = QFileDialog.getOpenFileName(
            self, "Open a project's config.yaml", str(Path.home()),
            "Config (config.yaml *.yaml *.yml);;All files (*)",
        )
        if picked:
            self.hub.open_project(picked)

    def _new_project(self) -> None:
        dialog = NewProjectDialog(self)
        if dialog.run():
            try:
                path = create_project(**dialog.values())
            except (OSError, FileExistsError) as exc:
                QMessageBox.warning(self, "Could not create project", str(exc))
                return
            self.hub.open_project(path)
            self.hub.go_to(1)
            self.status_message.emit(
                "project created - check the settings, then run detection", 8000
            )

    def _save_template(self) -> None:
        """Store the open project's settings for the next project to start from."""
        from ..templates import find_template, save_template

        if self.cfg is None:
            return
        suggested = self.cfg.path.parent.name
        name, ok = QInputDialog.getText(
            self, "Save as template", "Name for these settings:", text=suggested
        )
        name = (name or "").strip()
        if not ok or not name:
            return
        if find_template(name) is not None:
            answer = QMessageBox.question(
                self, "Save as template",
                f"A template named {name!r} already exists. Replace it?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        try:
            template = save_template(name, self.cfg.path, overwrite=True)
        except Exception as exc:
            QMessageBox.warning(self, "Save as template", str(exc))
            return
        self.status_message.emit(
            f"saved template {template.name!r} - New project can start from it", 8000
        )

    def _go_next(self) -> None:
        self.hub.go_to(self._next_page)

    def refresh_gate(self) -> None:
        # Cheap: repaint counts and advice from hub.status, no recents rebuild.
        if self.cfg is None or self.hub.status is None:
            return
        self._paint_summary()

    # -- display ------------------------------------------------------------ #
    def on_project(self) -> None:
        # Saving a template needs a config, nothing more: it copies the file and
        # does not care whether anything has been prepared or annotated.
        self.template_btn.setEnabled(self.cfg is not None)
        self.refresh_recent()
        self._paint_summary()

    def _paint_summary(self) -> None:
        if self.cfg is None or self.hub.status is None:
            for label in self._summary_values.values():
                label.setText("-")
            self.next_step.setText("Open or create a project to begin.")
            self.go_button.setVisible(False)
            return

        status = self.hub.status
        self._summary_values["project"].setText(str(self.cfg.path))
        self._summary_values["videos"].setText(
            f"{status.n_videos} in {self.cfg.paths.video_dir}"
            + ("" if status.video_dir_ok else "   (folder not found)")
        )
        self._summary_values["prepared"].setText(f"{status.n_prepared} movie(s)")
        self._summary_values["clips"].setText(f"{status.n_clips}")
        self._summary_values["annotated"].setText(
            f"{status.n_annotated}"
            + (f"  ({status.n_annotated / status.n_clips:.0%})" if status.n_clips else "")
        )
        self._summary_values["suggester"].setText(status.suggester or "not trained")

        advice, page, label = self.hub.next_step()
        self.next_step.setText(advice)
        self._next_page = page
        self.go_button.setText(label)
        self.go_button.setVisible(bool(label))


class NewProjectDialog(QDialog):
    """Deliberately a plain form: a project is a name and three paths."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New project")
        self.setMinimumWidth(560)

        self.name = QLineEdit("my-annotation-project")
        self.location = _Browse("dir", str(Path.home()))
        self.videos = _Browse("dir", "")
        self.model = _Browse("file", "", "PyTorch weights (*.pt);;All files (*)")

        self.template = QComboBox()
        self.forget_btn = QPushButton("Forget")
        self.forget_btn.setToolTip("Delete the selected template.")
        self.forget_btn.clicked.connect(self._forget_template)
        self._fill_templates()
        template_row = QHBoxLayout()
        template_row.addWidget(self.template, 1)
        template_row.addWidget(self.forget_btn)

        form = QFormLayout()
        form.addRow("project name", self.name)
        form.addRow("create in", self.location)
        form.addRow("start from", template_row)
        form.addRow("video folder", self.videos)
        form.addRow("detection model", self.model)

        note = QLabel(
            "The clip cache and annotations.csv are created inside the project "
            "folder. A template supplies every other setting, and 'Save as "
            "template' on the Project page makes one from a project you have "
            "already tuned. You can change any of this on the Configure page."
        )
        note.setWordWrap(True)
        note.setProperty("role", "muted")

        ok = QPushButton("Create")
        ok.setDefault(True)
        ok.clicked.connect(self._accept)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(cancel)
        buttons.addWidget(ok)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(note)
        layout.addLayout(buttons)

    def _accept(self) -> None:
        """Refuse what cannot work; warn about what merely will not work yet."""
        if not self.name.text().strip():
            QMessageBox.warning(self, "New project", "Give the project a name.")
            return
        if not self.location.value():
            QMessageBox.warning(self, "New project", "Choose where to create it.")
            return
        target = Path(self.location.value()) / self.name.text().strip()
        if _exists(target / CONFIG_NAME):
            QMessageBox.warning(
                self, "New project",
                f"There is already a project in\n{target}\n\nOpen it instead, or "
                "pick a different name.",
            )
            return

        missing = []
        if not self.videos.value():
            missing.append("no movie folder chosen")
        elif not _exists(Path(self.videos.value()), "dir"):
            missing.append(f"the movie folder does not exist:\n    {self.videos.value()}")
        if not self.model.value():
            missing.append("no detection model chosen")
        elif not _exists(Path(self.model.value()), "file"):
            missing.append(f"the model file does not exist:\n    {self.model.value()}")
        if missing:
            answer = QMessageBox.question(
                self, "Create anyway?",
                "The project will be created, but detection (step 3) cannot run until "
                "this is sorted:\n\n- " + "\n- ".join(missing)
                + "\n\nYou can set both on the Configure page. Create it anyway?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        self.accept()

    def run(self) -> bool:
        """Qt5 spells it exec_, Qt6 exec; qtpy does not alias it on QDialog."""
        runner = getattr(self, "exec_", None) or self.exec
        return bool(runner())

    def _fill_templates(self) -> None:
        from ..templates import list_templates

        current = self.template.currentData()
        self.template.clear()
        for template in list_templates():
            self.template.addItem(template.label, template.name)
        index = self.template.findData(current)
        self.template.setCurrentIndex(max(0, index))
        self._sync_forget()
        self.template.currentIndexChanged.connect(self._sync_forget)

    def _sync_forget(self) -> None:
        from ..templates import BUILTIN_NAME

        self.forget_btn.setEnabled(self.template.currentData() != BUILTIN_NAME)

    def _forget_template(self) -> None:
        from ..templates import delete_template

        name = self.template.currentData()
        answer = QMessageBox.question(
            self, "Forget template", f"Delete the template {name!r}?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if answer == QMessageBox.Yes and delete_template(name):
            self._fill_templates()

    def values(self) -> dict:
        directory = Path(self.location.value()) / self.name.text().strip()
        return {
            "directory": directory,
            "video_dir": self.videos.value() or None,
            "model": self.model.value() or None,
            "template": self.template.currentData(),
        }


class _Browse(QWidget):
    def __init__(self, kind: str, initial: str = "", filter_: str = "All files (*)"):
        super().__init__()
        self._kind, self._filter = kind, filter_
        self.edit = QLineEdit(initial)
        button = QPushButton("Browse")
        button.setFixedWidth(70)
        button.clicked.connect(self._browse)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.edit, 1)
        layout.addWidget(button)

    def value(self) -> str:
        return self.edit.text().strip()

    def _browse(self) -> None:
        start = self.edit.text().strip() or str(Path.home())
        if self._kind == "dir":
            picked = QFileDialog.getExistingDirectory(self, "Choose folder", start)
        else:
            picked, _ = QFileDialog.getOpenFileName(self, "Choose file", start, self._filter)
        if picked:
            self.edit.setText(str(Path(picked)).replace("\\", "/"))


# --------------------------------------------------------------------------- #
# 2. configure
# --------------------------------------------------------------------------- #
class ConfigPage(Page):
    def __init__(self, hub, parent=None):
        super().__init__(hub, parent)
        # Two forms over the same file: the decisions you have to make, and the
        # tuning knobs. One 54-row scroll is an unusable settings screen.
        self.form = ConfigForm("basic")
        self.advanced = ConfigForm("advanced")
        self.categories = CategoryEditor()
        self._dirty = False
        self._loading = False

        category_scroll = QScrollArea()
        category_scroll.setWidgetResizable(True)
        category_scroll.setWidget(self.categories)

        self.keypoints = KeypointEditor()
        self.keypoints.count_requested.connect(self._read_keypoints)
        keypoint_scroll = QScrollArea()
        keypoint_scroll.setWidgetResizable(True)
        keypoint_scroll.setWidget(self.keypoints)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.form, "Basics")
        self.tabs.addTab(category_scroll, "Labels")
        self.tabs.addTab(keypoint_scroll, "Keypoints")
        self.tabs.addTab(self.advanced, "Advanced")
        self.tabs.setTabToolTip(0, "What this project points at and how long a clip is")
        self.tabs.setTabToolTip(1, "The buttons you will press while annotating")
        self.tabs.setTabToolTip(2, "Only used when the task is 'pose'")
        self.tabs.setTabToolTip(3, "Detector, tracker and cropping knobs - defaults are fine")
        tabs = self.tabs
        self.KEYPOINT_TAB = 2

        # At the top and persistent: an error that scrolls off is one nobody reads.
        self.banner = QLabel("")
        self.banner.setWordWrap(True)
        self.banner.setTextFormat(Qt.RichText)
        self.banner.setVisible(False)
        self.banner.setStyleSheet(banner_style("danger"))

        # `task: pose` needs a keypoint count, and the checkpoint knows it.
        self.pose_notice = QLabel(
            "This project is set to <b>pose</b> but no keypoints are listed. "
            "The model file knows how many it predicts."
        )
        self.pose_notice.setWordWrap(True)
        self.pose_notice.setStyleSheet(banner_style("warning"))
        self.pose_button = QPushButton("Set up keypoints")
        self.pose_button.setProperty("accent", "true")
        self.pose_button.clicked.connect(lambda: self.tabs.setCurrentIndex(self.KEYPOINT_TAB))
        self.pose_row = QWidget()
        pose_layout = QHBoxLayout(self.pose_row)
        pose_layout.setContentsMargins(0, 0, 0, 0)
        pose_layout.addWidget(self.pose_notice, 1)
        pose_layout.addWidget(self.pose_button)
        self.pose_row.setVisible(False)
        self._keypoint_reader: TaskRunner | None = None

        revert = QPushButton("Revert to saved")
        revert.clicked.connect(self.on_project)
        self.state = QLabel("")
        self.state.setWordWrap(True)

        bar = QHBoxLayout()
        bar.addWidget(self.state, 1)
        bar.addWidget(revert)

        # The usual way to get a model is to download a shared one, so this sits
        # in the "detection model" row itself, ahead of Browse.
        self.registry_btn = QPushButton("Choose shared model...")
        self.registry_btn.setProperty("accent", "true")
        self.registry_btn.setToolTip(
            "Download a model from the shared registry. It is fetched once, "
            "cached for later, and fills in the settings it was trained with. "
            "Browse instead for a .pt already on this machine."
        )
        self.registry_btn.clicked.connect(self._pick_registry_model)
        model_field = self.form._widgets.get("paths.model")
        if model_field is not None:
            model_field.layout().insertWidget(1, self.registry_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(_heading(
            "Configure",
            "Saved automatically, straight back into config.yaml - comments and all, "
            "so the file stays readable and hand-editable. Settings not shown here "
            "are left untouched.",
        ))
        layout.addWidget(self.banner)
        layout.addWidget(self.pose_row)
        layout.addWidget(tabs, 1)
        layout.addLayout(bar)

        # Probed off the UI thread: importing torch costs a second or two.
        self._device_probe: TaskRunner | None = None
        QTimer.singleShot(0, self._probe_devices)

        # Pick a .pt and the project configures itself: task, count, names table.
        self._model_probe: TaskRunner | None = None
        self._probed_model: str | None = None
        self._model_timer = QTimer(self)
        self._model_timer.setSingleShot(True)
        self._model_timer.setInterval(400)
        self._model_timer.timeout.connect(self._probe_model)
        self.form._widgets["paths.model"].edit.textChanged.connect(
            lambda _t: self._model_timer.start()
        )

        # Autosave, debounced: every save re-reads the whole project, and a path
        # field fires a signal per keystroke.
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(700)
        self._save_timer.timeout.connect(self.save)

        self.form.changed.connect(self._mark_dirty)
        self.advanced.changed.connect(self._mark_dirty)
        self.categories.changed.connect(self._mark_dirty)
        self.keypoints.changed.connect(self._mark_dirty)

    def restyle(self) -> None:
        self.banner.setStyleSheet(banner_style("danger"))
        self.pose_notice.setStyleSheet(banner_style("warning"))
        self._show_warnings()

    def _pick_registry_model(self) -> None:
        """Choose shared weights, fetch them, and let them configure the project."""
        from ..models import apply_manifest, fetch, mismatch_warnings

        if self.cfg is None:
            QMessageBox.information(self, "From registry", "Open a project first.")
            return
        dialog = ModelRegistryDialog(self)
        if not dialog.run():
            return
        entry = dialog.chosen()
        if entry is None:
            return

        def work(reporter):
            path = fetch(entry, reporter)
            import hashlib

            sha = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    sha.update(block)
            return path, sha.hexdigest()

        self.save()
        runner = TaskRunner(work, self)

        def done(result):
            path, digest = result
            try:
                told = apply_manifest(self.cfg.path, entry, digest)
            except Exception as exc:
                QMessageBox.warning(self, "From registry", str(exc))
                return
            self.hub.open_project(self.cfg.path)
            summary = ", ".join(told) if told else "already matched the settings"
            self.status_message.emit(f"{entry.id}: {summary}   ->   {path}", 12000)
            for warning in mismatch_warnings(self.hub.cfg, entry):
                self.status_message.emit(warning, 15000)

        runner.succeeded.connect(done)
        runner.failed.connect(
            lambda message: QMessageBox.warning(self, "From registry", message)
        )
        self._fetch_runner = runner
        self.status_message.emit(f"fetching {entry.id}...", 0)
        runner.start()

    def _probe_model(self) -> None:
        """Read whatever the selected checkpoint says about itself."""
        if self.cfg is None or (self._model_probe and self._model_probe.isRunning()):
            return
        raw = self.form._widgets["paths.model"].value()
        if not raw:
            return
        path = Path(raw)
        if not path.is_absolute():
            path = (self.cfg.path.parent / path).resolve()
        # Guarded: an unguarded is_file() on a network share blocks the window.
        if not _exists(path, "file") or str(path) == self._probed_model:
            return
        self._probed_model = str(path)
        self.keypoints.state.setText("reading the model file...")

        def work(reporter):
            from ..tiling import read_model_info

            return read_model_info(path)

        runner = TaskRunner(work, self)
        runner.succeeded.connect(self._on_model_info)
        runner.failed.connect(
            lambda message: self.keypoints.state.setText(f"could not read it: {message}")
        )
        self._model_probe = runner
        runner.start()

    def _on_model_info(self, info) -> None:
        """Apply what the checkpoint said, and say what changed."""
        if not info or self.cfg is None:
            return
        told: list[str] = []
        task_widget = self.form._widgets.get("detection.task")
        model_task = info.get("task")
        count = info.get("n_keypoints")

        # Follow the file: a project set to 'detect' has simply not been told.
        if model_task in ("pose", "detect") and task_widget is not None:
            if task_widget.currentText() != model_task:
                task_widget.setCurrentText(model_task)
                told.append(f"task {model_task}")

        if count:
            if self.keypoints.count.value() != count:
                self.keypoints.set_count(count, f"{count} from {info.get('name', 'the model')}")
                told.append(f"{count} keypoints")
            else:
                self.keypoints.state.setText(f"{count} from {info.get('name', 'the model')}")
        else:
            self.keypoints.state.setText("")

        classes = info.get("classes") or []
        summary = ", ".join(told) if told else "already matched the settings"
        self.status_message.emit(
            f"{info.get('name', 'model')}: {model_task or 'unknown task'}"
            + (f", {count} keypoints" if count else "")
            + (f", {len(classes)} class(es)" if classes else "")
            + f"   ->   {summary}",
            10000,
        )
        if told:
            self._mark_dirty()
        self._show_warnings()

    def _probe_devices(self) -> None:
        if self._device_probe is not None:
            return

        def work(reporter):
            from ..tiling import describe_devices

            return describe_devices()

        runner = TaskRunner(work, self)
        runner.succeeded.connect(self._on_devices)
        self._device_probe = runner
        runner.start()

    def _on_devices(self, devices) -> None:
        widget = self.form._widgets.get("detection.device")
        if widget is None or not devices:
            return
        current = widget.currentText()
        widget.blockSignals(True)
        widget.clear()
        for value, label in devices:
            widget.addItem(value)
            widget.setItemData(widget.count() - 1, label, Qt.ToolTipRole)
        widget.setCurrentText(current or "auto")
        widget.blockSignals(False)
        lines = [f"  {label}" for _value, label in devices]
        widget.setToolTip(
            "Available on this machine:\n" + "\n".join(lines)
            + "\n\n'auto' falls back to the CPU where there is no GPU, so the "
              "project stays portable between machines."
        )

    def _mark_dirty(self) -> None:
        if self._loading:
            return
        self._dirty = True
        self.state.setText("saving...")
        self._save_timer.start()

    def on_project(self) -> None:
        if self.cfg is None:
            return
        import yaml

        # Reloading fires every widget's changed signal; the guard stops the
        # reload triggered BY a save from scheduling another one.
        self._loading = True
        try:
            raw = yaml.safe_load(self.cfg.path.read_text(encoding="utf-8")) or {}
            self.form.load(raw)
            self.advanced.load(raw)
            self.categories.load(raw.get("categories") or [])
            kp = raw.get("keypoints") or {}
            self.keypoints.load(kp.get("nodes"), kp.get("edges"))
        finally:
            self._loading = False
        self._dirty = False
        self._save_timer.stop()
        self.banner.setVisible(False)
        self.state.setText("")
        self._show_warnings()

    def _show_warnings(self) -> None:
        """Amber outlines for things that point at nothing. Never blocks a save."""
        root = self.cfg.path.parent if self.cfg else None
        warnings = {**self.form.warnings(root), **self.advanced.warnings(root)}
        self.form.mark(warnings)
        self.advanced.mark(warnings)
        picker = self.advanced._widgets.get("suggest.feature_groups")
        if picker is not None and self.cfg is not None:
            from ..clipfeatures import available_groups

            picker.set_available(available_groups(self.cfg))
        self.pose_row.setVisible(bool(self.cfg and self.cfg.needs_keypoints))
        self.tabs.setTabEnabled(self.KEYPOINT_TAB, bool(self.cfg and self.cfg.is_pose))
        return warnings

    def _read_keypoints(self) -> None:
        """Ask the checkpoint how many keypoints it predicts. Names stay yours."""
        if self.cfg is None or (self._keypoint_reader and self._keypoint_reader.isRunning()):
            return
        from ..models import local_model_path

        model_path = local_model_path(self.cfg)
        if not _exists(model_path, "file"):
            self._fail("Cannot read the keypoint count:",
                       [f"the model file was not found: {model_path}"])
            return
        self.keypoints.read_button.setEnabled(False)
        self.keypoints.state.setText("reading the model file...")

        def work(reporter):
            from ..tiling import read_keypoint_count

            return read_keypoint_count(model_path)

        runner = TaskRunner(work, self)
        runner.succeeded.connect(self._on_keypoints_read)
        runner.failed.connect(self._on_keypoints_failed)
        self._keypoint_reader = runner
        runner.start()

    def _on_keypoints_read(self, count) -> None:
        self.keypoints.read_button.setEnabled(True)
        if not count:
            self.keypoints.state.setText("")
            self._fail(
                "That model does not predict keypoints, so it cannot be used for pose:",
                [f"{self.cfg.paths.model} reports no keypoint shape. Either set the "
                 "task back to 'detect', or point at pose weights."],
            )
            return
        # Only the COUNT comes from the model. Names already typed are kept, and
        # any new rows get kp<N> placeholders to be renamed in the table.
        self.keypoints.set_count(count, f"{count} from {self.cfg.paths.model.name}")
        self.tabs.setCurrentIndex(self.KEYPOINT_TAB)
        self.status_message.emit(
            f"{self.cfg.paths.model.name} predicts {count} keypoints - name them in "
            "the table (order is what has to match the model)", 10000
        )

    def _on_keypoints_failed(self, message: str) -> None:
        self.keypoints.read_button.setEnabled(True)
        self.keypoints.state.setText("")
        self._fail("Could not read the model file:", [message])

    def save(self) -> bool:
        """Validate and write. Called by the autosave timer, not by a button."""
        self._save_timer.stop()
        if self.cfg is None:
            return False

        problems = self.categories.problems()
        if problems:
            self._fail(
                "These labels are not usable yet, so nothing was saved:",
                problems,
            )
            return False
        # EVERY tab: one file, so writing only the visible one drops the others.
        updates = {**self.form.values(), **self.advanced.values()}

        # Judged on the value about to be WRITTEN, not the loaded config: a pose
        # checkpoint changes task and keypoints in one save. Only when the project
        # is actually pose - a detect project with a stale keypoints block would
        # be unsaveable with the tab that fixes it greyed out.
        is_pose = str(updates.get("detection.task", "")) == "pose"
        if is_pose:
            problems = self.keypoints.problems()
            if problems:
                self._fail("The keypoint layout is not usable yet, so nothing was saved:",
                           problems)
                return False

        try:
            # Validated through the real loader first: a bad edit never reaches disk.
            write_config(self.cfg.path, updates,
                         categories=self.categories.values(),
                         keypoints=self.keypoints.values() if is_pose else None)
        except (ConfigError, ValueError, OSError) as exc:
            self._fail("This setting is not valid, so nothing was saved:", [str(exc)])
            return False

        self.banner.setVisible(False)
        self._dirty = False
        # Reload WITHOUT rebuilding this page: the category editor re-creates
        # every row, so an autosave mid-word destroys the field holding the caret.
        self.hub.reload_project(skip=self)
        warnings = self._show_warnings()
        self.state.setText(
            "saved" if not warnings
            else f"saved - but {len(warnings)} path(s) point at nothing (outlined above)"
        )
        self.status_message.emit(f"saved {self.cfg.path.name}", 4000)
        return True

    def _fail(self, headline: str, details: list[str]) -> None:
        items = "".join(f"<li>{d}</li>" for d in details[:5])
        self.banner.setText(f"<b>{headline}</b><ul>{items}</ul>")
        self.banner.setVisible(True)
        self.state.setText("NOT saved")
        self.status_message.emit(details[0][:120], 8000)

    def maybe_save(self) -> bool:
        """Flush any pending autosave on the way out. Never blocks navigation."""
        if self._dirty and self.cfg is not None:
            self.save()
        return True

    def shutdown(self) -> None:
        for runner in (self._keypoint_reader, self._device_probe, self._model_probe):
            if runner is not None:
                runner.cancel()
                runner.wait(3000)


# --------------------------------------------------------------------------- #
# 3. detect & crop
# --------------------------------------------------------------------------- #
class ModelRegistryDialog(QDialog):
    """Pick shared weights, and set where this machine looks for them.

    The registry URL is edited here rather than in ``config.yaml`` on purpose: a
    project someone sends you may name a model, but must not be able to decide
    which server the weights come from.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        from ..models import registry_url

        self.setWindowTitle("Choose a shared model")
        self.setMinimumWidth(640)
        self._entries: list = []

        self.url = QLineEdit(registry_url())
        self.url.setPlaceholderText(
            "https://github.com/<owner>/<repo>/releases/latest/download/index.json"
            "   or   github://owner/repo"
        )
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self._refresh)
        url_row = QHBoxLayout()
        url_row.addWidget(self.url, 1)
        url_row.addWidget(refresh)

        self.list = QListWidget()
        self.list.setAlternatingRowColors(True)
        self.list.itemSelectionChanged.connect(self._describe)
        self.detail = QLabel("")
        self.detail.setWordWrap(True)
        self.detail.setProperty("role", "muted")

        self.ok = QPushButton("Download and use")
        self.ok.setDefault(True)
        self.ok.setEnabled(False)
        self.ok.clicked.connect(self.accept)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(cancel)
        buttons.addWidget(self.ok)

        form = QFormLayout()
        form.addRow("registry", url_row)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.list, 1)
        layout.addWidget(self.detail)
        layout.addLayout(buttons)
        self._fill(refresh=False)

    def _refresh(self) -> None:
        from ..models import RegistryError, set_registry_url

        try:
            set_registry_url(self.url.text())
        except RegistryError as exc:
            QMessageBox.warning(self, "Detection models", str(exc))
            return
        self._fill(refresh=True)

    def _fill(self, refresh: bool) -> None:
        from ..models import RegistryError, cached_path, list_models, registry_url

        self.list.clear()
        if not registry_url():
            self.detail.setText(
                "No registry is set on this machine. Paste the index URL above "
                "and press Refresh. Models stay available offline once fetched."
            )
            return
        try:
            offered = list_models(refresh=refresh)
        except RegistryError as exc:
            self._entries = []
            self.detail.setText(str(exc))
            return
        # Only what this project can run. The rest of the registry is counted
        # rather than listed, so the choice stays a choice.
        self._entries = [e for e in offered if e.runnable]
        for entry in self._entries:
            here = cached_path(entry.sha256, entry.name) is not None
            self.list.addItem(QListWidgetItem(entry.label + ("   [cached]" if here else "")))
        others = len(offered) - len(self._entries)
        if not self._entries:
            self.detail.setText(
                f"This registry offers {len(offered)} model(s), none of which "
                "trackanno can run." if offered
                else "The registry index lists no models."
            )
        elif others:
            self.detail.setText(
                f"{others} further model(s) here are for other tools, not shown."
            )

    def _describe(self) -> None:
        entry = self.chosen()
        self.ok.setEnabled(entry is not None and entry.runnable)
        if entry is None:
            return
        size = f"{entry.size / 1e6:.0f} MB" if entry.size else "size unknown"
        trained = f"   |   trained at {entry.trained_at}" if entry.trained_at else ""
        self.detail.setText(f"{entry.notes or 'no description'}\n{size}{trained}")

    def chosen(self):
        row = self.list.currentRow()
        if 0 <= row < len(self._entries):
            return self._entries[row]
        return None

    def run(self) -> bool:
        runner = getattr(self, "exec_", None) or self.exec
        return bool(runner())


class PreparePage(Page):
    def __init__(self, hub, parent=None):
        super().__init__(hub, parent)
        self._scan: TaskRunner | None = None
        self._scanned_for: Path | None = None

        self.movies = QListWidget()
        self.movies.setSelectionMode(QListWidget.NoSelection)
        self.movies.setAlternatingRowColors(True)

        scan_btn = QPushButton("Scan video folder")
        scan_btn.clicked.connect(self.scan)
        self.select_all = QPushButton("select all")
        self.select_all.clicked.connect(lambda: self._set_all(True))
        self.select_none = QPushButton("select none")
        self.select_none.clicked.connect(lambda: self._set_all(False))
        self.scan_state = QLabel("not scanned yet")
        self.scan_state.setProperty("role", "muted")
        self.scan_state.setWordWrap(True)

        # A blocked page that only complains is a dead end; this jumps to the fix.
        self.fix_button = QPushButton("Fix this in step 2 - Configure")
        self.fix_button.setProperty("accent", "true")
        self.fix_button.setVisible(False)
        self.fix_button.clicked.connect(lambda: self.hub.go_to(1))

        movie_bar = QHBoxLayout()
        movie_bar.addWidget(scan_btn)
        movie_bar.addWidget(self.select_all)
        movie_bar.addWidget(self.select_none)
        movie_bar.addWidget(self.fix_button)
        movie_bar.addWidget(self.scan_state, 1)

        self.force = QCheckBox("redo movies that are already prepared")
        self.limit_frames = QSpinBox()
        self.limit_frames.setRange(0, 1_000_000)
        self.limit_frames.setSpecialValueText("all frames")
        self.limit_frames.setToolTip(
            "Stop after this many frames per movie. Useful for a smoke test "
            "before committing to a full run."
        )
        self.device = QLineEdit()
        self.device.setPlaceholderText("from config")
        self.device.setToolTip("Override detection.device for this run, e.g. cpu")

        options = QFormLayout()
        options.addRow(self.force)
        options.addRow("frame limit", self.limit_frames)
        options.addRow("device override", self.device)
        options_box = QGroupBox("Run options")
        options_box.setLayout(options)

        self.tasks = TaskPanel("Run detection")
        self.tasks.set_work(self._build_work)
        self.tasks.started.connect(lambda: self.hub.set_cache_job(True))
        self.tasks.finished.connect(self._on_run_finished)

        movies_box = QGroupBox("Movies")
        movies_layout = QVBoxLayout(movies_box)
        movies_layout.addLayout(movie_bar)
        movies_layout.addWidget(self.movies)

        splitter = QSplitter(Qt.Vertical)
        top = QWidget()
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.addWidget(movies_box, 2)
        top_layout.addWidget(options_box, 1)
        splitter.addWidget(top)
        splitter.addWidget(self.tasks)
        splitter.setSizes([300, 380])

        layout = QVBoxLayout(self)
        layout.addWidget(_heading(
            "Detect & crop",
            "Runs the detector over every frame, tracks the detections, and cuts "
            "the per-track clips the annotator will see. This is the slow, "
            "GPU-bound step; it is resumable, and finished movies are skipped.",
        ))
        layout.addWidget(splitter, 1)

    def blocker(self) -> str:
        if self.cfg is None or self.hub.status is None:
            return "Open or create a project first (step 1)."
        status = self.hub.status
        if not status.video_dir_ok:
            return (
                "The movie folder does not exist yet. Set it on the Configure page "
                f"(step 2, \"video folder\").\nCurrently: {self.cfg.paths.video_dir}"
            )
        if not status.model_ok:
            return (
                "The detection model file was not found. Set it on the Configure "
                f"page (step 2, \"detection model\").\nCurrently: {self.cfg.paths.model}"
            )
        if status.n_videos == 0:
            return (
                f"No movies in {self.cfg.paths.video_dir}.\nIt holds no files matching "
                f"{cfg_patterns(self.cfg)} - either point at a different folder or "
                "widen \"file patterns\" on the Configure page (step 2)."
            )
        return ""

    def on_project(self) -> None:
        # A scan started against the previous project would land on this one.
        if self._scan is not None and self._scan.isRunning():
            self._scan.cancel()
        blocker = self.blocker()
        self.tasks.run_button.setEnabled(self.tasks.busy or not blocker)
        self.fix_button.setVisible(bool(blocker) and self.cfg is not None)
        if blocker or self.cfg is None:
            self.movies.clear()
            self.scan_state.setText(blocker)
            set_role(self.scan_state, "warning")
            self._scanned_for = None
            return
        set_role(self.scan_state, "muted")
        if self._scanned_for != self.cfg.path:
            self.movies.clear()
            self.scan_state.setText("not scanned yet - press Scan to list the movies")
            self._scanned_for = None
            return
        # Same project: keep the list; rescanning a share is the page's slowest act.
        self._mark_prepared()

    # -- scanning ----------------------------------------------------------- #
    def _on_run_finished(self, ok: bool) -> None:
        self.hub.set_cache_job(False)
        self.hub.reload_project()

    def restyle(self) -> None:
        set_role(self.scan_state, "warning" if self.blocker() else "muted")

    def scan(self) -> None:
        blocker = self.blocker()
        if blocker:
            QMessageBox.information(self, "Cannot scan", blocker)
            return
        if self._scan is not None and self._scan.isRunning():
            return
        cfg = self.cfg
        self.scan_state.setText("scanning... (ND2 files on a share can take a moment)")

        def work(reporter):
            from ..movies import discover_movies

            out = []
            for movie in discover_movies(cfg):
                reporter.raise_if_cancelled()
                try:
                    out.append(movie.describe())
                finally:
                    movie.close()
            return out

        runner = TaskRunner(work, self)
        runner.succeeded.connect(self._on_scanned)
        runner.failed.connect(
            lambda message: self.scan_state.setText(f"scan failed: {message}")
        )
        self._scan = runner
        runner.start()

    def _on_scanned(self, movies) -> None:
        if self.cfg is None:
            return
        self.movies.clear()
        for movie in movies or []:
            item = QListWidgetItem()
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setData(Qt.UserRole, movie["movie_id"])
            item.setData(Qt.UserRole + 1, movie)
            self.movies.addItem(item)
        self._scanned_for = self.cfg.path if self.cfg else None
        self._mark_prepared(select_new=True)

    def _mark_prepared(self, select_new: bool = False) -> None:
        prepared = self._prepared_ids()
        for i in range(self.movies.count()):
            item = self.movies.item(i)
            movie = item.data(Qt.UserRole + 1)
            if not movie:
                continue
            done = movie["movie_id"] in prepared
            item.setText(
                f"{'[prepared]' if done else '[  new   ]'}  {movie['movie_id']}"
                f"      {movie['n_frames']} frames  {movie['width']}x{movie['height']}"
                f"  @ {movie['fps']:g} fps"
            )
            if select_new:
                item.setCheckState(Qt.Unchecked if done else Qt.Checked)
        n_new = self.movies.count() - sum(
            1 for i in range(self.movies.count())
            if (self.movies.item(i).data(Qt.UserRole) or "") in prepared
        )
        if self.movies.count():
            self.scan_state.setText(
                f"{self.movies.count()} movie(s), {n_new} not yet prepared"
            )

    def _prepared_ids(self) -> set[str]:
        if self.cfg is None:
            return set()
        work = self.cfg.paths.work_dir
        # Guarded: a work_dir on a share blocks here once per movie directory.
        if not _exists(work, "dir"):
            return set()
        try:
            entries = sorted(work.iterdir())
        except OSError:
            return set()
        return {
            d.name for d in entries
            if _exists(d, "dir") and _exists(d / "prepare_done.json")
        }

    def _set_all(self, checked: bool) -> None:
        for i in range(self.movies.count()):
            self.movies.item(i).setCheckState(Qt.Checked if checked else Qt.Unchecked)

    def _selected(self) -> list[str] | None:
        picked = [
            self.movies.item(i).data(Qt.UserRole)
            for i in range(self.movies.count())
            if self.movies.item(i).checkState() == Qt.Checked
        ]
        return picked or None

    # -- running ------------------------------------------------------------ #
    def _build_work(self):
        blocker = self.blocker()
        if blocker:
            QMessageBox.information(self, "Cannot run", blocker)
            return None
        if self.movies.count() and not self._selected():
            QMessageBox.information(
                self, "Nothing selected", "Tick at least one movie, or press Scan."
            )
            return None

        # Windows will not overwrite a file this window has memory-mapped.
        self.hub.release_caches()

        from copy import copy

        from ..prepare import prepare_all

        cfg = copy(self.cfg)
        if self.device.text().strip():
            cfg.detection = copy(self.cfg.detection)
            cfg.detection.device = self.device.text().strip()
        movie_ids = self._selected()
        force = self.force.isChecked()
        limit = self.limit_frames.value() or None

        def work(reporter):
            return prepare_all(cfg, movie_ids, force, limit, reporter=reporter)

        return work

    def shutdown(self) -> None:
        if self._scan is not None:
            self._scan.cancel()
            self._scan.wait(2000)
        self.tasks.shutdown()


def cfg_patterns(cfg) -> str:
    return ", ".join(cfg.paths.patterns)


# --------------------------------------------------------------------------- #
# 4. annotate
# --------------------------------------------------------------------------- #
class AnnotatePage(Page):
    def __init__(self, hub, parent=None):
        super().__init__(hub, parent)
        self.panel: AnnotationWidget | None = None
        self._store = None
        self._annotations = None
        self._built_from = None

        self.placeholder = QLabel("")
        self.placeholder.setWordWrap(True)
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setProperty("role", "muted")

        self.stack = QStackedWidget()
        self.stack.addWidget(self.placeholder)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.stack)

    def blocker(self) -> str:
        if self.cfg is None or self.hub.status is None:
            return "no project open"
        if self.hub.cache_job_running:
            return (
                "Detection is running and is rewriting the clips. This page "
                "reopens when it finishes."
            )
        if self.hub.status.n_clips == 0:
            return "no clips yet - run Detect & crop first"
        return ""

    def _fingerprint(self):
        """What the annotation panel is built from; it is rebuilt only when this
        moves, because a rebuild discards the clip on screen."""
        if self.cfg is None:
            return None
        ui = self.cfg.ui
        return (
            str(self.cfg.paths.work_dir), str(self.cfg.paths.csv),
            self.hub.status.n_clips if self.hub.status else 0,
            # Colours and keys included: the buttons are built from them.
            tuple((g.name, g.type, g.required,
                   tuple((o.name, o.key, o.color) for o in g.options))
                  for g in self.cfg.categories),
            tuple(sorted((k, tuple(v)) for k, v in self.cfg.keys.items())),
            tuple(self.cfg.keypoints.nodes), tuple(self.cfg.keypoints.edges),
            # Everything the panel or store reads at write time, ui.annotator too.
            ui.shuffle, ui.shuffle_seed, ui.resume, ui.annotator,
            ui.auto_advance, ui.follow, ui.view_zoom, ui.smooth_window, ui.fps,
            ui.show_overlay, ui.show_context_tracks, ui.show_skeleton,
            ui.show_minimap, ui.prefetch, ui.autoplay, ui.loop,
            self.cfg.clips.frames, self.cfg.detection.task,
            self.cfg.suggest.enabled,
        )

    def on_project(self) -> None:
        # Display-only settings first, BEFORE the fingerprint check: they are not
        # in the fingerprint (no rebuild, no lost clip), so push them in by hand.
        if self.panel is not None and self.cfg is not None:
            self.panel.apply_view_settings(self.cfg)

        fingerprint = self._fingerprint()
        if self.panel is not None and fingerprint == self._built_from:
            return  # nothing this panel depends on changed
        self._teardown()
        self._built_from = None
        blocker = self.blocker()
        if blocker:
            self.placeholder.setText(blocker)
            self.stack.setCurrentWidget(self.placeholder)
            return
        self.placeholder.setText("loading the clip cache...")
        self.stack.setCurrentWidget(self.placeholder)

    def restyle(self) -> None:
        if self.panel is not None:
            self.panel.restyle()

    def refresh_gate(self) -> None:
        # A cache job started while this page is open: let go of the panel now.
        if self.hub.cache_job_running and self.panel is not None:
            self._teardown()
            self._built_from = None
            self.placeholder.setText(self.blocker())
            self.stack.setCurrentWidget(self.placeholder)

    def ensure_loaded(self) -> None:
        """Built on first visit, not on project open: constructing it reads the
        whole manifest and starts the prefetch thread."""
        if self.panel is not None or self.blocker():
            if self.blocker():
                self.placeholder.setText(self.blocker())
                self.stack.setCurrentWidget(self.placeholder)
            return
        try:
            self._store = ClipStore(self.cfg.paths.work_dir)
            self._annotations = AnnotationStore(self.cfg.paths.csv, self.cfg)
            self.panel = AnnotationWidget(self.cfg, self._store, self._annotations)
        except Exception as exc:
            self.placeholder.setText(f"could not open the clip cache:\n{exc}")
            self.stack.setCurrentWidget(self.placeholder)
            return
        self._built_from = self._fingerprint()
        self.panel.status_message.connect(self.status_message)
        # NOT refresh_status: that calls on_project on every page and tears this
        # panel down, once per clip. The panel counts the CURRENT FILTER, so the
        # project-wide total that gates later steps comes off the store.
        self.panel.counts_changed.connect(
            lambda done, total: self.hub.note_progress(len(self._annotations))
        )
        self.stack.addWidget(self.panel)
        self.stack.setCurrentWidget(self.panel)
        self.panel.setFocus()

    def _teardown(self) -> None:
        if self.panel is not None:
            self.panel.shutdown()
            self.stack.removeWidget(self.panel)
            self.panel.deleteLater()
            self.panel = None
        if self._store is not None:
            self._store.close()
        self._store = None
        self._annotations = None

    def shutdown(self) -> None:
        self._teardown()


# --------------------------------------------------------------------------- #
# 5. model
# --------------------------------------------------------------------------- #
class ModelPage(Page):
    def __init__(self, hub, parent=None):
        super().__init__(hub, parent)

        self.report = QTextBrowser()
        self.report.setOpenExternalLinks(False)
        font = QFont("Consolas")
        font.setStyleHint(QFont.Monospace)
        self.report.setFont(font)

        self.tasks = TaskPanel("Train suggester")
        self.tasks.set_work(self._build_work)
        self.tasks.finished.connect(self._on_finished)

        # Re-running detection renumbers tracks, so older clip_ids stop resolving
        # and those labels drop silently out of the fit.
        self._orphans = 0
        self._orphan_cached = 0
        self._orphan_key: tuple | None = None
        self._repair_runner: TaskRunner | None = None
        self.repair_button = QPushButton("Repair label links...")
        self.repair_button.setEnabled(False)
        self.repair_button.setToolTip(
            "Match labels whose clip no longer exists back onto the current "
            "clips, by movie, frame range and crop position."
        )
        self.repair_button.clicked.connect(self._repair_links)
        self.repair_note = _muted(QLabel(""))
        self.repair_note.setWordWrap(True)

        repair_row = QHBoxLayout()
        repair_row.addWidget(self.repair_button)
        repair_row.addWidget(self.repair_note, 1)

        # Labelling everything shares the task panel below; only the gate differs.
        self._job = "train"
        self._predict_blocker = "no project open"
        self._predict_summary = ""
        self.predict_button = QPushButton("Label all clips with the model")
        self.predict_button.setEnabled(False)
        self.predict_button.setToolTip(
            "Score every clip with the trained model and write predictions.parquet. "
            "Your annotation CSV is never written to."
        )
        self.predict_button.clicked.connect(self._label_all)
        self.predict_note = _muted(QLabel(""))
        self.predict_note.setWordWrap(True)

        predict_row = QHBoxLayout()
        predict_row.addWidget(self.predict_button)
        predict_row.addWidget(self.predict_note, 1)

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.tasks)
        report_box = QGroupBox("Last report")
        report_layout = QVBoxLayout(report_box)
        report_layout.addWidget(self.report)
        splitter.addWidget(report_box)
        splitter.setSizes([300, 380])

        layout = QVBoxLayout(self)
        layout.addWidget(_heading(
            "Model",
            "Fits a classifier on the labels so far and cross-validates it leaving "
            "whole movies out. The suggestion it produces is only ever displayed, "
            "never pre-selected, and it stays silent unless agreement beats chance "
            "by a bootstrapped margin.",
        ))
        layout.addLayout(repair_row)
        layout.addLayout(predict_row)
        layout.addWidget(splitter, 1)

    def blocker(self) -> str:
        if self.cfg is None or self.hub.status is None:
            return "no project open"
        if self.hub.status.n_annotated < 2:
            return "nothing labelled yet - annotate some clips first"
        return ""

    def refresh_gate(self) -> None:
        # While busy this button IS Cancel, so a refresh must not grey it out.
        self.tasks.run_button.setEnabled(self.tasks.busy or not self.blocker())
        # The CACHED count: this runs after every annotation, off the I/O path.
        self.repair_button.setEnabled(
            self._orphans > 0 and self._repair_runner is None and not self.tasks.busy
        )
        # Cached too: a reliable model appears only when one is trained.
        self.predict_button.setEnabled(not self._predict_blocker and not self.tasks.busy)

    def on_project(self) -> None:
        self._count_orphans()
        self._refresh_predict_gate()
        self.refresh_gate()
        self._load_report()

    def _refresh_predict_gate(self) -> None:
        """Re-read whether any group cleared the reliability bar. Only from
        :meth:`on_project` and when a job finishes - never from
        :meth:`refresh_gate`, which runs on the annotation keystroke path."""
        if self.cfg is None:
            self._predict_blocker = "no project open"
        else:
            from ..predict import prediction_blocker, predictions_path, staleness

            self._predict_blocker = prediction_blocker(self.cfg)
            if not self._predict_blocker and predictions_path(self.cfg).exists():
                stale = staleness(self.cfg)
                self.predict_note.setText(
                    f"predictions.parquet exists but {stale}" if stale
                    else "predictions.parquet is up to date; the Export page can "
                         "colour movies by it"
                )
                return
        self.predict_note.setText(self._predict_blocker)

    def _count_orphans(self) -> None:
        """How many labels point at a clip_id the cache no longer has. Cheap on
        purpose, because a project reload happens on every autosave; the real
        plan is built on a worker in :meth:`_repair_links`."""
        self._orphans, key = 0, None
        if self.cfg is None or not self.cfg.paths.csv.exists():
            self._orphan_key = None
            self._paint_repair_note()
            return

        import pandas as pd

        csv_path = self.cfg.paths.csv
        manifests = sorted(self.cfg.paths.work_dir.glob(f"*/{MANIFEST_FILE}"))
        try:
            # Stat everything the answer depends on and reuse the last result.
            key = (csv_path.stat().st_mtime_ns, csv_path.stat().st_size,
                   tuple((str(p), p.stat().st_mtime_ns) for p in manifests))
            if key == self._orphan_key:
                self._orphans = self._orphan_cached
                self._paint_repair_note()
                return
            labels = pd.read_csv(csv_path, usecols=["clip_id"])["clip_id"].astype(str)
            known: set[str] = set()
            for path in manifests:
                known.update(pd.read_parquet(path, columns=["clip_id"])["clip_id"].astype(str))
            self._orphans = int((~labels.isin(known)).sum()) if known else 0
        except Exception as exc:
            # A bad CSV or manifest is not worth taking the page down for.
            print(f"[trackanno] could not check the label links: {exc}")
            self._orphans, key = 0, None
        self._orphan_key, self._orphan_cached = key, self._orphans
        self._paint_repair_note()

    def _paint_repair_note(self) -> None:
        self.repair_note.setText(
            f"{self._orphans} label(s) point at clips that no longer exist and are "
            "being ignored by every fit below." if self._orphans else ""
        )

    # -- repairing the label links ------------------------------------------ #
    def _repair_links(self) -> None:
        """Build the re-association plan off the UI thread, then ask."""
        if self.cfg is None or self._repair_runner is not None:
            return
        cfg = self.cfg

        def work(reporter):
            import pandas as pd

            from ..reassociate import plan_reassociation

            reporter.stage("reading the clip manifest", None)
            store = ClipStore(cfg.paths.work_dir)
            try:
                manifest = store.manifest.copy()
            finally:
                # An open memory map on Windows blocks the next detection run.
                store.close()
            reporter.stage("matching labels to clips", None)
            # dtype=str with NA disabled: this frame is written back over the
            # annotation log, so untouched values must round-trip exactly
            # (parsing turns "0.00" into 0.0 and an empty note into "nan").
            labels = pd.read_csv(cfg.paths.csv, dtype=str, keep_default_na=False)
            return labels, plan_reassociation(labels, manifest)

        self.repair_button.setEnabled(False)
        self.repair_button.setText("Checking...")
        runner = TaskRunner(work, self)
        runner.log_line.connect(self.tasks.append)
        runner.succeeded.connect(self._on_plan_ready)
        runner.failed.connect(self._on_plan_failed)
        runner.finished.connect(self._on_repair_finished)
        runner.finished.connect(runner.deleteLater)
        self._repair_runner = runner
        runner.start()

    def _on_repair_finished(self) -> None:
        self._repair_runner = None
        self.repair_button.setText("Repair label links...")
        self.refresh_gate()

    def _on_plan_failed(self, message: str) -> None:
        self.status_message.emit(f"could not check the label links: {message}", 10000)
        QMessageBox.warning(self, "Could not check the label links", message)

    def _on_plan_ready(self, result) -> None:
        labels, plan = result
        if not plan.n_orphaned:
            QMessageBox.information(
                self, "Nothing to repair",
                f"All {plan.already_valid} label(s) already point at clips that exist.",
            )
            self._orphans, self._orphan_key = 0, None
            self._paint_repair_note()
            return

        lines = [
            f"{plan.already_valid} label(s) are already linked correctly.",
            f"{plan.n_orphaned} point at clips that no longer exist:",
            "",
            f"  {len(plan.matched)} can be re-linked by movie, frame range and "
            "crop position",
        ]
        if plan.conflicted:
            lines.append(
                f"  {len(plan.conflicted)} match a clip you have ALREADY labelled since"
            )
        if plan.ambiguous:
            lines.append(f"  {len(plan.ambiguous)} are ambiguous (two clips fit equally well)")
        if plan.unmatched:
            lines.append(f"  {len(plan.unmatched)} have no candidate at all")
        lines += [
            "",
            "Only the first group is changed. The rest are left exactly as they are, "
            "because guessing between two clips would train the model on the wrong "
            "animal - which is worse than losing the label.",
        ]
        if not plan.matched:
            QMessageBox.information(self, "Nothing can be re-linked", "\n".join(lines))
            return

        lines += [
            "",
            f"Rewrite {len(plan.matched)} row(s) of {self.cfg.paths.csv.name}? "
            "The current file is copied to a numbered .bak first.",
        ]
        box = QMessageBox(self)
        box.setWindowTitle("Repair label links")
        box.setIcon(QMessageBox.Question)
        box.setText("\n".join(lines))
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setDefaultButton(QMessageBox.No)
        if box.exec() != QMessageBox.Yes:
            self.status_message.emit("label links left unchanged", 5000)
            return
        self._write_repair(labels, plan)

    def _write_repair(self, labels, plan) -> None:
        from ..reassociate import apply_reassociation, backup_labels, write_labels

        csv_path = self.cfg.paths.csv
        # The panel holds the CSV in memory and rewrites it on the next keystroke.
        self.hub.release_caches()
        try:
            backup = backup_labels(csv_path)
            write_labels(csv_path, apply_reassociation(labels, plan))
        except Exception as exc:
            QMessageBox.critical(
                self, "Could not write the annotations", f"{csv_path}\n\n{exc}"
            )
            self.status_message.emit(f"repair failed: {exc}", 10000)
            return

        self._orphans, self._orphan_key = 0, None
        self.hub.reload_project()
        message = (
            f"{len(plan.matched)} label(s) re-linked to their clips; the previous "
            f"file is at {backup.name}. Retrain to fold them in."
        )
        self.tasks.append(message, "good")
        self.status_message.emit(message, 15000)

    # -- labelling everything with the model --------------------------------- #
    def _label_all(self) -> None:
        """Score every clip with the trained model, off the UI thread. Never
        touches the annotation CSV: predictions land in predictions.parquet, so
        a hand label stays distinguishable from a guess."""
        if self.cfg is None or self.tasks.busy:
            return
        self._refresh_predict_gate()
        if self._predict_blocker:
            QMessageBox.information(self, "Cannot label yet", self._predict_blocker)
            self.refresh_gate()
            return

        cfg = self.cfg
        self._job = "predict"
        self._predict_summary = ""

        def work(reporter):
            from ..predict import predict_all, summarise

            frame = predict_all(cfg, reporter=reporter)
            # Stashed for the status bar; the full tally is already in the log.
            self._predict_summary = summarise(frame).splitlines()[0]
            return frame

        self.tasks.start(work, throttle=1)
        self.refresh_gate()

    def _build_work(self):
        blocker = self.blocker()
        if blocker:
            QMessageBox.information(self, "Cannot train", blocker)
            return None
        cfg = self.cfg
        self._job = "train"

        def work(reporter):
            from ..suggest import train

            return train(cfg, verbose=False, reporter=reporter)

        return work

    def _on_finished(self, ok: bool) -> None:
        if self._job == "predict":
            # Predicting changes no model and no label; only the gates move.
            self._job = "train"
            self._refresh_predict_gate()
            self.hub.reload_project()
            if ok:
                self.status_message.emit(
                    f"predictions written - {self._predict_summary}", 12000
                )
            return
        self._load_report()
        self.hub.reload_project()
        if not ok:
            return
        # reload_project does NOT reach the live annotation panel - the model is
        # not part of its fingerprint - so hand the new model over explicitly.
        annotate = self.hub.page(AnnotatePage)
        panel = getattr(annotate, "panel", None)
        if panel is not None:
            line = panel.reload_suggester()
            self.status_message.emit(f"model reloaded into the annotator - {line}", 8000)

    def restyle(self) -> None:
        self._load_report()      # the report is HTML built from theme colours

    def shutdown(self) -> None:
        self.tasks.shutdown()
        # Wait for it: a QThread still running at teardown aborts the process.
        runner = self._repair_runner
        if runner is not None:
            runner.cancel()
            runner.wait(5000)
            self._repair_runner = None

    def _load_report(self) -> None:
        if self.cfg is None:
            self.report.setPlainText("")
            return
        path = self.cfg.paths.work_dir / "suggest_report.json"
        if not path.exists():
            self.report.setHtml(
                "<p style='color:gray;'>No model has been trained in this project yet.</p>"
            )
            return
        try:
            reports = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self.report.setPlainText(f"could not read {path}: {exc}")
            return
        self.report.setHtml(_format_report(reports))


def _format_report(reports: dict) -> str:
    parts = []
    for group, rep in reports.items():
        live = rep.get("reliable")
        badge = (
            f"<span style='color:{tokens()['success']};'>suggesting</span>" if live
            else f"<span style='color:{tokens()['warning']};'>staying silent</span>"
        )
        parts.append(f"<h3>{group} &mdash; {badge}</h3>")
        parts.append(
            f"<p>{rep.get('n_labels', 0)} labels over {rep.get('n_classes', 0)} classes"
            f"<br>{rep.get('scope', '')}"
            f"<br>kappa {rep.get('kappa', 0):+.3f} "
            f"(5th percentile {rep.get('kappa_lower', 0):+.3f})"
            f"&nbsp;&nbsp;macro-F1 {rep.get('macro_f1', 0):.3f}"
            f"&nbsp;&nbsp;accuracy {rep.get('accuracy', 0):.3f} "
            f"vs {rep.get('baseline_accuracy', 0):.3f} majority</p>"
        )
        if not live:
            parts.append(f"<p><i>{rep.get('reason', '')}</i></p>")
        if rep.get("n_orphaned"):
            # Loud and in colour: this is data loss, not a modelling result.
            by_class = ", ".join(
                f"{k} ({v})" for k, v in rep.get("orphaned_by_class", {}).items()
            )
            parts.append(
                f"<p style='color:{tokens()['warning']};'><b>{rep['n_orphaned']} label(s) "
                "were ignored</b>: their clips are not in the feature table"
                + (f" - {by_class}" if by_class else "")
                + ". Use \"Repair label links\" above, then retrain.</p>"
            )
        if rep.get("dropped_classes"):
            dropped = ", ".join(f"{k} ({v})" for k, v in rep["dropped_classes"].items())
            parts.append(f"<p>not modelled, too few labels: {dropped}</p>")
        if rep.get("audit_n"):
            audit_kappa = rep.get("audit_kappa")
            audit_kappa = "n/a" if audit_kappa is None or audit_kappa != audit_kappa \
                else f"{audit_kappa:+.3f}"
            parts.append(
                f"<p>audit holdout ({rep['audit_n']} clips, never suggested nor "
                f"trained on): kappa {audit_kappa}</p>"
            )
        if rep.get("confusion") and rep.get("classes"):
            head = "".join(f"<th>{c}</th>" for c in rep["classes"])
            rows = "".join(
                f"<tr><th align='left'>{name}</th>"
                + "".join(f"<td align='right'>&nbsp;{v}&nbsp;</td>" for v in row)
                + "</tr>"
                for name, row in zip(rep["classes"], rep["confusion"])
            )
            parts.append(
                f"<p>confusion (rows = your label):</p>"
                f"<table border='1' cellspacing='0' cellpadding='2'>"
                f"<tr><th></th>{head}</tr>{rows}</table>"
            )
        parts.append("<hr>")
    return "".join(parts)


# --------------------------------------------------------------------------- #
# 6. export
# --------------------------------------------------------------------------- #
class DistributionPage(Page):
    """What the labels look like, before anything is trained or exported."""

    def __init__(self, hub, parent=None):
        super().__init__(hub, parent)
        self._stale = True
        self._charts: dict[str, BarChart] = {}

        self.coverage = QLabel("")
        self.coverage.setWordWrap(True)
        self.coverage.setTextFormat(Qt.RichText)

        self.notes = QLabel("")
        self.notes.setWordWrap(True)
        self.notes.setTextFormat(Qt.RichText)
        self.notes.setVisible(False)

        refresh = QPushButton("Refresh")
        refresh.setToolTip("Re-read the annotation CSV")
        refresh.clicked.connect(self.reload)

        head = QHBoxLayout()
        head.addWidget(self.coverage, 1)
        head.addWidget(refresh, 0, Qt.AlignTop)

        self.charts_box = QWidget()
        self.charts_layout = QVBoxLayout(self.charts_box)
        self.charts_layout.setContentsMargins(0, 0, 0, 0)
        self.charts_layout.setSpacing(12)
        self.charts_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(self.charts_box)

        layout = QVBoxLayout(self)
        layout.addWidget(_heading(
            "Distribution",
            "How your labels are spread, right now. Worth a look before training "
            "or exporting: a class with almost nothing in it, or one that "
            "swallows everything, decides what the next step is actually worth.",
        ))
        layout.addLayout(head)
        layout.addWidget(self.notes)
        layout.addWidget(scroll, 1)

    def blocker(self) -> str:
        if self.cfg is None or self.hub.status is None:
            return "no project open"
        if self.hub.status.n_annotated == 0:
            return "no annotations yet - label some clips first"
        return ""

    def refresh_gate(self) -> None:
        # Mark it stale and rebuild when shown, not once per annotation.
        self._stale = True

    def on_project(self) -> None:
        self._stale = True
        # Groups may have been renamed or removed; old charts cannot be reused.
        self._clear_charts()

    def restyle(self) -> None:
        for chart in self._charts.values():
            chart.update()
        self.reload()

    def _clear_charts(self) -> None:
        for box in list(self._charts.values()):
            holder = box.parentWidget()
            while holder is not None and not isinstance(holder, QGroupBox):
                holder = holder.parentWidget()
            (holder or box).setParent(None)
            (holder or box).deleteLater()
        self._charts.clear()

    def reload(self) -> None:
        """Re-read the CSV and repaint. Called when the page is shown."""
        self._stale = False
        blocker = self.blocker()
        if blocker:
            self._clear_charts()
            self.coverage.setText(f"<span style='color:{tokens()['muted']}'>{blocker}</span>")
            self.notes.setVisible(False)
            return

        rows, problem = _read_annotation_rows(self.cfg)
        if problem:
            self._clear_charts()
            self.coverage.setText(
                f"<span style='color:{tokens()['danger']}'>{problem}</span>"
            )
            self.notes.setVisible(False)
            return

        tallies = tally_annotations(rows, self.cfg)
        self._paint_summary(rows, tallies)

        minimum = self.cfg.suggest.min_per_class
        for tally in tallies:
            chart = self._charts.get(tally.name)
            if chart is None:
                chart = BarChart(tally, minimum)
                self._charts[tally.name] = chart
                title = tally.name + ("   (multi-select)" if tally.is_multi else "")
                box = QGroupBox(title)
                inner = QVBoxLayout(box)
                # Zero: the QGroupBox rule already reserves 34px for the title.
                inner.setContentsMargins(0, 0, 0, 0)
                inner.addWidget(chart)
                self.charts_layout.insertWidget(self.charts_layout.count() - 1, box)
            else:
                chart.set_tally(tally, minimum)

        lines = summary_lines(tallies, self.cfg)
        self.notes.setVisible(bool(lines))
        if lines:
            items = "".join(f"<li>{line}</li>" for line in lines)
            self.notes.setText(
                f"<div style='color:{tokens()['warning']}'>"
                f"<ul style='margin-left:-20px;'>{items}</ul></div>"
            )

    def _paint_summary(self, rows, tallies) -> None:
        status = self.hub.status
        n = len(rows)
        total = status.n_clips if status else 0
        share = f" ({100.0 * n / total:.0f}% of {total} clips)" if total else ""
        movies = {str(r.get("movie_id", "") or "") for r in rows} - {""}
        people = {str(r.get("annotator", "") or "") for r in rows} - {""}
        uncertain = sum(1 for r in rows if str(r.get("uncertain", "")).strip() == "1")

        parts = [f"<b>{n}</b> clip(s) labelled{share}",
                 f"across <b>{len(movies)}</b> movie(s)"]
        if len(people) > 1:
            parts.append(f"by <b>{len(people)}</b> annotators")
        elif people:
            parts.append(f"by {sorted(people)[0]}")
        if uncertain:
            parts.append(f"<b>{uncertain}</b> marked uncertain")
        self.coverage.setText(
            "   |   ".join(parts)
            + f"<br><span style='color:{tokens()['faint']}'>The dashed line marks "
            f"suggest.min_per_class ({self.cfg.suggest.min_per_class}); classes "
            "left of it are dropped when the model is trained.</span>"
        )


def _read_annotation_rows(cfg):
    """(rows, problem). A broken CSV must report itself, not raise into Qt."""
    import csv

    path = cfg.paths.csv
    if not path.exists():
        return [], f"no annotation file yet at {path}"
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh)), ""
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        return [], f"could not read {path.name}: {exc}"


class ExportPage(Page):
    def __init__(self, hub, parent=None):
        super().__init__(hub, parent)

        self.movies = QListWidget()
        self.movies.setAlternatingRowColors(True)
        self._list_stale = True

        self.group = QComboBox()
        # The combo carries the machine-readable value behind each plain label.
        self.source = QComboBox()
        for label, value in (("my labels", "labels"),
                             ("model predictions", "predictions"),
                             ("my labels and the model's", "both")):
            self.source.addItem(label, value)
        self.source.currentIndexChanged.connect(self._on_source_changed)
        self.source_note = _muted(QLabel(""))
        self.source_note.setWordWrap(True)

        self.hide_unlabelled = QCheckBox("draw only annotated tracks")
        self.max_frames = QSpinBox()
        self.max_frames.setRange(0, 1_000_000)
        self.max_frames.setSpecialValueText("whole movie")
        self.as_tif = QCheckBox("write a TIFF stack instead of mp4")

        options = QFormLayout()
        options.addRow("category group", self.group)
        options.addRow("colour by", self.source)
        options.addRow("", self.source_note)
        options.addRow(self.hide_unlabelled)
        options.addRow("frame limit", self.max_frames)
        options.addRow(self.as_tif)
        options_box = QGroupBox("Options")
        options_box.setLayout(options)

        self.movies_box = QGroupBox("Annotated movies")
        movies_box = self.movies_box
        movies_layout = QVBoxLayout(movies_box)
        movies_layout.addWidget(self.movies)

        self.tasks = TaskPanel("Render overlay movies")
        self.tasks.set_work(self._build_work)

        top = QWidget()
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.addWidget(movies_box, 2)
        top_layout.addWidget(options_box, 1)

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(top)
        splitter.addWidget(self.tasks)
        splitter.setSizes([260, 380])

        layout = QVBoxLayout(self)
        layout.addWidget(_heading(
            "Export",
            "Burns the labels back onto the original movie, colouring each track "
            "by its category. Works at any time, including on a half-finished CSV. "
            "Colour by your own labels, by the model's predictions, or by both - a "
            "predicted box is drawn dashed with its confidence, so it can never be "
            "mistaken for something you decided.",
        ))
        layout.addWidget(splitter, 1)

    def source_value(self) -> str:
        """"labels", "predictions" or "both"."""
        return self.source.currentData() or "labels"

    def blocker(self) -> str:
        if self.cfg is None or self.hub.status is None:
            return "no project open"
        if self.source_value() == "predictions":
            # A prediction-only export needs no hand labels, but it needs the file.
            from ..predict import predictions_path

            if not predictions_path(self.cfg).exists():
                return ("no predictions yet - run 'Label all clips with the model' "
                        "on the Model page")
            return ""
        if self.hub.status.n_annotated == 0:
            return "no annotations yet"
        return ""

    def refresh_gate(self) -> None:
        # Busy means this button is Cancel; leave it clickable.
        self.tasks.run_button.setEnabled(self.tasks.busy or not self.blocker())
        # The list comes from the CSV; rebuild it when the page is shown.
        self._list_stale = True

    def on_project(self) -> None:
        self._list_stale = True
        self.group.clear()
        if self.cfg is None:
            self.movies.clear()
            self._paint_source_note()
            return
        for group in self.cfg.categories:
            if not group.is_multi:
                self.group.addItem(group.name)
        self._paint_source_note()
        self.refresh_gate()
        self.reload_movies()

    def _on_source_changed(self) -> None:
        """The source decides both what the button may do and which movies exist."""
        self._paint_source_note()
        self.refresh_gate()
        self.reload_movies()

    def _paint_source_note(self) -> None:
        source = self.source_value()
        if source == "labels":
            self.source_note.setText("")
            return
        note = (
            "predicted values are drawn as dashed boxes with their confidence, "
            "so they cannot be mistaken for your labels"
        )
        if self.cfg is not None:
            from ..predict import predictions_path, staleness

            if predictions_path(self.cfg).exists():
                stale = staleness(self.cfg)
                if stale:
                    note += f".  Warning: {stale}"
            else:
                note = "there are no predictions in this project yet"
        self.source_note.setText(note)

    def reload_movies(self) -> None:
        """Re-read the CSV. Called when the page is shown, not per annotation."""
        if self.cfg is None:
            return
        self._list_stale = False
        self.movies.clear()
        blocker = self.blocker()
        if blocker:
            self.movies.addItem(blocker)
            return

        source = self.source_value()
        self.movies_box.setTitle(
            "Predicted movies" if source == "predictions" else "Annotated movies"
        )
        if source == "predictions":
            counts, unit = self._predicted_movies(), "predicted clips"
        else:
            from ..render import annotated_movies

            counts, unit = annotated_movies(self.cfg), "annotated clips"

        for movie_id, count in counts.items():
            item = QListWidgetItem(f"{movie_id}      {count} {unit}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            item.setData(Qt.UserRole, movie_id)
            self.movies.addItem(item)

    def _predicted_movies(self) -> dict:
        """{movie_id: predicted clip count}. Never raises, like annotated_movies."""
        from ..predict import load_predictions

        try:
            frame = load_predictions(self.cfg)
        except (FileNotFoundError, OSError, RuntimeError, ValueError):
            return {}
        counts = frame["movie_id"].astype(str).value_counts()
        return {str(k): int(v) for k, v in sorted(counts.items())}

    def _build_work(self):
        blocker = self.blocker()
        if blocker:
            QMessageBox.information(self, "Cannot render", blocker)
            return None
        source = self.source_value()
        group_name = self.group.currentText() or None
        if source != "labels":
            # Refuse before starting: an empty export looks like "nothing to say".
            from ..predict import load_predictions

            try:
                load_predictions(self.cfg, group_name)
            except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
                QMessageBox.warning(self, "No usable predictions", str(exc))
                return None

        picked = [
            self.movies.item(i).data(Qt.UserRole)
            for i in range(self.movies.count())
            if self.movies.item(i).checkState() == Qt.Checked
            and self.movies.item(i).data(Qt.UserRole)
        ]
        if not picked:
            QMessageBox.information(self, "Nothing selected", "Tick at least one movie.")
            return None

        cfg = self.cfg
        hide = self.hide_unlabelled.isChecked()
        limit = self.max_frames.value() or None
        as_tif = self.as_tif.isChecked()

        def work(reporter):
            from ..render import render_all

            return render_all(
                cfg, picked, group_name, hide_unlabelled=hide, max_frames=limit,
                write_tif=as_tif, reporter=reporter, source=source,
            )

        return work

    def shutdown(self) -> None:
        self.tasks.shutdown()


# --------------------------------------------------------------------------- #
# window
# --------------------------------------------------------------------------- #
class Hub(QMainWindow):
    def __init__(self, config_path: str | Path | None = None):
        super().__init__()
        self.setWindowTitle("trackanno")
        self.setWindowIcon(app_icon())
        self.resize(1280, 860)
        self.cfg = None
        self.status = None
        # Set while a job is rewriting the clip cache: release_caches() runs once,
        # before the worker, so this is what stops Annotate re-mapping those files.
        self.cache_job_running = False

        self.sidebar = QListWidget()
        self.sidebar.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.sidebar.setFocusPolicy(Qt.NoFocus)
        for name in PAGES:
            self.sidebar.addItem(QListWidgetItem(name))
        self.sidebar.setStyleSheet(
            f"QListWidget{{background:{tokens()['surface']};padding:8px;"
            f"border:1px solid {tokens()['border']};border-radius:8px;}}"
            "QListWidget::item{padding:10px 8px;margin:1px 0;}"
        )

        self.pages: list[Page] = [
            ProjectPage(self), ConfigPage(self), PreparePage(self),
            AnnotatePage(self), ModelPage(self), DistributionPage(self),
            ExportPage(self),
        ]
        self.stack = QStackedWidget()
        for page in self.pages:
            page.status_message.connect(
                lambda text, timeout=0: self.statusBar().showMessage(text, timeout)
            )
            self.stack.addWidget(page)

        # A permanent "what now" line, on every page.
        self.hint = QLabel("")
        self.hint.setWordWrap(True)
        self.hint.setTextFormat(Qt.RichText)
        self.hint_button = QPushButton("")
        self.hint_button.setProperty("accent", "true")
        self.hint_button.setVisible(False)
        self.hint_button.clicked.connect(lambda: self.go_to(self._hint_page))
        self._hint_page = 0

        hint_bar = self._hint_bar = QWidget()
        hint_bar.setObjectName("hintBar")
        hint_bar.setStyleSheet(
            f"#hintBar{{background:{tokens()['surface']};"
            f"border:1px solid {tokens()['border']};border-radius:8px;}}"
        )
        hint_layout = QHBoxLayout(hint_bar)
        hint_layout.setContentsMargins(10, 6, 10, 6)
        hint_layout.addWidget(self.hint, 1)
        hint_layout.addWidget(self.hint_button)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.stack, 1)
        right_layout.addWidget(hint_bar)

        # At the foot of the nav rail, where an appearance control belongs.
        self.theme_button = compact_button("", 30)
        self.theme_button.setIconSize(QSize(15, 15))
        self.theme_button.clicked.connect(self.toggle_theme)
        self._sync_theme_button()

        self.nav = QWidget()
        self.nav.setFixedWidth(200)
        nav_layout = QVBoxLayout(self.nav)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(8)
        nav_layout.addWidget(self.sidebar, 1)
        nav_layout.addWidget(self.theme_button, 0, Qt.AlignLeft)

        body = QWidget()
        layout = QHBoxLayout(body)
        layout.setContentsMargins(10, 10, 10, 6)
        layout.setSpacing(10)
        layout.addWidget(self.nav)
        layout.addWidget(right, 1)
        self.setCentralWidget(body)

        self.sidebar.currentRowChanged.connect(self._on_page_changed)
        self.sidebar.setCurrentRow(0)
        self.statusBar().showMessage("no project open")

        opened = config_path or (recent_projects()[0] if recent_projects() else None)
        if not (opened and self.open_project(opened, quiet=True)):
            # Nothing opened: the sidebar, pages and hint bar need a first paint.
            self.refresh_status(quiet=True)

    # -- project ------------------------------------------------------------ #
    def open_project(self, target: str | Path, quiet: bool = False,
                     skip: "Page | None" = None) -> bool:
        path = config_path_for(target)
        if not path.exists():
            if not quiet:
                QMessageBox.warning(self, "Not found", f"No {CONFIG_NAME} at {path}")
            return False
        try:
            self.cfg = load_config(path)
        except Exception as exc:  # a broken config must not take the window down
            self.cfg = None
            # Tell every page the project went away: returning early leaves a
            # live annotation panel writing into the previous project's CSV.
            self.status = None
            self.setWindowTitle("trackanno")
            self.refresh_status(quiet=True)
            if not quiet:
                QMessageBox.warning(self, "Could not open", f"{path}\n\n{exc}")
            self.statusBar().showMessage(f"could not open {path}: {exc}")
            return False
        remember_project(path)
        self.setWindowTitle(f"trackanno - {path.parent.name}")
        self.refresh_status(skip=skip)
        return True

    def page(self, kind):
        """The page of a given class - never by index, which shifts when a step
        is inserted and silently reaches for the wrong page."""
        return next((p for p in self.pages if isinstance(p, kind)), None)

    def reload_project(self, skip: "Page | None" = None) -> None:
        if self.cfg is not None:
            self.open_project(self.cfg.path, quiet=True, skip=skip)

    # -- appearance --------------------------------------------------------- #
    def _sync_theme_button(self) -> None:
        self.theme_button.setIcon(theme_icon(15))
        self.theme_button.setToolTip(
            f"Switch to the {'light' if current() == 'dark' else 'dark'} theme"
        )

    def toggle_theme(self) -> None:
        """Swap light/dark now, and remember it in the project."""
        name = "light" if current() == "dark" else "dark"
        apply_theme(QApplication.instance(), name)
        self._sync_theme_button()
        self.restyle()
        if self.cfg is not None:
            try:
                write_config(self.cfg.path, {"ui.theme": name})
                self.cfg.ui.theme = name
                # The Configure form writes every field back on its next save.
                config_page = self.pages[1]
                widget = config_page.form._widgets.get("ui.theme")
                if widget is not None:
                    widget.blockSignals(True)
                    widget.setCurrentText(name)
                    widget.blockSignals(False)
            except Exception as exc:      # a theme is never worth failing over
                self.statusBar().showMessage(f"theme changed but not saved: {exc}", 6000)
                return
        self.statusBar().showMessage(f"{name} theme", 3000)

    def restyle(self) -> None:
        """Re-apply the styles that are set in code rather than by the sheet."""
        self.sidebar.setStyleSheet(
            f"QListWidget{{background:{tokens()['surface']};padding:8px;"
            f"border:1px solid {tokens()['border']};border-radius:8px;}}"
            "QListWidget::item{padding:10px 8px;margin:1px 0;}"
        )
        self._hint_bar.setStyleSheet(
            f"#hintBar{{background:{tokens()['surface']};"
            f"border:1px solid {tokens()['border']};border-radius:8px;}}"
        )
        for page in self.pages:
            page.restyle()
        self._paint_hint()

    def release_caches(self) -> None:
        """Let go of every clip-cache memory map this window holds. Anything that
        rewrites the cache has to call this first on Windows."""
        for page in self.pages:
            if isinstance(page, AnnotatePage):
                page._teardown()
                page._built_from = None

    def refresh_status(self, quiet: bool = False, skip: "Page | None" = None) -> None:
        if self.cfg is not None:
            self.status = project_status(self.cfg)
        for page in self.pages:
            if page is skip:
                continue
            # One page's I/O must not take the window down.
            try:
                page.on_project()
            except Exception as exc:
                print(f"[trackanno] {type(page).__name__} failed to refresh: {exc}")
                self.statusBar().showMessage(
                    f"{type(page).__name__}: {exc}", 10000
                )
        self._paint_sidebar()
        self._paint_hint()
        if self.cfg is not None and not quiet:
            self.statusBar().showMessage(
                f"{self.cfg.path}   |   {self.status.n_clips} clips, "
                f"{self.status.n_annotated} annotated"
            )
        # on_project tore the annotation page down; rebuild it if it is on screen.
        current = self.pages[self.sidebar.currentRow()] if self.sidebar.currentRow() >= 0 else None
        if isinstance(current, AnnotatePage):
            current.ensure_loaded()

    def _paint_hint(self) -> None:
        advice, page, label = self.next_step()
        self.hint.setText(f"<b>Next:</b> {advice}")
        self._hint_page = page
        self.hint_button.setText(label)
        self.hint_button.setVisible(bool(label) and self.sidebar.currentRow() != page)

    def set_cache_job(self, running: bool) -> None:
        """Flag that something is rewriting the clip cache."""
        self.cache_job_running = bool(running)
        if running:
            self.release_caches()
        for page in self.pages:
            page.refresh_gate()
        self._paint_sidebar()
        if not running:
            # Bring the page back by itself, without waiting for a navigation.
            row = self.sidebar.currentRow()
            current = self.pages[row] if 0 <= row < len(self.pages) else None
            if isinstance(current, AnnotatePage):
                current.ensure_loaded()

    def note_progress(self, n_annotated: int) -> None:
        """Cheap counter update from the annotation panel, once per clip. Not
        ``refresh_status``, which rebuilds every page - but the gates must still
        be re-evaluated: Model and Export unlock on the first label."""
        if self.status is not None:
            self.status.n_annotated = n_annotated
        for page in self.pages:
            page.refresh_gate()
        self._paint_sidebar()
        self._paint_hint()

    def next_step(self) -> tuple[str, int, str]:
        """(advice, page index, button label) - what this project needs next."""
        if self.cfg is None:
            return ("Open or create a project to begin.", 0, "")
        status = self.status
        # Missing paths only block the NEXT detection run.
        if status.n_clips == 0 and (not status.video_dir_ok or not status.model_ok):
            missing = []
            if not status.video_dir_ok:
                missing.append(f"video folder <code>{self.cfg.paths.video_dir}</code>")
            if not status.model_ok:
                missing.append(f"model <code>{self.cfg.paths.model}</code>")
            return (
                "Point the project at its data: " + " and ".join(missing) + " not found.",
                1, "Go to Configure",
            )
        if status.n_clips == 0:
            return (
                f"{status.n_videos} movie(s) found and nothing prepared yet. "
                "Run detection to build the clip cache.",
                2, "Go to Detect & crop",
            )
        if status.n_annotated < status.n_clips:
            note = (
                "" if status.video_dir_ok and status.model_ok
                else "<br><i>(the video folder or model is missing, so no further "
                "movies can be prepared until that is fixed)</i>"
            )
            return (
                f"{status.n_annotated} of {status.n_clips} clips labelled. "
                "Keep going - one keypress per clip." + note,
                3, "Go to Annotate",
            )
        return (
            "Every clip is labelled. Check how they came out, train the "
            "suggester on them, or export the overlay movies.",
            PAGES.index("Distribution"), "Go to Distribution",
        )

    # -- navigation --------------------------------------------------------- #
    def go_to(self, index: int) -> None:
        self.sidebar.setCurrentRow(index)

    def _on_page_changed(self, index: int) -> None:
        if index < 0:
            return
        previous = self.stack.currentWidget()
        if isinstance(previous, ConfigPage) and not previous.maybe_save():
            self.sidebar.blockSignals(True)
            self.sidebar.setCurrentRow(self.pages.index(previous))
            self.sidebar.blockSignals(False)
            return
        page = self.pages[index]
        self.stack.setCurrentWidget(page)
        page.refresh_gate()
        if isinstance(page, AnnotatePage):
            page.ensure_loaded()
            if page.panel is not None:
                page.panel.setFocus()
        elif isinstance(page, ExportPage) and page._list_stale:
            page.reload_movies()
        elif isinstance(page, DistributionPage) and page._stale:
            page.reload()
        elif isinstance(page, ModelPage):
            page._load_report()
        self._paint_hint()
        blocker = page.blocker()
        if blocker:
            self.statusBar().showMessage(blocker)

    def _step_states(self) -> list[tuple[str, str]]:
        """(glyph, note) per step - a checklist, not six equal buttons."""
        if self.cfg is None:
            return [(">", "start here")] + [("", "needs a project")] * (len(PAGES) - 1)
        s = self.status
        paths_ok = s.video_dir_ok and s.model_ok
        return [
            ("*", ""),
            ("*" if paths_ok else "!", "" if paths_ok else "something is missing"),
            ("*" if s.n_clips else ">", f"{s.n_prepared} done" if s.n_clips else "not run yet"),
            ("*" if s.n_clips and s.n_annotated >= s.n_clips else
             (">" if s.n_clips else ""),
             f"{s.n_annotated} / {s.n_clips}" if s.n_clips else ""),
            ("*" if s.suggester else "", s.suggester.split(":")[0] if s.suggester else ""),
            ("*" if s.n_annotated else "", f"{s.n_annotated} labelled" if s.n_annotated else ""),
            ("*" if s.n_annotated else "", ""),
        ]

    def _paint_sidebar(self) -> None:
        states = self._step_states()
        for i, name in enumerate(PAGES):
            item = self.sidebar.item(i)
            glyph, note = states[i]
            item.setText(f" {i + 1}. {name}" + (f"   {glyph}" if glyph else ""))
            item.setToolTip(
                f"{PAGE_HINTS[i]}" + (f"\n{note}" if note else "")
            )
            blocked = self.cfg is None and i != 0
            item.setFlags(
                (item.flags() & ~Qt.ItemIsEnabled) if blocked
                else (item.flags() | Qt.ItemIsEnabled)
            )

    # -- close -------------------------------------------------------------- #
    def closeEvent(self, event):  # noqa: N802 (Qt naming)
        for page in self.pages:
            if isinstance(page, ConfigPage) and not page.maybe_save():
                event.ignore()
                return
        for page in self.pages:
            page.shutdown()
        super().closeEvent(event)


def _preferred_theme(project) -> str:
    """Read ui.theme from the project about to be opened, if there is one."""
    candidates = [project] if project else []
    candidates += recent_projects()[:1]
    for candidate in candidates:
        try:
            return load_config(config_path_for(candidate)).ui.theme
        except Exception:
            continue
    return "dark"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="trackanno", description="The whole annotation pipeline in one window."
    )
    parser.add_argument("project", nargs="?", default=None,
                        help="a project folder or its config.yaml (default: the last one)")
    args = parser.parse_args(argv)

    app = QApplication.instance() or QApplication(sys.argv[:1])
    # Before any widget exists: a stylesheet applied later has to re-polish the
    # whole tree, and some widgets keep the old metrics.
    apply_theme(app, _preferred_theme(args.project))
    window = Hub(args.project)
    window.show()
    return app.exec_() if hasattr(app, "exec_") else app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
