"""Step 1: pick or create a project."""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QComboBox, QDialog, QFileDialog, QFormLayout, QFrame,
    QGridLayout, QGroupBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget,
    QListWidgetItem,
    QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

from ...project import (
    CONFIG_NAME, _exists, create_project, recent_projects,
)
from .base import Page, _heading
from .configure import ModelRegistryDialog, _install_shared_model


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
            "Pick the folder to work in.",
            help="""
<p>A <b>project</b> is a folder holding <b>config.yaml</b>, <b>work/</b> (the
clip cache) and <b>annotations.csv</b>. This window operates on the open one.</p>

<p><i>New project</i> writes a fully commented config.yaml with the paths
filled in.</p>

<p><b>Presets.</b> <b>start from</b> in <i>New project</i> offers three shipped
settings, a ladder: <b>annotator</b> labels clips by hand from a box detector,
<b>suggester</b> moves to pose weights and trains a classifier on those labels,
<b>dense bouts</b> scores every frame and cuts it into bouts to review. Each step
up is a handful of settings the Configure page can also set by hand.</p>

<p><b>Templates.</b> <i>Save as template</i> stores the open project's settings
under a name, and <i>New project</i> offers them under <b>start from</b> beside
the presets.</p>

<p>A template is a whole config.yaml, comments included, kept outside any
project. It carries every setting except video_dir, work_dir and csv, which
describe one project on one disk and are reset.</p>
""",
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
            entry = dialog.chosen_model()
            if entry is None:
                self.status_message.emit(
                    "project created - check the settings, then run detection", 8000
                )
                return
            # The download is the slow half and the project is already on disk,
            # so it runs in the background against the config just written. Its
            # own reporting takes over from here, including what the manifest
            # decided about this project.
            _install_shared_model(self, entry, path)

    def _save_template(self) -> None:
        """Store the open project's settings for the next project to start from."""
        from ...templates import find_template, save_template

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
            f"{status.videos_text} in {self.cfg.paths.video_dir}"
            + ("" if status.video_dir_ok else "   (folder not found)")
        )
        if status.work_known:
            self._summary_values["prepared"].setText(
                f"{status.n_prepared} movie(s)"
                + (f", {status.n_stale} with changed settings" if status.n_stale else "")
            )
            self._summary_values["clips"].setText(f"{status.n_clips}")
        else:
            for field in ("prepared", "clips"):
                self._summary_values[field].setText("? (work folder slow, still counting)")
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

        # Most people do not have a .pt yet, which is the whole reason the
        # registry exists; offering it only after the project is made sends them
        # to the Configure page for the one setting they came here without.
        self._entry = None
        self.registry_btn = QPushButton("Choose shared model...")
        self.registry_btn.setProperty("accent", "true")
        self.registry_btn.setToolTip(
            "Download a model from the shared registry. It is fetched once the "
            "project is created, cached for later, and fills in the settings it "
            "was trained with. Browse instead for a .pt already on this machine."
        )
        self.registry_btn.clicked.connect(self._pick_shared_model)
        self.model.layout().insertWidget(1, self.registry_btn)
        self.model.picked.connect(self._drop_shared_model)
        self.model_note = QLabel()
        self.model_note.setWordWrap(True)
        self.model_note.setProperty("role", "muted")
        self.model_note.setVisible(False)

        self.template = QComboBox()
        self.forget_btn = QPushButton("Forget")
        self.forget_btn.setToolTip("Delete the selected template. Presets stay.")
        self.forget_btn.clicked.connect(self._forget_template)
        self.template_note = QLabel()
        self.template_note.setWordWrap(True)
        self.template_note.setProperty("role", "muted")
        self._fill_templates()
        template_row = QHBoxLayout()
        template_row.addWidget(self.template, 1)
        template_row.addWidget(self.forget_btn)

        form = QFormLayout()
        form.addRow("project name", self.name)
        form.addRow("create in", self.location)
        form.addRow("start from", template_row)
        form.addRow(self.template_note)
        form.addRow("video folder", self.videos)
        form.addRow("detection model", self.model)
        form.addRow(self.model_note)

        note = QLabel(
            "Everything else comes from the template, and can be changed later "
            "on the Configure page."
        )
        note.setWordWrap(True)
        note.setProperty("role", "muted")
        note.setToolTip(
            "The clip cache and annotations.csv are created inside the project "
            "folder.\n\n"
            "The presets are a ladder: annotator labels by hand from a box "
            "detector, suggester wants pose weights and trains a classifier, "
            "dense bouts scores every frame. Start low, and climb from Configure.\n\n"
            "'Save as template' on the Project page makes one from a project "
            "you have already tuned.\n\n"
            "None of it is final: every setting is on the Configure page."
        )

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
        if self._entry is not None:
            pass          # a shared model is chosen, and fetched after creation
        elif not self.model.value():
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
        from ...templates import list_templates

        current = self.template.currentData()
        self.template.clear()
        for template in list_templates():
            self.template.addItem(template.label, template.name)
            self.template.setItemData(
                self.template.count() - 1,
                template.about or template.summary,
                Qt.ToolTipRole,
            )
        index = self.template.findData(current)
        self.template.setCurrentIndex(max(0, index))
        self._sync_selection()
        self.template.currentIndexChanged.connect(self._sync_selection)

    def _sync_selection(self) -> None:
        """A preset says what it is under the picker, and cannot be forgotten."""
        from ...templates import find_template

        template = find_template(self.template.currentData())
        self.forget_btn.setEnabled(template is not None and not template.builtin)
        self.template_note.setText(template.summary if template else "")
        self.template_note.setToolTip(template.about if template else "")
        self.template_note.setVisible(bool(self.template_note.text()))

    def _forget_template(self) -> None:
        from ...templates import delete_template

        name = self.template.currentData()
        answer = QMessageBox.question(
            self, "Forget template", f"Delete the template {name!r}?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if answer == QMessageBox.Yes and delete_template(name):
            self._fill_templates()

    def _pick_shared_model(self) -> None:
        """Choose registry weights now; they are fetched once the project exists.

        Nothing is downloaded here. This dialog can still be cancelled, and a
        cancelled dialog that had already put 200 MB in the cache is a surprise.
        """
        dialog = ModelRegistryDialog(self)
        if not dialog.run():
            return
        entry = dialog.chosen()
        if entry is None:
            return
        self._entry = entry
        self.model.edit.setText("")
        self.model.edit.setEnabled(False)
        self.model_note.setText(
            f"will download <b>{entry.label}</b> when the project is created, and "
            "take the settings it was trained with. The path above is not used."
        )
        self.model_note.setVisible(True)

    def _drop_shared_model(self, _path: str = "") -> None:
        """Browsing to a .pt means run THAT file, shared model or not."""
        self._entry = None
        self.model.edit.setEnabled(True)
        self.model_note.clear()
        self.model_note.setVisible(False)

    def chosen_model(self):
        """The registry entry to fetch after the project is written, or None.

        Kept out of :meth:`values`, which goes straight to ``create_project``: a
        project is created from paths, and weights that are not on this disk yet
        are not one.
        """
        return self._entry

    def values(self) -> dict:
        directory = Path(self.location.value()) / self.name.text().strip()
        return {
            "directory": directory,
            "video_dir": self.videos.value() or None,
            "model": None if self._entry is not None else (self.model.value() or None),
            "template": self.template.currentData(),
        }


class _Browse(QWidget):
    # Picking a file is not the same event as the text changing: it is the one
    # that means "run THIS file", and a shared model has to give way to it.
    picked = Signal(str)

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
            text = str(Path(picked)).replace("\\", "/")
            self.edit.setText(text)
            self.picked.emit(text)
