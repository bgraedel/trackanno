"""Step 1: pick or create a project."""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QComboBox, QDialog, QFileDialog, QFormLayout, QFrame,
    QGridLayout, QGroupBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget,
    QListWidgetItem,
    QMenu, QMessageBox, QPushButton, QVBoxLayout, QWidget,
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
        self.template_btn = QPushButton("Template")
        self.template_btn.setToolTip(
            "Save this project's settings for new projects, or export a saved template to share."
        )
        template_menu = QMenu(self.template_btn)
        self.save_action = template_menu.addAction("Save as template...")
        self.save_action.triggered.connect(self._save_template)
        self.save_action.setEnabled(False)
        self.export_menu = template_menu.addMenu("Export")
        self.export_menu.aboutToShow.connect(self._fill_export_menu)
        self.template_btn.setMenu(template_menu)

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
clip cache) and <b>annotations.csv</b>.</p>

<p><b>Presets</b> under <b>start from</b>: <b>annotator</b> (label by hand from a
box detector), <b>suggester</b> (pose weights plus a trained classifier) and
<b>dense bouts</b> (score every frame and review bouts).</p>

<p><b>Templates</b>: <i>Template &gt; Save as template</i> stores this
project's config.yaml without video_dir, work_dir and csv, and <i>New
project</i> offers it under <b>start from</b>. <i>Template &gt; Export</i>
writes a saved template to a file to share; <i>Import</i> in the New project
dialog adds one.</p>
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
            self.recent.addItem("nothing yet: create or open a project above")

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
            except (OSError, ValueError) as exc:
                QMessageBox.warning(self, "Could not create project", str(exc))
                return
            self.hub.open_project(path)
            self.hub.go_to(1)
            entry = dialog.chosen_model()
            if entry is None:
                self.status_message.emit(
                    "project created: check the settings, then run detection", 8000
                )
                return
            # Download in the background into the config just written.
            _install_shared_model(self, entry, path)

    def _save_template(self) -> None:
        """Save the open project's settings as a named template."""
        if self.cfg is None:
            return
        template = _store_template(self, self.cfg.path, self.cfg.path.parent.name)
        if template is not None:
            self.status_message.emit(
                f"saved template {template.name!r}; Template > Export writes it to a file", 8000
            )

    def _fill_export_menu(self) -> None:
        """List the saved templates; presets ship with trackanno, so they are not offered."""
        from ...templates import list_templates

        self.export_menu.clear()
        saved = [t for t in list_templates() if t.path is not None]
        for template in saved:
            action = self.export_menu.addAction(template.name)
            action.triggered.connect(
                lambda _=False, name=template.name: self._export_template(name)
            )
        if not saved:
            self.export_menu.addAction("no saved templates").setEnabled(False)

    def _export_template(self, name: str) -> None:
        """Write a saved template to a file."""
        from ...templates import _slug, export_template, local_files

        picked, _ = QFileDialog.getSaveFileName(
            self, "Export template", str(Path.home() / f"{_slug(name)}.yaml"),
            "Template (*.yaml)",
        )
        if not picked:
            return
        try:
            dest = export_template(name, picked)
        except Exception as exc:
            QMessageBox.warning(self, "Export template", str(exc))
            return
        _tell_files(
            self, "Export template", local_files(dest),
            "These settings name files on this computer. Whoever imports the "
            "template has to set them",
        )
        self.status_message.emit(f"exported template to {dest}", 8000)

    def _go_next(self) -> None:
        self.hub.go_to(self._next_page)

    def refresh_gate(self) -> None:
        # Cheap repaint from hub.status.
        if self.cfg is None or self.hub.status is None:
            return
        self._paint_summary()

    # -- display ------------------------------------------------------------ #
    def on_project(self) -> None:
        self.save_action.setEnabled(self.cfg is not None)
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
    """Name, location, template, video folder and detection model for a new project."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New project")
        self.setMinimumWidth(560)

        self.name = QLineEdit("my-annotation-project")
        self.location = _Browse("dir", str(Path.home()))
        self.videos = _Browse("dir", "")
        self.model = _Browse("file", "", "PyTorch weights (*.pt);;All files (*)")

        self._entry = None
        self.registry_btn = QPushButton("Choose shared model...")
        self.registry_btn.setProperty("accent", "true")
        self.registry_btn.setToolTip(
            "Download a model and its settings from the shared registry after the project is created."
        )
        self.registry_btn.clicked.connect(self._pick_shared_model)
        self.model.layout().insertWidget(1, self.registry_btn)
        self.model.picked.connect(self._drop_shared_model)
        self.model_note = QLabel()
        self.model_note.setWordWrap(True)
        self.model_note.setProperty("role", "muted")
        self.model_note.setVisible(False)

        self.template = QComboBox()
        self.import_btn = QPushButton("Import...")
        self.import_btn.setToolTip("Add a template file, or another project's config.yaml.")
        self.import_btn.clicked.connect(self._import_template)
        self.forget_btn = QPushButton("Forget")
        self.forget_btn.setToolTip("Delete the selected template. Presets stay.")
        self.forget_btn.clicked.connect(self._forget_template)
        self.template_note = QLabel()
        self.template_note.setWordWrap(True)
        self.template_note.setProperty("role", "muted")
        self._fill_templates()
        self.template.currentIndexChanged.connect(self._sync_selection)
        template_row = QHBoxLayout()
        template_row.addWidget(self.template, 1)
        template_row.addWidget(self.import_btn)
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
            "Everything else comes from the template and can be changed on the "
            "Configure page."
        )
        note.setWordWrap(True)
        note.setProperty("role", "muted")
        note.setToolTip("The clip cache and annotations.csv go inside the project folder.")

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
        """Refuse a missing name or existing project; ask about missing paths."""
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
                f"A project already exists in {target}. Open it or pick another name.",
            )
            return

        missing = []
        if not self.videos.value():
            missing.append("no movie folder chosen")
        elif not _exists(Path(self.videos.value()), "dir"):
            missing.append(f"movie folder not found: {self.videos.value()}")
        if self._entry is not None:
            pass          # fetched after creation
        elif not self.model.value():
            missing.append("no detection model chosen")
        elif not _exists(Path(self.model.value()), "file"):
            missing.append(f"model file not found: {self.model.value()}")
        if missing:
            answer = QMessageBox.question(
                self, "Create anyway?",
                "Detection cannot run until this is fixed on the Configure page:\n\n- "
                + "\n- ".join(missing) + "\n\nCreate the project anyway?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        self.accept()

    def run(self) -> bool:
        """Run the dialog modally (exec_ on Qt5, exec on Qt6)."""
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

    def _sync_selection(self) -> None:
        """Show the template summary; only saved templates can be forgotten."""
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

    def _import_template(self) -> None:
        from ...templates import _stored_name, missing_files

        picked, _ = QFileDialog.getOpenFileName(
            self, "Import template", str(Path.home()),
            "Template or config (*.yaml *.yml);;All files (*)",
        )
        if not picked:
            return
        template = _store_template(self, picked, _stored_name(Path(picked)))
        if template is None:
            return
        self._fill_templates()
        self.template.setCurrentIndex(self.template.findData(template.name))
        _tell_files(
            self, "Import template", missing_files(template.path),
            "These settings name files that do not exist on this computer. Set them",
        )

    def _pick_shared_model(self) -> None:
        """Choose registry weights; they are downloaded after the project is created."""
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
            f"<b>{entry.label}</b> and its settings will be downloaded when the "
            "project is created; the path above is not used."
        )
        self.model_note.setVisible(True)

    def _drop_shared_model(self, _path: str = "") -> None:
        """A browsed .pt replaces the chosen shared model."""
        self._entry = None
        self.model.edit.setEnabled(True)
        self.model_note.clear()
        self.model_note.setVisible(False)

    def chosen_model(self):
        """The registry entry to fetch after the project is created, or None."""
        return self._entry

    def values(self) -> dict:
        directory = Path(self.location.value()) / self.name.text().strip()
        return {
            "directory": directory,
            "video_dir": self.videos.value() or None,
            "model": None if self._entry is not None else (self.model.value() or None),
            "template": self.template.currentData(),
        }


def _store_template(parent, source, suggested: str):
    """Ask for a name and save ``source``'s settings as a template; None if cancelled or refused."""
    from ...templates import find_template, save_template

    name, ok = QInputDialog.getText(
        parent, "Save as template", "Name for these settings:", text=suggested
    )
    name = (name or "").strip()
    if not ok or not name:
        return None
    if find_template(name) is not None:
        answer = QMessageBox.question(
            parent, "Save as template",
            f"A template named {name!r} already exists. Replace it?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return None
    try:
        return save_template(name, source, overwrite=True)
    except Exception as exc:
        QMessageBox.warning(parent, "Save as template", str(exc))
        return None


def _tell_files(parent, title: str, files: dict, intro: str) -> None:
    """List file settings the user has to set, if there are any."""
    if files:
        QMessageBox.information(
            parent, title,
            f"{intro} on the Configure page or in config.yaml:\n\n"
            + "\n".join(f"{key}: {value}" for key, value in files.items()),
        )


class _Browse(QWidget):
    # Emitted on Browse only, not on typing.
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
