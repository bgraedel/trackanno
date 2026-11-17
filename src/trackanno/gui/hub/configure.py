"""Step 2: edit config.yaml, and fetch shared model weights."""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem,
    QMessageBox, QPushButton, QScrollArea, QTabWidget, QVBoxLayout, QWidget,
)

from ...config import ConfigError
from ...project import (
    _exists, write_config,
)
from ..configform import CategoryEditor, ConfigForm, KeypointEditor
from ..theme import (
    banner_style,
)
from .base import Page, _heading


def _install_shared_model(page, entry, config_path) -> None:
    """Download shared weights in the background and apply their manifest to ``config_path``.

    The sha256 pin is computed from the downloaded file, not taken from the registry.
    """
    from ...video.models import apply_manifest, fetch, mismatch_warnings

    config_path = Path(config_path)

    def work(reporter):
        path = fetch(entry, reporter)
        import hashlib

        sha = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                sha.update(block)
        return path, sha.hexdigest()

    def done(result):
        path, digest = result
        # Settle unsaved edits first: applying rewrites the file.
        config = page.hub.page(ConfigPage)
        if (config is not None and config.cfg is not None
                and config.cfg.path == config_path and not config.confirm_leave()):
            page.status_message.emit(
                f"{entry.id} downloaded but not applied: save or revert the "
                "settings, then pick it again", 12000)
            return
        try:
            told = apply_manifest(config_path, entry, digest)
        except Exception as exc:
            QMessageBox.warning(page, "From registry", str(exc))
            return
        page.hub.open_project(config_path)
        summary = ", ".join(told) if told else "already matched the settings"
        page.status_message.emit(f"{entry.id}: {summary}   ->   {path}", 12000)
        for warning in mismatch_warnings(page.hub.cfg, entry):
            page.status_message.emit(warning, 15000)

    fetching = f"fetching {entry.id}..."

    def settle(text: str) -> None:
        """Replace the fetching message if it is still shown."""
        bar = page.hub.statusBar()
        if bar.currentMessage() == fetching:
            bar.showMessage(text, 5000)

    def failed(message):
        settle(f"{entry.id}: download failed")
        QMessageBox.warning(page, "From registry", message)

    page.status_message.emit(fetching, 0)
    page.run_bg("download", work, done, failed,
                lambda: settle(f"{entry.id}: download stopped"))


class ConfigPage(Page):
    def __init__(self, hub, parent=None):
        super().__init__(hub, parent)
        # Basic and advanced forms over the same file.
        self.form = ConfigForm("basic")
        self.advanced = ConfigForm("advanced")
        self.categories = CategoryEditor()
        self._dirty = False
        self._loading = False
        self._loaded_from: Path | None = None
        # Categories and keypoints as loaded, so a save writes them only if changed.
        self._categories_at_load: list = []
        self._keypoints_at_load: dict = {}
        # Keys written with the next save that are not form fields.
        self._extra: dict = {}

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
        self.tabs.setTabToolTip(0, "Paths and clip length.")
        self.tabs.setTabToolTip(1, "The label buttons used while annotating.")
        self.tabs.setTabToolTip(2, "Keypoint names, used only when the task is 'pose'.")
        self.tabs.setTabToolTip(3, "Detector, tracker, cropping and model settings.")
        tabs = self.tabs
        self.KEYPOINT_TAB = 2

        # Save errors, pinned at the top.
        self.banner = QLabel("")
        self.banner.setWordWrap(True)
        self.banner.setTextFormat(Qt.RichText)
        self.banner.setVisible(False)
        self.banner.setStyleSheet(banner_style("danger"))

        self.pose_notice = QLabel(
            "The task is <b>pose</b> but no keypoints are listed. Read them from the model."
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

        # Offer the checkpoint's task and keypoint count when they differ from the form.
        self.model_notice = QLabel("")
        self.model_notice.setWordWrap(True)
        self.model_notice.setStyleSheet(banner_style("warning"))
        self.model_apply = QPushButton("Use these")
        self.model_apply.setProperty("accent", "true")
        self.model_apply.clicked.connect(self._apply_model_info)
        self.model_row = QWidget()
        model_layout = QHBoxLayout(self.model_row)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.addWidget(self.model_notice, 1)
        model_layout.addWidget(self.model_apply)
        self.model_row.setVisible(False)
        self._model_offer: dict = {}

        self.revert_button = QPushButton("Revert")
        self.revert_button.setToolTip("Drop unsaved edits and reload config.yaml.")
        self.revert_button.clicked.connect(self.revert)
        self.save_button = QPushButton("Save")
        self.save_button.setProperty("accent", "true")
        self.save_button.clicked.connect(self.save)
        self.state = QLabel("")
        self.state.setWordWrap(True)

        bar = QHBoxLayout()
        bar.addWidget(self.state, 1)
        bar.addWidget(self.revert_button)
        bar.addWidget(self.save_button)

        # Sits in the "detection model" row, ahead of Browse.
        self.registry_btn = QPushButton("Choose shared model...")
        self.registry_btn.setProperty("accent", "true")
        self.registry_btn.setToolTip(
            "Download a model and its settings from the shared registry."
        )
        self.registry_btn.clicked.connect(self._pick_registry_model)
        model_field = self.form._widgets.get("paths.model")
        if model_field is not None:
            model_field.row.insertWidget(1, self.registry_btn)
            model_field.picked.connect(self._use_local_model)

        layout = QVBoxLayout(self)
        self.heading = _heading(
            "Configure",
            "Every setting for this project. Edit, then Save.",
            help="""
<p><b>Save</b> writes only the changed settings into <b>config.yaml</b>, keeping
its comments. <b>Revert</b> drops the edits and reloads the file.</p>

<p>A save that would not load is refused with the loader's message, and the file
is left untouched.</p>

<p>Settings marked <b>*</b> take effect only after detection is run again.</p>
""",
        )
        layout.addWidget(self.heading)
        layout.addWidget(self.banner)
        layout.addWidget(self.pose_row)
        layout.addWidget(self.model_row)
        layout.addWidget(tabs, 1)
        layout.addLayout(bar)

        # Probe a newly picked .pt for its task and keypoint count.
        self._probed_model: str | None = None
        self._model_info: dict | None = None
        self._model_timer = QTimer(self)
        self._model_timer.setSingleShot(True)
        self._model_timer.setInterval(400)
        self._model_timer.timeout.connect(self._probe_model)
        self.form._widgets["paths.model"].edit.textChanged.connect(
            lambda _t: self._model_timer.start()
        )

        self._paint_dirty()
        self.form.changed.connect(self._mark_dirty)
        self.advanced.changed.connect(self._mark_dirty)
        self.form.changed.connect(self._paint_centerline)
        self.categories.changed.connect(self._mark_dirty)
        self.keypoints.changed.connect(self._mark_dirty)

    def _sync_model_status(self) -> None:
        """Show under paths.model when a shared model is in use instead."""
        field = self.form._widgets.get("paths.model")
        if field is None or not hasattr(field, "set_status") or "model.source" in self._extra:
            return
        spec = getattr(self.cfg, "model", None) if self.cfg else None
        if spec is None or not spec.is_remote:
            field.set_status("")
            field.edit.setEnabled(True)
            return

        from ...video.models import cache_dir, find_model, local_model_path

        entry = find_model(spec.source, spec.version)
        name = " ".join(b for b in (spec.source, spec.version) if b)
        here = local_model_path(self.cfg)
        if here.exists() and cache_dir() in here.parents:
            where = f"cached, {here.stat().st_size / 1e6:.0f} MB"
        elif entry is None:
            where = "not in this registry"
        else:
            where = "downloads when detection runs"
        field.set_status(
            f"using shared model <b>{name}</b> ({where}); the path above is not used."
        )
        field.edit.setEnabled(False)

    def _use_local_model(self, path: str) -> None:
        """A browsed .pt replaces the shared model: clear ``model.source`` on save."""
        spec = getattr(self.cfg, "model", None) if self.cfg else None
        if spec is None or not spec.is_remote:
            return
        was = " ".join(b for b in (spec.source, spec.version) if b)
        self._extra.update({"model.source": "", "model.version": "", "model.sha256": ""})
        field = self.form._widgets["paths.model"]
        field.edit.setEnabled(True)
        field.set_status(f"Save to use {Path(path).name} instead of the shared "
                         f"model {was}.")
        self._mark_dirty()

    def _pick_registry_model(self) -> None:
        """Pick shared weights, then download and apply them."""
        if self.cfg is None:
            QMessageBox.information(self, "From registry", "Open a project first.")
            return
        dialog = ModelRegistryDialog(self)
        if not dialog.run():
            return
        entry = dialog.chosen()
        if entry is None:
            return
        # Settle unsaved edits first: applying rewrites the file.
        if not self.confirm_leave():
            return
        _install_shared_model(self, entry, self.cfg.path)

    def _probe_model(self) -> None:
        """Read the selected checkpoint's metadata off the UI thread."""
        if self.cfg is None:
            return
        raw = self.form._widgets["paths.model"].value()
        if not raw:
            return
        path = Path(raw)
        if not path.is_absolute():
            path = (self.cfg.path.parent / path).resolve()
        # Timeout-guarded: is_file() on a share can block.
        if not _exists(path, "file"):
            return
        if str(path) == self._probed_model:
            if self._model_info:
                self._offer_model_info(self._model_info)
            return
        self._probed_model = str(path)
        self._model_info = None
        self.keypoints.state.setText("reading the model file...")

        def work(reporter):
            from ...detect.tiling import read_model_info

            return read_model_info(path)

        def failed(message):
            self._probed_model = None
            self.keypoints.state.setText(f"could not read it: {message}")

        self.run_bg("probe", work, self._on_model_info, failed)

    def _on_model_info(self, info) -> None:
        """Report the checkpoint's metadata and offer it where the form differs."""
        if not info or self.cfg is None:
            return
        self._model_info = info
        offer = self._offer_model_info(info)
        count = info.get("n_keypoints")
        classes = info.get("classes") or []
        self.status_message.emit(
            f"{info.get('name', 'model')}: {info.get('task') or 'unknown task'}"
            + (f", {count} keypoints" if count else "")
            + (f", {len(classes)} class(es)" if classes else "")
            + ("   ->   differs from the settings" if offer
               else "   ->   matches the settings"),
            10000,
        )

    def _centerline_on(self) -> bool:
        """Whether the task is pose + centerline."""
        return self.form.centerline_on()

    def _paint_centerline(self) -> None:
        """With a centerline, the keypoints are its output, not the pose model's."""
        on = self._centerline_on()
        self.keypoints.read_button.setEnabled(not on and not self.bg_running("keypoints"))
        if on:
            self.keypoints.state.setText("named by the centerline model: the Head, then its points "
                                         "from Base to Tip")
            self.pose_row.setVisible(False)

    def _offer_model_info(self, info: dict) -> dict:
        """Show the checkpoint's task and keypoint count where the form differs."""
        offer: dict = {}
        task_widget = self.form._widgets.get("detection.task")
        # pose + centerline runs a pose model, so a pose checkpoint matches it.
        task_now = self.form.current().get("detection.task", "") if task_widget else ""
        model_task = info.get("task")
        count = info.get("n_keypoints") or 0
        name = info.get("name", "the model")
        if (model_task in ("pose", "detect") and task_widget is not None
                and task_now != model_task):
            offer["task"] = model_task
        pose = offer.get("task", task_now) == "pose"
        # A centerline names its own output; the pose model's count does not apply.
        if count and pose and not self._centerline_on() and self.keypoints.count.value() != count:
            offer["count"] = count
        if self._centerline_on():
            self._paint_centerline()
        else:
            self.keypoints.state.setText(f"{count} from {name}" if count else "")
        self._model_offer = offer
        if offer:
            said = [f"task {offer['task']}"] if "task" in offer else []
            said += [f"{offer['count']} keypoints"] if "count" in offer else []
            self.model_notice.setText(f"{name} reports {' and '.join(said)}; the "
                                      "settings differ.")
        self.model_row.setVisible(bool(offer))
        return offer

    def _apply_model_info(self) -> None:
        """Take the checkpoint's task and keypoint count into the form."""
        offer, self._model_offer = self._model_offer, {}
        name = (self._model_info or {}).get("name", "the model")
        if "task" in offer:
            self.form._widgets["detection.task"].setCurrentText(offer["task"])
        if "count" in offer:
            self.keypoints.set_count(offer["count"], f"{offer['count']} from {name}")
        self.model_row.setVisible(False)
        if offer:
            self._mark_dirty()
        self._show_warnings()

    def _mark_dirty(self) -> None:
        if self._loading:
            return
        self._refresh_auto()
        self._dirty = True
        self.state.setText("unsaved changes")
        self._paint_dirty()

    def _paint_dirty(self) -> None:
        """Enable Save and Revert and mark the title while there are unsaved edits."""
        self.save_button.setEnabled(self._dirty)
        self.revert_button.setEnabled(self._dirty)
        self.heading.title.setText("Configure *" if self._dirty else "Configure")
        window = self.window()
        if window is not None and window is not self:
            window.setWindowModified(self._dirty)

    def _refresh_auto(self) -> None:
        """Show what each 'auto' resolves to, across both forms."""
        values = {**self.form.current(), **self.advanced.current()}
        self.form.show_resolved(values)
        self.advanced.show_resolved(values)

    def on_project(self) -> None:
        if not self.bg_running("keypoints"):
            self.keypoints.read_button.setEnabled(True)
        if self.cfg is None:
            return
        if self._dirty and self.cfg.path == self._loaded_from:
            # Keep edits a refused save left behind.
            self._sync_model_status()
            return
        import yaml

        # _loading keeps the reload from marking the page dirty.
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
        self._loaded_from = self.cfg.path
        self._categories_at_load = self.categories.values()
        self._keypoints_at_load = self.keypoints.values()
        self._extra = {}
        self._dirty = False
        self._paint_dirty()
        self.banner.setVisible(False)
        self.model_row.setVisible(False)
        self.state.setText("")
        self._sync_model_status()
        self._refresh_auto()
        self._show_warnings()
        self._model_timer.start()

    def _show_warnings(self) -> None:
        """Outline paths that point at nothing (warnings never block a save)."""
        root = self.cfg.path.parent if self.cfg else None
        warnings = {**self.form.warnings(root), **self.advanced.warnings(root)}
        # paths.model is unused under a shared model.
        spec = getattr(self.cfg, "model", None) if self.cfg else None
        if spec is not None and spec.is_remote:
            warnings.pop("paths.model", None)
        # Problems are outlined too, but only warnings are returned.
        marks = {**warnings, **self.form.problems(), **self.advanced.problems()}
        self.form.mark(marks)
        self.advanced.mark(marks)
        picker = self.advanced._widgets.get("suggest.feature_groups")
        if picker is not None and self.cfg is not None:
            from ...measure.clipfeatures import available_groups

            picker.set_available(available_groups(self.cfg))
        self.pose_row.setVisible(bool(self.cfg and self.cfg.needs_keypoints))
        self.tabs.setTabEnabled(self.KEYPOINT_TAB, bool(self.cfg and self.cfg.is_pose))
        self._paint_centerline()
        return warnings

    def _read_keypoints(self) -> None:
        """Read the keypoint count from the checkpoint; names are kept."""
        if self.cfg is None or self.bg_running("keypoints"):
            return
        from ...video.models import local_model_path

        model_path = local_model_path(self.cfg)
        if not _exists(model_path, "file"):
            self._fail("Cannot read the keypoint count:",
                       [f"model file not found: {model_path}"])
            return
        self.keypoints.read_button.setEnabled(False)
        self.keypoints.state.setText("reading the model file...")

        def work(reporter):
            from ...detect.tiling import read_keypoint_count

            return read_keypoint_count(model_path)

        self.run_bg("keypoints", work, self._on_keypoints_read, self._on_keypoints_failed)

    def _on_keypoints_read(self, count) -> None:
        self.keypoints.read_button.setEnabled(True)
        if not count:
            self.keypoints.state.setText("")
            self._fail(
                "That model has no keypoints:",
                [f"{self.cfg.paths.model} is not a pose model. Set the task to "
                 "'detect' or choose pose weights."],
            )
            return
        # Only the count comes from the model; new rows get kp<N> placeholders.
        self.keypoints.set_count(count, f"{count} from {self.cfg.paths.model.name}")
        self.tabs.setCurrentIndex(self.KEYPOINT_TAB)
        self.status_message.emit(
            f"{self.cfg.paths.model.name} predicts {count} keypoints: name them in "
            "model order", 10000
        )

    def _on_keypoints_failed(self, message: str) -> None:
        self.keypoints.read_button.setEnabled(True)
        self.keypoints.state.setText("")
        self._fail("Could not read the model file:", [message])

    def save(self) -> bool:
        """Validate with the real loader, then write the changed settings."""
        if self.cfg is None:
            return False

        problems = self.categories.problems()
        if problems:
            self._fail(
                "Not saved, fix these labels:",
                problems,
            )
            return False
        # Malformed fields are refused here, by name.
        malformed = {**self.form.problems(), **self.advanced.problems()}
        if malformed:
            self._show_warnings()
            self._fail("Not saved, fix these settings:",
                       [f"{key}: {why}" for key, why in malformed.items()])
            return False
        # Every tab writes to the one file.
        updates = {**self.form.values(), **self.advanced.values(), **self._extra}
        categories = self.categories.values()

        # Check keypoints only if the task about to be written is pose.
        shown = {**self.form.current(), **self.advanced.current()}
        is_pose = str(shown.get("detection.task", "")) == "pose"
        if is_pose:
            problems = self.keypoints.problems()
            if problems:
                self._fail("Not saved, fix the keypoint layout:",
                           problems)
                return False

        keypoints = self.keypoints.values()
        try:
            write_config(
                self.cfg.path, updates,
                categories=categories if categories != self._categories_at_load else None,
                keypoints=keypoints if is_pose and keypoints != self._keypoints_at_load
                else None,
            )
        except (ConfigError, ValueError, OSError) as exc:
            self._fail("Not saved, invalid setting:", [str(exc)])
            return False

        self.form.saved()
        self.advanced.saved()
        self._categories_at_load = categories
        if is_pose:
            self._keypoints_at_load = keypoints
        self._extra = {}
        self.banner.setVisible(False)
        self._dirty = False
        self._paint_dirty()
        self.hub.config_saved()
        self._sync_model_status()
        warnings = self._show_warnings()
        self.state.setText(
            "saved" if not warnings
            else f"saved, but {len(warnings)} path(s) point at nothing (outlined above)"
        )
        self.status_message.emit(f"saved {self.cfg.path.name}", 4000)
        return True

    def _fail(self, headline: str, details: list[str]) -> None:
        items = "".join(f"<li>{d}</li>" for d in details[:5])
        self.banner.setText(f"<b>{headline}</b><ul>{items}</ul>")
        self.banner.setVisible(True)
        self.state.setText("NOT saved")
        self.status_message.emit(details[0][:120], 8000)

    def revert(self) -> None:
        """Drop the unsaved edits and reload the form from disk."""
        self._dirty = False
        self.on_project()

    def confirm_leave(self) -> bool:
        """Settle unsaved edits before leaving. False to stay on the page."""
        if not self._dirty or self.cfg is None:
            return True
        answer = self._ask_unsaved()
        if answer == "save":
            return self.save()
        if answer == "discard":
            self.revert()
            return True
        return False

    def _ask_unsaved(self) -> str:
        """"save", "discard" or "cancel"."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Unsaved settings")
        box.setText("Save the changes to the settings?")
        save = box.addButton("Save", QMessageBox.AcceptRole)
        discard = box.addButton("Discard", QMessageBox.DestructiveRole)
        cancel = box.addButton("Cancel", QMessageBox.RejectRole)
        box.setDefaultButton(save)
        box.setEscapeButton(cancel)
        box.exec()
        clicked = box.clickedButton()
        return "save" if clicked is save else "discard" if clicked is discard else "cancel"


class ModelRegistryDialog(QDialog):
    """Pick shared weights and set this machine's registry URL (not stored in config.yaml)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        from ...video.models import registry_url

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
        from ...video.models import RegistryError, set_registry_url

        try:
            set_registry_url(self.url.text())
        except RegistryError as exc:
            QMessageBox.warning(self, "Detection models", str(exc))
            return
        self._fill(refresh=True)

    def _fill(self, refresh: bool) -> None:
        from ...video.models import RegistryError, cached_path, list_models, registry_url

        self.list.clear()
        if not registry_url():
            self.detail.setText(
                "No registry set. Paste the index URL above and press Refresh."
            )
            return
        try:
            offered = list_models(refresh=refresh)
        except RegistryError as exc:
            self._entries = []
            self.detail.setText(str(exc))
            return
        # List runnable models only; count the rest.
        self._entries = [e for e in offered if e.runnable]
        for entry in self._entries:
            here = cached_path(entry.sha256, entry.name) is not None
            self.list.addItem(QListWidgetItem(entry.label + ("   [cached]" if here else "")))
        others = len(offered) - len(self._entries)
        if not self._entries:
            self.detail.setText(
                f"None of the {len(offered)} model(s) in this registry can run "
                "in trackanno." if offered
                else "The registry index lists no models."
            )
        elif others:
            self.detail.setText(
                f"{others} further model(s) for other tools are hidden."
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
