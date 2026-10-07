"""Step 7: render overlay movies."""

from __future__ import annotations

import os
from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem,
    QMessageBox, QPushButton, QSpinBox, QSplitter,
    QVBoxLayout, QWidget,
)

from ..tasks import TaskPanel
from .base import Page, _heading, _muted


def _reveal(path) -> str:
    """Open a folder in the desktop's file manager. Returns "" or why it failed."""
    import subprocess
    import sys as _sys

    path = Path(path)
    if not path.exists():
        return f"{path} does not exist yet"
    try:
        if _sys.platform == "win32":
            os.startfile(str(path))                       # noqa: S606 - a directory
        elif _sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as exc:
        return str(exc)
    return ""


class ExportPage(Page):
    def __init__(self, hub, parent=None):
        super().__init__(hub, parent)

        self.movies = QListWidget()
        self.movies.setAlternatingRowColors(True)
        self._list_stale = True

        self.group = QComboBox()
        self.source = QComboBox()
        for label, value in (("my labels", "labels"),
                             ("model predictions", "predictions"),
                             ("my labels and the model's", "both")):
            self.source.addItem(label, value)
        self.source.currentIndexChanged.connect(self._on_source_changed)
        self.source_note = _muted(QLabel(""))
        self.source_note.setWordWrap(True)

        self.hide_unlabelled = QCheckBox("draw only annotated tracks")
        self._hide_seed: tuple | None = None
        self.max_frames = QSpinBox()
        self.max_frames.setRange(0, 1_000_000)
        self.max_frames.setSpecialValueText("whole movie")

        options = QFormLayout()
        options.addRow("category group", self.group)
        options.addRow("colour by", self.source)
        options.addRow("", self.source_note)
        options.addRow(self.hide_unlabelled)
        options.addRow("frame limit", self.max_frames)
        options_box = QGroupBox("Options")
        options_box.setLayout(options)

        self.movies_box = QGroupBox("Annotated movies")
        movies_box = self.movies_box
        movies_layout = QVBoxLayout(movies_box)
        movies_layout.addWidget(self.movies)

        self.open_folder = QPushButton("Open project folder")
        self.open_folder.setToolTip(
            "Show the project in the file manager (movies go to render.out_dir)."
        )
        self.open_folder.clicked.connect(self._open_project_folder)
        folder_bar = QHBoxLayout()
        folder_bar.addWidget(self.open_folder)
        folder_bar.addStretch(1)
        movies_layout.addLayout(folder_bar)

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
            "Put the labels back onto the original movies.",
            help="""
<p>Draws the labels onto a copy of each movie, one colour per category. Works on
a half-finished CSV too.</p>

<p>Colour by <b>your labels</b>, the <b>model's predictions</b>, or both.
Predicted boxes are <b>dashed and show their confidence</b>.</p>

<p>Frames without a label or prediction are left unchanged.</p>
""",
        ))
        layout.addWidget(splitter, 1)

    def _open_project_folder(self) -> None:
        if self.cfg is None:
            return
        problem = _reveal(self.cfg.path.parent)
        if problem:
            self.status_message.emit(f"could not open the project folder: {problem}", 8000)

    def source_value(self) -> str:
        """"labels", "predictions" or "both"."""
        return self.source.currentData() or "labels"

    def blocker(self) -> str:
        if self.cfg is None or self.hub.status is None:
            return "no project open"
        if self.source_value() == "predictions":
            from ...infer.predict import predictions_path

            if not predictions_path(self.cfg).exists():
                return ("no predictions yet: run 'Label all clips with the model' "
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
        # Reseed the box only when render.hide_unlabelled or the project changes.
        seed = (self.cfg.path, bool(self.cfg.render.hide_unlabelled))
        if seed != self._hide_seed:
            self._hide_seed = seed
            self.hide_unlabelled.setChecked(seed[1])
        self._paint_source_note()
        self.refresh_gate()
        self.reload_movies()

    def _on_source_changed(self) -> None:
        """Update the note, gate and movie list for the new source."""
        self._paint_source_note()
        self.refresh_gate()
        self.reload_movies()

    def _paint_source_note(self) -> None:
        source = self.source_value()
        if source == "labels":
            self.source_note.setText("")
            return
        note = (
            "predictions are drawn as dashed boxes with their confidence"
        )
        if self.cfg is not None:
            from ...infer.predict import predictions_path, staleness

            if predictions_path(self.cfg).exists():
                stale = staleness(self.cfg)
                if stale:
                    note += f".  Warning: {stale}"
            else:
                note = "there are no predictions in this project yet"
        self.source_note.setText(note)

    def reload_movies(self) -> None:
        """Re-read the movie list (on show, not per annotation)."""
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
            counts, unit = self._predicted_movies()
        else:
            from ...view.render import annotated_movies

            counts, unit = annotated_movies(self.cfg), "annotated clips"

        for movie_id, count in counts.items():
            item = QListWidgetItem(f"{movie_id}      {count} {unit}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            item.setData(Qt.UserRole, movie_id)
            self.movies.addItem(item)

    def _predicted_movies(self) -> tuple[dict, str]:
        """({movie_id: row count}, what one row is). Never raises."""
        from ...infer.predict import load_predictions

        try:
            frame = load_predictions(self.cfg)
        except (FileNotFoundError, OSError, RuntimeError, ValueError):
            return {}, "predicted bouts"
        counts = frame["movie_id"].astype(str).value_counts()
        return ({str(k): int(v) for k, v in sorted(counts.items())},
                "predicted bouts")

    def _build_work(self):
        blocker = self.blocker()
        if blocker:
            QMessageBox.information(self, "Cannot render", blocker)
            return None
        source = self.source_value()
        group_name = self.group.currentText() or None
        if source != "labels":
            # Refuse up front rather than render an empty export.
            from ...infer.predict import load_predictions

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

        def work(reporter):
            from ...view.render import render_all

            return render_all(
                cfg, picked, group_name, hide_unlabelled=hide, max_frames=limit,
                reporter=reporter, source=source,
            )

        return work

    def shutdown(self) -> None:
        self.tasks.shutdown()
