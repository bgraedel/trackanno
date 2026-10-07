"""Step 5: train the suggester and label every clip."""

from __future__ import annotations

import json
from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QHBoxLayout, QLabel, QMessageBox, QPushButton, QSplitter,
    QTabWidget, QTextBrowser, QVBoxLayout,
)

from ...annotate.clips import MANIFEST_FILE
from ..distribution import DistributionView
from ..theme import (
    tokens,
)
from ..tasks import TaskPanel
from .annotate import AnnotatePage
from .base import Page, _heading, _muted


def count_orphans(csv_path: Path, work_dir: Path) -> int:
    """Labels in ``csv_path`` whose clip_id no manifest under ``work_dir`` has."""
    import pandas as pd

    if not csv_path.exists():
        return 0
    try:
        labels = pd.read_csv(csv_path, usecols=["clip_id"])["clip_id"].astype(str)
        known: set[str] = set()
        for path in sorted(work_dir.glob(f"*/{MANIFEST_FILE}")):
            known.update(pd.read_parquet(path, columns=["clip_id"])["clip_id"].astype(str))
    except Exception as exc:
        print(f"[trackanno] could not check the label links: {exc}")
        return 0
    return int((~labels.isin(known)).sum()) if known else 0


def predict_plan_line(cfg) -> str:
    """What "Label all clips with the model" would score under ``cfg``."""
    from ...infer.densepredict import count_windows, window_length, window_stride

    n_windows, n_tracks = count_windows(cfg)
    return (f"{n_windows:,} windows of {window_length(cfg)} frames every "
            f"{window_stride(cfg)} over {n_tracks:,} tracks")


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

        # Labels whose clip_id no longer resolves (e.g. after re-detection).
        self._orphans = 0
        self.repair_note = _muted(QLabel(""))
        self.repair_note.setWordWrap(True)
        self.repair_note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        # Training and predicting share the task panel.
        self._job = "train"
        self._predict_blocker = "no project open"
        self._predict_state = "no project open"
        self._predict_summary = ""
        # Prediction plan summary, built off-thread.
        self._plan_line = ""
        self.predict_button = QPushButton("Label all clips with the model")
        self.predict_button.setEnabled(False)
        self.predict_button.setToolTip(
            "Score every clip and write predictions.parquet (the annotation CSV is untouched)."
        )
        self.predict_button.clicked.connect(self._label_all)
        self.predict_note = _muted(QLabel(""))
        self.predict_note.setWordWrap(True)
        self.predict_note.setToolTip(
            "Window length, stride and count under the current settings."
        )

        self._has_predictions = False

        predict_row = QHBoxLayout()
        predict_row.addWidget(self.predict_button)
        predict_row.addWidget(self.predict_note, 1)

        self.distribution = DistributionView()
        results = QTabWidget()
        results.addTab(self.report, "Last report")
        results.addTab(self.distribution, "Label distribution")

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.tasks)
        splitter.addWidget(results)
        splitter.setSizes([300, 380])

        layout = QVBoxLayout(self)
        layout.addWidget(_heading(
            "Model",
            "Learn from your labels, and label everything with what it learnt.",
            help="""
<p>Fits a classifier on your labels and cross-validates it <b>leaving whole
movies out</b>, so the score is for an unseen movie.</p>

<p>The suggestion is <b>shown, never pre-selected</b>, and only when Cohen's
kappa beats chance by a bootstrapped margin.</p>

<p>About a tenth of clips are an audit holdout: never trained on, never
suggested. Agreement on those is the honest measure.</p>

<p><b>Label distribution</b> flags values with no labels, values below
<b>suggest.min_per_class</b>, values in only one movie, values no longer in the
config, and any value over 90% of the labels.</p>

<p><b>Label all clips with the model</b> writes predictions.parquet for the
Review and Export pages. It never writes to your annotation CSV.</p>
""",
        ))
        layout.addWidget(self.repair_note)
        layout.addLayout(predict_row)
        layout.addWidget(splitter, 1)

    def blocker(self) -> str:
        if self.cfg is None or self.hub.status is None:
            return "no project open"
        if self.hub.status.n_annotated < 2:
            return "nothing labelled yet: annotate some clips first"
        return ""

    def refresh_gate(self) -> None:
        # Rebuilt when shown, not per annotation.
        self.distribution.stale = True
        # Busy means this button is Cancel; leave it clickable.
        self.tasks.run_button.setEnabled(self.tasks.busy or not self.blocker())
        self.predict_button.setEnabled(not self._predict_blocker and not self.tasks.busy)

    def on_project(self) -> None:
        self.distribution.set_project(self.cfg, self.hub.status)
        self._count_orphans()
        self._refresh_predict_gate()
        self.refresh_gate()
        self._load_report()

    def on_shown(self) -> None:
        self._load_report()
        if self.distribution.stale:
            self.distribution.reload()

    def _refresh_predict_gate(self) -> None:
        """Re-read the prediction gate and re-plan (not on the per-annotation path)."""
        if self.cfg is None:
            self._predict_blocker = "no project open"
            self._has_predictions = False
            self._predict_state = self._predict_blocker
        else:
            from ...infer.predict import prediction_blocker, predictions_path, staleness

            self._predict_blocker = prediction_blocker(self.cfg)
            self._has_predictions = predictions_path(self.cfg).exists()
            self._predict_state = self._predict_blocker
            if not self._predict_blocker and self._has_predictions:
                stale = staleness(self.cfg)
                self._predict_state = (
                    f"predictions.parquet exists but {stale}" if stale
                    else "predictions.parquet is up to date; the Export page can "
                         "colour movies by it"
                )
        self._paint_predict_note()
        self._plan_preview()

    def _paint_predict_note(self) -> None:
        """Show the prediction state and the plan line."""
        self.predict_note.setText(
            "\n".join(line for line in (self._predict_state, self._plan_line) if line)
        )

    # -- what the settings resolve to ---------------------------------------- #
    def _plan_preview(self) -> None:
        """State the plan off the UI thread."""
        if self.cfg is None:
            self._plan_line = ""
            self._paint_predict_note()
            return
        cfg = self.cfg

        def work(reporter):
            reporter.raise_if_cancelled()
            return predict_plan_line(cfg)

        # Keep the previous line up while recomputing.
        if not self._plan_line:
            self._on_plan_line("working out what these settings would do...")
        self.run_bg("plan", work, self._on_plan_line, lambda message: self._on_plan_line(
            f"could not work out what these settings would do: {message}"))

    def _on_plan_line(self, line: str) -> None:
        self._plan_line = str(line)
        self._paint_predict_note()

    def _count_orphans(self) -> None:
        """Count orphaned labels off the UI thread."""
        if self.cfg is None:
            self._on_orphans(0)
            return
        csv_path, work_dir = self.cfg.paths.csv, self.cfg.paths.work_dir

        def work(reporter):
            return count_orphans(csv_path, work_dir)

        self.run_bg("orphans", work, self._on_orphans)

    def _on_orphans(self, count: int) -> None:
        self._orphans = int(count)
        self._paint_repair_note()

    def _paint_repair_note(self) -> None:
        self.repair_note.setText(
            f"{self._orphans} label(s) point at missing clips and are ignored. "
            f"Run {self._repair_command()} to re-link them." if self._orphans else ""
        )

    def _repair_command(self) -> str:
        """The repair CLI call for the open project."""
        if self.cfg is None:
            return "trackanno repair-links --config <config.yaml>"
        return f'trackanno repair-links --config "{self.cfg.path}"'

    # -- labelling everything with the model --------------------------------- #
    def _label_all(self) -> None:
        """Score every clip into predictions.parquet, off the UI thread."""
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
            from ...infer.predict import predict_all, summarise

            frame = predict_all(cfg, reporter=reporter)
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
            from ...learn.suggest import train

            return train(cfg, verbose=False, reporter=reporter)

        return work

    def _on_finished(self, ok: bool) -> None:
        if self._job == "predict":
            self._job = "train"
            self._refresh_predict_gate()
            self.hub.reload_project()
            if ok:
                self.status_message.emit(
                    f"predictions written: {self._predict_summary}", 12000
                )
            return
        self._load_report()
        self.hub.reload_project()
        if not ok:
            return
        # The model is not in the panel's fingerprint, so hand it over directly.
        annotate = self.hub.page(AnnotatePage)
        panel = getattr(annotate, "panel", None)
        if panel is not None:
            line = panel.reload_suggester()
            self.status_message.emit(f"model reloaded into the annotator: {line}", 8000)

    def shutdown(self) -> None:
        self.tasks.shutdown()

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
        parts.append(f"<h3>{group}: {badge}</h3>")
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
            by_class = ", ".join(
                f"{k} ({v})" for k, v in rep.get("orphaned_by_class", {}).items()
            )
            parts.append(
                f"<p style='color:{tokens()['warning']};'><b>{rep['n_orphaned']} label(s) "
                "were ignored</b>: their clips are not in the feature table"
                + (f" ({by_class})" if by_class else "")
                + ". Run trackanno repair-links, then retrain.</p>"
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
        parts.append(_feature_table(rep.get("top_features") or []))
        parts.append("<hr>")
    return "".join(parts)


def _feature_table(features: list, top: int = 10) -> str:
    """HTML table of the top feature importances, bars scaled to the largest."""
    if not features:
        return ""
    shown = features[:top]
    best = max((float(f.get("importance", 0.0)) for f in shown), default=0.0)
    rows = []
    for f in shown:
        weight = float(f.get("importance", 0.0))
        width = int(round(100 * weight / best)) if best > 0 else 0
        rows.append(
            f"<tr><td align='right'>&nbsp;{weight:.2f}&nbsp;</td>"
            f"<td>{f.get('feature', '')}</td>"
            f"<td width='120'><div style='background:{tokens()['accent']};"
            f"height:7px;width:{max(width, 1)}%;'></div></td></tr>"
        )
    return (
        f"<p>top {len(shown)} features (% of the fitted model's total importance):</p>"
        f"<table cellspacing='0' cellpadding='1'>{''.join(rows)}</table>"
    )
