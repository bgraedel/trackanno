"""The annotation screen: one clip at a time, a keypress labels it and moves on.

Every label is written to the CSV at once, so a session resumes after a crash.
"""

from __future__ import annotations

import json
import time

from qtpy.QtCore import QEvent, Qt, QTimer, Signal
from qtpy.QtGui import QKeySequence

try:  # QtGui on Qt6, QtWidgets on Qt5
    from qtpy.QtGui import QShortcut
except ImportError:  # pragma: no cover
    from qtpy.QtWidgets import QShortcut

from qtpy.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QProgressBar, QPushButton, QSizePolicy, QSplitter, QVBoxLayout, QWidget,
)

from ..annotate.clips import ClipStore
from ..config import FIXED_SETTINGS, Config
from ..infer.agreement import differs, load_cv_predictions
from ..view.overlay import FOLLOW_LABELS, FOLLOW_MODES, build_clip_view
from ..annotate.store import AnnotationStore
from ..learn.suggest import Suggester
from .contrast import ContrastBar
from .panels import CategoryPanel, MetadataPanel
from .prefetch import Prefetcher
from .theme import compact_button
from .kymograph import KymographView
from .viewer import SimpleViewer, follow_combo
from .viewprefs import load_view, save_view

STATE_FILE = ".gui_state.json"

ZOOM_STEPS = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0)

AUTOPLAY = FIXED_SETTINGS["ui.autoplay"]
LOOP = FIXED_SETTINGS["ui.loop"]
PREFETCH = FIXED_SETTINGS["ui.prefetch"]                # clips read ahead
RESUME = FIXED_SETTINGS["ui.resume"]                    # or "last_position"
SHUFFLE_SEED = FIXED_SETTINGS["ui.shuffle_seed"]
SHOW_KYMOGRAPH = FIXED_SETTINGS["ui.show_kymograph"]
KYMO_SMOOTH_PX = FIXED_SETTINGS["ui.kymo_smooth_px"]    # centreline fit tolerance
SELECT_BATCH_SIZE = FIXED_SETTINGS["suggest.select_batch_size"]
AUDIT_VISIBLE = FIXED_SETTINGS["suggest.audit_visible"]  # say when a clip is held out

# Shown in review mode in place of a suggestion.
REVIEW_SUGGESTION_NOTE = "no suggestion here: this model was fitted on this label"


class AnnotationWidget(QWidget):
    """The annotation screen as a plain widget; status goes out via ``status_message``."""

    status_message = Signal(str, int)
    counts_changed = Signal(int, int)      # labelled, total in the current filter
    model_refreshed = Signal()             # a new model is in place
    suggester_built = Signal(object, int)  # retrained suggester, its generation
    selection_warmed = Signal(object)

    def __init__(self, cfg: Config, store: ClipStore, annotations: AnnotationStore, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.store = store
        self.annotations = annotations
        # Needed for WidgetWithChildrenShortcut, which keeps keys off other pages.
        self.setFocusPolicy(Qt.StrongFocus)

        # "annotate" walks unlabelled clips; "review" walks labelled ones in label order.
        self.mode = "annotate"
        # Cross-validation predictions per labelled clip.
        self._answers: dict[str, dict] = {}
        self._agreement_problem = ""

        view = load_view()
        self.show_overlay = view["show_overlay"]
        self.show_context = view["show_context_tracks"]
        self.show_skeleton = view["show_skeleton"]
        self.show_contrast = view["show_contrast"]
        self.follow = view["follow"] if view["follow"] in FOLLOW_MODES else "centred"
        self.zoom = float(min(max(view["view_zoom"], ZOOM_STEPS[0]), ZOOM_STEPS[-1]))
        self._saved_contrast = (view["contrast"], view["gamma"])
        self._loaded_at = time.time()
        self._current_values: dict[str, str] = {}
        self._suppress_advance = False
        # A commit warning shown after the next _load().
        self._pending_message: str | None = None
        self._commit_refused = False
        self._clip = None
        self._boxes = None
        self._current_suggestion: dict = {}
        self._suggestion_note = ""
        self._labels_since_train = 0
        self._retrain_thread = None
        # Bumped on reload; retrains from older generations are dropped.
        self._suggester_generation = 0
        self.suggester = Suggester(cfg) if cfg.suggest.enabled else None
        # Selector records by clip_id.
        self._served: dict[str, dict] = {}

        self.order: list[int] = list(range(len(store)))
        self.position = 0
        # Plain list: a pandas lookup per keypress is slow.
        self._clip_ids: list[str] = store.manifest["clip_id"].astype(str).tolist()

        self._shuffled_index: list[int] = list(store.manifest.index)
        if cfg.ui.shuffle:
            # Seeded shuffle against order effects.
            import random

            random.Random(SHUFFLE_SEED).shuffle(self._shuffled_index)

        self._build_ui()
        self._bind_shortcuts()
        self._warm_spline_import()
        self.model_refreshed.connect(self._refresh_selection)
        self.suggester_built.connect(self._adopt_suggester)
        self.selection_warmed.connect(self._on_selection_warmed)

        self.prefetcher = Prefetcher(store, PREFETCH)
        self._warming = self._warm_selection()
        self._apply_filters(initial=True)
        if AUTOPLAY:
            self.viewer.play()

    # ------------------------------------------------------------------ ui -- #
    def _status(self, message: str, timeout: int = 0) -> None:
        self.status_message.emit(message, timeout)

    def _build_ui(self) -> None:
        self.viewer = SimpleViewer(fps=self.cfg.ui.fps, loop=LOOP)
        if hasattr(self.viewer, "set_edges"):
            self.viewer.set_edges(self.cfg.keypoints.edge_idx if self.cfg.is_pose else [])

        # -- top bar --
        self.movie_combo = QComboBox()
        self.movie_combo.addItem("all movies", None)
        for movie_id in self.store.movie_ids:
            self.movie_combo.addItem(movie_id, movie_id)
        self.movie_combo.setFocusPolicy(Qt.NoFocus)
        # Elide long movie ids.
        self.movie_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.movie_combo.setMinimumContentsLength(12)
        self.movie_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.movie_combo.currentIndexChanged.connect(lambda _: self._apply_filters())

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("annotate", "annotate")
        self.mode_combo.addItem("review labelled", "review")
        self.mode_combo.setFocusPolicy(Qt.NoFocus)
        self.mode_combo.setToolTip(
            "Annotate unlabelled clips, or review labelled ones (Tab jumps to a disagreement)."
        )
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)

        self.overlay_box = QCheckBox("marker (o)")
        self.overlay_box.setToolTip("Outline the animal this clip follows.")
        self.overlay_box.setChecked(self.show_overlay)
        self.overlay_box.setFocusPolicy(Qt.NoFocus)
        self.overlay_box.stateChanged.connect(self._on_overlay_toggled)

        self.context_box = QCheckBox("neighbours")
        self.context_box.setToolTip("Also outline other tracks in the crop.")
        self.context_box.setChecked(self.show_context)
        self.context_box.setFocusPolicy(Qt.NoFocus)
        self.context_box.stateChanged.connect(self._on_context_toggled)

        self.skeleton_box = QCheckBox("keypoints")
        self.skeleton_box.setToolTip("Draw the pose keypoints on the marker.")
        self.skeleton_box.setChecked(self.show_skeleton)
        self.skeleton_box.setFocusPolicy(Qt.NoFocus)
        self.skeleton_box.setVisible(self.cfg.is_pose)
        self.skeleton_box.stateChanged.connect(self._on_skeleton_toggled)

        self.contrast_box = QCheckBox("contrast")
        self.contrast_box.setToolTip(
            "Show the contrast bar under the clip; hidden, its settings still apply.")
        self.contrast_box.setChecked(self.show_contrast)
        self.contrast_box.setFocusPolicy(Qt.NoFocus)
        self.contrast_box.stateChanged.connect(self._on_contrast_toggled)

        self.follow_combo = follow_combo(self.follow)
        self.follow_combo.currentIndexChanged.connect(self._on_follow_changed)

        self.zoom_label = QLabel("1.0x")
        self.zoom_label.setMinimumWidth(38)
        self.zoom_label.setAlignment(Qt.AlignCenter)
        self.zoom_label.setToolTip("Zoom into the crop (+ / - / 0).")

        help_btn = compact_button("?", 28, "keyboard shortcuts")
        help_btn.clicked.connect(self._show_help)

        top = QHBoxLayout()
        top.addWidget(QLabel("mode"))
        top.addWidget(self.mode_combo)
        top.addSpacing(12)
        top.addWidget(QLabel("movie"))
        top.addWidget(self.movie_combo, 1)
        top.addSpacing(12)
        top.addWidget(self.overlay_box)
        top.addWidget(self.context_box)
        top.addWidget(self.skeleton_box)
        top.addWidget(self.contrast_box)
        top.addSpacing(12)
        top.addWidget(QLabel("view (s)"))
        top.addWidget(self.follow_combo)
        top.addWidget(self.zoom_label)
        top.addStretch(1)
        top.addWidget(help_btn)

        # -- side panel --
        self.categories = CategoryPanel(self.cfg)
        self.categories.changed.connect(self._on_category_changed)

        self.uncertain_box = QCheckBox("uncertain (u)")
        self.uncertain_box.setFocusPolicy(Qt.NoFocus)

        # A focused line edit eats shortcut keys: "n" enters, Esc/Enter leave.
        self.notes = QLineEdit()
        self.notes.setPlaceholderText("notes (n to type, Esc when done)")
        self.notes.installEventFilter(self)
        self.notes.returnPressed.connect(self._leave_notes)

        self.metadata = MetadataPanel()

        # Review mode only: who labelled the clip, and any model disagreement.
        self.review_note = QLabel("")
        self.review_note.setWordWrap(True)
        self.review_note.setTextFormat(Qt.RichText)
        self.review_note.setProperty("role", "muted")
        self.review_note.setVisible(False)

        source_title = QLabel("source")
        source_title.setStyleSheet("font-weight:600;")

        side_layout = QVBoxLayout()
        side_layout.setContentsMargins(10, 0, 4, 0)
        side_layout.addWidget(self.categories)
        side_layout.addWidget(self.uncertain_box)
        side_layout.addWidget(self.notes)
        side_layout.addSpacing(8)
        side_layout.addWidget(source_title)
        side_layout.addWidget(self.metadata)
        side_layout.addWidget(self.review_note)
        side = QWidget()
        side.setLayout(side_layout)
        side.setMinimumWidth(280)

        self.contrast = ContrastBar(*self._saved_contrast)
        self._saved_contrast = (self.contrast.mode, round(self.contrast.gamma, 3))
        self.contrast.changed.connect(self._on_contrast_changed)
        self.contrast.setVisible(self.show_contrast)
        self.viewer.frame_changed.connect(self.contrast.set_frame)
        # Remembers the contrast mode and gamma once a drag settles.
        self._contrast_timer = QTimer(self)
        self._contrast_timer.setSingleShot(True)
        self._contrast_timer.setInterval(500)
        self._contrast_timer.timeout.connect(self._save_contrast)

        clip_pane = QWidget()
        clip_layout = QVBoxLayout(clip_pane)
        clip_layout.setContentsMargins(0, 0, 0, 0)
        clip_layout.addWidget(self.viewer, 1)
        clip_layout.addWidget(self.contrast)

        self.kymo_view = KymographView()
        self.kymo_view.frame_picked.connect(self._scrub_to)
        self.kymo_view.set_sign_mode(self.cfg.ui.kymo_sign)
        self.kymo_view.setVisible(self._kymograph_wanted())
        if hasattr(self.viewer, "frame_changed"):
            self.viewer.frame_changed.connect(self.kymo_view.set_frame)

        left = QSplitter(Qt.Vertical)
        left.addWidget(clip_pane)
        left.addWidget(self.kymo_view)
        left.setStretchFactor(0, 3)
        left.setStretchFactor(1, 1)
        left.setSizes([560, 210])
        left.setCollapsible(0, False)
        self.kymo_splitter = left

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(side)
        splitter.setStretchFactor(0, 1)
        splitter.setSizes([820, 300])

        # -- bottom bar --
        self.prev_btn = QPushButton("< prev")
        self.prev_btn.setFocusPolicy(Qt.NoFocus)
        self.prev_btn.clicked.connect(self.previous_clip)
        self.next_btn = QPushButton("next >")
        self.next_btn.setFocusPolicy(Qt.NoFocus)
        self.next_btn.clicked.connect(self.next_clip)

        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.counter = QLabel("-")
        self.counter.setMinimumWidth(220)
        self.counter.setAlignment(Qt.AlignCenter)

        bottom = QHBoxLayout()
        bottom.addWidget(self.prev_btn)
        bottom.addWidget(self.progress, 1)
        bottom.addWidget(self.counter)
        bottom.addWidget(self.next_btn)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(10, 8, 10, 8)
        root_layout.addLayout(top)
        root_layout.addWidget(splitter, 1)
        root_layout.addLayout(bottom)

        self._sync_zoom_label()

    def _bind_shortcuts(self) -> None:
        def bind(sequence: str, handler):
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(handler)

        keys = self.cfg.keys
        for seq in keys["next"]:
            bind(seq, self.next_clip)
        for seq in keys["prev"]:
            bind(seq, self.previous_clip)
        for seq in keys["next_unannotated"]:
            bind(seq, self._on_next_gap)
        for seq in keys["play_pause"]:
            bind(seq, self.viewer.toggle_play)
        for seq in keys["overlay"]:
            bind(seq, lambda: self.overlay_box.setChecked(not self.overlay_box.isChecked()))
        for seq in keys["stabilise"]:
            bind(seq, self.cycle_follow)
        for seq in keys["context"]:
            bind(seq, lambda: self.context_box.setChecked(not self.context_box.isChecked()))
        for seq in keys["zoom_in"]:
            bind(seq, lambda: self.step_zoom(+1))
        for seq in keys["zoom_out"]:
            bind(seq, lambda: self.step_zoom(-1))
        for seq in keys["zoom_reset"]:
            bind(seq, lambda: self.set_zoom(1.0))
        for seq in keys["uncertain"]:
            bind(seq, lambda: self.uncertain_box.setChecked(not self.uncertain_box.isChecked()))
        for seq in keys["clear"]:
            bind(seq, self._clear_current)
        for seq in keys["undo"]:
            bind(seq, self._undo)
        for seq in keys["notes"]:
            bind(seq, self._focus_notes)

        for group in self.cfg.categories:
            for option in group.options:
                bind(
                    option.key,
                    lambda g=group.name, o=option.name: self.categories.toggle(g, o),
                )

    # -------------------------------------------------------------- filters -- #
    def _apply_filters(self, initial: bool = False) -> None:
        current_id = None
        if not initial and self.order:
            # Commit first so a half-typed note is kept.
            self._commit()
            current_id = self._clip_ids[self.order[self.position]]

        manifest = self.store.manifest
        mask = manifest["clip_id"].notna()
        movie_id = self.movie_combo.currentData()
        if movie_id:
            mask &= manifest["movie_id"] == movie_id
        if self.reviewing:
            mask &= manifest["clip_id"].isin(self.annotations.done_ids)

        # Filter the once-shuffled manifest, so the order is stable across filters.
        selected = set(manifest.index[mask].tolist())
        if self.reviewing:
            # Label order, without shuffle or selector batch.
            self.order = [
                index for index in map(self.store.index_of,
                                       self.annotations.labelled_order)
                if index in selected
            ]
        else:
            self.order = [i for i in self._shuffled_index if i in selected]
        if not self.order:
            self._show_empty()
            return

        index = self.store.index_of(current_id) if current_id is not None else None
        anchor = index if index in selected else None
        if not self.reviewing:
            self.order = self._queue_with_batch(self.order, anchor)

        self.position = 0
        if anchor is not None and anchor in self.order:
            self.position = self.order.index(anchor)
        elif initial and not self.reviewing:
            self.position = self._resume_position()
        self._load()

    def _resume_position(self) -> int:
        if RESUME == "last_position":
            state_path = self.cfg.paths.work_dir / STATE_FILE
            if state_path.exists():
                try:
                    clip_id = json.loads(state_path.read_text(encoding="utf-8")).get("clip_id")
                    index = self.store.index_of(clip_id) if clip_id else None
                    if index in self.order:
                        return self.order.index(index)
                except Exception:
                    pass
        done = self.annotations.done_ids
        for pos, index in enumerate(self.order):
            if self._clip_ids[index] not in done:
                return pos
        return 0

    def _empty_reason(self) -> str:
        """Why the current filter shows nothing."""
        if not self.reviewing:
            return "no clips match the current filter"
        if self.annotations.done_ids:
            return "nothing labelled in the current filter"
        return "nothing labelled here yet: switch back to annotate"

    def _show_empty(self) -> None:
        self.counter.setText(self._empty_reason())
        self.progress.setValue(0)
        self.kymo_view.clear()

    @property
    def reviewing(self) -> bool:
        """True in review mode."""
        return self.mode == "review"

    def _refresh_fold_answers(self) -> None:
        """Load held-out fold predictions from ``cv_predictions.parquet`` (not the saved model)."""
        self._answers, self._agreement_problem = {}, ""
        if not self.annotations.labelled_order:
            return
        try:
            self._answers, self._agreement_problem = load_cv_predictions(self.cfg)
        except Exception as exc:                     # noqa: BLE001
            self._answers = {}
            self._agreement_problem = f"the fold predictions could not be read ({exc})"

    def _on_mode_changed(self, _index: int) -> None:
        self.mode = self.mode_combo.currentData() or "annotate"
        self.review_note.setVisible(self.reviewing)
        if self.reviewing:
            self._refresh_fold_answers()
        self._apply_filters()
        self._status({
            "annotate": "annotating the clips still to do",
            "review": "reviewing labels, oldest first; Tab jumps to a disagreement",
        }[self.mode], 10000)

    # ------------------------------------------------------------ selection -- #
    def _queue_with_batch(self, order: list[int], anchor: int | None) -> list[int]:
        """``order`` with the selector's picks moved to just after ``anchor`` (or the front).

        ``anchor`` and unpicked clips keep their relative order.
        """
        batch = self._fresh_batch(order)
        if not batch:
            return order

        index_of = {self._clip_ids[i]: i for i in order}
        picked: list[int] = []
        for record in batch:
            index = index_of.get(str(record["clip_id"]))
            if index is None or index == anchor or index in picked:
                continue
            picked.append(index)
            self._served[self._clip_ids[index]] = record
        if not picked:
            return order
        taken = set(picked)

        rest = [i for i in order if i not in taken]
        splice = rest.index(anchor) + 1 if anchor is not None else 0
        return rest[:splice] + picked + rest[splice:]

    def _fresh_batch(self, order: list[int]) -> list[dict]:
        """The selector's next picks from the filtered clips, or ``[]``."""
        if self.suggester is None or not order or self._warming:
            return []
        try:
            # `order` holds positions, hence .iloc.
            batch = list(self.suggester.selection_batch(
                self.store.manifest.iloc[order],
                self.annotations.done_ids,
                int(SELECT_BATCH_SIZE),
            ))
        except Exception as exc:
            # Fall back to shuffle order.
            message = f"clip selection is off: {type(exc).__name__}: {exc}"
            print(f"[trackanno] {message}")
            self._status(message, 10000)
            return []
        return batch

    def _warm_selection(self) -> bool:
        """Score the selection pool off the UI thread. True if a worker started."""
        suggester, cfg = self.suggester, self.cfg
        if (suggester is None or not hasattr(suggester, "warm")
                or not (cfg.suggest.enabled and cfg.suggest.selection_on)
                or not getattr(suggester, "ready", False)):
            return False
        import threading

        def work():
            try:
                suggester.warm()
            except Exception as exc:
                print(f"[trackanno] scoring the selection pool failed: {exc}")
            try:
                self.selection_warmed.emit(suggester)
            except RuntimeError:          # the widget is gone
                pass

        threading.Thread(target=work, name="trackanno-selection-warm",
                         daemon=True).start()
        return True

    def _on_selection_warmed(self, _suggester) -> None:
        self._warming = False
        self._refresh_selection()

    def _refresh_selection(self) -> None:
        """Re-queue the clips after the current one when a new model arrives."""
        if not self.order:
            return
        if self.reviewing:
            # Re-read predictions; leave the review order alone.
            self._refresh_fold_answers()
            return
        anchor = self.order[self.position]
        before = self.order
        self.order = self._queue_with_batch(self.order, anchor)
        if self.order == before:
            return
        self.position = self.order.index(anchor)
        # Re-aim prefetch at the new queue.
        self.prefetcher.around(self.order, self.position)
        self._status("model updated: the queue ahead was re-selected", 6000)

    # ----------------------------------------------------------- clip load -- #
    def _build_view(self, n_frames: int, size: int):
        return build_clip_view(
            self._boxes, n_frames, size,
            mode=self.follow,
            zoom=self.zoom,
            with_context=self.show_context,
            n_kpts=self.cfg.n_keypoints if self.show_skeleton else 0,
        )

    def _load(self) -> None:
        if not self.order:
            return
        started = time.perf_counter()
        index = self.order[self.position]
        meta = self.store.meta(index)

        self._clip = self.store.clip(index)
        self._boxes = self.store.clip_boxes(index)
        view = self._build_view(self._clip.shape[0], self._clip.shape[1])

        self.viewer.set_overlay(
            self.show_overlay, self.show_context, self.show_skeleton
        )
        self.viewer.set_clip(self._clip, view)
        self.contrast.set_clip(
            self._clip, self.store.header(str(meta["movie_id"])).get("intensity"))
        if AUTOPLAY and not self.viewer.is_playing:
            self.viewer.play()

        self.metadata.update_from(meta, self.store.origins(index))
        self._update_kymograph(self._boxes, meta)

        clip_id = str(meta["clip_id"])
        self._resuggest(clip_id)

        existing = self.annotations.get(clip_id)
        self._suppress_advance = True
        self.categories.set_values(existing)
        # After set_values, which repaints the buttons.
        self._paint_suggestion()
        self.uncertain_box.setChecked(bool(existing and existing.get("uncertain") == "1"))
        self.notes.setText((existing or {}).get("notes", ""))
        self._suppress_advance = False
        self._current_values = self.categories.values()

        self._paint_review_note(clip_id, existing)

        self._loaded_at = time.time()
        self._update_progress(index, meta, (time.perf_counter() - started) * 1000)
        self.prefetcher.around(self.order, self.position)
        # The resume point tracks annotate mode only.
        if not self.reviewing:
            self._save_state(str(meta["clip_id"]))

    def _resuggest(self, clip_id: str) -> None:
        """Set the suggestion and its note for ``clip_id``."""
        self._current_suggestion = (
            self.suggester.suggest(clip_id) if self.suggester and self.suggester.ready else {}
        )
        self._suggestion_note = self._why_no_suggestion(clip_id)

    def _paint_suggestion(self) -> None:
        """Show the suggestion; in review mode show only the review note."""
        if self.reviewing:
            self.categories.set_suggestion({}, REVIEW_SUGGESTION_NOTE)
        else:
            self.categories.set_suggestion(self._current_suggestion, self._suggestion_note)

    def _disagreement(self, clip_id: str, suggestion: dict | None = None) -> list[str]:
        """Groups where the label differs from the fold prediction ([] if unscored)."""
        row = self.annotations.get(clip_id)
        if row is None:
            return []
        if suggestion is None:
            suggestion = self._answers.get(clip_id) or {}
        return differs(row, suggestion)

    def _paint_review_note(self, clip_id: str, existing: dict | None) -> None:
        """Show who labelled the clip and where the fold prediction differs."""
        if not self.reviewing:
            return
        if not existing:
            self.review_note.setText("no label recorded for this clip")
            return
        who = existing.get("annotator") or "an unrecorded annotator"
        when = (existing.get("timestamp_utc") or "")[:16].replace("T", " ")
        line = f"labelled by {who}" + (f" on {when} UTC" if when else "")
        if existing.get("uncertain") == "1":
            line += ", marked uncertain"

        folds = self._answers.get(clip_id) or {}
        disagrees = {entry.split(":", 1)[0] for entry in differs(existing, folds)}
        if disagrees:
            body = ", ".join(
                f"{group} <b>{value}</b>"
                + ("" if confidence is None else f" ({confidence:.0%} sure)")
                for group, (value, confidence) in folds.items()
                if group in disagrees
            )
            line += (f"<br><b>held out, the model said</b> {body}"
                     "<br>Check the clip to see which is right.")
        elif folds:
            line += "<br>held out, the model agreed with this label"
        else:
            line += "<br><i>no held-out prediction for this clip</i>"
        self.review_note.setText(line)

    def _refresh_view(self) -> None:
        """Re-centre / re-zoom / re-mark without re-reading the clip."""
        if self._clip is None:
            return
        view = self._build_view(self._clip.shape[0], self._clip.shape[1])
        self.viewer.set_overlay(
            self.show_overlay, self.show_context, self.show_skeleton
        )
        self.viewer.set_view(view)

    def _update_progress(self, index: int, meta, load_ms: float) -> None:
        done = self.annotations.done_ids
        done_here = sum(1 for i in self.order if self._clip_ids[i] in done)
        total = len(self.order)
        self.progress.setRange(0, max(1, total))
        if self.reviewing:
            # The bar shows review position.
            self.progress.setValue(self.position + 1)
            self.counter.setText(f"reviewing {self.position + 1} / {total} labelled")
        else:
            mark = "OK" if str(meta["clip_id"]) in done else "--"
            self.progress.setValue(done_here)
            self.counter.setText(
                f"[{mark}]  clip {self.position + 1} / {total}   |   {done_here} labelled "
                f"({done_here / max(1, total):.0%})"
            )
        self.counts_changed.emit(done_here, total)
        if self._pending_message:
            self._status(self._pending_message, 6000)
            self._pending_message = None
        else:
            self._status(
                f"load {load_ms:.1f} ms   |   "
                f"{self.annotations.path.name}: {len(self.annotations)} rows"
            )

    def _save_state(self, clip_id: str) -> None:
        try:
            (self.cfg.paths.work_dir / STATE_FILE).write_text(
                json.dumps({"clip_id": clip_id}), encoding="utf-8"
            )
        except OSError:
            pass

    # ------------------------------------------------------------ labelling -- #
    def _on_category_changed(self) -> None:
        if self._suppress_advance:
            return
        values = self.categories.values()
        changed_single = [
            g.name for g in self.cfg.categories
            if not g.is_multi and values.get(g.name) != self._current_values.get(g.name)
        ]
        self._current_values = values

        # Review mode never auto-advances; repaint the note against the buttons.
        if self.reviewing:
            clip_id = self._clip_ids[self.order[self.position]]
            self._paint_review_note(
                clip_id, {**(self.annotations.get(clip_id) or {}), **values}
            )
            return
        if not self.cfg.ui.auto_advance or not changed_single:
            return
        # Only a single-choice pick advances.
        if self.annotations.missing_required(values):
            return
        self._commit()
        self.next_clip()

    def _on_screen_matches(self, existing: dict | None) -> bool:
        """True if the buttons, uncertain box and notes equal ``existing``."""
        values = self.categories.values()
        uncertain = self.uncertain_box.isChecked()
        notes = self.notes.text().strip()
        if existing is None:
            return not (any(values.values()) or uncertain or notes)
        return (
            all(existing.get(g.name, "") == values.get(g.name, "") for g in self.cfg.categories)
            and existing.get("uncertain", "0") == ("1" if uncertain else "0")
            and existing.get("notes", "") == notes
        )

    def _commit(self) -> bool:
        """Write the clip on screen. ``_commit_refused`` says whether input was kept back."""
        self._commit_refused = False
        if not self.order:
            return False
        meta = self.store.meta(self.order[self.position])
        values = self.categories.values()
        uncertain = self.uncertain_box.isChecked()
        notes = self.notes.text().strip()
        has_any = any(values.values()) or uncertain or notes
        existing = self.annotations.get(str(meta["clip_id"]))
        if not has_any:
            # Deselecting everything un-labels the clip.
            if existing is not None:
                try:
                    self.annotations.clear(str(meta["clip_id"]))
                except OSError as exc:
                    self._pending_message = self._write_failed(exc)
                    self._commit_refused = True
                    return False
                self._pending_message = "annotation cleared"
                return True
            return False

        # Unchanged: skip the rewrite and the no-op undo entry.
        if existing is not None and self._on_screen_matches(existing):
            return False

        missing = self.annotations.missing_required(values)
        if missing and not self.uncertain_box.isChecked():
            self._pending_message = (
                f"not saved: {', '.join(missing)} still unset "
                f"(press u to mark uncertain instead)"
            )
            self._commit_refused = True
            return False

        # Record the shown suggestion, except on a revision.
        revising = self.annotations.get(str(meta["clip_id"])) is not None
        primary = None if revising else next(
            ((g.name, self._current_suggestion[g.name])
             for g in self.cfg.categories if g.name in self._current_suggestion),
            None,
        )
        try:
            self.annotations.record(
                meta, values, uncertain=uncertain, notes=notes,
                dwell_s=time.time() - self._loaded_at,
                suggestion=primary[1] if primary else None,
                suggestion_shown=bool(primary),
                selected_by=self._served.get(str(meta["clip_id"])),
            )
        except OSError as exc:
            # Report instead of raising out of a Qt slot.
            self._pending_message = self._write_failed(exc)
            self._commit_refused = True
            return False
        self._labels_since_train += 1
        self._maybe_retrain()
        return True

    def _maybe_retrain(self) -> None:
        """Refit in the background once enough new labels have accumulated."""
        if self.suggester is None or not self.cfg.suggest.enabled:
            return
        if self._labels_since_train < self.cfg.suggest.retrain_every:
            return
        # One retrain at a time.
        if self._retrain_thread is not None and self._retrain_thread.is_alive():
            return
        self._labels_since_train = 0

        import threading

        live = self.cfg
        generation = self._suggester_generation

        def work():
            try:
                from copy import copy

                from ..learn.suggest import train

                # Train on a snapshot so the live CSV stays writable on Windows.
                snapshot = live.paths.work_dir / ".annotations_snapshot.csv"
                self.annotations.snapshot(snapshot)
                cfg = copy(live)
                cfg.paths = copy(live.paths)
                cfg.paths.csv = snapshot
                train(cfg, verbose=False)
                built = Suggester(live, labels_csv=snapshot)
                built.warm()
                self.suggester_built.emit(built, generation)
            except Exception as exc:
                print(f"[trackanno] background retrain failed: {exc}")

        self._retrain_thread = threading.Thread(
            target=work, name="trackanno-retrain", daemon=True
        )
        self._retrain_thread.start()

    # ---------------------------------------------------------- kymograph -- #
    def _kymo_cutoff_px(self, meta) -> float | None:
        """A centerline's smoothing distance in cache pixels, or None for keypoints."""
        if not self.cfg.centerline.enabled or meta is None:
            return None
        from ..measure.instantaneous import SMOOTHING_UM

        try:
            movie = self.store.header(str(meta["movie_id"])).get("movie", {})
            um_per_px = float(movie.get("pixel_size_um") or self.cfg.tiff.pixel_size_um or 0)
        except (KeyError, TypeError, ValueError):
            um_per_px = 0.0
        if um_per_px <= 0:
            return None
        return SMOOTHING_UM / um_per_px * float(meta.get("scale", 1.0) or 1.0)

    def _kymograph_wanted(self) -> bool:
        """Whether to show the kymograph (pose projects with 2+ keypoints)."""
        return bool(SHOW_KYMOGRAPH and self.cfg.is_pose
                    and self.cfg.n_keypoints >= 2)

    def _why_no_suggestion(self, clip_id: str) -> str:
        """Why there is no suggestion, or "" (silent unless AUDIT_VISIBLE)."""
        if self._current_suggestion or self.suggester is None:
            return ""
        if not AUDIT_VISIBLE:
            return ""
        if self.suggester.is_audit_clip(clip_id):
            return "held out for audit: no suggestion on this clip"
        if not self.suggester.ready:
            return self.suggester.status_line()
        return "the model has nothing to say about this clip"

    def apply_view_settings(self, cfg) -> None:
        """Adopt a reloaded config without rebuilding the panel."""
        if cfg != self.cfg:
            self._suggester_generation += 1
        self.cfg = cfg
        self.kymo_view.set_sign_mode(cfg.ui.kymo_sign)

    def _warm_spline_import(self) -> None:
        """Import scipy off the UI thread ahead of the first kymograph."""
        if not self._kymograph_wanted():
            return
        import threading

        def work():
            try:
                import scipy.interpolate  # noqa: F401
            except Exception:
                pass          # surfaces at first use

        threading.Thread(target=work, name="trackanno-spline-warm",
                         daemon=True).start()

    def _update_kymograph(self, boxes, meta=None) -> None:
        """Recompute the kymograph for the clip on screen."""
        if not self._kymograph_wanted():
            return
        from ..measure.kymograph import chain_from_edges, kymograph_from_boxes

        # The skeleton's chain order sets the sign of the chart; node 0 is the Head, not flagellum.
        order = chain_from_edges(self.cfg.n_keypoints, self.cfg.keypoints.edge_idx)
        order = [i for i in order if i != 0]
        try:
            kymo = kymograph_from_boxes(
                boxes, self.cfg.n_keypoints, int(self._clip.shape[0]),
                min_conf=0.0, n_samples=128, order=order,
                smooth_px=KYMO_SMOOTH_PX, cutoff_px=self._kymo_cutoff_px(meta),
            )
        except Exception as exc:                 # never take the clip down
            print(f"[trackanno] kymograph failed: {exc}")
            self.kymo_view.set_kymograph(None)
            return
        self.kymo_view.set_kymograph(kymo)
        self.kymo_view.set_frame(0)

    def _scrub_to(self, frame: int) -> None:
        """Clicking the kymograph moves the clip to that frame."""
        viewer = self.viewer
        if hasattr(viewer, "pause"):
            viewer.pause()
        if hasattr(viewer, "seek"):
            viewer.seek(int(frame))

    def reload_suggester(self) -> str:
        """Load a newly trained model in place, keeping the current clip."""
        self._suggester_generation += 1
        if not self.cfg.suggest.enabled:
            self.suggester = None
            return "suggestions are switched off in the config"
        self._install_suggester(Suggester(self.cfg))
        return self.suggester.status_line()

    def _adopt_suggester(self, built, generation: int) -> None:
        """Swap in a retrained Suggester unless a reload happened since."""
        if generation != self._suggester_generation:
            return
        if self.suggester is None or not self.cfg.suggest.enabled:
            return
        self._install_suggester(built)

    def _install_suggester(self, suggester) -> None:
        """Make ``suggester`` current, refresh the clip on screen and the queue."""
        previous = self.suggester
        if previous is not None and hasattr(suggester, "model_epoch"):
            suggester.model_epoch = max(suggester.model_epoch,
                                        getattr(previous, "model_epoch", 0) + 1)
        self.suggester = suggester
        if self.order:
            self._resuggest(self._clip_ids[self.order[self.position]])
            self._paint_suggestion()
        self.model_refreshed.emit()

    def _clear_current(self) -> None:
        if not self.order:
            return
        meta = self.store.meta(self.order[self.position])
        self._suppress_advance = True
        self.categories.clear()
        self.uncertain_box.setChecked(False)
        self.notes.clear()
        self._suppress_advance = False
        self._current_values = self.categories.values()
        try:
            self.annotations.clear(str(meta["clip_id"]))
        except OSError as exc:
            self._pending_message = self._write_failed(exc)
        self._update_progress(self.order[self.position], meta, 0.0)

    def _write_failed(self, exc: OSError) -> str:
        """A one-line write error with a likely cause."""
        hint = ""
        if getattr(exc, "errno", None) in (13, 30) or "Permission" in type(exc).__name__:
            hint = " (is it open in Excel, or is the folder read-only?)"
        return f"could not write {self.annotations.path.name}: {exc}{hint}"

    def _undo(self) -> None:
        """Undo the last commit; in review, first revert an uncommitted pick."""
        if self.reviewing and self.order:
            clip_id = self._clip_ids[self.order[self.position]]
            existing = self.annotations.get(clip_id)
            if not self._on_screen_matches(existing):
                self._suppress_advance = True
                self.categories.set_values(existing)
                self._paint_suggestion()
                self.uncertain_box.setChecked(bool(existing and existing.get("uncertain") == "1"))
                self.notes.setText((existing or {}).get("notes", ""))
                self._suppress_advance = False
                self._current_values = self.categories.values()
                self._paint_review_note(clip_id, existing)
                self._status("reverted the change on screen", 3000)
                return
        try:
            clip_id = self.annotations.undo()
        except OSError as exc:
            self._status(self._write_failed(exc), 10000)
            return
        if clip_id is None:
            self._status("nothing to undo", 2000)
            return
        index = self.store.index_of(clip_id)
        if index is not None and index in self.order:
            self.position = self.order.index(index)
        self._load()
        self._status(f"undid {clip_id}", 3000)

    # ----------------------------------------------------------- navigation -- #
    def next_clip(self) -> None:
        self._commit()
        if self.position + 1 >= len(self.order):
            if self._commit_refused:
                # Do not reload: it would wipe the kept input.
                self._status(self._pending_message or "not saved", 6000)
                self._pending_message = None
                return
            self._status("last clip in the current filter", 3000)
            self._load()
            return
        self.position += 1
        self._load()

    def previous_clip(self) -> None:
        self._commit()
        if self.position == 0:
            return
        self.position -= 1
        self._load()

    def _on_next_gap(self) -> None:
        """Tab: next unlabelled clip, or next disagreement in review mode."""
        if self.reviewing:
            self.next_disagreement()
        else:
            self.next_unannotated()

    def next_disagreement(self) -> None:
        """Jump to the next labelled clip the fold prediction disagrees with."""
        self._commit()
        if not self._answers:
            self._status(
                self._agreement_problem
                or "no held-out predictions yet: train the model on the Model page",
                6000)
            return
        for pos in range(self.position + 1, len(self.order)):
            if self._disagreement(self._clip_ids[self.order[pos]]):
                self.position = pos
                self._load()
                return
        self._status(
            "no disagreements after this clip", 4000)

    def next_unannotated(self) -> None:
        self._commit()
        done = self.annotations.done_ids
        for pos in range(self.position + 1, len(self.order)):
            if self._clip_ids[self.order[pos]] not in done:
                self.position = pos
                self._load()
                return
        self._status("no unannotated clips after this one", 3000)

    # -------------------------------------------------------------- toggles -- #
    def _on_overlay_toggled(self, state) -> None:
        self.show_overlay = bool(self.overlay_box.isChecked())
        save_view(show_overlay=self.show_overlay)
        self._refresh_view()

    def _on_context_toggled(self, state) -> None:
        self.show_context = bool(self.context_box.isChecked())
        save_view(show_context_tracks=self.show_context)
        self._refresh_view()

    def _on_skeleton_toggled(self, state) -> None:
        self.show_skeleton = bool(self.skeleton_box.isChecked())
        save_view(show_skeleton=self.show_skeleton)
        self._refresh_view()

    def _on_contrast_toggled(self, state) -> None:
        self.show_contrast = bool(self.contrast_box.isChecked())
        save_view(show_contrast=self.show_contrast)
        self.contrast.setVisible(self.show_contrast)

    def _on_follow_changed(self, index: int) -> None:
        self.follow = self.follow_combo.currentData() or "centred"
        save_view(follow=self.follow)
        self._refresh_view()
        self._status(FOLLOW_LABELS[self.follow], 6000)

    def cycle_follow(self) -> None:
        nxt = (FOLLOW_MODES.index(self.follow) + 1) % len(FOLLOW_MODES)
        self.follow_combo.setCurrentIndex(nxt)

    def set_zoom(self, zoom: float) -> None:
        self.zoom = float(min(max(zoom, ZOOM_STEPS[0]), ZOOM_STEPS[-1]))
        save_view(view_zoom=self.zoom)
        self._sync_zoom_label()
        self._refresh_view()

    def step_zoom(self, direction: int) -> None:
        steps = list(ZOOM_STEPS)
        current = min(range(len(steps)), key=lambda i: abs(steps[i] - self.zoom))
        self.set_zoom(steps[min(max(current + direction, 0), len(steps) - 1)])

    def _sync_zoom_label(self) -> None:
        self.zoom_label.setText(f"{self.zoom:.2f}x".replace(".00x", ".0x"))

    def _on_contrast_changed(self) -> None:
        self.viewer.set_display(self.contrast.mapper())
        self._contrast_timer.start()

    def _save_contrast(self) -> None:
        current = (self.contrast.mode, round(self.contrast.gamma, 3))
        if current != self._saved_contrast:
            save_view(contrast=current[0], gamma=current[1])
            self._saved_contrast = current

    # ---------------------------------------------------------------- notes -- #
    def eventFilter(self, obj, event):  # noqa: N802 (Qt naming)
        if obj is self.notes and event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Escape:
                self._leave_notes()
                return True
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                # Return is also "next clip" and would pre-empt returnPressed.
                self._leave_notes()
                return True
        return super().eventFilter(obj, event)

    def _focus_notes(self) -> None:
        self.notes.setFocus(Qt.ShortcutFocusReason)
        self.notes.selectAll()
        self._status("typing notes: Esc or Enter to return to shortcuts", 4000)

    def _leave_notes(self) -> None:
        self.notes.clearFocus()
        self.setFocus(Qt.OtherFocusReason)
        self._status("shortcuts active", 1500)

    def _show_help(self) -> None:
        lines = ["<b>Navigation</b><table>"]
        for action, binds in self.cfg.keys.items():
            lines.append(
                f"<tr><td><code>{' / '.join(binds)}</code></td><td>&nbsp;{action}</td></tr>"
            )
        lines.append("</table><br><b>Categories</b><table>")
        for group in self.cfg.categories:
            for option in group.options:
                lines.append(
                    f"<tr><td><code>{option.key}</code></td>"
                    f"<td>&nbsp;{group.name}: {option.name}</td></tr>"
                )
        lines.append("</table>")

        lines.append(
            "<br><b>Modes</b><br>"
            "<i>annotate</i> walks unlabelled clips. <i>review labelled</i> walks "
            "labelled ones, oldest first; a pick rewrites the label and Tab jumps "
            "to the next disagreement."
        )

        # Explained here, not on the clip, so audit clips stay unmarked.
        share = self.cfg.suggest.audit_fraction
        if self.cfg.suggest.enabled and share > 0:
            lines.append(
                "<br><b>Why some clips show no suggestion</b><br>"
                f"About {share:.0%} of clips are an audit holdout: never trained on "
                "and never suggested, so agreement on them is unbiased. Set "
                "<i>suggest.audit_fraction</i> to 0 to turn this off."
            )
        QMessageBox.information(self, "Shortcuts", "".join(lines))

    # ---------------------------------------------------------------- close -- #
    def shutdown(self) -> None:
        # Release the prefetch thread and cache even if the commit fails.
        try:
            self._commit()
        except Exception as exc:
            print(f"[trackanno] could not save the last clip: {exc}")
        finally:
            if not self.prefetcher.stop():
                print("[trackanno] prefetch thread did not stop in time")
        self._contrast_timer.stop()
        self._save_contrast()
        # Drop views on the cache's memory map so Windows can rewrite it.
        self._clip = None
        self._boxes = None
        if hasattr(self.viewer, "clear"):
            self.viewer.clear()
        self.contrast.clear()
