"""The annotation surface.

    uv run annotate.py          # just this screen
    uv run gui.py               # this screen inside the full workflow window

One clip of one track at a time. A keypress labels it and jumps to the next; the
CSV is written and fsynced on every annotation, so a session resumes from any
point, including after a crash.

:class:`AnnotationWidget` is a plain ``QWidget``: the hub embeds it as a page,
:class:`AnnotationWindow` wraps it standalone. It emits ``status_message``
rather than touching a status bar, and lets its host place that.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from qtpy.QtCore import QEvent, Qt, Signal
from qtpy.QtGui import QKeySequence

try:  # Qt6 puts QShortcut in QtGui, Qt5 in QtWidgets; qtpy aliases but be explicit
    from qtpy.QtGui import QShortcut
except ImportError:  # pragma: no cover
    from qtpy.QtWidgets import QShortcut

from qtpy.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QSizePolicy, QSplitter, QVBoxLayout,
    QWidget,
)

from ..annotate.clips import ClipStore
from ..config import Config, load_config
from ..view.overlay import FOLLOW_MODES, build_clip_view
from ..annotate.store import AnnotationStore, encode_provenance
from ..learn.suggest import Suggester
from .panels import CategoryPanel, MetadataPanel, MinimapWidget
from .prefetch import Prefetcher
from .theme import app_icon, apply_theme, compact_button
from .kymograph import KymographView
from .viewer import SimpleViewer

STATE_FILE = ".gui_state.json"

# Short in the combo, long in the tooltip and the status line.
FOLLOW_LABELS = {
    "centred": "centred - one view on the track's median position, all motion visible",
    "smooth": "smooth - the view follows a moving average: drift removed, beat kept",
    "locked": "locked - re-centred every frame: best for shape, hides translation",
}
ZOOM_STEPS = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0)

# The recorded reason for a clip the annotator reached under their own steam: an
# arrow key back through the queue, a movie filter, a jump to the first unlabelled
# clip. Deliberately NOT one of `select.py`'s rules, so the labelled set can be
# split into the model's picks and the human's own detours.
MANUAL_RULE = "annotator_navigated"


class AnnotationWidget(QWidget):
    """Everything on the annotation screen. Hostable anywhere."""

    status_message = Signal(str, int)
    counts_changed = Signal(int, int)      # labelled, total in the current filter
    # Emitted from the background retrain thread when a NEW model is loaded.
    # `order` and every widget under this one belong to the UI thread, so a
    # queued signal is the only safe way to say the ranking has been replaced.
    model_refreshed = Signal()

    def __init__(self, cfg: Config, store: ClipStore, annotations: AnnotationStore, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.store = store
        self.annotations = annotations
        # Shortcuts use WidgetWithChildrenShortcut so a bare "1" cannot be stolen
        # by another page - which only works if this widget can hold focus.
        self.setFocusPolicy(Qt.StrongFocus)

        self.show_overlay = cfg.ui.show_overlay
        self.show_context = cfg.ui.show_context_tracks
        self.follow = cfg.ui.follow
        self.zoom = float(cfg.ui.view_zoom)
        self.show_minimap = cfg.ui.show_minimap
        self._loaded_at = time.time()
        self._current_values: dict[str, str] = {}
        self._suppress_advance = False
        # A warning raised while committing must outlive the _load() that follows.
        self._pending_message: str | None = None
        self._clip = None
        self._boxes = None
        self._current_suggestion: dict = {}
        self._suggestion_note = ""
        self._labels_since_train = 0
        self._retrain_thread = None
        self.suggester = Suggester(cfg) if cfg.suggest.enabled else None
        # The last batch actually spliced into the queue, and the accumulated
        # record of every clip the selector has put in front of this annotator.
        # `_served` is keyed by clip_id and never by queue position: adding a
        # movie renumbers every position and would re-point every reason.
        self._batch: list[dict] = []
        self._served: dict[str, dict] = {}

        self.order: list[int] = list(range(len(store)))
        self.position = 0
        # Plain list of clip ids: a pandas .iloc per clip makes a keypress O(n).
        self._clip_ids: list[str] = store.manifest["clip_id"].astype(str).tolist()

        self._shuffled_index: list[int] = list(store.manifest.index)
        if cfg.ui.shuffle:
            # Movie-by-movie invites order effects; seeded, so it is reproducible.
            import random

            random.Random(cfg.ui.shuffle_seed).shuffle(self._shuffled_index)

        self._build_ui()
        self._bind_shortcuts()
        self._warm_spline_import()
        self.model_refreshed.connect(self._refresh_selection)

        self.prefetcher = Prefetcher(store, cfg.ui.prefetch)
        self._apply_filters(initial=True)
        if cfg.ui.autoplay:
            self.viewer.play()

    # ------------------------------------------------------------------ ui -- #
    def _status(self, message: str, timeout: int = 0) -> None:
        self.status_message.emit(message, timeout)

    def _build_ui(self) -> None:
        self.viewer = SimpleViewer(fps=self.cfg.ui.fps, loop=self.cfg.ui.loop)
        if hasattr(self.viewer, "set_edges"):
            self.viewer.set_edges(self.cfg.keypoints.edge_idx if self.cfg.is_pose else [])

        # -- top bar --
        self.movie_combo = QComboBox()
        self.movie_combo.addItem("all movies", None)
        for movie_id in self.store.movie_ids:
            self.movie_combo.addItem(movie_id, movie_id)
        self.movie_combo.setFocusPolicy(Qt.NoFocus)
        # Movie ids are long: unelided the combo crushes everything to its right.
        self.movie_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.movie_combo.setMinimumContentsLength(12)
        self.movie_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.movie_combo.currentIndexChanged.connect(lambda _: self._apply_filters())

        self.unannotated_only = QCheckBox("unannotated only")
        self.unannotated_only.setFocusPolicy(Qt.NoFocus)
        self.unannotated_only.stateChanged.connect(lambda _: self._apply_filters())

        self.overlay_box = QCheckBox("marker (o)")
        self.overlay_box.setToolTip("outline the animal this clip is about")
        self.overlay_box.setChecked(self.show_overlay)
        self.overlay_box.setFocusPolicy(Qt.NoFocus)
        self.overlay_box.stateChanged.connect(self._on_overlay_toggled)

        self.context_box = QCheckBox("neighbours")
        self.context_box.setToolTip(
            "also outline other tracks in the crop. Off by default: the point of "
            "the marker is to say which animal is being judged."
        )
        self.context_box.setChecked(self.show_context)
        self.context_box.setFocusPolicy(Qt.NoFocus)
        self.context_box.stateChanged.connect(self._on_context_toggled)

        self.follow_combo = QComboBox()
        for mode in FOLLOW_MODES:
            self.follow_combo.addItem(mode, mode)
            self.follow_combo.setItemData(
                FOLLOW_MODES.index(mode), FOLLOW_LABELS[mode], Qt.ToolTipRole
            )
        self.follow_combo.setCurrentIndex(FOLLOW_MODES.index(self.follow))
        # Sized to its content: a fixed width below minimumSizeHint gets crushed.
        self.follow_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.follow_combo.setFocusPolicy(Qt.NoFocus)
        self.follow_combo.setToolTip(
            "\n".join(FOLLOW_LABELS[m] for m in FOLLOW_MODES)
            + "\n\nAll three read the same cached pixels, so switching is free."
        )
        self.follow_combo.currentIndexChanged.connect(self._on_follow_changed)

        self.zoom_label = QLabel("1.0x")
        self.zoom_label.setMinimumWidth(38)
        self.zoom_label.setAlignment(Qt.AlignCenter)
        self.zoom_label.setToolTip("zoom into the crop  (+ / - / 0)")

        help_btn = compact_button("?", 28, "keyboard shortcuts")
        help_btn.clicked.connect(self._show_help)

        top = QHBoxLayout()
        top.addWidget(QLabel("movie"))
        top.addWidget(self.movie_combo, 1)
        top.addWidget(self.unannotated_only)
        top.addSpacing(12)
        top.addWidget(self.overlay_box)
        top.addWidget(self.context_box)
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

        # A focused QLineEdit eats every plain keypress, so "1", "o", "s" become
        # text. Escape and Enter always release focus; "n" grabs it.
        self.notes = QLineEdit()
        self.notes.setPlaceholderText("notes  -  press n to type, Esc when done")
        self.notes.installEventFilter(self)
        self.notes.returnPressed.connect(self._leave_notes)

        self.metadata = MetadataPanel()
        self.minimap = MinimapWidget()
        self.minimap.setVisible(self.show_minimap)

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
        side_layout.addWidget(self.minimap)
        side = QWidget()
        side.setLayout(side_layout)
        side.setMinimumWidth(280)

        # The kymograph shares the clip's time axis, so it sits directly under it.
        self.kymo_view = KymographView()
        self.kymo_view.frame_picked.connect(self._scrub_to)
        self.kymo_view.set_interpolate(self.cfg.ui.kymo_interpolate)
        self.kymo_view.set_field_kind(self.cfg.ui.kymo_field)
        self.kymo_view.set_sign_mode(self.cfg.ui.kymo_sign)
        self.kymo_view.setVisible(self._kymograph_wanted())
        if hasattr(self.viewer, "frame_changed"):
            self.viewer.frame_changed.connect(self.kymo_view.set_frame)

        left = QSplitter(Qt.Vertical)
        left.addWidget(self.viewer)
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
            bind(seq, self.next_unannotated)
        for seq in keys["first_unannotated"]:
            bind(seq, lambda: self.goto_first_unannotated(wrap=True))
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
        for seq in keys["minimap"]:
            bind(seq, self._toggle_minimap)
        for seq in keys["uncertain"]:
            bind(seq, lambda: self.uncertain_box.setChecked(not self.uncertain_box.isChecked()))
        for seq in keys["clear"]:
            bind(seq, self._clear_current)
        for seq in keys["undo"]:
            bind(seq, self._undo)
        for seq in keys["save"]:
            bind(seq, lambda: self._commit(force=True))
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
            # Commit first: a filter change must not discard a half-entered note.
            self._commit()
            current_id = self._clip_ids[self.order[self.position]]

        manifest = self.store.manifest
        mask = manifest["clip_id"].notna()
        movie_id = self.movie_combo.currentData()
        if movie_id:
            mask &= manifest["movie_id"] == movie_id
        if self.unannotated_only.isChecked():
            done = self.annotations.done_ids
            mask &= ~manifest["clip_id"].isin(done)

        # Shuffle the FULL manifest once, then filter it. Re-permuting the
        # filtered list would re-randomise the order on every filter change and
        # strand clips behind the cursor, since navigation only scans forward.
        selected = set(manifest.index[mask].tolist())
        self.order = [i for i in self._shuffled_index if i in selected]
        if not self.order:
            self._show_empty()
            return

        index = self.store.index_of(current_id) if current_id is not None else None
        anchor = index if index in selected else None
        self.order = self._queue_with_batch(self.order, anchor)

        self.position = 0
        if anchor is not None:
            self.position = self.order.index(anchor)
        elif initial:
            self.position = self._resume_position()
        self._load()

    def _resume_position(self) -> int:
        if self.cfg.ui.resume == "last_position":
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

    def _show_empty(self) -> None:
        self.counter.setText("no clips match the current filter")
        self.progress.setValue(0)
        self.kymo_view.clear()

    # ------------------------------------------------------------ selection -- #
    def _queue_with_batch(self, order: list[int], anchor: int | None) -> list[int]:
        """``order`` with the selector's picks moved to just after ``anchor``.

        The queue is forward-only, so anything spliced behind the cursor is
        unreachable: ``anchor``, the clip on screen, keeps its place and the
        picks go immediately after it. With ``anchor`` ``None`` the splice point
        is the front, so the top-ranked pick opens the session.

        ``anchor`` is never moved into the batch even when the selector chose it:
        it was already in front of the annotator, so it keeps whatever reason it
        had. Everything not picked keeps its relative order, so an empty batch
        returns ``order`` unchanged.

        Records into ``_served`` as it goes, because commit time cannot tell the
        selector's picks from the annotator's own detours.
        """
        batch = self._fresh_batch(order)
        if not batch:
            return order

        index_of = {self._clip_ids[i]: i for i in order}
        picked: list[int] = []
        taken: set[int] = set()
        served: list[dict] = []
        for record in batch:
            index = index_of.get(str(record["clip_id"]))
            if index is None or index == anchor or index in taken:
                continue
            taken.add(index)
            picked.append(index)
            served.append(record)
        if not picked:
            return order

        self._batch = served
        for index, record in zip(picked, served):
            self._served[self._clip_ids[index]] = record

        rest = [i for i in order if i not in taken]
        splice = rest.index(anchor) + 1 if anchor is not None else 0
        return rest[:splice] + picked + rest[splice:]

    def _fresh_batch(self, order: list[int]) -> list[dict]:
        """The selector's next picks, or ``[]`` to leave the queue exactly as it is.

        The pool handed over is the clips the current filter makes REACHABLE and
        not the whole manifest, since a pick outside the movie filter could only
        be honoured by changing that filter under the annotator.
        """
        if self.suggester is None or not order:
            return []
        try:
            # `.iloc`, because `order` holds POSITIONS into the manifest, not
            # labels. A manifest re-indexed without a reset would otherwise give
            # a KeyError, or a pool scored against the wrong clip_ids.
            batch = list(self.suggester.selection_batch(
                self.store.manifest.iloc[order],
                self.annotations.done_ids,
                int(self.cfg.suggest.select_batch_size),
            ))
            for record in batch:
                if not str(record.get("clip_id", "")).strip():
                    raise ValueError(
                        f"a selection record names no clip_id ({record!r}), so "
                        "there is no clip to queue it against"
                    )
                # The same encoder `store.record` will use, run HERE, so a record
                # it refuses costs one batch instead of aborting the save of a
                # label somebody has just typed.
                encode_provenance(record)
        except Exception as exc:
            # Said out loud, and the annotator keeps working: shuffle order is a
            # worse queue, not a broken one.
            message = f"clip selection is off - {type(exc).__name__}: {exc}"
            print(f"[trackanno] {message}")
            self._status(message, 10000)
            return []
        return batch

    def _provenance_for(self, clip_id: str) -> dict | None:
        """Why this clip was in front of the annotator, or ``None`` for *not recorded*.

        Three answers, kept apart:

        * the selector put this clip in the queue -> its own record, the rule and
          score that chose it;
        * a batch has been served and this clip is not in it -> the annotator
          navigated here;
        * no batch has ever been served in this session -> nothing.

        The third is the weaker claim on purpose. ``selection_batch`` returns
        ``[]`` for "selection is switched off", "the model is not ready" and
        "nothing left to choose" alike, so ``annotator_navigated`` would assert
        a selector was running when only one of the three says so.
        """
        if not self._served:
            return None
        return self._served.get(clip_id) or {"rule": MANUAL_RULE}

    def _refresh_selection(self) -> None:
        """Re-queue after the model changed, without disturbing the clip on screen.

        Hooked to a refit COMPLETING rather than to each label: the ranking is a
        function of the model, so between refits it has nothing new to say.

        Nothing here calls ``_load``, so the clip on screen keeps its pixels, its
        half-typed note and its dwell clock; only what comes AFTER it moves.
        """
        if not self.order:
            return
        anchor = self.order[self.position]
        before = self.order
        self.order = self._queue_with_batch(self.order, anchor)
        if self.order == before:
            # Selection is off, the refit produced nothing to re-queue, or every
            # pick was already in place. Nothing moved, so say nothing.
            return
        self.position = self.order.index(anchor)
        # The queue after the cursor is new, so the clips being read ahead are the
        # wrong ones until this is re-aimed.
        self.prefetcher.around(self.order, self.position)
        self._status(
            f"model updated - the queue's next {len(self._batch)} clip(s) were "
            "re-selected", 6000
        )

    # ----------------------------------------------------------- clip load -- #
    def _build_view(self, n_frames: int, size: int):
        return build_clip_view(
            self._boxes, n_frames, size,
            mode=self.follow,
            zoom=self.zoom,
            smooth_window=self.cfg.ui.smooth_window,
            with_context=self.show_context,
            n_kpts=self.cfg.n_keypoints if self.cfg.ui.show_skeleton else 0,
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

        # Grayscale in, marker painted as vectors: no RGB stack is ever built.
        self.viewer.set_overlay(
            self.show_overlay, self.show_context, self.cfg.ui.show_skeleton
        )
        self.viewer.set_clip(self._clip, view)
        if self.cfg.ui.autoplay and not self.viewer.is_playing:
            self.viewer.play()

        self.metadata.update_from(meta, self.store.origins(index))
        self._update_kymograph(self._boxes)
        self._refresh_minimap(index, meta)

        clip_id = str(meta["clip_id"])
        self._current_suggestion = (
            self.suggester.suggest(clip_id) if self.suggester and self.suggester.ready else {}
        )
        self._suggestion_note = self._why_no_suggestion(clip_id)

        existing = self.annotations.get(clip_id)
        self._suppress_advance = True
        self.categories.set_values(existing)
        # AFTER set_values: it repaints every button, wiping the dashed outline.
        self.categories.set_suggestion(self._current_suggestion, self._suggestion_note)
        self.uncertain_box.setChecked(bool(existing and existing.get("uncertain") == "1"))
        self.notes.setText((existing or {}).get("notes", ""))
        self._suppress_advance = False
        self._current_values = self.categories.values()

        self._loaded_at = time.time()
        self._update_progress(index, meta, (time.perf_counter() - started) * 1000)
        self.prefetcher.around(self.order, self.position)
        self._save_state(str(meta["clip_id"]))

    def _refresh_view(self) -> None:
        """Re-centre / re-zoom / re-mark without re-reading the clip."""
        if self._clip is None:
            return
        view = self._build_view(self._clip.shape[0], self._clip.shape[1])
        self.viewer.set_overlay(
            self.show_overlay, self.show_context, self.cfg.ui.show_skeleton
        )
        self.viewer.set_view(view)

    def _update_progress(self, index: int, meta, load_ms: float) -> None:
        done = self.annotations.done_ids
        done_here = sum(1 for i in self.order if self._clip_ids[i] in done)
        total = len(self.order)
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done_here)
        mark = "OK" if str(meta["clip_id"]) in self.annotations.done_ids else "--"
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

        if not self.cfg.ui.auto_advance or not changed_single:
            return
        # Only a single-choice pick advances; a checkbox never yanks the clip away.
        if self.annotations.missing_required(values):
            return
        self._commit()
        self.next_clip()

    def _commit(self, force: bool = False) -> bool:
        if not self.order:
            return False
        meta = self.store.meta(self.order[self.position])
        values = self.categories.values()
        uncertain = self.uncertain_box.isChecked()
        notes = self.notes.text().strip()
        has_any = any(values.values()) or uncertain or notes
        existing_row = self.annotations.get(str(meta["clip_id"]))
        if not has_any:
            # De-selecting everything must actually un-label the clip.
            if existing_row is not None:
                try:
                    self.annotations.clear(str(meta["clip_id"]))
                except OSError as exc:
                    self._pending_message = self._write_failed(exc)
                    return False
                self._pending_message = "annotation cleared"
                return True
            if not force:
                return False

        # Revisiting an already-labelled clip must not rewrite the CSV: editing a
        # row is a full atomic rewrite, and the undo stack would fill with no-ops.
        existing = self.annotations.get(str(meta["clip_id"]))
        if existing is not None and not force:
            unchanged = (
                all(existing.get(g.name, "") == values.get(g.name, "") for g in self.cfg.categories)
                and existing.get("uncertain", "0") == ("1" if uncertain else "0")
                and existing.get("notes", "") == notes
            )
            if unchanged:
                return False

        missing = self.annotations.missing_required(values)
        if missing and not self.uncertain_box.isChecked():
            if force:
                QMessageBox.information(
                    self, "Incomplete", "Still unset: " + ", ".join(missing)
                )
            else:
                # Don't block navigation, but never drop input silently.
                self._pending_message = (
                    f"not saved - {', '.join(missing)} still unset "
                    f"(press u to mark uncertain instead)"
                )
            return False

        # Record the model's answer and whether it was visible: agreement is not
        # influence unless the two are recorded apart.
        primary = next(
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
                selected_by=self._provenance_for(str(meta["clip_id"])),
            )
        except OSError as exc:
            # The store rolls back and re-raises; escaping a Qt slot aborts.
            self._pending_message = self._write_failed(exc)
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
        # Two trainings at once race on the model file, so skip rather than queue.
        if self._retrain_thread is not None and self._retrain_thread.is_alive():
            return
        self._labels_since_train = 0

        import threading

        def work():
            try:
                from copy import copy

                from ..learn.suggest import train

                # Snapshot, never the live CSV: on Windows an open handle makes
                # the UI thread's atomic rewrite fail outright.
                snapshot = self.cfg.paths.work_dir / ".annotations_snapshot.csv"
                self.annotations.snapshot(snapshot)
                cfg = copy(self.cfg)
                cfg.paths = copy(self.cfg.paths)
                cfg.paths.csv = snapshot
                train(cfg, verbose=False)
                self.suggester.load()
                # AFTER load(), so the batch the slot goes on to ask for is
                # ranked by the new model. Emitted rather than called: this is
                # not the UI thread.
                self.model_refreshed.emit()
            except Exception as exc:
                print(f"[trackanno] background retrain failed: {exc}")

        self._retrain_thread = threading.Thread(
            target=work, name="trackanno-retrain", daemon=True
        )
        self._retrain_thread.start()

    # ---------------------------------------------------------- kymograph -- #
    def _kymograph_wanted(self) -> bool:
        """Only on a pose project: with no keypoints there is no curve to fit."""
        return bool(self.cfg.ui.show_kymograph and self.cfg.is_pose
                    and self.cfg.n_keypoints >= 2)

    def _why_no_suggestion(self, clip_id: str) -> str:
        """One line explaining an empty suggestion, or "" to stay silent. Only
        spoken when ``suggest.audit_visible`` is on, because knowing a clip is
        audited is the influence the holdout exists to exclude."""
        if self._current_suggestion or self.suggester is None:
            return ""
        if not self.cfg.suggest.audit_visible:
            return ""
        if self.suggester.is_audit_clip(clip_id):
            return ("held out for audit: no suggestion is shown on this clip at "
                    "any confidence, so your answer stays independent")
        if not self.suggester.ready:
            return self.suggester.status_line()
        return "the model has nothing to say about this clip"

    def apply_view_settings(self, cfg) -> None:
        """Adopt the display-only settings from a freshly reloaded config.

        They change what is drawn, not what is loaded, so they must not go
        through a panel rebuild, which would discard the clip and the filter."""
        ui = cfg.ui
        changed = (ui.kymo_field != self.cfg.ui.kymo_field
                   or ui.kymo_sign != self.cfg.ui.kymo_sign
                   or ui.kymo_interpolate != self.cfg.ui.kymo_interpolate
                   or ui.show_kymograph != self.cfg.ui.show_kymograph)
        self.cfg = cfg
        if not changed:
            return
        self.kymo_view.set_field_kind(ui.kymo_field)
        self.kymo_view.set_sign_mode(ui.kymo_sign)
        self.kymo_view.set_interpolate(ui.kymo_interpolate)
        wanted = self._kymograph_wanted()
        self.kymo_view.setVisible(wanted)
        # Turning it on mid-session has nothing to draw until a clip goes through.
        if wanted and self.kymo_view.kymo is None and self._boxes is not None:
            self._update_kymograph(self._boxes)

    def _warm_spline_import(self) -> None:
        """Import scipy off the UI thread, so the first kymograph does not pay
        for it."""
        if not self._kymograph_wanted():
            return
        import threading

        def work():
            try:
                import scipy.interpolate  # noqa: F401
            except Exception:
                pass          # the failure surfaces properly at the first use

        threading.Thread(target=work, name="trackanno-spline-warm",
                         daemon=True).start()

    def _update_kymograph(self, boxes) -> None:
        """Recompute for the clip now on screen. Cheap enough to run inline."""
        if not self._kymograph_wanted():
            return
        from ..measure.kymograph import chain_from_edges, kymograph_from_boxes

        # Which points form the flagellum, and in which direction, comes from the
        # project's skeleton: the order sets the sign of every value on the chart.
        order = chain_from_edges(self.cfg.n_keypoints, self.cfg.keypoints.edge_idx)
        try:
            kymo = kymograph_from_boxes(
                boxes, self.cfg.n_keypoints, int(self._clip.shape[0]),
                min_conf=0.0, n_samples=128, order=order,
                smooth_px=self.cfg.ui.kymo_smooth_px,
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
        """Pick up a model that was just (re)trained, without losing your place.
        The model is not part of what the panel is built from, so it is reloaded
        in place rather than by rebuilding."""
        if not self.cfg.suggest.enabled:
            self.suggester = None
            return "suggestions are switched off in the config"
        if self.suggester is None:
            self.suggester = Suggester(self.cfg)
        else:
            self.suggester.load()

        # Re-run the current clip so the change shows now, not on the next Next.
        if self.order:
            meta = self.store.meta(self.order[self.position])
            clip_id = str(meta["clip_id"])
            self._current_suggestion = (
                self.suggester.suggest(clip_id) if self.suggester.ready else {}
            )
            self.categories.set_suggestion(self._current_suggestion)
        # The same event as a background refit finishing, so the queue is
        # re-ranked the same way. Called, not emitted: already on the UI thread.
        self._refresh_selection()
        return self.suggester.status_line()

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
        """One sentence the annotator can act on, instead of a traceback."""
        hint = ""
        if getattr(exc, "errno", None) in (13, 30) or "Permission" in type(exc).__name__:
            hint = " - is it open in Excel, or is the folder read-only?"
        return f"could not write {self.annotations.path.name}: {exc}{hint}"

    def _undo(self) -> None:
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

    def next_unannotated(self) -> None:
        self._commit()
        done = self.annotations.done_ids
        for pos in range(self.position + 1, len(self.order)):
            if self._clip_ids[self.order[pos]] not in done:
                self.position = pos
                self._load()
                return
        self._status("no unannotated clips after this one", 3000)

    def goto_first_unannotated(self, wrap: bool = False) -> None:
        self._commit()
        done = self.annotations.done_ids
        for pos, index in enumerate(self.order):
            if self._clip_ids[index] not in done:
                self.position = pos
                self._load()
                return
        self._status("every clip in this filter is annotated", 4000)

    # -------------------------------------------------------------- toggles -- #
    def _on_overlay_toggled(self, state) -> None:
        self.show_overlay = bool(self.overlay_box.isChecked())
        self._refresh_view()

    def _on_context_toggled(self, state) -> None:
        self.show_context = bool(self.context_box.isChecked())
        self._refresh_view()

    def _on_follow_changed(self, index: int) -> None:
        self.follow = self.follow_combo.currentData() or "centred"
        self._refresh_view()
        self._status(FOLLOW_LABELS[self.follow] + self._headroom_note(), 6000)

    def _headroom_note(self) -> str:
        """Warn when the cached crop leaves the view nothing to do: with
        ``clips.follow: track`` and a tight ``clips.pad`` all three view modes
        land within a pixel of each other and the control looks broken."""
        if self._clip is None or self._boxes is None:
            return ""
        size = self._clip.shape[1]
        centres = [
            build_clip_view(self._boxes, self._clip.shape[0], size, mode=mode).centre
            for mode in FOLLOW_MODES
        ]
        spread = max(
            float(np.abs(a - b).max()) for a in centres for b in centres
        )
        if spread >= 2.0:
            return ""
        return (
            f"   -  but this clip's crop only allows {spread:.1f} px of movement, so the "
            f"three views look the same. The cache was cut with clips.follow: "
            f"{self.cfg.clips.follow!r} / pad {self.cfg.clips.pad:g}, which removed the "
            f"motion before the viewer saw it. Re-cut with follow: fixed and a larger pad."
        )

    def cycle_follow(self) -> None:
        nxt = (FOLLOW_MODES.index(self.follow) + 1) % len(FOLLOW_MODES)
        self.follow_combo.setCurrentIndex(nxt)

    def set_zoom(self, zoom: float) -> None:
        self.zoom = float(min(max(zoom, ZOOM_STEPS[0]), ZOOM_STEPS[-1]))
        self._sync_zoom_label()
        self._refresh_view()

    def step_zoom(self, direction: int) -> None:
        steps = list(ZOOM_STEPS)
        current = min(range(len(steps)), key=lambda i: abs(steps[i] - self.zoom))
        self.set_zoom(steps[min(max(current + direction, 0), len(steps) - 1)])

    def _sync_zoom_label(self) -> None:
        self.zoom_label.setText(f"{self.zoom:.2f}x".replace(".00x", ".0x"))

    def _refresh_minimap(self, index=None, meta=None) -> None:
        if not self.show_minimap or not self.order:
            return
        if index is None:
            index = self.order[self.position]
        if meta is None:
            meta = self.store.meta(index)
        header = self.store.header(str(meta["movie_id"]))["movie"]
        self.minimap.update_from(
            self.store.thumbnail(index), meta, (header["height"], header["width"])
        )

    def _toggle_minimap(self) -> None:
        self.show_minimap = not self.show_minimap
        self.minimap.setVisible(self.show_minimap)
        # Only the minimap: _load() would rewrite every widget from the CSV.
        self._refresh_minimap()

    # ---------------------------------------------------------------- notes -- #
    def eventFilter(self, obj, event):  # noqa: N802 (Qt naming)
        if obj is self.notes and event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Escape:
                self._leave_notes()
                return True
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                # `Return` is also bound to "next clip" and pre-empts returnPressed.
                self._leave_notes()
                return True
        return super().eventFilter(obj, event)

    def _focus_notes(self) -> None:
        self.notes.setFocus(Qt.ShortcutFocusReason)
        self.notes.selectAll()
        self._status("typing notes - Esc or Enter to return to shortcuts", 4000)

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

        # Why a confident model shows nothing on some clips. Explained HERE and
        # not on the clip: knowing it is audited is the influence to exclude.
        share = self.cfg.suggest.audit_fraction
        if self.cfg.suggest.enabled and share > 0:
            lines.append(
                "<br><b>Why some clips show no suggestion</b><br>"
                f"About {share:.0%} of clips are held out of the suggestion loop. "
                "They are never trained on and never carry a suggestion, whatever "
                "the model's confidence, so how often you and the model agree on "
                "them is an honest measure rather than a measure of how much the "
                "suggestion nudged you.<br>"
                "<i>suggest.audit_fraction</i> controls the share; set it to 0 to "
                "see a suggestion on every clip and give up that measure."
            )
        QMessageBox.information(self, "Shortcuts", "".join(lines))

    # ---------------------------------------------------------------- close -- #
    def restyle(self) -> None:
        """Repaint what is styled in code: the category buttons and the canvas."""
        self.categories.restyle()
        self.categories.set_suggestion(self._current_suggestion)
        self.kymo_view.restyle()
        if hasattr(self.viewer, "canvas"):
            self.viewer.canvas.update()

    def shutdown(self) -> None:
        # try/finally: a failing commit must still release the prefetch thread
        # and the clip cache.
        try:
            self._commit()
        except Exception as exc:
            print(f"[trackanno] could not save the last clip: {exc}")
        finally:
            if not self.prefetcher.stop():
                print("[trackanno] prefetch thread did not stop in time")
        # Every one of these is a VIEW on the clip cache's memory map, and while
        # any of them lives the file cannot be rewritten on Windows.
        self._clip = None
        self._boxes = None
        if hasattr(self.viewer, "clear"):
            self.viewer.clear()


class AnnotationWindow(QMainWindow):
    """Standalone host for :class:`AnnotationWidget` (``uv run annotate.py``)."""

    def __init__(self, cfg: Config, store: ClipStore, annotations: AnnotationStore):
        super().__init__()
        self.setWindowTitle("trackanno")
        self.setWindowIcon(app_icon())
        self.resize(1180, 780)
        self.panel = AnnotationWidget(cfg, store, annotations)
        self.panel.status_message.connect(
            lambda text, timeout: self.statusBar().showMessage(text, timeout)
        )
        self.setCentralWidget(self.panel)
        self.panel.setFocus()
        self.statusBar().showMessage(
            f"cache: {cfg.paths.work_dir}   |   csv: {cfg.paths.csv}"
        )

    def __getattr__(self, name):
        # The window is a thin shell; everything interesting lives on the panel.
        try:
            return getattr(object.__getattribute__(self, "panel"), name)
        except AttributeError as exc:
            raise AttributeError(name) from exc

    def closeEvent(self, event):  # noqa: N802 (Qt naming)
        self.panel.shutdown()
        super().closeEvent(event)


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="annotate", description="Annotate track clips.")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--csv", default=None, help="override paths.csv")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.csv:
        cfg.paths.csv = Path(args.csv).resolve()

    try:
        store = ClipStore(cfg.paths.work_dir)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    annotations = AnnotationStore(cfg.paths.csv, cfg)
    print(
        f"{len(store)} clips across {len(store.movie_ids)} movie(s); "
        f"{len(annotations)} already annotated -> {cfg.paths.csv}"
    )

    app = QApplication.instance() or QApplication(sys.argv[:1])
    apply_theme(app, cfg.ui.theme)
    window = AnnotationWindow(cfg, store, annotations)
    window.show()
    return app.exec_() if hasattr(app, "exec_") else app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
