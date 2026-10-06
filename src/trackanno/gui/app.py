"""The annotation surface, hosted as a page of the workflow window.

One clip of one track at a time. A keypress labels it and jumps to the next; the
CSV is written and fsynced on every annotation, so a session resumes from any
point, including after a crash.

:class:`AnnotationWidget` is a plain ``QWidget``. It emits ``status_message``
rather than touching a status bar, and lets its host place that.
"""

from __future__ import annotations

import json
import time

from qtpy.QtCore import QEvent, Qt, Signal
from qtpy.QtGui import QKeySequence

try:  # Qt6 puts QShortcut in QtGui, Qt5 in QtWidgets; qtpy aliases but be explicit
    from qtpy.QtGui import QShortcut
except ImportError:  # pragma: no cover
    from qtpy.QtWidgets import QShortcut

from qtpy.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QProgressBar, QPushButton, QSizePolicy, QSplitter, QVBoxLayout, QWidget,
)

from ..annotate.clips import ClipStore
from ..config import Config
from ..infer.agreement import differs, load_cv_predictions
from ..view.overlay import FOLLOW_LABELS, FOLLOW_MODES, build_clip_view
from ..annotate.store import AnnotationStore
from ..learn.suggest import Suggester
from .panels import CategoryPanel, MetadataPanel
from .prefetch import Prefetcher
from .theme import compact_button
from .kymograph import KymographView
from .viewer import SimpleViewer, follow_combo

STATE_FILE = ".gui_state.json"

ZOOM_STEPS = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0)

# Said once per review clip, in place of a suggestion, so the absence reads as a
# decision rather than a model that has nothing to say.
REVIEW_SUGGESTION_NOTE = "no suggestion here: this model was fitted on this label"


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

        # "annotate" walks the clips still to do; "review" walks the ones
        # already labelled, in the order they were labelled, so a session can be
        # read back. Everything else on this screen is shared.
        self.mode = "annotate"
        # What the folds said about each labelled clip, read once per mode switch.
        self._answers: dict[str, dict] = {}
        self._agreement_problem = ""

        self.show_overlay = cfg.ui.show_overlay
        self.show_context = cfg.ui.show_context_tracks
        self.follow = cfg.ui.follow
        self.zoom = float(cfg.ui.view_zoom)
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
        # Every clip the selector has queued, keyed by clip_id (positions shift
        # when a movie is added).
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

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("annotate", "annotate")
        self.mode_combo.addItem("review labelled", "review")
        self.mode_combo.setFocusPolicy(Qt.NoFocus)
        self.mode_combo.setToolTip(
            "annotate walks the clips still to do.\n"
            "review labelled walks the ones already labelled, oldest first, "
            "with who judged each and where the model would say otherwise. "
            "Tab jumps to the next label the model disagrees with."
        )
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)

        self.overlay_box = QCheckBox("marker (o)")
        self.overlay_box.setToolTip("outline the animal this clip is about")
        self.overlay_box.setChecked(self.show_overlay)
        self.overlay_box.setFocusPolicy(Qt.NoFocus)
        self.overlay_box.stateChanged.connect(self._on_overlay_toggled)

        self.context_box = QCheckBox("neighbours")
        self.context_box.setToolTip(
            "also outline other tracks in the crop. Off by default, so the "
            "marker says which animal is being judged."
        )
        self.context_box.setChecked(self.show_context)
        self.context_box.setFocusPolicy(Qt.NoFocus)
        self.context_box.stateChanged.connect(self._on_context_toggled)

        self.follow_combo = follow_combo(self.follow)
        self.follow_combo.currentIndexChanged.connect(self._on_follow_changed)

        self.zoom_label = QLabel("1.0x")
        self.zoom_label.setMinimumWidth(38)
        self.zoom_label.setAlignment(Qt.AlignCenter)
        self.zoom_label.setToolTip("zoom into the crop  (+ / - / 0)")

        help_btn = compact_button("?", 28, "keyboard shortcuts")
        help_btn.clicked.connect(self._show_help)

        top = QHBoxLayout()
        # Labelled like every other control in this row. Unlabelled it read as
        # part of the movie filter beside it, and nobody found it.
        top.addWidget(QLabel("mode"))
        top.addWidget(self.mode_combo)
        top.addSpacing(12)
        top.addWidget(QLabel("movie"))
        top.addWidget(self.movie_combo, 1)
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

        # Who judged this clip and when, and where the model would disagree.
        # Only in review mode: during annotation it would be the annotator's own
        # name under every clip, and a suggestion already has its own outline.
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

        # The kymograph shares the clip's time axis, so it sits directly under it.
        self.kymo_view = KymographView()
        self.kymo_view.frame_picked.connect(self._scrub_to)
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
            # Commit first: a filter change must not discard a half-entered note.
            self._commit()
            current_id = self._clip_ids[self.order[self.position]]

        manifest = self.store.manifest
        mask = manifest["clip_id"].notna()
        movie_id = self.movie_combo.currentData()
        if movie_id:
            mask &= manifest["movie_id"] == movie_id
        if self.reviewing:
            mask &= manifest["clip_id"].isin(self.annotations.done_ids)

        # Shuffle the FULL manifest once, then filter it. Re-permuting the
        # filtered list would re-randomise the order on every filter change and
        # strand clips behind the cursor, since navigation only scans forward.
        selected = set(manifest.index[mask].tolist())
        if self.reviewing:
            # The order they were labelled in. A review reads a session back,
            # and the shuffle would scatter that session through the whole
            # project; the selector's batch is not spliced in either, because
            # nothing here is waiting to be chosen.
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

    def _empty_reason(self) -> str:
        """Why there is nothing to show, which is a different sentence per mode."""
        if not self.reviewing:
            return "no clips match the current filter"
        if self.annotations.done_ids:
            return "nothing labelled in the current filter"
        return "nothing labelled here yet - switch back to annotate"

    def _show_empty(self) -> None:
        self.counter.setText(self._empty_reason())
        self.progress.setValue(0)
        self.kymo_view.clear()

    @property
    def reviewing(self) -> bool:
        """True in review mode: the queue is labels, not work to do."""
        return self.mode == "review"

    def _refresh_fold_answers(self) -> None:
        """Read what the last fit's folds predicted for each labelled clip.

        Read from ``cv_predictions.parquet`` and never from the saved model,
        which was fitted on these labels and so reproduces them.
        """
        self._answers, self._agreement_problem = {}, ""
        if not self.annotations.labelled_order:
            return
        try:
            self._answers, self._agreement_problem = load_cv_predictions(self.cfg)
        except Exception as exc:                     # noqa: BLE001
            # A model that will not answer is a status line, never a traceback
            # out of a mode switch.
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
            "review": (
                "reading back the labels already recorded, oldest first. A pick "
                "rewrites the label rather than advancing; Tab jumps to the "
                "next one the model disagrees with"
            ),
        }[self.mode], 10000)

    # ------------------------------------------------------------ selection -- #
    def _queue_with_batch(self, order: list[int], anchor: int | None) -> list[int]:
        """``order`` with the selector's picks moved to just after ``anchor``.

        The queue is forward-only, so anything spliced behind the cursor is
        unreachable: ``anchor``, the clip on screen, keeps its place and the
        picks go immediately after it. With ``anchor`` ``None`` the splice point
        is the front, so the top-ranked pick opens the session.

        ``anchor`` is never moved into the batch. Everything not picked keeps
        its relative order.
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
        except Exception as exc:
            # Said out loud, and the annotator keeps working: shuffle order is a
            # worse queue, not a broken one.
            message = f"clip selection is off - {type(exc).__name__}: {exc}"
            print(f"[trackanno] {message}")
            self._status(message, 10000)
            return []
        return batch

    def _refresh_selection(self) -> None:
        """Re-queue after the model changed, without disturbing the clip on screen.

        Hooked to a refit COMPLETING rather than to each label: the ranking is a
        function of the model, so between refits it has nothing new to say.

        Nothing here calls ``_load``, so the clip on screen keeps its pixels, its
        half-typed note and its dwell clock; only what comes AFTER it moves.
        """
        if not self.order:
            return
        if self.reviewing:
            # A new model has new opinions. Re-read, but the queue is left
            # alone: re-ordering under a reviewer mid-clip would lose their place.
            self._refresh_fold_answers()
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
        self._status("model updated - the queue ahead was re-selected", 6000)

    # ----------------------------------------------------------- clip load -- #
    def _build_view(self, n_frames: int, size: int):
        return build_clip_view(
            self._boxes, n_frames, size,
            mode=self.follow,
            zoom=self.zoom,
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

        clip_id = str(meta["clip_id"])
        self._current_suggestion = (
            self.suggester.suggest(clip_id) if self.suggester and self.suggester.ready else {}
        )
        self._suggestion_note = self._why_no_suggestion(clip_id)

        existing = self.annotations.get(clip_id)
        self._suppress_advance = True
        self.categories.set_values(existing)
        # AFTER set_values: it repaints every button, wiping the dashed outline.
        # Nothing is outlined while reviewing. The suggester was fitted ON these
        # labels, so its opinion of one reproduces it rather than testing it, and
        # the outline would land on the label already chosen beside a note saying
        # the held-out model disagreed. Only the note means anything here.
        if self.reviewing:
            self.categories.set_suggestion({}, REVIEW_SUGGESTION_NOTE)
        else:
            self.categories.set_suggestion(self._current_suggestion, self._suggestion_note)
        self.uncertain_box.setChecked(bool(existing and existing.get("uncertain") == "1"))
        self.notes.setText((existing or {}).get("notes", ""))
        self._suppress_advance = False
        self._current_values = self.categories.values()

        self._paint_review_note(clip_id, existing)

        self._loaded_at = time.time()
        self._update_progress(index, meta, (time.perf_counter() - started) * 1000)
        self.prefetcher.around(self.order, self.position)
        # Only from annotate mode: the resume point is where the work stopped,
        # and a detour through the labels already done is not where to come back.
        if not self.reviewing:
            self._save_state(str(meta["clip_id"]))

    def _disagreement(self, clip_id: str, suggestion: dict | None = None) -> list[str]:
        """Groups where the recorded label is not what the folds predicted.

        The comparison itself lives in :func:`trackanno.infer.agreement.differs`,
        which the review page's own walk uses too: two ideas of what counts as a
        disagreement would sooner or later disagree with each other.

        Empty for a clip no fold scored - the audit holdout, a clip with no
        features, or one labelled since the last fit.
        """
        row = self.annotations.get(clip_id)
        if row is None:
            return []
        if suggestion is None:
            suggestion = self._answers.get(clip_id) or {}
        return differs(row, suggestion)

    def _paint_review_note(self, clip_id: str, existing: dict | None) -> None:
        """Who judged this clip and when, and where the folds said otherwise.

        ``existing`` is a label row with the buttons' current values laid over
        it, so the disagreement clears as the label is changed.
        """
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

        # The fold prediction, from a model fitted without this clip's movie.
        folds = self._answers.get(clip_id) or {}
        disagrees = {entry.split(":", 1)[0] for entry in differs(existing, folds)}
        if disagrees:
            body = ", ".join(
                f"{group} <b>{value}</b>"
                + ("" if confidence is None else f" ({confidence:.0%} sure)")
                for group, (value, confidence) in folds.items()
                if group in disagrees
            )
            line += (
                f"<br><b>held out, the model said</b> {body}"
                "<br>One of the two is wrong. This clip is what says which."
            )
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
            self.show_overlay, self.show_context, self.cfg.ui.show_skeleton
        )
        self.viewer.set_view(view)

    def _update_progress(self, index: int, meta, load_ms: float) -> None:
        done = self.annotations.done_ids
        done_here = sum(1 for i in self.order if self._clip_ids[i] in done)
        total = len(self.order)
        self.progress.setRange(0, max(1, total))
        if self.reviewing:
            # Everything here is labelled, so the bar is how far through the
            # review this is rather than how much of it is done.
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

        # Reviewing is checking, not labelling: the clip being judged must not
        # be yanked away by the keypress that judged it, since the next thing
        # the reviewer usually does is look at it again.
        if self.reviewing:
            # Repaint against what is on the buttons NOW, not against the row on
            # disk: the point of changing a label here is to watch the
            # disagreement clear, and it has not been committed yet.
            clip_id = self._clip_ids[self.order[self.position]]
            self._paint_review_note(
                clip_id, {**(self.annotations.get(clip_id) or {}), **values}
            )
            return
        if not self.cfg.ui.auto_advance or not changed_single:
            return
        # Only a single-choice pick advances; a checkbox never yanks the clip away.
        if self.annotations.missing_required(values):
            return
        self._commit()
        self.next_clip()

    def _commit(self) -> bool:
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
            return False

        # Revisiting an already-labelled clip must not rewrite the CSV: editing a
        # row is a full atomic rewrite, and the undo stack would fill with no-ops.
        existing = self.annotations.get(str(meta["clip_id"]))
        if existing is not None:
            unchanged = (
                all(existing.get(g.name, "") == values.get(g.name, "") for g in self.cfg.categories)
                and existing.get("uncertain", "0") == ("1" if uncertain else "0")
                and existing.get("notes", "") == notes
            )
            if unchanged:
                return False

        missing = self.annotations.missing_required(values)
        if missing and not self.uncertain_box.isChecked():
            # Don't block navigation, but never drop input silently.
            self._pending_message = (
                f"not saved - {', '.join(missing)} still unset "
                f"(press u to mark uncertain instead)"
            )
            return False

        # Record the model's answer and whether it was visible: agreement is not
        # influence unless the two are recorded apart. A revision records neither:
        # the suggestion on screen now comes from today's model, and the original
        # judgement was not made under it.
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
        changed = (ui.kymo_sign != self.cfg.ui.kymo_sign
                   or ui.show_kymograph != self.cfg.ui.show_kymograph)
        self.cfg = cfg
        if not changed:
            return
        self.kymo_view.set_sign_mode(ui.kymo_sign)
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

    def _on_next_gap(self) -> None:
        """Tab: the next clip worth stopping at, which is not the same question
        in each mode. Annotating, it is the next one with no label; reviewing,
        the next one whose label the model would not have chosen."""
        if self.reviewing:
            self.next_disagreement()
        else:
            self.next_unannotated()

    def next_disagreement(self) -> None:
        """The next labelled clip the FOLDS labelled differently.

        Where to start a review: a label a held-out model reproduced is evidence
        about neither of them, and one it contradicted is either a mistake or
        the thing the model still has to learn.
        """
        self._commit()
        if not self._answers:
            self._status(
                self._agreement_problem
                or "no held-out predictions to hold these labels up against. "
                   "Train the model on the Model page", 6000)
            return
        for pos in range(self.position + 1, len(self.order)):
            if self._disagreement(self._clip_ids[self.order[pos]]):
                self.position = pos
                self._load()
                return
        self._status(
            "the model agrees with every label after this one, or has nothing "
            "to say about them", 4000)

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
        self._refresh_view()

    def _on_context_toggled(self, state) -> None:
        self.show_context = bool(self.context_box.isChecked())
        self._refresh_view()

    def _on_follow_changed(self, index: int) -> None:
        self.follow = self.follow_combo.currentData() or "centred"
        self._refresh_view()
        self._status(FOLLOW_LABELS[self.follow], 6000)

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

        lines.append(
            "<br><b>Two modes, same keys</b><br>"
            "<i>annotate</i> walks the clips still to do. <i>review labelled</i> "
            "walks the ones already done, oldest first, showing who judged each "
            "and where the model would now say otherwise. There a pick rewrites "
            "the label instead of advancing, and Tab jumps to the next label the "
            "model argues with."
        )

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
