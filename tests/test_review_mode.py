"""Reading a labelling session back, on the screen that produced it.

A separate suite from ``test_gui_headless``, which is one long scenario about
labelling. These claims are about the other direction of travel: the queue is
the clips already done rather than the ones still to do, a keypress rewrites
rather than advances, and the screen says who judged each clip and where the
model would have said otherwise.

The suggester is stubbed. Whether a real classifier would disagree with a label
is a statement about a classifier; what this suite is about is what the screen
does once something disagrees.
"""

from __future__ import annotations

import csv

import pytest

from harness import check
from trackanno.annotate.clips import ClipStore
from trackanno.annotate.store import PROVENANCE_COLUMN, AnnotationStore
from trackanno.config import load_config


@pytest.fixture(scope="module")
def cfg(root):
    """A generated project, never the repository's own."""
    from fixtures import make_project

    config = load_config(make_project(root))
    # Nothing here writes five labels, so no background refit can fire and make
    # the queue depend on a model that was fitted mid-test.
    config.suggest.retrain_every = 5
    return config


@pytest.fixture(scope="module")
def store(cfg):
    """The clip cache, closed on the way out: on Windows a mapped file cannot
    be overwritten, and a later suite re-runs detection over this one."""
    store = ClipStore(cfg.paths.work_dir)
    yield store
    store.close()


class StubSuggester:
    """Answers only what this screen asks of a suggester."""

    def __init__(self, cfg, answers=None):
        self.cfg = cfg
        self.ready = True
        self.problem = ""
        self.answers = answers or {}

    def load(self) -> bool:
        return True

    def suggest(self, clip_id):
        return self.answers.get(str(clip_id), {})

    def is_audit_clip(self, clip_id) -> bool:
        return False

    def status_line(self) -> str:
        return "stub suggester"

    def selection_batch(self, manifest, done_ids, n) -> list[dict]:
        return []


def build(cfg, store, root, name: str):
    """A real widget over a CSV of its own."""
    from trackanno.gui.app import AnnotationWidget

    path = root / f"{name}.csv"
    path.unlink(missing_ok=True)
    annotations = AnnotationStore(path, cfg)
    return AnnotationWidget(cfg, store, annotations), annotations, path


def label(widget, app, option: int = 0) -> str:
    """Label the clip on screen the way a keypress does; return its clip id."""
    clip_id = widget._clip_ids[widget.order[widget.position]]
    group = widget.cfg.categories[0]
    widget.categories.toggle(group.name, group.options[option].name)
    app.processEvents()
    return clip_id


def set_mode(widget, app, mode: str) -> None:
    widget.mode_combo.setCurrentIndex(
        widget.mode_combo.findData(mode)
    )
    app.processEvents()


def current(widget) -> str:
    return widget._clip_ids[widget.order[widget.position]]


def rows_of(path) -> dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["clip_id"]: r for r in csv.DictReader(fh)}


def arm(widget, answers) -> None:
    """Plant the fold predictions a training run would have left behind.

    The real file, not a stubbed method: what a label is checked against is the
    out-of-fold prediction on disk, and a stub in front of that would let the
    in-sample mistake back in without a test noticing.
    """
    import pandas as pd

    from trackanno.learn.suggest import CV_PREDICTIONS_FILE

    rows = [
        {"clip_id": clip_id, "group": group,
         "predicted": value, "confidence": confidence}
        for clip_id, groups in answers.items()
        for group, (value, confidence) in groups.items()
    ]
    pd.DataFrame(rows, columns=["clip_id", "group", "predicted", "confidence"]
                 ).to_parquet(widget.cfg.paths.work_dir / CV_PREDICTIONS_FILE,
                              index=False)


def disarm(cfg) -> None:
    """Take the fold predictions away again: the project is shared."""
    from trackanno.learn.suggest import CV_PREDICTIONS_FILE

    (cfg.paths.work_dir / CV_PREDICTIONS_FILE).unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def no_stale_folds(cfg):
    """No test here may inherit another's planted predictions."""
    disarm(cfg)
    yield
    disarm(cfg)


# --------------------------------------------------------------------------- #
# what the mode queues
# --------------------------------------------------------------------------- #
def test_review_walks_what_was_labelled_in_the_order_it_was_labelled(
    cfg, store, root, app,
) -> None:
    """A review reads a session back, and a session happened in an order.

    The annotate queue is a seeded shuffle of the whole manifest, which is right
    for labelling and wrong for reading three labels back: it would scatter them
    through hundreds of clips nobody has touched.
    """
    widget, annotations, _ = build(cfg, store, root, "review-order")
    try:
        first = label(widget, app)
        second = label(widget, app)
        third = label(widget, app)
        check("three clips were labelled", len(annotations) == 3)

        set_mode(widget, app, "review")
        check("the queue is exactly the clips already labelled",
              len(widget.order) == 3)
        check("...in the order they were labelled",
              [widget._clip_ids[i] for i in widget.order] == [first, second, third])
        check("and it opens on the first of them", current(widget) == first)
        check("the counter says what it is counting",
              "reviewing 1 / 3 labelled" in widget.counter.text(),
              widget.counter.text())

        widget.next_clip()
        app.processEvents()
        check("next moves through the session", current(widget) == second)

        set_mode(widget, app, "annotate")
        check("going back to annotate restores the working queue",
              len(widget.order) == len(store))
    finally:
        widget.close()
        app.processEvents()


def test_review_with_nothing_labelled_says_so_rather_than_opening_blank(
    cfg, store, root, app,
) -> None:
    """An empty review is a normal state on a project nobody has started."""
    widget, _, _ = build(cfg, store, root, "review-empty")
    try:
        set_mode(widget, app, "review")
        check("the queue is empty", widget.order == [])
        check("and the screen says why, and what to do instead",
              "nothing labelled here yet" in widget.counter.text(),
              widget.counter.text())
    finally:
        widget.close()
        app.processEvents()


# --------------------------------------------------------------------------- #
# labels that might be wrong
# --------------------------------------------------------------------------- #
def test_the_note_says_what_the_held_out_model_said_instead(
    cfg, store, root, app,
) -> None:
    """A wrong label is what teaches the next model to be wrong, so the review
    says where a held-out model doubts it."""
    widget, annotations, _ = build(cfg, store, root, "disagree-note")
    try:
        group = cfg.categories[0]
        first = label(widget, app, option=0)
        other = group.options[-1].name
        check("there are two values to disagree about",
              other != group.options[0].name)

        arm(widget, {first: {group.name: (other, 0.91)}})
        set_mode(widget, app, "review")
        check("it opens on the label", current(widget) == first)

        note = widget.review_note.text()
        check("the note says what the held-out model said instead",
              "held out, the model said" in note and other in note, note)
        check("...and how sure it was of that", "91%" in note, note)
        check("...and that only the clip settles it",
              "which" in note.lower(), note)
    finally:
        widget.close()
        app.processEvents()


def test_changing_the_label_clears_the_disagreement_on_the_spot(
    cfg, store, root, app,
) -> None:
    """The point of the mode is to fix the label, so the fix has to show.

    Repainted against the buttons rather than against the row on disk: nothing
    is committed until the clip is left, and a note that only caught up on the
    next navigation would leave a reviewer unsure whether the key registered.
    """
    widget, annotations, path = build(cfg, store, root, "disagree-fix")
    try:
        group = cfg.categories[0]
        first = label(widget, app, option=0)
        other = group.options[-1].name
        arm(widget, {first: {group.name: (other, 0.88)}})
        set_mode(widget, app, "review")

        check("it is queued for review", len(widget.order) == 1)
        check("and the note says so",
              "held out, the model said" in widget.review_note.text(),
              widget.review_note.text())

        widget.categories.toggle(group.name, other)
        app.processEvents()
        check("the clip did not move", current(widget) == first)
        note = widget.review_note.text()
        check("the note now says the two agree",
              "agreed with this label" in note
              and "the model said" not in note, note)

        widget._commit()
        app.processEvents()
        check("and the label on disk is the one that was picked",
              annotations.get(first)[group.name] == other,
              str(annotations.get(first)))
        check("...rewritten, not added", len(rows_of(path)) == 1)
    finally:
        widget.close()
        app.processEvents()


def test_the_mode_control_is_labelled_and_offers_both(cfg, store, root, app) -> None:
    """Unlabelled, it sat at the far left reading `annotate` and was taken for
    part of the movie filter beside it."""
    from qtpy.QtWidgets import QLabel

    widget, _, _ = build(cfg, store, root, "disagree-labelled")
    try:
        check("both modes are offered",
              [widget.mode_combo.itemData(i)
               for i in range(widget.mode_combo.count())]
              == ["annotate", "review"],
              str([widget.mode_combo.itemText(i)
                   for i in range(widget.mode_combo.count())]))
        top = widget.layout().itemAt(0).layout()
        before = [top.itemAt(i).widget() for i in range(top.count())]
        index = before.index(widget.mode_combo)
        label_widget = before[index - 1]
        check("and the control has a word beside it saying what it is",
              isinstance(label_widget, QLabel) and label_widget.text() == "mode",
              type(label_widget).__name__)
    finally:
        widget.close()
        app.processEvents()


# --------------------------------------------------------------------------- #
# what a keypress does here
# --------------------------------------------------------------------------- #
def test_a_pick_while_reviewing_rewrites_the_label_and_stays_put(
    cfg, store, root, app,
) -> None:
    """Auto-advance is for labelling. Reviewing, the next thing somebody does
    after changing a label is look at the clip again, so the keypress that
    changed it must not take the clip away."""
    widget, annotations, path = build(cfg, store, root, "review-edit")
    try:
        group = cfg.categories[0]
        first = label(widget, app, option=0)
        label(widget, app, option=0)

        set_mode(widget, app, "review")
        check("auto-advance is on for this project", cfg.ui.auto_advance)
        check("the review opens on the first label", current(widget) == first)

        widget.categories.toggle(group.name, group.options[-1].name)
        app.processEvents()
        check("the clip did not move", current(widget) == first)

        widget.next_clip()
        app.processEvents()
        check("the edit was written on the way out",
              annotations.get(first)[group.name] == group.options[-1].name,
              str(annotations.get(first)))
        check("and it rewrote the row rather than adding one",
              len(rows_of(path)) == 2)
    finally:
        widget.close()
        app.processEvents()


def test_revising_a_label_keeps_the_reason_its_clip_was_shown(
    cfg, store, root, app,
) -> None:
    """The selector's reason is the one thing a revisit cannot reconstruct.

    ``selected_by`` defaults to *not recorded*, and nobody correcting a label
    months later has a selector to ask. Writing that default over the original
    would quietly turn every corrected row into one that cannot say whether the
    model chose it - and a review session corrects rows by design.
    """
    from trackanno.annotate.store import encode_provenance

    widget, annotations, path = build(cfg, store, root, "review-provenance")
    try:
        first = label(widget, app)
        reason = encode_provenance({"rule": "margin", "score": 0.9, "rank": 0})
        row = dict(annotations.get(first))
        row[PROVENANCE_COLUMN] = reason
        annotations.rows[first] = row
        annotations.rewrite()
        check("the row starts out knowing why it was shown",
              rows_of(path)[first][PROVENANCE_COLUMN] == reason)

        set_mode(widget, app, "review")
        group = cfg.categories[0]
        widget.categories.toggle(group.name, group.options[-1].name)
        widget._commit()
        app.processEvents()

        after = rows_of(path)[first]
        check("the label was revised", after[group.name] == group.options[-1].name)
        check("...and the reason it was shown survived the revision",
              after[PROVENANCE_COLUMN] == reason, after[PROVENANCE_COLUMN])
    finally:
        widget.close()
        app.processEvents()


def test_a_revision_keeps_what_the_model_said_at_the_time(cfg, store, root, app) -> None:
    """The suggestion columns describe the screen the first judgement was made
    under. Today's model is a different model, and rewriting them would claim a
    suggestion had been visible when none was."""
    widget, annotations, path = build(cfg, store, root, "review-suggestion")
    try:
        first = label(widget, app)
        before = rows_of(path)[first]
        check("nothing was suggested when the label was made",
              before["suggested"] == "" and before["suggest_shown"] == "0",
              f"{before['suggested']!r}/{before['suggest_shown']!r}")

        group = cfg.categories[0]
        set_mode(widget, app, "review")
        # AFTER the mode switch, which reloads the clip and would clear it: a
        # model that now has an opinion about this clip, as a later fit would.
        widget._current_suggestion = {group.name: (group.options[0].name, 0.91)}
        widget.categories.toggle(group.name, group.options[-1].name)
        widget._commit()
        app.processEvents()

        after = rows_of(path)[first]
        check("the label was revised", after[group.name] == group.options[-1].name)
        check("...and the row still says no suggestion was shown",
              after["suggested"] == "" and after["suggest_shown"] == "0",
              f"{after['suggested']!r}/{after['suggest_shown']!r}")
    finally:
        widget.close()
        app.processEvents()


def test_review_does_not_outline_the_model_that_was_fitted_on_the_label(
    cfg, store, root, app,
) -> None:
    """Two model opinions on one screen, and the prominent one would be the
    misleading one: the saved model reproduces a label it was trained on, while
    the note beside the clip carries the held-out answer."""
    from trackanno.gui.app import REVIEW_SUGGESTION_NOTE

    widget, annotations, path = build(cfg, store, root, "review-no-outline")
    try:
        group = cfg.categories[0]
        label(widget, app)
        widget._current_suggestion = {group.name: (group.options[0].name, 0.91)}

        widget.categories.set_suggestion(widget._current_suggestion, "")
        check("an outline is painted while annotating",
              widget.categories._suggested.get(group.name) == group.options[0].name)

        set_mode(widget, app, "review")
        check("but reviewing paints none",
              not widget.categories._suggested, widget.categories._suggested)
        check("...and says why, so the absence is a decision not a silence",
              widget.categories.suggestion.text() == REVIEW_SUGGESTION_NOTE,
              widget.categories.suggestion.text())
    finally:
        widget.close()
        app.processEvents()


# --------------------------------------------------------------------------- #
# what the screen says about a label
# --------------------------------------------------------------------------- #
def test_the_review_says_who_judged_the_clip_and_where_the_model_differs(
    cfg, store, root, app,
) -> None:
    """Two things a label alone does not carry: whose judgement it is, and
    whether anything now disagrees with it."""
    widget, annotations, _ = build(cfg, store, root, "review-note")
    try:
        group = cfg.categories[0]
        agreed = label(widget, app, option=0)
        disputed = label(widget, app, option=0)
        other = group.options[-1].name
        check("the project has two values to disagree about",
              other != group.options[0].name)

        arm(widget, {
            agreed: {group.name: (group.options[0].name, 0.9)},
            disputed: {group.name: (other, 0.8)},
        })

        set_mode(widget, app, "review")
        check("the note is only shown while reviewing",
              widget.review_note.isVisibleTo(widget))
        note = widget.review_note.text()
        check("it names who judged the clip",
              "labelled by" in note, note)
        check("it dates the judgement",
              "UTC" in note, note)
        check("a label the folds reproduced is not flagged",
              "the model said" not in note, note)

        widget.next_clip()
        app.processEvents()
        check("the review opened the disputed clip", current(widget) == disputed)
        note = widget.review_note.text()
        check("a label the folds contradicted is flagged, with their answer",
              "held out, the model said" in note and other in note, note)

        check("the disagreement is per group and per clip",
              widget._disagreement(disputed) == [f"{group.name}: {other}"]
              and widget._disagreement(agreed) == [],
              str(widget._disagreement(disputed)))
    finally:
        widget.close()
        app.processEvents()


def test_tab_jumps_to_the_next_label_the_model_argues_with(
    cfg, store, root, app,
) -> None:
    """Where to start a review. A label the model agrees with is evidence about
    neither of them; the ones it does not are either mistakes or the thing the
    model still has to learn."""
    widget, _, _ = build(cfg, store, root, "review-tab")
    try:
        group = cfg.categories[0]
        first = label(widget, app, option=0)
        label(widget, app, option=0)
        third = label(widget, app, option=0)
        other = group.options[-1].name

        arm(widget, {
            first: {group.name: (group.options[0].name, 0.9)},
            third: {group.name: (other, 0.8)},
        })

        set_mode(widget, app, "review")
        check("it opens on the first label", current(widget) == first)
        widget._on_next_gap()
        app.processEvents()
        check("Tab skips the labels the model accepts", current(widget) == third)

        widget._on_next_gap()
        app.processEvents()
        check("with none left it says so and stays put", current(widget) == third)

        widget._answers = {}
        widget.next_disagreement()
        app.processEvents()
        check("with no held-out predictions it stays put rather than finding "
              "nothing", current(widget) == third)
    finally:
        widget.close()
        app.processEvents()


def test_reviewing_does_not_move_where_the_next_session_resumes(
    cfg, store, root, app,
) -> None:
    """The resume point is where the work stopped. A detour back through the
    labels already done is not where to come back to."""
    from trackanno.gui.app import STATE_FILE

    widget, _, _ = build(cfg, store, root, "review-resume")
    state = cfg.paths.work_dir / STATE_FILE
    try:
        label(widget, app)
        label(widget, app)
        app.processEvents()
        working_on = state.read_text(encoding="utf-8")

        set_mode(widget, app, "review")
        widget.next_clip()
        app.processEvents()
        check("the review left the resume point alone",
              state.read_text(encoding="utf-8") == working_on,
              state.read_text(encoding="utf-8"))
    finally:
        widget.close()
        app.processEvents()
