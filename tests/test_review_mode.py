"""Review mode of the annotation screen: queue, rewrites and disagreement notes."""

from __future__ import annotations

import csv

import pytest

from trackanno.annotate.clips import ClipStore
from trackanno.annotate.store import PROVENANCE_COLUMN, AnnotationStore
from trackanno.config import load_config


@pytest.fixture(scope="module")
def cfg(root):
    from fixtures import make_project

    config = load_config(make_project(root))
    # No test writes five labels, so no background refit fires mid-test.
    config.suggest.retrain_every = 5
    return config


@pytest.fixture(scope="module")
def store(cfg):
    """The clip cache, closed so a later suite can overwrite the mapped file."""
    store = ClipStore(cfg.paths.work_dir)
    yield store
    store.close()


def build(cfg, store, root, name: str):
    """A real widget over a CSV of its own."""
    from trackanno.gui.app import AnnotationWidget

    path = root / f"{name}.csv"
    path.unlink(missing_ok=True)
    annotations = AnnotationStore(path, cfg)
    return AnnotationWidget(cfg, store, annotations), annotations, path


@pytest.fixture
def screen(request, cfg, store, root, app):
    """A widget over a CSV named after the test, closed afterwards."""
    widget, annotations, path = build(cfg, store, root, request.node.name)
    yield widget, annotations, path
    widget.close()
    app.processEvents()


def label(widget, app, option: int = 0) -> str:
    """Label the clip on screen the way a keypress does; return its clip id."""
    clip_id = widget._clip_ids[widget.order[widget.position]]
    group = widget.cfg.categories[0]
    widget.categories.toggle(group.name, group.options[option].name)
    app.processEvents()
    return clip_id


def set_mode(widget, app, mode: str) -> None:
    widget.mode_combo.setCurrentIndex(widget.mode_combo.findData(mode))
    app.processEvents()


def current(widget) -> str:
    return widget._clip_ids[widget.order[widget.position]]


def rows_of(path) -> dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["clip_id"]: r for r in csv.DictReader(fh)}


def arm(widget, answers) -> None:
    """Write the out-of-fold predictions file a training run leaves behind."""
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
    """Remove the planted fold predictions from the shared project."""
    from trackanno.learn.suggest import CV_PREDICTIONS_FILE

    (cfg.paths.work_dir / CV_PREDICTIONS_FILE).unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def no_stale_folds(cfg):
    disarm(cfg)
    yield
    disarm(cfg)


# --------------------------------------------------------------------------- #
# what the mode queues
# --------------------------------------------------------------------------- #
def test_review_walks_what_was_labelled_in_the_order_it_was_labelled(
    screen, store, app,
) -> None:
    widget, annotations, _ = screen
    first = label(widget, app)
    second = label(widget, app)
    third = label(widget, app)
    assert len(annotations) == 3

    set_mode(widget, app, "review")
    assert [widget._clip_ids[i] for i in widget.order] == [first, second, third], (
        "the queue is the labelled clips in labelling order")
    assert current(widget) == first
    assert "reviewing 1 / 3 labelled" in widget.counter.text(), widget.counter.text()

    widget.next_clip()
    app.processEvents()
    assert current(widget) == second, "next moves through the session"

    set_mode(widget, app, "annotate")
    assert len(widget.order) == len(store), "annotate restores the working queue"


def test_review_with_nothing_labelled_says_so(screen, app) -> None:
    widget, _, _ = screen
    set_mode(widget, app, "review")
    assert widget.order == []
    assert "nothing labelled here yet" in widget.counter.text(), widget.counter.text()


def test_the_mode_control_is_labelled_and_offers_both(screen) -> None:
    from qtpy.QtWidgets import QLabel

    widget, _, _ = screen
    modes = [widget.mode_combo.itemData(i) for i in range(widget.mode_combo.count())]
    assert modes == ["annotate", "review"]
    top = widget.layout().itemAt(0).layout()
    before = [top.itemAt(i).widget() for i in range(top.count())]
    label_widget = before[before.index(widget.mode_combo) - 1]
    assert isinstance(label_widget, QLabel) and label_widget.text() == "mode"


# --------------------------------------------------------------------------- #
# labels that might be wrong
# --------------------------------------------------------------------------- #
def test_the_note_says_what_the_held_out_model_said_instead(cfg, screen, app) -> None:
    widget, _, _ = screen
    group = cfg.categories[0]
    first = label(widget, app, option=0)
    other = group.options[-1].name
    assert other != group.options[0].name, "there are two values to disagree about"

    arm(widget, {first: {group.name: (other, 0.91)}})
    set_mode(widget, app, "review")
    assert current(widget) == first

    note = widget.review_note.text()
    assert "held out, the model said" in note and other in note, note
    assert "91%" in note, f"the note gives the confidence: {note}"
    assert "which" in note.lower(), f"the note says only the clip settles it: {note}"


def test_changing_the_label_clears_the_disagreement_on_the_spot(cfg, screen, app) -> None:
    """The note repaints from the buttons, before anything is committed."""
    widget, annotations, path = screen
    group = cfg.categories[0]
    first = label(widget, app, option=0)
    other = group.options[-1].name
    arm(widget, {first: {group.name: (other, 0.88)}})
    set_mode(widget, app, "review")

    assert len(widget.order) == 1
    assert "held out, the model said" in widget.review_note.text()

    widget.categories.toggle(group.name, other)
    app.processEvents()
    assert current(widget) == first, "the clip did not move"
    note = widget.review_note.text()
    assert "agreed with this label" in note and "the model said" not in note, note

    widget._commit()
    app.processEvents()
    assert annotations.get(first)[group.name] == other, annotations.get(first)
    assert len(rows_of(path)) == 1, "rewritten, not added"


def test_the_review_says_who_judged_the_clip_and_where_the_model_differs(
    cfg, screen, app,
) -> None:
    widget, _, _ = screen
    group = cfg.categories[0]
    agreed = label(widget, app, option=0)
    disputed = label(widget, app, option=0)
    other = group.options[-1].name
    assert other != group.options[0].name

    arm(widget, {
        agreed: {group.name: (group.options[0].name, 0.9)},
        disputed: {group.name: (other, 0.8)},
    })

    set_mode(widget, app, "review")
    assert widget.review_note.isVisibleTo(widget)
    note = widget.review_note.text()
    assert "labelled by" in note, note
    assert "UTC" in note, note
    assert "the model said" not in note, "a label the folds reproduced is not flagged"

    widget.next_clip()
    app.processEvents()
    assert current(widget) == disputed
    note = widget.review_note.text()
    assert "held out, the model said" in note and other in note, note
    assert widget._disagreement(disputed) == [f"{group.name}: {other}"]
    assert widget._disagreement(agreed) == []


def test_tab_jumps_to_the_next_label_the_model_argues_with(cfg, screen, app) -> None:
    widget, _, _ = screen
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
    assert current(widget) == first
    widget._on_next_gap()
    app.processEvents()
    assert current(widget) == third, "Tab skips the labels the model accepts"

    widget._on_next_gap()
    app.processEvents()
    assert current(widget) == third, "with none left it stays put"

    widget._answers = {}
    widget.next_disagreement()
    app.processEvents()
    assert current(widget) == third, "with no held-out predictions it stays put"


# --------------------------------------------------------------------------- #
# what a keypress does here
# --------------------------------------------------------------------------- #
def test_a_pick_while_reviewing_rewrites_the_label_and_stays_put(cfg, screen, app) -> None:
    """Auto-advance applies to labelling only."""
    widget, annotations, path = screen
    group = cfg.categories[0]
    first = label(widget, app, option=0)
    label(widget, app, option=0)

    set_mode(widget, app, "review")
    assert cfg.ui.auto_advance, "auto-advance is on for this project"
    assert current(widget) == first

    widget.categories.toggle(group.name, group.options[-1].name)
    app.processEvents()
    assert current(widget) == first, "the clip did not move"

    widget.next_clip()
    app.processEvents()
    assert annotations.get(first)[group.name] == group.options[-1].name, (
        "the edit is written on leaving the clip")
    assert len(rows_of(path)) == 2, "it rewrote the row rather than adding one"


def test_revising_a_label_keeps_the_reason_its_clip_was_shown(cfg, screen, app) -> None:
    """A revision never overwrites ``selected_by`` with the not-recorded default."""
    from trackanno.annotate.store import encode_provenance

    widget, annotations, path = screen
    first = label(widget, app)
    reason = encode_provenance({"rule": "margin", "score": 0.9, "rank": 0})
    row = dict(annotations.get(first))
    row[PROVENANCE_COLUMN] = reason
    annotations.rows[first] = row
    annotations.rewrite()
    assert rows_of(path)[first][PROVENANCE_COLUMN] == reason

    set_mode(widget, app, "review")
    group = cfg.categories[0]
    widget.categories.toggle(group.name, group.options[-1].name)
    widget._commit()
    app.processEvents()

    after = rows_of(path)[first]
    assert after[group.name] == group.options[-1].name, "the label was revised"
    assert after[PROVENANCE_COLUMN] == reason, after[PROVENANCE_COLUMN]


def test_a_revision_keeps_what_the_model_said_at_the_time(cfg, screen, app) -> None:
    """The suggestion columns describe the screen of the first judgement."""
    widget, _, path = screen
    first = label(widget, app)
    before = rows_of(path)[first]
    assert before["suggested"] == "" and before["suggest_shown"] == "0", before

    group = cfg.categories[0]
    set_mode(widget, app, "review")
    # Set after the mode switch, which reloads the clip and would clear it.
    widget._current_suggestion = {group.name: (group.options[0].name, 0.91)}
    widget.categories.toggle(group.name, group.options[-1].name)
    widget._commit()
    app.processEvents()

    after = rows_of(path)[first]
    assert after[group.name] == group.options[-1].name, "the label was revised"
    assert after["suggested"] == "" and after["suggest_shown"] == "0", (
        f"the row still says no suggestion was shown: {after}")


def test_review_does_not_outline_the_model_that_was_fitted_on_the_label(
    cfg, screen, app,
) -> None:
    """The saved model was trained on the label, so review shows no outline."""
    from trackanno.gui.app import REVIEW_SUGGESTION_NOTE

    widget, _, _ = screen
    group = cfg.categories[0]
    label(widget, app)
    widget._current_suggestion = {group.name: (group.options[0].name, 0.91)}

    widget.categories.set_suggestion(widget._current_suggestion, "")
    assert widget.categories._suggested.get(group.name) == group.options[0].name, (
        "an outline is painted while annotating")

    set_mode(widget, app, "review")
    assert not widget.categories._suggested, "reviewing paints none"
    assert widget.categories.suggestion.text() == REVIEW_SUGGESTION_NOTE


def test_reviewing_does_not_move_where_the_next_session_resumes(cfg, screen, app) -> None:
    from trackanno.gui.app import STATE_FILE

    widget, _, _ = screen
    state = cfg.paths.work_dir / STATE_FILE
    label(widget, app)
    label(widget, app)
    app.processEvents()
    working_on = state.read_text(encoding="utf-8")

    set_mode(widget, app, "review")
    widget.next_clip()
    app.processEvents()
    assert state.read_text(encoding="utf-8") == working_on
