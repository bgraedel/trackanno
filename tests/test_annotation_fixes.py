"""Annotation screen: suggester swaps, review-mode undo and outline, refused commits."""

from __future__ import annotations

from pathlib import Path

import pytest

from trackanno.annotate.clips import ClipStore
from trackanno.annotate.store import AnnotationStore
from trackanno.config import load_config


@pytest.fixture(scope="module")
def cfg(root):
    from fixtures import make_project

    config = load_config(make_project(root))
    config.suggest.retrain_every = 50
    return config


@pytest.fixture(scope="module")
def store(cfg):
    store = ClipStore(cfg.paths.work_dir)
    yield store
    store.close()


class StubSuggester:
    """Suggests ``answer`` for every clip; records how it was built."""

    built: list["StubSuggester"] = []
    answer: dict = {}

    def __init__(self, cfg, labels_csv=None):
        self.cfg = cfg
        self.labels_csv = labels_csv
        self.ready = True
        self.problem = ""
        self.model_epoch = 1
        self.loads = 0
        self.warmed = False
        self.answer = dict(type(self).answer)
        StubSuggester.built.append(self)

    def load(self) -> bool:
        self.loads += 1
        return True

    def warm(self) -> None:
        self.warmed = True

    def suggest(self, clip_id):
        return dict(self.answer)

    def is_audit_clip(self, clip_id) -> bool:
        return False

    def status_line(self) -> str:
        return "stub"

    def selection_batch(self, manifest, done_ids, n) -> list[dict]:
        return []


@pytest.fixture
def widget(cfg, store, root, app, monkeypatch, request):
    from trackanno.gui.app import AnnotationWidget

    StubSuggester.built = []
    StubSuggester.answer = {}
    monkeypatch.setattr("trackanno.gui.app.Suggester", StubSuggester)
    path = root / f"{request.node.name}.csv"
    path.unlink(missing_ok=True)
    w = AnnotationWidget(cfg, store, AnnotationStore(path, cfg))
    yield w
    w.shutdown()
    w.close()
    app.processEvents()
    path.unlink(missing_ok=True)


def label(widget, app, option: int = 0) -> str:
    clip_id = widget._clip_ids[widget.order[widget.position]]
    group = widget.cfg.categories[0]
    widget.categories.toggle(group.name, group.options[option].name)
    app.processEvents()
    return clip_id


def set_mode(widget, app, mode: str) -> None:
    widget.mode_combo.setCurrentIndex(widget.mode_combo.findData(mode))
    app.processEvents()


def test_reload_paints_no_outline_while_reviewing(widget, app, cfg) -> None:
    from trackanno.gui.app import REVIEW_SUGGESTION_NOTE

    group = cfg.categories[0]
    StubSuggester.answer = {group.name: (group.options[1].name, 0.9)}
    label(widget, app)
    label(widget, app)
    set_mode(widget, app, "review")

    widget.reload_suggester()
    assert widget.categories._suggested == {}
    assert widget.categories.suggestion.text() == REVIEW_SUGGESTION_NOTE

    set_mode(widget, app, "annotate")
    widget.reload_suggester()
    assert widget.categories._suggested == {group.name: group.options[1].name}


def test_reload_keeps_the_note_for_an_empty_suggestion(widget, app, cfg,
                                                       monkeypatch) -> None:
    monkeypatch.setattr("trackanno.gui.app.AUDIT_VISIBLE", True)
    widget.reload_suggester()
    note = "the model has nothing to say about this clip"
    assert widget._suggestion_note == note
    assert widget.categories.suggestion.text() == note


def test_undo_in_review_reverts_the_pick_on_screen_first(widget, app, cfg) -> None:
    group = cfg.categories[0]
    first = label(widget, app, option=0)
    label(widget, app, option=0)
    annotations = widget.annotations
    set_mode(widget, app, "review")
    assert widget._clip_ids[widget.order[widget.position]] == first

    widget.categories.toggle(group.name, group.options[-1].name)
    app.processEvents()
    widget._undo()
    assert widget.categories.values()[group.name] == group.options[0].name
    assert widget._clip_ids[widget.order[widget.position]] == first
    assert len(annotations) == 2

    widget._undo()
    assert len(annotations) == 1


def test_a_refused_commit_at_the_last_clip_keeps_the_input(widget, app, cfg) -> None:
    optional = next(g for g in cfg.categories if not g.required)
    widget.position = len(widget.order) - 1
    widget._load()
    widget.categories.toggle(optional.name, optional.options[0].name)
    app.processEvents()

    widget.next_clip()
    assert widget.position == len(widget.order) - 1
    assert widget.categories.values()[optional.name] == optional.options[0].name
    assert len(widget.annotations) == 0


def test_a_retrained_suggester_is_built_off_thread_and_swapped_in(
    widget, app, cfg, monkeypatch
) -> None:
    import trackanno.learn.suggest as suggest

    trained: list[Path] = []
    monkeypatch.setattr(suggest, "train",
                        lambda c, **kw: trained.append(Path(c.paths.csv)))
    old = widget.suggester
    group = cfg.categories[0]
    StubSuggester.answer = {group.name: (group.options[2].name, 0.8)}
    label(widget, app)

    widget._labels_since_train = cfg.suggest.retrain_every
    widget._maybe_retrain()
    widget._retrain_thread.join(10)
    assert widget.suggester is old
    app.processEvents()

    new = widget.suggester
    assert new is not old
    assert old.loads == 0
    assert new.warmed
    assert new.labels_csv == trained[0]
    assert new.labels_csv != widget.annotations.path
    assert new.model_epoch > old.model_epoch
    assert widget._current_suggestion == StubSuggester.answer
    assert widget.categories._suggested == {group.name: group.options[2].name}


def test_a_retrain_overtaken_by_a_reload_is_dropped(widget, app, cfg, monkeypatch) -> None:
    import threading
    from copy import deepcopy

    import trackanno.learn.suggest as suggest

    release = threading.Event()
    monkeypatch.setattr(suggest, "train", lambda c, **kw: release.wait(10))
    label(widget, app)
    widget._labels_since_train = cfg.suggest.retrain_every
    widget._maybe_retrain()

    widget.reload_suggester()
    reloaded = widget.suggester
    release.set()
    widget._retrain_thread.join(10)
    app.processEvents()
    assert widget.suggester is reloaded
    assert len(StubSuggester.built) == 3       # startup, reload, the stale retrain

    widget._labels_since_train = cfg.suggest.retrain_every
    widget._maybe_retrain()
    widget.apply_view_settings(deepcopy(cfg))   # equal, so the retrain stands
    widget._retrain_thread.join(10)
    app.processEvents()
    assert widget.suggester is not reloaded
    adopted = widget.suggester

    edited = deepcopy(cfg)
    edited.suggest.min_confidence = 0.5
    widget._labels_since_train = cfg.suggest.retrain_every
    widget._maybe_retrain()
    widget.apply_view_settings(edited)
    widget._retrain_thread.join(10)
    app.processEvents()
    assert widget.suggester is adopted
    widget.apply_view_settings(cfg)


def test_selection_scores_the_pool_once_per_model(root: Path) -> None:
    from test_embeddings import synthetic_project
    from trackanno.learn.suggest import Suggester, train
    from trackanno.measure.clipfeatures import build_clip_features

    cfg = load_config(synthetic_project(root, "select-cache",
                                        {"suggest.select_rule": "margin",
                                         "suggest.audit_fraction": 0.1}))
    train(cfg, verbose=False)
    suggester = Suggester(cfg)
    assert suggester.ready, suggester.problem
    model = suggester.models[suggester.selection_group()]
    calls: list[int] = []
    real = model.predict_proba
    model.predict_proba = lambda X: (calls.append(len(X)), real(X))[1]

    manifest = build_clip_features(cfg, verbose=False, write=False)[["clip_id"]]
    suggester.warm()
    first = suggester.selection_batch(manifest, set(), 5)
    second = suggester.selection_batch(manifest.iloc[::2], set(), 5)
    assert len(calls) == 1
    assert len(first) == 5 and len(second) == 5

    suggester.load()
    assert suggester.selection_batch(manifest, set(), 5) == first
