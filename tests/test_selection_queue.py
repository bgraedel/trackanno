"""The annotation queue with a stubbed selector: batch placement and CSV provenance."""

from __future__ import annotations

import csv
import random

import pytest

from trackanno.annotate.clips import ClipStore
from trackanno.config import load_config
from trackanno.gui.app import SELECT_BATCH_SIZE
from trackanno.annotate.store import (
    PROVENANCE_COLUMN, AnnotationStore, decode_provenance,
)


@pytest.fixture(scope="module")
def cfg(root):
    from fixtures import make_project

    config = load_config(make_project(root))
    # No scenario writes five labels, so no background refit fires.
    config.suggest.retrain_every = 5
    return config


@pytest.fixture(scope="module")
def store(cfg):
    """The clip cache, closed explicitly so Windows releases the memory maps."""
    store = ClipStore(cfg.paths.work_dir)
    yield store
    store.close()


def selection(clip_id: str, rank: int, rule: str = "margin", **extra) -> dict:
    """One ``select.select_batch`` record, in its real shape."""
    out = {
        "clip_id": str(clip_id), "index": rank, "rank": rank, "rule": rule,
        "score": 0.9 - 0.1 * rank,
        "predicted_class": "swimming", "predicted_proba": 0.55, "pool_size": 120,
    }
    out.update(extra)
    return out


class StubSuggester:
    """Answers only what the queue asks of a suggester, and remembers being asked."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.ready = False
        self.problem = ""
        self.batch: list[dict] = []
        self.pools: list[list[str]] = []       # the clip ids offered, per call
        self.sizes: list[int] = []             # the `n` asked for, per call

    def load(self) -> bool:
        return False

    def suggest(self, clip_id):
        return {}

    def is_audit_clip(self, clip_id) -> bool:
        return False

    def status_line(self) -> str:
        return "stub suggester"

    def selection_batch(self, manifest, done_ids, n) -> list[dict]:
        self.pools.append([str(c) for c in manifest["clip_id"]])
        self.sizes.append(int(n))
        return [dict(r) for r in self.batch if r["clip_id"] not in done_ids][:n]


def shuffled_baseline(cfg, store) -> list[int]:
    """The seeded shuffle of the manifest that the GUI serves without a selector."""
    order = list(store.manifest.index)
    if cfg.ui.shuffle:
        random.Random(0).shuffle(order)
    return order


@pytest.fixture
def build(app, cfg, store, root):
    """Factory for a real widget over a CSV of its own; shut down after the test."""
    from trackanno.gui.app import AnnotationWidget

    made = []

    def make(name: str):
        path = root / f"{name}.csv"
        path.unlink(missing_ok=True)
        annotations = AnnotationStore(path, cfg)
        widget = AnnotationWidget(cfg, store, annotations)
        made.append((widget, path))
        app.processEvents()
        return widget, annotations, path

    yield make
    for widget, path in made:
        widget.shutdown()
        path.unlink(missing_ok=True)


@pytest.fixture
def stubbed(monkeypatch):
    """Replace the real Suggester with ``StubSuggester`` for widgets built afterwards."""
    monkeypatch.setattr("trackanno.gui.app.Suggester", StubSuggester)


def rows_of(path) -> dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["clip_id"]: r for r in csv.DictReader(fh)}


def label(widget, app, option: int = 0) -> str:
    """Label the clip on screen the way a keypress does; return its clip id."""
    clip_id = widget._clip_ids[widget.order[widget.position]]
    group = widget.cfg.categories[0]
    widget.categories.toggle(group.name, group.options[option].name)
    app.processEvents()
    return clip_id


# --------------------------------------------------------------------------- #
def test_the_default_config_leaves_the_queue_and_the_csv_exactly_as_they_were(
    app, cfg, store, build
) -> None:
    """With no selection configured the queue is the shuffle and ``selected_by`` stays empty."""
    widget, annotations, path = build("default")

    assert type(widget.suggester).__name__ == "Suggester", "the real suggester is in place"
    assert not cfg.suggest.selection_on, (
        f"selection is off out of the box: {cfg.suggest.select_rule}")
    assert not widget.suggester.ready, "no model is trained in this project"
    assert widget.suggester.selection_batch(store.manifest, set(), 5) == [], (
        "the selector has nothing to say")
    assert widget.order == shuffled_baseline(cfg, store), (
        f"the queue is the seeded shuffle, unchanged: {widget.order[:6]} vs {shuffled_baseline(cfg, store)[:6]}")
    assert widget._served == {}, "nothing has been served"

    clip_id = label(widget, app)
    assert annotations.is_done(clip_id), "the label was written"
    row = rows_of(path)[clip_id]
    assert PROVENANCE_COLUMN in row, "the provenance column exists"
    assert row[PROVENANCE_COLUMN] == "", (
        f"it says NOT RECORDED, not 'chosen at random': {row[PROVENANCE_COLUMN]!r}")
    assert decode_provenance(row[PROVENANCE_COLUMN]) == {}, (
        "which decodes to the empty answer")


def test_a_batch_leads_the_queue_and_leaves_everything_else_in_place(
    app, cfg, store, build, stubbed
) -> None:
    """The picks open the session; the rest of the shuffle keeps its own order."""
    widget, annotations, path = build("leads")

    baseline = list(widget.order)
    assert baseline == shuffled_baseline(cfg, store), (
        "an empty batch leaves the shuffle alone")

    # The last two clips of the shuffle, reversed, so leading the queue is a real move.
    picked = [baseline[-1], baseline[-2]]
    assert baseline[:2] != picked, "the picks are not where the shuffle already put them"
    widget.suggester.batch = [
        selection(widget._clip_ids[i], rank) for rank, i in enumerate(picked)
    ]
    widget._apply_filters(initial=True)
    app.processEvents()

    assert widget.order[:2] == picked, (
        f"the picks lead the queue, in rank order: {widget.order[:2]} vs {picked}")
    assert widget.position == 0, "the cursor opens on the top-ranked pick"
    assert sorted(widget.order) == sorted(baseline), (
        "every clip is still in the queue exactly once")
    assert ([i for i in widget.order if i not in picked]
            == [i for i in baseline if i not in picked]), (
        "everything else keeps its relative order")
    assert all(widget._clip_ids[i] in widget._served for i in picked), (
        "each pick's reason was kept")

    assert widget.suggester.pools[-1] == [widget._clip_ids[i] for i in baseline], (
        "the selector was offered the reachable clips and only those")
    assert widget.suggester.sizes[-1] == SELECT_BATCH_SIZE, (
        f"asked for the configured batch size: {widget.suggester.sizes[-1]}")


def test_the_csv_separates_the_selector_s_picks_from_the_annotator_s_own_detours(
    app, cfg, store, build, stubbed
) -> None:
    """A pick records the rule and score that chose it; any other clip records nothing."""
    widget, annotations, path = build("provenance")

    baseline = list(widget.order)
    picked = [baseline[-1], baseline[-2]]
    widget.suggester.batch = [
        selection(widget._clip_ids[picked[0]], 0, score=0.83),
        selection(widget._clip_ids[picked[1]], 1, score=0.71),
    ]
    widget._apply_filters(initial=True)
    app.processEvents()

    ranked_id = label(widget, app)
    second_id = label(widget, app)
    manual_id = label(widget, app)

    assert all(annotations.is_done(c) for c in (ranked_id, second_id, manual_id))
    assert manual_id not in {widget._clip_ids[i] for i in picked}, (
        "the third clip was not one of the picks")

    rows = rows_of(path)
    ranked = decode_provenance(rows[ranked_id][PROVENANCE_COLUMN])
    assert ranked["rule"] == "margin", f"a pick records the rule that chose it: {ranked!r}"
    assert ranked["score"] == pytest.approx(0.83), "the score that earned it"
    assert ranked["rank"] == 0, "its rank in the batch"
    assert ranked["pool_size"] == 120, "the pool it was chosen from"
    assert decode_provenance(rows[second_id][PROVENANCE_COLUMN])["rank"] == 1, (
        "the second pick records its own rank")
    assert rows[manual_id][PROVENANCE_COLUMN] == "", (
        f"a clip the selector did not pick records nothing: {rows[manual_id]!r}")


def test_a_refresh_re_queues_behind_the_cursor_and_never_under_it(
    app, cfg, store, build, stubbed
) -> None:
    """A refit mid-session queues fresh picks right after the clip on screen."""
    widget, annotations, path = build("refresh")

    baseline = list(widget.order)
    first = [baseline[-1], baseline[-2]]
    widget.suggester.batch = [
        selection(widget._clip_ids[i], rank) for rank, i in enumerate(first)
    ]
    widget._apply_filters(initial=True)
    app.processEvents()
    queued = list(widget.order)

    widget.next_clip()
    app.processEvents()
    assert widget.position == 1, "the cursor walked into the batch"

    # Re-pick one clip the cursor has passed and one further down.
    behind, ahead = queued[0], queued[3]
    widget.suggester.batch = [
        selection(widget._clip_ids[behind], 0, score=0.95),
        selection(widget._clip_ids[ahead], 1, score=0.91),
    ]
    labelled = label(widget, app)
    assert widget.order == queued, (
        f"a label alone does not re-rank the queue: {widget.order} vs {queued}")
    assert annotations.is_done(labelled), "the label was still written"
    assert widget.position == 2, "auto-advance left the cursor one further on"

    on_screen = widget._clip_ids[widget.order[widget.position]]
    widget.model_refreshed.emit()
    app.processEvents()

    assert widget._clip_ids[widget.order[widget.position]] == on_screen, (
        "the clip on screen did not move")
    assert widget.order[widget.position + 1:widget.position + 3] == [behind, ahead], (
        f"the fresh picks sit immediately after it: {widget.order} at {widget.position}")
    assert widget.order.index(behind) > widget.position, (
        "a pick the cursor had passed was brought back within reach")
    assert sorted(widget.order) == sorted(baseline), (
        "no clip was lost or duplicated by the re-queue")

    widget.next_clip()
    app.processEvents()
    assert widget.order[widget.position] == behind, "next lands on the top fresh pick"
    requeued_id = label(widget, app)
    requeued = decode_provenance(rows_of(path)[requeued_id][PROVENANCE_COLUMN])
    assert requeued["score"] == pytest.approx(0.95), "the re-queuing batch's reason is kept"


def test_the_clip_already_on_screen_is_never_claimed_by_the_selector(
    app, cfg, store, build, stubbed
) -> None:
    """A pick that is already on screen does not get the selector's reason."""
    widget, annotations, path = build("anchor")

    baseline = list(widget.order)
    anchor = widget.order[widget.position]
    on_screen = widget._clip_ids[anchor]
    other = baseline[-1]
    assert other != anchor, "the two are different clips"

    widget.suggester.batch = [
        selection(on_screen, 0, score=0.99),
        selection(widget._clip_ids[other], 1, score=0.98),
    ]
    widget.model_refreshed.emit()
    app.processEvents()

    assert widget.order[widget.position] == anchor, "the clip on screen stayed on screen"
    assert widget.order.count(anchor) == 1, "was not also queued a second time"
    assert widget.order[widget.position + 1] == other, (
        f"the other pick landed immediately after it: {widget.order}")
    assert widget._served != {}, "a batch really was served"
    assert on_screen not in widget._served, (
        "the selector's reason was not attached to the anchor")

    assert label(widget, app) == on_screen, "labelling it labels the same clip"
    recorded = rows_of(path)[on_screen][PROVENANCE_COLUMN]
    assert recorded == "", f"records no selector reason: {recorded!r}"


def test_a_selector_that_raises_is_reported_and_the_shuffle_carries_on(
    app, cfg, store, build, stubbed
) -> None:
    """A broken selector is reported and the shuffle carries on."""
    widget, annotations, path = build("broken")
    baseline = list(widget.order)

    said: list[str] = []
    widget.status_message.connect(lambda text, _timeout: said.append(text))

    def explode(manifest, done_ids, n):
        raise ValueError("the candidate mask excludes all clips")

    widget.suggester.selection_batch = explode
    widget._apply_filters(initial=True)
    app.processEvents()

    assert widget.order == baseline, "the queue falls back to the shuffle"
    assert any("clip selection is off" in m for m in said), (
        f"the annotator is told why: {said!r}")
    assert widget._served == {}, "nothing was served"

