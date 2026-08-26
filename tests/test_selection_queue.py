"""What the annotation queue does once a selector starts choosing clips.

A separate suite from ``test_gui_headless``, which is deliberately one long
scenario whose every step depends on the state the step before it left. These
claims are orthogonal to that: they need a stubbed ``Suggester.selection_batch``
and a fresh CSV per scenario, and threading them through the existing narrative
would make both harder to read than either is now.

The stub is the point rather than a shortcut. ``selection_batch`` is built
separately and needs a trained model to say anything, so stubbing it is what
makes these tests statements about the QUEUE - where a batch lands relative to a
forward-only cursor, and whose reason ends up in the CSV - instead of statements
about a classifier.
"""

from __future__ import annotations

import csv
import random

import pytest

from harness import check
from trackanno.clips import ClipStore
from trackanno.config import load_config
from trackanno.select import AUDIT_RULE
from trackanno.store import (
    PROVENANCE_COLUMN, AnnotationStore, decode_provenance, encode_provenance,
)


@pytest.fixture(scope="module")
def cfg(root):
    """A generated project, never the repository's own."""
    from fixtures import make_project

    config = load_config(make_project(root))
    # How many labels a scenario may write before a background refit would fire
    # and make the queue depend on a real model. Nothing here writes five.
    config.suggest.retrain_every = 5
    return config


@pytest.fixture(scope="module")
def store(cfg):
    """The clip cache, closed on the way out.

    `make_project` hands every suite the same generated project, so the memory
    maps opened here are maps of the file a later suite re-runs detection over -
    and on Windows a mapped file cannot be overwritten at all. Dropping the
    reference and leaving it to the collector makes that a matter of GC timing.
    """
    store = ClipStore(cfg.paths.work_dir)
    yield store
    store.close()


def selection(clip_id: str, rank: int, rule: str = "margin", **extra) -> dict:
    """One ``select.select_batch`` record, in its real shape."""
    out = {
        "clip_id": str(clip_id), "index": rank, "rank": rank, "rule": rule,
        "score": 0.9 - 0.1 * rank, "target_class": "",
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
    """The order the GUI has always produced: the seeded shuffle of the manifest."""
    order = list(store.manifest.index)
    if cfg.ui.shuffle:
        random.Random(cfg.ui.shuffle_seed).shuffle(order)
    return order


def build(cfg, store, root, name: str):
    """A real widget over a CSV of its own."""
    from trackanno.gui.app import AnnotationWidget

    path = root / f"{name}.csv"
    path.unlink(missing_ok=True)
    annotations = AnnotationStore(path, cfg)
    return AnnotationWidget(cfg, store, annotations), annotations, path


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
    app, cfg, store, root
) -> None:
    """No selection configured must be indistinguishable from no selector at all.

    Both halves matter. A queue that differs by one position has re-randomised
    somebody's session for no reason, and a `selected_by` cell that is anything
    but empty claims a sample was drawn a way it was not - the store spells the
    empty cell as NOT RECORDED precisely so that nothing has to guess.
    """
    widget, annotations, path = build(cfg, store, root, "default")
    app.processEvents()

    check("the real suggester is in place", type(widget.suggester).__name__ == "Suggester")
    check("selection is off out of the box", not cfg.suggest.selection_on,
          cfg.suggest.select_rule)
    check("no model is trained in this project", not widget.suggester.ready)
    check("so the selector has nothing to say",
          widget.suggester.selection_batch(store.manifest, set(), 5) == [])
    check("the queue is the seeded shuffle, unchanged",
          widget.order == shuffled_baseline(cfg, store),
          f"{widget.order[:6]} vs {shuffled_baseline(cfg, store)[:6]}")
    check("nothing has been served", widget._served == {})
    check("nothing is claimed about how a clip was chosen",
          widget._provenance_for(widget._clip_ids[widget.order[0]]) is None)

    clip_id = label(widget, app)
    check("the label was written", annotations.is_done(clip_id))
    row = rows_of(path)[clip_id]
    check("the provenance column exists", PROVENANCE_COLUMN in row)
    check("and it says NOT RECORDED, not 'chosen at random'",
          row[PROVENANCE_COLUMN] == "", repr(row[PROVENANCE_COLUMN]))
    check("which decodes to the empty answer",
          decode_provenance(row[PROVENANCE_COLUMN]) == {})

    widget.shutdown()
    path.unlink(missing_ok=True)


def test_a_batch_leads_the_queue_and_leaves_everything_else_in_place(
    app, cfg, store, root, monkeypatch
) -> None:
    """The picks open the session; the rest of the shuffle keeps its own order.

    The second half is what stops a batch from re-randomising the queue: a clip
    the selector said nothing about must be exactly where the shuffle left it
    relative to its neighbours, or every filter change becomes a reshuffle.
    """
    monkeypatch.setattr("trackanno.gui.app.Suggester", StubSuggester)
    widget, annotations, path = build(cfg, store, root, "leads")
    app.processEvents()

    baseline = list(widget.order)
    check("an empty batch leaves the shuffle alone",
          baseline == shuffled_baseline(cfg, store))

    # The LAST two clips of the shuffle, in reverse, so leading the queue is a
    # real move and a hook that quietly did nothing would fail this.
    picked = [baseline[-1], baseline[-2]]
    check("the picks are not where the shuffle already put them",
          baseline[:2] != picked)
    widget.suggester.batch = [
        selection(widget._clip_ids[i], rank) for rank, i in enumerate(picked)
    ]
    widget._apply_filters(initial=True)
    app.processEvents()

    check("the picks lead the queue, in rank order",
          widget.order[:2] == picked, f"{widget.order[:2]} vs {picked}")
    check("the cursor opens on the top-ranked pick", widget.position == 0)
    check("every clip is still in the queue exactly once",
          sorted(widget.order) == sorted(baseline))
    check("everything else keeps its relative order",
          [i for i in widget.order if i not in picked]
          == [i for i in baseline if i not in picked])
    check("each pick's reason was kept",
          all(widget._clip_ids[i] in widget._served for i in picked))

    check("the selector was offered the reachable clips and only those",
          widget.suggester.pools[-1] == [widget._clip_ids[i] for i in baseline])
    check("and asked for the configured batch size",
          widget.suggester.sizes[-1] == cfg.suggest.select_batch_size,
          str(widget.suggester.sizes[-1]))

    widget.shutdown()
    path.unlink(missing_ok=True)


def test_the_csv_separates_the_selector_s_picks_from_the_annotator_s_own_detours(
    app, cfg, store, root, monkeypatch
) -> None:
    """The distinction the column exists for, driven through the real GUI path.

    Labelling a pick must record the rule and score that chose it. Labelling a
    clip the selector never named must record that a human went and got it - the
    weaker claim, and a true one - because a labelled set that cannot be split
    into those two groups is a set on which no accuracy can be honestly computed.
    """
    monkeypatch.setattr("trackanno.gui.app.Suggester", StubSuggester)
    widget, annotations, path = build(cfg, store, root, "provenance")
    app.processEvents()

    from trackanno.gui.app import MANUAL_RULE

    baseline = list(widget.order)
    picked = [baseline[-1], baseline[-2]]
    widget.suggester.batch = [
        selection(widget._clip_ids[picked[0]], 0, rule="margin", score=0.83),
        # A reserved random draw. `selection_batch` hands its scoreless score
        # over as None, having already turned `select`'s NaN into the one JSON
        # value that means the same thing and survives the round trip.
        selection(widget._clip_ids[picked[1]], 1, rule=AUDIT_RULE,
                  score=None, target_class=""),
    ]
    widget._apply_filters(initial=True)
    app.processEvents()

    ranked_id = label(widget, app)             # auto-advance moves to the draw
    drawn_id = label(widget, app)              # and then off the end of the batch
    manual_id = label(widget, app)

    check("all three were written", all(annotations.is_done(c) for c in
                                        (ranked_id, drawn_id, manual_id)))
    check("the third clip was not one of the picks",
          manual_id not in {widget._clip_ids[i] for i in picked})

    rows = rows_of(path)
    ranked = decode_provenance(rows[ranked_id][PROVENANCE_COLUMN])
    check("a pick records the rule that chose it", ranked["rule"] == "margin",
          repr(ranked))
    check("and the score that earned it", ranked["score"] == pytest.approx(0.83))
    check("and its rank in the batch", ranked["rank"] == 0)
    check("and the pool it was chosen from", ranked["pool_size"] == 120)

    drawn = decode_provenance(rows[drawn_id][PROVENANCE_COLUMN])
    check("a random draw records itself as one", drawn["rule"] == AUDIT_RULE)
    check("with no score rather than an invented zero",
          drawn["score"] is None, repr(drawn))

    manual = decode_provenance(rows[manual_id][PROVENANCE_COLUMN])
    check("a clip the annotator navigated to says so",
          manual["rule"] == MANUAL_RULE, repr(manual))
    check("and never borrows the selector's reason",
          manual["rule"] not in {"margin", AUDIT_RULE})

    widget.shutdown()
    path.unlink(missing_ok=True)


def test_a_refresh_re_queues_behind_the_cursor_and_never_under_it(
    app, cfg, store, root, monkeypatch
) -> None:
    """A refit landing mid-session must not move the clip being judged.

    Navigation is forward-only, so the two failures are symmetric: a pick placed
    BEFORE the cursor is unreachable forever, and a queue rebuilt AROUND the
    cursor yanks away the clip somebody is halfway through reading. The picks
    therefore go immediately after the clip on screen, which is the earliest slot
    that is both reachable and not the one in use.
    """
    monkeypatch.setattr("trackanno.gui.app.Suggester", StubSuggester)
    widget, annotations, path = build(cfg, store, root, "refresh")
    app.processEvents()

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
    check("the cursor walked into the batch", widget.position == 1)

    # The next batch deliberately re-picks a clip the cursor has already passed
    # and one further down. The first is the unreachable case, and it is only
    # unreachable because navigation never goes backwards on its own. Neither is
    # the clip that will be on screen - that one is tested separately below.
    behind, ahead = queued[0], queued[3]
    widget.suggester.batch = [
        selection(widget._clip_ids[behind], 0, score=0.95),
        selection(widget._clip_ids[ahead], 1, score=0.91),
    ]
    labelled = label(widget, app)
    check("a label alone does not re-rank the queue", widget.order == queued,
          f"{widget.order} vs {queued}")
    check("the label was still written", annotations.is_done(labelled))
    check("and auto-advance left the cursor one further on", widget.position == 2)

    on_screen = widget._clip_ids[widget.order[widget.position]]
    widget.model_refreshed.emit()
    app.processEvents()

    check("the clip on screen is the same clip",
          widget._clip_ids[widget.order[widget.position]] == on_screen,
          f"{widget._clip_ids[widget.order[widget.position]]} vs {on_screen}")
    check("the fresh picks sit immediately after it",
          widget.order[widget.position + 1:widget.position + 3] == [behind, ahead],
          f"{widget.order} at {widget.position}")
    check("a pick the cursor had passed was brought back within reach",
          widget.order.index(behind) > widget.position)
    check("no clip was lost or duplicated by the re-queue",
          sorted(widget.order) == sorted(baseline))

    # And forward-only navigation really does reach them.
    widget.next_clip()
    app.processEvents()
    check("next lands on the top fresh pick",
          widget.order[widget.position] == behind)
    requeued_id = label(widget, app)
    check("which records the reason from the batch that re-queued it",
          decode_provenance(rows_of(path)[requeued_id][PROVENANCE_COLUMN])["score"]
          == pytest.approx(0.95))

    widget.shutdown()
    path.unlink(missing_ok=True)


def test_the_clip_already_on_screen_is_never_claimed_by_the_selector(
    app, cfg, store, root, monkeypatch
) -> None:
    """A pick the annotator was already looking at is not the selector's doing.

    The selector scores every unlabelled clip, so the one on screen is a
    candidate like any other and a refit can perfectly well name it. But it was
    put there by the queue that came before, not by this batch, and writing this
    batch's rule on it would claim the row entered the labelled set because the
    model asked for it. The weaker claim is the true one.
    """
    monkeypatch.setattr("trackanno.gui.app.Suggester", StubSuggester)
    widget, annotations, path = build(cfg, store, root, "anchor")
    app.processEvents()

    from trackanno.gui.app import MANUAL_RULE

    baseline = list(widget.order)
    anchor = widget.order[widget.position]
    on_screen = widget._clip_ids[anchor]
    other = baseline[-1]
    check("the two are different clips", other != anchor)

    widget.suggester.batch = [
        selection(on_screen, 0, score=0.99),
        selection(widget._clip_ids[other], 1, score=0.98),
    ]
    widget.model_refreshed.emit()
    app.processEvents()

    check("the clip on screen stayed on screen",
          widget.order[widget.position] == anchor)
    check("and was not also queued a second time",
          widget.order.count(anchor) == 1)
    check("the other pick landed immediately after it",
          widget.order[widget.position + 1] == other, str(widget.order))
    check("a batch really was served", widget._served != {})
    check("but the selector's reason was not attached to the anchor",
          on_screen not in widget._served)

    check("labelling it labels the same clip", label(widget, app) == on_screen)
    recorded = decode_provenance(rows_of(path)[on_screen][PROVENANCE_COLUMN])
    check("and records the weaker claim, that a human was already there",
          recorded["rule"] == MANUAL_RULE, repr(recorded))

    widget.shutdown()
    path.unlink(missing_ok=True)


def test_a_selector_that_raises_is_reported_and_the_shuffle_carries_on(
    app, cfg, store, root, monkeypatch
) -> None:
    """A broken selector costs the ranking, never the ability to annotate.

    Said out loud, because a selector that silently stops selecting is one nobody
    ever fixes - and every row written afterwards would carry a reason that had
    quietly stopped being true.
    """
    monkeypatch.setattr("trackanno.gui.app.Suggester", StubSuggester)
    widget, annotations, path = build(cfg, store, root, "broken")
    app.processEvents()
    baseline = list(widget.order)

    said: list[str] = []
    widget.status_message.connect(lambda text, _timeout: said.append(text))

    def explode(manifest, done_ids, n):
        raise ValueError("the candidate mask excludes all clips")

    widget.suggester.selection_batch = explode
    widget._apply_filters(initial=True)
    app.processEvents()

    check("the queue falls back to the shuffle", widget.order == baseline)
    check("and the annotator is told why",
          any("clip selection is off" in m for m in said), repr(said))
    check("a clip labelled now claims nothing about its selection",
          widget._provenance_for(widget._clip_ids[widget.order[0]]) is None)

    # A record the CSV could never hold back is caught HERE, where it costs one
    # batch, and not at the save of a label somebody has just typed. A NaN score
    # is not one of those: the codec writes it as `null`, because a clip drawn at
    # random earned no score and 0.0 would say the model was certain.
    widget.suggester.selection_batch = StubSuggester.selection_batch.__get__(
        widget.suggester
    )
    for what, bad in (
        ("a nameless rule", {"clip_id": widget._clip_ids[baseline[0]], "rule": ""}),
        ("a score json cannot spell", selection(widget._clip_ids[baseline[0]], 0,
                                                score=object())),
    ):
        said.clear()
        widget.suggester.batch = [bad]
        widget._apply_filters(initial=True)
        app.processEvents()
        check(f"{what} is refused before it can reach the log",
              widget.order == baseline
              and any("clip selection is off" in m for m in said), repr(said))

    said.clear()
    first = widget._clip_ids[baseline[0]]
    widget.suggester.batch = [selection(first, 0, rule="audit", score=float("nan"))]
    widget._apply_filters(initial=True)
    app.processEvents()
    check("a score nothing measured costs no batch, the cell being able to hold it",
          not any("clip selection is off" in m for m in said), repr(said))
    check("...and reaches the log as no value rather than as a score of zero",
          decode_provenance(
              encode_provenance(widget._provenance_for(first))
          )["score"] is None,
          str(widget._provenance_for(first)))

    widget.shutdown()
    path.unlink(missing_ok=True)
