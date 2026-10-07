"""The label distribution: tallies from the CSV and the summary lines over them."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def cfg(root: Path):
    from trackanno.config import load_config
    from trackanno.project import create_project

    return load_config(create_project(root / "dist"))


@pytest.fixture(scope="module")
def names(cfg):
    """The single group, the multi group, and option names from each."""
    single = cfg.categories[0]
    multi = next(g for g in cfg.categories if g.is_multi)
    a, b, c = (o.name for o in single.options)
    q0, q1 = (o.name for o in multi.options[:2])
    return single.name, multi.name, a, b, c, q0, q1


@pytest.fixture(scope="module")
def tallies(cfg, names):
    """Six a, two b with packed multi values, one renamed value, one blank."""
    from trackanno.gui.distribution import tally_annotations

    single, multi, a, b, _c, q0, q1 = names
    rows = (
        [{"clip_id": f"x{i}", "movie_id": "m1", single: a, multi: ""} for i in range(6)]
        + [{"clip_id": f"y{i}", "movie_id": "m2", single: b, multi: f"{q0}|{q1}"}
           for i in range(2)]
        + [{"clip_id": "z0", "movie_id": "m1", single: "renamed_away", multi: q0},
           {"clip_id": "z1", "movie_id": "m3", single: "", multi: ""}]
    )
    return tally_annotations(rows, cfg)


def _bars(tallies, group):
    return {bar.name: bar for bar in next(t for t in tallies if t.name == group).bars}


def test_every_option_gets_a_bar_with_its_count(tallies, names) -> None:
    single, _m, a, b, c, _q0, _q1 = names
    bars = _bars(tallies, single)
    assert {a, b, c} <= set(bars), sorted(bars)
    assert (bars[a].count, bars[b].count, bars[c].count) == (6, 2, 0)
    assert bars[a].n_movies == 1 and bars[b].n_movies == 1, "movies are counted per class"


def test_an_unknown_value_and_blank_cells_get_their_own_bars(tallies, names) -> None:
    from trackanno.gui.distribution import UNLABELLED

    bars = _bars(tallies, names[0])
    assert bars["renamed_away"].count == 1 and not bars["renamed_away"].known
    assert bars[UNLABELLED].count == 1


def test_multi_select_cells_are_split(tallies, names) -> None:
    _s, multi, *_rest, q0, q1 = names
    bars = _bars(tallies, multi)
    assert (bars[q0].count, bars[q1].count) == (3, 2)
    assert next(t for t in tallies if t.name == multi).total > 0


def test_the_summary_calls_out_the_awkward_cases(tallies, names, cfg) -> None:
    from trackanno.gui.distribution import summary_lines

    c = names[4]
    lines = " ".join(summary_lines(tallies, cfg))
    assert f"'{c}'" in lines, f"the empty class is called out: {lines}"
    assert "renamed_away" in lines, "the renamed value is called out"
    assert "only appears in one movie" in lines, lines
    assert "always answers" not in lines, f"ten labels are too few to call the balance: {lines}"


def test_only_a_labelled_group_is_called_lopsided(cfg, names) -> None:
    """An entirely unlabelled group makes no claim about its share."""
    from trackanno.gui.distribution import summary_lines, tally_annotations

    single, multi, a, b, *_ = names
    rows = (
        [{"clip_id": f"e{i}", "movie_id": f"m{i % 3}", single: a, multi: ""}
         for i in range(30)]
        + [{"clip_id": "e30", "movie_id": "m1", single: b, multi: ""}]
    )
    lines = summary_lines(tally_annotations(rows, cfg), cfg)
    assert not any("always answers" in ln and ln.startswith(multi) for ln in lines), lines
    assert any("always answers" in ln and ln.startswith(single) for ln in lines), lines


def test_a_broken_csv_is_reported_not_raised(app, root: Path) -> None:
    from trackanno.gui.hub import Hub, ModelPage
    from trackanno.project import create_project

    hub = Hub(str(create_project(root / "broken-dist")))
    hub.show()
    page = hub.page(ModelPage).distribution
    try:
        page.reload()
        assert "no annotation" in page.coverage.text().lower(), page.coverage.text()

        hub.cfg.paths.csv.parent.mkdir(parents=True, exist_ok=True)
        hub.cfg.paths.csv.write_bytes(b"clip_id,movie_id\n\xff\xfe not utf-8 \x00\n")
        if hub.status is not None:
            hub.status.n_annotated = 1
        page.reload()
        assert "could not read" in page.coverage.text(), page.coverage.text()
        assert not page._charts, "no stale charts are left behind"
    finally:
        hub.close()
