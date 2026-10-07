"""config.yaml as a form: what it exposes, how, and what it refuses."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

# Prediction keys the form has to offer and round-trip.
NEW_KEYS = {
    "predict.window_stride": 25,
    "predict.min_bout_frames": 10,
}


def _raw(values: dict) -> dict:
    """Dotted keys back into the nested shape ``ConfigForm.load`` reads."""
    out: dict = {}
    for dotted, value in values.items():
        section, _, key = dotted.partition(".")
        out.setdefault(section, {})[key] = value
    return out


def test_every_field_reaches_exactly_one_tab(app) -> None:
    """The page is split in two tabs, so a key could show twice or not at all."""
    from trackanno.gui.configform import FIELDS, GROUPS, ConfigForm

    defined = [f"{section}.{key}" for section, key, *_ in FIELDS]
    assert 25 <= len(defined) <= 32, f"the form stays short: {len(defined)}"
    assert all(tip for *_, tip, _e in FIELDS), "every field has help from the schema"
    assert len(defined) == len(set(defined)), (
        f"no field is defined twice: {sorted({d for d in defined if defined.count(d) > 1})}")

    basic, advanced = ConfigForm("basic"), ConfigForm("advanced")
    rendered = list(basic._widgets) + list(advanced._widgets)
    assert set(rendered) == set(defined), (
        f"every field defined reaches a tab: {sorted(set(defined) ^ set(rendered))}")
    assert len(rendered) == len(set(rendered)), "none reaches both"
    assert 5 < len(basic._widgets) < 20, (
        f"the basics tab is still the short list: {len(basic._widgets)}")
    assert len(ConfigForm("all")._widgets) == len(defined), "scope 'all' is both"

    # Every group is a heading plus one sentence saying what it is for.
    assert all(blurb.strip() and blurb.strip().endswith(".") for _t, blurb, _f in GROUPS), (
        f"every group says what it is for: {[t for t, b, _f in GROUPS if not b.strip().endswith('.')]}")
    assert all(fields for _t, _b, fields in GROUPS), "every group has fields in it"
    titles = [title for title, _b, _f in GROUPS]
    assert len(titles) == len(set(titles)), "no two groups share a title"

    # Keys the loader refuses over each other share one group.
    group_of = {dotted: title for title, _b, rows in GROUPS for dotted, *_ in rows}
    coupled = ["predict.window_stride", "predict.min_bout_frames"]
    assert len({group_of[k] for k in coupled}) == 1, (
        f"the keys that are refused over each other are one group: { {k: group_of[k] for k in coupled}}")


def test_the_prediction_keys_the_gui_can_break_are_on_it(app, root: Path) -> None:
    """The prediction keys are on the form and survive the real loader."""
    from trackanno.config import load_config
    from trackanno.gui.configform import ConfigForm
    from trackanno.project import create_project, write_config

    advanced = ConfigForm("advanced")
    assert all(key in advanced._widgets for key in NEW_KEYS), (
        f"all the keys are on a tab: {sorted(set(NEW_KEYS) - set(advanced._widgets))}")

    form = ConfigForm("all")
    form.load(_raw(NEW_KEYS))
    values = form.current()
    for dotted, value in NEW_KEYS.items():
        assert values[dotted] == value, (
            f"{dotted} round-trips through the form: {values.get(dotted)!r}")

    path = create_project(root / "predict-keys")
    coherent = {
        "clips.frames": 50, "clips.stride": 50,
        "predict.window_stride": 25, "predict.min_bout_frames": 25,
    }
    write_config(path, coherent)
    cfg = load_config(path)
    assert cfg.predict.resolved_stride(cfg.clips) == 25, (
        f"an overlapping project written from these fields loads: {cfg.predict.resolved_stride(cfg.clips)}")
    assert cfg.predict.min_bout_frames == 25, "every value survived the round trip"

    form = ConfigForm("all")
    form.load(yaml.safe_load(path.read_text(encoding="utf-8")))
    written = form.current()
    assert all(written[k] == v for k, v in coherent.items()), (
        f"the form reads back what it wrote: { {k: written[k] for k, v in coherent.items() if written[k] != v}}")


def test_a_refused_field_never_reaches_the_file(app, root: Path) -> None:
    """One unusable field stops the whole save rather than being dropped from it."""
    from trackanno.gui.hub import ConfigPage, Hub
    from trackanno.project import create_project

    path = create_project(root / "refusal")
    hub = Hub(str(path))
    hub.show()
    page = hub.page(ConfigPage)
    hub.go_to(hub.pages.index(page))
    app.processEvents()
    good = path.read_text(encoding="utf-8")

    stride = page.advanced._widgets["predict.window_stride"]
    stride.auto.setChecked(False)
    stride.edit.setValue(999)
    app.processEvents()
    assert not page.save() and not page.banner.isHidden(), "Save refuses it"
    assert "window_stride" in page.banner.text(), (
        f"the banner names the field: {page.banner.text()[:160]}")
    assert path.read_text(encoding="utf-8") == good, "the refused save left the file alone"

    stride.auto.setChecked(True)
    app.processEvents()
    assert page.save() and page.banner.isHidden(), "putting it back saves again"

    hub.close()
    app.processEvents()


@pytest.fixture
def default_form(app):
    """The form loaded from the shipped config, with 'auto' resolved."""
    from trackanno.gui.configform import ConfigForm
    from trackanno.project import DEFAULT_CONFIG

    form = ConfigForm("all")
    form.load(yaml.safe_load(DEFAULT_CONFIG))
    values = form.current()
    form.show_resolved(values)
    return form, values


def test_auto_says_what_it_resolves_to(default_form) -> None:
    form, values = default_form
    window = form._widgets["predict.window_stride"]
    assert window.auto.isChecked() and not window.edit.isEnabled(), "'auto' is a tick"
    assert window.auto.text() == "auto (50)", window.auto.text()
    assert form._widgets["clips.crop_size"].auto.text() == "auto", (
        "an auto with no answer yet says only 'auto'")
    form.show_resolved({**values, "clips.frames": 30, "clips.stride": 30})
    assert window.auto.text() == "auto (30)", "it follows clips.frames"


def test_the_clip_stride_says_whether_it_overlaps(default_form) -> None:
    form, values = default_form
    form.show_resolved({**values, "clips.frames": 50, "clips.stride": 25})
    assert "overlap" in form._status["clips.stride"].text()
    form.show_resolved({**values, "clips.frames": 50, "clips.stride": 50})
    assert "tile" in form._status["clips.stride"].text()


def test_leaving_auto_starts_from_the_number_it_stood_for(default_form) -> None:
    form, _values = default_form
    window = form._widgets["predict.window_stride"]
    assert window.edit.value() == 50
    window.auto.setChecked(False)
    assert window.edit.isEnabled()
    assert form.current()["predict.window_stride"] == 50
    window.edit.setValue(37)
    assert form.current()["predict.window_stride"] == 37
    window.auto.setChecked(True)
    assert form.current()["predict.window_stride"] == "auto", "ticking it back writes the word"


def test_a_fixed_set_of_values_is_never_free_text(app) -> None:
    """Every setting with a fixed set of values is a combo of exactly that set."""
    from trackanno.config import meta_for
    from trackanno.gui.configform import FIELDS, ConfigForm, _options

    form = ConfigForm("all")
    for section, key, kind, _label, _tip, extra in FIELDS:
        dotted = f"{section}.{key}"
        choices = meta_for(dotted).get("choices")
        if choices:
            assert kind == "choice", f"{dotted} is a combo: {kind}"
            assert _options(dotted, extra) == [str(c) for c in choices], (
                f"{dotted} offers the schema's choices")
            assert not form._widgets[dotted].isEditable(), f"{dotted} cannot be typed into"
    assert all(_options(f"{f[0]}.{f[1]}", f[5]) for f in FIELDS if f[2] == "choice"), (
        "every combo has options to offer")

    # A name is a name: an annotator called 2024 stays text, an empty box is "".
    annotator = form._widgets["ui.annotator"]
    annotator.setText("2024")
    assert form.current()["ui.annotator"] == "2024", "an annotator called 2024 stays a name"
    annotator.setText("")
    assert form.current()["ui.annotator"] == "", "an empty name is the empty name, not null"


def test_no_combo_offers_a_value_the_loader_refuses(app, root: Path) -> None:
    """Every option every combo offers is accepted by the real loader."""
    from trackanno.detect.tracking import TRACKERS
    from trackanno.gui.configform import FIELDS, _options
    from trackanno.project import create_project, write_config

    assert _options("tracking.backend", {}) == sorted(TRACKERS) + ["greedy"], (
        "the trackers are the registry itself")

    path = create_project(root / "combos")
    base = path.read_text(encoding="utf-8")
    offered = 0
    for section, key, kind, _label, _tip, extra in FIELDS:
        if kind != "choice":
            continue
        dotted = f"{section}.{key}"
        for option in _options(dotted, extra):
            path.write_text(base, encoding="utf-8")
            write_config(path, {dotted: option})        # raises on a refused value
            offered += 1
    path.write_text(base, encoding="utf-8")
    assert offered >= 15, (
        f"every value every combo offers is one the loader accepts: {offered}")


def test_tooltips_come_from_the_schema(app) -> None:
    """One help string per key, shown as the field's tooltip."""
    from trackanno.config import help_for
    from trackanno.gui.configform import ConfigForm

    form = ConfigForm("all")
    for dotted, widget in form._widgets.items():
        assert help_for(dotted) in widget.toolTip(), (
            f"{dotted} shows its schema help: {widget.toolTip()}")


def test_reprepare_keys_follow_the_fingerprint() -> None:
    """The '*' marks are the keys the detect and clip stages record."""
    from trackanno.detect.prepare import _CLIP_KEYS, _DETECT_KEYS
    from trackanno.gui.configform import NEEDS_REPREPARE

    assert set(_DETECT_KEYS) | set(_CLIP_KEYS) <= NEEDS_REPREPARE, (
        "every fingerprinted key is marked")
    assert "clips.frames" in NEEDS_REPREPARE, "clip length is one of them"
    assert "ui.fps" not in NEEDS_REPREPARE, "playback speed is not"


def test_only_changed_keys_are_written(app, root: Path) -> None:
    """An untouched field is never rewritten, even where its widget rounds it."""
    from trackanno.config import load_config
    from trackanno.gui.configform import ConfigForm
    from trackanno.project import create_project, write_config

    path = create_project(root / "untouched")
    write_config(path, {"detection.conf": 0.125})
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    form = ConfigForm("all")
    form.load(raw)
    assert form.current()["detection.conf"] == 0.13, (
        f"the spin box shows a rounded value: {form.current()['detection.conf']}")
    assert form.values() == {}, (
        f"nothing is offered for writing straight after loading: {form.values()}")
    before = path.read_text(encoding="utf-8")

    form._widgets["clips.frames"].setValue(40)
    form._widgets["clips.stride"].setValue(40)
    offered = form.values()
    assert set(offered) == {"clips.frames", "clips.stride"}, (
        f"only the touched keys are offered: {offered}")
    write_config(path, offered)
    cfg = load_config(path)
    assert cfg.detection.conf == 0.125, (
        f"the untouched value keeps its precision: {cfg.detection.conf}")
    assert cfg.clips.frames == 40, "the touched one changed"
    assert (path.read_text(encoding="utf-8").count("#")
          == before.count("#")), (
        "the comments survive")

    form.saved()
    assert form.values() == {}, "after a save nothing is pending"
    form._widgets["clips.frames"].setValue(50)
    assert form.values().get("clips.frames") == 50, "changing back after a save is a change"


def test_an_absent_key_is_shown_at_its_default(app) -> None:
    """A key the file does not mention shows the value the loader uses."""
    from trackanno.config import section_defaults
    from trackanno.gui.configform import ConfigForm

    form = ConfigForm("all")
    form.load({})
    shown = form.current()
    checked = 0
    for dotted, kind in form._kinds.items():
        section, _, key = dotted.partition(".")
        default = section_defaults(section).get(key)
        if default is None or kind in ("dir", "file"):
            continue
        assert shown[dotted] == default, (
            f"{dotted} is shown at its default: {shown[dotted]!r} vs {default!r}")
        checked += 1
    assert checked >= 15, f"there were defaults to check: {checked}"
    assert form.values() == {}, "none of them is offered for writing"


def test_the_form_shows_what_the_loader_uses(app) -> None:
    """Project B omits or retires what project A set; the form shows B's effective values."""
    from trackanno.gui.configform import ConfigForm

    form = ConfigForm("all")
    form.load({
        "paths": {"work_dir": "./elsewhere", "csv": "./a.csv"},
        "clips": {"frames": 40, "stride": 10, "min_clip_frames": 10},
        "suggest": {"select_rule": "margin", "feature_groups": ["kinematic"]},
        "detection": {"imgsz": 1280},
    })
    form.load({
        "clips": {"frames": 25, "min_clip_frames": 10},
        "suggest": {"select_rule": "entropy", "context_frames": 5},
    })
    shown = form.current()
    assert (shown["paths.work_dir"] == "./work"
          and shown["paths.csv"] == "./annotations.csv"), (
        f"an omitted path shows its default: {shown['paths.work_dir']}")
    assert shown["clips.stride"] == 25, (
        f"an omitted stride shows clips.frames: {shown['clips.stride']}")
    assert shown["suggest.select_rule"] == "margin", (
        f"a retired value shows its replacement: {shown['suggest.select_rule']}")
    assert shown["suggest.feature_groups"] == ["kinematic", "pose", "waveform", "context"], (
        f"omitted feature groups show the effective set: {shown['suggest.feature_groups']}")
    assert shown["detection.imgsz"] == 640, "an omitted number shows its default"
    assert form.values() == {}, "nothing is offered for writing"


def test_a_cleared_warning_gives_the_tooltip_back(app) -> None:
    from trackanno.gui.configform import ConfigForm

    form = ConfigForm("advanced")
    widget = form._widgets["predict.window_stride"]
    target = getattr(widget, "edit", widget)
    original = target.toolTip()
    form.mark({"predict.window_stride": "broken"})
    assert target.toolTip() == "broken", "the warning replaces the tooltip"
    form.mark({"predict.window_stride": "still broken"})
    form.mark({})
    assert target.toolTip() == original, (
        f"the original comes back once it clears: {target.toolTip()[:80]}")


@pytest.fixture
def refused(app, tmp_path: Path):
    """A Configure page holding a window stride its save refuses."""
    from trackanno.gui.hub import ConfigPage, Hub
    from trackanno.project import create_project

    path = create_project(tmp_path / "refused_kept")
    hub = Hub(str(path))
    hub.show()
    page = hub.page(ConfigPage)
    hub.go_to(hub.pages.index(page))
    app.processEvents()
    good = path.read_text(encoding="utf-8")
    stride = page.advanced._widgets["predict.window_stride"]
    stride.auto.setChecked(False)
    stride.edit.setValue(999)
    app.processEvents()
    assert not page.save(), "the save is refused"
    yield hub, page, stride, path, good
    page._ask_unsaved = lambda: "discard"
    hub.close()
    app.processEvents()


def test_a_background_reload_keeps_a_refused_edit(app, refused) -> None:
    hub, _page, stride, _path, _good = refused
    hub.reload_project()            # what Prepare or Model finishing does
    app.processEvents()
    assert not stride.auto.isChecked() and stride.edit.value() == 999, stride.edit.value()


@pytest.mark.parametrize("answer", ["save", "cancel"])
def test_leaving_with_a_refused_edit_stays_unless_discarded(app, refused, answer) -> None:
    hub, page, stride, path, good = refused
    page._ask_unsaved = lambda: answer
    hub.go_to(0)
    app.processEvents()
    assert hub.stack.currentWidget() is page
    assert stride.edit.value() == 999, "the edit is kept"
    assert path.read_text(encoding="utf-8") == good, "the file is untouched"


def test_discarding_a_refused_edit_leaves_and_reverts(app, refused) -> None:
    hub, page, stride, _path, _good = refused
    page._ask_unsaved = lambda: "discard"
    hub.go_to(0)
    app.processEvents()
    assert hub.stack.currentWidget() is not page
    assert stride.auto.isChecked(), "the field is back to the file"
    assert not page._dirty and page.confirm_leave()


def test_export_checkbox_follows_the_config(app, root: Path) -> None:
    from trackanno.gui.hub import ExportPage, Hub
    from trackanno.project import create_project, write_config

    path = create_project(root / "export_hide")
    write_config(path, {"render.hide_unlabelled": True})
    hub = Hub(str(path))
    page = hub.page(ExportPage)
    assert page.hide_unlabelled.isChecked(), "the box starts from render.hide_unlabelled"
    page.hide_unlabelled.setChecked(False)
    hub.reload_project()
    assert not page.hide_unlabelled.isChecked(), "a reload leaves a cleared box alone"
    hub.close()
    app.processEvents()


def test_the_category_editor_flags_a_reserved_name(app) -> None:
    from trackanno.gui.configform import CategoryEditor

    editor = CategoryEditor()
    editor.load([{"name": "notes", "type": "single", "required": True,
                  "options": [{"name": "a", "key": "1", "color": "#4C9F70"}]}])
    assert any("already uses" in p for p in editor.problems()), editor.problems()


@pytest.fixture
def picker(app, tmp_path: Path):
    """The feature-group picker on the Advanced tab of a fresh project."""
    from trackanno.gui.hub import ConfigPage, Hub
    from trackanno.project import create_project

    path = create_project(tmp_path / "picker")
    hub = Hub(str(path))
    hub.show()
    page = hub.page(ConfigPage)
    hub.go_to(hub.pages.index(page))
    page.tabs.setCurrentIndex(1)                  # Advanced, so the note shows
    app.processEvents()
    yield page, page.advanced._widgets["suggest.feature_groups"], path
    page.revert()
    hub.close()
    app.processEvents()


def test_unticking_every_group_keeps_one_and_still_saves(app, picker) -> None:
    from trackanno.config import load_config

    page, groups, path = picker
    for box in groups._boxes.values():
        box.setChecked(False)
    app.processEvents()
    assert len(groups.value()) == 1, groups.value()
    assert not groups.note.isHidden(), "it says why"
    assert page.save() and not page.banner.isVisible()
    assert len(load_config(path).suggest.feature_groups) == 1


def test_ticking_the_new_group_first_swaps_groups(app, picker) -> None:
    _page, groups, _path = picker
    groups._boxes["kinematic"].setChecked(True)
    app.processEvents()
    for name in ("context", "waveform"):
        groups._boxes[name].setChecked(False)
    app.processEvents()
    assert "kinematic" in groups.value(), groups.value()


def test_a_group_this_project_cannot_compute_is_flagged(app, picker) -> None:
    _page, groups, _path = picker
    groups.set_value(["pose"])
    groups._refresh_note()
    assert not groups.note.isHidden()
    assert "detection.task" in groups.note.text(), groups.note.text()
