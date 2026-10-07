"""Checks for config.yaml as a form: what it exposes, how, and what it refuses.

The form is the only place most people ever see these settings, so the claims
here are about the SHAPE of that presentation as much as the values: that every
key defined reaches exactly one tab, that a setting whose valid values are a
fixed set is offered as that set rather than as free text, that a combo's
options come from whatever module defines them, and that 'auto' says what it
stands for on this project instead of leaving it to be guessed.

Nothing here needs a movie or a clip cache; the projects are generated.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from harness import check

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
    """The page is split in two, so a key can be shown twice or not at all."""
    from trackanno.gui.configform import FIELDS, GROUPS, ConfigForm

    defined = [f"{section}.{key}" for section, key, *_ in FIELDS]
    check("the form stays short", 25 <= len(defined) <= 32, str(len(defined)))
    check("every field has help from the schema", all(tip for *_, tip, _e in FIELDS))
    check("no field is defined twice", len(defined) == len(set(defined)),
          str(sorted({d for d in defined if defined.count(d) > 1})))

    basic, advanced = ConfigForm("basic"), ConfigForm("advanced")
    rendered = list(basic._widgets) + list(advanced._widgets)
    check("every field defined reaches a tab", set(rendered) == set(defined),
          str(sorted(set(defined) ^ set(rendered))))
    check("...and none reaches both", len(rendered) == len(set(rendered)))
    check("the basics tab is still the short list",
          5 < len(basic._widgets) < 20, str(len(basic._widgets)))
    check("scope 'all' is both", len(ConfigForm("all")._widgets) == len(defined))

    # Every group is a heading plus one sentence about what the settings under
    # it produce together. A group with no sentence is a section title again.
    check("every group says what it is for",
          all(blurb.strip() and blurb.strip().endswith(".") for _t, blurb, _f in GROUPS),
          str([t for t, b, _f in GROUPS if not b.strip().endswith(".")]))
    check("every group has fields in it", all(fields for _t, _b, fields in GROUPS))
    titles = [title for title, _b, _f in GROUPS]
    check("no two groups share a title", len(titles) == len(set(titles)))

    # The point of the regrouping: the keys load_config refuses OVER EACH OTHER
    # are shown together, with one sentence about the combination, rather than
    # as adjacent rows each explaining only itself.
    group_of = {dotted: title for title, _b, rows in GROUPS for dotted, *_ in rows}
    coupled = ["predict.window_stride", "predict.min_bout_frames"]
    check("the keys that are refused over each other are one group",
          len({group_of[k] for k in coupled}) == 1,
          str({k: group_of[k] for k in coupled}))


def test_the_prediction_keys_the_gui_can_break_are_on_it(app, root: Path) -> None:
    """The prediction keys are on the form and survive the real loader."""
    from trackanno.config import load_config
    from trackanno.gui.configform import ConfigForm
    from trackanno.project import create_project, write_config

    advanced = ConfigForm("advanced")
    check("all the keys are on a tab",
          all(key in advanced._widgets for key in NEW_KEYS),
          str(sorted(set(NEW_KEYS) - set(advanced._widgets))))

    form = ConfigForm("all")
    form.load(_raw(NEW_KEYS))
    values = form.current()
    for dotted, value in NEW_KEYS.items():
        check(f"{dotted} round-trips through the form",
              values[dotted] == value, repr(values.get(dotted)))

    # And the combination they exist for has to survive the real loader.
    path = create_project(root / "predict-keys")
    coherent = {
        "clips.frames": 50, "clips.stride": 50,
        "predict.window_stride": 25, "predict.min_bout_frames": 25,
    }
    write_config(path, coherent)
    cfg = load_config(path)
    check("an overlapping project written from these fields loads",
          cfg.predict.resolved_stride(cfg.clips) == 25,
          str(cfg.predict.resolved_stride(cfg.clips)))
    check("...and every value survived the round trip",
          cfg.predict.min_bout_frames == 25)

    form = ConfigForm("all")
    form.load(yaml.safe_load(path.read_text(encoding="utf-8")))
    written = form.current()
    check("...and the form reads back what it wrote",
          all(written[k] == v for k, v in coherent.items()),
          str({k: written[k] for k, v in coherent.items() if written[k] != v}))


def test_a_refused_field_never_reaches_the_file(app, root: Path) -> None:
    """Save writes every tab at once, so one unusable field must stop the whole
    save rather than being dropped out of it."""
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
    check("the autosave refuses it", not page.save() and not page.banner.isHidden())
    check("the banner names the field",
          "window_stride" in page.banner.text(), page.banner.text()[:160])
    check("the refused save left the file alone",
          path.read_text(encoding="utf-8") == good)

    stride.auto.setChecked(True)
    app.processEvents()
    check("putting it back saves again", page.save() and page.banner.isHidden())

    hub.close()
    app.processEvents()


def test_auto_says_what_it_stands_for(app) -> None:
    """'auto' in a text box is two hidden facts: that it is one of exactly two
    shapes, and what it resolves to here. Neither was on screen."""
    from trackanno.gui.configform import ConfigForm
    from trackanno.project import DEFAULT_CONFIG

    form = ConfigForm("all")
    form.load(yaml.safe_load(DEFAULT_CONFIG))
    values = form.current()
    form.show_resolved(values)

    window = form._widgets["predict.window_stride"]
    check("the choice between 'auto' and a number is a tick, not typing",
          window.auto.isChecked() and not window.edit.isEnabled())
    check("'auto' says what it resolves to", window.auto.text() == "auto (50)",
          window.auto.text())
    check("an auto with no answer yet says only 'auto'",
          form._widgets["clips.crop_size"].auto.text() == "auto")

    # It has to follow the settings it resolves from, which live in another
    # config section and, on the split page, on the other tab.
    form.show_resolved({**values, "clips.frames": 30, "clips.stride": 30})
    check("it follows the setting it depends on", window.auto.text() == "auto (30)",
          window.auto.text())

    form.show_resolved({**values, "clips.frames": 50, "clips.stride": 25})
    check("an overlapping clip grid says so",
          "overlap" in form._status["clips.stride"].text(),
          form._status["clips.stride"].text())
    form.show_resolved({**values, "clips.frames": 50, "clips.stride": 50})
    check("a tiling grid says so instead",
          "tile" in form._status["clips.stride"].text(),
          form._status["clips.stride"].text())

    # Leaving 'auto' has to start somewhere, and the number it stood for is the
    # only defensible place.
    check("leaving 'auto' starts from the number it stood for", window.edit.value() == 50)
    window.auto.setChecked(False)
    check("...and only then is the number editable", window.edit.isEnabled())
    check("...and the form offers the number", form.current()["predict.window_stride"] == 50)
    window.edit.setValue(37)
    check("an explicit number is written as one",
          form.current()["predict.window_stride"] == 37)
    window.auto.setChecked(True)
    check("ticking it back writes the word", form.current()["predict.window_stride"] == "auto")


def test_a_fixed_set_of_values_is_never_free_text(app) -> None:
    """Every setting with a fixed set of values is a combo of exactly that set."""
    from trackanno.config import meta_for
    from trackanno.gui.configform import FIELDS, ConfigForm, _options

    form = ConfigForm("all")
    for section, key, kind, _label, _tip, extra in FIELDS:
        dotted = f"{section}.{key}"
        choices = meta_for(dotted).get("choices")
        if choices:
            check(f"{dotted} is a combo", kind == "choice", kind)
            check(f"{dotted} offers the schema's choices",
                  _options(dotted, extra) == [str(c) for c in choices])
            check(f"{dotted} cannot be typed into", not form._widgets[dotted].isEditable())
    check("every combo has options to offer",
          all(_options(f"{f[0]}.{f[1]}", f[5]) for f in FIELDS if f[2] == "choice"))

    # A name is a name: an annotator called 2024 stays text, an empty box is "".
    annotator = form._widgets["ui.annotator"]
    annotator.setText("2024")
    check("an annotator called 2024 stays a name", form.current()["ui.annotator"] == "2024")
    annotator.setText("")
    check("an empty name is the empty name, not null", form.current()["ui.annotator"] == "")


def test_no_combo_offers_a_value_the_loader_refuses(app, root: Path) -> None:
    """Every option every combo offers is accepted by the real loader."""
    from trackanno.config import ConfigError
    from trackanno.detect.tracking import TRACKERS
    from trackanno.gui.configform import FIELDS, _options
    from trackanno.project import create_project, write_config

    check("the trackers are the registry itself",
          _options("tracking.backend", {}) == sorted(TRACKERS) + ["greedy"])

    path = create_project(root / "combos")
    base = path.read_text(encoding="utf-8")
    offered = 0
    for section, key, kind, _label, _tip, extra in FIELDS:
        if kind != "choice":
            continue
        dotted = f"{section}.{key}"
        for option in _options(dotted, extra):
            path.write_text(base, encoding="utf-8")
            try:
                write_config(path, {dotted: option})
            except ConfigError as exc:
                check(f"{dotted} offers only values the loader takes", False,
                      f"{option!r}: {exc}")
            offered += 1
    path.write_text(base, encoding="utf-8")
    check("every value every combo offers is one the loader accepts", offered >= 15,
          str(offered))


def test_tooltips_come_from_the_schema(app) -> None:
    """One help string per key, shown as the field's tooltip."""
    from trackanno.config import help_for
    from trackanno.gui.configform import ConfigForm

    form = ConfigForm("all")
    for dotted, widget in form._widgets.items():
        check(f"{dotted} shows its schema help", help_for(dotted) in widget.toolTip(),
              widget.toolTip())


def test_reprepare_keys_follow_the_fingerprint() -> None:
    """The '*' marks are the keys the detect and clip stages record."""
    from trackanno.detect.prepare import _CLIP_KEYS, _DETECT_KEYS
    from trackanno.gui.configform import NEEDS_REPREPARE

    check("every fingerprinted key is marked",
          set(_DETECT_KEYS) | set(_CLIP_KEYS) <= NEEDS_REPREPARE)
    check("clip length is one of them", "clips.frames" in NEEDS_REPREPARE)
    check("playback speed is not", "ui.fps" not in NEEDS_REPREPARE)


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
    check("the spin box shows a rounded value",
          form.current()["detection.conf"] == 0.13, str(form.current()["detection.conf"]))
    check("nothing is offered for writing straight after loading", form.values() == {},
          str(form.values()))
    before = path.read_text(encoding="utf-8")

    form._widgets["clips.frames"].setValue(40)
    form._widgets["clips.stride"].setValue(40)
    offered = form.values()
    check("only the touched keys are offered", set(offered) == {"clips.frames", "clips.stride"},
          str(offered))
    write_config(path, offered)
    cfg = load_config(path)
    check("the untouched value keeps its precision", cfg.detection.conf == 0.125,
          str(cfg.detection.conf))
    check("the touched one changed", cfg.clips.frames == 40)
    check("the comments survive", path.read_text(encoding="utf-8").count("#")
          == before.count("#"))

    form.saved()
    check("after a save nothing is pending", form.values() == {})
    form._widgets["clips.frames"].setValue(50)
    check("changing back after a save is a change", form.values().get("clips.frames") == 50)


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
        check(f"{dotted} is shown at its default", shown[dotted] == default,
              f"{shown[dotted]!r} vs {default!r}")
        checked += 1
    check("there were defaults to check", checked >= 15, str(checked))
    check("and none of them is offered for writing", form.values() == {})


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
    check("an omitted path shows its default", shown["paths.work_dir"] == "./work"
          and shown["paths.csv"] == "./annotations.csv", str(shown["paths.work_dir"]))
    check("an omitted stride shows clips.frames", shown["clips.stride"] == 25,
          str(shown["clips.stride"]))
    check("a retired value shows its replacement", shown["suggest.select_rule"] == "margin",
          shown["suggest.select_rule"])
    check("omitted feature groups show the effective set",
          shown["suggest.feature_groups"] == ["kinematic", "pose", "waveform", "context"],
          str(shown["suggest.feature_groups"]))
    check("an omitted number shows its default", shown["detection.imgsz"] == 640)
    check("nothing is offered for writing", form.values() == {})


def test_a_cleared_warning_gives_the_tooltip_back(app) -> None:
    from trackanno.gui.configform import ConfigForm

    form = ConfigForm("advanced")
    widget = form._widgets["predict.window_stride"]
    target = getattr(widget, "edit", widget)
    original = target.toolTip()
    form.mark({"predict.window_stride": "broken"})
    check("the warning replaces the tooltip", target.toolTip() == "broken")
    form.mark({"predict.window_stride": "still broken"})
    form.mark({})
    check("...and the original comes back once it clears",
          target.toolTip() == original, target.toolTip()[:80])


def test_refused_edits_survive_reloads_and_guard_navigation(app, root: Path) -> None:
    """A refused save keeps the edit on screen through a background reload, and
    leaving the page asks before dropping it."""
    from trackanno.gui.hub import ConfigPage, Hub
    from trackanno.project import create_project

    path = create_project(root / "refused_kept")
    hub = Hub(str(path))
    hub.show()
    page = hub.page(ConfigPage)
    index = hub.pages.index(page)
    hub.go_to(index)
    app.processEvents()
    good = path.read_text(encoding="utf-8")

    stride = page.advanced._widgets["predict.window_stride"]
    stride.auto.setChecked(False)
    stride.edit.setValue(999)
    app.processEvents()
    check("the save is refused", not page.save())

    hub.reload_project()            # what Prepare or Model finishing does
    app.processEvents()
    check("a background reload keeps the refused edit",
          not stride.auto.isChecked() and stride.edit.value() == 999,
          str(stride.edit.value()))

    answers: list[str] = []
    page._ask_unsaved = lambda: answers.pop(0)
    answers.append("save")
    hub.go_to(0)
    app.processEvents()
    check("a refused Save keeps the Configure page", hub.stack.currentWidget() is page)
    check("...and the edit", stride.edit.value() == 999)
    check("...and the file", path.read_text(encoding="utf-8") == good)

    answers.append("cancel")
    hub.go_to(0)
    app.processEvents()
    check("Cancel keeps the Configure page", hub.stack.currentWidget() is page)

    answers.append("discard")
    hub.go_to(0)
    app.processEvents()
    check("Discard leaves the page", hub.stack.currentWidget() is not page)
    check("...and reverts the field to the file", stride.auto.isChecked())
    check("...and nothing is pending", not page._dirty and page.confirm_leave())

    hub.close()
    app.processEvents()


def test_export_checkbox_follows_the_config(app, root: Path) -> None:
    from trackanno.gui.hub import ExportPage, Hub
    from trackanno.project import create_project, write_config

    path = create_project(root / "export_hide")
    write_config(path, {"render.hide_unlabelled": True})
    hub = Hub(str(path))
    page = hub.page(ExportPage)
    check("the box starts from render.hide_unlabelled", page.hide_unlabelled.isChecked())
    page.hide_unlabelled.setChecked(False)
    hub.reload_project()
    check("...and a reload leaves a cleared box alone",
          not page.hide_unlabelled.isChecked())
    hub.close()
    app.processEvents()
