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
    "predict.min_frames": 7,
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
    check("no field is defined twice", len(defined) == len(set(defined)),
          str(sorted({d for d in defined if defined.count(d) > 1})))

    basic, advanced = ConfigForm("basic"), ConfigForm("advanced")
    rendered = list(basic._widgets) + list(advanced._widgets)
    check("every field defined reaches a tab", set(rendered) == set(defined),
          str(sorted(set(defined) ^ set(rendered))))
    check("...and none reaches both", len(rendered) == len(set(rendered)))
    check("the basics tab is still the short list",
          5 < len(basic._widgets) < 25, str(len(basic._widgets)))
    check("the advanced tab holds the rest",
          len(advanced._widgets) > len(basic._widgets))
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
    group_of = {f"{section}.{key}": title
                for title, _b, fields in GROUPS for section, key, *_ in fields}
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
    values = form.values()
    for dotted, value in NEW_KEYS.items():
        check(f"{dotted} round-trips through the form",
              values[dotted] == value, repr(values.get(dotted)))

    # And the combination they exist for has to survive the real loader.
    path = create_project(root / "predict-keys")
    coherent = {
        "clips.frames": 50, "clips.stride": 50,
        "predict.window_stride": 25, "predict.min_bout_frames": 25,
        "predict.min_frames": 4,
    }
    write_config(path, coherent)
    cfg = load_config(path)
    check("an overlapping project written from these fields loads",
          cfg.predict.resolved_stride(cfg.clips) == 25,
          str(cfg.predict.resolved_stride(cfg.clips)))
    check("...and every value survived the round trip",
          cfg.predict.min_frames == 4 and cfg.predict.min_bout_frames == 25)

    form = ConfigForm("all")
    form.load(yaml.safe_load(path.read_text(encoding="utf-8")))
    written = form.values()
    check("...and the form reads back what it wrote",
          all(written[k] == v for k, v in coherent.items()),
          str({k: written[k] for k, v in coherent.items() if written[k] != v}))


def test_a_refused_field_never_reaches_the_file(app, root: Path) -> None:
    """The autosave writes every tab at once, so one unusable field must stop
    the whole save rather than being dropped out of it."""
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
    values = form.values()
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
    check("...and the form offers the number", form.values()["predict.window_stride"] == 50)
    window.edit.setValue(37)
    check("an explicit number is written as one",
          form.values()["predict.window_stride"] == 37)
    window.auto.setChecked(True)
    check("ticking it back writes the word", form.values()["predict.window_stride"] == "auto")


def test_a_fixed_set_of_values_is_never_free_text(app) -> None:
    """Every setting whose valid values are a fixed set is offered as that set."""
    from trackanno.gui.configform import FIELDS, ConfigForm, _options

    kinds = {f"{section}.{key}": kind for section, key, kind, *_ in FIELDS}
    check("nothing is left as an untyped scalar box",
          not [k for k, kind in kinds.items() if kind == "scalar"],
          str([k for k, kind in kinds.items() if kind == "scalar"]))
    for dotted, expected in (("tiling.enabled", "tristate"),
                             ("clips.crop_size", "autonum")):
        check(f"{dotted} is not free text any more", kinds[dotted] == expected,
              kinds[dotted])

    form = ConfigForm("all")
    tiling = form._widgets["tiling.enabled"]
    check("tiling offers exactly its three values", tiling.count() == 3)
    # The item carries what it writes, not what it reads: `off` typed into a
    # text box arrives downstream as the string "off", which is TRUE.
    tiling.set_value(False)
    check("'off' writes a boolean, not the word",
          form.values()["tiling.enabled"] is False)
    tiling.set_value(True)
    check("'on' too", form.values()["tiling.enabled"] is True)
    tiling.set_value("auto")
    check("...and 'auto' stays the word it is compared against",
          form.values()["tiling.enabled"] == "auto")

    # A name is a name. `_parse_scalar` used to type these, so an annotator
    # called 2024 was written as an integer and an empty box as null - which
    # reaches ui.annotator as the string "None" and signs every CSV row with it.
    annotator = form._widgets["ui.annotator"]
    annotator.setText("2024")
    check("an annotator called 2024 stays a name",
          form.values()["ui.annotator"] == "2024")
    annotator.setText("")
    check("an empty name is the empty name, not null",
          form.values()["ui.annotator"] == "")

    check("every combo has options to offer",
          all(_options(field[5]) for field in FIELDS if field[2] == "choice"),
          str([f"{f[0]}.{f[1]}" for f in FIELDS
               if f[2] == "choice" and not _options(f[5])]))


def test_no_combo_offers_a_value_the_loader_refuses(app, root: Path) -> None:
    """A list of allowed values copied into the form rots silently in either
    direction. Where the module that defines them exports a registry, the form
    reads it; where it does not, this pins the direction that can be pinned."""
    from trackanno.config import ConfigError
    from trackanno.gui.configform import FIELDS, _options
    from trackanno.project import create_project, write_config
    from trackanno.detect.tracking import TRACKERS

    extras = {f"{section}.{key}": extra for section, key, _kind, _l, _t, extra in FIELDS}
    check("the trackers are the registry itself",
          _options(extras["tracking.backend"]) == sorted(TRACKERS) + ["greedy"])
    check("...and are read, not written down",
          "options_from" in extras["tracking.backend"]
          and "options" not in extras["tracking.backend"])

    # Everything else has its allowed values written out inside config.py's own
    # parse functions, with nothing importable to read. Every option is put
    # through the real loader instead, so a list that goes stale fails here.
    path = create_project(root / "combos")
    base = path.read_text(encoding="utf-8")
    companions: dict = {}
    offered = 0
    for section, key, kind, _label, _tip, extra in FIELDS:
        if kind != "choice":
            continue
        dotted = f"{section}.{key}"
        for option in _options(extra):
            path.write_text(base, encoding="utf-8")
            try:
                write_config(path, {dotted: option, **companions.get(dotted, {})})
            except ConfigError as exc:
                check(f"{dotted} offers only values the loader takes", False,
                      f"{option!r}: {exc}")
            offered += 1
        if dotted == "detection.device":
            continue  # any CUDA index is valid, so there is no value to refuse
        path.write_text(base, encoding="utf-8")
        try:
            write_config(path, {dotted: "not-a-real-option"})
        except ConfigError:
            pass
        else:
            check(f"{dotted} refuses a value it does not offer", False)
    check("every value every combo offers is one the loader accepts",
          offered >= 30, str(offered))


def test_an_autonum_widget_takes_the_shape_its_setting_does(app) -> None:
    """A field has to be able to represent what its setting accepts.

    `_AutoField` builds an integer spin box unless the row says
    `numeric: "float"`, so a setting declared `float | str` with that missing
    loads 0.05 as 0. The next autosave writes the 0, the loader refuses it, and
    the Configure page cannot save anything at all until the file is edited by
    hand - which is how `suggest.learning_rate` shipped.

    Read off the dataclass annotation rather than probed with a value, because
    the value that exposes it differs per setting and a probe that suits one
    is nonsense for the next.
    """
    from trackanno import config as config_module
    from trackanno.gui.configform import FIELDS, ConfigForm

    sections = {
        "detection": "Detection", "tiling": "Tiling", "tracking": "Tracking",
        "clips": "Clips", "suggest": "Suggest", "predict": "Predict",
        "ui": "UI", "normalize": "Normalize", "features": "Features",
        "paths": "Paths", "keypoints": "Keypoints",
    }
    form = ConfigForm("all")
    checked = 0
    for section, key, kind, *_ in FIELDS:
        if kind != "autonum":
            continue
        holder = getattr(config_module, sections.get(section, ""), None)
        annotation = str((getattr(holder, "__annotations__", {}) or {}).get(key, ""))
        check(f"{section}.{key} is a real setting on a real section",
              bool(annotation), f"no annotation for {section}.{key}")
        wants_float = "float" in annotation
        widget = form._widgets[f"{section}.{key}"]
        check(f"{section}.{key} takes {'a rate' if wants_float else 'a whole number'}, "
              f"and its box does too",
              bool(widget._float) == wants_float,
              f"declared {annotation!r}, widget float={widget._float}")
        checked += 1
    check("there were autonum fields to check", checked >= 3, str(checked))


def test_a_key_the_file_never_mentions_is_not_invented(app, root: Path) -> None:
    """The generated config.yaml documents most settings and leaves the rest to
    their defaults. Writing every widget back turned each of those into whatever
    its spin box range starts at - `suggest.iterations` 1000 -> 10, `depth` 6 -> 1,
    `balance_classes` on -> off - on the first autosave of a new project, which
    is a different model with nothing on screen to say so."""
    from trackanno.config import load_config
    from trackanno.gui.configform import ConfigForm
    from trackanno.project import create_project, write_config

    path = create_project(root / "untouched")
    before = path.read_text(encoding="utf-8")
    raw = yaml.safe_load(before)
    check("the generated project really does leave these out",
          "iterations" not in (raw.get("suggest") or {}))

    form = ConfigForm("all")
    form.load(raw)
    check("a key the file does not mention is not offered for writing",
          "suggest.iterations" not in form.values())
    check("...while one it does mention is",
          form.values()["clips.frames"] == raw["clips"]["frames"])

    write_config(path, form.values())
    cfg = load_config(path)
    check("saving the form keeps the default the loader would have used",
          cfg.suggest.iterations == 1000 and cfg.suggest.depth == 6
          and cfg.suggest.learning_rate == "auto"
          and cfg.suggest.balance_classes,
          f"{cfg.suggest.iterations}, {cfg.suggest.depth}, {cfg.suggest.balance_classes}")
    check("...and the comments survive a save of every field",
          path.read_text(encoding="utf-8").count("#") == before.count("#"),
          f"{path.read_text(encoding='utf-8').count('#')} vs {before.count('#')}")

    # Touching one is what makes it real, and only that one.
    form.load(yaml.safe_load(path.read_text(encoding="utf-8")))
    form._widgets["suggest.iterations"].setValue(1200)
    offered = form.values()
    check("touching an absent key writes it", offered.get("suggest.iterations") == 1200)
    check("...and its untouched neighbours stay absent",
          "suggest.depth" not in offered and "suggest.balance_classes" not in offered)
    write_config(path, offered)
    written = load_config(path)
    check("the touched key reached the file", written.suggest.iterations == 1200)
    check("...and the untouched one kept its default", written.suggest.depth == 6)


def test_a_key_the_file_never_mentions_is_shown_at_the_value_that_runs(app, root: Path) -> None:
    """Absent is not blank. The test above is the write half of this rule; this
    is the read half, and it was the half that was wrong: with the key missing,
    every one of those widgets displayed whatever its range began at, so the
    Suggestions tab reported 10 boosting iterations, depth 1 and unbalanced
    classes while the model being fitted was 1000, 6 and balanced. Reading it was
    misleading and acting on it was worse - raising the iterations from what the
    page showed would have cut the model to a hundredth of its size."""
    from trackanno.config import load_config, section_defaults
    from trackanno.gui.configform import ConfigForm
    from trackanno.project import create_project

    path = create_project(root / "shown")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = load_config(path)

    form = ConfigForm("all")
    form.load(raw)
    shown = form._raw_values()

    check("the generated file really does leave the boosting knobs out",
          "iterations" not in (raw.get("suggest") or {}))
    check("the page shows the iterations that will actually be fitted",
          shown["suggest.iterations"] == cfg.suggest.iterations == 1000,
          f"{shown['suggest.iterations']} shown, {cfg.suggest.iterations} fitted")
    check("...the depth those trees will have",
          shown["suggest.depth"] == cfg.suggest.depth == 6,
          f"{shown['suggest.depth']} vs {cfg.suggest.depth}")
    check("...the L2 penalty they will carry",
          shown["suggest.l2_leaf_reg"] == cfg.suggest.l2_leaf_reg == 3.0,
          f"{shown['suggest.l2_leaf_reg']} vs {cfg.suggest.l2_leaf_reg}")
    check("...and that the classes will be balanced",
          shown["suggest.balance_classes"] == cfg.suggest.balance_classes is True,
          f"{shown['suggest.balance_classes']} vs {cfg.suggest.balance_classes}")
    check("the rate is still the estimator's to choose",
          shown["suggest.learning_rate"] == cfg.suggest.learning_rate == "auto",
          f"{shown['suggest.learning_rate']!r}")

    # The general rule, so a knob added later cannot quietly go back to showing
    # the corner of its own range.
    checked = 0
    for dotted in sorted(form._absent):
        section, _, key = dotted.partition(".")
        default = section_defaults(section)
        if key not in default:
            continue
        check(f"{dotted} is shown at the value the loader would use",
              shown[dotted] == default[key],
              f"shows {shown[dotted]!r}, loader uses {default[key]!r}")
        checked += 1
    check("there were absent keys to check", checked >= 5, str(checked))

    check("and showing them is still not offering them for writing",
          not (set(form._absent) & set(form.values())),
          str(sorted(set(form._absent) & set(form.values()))))


def test_a_half_typed_pair_is_left_out_not_raised(app) -> None:
    """``0.5, -`` is a normal state mid-keystroke; the changed slot must not raise."""
    from trackanno.gui.configform import ConfigForm

    form = next(f for f in (ConfigForm("basic"), ConfigForm("advanced"))
                if "normalize.percentiles" in f._widgets)
    widget = form._widgets["normalize.percentiles"]
    for text in ("0.5, -", "0.5, 99.5, 1", "x, y"):
        widget.setText(text)
        check(f"{text!r} is left out of the values",
              "normalize.percentiles" not in form._raw_values())
        check("...and is warned about", "normalize.percentiles" in form.warnings())
    widget.setText("0.5, 99.5")
    check("a complete pair parses", form.values()["normalize.percentiles"] == [0.5, 99.5])


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

    answers: list[bool] = []
    page._ask_discard = lambda: answers.pop(0)
    answers.append(False)
    hub.go_to(0)
    app.processEvents()
    check("Stay keeps the Configure page", hub.stack.currentWidget() is page)
    check("...and the edit", stride.edit.value() == 999)

    answers.append(True)
    hub.go_to(0)
    app.processEvents()
    check("Discard leaves the page", hub.stack.currentWidget() is not page)
    check("...and reverts the field to the file", stride.auto.isChecked())
    check("...and nothing is pending", not page._dirty and page.maybe_save())

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
