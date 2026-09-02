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

# The keys the GUI could put a project into a state over and then not offer the
# fix for: clips.frames and clips.stride are on the form, and an overlapping
# clip grid is refused by load_config naming a predict.* key.
NEW_KEYS = {
    "predict.coverage": "manifest",
    "predict.window_scales": [25, 50, 100],
    "predict.min_frames": 7,
    "predict.self_transition": 0.98,
    "predict.save_posteriors": True,
    "suggest.feature_scope": "length_invariant",
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
    coupled = ["predict.window_frames", "predict.window_stride", "predict.window_scales",
               "predict.decode", "predict.min_bout_frames", "suggest.feature_scope"]
    check("the keys that are refused over each other are one group",
          len({group_of[k] for k in coupled}) == 1,
          str({k: group_of[k] for k in coupled}))
    check("...including the one from another config section",
          group_of["suggest.feature_scope"] == group_of["predict.window_scales"])


def test_the_prediction_keys_the_gui_can_break_are_on_it(app, root: Path) -> None:
    """Six keys the form did not offer, and one combination that needs all of them."""
    from trackanno.config import load_config
    from trackanno.gui.configform import ConfigForm
    from trackanno.project import create_project, write_config

    advanced = ConfigForm("advanced")
    check("all six new keys are on a tab",
          all(key in advanced._widgets for key in NEW_KEYS),
          str(sorted(set(NEW_KEYS) - set(advanced._widgets))))

    form = ConfigForm("all")
    form.load(_raw(NEW_KEYS))
    values = form.values()
    for dotted, value in NEW_KEYS.items():
        check(f"{dotted} round-trips through the form",
              values[dotted] == value, repr(values.get(dotted)))

    # And the combination they exist for has to survive the real loader: several
    # window scales need a decode, dense coverage and a length-invariant model.
    path = create_project(root / "predict-keys")
    coherent = {
        "clips.frames": 50, "clips.stride": 50,
        "predict.coverage": "dense", "predict.window_frames": 50,
        "predict.window_scales": [25, 50, 100], "predict.decode": "viterbi",
        "predict.min_bout_frames": 25, "predict.self_transition": 0.98,
        "predict.min_frames": 4, "predict.save_posteriors": True,
        "suggest.feature_scope": "length_invariant",
    }
    write_config(path, coherent)
    cfg = load_config(path)
    check("a multi-scale project written from these fields loads",
          cfg.predict.resolved_scales(cfg.clips) == [50, 25, 100],
          str(cfg.predict.resolved_scales(cfg.clips)))
    check("...and every value survived the round trip",
          cfg.predict.min_frames == 4 and cfg.predict.save_posteriors
          and cfg.suggest.feature_scope == "length_invariant"
          and abs(cfg.predict.resolved_self_transition() - 0.98) < 1e-9)

    form = ConfigForm("all")
    form.load(yaml.safe_load(path.read_text(encoding="utf-8")))
    written = form.values()
    check("...and the form reads back what it wrote",
          all(written[k] == v for k, v in coherent.items()),
          str({k: written[k] for k, v in coherent.items() if written[k] != v}))


def test_a_malformed_window_scales_is_refused_with_a_message(app) -> None:
    """A list of integers in a text box. The form has to refuse a bad one itself:
    handing it to load_config refuses the whole file over one stray character,
    on the keystroke that typed it."""
    from trackanno.gui.configform import ConfigForm

    form = ConfigForm("advanced")
    widget = form._widgets["predict.window_scales"]

    widget.edit.setText("25, fifty, 100")
    problems = form.problems()
    check("a malformed entry is refused", "predict.window_scales" in problems)
    check("...and the message names the entry that is wrong",
          "fifty" in problems["predict.window_scales"], problems["predict.window_scales"])
    check("...and nothing is offered for writing",
          "predict.window_scales" not in form.values())
    # isHidden, not isVisible: the form is not the page on screen.
    check("...and the field says so where it is typed",
          not widget.note.isHidden() and "fifty" in widget.note.text())

    widget.edit.setText("50, 50")
    check("a repeated scale is refused too, and says why",
          "twice" in form.problems()["predict.window_scales"])

    widget.edit.setText("0, 50")
    check("a scale of no frames is refused",
          "predict.window_scales" in form.problems())

    widget.edit.setText("25, 50, 100")
    check("a good list clears the refusal", not form.problems())
    check("...and parses to whole numbers",
          form.values()["predict.window_scales"] == [25, 50, 100])
    check("...and the note goes away", widget.note.isHidden())

    widget.edit.setText("  100  50 ")
    check("whitespace alone separates them too",
          form.values()["predict.window_scales"] == [100, 50])
    widget.edit.setText("")
    check("empty is the empty list, which is what 'one scale' means",
          form.values()["predict.window_scales"] == [])


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

    scales = page.advanced._widgets["predict.window_scales"]
    scales.edit.setText("25, fifty")
    app.processEvents()
    check("the autosave refuses it", not page.save() and not page.banner.isHidden())
    check("the banner names the field and the entry",
          "window_scales" in page.banner.text() and "fifty" in page.banner.text(),
          page.banner.text()[:160])
    check("the refused save left the file alone",
          path.read_text(encoding="utf-8") == good)
    check("...and the field is outlined", scales.edit.property("invalid") == "true")

    scales.edit.setText("")
    app.processEvents()
    check("clearing it saves again", page.save() and page.banner.isHidden())
    check("...and the outline goes", scales.edit.property("invalid") is None)

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

    window = form._widgets["predict.window_frames"]
    check("the choice between 'auto' and a number is a tick, not typing",
          window.auto.isChecked() and not window.edit.isEnabled())
    check("'auto' says what it resolves to", window.auto.text() == "auto (50)",
          window.auto.text())
    check("...for every auto field there is",
          form._widgets["suggest.select_frame_gap"].auto.text() == "auto (50)"
          and form._widgets["suggest.select_audit_share"].auto.text() == "auto (0.1)",
          form._widgets["suggest.select_audit_share"].auto.text())
    check("an auto with no answer yet says only 'auto'",
          form._widgets["clips.crop_size"].auto.text() == "auto")
    check("an auto that needs another setting first says which",
          "shortest bout" in form._widgets["predict.self_transition"].auto.text(),
          form._widgets["predict.self_transition"].auto.text())

    # It has to follow the settings it resolves from, which live in another
    # config section and, on the split page, on the other tab.
    form.show_resolved({**values, "clips.frames": 30, "clips.stride": 30})
    check("it follows the setting it depends on", window.auto.text() == "auto (30)",
          window.auto.text())
    check("...and so does the field the same number means something else in",
          form._widgets["suggest.select_frame_gap"].auto.text() == "auto (30)")

    # The refusal the user actually hit: clips.stride below clips.frames makes
    # predict.window_frames: auto unresolvable, and load_config says so about a
    # key on the OTHER tab.
    form.show_resolved({**values, "clips.frames": 50, "clips.stride": 25})
    check("an overlapping clip grid says 'auto' has no answer here",
          "overlap" in window.auto.text(), window.auto.text())
    check("...and the stride that caused it says what to change",
          "Bulk labelling" in form._status["clips.stride"].text(),
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
    check("...and the form offers the number", form.values()["predict.window_frames"] == 50)
    window.edit.setValue(37)
    check("an explicit number is written as one",
          form.values()["predict.window_frames"] == 37)
    window.auto.setChecked(True)
    check("ticking it back writes the word", form.values()["predict.window_frames"] == "auto")

    share = form._widgets["suggest.select_audit_share"]
    share.set_value(0.25)
    check("a share stays a fraction, not an integer",
          form.values()["suggest.select_audit_share"] == 0.25)
    share.set_value("auto")
    check("...and reading 'auto' back ticks the box", share.auto.isChecked())


def test_a_fixed_set_of_values_is_never_free_text(app) -> None:
    """Every setting whose valid values are a fixed set is offered as that set."""
    from trackanno.gui.configform import FIELDS, ConfigForm, _options

    kinds = {f"{section}.{key}": kind for section, key, kind, *_ in FIELDS}
    check("nothing is left as an untyped scalar box",
          not [k for k, kind in kinds.items() if kind == "scalar"],
          str([k for k, kind in kinds.items() if kind == "scalar"]))
    for dotted, expected in (("tiling.enabled", "tristate"),
                             ("clips.crop_size", "autonum"),
                             ("predict.coverage", "choice"),
                             ("suggest.feature_scope", "choice")):
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
    from trackanno.learn.select import TARGET_RULES, UNCERTAINTY_RULES
    from trackanno.detect.tracking import TRACKERS

    extras = {f"{section}.{key}": extra for section, key, _kind, _l, _t, extra in FIELDS}
    check("the selection rules are the registry itself",
          _options(extras["suggest.select_rule"]) == ["off"] + sorted(UNCERTAINTY_RULES))
    check("so are the targeting rules",
          _options(extras["suggest.select_target_rule"]) == sorted(TARGET_RULES))
    check("so are the trackers",
          _options(extras["tracking.backend"]) == sorted(TRACKERS) + ["greedy"])
    check("...and all three are read, not written down",
          all("options_from" in extras[k] and "options" not in extras[k]
              for k in ("suggest.select_rule", "suggest.select_target_rule",
                        "tracking.backend")))

    # Everything else has its allowed values written out inside config.py's own
    # parse functions, with nothing importable to read. Every option is put
    # through the real loader instead, so a list that goes stale fails here.
    path = create_project(root / "combos")
    base = path.read_text(encoding="utf-8")
    # predict.decode is the one option that cannot stand alone: min_bout and
    # viterbi are both refused without a shortest bout to enforce.
    companions = {"predict.decode": {"predict.min_bout_frames": 25}}
    offered = 0
    for section, key, kind, _label, _tip, extra in FIELDS:
        if kind != "choice":
            continue
        dotted = f"{section}.{key}"
        if dotted == "detection.device":
            continue  # filled in from whatever this machine reports
        for option in _options(extra):
            path.write_text(base, encoding="utf-8")
            try:
                write_config(path, {dotted: option, **companions.get(dotted, {})})
            except ConfigError as exc:
                check(f"{dotted} offers only values the loader takes", False,
                      f"{option!r}: {exc}")
            offered += 1
        path.write_text(base, encoding="utf-8")
        try:
            write_config(path, {dotted: "not-a-real-option"})
        except ConfigError:
            pass
        else:
            check(f"{dotted} refuses a value it does not offer", False)
    check("every value every combo offers is one the loader accepts",
          offered > 35, str(offered))


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
    check("there were autonum fields to check", checked >= 4, str(checked))


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
