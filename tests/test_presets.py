"""The shipped presets: what each one claims, and that all of them load.

A preset is the first thing a new project is made of, so a broken one is a broken
first hour. Each names only the keys it changes, which keeps it from drifting
away from `default_config.yaml` and also means nothing checks those keys: a
renamed section would be appended to the config as a new one and then ignored by
the loader, with a warning nobody reads. Hence the two general claims here, that
every key exists in the default and that every preset loads clean, and then one
test per rung of the ladder for what that rung is for.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import yaml
from harness import check
from trackanno.config import load_config
from trackanno.project import DEFAULT_CONFIG, create_project
from trackanno.templates import find_preset, list_presets, preset_text

LADDER = ["annotator", "suggester", "bouts"]


def _config(root: Path, preset: str, where: str = ""):
    """The config a project made from `preset` would be read as.

    A directory per caller: `root` is one scratch dir for the whole module and
    `create_project` refuses to write over an existing config.
    """
    directory = Path(root) / (where or f"preset-{preset}")
    return load_config(create_project(directory, template=preset))


def test_the_presets_are_the_ladder_they_are_documented_as() -> None:
    """Four of them, in order, each with a line saying what it is for."""
    presets = list_presets()

    check("the shipped presets are the four in the docs",
          [p.id for p in presets] == LADDER, str([p.id for p in presets]))
    check("each carries a one-line summary for the picker",
          all(p.summary for p in presets),
          str([p.id for p in presets if not p.summary]))
    check("and a longer description for the tooltip",
          all(len(p.about) > len(p.summary) for p in presets),
          str([p.id for p in presets if len(p.about) <= len(p.summary)]))
    check("each is found by its file name, its name and the name's slug",
          all(find_preset(x) is not None
              for p in presets for x in (p.id, p.name, p.name.replace(" ", "-"))))
    check("a name that is not a preset is not one",
          find_preset("no such preset") is None)


def test_every_preset_key_is_a_key_the_default_config_has() -> None:
    """An override of a key that moved is silently nothing. Catch it here.

    `set_values` appends a key it cannot find rather than failing, and the loader
    ignores an unknown section outright, so this is the only place the two files
    are held against each other.
    """
    default = yaml.safe_load(DEFAULT_CONFIG)

    for preset in list_presets():
        for dotted in preset.overrides:
            section, _, key = dotted.rpartition(".")
            check(f"{preset.id}: {dotted} names a section config.yaml has",
                  section in default, f"{section} is not in {sorted(default)}")
            check(f"{preset.id}: {dotted} names a key that section has",
                  key in (default.get(section) or {}),
                  f"{key} is not in {sorted(default.get(section) or {})}")


def test_every_preset_loads_without_a_single_warning(root: Path) -> None:
    """A preset that warns on load ships a broken project to a new user.

    Warnings are errors here for the reason `test_defaults_lock` makes them so:
    an unregistered key is ignored with a warning, and a warning in a passing
    suite is a warning nobody sees.
    """
    for preset in list_presets():
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cfg = _config(root, preset.id, where=f"load-{preset.id}")
        check(f"{preset.id} loads clean", cfg is not None)

    check("and the packaged default is still what an unnamed project gets",
          preset_text(None) == DEFAULT_CONFIG)


def test_a_preset_keeps_the_comments_of_the_default() -> None:
    """Overrides are rewritten in place, so the documentation survives them."""
    text = preset_text("annotator")

    check("the file is as long as the one it was made from",
          len(text.splitlines()) == len(DEFAULT_CONFIG.splitlines()))
    check("the comment above the key it changed is still there",
          "# A classifier trained on your labels suggests a category" in text)
    check("and the key itself changed",
          "suggest:\n  enabled: false" in text)


def test_the_annotator_preset_is_the_annotation_screen_and_nothing_else(root: Path) -> None:
    """The bottom rung: a detector, clips, and a person pressing keys.

    Features stay ON. They are pooled from the detection pass, so switching them
    off would save one file and cost a whole re-detection to the first person who
    then wants a suggestion.
    """
    cfg = _config(root, "annotator")

    check("no classifier runs", cfg.suggest.enabled is False)
    check("detection is boxes, not keypoints", cfg.detection.task == "detect")
    check("per-detection features are still written, so the next rung is free",
          cfg.features.enabled is True)


def test_the_suggester_preset_is_pose_from_the_start(root: Path) -> None:
    """One rung up: keypoints, and the classifier fitted on what they measure.

    Pose is not a separate rung. A model that can see the flagellum beat is the
    one worth training, so every preset that fits a classifier asks for keypoints
    and for the feature groups they unlock.
    """
    cfg = _config(root, "suggester")

    check("the suggester is on", cfg.suggest.enabled is True)
    check("detection predicts keypoints, not boxes alone",
          cfg.detection.task == "pose", cfg.detection.task)
    check("the node names are left to the checkpoint, not guessed here",
          cfg.keypoints.nodes == [], str(cfg.keypoints.nodes))
    check("the pose and waveform groups are asked for alongside the rest",
          set(cfg.suggest.feature_groups) == {"kinematic", "pose", "waveform"},
          str(cfg.suggest.feature_groups))
    check("the skeleton is drawn on the export", cfg.render.show_skeleton is True)
    check("no shortest bout is imposed", cfg.predict.min_bout_frames == 0)


def test_the_dense_bouts_preset_measures_25_frames_and_steps_5(root: Path) -> None:
    """The rung the boundary work exists for: frames, not clips.

    The two numbers are the whole preset. 25 frames is what every measurement is
    made over, and a step of 5 is what pins a change of behaviour to five frames
    rather than to a window.
    """
    cfg = _config(root, "bouts")

    check("it is the suggester preset underneath",
          cfg.detection.task == "pose" and cfg.suggest.enabled is True
          and set(cfg.suggest.feature_groups)
          == {"kinematic", "pose", "waveform"},
          f"{cfg.detection.task}, {cfg.suggest.feature_groups}")
    check("the dense window is 25 frames", cfg.clips.frames == 25,
          str(cfg.clips.frames))
    check("and steps 5, so the windows overlap five to one",
          cfg.predict.resolved_stride(cfg.clips) == 5,
          str(cfg.predict.resolved_stride(cfg.clips)))
    check("the decode has a shortest bout to be sticky about",
          cfg.predict.min_bout_frames >= 2, str(cfg.predict.min_bout_frames))


def test_the_dense_window_is_the_length_the_model_is_fitted_at(root: Path) -> None:
    """The dense window is `clips.frames`, so every preset's clips must fill."""
    for preset in list_presets():
        cfg = _config(root, preset.id, where=f"window-{preset.id}")
        check(f"{preset.id}: a window shorter than the clips it is cut from "
              "would never fill", cfg.clips.min_clip_frames <= cfg.clips.frames,
              f"{cfg.clips.min_clip_frames} > {cfg.clips.frames}")


def test_a_preset_cannot_be_deleted_or_written_over(root: Path) -> None:
    """They ship with the package, so the state directory has no say in them."""
    from trackanno.templates import delete_template, save_template

    source = create_project(Path(root) / "preset-clash")

    check("forgetting a preset does nothing", not delete_template("annotator"))
    check("it is still there", find_preset("annotator") is not None)

    refused = False
    try:
        save_template("dense bouts", source)
    except ValueError as exc:
        refused = "preset" in str(exc)
    check("and saving a template under a preset name is refused", refused)
