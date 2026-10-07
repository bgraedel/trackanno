"""The shipped presets: each key exists in the default, each loads clean, each rung does its job."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import yaml
from trackanno.config import load_config
from trackanno.project import DEFAULT_CONFIG, create_project
from trackanno.templates import find_preset, list_presets, preset_text

LADDER = ["annotator", "suggester", "bouts"]


def _config(root: Path, preset: str, where: str = ""):
    """The config a project made from ``preset`` is read as, in its own directory."""
    directory = Path(root) / (where or f"preset-{preset}")
    return load_config(create_project(directory, template=preset))


def test_the_presets_are_the_documented_ladder() -> None:
    presets = list_presets()
    assert [p.id for p in presets] == LADDER
    assert all(p.summary for p in presets), "a one-line summary for the picker"
    assert all(len(p.about) > len(p.summary) for p in presets), "a longer tooltip"


def test_a_preset_is_found_by_file_name_name_or_slug() -> None:
    for p in list_presets():
        for key in (p.id, p.name, p.name.replace(" ", "-")):
            assert find_preset(key) is not None, key
    assert find_preset("no such preset") is None


def test_every_preset_key_is_a_key_the_default_config_has() -> None:
    """An override of a missing key is appended and then ignored, so catch it here."""
    default = yaml.safe_load(DEFAULT_CONFIG)
    for preset in list_presets():
        for dotted in preset.overrides:
            section, _, key = dotted.rpartition(".")
            assert section in default, f"{preset.id}: {dotted}"
            assert key in (default.get(section) or {}), f"{preset.id}: {dotted}"


@pytest.mark.parametrize("preset", LADDER)
def test_every_preset_loads_without_a_single_warning(root: Path, preset: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _config(root, preset, where=f"load-{preset}")


def test_an_unnamed_project_gets_the_packaged_default() -> None:
    assert preset_text(None) == DEFAULT_CONFIG


def test_a_preset_keeps_the_comments_of_the_default() -> None:
    text = preset_text("annotator")
    assert len(text.splitlines()) == len(DEFAULT_CONFIG.splitlines())
    assert "# A classifier trained on your labels suggests a category" in text
    assert "suggest:\n  enabled: false" in text, "the key itself changed"


def test_the_annotator_preset_is_the_annotation_screen_and_nothing_else(root: Path) -> None:
    """Features stay on, so the next rung needs no re-detection."""
    cfg = _config(root, "annotator")
    assert cfg.suggest.enabled is False
    assert cfg.detection.task == "detect"
    assert cfg.features.enabled is True


def test_the_suggester_preset_is_pose_from_the_start(root: Path) -> None:
    cfg = _config(root, "suggester")
    assert cfg.suggest.enabled is True
    assert cfg.detection.task == "pose"
    assert cfg.keypoints.nodes == [], "node names are left to the checkpoint"
    assert set(cfg.suggest.feature_groups) == {"kinematic", "pose", "waveform"}
    assert cfg.render.show_skeleton is True
    assert cfg.predict.min_bout_frames == 0


def test_the_dense_bouts_preset_measures_25_frames_and_steps_5(root: Path) -> None:
    cfg = _config(root, "bouts")
    assert cfg.detection.task == "pose" and cfg.suggest.enabled is True
    assert set(cfg.suggest.feature_groups) == {"kinematic", "pose", "waveform"}
    assert cfg.clips.frames == 25
    assert cfg.predict.resolved_stride(cfg.clips) == 5
    assert cfg.predict.min_bout_frames >= 2, "a shortest bout to be sticky about"


@pytest.mark.parametrize("preset", LADDER)
def test_the_dense_window_is_the_length_the_model_is_fitted_at(root: Path, preset) -> None:
    """A window shorter than the clips it is cut from would never fill."""
    cfg = _config(root, preset, where=f"window-{preset}")
    assert cfg.clips.min_clip_frames <= cfg.clips.frames


def test_a_preset_cannot_be_deleted_or_written_over(root: Path) -> None:
    from trackanno.templates import delete_template, save_template

    source = create_project(Path(root) / "preset-clash")
    assert not delete_template("annotator")
    assert find_preset("annotator") is not None
    with pytest.raises(ValueError, match="preset"):
        save_template("dense bouts", source)
