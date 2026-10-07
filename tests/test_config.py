"""The config schema: defaults, coercion, migrations and the shipped yaml."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import yaml

from trackanno.config import (
    FIXED_SETTINGS, SECTIONS, VIEW_SETTINGS, ConfigError, load_config, schema_keys,
    section_defaults,
)
from trackanno.project import DEFAULT_CONFIG, create_project, write_config
from trackanno.yamledit import set_values


@pytest.fixture(autouse=True)
def _private_state(tmp_path, monkeypatch):
    """Keep create_project's recent list out of the real user state."""
    monkeypatch.setenv("TRACKANNO_STATE_DIR", str(tmp_path / "state"))


def _plain(value):
    if isinstance(value, tuple):
        return list(value)
    return value


def test_the_shipped_yaml_holds_every_default() -> None:
    data = yaml.safe_load(DEFAULT_CONFIG)
    for name in SECTIONS:
        defaults = section_defaults(name)
        for key, default in defaults.items():
            assert data[name][key] == _plain(default) or (
                key == "embeddings" and data[name][key] == [] and default == []), \
                f"{name}.{key}: yaml {data[name][key]!r}, dataclass {default!r}"
    paths = {f.name: f.default for f in SECTIONS["paths"].__dataclass_fields__.values()
             if not callable(f.default_factory)}
    for key in ("video_dir", "model", "work_dir", "csv"):
        assert data["paths"][key] == paths[key]


def test_the_shipped_yaml_documents_every_key() -> None:
    data = yaml.safe_load(DEFAULT_CONFIG)
    written = {f"{name}.{key}" for name, block in data.items() if isinstance(block, dict)
               for key in block if name in SECTIONS}
    assert written == set(schema_keys())
    for dotted in list(FIXED_SETTINGS) + list(VIEW_SETTINGS):
        section, _, key = dotted.partition(".")
        assert key not in (data.get(section) or {}), f"{dotted} is still in the yaml"


def _project(tmp_path: Path, **updates) -> Path:
    path = create_project(tmp_path / "p")
    if updates:
        text = set_values(path.read_text(encoding="utf-8"), updates)
        path.write_text(text, encoding="utf-8")
    return path


def _messages(path: Path) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_config(path)
    return [str(w.message) for w in caught]


def test_values_are_coerced(tmp_path: Path) -> None:
    path = _project(tmp_path, **{"ui.shuffle": "false", "paths.patterns": "*.tif",
                                 "clips.pad": "2", "tiling.enabled": "true",
                                 "detection.device": 0, "suggest.select_rule": False})
    cfg = load_config(path)
    assert cfg.ui.shuffle is False
    assert cfg.paths.patterns == ["*.tif"]
    assert cfg.clips.pad == 2.0
    assert cfg.tiling.enabled is True
    assert cfg.detection.device == "0"
    assert cfg.suggest.select_rule == "off"


@pytest.mark.parametrize("dotted, value, needle", [
    ("detection.conf", 1.5, "detection.conf must be in [0.0, 1.0]"),
    ("clips.frames", "many", "clips.frames must be a whole number"),
    ("ui.theme", "pink", "ui.theme must be one of"),
    ("suggest.audit_fraction", 1.0, "suggest.audit_fraction must be in [0.0, 1.0)"),
    ("predict.window_stride", "often", "predict.window_stride must be a whole number or 'auto'"),
])
def test_bad_values_are_refused_in_one_line(tmp_path: Path, dotted, value, needle) -> None:
    path = _project(tmp_path, **{dotted: value})
    with pytest.raises(ConfigError) as caught:
        load_config(path)
    assert needle in str(caught.value)
    assert "\n" not in str(caught.value)


def test_unknown_keys_and_sections_warn(tmp_path: Path) -> None:
    path = _project(tmp_path, **{"clips.framez": 3, "nonsense.key": 1})
    said = _messages(path)
    assert any("clips.framez" in m for m in said)
    assert any("unknown config section 'nonsense'" in m for m in said)


def test_fixed_settings_load_with_a_warning(tmp_path: Path) -> None:
    path = _project(tmp_path, **{"detection.iou": 0.5, "suggest.depth": 8,
                                 "clips.crop_from": "bbox"})
    said = _messages(path)
    assert "detection.iou is no longer configurable; using 0.5" in said
    assert "clips.crop_from is no longer configurable; using 'bbox'" in said
    loud = [m for m in said if m.startswith("suggest.depth")]
    assert loud and "NOT the 8" in loud[0]


# Values real projects set away from the defaults; each must take effect.
PROJECT_VALUES = {
    "detection.imgsz": 1280, "detection.half": True,
    "tiling.overlap": 0.4, "tiling.merge_iou": 0.65,
    "suggest.iterations": 1500, "suggest.learning_rate": 0.05, "suggest.l2_leaf_reg": 4.0,
    "predict.min_frames": 10, "ui.kymo_sign": "raw",
}


def test_project_tuned_settings_are_honoured(tmp_path: Path) -> None:
    path = _project(tmp_path, **PROJECT_VALUES)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = load_config(path)
    said = [str(w.message) for w in caught]
    assert not [m for m in said if "no longer configurable" in m], said
    for dotted, value in PROJECT_VALUES.items():
        section, _, key = dotted.partition(".")
        assert getattr(getattr(cfg, section), key) == value, dotted
        assert dotted not in FIXED_SETTINGS

    # And the code that runs reads them.
    from types import SimpleNamespace

    from trackanno.detect.tiling import TiledDetector, tile_grid
    from trackanno.learn.suggest import _build_model

    detector = SimpleNamespace(cfg=cfg, device="cpu", _grid_cache={})
    kwargs = TiledDetector._predict_kwargs(detector, cfg.detection.imgsz)
    assert kwargs["imgsz"] == 1280 and ("half" in kwargs or "quantize" in kwargs)
    cfg.tiling.enabled = True
    assert TiledDetector.tiles_for(detector, 2000, 2000) == tile_grid(
        2000, 2000, cfg.tiling.tile, 0.4)
    params = _build_model(cfg, 3).get_params()
    if "iterations" in params:
        assert (params["iterations"], params["learning_rate"], params["l2_leaf_reg"]) \
            == (1500, 0.05, 4.0)
    else:
        assert (params["max_iter"], params["learning_rate"],
                params["l2_regularization"]) == (1500, 0.05, 4.0)


def test_old_kymo_sign_maps_to_canonical(tmp_path: Path) -> None:
    path = _project(tmp_path, **{"ui.kymo_sign": "absolute"})
    with pytest.warns(UserWarning, match="'absolute' was removed"):
        cfg = load_config(path)
    assert cfg.ui.kymo_sign == "canonical"


def test_view_toggles_moved_out_of_the_config(tmp_path: Path) -> None:
    path = _project(tmp_path, **{"ui.show_overlay": False, "ui.view_zoom": 2.0})
    said = _messages(path)
    assert sum("per-user view setting" in m for m in said) == 2


def test_removed_values_map_to_their_replacement(tmp_path: Path) -> None:
    path = _project(tmp_path, **{"normalize.mode": "smooth"})
    with pytest.warns(UserWarning, match="'smooth' was removed"):
        cfg = load_config(path)
    assert cfg.normalize.mode == "per_frame"


def test_yamledit_block_lists_nesting_and_numpy(tmp_path: Path) -> None:
    text = ("paths:\n  patterns:\n    - '*.tif'\n    - '*.nd2'\n  csv: a.csv\n"
            "tracking:\n  overrides:\n    # camera motion\n    gmc_method: none\n"
            "  backend: greedy\n")
    out = set_values(text, {"paths.patterns": ["*.nd2"], "tracking.overrides.track_buffer": 5,
                            "tracking.overrides.gmc_method": "sof",
                            "detection.conf": np.float64(0.5), "detection.batch": np.int64(5)})
    data = yaml.safe_load(out)
    assert data["paths"] == {"patterns": ["*.nd2"], "csv": "a.csv"}
    assert data["tracking"]["overrides"] == {"gmc_method": "sof", "track_buffer": 5}
    assert data["tracking"]["backend"] == "greedy"
    assert "tracking.overrides" not in data
    assert "conf: 0.5" in out and "batch: 5" in out
    assert "# camera motion" in out

    flow = set_values("tracking:\n  overrides: {}\n", {"tracking.overrides.x": 1})
    assert yaml.safe_load(flow) == {"tracking": {"overrides": {"x": 1}}}


def test_a_written_config_still_loads(tmp_path: Path) -> None:
    path = create_project(tmp_path / "w")
    write_config(path, {"tracking.overrides.track_buffer": np.int64(60)})
    assert load_config(path).tracking.overrides["track_buffer"] == 60


def test_view_toggles_are_per_user_settings(app) -> None:
    from trackanno.gui.viewprefs import load_view, save_view

    assert load_view() == {k.split(".", 1)[1]: v for k, v in VIEW_SETTINGS.items()}
    save_view(show_overlay=False, view_zoom=2.5, follow="locked")
    view = load_view()
    assert view["show_overlay"] is False and view["view_zoom"] == 2.5
    assert view["follow"] == "locked"


def _one_option(name: str, kind: str = "single", key: str = "1") -> dict:
    return {"name": name, "type": kind, "required": kind == "single",
            "options": [{"name": f"o{key}", "key": key, "color": "#4C9F70"}]}


@pytest.mark.parametrize("bad", ["notes", "clip_id", "annotator"])
def test_a_group_named_after_a_csv_column_is_refused(tmp_path: Path, bad) -> None:
    path = create_project(tmp_path / "reserved")
    with pytest.raises(ConfigError, match="(?i)collides"):
        write_config(path, {}, categories=[_one_option(bad)])
    assert load_config(path) is not None, "the project still loads"


def test_two_groups_with_one_name_are_refused(tmp_path: Path) -> None:
    path = create_project(tmp_path / "dup")
    with pytest.raises(ConfigError, match="named"):
        write_config(path, {}, categories=[_one_option("dup"),
                                           _one_option("dup", "multi", "2")])


@pytest.mark.parametrize("bad", [0, -5])
def test_a_stride_below_one_is_refused(tmp_path: Path, bad) -> None:
    """clips.stride is the window loop's increment, so 0 would never end."""
    path = create_project(tmp_path / "stride")
    with pytest.raises(ConfigError, match="at least 1"):
        write_config(path, {"clips.stride": bad})
    assert load_config(path).clips.stride >= 1


def test_repeated_saves_do_not_grow_the_file(tmp_path: Path) -> None:
    path = create_project(tmp_path / "grow")
    cats = [{"name": "state", "type": "single", "required": True,
             "options": [{"name": "a", "key": "1", "color": "#4C9F70"}]}]
    sizes = []
    for i in range(6):
        write_config(path, {"clips.pad": 1.0 + i / 10}, categories=cats,
                     keypoints={"nodes": [], "edges": []})
        sizes.append(len(path.read_text(encoding="utf-8").splitlines()))
    text = path.read_text(encoding="utf-8")
    assert len(set(sizes[1:])) == 1, sizes
    assert text.count("#") > 40, "the comments are still there"
    assert yaml.safe_load(text)["categories"] == cats


def test_a_new_project_selects_nothing(tmp_path: Path) -> None:
    """No rule ranks the queue, so the labelled set stays a random sample."""
    cfg = load_config(create_project(tmp_path / "queue"))
    assert cfg.suggest.select_rule == "off"
    assert 0.0 < cfg.suggest.audit_fraction < 1.0, "a holdout exists to estimate against"


def test_the_shipped_template_loads_without_a_warning(tmp_path: Path) -> None:
    path = create_project(tmp_path / "quiet")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert load_config(path) is not None
