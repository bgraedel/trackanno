"""Choosing the model's input by group and by column, and refusing the choices that contradict."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pytest

GROUP = "behaviour"


def _raises(fn, needle: str) -> str:
    """Run ``fn`` and return the refusal's message, insisting it names ``needle``."""
    with pytest.raises(Exception) as caught:
        fn()
    message = str(caught.value)
    assert needle in message, message
    return message


def _project(root: Path, name: str, **settings) -> Path:
    """A fresh project with `settings` written into its config."""
    from trackanno.project import create_project, write_config

    path = create_project(Path(root) / name)
    if settings:
        write_config(path, settings)
    return path


def _hand_edit(path: Path, **settings) -> Path:
    """Write settings without validation, as a text editor would."""
    from trackanno.project import write_config

    return write_config(path, settings, validate=False)


# --------------------------------------------------------------------------- #
# the loader
# --------------------------------------------------------------------------- #
def test_a_misspelt_column_is_refused_with_a_suggestion(tmp_path: Path) -> None:
    from trackanno.config import load_config
    from trackanno.project import write_config

    path = _project(tmp_path, "vocabulary")
    _hand_edit(path, **{"suggest.feature_columns": ["speed_men", "area_mean"]})
    message = _raises(lambda: load_config(path), "speed_men")
    assert "speed_mean" in message, f"offers the column that was meant: {message}"
    _raises(lambda: write_config(path, {"suggest.feature_columns": ["speed_men"]}),
            "speed_men")


def test_a_repeated_column_is_refused(tmp_path: Path) -> None:
    from trackanno.config import load_config

    path = _project(tmp_path, "repeat")
    _hand_edit(path, **{"suggest.feature_columns": ["speed_mean", "speed_mean"]})
    _raises(lambda: load_config(path), "more than once")


def test_an_empty_column_list_points_at_null(tmp_path: Path) -> None:
    from trackanno.config import load_config

    path = _project(tmp_path, "empty")
    _hand_edit(path, **{"suggest.feature_columns": []})
    message = _raises(lambda: load_config(path), "empty list")
    assert "null" in message, message


def test_embedding_columns_match_by_shape_and_cnn_maps_to_det(tmp_path: Path) -> None:
    from trackanno.config import load_config
    from trackanno.project import write_config

    path = _project(tmp_path, "embedding")
    write_config(path, {"suggest.feature_groups": ["kinematic", "cnn"],
                        "suggest.feature_columns": ["speed_mean", "cnn0_mean",
                                                    "emb_det_31_delta"]})
    with pytest.warns(UserWarning, match="emb_det"):
        cfg = load_config(path)
    assert (cfg.suggest.feature_columns
            == ["speed_mean", "emb_det_0_mean", "emb_det_31_delta"]), (
        cfg.suggest.feature_columns)

    write_config(path, {"suggest.feature_columns": None,
                        "suggest.feature_groups": ["kinematic"]})
    assert load_config(path).suggest.feature_columns is None, "null turns the cut off"


def test_naming_a_column_from_a_group_that_is_off_is_refused(root: Path) -> None:
    """The two keys would disagree about what the model may see."""
    from trackanno.config import load_config
    from trackanno.project import write_config

    path = _project(root, "contradiction")
    _hand_edit(path, **{
        "suggest.feature_groups": ["kinematic"],
        "suggest.feature_columns": ["speed_mean", "wave_freq_hz"],
    })
    message = _raises(lambda: load_config(path), "wave_freq_hz")
    assert "waveform" in message, f"names the group it belongs to: {message}"
    assert "feature_groups" in message and "feature_columns" in message, (
        f"names both keys: {message}")
    assert "add the group" in message.lower() and "drop those" in message.lower(), (
        f"offers both fixes: {message}")

    # The same list under the group that owns it loads.
    write_config(path, {"suggest.feature_groups": ["kinematic", "waveform"]})
    cfg = load_config(path)
    assert cfg.suggest.feature_columns == ["speed_mean", "wave_freq_hz"], (
        f"with the group ticked the same list is fine: {cfg.suggest.feature_columns}")


# --------------------------------------------------------------------------- #
# the fit
# --------------------------------------------------------------------------- #
def _labelled_project(root: Path, name: str, **settings):
    """A copy of the shared fixture, labelled, ready to train."""
    from fixtures import make_project
    from trackanno.annotate.store import AnnotationStore
    from trackanno.config import load_config
    from trackanno.project import write_config

    scratch = Path(tempfile.mkdtemp(dir=root, prefix=f"{name}-"))
    fixture = make_project(Path(tempfile.mkdtemp(dir=root, prefix=f"{name}-fix-")))
    project = scratch / "project"
    project.mkdir(parents=True, exist_ok=True)
    config_path = project / "config.yaml"
    shutil.copy(fixture, config_path)
    write_config(config_path, {
        "paths.work_dir": str(project / "work").replace("\\", "/"),
        "paths.csv": str(project / "annotations.csv").replace("\\", "/"),
        **settings,
    })
    cfg = load_config(config_path)
    work = cfg.paths.work_dir
    work.mkdir(parents=True, exist_ok=True)

    manifests = []
    for movie_dir in sorted(p for p in load_config(fixture).paths.work_dir.iterdir()
                            if p.is_dir()):
        if not (movie_dir / "clips.parquet").exists():
            continue
        (work / movie_dir.name).mkdir(parents=True, exist_ok=True)
        for fname in ("clips.parquet", "tracks.parquet", "cache.json"):
            if (movie_dir / fname).exists():
                shutil.copy(movie_dir / fname, work / movie_dir.name / fname)
        manifests.append(pd.read_parquet(movie_dir / "clips.parquet"))
    manifest = pd.concat(manifests, ignore_index=True)

    store = AnnotationStore(cfg.paths.csv, cfg)
    for i, row in enumerate(manifest.to_dict("records")):
        store.record(row, {GROUP: ("swimming", "immotile")[i % 2]})
    return config_path, cfg


def test_the_named_columns_are_the_ones_the_payload_records(root: Path) -> None:
    """Later steps read the columns from the payload, so it must hold what was fitted."""
    import joblib

    from trackanno.config import load_config
    from trackanno.learn.suggest import MODEL_FILE, Suggester, train
    from trackanno.project import write_config

    config_path, cfg = _labelled_project(root, "named")
    train(cfg, rebuild_features=True, verbose=False)
    wide = list(joblib.load(cfg.paths.work_dir / MODEL_FILE)["columns"])
    assert len(wide) > 4, f"the unnarrowed fit uses many columns: {len(wide)}"

    picked = ["speed_mean", "area_mean", "straightness", "immotile_frac"]
    write_config(config_path, {
        "suggest.feature_groups": ["kinematic"],
        "suggest.feature_columns": picked,
    })
    train(load_config(config_path), rebuild_features=False, verbose=False)
    narrow = list(joblib.load(cfg.paths.work_dir / MODEL_FILE)["columns"])

    assert narrow == [c for c in wide if c in set(picked)], (
        f"the payload holds exactly what was named: {narrow}")
    assert len(narrow) < len(wide), (
        f"which is fewer than the groups alone would have given: {len(narrow)} vs {len(wide)}")

    suggester = Suggester(load_config(config_path))
    assert list(suggester.columns) == narrow, (
        f"the model loads against those columns: {suggester.columns}")


# --------------------------------------------------------------------------- #
# where the fit runs
# --------------------------------------------------------------------------- #
def test_an_unknown_or_impossible_device_is_refused(tmp_path: Path) -> None:
    """CatBoost's GPU fits a different model, so the device is a modelling setting."""
    from trackanno.config import load_config

    path = _project(tmp_path, "device-set")
    _hand_edit(path, **{"suggest.device": "cuda"})
    message = _raises(lambda: load_config(path), "cuda")
    assert all(v in message for v in ("cpu", "gpu", "auto")), message

    _hand_edit(path, **{"suggest.device": "gpu", "suggest.model": "hist"})
    message = _raises(lambda: load_config(path), "hist")
    assert "suggest.device" in message and "suggest.model" in message, message


def test_the_device_resolves_to_what_this_machine_has(tmp_path: Path) -> None:
    from trackanno.config import load_config
    from trackanno.learn.suggest import gpu_devices, resolved_device
    from trackanno.project import write_config

    path = _project(tmp_path, "device", **{"suggest.device": "cpu"})
    assert resolved_device(load_config(path)) == "cpu"

    write_config(path, {"suggest.device": "auto", "suggest.model": "catboost"})
    assert resolved_device(load_config(path)) == ("gpu" if gpu_devices() else "cpu")

    write_config(path, {"suggest.device": "gpu"})
    cfg = load_config(path)
    if gpu_devices():
        assert resolved_device(cfg) == "gpu"
    else:
        _raises(lambda: resolved_device(cfg), "no GPU")


# --------------------------------------------------------------------------- #
# feature groups
# --------------------------------------------------------------------------- #
def test_a_detect_project_offers_only_kinematic_groups(tmp_path: Path) -> None:
    from trackanno.config import load_config
    from trackanno.measure.clipfeatures import available_groups

    availability = available_groups(load_config(_project(tmp_path, "groups",
                                                         **{"detection.task": "detect"})))
    assert availability["kinematic"] == ""
    assert "task: pose" in availability["pose"], availability["pose"]
    assert availability["waveform"], "waveform is unavailable too"


def test_a_pose_project_offers_pose_and_waveform(tmp_path: Path) -> None:
    from trackanno.config import load_config
    from trackanno.measure.clipfeatures import available_groups

    path = _project(tmp_path, "pose", **{"detection.task": "pose",
                                         "keypoints.nodes": ["a", "b", "c"]})
    availability = available_groups(load_config(path))
    assert not availability["pose"] and not availability["waveform"]


def test_embedding_and_context_groups_follow_their_settings(tmp_path: Path) -> None:
    from trackanno.config import load_config
    from trackanno.measure.clipfeatures import available_groups

    path = _project(tmp_path, "followers", **{
        "features.enabled": False,
        "suggest.embeddings": [{"name": "det", "source": "detector"}]})
    availability = available_groups(load_config(path))
    assert availability["emb_det"], "a detector embedding follows features.enabled"
    assert availability["context"], "context follows suggest.context_frames"


@pytest.mark.parametrize("column", ["speed_mean", "area_mean", "body_len_mean",
                                    "wave_disp_amp_tip", "wave_amp_asym", "ctx_speed_mean",
                                    "conf_std"])
def test_every_feature_column_maps_to_a_group(column) -> None:
    from trackanno.measure.clipfeatures import FEATURE_GROUPS, group_of

    assert group_of(column) in FEATURE_GROUPS


def test_embedding_and_metadata_columns_map_as_expected() -> None:
    from trackanno.measure.clipfeatures import group_of

    assert group_of("emb_det_3_mean") == "emb_det"
    assert group_of("clip_id") is None


@pytest.mark.parametrize("groups, needle", [([], "no inputs at all"),
                                            (["nonsense"], "unknown entries")])
def test_a_bad_group_selection_is_refused(tmp_path: Path, groups, needle) -> None:
    from trackanno.config import ConfigError
    from trackanno.project import write_config

    path = _project(tmp_path, "bad-groups")
    with pytest.raises(ConfigError, match=needle):
        write_config(path, {"suggest.feature_groups": groups})
