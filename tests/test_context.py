"""The ``context`` feature family over ``suggest.context_frames``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fixtures import EMB_LENGTH as LENGTH, synthetic_project

CONTEXT = {"suggest.context_frames": 120,
           "suggest.feature_groups": ["kinematic", "context"]}


@pytest.fixture(scope="module")
def built(root):
    from trackanno.config import load_config
    from trackanno.measure.clipfeatures import build_clip_features

    cfg = load_config(synthetic_project(root, "ctx", CONTEXT))
    features = build_clip_features(cfg, verbose=False)
    return cfg, features.set_index(["movie_id", "track_id", "f_start"])


def test_context_columns_are_the_same_for_clips_and_dense_windows(built) -> None:
    from trackanno.infer.densepredict import build_dense_features
    from trackanno.measure.clipfeatures import context_names

    cfg, clips = built
    columns = context_names()
    assert set(columns) <= set(clips.columns), sorted(clips.columns)

    dense = build_dense_features(cfg).set_index(["movie_id", "track_id", "f_start"])
    shared = clips.index.intersection(dense.index)
    assert len(shared) == len(clips), "the dense grid lands on the clip windows"
    assert np.allclose(dense.loc[shared, columns].to_numpy(float),
                       clips.loc[shared, columns].to_numpy(float), equal_nan=True)


def _track(cfg):
    import pandas as pd

    tdf = pd.read_parquet(cfg.movie_dir("M0") / "tracks.parquet")
    return tdf[tdf["track_id"] == 1].sort_values("frame")


def test_context_is_the_kinematics_over_the_centred_span(built) -> None:
    """Clip [50, 100) is centred on 75, so its context is [15, 135)."""
    from trackanno.measure.clipfeatures import window_features

    cfg, clips = built
    wide = window_features(_track(cfg), cfg, 100.0, [(15, 135)])[0]
    row = clips.loc[("M0", 1, 50)]
    assert np.isclose(row["ctx_speed_mean"], wide["speed_mean"])
    assert np.isclose(row["ctx_path_length"], wide["path_length"])
    assert not np.isclose(row["ctx_path_length"], row["path_length"]), "not the clip's own"


def test_context_is_clipped_at_the_track_start(built) -> None:
    from trackanno.measure.clipfeatures import window_features

    cfg, clips = built
    early = window_features(_track(cfg), cfg, 100.0, [(0, 120)])[0]
    assert np.isclose(clips.loc[("M0", 1, 0)]["ctx_path_length"], early["path_length"])


def test_a_model_refuses_another_context_length(root: Path) -> None:
    import joblib

    from trackanno.config import load_config
    from trackanno.infer.predict import predict_all
    from trackanno.learn.suggest import MODEL_FILE, Suggester, train
    from trackanno.project import write_config

    cfg = load_config(synthetic_project(root, "ctxmodel", CONTEXT))
    train(cfg, verbose=False)
    payload = joblib.load(cfg.paths.work_dir / MODEL_FILE)
    assert payload["context_frames"] == 120, "the model records its context length"

    write_config(cfg.path, {"suggest.context_frames": 60})
    cfg = load_config(cfg.path)
    with pytest.raises(RuntimeError, match="context_frames"):
        Suggester(cfg).refuse_context_mismatch(cfg.suggest.context_frames)
    with pytest.raises(RuntimeError, match="context_frames"):
        predict_all(cfg)

    write_config(cfg.path, {"suggest.context_frames": 120})
    Suggester(load_config(cfg.path)).refuse_context_mismatch(120)      # the same length passes
    assert LENGTH == cfg.clips.frames
