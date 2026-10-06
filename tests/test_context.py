"""The ``context`` feature family over ``suggest.context_frames``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from harness import check
from test_embeddings import LENGTH, synthetic_project

CONTEXT = {"suggest.context_frames": 120,
           "suggest.feature_groups": ["kinematic", "context"]}


def test_context_columns_are_the_same_for_clips_and_dense_windows(root: Path) -> None:
    import pandas as pd

    from trackanno.config import load_config
    from trackanno.infer.densepredict import build_dense_features
    from trackanno.measure.clipfeatures import (
        build_clip_features, context_names, window_features,
    )

    cfg = load_config(synthetic_project(root, "ctx", CONTEXT))
    features = build_clip_features(cfg, verbose=False)
    columns = context_names()
    check("every context column is present",
          set(columns) <= set(features.columns), str(sorted(features.columns)))

    dense = build_dense_features(cfg).set_index(["movie_id", "track_id", "f_start"])
    clips = features.set_index(["movie_id", "track_id", "f_start"])
    shared = clips.index.intersection(dense.index)
    check("the dense grid lands on the clip windows", len(shared) == len(clips))
    check("...and its context columns are identical",
          np.allclose(dense.loc[shared, columns].to_numpy(float),
                      clips.loc[shared, columns].to_numpy(float), equal_nan=True))

    # Clip [50, 100) is centred on 75, so its context is [15, 135).
    tdf = pd.read_parquet(cfg.movie_dir("M0") / "tracks.parquet")
    tdf = tdf[tdf["track_id"] == 1].sort_values("frame")
    wide = window_features(tdf, cfg, 100.0, [(15, 135)])[0]
    row = clips.loc[("M0", 1, 50)]
    check("context is the kinematics over the centred span",
          np.isclose(row["ctx_speed_mean"], wide["speed_mean"])
          and np.isclose(row["ctx_path_length"], wide["path_length"]),
          f"{row['ctx_speed_mean']} vs {wide['speed_mean']}")
    check("...which is not the clip's own",
          not np.isclose(row["ctx_path_length"], row["path_length"]))
    # At the track's start the span is clipped, not shifted off the track.
    first = clips.loc[("M0", 1, 0)]
    early = window_features(tdf, cfg, 100.0, [(0, 120)])[0]
    check("clipped to the track's extent",
          np.isclose(first["ctx_path_length"], early["path_length"]))


def test_a_model_refuses_another_context_length(root: Path) -> None:
    import joblib

    from trackanno.config import load_config
    from trackanno.infer.predict import predict_all
    from trackanno.learn.suggest import MODEL_FILE, Suggester, train
    from trackanno.project import write_config

    cfg = load_config(synthetic_project(root, "ctxmodel", CONTEXT))
    train(cfg, verbose=False)
    payload = joblib.load(cfg.paths.work_dir / MODEL_FILE)
    check("the model records its context length", payload["context_frames"] == 120)

    write_config(cfg.path, {"suggest.context_frames": 60})
    cfg = load_config(cfg.path)
    with pytest.raises(RuntimeError, match="context_frames"):
        Suggester(cfg).refuse_context_mismatch(cfg.suggest.context_frames)
    with pytest.raises(RuntimeError, match="context_frames"):
        predict_all(cfg)

    write_config(cfg.path, {"suggest.context_frames": 120})
    Suggester(load_config(cfg.path)).refuse_context_mismatch(120)
    check("the same length passes", True)
    check("the window length is still the clip length", LENGTH == cfg.clips.frames)
