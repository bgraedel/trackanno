"""Clip features are rebuilt only when their inputs change, not per retrain."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from test_embeddings import FILE_SETTINGS, synthetic_project


@pytest.fixture
def builds(monkeypatch):
    """Count calls to the feature build that ``train`` makes."""
    from trackanno.learn import suggest

    calls: list[int] = []
    real = suggest.build_clip_features

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(suggest, "build_clip_features", counted)
    return calls


def test_train_reuses_features_until_an_input_changes(root: Path, builds) -> None:
    from trackanno.config import load_config
    from trackanno.learn.suggest import Suggester, train
    from trackanno.measure.clipfeatures import CLIP_FEATURES_KEY_FILE
    from trackanno.project import write_config

    path = synthetic_project(root, "cache", FILE_SETTINGS, file_source=True)
    cfg = load_config(path)
    work = cfg.paths.work_dir

    train(cfg, verbose=False)
    assert len(builds) == 1
    assert (work / CLIP_FEATURES_KEY_FILE).exists()

    # New labels alone do not touch the features.
    labels = pd.read_csv(cfg.paths.csv)
    labels.iloc[:-2].to_csv(cfg.paths.csv, index=False)
    train(cfg, verbose=False)
    assert len(builds) == 1
    assert Suggester(cfg).ready

    train(cfg, rebuild_features=True, verbose=False)
    assert len(builds) == 2

    # A rewritten tracks table is a new input.
    tracks = work / "M0" / "tracks.parquet"
    st = tracks.stat()
    os.utime(tracks, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000_000))
    train(cfg, verbose=False)
    assert len(builds) == 3

    # So is a feature-relevant config key.
    write_config(path, {"spectral.window": 60})
    train(load_config(path), verbose=False)
    assert len(builds) == 4
    train(load_config(path), verbose=False)
    assert len(builds) == 4
    assert Suggester(load_config(path)).ready


def test_a_cache_with_other_bases_on_disk_is_rebuilt(root: Path, builds) -> None:
    from trackanno.config import load_config
    from trackanno.learn.suggest import Suggester, train
    from trackanno.measure import embeddings as emb
    from trackanno.measure.clipfeatures import cached_clip_features

    cfg = load_config(synthetic_project(root, "bases", FILE_SETTINGS, file_source=True))
    train(cfg, verbose=False)
    assert cached_clip_features(cfg) is not None

    (cfg.paths.work_dir / emb.BASES_FILE).unlink()
    assert cached_clip_features(cfg) is None
    train(cfg, verbose=False)
    assert len(builds) == 2
    assert Suggester(cfg).ready


def test_cli_forces_a_rebuild_on_request(root: Path, builds, capsys) -> None:
    from trackanno.config import load_config
    from trackanno.learn.suggest import main

    path = synthetic_project(root, "cli", {})
    main(["--config", str(path), "--train"])
    main(["--config", str(path), "--train"])
    assert len(builds) == 1
    main(["--config", str(path), "--train", "--rebuild-features"])
    assert len(builds) == 2
    assert load_config(path).paths.work_dir.exists()
