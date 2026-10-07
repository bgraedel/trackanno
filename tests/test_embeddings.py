"""Per-detection embeddings as a pluggable feature input.

A synthetic project of tracks, a clip manifest and per-movie embedding sources,
built here without pixels: clip features, training and the dense table read
nothing else.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fixtures import (
    EMB_LENGTH as LENGTH,
    EMB_MOVIES as N_MOVIES, FILE_SETTINGS, synthetic_project,
)

PCA_WIDTH = 4


@pytest.fixture(autouse=True)
def _small_fits(monkeypatch):
    """A narrow PCA and a short fit keep these tests fast."""
    monkeypatch.setattr("trackanno.measure.embeddings.PCA_COMPONENTS", PCA_WIDTH)
    monkeypatch.setattr("trackanno.learn.suggest._iterations", lambda cfg: 20)


@pytest.fixture(scope="module")
def file_features(root):
    """A file-embedding project with a few vectors missing, and its clip features."""
    from trackanno.config import load_config
    from trackanno.measure.clipfeatures import build_clip_features

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("trackanno.measure.embeddings.PCA_COMPONENTS", PCA_WIDTH)
        cfg = load_config(synthetic_project(root, "file", FILE_SETTINGS,
                                            file_source=True, drop_every=7))
        features = build_clip_features(cfg, verbose=False)
    columns = [c for c in features.columns if c.startswith("emb_ae_")]
    return cfg, features, columns


def test_a_file_embedding_becomes_summary_columns(file_features) -> None:
    from trackanno.measure import embeddings as emb

    cfg, features, columns = file_features
    assert len(columns) == 3 * 4, f"three columns per component: {columns}"
    assert features[columns].notna().all().all(), "a few missing vectors leave it defined"
    assert set(emb.load_bases(cfg.paths.work_dir / emb.BASES_FILE)) == {"ae"}, (
        "the basis is saved for the dense grid")


def test_cross_validation_sees_the_saved_models_columns(file_features, monkeypatch) -> None:
    import joblib

    from trackanno.learn import suggest

    cfg, _, columns = file_features
    seen = []
    original = suggest._grouped_cv_predictions

    def spy(X, *args, **kwargs):
        seen.append(X.shape[1])
        return original(X, *args, **kwargs)

    monkeypatch.setattr(suggest, "_grouped_cv_predictions", spy)
    suggest.train(cfg, rebuild_features=False, verbose=False)
    payload = joblib.load(cfg.paths.work_dir / suggest.MODEL_FILE)
    assert set(columns) <= set(payload["columns"]), payload["columns"]
    assert seen == [len(payload["columns"])], f"{seen} vs {len(payload['columns'])}"


def test_the_dense_grid_summarises_embeddings_like_the_clips(file_features) -> None:
    from trackanno.infer.densepredict import build_dense_features

    cfg, features, columns = file_features
    dense = build_dense_features(cfg).set_index(["movie_id", "track_id", "f_start"])
    clips = features.set_index(["movie_id", "track_id", "f_start"])
    shared = clips.index.intersection(dense.index)
    assert len(shared) == len(clips), "the dense grid lands on the clip windows"
    assert np.allclose(dense.loc[shared, columns].to_numpy(float),
                       clips.loc[shared, columns].to_numpy(float), equal_nan=True)


def test_a_movie_without_its_embedding_file_is_refused(root: Path) -> None:
    from trackanno.config import load_config
    from trackanno.measure.clipfeatures import build_clip_features

    cfg = load_config(synthetic_project(root, "nofile", FILE_SETTINGS, file_source=True))
    (cfg.path.parent / "emb" / "M1.parquet").unlink()
    with pytest.raises(RuntimeError, match="M1"):
        build_clip_features(cfg, verbose=False)


def test_the_detector_source_matches_the_old_cnn_block(root: Path) -> None:
    """Same PCA over every detection, same mean/std/delta, same column count."""
    from sklearn.decomposition import PCA

    from trackanno.config import load_config
    from trackanno.measure.clipfeatures import build_clip_features, grid_delta

    settings = {"suggest.embeddings": [{"name": "det", "source": "detector"}],
                "suggest.feature_groups": ["kinematic", "emb_det"]}
    cfg = load_config(synthetic_project(root, "det", settings, detector=True))
    features = build_clip_features(cfg, verbose=False)
    columns = [c for c in features.columns if c.startswith("emb_det_")]
    stacked = np.concatenate([
        np.load(cfg.movie_dir(f"M{m}") / "instance_features.npy")
        for m in range(N_MOVIES)]).astype(np.float32)
    n_comp = min(PCA_WIDTH, *stacked.shape)
    assert len(columns) == 3 * n_comp

    pca = PCA(n_components=n_comp, random_state=0).fit(stacked)
    tracks = pd.read_parquet(cfg.movie_dir("M0") / "tracks.parquet")
    feats = np.load(cfg.movie_dir("M0") / "instance_features.npy")
    sel = (tracks["track_id"] == 3) & (tracks["frame"] >= 50) & (tracks["frame"] < 100)
    codes = pca.transform(feats[sel.to_numpy()].astype(np.float32))
    row = features[(features["movie_id"] == "M0") & (features["track_id"] == 3)
                   & (features["f_start"] == 50)].iloc[0]
    delta = grid_delta(codes, tracks.loc[sel, "frame"].to_numpy() - 50, LENGTH)
    for stat, want in (("mean", codes.mean(axis=0)), ("std", codes.std(axis=0)),
                       ("delta", delta)):
        got = [row[f"emb_det_{i}_{stat}"] for i in range(n_comp)]
        assert np.allclose(got, want, atol=1e-4), stat


def test_an_old_cnn_config_and_model_are_migrated_or_refused(root: Path) -> None:
    import joblib

    from trackanno.config import load_config
    from trackanno.learn.suggest import MODEL_FILE, Suggester
    from trackanno.measure.clipfeatures import CLIP_FEATURES_FILE, FEATURE_VERSION

    path = synthetic_project(root, "old", {
        "suggest.feature_groups": ["kinematic", "pose", "waveform", "cnn"]})
    with pytest.warns(UserWarning, match="emb_det"):
        cfg = load_config(path)
    assert [(e.name, e.source) for e in cfg.suggest.embeddings] == [("det", "detector")]
    assert cfg.suggest.feature_groups == ["kinematic", "pose", "waveform", "emb_det"]

    work = cfg.paths.work_dir
    pd.DataFrame({"clip_id": ["a"], "speed_mean": [1.0], "cnn0_mean": [0.5],
                  "_feature_version": [FEATURE_VERSION]}).to_parquet(
        work / CLIP_FEATURES_FILE, index=False)
    joblib.dump({"columns": ["speed_mean", "cnn0_mean"], "models": {"behaviour": None},
                 "feature_version": FEATURE_VERSION}, work / MODEL_FILE)
    suggester = Suggester(cfg)
    assert not suggester.ready, "a model fitted on cnn columns is refused"
    assert "retrain" in suggester.problem.lower(), suggester.problem


def test_prediction_reads_embeddings_in_the_models_own_basis(root: Path) -> None:
    """A refitted basis on disk must not change what the model is handed."""
    import joblib
    from sklearn.decomposition import PCA

    from trackanno.config import load_config
    from trackanno.infer.predict import _rows_to_score
    from trackanno.learn.suggest import MODEL_FILE, Suggester, train
    from trackanno.measure import embeddings as emb
    from trackanno.progress import NullReporter

    cfg = load_config(synthetic_project(root, "basis", FILE_SETTINGS, file_source=True))
    train(cfg, verbose=False)
    suggester = Suggester(cfg)
    assert set(suggester.embedding_bases) == {"ae"}, suggester.problem
    columns = [c for c in suggester.columns if c.startswith("emb_ae_")]
    assert columns
    before, _, _ = _rows_to_score(cfg, suggester, NullReporter())

    other = PCA(n_components=4).fit(np.random.default_rng(9).normal(size=(50, 12)))
    emb.save_bases(cfg.paths.work_dir / emb.BASES_FILE, {"ae": other})
    after, _, _ = _rows_to_score(cfg, Suggester(cfg), NullReporter())
    assert np.allclose(before[columns].to_numpy(float), after[columns].to_numpy(float),
                       equal_nan=True), "an overwritten basis on disk changes nothing"

    # The model's own source spec is used, whatever the config declares.
    from trackanno.project import write_config

    write_config(cfg.path, {"suggest.embeddings": [],
                            "suggest.feature_groups": ["kinematic"]})
    cfg = load_config(cfg.path)
    undeclared, _, _ = _rows_to_score(cfg, Suggester(cfg), NullReporter())
    assert np.allclose(before[columns].to_numpy(float),
                       undeclared[columns].to_numpy(float), equal_nan=True), (
        "an undeclared embedding is still read from the model's source")
    moved = cfg.path.parent / "emb" / "M0.parquet"
    moved.rename(moved.with_suffix(".bak"))
    with pytest.raises(RuntimeError, match="M0"):
        _rows_to_score(cfg, Suggester(cfg), NullReporter())
    moved.with_suffix(".bak").rename(moved)

    payload = joblib.load(cfg.paths.work_dir / MODEL_FILE)
    del payload["embedding_bases"]
    joblib.dump(payload, cfg.paths.work_dir / MODEL_FILE)
    stale = Suggester(cfg)
    assert not stale.ready, "emb_* columns and no basis"
    assert "retrain" in stale.problem.lower(), stale.problem


def test_a_rebuilt_table_the_model_was_not_fitted_on_is_refused(root: Path, monkeypatch) -> None:
    import hashlib

    from trackanno.config import load_config
    from trackanno.learn.suggest import Suggester, ablate, train
    from trackanno.measure import embeddings as emb
    from trackanno.measure.clipfeatures import CLIP_FEATURES_FILE, build_clip_features
    from trackanno.project import write_config

    cfg = load_config(synthetic_project(root, "stamps", FILE_SETTINGS, file_source=True))
    train(cfg, verbose=False)
    assert Suggester(cfg).ready, Suggester(cfg).problem

    work = cfg.paths.work_dir
    saved = {n: hashlib.sha256((work / n).read_bytes()).hexdigest()
             for n in (CLIP_FEATURES_FILE, emb.BASES_FILE)}
    ablate(cfg)
    assert all(hashlib.sha256((work / n).read_bytes()).hexdigest() == h
               for n, h in saved.items()), "ablate leaves the model's files alone"

    build_clip_features(cfg, verbose=False)
    assert Suggester(cfg).ready, f"the same settings rebuild the same basis: {Suggester(cfg).problem}"

    monkeypatch.setattr(emb, "PCA_COMPONENTS", 3)
    build_clip_features(load_config(cfg.path), verbose=False)
    stale = Suggester(load_config(cfg.path))
    assert not stale.ready, "another basis in the table"
    assert "basis" in stale.problem and "Retrain" in stale.problem, stale.problem

    monkeypatch.setattr(emb, "PCA_COMPONENTS", PCA_WIDTH)
    write_config(cfg.path, {"suggest.context_frames": 80})
    with pytest.warns(UserWarning, match="context"):
        changed = load_config(cfg.path)
    build_clip_features(changed, verbose=False)
    stale = Suggester(changed)
    assert not stale.ready and "context_frames" in stale.problem, stale.problem


def test_a_saved_index_column_is_not_part_of_the_vector(root: Path) -> None:
    from trackanno.config import load_config
    from trackanno.measure.clipfeatures import build_clip_features

    cfg = load_config(synthetic_project(root, "index", FILE_SETTINGS, file_source=True))
    path = cfg.path.parent / "emb" / "M0.parquet"
    pd.read_parquet(path).reset_index().rename(columns={"index": "Unnamed: 0"}).to_parquet(path)
    features = build_clip_features(cfg, verbose=False)
    assert len([c for c in features.columns if c.startswith("emb_ae_")]) == 12


@pytest.mark.parametrize("damage,match", [
    (lambda t: t.assign(frame=t["frame"] + 100_000), "none of its"),
    (lambda t: t.assign(frame=t["frame"].where(t.index != t.index[0])), "empty"),
], ids=["no-matching-detection", "missing-frame"])
def test_a_file_source_is_checked_against_the_detections(root: Path, damage, match) -> None:
    from trackanno.config import load_config
    from trackanno.measure.clipfeatures import build_clip_features

    cfg = load_config(synthetic_project(root, f"checks-{match[:4]}", FILE_SETTINGS,
                                        file_source=True))
    path = cfg.path.parent / "emb" / "M0.parquet"
    damage(pd.read_parquet(path)).to_parquet(path, index=False)
    with pytest.raises(RuntimeError, match=match):
        build_clip_features(cfg, verbose=False)


def test_unused_context_is_mentioned(root: Path) -> None:
    from trackanno.config import load_config
    from trackanno.project import create_project

    path = create_project(Path(root) / "warn-context")
    fresh = path.read_text(encoding="utf-8")
    path.write_text(fresh.replace("context_frames: 0", "context_frames: 50"), encoding="utf-8")
    with pytest.warns(UserWarning, match="context"):
        load_config(path)


def test_a_config_without_feature_groups_mentions_the_opt_in(root: Path) -> None:
    from trackanno.config import load_config
    from trackanno.project import create_project

    path = create_project(Path(root) / "warn-old")
    old = "\n".join(line for line in path.read_text(encoding="utf-8").splitlines()
                    if not line.startswith(("  feature_groups:", "  embeddings:")))
    path.write_text(old, encoding="utf-8")
    with pytest.warns(UserWarning, match="opt-in"):
        load_config(path)
