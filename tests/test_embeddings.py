"""Per-detection embeddings as a pluggable feature input.

A synthetic project of tracks, a clip manifest and per-movie embedding sources,
built here without pixels: clip features, training and the dense table read
nothing else.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from harness import check

N_MOVIES, N_TRACKS, N_FRAMES, LENGTH, DIM = 3, 8, 200, 50, 12


def synthetic_project(root: Path, name: str, settings: dict | None = None,
                      detector: bool = False, file_source: bool = False,
                      drop_every: int = 0):
    """A project of N_MOVIES movies whose odd tracks swim and even ones sit.

    ``detector`` writes ``instance_features.npy``; ``file_source`` writes
    ``emb/<movie>.parquet`` with every ``drop_every``-th detection left out.
    Returns ``(config_path, labels)``.
    """
    from trackanno.annotate.clips import clip_id_for
    from trackanno.config import load_config
    from trackanno.project import create_project, write_config

    path = create_project(Path(root) / name)
    write_config(path, {
        "clips.frames": LENGTH, "clips.stride": LENGTH, "clips.min_clip_frames": 20,
        "suggest.model": "hist", "suggest.iterations": 20,
        "suggest.audit_fraction": 0.0, "suggest.min_per_class": 2,
        "suggest.pca_components": 4,
        **(settings or {}),
    })
    cfg = load_config(path)
    rng = np.random.default_rng(1)
    labels = []
    for m in range(N_MOVIES):
        movie_id = f"M{m}"
        out = cfg.movie_dir(movie_id)
        out.mkdir(parents=True, exist_ok=True)
        (out / "cache.json").write_text(json.dumps({"movie": {
            "n_frames": N_FRAMES, "fps": 100.0, "height": 512, "width": 512}}),
            encoding="utf-8")
        rows, manifest = [], []
        for t in range(1, N_TRACKS + 1):
            moving = t % 2 == 1
            frames = np.arange(N_FRAMES)
            x = 50.0 + (3.0 if moving else 0.05) * frames + rng.normal(0, 0.5, N_FRAMES)
            y = 40.0 * t + rng.normal(0, 0.5, N_FRAMES)
            rows.append(pd.DataFrame({
                "frame": frames, "track_id": t, "x1": x, "y1": y,
                "x2": x + 20.0, "y2": y + 20.0, "conf": 0.9}))
            for f in range(0, N_FRAMES, LENGTH):
                clip_id = clip_id_for(movie_id, t, f)
                manifest.append({"clip_id": clip_id, "movie_id": movie_id,
                                 "track_id": t, "f_start": f, "f_end": f + LENGTH})
                labels.append({"clip_id": clip_id,
                               "behaviour": "swimming" if moving else "immotile"})
        tracks = pd.concat(rows, ignore_index=True)
        tracks.to_parquet(out / "tracks.parquet", index=False)
        pd.DataFrame(manifest).to_parquet(out / "clips.parquet", index=False)
        signal = np.where(tracks["track_id"] % 2 == 1, 3.0, -3.0)[:, None]
        vectors = (rng.normal(size=(len(tracks), DIM)) + signal).astype(np.float32)
        if detector:
            np.save(out / "instance_features.npy", vectors.astype(np.float16))
        if file_source:
            table = pd.DataFrame(vectors, columns=[f"v{i}" for i in range(DIM)])
            table.insert(0, "frame", tracks["frame"])
            table.insert(0, "track_id", tracks["track_id"])
            if drop_every:
                table = table.iloc[[i for i in range(len(table)) if i % drop_every]]
            (cfg.path.parent / "emb").mkdir(exist_ok=True)
            table.sample(frac=1.0, random_state=0).to_parquet(
                cfg.path.parent / "emb" / f"{movie_id}.parquet", index=False)
    pd.DataFrame(labels).to_csv(cfg.paths.csv, index=False)
    return path


FILE_SETTINGS = {
    "suggest.embeddings": [{"name": "ae", "source": "file",
                            "path": "emb/{movie_id}.parquet"}],
    "suggest.feature_groups": ["kinematic", "emb_ae"],
}


def test_a_file_embedding_goes_from_table_to_model_and_dense_grid(root: Path) -> None:
    from trackanno.config import load_config
    from trackanno.infer.densepredict import build_dense_features
    from trackanno.learn import suggest
    from trackanno.measure import embeddings as emb
    from trackanno.measure.clipfeatures import build_clip_features

    cfg = load_config(synthetic_project(root, "file", FILE_SETTINGS,
                                        file_source=True, drop_every=7))
    features = build_clip_features(cfg, verbose=False)
    columns = [c for c in features.columns if c.startswith("emb_ae_")]
    check("three columns per component", len(columns) == 3 * 4, str(columns))
    check("a few missing vectors leave the summary defined",
          features[columns].notna().all().all())
    check("the basis is saved for the dense grid",
          set(emb.load_bases(cfg.paths.work_dir / emb.BASES_FILE)) == {"ae"})

    # Every fold sees exactly the columns the saved model is fitted on.
    seen = []
    original = suggest._grouped_cv_predictions

    def spy(X, *args, **kwargs):
        seen.append(X.shape[1])
        return original(X, *args, **kwargs)

    suggest._grouped_cv_predictions = spy
    try:
        suggest.train(cfg, rebuild_features=False, verbose=False)
    finally:
        suggest._grouped_cv_predictions = original
    import joblib

    payload = joblib.load(cfg.paths.work_dir / suggest.MODEL_FILE)
    check("the model is fitted on the embedding columns",
          set(columns) <= set(payload["columns"]), str(payload["columns"]))
    check("cross-validation used the saved model's columns",
          seen == [len(payload["columns"])], f"{seen} vs {len(payload['columns'])}")

    dense = build_dense_features(cfg).set_index(["movie_id", "track_id", "f_start"])
    clips = features.set_index(["movie_id", "track_id", "f_start"])
    shared = clips.index.intersection(dense.index)
    check("the dense grid lands on the clip windows", len(shared) == len(clips))
    check("...and summarises them identically",
          np.allclose(dense.loc[shared, columns].to_numpy(float),
                      clips.loc[shared, columns].to_numpy(float), equal_nan=True))


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
    n_comp = min(cfg.suggest.pca_components, *stacked.shape)
    check("the old path's column count", len(columns) == 3 * n_comp,
          f"{len(columns)} vs {3 * n_comp}")

    pca = PCA(n_components=n_comp, random_state=cfg.suggest.seed).fit(stacked)
    tracks = pd.read_parquet(cfg.movie_dir("M0") / "tracks.parquet")
    feats = np.load(cfg.movie_dir("M0") / "instance_features.npy")
    sel = (tracks["track_id"] == 3) & (tracks["frame"] >= 50) & (tracks["frame"] < 100)
    codes = pca.transform(feats[sel.to_numpy()].astype(np.float32))
    row = features[(features["movie_id"] == "M0") & (features["track_id"] == 3)
                   & (features["f_start"] == 50)].iloc[0]
    delta = grid_delta(codes, tracks.loc[sel, "frame"].to_numpy() - 50, LENGTH)
    check("the same values the cnn block held",
          np.allclose([row[f"emb_det_{i}_mean"] for i in range(n_comp)],
                      codes.mean(axis=0), atol=1e-4)
          and np.allclose([row[f"emb_det_{i}_std"] for i in range(n_comp)],
                          codes.std(axis=0), atol=1e-4)
          and np.allclose([row[f"emb_det_{i}_delta"] for i in range(n_comp)],
                          delta, atol=1e-4))


def test_an_old_cnn_config_and_model_are_migrated_or_refused(root: Path) -> None:
    import joblib

    from trackanno.config import load_config
    from trackanno.learn.suggest import MODEL_FILE, Suggester
    from trackanno.measure.clipfeatures import CLIP_FEATURES_FILE, FEATURE_VERSION

    path = synthetic_project(root, "old", {
        "suggest.feature_groups": ["kinematic", "pose", "waveform", "cnn"]})
    with pytest.warns(UserWarning, match="emb_det"):
        cfg = load_config(path)
    check("cnn reads as the detector embedding 'det'",
          [(e.name, e.source) for e in cfg.suggest.embeddings] == [("det", "detector")])
    check("...and its group", cfg.suggest.feature_groups
          == ["kinematic", "pose", "waveform", "emb_det"], str(cfg.suggest.feature_groups))

    work = cfg.paths.work_dir
    pd.DataFrame({"clip_id": ["a"], "speed_mean": [1.0], "cnn0_mean": [0.5],
                  "_feature_version": [FEATURE_VERSION]}).to_parquet(
        work / CLIP_FEATURES_FILE, index=False)
    joblib.dump({"columns": ["speed_mean", "cnn0_mean"], "models": {"behaviour": None},
                 "feature_version": FEATURE_VERSION}, work / MODEL_FILE)
    suggester = Suggester(cfg)
    check("a model fitted on cnn columns is refused", not suggester.ready)
    check("...with a retrain message", "retrain" in suggester.problem.lower(),
          suggester.problem)


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
    check("the model carries its basis", set(suggester.embedding_bases) == {"ae"},
          suggester.problem)
    columns = [c for c in suggester.columns if c.startswith("emb_ae_")]
    check("...for columns it actually uses", len(columns) > 0)
    before, _, _ = _rows_to_score(cfg, suggester, NullReporter())

    other = PCA(n_components=4).fit(np.random.default_rng(9).normal(size=(50, 12)))
    emb.save_bases(cfg.paths.work_dir / emb.BASES_FILE, {"ae": other})
    after, _, _ = _rows_to_score(cfg, Suggester(cfg), NullReporter())
    check("an overwritten basis on disk changes nothing",
          np.allclose(before[columns].to_numpy(float), after[columns].to_numpy(float),
                      equal_nan=True))

    # The model's own source spec is used, whatever the config now declares.
    from trackanno.project import write_config

    write_config(cfg.path, {"suggest.embeddings": [],
                            "suggest.feature_groups": ["kinematic"]})
    cfg = load_config(cfg.path)
    undeclared, _, _ = _rows_to_score(cfg, Suggester(cfg), NullReporter())
    check("an undeclared embedding is still read from the model's source",
          np.allclose(before[columns].to_numpy(float),
                      undeclared[columns].to_numpy(float), equal_nan=True))
    moved = cfg.path.parent / "emb" / "M0.parquet"
    moved.rename(moved.with_suffix(".bak"))
    with pytest.raises(RuntimeError, match="M0"):
        _rows_to_score(cfg, Suggester(cfg), NullReporter())
    moved.with_suffix(".bak").rename(moved)

    payload = joblib.load(cfg.paths.work_dir / MODEL_FILE)
    del payload["embedding_bases"]
    joblib.dump(payload, cfg.paths.work_dir / MODEL_FILE)
    stale = Suggester(cfg)
    check("a model with emb_* columns and no basis is refused",
          not stale.ready and "retrain" in stale.problem.lower(), stale.problem)


def test_a_rebuilt_table_the_model_was_not_fitted_on_is_refused(root: Path) -> None:
    import hashlib

    from trackanno.config import load_config
    from trackanno.learn.suggest import Suggester, ablate, train
    from trackanno.measure import embeddings as emb
    from trackanno.measure.clipfeatures import CLIP_FEATURES_FILE, build_clip_features
    from trackanno.project import write_config

    cfg = load_config(synthetic_project(root, "stamps", FILE_SETTINGS, file_source=True))
    train(cfg, verbose=False)
    check("a fresh model loads", Suggester(cfg).ready, Suggester(cfg).problem)

    work = cfg.paths.work_dir
    saved = {n: hashlib.sha256((work / n).read_bytes()).hexdigest()
             for n in (CLIP_FEATURES_FILE, emb.BASES_FILE)}
    ablate(cfg)
    check("ablate leaves the model's files alone",
          all(hashlib.sha256((work / n).read_bytes()).hexdigest() == h
              for n, h in saved.items()))

    build_clip_features(cfg, verbose=False)
    check("the same settings rebuild the same basis", Suggester(cfg).ready,
          Suggester(cfg).problem)

    write_config(cfg.path, {"suggest.pca_components": 3})
    build_clip_features(load_config(cfg.path), verbose=False)
    stale = Suggester(load_config(cfg.path))
    check("another basis in the table is refused",
          not stale.ready and "basis" in stale.problem and "Retrain" in stale.problem,
          stale.problem)

    write_config(cfg.path, {"suggest.pca_components": 4, "suggest.context_frames": 80})
    with pytest.warns(UserWarning, match="context"):
        changed = load_config(cfg.path)
    build_clip_features(changed, verbose=False)
    stale = Suggester(changed)
    check("another context_frames in the table is refused",
          not stale.ready and "context_frames" in stale.problem, stale.problem)


def test_a_file_source_is_checked_against_the_detections(root: Path) -> None:
    from trackanno.config import load_config
    from trackanno.measure.clipfeatures import build_clip_features

    cfg = load_config(synthetic_project(root, "checks", FILE_SETTINGS, file_source=True))
    path = cfg.path.parent / "emb" / "M0.parquet"
    good = pd.read_parquet(path)

    good.reset_index().rename(columns={"index": "Unnamed: 0"}).to_parquet(path)
    features = build_clip_features(cfg, verbose=False)
    check("a saved index column is not part of the vector",
          len([c for c in features.columns if c.startswith("emb_ae_")]) == 12)

    good.assign(frame=good["frame"] + 100_000).to_parquet(path, index=False)
    with pytest.raises(RuntimeError, match="none of its"):
        build_clip_features(cfg, verbose=False)

    good.assign(frame=good["frame"].where(good.index != good.index[0])).to_parquet(
        path, index=False)
    with pytest.raises(RuntimeError, match="empty"):
        build_clip_features(cfg, verbose=False)


def test_unused_context_and_old_configs_are_mentioned(root: Path) -> None:
    from trackanno.config import load_config
    from trackanno.project import create_project

    path = create_project(Path(root) / "warn")
    fresh = path.read_text(encoding="utf-8")
    path.write_text(fresh.replace("context_frames: 0", "context_frames: 50"),
                    encoding="utf-8")
    with pytest.warns(UserWarning, match="context"):
        load_config(path)

    old = "\n".join(line for line in fresh.splitlines()
                    if not line.startswith(("  feature_groups:", "  embeddings:")))
    path.write_text(old, encoding="utf-8")
    with pytest.warns(UserWarning, match="opt-in"):
        load_config(path)
