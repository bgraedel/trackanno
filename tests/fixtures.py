"""Shared builders for the suites: synthetic movies, generated projects and stubs."""

from __future__ import annotations

import atexit
import csv
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from trackanno.video.movies import MovieSource as _MovieSource  # noqa: E402

GROUP = "behaviour"                       # the default project's single-choice group
CLASSES = ("swimming", "immotile")
LENGTH = 50                               # clips.frames, the dense window

# The planted change sits on neither grid, so no window boundary lands on it by luck.
CUT, PLANTED_FRAMES = 137, 300
BIG, SMALL = 20.0, 10.0                   # box side while moving, and once stopped

MOVIE_FRAMES, MOVIE_HEIGHT, MOVIE_WIDTH = 120, 512, 512
EMB_MOVIES, EMB_TRACKS, EMB_FRAMES, EMB_LENGTH, EMB_DIM = 3, 8, 200, 50, 12

_SHARED: list[Path] = []


def _shared_root() -> Path:
    """One temporary folder per run for the cached fixture projects."""
    if not _SHARED:
        path = Path(tempfile.mkdtemp(prefix="trackanno-fixtures-"))
        atexit.register(shutil.rmtree, path, True)
        _SHARED.append(path)
    return _SHARED[0]


def pump(app, predicate, timeout: float = 30.0) -> bool:
    """Spin the event loop until ``predicate`` holds or ``timeout`` passes."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


# --------------------------------------------------------------------------- #
# a synthetic movie
# --------------------------------------------------------------------------- #

PIPELINE_CONFIG = """
paths:  {video_dir: ./videos, model: ./model.pt, work_dir: ./work, csv: ./annotations.csv}
tiff:   {fps: 100}
detection: {task: detect}
clips:  {frames: 50, stride: 50, window_mode: all, follow: fixed,
         pad: 1.6, crop_size: 96, min_clip_frames: 40}
normalize: {mode: fixed}
tracking: {min_track_length: 50}
categories:
  - name: behaviour
    type: single
    required: true
    options:
      - {name: swimming, key: "1", color: "#4C9F70"}
      - {name: tumbling, key: "2", color: "#D95F02"}
  - name: quality
    type: multi
    required: false
    options:
      - {name: blurry, key: "q", color: "#999999"}
ui: {annotator: tester}
"""


class SyntheticMovie(_MovieSource):
    """Two bright squares drifting across a textured background."""

    def __init__(self):
        self.id = "synthetic"
        self.path = Path("synthetic.tif")
        self.series = 0
        self.channel = 0
        self.n_frames = MOVIE_FRAMES
        self.height, self.width = MOVIE_HEIGHT, MOVIE_WIDTH
        self.dtype = np.dtype(np.uint8)
        self.fps = 100.0
        self.pixel_size_um = 0.1

        rng = np.random.default_rng(0)
        base = rng.integers(20, 60, size=(MOVIE_HEIGHT, MOVIE_WIDTH), dtype=np.uint8)
        self._data = np.repeat(base[None], MOVIE_FRAMES, axis=0).copy()
        self.truth: list[dict] = []
        for track_id, (x0, y0, dx, dy, size) in enumerate(
            [(60, 80, 2.0, 0.5, 20), (300, 200, -1.0, 1.5, 16)], start=1
        ):
            for f in range(MOVIE_FRAMES):
                x = int(round(x0 + dx * f))
                y = int(round(y0 + dy * f))
                x = int(np.clip(x, 0, MOVIE_WIDTH - size))
                y = int(np.clip(y, 0, MOVIE_HEIGHT - size))
                self._data[f, y : y + size, x : x + size] = 200 + track_id * 20
                self.truth.append(
                    {"frame": f, "track_id": track_id,
                     "x1": float(x), "y1": float(y),
                     "x2": float(x + size), "y2": float(y + size), "conf": 0.9}
                )

    def frame(self, index: int) -> np.ndarray:
        return self._data[index]

    def tracks(self) -> pd.DataFrame:
        return pd.DataFrame(self.truth)


# --------------------------------------------------------------------------- #
# the cached GUI fixture project
# --------------------------------------------------------------------------- #

N_KEYPOINTS = 8            # Head, Base, Flag1..Flag5, Tip

_CACHE: dict = {}


def _node_names() -> list:
    return (["Head", "Base"]
            + [f"Flag{i}" for i in range(1, N_KEYPOINTS - 2)] + ["Tip"])


def _settings(pose: bool) -> dict:
    """Settings applied with the real ``write_config`` over a created project."""
    return {
        "tiff.fps": 100,
        "normalize.mode": "fixed",
        "tracking.min_track_length": 50,
        "detection.task": "pose" if pose else "detect",
        "clips.frames": 50,
        "clips.stride": 50,
        "clips.window_mode": "all",
        "clips.follow": "fixed",
        "clips.pad": 1.6,
        "clips.crop_size": 96,
        "clips.min_clip_frames": 40,
        "ui.annotator": "tester",
    }


def _add_keypoints(tracks):
    """A beating flagellum hung off every detection, in movie pixels."""
    s = np.linspace(0.0, 1.0, N_KEYPOINTS - 1)
    centre_x = (tracks["x1"] + tracks["x2"]) / 2.0
    centre_y = (tracks["y1"] + tracks["y2"]) / 2.0
    phase = 2 * np.pi * tracks["frame"].to_numpy() / 11.0
    tracks["kp_x_0"] = centre_x - 6.0                    # Head
    tracks["kp_y_0"] = centre_y
    tracks["kp_c_0"] = 0.9
    for k in range(N_KEYPOINTS - 1):                     # Base..Tip
        tracks[f"kp_x_{k + 1}"] = centre_x + s[k] * 22.0
        tracks[f"kp_y_{k + 1}"] = centre_y + 5.0 * np.sin(
            2 * np.pi * 1.4 * s[k] - phase)
        tracks[f"kp_c_{k + 1}"] = 0.9
    return tracks


def make_project(root: Path | None = None, pose: bool = False) -> Path:
    """A project with a real clip cache, built once per pose flag per run; returns its config."""
    key = bool(pose)
    cached = _CACHE.get(key)
    if cached is not None and cached.exists():
        return cached

    from trackanno.annotate.clips import (
        build_box_table, choose_crop_size, finalise_geometry, plan_clips,
        specs_to_frame, write_clip_cache,
    )
    from trackanno.config import load_config
    from trackanno.video.movies import Normalizer
    from trackanno.project import create_project, write_config

    if root is None:
        root = _shared_root()
    project = Path(root) / f"fixture{'_pose' if pose else ''}"
    config_path = create_project(project)
    names = _node_names()
    write_config(
        config_path, _settings(pose),
        keypoints={"nodes": names,
                   "edges": [[names[i], names[i + 1]] for i in range(1, len(names) - 1)]}
        if pose else None,
    )
    cfg = load_config(config_path)

    movie = SyntheticMovie()
    tracks = movie.tracks()
    if pose:
        tracks = _add_keypoints(tracks)

    specs = plan_clips(tracks, movie.id, movie.n_frames, cfg)
    size = choose_crop_size(specs, cfg)
    origins = finalise_geometry(specs, size, cfg.clips.frames)
    out = cfg.movie_dir(movie.id)
    out.mkdir(parents=True, exist_ok=True)
    build_box_table(
        specs, origins, tracks, cfg, N_KEYPOINTS if pose else 0
    ).to_parquet(out / "clip_boxes.parquet", index=False)
    write_clip_cache(movie, specs, origins, size, out, Normalizer(0.0, 255.0), cfg)

    # The source columns prepare_movie adds; the store writes them into every row.
    manifest = specs_to_frame(specs)
    manifest["source_file"] = str(movie.path)
    manifest["series"] = movie.series
    manifest["channel"] = movie.channel
    manifest.to_parquet(out / "clips.parquet", index=False)

    # The dense grid covers every track in tracks.parquet, not only clipped ones.
    tracks.to_parquet(out / "tracks.parquet", index=False)

    # project_status counts prepare_done.json.
    (out / "prepare_done.json").write_text(
        json.dumps({"n_clips": len(specs), "movie_id": movie.id}), encoding="utf-8")

    _CACHE[key] = config_path
    return config_path


# --------------------------------------------------------------------------- #
# the two prepare stages
# --------------------------------------------------------------------------- #

PREPARE_CONFIG = """
paths:  {video_dir: ./videos, model: ./model.pt, work_dir: ./work, csv: ./annotations.csv}
tiff:   {fps: 100}
detection: {task: detect, device: cpu}
features: {enabled: false}
clips:  {frames: 50, stride: 50, window_mode: first, follow: smooth, pad: 1.6,
         crop_size: 96, min_clip_frames: 40}
normalize: {mode: fixed}
tracking: {backend: greedy, min_track_length: 40}
categories:
  - name: behaviour
    type: single
    required: true
    options: [{name: swimming, key: "1", color: "#4C9F70"}]
ui: {annotator: tester}
"""


class StubDetector:
    """Replays a synthetic movie's ground truth, one ``detect`` call per frame."""

    def __init__(self, by_frame: dict, n_frames: int):
        self.by_frame, self.n_frames, self.calls = by_frame, n_frames, 0

    def is_tiled(self, height, width):
        return False

    def tiles_for(self, height, width):
        return [(0, 0, width, height)]

    def detect(self, frame_rgb):
        from trackanno.detect.tiling import Detections

        xyxy, conf = self.by_frame[self.calls % self.n_frames]
        self.calls += 1
        return Detections(
            xyxy=xyxy, conf=conf, cls=np.zeros(len(conf), np.int32), kpts=None
        )


def prep_project(root: Path, name: str, config: str = PREPARE_CONFIG):
    """A project folder, a synthetic movie and its ground truth, ready to run."""
    from trackanno.config import load_config

    project = root / name
    project.mkdir()
    (project / "config.yaml").write_text(config, encoding="utf-8")
    cfg = load_config(project / "config.yaml")
    movie = SyntheticMovie()
    by_frame = {
        int(f): (df[["x1", "y1", "x2", "y2"]].to_numpy(np.float32),
                 df["conf"].to_numpy(np.float32))
        for f, df in movie.tracks().groupby("frame")
    }
    return cfg, movie, by_frame


def no_detector():
    raise AssertionError("the clip stage built a detector, which needs the weights")


# --------------------------------------------------------------------------- #
# prediction cases: a project with a stub model
# --------------------------------------------------------------------------- #

class StubModel:
    """A deterministic classifier: the first feature column decides the class."""

    classes_ = np.array(CLASSES)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        swimming = np.nan_to_num(X[:, 0]) % 2.0 < 1.0
        p = np.where(swimming, 0.9, 0.25)
        return np.column_stack([p, 1.0 - p])


class AreaModel:
    """Exactly right on `planted_case`: inverts area_mean to the fraction moving."""

    classes_ = np.array(CLASSES)

    def predict_proba(self, X):
        area = np.asarray(X, dtype=float)[:, 1]        # area_mean, by position
        moving = np.clip(
            (area - SMALL ** 2) / (BIG ** 2 - SMALL ** 2), 0.001, 0.999
        )
        return np.column_stack([moving, 1.0 - moving])


def report_entry(group: str, reliable: bool) -> dict:
    """The GroupReport fields anything downstream reads."""
    return {
        "group": group,
        "n_labels": 40,
        "n_classes": 2,
        "classes": list(CLASSES),
        "macro_f1": 0.81 if reliable else 0.12,
        "accuracy": 0.84 if reliable else 0.5,
        "baseline_accuracy": 0.5,
        "kappa": 0.62 if reliable else 0.01,
        "kappa_lower": 0.31 if reliable else -0.20,
        "reliable": bool(reliable),
        "reason": "ok" if reliable else
                  "agreement is not distinguishable from chance (kappa 0.01)",
        "confusion": [[10, 2], [3, 9]],
        "classes_dropped": {},
    }


def build_case(
    root: Path,
    name: str,
    *,
    model: bool = True,
    report: bool = True,
    reliable: bool = True,
    groups: dict | None = None,
    labels: int = 0,
    model_class: type = StubModel,
    trained_window: int | None = None,
):
    """A project with clip features and a stub model per group; returns (cfg, manifest)."""
    import joblib

    from trackanno.measure.clipfeatures import CLIP_FEATURES_FILE, FEATURE_VERSION
    from trackanno.config import load_config
    from trackanno.project import write_config
    from trackanno.annotate.store import AnnotationStore
    from trackanno.learn.suggest import MODEL_FILE, REPORT_FILE

    fixture = make_project()
    project = Path(root) / name
    project.mkdir(parents=True, exist_ok=True)
    config_path = project / "config.yaml"
    shutil.copy(fixture, config_path)
    write_config(config_path, {
        "paths.work_dir": str(project / "work").replace("\\", "/"),
        "paths.csv": str(project / "annotations.csv").replace("\\", "/"),
    })
    cfg = load_config(config_path)
    work = cfg.paths.work_dir
    work.mkdir(parents=True, exist_ok=True)

    # Tables only: the dense pass reads tracks.parquet and cache.json, never pixels.
    manifests = []
    for movie_dir in sorted(p for p in load_config(fixture).paths.work_dir.iterdir()
                            if p.is_dir()):
        source = movie_dir / "clips.parquet"
        if not source.exists():
            continue
        (work / movie_dir.name).mkdir(parents=True, exist_ok=True)
        for name in ("clips.parquet", "tracks.parquet", "cache.json"):
            if (movie_dir / name).exists():
                shutil.copy(movie_dir / name, work / movie_dir.name / name)
        manifests.append(pd.read_parquet(source))
    manifest = pd.concat(manifests, ignore_index=True)

    features = pd.DataFrame({
        "clip_id": manifest["clip_id"].astype(str),
        "movie_id": manifest["movie_id"].astype(str),
        "track_id": manifest["track_id"].astype(int),
        "f_start": manifest["f_start"].astype(int),
        "speed_mean": np.arange(len(manifest), dtype=float),
        "area_mean": np.linspace(10.0, 20.0, len(manifest)),
        "_feature_version": FEATURE_VERSION,
    })
    features.to_parquet(work / CLIP_FEATURES_FILE, index=False)

    wanted = groups if groups is not None else {GROUP: reliable}
    if model:
        payload = {"columns": ["speed_mean", "area_mean"],
                   "models": {g: model_class() for g in wanted},
                   "feature_version": FEATURE_VERSION}
        if trained_window is not None:
            payload["trained_window"] = int(trained_window)
        joblib.dump(payload, work / MODEL_FILE)
    if report:
        (work / REPORT_FILE).write_text(
            json.dumps({g: report_entry(g, ok) for g, ok in wanted.items()}, indent=2),
            encoding="utf-8",
        )

    if labels:
        store = AnnotationStore(cfg.paths.csv, cfg)
        for i, row in enumerate(manifest.head(labels).to_dict("records")):
            store.record(row, {GROUP: CLASSES[i % len(CLASSES)]})
    return cfg, manifest


def planted_case(root: Path, name: str, stride: int, **settings):
    """A project whose one 300-frame track stops moving at frame `CUT`; returns (cfg, movie)."""
    import joblib

    from trackanno.config import load_config
    from trackanno.project import write_config
    from trackanno.learn.suggest import MODEL_FILE

    cfg, _ = build_case(root, name, model_class=AreaModel, trained_window=LENGTH)
    work = cfg.paths.work_dir
    movie = next(p for p in sorted(work.iterdir())
                 if p.is_dir() and (p / "tracks.parquet").exists())
    frames = np.arange(PLANTED_FRAMES)
    side = np.where(frames < CUT, BIG, SMALL)
    left = 20.0 + 0.5 * frames                      # drifting, so speed_* exist too
    pd.DataFrame({
        "frame": frames, "track_id": 1,
        "x1": left, "y1": 40.0, "x2": left + side, "y2": 40.0 + side,
        "conf": 0.9,
    }).to_parquet(movie / "tracks.parquet", index=False)

    payload = joblib.load(work / MODEL_FILE)
    payload["models"] = {g: AreaModel() for g in payload["models"]}
    joblib.dump(payload, work / MODEL_FILE)

    write_config(cfg.path, {
        "predict.window_stride": stride,
        "predict.min_bout_frames": 20,
        **settings,
    })
    return load_config(cfg.path), movie.name


# --------------------------------------------------------------------------- #
# embedding projects
# --------------------------------------------------------------------------- #

def synthetic_project(root: Path, name: str, settings: dict | None = None,
                      detector: bool = False, file_source: bool = False,
                      drop_every: int = 0):
    """Movies whose odd tracks swim and even ones sit, with optional embedding sources."""
    from trackanno.annotate.clips import clip_id_for
    from trackanno.config import load_config
    from trackanno.project import create_project, write_config

    path = create_project(Path(root) / name)
    write_config(path, {
        "clips.frames": EMB_LENGTH, "clips.stride": EMB_LENGTH, "clips.min_clip_frames": 20,
        "suggest.model": "hist",
        "suggest.audit_fraction": 0.0, "suggest.min_per_class": 2,
        **(settings or {}),
    })
    cfg = load_config(path)
    rng = np.random.default_rng(1)
    labels = []
    for m in range(EMB_MOVIES):
        movie_id = f"M{m}"
        out = cfg.movie_dir(movie_id)
        out.mkdir(parents=True, exist_ok=True)
        (out / "cache.json").write_text(json.dumps({"movie": {
            "n_frames": EMB_FRAMES, "fps": 100.0, "height": 512, "width": 512}}),
            encoding="utf-8")
        rows, manifest = [], []
        for t in range(1, EMB_TRACKS + 1):
            moving = t % 2 == 1
            frames = np.arange(EMB_FRAMES)
            x = 50.0 + (3.0 if moving else 0.05) * frames + rng.normal(0, 0.5, EMB_FRAMES)
            y = 40.0 * t + rng.normal(0, 0.5, EMB_FRAMES)
            rows.append(pd.DataFrame({
                "frame": frames, "track_id": t, "x1": x, "y1": y,
                "x2": x + 20.0, "y2": y + 20.0, "conf": 0.9}))
            for f in range(0, EMB_FRAMES, EMB_LENGTH):
                clip_id = clip_id_for(movie_id, t, f)
                manifest.append({"clip_id": clip_id, "movie_id": movie_id,
                                 "track_id": t, "f_start": f, "f_end": f + EMB_LENGTH})
                labels.append({"clip_id": clip_id,
                               "behaviour": "swimming" if moving else "immotile"})
        tracks = pd.concat(rows, ignore_index=True)
        tracks.to_parquet(out / "tracks.parquet", index=False)
        pd.DataFrame(manifest).to_parquet(out / "clips.parquet", index=False)
        signal = np.where(tracks["track_id"] % 2 == 1, 3.0, -3.0)[:, None]
        vectors = (rng.normal(size=(len(tracks), EMB_DIM)) + signal).astype(np.float32)
        if detector:
            np.save(out / "instance_features.npy", vectors.astype(np.float16))
        if file_source:
            table = pd.DataFrame(vectors, columns=[f"v{i}" for i in range(EMB_DIM)])
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


# --------------------------------------------------------------------------- #
# annotation CSV rows
# --------------------------------------------------------------------------- #

STORE_CONFIG = """
paths:  {video_dir: ./videos, model: ./model.pt, work_dir: ./work, csv: ./annotations.csv}
tiff:   {fps: 100}
categories:
  - name: behaviour
    type: single
    required: false
    options:
      - {name: swimming, key: "1", color: "#4C9F70"}
      - {name: tumbling, key: "2", color: "#D95F02"}
ui: {annotator: tester}
"""

# Comma, quote, backslash, newline and tab in one string.
AWKWARD = 'comma, "quoted" \\ backslash\nnewline\ttab'


def clip_meta(i: int) -> dict:
    return {
        "clip_id": f"MOVIE__t{i:04d}__f000000", "movie_id": "MOVIE",
        "source_file": "/movies/MOVIE.tif", "series": 0, "channel": 0,
        "track_id": i, "f_start": i * 50, "f_end": i * 50 + 49,
        "crop_x0": 10 + i, "crop_y0": 20 + i, "crop_w": 96, "crop_h": 96,
        "scale": 1.0,
    }


def old_schema(cfg) -> list[str]:
    """A header without the provenance and audit columns."""
    from trackanno.annotate import store as _store

    columns = _store.IDENTITY_COLUMNS + [g.name for g in cfg.categories] + _store.TRAILING_COLUMNS
    return [c for c in columns if c not in (_store.PROVENANCE_COLUMN, _store.AUDIT_COLUMN)]


def write_old_csv(path: Path, cfg, rows: list[dict]) -> None:
    """A CSV in the old schema, without the provenance and audit columns."""
    columns = old_schema(cfg)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def old_row(i: int, behaviour: str = "swimming", **extra) -> dict:
    row = dict(clip_meta(i))
    row.update({
        "behaviour": behaviour, "uncertain": "0", "notes": "", "annotator": "benny",
        "timestamp_utc": "2025-11-04T09:12:33+00:00", "dwell_s": "12.75",
        "suggested": "", "suggest_conf": "", "suggest_shown": "0",
    })
    row.update(extra)
    return row


# --------------------------------------------------------------------------- #
# bout review drawing
# --------------------------------------------------------------------------- #

def painted(widget) -> np.ndarray:
    """The widget's own rendered pixels as an RGB array."""
    from qtpy.QtGui import QPixmap

    pixmap = QPixmap(widget.size())
    widget.render(pixmap)
    image = pixmap.toImage()
    raw = image.constBits().asstring(image.sizeInBytes())
    return np.frombuffer(raw, np.uint8).reshape(
        image.height(), image.bytesPerLine() // 4, 4
    )[..., :3]


def loaded(app, panel, tries: int = 5000) -> None:
    """Spin the loop until the panel has read its predictions off-thread."""
    for _ in range(tries):
        app.processEvents()
        if panel.loaded:
            app.processEvents()
            return
        time.sleep(0.002)
    raise AssertionError("the predictions never finished loading")


def ink(frame: np.ndarray) -> int:
    """Pixels that are not the commonest colour in the image."""
    flat = frame.reshape(-1, 3)
    colours, counts = np.unique(flat, axis=0, return_counts=True)
    return int(len(flat) - counts.max())


def bout_row(f_start, f_end, label, conf, reason="",
             movie="m", track=1, group=GROUP, groups=(GROUP,)) -> dict:
    """One row of the bout schema, with every other group's columns left empty."""
    row = {
        "bout_id": f"{movie}__t{track:04d}__f{f_start:06d}-{f_end:06d}",
        "movie_id": movie, "track_id": track,
        "f_start": f_start, "f_end": f_end, "n_frames": f_end - f_start,
        "group": group, "skip_reason": reason,
    }
    for other in groups:
        row[f"{other}_pred"] = label if other == group else ""
        row[f"{other}_conf"] = conf if other == group else np.nan
    return row
