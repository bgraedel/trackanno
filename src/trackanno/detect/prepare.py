"""Entry point 1: detection + tracking + clip cache.

    uv run prepare.py [--movies ID ...] [--force] [--limit-frames N] [--self-test]

Two stages, because only the first needs a GPU. Once per movie:

    work/<movie_id>/
      stage 1, `--stage detect`:
        norm.json          the movie's fixed intensity mapping
        tracks.parquet     every detection, in ORIGINAL full-frame coordinates
        instance_features.npy  (N, D) CNN features, ROW-ALIGNED with
                               tracks.parquet; absent unless every frame produced
                               them
        detect_done.json   the handoff: what stage 2 cannot work out for itself
      stage 2, `--stage clips`:
        clips.parquet      one row per annotatable clip + its crop geometry
        clip_boxes.parquet overlay geometry, in cache coordinates
        clips.dat          the pre-cropped pixels (uncompressed memmap)
        thumbs.dat         optional full-frame thumbnails for the GUI minimap
        cache.json         cache shape + movie metadata
        prepare_done.json  resume marker

Stage 2 needs no model, no GPU and no network, so detection can run on a remote
box and the clips - the multi-gigabyte artefact - be cut where they are read.
Copy ``work/<movie_id>/`` back between the two; the four stage-1 files are small.

Without ``--stage``, each movie gets whatever it is still missing, so a run
started elsewhere is finished rather than repeated. Finished movies are skipped
unless ``--force``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ..annotate.clips import (
    DETECT_FILE, DONE_FILE, FEATURES_FILE, MANIFEST_FILE, NORM_FILE, TRACKS_FILE,
    BOXES_FILE,
    StagedWrite, build_box_table, choose_crop_size, drop_border_tracks,
    finalise_geometry, plan_clips,
    specs_to_frame, write_clip_cache,
)
from ..config import Config, load_config
from ..video.movies import MovieSource, Normalizer, discover_movies
from ..progress import Cancelled, ConsoleReporter, NullReporter, Reporter
from .tiling import TiledDetector
from .tracking import build_tracker


# --------------------------------------------------------------------------- #
# tracking pass
# --------------------------------------------------------------------------- #
def run_tracking(
    movie: MovieSource,
    detector: TiledDetector,
    cfg: Config,
    normalizer: Normalizer,
    limit_frames: int | None = None,
    reporter: Reporter | None = None,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """One sequential pass: detect, then track on the merged full-frame boxes.

    Returns ``(tracks, instance_features)``; features are None unless
    ``features.enabled`` and every frame produced them."""
    reporter = reporter or NullReporter()
    tracker = build_tracker(cfg, movie.fps, (movie.height, movie.width))
    n_frames = min(movie.n_frames, limit_frames) if limit_frames else movie.n_frames
    n_kpts = cfg.n_keypoints

    frames_c, ids_c, box_c, conf_c, kpt_c, feat_c = [], [], [], [], [], []
    for f in reporter.track(range(n_frames), f"{movie.id}  detect + track", n_frames):
        rgb = normalizer.to_rgb_u8(movie.frame(f), f)
        dets = detector.detect(rgb)
        ids, idx = tracker.update(dets)
        if len(ids) == 0:
            continue
        frames_c.append(np.full(len(ids), f, dtype=np.int32))
        ids_c.append(ids.astype(np.int32))
        box_c.append(dets.xyxy[idx])
        conf_c.append(dets.conf[idx])
        if n_kpts and dets.kpts is not None:
            kpt_c.append(dets.kpts[idx])
        if dets.feats is not None:
            feat_c.append(dets.feats[idx])

    if not frames_c:
        return pd.DataFrame(
            columns=["frame", "track_id", "x1", "y1", "x2", "y2", "conf"]
        ), None

    boxes = np.concatenate(box_c, axis=0)
    data = {
        "frame": np.concatenate(frames_c),
        "track_id": np.concatenate(ids_c),
        "x1": boxes[:, 0], "y1": boxes[:, 1], "x2": boxes[:, 2], "y2": boxes[:, 3],
        "conf": np.concatenate(conf_c),
    }
    if kpt_c and len(kpt_c) == len(frames_c):
        kpts = np.concatenate(kpt_c, axis=0)  # (N, K, 3)
        for k in range(kpts.shape[1]):
            data[f"kp_x_{k}"] = kpts[:, k, 0]
            data[f"kp_y_{k}"] = kpts[:, k, 1]
            data[f"kp_c_{k}"] = kpts[:, k, 2]

    # Keep features only if EVERY frame produced them; a partial array silently
    # misaligns against the tracks table.
    feats = (
        np.concatenate(feat_c, axis=0)
        if feat_c and len(feat_c) == len(frames_c)
        else None
    )
    return pd.DataFrame(data), feats


# --------------------------------------------------------------------------- #
# the stage handoff
# --------------------------------------------------------------------------- #
# What the detection stage's outputs depend on. A change to any of these makes an
# existing tracks.parquet stale; the rest of the config - clip geometry, spectral,
# categories - belongs to the clip stage and may change freely between the two.
# `device` and `batch` are excluded deliberately: they decide where and how the
# work runs, not what comes out of it, so detecting on a GPU box and cutting the
# clips on a laptop is not a mismatch.
_FINGERPRINTED = ("detection", "tiling", "tracking", "features", "normalize", "nd2", "tiff")
_FINGERPRINT_SKIP = {"detection": ("device", "batch")}


def _config_fingerprint(cfg: Config) -> dict:
    """Per-section digests, so a stale handoff can name what changed."""
    from dataclasses import asdict, is_dataclass

    out: dict[str, str] = {}
    for name in _FINGERPRINTED:
        section = getattr(cfg, name, None)
        if section is None:
            continue
        values = asdict(section) if is_dataclass(section) else dict(section)
        for key in _FINGERPRINT_SKIP.get(name, ()):
            values.pop(key, None)
        text = json.dumps(values, sort_keys=True, default=str)
        out[name] = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return out


def read_handoff(cfg: Config, movie_id: str) -> dict:
    """What the detection stage recorded for one movie.

    Raises if it is absent or was made under different settings.
    """
    path = cfg.movie_dir(movie_id) / DETECT_FILE
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise FileNotFoundError(
            f"{movie_id}: no {DETECT_FILE} in {path.parent}. Detection has not run for "
            f"this movie - run it here, or copy work/{movie_id}/ back from the machine "
            "that did."
        ) from exc
    except ValueError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc

    was = info.get("config_fingerprint") or {}
    now = _config_fingerprint(cfg)
    if not was:
        raise ValueError(
            f"{movie_id}: {DETECT_FILE} records no config fingerprint, so nothing here "
            "can say whether these detections were made under this project's settings. "
            "Re-run detection for this movie."
        )
    # Over the union: a section missing from either side is a difference, not
    # something to skip. Skipping it would accept exactly the handoff that cannot
    # account for itself.
    changed = sorted(k for k in set(was) | set(now) if was.get(k) != now.get(k))
    if changed:
        raise ValueError(
            f"{movie_id}: these detections were made with a different config "
            f"({', '.join(changed)} changed since). Re-run detection for this movie, "
            "or put those settings back the way they were."
        )
    return info


def _needs_detection(cfg: Config, movie_id: str, force: bool = False) -> bool:
    """Whether stage 1 has to run before this movie's clips can be cut.

    A prepared movie needs nothing, handoff or not: `prepare_movie` returns at its
    marker without ever looking at a detector.
    """
    if not force and (cfg.movie_dir(movie_id) / DONE_FILE).exists():
        return False
    return force or not handoff_is_fresh(cfg, movie_id)


def handoff_is_fresh(cfg: Config, movie_id: str) -> bool:
    """Whether the clip stage could run for this movie without detecting again."""
    try:
        read_handoff(cfg, movie_id)
    except (OSError, ValueError):
        return False
    return True


def _detector(source):
    """Accept either a detector or something that builds one on demand."""
    return source() if callable(source) else source


# --------------------------------------------------------------------------- #
# stage 1: detect and track
# --------------------------------------------------------------------------- #
def detect_movie(
    movie: MovieSource,
    detector: TiledDetector,
    cfg: Config,
    force: bool = False,
    limit_frames: int | None = None,
    reporter: Reporter | None = None,
) -> dict:
    """Normalise, detect and track. Writes the tracks, the features and the handoff.

    The half that needs the model, and so the half that can run on a GPU box.
    """
    reporter = reporter or NullReporter()
    out_dir = cfg.movie_dir(movie.id)
    handoff_path = out_dir / DETECT_FILE
    if handoff_path.exists() and not force:
        info = json.loads(handoff_path.read_text(encoding="utf-8"))
        reporter.log(
            f"{movie.id}: already detected ({info.get('n_detections', '?')} detections)"
        )
        return info

    out_dir.mkdir(parents=True, exist_ok=True)
    # Clear the markers BEFORE touching anything: the artefacts are only
    # self-consistent once all of them are there, so one surviving a cancelled
    # rebuild pairs a new cache with the old manifest. Re-detecting also
    # invalidates any clips already cut, so that marker goes too.
    handoff_path.unlink(missing_ok=True)
    (out_dir / DONE_FILE).unlink(missing_ok=True)
    started = time.time()

    tiled = detector.is_tiled(movie.height, movie.width)
    n_tiles = len(detector.tiles_for(movie.height, movie.width))
    reporter.log(
        f"{movie.id}: {movie.n_frames} frames {movie.width}x{movie.height} "
        f"@ {movie.fps:g} fps | {'tiled ' + str(n_tiles) + ' tiles/frame' if tiled else 'single tile'}"
    )

    normalizer = Normalizer.from_movie(
        movie, cfg.normalize,
        progress=lambda it: reporter.track(it, f"{movie.id}  intensity levels"),
        n_frames=limit_frames,
    )
    normalizer.save(out_dir / NORM_FILE)

    tracks, feats = run_tracking(movie, detector, cfg, normalizer, limit_frames, reporter)
    tracks.to_parquet(out_dir / TRACKS_FILE, index=False)

    feature_path = out_dir / FEATURES_FILE
    feature_dim = 0
    if feats is not None and len(feats) == len(tracks):
        np.save(feature_path, feats)
        feature_dim = int(feats.shape[1])
    else:
        if feats is not None:
            reporter.log(
                f"feature rows ({len(feats)}) != detections ({len(tracks)}); "
                "features not written",
                "warning",
            )
        feature_path.unlink(missing_ok=True)

    n_frames = min(movie.n_frames, limit_frames) if limit_frames else movie.n_frames
    info = {
        "movie_id": movie.id,
        "source_file": str(movie.path),
        "series": movie.series,
        "n_frames": n_frames,
        "height": movie.height,
        "width": movie.width,
        "n_detections": int(len(tracks)),
        "n_tracks": int(tracks["track_id"].nunique()) if not tracks.empty else 0,
        # Resolved from the checkpoint's kpt_shape when the detector was built.
        # The machine that cuts the clips may never load that checkpoint, so the
        # names have to travel with the detections.
        "keypoint_nodes": list(cfg.keypoints.nodes),
        "tiled": bool(tiled),
        "n_tiles": n_tiles,
        "feature_dim": feature_dim,
        "device": str(getattr(detector, "device", "")),
        "model": cfg.paths.model.name,
        "model_sha256": str(getattr(getattr(cfg, "model", None), "sha256", "")),
        "config_fingerprint": _config_fingerprint(cfg),
        "detect_seconds": round(time.time() - started, 1),
    }
    # Written LAST: its presence is the promise that everything above it landed.
    handoff_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    reporter.log(
        f"-> {info['n_tracks']} tracks, {info['n_detections']} detections "
        f"in {info['detect_seconds']:.0f}s"
    )
    return info


# --------------------------------------------------------------------------- #
# stage 2: plan and cut the clips
# --------------------------------------------------------------------------- #
def cut_movie(
    movie: MovieSource,
    cfg: Config,
    force: bool = False,
    write_sleap: bool = False,
    reporter: Reporter | None = None,
) -> dict:
    """Plan the clips and cut their pixels, from what the detection stage left.

    No model, no GPU, no network. Its output is the clip cache, so it runs where
    the annotating does.
    """
    reporter = reporter or NullReporter()
    out_dir = cfg.movie_dir(movie.id)
    done_path = out_dir / DONE_FILE
    if done_path.exists() and not force:
        info = json.loads(done_path.read_text(encoding="utf-8"))
        reporter.log(
            f"{movie.id}: already prepared ({info.get('n_clips', '?')} clips) - "
            "tick 'redo finished movies' to rebuild"
        )
        return info

    detected = read_handoff(cfg, movie.id)
    out_dir.mkdir(parents=True, exist_ok=True)
    done_path.unlink(missing_ok=True)
    started = time.time()

    # The names came from the checkpoint, not from config.yaml. Without this a
    # pose project whose config lists no `keypoints.nodes` would cut a box table
    # with no keypoint columns at all, and say nothing about it.
    nodes = detected.get("keypoint_nodes") or []
    if cfg.is_pose and nodes and list(cfg.keypoints.nodes) != list(nodes):
        cfg.keypoints.nodes = list(nodes)

    tracks = pd.read_parquet(out_dir / TRACKS_FILE)
    # Loaded rather than remeasured, even when detection just ran in this
    # process: the mapping the labels were applied under is the one the review
    # view reloads, and one route through it is easier to keep honest than two.
    normalizer = Normalizer.load(out_dir / NORM_FILE)

    n_frames = int(detected.get("n_frames") or movie.n_frames)
    frame_size = (movie.height, movie.width)
    if cfg.tracking.border_margin_px > 0 and not tracks.empty:
        # Over the tracks that were still candidates for a clip: plan_clips
        # applies min_track_length first, so counting before it reports
        # discards that changed nothing.
        counts = tracks.groupby("track_id").size()
        long_enough = counts[counts >= cfg.tracking.min_track_length].index
        _, at_border = drop_border_tracks(
            tracks[tracks["track_id"].isin(long_enough)], frame_size, cfg
        )
        if at_border:
            worst = max(at_border.values())
            reporter.log(
                f"{len(at_border)} track(s) discarded for reaching within "
                f"{cfg.tracking.border_margin_px}px of the frame edge "
                f"(tracking.border_margin_px); the worst spent "
                f"{100 * worst:.0f}% of its life there. Their rows are still in "
                f"{TRACKS_FILE}, they just become no clips",
                "warning",
            )
    specs = plan_clips(tracks, movie.id, n_frames, cfg, frame_size=frame_size)
    size = choose_crop_size(specs, cfg)
    origins = finalise_geometry(specs, size, cfg.clips.frames)

    manifest = specs_to_frame(specs)
    if not manifest.empty:
        manifest["source_file"] = str(movie.path)
        manifest["series"] = movie.series
        manifest["channel"] = movie.channel

    # One commit: the pixels, the manifest and the box table only make sense
    # together.
    staged = StagedWrite(out_dir)
    boxes = build_box_table(specs, origins, tracks, cfg, cfg.n_keypoints)
    try:
        if not boxes.empty:
            boxes.to_parquet(staged.path(BOXES_FILE), index=False)
        elif (out_dir / BOXES_FILE).exists():
            (out_dir / BOXES_FILE).unlink()

        write_clip_cache(
            movie, specs, origins, size, out_dir, normalizer, cfg,
            progress=lambda it: reporter.track(it, f"{movie.id}  cutting clips"),
            staged=staged,
        )
        manifest.to_parquet(staged.path(MANIFEST_FILE), index=False)
    except BaseException:
        staged.discard()
        raise
    staged.commit()

    if write_sleap:
        _write_sleap(movie, tracks, cfg, out_dir)

    n_tracks = int(tracks["track_id"].nunique()) if not tracks.empty else 0
    info = {
        "movie_id": movie.id,
        "source_file": str(movie.path),
        "series": movie.series,
        "n_frames": n_frames,
        "n_detections": int(len(tracks)),
        "n_tracks": n_tracks,
        "n_clips": len(specs),
        "crop_size": size,
        "clip_frames": cfg.clips.frames,
        "window_mode": cfg.clips.window_mode,
        # Carried through from the detection stage: these describe how the boxes
        # were found, and this half never built a detector to ask.
        "tiled": bool(detected.get("tiled", False)),
        "n_tiles": int(detected.get("n_tiles", 1)),
        "feature_dim": int(detected.get("feature_dim", 0)),
        "seconds": round(time.time() - started, 1),
    }
    (out_dir / DONE_FILE).write_text(json.dumps(info, indent=2), encoding="utf-8")

    if specs:
        mb = len(specs) * cfg.clips.frames * size * size / 1e6
        reporter.log(
            f"-> {n_tracks} tracks, {len(specs)} clips, crop {size}px "
            f"({mb:.0f} MB cache) in {info['seconds']:.0f}s"
        )
    else:
        border = (
            f", or reach within tracking.border_margin_px "
            f"({cfg.tracking.border_margin_px}) of the frame edge"
            if cfg.tracking.border_margin_px > 0 else ""
        )
        reporter.log(
            f"{n_tracks} tracks but 0 clips. Tracks may be shorter than "
            f"clips.min_clip_frames ({cfg.clips.min_clip_frames}) or "
            f"tracking.min_track_length ({cfg.tracking.min_track_length}){border}.",
            "warning",
        )
    return info


# --------------------------------------------------------------------------- #
# both stages
# --------------------------------------------------------------------------- #
def prepare_movie(
    movie: MovieSource,
    detector,
    cfg: Config,
    force: bool = False,
    limit_frames: int | None = None,
    write_sleap: bool = False,
    reporter: Reporter | None = None,
) -> dict:
    """Whatever this movie still needs in order to have a clip cache.

    Not "run both stages": a movie detected elsewhere already has its handoff, so
    only its clips are cut. ``detector`` may be a callable returning one, and is
    not called when there is nothing left to detect.
    """
    reporter = reporter or NullReporter()
    out_dir = cfg.movie_dir(movie.id)
    done_path = out_dir / DONE_FILE
    if done_path.exists() and not force:
        info = json.loads(done_path.read_text(encoding="utf-8"))
        reporter.log(
            f"{movie.id}: already prepared ({info.get('n_clips', '?')} clips) - "
            "tick 'redo finished movies' to rebuild"
        )
        return info

    if force or not handoff_is_fresh(cfg, movie.id):
        detect_movie(movie, _detector(detector), cfg, force, limit_frames, reporter)
    else:
        reporter.log(f"{movie.id}: detections already here, cutting clips only")
    return cut_movie(movie, cfg, force, write_sleap, reporter)


# --------------------------------------------------------------------------- #
# batch driver - shared by the CLI and the GUI
# --------------------------------------------------------------------------- #
class _LazyDetector:
    """Builds the detector on first use, and not at all if nothing needs one.

    So a batch with nothing left to detect needs no weights, torch or registry.
    """

    def __init__(self, cfg: Config, reporter: Reporter):
        self._cfg, self._reporter, self._made = cfg, reporter, None

    def __call__(self) -> TiledDetector:
        if self._made is None:
            from ..video.models import resolve_model

            # Resolve a registry model to a cached file first, then carry on as if
            # it had always been local: nothing downstream needs to know a network
            # was involved.
            self._cfg.paths.model = resolve_model(self._cfg, self._reporter)
            if not self._cfg.paths.model.exists():
                raise FileNotFoundError(
                    f"detection model not found: {self._cfg.paths.model}\n"
                    "Set paths.model to the .pt weights file, or model.source to a "
                    "model from the registry."
                )
            self._reporter.log(f"loading model {self._cfg.paths.model.name}...")
            self._made = TiledDetector(self._cfg, reporter=self._reporter)
        return self._made


def prepare_all(
    cfg: Config,
    movie_ids: list[str] | None = None,
    force: bool = False,
    limit_frames: int | None = None,
    write_sleap: bool = False,
    reporter: Reporter | None = None,
    stage: str = "all",
) -> list[dict]:
    """Prepare every requested movie; one bad movie never stops the batch. The
    CLI and the GUI both come through here, differing only in the Reporter.

    ``stage`` is ``all`` (do whatever each movie still needs), ``detect`` (stop
    after the tracks, for a GPU box) or ``clips`` (cut from tracks that are
    already there, for the machine that annotates).
    """
    if stage not in ("all", "detect", "clips"):
        raise ValueError(f"stage must be 'all', 'detect' or 'clips', not {stage!r}")
    reporter = reporter or NullReporter()

    cfg.paths.work_dir.mkdir(parents=True, exist_ok=True)
    movies = discover_movies(cfg, only=movie_ids)
    reporter.log(f"found {len(movies)} movie(s) in {cfg.paths.video_dir}")
    if not movies:
        return []

    detector = _LazyDetector(cfg, reporter)
    # Up front when anything in this batch will actually detect, and only then.
    # Deliberately outside the per-movie try/except below: a missing checkpoint is
    # one thing wrong with the whole run, and finding that out before the first
    # movie beats finding it out once per movie after an hour of work.
    if stage != "clips" and any(_needs_detection(cfg, m.id, force) for m in movies):
        detector()

    results: list[dict] = []
    try:
        for i, movie in enumerate(movies):
            reporter.overall(i, len(movies), movie.id)
            try:
                reporter.raise_if_cancelled()
                if stage == "detect":
                    results.append(
                        detect_movie(movie, detector(), cfg, force, limit_frames, reporter)
                    )
                elif stage == "clips":
                    results.append(cut_movie(movie, cfg, force, write_sleap, reporter))
                else:
                    results.append(
                        prepare_movie(
                            movie, detector, cfg, force, limit_frames, write_sleap, reporter
                        )
                    )
            except Cancelled:
                reporter.log("cancelled", "warning")
                raise
            except Exception as exc:  # keep going; one bad movie shouldn't stop the batch
                reporter.log(f"{movie.id}: FAILED - {type(exc).__name__}: {exc}", "error")
            finally:
                movie.close()
        reporter.overall(len(movies), len(movies), "done")
    finally:
        for movie in movies:
            try:
                movie.close()
            except Exception:
                pass
    return results


def _write_sleap(movie: MovieSource, tracks: pd.DataFrame, cfg: Config, out_dir: Path) -> None:
    """Optional SLEAP-analysis .h5, so notebook 02 (keypoint-MoSeq) keeps working."""
    try:
        import sleap_io as sio
    except ImportError:
        warnings.warn("--sleap requested but sleap-io is not installed (uv sync --extra sleap)")
        return
    if tracks.empty or not cfg.is_pose:
        return

    nodes = [sio.Node(n) for n in cfg.keypoints.nodes]
    nmap = {n.name: n for n in nodes}
    skeleton = sio.Skeleton(
        nodes=nodes, edges=[sio.Edge(nmap[a], nmap[b]) for a, b in cfg.keypoints.edges],
        name="animal",
    )
    counts = tracks.groupby("track_id").size()
    kept = counts[counts >= cfg.tracking.min_track_length].sort_values(ascending=False).index
    track_objs = {tid: sio.Track(name=f"track{i}") for i, tid in enumerate(kept)}
    video = sio.Video(filename=str(movie.path), open_backend=False)

    n_kpts = cfg.n_keypoints
    by_frame: dict[int, list] = {}
    subset = tracks[tracks["track_id"].isin(kept)]
    for rec in subset.itertuples(index=False):
        xy = np.array([[getattr(rec, f"kp_x_{k}"), getattr(rec, f"kp_y_{k}")] for k in range(n_kpts)],
                      dtype=np.float32)
        conf = np.array([getattr(rec, f"kp_c_{k}") for k in range(n_kpts)], dtype=np.float32)
        by_frame.setdefault(int(rec.frame), []).append(
            sio.PredictedInstance.from_numpy(
                points_data=xy, skeleton=skeleton, point_scores=conf,
                score=float(np.nanmean(conf)), track=track_objs[rec.track_id],
            )
        )
    lfs = [sio.LabeledFrame(video=video, frame_idx=f, instances=i) for f, i in sorted(by_frame.items())]
    for anchor in (0, movie.n_frames - 1):
        if anchor not in by_frame:
            lfs.append(sio.LabeledFrame(video=video, frame_idx=anchor, instances=[]))
    sio.save_analysis_h5(
        sio.Labels(labeled_frames=lfs, videos=[video], skeletons=[skeleton],
                   tracks=list(track_objs.values())),
        str(out_dir / f"{movie.id}.h5"),
    )


# --------------------------------------------------------------------------- #
# self test
# --------------------------------------------------------------------------- #
def self_test(movie: MovieSource, detector: TiledDetector, cfg: Config, n_frames: int = 5) -> None:
    """Compare tiled vs. whole-frame inference on the same frames. Real movies
    are larger than the model input, so this only says anything on a small movie
    with tiling forced (``--tile 320``)."""
    from .tiling import _iou_matrix

    if not detector.is_tiled(movie.height, movie.width):
        print(
            f"  {movie.id}: {movie.width}x{movie.height} fits in one {cfg.tiling.tile}px tile, "
            f"so there is nothing to compare. Re-run with e.g. --tile {max(64, movie.width // 2)}."
        )
        return

    normalizer = Normalizer.from_movie(movie, cfg.normalize)
    idx = np.unique(np.linspace(0, movie.n_frames - 1, n_frames).astype(int))
    print(f"  {movie.id}: comparing tiled vs untiled on {len(idx)} frames")
    for f in idx:
        rgb = normalizer.to_rgb_u8(movie.frame(int(f)), int(f))
        tiled = detector.detect(rgb)
        full = detector.detect_untiled(rgb)
        if len(tiled) and len(full):
            iou = _iou_matrix(tiled.xyxy, full.xyxy)
            matched = int((iou.max(axis=1) > 0.5).sum())
            median_iou = float(np.median(iou.max(axis=1)))
        else:
            matched, median_iou = 0, float("nan")
        print(
            f"    frame {f:>5}: tiled={len(tiled):>4}  untiled={len(full):>4}  "
            f"matched@IoU0.5={matched:>4}  median best IoU={median_iou:.3f}"
        )


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="prepare",
        description="Run tiled YOLO detection + tracking and build the annotation clip cache.",
    )
    p.add_argument("--config", default=None, help="path to config.yaml")
    p.add_argument("--movies", nargs="*", default=None, metavar="ID",
                   help="only these movie ids (default: all found in paths.video_dir)")
    p.add_argument("--force", action="store_true", help="redo movies that are already prepared")
    p.add_argument("--stage", choices=("all", "detect", "clips"), default="all",
                   help="all (default): give each movie whatever it still needs. "
                        "detect: stop after the tracks, for a GPU box. "
                        "clips: cut clips from tracks that are already there.")
    p.add_argument("--limit-frames", type=int, default=None,
                   help="only process the first N frames (quick smoke test)")
    # A config.yaml carries whatever the Configure page's Browse button put in it,
    # which is an absolute path on the machine that picked it. These three let the
    # same project run on a second machine without editing the file - the point of
    # a split run, where the GPU box mounts the movies somewhere else entirely.
    p.add_argument("--video-dir", default=None, help="override paths.video_dir")
    p.add_argument("--work-dir", default=None, help="override paths.work_dir")
    p.add_argument("--model", default=None,
                   help="override paths.model; also clears model.source, since naming "
                        "a file says the shared model is not wanted")
    p.add_argument("--device", default=None, help="override detection.device, e.g. cpu")
    p.add_argument("--tile", type=int, default=None, help="override tiling.tile")
    p.add_argument("--batch", type=int, default=None, help="override detection.batch")
    p.add_argument("--sleap", action="store_true",
                   help="also write a SLEAP .h5 per movie (needs --extra sleap)")
    p.add_argument("--self-test", action="store_true",
                   help="compare tiled vs whole-frame inference instead of preparing")
    p.add_argument("--list", action="store_true", help="list discoverable movies and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)

    if args.video_dir:
        cfg.paths.video_dir = Path(args.video_dir).expanduser().resolve()
    if args.work_dir:
        cfg.paths.work_dir = Path(args.work_dir).expanduser().resolve()
    if args.model:
        cfg.paths.model = Path(args.model).expanduser().resolve()
        # model.source outranks paths.model, so leaving it set would silently
        # ignore the file just named on the command line.
        if getattr(cfg, "model", None) is not None:
            cfg.model.source = ""
    if args.device:
        cfg.detection.device = args.device
    if args.tile:
        cfg.tiling.tile = args.tile
        cfg.tiling.enabled = True
    if args.batch:
        cfg.detection.batch = args.batch

    work = cfg.paths.work_dir
    if str(work).startswith("//") or str(work).startswith("\\\\"):
        warnings.warn(
            f"paths.work_dir is on a network share ({work}). The GUI does ~1 MB random reads "
            "per clip; put the cache on a local disk or annotation will feel sluggish.",
            stacklevel=1,
        )

    if args.list:
        for m in discover_movies(cfg, only=args.movies):
            print(f"  {m.id:<45} {m.n_frames:>6} frames  {m.width}x{m.height}  "
                  f"{m.dtype}  {m.fps:g} fps  [{m.path.suffix}]")
            m.close()
        return 0

    if args.self_test:
        if not cfg.paths.model.exists():
            print(f"error: model not found at {cfg.paths.model}", file=sys.stderr)
            return 2
        detector = TiledDetector(cfg)
        for movie in discover_movies(cfg, only=args.movies):
            try:
                self_test(movie, detector, cfg)
            finally:
                movie.close()
        return 0

    try:
        prepare_all(
            cfg, args.movies, args.force, args.limit_frames, args.sleap,
            ConsoleReporter(), stage=args.stage,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Cancelled:
        return 130

    if args.stage == "detect":
        print(f"\nDetections: {work}\nNext: copy work/ to the annotating machine, "
              "then run there with --stage clips (or just open trackanno).")
    else:
        print(f"\nClip cache: {work}\nNext: uv run annotate.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
