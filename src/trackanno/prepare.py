"""Entry point 1: detection + tracking + clip cache.

    uv run prepare.py [--movies ID ...] [--force] [--limit-frames N] [--self-test]

The slow, GPU-bound half of the pipeline. Once per movie, it leaves behind:

    work/<movie_id>/
        tracks.parquet     every detection, in ORIGINAL full-frame coordinates
        clips.parquet      one row per annotatable clip + its crop geometry
        clip_boxes.parquet overlay geometry, in cache coordinates
        clips.dat          the pre-cropped pixels (uncompressed memmap)
        thumbs.dat         optional full-frame thumbnails for the GUI minimap
        norm.json          the movie's fixed intensity mapping
        cache.json         cache shape + movie metadata
        prepare_done.json  resume marker

Re-running skips finished movies unless ``--force``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .clips import (
    DONE_FILE, FEATURES_FILE, MANIFEST_FILE, NORM_FILE, TRACKS_FILE, BOXES_FILE,
    StagedWrite, build_box_table, choose_crop_size, finalise_geometry, plan_clips,
    specs_to_frame, write_clip_cache,
)
from .config import Config, load_config
from .movies import MovieSource, Normalizer, discover_movies
from .progress import Cancelled, ConsoleReporter, NullReporter, Reporter
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
# per-movie driver
# --------------------------------------------------------------------------- #
def prepare_movie(
    movie: MovieSource,
    detector: TiledDetector,
    cfg: Config,
    force: bool = False,
    limit_frames: int | None = None,
    write_sleap: bool = False,
    reporter: Reporter | None = None,
) -> dict:
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

    out_dir.mkdir(parents=True, exist_ok=True)
    # Clear the marker BEFORE touching anything: the artefacts are only
    # self-consistent once all of them are there, so one surviving a cancelled
    # rebuild pairs a new cache with the old manifest.
    done_path.unlink(missing_ok=True)
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
    if feats is not None and len(feats) == len(tracks):
        np.save(feature_path, feats)
    else:
        if feats is not None:
            reporter.log(
                f"feature rows ({len(feats)}) != detections ({len(tracks)}); "
                "features not written",
                "warning",
            )
        feature_path.unlink(missing_ok=True)

    n_frames = min(movie.n_frames, limit_frames) if limit_frames else movie.n_frames
    specs = plan_clips(tracks, movie.id, n_frames, cfg)
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
        "tiled": bool(tiled),
        "n_tiles": n_tiles,
        "feature_dim": int(feats.shape[1]) if feats is not None else 0,
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
        reporter.log(
            f"{n_tracks} tracks but 0 clips. Tracks may be shorter than "
            f"clips.min_clip_frames ({cfg.clips.min_clip_frames}) or "
            f"tracking.min_track_length ({cfg.tracking.min_track_length}).",
            "warning",
        )
    return info


# --------------------------------------------------------------------------- #
# batch driver - shared by the CLI and the GUI
# --------------------------------------------------------------------------- #
def prepare_all(
    cfg: Config,
    movie_ids: list[str] | None = None,
    force: bool = False,
    limit_frames: int | None = None,
    write_sleap: bool = False,
    reporter: Reporter | None = None,
) -> list[dict]:
    """Prepare every requested movie; one bad movie never stops the batch. The
    CLI and the GUI both come through here, differing only in the Reporter."""
    reporter = reporter or NullReporter()
    # Resolve a registry model to a cached file first, then carry on as if it had
    # always been local: nothing downstream needs to know a network was involved.
    from .models import resolve_model

    cfg.paths.model = resolve_model(cfg, reporter)
    if not cfg.paths.model.exists():
        raise FileNotFoundError(
            f"detection model not found: {cfg.paths.model}\n"
            "Set paths.model to the .pt weights file, or model.source to a model "
            "from the registry."
        )

    cfg.paths.work_dir.mkdir(parents=True, exist_ok=True)
    movies = discover_movies(cfg, only=movie_ids)
    reporter.log(f"found {len(movies)} movie(s) in {cfg.paths.video_dir}")
    if not movies:
        return []

    reporter.log(f"loading model {cfg.paths.model.name}...")
    detector = TiledDetector(cfg, reporter=reporter)
    results: list[dict] = []
    try:
        for i, movie in enumerate(movies):
            reporter.overall(i, len(movies), movie.id)
            try:
                reporter.raise_if_cancelled()
                results.append(
                    prepare_movie(movie, detector, cfg, force, limit_frames, write_sleap, reporter)
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
    p.add_argument("--limit-frames", type=int, default=None,
                   help="only process the first N frames (quick smoke test)")
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
            cfg, args.movies, args.force, args.limit_frames, args.sleap, ConsoleReporter()
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Cancelled:
        return 130

    print(f"\nClip cache: {work}\nNext: uv run annotate.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
