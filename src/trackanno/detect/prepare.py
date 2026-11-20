"""``trackanno prepare``: detect and track (stage 1, GPU), then cut the clip cache (stage 2).

Stage 2 needs only stage 1's files in ``work/<movie_id>/``, so it can run on another machine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..annotate.clips import (
    DETECT_FILE, DONE_FILE, FEATURES_FILE, MANIFEST_FILE, NORM_FILE, TRACKS_FILE,
    BOXES_FILE,
    StagedWrite, build_box_table, cache_normalizer, choose_crop_size,
    finalise_geometry, plan_clips,
    specs_to_frame, write_clip_cache,
)
from ..config import FIXED_SETTINGS, Config, load_config
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
    """Detect and track every frame; returns ``(tracks, features)``, features possibly None."""
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

    # Keep features only if every frame produced them, so rows stay aligned.
    feats = (
        np.concatenate(feat_c, axis=0)
        if feat_c and len(feat_c) == len(frames_c)
        else None
    )
    return pd.DataFrame(data), feats


# --------------------------------------------------------------------------- #
# the stage handoff
# --------------------------------------------------------------------------- #
# Settings that change what detection writes (not clip-stage keys, device or batch).
_DETECT_KEYS = (
    "detection.task", "detection.conf", "detection.imgsz", "detection.half",
    "tiling.enabled", "tiling.tile", "tiling.overlap", "tiling.merge_iou",
    "tracking.backend", "tracking.overrides",
    "features.enabled",
    "normalize.mode", "normalize.percentiles",
    "nd2.channel", "nd2.z", "tiff.channel",
    "nd2.fps", "tiff.fps",   # scale the tracker's track_buffer
    "centerline.enabled", "centerline.checkpoint", "centerline.hint_model",
    "centerline.stack_stride", "centerline.hint_conf", "centerline.hint_imgsz",
    "centerline.hint_tile",
)
# Settings that change what the clip stage cuts from the same tracks.
_CLIP_KEYS = (
    "clips.frames", "clips.stride", "clips.window_mode", "clips.follow",
    "clips.pad", "clips.crop_size", "clips.min_clip_frames",
    "tracking.min_track_length", "tracking.border_margin_px",
    "tracking.border_max_frac",
)
_MODEL_KEY = "model.sha256"
_SHA_CACHE: dict[tuple, str] = {}


def pin_configured_tiling(cfg: Config) -> None:
    """Keep the config.yaml tiling for the fingerprint before a ``--tile`` override."""
    from dataclasses import replace

    cfg._configured_tiling = replace(cfg.tiling)


def _plain(value):
    """A JSON round-trip, so tuples and lists compare equal to their stored form."""
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _section(cfg: Config, name: str):
    if name == "tiling":
        return getattr(cfg, "_configured_tiling", None) or cfg.tiling
    return getattr(cfg, name, None)


def _settings(cfg: Config, keys: tuple[str, ...]) -> dict:
    out = {}
    for dotted in keys:
        name, key = dotted.split(".", 1)
        section = _section(cfg, name)
        if section is not None and hasattr(section, key):
            out[dotted] = _plain(getattr(section, key))
    return out


def _default(cfg: Config, dotted: str):
    """The dataclass default for a key, or ``MISSING`` when there is none."""
    from dataclasses import MISSING, fields, is_dataclass

    name, key = dotted.split(".", 1)
    section = _section(cfg, name)
    if is_dataclass(section):
        for f in fields(section):
            if f.name == key:
                if f.default is not MISSING:
                    return _plain(f.default)
                if f.default_factory is not MISSING:
                    return _plain(f.default_factory())
    return MISSING


def _model_identity(cfg: Config) -> str:
    """sha256 of the weights when they are here, else the pinned ``model.sha256``."""
    path = Path(cfg.paths.model)
    if path.is_file():
        st = path.stat()
        key = (str(path), st.st_size, st.st_mtime_ns)
        if key not in _SHA_CACHE:
            from ..video.models import _digest

            _SHA_CACHE[key] = _digest(path)
        return _SHA_CACHE[key]
    return str(getattr(getattr(cfg, "model", None), "sha256", "") or "")


def _config_fingerprint(cfg: Config) -> dict:
    """The detection settings by dotted key, so a stale handoff can name them."""
    out = _settings(cfg, _DETECT_KEYS)
    out[_MODEL_KEY] = _model_identity(cfg)
    return out


def _legacy_sections(cfg: Config) -> dict[str, dict]:
    """The sections older handoffs digested whole."""
    from dataclasses import is_dataclass

    out: dict[str, dict] = {}
    for name in ("detection", "tiling", "tracking", "features", "normalize", "nd2", "tiff"):
        section = getattr(cfg, name, None)
        if section is None:
            continue
        values = asdict(section) if is_dataclass(section) else dict(section)
        # Keys that are now fixed, at the values older handoffs were hashed with.
        for dotted, fixed in FIXED_SETTINGS.items():
            section_name, _, key = dotted.partition(".")
            if section_name == name:
                values.setdefault(key, fixed)
        if name == "normalize":
            values.setdefault("sample_every", "auto")
            values.setdefault("smooth_window", 9)
        if name == "detection":
            values.pop("device", None)
            values.pop("batch", None)
        out[name] = values
    return out


def _legacy_digest(values: dict) -> str:
    text = json.dumps(values, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _legacy_fingerprint(cfg: Config) -> dict:
    """The per-section digests older handoffs recorded, to keep accepting them."""
    return {name: _legacy_digest(v) for name, v in _legacy_sections(cfg).items()}


_ABSENT = object()


def _legacy_matches_without_clip_keys(
    cfg: Config, movie_id: str, name: str, values: dict, digest: str
) -> bool:
    """Whether absent, default or recorded clip-key values reproduce the old ``digest``."""
    import itertools
    from dataclasses import MISSING

    keys = [d.split(".", 1)[1] for d in _CLIP_KEYS
            if d.startswith(name + ".") and d.split(".", 1)[1] in values]
    if not keys:
        return False
    marker = _read_json(cfg.movie_dir(movie_id) / DONE_FILE) or {}
    recorded = marker.get("clip_settings")
    recorded = recorded if isinstance(recorded, dict) else {}
    options = []
    for key in keys:
        dotted = f"{name}.{key}"
        opts = [values[key], _ABSENT]
        default = _default(cfg, dotted)
        if default is not MISSING:
            opts.append(default)
        if dotted in recorded:
            opts.append(recorded[dotted])
        options.append(opts)
    for combo in itertools.product(*options):
        trial = dict(values)
        for key, value in zip(keys, combo):
            if value is _ABSENT:
                trial.pop(key)
            else:
                trial[key] = value
        if _legacy_digest(trial) == digest:
            return True
    return False


# Removed normalize modes whose replacement gives the same pixels (not `none`).
_SAME_PIXELS = {"smooth": "per_frame"}


def _same_mode(was, now) -> bool:
    """Whether a recorded normalize.mode gives the pixels the current one does."""
    return was == now or _SAME_PIXELS.get(was) == now


def _legacy_variants(name: str, values: dict) -> list[dict]:
    """``values``, plus the removed normalize modes that map onto its mode."""
    out = [values]
    if name == "normalize":
        out += [{**values, "mode": old} for old, new in _SAME_PIXELS.items()
                if new == values.get("mode")]
    return out


def _legacy_changes(cfg: Config, movie_id: str, was: dict) -> list[str]:
    """Sections whose old digest no longer matches, as ``section.*``."""
    now = _legacy_sections(cfg)
    changed = []
    # Over the union, so a partial record fails.
    for name in sorted(set(was) | set(now)):
        if name in now and name in was and any(
            was[name] == _legacy_digest(values)
            or _legacy_matches_without_clip_keys(cfg, movie_id, name, values, was[name])
            for values in _legacy_variants(name, now[name])
        ):
            continue
        changed.append(f"{name}.*")
    return changed


def _changed(cfg: Config, was: dict, now: dict) -> list[str]:
    """Keys that differ. A key the record lacks differs only if not at its default."""
    from dataclasses import MISSING

    changed = []
    for key, value in now.items():
        if key == _MODEL_KEY:
            # Skip when unknown on one side (no weights on this machine).
            if was.get(key) and value and was[key] != value:
                changed.append(key)
        elif key in was:
            if key == "normalize.mode" and _same_mode(was[key], value):
                continue
            if was[key] != value:
                changed.append(key)
        else:
            default = _default(cfg, key)
            if default is MISSING or default != value:
                changed.append(key)
    # A recorded key that is now fixed compares against the fixed value.
    for key in set(was) - set(now):
        if key in FIXED_SETTINGS and was[key] != _plain(FIXED_SETTINGS[key]):
            changed.append(key)
    return sorted(changed)


def _read_json(path: Path) -> dict | None:
    """A marker's contents, or None when it is absent or unreadable."""
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return info if isinstance(info, dict) else None


def _write_json(path: Path, info: dict) -> None:
    """Write through a temporary file, so a crash never leaves a truncated marker."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(info, indent=2), encoding="utf-8")
    tmp.replace(path)


def _load_handoff(cfg: Config, movie_id: str) -> dict:
    """The handoff as written. Raises if it is absent, unreadable or unchecked."""
    path = cfg.movie_dir(movie_id) / DETECT_FILE
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise FileNotFoundError(
            f"{movie_id}: no {DETECT_FILE} in {path.parent}; run detection or copy "
            f"work/{movie_id}/ from the machine that did"
        ) from exc
    except ValueError as exc:
        raise ValueError(f"{path.name} is not valid JSON: {exc}") from exc

    was = info.get("config_fingerprint") if isinstance(info, dict) else None
    if not was or not isinstance(was, dict):
        raise ValueError(f"{DETECT_FILE} records no config fingerprint")
    return info


def _detection_changes(
    cfg: Config, movie_id: str, info: dict, check_model: bool = True
) -> list[str]:
    was = info["config_fingerprint"]
    if any("." in key for key in was):
        now = _config_fingerprint(cfg) if check_model else _settings(cfg, _DETECT_KEYS)
        return _changed(cfg, was, now)
    return _legacy_changes(cfg, movie_id, was)


def detection_changes(cfg: Config, movie_id: str, check_model: bool = True) -> list[str]:
    """Detection keys changed since the handoff; raises without a readable handoff.

    ``check_model=False`` skips hashing the weights."""
    return _detection_changes(cfg, movie_id, _load_handoff(cfg, movie_id), check_model)


# How to ask for a redo, from the CLI and from the Prepare page.
_REDO = "--force (Prepare page: 'redo movies that are already prepared')"


def read_handoff(cfg: Config, movie_id: str) -> dict:
    """The movie's detection handoff; raises if absent or made under different settings."""
    info = _load_handoff(cfg, movie_id)
    changed = _detection_changes(cfg, movie_id, info)
    if changed:
        raise ValueError(
            f"{movie_id}: detections were made with a different config "
            f"({', '.join(changed)}); re-detect with {_REDO} or restore those settings"
        )
    return info


def clip_changes(cfg: Config, movie_id: str) -> list[str] | None:
    """Clip keys changed since the clips were cut; None when nothing is cut."""
    path = cfg.movie_dir(movie_id) / DONE_FILE
    if not path.exists():
        return None
    info = _read_json(path)
    if info is None:
        raise ValueError(f"{DONE_FILE} is unreadable")
    was = info.get("clip_settings")
    if not isinstance(was, dict):
        # No recorded clip settings: trust the clips.
        return []
    return _changed(cfg, was, _settings(cfg, _CLIP_KEYS))


def clips_are_fresh(cfg: Config, movie_id: str) -> dict | None:
    """The prepared marker if the clips match the current clip settings, else None."""
    try:
        changed = clip_changes(cfg, movie_id)
    except ValueError:
        return None
    if changed is None or changed:
        return None
    return _read_json(cfg.movie_dir(movie_id) / DONE_FILE)


def _detection_stale(cfg: Config, movie_id: str, check_model: bool = True) -> str:
    """Why the handoff no longer matches, or "" when it matches or is absent."""
    try:
        changed = detection_changes(cfg, movie_id, check_model)
    except FileNotFoundError:
        return ""
    except ValueError as exc:
        return f"detections cannot be checked ({exc})"
    return f"detection settings changed ({', '.join(changed)})" if changed else ""


def _clips_stale(cfg: Config, movie_id: str) -> str:
    """Why the cut clips no longer match, or "" when they match or are absent."""
    try:
        changed = clip_changes(cfg, movie_id)
    except ValueError as exc:
        return f"clips cannot be checked ({exc})"
    return f"clip settings changed ({', '.join(changed)})" if changed else ""


def stale_reasons(cfg: Config, movie_id: str, check_model: bool = True) -> list[str]:
    """What no longer matches the config for this movie; empty when nothing does."""
    return [r for r in (_detection_stale(cfg, movie_id, check_model),
                        _clips_stale(cfg, movie_id)) if r]


def _skip_stale(movie_id: str, reasons: list[str], reporter: Reporter) -> dict:
    """Report a stale movie and leave its detections and clips as they are."""
    if any(r.startswith("detection") for r in reasons):
        redo = f"run again with {_REDO} to re-detect"
    else:
        redo = f"re-cut with --stage clips {_REDO}"
    reporter.log(f"{movie_id}: {'; '.join(reasons)}. Skipped; {redo}.", "warning")
    return {"movie_id": movie_id, "skipped": True, "stale": reasons}


def _needs_detection(cfg: Config, movie_id: str, force: bool = False) -> bool:
    """Whether stage 1 must run: with ``force``, or with neither clips nor a handoff."""
    if force:
        return True
    out = cfg.movie_dir(movie_id)
    return not (out / DONE_FILE).exists() and not (out / DETECT_FILE).exists()


def handoff_is_fresh(cfg: Config, movie_id: str) -> bool:
    """Whether the clip stage could run for this movie without detecting again."""
    try:
        return not detection_changes(cfg, movie_id)
    except (OSError, ValueError):
        return False


def _centerline_model(detector, cfg: Config, reporter: Reporter):
    """The centerline model, loaded once per detector."""
    from .centerline import CenterlineModel

    made = getattr(detector, "centerline_model", None)
    if made is None:
        reporter.stage("loading the centerline model", None)
        made = CenterlineModel(cfg, reporter)
        detector.centerline_model = made
    if cfg.keypoints.n != made.K + 1:
        from ..project import write_config
        from .centerline import node_names

        nodes = node_names(made.K)
        edges = [[a, b] for a, b in zip(nodes, nodes[1:])]
        write_config(cfg.path, {}, keypoints={"nodes": nodes, "edges": edges})
        cfg.keypoints.nodes, cfg.keypoints.edges = nodes, [tuple(e) for e in edges]
        reporter.log(f"keypoints.nodes set to the Head and the centerline's {made.K} points")
    return made


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
    """Stage 1: normalise, detect and track; writes tracks, features and the handoff."""
    reporter = reporter or NullReporter()
    out_dir = cfg.movie_dir(movie.id)
    handoff_path = out_dir / DETECT_FILE
    if handoff_path.exists() and not force:
        stale = _detection_stale(cfg, movie.id)
        if stale:
            return _skip_stale(movie.id, [stale], reporter)
        info = _read_json(handoff_path) or {}
        reporter.log(
            f"{movie.id}: already detected ({info.get('n_detections', '?')} detections)"
        )
        return info

    out_dir.mkdir(parents=True, exist_ok=True)
    # Clear both markers first; re-detecting invalidates any cut clips.
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

    # Loaded first: it names the output nodes, and tracking keeps the pose Head only with names.
    centerline = _centerline_model(detector, cfg, reporter) if cfg.centerline.enabled else None
    tracks, feats = run_tracking(movie, detector, cfg, normalizer, limit_frames, reporter)
    if centerline is not None:
        from .centerline import refine_tracks

        tracks = refine_tracks(movie, tracks, cfg, normalizer, centerline, reporter)
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
        # Names may come from the checkpoint, which stage 2 never loads.
        "keypoint_nodes": list(cfg.keypoints.nodes),
        "centerline": ({"checkpoint": centerline.name, "epoch": centerline.epoch}
                       if centerline is not None else None),
        "tiled": bool(tiled),
        "n_tiles": n_tiles,
        "feature_dim": feature_dim,
        "device": str(getattr(detector, "device", "")),
        "model": cfg.paths.model.name,
        "model_sha256": _model_identity(cfg),
        "config_fingerprint": _config_fingerprint(cfg),
        # Records a `--tile` override, which the fingerprint leaves out.
        "tiling_used": _plain(asdict(cfg.tiling)),
        "detect_seconds": round(time.time() - started, 1),
    }
    # Written last: it marks everything above as complete.
    _write_json(handoff_path, info)
    reporter.log(
        f"-> {info['n_tracks']} tracks, {info['n_detections']} detections "
        f"in {info['detect_seconds']:.0f}s"
    )
    return info


def _already_prepared(cfg: Config, movie_id: str, reporter: Reporter) -> dict | None:
    """The marker of a movie with current clips, a skip record if stale, or None without clips."""
    if not (cfg.movie_dir(movie_id) / DONE_FILE).exists():
        return None
    reasons = stale_reasons(cfg, movie_id)
    if reasons:
        return _skip_stale(movie_id, reasons, reporter)
    info = _read_json(cfg.movie_dir(movie_id) / DONE_FILE) or {}
    reporter.log(
        f"{movie_id}: already prepared ({info.get('n_clips', '?')} clips); "
        f"redo with {_REDO}"
    )
    return info


# --------------------------------------------------------------------------- #
# stage 2: plan and cut the clips
# --------------------------------------------------------------------------- #
def cut_movie(
    movie: MovieSource,
    cfg: Config,
    force: bool = False,
    reporter: Reporter | None = None,
) -> dict:
    """Stage 2: plan the clips and cut the clip cache from the stage-1 handoff."""
    reporter = reporter or NullReporter()
    out_dir = cfg.movie_dir(movie.id)
    done_path = out_dir / DONE_FILE
    if not force:
        info = _already_prepared(cfg, movie.id, reporter)
        if info is not None:
            return info

    # Clips are never cut from stale tracks, forced or not.
    stale = _detection_stale(cfg, movie.id)
    if stale:
        return _skip_stale(movie.id, [stale], reporter)
    detected = _load_handoff(cfg, movie.id)
    out_dir.mkdir(parents=True, exist_ok=True)
    done_path.unlink(missing_ok=True)
    started = time.time()

    # Keypoint names recorded by stage 1 (possibly from the checkpoint).
    nodes = detected.get("keypoint_nodes") or []
    if cfg.is_pose and nodes and list(cfg.keypoints.nodes) != list(nodes):
        cfg.keypoints.nodes = list(nodes)

    reporter.stage(f"{movie.id}  planning clips", None)
    tracks = pd.read_parquet(out_dir / TRACKS_FILE)
    norm_path = out_dir / NORM_FILE
    detected_norm = Normalizer.load(norm_path) if norm_path.exists() else None

    n_frames = int(detected.get("n_frames") or movie.n_frames)
    frame_size = (movie.height, movie.width)
    at_border: dict = {}
    specs = plan_clips(tracks, movie.id, n_frames, cfg, frame_size=frame_size,
                       at_border=at_border)
    if at_border:
        worst = max(at_border.values())
        reporter.log(
            f"{len(at_border)} track(s) come within {cfg.tracking.border_margin_px}px "
            f"of the frame edge (worst {100 * worst:.0f}% of frames); those frames "
            "are excluded from clips",
            "warning",
        )
    size = choose_crop_size(specs, cfg)
    origins = finalise_geometry(specs, size, cfg.clips.frames)
    if specs:
        reporter.log(
            f"{movie.id}: {len(specs)} clips, crop {size}px, "
            f"{len(specs) * cfg.clips.frames * size * size / 1e6:.0f} MB of cache"
        )

    manifest = specs_to_frame(specs)
    if not manifest.empty:
        manifest["source_file"] = str(movie.path)
        manifest["series"] = movie.series
        manifest["channel"] = movie.channel

    # Pixels, manifest and box table commit together.
    staged = StagedWrite(out_dir)
    boxes = build_box_table(specs, origins, tracks, cfg, cfg.n_keypoints)
    try:
        if not boxes.empty:
            boxes.to_parquet(staged.path(BOXES_FILE), index=False)
        elif (out_dir / BOXES_FILE).exists():
            (out_dir / BOXES_FILE).unlink()

        write_clip_cache(
            movie, specs, origins, size, out_dir, cache_normalizer(movie, detected_norm), cfg,
            progress=lambda it: reporter.track(it, f"{movie.id}  cutting clips"),
            staged=staged,
        )
        manifest.to_parquet(staged.path(MANIFEST_FILE), index=False)
    except BaseException:
        staged.discard()
        raise
    staged.commit()

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
        "clip_settings": _settings(cfg, _CLIP_KEYS),
        # Carried over from the detection handoff.
        "tiled": bool(detected.get("tiled", False)),
        "n_tiles": int(detected.get("n_tiles", 1)),
        "feature_dim": int(detected.get("feature_dim", 0)),
        "seconds": round(time.time() - started, 1),
    }
    _write_json(out_dir / DONE_FILE, info)

    if specs:
        mb = len(specs) * cfg.clips.frames * size * size / 1e6
        reporter.log(
            f"-> {n_tracks} tracks, {len(specs)} clips, crop {size}px "
            f"({mb:.0f} MB cache) in {info['seconds']:.0f}s"
        )
    else:
        border = (
            f", or spend them within tracking.border_margin_px "
            f"({cfg.tracking.border_margin_px}) of the frame edge"
            if cfg.tracking.border_margin_px > 0 else ""
        )
        reporter.log(
            f"{n_tracks} tracks but 0 clips; tracks may be shorter than "
            f"clips.min_clip_frames ({cfg.clips.min_clip_frames}) or "
            f"tracking.min_track_length ({cfg.tracking.min_track_length}){border}",
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
    reporter: Reporter | None = None,
) -> dict:
    """Run the stages this movie still needs; ``detector`` may be a factory, called only if needed."""
    reporter = reporter or NullReporter()
    if not force:
        info = _already_prepared(cfg, movie.id, reporter)
        if info is not None:
            return info
        stale = _detection_stale(cfg, movie.id)
        if stale:
            return _skip_stale(movie.id, [stale], reporter)

    if force or not (cfg.movie_dir(movie.id) / DETECT_FILE).exists():
        detect_movie(movie, _detector(detector), cfg, force, limit_frames, reporter)
    else:
        reporter.log(f"{movie.id}: detections already here, cutting clips only")
    # The marker was checked above, so force the cut.
    return cut_movie(movie, cfg, True, reporter)


# --------------------------------------------------------------------------- #
# batch driver, shared by the CLI and the GUI
# --------------------------------------------------------------------------- #
class _LazyDetector:
    """Builds the detector on first use, so a batch with nothing to detect needs no weights."""

    def __init__(self, cfg: Config, reporter: Reporter):
        self._cfg, self._reporter, self._made = cfg, reporter, None

    def __call__(self) -> TiledDetector:
        if self._made is None:
            self._reporter.stage("loading the detection model", None)
            from ..video.models import resolve_model

            self._cfg.paths.model = resolve_model(self._cfg, self._reporter)
            if not self._cfg.paths.model.exists():
                raise FileNotFoundError(
                    f"detection model not found: {self._cfg.paths.model}; set paths.model "
                    "or model.source"
                )
            self._reporter.log(f"loading model {self._cfg.paths.model.name}...")
            self._made = TiledDetector(self._cfg, reporter=self._reporter)
        return self._made


def prepare_all(
    cfg: Config,
    movie_ids: list[str] | None = None,
    force: bool = False,
    limit_frames: int | None = None,
    reporter: Reporter | None = None,
    stage: str = "all",
) -> list[dict]:
    """Prepare each requested movie, continuing past failures.

    ``stage``: ``all`` (whatever each movie needs), ``detect`` (stage 1) or ``clips`` (stage 2)."""
    if stage not in ("all", "detect", "clips"):
        raise ValueError(f"stage must be 'all', 'detect' or 'clips', not {stage!r}")
    reporter = reporter or NullReporter()

    cfg.paths.work_dir.mkdir(parents=True, exist_ok=True)
    reporter.stage(f"looking for movies in {cfg.paths.video_dir}", None)
    movies = discover_movies(cfg, only=movie_ids)
    reporter.log(f"found {len(movies)} movie(s) in {cfg.paths.video_dir}")
    if not movies:
        return []

    detector = _LazyDetector(cfg, reporter)
    # Load the model up front, outside the per-movie try, so a missing one fails once.
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
                    results.append(cut_movie(movie, cfg, force, reporter))
                else:
                    results.append(
                        prepare_movie(
                            movie, detector, cfg, force, limit_frames, reporter
                        )
                    )
            except Cancelled:
                reporter.log("cancelled", "warning")
                raise
            except Exception as exc:  # one bad movie does not stop the batch
                reporter.log(f"{movie.id}: FAILED - {type(exc).__name__}: {exc}", "error")
            finally:
                movie.close()
        skipped = [r["movie_id"] for r in results if r.get("skipped")]
        if skipped:
            reporter.log(
                f"{len(skipped)} stale movie(s) left as they were: {', '.join(skipped)}",
                "warning",
            )
        reporter.overall(len(movies), len(movies), "done")
    finally:
        for movie in movies:
            try:
                movie.close()
            except Exception:
                pass
    return results


# --------------------------------------------------------------------------- #
# self test
# --------------------------------------------------------------------------- #
def self_test(movie: MovieSource, detector: TiledDetector, cfg: Config, n_frames: int = 5) -> None:
    """Compare tiled and whole-frame inference on a few frames (small movie, ``--tile``)."""
    from .tiling import _iou_matrix

    if not detector.is_tiled(movie.height, movie.width):
        print(
            f"  {movie.id}: {movie.width}x{movie.height} fits in one {cfg.tiling.tile}px tile; "
            f"re-run with e.g. --tile {max(64, movie.width // 2)}"
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
        prog="trackanno prepare",
        description="Detect and track animals, then build the annotation clip cache.",
    )
    p.add_argument("--config", default=None, help="path to config.yaml")
    p.add_argument("--movies", nargs="*", default=None, metavar="ID",
                   help="only these movie ids (default: all)")
    p.add_argument("--force", action="store_true", help="redo movies that are already prepared")
    p.add_argument("--stage", choices=("all", "detect", "clips"), default="all",
                   help="all: whatever each movie needs; detect: tracks only; "
                        "clips: cut clips from existing tracks")
    p.add_argument("--limit-frames", type=int, default=None,
                   help="only process the first N frames")
    # Path overrides let the same project run on another machine.
    p.add_argument("--video-dir", default=None, help="override paths.video_dir")
    p.add_argument("--work-dir", default=None, help="override paths.work_dir")
    p.add_argument("--model", default=None,
                   help="override paths.model (clears model.source)")
    p.add_argument("--device", default=None, help="override detection.device, e.g. cpu")
    p.add_argument("--tile", type=int, default=None, help="override tiling.tile")
    p.add_argument("--batch", type=int, default=None, help="override detection.batch")
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
        # model.source outranks paths.model.
        if getattr(cfg, "model", None) is not None:
            cfg.model.source = ""
    if args.device:
        cfg.detection.device = args.device
    if args.tile:
        pin_configured_tiling(cfg)
        cfg.tiling.tile = args.tile
        cfg.tiling.enabled = True
    if args.batch:
        cfg.detection.batch = args.batch

    work = cfg.paths.work_dir
    if str(work).startswith("//") or str(work).startswith("\\\\"):
        warnings.warn(
            f"paths.work_dir is on a network share ({work}); use a local disk for fast annotation",
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
            cfg, args.movies, args.force, args.limit_frames,
            ConsoleReporter(), stage=args.stage,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Cancelled:
        return 130

    if args.stage == "detect":
        print(f"\nDetections: {work}\nNext: copy work/ to the annotating machine and run "
              "--stage clips there (or open trackanno).")
    else:
        print(f"\nClip cache: {work}\nNext: open trackanno and annotate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
