"""Entry point 1: detection + tracking + clip cache.

    trackanno prepare [--movies ID ...] [--force] [--limit-frames N] [--self-test]

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
        cache.json         cache shape + movie metadata
        prepare_done.json  resume marker

Stage 2 needs no model, no GPU and no network, so detection can run on a remote
box and the clips - the multi-gigabyte artefact - be cut where they are read.
Copy ``work/<movie_id>/`` back between the two; the four stage-1 files are small.

Without ``--stage``, each movie gets whatever it is still missing, so a run
started elsewhere is finished rather than repeated. Finished movies are skipped
unless ``--force``. So are movies whose detections or clips no longer match the
config: the changed keys are reported, and only ``--force`` detects or cuts again.
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
    StagedWrite, build_box_table, choose_crop_size,
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
# Settings that change what detection writes, fps included. Clip-stage keys
# (min_track_length, border_*) and where it runs (device, batch) are left out.
_DETECT_KEYS = (
    "detection.task", "detection.conf", "detection.imgsz", "detection.half",
    "tiling.enabled", "tiling.tile", "tiling.overlap", "tiling.merge_iou",
    "tracking.backend", "tracking.overrides",
    "features.enabled",
    "normalize.mode", "normalize.percentiles",
    "nd2.channel", "nd2.z", "tiff.channel",
    "nd2.fps", "tiff.fps",   # scale the tracker's track_buffer
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
    """Whether some old value of the section's clip-stage keys reproduces ``digest``.

    Old digests covered clip-stage keys too. Candidates for their old values are
    the key being absent, its default, and what the clip marker recorded.
    """
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


# Removed normalize modes whose replacement gives the same pixels. `none` maps
# to `fixed` on load but rescales differently, so a record of it is stale.
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
            # Unknown on one side, e.g. no weights where the clips are cut: skip.
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
            f"{movie_id}: no {DETECT_FILE} in {path.parent}. Detection has not run for "
            f"this movie - run it here, or copy work/{movie_id}/ back from the machine "
            "that did."
        ) from exc
    except ValueError as exc:
        raise ValueError(f"{path.name} is not valid JSON: {exc}") from exc

    was = info.get("config_fingerprint") if isinstance(info, dict) else None
    if not was or not isinstance(was, dict):
        raise ValueError(
            f"{DETECT_FILE} records no config fingerprint, so nothing here can say "
            "whether these detections were made under this project's settings"
        )
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
    """Detection keys changed since the handoff was written; empty when it is fresh.

    Raises FileNotFoundError without a handoff, ValueError when it cannot be
    checked. ``check_model=False`` skips hashing the weights.
    """
    return _detection_changes(cfg, movie_id, _load_handoff(cfg, movie_id), check_model)


# How to ask for a redo, from the CLI and from the Prepare page.
_REDO = "--force (Prepare page: 'redo movies that are already prepared')"


def read_handoff(cfg: Config, movie_id: str) -> dict:
    """What the detection stage recorded for one movie.

    Raises if it is absent or was made under different settings.
    """
    info = _load_handoff(cfg, movie_id)
    changed = _detection_changes(cfg, movie_id, info)
    if changed:
        raise ValueError(
            f"{movie_id}: these detections were made with a different config "
            f"({', '.join(changed)} changed since). Re-detect with {_REDO}, "
            "or put those settings back the way they were."
        )
    return info


def clip_changes(cfg: Config, movie_id: str) -> list[str] | None:
    """Clip keys changed since the clips were cut; None when nothing is cut.

    Raises ValueError when the prepared marker is there but unreadable.
    """
    path = cfg.movie_dir(movie_id) / DONE_FILE
    if not path.exists():
        return None
    info = _read_json(path)
    if info is None:
        raise ValueError(f"{DONE_FILE} is unreadable")
    was = info.get("clip_settings")
    if not isinstance(was, dict):
        # Cut before clip settings were recorded: trusted rather than re-cut.
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
    """Whether stage 1 has to run before this movie's clips can be cut.

    Without ``force`` only a movie with neither clips nor a handoff does; a stale
    one is reported and skipped, never detected again.
    """
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
        stale = _detection_stale(cfg, movie.id)
        if stale:
            return _skip_stale(movie.id, [stale], reporter)
        info = _read_json(handoff_path) or {}
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
        "model_sha256": _model_identity(cfg),
        "config_fingerprint": _config_fingerprint(cfg),
        # A `--tile` override is not in the fingerprint, so it is recorded here.
        "tiling_used": _plain(asdict(cfg.tiling)),
        "detect_seconds": round(time.time() - started, 1),
    }
    # Written LAST: its presence is the promise that everything above it landed.
    _write_json(handoff_path, info)
    reporter.log(
        f"-> {info['n_tracks']} tracks, {info['n_detections']} detections "
        f"in {info['detect_seconds']:.0f}s"
    )
    return info


def _already_prepared(cfg: Config, movie_id: str, reporter: Reporter) -> dict | None:
    """What to return for a movie with clips, or None when it has none yet.

    Current clips return their marker; stale ones are reported and skipped.
    """
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
    """Plan the clips and cut their pixels, from what the detection stage left.

    No model, no GPU, no network. Its output is the clip cache, so it runs where
    the annotating does.
    """
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

    # The names came from the checkpoint, not from config.yaml. Without this a
    # pose project whose config lists no `keypoints.nodes` would cut a box table
    # with no keypoint columns at all, and say nothing about it.
    nodes = detected.get("keypoint_nodes") or []
    if cfg.is_pose and nodes and list(cfg.keypoints.nodes) != list(nodes):
        cfg.keypoints.nodes = list(nodes)

    # Named before it starts, not after: reading a big tracks table and laying
    # the window grid over it is seconds to a minute with nothing to count, and
    # an unlabelled bar there reads as a hang.
    reporter.stage(f"{movie.id}  planning clips", None)
    tracks = pd.read_parquet(out_dir / TRACKS_FILE)
    # Loaded rather than remeasured, even when detection just ran in this
    # process: the mapping the labels were applied under is the one the review
    # view reloads, and one route through it is easier to keep honest than two.
    normalizer = Normalizer.load(out_dir / NORM_FILE)

    n_frames = int(detected.get("n_frames") or movie.n_frames)
    frame_size = (movie.height, movie.width)
    # Reported by plan_clips as it goes, since it decides which detections were
    # candidates and counting them here would restate that rule.
    at_border: dict = {}
    specs = plan_clips(tracks, movie.id, n_frames, cfg, frame_size=frame_size,
                       at_border=at_border)
    if at_border:
        worst = max(at_border.values())
        reporter.log(
            f"{len(at_border)} track(s) reach within "
            f"{cfg.tracking.border_margin_px}px of the frame edge "
            f"(tracking.border_margin_px); the worst spent "
            f"{100 * worst:.0f}% of its life there. Those frames are excluded "
            f"and the rest of each track still becomes clips. Their rows are "
            f"still in {TRACKS_FILE}",
            "warning",
        )
    size = choose_crop_size(specs, cfg)
    origins = finalise_geometry(specs, size, cfg.clips.frames)
    if specs:
        # Said BEFORE the pixels are cut, because this is the number that decides
        # how long the rest takes and how much disk it wants, and finding it out
        # afterwards is finding it out too late.
        reporter.log(
            f"{movie.id}: {len(specs)} clips, crop {size}px, "
            f"{len(specs) * cfg.clips.frames * size * size / 1e6:.0f} MB of cache"
        )

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
        # Carried through from the detection stage: these describe how the boxes
        # were found, and this half never built a detector to ask.
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
    reporter: Reporter | None = None,
) -> dict:
    """Whatever this movie still needs in order to have a clip cache.

    Not "run both stages": a movie detected elsewhere already has its handoff, so
    only its clips are cut. ``detector`` may be a callable returning one, and is
    not called when there is nothing left to detect.
    """
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
    # The marker was checked above, so the clips are cut regardless of it.
    return cut_movie(movie, cfg, True, reporter)


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
            # Before the import, not after: resolving a registry model can go to
            # the network and loading torch takes tens of seconds, and both are
            # otherwise a silent gap that reads as a hang.
            self._reporter.stage("loading the detection model", None)
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
    # Every movie in the folder is opened to be identified, ND2 positions and
    # all, so on a share this is minutes before the first frame is read. Named,
    # or the run looks stuck before it has started.
    reporter.stage(f"looking for movies in {cfg.paths.video_dir}", None)
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
            except Exception as exc:  # keep going; one bad movie shouldn't stop the batch
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
        prog="trackanno prepare",
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
        pin_configured_tiling(cfg)
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
            cfg, args.movies, args.force, args.limit_frames,
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
        print(f"\nClip cache: {work}\nNext: open trackanno and annotate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
