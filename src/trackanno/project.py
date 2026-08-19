"""Projects: a folder, a config, and the state of the work done in it.

A *project* is a directory holding

    config.yaml        every setting, with its comments intact
    work/              the clip cache and the trained suggester
    annotations.csv    the labels

This module creates one from a template, remembers the ones you have opened,
and summarises how far each has got without opening a single movie file.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .yamledit import replace_section, set_values

__all__ = [
    "DEFAULT_CONFIG", "ProjectStatus", "create_project", "project_status",
    "recent_projects", "remember_project", "forget_project", "write_config",
    "candidate_config", "config_path_for",
]

CONFIG_NAME = "config.yaml"
RECENT_FILE = "recent.json"
MAX_RECENT = 12


DEFAULT_CONFIG = """\
# =============================================================================
# trackanno project configuration
#
# Created by the New Project screen. Every value here is editable from the
# Configure tab, which rewrites this file in place and keeps these comments.
# Paths are resolved relative to THIS file.
# =============================================================================

paths:
  # Folder containing the movies. .tif/.tiff and .nd2 are both read natively.
  video_dir: "./videos"
  # YOLO weights used for detection (and pose, if detection.task is 'pose').
  model: "./best.pt"
  # Clip cache + per-movie tables. KEEP THIS ON A LOCAL DISK: the GUI does
  # ~1 MB random reads per clip and over a network share that latency is what
  # you feel.
  work_dir: "./work"
  # The annotation log. One row per annotated clip; this file IS the state.
  csv: "./annotations.csv"
  patterns: ["*.tif", "*.tiff", "*.nd2"]


# --- ND2-specific reading ----------------------------------------------------
# ND2 files can carry P (multipoint), T, C and Z axes. We need a single 2D
# grayscale time series, so choose how to collapse the extra axes.
nd2:
  channel: 0          # channel index, or its name as a string (e.g. "BF")
  z: 0                # z index, or "max" / "mean" / "min" for a projection
  positions: all      # "all", or a list like [0, 2, 5]. Each P becomes a movie.
  fps: auto           # "auto" reads the frame interval from the ND2 metadata


# --- TIFF-specific reading ---------------------------------------------------
tiff:
  fps: 100            # TIFF carries no frame rate; set the acquisition rate
  channel: 0


# --- Raw -> uint8 intensity mapping ------------------------------------------
#   per_frame -> own mapping per frame (default; flattest)
#   smooth    -> sampled + median-filtered over time; corrects slow drift only
#   fixed     -> one mapping for the whole movie (faithful, including flicker)
#   none      -> pass raw uint8 straight through
normalize:
  mode: per_frame
  percentiles: [0.5, 99.5]
  sample_every: auto
  smooth_window: 9
  subsample: 4


# --- Detection ---------------------------------------------------------------
detection:
  task: detect        # "pose" (model predicts keypoints) or "detect" (boxes)
  conf: 0.25
  iou: 0.5            # per-tile NMS
  imgsz: 640          # inference size - match what the model was trained at
  max_det: 300
  device: auto        # auto = first CUDA GPU, else Apple MPS, else the CPU.
                      # "0" pins the first CUDA device, "mps", or "cpu".
  batch: 8            # tiles per forward pass (lower if the GPU runs out of memory)
  half: false


# --- Tiled inference ---------------------------------------------------------
# Movies larger than one model input get cut into overlapping tiles, run as a
# batch, mapped back to full-frame coordinates and merged with a global NMS.
# Frames that already fit in one tile skip all of this automatically.
tiling:
  enabled: auto       # auto | true | false
  tile: 640           # how the frame is CUT; detection.imgsz is the resolution
                      # each tile is SEEN at
  overlap: 0.25
  merge_iou: 0.6
  drop_edge_truncated: true
  edge_margin_px: 2


# --- Tracking ----------------------------------------------------------------
tracking:
  backend: tracktrack       # tracktrack | bytetrack | botsort | ocsort | greedy
  tracker_cfg: ""           # "" = the backend's bundled ultralytics yaml
  overrides:
    # Compensates for CAMERA motion. On a static stage everything that moves is
    # an animal, so leave this off or a global transform gets fitted to them.
    gmc_method: none
  min_track_length: 40      # drop tracks seen in fewer frames than this
  greedy_max_dist_px: 25


# --- Keypoints (only used when detection.task is 'pose') ---------------------
# The order MUST match the trained model.
keypoints:
  nodes: []
  edges: []


# --- Clip extraction ---------------------------------------------------------
clips:
  frames: 50          # clip length -> "50 frames of a single track"
  stride: 50          # == frames means non-overlapping windows

  # How much of each movie becomes clips:
  #   first           -> ONLY the movie's first `frames` frames
  #   first_per_track -> the first full window of each track
  #   all             -> every non-overlapping window of every track
  window_mode: all

  # The crop follows the animal; `follow` sets how closely:
  #   track  -> re-centred every frame. Smallest crop, translation invisible.
  #   smooth -> centre follows a moving average.  (default)
  #   fixed  -> one crop for the window. All motion visible, biggest crop.
  follow: smooth
  crop_from: bbox     # bbox | union
  smooth_window: 25

  pad: 1.6            # crop = per-frame box extent x this
  crop_size: auto     # "auto" = p99 of the required sizes, rounded to /16
  max_size: 256
  min_clip_frames: 40
  max_clips_per_track: 0   # 0 = unlimited
  context_tracks: true     # record neighbouring tracks that fall in the crop
  thumbnails: false        # full-frame thumbnails for the minimap (~65 kB/clip)
  thumbnail_size: 256


# --- Per-detection CNN features ----------------------------------------------
# Pooled from the SAME forward pass as detection - no second pass. Used by the
# suggestion model.
features:
  enabled: true
  levels: [P3, P4, P5]
  pool: 1
  dtype: float16


# --- Annotation suggestions ---------------------------------------------------
# Trains a classifier on your labels and SUGGESTS a category. The suggestion is
# displayed only - never pre-selected - so your labels stay independent of the
# model trained on them. It stays silent until it clears a chance-corrected bar.
suggest:
  enabled: true
  model: hist           # hist (sklearn) | catboost (uv sync --extra catboost)
  # Which families of measurement the model may use. Drop 'cnn' if you want a
  # behaviour model rather than one that can re-identify individuals; drop
  # 'pose'/'waveform' automatically happens when detection.task is 'detect'.
  #   kinematic  trajectory and box shape   (always available)
  #   pose       body/flagellum, beat       (needs task: pose)
  #   waveform   beat shape after Geyer, Howard & Sartori 2022 (needs task: pose)
  #   cnn        detector embeddings        (needs features.enabled)
  feature_groups: [kinematic, pose, waveform, cnn]
  pca_components: 32
  retrain_every: 10     # retrain in the background after this many new labels
  min_per_class: 5
  min_macro_f1: 0.45
  audit_fraction: 0.10
  # Held-out clips show NO suggestion at any confidence. Set audit_fraction
  # to 0 to suggest on every clip and give up the agreement measure; set
  # audit_visible to true to be told which clips those are.
  audit_visible: false  # share of clips that never show a suggestion, so
                        # human-vs-model agreement stays measurable
  seed: 0


# --- Bulk labelling ----------------------------------------------------------
# What `python -m trackanno.predict` scores when it writes predictions.parquet.
#   dense     -> its OWN window grid, tiling every frame of every track. The
#                clip grid above is a sampling chosen for a human's attention,
#                so scoring only those clips leaves most of the movie
#                unlabelled and unpainted in the render. Needs no pixels: clip
#                features come from tracks.parquet and instance_features.npy,
#                so nothing is re-cropped or re-detected.  (default)
#   manifest  -> exactly the clips in clip_features.parquet, and nothing else.
# Dense windows tile EXACTLY (stride == length), because the renderer paints
# range(f_start, f_end) per track and two windows sharing a frame means one
# prediction silently overwrites the other.
predict:
  coverage: dense
  # Length of one dense window, which is also its stride. auto =
  # min(clips.frames, clips.stride): the longest window that both tiles and
  # lands on the annotation grid's own starts, so a dense window coinciding
  # with an annotated one keeps its clip_id and hand labels still join to it.
  window_frames: auto
  # Detections a dense window needs before it is featurised. NOT
  # clips.min_clip_frames - that guarantees a watchable video clip, and nothing
  # here is watched. 2 is the floor at which any motion feature can exist.
  # Windows below it still get a ROW, with an empty prediction and a
  # skip_reason saying why.
  min_frames: 2


# --- Annotation categories ---------------------------------------------------
# Each group becomes a block of buttons. `key` is the keyboard shortcut and
# `color` is used for the buttons AND the rendered overlay movie.
# EDIT THESE - the ones below are placeholders.
categories:
  - name: behaviour
    type: single          # single = one choice | multi = checkboxes
    required: true        # auto-advance waits until every required group is set
    options:
      - {name: swimming, key: "1", color: "#4C9F70"}
      - {name: immotile, key: "2", color: "#D95F02"}
      - {name: unclear,  key: "3", color: "#7570B3"}

  - name: quality
    type: multi
    required: false
    options:
      - {name: blurry,       key: "q", color: "#999999"}
      - {name: occluded,     key: "w", color: "#777777"}
      - {name: track_switch, key: "e", color: "#BB2222"}


# --- GUI ---------------------------------------------------------------------
ui:
  theme: dark          # dark | light | system (follow the desktop)
  backend: simple       # simple = built-in player, and the only backend that
                        # draws the marker as vectors rather than into pixels
                        # ndv    = embed ndv.ArrayViewer  |  auto = try ndv first
  fps: 20               # playback rate in the GUI (not the acquisition rate)
  autoplay: true
  loop: true
  auto_advance: true    # one keypress labels and jumps to the next clip

  # Annotate in a seeded random order. Going movie-by-movie invites order
  # effects - you calibrate on the first movie and drift.
  shuffle: true
  shuffle_seed: 0

  show_overlay: true          # mark the animal this clip is about (key: o)
  show_skeleton: false        # pose keypoints on top of the marker
  show_context_tracks: false  # ALSO outline other tracks in the crop (key: c).
                              # Off by default: the marker's job is to say which
                              # animal is being judged.

  # What the view follows (key: s cycles):
  #   centred -> one view on the track's median position; all motion visible
  #   smooth  -> view follows a moving average; drift removed, beat kept
  #   locked  -> re-centred every frame; translation invisible by construction
  follow: centred
  smooth_window: 15
  view_zoom: 1.0        # magnification into the crop (keys: + / - / 0)

  show_minimap: false   # needs clips.thumbnails: true
  # Curvature of the flagellum along its length against time, drawn under
  # the clip. Only appears when detection.task is 'pose'.
  show_kymograph: true
  kymo_interpolate: true   # false shows the raw one-block-per-frame grid
  kymo_smooth_px: 0.6      # how far the fitted centreline may leave a keypoint
  kymo_field: curvature    # curvature | tangent (the unwrapped angle psi)
  # A 2D image cannot tell a cell from its mirror, so the same beat can come
  # out blue in one clip and red in the next. canonical | raw | absolute
  kymo_sign: canonical
  prefetch: 8           # clips read ahead in the background
  resume: first_unannotated   # first_unannotated | last_position
  annotator: ""

# Keyboard shortcuts. Category keys are defined per option above.
keys:
  next: ["Right", "Return"]
  prev: ["Left"]
  next_unannotated: ["Tab"]
  play_pause: ["Space"]
  overlay: ["o"]
  stabilise: ["s"]
  context: ["c"]
  zoom_in: ["+", "="]
  zoom_out: ["-"]
  zoom_reset: ["0"]
  minimap: ["m"]
  uncertain: ["u"]
  notes: ["n"]
  clear: ["Backspace"]
  undo: ["Ctrl+Z"]
  save: ["Ctrl+S"]
  first_unannotated: ["Ctrl+U"]


# --- Overlay movie rendering -------------------------------------------------
render:
  fps: auto             # auto = the movie's own frame rate
  out_dir: "./renders"
  thickness: 2
  font_scale: 0.4
  show_track_id: true
  show_category: true
  show_skeleton: false
  hide_unlabelled: false
  unlabelled_color: "#555555"
  codec: libx264
  quality: 8
"""


# --------------------------------------------------------------------------- #
# creating and locating
# --------------------------------------------------------------------------- #
def config_path_for(target: str | Path) -> Path:
    """Accept either a project folder or the config file itself."""
    path = Path(target).expanduser()
    return path / CONFIG_NAME if path.is_dir() else path


def create_project(
    directory: str | Path,
    video_dir: str | Path | None = None,
    model: str | Path | None = None,
    work_dir: str | Path | None = None,
    csv: str | Path | None = None,
    categories: list[dict] | None = None,
    overwrite: bool = False,
) -> Path:
    """Write a fresh, fully commented ``config.yaml``. Returns its path."""
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / CONFIG_NAME
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists")

    text = DEFAULT_CONFIG
    updates: dict[str, object] = {}
    if video_dir:
        updates["paths.video_dir"] = _relative_to(Path(video_dir), directory)
    if model:
        updates["paths.model"] = _relative_to(Path(model), directory)
    updates["paths.work_dir"] = _relative_to(Path(work_dir), directory) if work_dir else "./work"
    updates["paths.csv"] = (
        _relative_to(Path(csv), directory) if csv else "./annotations.csv"
    )
    text = set_values(text, updates)
    if categories:
        text = replace_section(
            text, "categories", categories,
            "# --- Annotation categories ---------------------------------------------------\n"
            "# Each group becomes a block of buttons; `key` is its keyboard shortcut.",
        )

    path.write_text(text, encoding="utf-8")
    (directory / "work").mkdir(exist_ok=True)
    remember_project(path)
    return path


def _relative_to(path: Path, root: Path) -> str:
    """Keep a project portable when its pieces live inside it."""
    path = path.expanduser()
    try:
        resolved = path.resolve()
        if resolved.is_relative_to(root):
            return "./" + resolved.relative_to(root).as_posix()
    except (OSError, ValueError):
        pass
    return str(path).replace("\\", "/")


def candidate_config(path: str | Path, updates: dict, categories: list[dict] | None = None,
                     keypoints: dict | None = None) -> str:
    """The text an edit *would* produce, without writing anything."""
    text = Path(path).read_text(encoding="utf-8")
    if updates:
        text = set_values(text, updates)
    if categories is not None:
        text = replace_section(
            text, "categories", categories,
            "# --- Annotation categories ---------------------------------------------------\n"
            "# Each group becomes a block of buttons; `key` is its keyboard shortcut and\n"
            "# `color` is used for the buttons AND the rendered overlay movie.",
        )
    if keypoints is not None:
        text = replace_section(
            text, "keypoints", keypoints,
            "# --- Keypoints (order MUST match the trained model) --------------------------",
        )
    return text


def write_config(path: str | Path, updates: dict, categories: list[dict] | None = None,
                 keypoints: dict | None = None, validate: bool = True) -> Path:
    """Apply edits to an existing config.yaml, comments and all.

    With ``validate`` the candidate is parsed by the real loader *before* it
    replaces anything, so an edit that would not load is refused outright rather
    than written and rolled back. That is what lets the GUI autosave.
    """
    path = Path(path)
    text = candidate_config(path, updates, categories, keypoints)

    # Same folder, so relative paths inside resolve exactly as they will after
    # the rename - validating somewhere else would check a different config.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    try:
        if validate:
            from .config import load_config

            load_config(tmp)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(path)
    return path


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #
@dataclass
class ProjectStatus:
    config_path: Path
    n_videos: int = 0
    n_prepared: int = 0
    n_clips: int = 0
    n_annotated: int = 0
    model_ok: bool = False
    video_dir_ok: bool = False
    suggester: str = ""

    @property
    def can_prepare(self) -> bool:
        return self.video_dir_ok and self.model_ok and self.n_videos > 0

    @property
    def can_annotate(self) -> bool:
        return self.n_clips > 0


def _probe(path, kind: str) -> bool:
    """One raw filesystem question, with "unreachable" folded into "no".

    ``Path.is_dir()`` swallows the ordinary not-found errors but not the ones a
    dropped network share raises, so a bare is_dir() can crash.
    """
    try:
        if kind == "dir":
            return path.is_dir()
        if kind == "file":
            return path.is_file()
        return path.exists()
    except OSError:
        return False


# A path that answered once, and when. On a disconnected share a single is_dir()
# blocks for 123 SECONDS and is_file() for 53 before Windows gives up, and the
# window calls project_status on every refresh, so the verdict is cached and
# _PROBE_TIMEOUT keeps even the first probe off the UI thread.
_PROBE_CACHE: dict = {}
_PROBE_RUNNING: set = set()
_PROBE_LOCK = __import__("threading").Lock()
_PROBE_TTL = 30.0            # seconds a "not found" verdict is trusted for
_PROBE_TIMEOUT = 0.6         # seconds the caller is willing to wait at all
# 0.6s because a local path answers in microseconds and a dead share never
# answers at all; longer only helps a merely-slow share, which the cache picks
# up on the call after next anyway.


def forget_path_probes() -> None:
    """Drop the cache, so a reconnected share is noticed immediately."""
    with _PROBE_LOCK:
        _PROBE_CACHE.clear()


def _exists(path, kind: str = "any", timeout: float = _PROBE_TIMEOUT) -> bool:
    """Does this path exist? Never blocks the caller for more than ``timeout``.

    A local path answers in microseconds; a share that has gone away answers in
    minutes, so it is probed on a throwaway thread and reported absent until the
    real answer lands in the cache.
    """
    import threading
    import time

    key = (str(path), kind)
    now = time.monotonic()
    with _PROBE_LOCK:
        cached = _PROBE_CACHE.get(key)
        # A positive verdict is kept until the TTL as well; re-probing a live
        # share on every repaint is what makes the window feel slow.
        if cached is not None and now - cached[1] < _PROBE_TTL:
            return cached[0]
        in_flight = key in _PROBE_RUNNING
        if not in_flight:
            _PROBE_RUNNING.add(key)

    if in_flight:
        # Another caller is already waiting on this exact question. Starting a
        # second thread would just queue a second multi-minute SMB timeout.
        return False

    result: list = []

    def work():
        answer = _probe(path, kind)
        with _PROBE_LOCK:
            _PROBE_CACHE[key] = (answer, time.monotonic())
            _PROBE_RUNNING.discard(key)
        result.append(answer)

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    worker.join(timeout)
    if result:
        return result[0]

    # Still blocked after the timeout: write a provisional "absent" so the next
    # call answers instantly instead of paying the timeout again. The live thread
    # overwrites it with the truth, and is a daemon so it cannot hold the app open.
    with _PROBE_LOCK:
        _PROBE_CACHE.setdefault(key, (False, time.monotonic()))
    return False


def project_status(cfg) -> ProjectStatus:
    """Summarise a project without opening a movie or a clip cache.

    Directory listings and small JSON reads only: this runs every time the
    workflow window is shown, where one ND2 open on a share is a visible stall.
    """
    status = ProjectStatus(config_path=cfg.path)
    video_dir = cfg.paths.video_dir
    # A path on a disconnected share does not answer "no", it RAISES: Windows
    # reports WinError 64 out of is_dir(), glob() and is_file() alike. Never let
    # that propagate - the clip cache is local, so annotating needs no share.
    status.video_dir_ok = _exists(video_dir, "dir")
    if status.video_dir_ok:
        seen = set()
        try:
            for pattern in cfg.paths.patterns:
                seen.update(p for p in video_dir.glob(pattern) if p.is_file())
        except OSError:
            status.video_dir_ok = False
        status.n_videos = len(seen)
    status.model_ok = _exists(cfg.paths.model, "file")

    work = cfg.paths.work_dir
    if _exists(work, "dir"):
        try:
            movie_dirs = sorted(work.iterdir())
        except OSError:
            movie_dirs = []
        for movie_dir in movie_dirs:
            if not _exists(movie_dir, "dir"):
                continue
            done = movie_dir / "prepare_done.json"
            if not _exists(done):
                continue
            status.n_prepared += 1
            try:
                status.n_clips += int(json.loads(done.read_text(encoding="utf-8"))["n_clips"])
            except (OSError, KeyError, ValueError):
                pass

        report = work / "suggest_report.json"
        if _exists(report):
            try:
                reports = json.loads(report.read_text(encoding="utf-8"))
                live = [g for g, r in reports.items() if r.get("reliable")]
                status.suggester = (
                    "suggesting: " + ", ".join(live) if live
                    else "trained, staying silent"
                )
            except (OSError, ValueError):
                status.suggester = "unreadable report"

    csv = cfg.paths.csv
    if _exists(csv):
        try:
            with open(csv, "r", encoding="utf-8") as fh:
                status.n_annotated = max(0, sum(1 for _ in fh) - 1)
        except OSError:
            pass
    return status


# --------------------------------------------------------------------------- #
# recent list
# --------------------------------------------------------------------------- #
def _state_dir() -> Path:
    base = (
        os.environ.get("TRACKANNO_STATE_DIR")
        or os.environ.get("APPDATA")
        or os.environ.get("XDG_CONFIG_HOME")
        or str(Path.home() / ".config")
    )
    return Path(base) / "trackanno"


def recent_projects() -> list[Path]:
    """Most recently opened first. Paths that have vanished are dropped."""
    path = _state_dir() / RECENT_FILE
    if not path.exists():
        return []
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out, seen = [], set()
    for entry in entries:
        candidate = Path(str(entry))
        if candidate.exists() and candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
    return out[:MAX_RECENT]


def remember_project(config_path: str | Path) -> None:
    path = Path(config_path).resolve()
    entries = [str(path)] + [str(p) for p in recent_projects() if p != path]
    _write_recent(entries[:MAX_RECENT])


def forget_project(config_path: str | Path) -> None:
    path = Path(config_path).resolve()
    _write_recent([str(p) for p in recent_projects() if p != path])


def _write_recent(entries: list[str]) -> None:
    try:
        directory = _state_dir()
        directory.mkdir(parents=True, exist_ok=True)
        (directory / RECENT_FILE).write_text(json.dumps(entries, indent=2), encoding="utf-8")
    except OSError:
        pass  # a missing recent list is never worth an error dialog
