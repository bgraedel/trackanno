"""Tracking of merged full-frame detections: ultralytics trackers, or greedy nearest-centroid."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from ..config import FIXED_SETTINGS, Config

# Distance gate of the greedy fallback tracker, in pixels.
GREEDY_MAX_DIST_PX = FIXED_SETTINGS["tracking.greedy_max_dist_px"]
from .tiling import Detections

__all__ = ["Tracker", "build_tracker"]


def _as_boxes(dets: Detections, shape: tuple[int, int]):
    """Wrap detections in ultralytics' ``Boxes``, which its trackers slice by mask."""
    from ultralytics.engine.results import Boxes

    data = np.concatenate(
        [dets.xyxy, dets.conf[:, None], dets.cls[:, None].astype(np.float32)], axis=1
    ).astype(np.float32)
    return Boxes(data, orig_shape=shape)


class Tracker:
    """Common interface: ``update(dets)`` -> (track_ids, index into ``dets``)."""

    def reset(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def update(self, dets: Detections) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# ByteTrack
# --------------------------------------------------------------------------- #
_BYTETRACK_DEFAULTS = {
    "tracker_type": "bytetrack",
    "track_high_thresh": 0.25,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.25,
    "track_buffer": 30,
    "match_thresh": 0.8,
    "fuse_score": True,
}

# name -> (ultralytics class, bundled config).
TRACKERS = {
    "bytetrack": ("BYTETracker", "bytetrack.yaml"),
    "tracktrack": ("TRACKTRACK", "tracktrack.yaml"),
    "botsort": ("BOTSORT", "botsort.yaml"),
    "ocsort": ("OCSORT", "ocsort.yaml"),
    "deepocsort": ("DeepOCSORT", "deepocsort.yaml"),
    "fasttrack": ("FASTTracker", "fasttrack.yaml"),
}


def _load_tracker_yaml(name: str) -> dict:
    """Read a tracker YAML by path, or by name from ultralytics' bundled configs."""
    import yaml

    path = Path(name)
    if not path.exists():
        import ultralytics

        path = Path(ultralytics.__file__).parent / "cfg" / "trackers" / name
    if not path.exists():
        raise FileNotFoundError(f"tracker config {name!r} not found (looked at {path})")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _tracker_args(cfg: Config, fps: float):
    from types import SimpleNamespace

    backend = cfg.tracking.backend
    wanted = TRACKERS.get(backend, ("", "bytetrack.yaml"))[1]

    params: dict = {}
    try:
        params.update({k: v for k, v in _load_tracker_yaml(wanted).items() if v is not None})
    except Exception as exc:
        if backend != "bytetrack":
            raise RuntimeError(
                f"could not load tracker config {wanted!r} for backend {backend!r}: {exc}"
            ) from exc
        warnings.warn(
            f"could not load {wanted!r} ({exc}); using built-in ByteTrack defaults",
            stacklevel=2,
        )
        params = dict(_BYTETRACK_DEFAULTS)

    # Bundled yamls assume 30 fps; scale track_buffer so it keeps the same duration.
    if "track_buffer" in params:
        params["track_buffer"] = int(round(params["track_buffer"] * max(1.0, fps) / 30.0))

    params.update(cfg.tracking.overrides)
    return SimpleNamespace(**params)


class UltralyticsTracker(Tracker):
    """Drives any bundled ultralytics tracker on our merged full-frame boxes."""

    def __init__(self, cfg: Config, fps: float, shape: tuple[int, int] = (0, 0)):
        import ultralytics.trackers as ut

        backend = cfg.tracking.backend
        if backend not in TRACKERS:
            raise ValueError(
                f"unknown tracking.backend {backend!r}; available: "
                f"{sorted(TRACKERS)} + 'greedy'"
            )
        class_name = TRACKERS[backend][0]
        self._cls = getattr(ut, class_name, None)
        if self._cls is None:
            raise RuntimeError(
                f"installed ultralytics has no {class_name}; "
                f"{backend!r} needs a newer version"
            )
        self._args = _tracker_args(cfg, fps)
        self._fps = int(round(max(1.0, fps)))
        self.shape = shape
        self._impl = self._make()

        # Probe with one detection so an API change fails here, not mid-run.
        probe = Detections(
            xyxy=np.array([[10, 10, 30, 30]], np.float32),
            conf=np.array([0.9], np.float32),
            cls=np.zeros(1, np.int32),
            kpts=None,
        )
        ids, idx = self.update(probe)
        if len(ids) != 1 or len(idx) != 1:
            raise RuntimeError(
                f"{class_name} returned {len(ids)} tracks for a single detection; "
                "the ultralytics tracker API has changed"
            )
        self.reset()

    def reset(self) -> None:
        self._impl = self._make()
        # Track ids are a process-wide counter used in clip_ids; restart it per movie.
        reset_id = getattr(self._impl, "reset_id", None)
        if callable(reset_id):
            reset_id()
        try:
            from ultralytics.trackers.basetrack import BaseTrack
        except ImportError:
            return
        BaseTrack.reset_id()

    def _make(self):
        try:
            return self._cls(self._args, frame_rate=self._fps)   # ultralytics <= 8.3
        except TypeError:
            return self._cls(self._args)                          # ultralytics >= 8.4

    def update(self, dets: Detections, img: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        empty = (np.zeros((0,), np.int64), np.zeros((0,), np.int64))
        boxes = _as_boxes(dets, self.shape)
        # Update on empty frames too, so lost-track ages advance.
        try:
            out = np.asarray(self._impl.update(boxes, img))
        except TypeError:
            out = np.asarray(self._impl.update(boxes))
        if out.size == 0:
            return empty

        ids = out[:, 4].astype(np.int64)
        if out.shape[1] >= 8:
            # [x1, y1, x2, y2, id, score, cls, det_idx]
            det_idx = np.clip(out[:, 7].astype(np.int64), 0, max(len(dets) - 1, 0))
        else:
            # No index column: recover it by matching boxes.
            from .tiling import _iou_matrix

            det_idx = _iou_matrix(out[:, :4].astype(np.float32), dets.xyxy).argmax(axis=1)
        return ids, det_idx


# --------------------------------------------------------------------------- #
# greedy fallback
# --------------------------------------------------------------------------- #
class GreedyTracker(Tracker):
    """Nearest-centroid matching with a hard distance gate."""

    def __init__(self, cfg: Config, fps: float):
        self.max_dist = float(GREEDY_MAX_DIST_PX)
        self.max_age = max(1, int(round(fps / 30.0 * 30)))
        self.reset()

    def reset(self) -> None:
        self._next_id = 1
        self._centres: dict[int, np.ndarray] = {}
        self._age: dict[int, int] = {}

    def update(self, dets: Detections, img: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        for tid in list(self._age):
            self._age[tid] += 1
            if self._age[tid] > self.max_age:
                self._age.pop(tid, None)
                self._centres.pop(tid, None)

        if len(dets) == 0:
            return np.zeros((0,), np.int64), np.zeros((0,), np.int64)

        centres = np.stack(
            [(dets.xyxy[:, 0] + dets.xyxy[:, 2]) * 0.5, (dets.xyxy[:, 1] + dets.xyxy[:, 3]) * 0.5],
            axis=1,
        )
        live = list(self._centres)
        assigned_det: set[int] = set()
        assigned_trk: set[int] = set()
        ids = np.zeros(len(dets), np.int64)

        if live:
            prev = np.stack([self._centres[t] for t in live], axis=0)
            dist = np.linalg.norm(centres[:, None, :] - prev[None, :, :], axis=2)
            # Greedy over ascending distance: cheap and stable at ~1 px/frame motion.
            for flat in np.argsort(dist, axis=None):
                di, ti = np.unravel_index(flat, dist.shape)
                if dist[di, ti] > self.max_dist:
                    break
                if di in assigned_det or ti in assigned_trk:
                    continue
                tid = live[ti]
                ids[di] = tid
                self._centres[tid] = centres[di]
                self._age[tid] = 0
                assigned_det.add(int(di))
                assigned_trk.add(int(ti))

        for di in range(len(dets)):
            if di in assigned_det:
                continue
            tid = self._next_id
            self._next_id += 1
            ids[di] = tid
            self._centres[tid] = centres[di]
            self._age[tid] = 0

        return ids, np.arange(len(dets), dtype=np.int64)


def build_tracker(cfg: Config, fps: float, shape: tuple[int, int] = (0, 0)) -> Tracker:
    if cfg.tracking.backend == "greedy":
        return GreedyTracker(cfg, fps)
    try:
        return UltralyticsTracker(cfg, fps, shape)
    except Exception as exc:
        raise RuntimeError(
            f"tracker {cfg.tracking.backend!r} could not be initialised "
            f"({type(exc).__name__}: {exc}); try tracking.backend: greedy"
        ) from exc
