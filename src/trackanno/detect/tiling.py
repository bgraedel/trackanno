"""Tiled YOLO inference.

Movies are much larger than the model's input, so each frame is cut into
overlapping tiles, run as a batch, mapped back to full-frame coordinates and
merged with one class-agnostic NMS. Frames that fit inside a single tile skip
the mechanism entirely.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..config import Config

__all__ = ["Detections", "tile_grid", "TiledDetector", "read_keypoint_count",
           "default_node_names", "resolve_device", "describe_devices", "device_label",
           "describe_auto", "cuda_status", "cuda_devices", "read_model_info"]


def cuda_status() -> tuple[bool, str]:
    """``(usable, why not)`` for CUDA on this machine, with the reason kept.

    ``torch.cuda.is_available()`` answers False for a CPU-only wheel, an absent
    driver, a card asleep behind hybrid graphics and a driver that has fallen
    over alike, and says which only in a warning it raises on the way past. That
    warning is caught here so the reason can be put on screen.
    """
    try:
        import torch
    except ImportError:
        return False, "torch is not installed"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            usable = bool(torch.cuda.is_available())
            count = int(torch.cuda.device_count()) if usable else 0
        except Exception as exc:  # noqa: BLE001 - any failure here means "no GPU"
            return False, f"{type(exc).__name__}: {exc}"
    if usable and count > 0:
        return True, ""
    if usable:
        # The two ask different layers and are allowed to disagree; only one of
        # them reads CUDA_VISIBLE_DEVICES. Nothing can run on a device that
        # cannot be enumerated, so this counts as no GPU.
        return False, ("torch reports CUDA available but can enumerate no device "
                       "(CUDA_VISIBLE_DEVICES is the usual cause)")

    said = "; ".join(
        str(w.message).strip().splitlines()[0] for w in caught if str(w.message).strip()
    )
    if said:
        return False, said
    if not getattr(torch.version, "cuda", None):
        return False, f"torch {torch.__version__} is a CPU-only build"
    # The default check inside torch never raises - a missing or unresponsive
    # driver is simply a device count of zero. So the interesting question is
    # whether the driver disagrees, which is the difference between "no card
    # here" and "the card is there and torch could not reach it".
    cards = _driver_sees()
    if cards:
        return False, (
            f"torch {torch.__version__} could not initialise CUDA, though the "
            f"driver reports {cards}"
        )
    return False, f"no CUDA device is visible to torch {torch.__version__}"


def _driver_sees() -> str:
    """What ``nvidia-smi`` lists, or "" if it is absent or says nothing.

    Only called once torch has already said no, so the happy path spawns
    nothing. Bounded and fully guarded, a diagnosis being no use if it can hang.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:  # noqa: BLE001 - no nvidia-smi, no answer, no problem
        return ""
    names = [line.strip() for line in (out.stdout or "").splitlines() if line.strip()]
    return ", ".join(names)


def cuda_devices() -> list[int]:
    """The CUDA indices this machine can actually be told to use.

    The one place either half of the app asks, so the settings list and
    ``resolve_device`` cannot disagree about what is on offer.
    """
    usable, _why = cuda_status()
    if not usable:
        return []
    try:
        import torch

        return list(range(int(torch.cuda.device_count())))
    except Exception:
        return []


def _mps_available() -> bool:
    try:
        import torch

        return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    except Exception:
        return False


def describe_devices() -> list[tuple[str, str]]:
    """[(value, human label)] for every device this machine can actually use.

    Importing torch takes a second or two, so callers populating a dropdown
    should do this off the UI thread.
    """
    out = [("auto", "auto (pick the best available)")]
    try:
        import torch
    except ImportError:
        return out + [("cpu", "cpu (torch is not installed)")]

    for i in cuda_devices():
        try:
            name = torch.cuda.get_device_name(i)
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            out.append((str(i), f"{i}: {name} ({total:.1f} GB)"))
        except Exception:
            out.append((str(i), f"{i}: CUDA device {i}"))
    if _mps_available():
        out.append(("mps", "mps (Apple GPU)"))
    out.append(("cpu", "cpu"))
    return out


def resolve_device(device: str) -> str:
    """Turn ``auto`` into something ultralytics can use, falling back safely.

    A config that names a GPU is a promise about the machine it was written on;
    on a machine without that device this says so and continues on the CPU
    rather than dying inside ultralytics.
    """
    requested = str(device).strip()
    try:
        import torch  # noqa: F401 - the probes below need it importable
    except ImportError:
        return "cpu"

    # The same list the settings page is shown, so the two cannot disagree.
    cuda = cuda_devices()
    mps = _mps_available()

    if requested in ("", "auto"):
        if cuda:
            return str(cuda[0])
        return "mps" if mps else "cpu"
    if requested == "cpu":
        return "cpu"
    if requested == "mps":
        return "mps" if mps else "cpu"
    if requested.replace("cuda:", "").isdigit():
        index = int(requested.replace("cuda:", ""))
        if index in cuda:
            return str(index)
        print(
            f"[trackanno] detection.device is {requested!r} but no such CUDA device "
            "is available here; using the CPU instead."
        )
        return "cpu"
    return requested


def device_label(device: str) -> str:
    """Name a resolved device the way a log line should read it."""
    value = str(device).strip()
    index = value.replace("cuda:", "")
    if index.isdigit():
        try:
            import torch

            return f"CUDA device {index} ({torch.cuda.get_device_name(int(index))})"
        except Exception:
            return f"CUDA device {index}"
    if value == "mps":
        return "the Apple GPU (mps)"
    if value == "cpu":
        return "the CPU"
    return value


def describe_auto() -> str:
    """One sentence on what ``auto`` picks here, and - if it is the CPU - why.

    No GPU, a sleeping driver and a CPU-only torch wheel need different answers
    from the person, so they are told apart rather than all reading as ``cpu``.
    """
    resolved = resolve_device("auto")
    if resolved != "cpu":
        return f"auto uses {device_label(resolved)} on this machine."
    usable, why = cuda_status()
    if usable:  # a CUDA card exists but auto still declined it - say so plainly
        return "auto uses the CPU here, though CUDA reports itself usable."
    extra = ""
    if "CPU-only build" in why:
        # The fix differs by platform: only Windows ships a CPU-only wheel on
        # PyPI, so only there is an extra the answer. Elsewhere a CPU-only build
        # means `[cpu]` was asked for, and the plain install is the way back.
        extra = (
            ' Reinstall with the GPU extra - uv tool install "trackanno[cuda]" - '
            "if this machine has an NVIDIA card."
            if sys.platform == "win32" else
            " Reinstall without the [cpu] extra if this machine has an NVIDIA card."
        )
    elif "could not initialise" in why or "visible to torch" in why:
        # The two states a click can change: a card asleep behind hybrid
        # graphics, and a driver that has just come back.
        extra = " Re-check once the card is awake."
    return f"auto uses the CPU here: {why}.{extra}"


def default_node_names(n: int) -> list[str]:
    """Placeholder keypoint names for a model that reports ``n`` of them.

    No ultralytics checkpoint stores keypoint *names*, only ``kpt_shape``. Order
    is what has to match the model, so generated names are correct by
    construction and can be renamed for readability afterwards.
    """
    return [f"kp{i}" for i in range(int(n))]


def read_model_info(model_path) -> dict:
    """What a checkpoint says about itself: task, keypoint count, class names.

    Read from the file rather than through ``YOLO(...)`` so the GUI can answer
    "what is this model?" the moment you pick one, without building a detector
    or touching the GPU.
    """
    import torch

    try:
        checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"could not read {model_path}: {exc}") from exc

    if isinstance(checkpoint, dict):
        # EMA first, then "model": a checkpoint written mid-training keeps the
        # averaged weights under "ema" and leaves "model" set to None, so
        # reading "model" alone finds nothing on a user's own best.pt.
        # Ultralytics resolves it the same way round.
        model = checkpoint.get("ema") or checkpoint.get("model")
        args = checkpoint.get("train_args") or {}
    else:
        model, args = checkpoint, {}

    layers = getattr(model, "model", None)
    try:
        head = layers[-1] if layers is not None and len(layers) else None
    except TypeError:
        head = None

    # kpt_shape is (n_keypoints, n_dims). It lives on the model, in the yaml it
    # was built from, or - on a checkpoint carrying neither - on the head module.
    shape = getattr(model, "kpt_shape", None)
    if shape is None:
        shape = (getattr(model, "yaml", None) or {}).get("kpt_shape")
    if shape is None:
        shape = getattr(head, "kpt_shape", None)
    n_keypoints = None
    if shape is not None:
        try:
            n_keypoints = int(shape[0])
        except (TypeError, IndexError, ValueError):
            n_keypoints = None

    task = args.get("task") or getattr(model, "task", None)
    if not task and head is not None:
        # Same rule ultralytics uses: the head module names the task. A segment
        # or classify checkpoint reported as a detector would fail much later
        # and much less clearly.
        kind = type(head).__name__.lower()
        for candidate in ("pose", "segment", "classify", "obb", "detect"):
            if candidate in kind:
                task = candidate
                break
    if not task:
        task = "pose" if n_keypoints else None

    names = getattr(model, "names", None)
    return {
        "task": str(task) if task else None,
        "n_keypoints": n_keypoints,
        "classes": list(names.values()) if isinstance(names, dict) else None,
        "imgsz": args.get("imgsz"),
        "name": Path(str(model_path)).name,
    }


def read_keypoint_count(model_path) -> int | None:
    """How many keypoints a checkpoint predicts, or None if it is not a pose model."""
    return read_model_info(model_path)["n_keypoints"]


_PRECISION_KWARG: dict | None = None


def _precision_kwarg() -> dict:
    """FP16 inference, spelled the way the installed ultralytics wants it.

    `quantize=16` where it exists, `half=True` otherwise. Resolved once: this is
    called per frame.
    """
    global _PRECISION_KWARG
    if _PRECISION_KWARG is None:
        try:
            from ultralytics.cfg import DEFAULT_CFG_DICT

            modern = "quantize" in DEFAULT_CFG_DICT
        except Exception:
            modern = False
        _PRECISION_KWARG = {"quantize": 16} if modern else {"half": True}
    return dict(_PRECISION_KWARG)


@dataclass
class Detections:
    """Full-frame detections for one frame."""

    xyxy: np.ndarray          # (N, 4) float32, original pixel coords
    conf: np.ndarray          # (N,)  float32
    cls: np.ndarray           # (N,)  int32
    kpts: np.ndarray | None   # (N, K, 3) float32 (x, y, conf) or None
    feats: np.ndarray | None = None   # (N, D) ROI-pooled CNN features

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])

    def take(self, idx: np.ndarray) -> "Detections":
        return Detections(
            xyxy=self.xyxy[idx],
            conf=self.conf[idx],
            cls=self.cls[idx],
            kpts=None if self.kpts is None else self.kpts[idx],
            feats=None if self.feats is None else self.feats[idx],
        )

    @classmethod
    def empty(cls, n_kpts: int = 0) -> "Detections":
        return cls(
            xyxy=np.zeros((0, 4), np.float32),
            conf=np.zeros((0,), np.float32),
            cls=np.zeros((0,), np.int32),
            kpts=np.zeros((0, n_kpts, 3), np.float32) if n_kpts else None,
            feats=None,
        )

    @classmethod
    def concat(cls, parts: Sequence["Detections"]) -> "Detections":
        parts = [p for p in parts if len(p)]
        if not parts:
            return cls.empty()
        kpts = None
        if all(p.kpts is not None for p in parts):
            kpts = np.concatenate([p.kpts for p in parts], axis=0)
        # Features are dropped unless every part has them: a partial failure must
        # never silently misalign rows against boxes.
        feats = None
        if all(p.feats is not None for p in parts):
            feats = np.concatenate([p.feats for p in parts], axis=0)
        return cls(
            xyxy=np.concatenate([p.xyxy for p in parts], axis=0),
            conf=np.concatenate([p.conf for p in parts], axis=0),
            cls=np.concatenate([p.cls for p in parts], axis=0),
            kpts=kpts,
            feats=feats,
        )


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def tile_grid(height: int, width: int, tile: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """Overlapping (x0, y0, x1, y1) tiles covering the frame.

    Tiles are ``tile`` px square where possible; the last row/column is snapped
    back inside the frame instead of padded, so every tile holds real pixels.
    """
    if tile <= 0:
        raise ValueError("tiling.tile must be positive")
    step = max(1, int(round(tile * (1.0 - overlap))))

    def starts(extent: int) -> list[int]:
        if extent <= tile:
            return [0]
        out = list(range(0, extent - tile + 1, step))
        if out[-1] != extent - tile:
            out.append(extent - tile)
        return out

    tw, th = min(tile, width), min(tile, height)
    return [
        (x0, y0, x0 + tw, y0 + th)
        for y0 in starts(height)
        for x0 in starts(width)
    ]


def _nms(xyxy: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    """Class-agnostic NMS. Uses torchvision when present, else a numpy fallback."""
    if xyxy.shape[0] == 0:
        return np.zeros((0,), np.int64)
    try:
        import torch
        from torchvision.ops import nms as tv_nms

        keep = tv_nms(
            torch.as_tensor(xyxy, dtype=torch.float32),
            torch.as_tensor(scores, dtype=torch.float32),
            float(iou_thresh),
        )
        return keep.cpu().numpy().astype(np.int64)
    except Exception:
        pass

    order = scores.argsort()[::-1]
    x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        order = rest[iou <= iou_thresh]
    return np.asarray(keep, dtype=np.int64)


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), np.float32)
    area_a = np.maximum(0, a[:, 2] - a[:, 0]) * np.maximum(0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0, b[:, 2] - b[:, 0]) * np.maximum(0, b[:, 3] - b[:, 1])
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    return (inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)).astype(np.float32)


# --------------------------------------------------------------------------- #
# detector
# --------------------------------------------------------------------------- #
class TiledDetector:
    """Runs a YOLO model over a frame, tiling it when it does not fit."""

    def __init__(self, cfg: Config, model=None, reporter=None):
        from ultralytics import YOLO

        self.cfg = cfg
        # Resolved once, here, so every predict call and every log line agree,
        # and logged every time: "auto" on its own never tells you whether a slow
        # run is on the GPU or the CPU.
        self.device = resolve_device(cfg.detection.device)
        wanted = str(cfg.detection.device).strip()
        told = f"detection.device {wanted!r} -> {device_label(self.device)}"
        level = "info"
        # Asking for a GPU and silently getting the CPU is the failure that
        # looks like nothing at all - the run just takes twenty times as long.
        if self.device == "cpu" and wanted not in ("cpu",):
            _usable, why = cuda_status()
            told += f" - NOT the GPU: {why}"
            level = "warning"
        if reporter is not None:
            reporter.log(told, level)
        else:
            print(f"[trackanno] {told}")
        self.model = model if model is not None else YOLO(str(cfg.paths.model))
        if cfg.is_pose and getattr(self.model, "task", None) != "pose":
            raise ValueError(
                f"detection.task is 'pose' but the model at {cfg.paths.model} has "
                f"task={getattr(self.model, 'task', None)!r}"
            )
        model_kpts = getattr(getattr(self.model, "model", None), "kpt_shape", None)
        model_kpts = int(model_kpts[0]) if model_kpts else None

        if cfg.needs_keypoints:
            # The checkpoint is the authority on the count, and it is already
            # loaded here, so resolve rather than refuse.
            if not model_kpts:
                raise ValueError(
                    f"detection.task is 'pose' but {cfg.paths.model} does not report a "
                    "keypoint shape, so it is not a pose model. Set detection.task to "
                    "'detect', or point paths.model at pose weights."
                )
            cfg.keypoints.nodes = default_node_names(model_kpts)
            print(
                f"[trackanno] {cfg.paths.model.name} predicts {model_kpts} keypoints; "
                f"using placeholder names {cfg.keypoints.nodes[0]}..{cfg.keypoints.nodes[-1]}. "
                "Rename them under keypoints.nodes if you want the skeleton labelled."
            )
        elif cfg.is_pose and model_kpts and model_kpts != cfg.n_keypoints:
            raise ValueError(
                f"model has {model_kpts} keypoints but keypoints.nodes lists "
                f"{cfg.n_keypoints}. Fix keypoints.nodes (order matters)."
            )
        self.n_kpts = cfg.n_keypoints
        self._grid_cache: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}

        # Per-detection CNN features come out of the same forward pass; the hook
        # is installed once here and lives for the detector's lifetime.
        self.extractor = None
        if getattr(cfg, "features", None) is not None and cfg.features.enabled:
            from .features import InstanceFeatures

            self.extractor = InstanceFeatures(
                self.model, levels=cfg.features.levels,
                pool=cfg.features.pool, dtype=cfg.features.dtype,
            )

    # -- grid --------------------------------------------------------------- #
    def tiles_for(self, height: int, width: int) -> list[tuple[int, int, int, int]]:
        key = (height, width)
        if key not in self._grid_cache:
            t = self.cfg.tiling
            enabled = t.enabled
            if enabled == "auto":
                enabled = height > t.tile or width > t.tile
            self._grid_cache[key] = (
                tile_grid(height, width, t.tile, t.overlap)
                if enabled
                else [(0, 0, width, height)]
            )
        return self._grid_cache[key]

    def is_tiled(self, height: int, width: int) -> bool:
        return len(self.tiles_for(height, width)) > 1

    # -- inference ---------------------------------------------------------- #
    def _predict_kwargs(self, imgsz: int) -> dict:
        d = self.cfg.detection
        kwargs = dict(
            imgsz=imgsz, conf=d.conf, iou=d.iou, max_det=d.max_det,
            device=self.device, verbose=False,
        )
        if d.half:
            # ultralytics replaced `half` with a unified `quantize` scheme and
            # logs a deprecation notice for every call that still uses the old
            # name - at one call per frame that buries everything else. Send
            # whichever the installed version actually understands.
            kwargs.update(_precision_kwarg())
        return kwargs

    def _predict(self, crops: list[np.ndarray], imgsz: int) -> list:
        kwargs = self._predict_kwargs(imgsz)
        out = []
        for i in range(0, len(crops), self.cfg.detection.batch):
            out.extend(self.model.predict(crops[i : i + self.cfg.detection.batch], **kwargs))
        return out

    def _predict_with_features(self, crops: list[np.ndarray], imgsz: int) -> tuple[list, list]:
        """Predict, capturing per-detection features from the same forward pass.

        Features must be pooled while the hook still holds *this* sub-batch's
        maps, so extraction happens inside the batching loop rather than after.
        """
        kwargs = self._predict_kwargs(imgsz)
        batch = self.cfg.detection.batch
        results, feats = [], []
        for i in range(0, len(crops), batch):
            chunk = crops[i : i + batch]
            # Drop the previous chunk's maps BEFORE predicting: a forward pass
            # that never reaches the head would otherwise leave them in place and
            # pooled() would return the PREVIOUS tile's features, correctly
            # shaped and correctly row-counted.
            self.extractor.reset()
            chunk_results = self.model.predict(chunk, **kwargs)
            results.extend(chunk_results)
            for j, r in enumerate(chunk_results):
                boxes = getattr(r, "boxes", None)
                if boxes is None or len(boxes) == 0:
                    feats.append(None)
                    continue
                # Tile-local coordinates: the hook's maps are for this tile.
                feats.append(
                    self.extractor.pooled(
                        boxes.xyxy.cpu().numpy().astype(np.float32),
                        batch_index=j,
                        tile_size=chunk[j].shape[:2],
                        imgsz=imgsz,
                        stride=int(self.model.model.stride.max()),
                    )
                )
        return results, feats

    def _result_to_dets(self, r, x0: int, y0: int, feats: np.ndarray | None = None) -> Detections:
        boxes = getattr(r, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return Detections.empty(self.n_kpts)
        xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
        conf = boxes.conf.cpu().numpy().astype(np.float32)
        cls = boxes.cls.cpu().numpy().astype(np.int32)
        xyxy[:, [0, 2]] += x0
        xyxy[:, [1, 3]] += y0

        kpts = None
        if self.n_kpts and getattr(r, "keypoints", None) is not None:
            data = r.keypoints.data.cpu().numpy().astype(np.float32)  # (N, K, 3)
            if data.size:
                data = data.copy()
                # (0, 0) is ultralytics' "not visible" marker; keep it out of the offset.
                missing = (data[..., 0] == 0) & (data[..., 1] == 0)
                data[..., 0] += x0
                data[..., 1] += y0
                data[missing] = np.nan
                data[..., 2] = np.where(missing, 0.0, data[..., 2])
                kpts = data
        if feats is not None and feats.shape[0] != xyxy.shape[0]:
            feats = None  # never risk misaligning features against boxes
        return Detections(xyxy=xyxy, conf=conf, cls=cls, kpts=kpts, feats=feats)

    def _drop_edge_truncated(
        self, dets: Detections, tile: tuple[int, int, int, int], height: int, width: int
    ) -> tuple[Detections, Detections]:
        """Split into (kept, suspect) where suspect boxes are clipped by an interior tile border.

        An object straddling a tile border is detected twice: truncated in this
        tile, complete in the overlapping neighbour. Dropping the truncated copy
        avoids a merged NMS having to choose between two poor boxes. Boxes that
        touch a *frame* border are genuinely at the edge and are kept.
        """
        if len(dets) == 0:
            return dets, Detections.empty(self.n_kpts)
        x0, y0, x1, y1 = tile
        m = self.cfg.tiling.edge_margin_px
        b = dets.xyxy
        touches = np.zeros(len(dets), bool)
        if x0 > 0:
            touches |= b[:, 0] <= x0 + m
        if y0 > 0:
            touches |= b[:, 1] <= y0 + m
        if x1 < width:
            touches |= b[:, 2] >= x1 - m
        if y1 < height:
            touches |= b[:, 3] >= y1 - m
        return dets.take(~touches), dets.take(touches)

    def detect(self, frame_rgb: np.ndarray) -> Detections:
        """Detections for one full RGB uint8 frame, in original pixel coordinates."""
        height, width = frame_rgb.shape[:2]
        tiles = self.tiles_for(height, width)
        tiled = len(tiles) > 1

        crops = [np.ascontiguousarray(frame_rgb[y0:y1, x0:x1]) for x0, y0, x1, y1 in tiles]
        # Always run at detection.imgsz, tiled or not. `tiling.tile` says how the
        # frame is CUT; `detection.imgsz` says at what resolution each piece is
        # SEEN, so the tiled and untiled paths show the model the same scale.
        imgsz = self.cfg.detection.imgsz
        if self.extractor is not None:
            results, feat_blocks = self._predict_with_features(crops, imgsz)
        else:
            results = self._predict(crops, imgsz)
            feat_blocks = [None] * len(results)

        kept_parts, suspect_parts = [], []
        for r, tile, feats in zip(results, tiles, feat_blocks):
            dets = self._result_to_dets(r, tile[0], tile[1], feats)
            if tiled and self.cfg.tiling.drop_edge_truncated:
                keep, suspect = self._drop_edge_truncated(dets, tile, height, width)
                kept_parts.append(keep)
                suspect_parts.append(suspect)
            else:
                kept_parts.append(dets)

        merged = Detections.concat(kept_parts)

        # Safety net: an object bigger than the tile overlap is truncated in
        # *every* tile that sees it and would otherwise vanish. Re-admit any
        # suspect box that no surviving detection covers.
        if suspect_parts:
            suspect = Detections.concat(suspect_parts)
            if len(suspect):
                if len(merged):
                    unmatched = _iou_matrix(suspect.xyxy, merged.xyxy).max(axis=1) < 0.3
                else:
                    unmatched = np.ones(len(suspect), bool)
                if unmatched.any():
                    merged = Detections.concat([merged, suspect.take(unmatched)])

        if not tiled or len(merged) == 0:
            return merged
        keep = _nms(merged.xyxy, merged.conf, self.cfg.tiling.merge_iou)
        merged = merged.take(keep)
        # Highest-confidence first keeps downstream behaviour deterministic.
        return merged.take(np.argsort(-merged.conf, kind="stable"))

    def detect_untiled(self, frame_rgb: np.ndarray) -> Detections:
        """Single-shot whole-frame inference. Used by ``prepare.py --self-test``."""
        r = self.model.predict(
            np.ascontiguousarray(frame_rgb), **self._predict_kwargs(self.cfg.detection.imgsz)
        )[0]
        return self._result_to_dets(r, 0, 0)
