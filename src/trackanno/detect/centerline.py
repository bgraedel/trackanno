"""Centerline refinement: the flagellum from a centerline checkpoint in place of the pose chain.

Runs on the tracks after tracking. Each tracked box gets a frame-stack crop, body and flagellum
hint boxes from a 3-class detector, and the model's own curves from earlier frames of the same
track. Node 0 keeps the pose model's Head; nodes 1.. are the model's points from Base to Tip, with
the visibility head as their confidence.

The model code is the ``centerline`` package, installed or found at ``centerline.code``.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..progress import NullReporter, Reporter
from .tiling import Detections, merge_tiles, tile_grid

__all__ = ["CenterlineModel", "refine_tracks", "match_cells", "assign_parts", "node_names",
           "HEAD"]

# The pose node kept as node 0 of the output.
HEAD = 0
# A tracked box takes the hint cell it overlaps by at least this IoU.
MATCH_IOU = 0.3
# An animal box this far inside a stronger one is the body seen twice.
DEDUP_CONTAIN = 0.9

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
GREY_MEAN, GREY_STD = float(IMAGENET_MEAN.mean()), float(IMAGENET_STD.mean())


def _resolve(cfg, value: str) -> Path:
    p = Path(str(value).replace("\\", "/"))
    return p if p.is_absolute() or str(p).startswith("//") else (cfg.path.parent / p).resolve()


def _import_centerline(code: Path | None):
    """The ``centerline`` modules this needs, from the environment or from ``code``."""
    try:
        importlib.import_module("centerline.model")
    except ImportError as first:
        if code is None or not code.is_dir():
            raise ImportError(
                f"the centerline package is not importable ({first}); install it next to "
                "trackanno or set centerline.code to its repository"
            ) from first
        sys.path.insert(0, str(code))
        for name in [m for m in sys.modules if m == "centerline" or m.startswith("centerline.")]:
            del sys.modules[name]
        try:
            importlib.import_module("centerline.model")
        except ImportError as exc:
            hint = " (pip install timm, or trackanno[centerline])" if "timm" in str(exc) else ""
            raise ImportError(f"could not import centerline from {code}: {exc}{hint}") from exc
    crop = importlib.import_module("centerline.crop_transform")
    model = importlib.import_module("centerline.model")
    return crop, model.CenterlineNet


def _torch_device(device: str) -> str:
    from .tiling import resolve_device

    resolved = resolve_device(device)
    return f"cuda:{resolved}" if resolved.isdigit() else resolved


# --------------------------------------------------------------------------- #
# hint boxes
# --------------------------------------------------------------------------- #
def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _dedup(xyxy: np.ndarray, cls: np.ndarray, conf: np.ndarray, animal: int) -> np.ndarray:
    """Indices to keep: an animal box mostly inside a stronger animal box is dropped."""
    keep = np.ones(len(cls), bool)
    order = [i for i in np.argsort(-conf) if cls[i] == animal]
    for n, i in enumerate(order):
        a = xyxy[i]
        area = max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)
        for j in order[:n]:
            if not keep[j]:
                continue
            b = xyxy[j]
            ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            if ix * iy / area >= DEDUP_CONTAIN:
                keep[i] = False
                break
    return np.flatnonzero(keep)


def assign_parts(xyxy: np.ndarray, cls: np.ndarray, conf: np.ndarray, names: dict) -> list[dict]:
    """Animal boxes, each with the body and flagellum box whose centre lies inside it.

    A part inside several animals goes to the one holding most of its area; per animal the
    most confident part wins."""
    ids = {v: k for k, v in names.items()}
    cells = [{"box": xyxy[i].tolist(), "conf": float(conf[i]), "body": None, "flag": None,
              "body_conf": 0.0, "flag_conf": 0.0}
             for i in range(len(cls)) if cls[i] == ids["animal"]]
    for kind, key in (("body", "body"), ("flagellum", "flag")):
        for j in np.flatnonzero(cls == ids[kind]):
            p = xyxy[j]
            cx, cy = 0.5 * (p[0] + p[2]), 0.5 * (p[1] + p[3])
            area = max((p[2] - p[0]) * (p[3] - p[1]), 1e-6)
            best, best_ov = None, 0.0
            for c in cells:
                b = c["box"]
                if not (b[0] <= cx <= b[2] and b[1] <= cy <= b[3]):
                    continue
                ix = max(0.0, min(b[2], p[2]) - max(b[0], p[0]))
                iy = max(0.0, min(b[3], p[3]) - max(b[1], p[1]))
                if ix * iy / area > best_ov:
                    best, best_ov = c, ix * iy / area
            if best is not None and float(conf[j]) > best[f"{key}_conf"]:
                best[key], best[f"{key}_conf"] = p.tolist(), float(conf[j])
    return cells


def match_cells(boxes: np.ndarray, cells: list[dict]) -> list[int | None]:
    """For each tracked box, the index of its hint cell (one-to-one by IoU), or None."""
    if not len(boxes) or not cells:
        return [None] * len(boxes)
    from scipy.optimize import linear_sum_assignment

    iou = np.array([[_iou(b, c["box"]) for c in cells] for b in boxes])
    out: list[int | None] = [None] * len(boxes)
    for r, c in zip(*linear_sum_assignment(-iou)):
        if iou[r, c] >= MATCH_IOU:
            out[int(r)] = int(c)
    return out


def node_names(n_points: int) -> list[str]:
    """Head, then ``n_points`` along the flagellum: Base, C1.., Tip."""
    return ["Head", "Base"] + [f"C{i}" for i in range(1, n_points - 1)] + ["Tip"]


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #
@dataclass
class _Cell:
    track_id: int
    box: list
    body: list | None
    flag: list | None


class CenterlineModel:
    """A centerline checkpoint, its 3-class hint detector, and the crop recipe it was trained on."""

    def __init__(self, cfg, reporter: Reporter | None = None):
        import torch
        from ultralytics import YOLO

        self.reporter = reporter or NullReporter()
        section = cfg.centerline
        code = _resolve(cfg, section.code) if section.code else None
        self.crop, net_cls = _import_centerline(code)

        ckpt_path = _resolve(cfg, section.checkpoint)
        hint_path = _resolve(cfg, section.hint_model)
        for what, path in (("centerline.checkpoint", ckpt_path),
                           ("centerline.hint_model", hint_path)):
            if not path.is_file():
                raise FileNotFoundError(f"{what} not found: {path}")

        self.device = torch.device(_torch_device(cfg.detection.device))
        # A Lightning checkpoint is a pickle: trusted like the YOLO weights.
        ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        hp = ck["hyper_parameters"]
        net = net_cls.from_config({**hp["model"], "pretrained": False}, hp["crop"])
        net.load_state_dict({k[4:]: v for k, v in ck["state_dict"].items()
                             if k.startswith("net.")}, strict=True)
        self.net = net.eval().to(self.device)
        self.epoch = int(ck.get("epoch", -1))
        self.name = ckpt_path.name

        m, c = hp["model"], hp["crop"]
        self.W, self.H = int(c["width"]), int(c["height"])
        self.pad = float(c.get("box_pad", 1.15))
        self.hint = str(c.get("target_hint", "none"))
        self.edge_extend = float(c.get("edge_extend", 0.0))
        self.K = int(m["n_ctrl"])
        self.F = int(m.get("in_frames", 1))
        half = self.F // 2
        self.offsets = [int(v) for v in (m.get("frame_offsets") or range(-half, half + 1))]
        self.half = self.offsets.index(0)
        prev = bool(m.get("prev_channel", False))
        lags = sorted(int(v) for v in (m.get("prev_lags") or [])) if prev else []
        if prev and not lags:
            lags = [int(((hp.get("data") or {}).get("aug") or {}).get("prev_ctrl_lag", 2))]
        self.lags = lags

        self.hinter = YOLO(str(hint_path))
        missing = {"animal", "body", "flagellum"} - set(self.hinter.names.values())
        if missing:
            raise ValueError(f"centerline.hint_model {hint_path.name} has no "
                             f"{', '.join(sorted(missing))} class; it needs animal, body "
                             "and flagellum")
        self.hint_conf = float(section.hint_conf)
        self.hint_imgsz = int(section.hint_imgsz)
        self.hint_tile = int(section.hint_tile)
        self.hint_overlap = float(cfg.tiling.overlap)
        self.hint_merge_iou = float(cfg.tiling.merge_iou)
        self.batch = int(section.batch)
        self.reporter.log(
            f"centerline {self.name} (epoch {self.epoch}): {self.F} planes {self.offsets}, "
            f"history lags {self.lags or 'none'}, hint {self.hint}, crop {self.W}x{self.H}, "
            f"hint detector {f'{self.hint_tile} px tiles' if self.hint_tile else 'whole frame'}"
            f" at {self.hint_imgsz}, on {self.device}")

    @property
    def reach(self) -> int:
        """How many frames back and forth one prediction reads, in stack strides."""
        return max(max(abs(o) for o in self.offsets), max(self.lags, default=0))

    def hint_tiles(self, height: int, width: int) -> list[tuple[int, int, int, int]]:
        """The hint detector's tiles: native ``hint_tile`` squares, as the detect3 weights were
        trained on, or the whole frame when it fits in one or ``hint_tile`` is 0."""
        if self.hint_tile <= 0 or (height <= self.hint_tile and width <= self.hint_tile):
            return [(0, 0, width, height)]
        return tile_grid(height, width, self.hint_tile, self.hint_overlap)

    def hint_cells(self, rgb: np.ndarray) -> list[dict]:
        height, width = rgb.shape[:2]
        tiles = self.hint_tiles(height, width)
        crops = [np.ascontiguousarray(rgb[y0:y1, x0:x1]) for x0, y0, x1, y1 in tiles]
        device = str(self.device).replace("cuda:", "")
        parts = []
        for i in range(0, len(crops), self.batch):
            results = self.hinter.predict(crops[i:i + self.batch], imgsz=self.hint_imgsz,
                                          conf=self.hint_conf, device=device, verbose=False)
            for r, (x0, y0, _, _) in zip(results, tiles[i:i + self.batch]):
                xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)
                xyxy[:, [0, 2]] += x0
                xyxy[:, [1, 3]] += y0
                parts.append(Detections(xyxy=xyxy, conf=r.boxes.conf.cpu().numpy(),
                                        cls=r.boxes.cls.cpu().numpy().astype(np.int32),
                                        kpts=None))
        # Per class: an animal and its own body overlap, and must not suppress each other.
        dets = merge_tiles(parts, tiles, height, width, self.hint_merge_iou, per_class=True)
        xyxy, cls, conf = dets.xyxy.astype(float), dets.cls.astype(int), dets.conf.astype(float)
        animal = {v: k for k, v in self.hinter.names.items()}["animal"]
        keep = _dedup(xyxy, cls, conf, animal)
        return assign_parts(xyxy[keep], cls[keep], conf[keep], self.hinter.names)

    def _box_n(self, box, M) -> list[float]:
        ct = self.crop
        x0, y0, x1, y1 = (float(v) for v in box)
        corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        c = ct.normalize_points(ct.transform_points(corners, M), self.W, self.H)
        return [float(c[:, 0].min()), float(c[:, 1].min()),
                float(c[:, 0].max()), float(c[:, 1].max())]

    def inputs(self, cell: _Cell, planes_f: list[np.ndarray], frame_hw, history: list):
        """One crop's input tensor and its fit; ``history`` holds a frame-px curve or None per lag."""
        import cv2
        import torch

        ct = self.crop
        fit = ct.fit_crop_to_box(ct.extend_box_at_frame_edge(cell.box, frame_hw, self.edge_extend),
                                 crop_w=self.W, crop_h=self.H, pad=self.pad)
        planes = np.stack([(np.clip(ct.warp_image(g, fit.M, self.W, self.H), 0.0, 1.0) * 255.0)
                           .astype(np.uint8) for g in planes_f])
        rgb = cv2.cvtColor(planes[self.half], cv2.COLOR_GRAY2RGB)
        body = None if cell.body is None else self._box_n(cell.body, fit.M)
        flag = None if cell.flag is None else self._box_n(cell.flag, fit.M)
        rgb = ct.apply_target_hint(rgb, self.hint, self._box_n(cell.box, fit.M), body, flag)
        x_rgb = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        x_rgb = torch.from_numpy(np.ascontiguousarray(x_rgb.transpose(2, 0, 1)))
        if self.F > 1:
            x_grey = torch.from_numpy((planes.astype(np.float32) / 255.0 - GREY_MEAN) / GREY_STD)
            x = torch.cat([x_grey, x_rgb[2:3]], 0)
        else:
            x = x_rgb
        for curve in history:
            n = (None if curve is None
                 else ct.normalize_points(ct.transform_points(curve, fit.M), self.W, self.H))
            x = torch.cat([x, torch.from_numpy(ct.draw_prev_curve(n, self.W, self.H))[None]], 0)
        return x, fit

    def predict(self, xs: list, dt: float):
        """``(points (B, K, 2), visibility (B, K))``, crop pixels."""
        import torch

        ctrl, vis = [], []
        for i in range(0, len(xs), self.batch):
            x = torch.stack(xs[i:i + self.batch]).to(self.device)
            b = x.shape[0]
            with torch.no_grad():
                o = self.net(x, torch.zeros(b, self.K, 2, device=self.device),
                             torch.zeros(b, device=self.device),
                             dt=torch.full((b,), dt, device=self.device))
            ctrl.append(o["ctrl"].float().cpu())
            vis.append(torch.sigmoid(o["vis_pts_logits"].float()).cpu()
                       if "vis_pts_logits" in o else torch.ones(b, self.K))
        scale = np.array([self.W, self.H], np.float32)
        return torch.cat(ctrl).numpy() * scale, torch.cat(vis).numpy()


# --------------------------------------------------------------------------- #
# the pass over a movie
# --------------------------------------------------------------------------- #
def refine_tracks(
    movie,
    tracks: pd.DataFrame,
    cfg,
    normalizer,
    model: CenterlineModel,
    reporter: Reporter | None = None,
) -> pd.DataFrame:
    """``tracks`` with the keypoint columns replaced: pose Head, then the centerline Base to Tip."""
    reporter = reporter or NullReporter()
    n_out = cfg.n_keypoints
    if n_out != model.K + 1:
        raise ValueError(f"keypoints.nodes lists {n_out} nodes; the centerline needs the Head "
                         f"and its {model.K} points")
    out = np.full((len(tracks), n_out, 3), np.nan, np.float32)
    head = [f"kp_{a}_{HEAD}" for a in ("x", "y", "c")]
    if all(c in tracks.columns for c in head):
        out[:, 0] = tracks[head].to_numpy(np.float32)
    if tracks.empty:
        return _with_keypoints(tracks, out)

    stride = int(cfg.centerline.stack_stride)
    dt = stride / float(movie.fps or 100.0)
    last = movie.n_frames - 1
    frame_hw = (movie.height, movie.width)
    grey: dict[int, np.ndarray] = {}

    def plane(i: int) -> np.ndarray:
        i = min(max(i, 0), last)
        if i not in grey:
            grey[i] = normalizer.to_rgb_u8(movie.frame(i), i)[..., 0].astype(np.float32) / 255.0
        return grey[i]

    frames = tracks["frame"].to_numpy()
    ids = tracks["track_id"].to_numpy()
    xyxy = tracks[["x1", "y1", "x2", "y2"]].to_numpy(np.float64)
    rows_at = pd.Series(np.arange(len(tracks))).groupby(frames).indices
    history: dict[tuple[int, int], np.ndarray] = {}
    reach = model.reach * stride
    n_hinted = 0

    for t in reporter.track(sorted(rows_at), f"{movie.id}  centerline", len(rows_at)):
        rows = rows_at[t]
        cells = model.hint_cells(normalizer.to_rgb_u8(movie.frame(int(t)), int(t)))
        matched = match_cells(xyxy[rows], cells)
        planes_f = [plane(int(t) + k * stride) for k in model.offsets]
        xs, fits = [], []
        for r, m in zip(rows, matched):
            hint = cells[m] if m is not None else None
            n_hinted += hint is not None
            cell = _Cell(int(ids[r]), hint["box"] if hint else xyxy[r].tolist(),
                         hint["body"] if hint else None, hint["flag"] if hint else None)
            past = [history.get((cell.track_id, int(t) - lag * stride)) for lag in model.lags]
            x, fit = model.inputs(cell, planes_f, frame_hw, past)
            xs.append(x)
            fits.append(fit)
        points, vis = model.predict(xs, dt)
        for j, r in enumerate(rows):
            curve = model.crop.transform_points(points[j], model.crop.invert_affine(fits[j].M))
            history[(int(ids[r]), int(t))] = curve
            out[r, 1:, :2] = curve
            out[r, 1:, 2] = vis[j]

        # Frames and curves no later prediction reads.
        for i in [i for i in grey if i < t - reach]:
            del grey[i]
        for key in [k for k in history if k[1] < t - reach]:
            del history[key]

    reporter.log(f"{movie.id}: centerline on {len(tracks)} boxes, "
                 f"{n_hinted / len(tracks):.0%} with a matched hint cell")
    return _with_keypoints(tracks, out)


def _with_keypoints(tracks: pd.DataFrame, kpts: np.ndarray) -> pd.DataFrame:
    keep = [c for c in tracks.columns if not str(c).startswith("kp_")]
    # All keypoint columns at once: one at a time fragments the frame (pandas warns per
    # column, ~390 of them for a 129-point centerline).
    cols = {}
    for k in range(kpts.shape[1]):
        cols[f"kp_x_{k}"] = kpts[:, k, 0]
        cols[f"kp_y_{k}"] = kpts[:, k, 1]
        cols[f"kp_c_{k}"] = kpts[:, k, 2]
    return pd.concat([tracks[keep].reset_index(drop=True),
                      pd.DataFrame(cols)], axis=1).set_axis(tracks.index, axis=0)
