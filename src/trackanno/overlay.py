"""Shared overlay geometry and drawing.

Two consumers, two jobs:

* The **GUI** draws no pixels. ``build_clip_view`` turns a clip's box table into
  a :class:`ClipView` and the Qt canvas paints the marker as vector graphics, so
  the reticle stays one hairline wide at 8x zoom.
* ``render.py`` burns the category-labelled overlay onto the full-resolution
  movie.

``render_clip_rgb`` rasterises the same geometry for a caller that wants pixels.
Everything that returns pixels returns RGB, which is what imageio and Qt want.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import Config

__all__ = [
    "hex_to_rgb",
    "gray_to_rgb",
    "ClipView",
    "build_clip_view",
    "render_clip_rgb",
    "draw_detections",
    "TARGET_RGB",
    "CONTEXT_RGB",
    "FOLLOW_MODES",
]

TARGET_RGB = (255, 196, 0)     # the track you are annotating
CONTEXT_RGB = (0, 168, 255)    # other tracks that happen to be in the crop

# How the view centre is derived from the target's trajectory. Cycled with "s".
FOLLOW_MODES = ("centred", "smooth", "locked")


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    s = color.lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    return tuple(int(s[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def gray_to_rgb(img: np.ndarray) -> np.ndarray:
    """Grayscale -> RGB for a single (H, W) frame or a whole (T, H, W) stack.

    Always writable - the caller draws into it and the clip cache hands out
    read-only memmaps.
    """
    import cv2

    if img.ndim >= 3 and img.shape[-1] == 3:
        return np.array(img, copy=True)          # already RGB
    if img.ndim >= 3 and img.shape[-1] == 1:
        img = img[..., 0]
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    out = np.empty(img.shape + (3,), np.uint8)
    for t in range(img.shape[0]):
        cv2.cvtColor(img[t], cv2.COLOR_GRAY2RGB, dst=out[t])
    return out


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
def _draw_box(img, x1, y1, x2, y2, color, thickness=1, label=None, font_scale=0.35,
              line=None):
    import cv2

    if not all(np.isfinite([x1, y1, x2, y2])):
        return
    p1 = (int(round(x1)), int(round(y1)))
    p2 = (int(round(x2)), int(round(y2)))
    cv2.rectangle(img, p1, p2, color, thickness, line if line is not None else cv2.LINE_8)
    if label:
        cv2.putText(
            img, label, (p1[0], max(9, p1[1] - 3)),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA,
        )


def _draw_dashed_box(img, x1, y1, x2, y2, color, thickness=1, label=None, font_scale=0.35,
                     dash=7, gap=5):
    """A box drawn in dashes: the mark for a model value nobody has confirmed.

    The categories already own the colours, so the outline is what separates a
    prediction from something a person decided.
    """
    import cv2

    if not all(np.isfinite([x1, y1, x2, y2])):
        return
    x1, y1, x2, y2 = (int(round(v)) for v in (x1, y1, x2, y2))
    period = max(2, int(dash) + int(gap))
    for a, b in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
                 ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        span = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        if span < 1.0:
            continue
        steps = max(1, int(span // period))
        for s in range(steps + 1):
            start = min(s * period, span)
            end = min(start + dash, span)
            if end <= start:
                break
            p = (int(round(a[0] + (b[0] - a[0]) * start / span)),
                 int(round(a[1] + (b[1] - a[1]) * start / span)))
            q = (int(round(a[0] + (b[0] - a[0]) * end / span)),
                 int(round(a[1] + (b[1] - a[1]) * end / span)))
            cv2.line(img, p, q, color, thickness, cv2.LINE_8)
    if label:
        cv2.putText(
            img, label, (x1, max(9, y1 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA,
        )


def _draw_reticle(img, x1, y1, x2, y2, color, thickness=1, halo=True):
    """Four corner brackets instead of a closed box, so the silhouette stays
    clear where the flagellum is; the halo keeps them readable on cell and
    background alike."""
    import cv2

    if not all(np.isfinite([x1, y1, x2, y2])):
        return
    x1, y1, x2, y2 = (int(round(v)) for v in (x1, y1, x2, y2))
    arm = max(3, int(round(min(x2 - x1, y2 - y1) * 0.28)))
    segments = [
        ((x1, y1), (x1 + arm, y1)), ((x1, y1), (x1, y1 + arm)),
        ((x2, y1), (x2 - arm, y1)), ((x2, y1), (x2, y1 + arm)),
        ((x1, y2), (x1 + arm, y2)), ((x1, y2), (x1, y2 - arm)),
        ((x2, y2), (x2 - arm, y2)), ((x2, y2), (x2, y2 - arm)),
    ]
    if halo:
        for a, b in segments:
            cv2.line(img, a, b, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    for a, b in segments:
        cv2.line(img, a, b, color, thickness, cv2.LINE_AA)


def _draw_skeleton(img, xy, edges, color, radius=2, thickness=1, line=None):
    import cv2

    line = cv2.LINE_8 if line is None else line
    valid = np.isfinite(xy).all(axis=1)
    pts = [None if not valid[k] else (int(xy[k, 0]), int(xy[k, 1])) for k in range(len(xy))]
    for ia, ib in edges:
        if pts[ia] is not None and pts[ib] is not None:
            cv2.line(img, pts[ia], pts[ib], color, thickness, line)
    for p in pts:
        if p is not None:
            cv2.circle(img, p, radius, color, -1, line)


def _kpts_from_row(row, n_kpts: int) -> np.ndarray:
    xy = np.full((n_kpts, 2), np.nan, dtype=np.float32)
    for k in range(n_kpts):
        try:
            xy[k, 0] = row[f"kp_x_{k}"]
            xy[k, 1] = row[f"kp_y_{k}"]
        except (KeyError, IndexError):
            return xy
    return xy


def _kpt_columns(frame, n_kpts: int) -> np.ndarray | None:
    """(N, K, 2) keypoint array pulled from the DataFrame in one shot, rather
    than by per-row lookup."""
    if n_kpts <= 0:
        return None
    xcols = [f"kp_x_{k}" for k in range(n_kpts)]
    ycols = [f"kp_y_{k}" for k in range(n_kpts)]
    if not all(c in frame.columns for c in xcols + ycols):
        return None
    return np.stack(
        [frame[xcols].to_numpy(dtype=np.float32), frame[ycols].to_numpy(dtype=np.float32)],
        axis=-1,
    )


# --------------------------------------------------------------------------- #
# GUI: clip overlay
# --------------------------------------------------------------------------- #
def render_clip_rgb(
    clip: np.ndarray,
    boxes,
    cfg: Config,
    show_boxes: bool = True,
    show_skeleton: bool = True,
    show_context: bool = False,
) -> np.ndarray:
    """(T, S, S) uint8 -> (T, S, S, 3) uint8 RGB with the overlay burned in."""
    rgb = np.ascontiguousarray(gray_to_rgb(clip))
    if not show_boxes or boxes is None or len(boxes) == 0:
        return rgb

    edges = cfg.keypoints.edge_idx if cfg.is_pose else []
    n_kpts = cfg.n_keypoints
    draw_skeleton = show_skeleton and cfg.is_pose and n_kpts > 0

    # Columns out as numpy once; the per-row Python work is what costs here.
    t_col = boxes["t"].to_numpy(dtype=np.int32)
    target_col = boxes["is_target"].to_numpy(dtype=bool)
    xyxy = boxes[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32)
    track_col = boxes["track_id"].to_numpy(dtype=np.int32)
    kpts = _kpt_columns(boxes, n_kpts) if draw_skeleton else None
    n_t = rgb.shape[0]

    for i in range(len(boxes)):
        is_target = bool(target_col[i])
        if not is_target and not show_context:
            continue
        t = int(t_col[i])
        if not 0 <= t < n_t:
            continue
        color = TARGET_RGB if is_target else CONTEXT_RGB
        frame = rgb[t]
        if is_target:
            _draw_reticle(frame, xyxy[i, 0], xyxy[i, 1], xyxy[i, 2], xyxy[i, 3], color, 1)
        else:
            _draw_box(
                frame, xyxy[i, 0], xyxy[i, 1], xyxy[i, 2], xyxy[i, 3], color,
                thickness=1, label=f"t{int(track_col[i])}",
            )
        if kpts is not None:
            _draw_skeleton(frame, kpts[i], edges, color)
    return rgb


# --------------------------------------------------------------------------- #
# GUI: view geometry (no pixels)
# --------------------------------------------------------------------------- #
@dataclass
class ClipView:
    """Where to look, and what to mark, for every frame of one clip.

    Coordinates are cache pixels. The viewer maps ``(centre[t] - side/2, side)``
    onto its canvas, so centring and zoom are display-time decisions: no pixel is
    resampled and one cached array serves every mode.
    """

    size: int                       # cache side length, px
    n_frames: int
    target: np.ndarray              # (T, 4) xyxy, NaN where the track was never seen
    centre: np.ndarray              # (T, 2) view centre
    side: float                     # side of the visible square, in cache px
    context: np.ndarray = field(default_factory=lambda: np.zeros((0, 5), np.float32))
    detected: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    kpts: np.ndarray | None = None  # (T, K, 2) target keypoints, NaN where absent
    held: np.ndarray | None = None  # (T, 4) target with gaps held from the nearest hit

    def box(self, t: int) -> np.ndarray | None:
        """The target's box on frame ``t``, or None if it has none to draw."""
        if not 0 <= t < self.n_frames:
            return None
        b = self.target[t]
        return None if not np.isfinite(b).all() else b

    def held_box(self, t: int) -> np.ndarray | None:
        """The box to draw on a frame with no detection, or None.

        Drawn dimmed rather than dropped: a marker that vanishes strobes at
        playback speed and reads as an event in the animal, not a missed frame.
        """
        if self.held is None or not 0 <= t < self.n_frames:
            return None
        b = self.held[t]
        return None if not np.isfinite(b).all() else b

    def source_rect(self, t: int) -> tuple[float, float, float, float]:
        """(x, y, w, h) of the visible square in cache pixels for frame ``t``."""
        t = int(np.clip(t, 0, self.n_frames - 1))
        cx, cy = self.centre[t]
        return (cx - self.side / 2.0, cy - self.side / 2.0, self.side, self.side)

    def context_for(self, t: int) -> np.ndarray:
        if len(self.context) == 0:
            return self.context
        return self.context[self.context[:, 0] == t][:, 1:]


def _hold_fill_rows(values: np.ndarray) -> np.ndarray:
    """Fill all-NaN rows from the nearest row that has data, or a track that
    drops out loses stabilisation for exactly those frames and the animal jumps
    sideways and back."""
    present = np.isfinite(values).all(axis=1)
    if present.all() or not present.any():
        return values
    idx = np.flatnonzero(present)
    all_idx = np.arange(values.shape[0])
    after = idx[np.clip(np.searchsorted(idx, all_idx), 0, idx.size - 1)]
    before = idx[np.clip(np.searchsorted(idx, all_idx, side="right") - 1, 0, idx.size - 1)]
    pick = np.where(np.abs(after - all_idx) <= np.abs(before - all_idx), after, before)
    out = values.copy()
    out[~present] = values[pick[~present]]
    return out


def _smooth_path(path: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average with edge padding, along axis 0."""
    if window <= 1 or path.shape[0] <= 2:
        return path
    half = max(1, min(window // 2, path.shape[0] - 1))
    padded = np.pad(path, ((half, half), (0, 0)), mode="edge")
    kernel = np.ones(2 * half + 1) / (2 * half + 1)
    return np.stack(
        [np.convolve(padded[:, c], kernel, mode="valid") for c in range(path.shape[1])], axis=1
    )


def build_clip_view(
    boxes,
    n_frames: int,
    size: int,
    mode: str = "centred",
    zoom: float = 1.0,
    smooth_window: int = 15,
    with_context: bool = False,
    n_kpts: int = 0,
) -> ClipView:
    """Turn a clip's box table into per-frame view geometry.

    ``mode`` decides what the view follows:

    ``centred``  one fixed view on the *median of the track's own path*, not the
                 crop centre, so a border clip's animal is not parked in a corner.
    ``smooth``   a moving average of the path: drift removed, beat kept.
    ``locked``   re-centred every frame - translation invisible by construction.

    ``zoom`` > 1 shrinks the visible square around that centre. Nothing is
    clamped to the cached crop; the viewer pads outside it.
    """
    target = np.full((n_frames, 4), np.nan, dtype=np.float64)
    context = np.zeros((0, 5), dtype=np.float32)
    detected = np.zeros(n_frames, dtype=bool)
    kpts = None

    if boxes is not None and len(boxes) > 0:
        t_col = boxes["t"].to_numpy(dtype=np.int64)
        is_target = boxes["is_target"].to_numpy(dtype=bool)
        xyxy = boxes[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float64)
        inside = (t_col >= 0) & (t_col < n_frames)
        sel = is_target & inside
        target[t_col[sel]] = xyxy[sel]
        detected[t_col[sel]] = True
        if with_context:
            other = (~is_target) & inside
            context = np.column_stack(
                [t_col[other].astype(np.float32), xyxy[other].astype(np.float32)]
            )
        all_kpts = _kpt_columns(boxes, n_kpts)
        if all_kpts is not None:
            kpts = np.full((n_frames, all_kpts.shape[1], 2), np.nan, dtype=np.float32)
            kpts[t_col[sel]] = all_kpts[sel]

    filled = _hold_fill_rows(target)
    if np.isfinite(filled).all():
        path = np.stack([(filled[:, 0] + filled[:, 2]) * 0.5,
                         (filled[:, 1] + filled[:, 3]) * 0.5], axis=1)
    else:
        # No detection anywhere in the clip: fall back to the crop's centre.
        path = np.full((n_frames, 2), size / 2.0)

    if mode == "locked":
        centre = path
    elif mode == "smooth":
        centre = _smooth_path(path, smooth_window)
    else:
        centre = np.repeat(np.median(path, axis=0, keepdims=True), n_frames, axis=0)

    return ClipView(
        size=int(size),
        n_frames=int(n_frames),
        target=target,
        centre=centre,
        side=float(size) / max(1e-6, float(zoom)),
        context=context,
        detected=detected,
        kpts=kpts,
        held=filled,
    )


# --------------------------------------------------------------------------- #
# renderer: full-frame overlay
# --------------------------------------------------------------------------- #
def draw_detections(
    frame_rgb: np.ndarray,
    records,
    cfg: Config,
    colors: dict[int, tuple[int, int, int]],
    labels: dict[int, str],
    thickness: int = 2,
    font_scale: float = 0.4,
    show_skeleton: bool = False,
    dashed: set | None = None,
) -> np.ndarray:
    """Draw one frame's tracked boxes onto a full-resolution RGB frame, in place.

    ``dashed`` holds the track ids whose value came from the model rather than a
    person; those are outlined in dashes so the two are never confused.
    """
    edges = cfg.keypoints.edge_idx if cfg.is_pose else []
    n_kpts = cfg.n_keypoints
    dashed = dashed or set()
    for row in records.itertuples(index=False):
        tid = int(row.track_id)
        color = colors.get(tid)
        if color is None:
            continue
        draw = _draw_dashed_box if tid in dashed else _draw_box
        draw(
            frame_rgb, row.x1, row.y1, row.x2, row.y2, color,
            thickness=thickness, label=labels.get(tid), font_scale=font_scale,
        )
        if show_skeleton and cfg.is_pose and n_kpts:
            _draw_skeleton(frame_rgb, _kpts_from_row(row._asdict(), n_kpts), edges, color)
    return frame_rgb
