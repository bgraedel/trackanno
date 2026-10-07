"""Per-detection embeddings, ROI-pooled from the YOLO head's P3/P4/P5 input maps."""

from __future__ import annotations

import numpy as np

__all__ = ["InstanceFeatures", "LEVEL_NAMES"]

LEVEL_NAMES = ("P3", "P4", "P5")


class InstanceFeatures:
    """Captures head-input feature maps and ROI-pools them per detection."""

    def __init__(self, model, levels=LEVEL_NAMES, pool: int = 1, dtype: str = "float16"):
        self.levels = [LEVEL_NAMES.index(str(l).upper()) for l in levels]
        if not self.levels:
            raise ValueError("features.levels must name at least one of P3/P4/P5")
        self.pool = max(1, int(pool))
        self.dtype = np.dtype(dtype)
        self._maps: list | None = None
        self._dim: int | None = None

        head = model.model.model[-1]
        self._handle = head.register_forward_pre_hook(self._capture)

    # -- hook --------------------------------------------------------------- #
    def _capture(self, module, args):
        maps = args[0]
        # The head takes a list/tuple of the three pyramid levels.
        if isinstance(maps, (list, tuple)) and len(maps) >= 3:
            self._maps = list(maps)
        else:
            self._maps = None

    def reset(self) -> None:
        """Drop captured maps before a predict, so a pass that misses the head returns None."""
        self._maps = None

    def close(self) -> None:
        self._maps = None
        try:
            self._handle.remove()
        except Exception:
            pass

    # -- extraction ---------------------------------------------------------- #
    @property
    def dim(self) -> int | None:
        return self._dim

    def pooled(
        self, boxes_xyxy: np.ndarray, batch_index: int, tile_size: tuple[int, int],
        imgsz: int | None = None, stride: int = 32,
    ) -> np.ndarray | None:
        """ROI-pool tile-local ``boxes_xyxy`` of tile ``batch_index`` to ``(N, D)``, or None.

        ``imgsz`` is the requested predictor size, used to reproduce the letterbox."""
        if self._maps is None or boxes_xyxy.shape[0] == 0:
            return None

        import torch
        from torchvision.ops import roi_align

        maps = self._maps
        # Below assumes maps[0] is P3 at stride 8.
        if len(maps) != len(LEVEL_NAMES):
            raise RuntimeError(
                f"the detection head received {len(maps)} feature maps, not "
                f"{len(LEVEL_NAMES)}; set features.enabled: false for this model"
            )
        net_h = int(maps[0].shape[-2]) * 8
        net_w = int(maps[0].shape[-1]) * 8
        tile_h, tile_w = tile_size

        # Reproduce ultralytics' integer LetterBox rounding; r comes from the requested
        # imgsz, and min-rectangle vs square padding is whichever matches the map size.
        target = int(imgsz) if imgsz else max(net_h, net_w)
        r = min(target / max(tile_h, 1), target / max(tile_w, 1))
        unpad_w, unpad_h = int(round(tile_w * r)), int(round(tile_h * r))
        for reduce_to_stride in (True, False):
            dw, dh = target - unpad_w, target - unpad_h
            if reduce_to_stride:
                dw, dh = dw % stride, dh % stride
            if (unpad_h + dh, unpad_w + dw) == (net_h, net_w):
                break
        else:
            raise RuntimeError(
                f"cannot reconstruct the letterbox for tile {tile_w}x{tile_h} at imgsz "
                f"{target} (net {net_w}x{net_h}); set features.enabled: false"
            )
        # LetterBox splits the padding with this exact rounding.
        pad_x = int(round(dw / 2.0 - 0.1))
        pad_y = int(round(dh / 2.0 - 0.1))

        device = maps[0].device
        # Per-axis scale: the rounded extents need not share one.
        scale_x, scale_y = unpad_w / max(tile_w, 1), unpad_h / max(tile_h, 1)
        boxes = torch.as_tensor(boxes_xyxy, dtype=torch.float32, device=device)
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_x + pad_x
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_y + pad_y
        rois = torch.cat(
            [torch.full((boxes.shape[0], 1), float(batch_index), device=device), boxes], dim=1
        )

        parts = []
        with torch.no_grad():
            for lvl in self.levels:
                fmap = maps[lvl]
                if batch_index >= fmap.shape[0]:
                    return None
                pooled = roi_align(
                    fmap.float(), rois,
                    output_size=self.pool,
                    spatial_scale=1.0 / (8 * (2 ** lvl)),
                    aligned=True,
                )
                parts.append(pooled.flatten(1))
            out = torch.cat(parts, dim=1).cpu().numpy()

        self._dim = int(out.shape[1])
        return out.astype(self.dtype, copy=False)
