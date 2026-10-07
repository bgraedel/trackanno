"""Tiling geometry, NMS and the checkpoint probe in :mod:`trackanno.detect.tiling`."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from trackanno.detect.tiling import (
    Detections, TiledDetector, _iou_matrix, _nms, tile_grid,
)


def test_the_grid_covers_every_pixel_with_full_tiles() -> None:
    """Awkward sizes included: no hole, no short tile, none outside the frame."""
    cases = [(2304, 2304, 640, 0.25), (2044, 2048, 640, 0.25),
             (1000, 3000, 640, 0.25), (641, 641, 640, 0.25),
             (640, 640, 640, 0.25), (300, 5000, 640, 0.1),
             (2048, 2048, 640, 0.0), (777, 1234, 512, 0.33)]
    for height, width, tile, overlap in cases:
        tiles = tile_grid(height, width, tile, overlap)
        cover = np.zeros((height, width), dtype=np.uint8)
        shaped = True
        for x0, y0, x1, y1 in tiles:
            shaped &= 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height
            shaped &= (x1 - x0) == min(tile, width) and (y1 - y0) == min(tile, height)
            cover[y0:y1, x0:x1] += 1
        assert shaped and int(cover.min()) >= 1, (
            f"{height}x{width} tile {tile} overlap {overlap}: full tiles, no hole")


def test_an_object_smaller_than_the_overlap_always_sits_clear_of_a_cut() -> None:
    """In 1-D: an object up to tile*overlap - 2*margin px is inside some tile, clear of its cuts."""
    tile, overlap, margin, extent = 640, 0.25, 2, 2304
    step = max(1, int(round(tile * (1.0 - overlap))))
    spans = sorted({(x0, x1) for x0, _, x1, _ in tile_grid(1, extent, tile, overlap)})
    size = (tile - step) - 2 * margin
    clear = all(
        any((x0 == 0 or a >= x0 + margin) and (x1 == extent or a + size <= x1 - margin)
            for x0, x1 in spans)
        for a in range(0, extent - size)
    )
    assert clear, f"a{size} px object lands clear of an interior border at every position"


def test_the_numpy_nms_matches_torchvision_where_it_matters() -> None:
    """Jittered near-duplicates through both, so suppression really happens."""
    rng = np.random.default_rng(1)
    boxes, scores = [], []
    for cx, cy in rng.uniform(50, 450, (80, 2)):
        for _ in range(4):
            jx, jy = rng.normal(0, 3, 2)
            w, h = rng.uniform(20, 40, 2)
            boxes.append([cx + jx - w / 2, cy + jy - h / 2,
                          cx + jx + w / 2, cy + jy + h / 2])
            scores.append(rng.uniform(0.1, 1.0))
    boxes = np.array(boxes, np.float32)
    scores = np.array(scores, np.float32)

    for thresh in (0.3, 0.6, 0.8):
        keep_tv = _nms(boxes, scores, thresh)
        saved = sys.modules.get("torchvision.ops")
        sys.modules["torchvision.ops"] = None  # forces the numpy fallback
        try:
            keep_np = _nms(boxes, scores, thresh)
        finally:
            if saved is None:
                sys.modules.pop("torchvision.ops", None)
            else:
                sys.modules["torchvision.ops"] = saved
        assert (len(keep_tv) < len(boxes)
              and set(keep_tv.tolist()) == set(keep_np.tolist())), (
            f"IoU{thresh}: fallback keeps the same {len(keep_tv)} of {len(boxes)} boxes")


def test_iou_and_the_edge_split() -> None:
    """IoU against hand values; interior cuts flag boxes, the frame edge does not."""
    a = np.array([[0, 0, 10, 10]], np.float32)
    b = np.array([[5, 5, 15, 15], [0, 0, 10, 10], [20, 20, 30, 30]], np.float32)
    assert np.allclose(_iou_matrix(a, b)[0], [25 / 175, 1.0, 0.0], atol=1e-6), (
        "IoU: 25/175, identity 1, disjoint 0")

    def dets(*rows):
        xyxy = np.array(rows, np.float32)
        return Detections(xyxy=xyxy, conf=np.full(len(xyxy), 0.9, np.float32),
                          cls=np.zeros(len(xyxy), np.int32), kpts=None)

    stub = SimpleNamespace(
        cfg=SimpleNamespace(tiling=SimpleNamespace(edge_margin_px=2)), n_kpts=0
    )
    # Frame 1000x2000, tile (480, 0, 1120, 640): left and right cuts are
    # interior, the top is the frame's own edge.
    d = dets([481, 10, 530, 60],       # within the margin of the left cut
             [700, 10, 750, 60],       # interior
             [1090, 10, 1119, 60],     # within the margin of the right cut
             [700, 0, 750, 50])        # touches y=0, the frame edge
    kept, suspect = TiledDetector._drop_edge_truncated(
        stub, d, (480, 0, 1120, 640), 1000, 2000
    )
    assert (len(kept) == 2 and len(suspect) == 2
          and kept.xyxy[0, 0] == 700 and kept.xyxy[1, 1] == 0), (
        "interior cuts flag two boxes, the interior and frame-edge boxes stay")

    survivors = dets([700, 10, 750, 60]).xyxy
    truncated = dets([470, 100, 640, 300]).xyxy
    assert bool((_iou_matrix(truncated, survivors).max(axis=1) < 0.3)[0]), (
        "a suspect no survivor covers (IoU < 0.3) is re-admitted")


def _probe(tmp_path, checkpoint: dict) -> dict:
    """Save ``checkpoint`` as a .pt file and read it back with read_model_info."""
    torch = pytest.importorskip("torch")
    from trackanno.detect.tiling import read_model_info

    path = tmp_path / "model.pt"
    torch.save(checkpoint, path)
    return read_model_info(path)


def test_the_probe_reads_the_ema_weights(tmp_path) -> None:
    """Mid-training checkpoints keep the weights in "ema" and leave "model" None."""
    ema = SimpleNamespace(kpt_shape=[8, 3], names={0: "cell"}, yaml={}, model=None)
    info = _probe(tmp_path, {"model": None, "ema": ema,
                             "train_args": {"task": "pose", "imgsz": 1280}})
    assert info["n_keypoints"] == 8, info
    assert info["task"] == "pose"
    assert info["classes"] == ["cell"]
    assert info["imgsz"] == 1280


def test_the_probe_falls_back_to_the_head_module(tmp_path) -> None:
    head = SimpleNamespace(kpt_shape=[5, 3])
    info = _probe(tmp_path, {"model": SimpleNamespace(names={0: "cell"}, model=[head])})
    assert info["n_keypoints"] == 5, info
    assert info["task"] == "pose"


def test_the_probe_reports_a_plain_detector_as_one(tmp_path) -> None:
    plain = SimpleNamespace(names={0: "cell"}, model=[SimpleNamespace()], task="detect")
    info = _probe(tmp_path, {"model": plain, "train_args": {"task": "detect"}})
    assert not info["n_keypoints"], info
    assert info["task"] == "detect"
