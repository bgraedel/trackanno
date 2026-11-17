"""The centerline wrapper: hint matching, the pass that rewrites the keypoints, and its smoothing."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fixtures import prep_project

from trackanno.config import ConfigError, load_config
from trackanno.detect.centerline import assign_parts, match_cells, node_names, refine_tracks
from trackanno.measure import wavespectra as ws

NAMES = {0: "animal", 1: "body", 2: "flagellum"}
K = 6  # the stub model's points from Base to Tip


def test_parts_go_to_the_animal_that_holds_them() -> None:
    xyxy = np.array([[0, 0, 50, 50], [100, 0, 150, 50],      # two animals
                     [10, 10, 20, 20], [110, 10, 120, 20],   # a body in each
                     [20, 20, 45, 45], [22, 22, 44, 44]],    # two flagella in the first
                    float)
    cls = np.array([0, 0, 1, 1, 2, 2])
    conf = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.95])
    cells = assign_parts(xyxy, cls, conf, NAMES)
    assert [c["body"] for c in cells] == [[10, 10, 20, 20], [110, 10, 120, 20]]
    assert cells[0]["flag"] == [22, 22, 44, 44], "the more confident flagellum wins"
    assert cells[1]["flag"] is None


def test_tracked_boxes_match_hint_cells_one_to_one() -> None:
    cells = [{"box": [0, 0, 10, 10]}, {"box": [100, 100, 110, 110]}]
    boxes = np.array([[101, 101, 111, 111], [1, 1, 11, 11], [500, 500, 510, 510]], float)
    assert match_cells(boxes, cells) == [1, 0, None]
    assert match_cells(boxes, []) == [None, None, None]


def test_nodes_are_the_head_then_every_point() -> None:
    assert node_names(4) == ["Head", "Base", "C1", "C2", "Tip"]


class _FakeModel:
    """The model's interface, placing its K points along each box's diagonal."""

    offsets, lags = [-2, 0, 2], [1, 2]
    reach = 2
    K = K
    crop = SimpleNamespace(invert_affine=lambda M: M, transform_points=lambda p, M: np.asarray(p))

    def __init__(self):
        self.fed: list[tuple[int, list[bool]]] = []

    def hint_cells(self, rgb):
        return []

    def inputs(self, cell, planes, frame_hw, history):
        assert len(planes) == len(self.offsets)
        self.fed.append((cell.track_id, [h is not None for h in history]))
        return np.asarray(cell.box, float), SimpleNamespace(M=None)

    def predict(self, xs, dt):
        t = np.linspace(0, 1, self.K)[:, None]
        points = np.stack([b[:2] + t * (b[2:] - b[:2]) for b in xs])
        return points, np.tile(1.0 - t[:, 0], (len(xs), 1))


@pytest.fixture
def pose_case(tmp_path: Path):
    from trackanno.video.movies import Normalizer

    cfg, movie, _ = prep_project(tmp_path, "centerline")
    cfg.detection.task = "pose"
    cfg.keypoints.nodes = node_names(K)
    tracks = movie.tracks().query("frame < 20").reset_index(drop=True)
    for k in range(8):  # the pose model's own 8 nodes
        tracks[f"kp_x_{k}"] = tracks["x1"] + k
        tracks[f"kp_y_{k}"] = tracks["y1"] - k
        tracks[f"kp_c_{k}"] = 0.9
    return cfg, movie, tracks, Normalizer.from_movie(movie, cfg.normalize)


def test_refine_keeps_the_head_and_writes_every_point(pose_case) -> None:
    cfg, movie, tracks, normalizer = pose_case
    out = refine_tracks(movie, tracks, cfg, normalizer, _FakeModel())

    kp = sorted(c for c in out.columns if c.startswith("kp_"))
    assert kp == sorted(f"kp_{a}_{k}" for k in range(K + 1) for a in "xyc")
    assert len(out) == len(tracks)
    np.testing.assert_allclose(out["kp_x_0"], tracks["kp_x_0"])
    np.testing.assert_allclose(out["kp_y_0"], tracks["kp_y_0"])
    np.testing.assert_allclose(out["kp_x_1"], tracks["x1"], atol=1e-4)  # Base
    np.testing.assert_allclose(out[f"kp_y_{K}"], tracks["y2"], atol=1e-4)  # Tip
    assert (out["kp_c_1"] > out[f"kp_c_{K}"]).all(), "confidence is the visibility"


def test_a_node_count_that_does_not_fit_the_model_is_refused(pose_case) -> None:
    cfg, movie, tracks, normalizer = pose_case
    cfg.keypoints.nodes = node_names(K + 1)
    with pytest.raises(ValueError, match="keypoints.nodes"):
        refine_tracks(movie, tracks, cfg, normalizer, _FakeModel())


def test_each_track_is_fed_only_its_own_history(pose_case) -> None:
    cfg, movie, tracks, normalizer = pose_case
    model = _FakeModel()
    refine_tracks(movie, tracks, cfg, normalizer, model)
    per_track: dict[int, list] = {}
    for track_id, fed in model.fed:
        per_track.setdefault(track_id, []).append(fed)
    for fed in per_track.values():
        assert fed[0] == [False, False], "a track starts with no history"
        assert fed[1] == [True, False], "lag 1 first, lag 2 a frame later"
        assert all(f == [True, True] for f in fed[2:])


def _wavy(length_um: float, um_per_px: float, wave_um: float = 2.0, n: int = 129):
    """A straight flagellum along x with a small bend every ``wave_um``, as one frame of nodes."""
    s = np.linspace(0.0, length_um, n)
    x = np.concatenate([[-5.0], s]) / um_per_px  # node 0 is the Head, behind the Base
    y = np.concatenate([[0.0], 0.05 * np.sin(2 * np.pi * s / wave_um)]) / um_per_px
    return x[None], y[None]


def _kept(length_um: float, um_per_px: float) -> float:
    px, py = _wavy(length_um, um_per_px)
    curves = ws.frame_curves(px, py, cutoff_px=2.0 / um_per_px)
    y = curves["y"][0]
    return float(np.ptp(y[y.size // 4: -(y.size // 4)]) * um_per_px / 0.1)


def test_centerline_smoothing_is_the_same_distance_at_every_magnification() -> None:
    assert _kept(12.0, 0.325) == pytest.approx(_kept(12.0, 0.0621), rel=0.02)


def test_centerline_smoothing_does_not_depend_on_the_cell_length() -> None:
    assert _kept(8.0, 0.325) == pytest.approx(_kept(16.0, 0.325), rel=0.02)


def test_enabling_it_needs_a_pose_project(root: Path) -> None:
    from trackanno.yamledit import set_values

    cfg, _, _ = prep_project(root, "centerline-config")
    path = cfg.path
    base = path.read_text(encoding="utf-8")
    on = {"centerline.enabled": True, "centerline.checkpoint": "a.ckpt",
          "centerline.hint_model": "b.pt"}

    path.write_text(set_values(base, on), encoding="utf-8")
    with pytest.raises(ConfigError, match="pose"):
        load_config(path)

    path.write_text(set_values(base, {**on, "detection.task": "pose"}), encoding="utf-8")
    assert load_config(path).centerline.enabled


def test_hint_detector_tiles_and_maps_back_to_the_frame() -> None:
    from trackanno.detect.centerline import CenterlineModel

    class _Boxes:
        def __init__(self, rows):
            import torch
            a = torch.tensor(rows, dtype=torch.float32).reshape(-1, 6)
            self.xyxy, self.conf, self.cls = a[:, :4], a[:, 4], a[:, 5]

    class _Hinter:
        names = NAMES

        def predict(self, crops, **kw):
            # Replays one prepared answer per tile, in tile order.
            out = []
            for c in crops:
                out.append(SimpleNamespace(boxes=_Boxes(self.at.pop(0))))
            return out

    m = object.__new__(CenterlineModel)
    m.hint_tile, m.hint_overlap, m.hint_merge_iou = 640, 0.4, 0.6
    m.hint_imgsz, m.hint_conf, m.batch, m.device = 640, 0.3, 4, "cpu"
    assert m.hint_tiles(600, 600) == [(0, 0, 600, 600)], "a frame that fits is one tile"
    m.hint_tile = 0
    assert m.hint_tiles(2400, 2400) == [(0, 0, 2400, 2400)], "hint_tile 0 is the whole frame"
    m.hint_tile = 640
    tiles = m.hint_tiles(1300, 1300)
    assert len(tiles) == 9 and all(x1 - x0 == 640 for x0, _, x1, _ in tiles)

    cell = [(600, 100, 660, 160, 0.9, 0), (602, 102, 620, 120, 0.8, 1)]  # animal, body
    at = []
    for x0, y0, x1, y1 in tiles:
        inside = x0 <= 600 and 660 <= x1 and y0 <= 100 and 160 <= y1
        at.append([(a - x0, b - y0, c - x0, d - y0, s, k) for a, b, c, d, s, k in cell]
                  if inside else [])
    hinter = _Hinter()
    hinter.at = at
    m.hinter = hinter
    cells = m.hint_cells(np.zeros((1300, 1300, 3), np.uint8))
    assert len(cells) == 1, "the cell seen by two tiles is one cell"
    assert cells[0]["box"] == [600, 100, 660, 160] and cells[0]["body"] == [602, 102, 620, 120]
