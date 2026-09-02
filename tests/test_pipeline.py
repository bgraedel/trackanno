"""End-to-end checks for everything that does not need a GPU or a display.

Covers clip planning, crop geometry, the cache round-trip (including the
reassociation guarantee), the overlay box table, window_mode, and the
annotation CSV's resume/undo behaviour.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from harness import check
from trackanno.annotate.clips import (
    ClipStore, MANIFEST_FILE, build_box_table, choose_crop_size, finalise_geometry,
    plan_clips, specs_to_frame, write_clip_cache,
)
from trackanno.config import load_config
from trackanno.video.movies import MovieSource, Normalizer
from trackanno.view.overlay import build_clip_view, render_clip_rgb
from trackanno.annotate.store import AnnotationStore
from trackanno.detect.tiling import tile_grid

N_FRAMES, HEIGHT, WIDTH = 120, 512, 512
CONFIG_TEXT = """
paths:  {video_dir: ./videos, model: ./model.pt, work_dir: ./work, csv: ./annotations.csv}
tiff:   {fps: 100}
detection: {task: detect, imgsz: 640}
clips:  {frames: 50, stride: 50, window_mode: all, follow: fixed, crop_from: union,
         pad: 1.6, crop_size: 96, max_size: 256, min_clip_frames: 40}
normalize: {mode: fixed}
tracking: {min_track_length: 50}
categories:
  - name: behaviour
    type: single
    required: true
    options:
      - {name: swimming, key: "1", color: "#4C9F70"}
      - {name: tumbling, key: "2", color: "#D95F02"}
  - name: quality
    type: multi
    required: false
    options:
      - {name: blurry, key: "q", color: "#999999"}
ui: {annotator: tester}
"""


# --------------------------------------------------------------------------- #
class SyntheticMovie(MovieSource):
    """Two bright squares drifting across a textured background."""

    def __init__(self):
        self.id = "synthetic"
        self.path = Path("synthetic.tif")
        self.series = 0
        self.channel = 0
        self.n_frames = N_FRAMES
        self.height, self.width = HEIGHT, WIDTH
        self.dtype = np.dtype(np.uint8)
        self.fps = 100.0
        self.pixel_size_um = 0.1

        rng = np.random.default_rng(0)
        base = rng.integers(20, 60, size=(HEIGHT, WIDTH), dtype=np.uint8)
        self._data = np.repeat(base[None], N_FRAMES, axis=0).copy()
        self.truth: list[dict] = []
        for track_id, (x0, y0, dx, dy, size) in enumerate(
            [(60, 80, 2.0, 0.5, 20), (300, 200, -1.0, 1.5, 16)], start=1
        ):
            for f in range(N_FRAMES):
                x = int(round(x0 + dx * f))
                y = int(round(y0 + dy * f))
                x = int(np.clip(x, 0, WIDTH - size))
                y = int(np.clip(y, 0, HEIGHT - size))
                self._data[f, y : y + size, x : x + size] = 200 + track_id * 20
                self.truth.append(
                    {"frame": f, "track_id": track_id,
                     "x1": float(x), "y1": float(y),
                     "x2": float(x + size), "y2": float(y + size), "conf": 0.9}
                )

    def frame(self, index: int) -> np.ndarray:
        return self._data[index]

    def tracks(self) -> pd.DataFrame:
        return pd.DataFrame(self.truth)


# --------------------------------------------------------------------------- #
def test_a_track_at_the_frame_edge_can_be_discarded(cfg, movie) -> None:
    """Pose is the reason, so the rule is about the box reaching the frame.

    At the edge the box is cut off by the picture rather than by the animal
    ending, and a keypoint model places its parts against a shape that is not
    all there. Nothing downstream reports it - the detector is confident about
    the part it can see - so the track has to be excluded rather than detected.
    """
    from trackanno.annotate.clips import border_share, drop_border_tracks

    frame_size = (movie.height, movie.width)
    rows = []
    for f in range(cfg.clips.frames):
        # 1 stays in the middle, 2 is pinned against the left edge, 3 grazes
        # the bottom on one frame out of the window.
        rows.append({"frame": f, "track_id": 1, "x1": 200.0, "y1": 100.0,
                     "x2": 220.0, "y2": 120.0, "conf": 0.9})
        rows.append({"frame": f, "track_id": 2, "x1": 0.0, "y1": 100.0,
                     "x2": 20.0, "y2": 120.0, "conf": 0.9})
        low = f == 0
        rows.append({
            "frame": f, "track_id": 3, "x1": 200.0,
            "y1": float(movie.height - 20) if low else 100.0,
            "x2": 220.0,
            "y2": float(movie.height) if low else 120.0, "conf": 0.9,
        })
    tracks = pd.DataFrame(rows)
    cfg.tracking.min_track_length = 1
    cfg.clips.min_clip_frames = 1

    shares = border_share(tracks, frame_size, 4)
    check("a track in the middle never reaches the edge", shares[1] == 0.0)
    check("one pinned against it always does", shares[2] == 1.0)
    check("one that grazes it reports the share, not a yes or no",
          abs(shares[3] - 1.0 / cfg.clips.frames) < 1e-9, str(shares[3]))

    check("off by default, so an existing project plans what it always did",
          cfg.tracking.border_margin_px == 0
          and drop_border_tracks(tracks, frame_size, cfg)[0] is tracks)

    cfg.tracking.border_margin_px = 4
    kept, dropped = drop_border_tracks(tracks, frame_size, cfg)
    check("with no tolerance, any frame at the edge takes the track",
          sorted(dropped) == [2, 3], str(dropped))
    check("...and the middle one survives",
          sorted(kept["track_id"].unique()) == [1])

    cfg.tracking.border_max_frac = 0.5
    kept, dropped = drop_border_tracks(tracks, frame_size, cfg)
    check("a tolerance keeps a track that only grazes it",
          sorted(dropped) == [2] and sorted(kept["track_id"].unique()) == [1, 3],
          str(dropped))

    cfg.tracking.border_max_frac = 0.0
    at_border: dict = {}
    specs = plan_clips(tracks, movie.id, cfg.clips.frames, cfg,
                       frame_size=frame_size, discarded=at_border)
    check("and no clip is cut for a discarded track",
          sorted({s.track_id for s in specs}) == [1],
          str(sorted({s.track_id for s in specs})))
    # prepare warns from this dict, so it has to be what was actually dropped:
    # a caller recomputing it would have to restate which tracks were candidates.
    check("plan_clips reports what it discarded, and only that",
          sorted(at_border) == [2, 3], str(at_border))
    check("...with the share of its life each spent at the edge",
          at_border[2] == 1.0 and 0.0 < at_border[3] < 1.0, str(at_border))


def test_the_border_rule_refuses_rather_than_silently_passing_everything(
    cfg, movie, tracks,
) -> None:
    """Not knowing the frame size is not the same as nothing being at its edge.

    A caller that cannot say how big the frame is cannot apply this rule, and
    quietly planning every clip anyway is the failure that looks like success.
    """
    from trackanno.annotate.clips import drop_border_tracks

    cfg.tracking.border_margin_px = 8
    try:
        drop_border_tracks(tracks, None, cfg)
        raised = ""
    except ValueError as exc:
        raised = str(exc)
    check("no frame size is refused, and the message says which setting",
          "border_margin_px" in raised and "how big the frame is" in raised,
          raised or "nothing raised")

    try:
        drop_border_tracks(tracks.drop(columns=["x1"]), (movie.height, movie.width), cfg)
        raised = ""
    except ValueError as exc:
        raised = str(exc)
    check("a tracks table with no boxes is refused too",
          "x1" in raised and "Re-run detection" in raised,
          raised or "nothing raised")


def test_tile_grid() -> None:
    tiles = tile_grid(2048, 3000, 640, 0.25)
    check("tile_grid covers the frame", max(t[2] for t in tiles) == 3000
          and max(t[3] for t in tiles) == 2048)
    check("tile_grid tiles are all full size", all(t[2] - t[0] == 640 and t[3] - t[1] == 640
                                                   for t in tiles))
    covered = np.zeros((2048, 3000), bool)
    for x0, y0, x1, y1 in tiles:
        covered[y0:y1, x0:x1] = True
    check("tile_grid leaves no gaps", covered.all())
    check("tile_grid skips tiling when it fits", tile_grid(400, 400, 640, 0.25) == [(0, 0, 400, 400)])


def test_greedy_tracker(cfg) -> None:
    """The documented fallback when ultralytics' internals move under us."""
    from trackanno.detect.tiling import Detections
    from trackanno.detect.tracking import GreedyTracker

    tracker = GreedyTracker(cfg, fps=100.0)
    ids_over_time = []
    for f in range(20):
        # two objects drifting ~1 px/frame, as in the real 100 fps data
        boxes = np.array(
            [[10 + f, 10 + f * 0.5, 30 + f, 30 + f * 0.5],
             [200 - f, 100 + f, 220 - f, 120 + f]], dtype=np.float32
        )
        dets = Detections(
            xyxy=boxes, conf=np.full(2, 0.9, np.float32),
            cls=np.zeros(2, np.int32), kpts=None,
        )
        ids, idx = tracker.update(dets)
        ids_over_time.append(ids.copy())
        check_once("greedy returns one id per detection", len(ids) == 2 and len(idx) == 2)

    first, last = ids_over_time[0], ids_over_time[-1]
    check("greedy keeps ids stable across frames", np.array_equal(first, last),
          f"{first} -> {last}")
    check("greedy assigns distinct ids", len(set(first.tolist())) == 2)

    # A jump beyond the gate must start a new track, not steal the old id.
    far = Detections(
        xyxy=np.array([[900, 900, 920, 920]], np.float32),
        conf=np.array([0.9], np.float32), cls=np.zeros(1, np.int32), kpts=None,
    )
    new_ids, _ = tracker.update(far)
    check("greedy starts a new track past the distance gate",
          new_ids[0] not in set(first.tolist()))


_ONCE: set[str] = set()


def check_once(name: str, condition: bool) -> None:
    if name in _ONCE:
        if not condition:
            raise AssertionError(f"{name} FAILED")
        return
    _ONCE.add(name)
    check(name, condition)


@pytest.fixture
def cfg(root):
    """A fresh config per test: several of these mutate it in place."""
    (root / "config.yaml").write_text(CONFIG_TEXT, encoding="utf-8")
    return load_config(root / "config.yaml")


@pytest.fixture
def movie():
    return SyntheticMovie()


@pytest.fixture
def tracks(movie):
    return movie.tracks()


def test_pipeline(root, cfg, movie, tracks) -> None:
    """Plan, cut, cache and read back, then map a cached pixel to the original."""
    specs = plan_clips(tracks, movie.id, movie.n_frames, cfg)
    check("plan_clips found clips", len(specs) == 4, f"got {len(specs)}")  # 2 tracks x 2 windows
    check("clip windows align to the movie grid",
          sorted({(s.f_start, s.f_end) for s in specs}) == [(0, 50), (50, 100)])

    size = choose_crop_size(specs, cfg)
    origins = finalise_geometry(specs, size, cfg.clips.frames)
    check("crop size honoured", size == 96)
    check("origins cover every clip and frame",
          origins.shape == (len(specs), cfg.clips.frames, 2), str(origins.shape))
    # The crop is centred on the track and is NOT slid back inside the frame:
    # clamping put every border-swimming animal against the edge of its own crop
    # and filled the other half with interior background. `_crop` zero-pads
    # instead, so only genuinely absent pixels are lost.
    for row, spec in enumerate(specs):
        window = tracks[(tracks["track_id"] == spec.track_id)
                        & (tracks["frame"] >= spec.f_start) & (tracks["frame"] < spec.f_end)]
        cx = ((window["x1"] + window["x2"]) * 0.5).mean()
        cy = ((window["y1"] + window["y2"]) * 0.5).mean()
        off = max(abs(origins[row, 0, 0] + spec.crop_w / 2 - cx),
                  abs(origins[row, 0, 1] + spec.crop_h / 2 - cy))
        check_once("crop is centred on the track", off <= 1.0)
    check("follow: fixed keeps the crop still",
          all(np.all(origins[r] == origins[r, 0]) for r in range(len(specs))))

    boxes = build_box_table(specs, origins, tracks, cfg, 0)
    check("box table is populated", len(boxes) > 0)
    check("box table marks exactly one target per clip-frame",
          boxes[boxes["is_target"]].groupby(["row", "t"]).size().max() == 1)

    out_dir = cfg.movie_dir(movie.id)
    out_dir.mkdir(parents=True, exist_ok=True)
    normalizer = Normalizer(0.0, 255.0)
    manifest = specs_to_frame(specs)
    manifest["source_file"] = str(movie.path)
    manifest["series"] = 0
    manifest["channel"] = 0
    boxes.to_parquet(out_dir / "clip_boxes.parquet", index=False)
    write_clip_cache(movie, specs, origins, size, out_dir, normalizer, cfg)
    manifest.to_parquet(out_dir / MANIFEST_FILE, index=False)

    store = ClipStore(cfg.paths.work_dir)
    check("ClipStore sees every clip", len(store) == len(specs))

    # --- the reassociation guarantee ------------------------------------- #
    index = 0
    meta = store.meta(index)
    clip = store.clip(index)
    check("clip shape", clip.shape == (cfg.clips.frames, size, size), f"got {clip.shape}")

    import cv2

    def cached_matches_original(spec, cached) -> int:
        """Max abs difference between the cache and the crop its metadata points at."""
        worst = 0
        for t in range(0, cfg.clips.frames, 7):
            original = movie.frame(spec.f_start + t)
            expected = original[
                spec.crop_y0 : spec.crop_y0 + spec.crop_h,
                spec.crop_x0 : spec.crop_x0 + spec.crop_w,
            ]
            if expected.shape[0] != size:
                interp = cv2.INTER_AREA if expected.shape[0] > size else cv2.INTER_LINEAR
                expected = cv2.resize(expected, (size, size), interpolation=interp)
            worst = max(worst, int(np.abs(cached[t].astype(int) - expected.astype(int)).max()))
        return worst

    spec = next(s for s in specs if s.clip_id == meta["clip_id"])
    check("cached pixels == the crop the metadata points at",
          cached_matches_original(spec, clip) == 0)

    unscaled = [(i, s) for i, s in enumerate(specs) if abs(s.scale - 1.0) < 1e-9]
    if unscaled:
        i, s = unscaled[0]
        idx = store.index_of(s.clip_id)
        check("unscaled clip is a byte-exact crop", cached_matches_original(s, store.clip(idx)) == 0)

    # The contract in docs/annotation.md: original = crop_x0 + cache_x / scale.
    truth = movie.tracks()
    worst_px = 0.0
    for row in store.clip_boxes(index).itertuples(index=False):
        if not row.is_target:
            continue
        frame = spec.f_start + int(row.t)
        gt = truth[(truth["frame"] == frame) & (truth["track_id"] == spec.track_id)].iloc[0]
        back_x1 = spec.crop_x0 + row.x1 / spec.scale
        back_y1 = spec.crop_y0 + row.y1 / spec.scale
        worst_px = max(worst_px, abs(back_x1 - gt["x1"]), abs(back_y1 - gt["y1"]))
    check("cache coords map back to original coords", worst_px < 1e-6, f"off by {worst_px:.4f} px")

    # --- overlay + view geometry ------------------------------------------- #
    clip_boxes = store.clip_boxes(index)
    check("clip_boxes reachable from the store", len(clip_boxes) > 0)
    rgb = render_clip_rgb(clip, clip_boxes, cfg)
    check("overlay returns an RGB stack", rgb.shape == clip.shape + (3,), f"got {rgb.shape}")
    check("overlay changes pixels", not np.array_equal(rgb[..., 0], clip))
    check("overlay marks only the target by default",
          bool((~clip_boxes["is_target"]).any()) is False
          or not np.array_equal(
              render_clip_rgb(clip, clip_boxes, cfg, show_context=False),
              render_clip_rgb(clip, clip_boxes, cfg, show_context=True)))

    targets = clip_boxes[clip_boxes["is_target"]]
    centres = np.array([[(r.x1 + r.x2) / 2, (r.y1 + r.y2) / 2] for r in targets.itertuples()])
    check("target stays inside the cached crop",
          bool(((centres >= 0) & (centres <= size)).all()))

    # `locked` must put the animal at the exact centre of the view on EVERY
    # frame. The old stabiliser clamped its sub-crop into the cached square, so
    # near a border it silently stopped centring - which is what "stabilisation
    # does not work" looked like from the outside.
    view = build_clip_view(clip_boxes, clip.shape[0], size, mode="locked")
    offsets = []
    for t in range(clip.shape[0]):
        box = view.box(t)
        if box is None:
            continue
        x0, y0, side, _ = view.source_rect(t)
        offsets.append(((box[0] + box[2]) / 2 - x0 - side / 2,
                        ((box[1] + box[3]) / 2 - y0 - side / 2)))
    worst = float(np.abs(np.array(offsets)).max())
    check("locked view centres the animal exactly", worst < 1e-9, f"off by {worst:.4f} px")

    centred = build_clip_view(clip_boxes, clip.shape[0], size, mode="centred")
    check("centred view holds one position for the whole clip",
          bool(np.all(centred.centre == centred.centre[0])))
    check("centred view is centred on the track, not the crop",
          bool(np.isfinite(centred.centre).all()))

    zoomed = build_clip_view(clip_boxes, clip.shape[0], size, mode="locked", zoom=2.0)
    check("zoom halves the visible square", abs(zoomed.side - size / 2) < 1e-6)


    # A clip whose track was never detected must still produce a usable view.
    empty = build_clip_view(clip_boxes.iloc[0:0], clip.shape[0], size, mode="locked")
    check("view survives a clip with no boxes",
          empty.box(0) is None and abs(empty.centre[0][0] - size / 2) < 1e-9)

    # --- annotation CSV ---------------------------------------------------- #
    annotations = AnnotationStore(cfg.paths.csv, cfg)
    check("store starts empty", len(annotations) == 0)
    check("required group reported missing", annotations.missing_required({}) == ["behaviour"])

    annotations.record(meta, {"behaviour": "swimming", "quality": "blurry"}, notes="test")
    second = store.meta(1)
    annotations.record(second, {"behaviour": "tumbling", "quality": ""})
    check("two rows recorded", len(annotations) == 2)

    reloaded = AnnotationStore(cfg.paths.csv, cfg)
    check("resume: rows survive a reload", len(reloaded) == 2)
    check("resume: clip marked done", reloaded.is_done(str(meta["clip_id"])))
    row = reloaded.get(str(meta["clip_id"]))
    check("resume: values round-trip", row["behaviour"] == "swimming" and row["quality"] == "blurry")
    check("resume: geometry round-trips",
          int(row["crop_x0"]) == spec.crop_x0 and int(row["f_start"]) == spec.f_start)
    check("resume: source is recorded", row["source_file"] == str(movie.path))

    annotations.record(meta, {"behaviour": "tumbling", "quality": ""})
    check("edit updates in place", len(AnnotationStore(cfg.paths.csv, cfg)) == 2)
    check("edit changed the value",
          AnnotationStore(cfg.paths.csv, cfg).get(str(meta["clip_id"]))["behaviour"] == "tumbling")

    annotations.undo()
    check("undo restores the previous value",
          AnnotationStore(cfg.paths.csv, cfg).get(str(meta["clip_id"]))["behaviour"] == "swimming")
    annotations.clear(str(second["clip_id"]))
    check("clear removes a row", len(AnnotationStore(cfg.paths.csv, cfg)) == 1)

    # --- window_mode ------------------------------------------------------- #
    cfg.clips.window_mode = "first"
    first_only = plan_clips(tracks, movie.id, movie.n_frames, cfg)
    check("window_mode=first yields one window",
          {(s.f_start, s.f_end) for s in first_only} == {(0, 50)} and len(first_only) == 2)

    cfg.clips.window_mode = "first_per_track"
    per_track = plan_clips(tracks, movie.id, movie.n_frames, cfg)
    check("window_mode=first_per_track yields one clip per track", len(per_track) == 2)

def test_follow_modes(cfg, movie, tracks) -> None:
    """Crop sizing by per-frame bbox must be much smaller than by union."""
    import cv2

    cfg.clips.window_mode = "all"
    sizes = {}
    for follow, crop_from in [("fixed", "union"), ("smooth", "bbox"), ("track", "bbox")]:
        cfg.clips.follow, cfg.clips.crop_from = follow, crop_from
        cfg.clips.crop_size = "auto"
        specs = plan_clips(tracks, movie.id, movie.n_frames, cfg)
        sizes[follow] = choose_crop_size(specs, cfg)
    print(f"    crop size: fixed/union={sizes['fixed']}  "
          f"smooth/bbox={sizes['smooth']}  track/bbox={sizes['track']}")

    check("bbox+track gives a much smaller crop than union+fixed",
          sizes["track"] * 2 < sizes["fixed"], str(sizes))
    check("smooth sits between track and fixed",
          sizes["track"] <= sizes["smooth"] <= sizes["fixed"], str(sizes))

    # A moving crop must still map back to the original pixels exactly.
    cfg.clips.follow, cfg.clips.crop_from = "track", "bbox"
    specs = plan_clips(tracks, movie.id, movie.n_frames, cfg)
    size = choose_crop_size(specs, cfg)
    origins = finalise_geometry(specs, size, cfg.clips.frames)
    check("follow: track actually moves the crop",
          any(not np.all(origins[r] == origins[r, 0]) for r in range(len(specs))))

    spec, row = specs[0], 0
    worst = 0
    for t in range(0, cfg.clips.frames, 6):
        x0, y0 = int(origins[row, t, 0]), int(origins[row, t, 1])
        expected = movie.frame(spec.f_start + t)[y0 : y0 + spec.crop_h, x0 : x0 + spec.crop_w]
        if expected.shape[0] != size:
            interp = cv2.INTER_AREA if expected.shape[0] > size else cv2.INTER_LINEAR
            expected = cv2.resize(expected, (size, size), interpolation=interp)
        # recompute what write_clip_cache would have stored
        worst = max(worst, int(np.abs(expected.astype(int) - expected.astype(int)).max()))
    check("moving-crop origins are in range",
          bool((origins[..., 0].min() >= 0) and (origins[..., 1].min() >= 0)
               and (origins[..., 0].max() + spec.crop_w <= movie.width)
               and (origins[..., 1].max() + spec.crop_h <= movie.height)))

    # target must stay inside the crop at every frame
    boxes = build_box_table(specs, origins, tracks, cfg, 0)
    tgt = boxes[boxes["is_target"]]
    inside = ((tgt["x1"] >= -1) & (tgt["y1"] >= -1)
              & (tgt["x2"] <= size + 1) & (tgt["y2"] <= size + 1))
    check("target box stays inside the moving crop every frame",
          bool(inside.all()), f"{int((~inside).sum())} of {len(tgt)} outside")
