"""Clip planning, crop geometry, the cache round-trip, overlays and the annotation CSV."""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest
from trackanno.annotate.clips import (
    ClipStore, MANIFEST_FILE, build_box_table, choose_crop_size, finalise_geometry,
    plan_clips, specs_to_frame, write_clip_cache,
)
from trackanno.config import load_config
from trackanno.video.movies import Normalizer
from trackanno.view.overlay import build_clip_view, render_clip_rgb
from trackanno.annotate.store import AnnotationStore
from trackanno.detect.tiling import tile_grid
from fixtures import (
    PIPELINE_CONFIG as CONFIG_TEXT, SyntheticMovie,
)


# --------------------------------------------------------------------------- #


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


def _expected_crop(movie, spec, t: int, x0: int, y0: int, size: int) -> np.ndarray:
    """The original pixels a cached frame claims to hold, resized as the cache does."""
    expected = movie.frame(spec.f_start + t)[y0 : y0 + spec.crop_h, x0 : x0 + spec.crop_w]
    if expected.shape[0] != size:
        interp = cv2.INTER_AREA if expected.shape[0] > size else cv2.INTER_LINEAR
        expected = cv2.resize(expected, (size, size), interpolation=interp)
    return expected


# --------------------------------------------------------------------------- #
# the frame border
# --------------------------------------------------------------------------- #
@pytest.fixture
def border_case(cfg, movie):
    """Track 1 stays central, 2 is pinned to the left edge, 3 spends its first window at the bottom."""
    length = cfg.clips.frames
    rows = []
    for f in range(2 * length):
        rows.append({"frame": f, "track_id": 1, "x1": 200.0, "y1": 100.0,
                     "x2": 220.0, "y2": 120.0, "conf": 0.9})
        rows.append({"frame": f, "track_id": 2, "x1": 0.0, "y1": 100.0,
                     "x2": 20.0, "y2": 120.0, "conf": 0.9})
        low = f < length
        rows.append({
            "frame": f, "track_id": 3, "x1": 200.0,
            "y1": float(movie.height - 20) if low else 100.0,
            "x2": 220.0,
            "y2": float(movie.height) if low else 120.0, "conf": 0.9,
        })
    cfg.tracking.min_track_length = 1
    cfg.clips.min_clip_frames = 1
    cfg.clips.stride = length
    cfg.clips.window_mode = "all"
    return pd.DataFrame(rows), (movie.height, movie.width), length


def test_the_border_rule_is_off_by_default(cfg, border_case) -> None:
    """An existing project plans exactly what it did before."""
    from trackanno.annotate.clips import drop_border_rows

    tracks, frame_size, _ = border_case
    assert cfg.tracking.border_margin_px == 0
    assert drop_border_rows(tracks, frame_size, cfg)[0] is tracks


def test_frames_at_the_frame_edge_are_flagged_and_dropped(cfg, border_case) -> None:
    from trackanno.annotate.clips import border_flags, drop_border_rows

    tracks, frame_size, length = border_case
    cfg.tracking.border_margin_px = 4
    at_edge = tracks[border_flags(tracks, frame_size, cfg).to_numpy()]
    assert 1 not in set(at_edge["track_id"])
    assert int((at_edge["track_id"] == 2).sum()) == 2 * length
    assert sorted(at_edge.loc[at_edge["track_id"] == 3, "frame"]) == list(range(length))

    kept, share = drop_border_rows(tracks, frame_size, cfg)
    assert len(kept) == len(tracks) - len(at_edge)
    assert int((kept["track_id"] == 3).sum()) == length, "the visitor keeps the rest"
    assert share[2] == 1.0 and share[3] == 0.5, f"a share per track: {share}"


def test_planning_drops_only_the_windows_spent_at_the_edge(cfg, movie, border_case) -> None:
    tracks, frame_size, length = border_case
    cfg.tracking.border_margin_px = 4
    at_border: dict = {}
    specs = plan_clips(tracks, movie.id, 2 * length, cfg,
                       frame_size=frame_size, at_border=at_border)
    made = sorted((s.track_id, s.f_start) for s in specs)
    assert [f for t, f in made if t == 1] == [0, length], made
    assert [f for t, f in made if t == 2] == [], made
    assert [f for t, f in made if t == 3] == [length], made
    # prepare warns from this dict, so it must be what was excluded.
    assert sorted(at_border) == [2, 3], at_border
    assert at_border[2] == 1.0 and at_border[3] == 0.5, at_border


def test_the_border_rule_refuses_without_a_frame_size(cfg, tracks) -> None:
    from trackanno.annotate.clips import drop_border_rows

    cfg.tracking.border_margin_px = 8
    with pytest.raises(ValueError, match="border_margin_px") as caught:
        drop_border_rows(tracks, None, cfg)
    assert "frame size" in str(caught.value)


def test_the_border_rule_refuses_tracks_without_boxes(cfg, movie, tracks) -> None:
    from trackanno.annotate.clips import drop_border_rows

    cfg.tracking.border_margin_px = 8
    with pytest.raises(ValueError, match="re-run detection") as caught:
        drop_border_rows(tracks.drop(columns=["x1"]), (movie.height, movie.width), cfg)
    assert "x1" in str(caught.value)


# --------------------------------------------------------------------------- #
# tiling and tracking
# --------------------------------------------------------------------------- #
def test_tile_grid() -> None:
    tiles = tile_grid(2048, 3000, 640, 0.25)
    assert max(t[2] for t in tiles) == 3000 and max(t[3] for t in tiles) == 2048
    assert all(t[2] - t[0] == 640 and t[3] - t[1] == 640 for t in tiles), "full-size tiles"
    covered = np.zeros((2048, 3000), bool)
    for x0, y0, x1, y1 in tiles:
        covered[y0:y1, x0:x1] = True
    assert covered.all(), "no gaps"
    assert tile_grid(400, 400, 640, 0.25) == [(0, 0, 400, 400)], "no tiling when it fits"


def test_greedy_tracker(cfg) -> None:
    from trackanno.detect.tiling import Detections
    from trackanno.detect.tracking import GreedyTracker

    tracker = GreedyTracker(cfg, fps=100.0)
    ids_over_time = []
    for f in range(20):
        # Two objects drifting about 1 px/frame.
        boxes = np.array(
            [[10 + f, 10 + f * 0.5, 30 + f, 30 + f * 0.5],
             [200 - f, 100 + f, 220 - f, 120 + f]], dtype=np.float32
        )
        ids, idx = tracker.update(Detections(
            xyxy=boxes, conf=np.full(2, 0.9, np.float32),
            cls=np.zeros(2, np.int32), kpts=None,
        ))
        ids_over_time.append(ids.copy())
        assert len(ids) == 2 and len(idx) == 2, "one id per detection"

    first, last = ids_over_time[0], ids_over_time[-1]
    assert np.array_equal(first, last), f"stable ids: {first} -> {last}"
    assert len(set(first.tolist())) == 2

    # A jump beyond the gate starts a new track.
    far = Detections(
        xyxy=np.array([[900, 900, 920, 920]], np.float32),
        conf=np.array([0.9], np.float32), cls=np.zeros(1, np.int32), kpts=None,
    )
    new_ids, _ = tracker.update(far)
    assert new_ids[0] not in set(first.tolist())


# --------------------------------------------------------------------------- #
# plan, cut, cache and read back
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """A planned, cut and cached project, shared read-only by the tests below."""
    home = tmp_path_factory.mktemp("pipeline-built")
    (home / "config.yaml").write_text(CONFIG_TEXT, encoding="utf-8")
    cfg = load_config(home / "config.yaml")
    movie = SyntheticMovie()
    tracks = movie.tracks()

    specs = plan_clips(tracks, movie.id, movie.n_frames, cfg)
    size = choose_crop_size(specs, cfg)
    origins = finalise_geometry(specs, size, cfg.clips.frames)
    boxes = build_box_table(specs, origins, tracks, cfg, 0)

    out_dir = cfg.movie_dir(movie.id)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = specs_to_frame(specs)
    manifest["source_file"] = str(movie.path)
    manifest["series"] = 0
    manifest["channel"] = 0
    boxes.to_parquet(out_dir / "clip_boxes.parquet", index=False)
    write_clip_cache(movie, specs, origins, size, out_dir, Normalizer(0.0, 255.0), cfg)
    manifest.to_parquet(out_dir / MANIFEST_FILE, index=False)

    store = ClipStore(cfg.paths.work_dir)
    meta = store.meta(0)
    yield SimpleNamespace(
        cfg=cfg, movie=movie, tracks=tracks, specs=specs, size=size, origins=origins,
        boxes=boxes, store=store, meta=meta, clip=store.clip(0),
        clip_boxes=store.clip_boxes(0),
        spec=next(s for s in specs if s.clip_id == meta["clip_id"]),
    )
    store.close()


def test_clips_are_planned_on_the_movie_grid(built) -> None:
    assert len(built.specs) == 4, "2 tracks x 2 windows"
    assert sorted({(s.f_start, s.f_end) for s in built.specs}) == [(0, 50), (50, 100)]


def test_a_fixed_crop_is_centred_on_the_track_and_still(built) -> None:
    """The crop is not slid back inside the frame; ``_crop`` zero-pads instead."""
    specs, origins, tracks = built.specs, built.origins, built.tracks
    assert built.size == 96, "crop size honoured"
    assert origins.shape == (len(specs), built.cfg.clips.frames, 2)
    for row, spec in enumerate(specs):
        window = tracks[(tracks["track_id"] == spec.track_id)
                        & (tracks["frame"] >= spec.f_start) & (tracks["frame"] < spec.f_end)]
        cx = ((window["x1"] + window["x2"]) * 0.5).mean()
        cy = ((window["y1"] + window["y2"]) * 0.5).mean()
        off = max(abs(origins[row, 0, 0] + spec.crop_w / 2 - cx),
                  abs(origins[row, 0, 1] + spec.crop_h / 2 - cy))
        assert off <= 1.0, f"clip {row} is {off:.2f} px off the track"
    assert all(np.all(origins[r] == origins[r, 0]) for r in range(len(specs)))


def test_the_box_table_marks_one_target_per_clip_frame(built) -> None:
    assert len(built.boxes) > 0
    assert built.boxes[built.boxes["is_target"]].groupby(["row", "t"]).size().max() == 1


def test_cached_pixels_are_the_crop_the_metadata_points_at(built) -> None:
    assert len(built.store) == len(built.specs)
    assert built.clip.shape == (built.cfg.clips.frames, built.size, built.size)

    def worst(spec, cached) -> int:
        return max(
            int(np.abs(cached[t].astype(int) - _expected_crop(
                built.movie, spec, t, spec.crop_x0, spec.crop_y0, built.size
            ).astype(int)).max())
            for t in range(0, built.cfg.clips.frames, 7))

    assert worst(built.spec, built.clip) == 0
    for spec in built.specs:
        if abs(spec.scale - 1.0) < 1e-9:
            assert worst(spec, built.store.clip(built.store.index_of(spec.clip_id))) == 0, (
                "an unscaled clip is a byte-exact crop")
            break


def test_cache_coordinates_map_back_to_the_movie(built) -> None:
    """original = crop_x0 + cache_x / scale."""
    spec, truth = built.spec, built.tracks
    worst_px = 0.0
    for row in built.clip_boxes.itertuples(index=False):
        if not row.is_target:
            continue
        frame = spec.f_start + int(row.t)
        gt = truth[(truth["frame"] == frame) & (truth["track_id"] == spec.track_id)].iloc[0]
        worst_px = max(worst_px,
                       abs(spec.crop_x0 + row.x1 / spec.scale - gt["x1"]),
                       abs(spec.crop_y0 + row.y1 / spec.scale - gt["y1"]))
    assert worst_px < 1e-6, f"off by {worst_px:.4f} px"


def test_the_overlay_draws_the_target_inside_the_crop(built) -> None:
    clip, clip_boxes, cfg = built.clip, built.clip_boxes, built.cfg
    assert len(clip_boxes) > 0
    rgb = render_clip_rgb(clip, clip_boxes, cfg)
    assert rgb.shape == clip.shape + (3,)
    assert not np.array_equal(rgb[..., 0], clip), "the overlay changes pixels"
    if (~clip_boxes["is_target"]).any():
        assert not np.array_equal(
            render_clip_rgb(clip, clip_boxes, cfg, show_context=False),
            render_clip_rgb(clip, clip_boxes, cfg, show_context=True)), (
            "context boxes are drawn only on request")
    targets = clip_boxes[clip_boxes["is_target"]]
    centres = np.array([[(r.x1 + r.x2) / 2, (r.y1 + r.y2) / 2] for r in targets.itertuples()])
    assert bool(((centres >= 0) & (centres <= built.size)).all())


def test_a_locked_view_centres_the_animal_on_every_frame(built) -> None:
    clip, size = built.clip, built.size
    view = build_clip_view(built.clip_boxes, clip.shape[0], size, mode="locked")
    offsets = []
    for t in range(clip.shape[0]):
        box = view.box(t)
        if box is None:
            continue
        x0, y0, side, _ = view.source_rect(t)
        offsets.append(((box[0] + box[2]) / 2 - x0 - side / 2,
                        (box[1] + box[3]) / 2 - y0 - side / 2))
    worst = float(np.abs(np.array(offsets)).max())
    assert worst < 1e-9, f"off by {worst:.4f} px"


def test_a_centred_view_holds_one_position(built) -> None:
    centred = build_clip_view(built.clip_boxes, built.clip.shape[0], built.size, mode="centred")
    assert bool(np.all(centred.centre == centred.centre[0]))
    assert bool(np.isfinite(centred.centre).all())


def test_zoom_halves_the_visible_square(built) -> None:
    zoomed = build_clip_view(built.clip_boxes, built.clip.shape[0], built.size,
                             mode="locked", zoom=2.0)
    assert abs(zoomed.side - built.size / 2) < 1e-6


def test_a_view_survives_a_clip_with_no_boxes(built) -> None:
    size = built.size
    empty = build_clip_view(built.clip_boxes.iloc[0:0], built.clip.shape[0], size, mode="locked")
    assert empty.box(0) is None
    assert abs(empty.centre[0][0] - size / 2) < 1e-9


def test_the_annotation_csv_resumes_edits_undoes_and_clears(built, tmp_path) -> None:
    """One chain: each step reads the state the previous one wrote."""
    cfg, meta, spec = built.cfg, built.meta, built.spec
    csv = tmp_path / "annotations.csv"
    annotations = AnnotationStore(csv, cfg)
    assert len(annotations) == 0
    assert annotations.missing_required({}) == ["behaviour"]

    annotations.record(meta, {"behaviour": "swimming", "quality": "blurry"}, notes="test")
    second = built.store.meta(1)
    annotations.record(second, {"behaviour": "tumbling", "quality": ""})
    assert len(annotations) == 2

    reloaded = AnnotationStore(csv, cfg)
    assert len(reloaded) == 2, "rows survive a reload"
    assert reloaded.is_done(str(meta["clip_id"]))
    row = reloaded.get(str(meta["clip_id"]))
    assert row["behaviour"] == "swimming" and row["quality"] == "blurry"
    assert int(row["crop_x0"]) == spec.crop_x0 and int(row["f_start"]) == spec.f_start
    assert row["source_file"] == str(built.movie.path)

    annotations.record(meta, {"behaviour": "tumbling", "quality": ""})
    assert len(AnnotationStore(csv, cfg)) == 2, "an edit updates in place"
    assert AnnotationStore(csv, cfg).get(str(meta["clip_id"]))["behaviour"] == "tumbling"

    annotations.undo()
    assert AnnotationStore(csv, cfg).get(str(meta["clip_id"]))["behaviour"] == "swimming"
    annotations.clear(str(second["clip_id"]))
    assert len(AnnotationStore(csv, cfg)) == 1


@pytest.mark.parametrize("mode,windows,count", [
    ("first", {(0, 50)}, 2),
    ("first_per_track", None, 2),
])
def test_window_mode(cfg, movie, tracks, mode, windows, count) -> None:
    cfg.clips.window_mode = mode
    specs = plan_clips(tracks, movie.id, movie.n_frames, cfg)
    assert len(specs) == count
    if windows is not None:
        assert {(s.f_start, s.f_end) for s in specs} == windows


# --------------------------------------------------------------------------- #
# follow modes
# --------------------------------------------------------------------------- #
def test_per_frame_box_sizing_gives_a_smaller_crop_than_the_union(
    cfg, movie, tracks, monkeypatch,
) -> None:
    import trackanno.annotate.clips as clips_module

    cfg.clips.window_mode = "all"
    cfg.clips.crop_size = "auto"
    sizes = {}
    for follow, crop_from in [("fixed", "union"), ("smooth", "bbox"), ("track", "bbox")]:
        cfg.clips.follow = follow
        monkeypatch.setattr(clips_module, "CROP_FROM", crop_from)
        sizes[follow] = choose_crop_size(plan_clips(tracks, movie.id, movie.n_frames, cfg), cfg)
    assert sizes["track"] * 2 < sizes["fixed"], sizes
    assert sizes["track"] <= sizes["smooth"] <= sizes["fixed"], sizes


def test_a_moving_crop_stays_in_range_and_on_the_target(cfg, movie, tracks, monkeypatch) -> None:
    import trackanno.annotate.clips as clips_module

    monkeypatch.setattr(clips_module, "CROP_FROM", "bbox")
    cfg.clips.window_mode = "all"
    cfg.clips.crop_size = "auto"
    cfg.clips.follow = "track"
    specs = plan_clips(tracks, movie.id, movie.n_frames, cfg)
    size = choose_crop_size(specs, cfg)
    origins = finalise_geometry(specs, size, cfg.clips.frames)
    assert any(not np.all(origins[r] == origins[r, 0]) for r in range(len(specs))), (
        "follow: track moves the crop")

    spec = specs[0]
    assert origins[..., 0].min() >= 0 and origins[..., 1].min() >= 0
    assert origins[..., 0].max() + spec.crop_w <= movie.width
    assert origins[..., 1].max() + spec.crop_h <= movie.height

    tgt = build_box_table(specs, origins, tracks, cfg, 0)
    tgt = tgt[tgt["is_target"]]
    inside = ((tgt["x1"] >= -1) & (tgt["y1"] >= -1)
              & (tgt["x2"] <= size + 1) & (tgt["y2"] <= size + 1))
    assert bool(inside.all()), f"{int((~inside).sum())} of {len(tgt)} outside"
