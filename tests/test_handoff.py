"""The detect -> clips handoff: what makes detections or clips stale, and what not."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from fixtures import StubDetector, no_detector, prep_project


def _detect(cfg, movie, frames):
    from trackanno.detect.prepare import detect_movie

    stub = StubDetector(frames, movie.n_frames)
    detect_movie(movie, stub, cfg, limit_frames=60)
    return stub


def test_tracker_ids_restart_for_every_movie(root: Path) -> None:
    """Track ids go into clip_ids, so two movies in one process must number alike."""
    from trackanno.detect.tiling import Detections
    from trackanno.detect.tracking import build_tracker

    cfg, _, _ = prep_project(root, "tracker-ids")
    cfg.tracking.backend = "tracktrack"

    def first_ids() -> list[int]:
        tracker = build_tracker(cfg, 100.0, (512, 512))
        seen: list[int] = []
        for f in range(10):
            boxes = np.array([[10 + f, 10, 30 + f, 30], [200, 100 + f, 220, 120 + f]],
                             np.float32)
            ids, _ = tracker.update(Detections(
                xyxy=boxes, conf=np.full(2, 0.9, np.float32),
                cls=np.zeros(2, np.int32), kpts=None,
            ))
            seen += [int(i) for i in ids if int(i) not in seen]
        return sorted(seen)

    one, two = first_ids(), first_ids()
    assert len(one) == 2, f"the first movie gets ids: {one}"
    assert one == two, f"the second movie starts from the same id: {one} vs {two}"
    assert one[0] == 1, f"the construction probe used none of them: {one}"


@pytest.fixture(scope="module")
def _detected_once(root: Path):
    cfg, movie, frames = prep_project(root, "fingerprint")
    _detect(cfg, movie, frames)
    path = cfg.movie_dir(movie.id) / "detect_done.json"
    return cfg.path, movie, path.read_text(encoding="utf-8")


@pytest.fixture
def detected(_detected_once):
    """A freshly loaded config over one detected movie, its handoff record restored."""
    from trackanno.config import load_config

    config_path, movie, text = _detected_once
    path = load_config(config_path).movie_dir(movie.id) / "detect_done.json"
    path.write_text(text, encoding="utf-8")
    return load_config(config_path), movie, path, json.loads(text)


def _write_record(path: Path, info: dict, record: dict) -> None:
    path.write_text(json.dumps({**info, "config_fingerprint": record}), encoding="utf-8")


def test_a_handoff_is_fresh_after_detecting(detected) -> None:
    from trackanno.detect.prepare import handoff_is_fresh

    cfg, movie, _, _ = detected
    assert handoff_is_fresh(cfg, movie.id)


def test_a_changed_detection_key_marks_the_movie_stale(detected) -> None:
    from trackanno.detect.prepare import detection_changes, handoff_is_fresh

    cfg, movie, _, _ = detected
    cfg.detection.conf = 0.9
    assert not handoff_is_fresh(cfg, movie.id)
    assert detection_changes(cfg, movie.id) == ["detection.conf"]


def test_clip_stage_keys_do_not_mark_the_movie_stale(detected) -> None:
    from trackanno.detect.prepare import handoff_is_fresh

    cfg, movie, _, info = detected
    cfg.tracking.min_track_length = 7
    cfg.tracking.border_margin_px = 5
    cfg.tracking.border_max_frac = 0.5
    cfg.clips.pad = 3.0
    cfg.clips.crop_size = 64
    assert handoff_is_fresh(cfg, movie.id)
    assert not any(k.startswith("clips.") or k == "tracking.min_track_length"
                   for k in info["config_fingerprint"]), "the clip stage is not recorded"


def test_fps_marks_the_movie_stale(detected) -> None:
    """fps reaches the tracker through its track buffer."""
    from trackanno.detect.prepare import handoff_is_fresh

    cfg, movie, _, _ = detected
    cfg.tiff.fps = 33.0
    assert not handoff_is_fresh(cfg, movie.id)


def test_a_tile_override_does_not_change_the_fingerprint(detected) -> None:
    from trackanno.detect.prepare import (
        _config_fingerprint, handoff_is_fresh, pin_configured_tiling,
    )

    cfg, movie, _, _ = detected
    before = _config_fingerprint(cfg)
    pin_configured_tiling(cfg)
    cfg.tiling.tile, cfg.tiling.enabled = 128, True
    assert _config_fingerprint(cfg) == before
    assert handoff_is_fresh(cfg, movie.id)
    del cfg._configured_tiling
    assert not handoff_is_fresh(cfg, movie.id), "the same change in config.yaml is stale"


def test_the_fingerprint_names_the_inference_keys(detected) -> None:
    _, _, _, info = detected
    assert {"detection.conf", "detection.imgsz", "detection.half", "tiling.overlap",
            "tiling.merge_iou"} <= set(info["config_fingerprint"])


@pytest.mark.parametrize("key, other", [("tiling.tile", 320), ("detection.imgsz", 1280)])
def test_a_key_an_old_record_lacks_compares_at_its_default(detected, key, other) -> None:
    from trackanno.detect.prepare import handoff_is_fresh

    cfg, movie, path, info = detected
    record = dict(info["config_fingerprint"])
    record.pop(key)
    _write_record(path, info, record)
    assert handoff_is_fresh(cfg, movie.id), "fresh at the default"
    section, name = key.split(".")
    setattr(getattr(cfg, section), name, other)
    assert not handoff_is_fresh(cfg, movie.id), "stale away from it"
    record[key] = other
    _write_record(path, info, record)
    assert handoff_is_fresh(cfg, movie.id), "a record made at that value is fresh"


@pytest.mark.parametrize("value, fresh", [(0.5, True), (0.7, False)])
def test_a_retired_key_compares_against_its_constant(detected, value, fresh) -> None:
    from trackanno.detect.prepare import handoff_is_fresh

    cfg, movie, path, info = detected
    _write_record(path, info, {**info["config_fingerprint"], "detection.iou": value})
    assert handoff_is_fresh(cfg, movie.id) is fresh


def test_a_legacy_fingerprint_is_still_checked(detected) -> None:
    from trackanno.detect.prepare import _legacy_fingerprint, handoff_is_fresh

    cfg, movie, path, info = detected
    _write_record(path, info, _legacy_fingerprint(cfg))
    assert handoff_is_fresh(cfg, movie.id)
    cfg.detection.conf = 0.9
    assert not handoff_is_fresh(cfg, movie.id)


def test_model_weights_are_part_of_the_fingerprint(detected) -> None:
    from trackanno.detect.prepare import _config_fingerprint, handoff_is_fresh

    cfg, movie, path, info = detected
    record = _config_fingerprint(cfg)
    record["model.sha256"] = "0" * 64
    _write_record(path, info, record)
    assert handoff_is_fresh(cfg, movie.id), "unknown weights here are not a mismatch"
    cfg.model.sha256 = "1" * 64
    assert not handoff_is_fresh(cfg, movie.id), "different weights are"


def test_a_removed_normalize_mode_none_is_stale(detected) -> None:
    """`none` gave other pixels than `fixed`."""
    from trackanno.detect.prepare import (
        _legacy_digest, _legacy_fingerprint, _legacy_sections, handoff_is_fresh,
    )

    cfg, movie, path, info = detected
    cfg.normalize.mode = "fixed"
    _write_record(path, info, {**info["config_fingerprint"], "normalize.mode": "none"})
    assert not handoff_is_fresh(cfg, movie.id)
    normalize = {**_legacy_sections(cfg)["normalize"], "mode": "none"}
    _write_record(path, info, {**_legacy_fingerprint(cfg), "normalize": _legacy_digest(normalize)})
    assert not handoff_is_fresh(cfg, movie.id), "nor does a legacy digest made under it"


def test_a_removed_normalize_mode_smooth_matches_per_frame(detected) -> None:
    from trackanno.detect.prepare import (
        _legacy_digest, _legacy_fingerprint, _legacy_sections, handoff_is_fresh,
    )

    cfg, movie, path, info = detected
    cfg.normalize.mode = "per_frame"
    _write_record(path, info, {**info["config_fingerprint"], "normalize.mode": "smooth"})
    assert handoff_is_fresh(cfg, movie.id)
    cfg.normalize.mode = "fixed"
    assert not handoff_is_fresh(cfg, movie.id), "it does not match another mode"

    cfg.normalize.mode = "per_frame"
    normalize = {**_legacy_sections(cfg)["normalize"], "mode": "smooth"}
    _write_record(path, info, {**_legacy_fingerprint(cfg), "normalize": _legacy_digest(normalize)})
    assert handoff_is_fresh(cfg, movie.id), "a legacy digest made under it matches too"


def _logs():
    from trackanno.progress import CallbackReporter

    logs: list[str] = []
    return logs, CallbackReporter(on_log=lambda m, _l: logs.append(m))


def _stale(root: Path, name: str):
    """A detected movie whose config has since changed two detection keys."""
    cfg, movie, frames = prep_project(root, name)
    _detect(cfg, movie, frames)
    cfg.detection.conf = 0.3
    cfg.tiling.tile = 320
    return cfg, movie, frames


def test_a_stale_handoff_is_reported_not_redetected(root: Path) -> None:
    from trackanno.detect.prepare import detect_movie, prepare_movie, stale_reasons

    cfg, movie, frames = _stale(root, "stale-skip")
    tracks = cfg.movie_dir(movie.id) / "tracks.parquet"
    before = tracks.read_bytes()

    logs, reporter = _logs()
    stub = StubDetector(frames, movie.n_frames)
    info = prepare_movie(movie, stub, cfg, limit_frames=60, reporter=reporter)
    assert stub.calls == 0
    assert info.get("skipped") is True, info
    assert any("detection.conf" in m and "tiling.tile" in m and "--force" in m
               for m in logs), f"the log names the changed keys and the way out: {logs}"
    assert not any("FAILED" in m for m in logs), logs
    assert tracks.read_bytes() == before
    assert any("detection.conf" in r for r in stale_reasons(cfg, movie.id))

    logs, reporter = _logs()
    detect_movie(movie, stub, cfg, limit_frames=60, reporter=reporter)
    assert stub.calls == 0, "--stage detect skips it too"
    assert not any("already detected" in m for m in logs), logs


def test_force_redetects_a_stale_handoff(root: Path) -> None:
    from trackanno.detect.prepare import handoff_is_fresh, prepare_movie

    cfg, movie, frames = _stale(root, "stale-force")
    stub = StubDetector(frames, movie.n_frames)
    info = prepare_movie(movie, stub, cfg, force=True, limit_frames=60)
    assert stub.calls > 0
    assert info["n_clips"] > 0
    assert handoff_is_fresh(cfg, movie.id)


def test_a_truncated_handoff_is_skipped_and_force_replaces_it(root: Path) -> None:
    from trackanno.detect.prepare import prepare_movie

    cfg, movie, frames = prep_project(root, "truncated")
    _detect(cfg, movie, frames)
    path = cfg.movie_dir(movie.id) / "detect_done.json"
    path.write_text(path.read_text(encoding="utf-8")[:40], encoding="utf-8")
    logs, reporter = _logs()
    info = prepare_movie(movie, no_detector, cfg, limit_frames=60, reporter=reporter)
    assert info.get("skipped") is True, logs

    stub = StubDetector(frames, movie.n_frames)
    info = prepare_movie(movie, stub, cfg, force=True, limit_frames=60)
    assert stub.calls > 0 and info["n_clips"] > 0
    assert not list(cfg.movie_dir(movie.id).glob("*.tmp")), "no temporary file is left"


def _prepared(root: Path, name: str):
    from trackanno.detect.prepare import prepare_movie

    cfg, movie, frames = prep_project(root, name)
    prepare_movie(movie, StubDetector(frames, movie.n_frames), cfg, limit_frames=60)
    done = cfg.movie_dir(movie.id) / "prepare_done.json"
    return cfg, movie, done, json.loads(done.read_text(encoding="utf-8"))


def test_unchanged_clip_settings_keep_the_cache(root: Path) -> None:
    from trackanno.detect.prepare import prepare_movie

    cfg, movie, _, before = _prepared(root, "unchanged")
    assert before["clip_settings"]["clips.pad"] == 1.6, "the clip settings are recorded"
    assert prepare_movie(movie, no_detector, cfg) == before


def test_changed_clip_settings_are_reported_not_recut(root: Path) -> None:
    from trackanno.detect.prepare import cut_movie, prepare_movie, stale_reasons
    from trackanno.project import forget_path_probes, project_status

    cfg, movie, done, before = _prepared(root, "recut")
    cfg.clips.crop_size = 64
    logs, reporter = _logs()
    after = prepare_movie(movie, no_detector, cfg, reporter=reporter)
    assert after.get("skipped") is True, after
    assert any("clips.crop_size" in m and "--force" in m for m in logs), logs
    assert json.loads(done.read_text(encoding="utf-8")) == before, "the cache is left as it was"
    assert stale_reasons(cfg, movie.id) == ["clip settings changed (clips.crop_size)"]

    forget_path_probes()
    status = project_status(cfg)
    assert status.n_stale == 1 and status.n_prepared == 1, (
        f"counted as stale but still prepared: {status}")

    after = cut_movie(movie, cfg, force=True)
    assert after["crop_size"] == 64, "a forced cut re-cuts without a detector"
    forget_path_probes()
    assert project_status(cfg).n_stale == 0


def test_an_unreadable_clip_marker_is_skipped_and_a_forced_cut_replaces_it(root: Path) -> None:
    from trackanno.detect.prepare import cut_movie, prepare_movie

    cfg, movie, done, _ = _prepared(root, "unreadable")
    done.write_text("{", encoding="utf-8")
    assert prepare_movie(movie, no_detector, cfg).get("skipped") is True
    assert cut_movie(movie, cfg, force=True)["n_clips"] > 0


def test_a_legacy_handoff_survives_clip_stage_changes(root: Path) -> None:
    """Old per-section digests covered clip-stage keys; changing those is fresh."""
    from dataclasses import asdict

    from trackanno.detect.prepare import (
        _legacy_digest, _legacy_fingerprint, detection_changes, handoff_is_fresh,
        prepare_movie,
    )

    cfg, movie, frames = prep_project(root, "legacy")
    prepare_movie(movie, StubDetector(frames, movie.n_frames), cfg, limit_frames=60)
    path = cfg.movie_dir(movie.id) / "detect_done.json"
    info = json.loads(path.read_text(encoding="utf-8"))

    def write(record):
        path.write_text(json.dumps({**info, "config_fingerprint": record}),
                        encoding="utf-8")

    write(_legacy_fingerprint(cfg))
    cfg.tracking.min_track_length = 7
    cfg.tracking.border_margin_px = 5
    assert handoff_is_fresh(cfg, movie.id), (
        f"a clip-stage change is fresh: {detection_changes(cfg, movie.id)}")
    cfg.tracking.min_track_length = 40
    cfg.tracking.border_margin_px = 0

    tracking = {**asdict(cfg.tracking), "tracker_cfg": "", "greedy_max_dist_px": 25.0}
    tracking.pop("border_margin_px")
    tracking.pop("border_max_frac")
    write({**_legacy_fingerprint(cfg), "tracking": _legacy_digest(tracking)})
    assert handoff_is_fresh(cfg, movie.id), (
        f"a digest without the border keys is fresh: {detection_changes(cfg, movie.id)}")

    cfg.tracking.overrides = {"gmc_method": "sparseOptFlow"}
    assert detection_changes(cfg, movie.id) == ["tracking.*"], (
        "a detect-stage change in the same section is not")
    cfg.tracking.overrides = {"gmc_method": "none"}

    write({**_legacy_fingerprint(cfg), "tracking": "0" * 12})
    stub = StubDetector(frames, movie.n_frames)
    out = prepare_movie(movie, stub, cfg, limit_frames=60)
    assert out.get("skipped") is True, (
        f"an unrecoverable legacy digest is reported stale: {out}")
    assert stub.calls == 0, "never re-detected"


def test_cache_size_must_match_exactly(root: Path) -> None:
    from trackanno.annotate.clips import ClipStore
    from trackanno.detect.prepare import prepare_movie

    cfg, movie, frames = prep_project(root, "exact-size")
    prepare_movie(movie, StubDetector(frames, movie.n_frames), cfg, limit_frames=60)
    cache = cfg.movie_dir(movie.id) / "clips.dat"
    with open(cache, "ab") as fh:
        fh.write(b"\0" * 7)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        try:
            loaded = len(ClipStore(cfg.paths.work_dir))
        except FileNotFoundError:
            loaded = 0
    assert loaded == 0, "an oversized clips.dat is refused"


def _staged(out: Path):
    from trackanno.annotate.clips import HEADER_FILE, MANIFEST_FILE, StagedWrite

    out.mkdir()
    (out / HEADER_FILE).write_text("old", encoding="utf-8")
    staged = StagedWrite(out)
    for name in (HEADER_FILE, MANIFEST_FILE, "clips.dat"):
        staged.path(name).write_text("new", encoding="utf-8")
    return staged


def test_an_interrupted_commit_leaves_no_header(root: Path, monkeypatch) -> None:
    """A commit that dies part-way cannot load."""
    from trackanno.annotate.clips import HEADER_FILE

    out = root / "interrupted"
    staged = _staged(out)
    order: list[str] = []
    real = Path.replace

    def failing(self, target):
        order.append(Path(target).name)
        if len(order) == 2:
            raise OSError("killed")
        return real(self, target)

    monkeypatch.setattr(Path, "replace", failing)
    with pytest.raises(OSError, match="killed"):
        staged.commit()
    assert order[0] == "clips.dat", f"pixels go first: {order}"
    assert not (out / HEADER_FILE).exists()


def test_a_commit_renames_the_header_last(root: Path, monkeypatch) -> None:
    from trackanno.annotate.clips import HEADER_FILE, MANIFEST_FILE

    staged = _staged(root / "ordered")
    order: list[str] = []
    real = Path.replace
    monkeypatch.setattr(
        Path, "replace", lambda self, t: (order.append(Path(t).name), real(self, t))[1]
    )
    staged.commit()
    assert order == ["clips.dat", MANIFEST_FILE, HEADER_FILE]


class _Broken(Exception):
    pass


def test_a_failed_forced_redetect_leaves_no_done_marker(root: Path) -> None:
    from trackanno.detect.prepare import detect_movie

    cfg, movie, _, _ = _prepared(root, "redetect-fails")
    out = cfg.movie_dir(movie.id)

    class Failing:
        def is_tiled(self, *_):
            raise _Broken("detector died")

    with pytest.raises(_Broken):
        detect_movie(movie, Failing(), cfg, force=True, limit_frames=60)
    assert not (out / "detect_done.json").exists()
    assert not (out / "prepare_done.json").exists(), "clips cut from old tracks are not done"


def test_a_failed_forced_recut_leaves_no_done_marker(root: Path, monkeypatch) -> None:
    from trackanno.detect import prepare

    cfg, movie, done, _ = _prepared(root, "recut-fails")

    def broken(*_a, **_k):
        raise _Broken("cut died")

    monkeypatch.setattr(prepare, "plan_clips", broken)
    with pytest.raises(_Broken):
        prepare.cut_movie(movie, cfg, force=True)
    assert not done.exists()
