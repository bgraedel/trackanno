"""The detect -> clips handoff: what makes detections or clips stale, and what not."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from harness import check
from test_workflow import StubDetector, _no_detector, _prep_project


def _detect(cfg, movie, frames):
    from trackanno.detect.prepare import detect_movie

    stub = StubDetector(frames, movie.n_frames)
    detect_movie(movie, stub, cfg, limit_frames=60)
    return stub


def test_tracker_ids_restart_for_every_movie(root: Path) -> None:
    """Track ids go into clip_ids, so two movies in one process must number alike."""
    from trackanno.detect.tiling import Detections
    from trackanno.detect.tracking import build_tracker

    cfg, _, _ = _prep_project(root, "tracker-ids")
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
    check("the first movie gets ids", len(one) == 2, str(one))
    check("the second movie starts from the same id", one == two, f"{one} vs {two}")
    check("and the construction probe used none of them", one[0] == 1, str(one))


def test_what_makes_a_handoff_stale(root: Path) -> None:
    from trackanno.detect.prepare import (
        _config_fingerprint, _legacy_fingerprint, handoff_is_fresh, pin_configured_tiling,
    )

    cfg, movie, frames = _prep_project(root, "fingerprint")
    _detect(cfg, movie, frames)
    check("fresh after detecting", handoff_is_fresh(cfg, movie.id))

    cfg.tracking.min_track_length = 7
    cfg.tracking.border_margin_px = 5
    cfg.tracking.border_max_frac = 0.5
    cfg.clips.pad = 3.0
    check("clip-stage keys do not invalidate detections",
          handoff_is_fresh(cfg, movie.id))
    cfg.tiff.fps = 33.0
    check("fps does, through the tracker's track_buffer",
          not handoff_is_fresh(cfg, movie.id))
    cfg.tiff.fps = 100.0

    pin_configured_tiling(cfg)
    cfg.tiling.tile, cfg.tiling.enabled = 128, True
    check("a --tile override does not either", handoff_is_fresh(cfg, movie.id))
    del cfg._configured_tiling
    check("but the same change in config.yaml does", not handoff_is_fresh(cfg, movie.id))
    cfg.tiling.tile, cfg.tiling.enabled = 640, "auto"

    path = cfg.movie_dir(movie.id) / "detect_done.json"
    info = json.loads(path.read_text(encoding="utf-8"))
    check("the fingerprint names its keys", "detection.conf" in info["config_fingerprint"])
    check("and leaves the clip stage out",
          not any(k.startswith("clips.") or k == "tracking.min_track_length"
                  for k in info["config_fingerprint"]))

    # A key added in a later version is absent from older records.
    record = dict(info["config_fingerprint"])
    record.pop("tiling.tile")
    path.write_text(json.dumps({**info, "config_fingerprint": record}), encoding="utf-8")
    check("a key the record lacks is fine at its default", handoff_is_fresh(cfg, movie.id))
    cfg.tiling.tile = 320
    check("but not away from it", not handoff_is_fresh(cfg, movie.id))
    cfg.tiling.tile = 640

    # A recorded key that is no longer configurable compares against its constant.
    for value, fresh in ((0.5, True), (0.7, False)):
        path.write_text(json.dumps({**info, "config_fingerprint": {
            **info["config_fingerprint"], "detection.iou": value}}), encoding="utf-8")
        check(f"a recorded detection.iou of {value} is {'fresh' if fresh else 'stale'}",
              handoff_is_fresh(cfg, movie.id) is fresh)

    # Inference size: records without it were detected at the default.
    check("the inference keys are fingerprinted",
          {"detection.imgsz", "detection.half", "tiling.overlap", "tiling.merge_iou"}
          <= set(info["config_fingerprint"]))
    record = dict(info["config_fingerprint"])
    record.pop("detection.imgsz")
    path.write_text(json.dumps({**info, "config_fingerprint": record}), encoding="utf-8")
    check("no recorded imgsz is fresh at the default", handoff_is_fresh(cfg, movie.id))
    cfg.detection.imgsz = 1280
    check("...and stale at another size", not handoff_is_fresh(cfg, movie.id))
    record["detection.imgsz"] = 1280
    path.write_text(json.dumps({**info, "config_fingerprint": record}), encoding="utf-8")
    check("a record made at that size is fresh", handoff_is_fresh(cfg, movie.id))
    cfg.detection.imgsz = 640

    path.write_text(json.dumps({**info, "config_fingerprint": _legacy_fingerprint(cfg)}),
                    encoding="utf-8")
    check("an old per-section fingerprint is still accepted", handoff_is_fresh(cfg, movie.id))
    cfg.detection.conf = 0.9
    check("and still checked", not handoff_is_fresh(cfg, movie.id))
    cfg.detection.conf = 0.25

    record = _config_fingerprint(cfg)
    record["model.sha256"] = "0" * 64
    path.write_text(json.dumps({**info, "config_fingerprint": record}), encoding="utf-8")
    check("unknown weights here are not a mismatch", handoff_is_fresh(cfg, movie.id))
    cfg.model.sha256 = "1" * 64
    check("different weights are", not handoff_is_fresh(cfg, movie.id))


def test_a_removed_normalize_mode_is_not_stale(root: Path) -> None:
    """`smooth` matches per_frame; `none` gave other pixels than fixed, so it is stale."""
    from trackanno.detect.prepare import (
        _legacy_digest, _legacy_fingerprint, _legacy_sections, handoff_is_fresh,
    )

    cfg, movie, frames = _prep_project(root, "removed-mode")
    _detect(cfg, movie, frames)
    path = cfg.movie_dir(movie.id) / "detect_done.json"
    info = json.loads(path.read_text(encoding="utf-8"))

    def write(record):
        path.write_text(json.dumps({**info, "config_fingerprint": record}),
                        encoding="utf-8")

    cfg.normalize.mode = "fixed"
    write({**info["config_fingerprint"], "normalize.mode": "none"})
    check("'none' does not match 'fixed'", not handoff_is_fresh(cfg, movie.id))
    normalize = {**_legacy_sections(cfg)["normalize"], "mode": "none"}
    write({**_legacy_fingerprint(cfg), "normalize": _legacy_digest(normalize)})
    check("...nor does a legacy digest made under it", not handoff_is_fresh(cfg, movie.id))

    for old, new in (("smooth", "per_frame"),):
        cfg.normalize.mode = new
        write({**info["config_fingerprint"], "normalize.mode": old})
        check(f"{old!r} matches {new!r}", handoff_is_fresh(cfg, movie.id))
        cfg.normalize.mode = "fixed" if new == "per_frame" else "per_frame"
        check(f"{old!r} does not match another mode", not handoff_is_fresh(cfg, movie.id))

        cfg.normalize.mode = new
        normalize = {**_legacy_sections(cfg)["normalize"], "mode": old}
        write({**_legacy_fingerprint(cfg), "normalize": _legacy_digest(normalize)})
        check(f"a legacy digest made under {old!r} matches too",
              handoff_is_fresh(cfg, movie.id))
    cfg.normalize.mode = "per_frame"


def _logs():
    from trackanno.progress import CallbackReporter

    logs: list[str] = []
    return logs, CallbackReporter(on_log=lambda m, _l: logs.append(m))


def test_a_stale_handoff_is_reported_not_redetected(root: Path) -> None:
    from trackanno.detect.prepare import (
        detect_movie, handoff_is_fresh, prepare_movie, stale_reasons,
    )

    cfg, movie, frames = _prep_project(root, "stale-skip")
    _detect(cfg, movie, frames)
    tracks = cfg.movie_dir(movie.id) / "tracks.parquet"
    before = tracks.read_bytes()
    cfg.detection.conf = 0.3
    cfg.tiling.tile = 320

    logs, reporter = _logs()
    stub = StubDetector(frames, movie.n_frames)
    info = prepare_movie(movie, stub, cfg, limit_frames=60, reporter=reporter)
    check("a stale handoff is not re-detected", stub.calls == 0)
    check("the movie is reported as skipped", info.get("skipped") is True, str(info))
    check("the log names the changed keys and the way out",
          any("detection.conf" in m and "tiling.tile" in m and "--force" in m
              for m in logs), str(logs))
    check("and no FAILED", not any("FAILED" in m for m in logs), str(logs))
    check("the tracks are untouched", tracks.read_bytes() == before)
    check("stale_reasons names it",
          any("detection.conf" in r for r in stale_reasons(cfg, movie.id)))

    logs, reporter = _logs()
    detect_movie(movie, stub, cfg, limit_frames=60, reporter=reporter)
    check("--stage detect skips it too", stub.calls == 0)
    check("without calling it already detected",
          not any("already detected" in m for m in logs), str(logs))

    stub = StubDetector(frames, movie.n_frames)
    info = prepare_movie(movie, stub, cfg, force=True, limit_frames=60)
    check("--force re-detects", stub.calls > 0)
    check("and cuts the clips", info["n_clips"] > 0)
    check("leaving it fresh", handoff_is_fresh(cfg, movie.id))

    path = cfg.movie_dir(movie.id) / "detect_done.json"
    path.write_text(path.read_text(encoding="utf-8")[:40], encoding="utf-8")
    (cfg.movie_dir(movie.id) / "prepare_done.json").unlink(missing_ok=True)
    logs, reporter = _logs()
    info = prepare_movie(movie, _no_detector, cfg, limit_frames=60, reporter=reporter)
    check("a truncated handoff is skipped, not re-detected",
          info.get("skipped") is True, str(logs))
    stub = StubDetector(frames, movie.n_frames)
    info = prepare_movie(movie, stub, cfg, force=True, limit_frames=60)
    check("--force replaces it", stub.calls > 0 and info["n_clips"] > 0)
    check("and no temporary file is left behind",
          not list(cfg.movie_dir(movie.id).glob("*.tmp")))


def test_changed_clip_settings_are_reported_not_recut(root: Path) -> None:
    from trackanno.detect.prepare import cut_movie, prepare_movie, stale_reasons

    cfg, movie, frames = _prep_project(root, "recut")
    prepare_movie(movie, StubDetector(frames, movie.n_frames), cfg, limit_frames=60)
    done = cfg.movie_dir(movie.id) / "prepare_done.json"
    before = json.loads(done.read_text(encoding="utf-8"))
    check("the clip settings are recorded", before["clip_settings"]["clips.pad"] == 1.6)

    again = prepare_movie(movie, _no_detector, cfg)
    check("unchanged settings keep the cache", again == before)

    cfg.clips.crop_size = 64
    logs, reporter = _logs()
    after = prepare_movie(movie, _no_detector, cfg, reporter=reporter)
    check("a changed clip setting is skipped", after.get("skipped") is True, str(after))
    check("naming the key and how to re-cut",
          any("clips.crop_size" in m and "--force" in m for m in logs), str(logs))
    check("the cache is left as it was",
          json.loads(done.read_text(encoding="utf-8")) == before)
    check("stale_reasons names it",
          stale_reasons(cfg, movie.id) == ["clip settings changed (clips.crop_size)"],
          str(stale_reasons(cfg, movie.id)))

    from trackanno.project import forget_path_probes, project_status

    forget_path_probes()
    status = project_status(cfg)
    check("the project status counts it as stale but still prepared",
          status.n_stale == 1 and status.n_prepared == 1, str(status))

    after = cut_movie(movie, cfg, force=True)
    check("a forced cut re-cuts without a detector", after["crop_size"] == 64,
          str(after["crop_size"]))
    forget_path_probes()
    check("and is no longer stale", project_status(cfg).n_stale == 0)

    done.write_text("{", encoding="utf-8")
    after = prepare_movie(movie, _no_detector, cfg)
    check("an unreadable marker is skipped too", after.get("skipped") is True)
    after = cut_movie(movie, cfg, force=True)
    check("and a forced cut replaces it", after["n_clips"] > 0)


def test_a_legacy_handoff_survives_clip_stage_changes(root: Path) -> None:
    """Old per-section digests covered clip-stage keys; changing those is fresh."""
    from dataclasses import asdict

    from trackanno.detect.prepare import (
        _legacy_digest, _legacy_fingerprint, detection_changes, handoff_is_fresh,
        prepare_movie,
    )

    cfg, movie, frames = _prep_project(root, "legacy")
    prepare_movie(movie, StubDetector(frames, movie.n_frames), cfg, limit_frames=60)
    path = cfg.movie_dir(movie.id) / "detect_done.json"
    info = json.loads(path.read_text(encoding="utf-8"))

    def write(record):
        path.write_text(json.dumps({**info, "config_fingerprint": record}),
                        encoding="utf-8")

    write(_legacy_fingerprint(cfg))
    cfg.tracking.min_track_length = 7
    cfg.tracking.border_margin_px = 5
    check("a clip-stage change recovered from the clip marker is fresh",
          handoff_is_fresh(cfg, movie.id), str(detection_changes(cfg, movie.id)))
    cfg.tracking.min_track_length = 40
    cfg.tracking.border_margin_px = 0

    tracking = {**asdict(cfg.tracking), "tracker_cfg": "", "greedy_max_dist_px": 25.0}
    tracking.pop("border_margin_px")
    tracking.pop("border_max_frac")
    write({**_legacy_fingerprint(cfg), "tracking": _legacy_digest(tracking)})
    check("a digest from before the border keys existed is fresh",
          handoff_is_fresh(cfg, movie.id), str(detection_changes(cfg, movie.id)))

    cfg.tracking.overrides = {"gmc_method": "sparseOptFlow"}
    check("a detect-stage change in the same section is not",
          detection_changes(cfg, movie.id) == ["tracking.*"],
          str(detection_changes(cfg, movie.id)))
    cfg.tracking.overrides = {"gmc_method": "none"}

    write({**_legacy_fingerprint(cfg), "tracking": "0" * 12})
    stub = StubDetector(frames, movie.n_frames)
    out = prepare_movie(movie, stub, cfg, limit_frames=60)
    check("an unrecoverable legacy digest is reported stale",
          out.get("skipped") is True, str(out))
    check("but never re-detected", stub.calls == 0)


def test_cache_size_must_match_exactly(root: Path) -> None:
    from trackanno.annotate.clips import ClipStore
    from trackanno.detect.prepare import prepare_movie

    cfg, movie, frames = _prep_project(root, "exact-size")
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
    check("an oversized clips.dat is refused", loaded == 0)


def test_an_interrupted_commit_leaves_no_header(root: Path, monkeypatch) -> None:
    """The header is renamed last, so a commit that dies part-way cannot load."""
    from trackanno.annotate.clips import HEADER_FILE, MANIFEST_FILE, StagedWrite

    out = root / "commit"
    out.mkdir()
    (out / HEADER_FILE).write_text("old", encoding="utf-8")
    staged = StagedWrite(out)
    for name in (HEADER_FILE, MANIFEST_FILE, "clips.dat"):
        staged.path(name).write_text("new", encoding="utf-8")

    order: list[str] = []
    real = Path.replace

    def failing(self, target):
        order.append(Path(target).name)
        if len(order) == 2:
            raise OSError("killed")
        return real(self, target)

    monkeypatch.setattr(Path, "replace", failing)
    try:
        staged.commit()
    except OSError:
        pass
    check("pixels go first", order[0] == "clips.dat", str(order))
    check("the live header is gone mid-commit", not (out / HEADER_FILE).exists())

    monkeypatch.setattr(Path, "replace", real)
    staged = StagedWrite(out)
    for name in (HEADER_FILE, MANIFEST_FILE, "clips.dat"):
        staged.path(name).write_text("new", encoding="utf-8")
    order.clear()
    monkeypatch.setattr(
        Path, "replace", lambda self, t: (order.append(Path(t).name), real(self, t))[1]
    )
    staged.commit()
    check("the header is renamed last", order == ["clips.dat", MANIFEST_FILE, HEADER_FILE],
          str(order))
