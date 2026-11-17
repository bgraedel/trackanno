"""Projects, config editing, prepare stages, background jobs and the workflow window."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
import pytest
import yaml
from qtpy.QtCore import Qt
from fixtures import (
    PREPARE_CONFIG, StubDetector, make_project, no_detector, prep_project, pump,
)

# Generated GUI fixture projects live here for the whole run.


def _create_edited_project(root: Path) -> Path:
    """A project with two movies, a model and one edited category group."""
    from trackanno.project import create_project, write_config

    videos = root / "movies"
    videos.mkdir()
    (videos / "a.tif").write_bytes(b"")
    (videos / "b.tif").write_bytes(b"")
    (videos / "notes.txt").write_bytes(b"")
    weights = root / "best.pt"
    weights.write_bytes(b"")

    path = create_project(root / "proj", video_dir=videos, model=weights)
    write_config(path, {"clips.frames": 25, "clips.min_clip_frames": 20, "ui.fps": 12},
                 categories=[{"name": "state", "type": "single", "required": True,
                              "options": [{"name": "a", "key": "1", "color": "#4C9F70"},
                                          {"name": "b", "key": "2", "color": "#D95F02"}]}])
    return path


@pytest.fixture(scope="module")
def project_config(root):
    """An edited project shared by the read-only tests."""
    return _create_edited_project(root)


@pytest.fixture
def hub(app, tmp_path):
    """A workflow window on a fresh edited project, closed afterwards."""
    from trackanno.gui.hub import Hub

    window = Hub(str(_create_edited_project(tmp_path)))
    window.show()
    app.processEvents()
    yield window
    window.close()
    app.processEvents()


# --------------------------------------------------------------------------- #
# yamledit and the shipped config
# --------------------------------------------------------------------------- #
EDITS = {
    "clips.frames": 33,
    "ui.view_zoom": 2.5,
    "ui.show_context_tracks": True,
    "paths.model": "D:/weights/best.pt",
    "normalize.percentiles": [1.0, 99.0],
    "ui.a_brand_new_key": "hello",
    "an_entirely_new_section.key": 7,
}


def _edited_default() -> tuple[str, dict, dict]:
    from trackanno.project import DEFAULT_CONFIG
    from trackanno.yamledit import set_values

    text = set_values(DEFAULT_CONFIG, EDITS)
    return text, yaml.safe_load(DEFAULT_CONFIG), yaml.safe_load(text)


def test_set_values_lands_every_kind_of_edit() -> None:
    _, _, after = _edited_default()
    assert after["clips"]["frames"] == 33 and after["ui"]["view_zoom"] == 2.5
    assert after["paths"]["model"] == "D:/weights/best.pt", "a path with a colon survives quoting"
    assert after["normalize"]["percentiles"] == [1.0, 99.0]
    assert after["ui"]["a_brand_new_key"] == "hello", "new keys go into their section"
    assert after["an_entirely_new_section"]["key"] == 7, "new sections are appended"


def test_set_values_leaves_the_rest_and_the_comments() -> None:
    from trackanno.project import DEFAULT_CONFIG

    text, before, after = _edited_default()
    assert after["tracking"] == before["tracking"] and after["keys"] == before["keys"]
    assert text.count("#") >= DEFAULT_CONFIG.count("#"), "comments survive an edit"
    assert "33#" not in text and "33 #" in text or "frames: 33" in text, (
        "trailing comments stay separated from their value")


def test_replace_section_swaps_one_block() -> None:
    from trackanno.yamledit import replace_section

    text, before, _ = _edited_default()
    groups = [{"name": "state", "type": "single", "required": True,
               "options": [{"name": "a", "key": "1", "color": "#4C9F70"}]}]
    replaced = yaml.safe_load(replace_section(text, "categories", groups))
    assert replaced["categories"] == groups
    assert replaced["ui"]["view_zoom"] == 2.5 and replaced["render"] == before["render"], (
        "neighbouring blocks are untouched")


def test_the_default_config_is_a_shipped_yaml_file() -> None:
    """The package carries it as a yaml file read through importlib.resources."""
    from importlib import resources

    from trackanno.project import DEFAULT_CONFIG, DEFAULT_CONFIG_FILE

    shipped = resources.files("trackanno").joinpath(DEFAULT_CONFIG_FILE)
    assert shipped.is_file(), str(shipped)
    assert str(shipped).endswith(".yaml"), str(shipped)
    assert DEFAULT_CONFIG == shipped.read_text(encoding="utf-8")

    parsed = yaml.safe_load(DEFAULT_CONFIG)
    assert isinstance(parsed, dict) and "paths" in parsed and "clips" in parsed
    assert DEFAULT_CONFIG.count("#") > 50, "the shipped text is commented"
    assert "\t" not in DEFAULT_CONFIG and "\x0c" not in DEFAULT_CONFIG, (
        "no stray tab or control character")


def test_a_new_project_starts_from_the_shipped_text(root) -> None:
    from trackanno.project import DEFAULT_CONFIG, create_project

    written = Path(create_project(root / "from-the-file")).read_text(encoding="utf-8")
    assert written.count("#") == DEFAULT_CONFIG.count("#"), "comments and all"


# --------------------------------------------------------------------------- #
# projects
# --------------------------------------------------------------------------- #
def test_a_created_project_loads_with_its_edits(root, project_config) -> None:
    from trackanno.config import load_config

    path = project_config
    assert path.exists()
    cfg = load_config(path)
    assert cfg.paths.video_dir == (root / "movies").resolve()
    assert cfg.paths.work_dir == (root / "proj" / "work").resolve(), (
        "work_dir resolves inside the project")
    assert path.read_text(encoding="utf-8").count("#") > 40, "the config is documented"
    assert cfg.clips.frames == 25 and cfg.ui.fps == 12
    assert [g.name for g in cfg.categories] == ["state"]
    assert [o.name for o in cfg.categories[0].options] == ["a", "b"]


def test_the_status_of_an_unprepared_project(project_config) -> None:
    from trackanno.config import load_config
    from trackanno.project import project_status

    status = project_status(load_config(project_config))
    assert status.n_videos == 2, "only matching files are counted"
    assert status.model_ok
    assert status.n_prepared == 0 and status.n_clips == 0 and not status.can_annotate
    assert status.can_prepare


def test_an_incoherent_edit_never_reaches_disk(project_config) -> None:
    from trackanno.config import ConfigError, load_config
    from trackanno.project import write_config

    good = project_config.read_text(encoding="utf-8")
    with pytest.raises(ConfigError, match="min_clip_frames") as exc:
        write_config(project_config, {"clips.min_clip_frames": 999})
    assert "no window could ever qualify" in str(exc.value)
    assert project_config.read_text(encoding="utf-8") == good
    assert load_config(project_config).clips.frames == 25
    assert not list(project_config.parent.glob("*.tmp")), "no temp file is left behind"


def test_pose_without_keypoints_loads_and_is_resolvable(root: Path) -> None:
    """`task: pose` with no keypoints listed loads and asks for them."""
    from trackanno.config import load_config
    from trackanno.project import create_project, write_config

    path = create_project(root / "pose")
    write_config(path, {"detection.task": "pose"})
    cfg = load_config(path)
    assert cfg.is_pose
    assert cfg.needs_keypoints and cfg.n_keypoints == 0

    write_config(path, {"keypoints.nodes": ["nose", "tail"]})
    resolved = load_config(path)
    assert not resolved.needs_keypoints and resolved.n_keypoints == 2


def test_generated_keypoint_names() -> None:
    from trackanno.detect.tiling import default_node_names

    assert default_node_names(4) == ["kp0", "kp1", "kp2", "kp3"]


def test_an_unreadable_model_has_no_keypoint_count(root: Path) -> None:
    from trackanno.detect.tiling import read_keypoint_count

    fake = root / "not-a-model.pt"
    fake.write_bytes(b"not a checkpoint")
    with pytest.raises(RuntimeError, match="could not read"):
        read_keypoint_count(fake)


# --------------------------------------------------------------------------- #
# keypoint editor
# --------------------------------------------------------------------------- #
@pytest.fixture
def keypoints(app):
    from trackanno.gui.configform import KeypointEditor

    editor = KeypointEditor()
    editor.load(["Head", "Base", "Tip"], [["Head", "Base"], ["Base", "Tip"]])
    return editor


def test_the_keypoint_table_loads(keypoints) -> None:
    assert keypoints.table.rowCount() == 3
    assert keypoints.count.value() == 3, "the count follows the names"
    assert not (keypoints.table.item(0, 0).flags() & Qt.ItemIsEditable), (
        "the position column is read-only")
    assert keypoints.edges.rowCount() == 2


def test_a_model_count_resizes_the_names(keypoints) -> None:
    keypoints.set_count(5, "5 from model.pt")
    assert keypoints.names()[:3] == ["Head", "Base", "Tip"], "typed names are kept"
    assert keypoints.names()[3:] == ["kp3", "kp4"], "new rows get placeholders"
    assert "model.pt" in keypoints.state.text(), "the source is shown"
    keypoints.set_count(2)
    assert keypoints.names() == ["Head", "Base"], "shrinking drops from the end"


def test_chained_links_follow_a_rename(app, keypoints) -> None:
    keypoints.set_count(2)
    keypoints.set_count(4)
    keypoints._chain()
    assert keypoints.values()["edges"] == [["Head", "Base"], ["Base", "kp2"], ["kp2", "kp3"]]
    keypoints.table.item(2, 1).setText("Middle")
    app.processEvents()
    assert keypoints.values()["edges"][1] == ["Base", "Middle"], keypoints.values()["edges"]


def test_bad_keypoint_names_are_problems(app, keypoints) -> None:
    assert keypoints.problems() == []
    keypoints.table.item(1, 1).setText("Head")
    app.processEvents()
    assert any("duplicate" in p for p in keypoints.problems()), keypoints.problems()
    keypoints.table.item(1, 1).setText("")
    app.processEvents()
    assert any("needs a name" in p for p in keypoints.problems()), keypoints.problems()


# --------------------------------------------------------------------------- #
# theme and progress
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["dark", "light"])
def test_a_theme_applies_completely(app, name) -> None:
    from trackanno.gui.theme import THEMES, apply_theme, current, tokens

    try:
        assert apply_theme(app, name) == name and current() == name
        sheet = app.styleSheet()
        assert len(sheet) > 2000, len(sheet)
        assert "{t[" not in sheet and "None" not in sheet.split("QToolTip")[0][:200], (
            "no unresolved tokens")
        colours = {k for k in tokens() if not k.startswith("_")}
        assert colours == set(THEMES["dark"]), set(THEMES["dark"]) ^ colours
        assets = [Path(v) for k, v in tokens().items() if k.startswith("_arrow")]
        assert len(assets) == 3 and all(p.is_file() and p.stat().st_size for p in assets), (
            f"arrow images generated: {[(p.name, p.exists()) for p in assets]}")
        assert all(str(p).replace("\\", "/") in sheet for p in assets), (
            "the sheet points at the generated arrows")
    finally:
        apply_theme(app, "dark")


def test_system_and_unknown_themes_resolve(app) -> None:
    from trackanno.gui.theme import apply_theme

    assert apply_theme(app, "system") in ("dark", "light")
    assert apply_theme(app, "chartreuse") == "dark", "an unknown theme falls back"
    apply_theme(app, "dark")


def test_a_reporter_tracks_a_loop() -> None:
    from trackanno.progress import CallbackReporter

    seen: list = []
    reporter = CallbackReporter(
        on_log=lambda m, l: seen.append(("log", m, l)),
        on_stage=lambda label, total: seen.append(("stage", label, total)),
        on_progress=lambda done, total: seen.append(("progress", done, total)),
    )
    assert list(reporter.track(range(5), "counting", 5)) == [0, 1, 2, 3, 4]
    assert ("stage", "counting", 5) in seen
    assert ("progress", 5, 5) in seen


def test_a_cancelled_reporter_stops_the_loop() -> None:
    from trackanno.progress import CallbackReporter, Cancelled

    reporter = CallbackReporter()
    reporter.cancel()
    with pytest.raises(Cancelled):
        list(reporter.track(range(5), "counting", 5))


# --------------------------------------------------------------------------- #
# prepare
# --------------------------------------------------------------------------- #


POSE_PREPARE_CONFIG = PREPARE_CONFIG.replace(
    "detection: {task: detect, device: cpu}",
    "detection: {task: pose, device: cpu}\n"
    "keypoints: {nodes: [head, base], edges: [[head, base]]}",
)


class PoseStubDetector(StubDetector):
    """As StubDetector, with two keypoints on every box."""

    def detect(self, frame_rgb):
        from trackanno.detect.tiling import Detections

        xyxy, conf = self.by_frame[self.calls % self.n_frames]
        self.calls += 1
        mid = ((xyxy[:, 0] + xyxy[:, 2]) / 2).astype(np.float32)
        seen = np.ones(len(conf), np.float32)
        kpts = np.stack(
            [np.stack([mid, xyxy[:, 1], seen], axis=1),
             np.stack([mid, xyxy[:, 3], seen], axis=1)], axis=1
        ).astype(np.float32)
        return Detections(
            xyxy=xyxy, conf=conf, cls=np.zeros(len(conf), np.int32), kpts=kpts
        )


def test_prepare_reports_its_stages_and_progress(root: Path) -> None:
    """The whole detect, track and cut path, driven the way the GUI drives it."""
    from trackanno.annotate.clips import ClipStore
    from trackanno.detect.prepare import prepare_movie
    from trackanno.progress import CallbackReporter

    cfg, movie, frames = prep_project(root, "prep")
    stages, progress = [], []
    reporter = CallbackReporter(
        on_stage=lambda label, total: stages.append((label, total)),
        on_progress=lambda done, total: progress.append(done),
    )
    info = prepare_movie(movie, StubDetector(frames, movie.n_frames), cfg,
                         limit_frames=60, reporter=reporter)

    assert info["n_clips"] > 0, info
    names = [s for s, _ in stages]
    assert any("detect + track" in s for s in names), names
    assert any("cutting clips" in s for s in names), names
    assert any(total for _, total in stages), f"a total to count against: {stages}"
    assert max(progress) > 1, "per-frame progress"
    assert (cfg.movie_dir(movie.id) / "prepare_done.json").exists(), "a resume marker"

    store = ClipStore(cfg.paths.work_dir)
    assert len(store) == info["n_clips"]
    assert len(store.clip_boxes(0)) > 0, "the clips carry overlay geometry"


def test_a_cancelled_prepare_stops_inside_the_movie(root: Path) -> None:
    from trackanno.detect.prepare import prepare_movie
    from trackanno.progress import CallbackReporter, Cancelled

    cfg, movie, frames = prep_project(root, "prep-cancel")
    stopper = CallbackReporter(on_progress=lambda done, total: None)
    stopper.cancel()
    with pytest.raises(Cancelled):
        prepare_movie(movie, StubDetector(frames, movie.n_frames), cfg,
                      force=True, limit_frames=60, reporter=stopper)


def test_the_two_stages_match_one_pass(root: Path) -> None:
    """Detecting and cutting separately gives the same cache as one pass."""
    from trackanno.annotate.clips import ClipStore
    from trackanno.detect.prepare import cut_movie, detect_movie, prepare_movie

    one, movie, frames = prep_project(root, "one-pass")
    whole = prepare_movie(movie, StubDetector(frames, movie.n_frames), one, limit_frames=60)

    split, movie2, frames2 = prep_project(root, "split")
    handoff = detect_movie(
        movie2, StubDetector(frames2, movie2.n_frames), split, limit_frames=60
    )
    out = split.movie_dir(movie2.id)
    assert (out / "detect_done.json").exists(), "stage 1 leaves a handoff"
    assert not (out / "prepare_done.json").exists(), "stage 1 does not mark the movie prepared"
    assert (out / "tracks.parquet").exists() and (out / "norm.json").exists()
    assert not (out / "clips.dat").exists(), "stage 1 writes no clip pixels"
    assert {"keypoint_nodes", "tiled", "n_tiles", "feature_dim", "n_frames",
            "config_fingerprint"} <= set(handoff), sorted(handoff)

    part = cut_movie(movie2, split)
    for field in ("n_clips", "crop_size", "n_tracks", "n_detections", "tiled", "n_tiles"):
        assert part[field] == whole[field], f"{field}: {part[field]} vs {whole[field]}"

    a, b = ClipStore(one.paths.work_dir), ClipStore(split.paths.work_dir)
    assert len(a) == len(b) and len(a) > 0
    assert len(a.clip_boxes(0)) == len(b.clip_boxes(0)), "the same overlay geometry"


def test_a_run_started_elsewhere_is_finished_not_repeated(root: Path) -> None:
    """A movie detected elsewhere is cut without ever building a detector."""
    from trackanno.detect.prepare import detect_movie, prepare_movie
    from trackanno.project import forget_path_probes, project_status

    cfg, movie, frames = prep_project(root, "from-gpu-box")
    detect_movie(movie, StubDetector(frames, movie.n_frames), cfg, limit_frames=60)

    status = project_status(cfg)
    assert status.n_prepared == 0, "a detected movie is not prepared"
    assert status.n_clips == 0
    assert status.n_detected == 1

    info = prepare_movie(movie, no_detector, cfg)
    assert info["n_clips"] > 0
    assert info["n_tiles"] == 1, "the recorded tiling is used"

    # The status above cached a "no prepare_done.json" verdict.
    forget_path_probes()
    after = project_status(cfg)
    assert after.n_prepared == 1, after
    assert after.n_clips == info["n_clips"]


def test_pose_keypoint_names_survive_the_handoff(root: Path) -> None:
    """A machine that never loads the checkpoint gets the names from the handoff."""
    import pandas as pd

    from trackanno.config import load_config
    from trackanno.detect.prepare import cut_movie, detect_movie

    cfg, movie, frames = prep_project(root, "pose-handoff", POSE_PREPARE_CONFIG)
    handoff = detect_movie(
        movie, PoseStubDetector(frames, movie.n_frames), cfg, limit_frames=60
    )
    assert handoff["keypoint_nodes"] == ["head", "base"]

    laptop = load_config(cfg.path)
    laptop.keypoints.nodes = []
    assert laptop.n_keypoints == 0

    cut_movie(movie, laptop)
    assert list(laptop.keypoints.nodes) == ["head", "base"], "names come from the handoff"
    boxes = pd.read_parquet(cfg.movie_dir(movie.id) / "clip_boxes.parquet")
    assert {"kp_x_0", "kp_y_0", "kp_x_1", "kp_y_1"} <= set(boxes.columns), (
        sorted(c for c in boxes.columns if c.startswith("kp_")))


def test_detections_from_a_different_config_are_refused(root: Path) -> None:
    """A changed detection setting makes the handoff stale and names the section."""
    from trackanno.detect.prepare import (
        cut_movie, detect_movie, handoff_is_fresh, read_handoff,
    )

    cfg, movie, frames = prep_project(root, "stale")
    detect_movie(movie, StubDetector(frames, movie.n_frames), cfg, limit_frames=60)
    assert handoff_is_fresh(cfg, movie.id)

    cfg.detection.device = "cuda:0"
    cfg.detection.batch = 32
    assert handoff_is_fresh(cfg, movie.id), "device and batch size do not invalidate it"

    cfg.detection.conf = 0.9
    assert not handoff_is_fresh(cfg, movie.id), "a changed confidence threshold does"
    with pytest.raises(ValueError, match="detection"):
        read_handoff(cfg, movie.id)

    for force in (False, True):
        info = cut_movie(movie, cfg, force=force)
        assert info.get("skipped") is True, f"force={force}: {info}"
        assert not (cfg.movie_dir(movie.id) / "prepare_done.json").exists()


def test_border_exclusions_are_reported_by_the_planner(root: Path) -> None:
    """The warning names the tracks plan_clips excluded frames from."""
    from trackanno.detect.prepare import prepare_movie
    from trackanno.progress import CallbackReporter

    # Track 1 comes within 60px of the left edge; track 2 stays 181px clear.
    config = PREPARE_CONFIG.replace(
        "tracking: {backend: greedy, min_track_length: 40}",
        "tracking: {backend: greedy, min_track_length: 40,\n"
        "           border_margin_px: 70}",
    )
    cfg, movie, frames = prep_project(root, "border", config)
    assert cfg.tracking.border_margin_px == 70

    logs: list[str] = []
    info = prepare_movie(
        movie, StubDetector(frames, movie.n_frames), cfg, limit_frames=60,
        reporter=CallbackReporter(on_log=lambda m, level: logs.append(m)),
    )
    warned = [m for m in logs if "come within" in m]
    assert len(warned) == 1, logs
    assert warned[0].startswith("1 track(s)"), warned[0]
    assert info["n_clips"] > 0, "the track that stayed clear still became clips"


@pytest.fixture
def two_movie_folder(tmp_path: Path):
    """A folder with one readable TIFF and one file that is not a TIFF."""
    import tifffile

    cfg, movie, _ = prep_project(tmp_path, "one-of-many")
    cfg.paths.video_dir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(cfg.paths.video_dir / "wanted.tif"),
                     np.stack([movie.frame(i) for i in range(4)]),
                     photometric="minisblack")
    (cfg.paths.video_dir / "unreadable.tif").write_text("not a tiff", encoding="utf-8")
    return cfg


def test_asking_for_one_movie_does_not_open_the_others(two_movie_folder) -> None:
    import warnings as W

    from trackanno.video.movies import discover_movies

    with W.catch_warnings(record=True) as caught:
        W.simplefilter("always")
        picked = discover_movies(two_movie_folder, only=["wanted"])
    for source in picked:
        source.close()
    assert [s.id for s in picked] == ["wanted"]
    assert not [w for w in caught if "unreadable" in str(w.message)], (
        f"the rest of the folder was opened: {[str(w.message) for w in caught]}")


def test_an_unknown_movie_id_lists_what_is_there(two_movie_folder) -> None:
    import warnings as W

    from trackanno.video.movies import discover_movies

    with W.catch_warnings(record=True):
        W.simplefilter("always")
        with pytest.raises(ValueError, match="typo") as exc:
            discover_movies(two_movie_folder, only=["typo"])
    assert "wanted" in str(exc.value)


def test_a_cut_only_run_is_given_the_clip_stage(app, root: Path, monkeypatch) -> None:
    """A "Cut clips" run must not ask detected movies whether they need detecting."""
    import tifffile
    from qtpy.QtWidgets import QListWidgetItem

    from trackanno.annotate.clips import DONE_FILE
    from trackanno.detect import prepare as prepare_module
    from trackanno.detect.prepare import detect_movie
    from trackanno.gui.hub import PreparePage
    from trackanno.progress import NullReporter
    from trackanno.project import project_status

    cfg, movie, frames = prep_project(root, "cut-only")
    # A real movie file and deliberately no model.pt.
    cfg.paths.video_dir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        str(cfg.paths.video_dir / f"{movie.id}.tif"),
        np.stack([movie.frame(i) for i in range(60)]),
    )
    detect_movie(movie, StubDetector(frames, movie.n_frames), cfg, limit_frames=60)
    assert not cfg.paths.model.exists()

    class StubHub:
        def __init__(self):
            self.cfg, self.status = cfg, project_status(cfg)

        def release_caches(self):
            pass

    page = PreparePage(StubHub())
    assert page.blocker() == "", "detections without a model can open the page"

    def listed(movie_id: str, state: str) -> None:
        item = QListWidgetItem()
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setData(Qt.UserRole, movie_id)
        item.setData(Qt.UserRole + 1, {
            "movie_id": movie_id, "n_frames": 60, "width": movie.width,
            "height": movie.height, "fps": movie.fps,
        })
        page.movies.addItem(item)
        item.setCheckState(Qt.Checked)
        page._states[movie_id] = state

    listed(movie.id, "detected")
    page._update_run_label(page._states)
    assert page.tasks.run_button.text() == "Cut clips"

    seen: dict = {}
    monkeypatch.setattr(
        prepare_module, "prepare_all",
        lambda cfg, ids=None, force=False, limit=None,
        reporter=None, stage="all": seen.update(stage=stage, ids=ids) or [],
    )
    page._build_work()(NullReporter())
    assert seen.get("stage") == "clips", seen

    # One movie that still needs detecting puts the run back to 'all'.
    listed("undetected", "new")
    page._update_run_label(page._states)
    assert page.tasks.run_button.text() == "Run detection"
    page._build_work()(NullReporter())
    assert seen.get("stage") == "all", seen

    # With no weights anywhere, the clip stage still cuts.
    monkeypatch.undo()
    page.movies.takeItem(1)
    page._build_work()(NullReporter())
    assert (cfg.movie_dir(movie.id) / DONE_FILE).exists()


@pytest.mark.parametrize("record", [{}, {"detection": "x"}], ids=["none", "partial"])
def test_a_handoff_without_a_full_fingerprint_is_refused(root: Path, record) -> None:
    import json

    from trackanno.detect.prepare import detect_movie, handoff_is_fresh, read_handoff

    cfg, movie, frames = prep_project(root, f"no-fingerprint-{len(record)}")
    detect_movie(movie, StubDetector(frames, movie.n_frames), cfg, limit_frames=60)

    path = cfg.movie_dir(movie.id) / "detect_done.json"
    info = json.loads(path.read_text(encoding="utf-8"))
    info["config_fingerprint"] = record
    path.write_text(json.dumps(info), encoding="utf-8")
    assert not handoff_is_fresh(cfg, movie.id)
    with pytest.raises(ValueError):
        read_handoff(cfg, movie.id)


def test_a_prepared_movie_never_asks_for_the_model(root: Path) -> None:
    """A movie with clips but no handoff does not need detection."""
    from trackanno.detect.prepare import (
        _needs_detection, handoff_is_fresh, prepare_movie,
    )

    cfg, movie, frames = prep_project(root, "old-project")
    prepare_movie(movie, StubDetector(frames, movie.n_frames), cfg, limit_frames=60)

    (cfg.movie_dir(movie.id) / "detect_done.json").unlink()
    assert not _needs_detection(cfg, movie.id)
    assert not handoff_is_fresh(cfg, movie.id), "even though it has no handoff"
    assert _needs_detection(cfg, movie.id, force=True), "--force means it does"
    assert _needs_detection(prep_project(root, "untouched")[0], "synthetic"), (
        "a movie with neither needs detection")


def test_cutting_without_detections_says_what_to_do(root: Path) -> None:
    from trackanno.detect.prepare import cut_movie, handoff_is_fresh

    cfg, movie, _ = prep_project(root, "no-handoff")
    assert not handoff_is_fresh(cfg, movie.id)
    with pytest.raises(FileNotFoundError, match="detect_done.json") as exc:
        cut_movie(movie, cfg)
    assert "copy" in str(exc.value), "the message names the way out"


# --------------------------------------------------------------------------- #
# background jobs
# --------------------------------------------------------------------------- #
def _slow(reporter):
    for _ in reporter.track(range(100000), "grinding", 100000):
        time.sleep(0.001)
    return "should not happen"


def _last_line_colour(panel) -> tuple[str, str]:
    from qtpy.QtGui import QTextFormat

    document = panel.log.document()
    last = document.findBlockByNumber(document.blockCount() - 1)
    fragment = last.begin()
    fmt = None if fragment.atEnd() else fragment.fragment().charFormat()
    colour = (
        fmt.foreground().color().name().upper()
        if fmt is not None and fmt.hasProperty(QTextFormat.ForegroundBrush)
        else "plain"
    )
    return last.text(), colour


def test_a_task_panel_job_finishes(app) -> None:
    from trackanno.gui.tasks import TaskPanel

    panel = TaskPanel("Go")
    results: list = []

    def work(reporter):
        reporter.log("starting")
        for _ in reporter.track(range(20), "counting", 20):
            time.sleep(0.005)
        return "done"

    panel.finished.connect(lambda ok: results.append(ok))
    panel.start(work, throttle=1)
    assert pump(app, lambda: results) and results[0] is True
    assert "starting" in panel.log.toPlainText(), "the log keeps the job's output"
    panel.shutdown()


def test_a_cancelled_job_stops_and_is_not_a_failure(app) -> None:
    """Cancel stops the worker and the last line is amber, not red."""
    from trackanno.gui.tasks import TaskPanel
    from trackanno.gui.theme import tokens

    panel = TaskPanel("Go")
    results: list = []
    stopped: list = []

    def slow(reporter):
        _slow(reporter)
        stopped.append("ran to completion")

    panel.finished.connect(lambda ok: results.append(ok))
    panel.start(slow, throttle=1)
    pump(app, lambda: panel.bar.value() > 2, timeout=10)
    panel.cancel()
    assert pump(app, lambda: results) and results[0] is False and not stopped

    text, colour = _last_line_colour(panel)
    assert text == "cancelled"
    assert colour == tokens()["warning"].upper() != tokens()["danger"].upper(), colour
    panel.shutdown()


def test_a_colour_in_the_log_belongs_to_one_line(app) -> None:
    """A warning or error colour does not spread to the lines after it."""
    from qtpy.QtGui import QTextFormat

    from trackanno.gui.tasks import TaskPanel
    from trackanno.gui.theme import tokens

    panel = TaskPanel("Go")
    written = [
        ("ordinary progress", "info"),
        ("something worth knowing", "warning"),
        ("more ordinary progress", "info"),
        ("it broke\nwith a second line", "error"),
        ("still ordinary", "info"),
        ("finished", "good"),
    ]
    for message, level in written:
        panel.append(message, level)

    palette = tokens()
    document = panel.log.document()
    colours: list[tuple[str, str]] = []
    for i in range(document.blockCount()):
        block = document.findBlockByNumber(i)
        fragment = block.begin()
        fmt = None if fragment.atEnd() else fragment.fragment().charFormat()
        colours.append((
            block.text(),
            fmt.foreground().color().name().upper()
            if fmt is not None and fmt.hasProperty(QTextFormat.ForegroundBrush)
            else "plain",
        ))
    by_line = dict(colours)

    assert len(colours) == 7, colours
    assert by_line["something worth knowing"] == palette["warning"].upper(), colours
    assert by_line["more ordinary progress"] == "plain", colours
    assert by_line["it broke"] == palette["danger"].upper(), colours
    assert by_line["with a second line"] == palette["danger"].upper(), (
        f"all the way down a traceback: {colours}")
    assert by_line["still ordinary"] == "plain", colours
    assert by_line["finished"] == palette["success"].upper(), colours
    panel.shutdown()


# --------------------------------------------------------------------------- #
# the workflow window
# --------------------------------------------------------------------------- #
def test_the_model_report_lists_the_top_features(app) -> None:
    from trackanno.gui.hub import _format_report

    features = [{"feature": f"feat_{i}", "importance": round(20.0 - i, 2)}
                for i in range(15)]
    html = _format_report({"behaviour": {
        "reliable": True, "n_labels": 240, "n_classes": 3, "macro_f1": 0.52,
        "kappa": 0.33, "kappa_lower": 0.29, "accuracy": 0.57,
        "baseline_accuracy": 0.45, "scope": "leave-movie-out over 2 movies",
        "top_features": features,
    }})
    assert "feat_0" in html
    assert "feat_9" in html and "feat_10" not in html, "stops at ten"
    assert "20.00" in html, "with the importance beside it"
    assert "top" not in _format_report({"behaviour": {"top_features": []}}).lower(), (
        "no importances, no mention")


def test_cache_is_released_before_detection(app) -> None:
    """Windows refuses to overwrite a memory-mapped cache file, so the hub unmaps it."""
    from trackanno.config import load_config
    from trackanno.gui.hub import Hub

    config_path = make_project()
    cfg = load_config(config_path)

    hub = Hub(str(config_path))
    hub.show()
    hub.go_to(3)
    app.processEvents()
    panel = hub.pages[3].panel
    assert panel is not None

    movie = str(panel.store.meta(panel.order[panel.position])["movie_id"])
    origins = cfg.paths.work_dir / movie / "clip_origins.npy"
    assert movie in panel.store._origins, "the store maps the movie being annotated"

    def writable() -> bool:
        try:
            np.save(origins, np.load(origins))
            return True
        except OSError:
            return False

    assert not writable(), "a mapped cache cannot be overwritten"
    hub.release_caches()
    app.processEvents()
    assert writable()
    assert not panel.store._origins and not panel.store._mm, "every map is dropped"

    hub.close()
    app.processEvents()


def test_no_page_keeps_a_paragraph_on_screen(app, project_config) -> None:
    """Page headings are one line; the long form is behind a `?`."""
    from trackanno.gui.hub import PAGES, Hub
    from trackanno.gui.theme import Heading

    hub = Hub(str(project_config))
    hub.show()
    app.processEvents()
    try:
        long_ones, helped = [], 0
        for name, page in zip(PAGES, hub.pages):
            for heading in page.findChildren(Heading):
                subtitle = heading.subtitle.text()
                if len(subtitle) > 110:
                    long_ones.append((name, len(subtitle), subtitle))
                if heading.help is not None:
                    helped += 1
        assert not long_ones, long_ones
        assert helped >= 4, f"{helped} heading(s) carry a ?"
    finally:
        for page in hub.pages:
            page.shutdown()
        hub.close()
        app.processEvents()


def test_the_help_button_opens_and_closes(app) -> None:
    from qtpy.QtCore import QTimer
    from qtpy.QtWidgets import QDialog, QTextBrowser

    from trackanno.gui.theme import help_button

    opened: list = []

    def close_it():
        for widget in app.topLevelWidgets():
            if isinstance(widget, QDialog) and widget.isVisible():
                body = widget.findChild(QTextBrowser)
                opened.append("" if body is None else body.toPlainText())
                widget.reject()

    button = help_button(None, "Help", "<h3>A heading</h3><p>and a paragraph.</p>")
    QTimer.singleShot(0, close_it)
    button.click()
    app.processEvents()

    assert opened, "nothing opened"
    assert "A heading" in opened[0] and "and a paragraph." in opened[0], opened[0]
    assert "<h3>" not in opened[0], "the text is rendered, not escaped"
    assert not [w for w in app.topLevelWidgets()
                if isinstance(w, QDialog) and w.isVisible()], "it closed"


def test_the_hub_opens_a_project_on_every_page(app, hub) -> None:
    from trackanno.gui.hub import PAGES, ExportPage, ModelPage

    assert hub.cfg is not None and hub.cfg.path.name == "config.yaml"
    assert len(hub.pages) == len(PAGES)

    advice, page, _ = hub.next_step()
    assert page == 2, f"an empty project is told to run detection: {advice}"

    for index in range(len(PAGES)):
        hub.go_to(index)
        app.processEvents()

    assert "no clips" in hub.pages[3].blocker(), hub.pages[3].blocker()
    assert "nothing labelled" in hub.pages[4].blocker(), hub.pages[4].blocker()
    assert "no annotations" in hub.page(ExportPage).blocker()
    assert "no annotations" in hub.page(ModelPage).distribution.blocker()


@pytest.fixture
def config_page(app, hub):
    hub.go_to(1)
    app.processEvents()
    return hub.pages[1]


def test_configure_splits_basics_from_advanced(config_page) -> None:
    basics, advanced = config_page.form.current(), config_page.advanced.current()
    assert 5 < len(basics) < 25, len(basics)
    # Not "more than the basics": the centerline task's fields made the two tabs about
    # equal. test_configform checks that every field reaches exactly one tab.
    assert advanced, "advanced holds the rest"
    assert not (set(basics) & set(advanced)), "no setting appears on both tabs"


def test_configure_saves_edits_from_both_tabs(app, hub, config_page) -> None:
    config_page.form._widgets["clips.pad"].setValue(2.75)
    config_page.advanced._widgets["detection.conf"].setValue(0.4)
    app.processEvents()
    assert config_page._dirty and config_page.save_button.isEnabled()
    assert config_page.heading.title.text().endswith("*"), "the title marks unsaved edits"
    assert config_page.save() and not config_page._dirty
    assert abs(hub.cfg.clips.pad - 2.75) < 1e-9, hub.cfg.clips.pad
    assert abs(hub.cfg.detection.conf - 0.4) < 1e-9, hub.cfg.detection.conf
    assert hub.cfg.path.read_text(encoding="utf-8").count("#") > 40, "comments are kept"
    assert config_page.form.current()["clips.pad"] == 2.75, "the form is not wiped"
    assert not config_page.save_button.isEnabled()


def test_configure_refuses_an_unusable_label_set(app, hub, config_page) -> None:
    good_text = hub.cfg.path.read_text(encoding="utf-8")
    name_field = config_page.categories._groups[0].name
    name_field.setText("")
    app.processEvents()
    assert not config_page.save() and config_page.banner.isVisible()
    assert "no name" in config_page.banner.text(), config_page.banner.text()[:120]
    assert hub.cfg.path.read_text(encoding="utf-8") == good_text

    name_field.setText("state")
    app.processEvents()
    assert config_page.save() and not config_page._dirty
    assert not config_page.banner.isVisible(), "the error banner clears"


def test_configure_saves_a_missing_path_with_a_warning(app, hub, config_page) -> None:
    """A model on an offline share must not block editing anything else."""
    config_page.form._widgets["paths.model"].set_value(str(hub.cfg.path.parent / "nope.pt"))
    app.processEvents()
    assert config_page.save() and not config_page._dirty
    assert "paths.model" in config_page._show_warnings()
    assert not config_page.banner.isVisible()


def test_leaving_configure_with_unsaved_edits_asks(app, hub, config_page) -> None:
    config_page.form._widgets["clips.pad"].setValue(3.0)
    asked: list[str] = []
    config_page._ask_unsaved = lambda: (asked.append("asked"), "save")[1]
    hub.go_to(0)
    app.processEvents()
    del config_page._ask_unsaved
    assert asked == ["asked"]
    assert not config_page._dirty and abs(hub.cfg.clips.pad - 3.0) < 1e-9, "Save writes them"


def test_the_model_row_offers_a_shared_model(app, hub, config_page) -> None:
    from qtpy.QtWidgets import QPushButton

    from trackanno.gui.hub import ConfigPage
    from trackanno.project import write_config

    row = config_page.form._widgets["paths.model"].row
    labels = [row.itemAt(i).widget().text() for i in range(row.count())
              if isinstance(row.itemAt(i).widget(), QPushButton)]
    assert labels == ["Choose shared model...", "Browse"], labels

    field = config_page.form._widgets["paths.model"]
    # isHidden, not isVisible: the page may not be the one on screen.
    assert field.status.isHidden() and not field.status.text(), (
        "nothing is claimed for a local file")

    write_config(hub.cfg.path, {"model.source": "some-model", "model.version": "v3"})
    hub.open_project(hub.cfg.path)
    app.processEvents()
    page = next(p for p in hub.pages if isinstance(p, ConfigPage))
    field = page.form._widgets["paths.model"]
    assert "some-model v3" in field.status.text(), field.status.text()
    assert "not used" in field.status.text() and not field.status.isHidden()
    assert not field.edit.isEnabled(), "the unused path is greyed out"
    assert "paths.model" not in (page._show_warnings() or {}), (
        "no amber outline for an unused path")


def test_the_scan_message_sits_below_the_buttons(hub) -> None:
    from trackanno.gui.hub import PreparePage

    prepare_page = next(p for p in hub.pages if isinstance(p, PreparePage))
    box = prepare_page.scan_state.parentWidget().layout()
    rows = [box.itemAt(i) for i in range(box.count())]
    button_row = next(i for i, it in enumerate(rows) if it.layout() is not None)
    label_row = next(i for i, it in enumerate(rows)
                     if it.widget() is prepare_page.scan_state)
    assert label_row == button_row + 1, f"buttons at {button_row}, message at {label_row}"
    assert prepare_page.scan_state.wordWrap()


@pytest.fixture
def cache_hub(app):
    """A hub on the generated clip cache, open on the Annotate page."""
    from trackanno.gui.hub import Hub

    window = Hub(str(make_project()))
    window.show()
    window.go_to(3)
    app.processEvents()
    yield window
    window.close()
    app.processEvents()


def test_annotate_keeps_its_panel_across_navigation(app, cache_hub) -> None:
    panel = cache_hub.pages[3].panel
    assert panel is not None
    assert len(panel.order) == cache_hub.status.n_clips, "the whole cache is loaded"

    position = panel.position
    cache_hub.go_to(0)
    app.processEvents()
    cache_hub.go_to(3)
    app.processEvents()
    assert cache_hub.pages[3].panel is panel
    assert cache_hub.pages[3].panel.position == position


def test_a_config_reload_keeps_the_annotation_panel(app, cache_hub) -> None:
    panel = cache_hub.pages[3].panel
    panel.position = min(3, len(panel.order) - 1)
    panel._load()
    at = panel.position
    cache_hub.reload_project()
    app.processEvents()
    assert cache_hub.pages[3].panel is panel
    assert cache_hub.pages[3].panel.position == at, "the current clip is kept"


def test_later_steps_unlock_while_annotating(app, tmp_path) -> None:
    """Model and Export unlock as labels arrive, without a project reopen."""
    from trackanno.config import load_config
    from trackanno.gui.hub import ExportPage, Hub, ModelPage
    from trackanno.project import write_config

    config_path = make_project()
    cfg = load_config(config_path)
    project = tmp_path / "p"
    project.mkdir()
    shutil.copy(config_path, project / "config.yaml")
    # Same clip cache, empty CSV.
    write_config(project / "config.yaml", {
        "paths.work_dir": str(cfg.paths.work_dir).replace("\\", "/"),
        "paths.csv": str(project / "fresh.csv").replace("\\", "/"),
    })
    hub = Hub(str(project / "config.yaml"))
    hub.show()
    app.processEvents()
    try:
        assert not hub.pages[4].tasks.run_button.isEnabled(), "Model starts locked"
        assert not hub.page(ExportPage).tasks.run_button.isEnabled(), "Export starts locked"

        hub.go_to(3)
        app.processEvents()
        panel = hub.pages[3].panel
        group = hub.cfg.categories[0]
        for _ in range(3):
            panel.categories.toggle(group.name, group.options[0].name)
            app.processEvents()

        assert len(panel.annotations) == 3
        assert hub.status.n_annotated == len(panel.annotations), "the project-wide count"
        assert hub.pages[4].tasks.run_button.isEnabled(), "Model unlocks on the Annotate page"
        assert hub.page(ExportPage).tasks.run_button.isEnabled()

        hub.go_to(hub.pages.index(hub.page(ExportPage)))
        app.processEvents()
        assert hub.page(ExportPage).movies.count() > 0, "Export lists movies to render"

        hub.go_to(hub.pages.index(hub.page(ModelPage)))
        app.processEvents()
        dist = hub.page(ModelPage).distribution
        assert len(dist._charts) == len(hub.cfg.categories), list(dist._charts)
        counted = sum(b.count for b in dist._charts[group.name].tally.bars
                      if b.name == group.options[0].name)
        assert counted == 3, counted
    finally:
        hub.close()
        app.processEvents()


# --------------------------------------------------------------------------- #
# config templates
# --------------------------------------------------------------------------- #
def _tuned_source(root: Path) -> tuple[Path, Path]:
    from trackanno.project import create_project, write_config

    videos = root / "tpl-videos"
    videos.mkdir()
    weights = root / "tpl-model.pt"
    weights.write_bytes(b"")
    source = create_project(root / "tpl-a", video_dir=videos, model=weights)
    write_config(source, {
        "clips.frames": 33, "clips.min_clip_frames": 25, "clips.crop_size": 128,
        "tiling.tile": 1280, "detection.conf": 0.4,
        "tracking.backend": "botsort", "ui.annotator": "someone",
    }, categories=[{"name": "state", "type": "single", "required": True,
                    "options": [{"name": "swimming", "key": "1", "color": "#4C9F70"},
                                {"name": "stuck", "key": "2", "color": "#D95F02"}]}])
    return source, weights


def test_a_template_carries_settings_but_not_paths(tmp_path) -> None:
    """A template keeps the tuning and the model, not the folders or the annotator."""
    from trackanno.config import load_config
    from trackanno.project import create_project
    from trackanno.templates import delete_template, list_templates, save_template

    source, weights = _tuned_source(tmp_path)
    template = save_template("Trypanosome 100 fps", source)
    try:
        assert template.name == "Trypanosome 100 fps", "the typed name is kept"
        assert template.path.name == "trypanosome-100-fps.yaml", "the file name is a slug"
        names = [t.name for t in list_templates()]
        assert names[0] == "default" and names[-1] == "Trypanosome 100 fps", (
            f"listed after the built-in and the shipped presets: {names}")

        other = tmp_path / "tpl-videos-2"
        other.mkdir()
        made = create_project(tmp_path / "tpl-b", video_dir=other,
                              template="Trypanosome 100 fps")
        cfg = load_config(made)
        assert (cfg.clips.frames, cfg.tiling.tile, cfg.tracking.backend) == (33, 1280, "botsort")
        assert [g.name for g in cfg.categories] == ["state"]
        assert [o.name for o in cfg.categories[0].options] == ["swimming", "stuck"]
        assert cfg.paths.model == weights.resolve(), "the model comes across"
        assert cfg.paths.video_dir == other.resolve(), "the movie folder does not"
        assert cfg.paths.work_dir == (tmp_path / "tpl-b" / "work").resolve()
        assert cfg.paths.csv == (tmp_path / "tpl-b" / "annotations.csv").resolve()
        assert cfg.ui.annotator == "", "a template is a protocol, not a person"

        text = made.read_text(encoding="utf-8")
        assert text.count("#") > 40, "the documenting comments survive"
        assert not text.startswith("# trackanno template:"), (
            "the template header does not leak into the project")
    finally:
        delete_template("Trypanosome 100 fps")


def test_a_template_can_be_found_however_it_is_named(root) -> None:
    from trackanno.config import load_config
    from trackanno.project import create_project
    from trackanno.templates import delete_template, find_template, save_template

    source = create_project(root / "tpl-c")
    save_template("Wide Field 20x", source)
    for spelling in ("Wide Field 20x", "wide field 20x", "wide-field-20x"):
        found = find_template(spelling)
        assert found is not None and found.name == "Wide Field 20x", spelling
    assert find_template("no such thing") is None
    made = create_project(root / "tpl-d", template="no such thing")
    assert load_config(made).clips.frames == 50, "an unknown template falls back to the built-in"

    assert not delete_template("default"), "the built-in cannot be deleted"
    assert delete_template("Wide Field 20x")
    assert find_template("Wide Field 20x") is None


def test_a_template_that_would_not_load_is_refused(root) -> None:
    from trackanno.project import create_project
    from trackanno.templates import find_template, save_template

    source = create_project(root / "tpl-e")
    text = source.read_text(encoding="utf-8")
    source.write_text(text.replace("frames: 50", "frames: not-a-number"), encoding="utf-8")

    with pytest.raises(Exception):
        save_template("broken", source)
    assert find_template("broken") is None, "nothing was stored"


def test_saving_a_template_twice_needs_permission(root) -> None:
    from trackanno.project import create_project
    from trackanno.templates import delete_template, save_template

    source = create_project(root / "tpl-f")
    save_template("dup", source)
    try:
        with pytest.raises(FileExistsError):
            save_template("dup", source)
        assert save_template("dup", source, overwrite=True).name == "dup"
    finally:
        delete_template("dup")


def test_the_new_project_dialog_offers_the_templates(app, root) -> None:
    from trackanno.gui.hub import NewProjectDialog
    from trackanno.project import create_project
    from trackanno.templates import BUILTIN_NAME, delete_template, save_template

    source = create_project(root / "tpl-g")
    save_template("from the dialog", source)

    dialog = NewProjectDialog()
    try:
        items = [dialog.template.itemText(i) for i in range(dialog.template.count())]
        assert items[0] == "default (built in)", items
        assert "from the dialog" in items, items
        assert "annotator (preset)" in items, items
        assert items.index("annotator (preset)") < items.index("from the dialog"), (
            "shipped presets come before saved templates")
        assert dialog.values()["template"] == BUILTIN_NAME
        assert not dialog.forget_btn.isEnabled(), "the built-in cannot be forgotten"

        dialog.template.setCurrentIndex(items.index("from the dialog"))
        assert dialog.forget_btn.isEnabled()
        assert dialog.values()["template"] == "from the dialog"
    finally:
        dialog.close()
        delete_template("from the dialog")


# --------------------------------------------------------------------------- #
# export and shared models
# --------------------------------------------------------------------------- #
def test_export_can_open_the_project_folder(app, root, monkeypatch) -> None:
    from trackanno.gui import hub as H
    from trackanno.gui.hub import ExportPage, Hub

    project = make_project(root / "reveal", pose=True)
    window = Hub(str(project))
    try:
        page = next(p for p in window.pages if isinstance(p, ExportPage))
        assert page.open_folder.text() == "Open project folder"

        opened: list = []
        monkeypatch.setattr(H.export.os, "startfile", opened.append, raising=False)
        monkeypatch.setattr("sys.platform", "win32")
        page._open_project_folder()
        assert opened == [str(project.parent)], opened

        assert "does not exist" in H._reveal(root / "gone"), "a missing folder is reported"
    finally:
        window.close()
        app.processEvents()


def test_browsing_to_a_file_stops_using_the_shared_model(app, root) -> None:
    """Picking a local .pt drops model.source and its pin."""
    from trackanno.config import load_config
    from trackanno.gui.hub import ConfigPage, Hub
    from trackanno.project import write_config

    project = make_project(root / "swap")
    write_config(project, {"model.source": "some-model", "model.version": "v3",
                           "model.sha256": "ab" * 32})
    local = project.parent / "small.pt"
    local.write_bytes(b"not really weights")

    window = Hub(str(project))
    try:
        page = next(p for p in window.pages if isinstance(p, ConfigPage))
        field = page.form._widgets["paths.model"]
        assert not field.edit.isEnabled(), "the shared model is in use to begin with"

        # What the Browse dialog does once a file has been chosen.
        field.edit.setText(str(local).replace("\\", "/"))
        field.picked.emit(str(local).replace("\\", "/"))
        assert page._dirty and "Save" in field.status.text(), field.status.text()
        assert page.save() and not page._dirty

        after = load_config(project)
        assert not after.model.is_remote, after.model.source
        assert not after.model.version and not after.model.sha256, "the pin goes with it"
        assert after.paths.model.name == "small.pt", after.paths.model
        assert not field.status.text(), field.status.text()
        assert field.edit.isEnabled()
    finally:
        window.close()
        app.processEvents()


class _Entry:
    id, version, sha256 = "shared-pose", "v2", "cd" * 32
    label = "shared-pose v2"


def test_a_new_project_can_start_from_a_shared_model(app, root, monkeypatch) -> None:
    """The dialog records the choice; nothing is downloaded while it is open."""
    from trackanno.gui import hub as H

    entry = _Entry()

    class StubRegistry:
        def __init__(self, parent=None):
            pass

        def run(self):
            return True

        def chosen(self):
            return entry

    monkeypatch.setattr(H.project, "ModelRegistryDialog", StubRegistry)
    dialog = H.NewProjectDialog()
    try:
        dialog.name.setText("from-registry")
        dialog.location.edit.setText(str(root))
        dialog._pick_shared_model()

        assert dialog.values()["model"] is None, "create_project is not handed the model"
        assert dialog.chosen_model() is entry
        assert not dialog.model.edit.isEnabled()
        assert "shared-pose v2" in dialog.model_note.text(), dialog.model_note.text()
        assert "not used" in dialog.model_note.text(), dialog.model_note.text()

        asked: list[str] = []
        monkeypatch.setattr(
            H.project.QMessageBox, "question",
            lambda *a, **k: (asked.append(str(a[2])), H.project.QMessageBox.No)[1],
        )
        dialog._accept()
        assert not any("no detection model" in text for text in asked), (
            f"a chosen shared model counts as a model: {asked}")

        picked = str(root / "small.pt")
        dialog.model.edit.setText(picked)
        dialog.model.picked.emit(picked)
        assert dialog.chosen_model() is None, "browsing to a file drops the shared model"
        assert dialog.model.edit.isEnabled()
        assert dialog.values()["model"] == picked
    finally:
        dialog.close()


def test_the_shared_model_is_fetched_after_the_project_is_written(app, root, monkeypatch) -> None:
    """Create first, then download into the config just created."""
    from trackanno.gui import hub as H

    entry = _Entry()
    target = root / "fetched"

    class StubDialog:
        def __init__(self, parent=None):
            pass

        def run(self):
            return True

        def values(self):
            return {"directory": target, "video_dir": None, "model": None,
                    "template": "annotator"}

        def chosen_model(self):
            return entry

    installed: list = []
    monkeypatch.setattr(H.project, "NewProjectDialog", StubDialog)
    monkeypatch.setattr(H.project, "_install_shared_model",
                        lambda page, e, path: installed.append((e, Path(path))))

    window = H.Hub(str(make_project(root / "host")))
    try:
        page = next(p for p in window.pages if isinstance(p, H.ProjectPage))
        page._new_project()

        assert (target / "config.yaml").exists()
        assert installed == [(entry, target / "config.yaml")], installed
        assert window.cfg is not None
        assert window.cfg.path == (target / "config.yaml").resolve(), window.cfg.path
    finally:
        window.close()
        app.processEvents()


def test_the_device_list_and_the_resolver_cannot_disagree() -> None:
    """``auto`` only resolves to a device the list offers."""
    from trackanno.detect import tiling

    indices = tiling.cuda_devices()
    resolved = tiling.resolve_device("auto")
    assert resolved in {str(i) for i in indices} | {"cpu", "mps"}, resolved
    if indices:
        assert resolved == str(indices[0]), f"auto takes the first usable GPU: {resolved}"
    else:
        assert resolved != "0", f"auto does not assume device 0: {resolved}"

    assert tiling.resolve_device("99") == "cpu", "an index nobody offered is refused"

    usable, why = tiling.cuda_status()
    assert usable or bool(why), "a refusal carries a reason"
    assert not usable or not why, f"a working GPU carries none: {why!r}"
    assert tiling.device_label("cpu") == "the CPU"
    assert "mps" in tiling.device_label("mps")
