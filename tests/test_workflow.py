"""Checks for the GUI-only workflow: projects, config editing, background jobs.

Nothing here needs a GPU, a movie or a display - the window is built offscreen.
Every part that needs a prepared clip cache builds its own; nothing here
reads the repository's project or any real data.
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
import yaml
from qtpy.QtCore import Qt
from fixtures import make_project
from harness import check

# Every GUI test below runs against a project this suite GENERATES: config,
# synthetic movie, real clip cache. It used to load the repository's own
# config.yaml behind a try/except that printed "skipped" - which meant the
# tests passed without running on any machine that lacked that data, and broke
# on the one that had it whenever its network share went away.
FIXTURES = Path(tempfile.mkdtemp(prefix="trackanno-fixtures-"))
atexit.register(shutil.rmtree, FIXTURES, True)


def pump(app, predicate, timeout: float = 30.0) -> bool:
    """Spin the event loop until a worker thread's signals have landed."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


# --------------------------------------------------------------------------- #
def test_yamledit() -> None:
    from trackanno.project import DEFAULT_CONFIG
    from trackanno.yamledit import replace_section, set_values

    before = yaml.safe_load(DEFAULT_CONFIG)
    text = set_values(DEFAULT_CONFIG, {
        "clips.frames": 33,
        "ui.view_zoom": 2.5,
        "ui.show_context_tracks": True,
        "paths.model": "D:/weights/best.pt",
        "normalize.percentiles": [1.0, 99.0],
        "ui.a_brand_new_key": "hello",
        "an_entirely_new_section.key": 7,
    })
    after = yaml.safe_load(text)
    check("scalar edits land", after["clips"]["frames"] == 33 and after["ui"]["view_zoom"] == 2.5)
    check("a path with a colon survives quoting", after["paths"]["model"] == "D:/weights/best.pt")
    check("list edits land", after["normalize"]["percentiles"] == [1.0, 99.0])
    check("new keys are appended to their section", after["ui"]["a_brand_new_key"] == "hello")
    check("new sections are appended", after["an_entirely_new_section"]["key"] == 7)
    check("untouched values are untouched",
          after["tracking"] == before["tracking"] and after["keys"] == before["keys"])
    check("comments survive an edit",
          text.count("#") >= DEFAULT_CONFIG.count("#"),
          f"{text.count('#')} vs {DEFAULT_CONFIG.count('#')}")
    check("trailing comments stay separated from their value",
          "33#" not in text and "33 #" in text or "frames: 33" in text)

    groups = [{"name": "state", "type": "single", "required": True,
               "options": [{"name": "a", "key": "1", "color": "#4C9F70"}]}]
    replaced = yaml.safe_load(replace_section(text, "categories", groups))
    check("a whole block can be replaced", replaced["categories"] == groups)
    check("replacing a block leaves its neighbours alone",
          replaced["ui"]["view_zoom"] == 2.5 and replaced["render"] == before["render"])


@pytest.fixture(scope="module")
def project_config(root):
    """A project created and then edited, the way the GUI leaves one.

    Shared with the hub tests below on purpose: the workflow window has to open
    a project that has been through an edit, not only a freshly created one.
    """
    from trackanno.project import create_project, write_config

    videos = root / "movies"
    videos.mkdir()
    (videos / "a.tif").write_bytes(b"")
    (videos / "b.tif").write_bytes(b"")
    (videos / "notes.txt").write_bytes(b"")
    weights = root / "best.pt"
    weights.write_bytes(b"")

    path = create_project(root / "proj", video_dir=videos, model=weights)
    write_config(path, {"clips.frames": 25, "clips.min_clip_frames": 20, "ui.follow": "locked"},
                 categories=[{"name": "state", "type": "single", "required": True,
                              "options": [{"name": "a", "key": "1", "color": "#4C9F70"},
                                          {"name": "b", "key": "2", "color": "#D95F02"}]}])
    return path


def test_project(root, project_config) -> None:
    """Creation, the status readout, and an edit that must never reach disk."""
    from trackanno.config import load_config
    from trackanno.project import project_status, write_config

    videos = root / "movies"
    path = project_config
    check("create_project writes a config", path.exists())
    cfg = load_config(path)
    check("the generated config loads", cfg.paths.video_dir == videos.resolve())
    check("work_dir defaults inside the project",
          cfg.paths.work_dir == (root / "proj" / "work").resolve())
    check("the generated config is documented", path.read_text(encoding="utf-8").count("#") > 40)

    status = project_status(cfg)
    check("status counts only matching files", status.n_videos == 2, str(status.n_videos))
    check("status sees the model", status.model_ok)
    check("status knows nothing is prepared",
          status.n_prepared == 0 and status.n_clips == 0 and not status.can_annotate)
    check("status says preparation is possible", status.can_prepare)

    edited = load_config(path)
    check("edits round-trip through the loader",
          edited.clips.frames == 25 and edited.ui.follow == "locked")
    check("category edits round-trip",
          [g.name for g in edited.categories] == ["state"]
          and [o.name for o in edited.categories[0].options] == ["a", "b"])

    # An incoherent edit must be refused BEFORE it reaches disk. Autosave relies
    # on this: writing first and rolling back would leave the project briefly
    # unopenable, and a crash in that window would leave it broken for good.
    good = path.read_text(encoding="utf-8")
    from trackanno.config import ConfigError

    try:
        write_config(path, {"clips.min_clip_frames": 999})
    except ConfigError as exc:
        check("an incoherent edit is refused with an explanation",
              "min_clip_frames" in str(exc) and "no window could ever qualify" in str(exc))
    else:
        check("an incoherent edit is refused with an explanation", False)
    check("the refused edit never touched the file",
          path.read_text(encoding="utf-8") == good)
    check("the project still loads", load_config(path).clips.frames == 25)
    check("no temp file was left behind",
          not list(path.parent.glob("*.tmp")), str(list(path.parent.glob("*.tmp"))))


def test_pose_is_resolvable(root: Path) -> None:
    """`task: pose` with no keypoints listed must be fixable, not fatal.

    It used to raise in load_config, which meant the project would no longer
    open - so switching the task to 'pose' from the GUI was impossible, and
    hand-editing it bricked the project until you edited the YAML back.
    """
    from trackanno.config import load_config
    from trackanno.project import create_project, write_config
    from trackanno.tiling import default_node_names, read_keypoint_count

    project = root / "pose"
    path = create_project(project)
    write_config(path, {"detection.task": "pose"})

    cfg = load_config(path)
    check("a pose config with no keypoints still loads", cfg.is_pose)
    check("...and says so", cfg.needs_keypoints and cfg.n_keypoints == 0)

    check("generated names are one per keypoint", default_node_names(4) == ["kp0", "kp1", "kp2", "kp3"])

    # Writing the names resolves it, and the count is what has to match.
    write_config(path, {"keypoints.nodes": ["nose", "tail"]})
    resolved = load_config(path)
    check("naming the keypoints resolves it",
          not resolved.needs_keypoints and resolved.n_keypoints == 2)

    fake = root / "not-a-model.pt"
    fake.write_bytes(b"not a checkpoint")
    try:
        read_keypoint_count(fake)
    except RuntimeError as exc:
        check("an unreadable model fails with a clear message", "could not read" in str(exc))
    else:
        check("an unreadable model fails with a clear message", False)


def test_keypoint_table(app) -> None:
    """The count comes from the model; the names are typed into a table."""
    from trackanno.gui.configform import KeypointEditor

    editor = KeypointEditor()
    editor.load(["Head", "Base", "Tip"], [["Head", "Base"], ["Base", "Tip"]])
    check("the table loads one row per keypoint", editor.table.rowCount() == 3)
    check("the count follows the names", editor.count.value() == 3)
    check("the position column is not editable",
          not (editor.table.item(0, 0).flags() & Qt.ItemIsEditable))
    check("links load too", editor.edges.rowCount() == 2)

    # Reading the count from a model must not throw away names already typed.
    editor.set_count(5, "5 from model.pt")
    check("growing keeps the names you typed", editor.names()[:3] == ["Head", "Base", "Tip"])
    check("new rows get placeholders", editor.names()[3:] == ["kp3", "kp4"])
    check("the source is shown", "model.pt" in editor.state.text())

    editor.set_count(2)
    check("shrinking drops from the end", editor.names() == ["Head", "Base"])

    editor.set_count(4)
    editor._chain()
    check("connect in order chains them", editor.values()["edges"] ==
          [["Head", "Base"], ["Base", "kp2"], ["kp2", "kp3"]])

    # Renaming a keypoint must follow through into the links, not orphan them.
    editor.table.item(2, 1).setText("Middle")
    app.processEvents()
    check("renaming updates the links",
          editor.values()["edges"][1] == ["Base", "Middle"], str(editor.values()["edges"]))

    check("a clean layout has no problems", editor.problems() == [])
    editor.table.item(1, 1).setText("Head")
    app.processEvents()
    check("duplicate names are caught",
          any("duplicate" in p for p in editor.problems()), str(editor.problems()))
    editor.table.item(1, 1).setText("")
    app.processEvents()
    check("empty names are caught",
          any("needs a name" in p for p in editor.problems()), str(editor.problems()))


def test_cache_is_released_before_detection(app) -> None:
    """Detection must be able to overwrite a cache the window has been reading.

    Windows refuses to overwrite a memory-mapped file, so annotating a movie and
    then re-running detection on it failed with a bare
    ``OSError: [Errno 22] Invalid argument`` on clip_origins.npy - and only for
    the movies that had actually been on screen, which made it look random.
    """
    from trackanno.config import load_config
    from trackanno.gui.hub import DistributionPage, ExportPage, Hub

    config_path = make_project(FIXTURES)
    repo = config_path.parent
    cfg = load_config(config_path)

    hub = Hub(str(config_path))
    hub.show()
    hub.go_to(3)
    app.processEvents()
    panel = hub.pages[3].panel
    check("the annotation panel opened on the generated cache", panel is not None)

    movie = str(panel.store.meta(panel.order[panel.position])["movie_id"])
    origins = cfg.paths.work_dir / movie / "clip_origins.npy"
    check("the store maps the movie being annotated", movie in panel.store._origins)

    def writable() -> bool:
        try:
            np.save(origins, np.load(origins))
            return True
        except OSError:
            return False

    check("a mapped cache cannot be overwritten", not writable())
    hub.release_caches()
    app.processEvents()
    check("release_caches lets detection rewrite it", writable())
    check("...and drops every map", not panel.store._origins and not panel.store._mm)

    hub.close()
    app.processEvents()


def test_theme(app) -> None:
    """Both themes apply cleanly and every widget picks up the palette."""
    from trackanno.gui.theme import THEMES, apply_theme, current, tokens

    for name in ("dark", "light"):
        used = apply_theme(app, name)
        check(f"{name} theme applies", used == name and current() == name)
        sheet = app.styleSheet()
        check(f"{name} sheet is substantial", len(sheet) > 2000, str(len(sheet)))
        check(f"{name} sheet has no unresolved tokens",
              "{t[" not in sheet and "None" not in sheet.split("QToolTip")[0][:200])
        colours = {k for k in tokens() if not k.startswith("_")}
        check(f"{name} defines every colour the other does",
              colours == set(THEMES["dark"]), str(set(THEMES["dark"]) ^ colours))
        assets = [Path(v) for k, v in tokens().items() if k.startswith("_arrow")]
        check(f"{name} generated its arrow images",
              len(assets) == 3 and all(p.is_file() and p.stat().st_size for p in assets),
              str([(p.name, p.exists()) for p in assets]))
        check(f"{name} sheet points at the arrows it generated",
              all(str(p).replace("\\", "/") in sheet for p in assets))

    # And the live toggle must actually repaint, not just rewrite the config.
    from trackanno.gui.hub import Hub
    from trackanno.project import create_project

    scratch = Path(tempfile.mkdtemp(prefix="trackanno-theme-"))
    try:
        path = create_project(scratch / "p")
        hub = Hub(str(path))
        hub.show()
        app.processEvents()
        started = current()
        check("the toggle is a compact icon button, not a wide label",
              not hub.theme_button.text() and not hub.theme_button.icon().isNull()
              and hub.theme_button.width() <= 34)
        check("the toggle says what it does",
              ("light" if started == "dark" else "dark") in hub.theme_button.toolTip().lower(),
              hub.theme_button.toolTip())
        hub.toggle_theme()
        app.processEvents()
        check("toggling swaps the theme", current() != started)
        check("the tooltip now offers the way back",
              ("dark" if current() == "light" else "light") in hub.theme_button.toolTip().lower(),
              hub.theme_button.toolTip())
        from trackanno.config import load_config
        check("the choice is remembered in the project",
              load_config(path).ui.theme == current(), load_config(path).ui.theme)
        hub.toggle_theme()
        app.processEvents()
        check("and back again", current() == started)
        hub.close()
        app.processEvents()
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    applied = apply_theme(app, "system")
    check("system resolves to a real theme", applied in ("dark", "light"))
    check("an unknown theme falls back rather than crashing",
          apply_theme(app, "chartreuse") == "dark")
    apply_theme(app, "dark")


def test_progress() -> None:
    from trackanno.progress import CallbackReporter, Cancelled

    seen: list = []
    reporter = CallbackReporter(
        on_log=lambda m, l: seen.append(("log", m, l)),
        on_stage=lambda label, total: seen.append(("stage", label, total)),
        on_progress=lambda done, total: seen.append(("progress", done, total)),
    )
    out = list(reporter.track(range(5), "counting", 5))
    check("track yields everything", out == [0, 1, 2, 3, 4])
    check("track announces its stage", ("stage", "counting", 5) in seen)
    check("track reports progress", ("progress", 5, 5) in seen)

    reporter.cancel()
    try:
        list(reporter.track(range(5), "counting", 5))
    except Cancelled:
        check("a cancelled reporter stops the loop", True)
    else:
        check("a cancelled reporter stops the loop", False)


PREPARE_CONFIG = """
paths:  {video_dir: ./videos, model: ./model.pt, work_dir: ./work, csv: ./annotations.csv}
tiff:   {fps: 100}
detection: {task: detect, imgsz: 640, device: cpu}
features: {enabled: false}
clips:  {frames: 50, stride: 50, window_mode: first, follow: smooth, pad: 1.6,
         crop_size: 96, min_clip_frames: 40}
normalize: {mode: fixed}
tracking: {backend: greedy, min_track_length: 40, greedy_max_dist_px: 30}
categories:
  - name: behaviour
    type: single
    required: true
    options: [{name: swimming, key: "1", color: "#4C9F70"}]
ui: {annotator: tester}
"""


class StubDetector:
    """Replays a synthetic movie's ground truth. No GPU, no weights, no torch.

    ``prepare`` calls ``detect`` once per frame in order, so counting calls is
    enough to know which frame is being asked for.
    """

    def __init__(self, by_frame: dict, n_frames: int):
        self.by_frame, self.n_frames, self.calls = by_frame, n_frames, 0

    def is_tiled(self, height, width):
        return False

    def tiles_for(self, height, width):
        return [(0, 0, width, height)]

    def detect(self, frame_rgb):
        from trackanno.tiling import Detections

        xyxy, conf = self.by_frame[self.calls % self.n_frames]
        self.calls += 1
        return Detections(
            xyxy=xyxy, conf=conf, cls=np.zeros(len(conf), np.int32), kpts=None
        )


def test_prepare_reports_and_cancels(root: Path) -> None:
    """The whole detect->track->cut path, driven the way the GUI drives it."""
    from test_pipeline import SyntheticMovie

    from trackanno.config import load_config
    from trackanno.prepare import prepare_movie
    from trackanno.progress import CallbackReporter, Cancelled

    project = root / "prep"
    project.mkdir()
    (project / "config.yaml").write_text(PREPARE_CONFIG, encoding="utf-8")
    cfg = load_config(project / "config.yaml")

    movie = SyntheticMovie()
    truth = movie.tracks()
    by_frame = {
        int(f): (df[["x1", "y1", "x2", "y2"]].to_numpy(np.float32),
                 df["conf"].to_numpy(np.float32))
        for f, df in truth.groupby("frame")
    }

    stages, progress, logs = [], [], []
    reporter = CallbackReporter(
        on_log=lambda m, l: logs.append(m),
        on_stage=lambda label, total: stages.append((label, total)),
        on_progress=lambda done, total: progress.append(done),
    )
    info = prepare_movie(movie, StubDetector(by_frame, movie.n_frames), cfg, limit_frames=60, reporter=reporter)

    check("prepare produced clips", info["n_clips"] > 0, str(info))
    check("prepare reported named stages",
          any("detect + track" in s for s, _ in stages) and any("cutting clips" in s for s, _ in stages),
          str([s for s, _ in stages]))
    check("prepare reported a total to count against",
          any(total for _, total in stages), str(stages))
    check("prepare reported per-frame progress", max(progress) > 1, str(max(progress or [0])))
    check("prepare wrote a resume marker", (cfg.movie_dir(movie.id) / "prepare_done.json").exists())

    from trackanno.clips import ClipStore

    store = ClipStore(cfg.paths.work_dir)
    check("the cache it wrote is readable", len(store) == info["n_clips"])
    check("the clips it wrote carry overlay geometry", len(store.clip_boxes(0)) > 0)

    # And the same call must be interruptible partway through, not only between
    # movies - a forty-minute movie is otherwise uncancellable.
    stopper = CallbackReporter(on_progress=lambda done, total: None)
    stopper.cancel()
    try:
        prepare_movie(
            movie, StubDetector(by_frame, movie.n_frames), cfg,
            force=True, limit_frames=60, reporter=stopper,
        )
    except Cancelled:
        check("a cancelled prepare stops inside the movie", True)
    else:
        check("a cancelled prepare stops inside the movie", False)


def test_task_panel(app) -> None:
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
    check("a finished job reports success", pump(app, lambda: results) and results[0] is True)
    check("the log kept the job's output", "starting" in panel.log.toPlainText())

    # And it must be stoppable, not merely have a button that says Cancel.
    results.clear()
    stopped: list = []

    def slow(reporter):
        for _ in reporter.track(range(100000), "grinding", 100000):
            time.sleep(0.001)
        stopped.append("ran to completion")
        return "should not happen"

    panel.start(slow, throttle=1)
    pump(app, lambda: panel.bar.value() > 2, timeout=10)
    panel.cancel()
    check("cancel actually stops the worker",
          pump(app, lambda: results) and results[0] is False and not stopped)
    panel.shutdown()


def test_hub(app, project_config: Path) -> None:
    from trackanno.gui.hub import PAGES, DistributionPage, ExportPage, Hub

    hub = Hub(str(project_config))
    hub.show()
    app.processEvents()
    check("the hub opens a project", hub.cfg is not None and hub.cfg.path == project_config)
    check("the hub has every page", len(hub.pages) == len(PAGES))

    advice, page, _ = hub.next_step()
    check("an empty project is told to run detection", page == 2, f"page {page}: {advice}")

    for index in range(len(PAGES)):
        hub.go_to(index)
        app.processEvents()
    check("every page builds", True)

    check("annotate is blocked without a cache",
          "no clips" in hub.pages[3].blocker(), hub.pages[3].blocker())
    check("model is blocked without labels",
          "nothing labelled" in hub.pages[4].blocker(), hub.pages[4].blocker())
    export = hub.page(ExportPage)
    check("export is blocked without annotations",
          "no annotations" in export.blocker(), export.blocker())
    check("distribution is blocked without annotations",
          "no annotations" in hub.page(DistributionPage).blocker(),
          hub.page(DistributionPage).blocker())

    # The config page is a real editor: change a value, save, and the file
    # must both change and still load.
    hub.go_to(1)
    app.processEvents()
    config_page = hub.pages[1]
    check("basics tab is a short list",
          5 < len(config_page.form.values()) < 25, str(len(config_page.form.values())))
    check("advanced tab holds the rest",
          len(config_page.advanced.values()) > len(config_page.form.values()))
    check("no setting appears on both tabs",
          not (set(config_page.form.values()) & set(config_page.advanced.values())))

    config_page.form._widgets["clips.pad"].setValue(2.75)
    # ...and one from the OTHER tab, which an earlier version silently dropped.
    config_page.advanced._widgets["detection.conf"].setValue(0.4)
    app.processEvents()
    check("editing schedules an autosave",
          config_page._dirty and config_page._save_timer.isActive())
    check("the autosave fires", pump(app, lambda: not config_page._dirty, timeout=10))
    check("the edit reached the file", abs(hub.cfg.clips.pad - 2.75) < 1e-9, str(hub.cfg.clips.pad))
    check("an advanced-tab edit is saved too",
          abs(hub.cfg.detection.conf - 0.4) < 1e-9, str(hub.cfg.detection.conf))
    check("saving keeps the comments",
          project_config.read_text(encoding="utf-8").count("#") > 40)
    check("saving does not wipe the form", config_page.form.values()["clips.pad"] == 2.75)

    # Reloading after a save must not re-arm the timer, or autosave loops.
    check("a save does not schedule another save", not config_page._save_timer.isActive())

    # An unusable label set must fail LOUDLY and leave the file alone.
    good_text = project_config.read_text(encoding="utf-8")
    name_field = config_page.categories._groups[0].name
    name_field.setText("")
    app.processEvents()
    check("the autosave refuses an unusable label set",
          not config_page.save() and config_page.banner.isVisible())
    check("the refusal says why",
          "no name" in config_page.banner.text(), config_page.banner.text()[:120])
    check("the refused save left the file alone",
          project_config.read_text(encoding="utf-8") == good_text)
    name_field.setText("state")
    app.processEvents()
    check("fixing it saves again", pump(app, lambda: not config_page._dirty, timeout=10))
    check("the error banner clears", not config_page.banner.isVisible())

    # A path that points at nothing is a WARNING, not a blocked save: a model on
    # an offline share must not stop you editing anything else.
    config_page.form._widgets["paths.model"].set_value(str(project_config.parent / "nope.pt"))
    app.processEvents()
    check("a missing path still saves", pump(app, lambda: not config_page._dirty, timeout=10))
    check("but it is flagged", "paths.model" in config_page._show_warnings())
    check("and it is not an error banner", not config_page.banner.isVisible())

    # Leaving the page must never prompt - it just flushes.
    config_page.form._widgets["clips.pad"].setValue(3.0)
    hub.go_to(0)
    app.processEvents()
    check("navigating away flushes instead of prompting",
          not config_page._dirty and abs(hub.cfg.clips.pad - 3.0) < 1e-9)

    hub.close()
    app.processEvents()


def test_hub_with_cache(app) -> None:
    """The annotate page inside the hub, against a generated clip cache."""
    from trackanno.config import load_config
    from trackanno.gui.hub import DistributionPage, ExportPage, Hub

    repo_config = make_project(FIXTURES)
    cfg = load_config(repo_config)

    hub = Hub(str(repo_config))
    hub.show()
    hub.go_to(3)
    app.processEvents()
    panel = hub.pages[3].panel
    check("the annotate page builds inside the hub", panel is not None)
    check("it loaded the whole cache", len(panel.order) == hub.status.n_clips,
          f"{len(panel.order)} vs {hub.status.n_clips}")

    # Leaving the page and coming back must not lose the position or leak a panel.
    position = panel.position
    hub.go_to(0)
    app.processEvents()
    hub.go_to(3)
    app.processEvents()
    check("returning to annotate keeps the same panel", hub.pages[3].panel is panel)
    check("returning keeps the position", hub.pages[3].panel.position == position)

    # Autosave reloads the project every few seconds while you edit settings.
    # That must not throw away the clip you were looking at.
    panel.position = min(3, len(panel.order) - 1)
    panel._load()
    at = panel.position
    hub.reload_project()
    app.processEvents()
    check("an unrelated config reload keeps the annotation panel",
          hub.pages[3].panel is panel)
    check("...and keeps the clip you were on", hub.pages[3].panel.position == at)

    hub.close()
    app.processEvents()


def test_later_steps_unlock(app) -> None:
    """Model and Export must unlock as you annotate, not only on reopen.

    They gate on the annotation count, but the per-clip update deliberately
    skips the full project refresh (it would tear the annotation panel down), so
    nothing re-checked them: the buttons stayed greyed out for the whole session
    however much you labelled.
    """
    import shutil as _shutil

    from trackanno.config import load_config
    from trackanno.gui.hub import DistributionPage, ExportPage, Hub
    from trackanno.project import write_config

    config_path = make_project(FIXTURES)
    repo = config_path.parent
    cfg = load_config(config_path)

    scratch = Path(tempfile.mkdtemp(prefix="trackanno-unlock-"))
    try:
        project = scratch / "p"
        project.mkdir()
        _shutil.copy(config_path, project / "config.yaml")
        # Same clip cache, but a CSV that starts empty.
        write_config(project / "config.yaml", {
            "paths.work_dir": str(cfg.paths.work_dir).replace("\\", "/"),
            "paths.csv": str(project / "fresh.csv").replace("\\", "/"),
        })
        hub = Hub(str(project / "config.yaml"))
        hub.show()
        app.processEvents()
        check("Model starts locked with no labels",
              not hub.pages[4].tasks.run_button.isEnabled())
        check("Export starts locked with no labels",
              not hub.page(ExportPage).tasks.run_button.isEnabled())

        hub.go_to(3)
        app.processEvents()
        panel = hub.pages[3].panel
        group = hub.cfg.categories[0]
        for _ in range(3):
            panel.categories.toggle(group.name, group.options[0].name)
            app.processEvents()

        check("labels reached the CSV", len(panel.annotations) == 3)
        check("the project-wide count is used, not the filtered one",
              hub.status.n_annotated == len(panel.annotations))
        check("Model unlocks while still on the Annotate page",
              hub.pages[4].tasks.run_button.isEnabled())
        check("Export unlocks too",
              hub.page(ExportPage).tasks.run_button.isEnabled())

        hub.go_to(hub.pages.index(hub.page(ExportPage)))
        app.processEvents()
        check("Export lists the movies it can render",
              hub.page(ExportPage).movies.count() > 0)

        # The new step between Model and Export reads the same CSV.
        hub.go_to(hub.pages.index(hub.page(DistributionPage)))
        app.processEvents()
        dist = hub.page(DistributionPage)
        check("Distribution charts every category group",
              len(dist._charts) == len(hub.cfg.categories), repr(list(dist._charts)))
        counted = sum(b.count for b in dist._charts[group.name].tally.bars
                      if b.name == group.options[0].name)
        check("and counts what was actually labelled", counted == 3, str(counted))
        hub.close()
        app.processEvents()
    finally:
        _shutil.rmtree(scratch, ignore_errors=True)

    hub.close()
    app.processEvents()
