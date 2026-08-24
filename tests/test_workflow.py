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
    from trackanno.gui.hub import PAGES, ConfigPage, DistributionPage, ExportPage, Hub

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

    # Downloading a shared model is the usual way to get one, so its button sits
    # in the detection model row rather than off in a corner.
    from qtpy.QtWidgets import QPushButton

    row = config_page.form._widgets["paths.model"].row
    labels = [row.itemAt(i).widget().text() for i in range(row.count())
              if isinstance(row.itemAt(i).widget(), QPushButton)]
    check("the shared-model button leads the detection model row",
          labels == ["Choose shared model...", "Browse"], str(labels))
    check("and says what it does rather than where it comes from",
          "shared model" in labels[0].lower())

    # A project running a downloaded model still has a file path in
    # paths.model, so the field on its own reads as the wrong answer.
    field = config_page.form._widgets["paths.model"]
    # isHidden, not isVisible: the page itself may not be the one on screen.
    check("nothing is claimed while the project uses a local file",
          field.status.isHidden() and not field.status.text(), field.status.text())

    from trackanno.project import write_config as _write

    _write(project_config, {"model.source": "some-model", "model.version": "v3"})
    hub.open_project(project_config)
    app.processEvents()
    field = next(p for p in hub.pages if isinstance(p, ConfigPage)).form._widgets["paths.model"]
    check("with a shared model it names the one in use",
          "some-model v3" in field.status.text(), field.status.text())
    check("and says the path is not what runs",
          "not used" in field.status.text() and not field.status.isHidden())
    check("the path is greyed out, since nothing reads it", not field.edit.isEnabled())
    check("and carries no amber outline for pointing at nothing",
          "paths.model" not in (config_page._show_warnings() or {}))
    _write(project_config, {"model.source": ""})
    hub.open_project(project_config)
    app.processEvents()

    # What is wrong on Detect & crop is a sentence, so it goes under the
    # buttons rather than into the strip left over beside them.
    from trackanno.gui.hub import PreparePage

    prepare_page = next(p for p in hub.pages if isinstance(p, PreparePage))
    box = prepare_page.scan_state.parentWidget().layout()
    rows = [box.itemAt(i) for i in range(box.count())]
    button_row = next(i for i, it in enumerate(rows) if it.layout() is not None)
    label_row = next(i for i, it in enumerate(rows)
                     if it.widget() is prepare_page.scan_state)
    check("the scan message sits below the button row, not in it",
          label_row == button_row + 1, f"buttons at {button_row}, message at {label_row}")
    check("and wraps rather than being clipped", prepare_page.scan_state.wordWrap())

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


# --------------------------------------------------------------------------- #
# config templates
# --------------------------------------------------------------------------- #
def test_a_template_carries_settings_but_not_paths(root) -> None:
    """The point of a template: keep the tuning, drop the project it came from.

    Tuning the detector, the crop geometry and the category buttons is most of
    the setup work for a dataset, and the next dataset wants the same answers.
    What it must NOT bring is the previous project's movie folder or clip cache.
    """
    from trackanno.config import load_config
    from trackanno.project import create_project, write_config
    from trackanno.templates import list_templates, save_template

    videos = root / "tpl-videos"
    videos.mkdir()
    weights = root / "tpl-model.pt"
    weights.write_bytes(b"")

    source = create_project(root / "tpl-a", video_dir=videos, model=weights)
    write_config(source, {
        "clips.frames": 33, "clips.min_clip_frames": 25, "clips.crop_size": 128,
        "detection.imgsz": 1280, "detection.conf": 0.4,
        "tracking.backend": "botsort", "ui.annotator": "someone",
    }, categories=[{"name": "state", "type": "single", "required": True,
                    "options": [{"name": "swimming", "key": "1", "color": "#4C9F70"},
                                {"name": "stuck", "key": "2", "color": "#D95F02"}]}])

    template = save_template("Trypanosome 100 fps", source)
    check("the typed name is kept, punctuation and all",
          template.name == "Trypanosome 100 fps", template.name)
    check("the file name is a slug of it",
          template.path.name == "trypanosome-100-fps.yaml", template.path.name)
    check("it appears in the picker after the built-in",
          [t.name for t in list_templates()] == ["default", "Trypanosome 100 fps"])

    other = root / "tpl-videos-2"
    other.mkdir()
    made = create_project(root / "tpl-b", video_dir=other, template="Trypanosome 100 fps")
    cfg = load_config(made)

    check("the tuning came across",
          (cfg.clips.frames, cfg.detection.imgsz, cfg.tracking.backend)
          == (33, 1280, "botsort"),
          f"{cfg.clips.frames}, {cfg.detection.imgsz}, {cfg.tracking.backend}")
    check("so did the category buttons",
          [g.name for g in cfg.categories] == ["state"]
          and [o.name for o in cfg.categories[0].options] == ["swimming", "stuck"])
    check("the model came across, since weights are reused between projects",
          cfg.paths.model == weights.resolve(), str(cfg.paths.model))
    check("the movie folder did NOT: this project has its own",
          cfg.paths.video_dir == other.resolve(), str(cfg.paths.video_dir))
    check("nor did the clip cache",
          cfg.paths.work_dir == (root / "tpl-b" / "work").resolve(), str(cfg.paths.work_dir))
    check("nor did the annotation log",
          cfg.paths.csv == (root / "tpl-b" / "annotations.csv").resolve(), str(cfg.paths.csv))
    check("nor the annotator: a template is a protocol, not a person",
          cfg.ui.annotator == "", repr(cfg.ui.annotator))

    text = made.read_text(encoding="utf-8")
    check("the comments that document every setting survived", text.count("#") > 40)
    check("and the template's own header line did not leak into the project",
          not text.startswith("# trackanno template:"), text.splitlines()[0][:40])


def test_a_template_can_be_found_however_it_is_named(root) -> None:
    """The typed name, its slug and the file name all mean the same template."""
    from trackanno.config import load_config
    from trackanno.project import create_project
    from trackanno.templates import delete_template, find_template, save_template

    source = create_project(root / "tpl-c")
    save_template("Wide Field 20x", source)
    for spelling in ("Wide Field 20x", "wide field 20x", "wide-field-20x"):
        found = find_template(spelling)
        check(f"{spelling!r} finds it", found is not None and found.name == "Wide Field 20x")
    check("an unknown name finds nothing", find_template("no such thing") is None)
    check("and creating from it falls back to the built-in rather than failing",
          load_config(create_project(root / "tpl-d", template="no such thing")).clips.frames
          == 50)

    check("the built-in cannot be deleted", not delete_template("default"))
    check("a saved one can", delete_template("Wide Field 20x"))
    check("and is then gone", find_template("Wide Field 20x") is None)



def test_a_template_that_would_not_load_is_refused(root) -> None:
    """Storing a broken template only moves the failure to project creation."""
    from trackanno.project import create_project
    from trackanno.templates import find_template, save_template

    source = create_project(root / "tpl-e")
    text = source.read_text(encoding="utf-8")
    source.write_text(text.replace("frames: 50", "frames: not-a-number"), encoding="utf-8")

    try:
        save_template("broken", source)
        refused = False
    except Exception:
        refused = True
    check("a template that cannot be parsed is refused at save time", refused)
    check("and nothing was stored", find_template("broken") is None)


def test_saving_a_template_twice_needs_permission(root) -> None:
    """Overwriting somebody's tuned settings is a decision, not a default."""
    from trackanno.project import create_project
    from trackanno.templates import delete_template, save_template

    source = create_project(root / "tpl-f")
    save_template("dup", source)
    try:
        save_template("dup", source)
        refused = False
    except FileExistsError:
        refused = True
    check("a second save under the same name is refused", refused)
    check("unless overwrite is asked for",
          save_template("dup", source, overwrite=True).name == "dup")
    delete_template("dup")


def test_the_new_project_dialog_offers_the_templates(app, root) -> None:
    """The picker is the only place most people will meet this."""
    from trackanno.gui.hub import NewProjectDialog
    from trackanno.project import create_project
    from trackanno.templates import BUILTIN_NAME, delete_template, save_template

    source = create_project(root / "tpl-g")
    save_template("from the dialog", source)

    dialog = NewProjectDialog()
    items = [dialog.template.itemText(i) for i in range(dialog.template.count())]
    check("the built-in is offered first and marked as such",
          items[0] == "default (built in)", str(items))
    check("the saved one is offered too", "from the dialog" in items, str(items))
    check("the default selection is the built-in",
          dialog.values()["template"] == BUILTIN_NAME)
    check("the built-in cannot be forgotten", not dialog.forget_btn.isEnabled())

    dialog.template.setCurrentIndex(items.index("from the dialog"))
    check("a saved one can be", dialog.forget_btn.isEnabled())
    check("and choosing it is what create_project is handed",
          dialog.values()["template"] == "from the dialog")
    dialog.close()
    delete_template("from the dialog")


def test_export_can_open_the_project_folder(app, root, monkeypatch) -> None:
    """The rendered movies are files somebody then does something else with."""
    from trackanno.gui import hub as H
    from trackanno.gui.hub import ExportPage, Hub

    project = make_project(root / "reveal", pose=True)
    window = Hub(str(project))
    try:
        page = next(p for p in window.pages if isinstance(p, ExportPage))
        check("the button is on the Export page", page.open_folder.text() == "Open project folder")

        opened: list = []
        monkeypatch.setattr(H.os, "startfile", opened.append, raising=False)
        monkeypatch.setattr(H.sys, "platform", "win32")
        page._open_project_folder()
        check("it opens the folder the config lives in",
              opened == [str(project.parent)], str(opened))

        check("a folder that is not there is reported, not raised",
              "does not exist" in H._reveal(root / "gone"))
    finally:
        window.close()
        app.processEvents()


def test_browsing_to_a_file_stops_using_the_shared_model(app, root) -> None:
    """paths.model is what you see; model.source is what runs.

    Picking a local .pt while a downloaded model is configured used to change
    only the visible half: the field showed the file, the project kept running
    the shared weights, and the page kept saying so.
    """
    from trackanno.config import load_config
    from trackanno.gui.hub import ConfigPage, Hub
    from trackanno.project import write_config as _write

    project = make_project(root / "swap")
    _write(project, {"model.source": "some-model", "model.version": "v3",
                     "model.sha256": "ab" * 32})
    local = project.parent / "small.pt"
    local.write_bytes(b"not really weights")

    window = Hub(str(project))
    try:
        page = next(p for p in window.pages if isinstance(p, ConfigPage))
        field = page.form._widgets["paths.model"]
        check("the shared model is in use to begin with", not field.edit.isEnabled())

        # Exactly what the Browse dialog does once a file has been chosen.
        field.edit.setText(str(local).replace("\\", "/"))
        field.picked.emit(str(local).replace("\\", "/"))
        check("the save lands", pump(app, lambda: not page._dirty, timeout=10))

        after = load_config(project)
        check("the shared model is dropped", not after.model.is_remote,
              repr(after.model.source))
        check("its pin goes with it", not after.model.version and not after.model.sha256)
        check("and the picked file is what the project runs",
              after.paths.model.name == "small.pt", str(after.paths.model))
        check("the page stops claiming a shared model", not field.status.text(),
              field.status.text())
        check("and the path is editable again", field.edit.isEnabled())
    finally:
        window.close()
        app.processEvents()


def test_the_device_list_says_when_it_is_still_looking(app, root) -> None:
    """Importing torch takes seconds, and the list is auto/cpu until it lands.

    A settings page showing only 'cpu' reads as "no GPU on this machine", which
    is the wrong answer while the probe has simply not finished.
    """
    from trackanno.gui.hub import ConfigPage, Hub

    project = make_project(root / "devices")
    window = Hub(str(project))
    try:
        page = next(p for p in window.pages if isinstance(p, ConfigPage))
        app.processEvents()
        status = page.form._status["detection.device"]
        check("the device row starts by saying the probe is running",
              "checking" in status.text().lower(), status.text())
        check("and it is visible, not a tooltip nobody hovers", not status.isHidden())

        check("the probe answers", pump(app, lambda: "checking" not in status.text().lower(),
                                        timeout=120))
        check("and it says what auto resolves to here",
              status.text().startswith("auto uses"), status.text())
        items = [page.form._widgets["detection.device"].itemText(i)
                 for i in range(page.form._widgets["detection.device"].count())]
        check("with auto and cpu always offered", {"auto", "cpu"} <= set(items), str(items))
    finally:
        window.close()
        app.processEvents()


def test_the_device_list_and_the_resolver_cannot_disagree() -> None:
    """One probe answers both, or the page shows one thing and the run does another.

    The list counted CUDA devices while ``resolve_device`` only asked whether
    CUDA was available at all and then assumed device 0. Those two questions are
    answered by different layers of torch and are allowed to differ - so the
    Configure page could offer no GPU while ``auto`` picked one regardless, which
    is a run landing somewhere nobody was ever shown.
    """
    from trackanno import tiling

    offered = {value for value, _label in tiling.describe_devices()}
    check("the list always offers auto and cpu", {"auto", "cpu"} <= offered, str(offered))

    indices = tiling.cuda_devices()
    check("the listed GPUs are exactly the usable ones",
          {str(i) for i in indices} <= offered, f"{indices} vs {offered}")

    resolved = tiling.resolve_device("auto")
    check("and auto only ever resolves to something on that list",
          resolved in offered or resolved == "cpu", f"auto -> {resolved}, offered {offered}")
    if indices:
        check("auto takes the first usable GPU", resolved == str(indices[0]), resolved)
    else:
        check("auto falls back rather than assuming device 0",
              resolved != "0", resolved)

    check("an index nobody offered is refused, not handed to ultralytics",
          tiling.resolve_device("99") == "cpu", tiling.resolve_device("99"))


def test_describe_auto_says_why_it_landed_on_the_cpu() -> None:
    """"cpu" alone cannot be acted on; the reason is what tells you the fix."""
    from trackanno import tiling

    sentence = tiling.describe_auto()
    check("it is one sentence about auto", sentence.startswith("auto uses"), sentence)
    if tiling.resolve_device("auto") == "cpu":
        check("and a CPU answer explains itself", ":" in sentence, sentence)

    usable, why = tiling.cuda_status()
    check("a refusal always carries a reason", usable or bool(why), repr((usable, why)))
    check("and a working GPU carries none", not usable or not why, repr(why))

    check("a CUDA index is named with its card",
          tiling.device_label("cpu") == "the CPU" and "mps" in tiling.device_label("mps"),
          tiling.device_label("mps"))
