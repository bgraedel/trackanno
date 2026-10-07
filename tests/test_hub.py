"""The hub window: background jobs, project switching, and the Configure page."""

from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path

import pytest
from qtpy.QtCore import Qt
from fixtures import pump


@pytest.fixture
def project(tmp_path: Path) -> Path:
    from trackanno.project import create_project

    return create_project(tmp_path / "proj")


@pytest.fixture
def hub(app, project):
    from trackanno.gui.hub import Hub

    hub = Hub(str(project))
    yield hub
    hub.close()
    app.processEvents()


@pytest.fixture
def config_page(app, hub):
    """The Configure page, shown, with a scripted unsaved-edits answer queue."""
    from trackanno.gui.hub import ConfigPage

    hub.show()
    page = hub.page(ConfigPage)
    hub.go_to(hub.pages.index(page))
    app.processEvents()
    page.answers = []
    page.asked = []

    def ask():
        page.asked.append(1)
        return page.answers.pop(0)

    page._ask_unsaved = ask
    yield page
    page.answers[:] = ["discard"] * 4


def _held(gate: threading.Event, value):
    def work(reporter):
        gate.wait(5)
        return value
    return work


# --------------------------------------------------------------------------- #
# background jobs
# --------------------------------------------------------------------------- #
def test_a_job_from_a_closed_project_is_dropped(app, hub, tmp_path: Path) -> None:
    from trackanno.gui.hub import ProjectPage
    from trackanno.project import create_project

    page = hub.page(ProjectPage)
    landed: list = []
    gate = threading.Event()
    page.run_bg("probe", _held(gate, "old project"), landed.append)
    hub.open_project(create_project(tmp_path / "other"))
    gate.set()
    pump(app, lambda: not page.bg_running("probe"), timeout=1)
    time.sleep(0.1)
    app.processEvents()
    assert landed == []


def test_a_superseded_job_of_the_same_name_is_dropped(app, hub) -> None:
    from trackanno.gui.hub import ProjectPage

    page = hub.page(ProjectPage)
    landed: list = []
    gate = threading.Event()
    try:
        page.run_bg("probe", _held(gate, "superseded"), landed.append)
        page.run_bg("probe", lambda reporter: "newest", landed.append)
        assert pump(app, lambda: "newest" in landed), "the newest job lands"
        gate.set()
        time.sleep(0.1)
        app.processEvents()
        assert landed == ["newest"]
    finally:
        gate.set()


def test_a_failed_job_is_delivered_as_a_message(app, hub) -> None:
    from trackanno.gui.hub import ProjectPage

    page = hub.page(ProjectPage)
    failures: list = []
    page.run_bg("broken", lambda reporter: 1 / 0, lambda _r: None, failures.append)
    assert pump(app, lambda: bool(failures))
    assert "ZeroDivisionError" in failures[0], failures


def test_closing_joins_a_running_job(app, project) -> None:
    from trackanno.gui.hub import Hub, ProjectPage

    hub = Hub(str(project))
    page = hub.page(ProjectPage)
    landed: list = []

    def until_cancelled(reporter):
        while True:
            reporter.raise_if_cancelled()
            time.sleep(0.01)

    page.run_bg("slow", until_cancelled, landed.append)
    hub.close()
    app.processEvents()
    assert not any(t.name == "trackanno-slow" and t.is_alive()
                   for t in threading.enumerate())
    app.processEvents()
    assert landed == [], "nothing lands after close"


def test_a_dropped_download_clears_its_status(app, hub, project, monkeypatch) -> None:
    from types import SimpleNamespace

    from trackanno.gui.hub import ConfigPage, _install_shared_model

    gate = threading.Event()
    monkeypatch.setattr("trackanno.video.models.fetch",
                        lambda entry, reporter: gate.wait(5) and project)
    page = hub.page(ConfigPage)
    bar = hub.statusBar()
    try:
        _install_shared_model(page, SimpleNamespace(id="shared"), project)
        app.processEvents()
        assert bar.currentMessage() == "fetching shared...", bar.currentMessage()
        page._jobs.cancel()
        app.processEvents()
        assert bar.currentMessage() == "shared: download stopped", bar.currentMessage()

        _install_shared_model(page, SimpleNamespace(id="shared"), project)
        hub.project_gen += 1                  # the project changed under the job
        gate.set()
        assert pump(app, lambda: not page.bg_running("download")), "the download lands"
        app.processEvents()
        assert bar.currentMessage() == "shared: download stopped", bar.currentMessage()
    finally:
        gate.set()


def test_a_refresh_leaves_a_running_cancel_enabled(app, hub) -> None:
    from trackanno.gui.hub import ExportPage, ModelPage

    hub.show()
    app.processEvents()
    pages = (hub.page(ModelPage), hub.page(ExportPage))
    try:
        for page in pages:
            page.tasks.start(lambda reporter: time.sleep(1.5))
        app.processEvents()
        assert all(page.tasks.busy for page in pages), "both jobs are running"
        hub.note_progress(7)              # fires refresh_gate on every page
        app.processEvents()
        assert all(page.tasks.run_button.isEnabled() for page in pages), "labelling"
        hub.go_to(4)
        app.processEvents()
        assert pages[0].tasks.run_button.isEnabled(), "navigating"
    finally:
        for page in pages:
            page.tasks.shutdown()


# --------------------------------------------------------------------------- #
# opening projects
# --------------------------------------------------------------------------- #
def test_a_window_with_no_project_still_paints_its_sidebar(app, tmp_path, monkeypatch) -> None:
    from trackanno.gui.hub import Hub

    monkeypatch.setenv("TRACKANNO_STATE_DIR", str(tmp_path / "empty-state"))
    hub = Hub(None)
    try:
        hub.show()
        app.processEvents()
        assert hub.sidebar.item(0).text().strip().startswith("1."), (
            hub.sidebar.item(0).text())
        assert not (hub.sidebar.item(3).flags() & Qt.ItemIsEnabled), (
            "later steps are disabled")
    finally:
        hub.close()


def test_a_failed_open_leaves_no_project_live(app, tmp_path: Path) -> None:
    from trackanno.gui.hub import Hub
    from trackanno.project import create_project

    good = create_project(tmp_path / "good")
    bad = create_project(tmp_path / "bad")
    bad.write_text("categories: [", encoding="utf-8")
    hub = Hub(str(good))
    try:
        hub.show()
        app.processEvents()
        assert hub.cfg is not None
        assert not hub.open_project(str(bad), quiet=True)
        app.processEvents()
        assert hub.cfg is None and hub.status is None
        assert "good" not in hub.windowTitle()
        assert hub.pages[3].panel is None, "no annotation panel survives"
    finally:
        hub.close()


# --------------------------------------------------------------------------- #
# the annotate page
# --------------------------------------------------------------------------- #
@pytest.fixture
def fixture_hub(app, root):
    """A hub on the generated project with a clip cache, Annotate page shown."""
    from fixtures import make_project

    from trackanno.gui.hub import Hub

    hub = Hub(str(make_project(root)))
    hub.show()
    hub.go_to(3)
    app.processEvents()
    yield hub
    hub.close()
    app.processEvents()


def test_a_cache_job_locks_the_annotate_page(app, fixture_hub) -> None:
    hub = fixture_hub
    assert hub.pages[3].panel is not None, "the annotate page loads normally"

    hub.set_cache_job(True)
    app.processEvents()
    assert hub.pages[3].panel is None, "a running cache job tears the panel down"
    assert "detection is rewriting" in hub.pages[3].blocker()
    hub.go_to(3)
    app.processEvents()
    assert hub.pages[3].panel is None, "navigating back does not reopen it"

    hub.set_cache_job(False)
    hub.go_to(3)
    app.processEvents()
    assert hub.pages[3].panel is not None, "it reopens once the job finishes"


def test_a_new_model_file_keeps_the_annotate_panel(app, fixture_hub) -> None:
    from trackanno.learn.suggest import MODEL_FILE

    page = fixture_hub.pages[3]
    panel, fingerprint = page.panel, page._fingerprint()
    model = fixture_hub.cfg.paths.work_dir / MODEL_FILE
    model.write_bytes(b"not a model")
    fixture_hub.reload_project()
    app.processEvents()
    assert page._fingerprint() == fingerprint
    assert page.panel is panel


def test_training_hands_the_new_model_to_the_open_annotator(app, fixture_hub) -> None:
    from trackanno.gui.hub import ModelPage

    panel = fixture_hub.pages[3].panel
    calls: list = []
    panel.reload_suggester = lambda: calls.append(1) or "reloaded"
    page = fixture_hub.page(ModelPage)
    page._job = "train"
    page._on_finished(True)
    assert calls == [1]


def test_an_annotator_rename_reaches_the_csv(app, root, tmp_path) -> None:
    from fixtures import make_project

    from trackanno.config import load_config
    from trackanno.gui.hub import Hub
    from trackanno.project import write_config

    fixture = make_project(root)
    config = tmp_path / "config.yaml"
    shutil.copy(fixture, config)
    write_config(config, {
        "paths.work_dir": str(load_config(fixture).paths.work_dir).replace("\\", "/"),
        "paths.csv": str(tmp_path / "fresh.csv").replace("\\", "/"),
        "ui.annotator": "alice",
    })
    hub = Hub(str(config))
    try:
        hub.show()
        hub.go_to(3)
        app.processEvents()
        before = hub.pages[3]._fingerprint()

        write_config(config, {"ui.annotator": "bob"})
        hub.reload_project()
        app.processEvents()
        assert hub.pages[3]._fingerprint() != before, "the rename moves the fingerprint"

        hub.go_to(3)
        app.processEvents()
        panel = hub.pages[3].panel
        group = hub.cfg.categories[0]
        panel.categories.toggle(group.name, group.options[0].name)
        app.processEvents()
        rows = list(panel.annotations.rows.values())
        assert rows and rows[-1]["annotator"] == "bob", rows[-1:] or "no rows"
    finally:
        hub.close()
        app.processEvents()


# --------------------------------------------------------------------------- #
# Configure: Save and Revert
# --------------------------------------------------------------------------- #
def test_an_edit_enables_save_and_writes_nothing(app, config_page, project) -> None:
    from trackanno.config import load_config

    page = config_page
    assert not page.save_button.isEnabled() and not page.revert_button.isEnabled()
    page.form._widgets["clips.pad"].setValue(2.5)
    app.processEvents()
    assert page.save_button.isEnabled() and page.revert_button.isEnabled()
    assert page.heading.title.text() == "Configure *", page.heading.title.text()
    assert load_config(project).clips.pad != 2.5


def test_save_writes_only_the_changed_key(app, hub, config_page, project) -> None:
    from trackanno.config import load_config
    from trackanno.gui.hub import ExportPage
    from trackanno.project import write_config

    page = config_page
    write_config(project, {"detection.conf": 0.125})    # a value the spin box rounds
    page.revert()
    app.processEvents()
    page.form._widgets["clips.pad"].setValue(2.5)
    app.processEvents()

    before = project.read_text(encoding="utf-8")
    assert page.save()
    after = project.read_text(encoding="utf-8")
    changed = [(a, b) for a, b in zip(before.splitlines(), after.splitlines()) if a != b]
    assert len(changed) == 1 and "pad" in changed[0][1], changed
    assert load_config(project).detection.conf == 0.125, "the rounded value keeps its precision"
    assert not page.save_button.isEnabled() and page.heading.title.text() == "Configure"
    assert hub.cfg.clips.pad == 2.5, "the window holds the saved config"

    export = hub.page(ExportPage)
    assert export.rebuild_pending and not page.rebuild_pending, "other pages wait to be shown"
    hub.go_to(hub.pages.index(export))
    app.processEvents()
    assert not export.rebuild_pending, "rebuilt on arrival"


def test_revert_reads_the_file_back(app, config_page, project) -> None:
    page = config_page
    before = project.read_text(encoding="utf-8")
    saved_pad = page.form.current()["clips.pad"]
    page.form._widgets["clips.pad"].setValue(4.0)
    page.categories._groups[0].name.setText("renamed")
    app.processEvents()
    page.revert()
    app.processEvents()
    assert page.form.current()["clips.pad"] == saved_pad
    assert page.categories._groups[0].name.text() != "renamed"
    assert not page._dirty and not page.save_button.isEnabled()
    assert project.read_text(encoding="utf-8") == before


def test_a_clean_page_leaves_without_asking(app, hub, config_page) -> None:
    hub.go_to(0)
    app.processEvents()
    assert not config_page.asked and hub.stack.currentWidget() is not config_page


def _edit_and_leave(app, hub, page, answer: str) -> None:
    page.form._widgets["clips.pad"].setValue(1.5)
    app.processEvents()
    page.answers.append(answer)
    hub.go_to(0)
    app.processEvents()


def test_cancel_stays_on_the_page(app, hub, config_page) -> None:
    index = hub.pages.index(config_page)
    _edit_and_leave(app, hub, config_page, "cancel")
    assert hub.stack.currentWidget() is config_page and config_page._dirty
    assert hub.sidebar.currentRow() == index, "the sidebar stays on the page"


def test_discard_leaves_without_writing(app, hub, config_page, project) -> None:
    from trackanno.config import load_config

    _edit_and_leave(app, hub, config_page, "discard")
    assert hub.stack.currentWidget() is not config_page
    assert load_config(project).clips.pad != 1.5
    pad = config_page.form._widgets["clips.pad"]
    assert pad.value() != 1.5 and not config_page._dirty, "the form is reverted"


def test_save_leaves_and_writes(app, hub, config_page, project) -> None:
    from trackanno.config import load_config

    _edit_and_leave(app, hub, config_page, "save")
    assert hub.stack.currentWidget() is not config_page
    assert load_config(project).clips.pad == 1.5


def test_closing_with_unsaved_edits_asks(app, hub, config_page) -> None:
    config_page.form._widgets["clips.pad"].setValue(2.25)
    app.processEvents()
    config_page.answers.append("cancel")
    hub.close()
    app.processEvents()
    assert hub.isVisible() and len(config_page.asked) == 1, "Cancel keeps it open"
    config_page.answers.append("discard")
    hub.close()
    app.processEvents()
    assert not hub.isVisible(), "Discard lets the window close"


# --------------------------------------------------------------------------- #
# Configure: the model probe
# --------------------------------------------------------------------------- #
def test_loading_the_form_emits_no_change(app) -> None:
    from trackanno.gui.configform import ConfigForm

    form = ConfigForm("all")
    fired: list[int] = []
    form.changed.connect(lambda: fired.append(1))
    form.load({"paths": {"video_dir": "/a", "model": "/b.pt", "work_dir": "/c",
                         "csv": "/d.csv"}, "detection": {"task": "pose"}})
    assert fired == []


def test_the_model_probe_offers_rather_than_applies(app, project, monkeypatch) -> None:
    from trackanno.gui.hub import ConfigPage, Hub

    (project.parent / "best.pt").write_bytes(b"weights")
    monkeypatch.setattr("trackanno.detect.tiling.read_model_info", lambda p: {
        "task": "pose", "n_keypoints": 8, "classes": ["cell"], "name": "best.pt"})
    before = project.read_text(encoding="utf-8")
    hub = Hub(str(project))
    page = hub.page(ConfigPage)
    hub.go_to(hub.pages.index(page))
    try:
        assert pump(app, lambda: not page.model_row.isHidden()), "the probe offers"
        assert (not page._dirty
                and page.form._widgets["detection.task"].currentText() == "detect"), (
            "without touching the form")
        page.model_apply.click()
        app.processEvents()
        assert (page._dirty
                and page.form._widgets["detection.task"].currentText() == "pose"
                and page.keypoints.count.value() == 8), "a click applies them"
        page.revert()
        app.processEvents()
        assert not page._dirty
        assert project.read_text(encoding="utf-8") == before, "the file was never written"
    finally:
        page._dirty = False
        hub.close()
        app.processEvents()


def _apply_checkpoint(app, page, task: str, n_keypoints: int) -> None:
    page._on_model_info({"task": task, "n_keypoints": n_keypoints, "classes": ["cell"],
                         "imgsz": 640, "name": f"yolo11n-{task}.pt"})
    app.processEvents()
    page.model_apply.click()
    app.processEvents()


def test_a_pose_checkpoint_saves_task_and_keypoints_together(app, config_page, project) -> None:
    from trackanno.config import load_config

    page = config_page
    assert page.form._widgets["detection.task"].currentText() == "detect"
    _apply_checkpoint(app, page, "pose", 17)
    assert page.form._widgets["detection.task"].currentText() == "pose"
    assert page.keypoints.count.value() == 17, "the names table is sized to match"
    assert page.save()
    saved = load_config(project)
    assert saved.detection.task == "pose"
    assert len(saved.keypoints.nodes) == 17, saved.keypoints.nodes
    assert not saved.needs_keypoints


def test_a_detect_checkpoint_keeps_the_typed_keypoint_names(app, hub, project) -> None:
    from trackanno.config import load_config
    from trackanno.gui.hub import ConfigPage
    from trackanno.project import write_config

    nodes = [f"k{i}" for i in range(17)]
    write_config(project, {"detection.task": "pose"},
                 keypoints={"nodes": nodes, "edges": []})
    hub.reload_project()
    page = hub.page(ConfigPage)
    hub.go_to(hub.pages.index(page))
    app.processEvents()
    _apply_checkpoint(app, page, "detect", 0)
    assert page.form._widgets["detection.task"].currentText() == "detect"
    assert page.save()
    back = load_config(project)
    assert back.detection.task == "detect"
    assert len(back.keypoints.nodes) == 17, "the names survive"
    assert not back.needs_keypoints


def test_a_dead_share_is_never_probed_on_the_ui_thread(app, tmp_path, monkeypatch) -> None:
    from trackanno.gui.hub import Hub
    from trackanno.project import create_project, write_config

    config = create_project(tmp_path / "proj")
    dead = tmp_path / "deadshare"
    write_config(config, {"paths.video_dir": str(dead / "movies"),
                          "paths.model": str(dead / "model.pt")})
    seen: list[str] = []
    for name in ("is_dir", "is_file", "exists"):
        real = getattr(Path, name)

        def probe(self, *a, _real=real, _name=name, **kw):
            if "deadshare" in str(self) and threading.current_thread() is threading.main_thread():
                seen.append(f"{_name}({self})")
            return _real(self, *a, **kw)

        monkeypatch.setattr(Path, name, probe)
    hub = Hub(str(config))
    try:
        hub.show()
        for index in range(len(hub.pages)):
            hub.go_to(index)
            pump(app, lambda: False, timeout=0.3)
    finally:
        hub.close()
        app.processEvents()
    assert not seen, seen
