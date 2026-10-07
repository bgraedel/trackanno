"""The hub window: background jobs, the Configure page's Save and Revert."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from harness import check


def pump(app, predicate, timeout: float = 10.0) -> bool:
    """Spin the event loop until ``predicate`` holds or ``timeout`` passes."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_a_job_from_a_closed_project_is_dropped(app, root: Path) -> None:
    """A result lands only on the project it was started for, and only if it
    is the newest job of its name."""
    from trackanno.gui.hub import Hub, ProjectPage
    from trackanno.project import create_project

    first = create_project(root / "jobs_a")
    second = create_project(root / "jobs_b")
    hub = Hub(str(first))
    page = hub.page(ProjectPage)
    try:
        landed: list = []
        gate = threading.Event()

        def held(value):
            def work(reporter):
                gate.wait(5)
                return value
            return work

        page.run_bg("probe", held("old project"), landed.append)
        hub.open_project(second)
        gate.set()
        pump(app, lambda: not page.bg_running("probe"), timeout=1)
        time.sleep(0.1)
        app.processEvents()
        check("a result for a closed project is dropped", landed == [], str(landed))

        gate.clear()
        page.run_bg("probe", held("superseded"), landed.append)
        page.run_bg("probe", lambda reporter: "newest", landed.append)
        check("the newest job lands", pump(app, lambda: "newest" in landed))
        gate.set()
        time.sleep(0.1)
        app.processEvents()
        check("a superseded job of the same name is dropped", landed == ["newest"],
              str(landed))

        failures: list = []
        page.run_bg("broken", lambda reporter: 1 / 0, landed.append, failures.append)
        check("a failure is delivered as a message",
              pump(app, lambda: bool(failures)) and "ZeroDivisionError" in failures[0],
              str(failures))

        def until_cancelled(reporter):
            while True:
                reporter.raise_if_cancelled()
                time.sleep(0.01)

        page.run_bg("slow", until_cancelled, landed.append)
    finally:
        hub.close()
        gate.set()
        app.processEvents()
    check("closing joins what is running", not any(
        t.name == "trackanno-slow" and t.is_alive() for t in threading.enumerate()))
    app.processEvents()
    check("and nothing lands after close", landed == ["newest"], str(landed))


def test_a_dropped_download_clears_its_status(app, root: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    from trackanno.gui.hub import ConfigPage, Hub, _install_shared_model
    from trackanno.project import create_project

    first = create_project(root / "fetch_a")
    gate = threading.Event()

    def fetch(entry, reporter):
        gate.wait(5)
        return first

    monkeypatch.setattr("trackanno.video.models.fetch", fetch)
    hub = Hub(str(first))
    page = hub.page(ConfigPage)
    bar = hub.statusBar()
    try:
        _install_shared_model(page, SimpleNamespace(id="shared"), first)
        app.processEvents()
        check("the download says it is fetching",
              bar.currentMessage() == "fetching shared...", bar.currentMessage())
        page._jobs.cancel()
        app.processEvents()
        check("a cancelled download takes the message down",
              bar.currentMessage() == "shared: download stopped", bar.currentMessage())

        _install_shared_model(page, SimpleNamespace(id="shared"), first)
        hub.project_gen += 1                  # the project changed under the job
        gate.set()
        check("a download for a closed project lands",
              pump(app, lambda: not page.bg_running("download")))
        app.processEvents()
        check("...and takes the message down too",
              bar.currentMessage() == "shared: download stopped", bar.currentMessage())
    finally:
        gate.set()
        hub.close()
        app.processEvents()


def test_save_writes_only_changed_keys_and_revert_reads_the_file(app, root: Path) -> None:
    from trackanno.config import load_config
    from trackanno.gui.hub import ConfigPage, ExportPage, Hub
    from trackanno.project import create_project, write_config

    path = create_project(root / "save_revert")
    # A value the spin box rounds, which a save must not rewrite.
    write_config(path, {"detection.conf": 0.125})
    hub = Hub(str(path))
    page = hub.page(ConfigPage)
    hub.go_to(hub.pages.index(page))
    app.processEvents()
    try:
        check("nothing to save at first", not page.save_button.isEnabled()
              and not page.revert_button.isEnabled())
        page.form._widgets["clips.pad"].setValue(2.5)
        app.processEvents()
        check("an edit enables Save and Revert",
              page.save_button.isEnabled() and page.revert_button.isEnabled())
        check("the title says there are unsaved edits",
              page.heading.title.text() == "Configure *", page.heading.title.text())
        check("an edit alone writes nothing", load_config(path).clips.pad != 2.5)

        before = path.read_text(encoding="utf-8")
        check("Save succeeds", page.save())
        after = path.read_text(encoding="utf-8")
        changed = [(a, b) for a, b in zip(before.splitlines(), after.splitlines()) if a != b]
        check("exactly one line changed", len(changed) == 1
              and "pad" in changed[0][1], str(changed))
        check("the rounded value kept its precision",
              load_config(path).detection.conf == 0.125)
        check("Save leaves nothing to save", not page.save_button.isEnabled()
              and page.heading.title.text() == "Configure")
        check("the window holds the saved config", hub.cfg.clips.pad == 2.5)
        export = hub.page(ExportPage)
        check("other pages wait to be shown", export.rebuild_pending
              and not page.rebuild_pending)
        hub.go_to(hub.pages.index(export))
        app.processEvents()
        check("...and rebuild on arrival", not export.rebuild_pending)
        hub.go_to(hub.pages.index(page))
        app.processEvents()

        page.form._widgets["clips.pad"].setValue(4.0)
        page.categories._groups[0].name.setText("renamed")
        app.processEvents()
        page.revert()
        app.processEvents()
        check("Revert reads the field back from disk",
              page.form.current()["clips.pad"] == 2.5, str(page.form.current()["clips.pad"]))
        check("...and the labels", page.categories._groups[0].name.text() != "renamed")
        check("...and leaves nothing to save", not page._dirty
              and not page.save_button.isEnabled())
        check("...and the file is untouched", path.read_text(encoding="utf-8") == after)
    finally:
        hub.close()
        app.processEvents()


def test_unsaved_edits_ask_before_leaving(app, root: Path) -> None:
    from trackanno.config import load_config
    from trackanno.gui.hub import ConfigPage, Hub
    from trackanno.project import create_project

    path = create_project(root / "unsaved")
    hub = Hub(str(path))
    hub.show()
    page = hub.page(ConfigPage)
    index = hub.pages.index(page)
    hub.go_to(index)
    app.processEvents()
    answers: list[str] = []
    asked: list[int] = []

    def ask():
        asked.append(1)
        return answers.pop(0)

    page._ask_unsaved = ask
    pad = page.form._widgets["clips.pad"]
    try:
        hub.go_to(0)
        app.processEvents()
        check("a clean page leaves without asking", not asked
              and hub.stack.currentWidget() is not page)

        hub.go_to(index)
        pad.setValue(1.5)
        app.processEvents()
        answers.append("cancel")
        hub.go_to(0)
        app.processEvents()
        check("Cancel stays", hub.stack.currentWidget() is page and page._dirty)
        check("...with the sidebar on the page", hub.sidebar.currentRow() == index)

        answers.append("discard")
        hub.go_to(0)
        app.processEvents()
        check("Discard leaves", hub.stack.currentWidget() is not page)
        check("...without writing", load_config(path).clips.pad != 1.5)
        check("...and reverts the form", pad.value() != 1.5 and not page._dirty)

        hub.go_to(index)
        pad.setValue(1.5)
        app.processEvents()
        answers.append("save")
        hub.go_to(0)
        app.processEvents()
        check("Save leaves and writes", hub.stack.currentWidget() is not page
              and load_config(path).clips.pad == 1.5)

        hub.go_to(index)
        pad.setValue(2.25)
        app.processEvents()
        answers.append("cancel")
        hub.close()
        app.processEvents()
        check("closing the window asks too, and Cancel keeps it open",
              hub.isVisible() and len(asked) == 4, str(len(asked)))
        answers.append("discard")
    finally:
        hub.close()
        app.processEvents()
    check("Discard lets the window close", not hub.isVisible())


def test_loading_and_the_model_probe_leave_the_form_clean(app, root: Path,
                                                          monkeypatch) -> None:
    """A load emits no change, and a disagreeing checkpoint is offered, not applied."""
    from trackanno.gui.configform import ConfigForm
    from trackanno.gui.hub import ConfigPage, Hub
    from trackanno.project import create_project

    form = ConfigForm("all")
    fired: list[int] = []
    form.changed.connect(lambda: fired.append(1))
    form.load({"paths": {"video_dir": "/a", "model": "/b.pt", "work_dir": "/c",
                         "csv": "/d.csv"}, "detection": {"task": "pose"}})
    check("loading emits no change", fired == [], str(len(fired)))

    path = create_project(root / "probe_offer")
    (path.parent / "best.pt").write_bytes(b"weights")
    monkeypatch.setattr("trackanno.detect.tiling.read_model_info", lambda p: {
        "task": "pose", "n_keypoints": 8, "classes": ["cell"], "name": "best.pt"})
    before = path.read_text(encoding="utf-8")
    hub = Hub(str(path))
    page = hub.page(ConfigPage)
    hub.go_to(hub.pages.index(page))
    try:
        check("the probe offers the checkpoint's settings",
              pump(app, lambda: not page.model_row.isHidden()))
        check("...without touching the form", not page._dirty
              and page.form._widgets["detection.task"].currentText() == "detect")
        page.model_apply.click()
        app.processEvents()
        check("a click applies them", page._dirty
              and page.form._widgets["detection.task"].currentText() == "pose"
              and page.keypoints.count.value() == 8)
        page.revert()
        app.processEvents()
        check("Revert leaves nothing to save", not page._dirty)
        check("the file was never written", path.read_text(encoding="utf-8") == before)
    finally:
        page._dirty = False
        hub.close()
        app.processEvents()
