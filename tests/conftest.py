"""Shared setup: ``src/`` on the path, offscreen Qt, and per-run state directories."""

from __future__ import annotations

import os
import contextlib
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# Per-user state (recent list, view toggles) never touches the real profile.
os.environ.setdefault("TRACKANNO_STATE_DIR", tempfile.mkdtemp(prefix="trackanno-state-"))

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--project-root", action="append", default=[], metavar="PATH",
        help="a prepared real project to run the window-independence checks over too",
    )


@pytest.fixture(scope="session")
def app():
    """One QApplication for the whole run; Qt allows one per process."""
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv[:1])
    yield app
    app.processEvents()

    # A live QThread at interpreter exit kills the process, so shut pages down first.
    for widget in list(app.topLevelWidgets()):
        for page in getattr(widget, "pages", ()) or ():
            with contextlib.suppress(Exception):
                page.shutdown()
        with contextlib.suppress(Exception):
            widget.close()
    app.processEvents()


@pytest.fixture(scope="module")
def root(tmp_path_factory, request):
    """A scratch directory for one suite, which also holds the GUI's state file."""
    root = tmp_path_factory.mktemp(request.module.__name__.replace("test_", "") + "-")
    previous = os.environ.get("TRACKANNO_STATE_DIR")
    os.environ["TRACKANNO_STATE_DIR"] = str(root / "state")
    yield root
    if previous is None:
        os.environ.pop("TRACKANNO_STATE_DIR", None)
    else:
        os.environ["TRACKANNO_STATE_DIR"] = previous


@pytest.fixture(autouse=True)
def _no_unsaved_prompt(request, monkeypatch):
    """Unsaved Configure edits are discarded instead of opening a modal box."""
    if "app" in request.fixturenames:
        monkeypatch.setattr("trackanno.gui.hub.ConfigPage._ask_unsaved",
                            lambda self: "discard")
    yield
