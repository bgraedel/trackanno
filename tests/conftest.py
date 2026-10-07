"""Shared setup for every suite.

pytest imports this before any test module, which is what makes it the right
place for the three things that must happen first: ``src/`` on ``sys.path`` so
the tests run against the working tree rather than an installed copy, the
offscreen Qt platform so a test run never steals focus, and the tests directory
itself on the path so ``fixtures`` and ``harness`` import by bare name.

Nothing here reads data the repository does not carry. Projects are generated
into a temporary directory by :mod:`fixtures` and deleted afterwards. To run the
window-independence checks against a real prepared project as well, pass
``--project-root PATH`` (repeatable).
"""

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

import harness  # noqa: E402


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--project-root", action="append", default=[], metavar="PATH",
        help="a real project (config.yaml + a prepared work/) to run the "
             "window-independence checks over as well; repeatable",
    )


def pytest_terminal_summary(terminalreporter) -> None:
    """Report the check count, which is finer grained than the test count."""
    terminalreporter.write_line(f"{len(harness.PASSED)} checks passed")


@pytest.fixture(scope="session")
def app():
    """One QApplication for the whole run.

    Session-scoped because Qt allows exactly one per process and tears down
    badly if it is rebuilt. Tests that need the event loop spun take this and
    call ``processEvents`` themselves.
    """
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv[:1])
    yield app
    app.processEvents()

    # Shut down anything a test left open before the QApplication goes. Pages
    # own background QThreads that close() does not cancel, and a live QThread
    # at interpreter exit kills the process after every test has passed, which
    # reads as a crash with no failing test.
    for widget in list(app.topLevelWidgets()):
        for page in getattr(widget, "pages", ()) or ():
            with contextlib.suppress(Exception):
                page.shutdown()
        with contextlib.suppress(Exception):
            widget.close()
    app.processEvents()


@pytest.fixture(scope="module")
def root(tmp_path_factory, request):
    """A scratch directory for one suite, with the GUI's state file inside it.

    ``TRACKANNO_STATE_DIR`` is redirected so a test run cannot read or write the
    recently-opened list of whoever is running it.
    """
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
