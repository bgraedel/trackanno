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
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

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
