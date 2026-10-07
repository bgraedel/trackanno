"""TaskPanel: background jobs that cannot be stopped are let go, never destroyed."""

from __future__ import annotations

import time


def _pump(app, predicate, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_shutdown_lets_go_of_a_job_it_cannot_stop(app) -> None:
    """A job with no cancellation point outlives shutdown's timeout without an abort."""
    from trackanno.gui.tasks import TaskPanel

    panel = TaskPanel("Go")
    panel.start(lambda reporter: time.sleep(2.5) or "done")
    _pump(app, lambda: panel.busy, timeout=5)
    assert panel.busy, "the job is running"
    assert panel.shutdown(timeout=200) is False, "shutdown reports it could not stop it"
    assert panel._runner is None, "it let go rather than destroying it"
    _pump(app, lambda: False, timeout=3)
