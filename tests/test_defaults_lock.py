"""Everything added by the boundary-and-selection work is OFF in a fresh project.

Each feature on this branch was built to be optional, and each was checked once,
by hand, on the day it landed. That is not a guarantee - it is a memory. A knob
whose default drifts costs more than a broken feature, because a broken feature
announces itself and a changed default does not: predictions quietly become a
different kind of row, the annotation queue quietly stops being a random sample,
and every number computed from either quietly means something else.

So the claims here are about the SHIPPED TEMPLATE, not about a config a test
wrote. `project.create_project` is what a new user gets, and these are the values
it has to keep giving them. Together they say: a project created today behaves
the way it did before any of this existed.
"""

from __future__ import annotations

from pathlib import Path

from harness import check
from trackanno.config import load_config
from trackanno.project import create_project


def _fresh(root: Path, name: str):
    """The config a new user gets, read back the way the program reads it.

    A directory per caller: `root` is one scratch dir for the whole module, and
    `create_project` is not idempotent over an existing one.
    """
    return load_config(create_project(Path(root) / name))


def test_a_new_project_tiles_its_windows(root: Path) -> None:
    """The dense predictor tiles and keeps every run of the argmax."""
    cfg = _fresh(root, "windows")

    check("the dense window stride is not written out by hand",
          cfg.predict.window_stride == "auto", repr(cfg.predict.window_stride))
    check("...and it resolves to the window length, which is a tiling",
          cfg.predict.resolved_stride(cfg.clips) == cfg.clips.frames,
          str(cfg.predict.resolved_stride(cfg.clips)))
    check("the decode has no bout length to absorb toward",
          cfg.predict.min_bout_frames == 0, repr(cfg.predict.min_bout_frames))


def test_a_new_project_hands_out_clips_in_its_own_order(root: Path) -> None:
    """Nothing selects, so the labelled set stays a random sample of the data.

    This is the one default whose drift would be invisible AND unrecoverable.
    Directed selection is worth having, but the moment it runs, every statistic
    computed over the labelled set describes the selector as well as the animals,
    and a project that was never told to turn it on would have no record saying
    when that started.
    """
    cfg = _fresh(root, "queue")

    check("no rule ranks the queue", cfg.suggest.select_rule == "off",
          repr(cfg.suggest.select_rule))
    check("the holdout still exists to be an unbiased estimate of something",
          0.0 < cfg.suggest.audit_fraction < 1.0, repr(cfg.suggest.audit_fraction))


def test_the_shipped_template_still_loads_without_a_single_warning(root: Path) -> None:
    """A default that warns is a default nobody reads twice.

    Every key added on this branch had to be registered in `_known`, and an
    unregistered key is IGNORED with a warning rather than refused - this file
    has prior art for exactly that bug. Turning the warning into a failure is the
    only way that stays true, because a warning in a test suite that already
    passes is a warning nobody sees.
    """
    import warnings

    project = create_project(Path(root) / "quiet")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg = load_config(project)

    check("the template a new user gets loads clean", cfg is not None)
